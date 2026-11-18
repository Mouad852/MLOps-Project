from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import settings
from app.ml.pipelines.training import build_reference_profile


def _status(score: float, warning_threshold: float, drift_threshold: float) -> str:
    if score >= drift_threshold:
        return "drifted"
    if score >= warning_threshold:
        return "warning"
    return "stable"


def _psi(reference: list[float], incoming: list[float]) -> float:
    reference_values = np.clip(np.asarray(reference, dtype=float), 1e-6, None)
    incoming_values = np.clip(np.asarray(incoming, dtype=float), 1e-6, None)
    reference_values /= reference_values.sum()
    incoming_values /= incoming_values.sum()
    return float(np.sum((incoming_values - reference_values) * np.log(incoming_values / reference_values)))


def compare_with_reference(
    incoming_features: pd.DataFrame,
    reference_profile: dict[str, Any],
    training_run_id: str | None = None,
) -> dict[str, Any]:
    """Compare incoming feature distributions using fixed training-time bins/categories."""
    row_count = int(len(incoming_features))
    timestamp = datetime.now(timezone.utc).isoformat()
    if row_count < settings.drift_min_batch_size:
        return {
            "status": "insufficient_data",
            "incoming_row_count": row_count,
            "drifted_feature_count": 0,
            "drifted_feature_percentage": 0.0,
            "features": {},
            "timestamp": timestamp,
            "training_run_id": training_run_id,
        }

    features: dict[str, Any] = {}
    for column, config in reference_profile.get("numeric", {}).items():
        if column not in incoming_features.columns or not config.get("bin_edges"):
            features[column] = {"method": "psi", "status": "insufficient_data", "score": None}
            continue
        series = incoming_features[column]
        values = pd.to_numeric(series, errors="coerce")
        observed = values.dropna().to_numpy(dtype=float)
        if observed.size == 0:
            incoming_proportions = [0.0] * len(config["proportions"])
        else:
            # Finite JSON-safe edge bounds represent the open-ended first and
            # last bins. Clamp out-of-range values into those boundary bins so
            # a shifted tail is counted instead of silently discarded.
            edges = np.asarray(config["bin_edges"], dtype=float)
            observed = np.clip(observed, edges[0], edges[-1])
            counts, _ = np.histogram(observed, bins=edges)
            incoming_proportions = (counts / observed.size).tolist()
        score = _psi(config["proportions"], incoming_proportions)
        missing_change = abs(float(values.isna().mean()) - float(config.get("missing_rate", 0.0)))
        feature_score = max(score, missing_change)
        features[column] = {
            "method": "psi",
            "score": feature_score,
            "psi": score,
            "missing_rate_change": missing_change,
            "threshold": {"warning": 0.10, "drifted": 0.25},
            "status": _status(feature_score, 0.10, 0.25),
        }

    for column, config in reference_profile.get("categorical", {}).items():
        if column not in incoming_features.columns:
            features[column] = {"method": "tvd", "status": "insufficient_data", "score": None}
            continue
        series = incoming_features[column].astype("string").fillna("<MISSING>")
        reference_proportions = config.get("proportions", {})
        incoming_proportions = series.value_counts(normalize=True).to_dict()
        categories = set(reference_proportions) | set(incoming_proportions)
        tvd = 0.5 * sum(
            abs(float(incoming_proportions.get(category, 0.0)) - float(reference_proportions.get(category, 0.0)))
            for category in categories
        )
        non_missing = series[series != "<MISSING>"]
        unseen_rate = (
            float((~non_missing.isin(reference_proportions)).mean()) if len(non_missing) else 0.0
        )
        missing_change = abs(float((series == "<MISSING>").mean()) - float(config.get("missing_rate", 0.0)))
        feature_score = max(tvd, unseen_rate, missing_change)
        features[column] = {
            "method": "tvd",
            "score": feature_score,
            "total_variation_distance": tvd,
            "unseen_category_rate": unseen_rate,
            "missing_rate_change": missing_change,
            "threshold": {"warning": 0.10, "drifted": 0.25},
            "status": _status(feature_score, 0.10, 0.25),
        }

    drifted = [name for name, result in features.items() if result.get("status") == "drifted"]
    warnings = [name for name, result in features.items() if result.get("status") == "warning"]
    status = "drifted" if drifted else "warning" if warnings else "stable"
    return {
        "status": status,
        "incoming_row_count": row_count,
        "drifted_feature_count": len(drifted),
        "drifted_feature_percentage": round(100 * len(drifted) / max(len(features), 1), 2),
        "features": features,
        "timestamp": timestamp,
        "training_run_id": training_run_id,
    }
