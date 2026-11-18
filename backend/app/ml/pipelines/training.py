from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from app.core.config import settings
from app.ml.pipelines.preprocessing import build_preprocessor, infer_column_types

RANDOM_SEED = 42


def get_candidate_models(task_type: str) -> dict[str, Any]:
    if task_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1_000, random_state=RANDOM_SEED),
            "decision_tree": DecisionTreeClassifier(random_state=RANDOM_SEED),
            "random_forest": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1),
            "xgboost": XGBClassifier(
                random_state=RANDOM_SEED, n_jobs=1, eval_metric="logloss", verbosity=0
            ),
        }
    if task_type == "regression":
        return {
            "linear_regression": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(random_state=RANDOM_SEED),
            "random_forest": RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=1),
            "xgboost": XGBRegressor(random_state=RANDOM_SEED, n_jobs=1, verbosity=0),
        }
    raise ValueError("task_type must be 'classification' or 'regression'")


def get_parameter_spaces(task_type: str) -> dict[str, dict[str, list[Any]]]:
    if task_type == "classification":
        return {
            "logistic_regression": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__solver": ["lbfgs"],
            },
            "decision_tree": {
                "model__max_depth": [None, 3, 5, 10],
                "model__min_samples_split": [2, 5, 10],
                "model__criterion": ["gini", "entropy"],
            },
            "random_forest": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": ["sqrt", "log2"],
            },
            "xgboost": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [2, 3, 5],
                "model__learning_rate": [0.03, 0.1, 0.2],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
        }
    if task_type == "regression":
        return {
            "linear_regression": {"model__fit_intercept": [True, False]},
            "decision_tree": {
                "model__max_depth": [None, 3, 5, 10],
                "model__min_samples_split": [2, 5, 10],
                "model__criterion": ["squared_error", "absolute_error"],
            },
            "random_forest": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__max_features": [1.0, "sqrt"],
            },
            "xgboost": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [2, 3, 5],
                "model__learning_rate": [0.03, 0.1, 0.2],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
        }
    raise ValueError("task_type must be 'classification' or 'regression'")


def validate_training_data(df: pd.DataFrame, target_column: str, task_type: str) -> pd.DataFrame:
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataset")
    if len(df.columns) < 2:
        raise ValueError("The dataset must contain at least one feature column")

    cleaned = df.dropna(subset=[target_column]).copy()
    if cleaned.empty:
        raise ValueError("The target column has no usable values")
    if task_type == "classification":
        if cleaned[target_column].nunique(dropna=True) < 2:
            raise ValueError("Classification requires at least two target classes")
    elif not pd.api.types.is_numeric_dtype(cleaned[target_column]):
        raise ValueError("Regression targets must be numeric")
    return cleaned


def _effective_cv(y_train: pd.Series, task_type: str) -> int:
    requested = max(2, settings.cv_folds)
    if task_type == "classification":
        requested = min(requested, int(y_train.value_counts().min()))
    if requested < 2:
        raise ValueError("Not enough samples per class for cross-validation")
    return requested


def evaluate_classification(model: Any, X_holdout: pd.DataFrame, y_holdout: np.ndarray) -> dict[str, Any]:
    predictions = model.predict(X_holdout)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_holdout, predictions)),
        "precision": float(precision_score(y_holdout, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_holdout, predictions, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_holdout, predictions, average="weighted", zero_division=0)),
    }
    matrix = confusion_matrix(y_holdout, predictions)
    metrics["confusion_matrix"] = matrix.tolist()
    if matrix.shape == (2, 2):
        metrics.update(
            {
                "confusion_matrix_tn": int(matrix[0, 0]),
                "confusion_matrix_fp": int(matrix[0, 1]),
                "confusion_matrix_fn": int(matrix[1, 0]),
                "confusion_matrix_tp": int(matrix[1, 1]),
            }
        )
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_holdout)
        if len(np.unique(y_holdout)) == 2 and probabilities.shape[1] == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_holdout, probabilities[:, 1]))
    return metrics


def evaluate_regression(model: Any, X_holdout: pd.DataFrame, y_holdout: np.ndarray) -> dict[str, float]:
    predictions = model.predict(X_holdout)
    return {
        "rmse": float(root_mean_squared_error(y_holdout, predictions)),
        "mae": float(mean_absolute_error(y_holdout, predictions)),
        "r2": float(r2_score(y_holdout, predictions)),
    }


def calculate_feature_importance(
    model: Any, X_holdout: pd.DataFrame, y_holdout: np.ndarray, feature_columns: list[str]
) -> dict[str, float]:
    result = permutation_importance(
        model, X_holdout, y_holdout, n_repeats=3, random_state=RANDOM_SEED, n_jobs=settings.search_n_jobs
    )
    return {
        feature: float(importance)
        for feature, importance in zip(feature_columns, result.importances_mean, strict=True)
    }


def build_reference_profile(training_features: pd.DataFrame) -> dict[str, Any]:
    """Capture compact training distributions without retaining raw training rows."""
    profile: dict[str, Any] = {"row_count": int(len(training_features)), "numeric": {}, "categorical": {}}
    for column in training_features.columns:
        series = training_features[column]
        missing_rate = float(series.isna().mean())
        if pd.api.types.is_numeric_dtype(series):
            values = series.dropna().astype(float)
            values = values[np.isfinite(values)]
            if values.empty:
                profile["numeric"][column] = {
                    "bin_edges": [], "proportions": [], "missing_rate": missing_rate
                }
                continue
            minimum = float(values.min())
            maximum = float(values.max())
            if minimum == maximum:
                margin = max(1.0, abs(minimum) * 1e-6)
                edges = [minimum - margin, maximum + margin]
            else:
                quantiles = values.quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
                inner_edges = sorted(
                    {
                        float(value)
                        for value in quantiles[1:-1]
                        if minimum < float(value) < maximum
                    }
                )
                lower_margin = max(1.0, abs(minimum) * 1e-6)
                upper_margin = max(1.0, abs(maximum) * 1e-6)
                edges = [minimum - lower_margin, *inner_edges, maximum + upper_margin]
            counts, _ = np.histogram(values.to_numpy(), bins=edges)
            proportions = (counts / max(len(values), 1)).tolist()
            profile["numeric"][column] = {
                "bin_edges": edges,
                "proportions": [float(value) for value in proportions],
                "missing_rate": missing_rate,
            }
        else:
            frequencies = series.astype("string").fillna("<MISSING>").value_counts(normalize=True)
            profile["categorical"][column] = {
                "proportions": {str(key): float(value) for key, value in frequencies.items()},
                "missing_rate": missing_rate,
            }
    return profile


def _json_safe_label(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def train_and_compare(df: pd.DataFrame, target_column: str, task_type: str) -> dict[str, Any]:
    cleaned = validate_training_data(df, target_column, task_type)
    features = cleaned.drop(columns=[target_column])
    numeric_columns, categorical_columns = infer_column_types(cleaned, target_column)
    if not numeric_columns and not categorical_columns:
        raise ValueError("The dataset must contain at least one feature column")

    labels: list[Any] = []
    target = cleaned[target_column]
    if task_type == "classification":
        encoder = LabelEncoder()
        model_target: np.ndarray = encoder.fit_transform(target)
        labels = [_json_safe_label(value) for value in encoder.classes_]
    else:
        model_target = target.to_numpy(dtype=float)

    stratify = model_target if task_type == "classification" else None
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        features,
        model_target,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )
    cv_folds = _effective_cv(pd.Series(y_train), task_type)
    candidates = get_candidate_models(task_type)
    spaces = get_parameter_spaces(task_type)
    results: dict[str, dict[str, Any]] = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_columns, categorical_columns)),
                ("model", estimator),
            ]
        )
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=spaces[name],
            n_iter=settings.tuning_iterations,
            cv=cv_folds,
            n_jobs=settings.search_n_jobs,
            random_state=RANDOM_SEED,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        if task_type == "classification":
            metrics = evaluate_classification(best_pipeline, X_holdout, y_holdout)
        else:
            metrics = evaluate_regression(best_pipeline, X_holdout, y_holdout)
        results[name] = {
            "pipeline": best_pipeline,
            "best_params": search.best_params_,
            "metrics": metrics,
            "feature_importance": calculate_feature_importance(
                best_pipeline, X_holdout, y_holdout, features.columns.tolist()
            ),
            "input_example": X_holdout.head(1),
        }

    if task_type == "classification":
        if len(labels) == 2:
            best_model = max(
                results,
                key=lambda name: results[name]["metrics"].get("roc_auc", float("-inf")),
            )
        else:
            best_model = max(results, key=lambda name: results[name]["metrics"]["f1"])
    else:
        best_model = min(results, key=lambda name: results[name]["metrics"]["rmse"])

    return {
        "best_model": best_model,
        "feature_columns": features.columns.tolist(),
        "target_labels": labels,
        "reference_profile": build_reference_profile(X_train),
        "results": results,
    }
