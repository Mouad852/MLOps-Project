import pandas as pd

from app.core.config import settings
from app.ml.drift import build_reference_profile, compare_with_reference


def test_drift_reports_insufficient_data(monkeypatch):
    monkeypatch.setattr(settings, "drift_min_batch_size", 30)
    training = pd.DataFrame({"amount": range(100), "plan": ["basic", "pro"] * 50})
    report = compare_with_reference(training.head(10), build_reference_profile(training))
    assert report["status"] == "insufficient_data"


def test_drift_reports_stable_for_same_distribution(monkeypatch):
    monkeypatch.setattr(settings, "drift_min_batch_size", 30)
    training = pd.DataFrame({"amount": range(100), "plan": ["basic", "pro"] * 50})
    incoming = training.sample(60, random_state=42).reset_index(drop=True)
    report = compare_with_reference(incoming, build_reference_profile(training))
    assert report["status"] in {"stable", "warning"}
    assert report["features"]["amount"]["method"] == "psi"


def test_drift_reports_shifted_numeric_and_unseen_categories(monkeypatch):
    monkeypatch.setattr(settings, "drift_min_batch_size", 30)
    training = pd.DataFrame({"amount": range(100), "plan": ["basic", "pro"] * 50})
    incoming = pd.DataFrame({"amount": range(500, 560), "plan": ["enterprise"] * 60})
    report = compare_with_reference(incoming, build_reference_profile(training))
    assert report["status"] == "drifted"
    assert report["features"]["amount"]["status"] == "drifted"
    assert report["features"]["plan"]["unseen_category_rate"] == 1.0
