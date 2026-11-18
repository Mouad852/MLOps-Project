import pandas as pd

from app.core.config import settings
from app.ml.pipelines.training import train_and_compare


def test_classification_training_keeps_labels_and_profile(monkeypatch):
    monkeypatch.setattr(settings, "tuning_iterations", 1)
    monkeypatch.setattr(settings, "cv_folds", 2)
    monkeypatch.setattr(settings, "search_n_jobs", 1)
    df = pd.DataFrame(
        {
            "age": list(range(30)),
            "plan": ["basic", "pro"] * 15,
            "churn": ["no", "yes"] * 15,
        }
    )

    outcome = train_and_compare(df, "churn", "classification")

    assert set(outcome["results"]) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    }
    assert outcome["target_labels"] == ["no", "yes"]
    assert "age" in outcome["reference_profile"]["numeric"]
    assert "plan" in outcome["reference_profile"]["categorical"]
    assert "roc_auc" in outcome["results"][outcome["best_model"]]["metrics"]


def test_regression_training_returns_holdout_metrics(monkeypatch):
    monkeypatch.setattr(settings, "tuning_iterations", 1)
    monkeypatch.setattr(settings, "cv_folds", 2)
    monkeypatch.setattr(settings, "search_n_jobs", 1)
    df = pd.DataFrame(
        {
            "x": list(range(30)),
            "segment": ["a", "b"] * 15,
            "target": [float(i * 2 + 1) for i in range(30)],
        }
    )

    outcome = train_and_compare(df, "target", "regression")

    assert set(outcome["results"]) == {
        "linear_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    }
    assert {"rmse", "mae", "r2"}.issubset(
        outcome["results"][outcome["best_model"]]["metrics"]
    )
