from __future__ import annotations

import numbers
import time
import uuid

import mlflow
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from mlflow.models import infer_signature
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import SessionLocal, get_db
from app.ml.drift import compare_with_reference
from app.ml.pipelines.training import train_and_compare
from app.models.dataset import Dataset, PredictionEvent, TrainingRun
from app.schemas.dataset import (
    PredictRequest,
    PredictResponse,
    TrainingRunOut,
    TrainingRunSummary,
    TrainRequest,
)

router = APIRouter(prefix="/training", tags=["training"])

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.api_key:
        raise HTTPException(401, "Invalid or missing API key")


@router.post("/run", response_model=TrainingRunOut, status_code=202)
def run_training(req: TrainRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == req.dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    try:
        columns = pd.read_csv(dataset.file_path, nrows=0).columns
    except Exception as exc:
        raise HTTPException(400, "Could not read dataset") from exc
    if req.target_column not in columns:
        raise HTTPException(400, f"Target column '{req.target_column}' not in dataset")

    run = TrainingRun(id=uuid.uuid4(), dataset_id=dataset.id, status="pending")
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(
        _execute_training,
        run.id,
        dataset.id,
        dataset.file_path,
        dataset.dataset_hash,
        req.target_column,
        req.task_type,
    )
    return run


def _scalar_metrics(metrics: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, numbers.Real) and not isinstance(value, bool)
    }


def _execute_training(
    run_id: uuid.UUID,
    dataset_id: uuid.UUID,
    file_path: str,
    dataset_hash: str,
    target_column: str,
    task_type: str,
):
    """Execute training with an independent session because the request session is closed."""
    db = SessionLocal()
    try:
        run = db.get(TrainingRun, run_id)
        if run is None:
            return
        run.status = "running"
        db.commit()

        df = pd.read_csv(file_path)
        mlflow.set_experiment(f"dataset_{dataset_id}")
        outcome = train_and_compare(df, target_column, task_type)

        all_metrics = {}
        best_model_uri = None
        best_feature_importance = None
        for name, result in outcome["results"].items():
            with mlflow.start_run(run_name=name) as mlflow_run:
                mlflow.log_params(
                    {
                        "dataset_id": str(dataset_id),
                        "dataset_hash": dataset_hash,
                        "task_type": task_type,
                        "model": name,
                        "random_seed": 42,
                        **{key: str(value) for key, value in result["best_params"].items()},
                    }
                )
                mlflow.log_metrics(_scalar_metrics(result["metrics"]))
                mlflow.log_dict(result["feature_importance"], "feature_importance.json")
                mlflow.log_dict(outcome["reference_profile"], "reference_profile.json")

                input_example = result["input_example"]
                predictions = result["pipeline"].predict(input_example)
                signature = infer_signature(input_example, predictions)
                mlflow.sklearn.log_model(
                    result["pipeline"],
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                )

                if name == outcome["best_model"]:
                    best_model_uri = f"runs:/{mlflow_run.info.run_id}/model"
                    run.mlflow_run_id = mlflow_run.info.run_id
                    best_feature_importance = result["feature_importance"]
            all_metrics[name] = result["metrics"]

        run.status = "completed"
        run.best_model_name = outcome["best_model"]
        run.best_model_uri = best_model_uri
        run.feature_columns = outcome["feature_columns"]
        run.target_labels = outcome["target_labels"]
        run.reference_profile = outcome["reference_profile"]
        run.metrics = all_metrics
        run.feature_importance = best_feature_importance

        dataset = db.get(Dataset, dataset_id)
        if dataset is not None:
            dataset.target_column = target_column
            dataset.task_type = task_type
        db.commit()
    except Exception as exc:
        db.rollback()
        failed_run = db.get(TrainingRun, run_id)
        if failed_run is not None:
            failed_run.status = "failed"
            failed_run.error_message = f"{type(exc).__name__}: {exc}"
            db.commit()
    finally:
        db.close()


@router.get("/runs", response_model=list[TrainingRunSummary])
def list_training_runs(
    dataset_id: uuid.UUID | None = None, limit: int = 10, db: Session = Depends(get_db)
):
    query = db.query(TrainingRun)
    if dataset_id is not None:
        query = query.filter(TrainingRun.dataset_id == dataset_id)
    return query.order_by(TrainingRun.created_at.desc()).limit(limit).all()


@router.get("/{run_id}/monitoring")
def get_monitoring(run_id: uuid.UUID, db: Session = Depends(get_db)):
    run = db.get(TrainingRun, run_id)
    if not run:
        raise HTTPException(404, "Training run not found")
    events = (
        db.query(PredictionEvent)
        .filter(PredictionEvent.training_run_id == run_id)
        .order_by(PredictionEvent.request_time.desc())
        .all()
    )
    successful = [event for event in events if event.success]
    latest = events[0] if events else None
    return {
        "training_run_id": str(run_id),
        "total_prediction_requests": len(events),
        "total_predicted_rows": sum(event.record_count for event in successful),
        "average_latency_ms": round(
            sum(event.latency_ms for event in successful) / max(len(successful), 1), 2
        ),
        "most_recent_prediction_time": latest.request_time.isoformat() if latest else None,
        "most_recent_drift_status": latest.drift_status if latest else None,
        "drifted_feature_count": latest.drifted_feature_count if latest else 0,
        "most_recent_drift_report": latest.drift_report if latest else None,
    }


@router.get("/{run_id}", response_model=TrainingRunOut)
def get_training_run(run_id: uuid.UUID, db: Session = Depends(get_db)):
    run = db.get(TrainingRun, run_id)
    if not run:
        raise HTTPException(404, "Training run not found")
    return run


@router.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    run = db.get(TrainingRun, req.training_run_id)
    if not run or run.status != "completed":
        raise HTTPException(404, "Completed training run not found")
    if not req.records:
        raise HTTPException(400, "At least one record is required")

    started = time.perf_counter()
    input_df = pd.DataFrame(req.records)
    drift_report = compare_with_reference(
        input_df, run.reference_profile or {}, training_run_id=str(run.id)
    )

    def record_event(success: bool, report: dict):
        event = PredictionEvent(
            training_run_id=run.id,
            record_count=len(input_df),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
            success=success,
            drift_status=report.get("status"),
            drifted_feature_count=report.get("drifted_feature_count", 0),
            drift_report=report,
        )
        db.add(event)
        db.commit()

    missing = set(run.feature_columns or []) - set(input_df.columns)
    if missing:
        record_event(False, drift_report)
        raise HTTPException(400, f"Missing feature columns: {sorted(missing)}")

    try:
        model = mlflow.sklearn.load_model(run.best_model_uri)
        predictions = model.predict(input_df[run.feature_columns])
    except (ValueError, TypeError) as exc:
        record_event(False, drift_report)
        raise HTTPException(400, f"Invalid input data: {exc}") from exc
    except Exception as exc:
        record_event(False, drift_report)
        raise HTTPException(500, "The trained model could not be loaded") from exc

    record_event(True, drift_report)
    if run.target_labels:
        try:
            decoded = [run.target_labels[int(prediction)] for prediction in predictions]
        except (IndexError, TypeError, ValueError) as exc:
            raise HTTPException(500, "The stored classification labels are invalid") from exc
        return {"predictions": decoded, "drift_report": drift_report}
    return {
        "predictions": [prediction.item() if hasattr(prediction, "item") else prediction for prediction in predictions],
        "drift_report": drift_report,
    }
