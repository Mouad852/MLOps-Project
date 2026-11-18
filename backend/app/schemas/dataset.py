from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from typing import Literal


class DatasetOut(BaseModel):
    id: UUID
    filename: str
    n_rows: int
    n_cols: int
    column_summary: dict
    dataset_hash: str

    model_config = ConfigDict(from_attributes=True)


class DatasetPreview(BaseModel):
    columns: list[str]
    rows: list[dict]


class TrainingRunSummary(BaseModel):
    id: UUID
    status: str
    best_model_name: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TrainRequest(BaseModel):
    dataset_id: UUID
    target_column: str
    task_type: Literal["classification", "regression"]


class TrainingRunOut(BaseModel):
    id: UUID
    status: str
    mlflow_run_id: str | None
    best_model_name: str | None
    feature_columns: list[str] | None
    metrics: dict | None
    feature_importance: dict | None
    error_message: str | None

    model_config = ConfigDict(from_attributes=True)


class PredictRequest(BaseModel):
    training_run_id: UUID
    records: list[dict]


class PredictResponse(BaseModel):
    predictions: list
    drift_report: dict | None = None
