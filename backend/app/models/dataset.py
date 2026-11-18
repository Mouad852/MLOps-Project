import uuid
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Integer, Float, Boolean, JSON, ForeignKey, Index, Uuid

from app.core.db import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    n_rows = Column(Integer)
    n_cols = Column(Integer)
    column_summary = Column(JSON)
    dataset_hash = Column(String(64), nullable=False)
    target_column = Column(String, nullable=True)
    task_type = Column(String, nullable=True)  # "classification" | "regression"
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(Uuid(as_uuid=True), ForeignKey("datasets.id"), nullable=False, index=True)
    mlflow_run_id = Column(String, nullable=True)
    status = Column(String, default="pending")  # pending | running | completed | failed
    best_model_name = Column(String, nullable=True)
    best_model_uri = Column(String, nullable=True)
    feature_columns = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    feature_importance = Column(JSON, nullable=True)
    target_labels = Column(JSON, nullable=True)
    reference_profile = Column(JSON, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


Index("ix_datasets_created_at", Dataset.created_at)
Index("ix_training_runs_created_at", TrainingRun.created_at)


class PredictionEvent(Base):
    __tablename__ = "prediction_events"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    training_run_id = Column(Uuid(as_uuid=True), ForeignKey("training_runs.id"), nullable=False, index=True)
    request_time = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    record_count = Column(Integer, nullable=False)
    latency_ms = Column(Float, nullable=False)
    success = Column(Boolean, nullable=False)
    drift_status = Column(String, nullable=True)
    drifted_feature_count = Column(Integer, nullable=False, default=0)
    drift_report = Column(JSON, nullable=True)
