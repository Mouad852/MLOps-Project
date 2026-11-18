import hashlib
import os
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db import get_db
from app.ml.pipelines.preprocessing import profile_dataframe
from app.models.dataset import Dataset
from app.schemas.dataset import DatasetOut, DatasetPreview

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=DatasetOut)
def upload_dataset(file: UploadFile, db: Session = Depends(get_db)):
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(400, "Only CSV files are supported")

    contents = file.file.read()
    if not contents:
        raise HTTPException(400, "The CSV file is empty")
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(413, f"CSV files must be smaller than {settings.max_upload_bytes} bytes")

    os.makedirs(settings.upload_dir, exist_ok=True)
    dataset_id = uuid.uuid4()
    file_path = os.path.join(settings.upload_dir, f"{dataset_id}.csv")

    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        df = pd.read_csv(file_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        Path(file_path).unlink(missing_ok=True)
        raise HTTPException(400, f"Could not parse CSV file: {exc}") from exc
    if df.empty or len(df.columns) == 0:
        Path(file_path).unlink(missing_ok=True)
        raise HTTPException(400, "The CSV file must contain at least one data row and one column")

    summary = profile_dataframe(df)
    dataset_hash = hashlib.sha256(contents).hexdigest()

    dataset = Dataset(
        id=dataset_id,
        filename=filename,
        file_path=file_path,
        n_rows=len(df),
        n_cols=len(df.columns),
        column_summary=summary,
        dataset_hash=dataset_hash,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: uuid.UUID, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    return dataset


@router.get("", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).order_by(Dataset.created_at.desc()).all()


@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
def preview_dataset(dataset_id: uuid.UUID, rows: int = 10, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    rows = max(1, min(rows, settings.max_preview_rows))
    df = pd.read_csv(dataset.file_path).head(rows)
    df = df.where(pd.notnull(df), None)
    return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}
