from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


def test_upload_valid_csv_returns_hash_and_preview(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    client = TestClient(app)
    content = b"age,city,target\n30,Rabat,yes\n31,Fes,no\n"

    response = client.post(
        "/datasets/upload",
        files={"file": ("demo.csv", content, "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["dataset_hash"]) == 64
    preview = client.get(f"/datasets/{payload['id']}/preview?rows=999")
    assert preview.status_code == 200
    assert preview.json()["rows"][0]["city"] == "Rabat"


def test_upload_rejects_non_csv_and_empty_files(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    client = TestClient(app)

    extension_response = client.post(
        "/datasets/upload", files={"file": ("demo.txt", b"a,b\n1,2\n", "text/csv")}
    )
    empty_response = client.post(
        "/datasets/upload", files={"file": ("empty.csv", b"", "text/csv")}
    )

    assert extension_response.status_code == 400
    assert empty_response.status_code == 400


def test_invalid_task_type_returns_422():
    client = TestClient(app)
    response = client.post(
        "/training/run",
        json={
            "dataset_id": "00000000-0000-0000-0000-000000000000",
            "target_column": "target",
            "task_type": "forecasting",
        },
    )
    assert response.status_code == 422
