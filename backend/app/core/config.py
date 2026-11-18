from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./mlops.db"
    mlflow_tracking_uri: str = "http://localhost:5000"
    upload_dir: str = "./data/uploads"
    api_key: str = "changeme"
    tuning_iterations: int = 6
    cv_folds: int = 3
    search_n_jobs: int = -1
    drift_min_batch_size: int = 30
    max_upload_bytes: int = 25_000_000
    max_preview_rows: int = 100

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
