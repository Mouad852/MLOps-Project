# MLOps Portfolio Demo — Implementation Plan

The target is a local, four-service portfolio demo that proves the MLOps lifecycle without pretending to be production infrastructure.

## Target result

A reviewer should be able to clone the repository, run:

```bash
docker compose up --build
```

Then:

1. Open Streamlit.
2. Upload a supplied classification or regression CSV.
3. Profile the dataset.
4. Train and tune four model families.
5. Compare holdout metrics.
6. Inspect every experiment in MLflow.
7. Load the winning model for single or batch predictions.
8. Run a realistic customer-churn case study and inspect its measured results.
9. Compare incoming prediction data with the training distribution for drift.
10. See automated tests and Docker builds passing in GitHub Actions.

### What this will demonstrate

| MLOps concept | Project evidence |
|---|---|
| Data ingestion | CSV upload, validation, preview and profiling |
| Data lineage | Dataset UUID and SHA-256 fingerprint |
| Reproducibility | Fixed seeds and versioned preprocessing pipelines |
| Model development | Classification and regression model comparison |
| Hyperparameter tuning | `RandomizedSearchCV` |
| Experiment tracking | MLflow parameters, metrics and artifacts |
| Model packaging | Complete preprocessing and estimator pipeline |
| Model serving | FastAPI prediction endpoint |
| Real ML use case | Customer-churn case study with class-aware evaluation |
| Post-deployment monitoring | Statistical drift reports for incoming prediction batches |
| Artifact persistence | MLflow artifact store and Docker volumes |
| Orchestration | Docker Compose |
| Quality automation | pytest and GitHub Actions |
| Basic monitoring | Optional prediction usage statistics |

## Explicit non-goals

Do not add these merely to make the repository look bigger:

- Kubernetes
- Cloud deployment
- Redis, Celery or Kafka
- Feature stores
- DVC
- Prometheus/Grafana
- Authentication systems
- Canary deployments
- Distributed training
- A full MLflow Model Registry workflow

FastAPI explicitly notes that heavy durable workloads need a real job queue, but its in-process `BackgroundTasks` is acceptable when documented as a demo limitation. See [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/).

---

## Phase 0 — Lock the design and APIs

Use the versions already pinned in the repository:

- Python 3.11
- scikit-learn 1.5.2
- XGBoost 2.1.1
- MLflow 2.17.2
- FastAPI 0.115
- SQLAlchemy 2.0
- Pydantic 2

The implementation should stay within these documented APIs:

- `Pipeline`, `ColumnTransformer`, `RandomizedSearchCV`
- `train_test_split`, classification/regression metrics
- `permutation_importance`
- `mlflow.start_run`, `log_params`, `log_metrics`, `log_dict`
- `mlflow.sklearn.log_model`, `load_model`
- `infer_signature`
- FastAPI `BackgroundTasks`, `UploadFile`, `Depends`, `TestClient`
- SQLAlchemy `Uuid`, `ForeignKey`, `Session.get`
- Docker Compose health checks, named volumes and `depends_on`

References: [scikit-learn Pipeline](https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html), [RandomizedSearchCV](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.RandomizedSearchCV.html), [MLflow sklearn API](https://mlflow.org/docs/2.17.0/python_api/mlflow.sklearn.html).

---

## Phase 1 — Repair the application foundation

### Implement

1. Update `backend/app/core/config.py`:

   - Replace Pydantic `class Config` with `SettingsConfigDict`.
   - Use SQLite as the no-Docker default: `sqlite:///./mlops.db`.
   - Let Compose override it with PostgreSQL.
   - Add settings for tuning iterations and CV folds.

2. Update `backend/app/schemas/dataset.py`:

   - Replace deprecated configuration with `ConfigDict(from_attributes=True)`.
   - Define task type as `Literal["classification", "regression"]`.

3. Update `backend/app/models/dataset.py`:

   - Replace PostgreSQL-only `UUID` with portable `sqlalchemy.Uuid`.
   - Add a foreign key from `TrainingRun.dataset_id` to `datasets.id`.
   - Add `dataset_hash`.
   - Add `target_labels` for decoding classification predictions.
   - Add `reference_profile` for the training-data distributions used by drift monitoring.
   - Add indexes to frequently queried IDs.

4. Improve dataset upload:

   - Reject non-CSV files.
   - Handle malformed and empty CSV files cleanly.
   - Compute a SHA-256 fingerprint after saving.
   - Put an upper bound on preview rows.
   - Keep uploads deliberately limited to small demo datasets.

5. Rename `backend/test` to `backend/tests`.

6. Add `backend/pytest.ini` with `testpaths = tests`.

References: [Pydantic model configuration](https://docs.pydantic.dev/latest/api/config/), [SQLAlchemy portable `Uuid`](https://docs.sqlalchemy.org/en/20/core/type_basics.html).

### Verification

- Backend imports without Pydantic warnings.
- SQLite tables are created successfully.
- Valid CSV upload returns metadata and a hash.
- Invalid task types return HTTP 422.
- Training runs cannot reference a nonexistent dataset.

### Guardrails

- Do not add Alembic for this demo.
- Do not pass SQLAlchemy sessions into background tasks.
- Do not trust MIME type alone for validating CSVs.

---

## Phase 2 — Implement the actual training engine

Completely replace the duplicated contents of `backend/app/ml/pipelines/training.py`.

### Required functions

Keep the module readable by separating it into functions such as:

```text
get_candidate_models()
get_parameter_spaces()
validate_training_data()
evaluate_classification()
evaluate_regression()
calculate_feature_importance()
train_and_compare()
```

### Candidate models

Classification:

- Logistic regression
- Decision tree classifier
- Random forest classifier
- XGBoost classifier

Regression:

- Linear regression
- Decision tree regressor
- Random forest regressor
- XGBoost regressor

Every stochastic estimator should use `random_state=42`.

### Training flow

For every model:

1. Remove rows with missing targets.
2. Separate features and target.
3. Encode classification labels and retain the original classes.
4. Split data into 80% training and 20% holdout.
5. Stratify classification splits.
6. Build one complete pipeline: `preprocessing -> estimator`.
7. Run a small `RandomizedSearchCV`:

   - `n_iter=6`
   - `cv=3`, reduced for very small class counts
   - `n_jobs=-1`
   - `random_state=42`
   - `error_score="raise"`

8. Evaluate the best estimator on the untouched holdout set.

Classification metrics:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1
- Binary ROC-AUC when probability estimates are available
- Confusion-matrix values for binary classification

Choose the best binary classification model by maximum ROC-AUC and the best multiclass model by maximum weighted F1. Keep F1, precision and recall visible so an apparently strong accuracy score cannot hide poor minority-class performance.

Regression metrics:

- RMSE
- MAE
- R-squared

Choose the best regression model by minimum RMSE.

9. Calculate model-agnostic permutation importance on the holdout set.

10. Build and return a compact reference profile of the training features for later drift comparison:

    - Numeric feature quantile boundaries and bin proportions
    - Categorical feature frequency distributions
    - Missing-value rates
    - Training row count

11. Return the exact structure expected by the API:

```python
{
    "best_model": "...",
    "feature_columns": [...],
    "target_labels": [...],
    "reference_profile": {...},
    "results": {
        "model_name": {
            "pipeline": fitted_pipeline,
            "best_params": {...},
            "metrics": {...},
            "feature_importance": {...},
            "input_example": ...,
        }
    },
}
```

References: [scikit-learn model evaluation](https://scikit-learn.org/1.5/modules/model_evaluation.html), [permutation importance](https://scikit-learn.org/1.5/modules/permutation_importance.html), [XGBoost sklearn API](https://xgboost.readthedocs.io/en/release_2.1.0/python/python_api.html).

### Verification

- A small Iris-style dataset completes classification training.
- A small regression dataset completes regression training.
- Every returned model can predict directly from raw feature rows.
- Unseen categorical values do not crash prediction.
- Original string classification labels are restored.
- Running twice with the same data gives equivalent results.
- Binary classification reports ROC-AUC and confusion-matrix values.
- A reusable reference profile is produced without storing the entire training dataset in the model record.

### Guardrails

- Never preprocess before the train/test split.
- Never call `pd.get_dummies` separately at inference time.
- Use `model__parameter` names inside search spaces.
- Do not give both XGBoost and `RandomizedSearchCV` full parallelism; set XGBoost `n_jobs=1`.
- Use `root_mean_squared_error`, not the deprecated `squared=False` pattern.

---

## Phase 3 — Complete MLflow tracking and prediction

Update `backend/app/api/training.py`.

### Implement

For each candidate model, log:

- Dataset ID and dataset hash
- Task type
- Model name
- Random seed
- Best hyperparameters
- Holdout metrics
- Feature importance JSON
- Training-data reference profile
- Complete fitted pipeline
- Model signature
- Input example

Use:

```python
signature = infer_signature(input_example, predictions)
```

Then log with:

```python
mlflow.sklearn.log_model(
    pipeline,
    artifact_path="model",
    signature=signature,
    input_example=input_example,
)
```

Store the winning:

- MLflow run ID
- `runs:/.../model` URI
- Model name
- Metrics
- Feature columns
- Label mapping
- Training-data reference profile for drift monitoring

Improve background execution:

- Use its own context-managed database session.
- Roll back before recording a failure.
- Set status through `pending -> running -> completed/failed`.
- Return readable validation failures.
- Keep raw stack traces out of API responses.

Prediction should:

1. Require the API key.
2. Validate required columns.
3. Order the input columns correctly.
4. Load the winning pipeline from MLflow.
5. Decode classification labels.
6. Return JSON-safe values.

References: [MLflow experiment tracking](https://mlflow.org/docs/2.17.0/tracking.html), [MLflow model signatures](https://mlflow.org/docs/2.17.0/python_api/mlflow.models.html).

### Verification

- Four MLflow runs appear per training request.
- Every run contains parameters, metrics and a model artifact.
- The winning URI loads after a backend restart.
- Single and batch predictions work.
- Failed training leaves a `failed` database record with a useful message.

---

## Phase 4 — Finish the frontend experience

Preserve the existing visual design in `frontend/app.py`.

### Implement

- Add timeouts and connection-error handling to every backend request.
- Fix metric direction:
  - Highlight maximum F1, accuracy and R-squared.
  - Highlight minimum RMSE and MAE.
- Show:
  - Dataset fingerprint
  - Training run ID
  - MLflow link
  - Winning model
  - Model comparison table
  - Feature importance
  - Clear failure messages
- Allow users to reload an existing dataset and completed run after refreshing Streamlit.
- Use numeric inputs for numeric features.
- Keep text inputs for categorical features.
- Separate internal and public URLs:

  ```text
  API_URL=http://backend:8000
  PUBLIC_API_URL=http://localhost:8000
  MLFLOW_UI_URL=http://localhost:5000
  ```

This matters because Streamlit uses the internal URL server-side, while browser links need the public URL.

### Verification

- Refreshing the page does not make stored runs inaccessible.
- Both single-record and batch prediction work.
- API documentation and MLflow links open from the browser.
- Backend downtime displays a friendly error instead of a traceback.

### Guardrails

- Do not rewrite the frontend framework.
- Do not add user accounts.
- Do not let frontend presentation logic decide which model is best.

---

## Phase 5 — Add one-command Docker orchestration

Add:

```text
compose.yaml
.dockerignore
```

Compose services:

- `db`: PostgreSQL 16
- `mlflow`: MLflow 2.17.2 with SQLite metadata and artifact storage
- `backend`
- `frontend`

Named volumes:

- `postgres_data`
- `mlflow_data`
- `uploads`

Use health checks and wait for healthy dependencies. Container-to-container addresses must use service names:

```text
postgresql://mlops:mlops@db:5432/mlops
http://mlflow:5000
http://backend:8000
```

Expose:

- Streamlit: `8501`
- FastAPI: `8000`
- MLflow: `5000`

Update `.env.example` with demo credentials and ensure the same API key reaches backend and frontend.

References: [Docker Compose startup order](https://docs.docker.com/compose/how-tos/startup-order/), [MLflow tracking server](https://mlflow.org/docs/2.17.0/tracking/server.html).

### Verification

```bash
docker compose config
docker compose up --build -d
docker compose ps
```

All four services should become healthy. Data, uploads and MLflow runs should survive a normal `docker compose down` and restart.

### Guardrails

- Never use `localhost` for container-to-container communication.
- Do not use unpinned `latest` images.
- Do not store persistent data only in container writable layers.
- Reserve `docker compose down -v` for an explicit full reset.

---

## Phase 6 — Build a meaningful test suite and CI

### Tests to add

Backend/API:

- Health endpoint
- Valid CSV upload
- Invalid extension
- Malformed/empty CSV
- Dataset preview
- Invalid dataset ID
- Invalid target column
- Invalid task type
- Training run status transitions
- Prediction with valid API key
- Prediction with invalid API key
- Prediction with missing columns

ML:

- Numeric and categorical preprocessing
- Missing-value handling
- Classification training
- Regression training
- Reproducibility
- Feature importance
- Original classification label restoration

Use:

- Temporary SQLite databases
- Temporary upload folders
- FastAPI dependency overrides
- MLflow mocks for most API tests
- One real Compose-based MLflow smoke test documented for manual execution

Update GitHub Actions to:

1. Install Python 3.11 and cache pip.
2. Install backend dependencies.
3. Run `pytest tests -q`.
4. Validate `docker compose config`.
5. Build backend and frontend images on pull requests and `main`.

References: [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/), [GitHub Actions Python testing](https://docs.github.com/en/actions/tutorials/build-and-test-code/python).

### Definition of done

- Tests are isolated and order-independent.
- CI catches broken imports.
- CI catches broken Dockerfiles.
- ML tests use reduced search iterations so the suite remains fast.

---

## Phase 7 — Add a realistic ML case study and make it portfolio-ready

### Primary use case: customer churn prediction

The headline demonstration should be a realistic binary classification problem, not Iris. Use a customer-churn dataset with fields such as tenure, contract type, monthly charges, payment method and service subscriptions, with `Churn` as the target.

- Prefer a public dataset such as IBM Telco Customer Churn.
- Document the source, license and data dictionary in the README.
- Either commit a permitted, cleaned demo copy or provide a deterministic download/preparation script.
- Keep personally identifiable information out of the repository.
- Retain Iris and Diabetes only as small automated-test or smoke-test datasets.

Suggested data layout:

```text
data/samples/customer_churn.csv
data/samples/incoming_churn_stable.csv
data/samples/incoming_churn_shifted.csv
data/samples/diabetes.csv
```

If the source dataset cannot be redistributed, add `scripts/prepare_customer_churn.py` and clear download instructions instead of committing it.

### Churn case-study requirements

1. Explain the business question: identify customers at risk of leaving so retention action can be prioritized.
2. Report dataset size, churn rate, missing values and important feature distributions.
3. Compare the trained models with a majority-class `DummyClassifier` baseline.
4. Report metrics that are appropriate for imbalanced classification:

   - ROC-AUC
   - F1
   - Precision
   - Recall
   - Accuracy for context only
   - Confusion matrix

5. Explain the precision/recall tradeoff instead of claiming that accuracy alone defines the best model.
6. Show the winning model's feature importance and discuss two or three plausible drivers.
7. Include a README results table populated from an actual completed MLflow run:

   ```text
   | Model | ROC-AUC | F1 | Precision | Recall | Accuracy |
   |---|---:|---:|---:|---:|---:|
   | Actual measured results copied from MLflow — never invented |
   ```

8. Record the dataset hash, random seed and library versions alongside the reported result so it can be reproduced.

### README requirements

Add a strong `README.md` containing:

1. One-sentence project description.
2. Screenshot or GIF.
3. Customer-churn problem statement and dataset source.
4. Actual evaluation results and interpretation.
5. MLOps concepts demonstrated.
6. Mermaid architecture diagram.
7. One-command quick start.
8. URLs for Streamlit, FastAPI docs and MLflow.
9. Two-minute demonstration walkthrough.
10. API endpoint table.
11. Testing instructions.
12. Repository structure.
13. Intentional limitations.
14. Small future-improvements section.

### Recommended demo sequence

1. Show the passing CI workflow.
2. Start Compose.
3. Upload the customer-churn dataset.
4. Show its class balance, data profile and dataset fingerprint.
5. Train classification models and compare them with the baseline.
6. Discuss the measured ROC-AUC, F1, precision, recall and confusion matrix.
7. Open MLflow and show parameters, metrics, artifacts and the model signature.
8. Make single and batch predictions.
9. Submit the stable and deliberately shifted incoming batches and show the drift report changing.
10. Restart the stack and demonstrate persistence.
11. Optionally repeat briefly with the regression sample.

### Verification

A reviewer following only the README should complete the demo in under ten minutes. Every metric displayed in the README must match a stored MLflow run; do not use placeholder or fabricated scores in the finished project.

---

## Phase 8 — Add prediction monitoring and basic data drift detection

Implement this with small, transparent statistical functions rather than adding a large monitoring stack. This keeps the demo explainable and avoids making Evidently a required runtime dependency.

### Drift implementation

Add `backend/app/ml/drift.py` with two main operations:

```text
build_reference_profile(training_features)
compare_with_reference(incoming_features, reference_profile)
```

For numeric features:

- Store quantile-based bin boundaries and training proportions.
- Calculate Population Stability Index (PSI) using those fixed training bins.
- Compare missing-value rates.
- Suggested interpretation: below `0.10` stable, `0.10–0.25` warning, above `0.25` drifted.

For categorical features:

- Store training category proportions.
- Calculate total variation distance between training and incoming proportions.
- Calculate unseen-category rate.
- Compare missing-value rates.

The overall report should include:

- Status: `stable`, `warning`, `drifted` or `insufficient_data`
- Incoming row count
- Number and percentage of drifted features
- Per-feature method, score, threshold and status
- Missing-value-rate changes
- Timestamp and training run ID

Require a configurable minimum batch size, such as 30 rows, before making a drift claim. A single prediction should be logged but reported as insufficient data for drift analysis.

### Monitoring persistence

Add a `PredictionEvent` table storing only:

- Training run ID
- Request time
- Number of records
- Prediction latency
- Success/failure
- Overall drift status
- Number of drifted features
- Drift report JSON

Do not store raw customer records in monitoring tables. Persist aggregate distributions and drift results only.

Expose `GET /training/{run_id}/monitoring` and show:

- Total prediction requests
- Total predicted rows
- Average latency
- Most recent prediction time
- Most recent drift status
- Drifted-feature count
- Per-feature drift chart or table

The batch prediction flow should automatically run the drift comparison against the winning run's saved training reference profile. The frontend should display the report alongside the predictions.

### Drift demonstration data

Supply two incoming churn batches:

- A stable batch sampled from a distribution similar to training data.
- A deliberately shifted batch with changes such as higher monthly charges, shorter tenure or altered contract proportions.

The first should normally report stable while the second should flag the intentionally shifted features.

### Verification

- Identical or similarly sampled data reports stable.
- Deliberately shifted numeric data crosses the configured PSI threshold.
- Deliberately shifted categorical proportions are detected.
- Previously unseen categories are reported.
- Small batches return `insufficient_data` rather than a misleading drift decision.
- Drift reports persist and remain visible after restart.
- Unit tests cover stable, warning, drifted and insufficient-data cases.

### Guardrails

- Document thresholds as configurable heuristics, not universal scientific laws.
- Never compare incoming bins calculated independently from training bins.
- Do not call this model-performance monitoring.
- This detects feature-distribution or covariate drift. It cannot detect concept drift or accuracy degradation without delayed ground-truth labels.
- Do not store raw prediction payloads solely for the monitoring demo.

---

## Recommended execution order and effort

| Phase | Estimate |
|---|---:|
| Foundation and validation | 2–3 hours |
| Training engine | 6–8 hours |
| MLflow/API integration | 4–5 hours |
| Frontend completion | 3–4 hours |
| Docker Compose | 2–3 hours |
| Tests and CI | 4–6 hours |
| Real churn case study, README and demo assets | 4–6 hours |
| Prediction monitoring and data drift | 4–6 hours |

Core completion: approximately **29–39 focused hours**.

## Final acceptance checklist

- [ ] `docker compose up --build` starts all services.
- [ ] Classification and regression both work.
- [ ] Four models are tuned and compared.
- [ ] Preprocessing is packaged inside every model.
- [ ] MLflow records parameters, metrics, importance, signatures and artifacts.
- [ ] The winning artifact reloads for prediction.
- [ ] Single and batch predictions work.
- [ ] The customer-churn case study reports real metrics from a traceable MLflow run.
- [ ] The churn model is compared with a simple baseline and evaluated beyond accuracy.
- [ ] Incoming batches are compared with the saved training reference profile.
- [ ] Stable, shifted and insufficient-data drift cases behave as documented.
- [ ] Drift reports and prediction usage statistics survive restarts.
- [ ] Data and runs survive restarts.
- [ ] Tests pass locally and in GitHub Actions.
- [ ] A fresh user can reproduce the demonstration from the README.
- [ ] The README clearly states the demo's limitations.
