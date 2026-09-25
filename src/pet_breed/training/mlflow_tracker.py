"""
MLflow experiment tracking and model registry interface.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from pet_breed.config import CHECKPOINTS_DIR, PROJECT_ROOT

EXPERIMENT_NAME = "pet-breed-classification"


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """Initialize MLflow tracking URI and experiment."""
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

    if tracking_uri is None:
        db_path = (PROJECT_ROOT / "mlflow.db").as_posix()
        tracking_uri = f"sqlite:///{db_path}"

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        exp_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=(PROJECT_ROOT / "mlruns").as_uri(),
        )
    else:
        exp_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return exp_id


def log_run_to_mlflow(
    run_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model: Optional[Any] = None,
    artifacts: Optional[Dict[str, Path]] = None,
    register_model_name: Optional[str] = None,
) -> str:
    """Log parameters, metrics, artifacts, and optionally register model in MLflow."""
    setup_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        # Log Hyperparameters
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log Metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log Artifacts
        if artifacts:
            for art_name, art_path in artifacts.items():
                if Path(art_path).exists():
                    if Path(art_path).is_file():
                        mlflow.log_artifact(str(art_path))
                    elif Path(art_path).is_dir():
                        mlflow.log_artifacts(str(art_path), artifact_path=art_name)

        # Log PyTorch Model
        if model is not None:
            try:
                import torch
                dummy_input = torch.randn(1, 3, 224, 224)
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    registered_model_name=register_model_name,
                    input_example=dummy_input,
                )
            except Exception as e:
                print(f"MLflow model log notice: {e}")
                try:
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path="model",
                        registered_model_name=register_model_name,
                        serialization_format="cloudpickle",
                    )
                except Exception as ex:
                    print(f"Fallback model log: {ex}")

        run_id = run.info.run_id
        print(f"Logged run '{run_name}' to MLflow (Run ID: {run_id})")
        return run_id


def promote_model_to_production(
    model_name: str = "PetBreedClassifier",
    stage: str = "Production",
) -> None:
    """Promote the latest registered model version to Production stage."""
    client = MlflowClient()
    try:
        latest_versions = client.get_latest_versions(model_name)
        if latest_versions:
            latest_version = latest_versions[-1].version
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version,
                stage=stage,
                archive_existing_versions=True,
            )
            print(f"Model '{model_name}' version {latest_version} promoted to '{stage}'.")
    except Exception as e:
        print(f"Notice: Model promotion in MLflow registry: {e}")
