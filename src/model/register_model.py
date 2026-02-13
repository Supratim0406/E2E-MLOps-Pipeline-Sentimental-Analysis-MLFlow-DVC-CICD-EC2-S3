# register_model.py
import os
import json
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from src.logger import logging
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# Initialize DagsHub MLflow tracking
# -------------------------------------------------------------

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Supratim0406"
repo_name = "E2E-MLOps-Pipeline-Sentimental-Analysis-MLFlow-DVC-CICD-EC2-S3"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# dagshub.init(
#     repo_owner="Supratim0406",
#     repo_name="E2E-MLOps-Pipeline-Sentimental-Analysis-MLFlow-DVC-CICD-EC2-S3",
#     mlflow=True
# )
# -------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load run_id from experiment_info.json"""
    try:
        with open(file_path, "r") as f:
            model_info = json.load(f)
        logging.info("Model info loaded from %s", file_path)
        return model_info
    except Exception as e:
        logging.error("Failed to load model info: %s", e)
        raise


def register_model(model_name: str, model_info: dict):
    try:
        run_id = model_info["run_id"]

        # Must match artifact_path used in log_model()
        model_uri = f"runs:/{run_id}/model"
        print("Registering model from:", model_uri)

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        client = MlflowClient()

        # Move to Staging
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        print(
            f"Model {model_name} v{model_version.version} moved to Staging."
        )

    except Exception as e:
        logging.error("Model registration failed: %s", e)
        raise


def main():
    try:
        model_info = load_model_info("reports/experiment_info.json")

        register_model(
            model_name="final_model",
            model_info=model_info
        )

    except Exception as e:
        logging.error("Registration pipeline failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
