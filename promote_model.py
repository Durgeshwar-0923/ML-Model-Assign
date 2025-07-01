import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:8080")
client = MlflowClient()

model_name = "BankLoanBestModel"
version = "1"

client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=version
)

print(f"✅ Model '{model_name}' version {version} is now aliased as 'champion'")
