from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

model_name = "BankLoanBestModel"  # <-- Replace with exact model name

versions = client.search_model_versions(f"name='{model_name}'")

best_f1 = 0
best_version = None

for v in versions:
    run = client.get_run(v.run_id)
    f1 = run.data.metrics.get("val_f1", 0)
    print(f"Version {v.version} - val_f1: {f1}")
    if f1 > best_f1:
        best_f1 = f1
        best_version = v.version

print(f"\n✅ Best version: {best_version} with val_f1: {best_f1}")
