import json
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent
RESULT_FILES = [
    "results/baseline_qwen.json",
    "results/reranking_qwen.json",
    "results/abstention_qwen.json",
    "results/final_reranking_abstention.json",
]

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("RBI-RAG-Evaluation")

for relative_path in RESULT_FILES:
    result_path = ROOT / relative_path

    if not result_path.exists():
        print(f"SKIPPED: {relative_path} not found")
        continue

    data = json.loads(result_path.read_text())
    metrics = data.get("metrics", {})
    run_name = result_path.stem

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("result_file", relative_path)
        mlflow.log_param(
            "evaluation_mode",
            metrics.get("evaluation_mode", "unknown"),
        )
        mlflow.log_param(
            "reranking_enabled",
            "true" if "reranking" in run_name else "unknown",
        )

        def log_numeric_metrics(values, prefix=""):
            for key, value in values.items():
                name = f"{prefix}{key}"

                if isinstance(value, dict):
                    log_numeric_metrics(value, f"{name}_")
                elif (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    mlflow.log_metric(name, value)
        
        log_numeric_metrics(metrics)
        mlflow.log_artifact(
            str(result_path),
            artifact_path="evaluation_results",
        )

    print(f"Logged: {run_name}")

print("\nMLflow tracking complete.")
print(f"Tracking directory: {ROOT / 'mlruns'}")
