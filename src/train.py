"""
Main training entry point.

Trains a detection model on the Combined SAR Ship Detection Dataset
and logs everything to DagsHub-hosted MLFlow.

Usage:
    python src/train.py                          # YOLOv8n, default config
    python src/train.py --model yolov8s          # YOLOv8s
    python src/train.py --model rtdetr-l         # RT-DETR
    python src/train.py --epochs 50 --imgsz 1024 # custom settings

MLFlow experiment structure:
    Experiment: "sar-ship-detection"
    Run name:   "<model>_<imgsz>_ep<epochs>_<timestamp>"
    Tags:       model, dataset, framework
    Params:     all training hyperparameters
    Metrics:    mAP50, mAP50-95, precision, recall (per epoch)
    Artifacts:  confusion matrix, PR curve, sample predictions, weights
"""

import argparse
import os
import time
from pathlib import Path

import mlflow

from src.utils.mlflow_utils import log_system_info, setup_mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "dataset.yaml"
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
EXPERIMENT_NAME = "sar-ship-detection"


def train_yolo(
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    run_name: str,
) -> None:
    """Train a YOLO or RT-DETR model via Ultralytics and log to MLFlow."""
    from ultralytics import YOLO

    model = YOLO(f"{model_name}.pt")

    with mlflow.start_run(run_name=run_name):
        # Tags
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("framework", "ultralytics")
        mlflow.set_tag("dataset", "combined-sar-ship-11k")
        mlflow.set_tag("task", "detection")
        log_system_info()

        # Params
        mlflow.log_params({
            "model": model_name,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "optimizer": "auto",
            "dataset": str(DATASET_YAML),
        })

        print(f"\nTraining {model_name} for {epochs} epochs at {imgsz}px ...")
        results = model.train(
            data=str(DATASET_YAML),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(RUNS_DIR),
            name=run_name,
            exist_ok=True,
            verbose=True,
        )

        # Log final metrics
        metrics = results.results_dict
        mlflow.log_metrics({
            "mAP50":    metrics.get("metrics/mAP50(B)", 0),
            "mAP50_95": metrics.get("metrics/mAP50-95(B)", 0),
            "precision": metrics.get("metrics/precision(B)", 0),
            "recall":    metrics.get("metrics/recall(B)", 0),
            "box_loss":  metrics.get("val/box_loss", 0),
            "cls_loss":  metrics.get("val/cls_loss", 0),
        })

        # Log artifacts
        run_dir = RUNS_DIR / run_name
        for artifact in [
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "PR_curve.png",
            "F1_curve.png",
            "results.png",
            "val_batch0_pred.jpg",
        ]:
            p = run_dir / artifact
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="plots")

        # Log best weights
        best_weights = run_dir / "weights" / "best.pt"
        if best_weights.exists():
            mlflow.log_artifact(str(best_weights), artifact_path="weights")

        # Register model in MLFlow registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/weights/best.pt"
        registry_name = model_name.replace("-", "_").upper()
        try:
            mlflow.register_model(model_uri, registry_name)
            print(f"Model registered as '{registry_name}' in MLFlow registry.")
        except Exception as e:
            print(f"Model registration skipped: {e}")

        map50 = metrics.get("metrics/mAP50(B)", 0)
        print(f"\nRun complete. mAP50={map50:.4f}")
        print(f"MLFlow run: {mlflow.active_run().info.run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAR ship detector")
    parser.add_argument("--model", default="yolov8n", help="Model name (yolov8n/s/m/l, rtdetr-l)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="GPU device id (0) or 'cpu'")
    args = parser.parse_args()

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at {DATASET_YAML}. Run: python src/data/preprocess.py"
        )

    setup_mlflow()
    mlflow.set_experiment(EXPERIMENT_NAME)

    timestamp = time.strftime("%m%d_%H%M")
    run_name = f"{args.model}_sz{args.imgsz}_ep{args.epochs}_{timestamp}"

    train_yolo(
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
