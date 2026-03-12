"""
Main training entry point.

Trains a detection model on the Combined SAR Ship Detection Dataset
and logs everything to DagsHub-hosted MLFlow.

Usage:
    python src/train.py                          # YOLOv8n, default config
    python src/train.py --model yolov8s          # YOLOv8s
    python src/train.py --model rtdetr-l         # RT-DETR
    python src/train.py --model fasterrcnn       # Faster R-CNN (ResNet-50 FPN)
    python src/train.py --epochs 50 --imgsz 1024 # custom settings
    python src/train.py --resume                 # resume most recent interrupted run
    python src/train.py --resume yolov8n_sz1024_ep50_0309_0905  # resume specific YOLO run
    python src/train.py --resume fasterrcnn_sz1024_ep50_0311_0900  # resume Faster R-CNN

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
RUN_ID_FILE = ".mlflow_run_id"


def _mlflow_key(k: str) -> str:
    """Sanitize an Ultralytics metric key for MLFlow (no parens, no prefix)."""
    return k.replace("metrics/", "").replace("(B)", "").replace("-", "_").strip("/")


def _make_epoch_callbacks():
    """Return (on_train_epoch_end, on_fit_epoch_end) that stream metrics to MLFlow."""

    def on_train_epoch_end(trainer) -> None:
        if mlflow.active_run() is None:
            return
        mlflow.log_metrics(
            {
                **{k: float(v) for k, v in trainer.lr.items()},
                **{_mlflow_key(k): float(v)
                   for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items()},
            },
            step=trainer.epoch,
        )

    def on_fit_epoch_end(trainer) -> None:
        if mlflow.active_run() is None:
            return
        mlflow.log_metrics(
            {_mlflow_key(k): float(v)
             for k, v in trainer.metrics.items()
             if k != "fitness"},
            step=trainer.epoch,
        )

    return on_train_epoch_end, on_fit_epoch_end


def _log_run_results(results, run_dir: Path, model_name: str) -> None:
    """Log final metrics, artifacts, and register model after training completes."""
    # Final summary metrics
    metrics = results.results_dict
    mlflow.log_metrics({
        "mAP50":     metrics.get("metrics/mAP50(B)", 0),
        "mAP50_95":  metrics.get("metrics/mAP50-95(B)", 0),
        "precision": metrics.get("metrics/precision(B)", 0),
        "recall":    metrics.get("metrics/recall(B)", 0),
        "box_loss":  metrics.get("val/box_loss", 0),
        "cls_loss":  metrics.get("val/cls_loss", 0),
    })

    # Inference speed on validation set (ms per image)
    if hasattr(results, "speed") and results.speed:
        mlflow.log_metrics({
            "speed_preprocess_ms":  results.speed.get("preprocess", 0),
            "speed_inference_ms":   results.speed.get("inference", 0),
            "speed_postprocess_ms": results.speed.get("postprocess", 0),
        })

    # Artifacts
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

    # Best weights
    best_weights = run_dir / "weights" / "best.pt"
    if best_weights.exists():
        mlflow.log_artifact(str(best_weights), artifact_path="weights")

    # Model registry
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
    from ultralytics import settings as ultralytics_settings

    # Disable Ultralytics' built-in MLFlow callback — we do our own logging
    ultralytics_settings.update({"mlflow": False})

    model = YOLO(f"{model_name}.pt")

    with mlflow.start_run(run_name=run_name):
        # Save run ID to disk so --resume can reopen this run
        run_dir = RUNS_DIR / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / RUN_ID_FILE).write_text(mlflow.active_run().info.run_id)

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

        # Register per-epoch callbacks
        on_train_epoch_end, on_fit_epoch_end = _make_epoch_callbacks()
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

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

        _log_run_results(results, run_dir, model_name)


def resume_yolo(run_dir_name: str) -> None:
    """Resume an interrupted training run, continuing MLFlow logging."""
    from ultralytics import YOLO
    from ultralytics import settings as ultralytics_settings

    ultralytics_settings.update({"mlflow": False})

    run_dir = RUNS_DIR / run_dir_name
    last_pt = run_dir / "weights" / "last.pt"
    run_id_file = run_dir / RUN_ID_FILE

    if not last_pt.exists():
        raise FileNotFoundError(f"No checkpoint found at {last_pt}")
    if not run_id_file.exists():
        raise FileNotFoundError(
            f"No MLFlow run ID file at {run_id_file}. "
            "This run may have been started before resume support was added."
        )

    run_id = run_id_file.read_text().strip()
    # Extract model name from the run directory name (e.g. "yolov8n_sz1024_ep50_...")
    model_name = run_dir_name.split("_sz")[0]

    print(f"\nResuming run '{run_dir_name}' (MLFlow run_id: {run_id}) ...")

    model = YOLO(str(last_pt))

    with mlflow.start_run(run_id=run_id):
        on_train_epoch_end, on_fit_epoch_end = _make_epoch_callbacks()
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        results = model.train(resume=True)

        _log_run_results(results, run_dir, model_name)


def _find_latest_interrupted_run() -> str:
    """Return the name of the most recently modified run dir with a last.pt and run ID file."""
    candidates = [
        d for d in sorted(RUNS_DIR.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
        if d.is_dir()
        and (d / "weights" / "last.pt").exists()
        and (d / RUN_ID_FILE).exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No resumable runs found in {RUNS_DIR}. "
            "A resumable run must have weights/last.pt and .mlflow_run_id."
        )
    return candidates[0].name


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAR ship detector")
    parser.add_argument("--model", default="yolov8n", help="Model name (yolov8n/s/m/l, rtdetr-l, fasterrcnn)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="GPU device id (0) or 'cpu'")
    parser.add_argument(
        "--resume", nargs="?", const="auto", default=None, metavar="RUN_DIR",
        help="Resume interrupted run. Optionally specify run dir name; defaults to most recent."
    )
    args = parser.parse_args()

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found at {DATASET_YAML}. Run: python src/data/preprocess.py"
        )

    setup_mlflow()
    mlflow.set_experiment(EXPERIMENT_NAME)

    if args.resume is not None:
        run_dir_name = (
            _find_latest_interrupted_run() if args.resume == "auto" else args.resume
        )
        if run_dir_name.startswith("fasterrcnn"):
            from src.models.faster_rcnn import resume_faster_rcnn
            resume_faster_rcnn(run_dir_name)
        else:
            resume_yolo(run_dir_name)
    else:
        timestamp = time.strftime("%m%d_%H%M")
        run_name = f"{args.model}_sz{args.imgsz}_ep{args.epochs}_{timestamp}"
        if args.model == "fasterrcnn":
            from src.models.faster_rcnn import train_faster_rcnn
            train_faster_rcnn(
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                run_name=run_name,
            )
        else:
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
