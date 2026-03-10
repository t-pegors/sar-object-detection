"""
Backfill inference speed metrics into existing MLFlow runs.

Loads best.pt from each completed run, runs validation to measure
inference speed, and logs speed_preprocess_ms / speed_inference_ms /
speed_postprocess_ms to the existing MLFlow run.

Usage:
    # Backfill all runs that have a .mlflow_run_id file
    python src/utils/backfill_speed.py

    # Backfill specific runs
    python src/utils/backfill_speed.py yolov8n_sz1024_ep50_0309_0912 yolov8s_sz1024_ep50_0309_1301
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"
DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "dataset.yaml"
RUN_ID_FILE = ".mlflow_run_id"

sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.mlflow_utils import setup_mlflow
import mlflow


def backfill_run(run_dir: Path) -> None:
    from ultralytics import YOLO
    from ultralytics import settings as ultralytics_settings

    ultralytics_settings.update({"mlflow": False})

    best_pt = run_dir / "weights" / "best.pt"
    run_id_file = run_dir / RUN_ID_FILE

    if not best_pt.exists():
        print(f"  SKIP {run_dir.name} — no best.pt found")
        return
    if not run_id_file.exists():
        print(f"  SKIP {run_dir.name} — no .mlflow_run_id file")
        return

    run_id = run_id_file.read_text().strip()

    # Parse imgsz from run directory name (e.g. yolov8n_sz1024_ep50_...)
    try:
        imgsz = int(run_dir.name.split("_sz")[1].split("_")[0])
    except (IndexError, ValueError):
        imgsz = 1024
        print(f"  WARN {run_dir.name} — could not parse imgsz, defaulting to {imgsz}")

    print(f"  Processing {run_dir.name} (run_id={run_id}, imgsz={imgsz}) ...")

    model = YOLO(str(best_pt))
    val_results = model.val(
        data=str(DATASET_YAML),
        imgsz=imgsz,
        verbose=False,
    )

    speed = val_results.speed
    metrics = {
        "speed_preprocess_ms":  speed.get("preprocess", 0),
        "speed_inference_ms":   speed.get("inference", 0),
        "speed_postprocess_ms": speed.get("postprocess", 0),
    }
    print(f"    preprocess={metrics['speed_preprocess_ms']:.2f}ms  "
          f"inference={metrics['speed_inference_ms']:.2f}ms  "
          f"postprocess={metrics['speed_postprocess_ms']:.2f}ms")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

    print(f"    Logged to MLFlow run {run_id}")


def find_all_runs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [d for d in RUNS_DIR.iterdir()
         if d.is_dir() and (d / RUN_ID_FILE).exists() and (d / "weights" / "best.pt").exists()],
        key=lambda d: d.stat().st_mtime,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill inference speed into MLFlow runs")
    parser.add_argument("runs", nargs="*", metavar="RUN_DIR",
                        help="Run directory names to backfill (default: all)")
    args = parser.parse_args()

    setup_mlflow()
    mlflow.set_experiment("sar-ship-detection")

    if args.runs:
        run_dirs = [RUNS_DIR / name for name in args.runs]
    else:
        run_dirs = find_all_runs()
        if not run_dirs:
            print(f"No completed runs found in {RUNS_DIR}")
            return
        print(f"Found {len(run_dirs)} run(s) to backfill:")
        for d in run_dirs:
            print(f"  {d.name}")
        print()

    for run_dir in run_dirs:
        backfill_run(run_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
