# SAR Object Detection — Claude Project Context

MLOps portfolio project: comparing detection architectures on SAR ship imagery.
All experiments tracked in DagsHub-hosted MLFlow. GitHub-publishable.

## Key Commands

```bash
# Activate environment (always required)
source .env                               # load credentials
export PYTHONPATH=$(pwd)

# Training (preferred — handles env and PYTHONPATH automatically)
./train.sh --model yolov8n --epochs 50 --imgsz 1024 --batch 8
./train.sh --model yolov8s --epochs 50 --imgsz 1024 --batch 8
./train.sh --model rtdetr-l --epochs 50 --imgsz 640  --batch 8
./train.sh --model yolov8n --epochs 3   --imgsz 640  --batch 16   # quick smoke test

# Preprocessing (run once after raw data is present)
.venv/bin/python3.12 src/data/preprocess.py

# DVC
.venv/bin/dvc push    # push data to S3 after adding/changing datasets
.venv/bin/dvc pull    # restore data on a new machine

# Jupyter
.venv/bin/jupyter lab notebooks/

# Tests
.venv/bin/pytest tests/
```

## Python Environment

- Venv: `.venv/` — always use `.venv/bin/python3.12` explicitly (NOT `python` or `python3`)
- Credentials: sourced from `.env` (never committed — gitignored)
- Run scripts from project root `/home/saltqueen/Projects/sar-object-detection/`

## Dataset

**Combined SAR Ship Detection Dataset** (Kaggle, 11.8K images)
- Source: https://www.kaggle.com/datasets/hewitleoj/combined-sar-ship-detection-dataset-11-8k-images
- Raw: `data/raw/COMBINED_SHIP_DETECTION_DATASET/COMBINED_SHIP_DETECTION_DATASET/`
- Splits (pre-made): `images/train_split` (9,481) / `images/val_split` (2,371)
- Format: YOLO (already, no XML parsing needed)
- YOLO config: `data/processed/dataset.yaml`
- COCO annotations: `data/processed/coco/train.json`, `val.json`
- DVC-tracked and pushed to S3 `sar-dvc` bucket

**Critical dataset facts:**
- Images: 1024×1024, RGB (SAR stored as pseudo-color — 3 channels)
- Single class: `ship` (class 0)
- Ships are extremely small: median ~25.6×25.0 px in a 1024px image (~0.17% of image area; bottom quartile ~14×16 px)
- No empty labels — every image has at least one ship

## MLFlow / DagsHub

- Tracking URI: `https://dagshub.com/t-pegors/sar-object-detection.mlflow`
- Experiment name: `sar-ship-detection`
- All runs log: hyperparams, per-epoch metrics, confusion matrix, PR curve, best weights
- Model registry: models promoted through `None → Staging → Production`

## Infrastructure

| Component | Details |
|---|---|
| GPU | RTX 5060 Ti 16GB VRAM (Blackwell — needs CUDA 12.8+, PyTorch 2.6+) |
| DVC remote | S3 `sar-dvc` bucket, `us-east-2`, IAM user `sar-detection-bot` |
| MLFlow | DagsHub hosted (publicly browsable) |
| Framework | Ultralytics (YOLO, RT-DETR) + torchvision (Faster R-CNN, Mask R-CNN) |

## Project Structure

```
src/
  train.py              main training entry point (all models)
  data/
    download.py         dataset download helper
    preprocess.py       YOLO-format input → COCO JSON + fixed dataset.yaml
  models/               model wrappers (to be built in Phase 2)
  utils/
    mlflow_utils.py     setup_mlflow() and log_system_info() helpers
notebooks/
  01_eda_ssdd.ipynb     EDA: image viewer, stats, intensity, bbox analysis
data/
  raw/                  DVC-tracked, gitignored
  processed/            dataset.yaml + COCO JSON (gitignored)
runs/                   Ultralytics training outputs (gitignored)
```

## Model Roadmap

| Phase | Model | Framework | Status |
|---|---|---|---|
| 2 | YOLOv8 n/s/m/l | Ultralytics | next |
| 2 | RT-DETR-L | Ultralytics | next |
| 2 | Faster R-CNN ResNet-50 | torchvision | next |
| 3 | YOLOv8-seg | Ultralytics | planned |
| 3 | Mask R-CNN | torchvision | planned |

## Modeling Notes

- Prefer `imgsz=1024` (native resolution) — small objects need full resolution
- `batch=8` fits comfortably in 16GB VRAM at 1024px
- Log-transform normalization experiment planned — compare vs linear in MLFlow
- Speckle noise augmentation planned for Phase 2 ablation studies
- MLFlow experiment `sar-ship-detection` is the single experiment for all runs

## Phase Status

- [x] Phase 1: Environment, data pipeline, first MLFlow run ← COMPLETE
- [ ] Phase 2: Model comparison (YOLOv8 sizes, RT-DETR, Faster R-CNN)
- [ ] Phase 3: MLOps hardening (MLproject, Docker, CI)
- [ ] Phase 4: Segmentation
- [ ] Phase 5: Polish and publish
