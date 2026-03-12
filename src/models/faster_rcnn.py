"""
Faster R-CNN (ResNet-50 FPN) training for SAR ship detection.

Uses torchvision's Faster R-CNN with a custom anchor configuration tuned
for small ships (~25px median in 1024px images). Logs all metrics and
artifacts to the shared MLFlow experiment via the caller (train.py).

Public API:
    train_faster_rcnn(epochs, imgsz, batch, device, run_name)
    resume_faster_rcnn(run_dir_name)
"""

import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNet50_Weights
from torchvision.ops import MultiScaleRoIAlign

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FASTER_RCNN_RUNS_DIR = PROJECT_ROOT / "runs" / "faster_rcnn"
COCO_TRAIN = PROJECT_ROOT / "data" / "processed" / "coco" / "train.json"
COCO_VAL = PROJECT_ROOT / "data" / "processed" / "coco" / "val.json"
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "COMBINED_SHIP_DETECTION_DATASET" / "COMBINED_SHIP_DETECTION_DATASET"
TRAIN_IMG_DIR = RAW_ROOT / "images" / "train_split"
VAL_IMG_DIR = RAW_ROOT / "images" / "val_split"
RUN_ID_FILE = ".mlflow_run_id"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SARCocoDataset(Dataset):
    """COCO-format dataset for SAR ship detection.

    Returns images as float tensors [3, H, W] in [0, 1] range and target
    dicts compatible with torchvision Faster R-CNN.
    """

    def __init__(self, img_dir: Path, ann_file: Path):
        from pycocotools.coco import COCO
        from PIL import Image
        self._Image = Image
        self.img_dir = img_dir
        self.coco = COCO(str(ann_file))
        self.img_ids = sorted(self.coco.getImgIds())

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        import torchvision.transforms.functional as F
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info["file_name"]

        img = self._Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img)  # [3, H, W], float32 in [0, 1]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        if anns:
            # COCO bbox is [x, y, w, h] — convert to [x1, y1, x2, y2]
            boxes = []
            labels = []
            areas = []
            iscrowd = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])  # already 1-indexed
                areas.append(ann["area"])
                iscrowd.append(ann["iscrowd"])

            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([img_id], dtype=torch.int64),
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "image_id": torch.tensor([img_id], dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64),
            }

        return img_tensor, target


def _collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 2, min_size: int = 800) -> nn.Module:
    """Build Faster R-CNN with small anchors tuned for ~25px SAR ships.

    Anchor sizes span 8–128px across the 5 FPN levels (P2–P5 + pooled),
    which covers ships from ~10px up to ~100px. The default 32–512px range
    would miss most of the small-object detections.
    """
    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )
    backbone = resnet_fpn_backbone("resnet50", weights=ResNet50_Weights.DEFAULT)
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=min_size,
        max_size=1333,
    )
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Run one training epoch. Returns mean loss values for MLFlow logging."""
    model.train()
    accum = {
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_total": 0.0,
    }
    n_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        accum["loss_objectness"] += loss_dict["loss_objectness"].item()
        accum["loss_rpn_box_reg"] += loss_dict["loss_rpn_box_reg"].item()
        accum["loss_classifier"] += loss_dict["loss_classifier"].item()
        accum["loss_box_reg"] += loss_dict["loss_box_reg"].item()
        accum["loss_total"] += total_loss.item()
        n_batches += 1

    if n_batches == 0:
        return accum
    return {k: v / n_batches for k, v in accum.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    coco_gt,
) -> dict[str, float]:
    """Evaluate on val set. Returns mAP50, mAP50_95, speed_inference_ms."""
    from pycocotools.cocoeval import COCOeval
    import copy

    model.eval()
    results = []
    total_time = 0.0
    n_images = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]

            t0 = time.perf_counter()
            outputs = model(images)
            total_time += time.perf_counter() - t0
            n_images += len(images)

            for target, output in zip(targets, outputs):
                img_id = target["image_id"].item()
                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    results.append({
                        "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # back to COCO [x,y,w,h]
                        "score": float(score),
                    })

    speed_ms = (total_time / n_images * 1000) if n_images > 0 else 0.0

    if not results:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "speed_inference_ms": speed_ms}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # stats[0] = mAP@[0.50:0.95], stats[1] = mAP@0.50
    return {
        "mAP50_95": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
        "speed_inference_ms": speed_ms,
    }


# ---------------------------------------------------------------------------
# Per-epoch loop (shared by train and resume)
# ---------------------------------------------------------------------------

def _run_epochs(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    coco_val_gt,
    device: torch.device,
    start_epoch: int,
    total_epochs: int,
    run_dir: Path,
    best_mAP50: float = 0.0,
) -> None:
    """Core epoch loop used by both train_faster_rcnn and resume_faster_rcnn."""
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, total_epochs):
        print(f"\n[Epoch {epoch + 1}/{total_epochs}]")

        # Train
        train_losses = train_one_epoch(model, optimizer, train_loader, device, epoch)
        mlflow.log_metrics(
            {
                "loss_objectness": train_losses["loss_objectness"],
                "loss_rpn_box_reg": train_losses["loss_rpn_box_reg"],
                "loss_classifier": train_losses["loss_classifier"],
                "loss_box_reg": train_losses["loss_box_reg"],
                "loss_total": train_losses["loss_total"],
            },
            step=epoch,
        )
        print(
            f"  losses — total={train_losses['loss_total']:.4f}  "
            f"obj={train_losses['loss_objectness']:.4f}  "
            f"rpn_box={train_losses['loss_rpn_box_reg']:.4f}  "
            f"cls={train_losses['loss_classifier']:.4f}  "
            f"box={train_losses['loss_box_reg']:.4f}"
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, coco_val_gt)
        mlflow.log_metrics(val_metrics, step=epoch)
        print(
            f"  val    — mAP50={val_metrics['mAP50']:.4f}  "
            f"mAP50-95={val_metrics['mAP50_95']:.4f}  "
            f"speed={val_metrics['speed_inference_ms']:.1f}ms/img"
        )

        scheduler.step()

        # Save checkpoint
        ckpt_path = checkpoint_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mAP50": best_mAP50,
            },
            ckpt_path,
        )

        # Save best weights
        if val_metrics["mAP50"] > best_mAP50:
            best_mAP50 = val_metrics["mAP50"]
            torch.save(model.state_dict(), weights_dir / "best.pt")
            print(f"  ** New best mAP50={best_mAP50:.4f} — saved best.pt")

    # Final: log best.pt artifact and register model
    best_pt = weights_dir / "best.pt"
    if best_pt.exists():
        mlflow.log_artifact(str(best_pt), artifact_path="weights")

    mlflow.log_metric("best_mAP50", best_mAP50)

    model_uri = f"runs:/{mlflow.active_run().info.run_id}/weights/best.pt"
    try:
        mlflow.register_model(model_uri, "FASTER_RCNN")
        print("Model registered as 'FASTER_RCNN' in MLFlow registry.")
    except Exception as e:
        print(f"Model registration skipped: {e}")

    print(f"\nTraining complete. Best mAP50={best_mAP50:.4f}")
    print(f"MLFlow run: {mlflow.active_run().info.run_id}")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def train_faster_rcnn(
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    run_name: str,
) -> None:
    """Train Faster R-CNN and log everything to the active MLFlow experiment."""
    from src.utils.mlflow_utils import log_system_info

    if device == "cpu":
        torch_device = torch.device("cpu")
    else:
        torch_device = torch.device(f"cuda:{device}")

    # Datasets
    train_ds = SARCocoDataset(TRAIN_IMG_DIR, COCO_TRAIN)
    val_ds = SARCocoDataset(VAL_IMG_DIR, COCO_VAL)
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=4, collate_fn=_collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=4, collate_fn=_collate_fn, pin_memory=True,
    )

    # Model
    model = build_model(num_classes=2, min_size=imgsz)
    model.to(torch_device)

    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    run_dir = FASTER_RCNN_RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        (run_dir / RUN_ID_FILE).write_text(mlflow.active_run().info.run_id)

        mlflow.set_tag("model", "fasterrcnn")
        mlflow.set_tag("framework", "torchvision")
        mlflow.set_tag("dataset", "combined-sar-ship-11k")
        mlflow.set_tag("task", "detection")
        log_system_info()

        mlflow.log_params({
            "model": "fasterrcnn",
            "backbone": "resnet50-fpn",
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "optimizer": "SGD",
            "lr": 0.005,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "lr_milestones": "30,40",
            "lr_gamma": 0.1,
            "anchor_sizes": "8,16,32,64,128",
            "num_classes": 2,
            "dataset": str(COCO_TRAIN.parent),
        })

        print(f"\nTraining Faster R-CNN for {epochs} epochs at min_size={imgsz}px ...")

        _run_epochs(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            coco_val_gt=val_ds.coco,
            device=torch_device,
            start_epoch=0,
            total_epochs=epochs,
            run_dir=run_dir,
        )


def resume_faster_rcnn(run_dir_name: str) -> None:
    """Resume an interrupted Faster R-CNN training run."""
    from src.utils.mlflow_utils import log_system_info

    run_dir = FASTER_RCNN_RUNS_DIR / run_dir_name
    run_id_file = run_dir / RUN_ID_FILE
    checkpoint_dir = run_dir / "checkpoints"

    if not run_id_file.exists():
        raise FileNotFoundError(f"No MLFlow run ID file at {run_id_file}")

    # Find highest checkpoint
    ckpts = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest_ckpt = ckpts[-1]

    run_id = run_id_file.read_text().strip()
    checkpoint = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
    start_epoch = checkpoint["epoch"] + 1
    best_mAP50 = checkpoint.get("best_mAP50", 0.0)

    # Parse device and imgsz from run dir name
    try:
        imgsz = int(run_dir_name.split("_sz")[1].split("_")[0])
    except (IndexError, ValueError):
        imgsz = 800
        print(f"  WARN: could not parse imgsz from '{run_dir_name}', defaulting to {imgsz}")

    # Parse total epochs from run dir name (e.g. fasterrcnn_sz1024_ep50_...)
    try:
        total_epochs = int(run_dir_name.split("_ep")[1].split("_")[0])
    except (IndexError, ValueError):
        total_epochs = 50
        print(f"  WARN: could not parse total_epochs from '{run_dir_name}', defaulting to {total_epochs}")

    print(f"\nResuming '{run_dir_name}' from epoch {start_epoch + 1}/{total_epochs} "
          f"(MLFlow run_id: {run_id}) ...")

    # Rebuild everything
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SARCocoDataset(TRAIN_IMG_DIR, COCO_TRAIN)
    val_ds = SARCocoDataset(VAL_IMG_DIR, COCO_VAL)
    # Use batch=1 as safe default for resume; real batch was saved in MLFlow params
    batch = 4
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=4, collate_fn=_collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=4, collate_fn=_collate_fn, pin_memory=True,
    )

    model = build_model(num_classes=2, min_size=imgsz)
    model.to(device)
    model.load_state_dict(checkpoint["model"])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    optimizer.load_state_dict(checkpoint["optimizer"])

    scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        # Fast-forward scheduler to the correct step
        for _ in range(start_epoch):
            scheduler.step()

    with mlflow.start_run(run_id=run_id):
        _run_epochs(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            coco_val_gt=val_ds.coco,
            device=device,
            start_epoch=start_epoch,
            total_epochs=total_epochs,
            run_dir=run_dir,
            best_mAP50=best_mAP50,
        )
