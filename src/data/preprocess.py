"""
Prepare the Combined SAR Ship Detection Dataset for training.

Source: https://www.kaggle.com/datasets/hewitleoj/combined-sar-ship-detection-dataset-11-8k-images
The raw data is already in YOLO format with pre-made train/val splits.

This script:
  1. Writes a corrected data.yaml with absolute paths for this machine
  2. Converts YOLO labels → COCO JSON (for torchvision Faster R-CNN / Mask R-CNN)

No image copying is done — processed/ just holds the yaml and COCO annotations,
pointing back at the raw images to avoid duplicating 1.5GB of data.

Output structure:
    data/processed/
    ├── dataset.yaml          YOLO dataset config (corrected paths)
    └── coco/
        ├── train.json        COCO annotations for train_split
        └── val.json          COCO annotations for val_split

Usage:
    python src/data/preprocess.py
"""

import json
from pathlib import Path

RAW_ROOT = Path("data/raw/COMBINED_SHIP_DETECTION_DATASET/COMBINED_SHIP_DETECTION_DATASET")
PROCESSED_ROOT = Path("data/processed")
CLASS_NAMES = ["ship"]


# ---------------------------------------------------------------------------
# YOLO label parsing
# ---------------------------------------------------------------------------

def parse_yolo_label(label_path: Path) -> list[dict]:
    """Parse a YOLO .txt label file into a list of box dicts."""
    boxes = []
    text = label_path.read_text().strip()
    if not text:
        return boxes
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append({"class": cls, "cx": cx, "cy": cy, "bw": bw, "bh": bh})
    return boxes


# ---------------------------------------------------------------------------
# YOLO dataset.yaml
# ---------------------------------------------------------------------------

def write_dataset_yaml(out_path: Path) -> None:
    """Write a corrected dataset.yaml pointing to absolute paths on this machine."""
    raw_abs = RAW_ROOT.resolve()
    content = (
        f"# Combined SAR Ship Detection Dataset\n"
        f"# Source: https://www.kaggle.com/datasets/hewitleoj/combined-sar-ship-detection-dataset-11-8k-images\n"
        f"\n"
        f"path: {raw_abs}\n"
        f"train: images/train_split\n"
        f"val: images/val_split\n"
        f"\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content)
    print(f"Wrote dataset.yaml → {out_path}")


# ---------------------------------------------------------------------------
# YOLO → COCO conversion
# ---------------------------------------------------------------------------

def yolo_to_coco(
    img_dir: Path,
    label_dir: Path,
    split_name: str,
    out_path: Path,
) -> None:
    """Convert a YOLO split to a COCO JSON annotation file."""
    categories = [
        {"id": i + 1, "name": n, "supercategory": "object"}
        for i, n in enumerate(CLASS_NAMES)
    ]
    coco: dict = {
        "info": {
            "description": "Combined SAR Ship Detection Dataset",
            "url": "https://www.kaggle.com/datasets/hewitleoj/combined-sar-ship-detection-dataset-11-8k-images",
            "version": "1.0",
        },
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    ann_id = 1

    for img_id, img_path in enumerate(img_files, start=1):
        # Get image dimensions without loading the full image
        from PIL import Image
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })

        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        for box in parse_yolo_label(label_path):
            # YOLO cx/cy/bw/bh → COCO xmin/ymin/w/h (absolute pixels)
            abs_w = box["bw"] * img_w
            abs_h = box["bh"] * img_h
            abs_x = (box["cx"] - box["bw"] / 2) * img_w
            abs_y = (box["cy"] - box["bh"] / 2) * img_h

            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": box["class"] + 1,
                "bbox": [abs_x, abs_y, abs_w, abs_h],
                "area": abs_w * abs_h,
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coco, indent=2))
    n_imgs = len(coco["images"])
    n_ann = len(coco["annotations"])
    print(f"  {split_name}: {n_imgs} images, {n_ann} annotations → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not RAW_ROOT.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_ROOT}. "
            "Download from: https://www.kaggle.com/datasets/hewitleoj/combined-sar-ship-detection-dataset-11-8k-images"
        )

    print(f"Source: {RAW_ROOT}")

    # 1. Corrected dataset.yaml
    write_dataset_yaml(PROCESSED_ROOT / "dataset.yaml")

    # 2. COCO conversion for both splits
    print("\nConverting to COCO format ...")
    splits = {
        "train": ("images/train_split", "labels/train_split"),
        "val": ("images/val_split", "labels/val_split"),
    }
    for split_name, (img_sub, lbl_sub) in splits.items():
        yolo_to_coco(
            img_dir=RAW_ROOT / img_sub,
            label_dir=RAW_ROOT / lbl_sub,
            split_name=split_name,
            out_path=PROCESSED_ROOT / "coco" / f"{split_name}.json",
        )

    print("\nPreprocessing complete.")
    print(f"  YOLO config: {PROCESSED_ROOT / 'dataset.yaml'}")
    print(f"  COCO output: {PROCESSED_ROOT / 'coco/'}")


if __name__ == "__main__":
    main()
