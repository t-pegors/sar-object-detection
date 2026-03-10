# Model Reference & Evaluation Metrics

Reference document for all models tested in the SAR ship detection experiment.
Updated as each model phase completes.

---

## Evaluation Metrics

All metrics are logged to MLFlow under the experiment `sar-ship-detection`.

### Intersection over Union (IoU)

IoU measures the overlap between a predicted bounding box and the ground-truth box:

```
IoU = Area of Overlap / Area of Union
```

A prediction is counted as a true positive only when IoU ≥ some threshold. The choice of threshold determines how strict "correct" is — a box that is roughly right vs one that must be tightly aligned.

---

### mAP50 — Mean Average Precision at IoU ≥ 0.50

**MLFlow key:** `metrics/mAP50(B)`

The most common detection metric. For each class:
1. Sort all predictions by confidence score (descending)
2. At each threshold, compute precision and recall
3. Compute the area under the precision-recall curve → that is Average Precision (AP)
4. Average AP across all classes → mAP

At IoU = 0.50 a predicted box just needs to cover ≥ 50% of the ground truth. This is a relatively forgiving threshold — a box that is somewhat offset but approximately correct will still count. Good for comparing models at a coarse level.

For this dataset (single class: `ship`), mAP50 = AP50 for the ship class.

---

### mAP50-95 — Mean Average Precision, COCO Standard

**MLFlow key:** `metrics/mAP50-95(B)`

The primary COCO benchmark metric. Computes mAP at each IoU threshold from 0.50 to 0.95 in steps of 0.05, then averages:

```
mAP50-95 = mean(mAP@0.50, mAP@0.55, mAP@0.60, ..., mAP@0.95)
```

This is significantly harder than mAP50. A model must localize boxes precisely to score well here — a rough box that passes at 0.50 may fail at 0.75+. This metric penalizes models that detect the right region but draw sloppy boxes.

**Interpretation:** mAP50-95 is the honest test of localization quality. For tiny SAR ships (~25px), high mAP50-95 requires both detecting the ship and drawing an accurate box around it.

---

### Precision

**MLFlow key:** `metrics/precision(B)`

```
Precision = TP / (TP + FP)
```

Of all the boxes the model predicted, what fraction were real ships? Low precision → many false alarms (the model is triggering on noise/clutter in the SAR image).

Reported at the confidence threshold that maximizes F1.

---

### Recall

**MLFlow key:** `metrics/recall(B)`

```
Recall = TP / (TP + FN)
```

Of all the real ships in the dataset, what fraction did the model find? Low recall → the model is missing ships (false negatives). In a surveillance context, missed detections are typically more costly than false alarms.

**Precision-Recall tradeoff:** Lowering the confidence threshold increases recall (catch more ships) but decreases precision (more false alarms). The PR curve in MLFlow artifacts shows the full tradeoff across all thresholds.

---

### Box Loss (val/box_loss)

**MLFlow key:** `val/box_loss`

Validation loss from the bounding box regression head. In YOLOv8 this is **Distribution Focal Loss (DFL)**, which models the box edge locations as distributions rather than single point estimates, improving localization accuracy on small objects.

A decreasing box loss over epochs indicates the model is learning to localize ships better. Watch for plateau or divergence relative to training loss (sign of overfitting).

---

### Class Loss (val/cls_loss)

**MLFlow key:** `val/cls_loss`

Validation loss from the classification head. Uses **binary cross-entropy (BCE)**. With a single class (`ship`), this is essentially the model's confidence in predicting "ship" vs "background."

For this dataset (no empty images, single class), cls_loss is expected to converge quickly — the harder problem is localization.

---

### Small-Object AP (APs) — Context Only

Not currently logged to MLFlow but relevant background: the COCO evaluation suite defines:
- **APs**: AP for objects with area < 32² = 1,024 px²
- **APm**: AP for objects with area 32²–96²
- **APl**: AP for objects with area > 96²

Ships in this dataset have median area of ~640 px² (25.6 × 25 px), placing most detections squarely in the **APs** category. Models that perform well on COCO large objects may score much lower on APs — worth tracking if formal COCO eval is added in a later phase.

---

## Models

---

### YOLOv8 (n / s / m / l)

**Framework:** Ultralytics
**Phase:** 2
**Status:** In progress

#### Architecture

YOLOv8 is a **single-stage, anchor-free** object detector. A single forward pass through the network produces all detections — no separate region proposal step.

**Backbone — CSP with C2f modules**

The backbone uses a **Cross-Stage Partial (CSP)** design, which splits feature maps into two paths and merges them, reducing computation while preserving gradient flow. The core building block is the **C2f module** (Cross-Stage Partial with 2 bottlenecks + feature fusion), replacing the C3 modules of YOLOv5. C2f provides richer gradient flow from all bottleneck layers to the output, improving feature learning with fewer parameters.

**Neck — Path Aggregation FPN**

A **Feature Pyramid Network (FPN)** + **PAN (Path Aggregation Network)** neck fuses features from the backbone at three scales:

| Layer | Stride | Receptive Field | Best For |
|-------|--------|-----------------|----------|
| P3    | 8×     | Small           | Small objects |
| P4    | 16×    | Medium          | Medium objects |
| P5    | 32×    | Large           | Large objects |

For the SAR ship dataset, P3 (stride 8) is the critical output — ships are ~25px wide, producing a 3×3 px feature footprint at stride 8. This is why `imgsz=1024` is required; at 640px the same ship would be ~16px → 2×2 px at stride 8, losing significant spatial detail.

A **P2 head** (stride 4) can be added for ultra-small object detection but was not used in Phase 2 runs.

**Detection Head — Decoupled**

YOLOv8 uses **separate (decoupled) heads** for classification and box regression — one branch predicts class probabilities, another predicts box coordinates. This decoupling (borrowed from FCOS/YOLOX) outperforms the coupled heads of YOLOv5 at the cost of slightly more parameters.

#### Anchor-Free Design

Prior YOLO versions (v3–v5) used **anchor boxes** — predefined aspect ratio templates. Each grid cell predicted offsets relative to these anchors. This required manual anchor tuning per dataset.

YOLOv8 is **anchor-free**: each grid point directly predicts the absolute distances from the point to the four box edges (`top`, `left`, `bottom`, `right`). Benefits:
- No anchor hyperparameter tuning
- Better recall on unusual aspect ratios (ships can be any orientation)
- Simpler training objective

#### Loss Functions

| Component | Loss | Notes |
|-----------|------|-------|
| Box regression | **Distribution Focal Loss (DFL)** | Models edge locations as categorical distributions over discrete offsets; improves small-object localization |
| Classification | **Binary Cross-Entropy (BCE)** | One sigmoid per class; supports multi-label (not needed here but architecture allows it) |
| Objectness | Removed in v8 | Earlier YOLO versions had a separate objectness branch; v8 folds this into the classification head |

DFL is particularly relevant here: rather than predicting a single x/y offset, DFL predicts a probability distribution over a set of possible offsets and takes the expectation. This produces more accurate box edges on small objects where single-point regression is noisy.

#### Scale Variants

All four variants share identical architecture — only depth and width multipliers change:

| Variant | Depth | Width | Params | Notes |
|---------|-------|-------|--------|-------|
| YOLOv8n | 0.33  | 0.25  | ~3.2M  | Nano — fastest, weakest |
| YOLOv8s | 0.33  | 0.50  | ~11.2M | Small — good speed/accuracy tradeoff |
| YOLOv8m | 0.67  | 0.75  | ~25.9M | Medium |
| YOLOv8l | 1.00  | 1.00  | ~43.7M | Large — slowest, strongest |

**Depth multiplier** scales the number of bottleneck repeats inside each C2f block.
**Width multiplier** scales the number of channels throughout the network.

All variants are initialized from COCO-pretrained weights (`yolov8{n/s/m/l}.pt`), meaning they start with learned features for general object detection. Domain shift from natural images to SAR is a factor but the low-level edge/texture features still transfer.

#### Inference Pipeline

```
Image → Backbone (C2f blocks) → Neck (FPN+PAN) → Detection heads (P3/P4/P5)
     → Raw predictions (boxes + class scores) → NMS → Final detections
```

**Non-Maximum Suppression (NMS):** After the forward pass, many overlapping boxes are predicted for each ship. NMS filters these by keeping only the highest-confidence box when IoU between two boxes exceeds a threshold (default 0.45). Too-high NMS IoU → duplicate detections; too-low → suppresses valid detections in dense scenes.

For dense scenes (up to 289 ships per image), NMS threshold tuning matters.

#### Positives

- **Speed**: Real-time inference; nano variant runs >100 FPS on a modern GPU
- **Pretrained weights**: COCO pretraining gives a strong starting point even on domain-shifted SAR data
- **Ultralytics integration**: Built-in per-epoch metric logging, artifact saving (PR curve, confusion matrix), automatic mixed precision
- **Anchor-free**: No per-dataset anchor optimization required
- **Scales cleanly**: Same codebase, four capability levels for MLFlow comparison

#### Negatives / Watch-Outs

- **Resolution sensitivity**: Small-object performance degrades sharply with reduced input size. At 640px, ~25px ships become ~16px features — borderline for detection. Always use `imgsz=1024` on this dataset.
- **NMS sensitivity on dense scenes**: Images with 100+ ships require careful IoU threshold tuning. The default 0.45 may suppress valid nearby ships.
- **Domain shift**: Pretrained on RGB natural images; SAR pseudo-color has different statistical properties. Expect lower initial precision, especially for low-backscatter ships (ships that appear faint in SAR imagery).
- **No explicit attention**: Unlike transformer-based models (RT-DETR), YOLOv8 has no global attention mechanism. It struggles when context (surrounding sea clutter, ship wakes) is needed to disambiguate detections.
- **Confidence calibration**: Raw confidence scores may not be well-calibrated on out-of-domain data. A high-confidence SAR prediction is not necessarily more reliable than the same score on natural images.

#### What to Watch in MLFlow

| Signal | What it tells you |
|--------|-------------------|
| mAP50 n→s→m→l | Diminishing returns curve — where does scale stop helping? |
| mAP50-95 | Whether larger models produce tighter boxes, not just more detections |
| Precision vs recall gap | Low recall = model missing ships; low precision = false alarms on sea clutter |
| Training vs val curves diverging | Overfitting — consider early stopping or increased regularization |
| Box loss plateau | Model has stopped improving localization — check if resolution is the bottleneck |

#### Phase 2 Runs

| Run name | Epochs | imgsz | Batch | mAP50 | mAP50-95 | Precision | Recall | Inference | Train time |
|----------|--------|-------|-------|-------|----------|-----------|--------|-----------|------------|
| yolov8n_sz640_ep3_* | 3 | 640 | 16 | 0.814 | 0.484 | — | — | — | — | Smoke test only |
| yolov8n_sz1024_ep50_0309_0912 | 50 | 1024 | 8 | 0.913 | 0.644 | 0.903 | 0.826 | 4.9ms | 1.3h |
| yolov8s_sz1024_ep50_0309_1301 | 50 | 1024 | 8 | 0.918 | 0.656 | 0.901 | 0.837 | 10.2ms | 2.7h |
| yolov8m_sz1024_ep50_0309_1553 | 50 | 1024 | 8 | 0.920 | 0.662 | 0.911 | 0.833 | 22.1ms | 5.5h |
| yolov8l_sz1024_ep50_0310_0000 | 50 | 1024 | 8 | 0.920 | 0.663 | 0.911 | 0.829 | 36.3ms | 9.2h |

**Key finding:** mAP50 plateaus completely at YOLOv8m. Going from n→l yields only +0.7% mAP50 at 7.4× inference cost. YOLOv8n is the Pareto-optimal choice for this dataset — the bottleneck is the inherent difficulty of detecting ~25px ships, not model capacity. Recall slightly decreases from s→l, suggesting larger models marginally overfit the training distribution.

---

### RT-DETR-L

**Framework:** Ultralytics
**Phase:** 2
**Status:** In progress

#### Background

RT-DETR (Real-Time Detection Transformer) was published by Baidu in 2023 with the provocative subtitle *"DETRs Beat YOLOs on Real-time Object Detection."* It is the first transformer-based detector to achieve YOLO-competitive inference speeds while retaining the global attention mechanism that makes transformers powerful. The `-L` variant uses a HGNetv2-L (High-Performance GPU Network v2) backbone.

#### Architecture

RT-DETR is a **hybrid CNN + transformer** architecture. It does not replace the CNN entirely — instead it uses a CNN backbone to extract features, then applies transformer attention at the feature level.

**Backbone — HGNetv2-L**

Baidu's HGNetv2 is a CNN backbone optimised for GPU throughput, using depthwise separable convolutions and a hierarchical feature design. It produces multi-scale feature maps at strides 8, 16, and 32 — the same P3/P4/P5 pyramid as YOLOv8.

**Hybrid Encoder**

This is the architectural centrepiece. Two modules operate in sequence:

1. **AIFI (Attention-based Intra-scale Feature Interaction):** A transformer encoder applied to the P5 (stride 32) feature map only. This is the key insight — applying full self-attention at the coarsest scale keeps the sequence length manageable (32×32 = 1024 tokens at 1024px input) while still enabling global reasoning. Every position in the P5 map can attend to every other position — the model can see the whole scene at once.

2. **CCFM (CNN-based Cross-scale Feature Fusion Module):** A CNN-based neck that fuses the attention-enhanced P5 features back down to P3 and P4 scales. This is similar to YOLOv8's FPN+PAN neck but informed by the global context from AIFI.

**Transformer Decoder**

6 decoder layers using cross-attention between a set of learned **object queries** and the encoder features. Each query iteratively refines its predicted box location and class score by attending to the relevant regions of the feature map. At inference, the top-N queries by confidence score become the final detections.

**NMS-Free Inference**

This is RT-DETR's most operationally significant property. During training, predictions are matched to ground truth using the **Hungarian algorithm** (bipartite matching) — each ground truth is assigned to exactly one query, with no duplicates. Because of this training objective, the model learns not to produce duplicate predictions, so **no NMS post-processing is required at inference.**

For YOLO, NMS is a hyperparameter-sensitive step: the IoU threshold controls how aggressively overlapping boxes are suppressed. In dense SAR scenes (up to 289 ships), NMS can suppress valid nearby ships. RT-DETR eliminates this problem entirely.

#### Key Differences from YOLOv8

| Property | YOLOv8 | RT-DETR-L |
|---|---|---|
| Detection paradigm | Dense grid predictions + NMS | Sparse object queries, NMS-free |
| Attention | None (pure CNN) | Global self-attention at P5 scale |
| Backbone | CSP + C2f | HGNetv2 (depthwise conv) |
| Decoder | Single forward pass to head | 6-layer iterative query refinement |
| Convergence speed | Fast (~20 epochs) | Slower (~50–100 epochs) |
| Params (L variant) | 43.7M (YOLOv8l) | ~32M |

#### Why Global Attention May Help SAR

YOLOv8 processes each grid cell largely in isolation — it can use local context from its receptive field, but has no explicit mechanism to reason about the global scene. Global attention in AIFI means the model can implicitly learn:

- **Sea state context**: High-clutter sea regions vs calm open water — the model can suppress detections in known-noisy areas
- **Ship clustering patterns**: Real vessel traffic has structure (shipping lanes, port approaches) — attention can learn these priors
- **Wake signatures**: Ship wakes extend far from the vessel — global attention can connect a wake to the ship that caused it

Whether this translates to measurable mAP improvement on our dataset is the empirical question this run answers.

#### Positives

- **NMS-free**: Eliminates threshold tuning and duplicate suppression issues in dense scenes
- **Global context**: AIFI attention lets the model reason about the whole scene, not just local patches
- **Strong COCO baseline**: Matches or beats YOLOv8l on COCO at lower parameter count
- **Same Ultralytics API**: `YOLO("rtdetr-l.pt")` — zero code changes to our pipeline

#### Negatives / Watch-Outs

- **Slower convergence**: Transformer decoders need more epochs to learn stable query assignments. 50 epochs may not be enough for full convergence (COCO training uses 72–120 epochs)
- **Higher compute per epoch**: The AIFI attention block and 6-layer decoder add significant compute vs the single-pass YOLO head. Expect ~2–3× slower training than YOLOv8m per epoch
- **VRAM**: At 1024px the P5 attention sequence is longer — may require reducing batch size
- **Small object challenge**: AIFI only applies attention at P5 (stride 32). A 25px ship at 1024px produces a ~0.8px footprint at P5 — the ship may not even register at the attention scale. Detection still relies on P3/P4 CNN features, same as YOLO
- **Quadratic attention cost**: Self-attention scales O(n²) with sequence length. At 1024px, P5 sequence = 1024 tokens vs 400 at 640px — 2.56× more attention compute

#### What to Watch in MLFlow

| Signal | What it tells you |
|--------|-------------------|
| mAP50 vs YOLOv8n | Does global attention help find more ships than a pure CNN? |
| mAP50-95 vs YOLO | Does iterative query refinement produce tighter boxes? |
| speed_inference_ms | Cost of the transformer decoder — expect 2–4× slower than YOLOv8n |
| Training loss curve shape | Slower initial convergence is normal; watch for plateau before epoch 50 |
| Recall vs YOLO recall | NMS-free should improve recall in dense scenes |

#### Phase 2 Runs

| Run name | Epochs | imgsz | Batch | mAP50 | mAP50-95 | Precision | Recall | Inference | Train time |
|----------|--------|-------|-------|-------|----------|-----------|--------|-----------|------------|
| *(TBD)* | 50 | 640 | 8 | — | — | — | — | — | — |

---

### Faster R-CNN (ResNet-50 FPN)

**Framework:** torchvision
**Phase:** 2
**Status:** Planned

*Section to be completed after Phase 2 Faster R-CNN runs.*

---

### YOLOv8-seg

**Framework:** Ultralytics
**Phase:** 4
**Status:** Planned

*Section to be completed in Phase 4 (segmentation).*

---

### Mask R-CNN

**Framework:** torchvision
**Phase:** 4
**Status:** Planned

*Section to be completed in Phase 4 (segmentation).*
