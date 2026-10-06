# BEVFormer PyTorch: Pure Native Re-implementation
A modular, native PyTorch reimplementation of BEVFormer for 3D Bird’s-Eye-View (BEV) spatial-temporal perception. This repository provides a transparent, dependency-free implementation designed for research, architecture verification, and educational exploration of multi-camera 3D perception.

---

## Key Features

- **Pure PyTorch Design:** Reimplemented without framework overhead (e.g., MMCV or MMDet3D). All deformable attention and temporal warping operations use native `torch.nn.functional.grid_sample`.
- **Spatial Cross-Attention (SCA):** Projects 3D anchor points from ego space into 2D feature maps across 6 surround cameras (CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT).
- **Temporal Self-Attention (TSA):** Warps past BEV queries into the current ego frame using ego-motion transformation matrices ($R_{\text{curr2prev}}$, $t_{\text{curr2prev}}$) to maintain temporal continuity.
- **ResNet-18 Backbone & BEV Head:** Multi-camera feature extraction combined with a multi-class BEV segmentation head.
- **Pipeline Optimization:** Includes automatic mixed precision training (`torch.amp`), gradient accumulation, layer-wise learning rate decay, and custom class-wise IoU metrics.

---

## Repository Structure

```
├── generate_gt_channels.py  # transforms data into 200x200 grids used as ground-truths for model training
├── preprocess.py            # Preprocesses nuScenes: resizes images, rescales K, computes extrinsics & ego-motion
├── data_loader.py           # PyTorch Dataset loader for preprocessed PyTorch tensors (.pt)
├── Model.py                 # Native PyTorch implementation of BEVFormer (Backbone, TSA, SCA, Encoder, Head)
├── train.py                 # Training loop with AMP, gradient accumulation, Cosine Annealing, and mIoU metric
└── README.md                # Project documentation
```
---

## Module Breakdown

* **`generate_gt_channels`**: builds 5-channel BEV ground-truth grids ($200 \times 200$, $102.4\text{m} \times 102.4\text{m}$ centered on ego) by rasterizing map drivable-area polygons into Channel 0 and projecting nuScenes object bounding boxes into ego frame for Channels 1–4 (cars/vehicles, pedestrians, two-wheelers, static objects/barriers), then stacks and saves all samples as `channels.npy` with shape $(N, 5, 200, 200)$.
* **`preprocess.py`**: Parses the nuScenes dataset (`v1.0-mini`), resizes images from $1600 \times 900$ to $800 \times 450$, rescales intrinsic matrix $K$, computes camera extrinsics ($R, t$), extracts relative ego-motion transformation matrices ($R_{\text{curr2prev}}$, $t_{\text{curr2prev}}$), and serializes per-frame tensors as `.pt` files.
* **`data_loader.py`**: Defines `NuScenesFrameLoader` PyTorch Dataset to load preprocessed frame tensors for training and evaluation splits.
* **`Model.py`**: Native PyTorch implementation of the BEVFormer architecture:
  * `Backbone`: ResNet-18 multi-view feature extractor operating across 6 surround cameras.
  * `DeformableAttention`: Native single-scale deformable attention using `F.grid_sample` without custom CUDA compilation dependencies.
  * `TemporalSelfAttention` (TSA): Warps past BEV queries into current ego frame coordinates using relative pose matrices ($R_{\text{curr2prev}}$, $t_{\text{curr2prev}}$) and fuses temporal features.
  * `SpatialCrossAttention` (SCA): Projects 3D physical grid anchors into 2D camera coordinates and aggregates multi-view features.
  * `BEVSegmentationHead`: Convolutional refinement blocks and $1 \times 1$ classification head for multi-class BEV segmentation.
* **`train.py`**: End-to-end training pipeline featuring class-weighted `BCEWithLogitsLoss`, mixed precision (`torch.amp`), gradient accumulation (`accum_steps = 4`), `CosineAnnealingLR` scheduler, and `BEVIoUMetric` evaluation.

---

## Experimental Setup & Scope

* **Dataset**: Evaluated exclusively on the **nuScenes mini dataset** (`v1.0-mini`).
* **Loss Function**: Trained using class-weighted **Binary Cross-Entropy Loss (`BCEWithLogitsLoss`)** with positive weights (`pos_weight=[10.0, 20.0, 20.0, 20.0, 20.0]`) to rapidly test and verify architecture convergence, spatial alignment, and gradient flow.

---

# Model Architecture Overview

- **Backbone:** Extracts feature maps from $6 \times \text{camera views}$ using ResNet-18.
- **3D Reference Grid:** Generates 3D physical coordinates $(X, Y, Z)$ in ego space.
- **Temporal Self-Attention (TSA):** Warps previous BEV representations using relative ego-motion ($R_{\text{curr2prev}}$, $t_{\text{curr2prev}}$) and fuses them with current queries using single-scale deformable attention.
- **Spatial Cross-Attention (SCA):** Samples image features across camera channels using 3D-to-2D geometric projections.
- **Decoder Head:** Maps refined BEV query tokens to multi-class spatial logits via convolutional refinement blocks.
