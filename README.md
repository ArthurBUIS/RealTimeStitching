# Real-Time Panoramic Surveillance Video Stitching

Implementation of:

> **"Real-Time Panoramic Surveillance Video Stitching Method for Complex Industrial Environments"**
> Jiuteng Zhu, Jianyu Guo, Kailun Ding, Gening Wang, Youxuan Zhou, Wenhong Li
> *Sensors 2026, 26, 186* — https://doi.org/10.3390/s26010186

---

## Project Structure

```
panoramic_stitching/
│
├── models/
│   ├── feature_extractor.py   # Siamese ResNet-50 + ECA + CA attention
│   └── homography_net.py      # Full two-scale homography estimation network
│
├── fusion/
│   ├── motion_detector.py     # MOG2 background subtraction → M_Ω mask
│   ├── graph_cut.py           # Motion-aware energy function + min-cut seam search
│   ├── blending.py            # Weighted average & seam-guided compositing
│   └── stitcher.py            # Frame-level real-time pipeline orchestrator
│
├── utils/
│   ├── warp_coco.py           # Generate Warped MS-COCO dataset from raw COCO
│   └── video_utils.py         # VideoReader/Writer, frame extraction, benchmarking
│
├── dataset.py                 # Dataset loaders (WarpedCOCO, UDIS-D, composite)
├── losses.py                  # L_sim + L_smooth loss functions (eqs. 1–5)
├── train.py                   # Two-stage training loop
├── evaluate.py                # RMSE / PSNR / SSIM evaluation (Tables 1–3)
├── visualize.py               # Qualitative comparison figures (Figs 8–12)
└── requirements.txt
```

---

## Installation

```bash
git clone RealTimeStitching
cd RealTimeStitching
pip install -r requirements.txt
```

> **PyMaxflow** is required for graph-cut seam search.
> Without it the system falls back to dynamic programming automatically.
> Install with: `pip install PyMaxflow`

---

## Data Preparation

### 1. Warped MS-COCO (Stage-1 pre-training)

Download raw MS-COCO images (train2017) then generate synthetic pairs:

```bash
python utils/warp_coco.py \
    --coco_dir   /data/coco/train2017 \
    --output_dir /data/WarpedCOCO \
    --n_train    10000 \
    --n_test     1000  \
    --patch_size 512   \
    --rho        32
```

This creates:
```
/data/WarpedCOCO/
    train/input1/  ← reference patches
    train/input2/  ← warped patches (known homography)
    test/input1|2/
```

### 2. UDIS-D (Stage-2 fine-tuning)

Download from the official repository:
```
https://github.com/nie-lang/UnsupervisedDeepImageStitching
```

Expected layout:
```
/data/UDIS-D/
    training/input1/
    training/input2/
    testing/input1/
    testing/input2/
```

### 3. Industrial dataset

Organise your real-world industrial images in the same layout:
```
/data/industrial/
    training/input1/   ← camera A frames
    training/input2/   ← camera B frames (overlapping)
    testing/input1|2/
```

---

## Training

### Two-stage training (Section 4.1.1)

**Stage 1** — 100 epochs pre-training on Warped MS-COCO:

**Stage 2** — 50 epochs fine-tuning on composite dataset:

Both stages run automatically with a single command:

```bash
python train.py \
    --coco_root        /data/WarpedCOCO \
    --udis_root        /data/UDIS-D \
    --industrial_root  /data/industrial \
    --output_dir       checkpoints/ \
    --batch_size       8 \
    --stage1_epochs    100 \
    --stage2_epochs    50 \
    --lr               1e-4 \
    --lambda_smooth    10.0
```

Resume from checkpoint:
```bash
python train.py ... --resume checkpoints/stage1_latest.pth
```

Key hyperparameters (paper values):

| Parameter         | Value  | Description                       |
|-------------------|--------|-----------------------------------|
| `lr`              | 1e-4   | Initial learning rate             |
| `lambda_smooth`   | 10.0   | Smoothness loss weight λ (eq. 1)  |
| `stage1_epochs`   | 100    | MS-COCO pre-training epochs       |
| `stage2_epochs`   | 50     | Fine-tuning epochs                |
| LR decay          | 0.97   | ExponentialLR γ per epoch         |
| `batch_size`      | 8      | Batch size                        |
| `img_size`        | 512    | Input resolution (px)             |

---

## Evaluation

Reproduces Tables 1–3 (RMSE, PSNR, SSIM) and Table 4 (runtime):

```bash
python evaluate.py \
    --checkpoint       checkpoints/stage2_best.pth \
    --coco_root        /data/WarpedCOCO \
    --udis_root        /data/UDIS-D \
    --industrial_root  /data/industrial \
    --batch_size       8
```

Expected results (from Table 1–3 of the paper):

| Metric   | 0–30%  | 30–60% | 60–100% | Average |
|----------|--------|--------|---------|---------|
| RMSE ↓   | 1.14   | 1.45   | 2.97    | 1.965   |
| PSNR ↑   | 30.02  | 25.73  | 21.57   | 25.338  |
| SSIM ↑   | 0.9312 | 0.8728 | 0.7385  | 0.8366  |

Runtime: **53 ms/pair** → ~19 fps registration only, **23 fps** full pipeline.

---

## Inference — Real-Time Video Stitching

```bash
python fusion/stitcher.py \
    --checkpoint checkpoints/stage2_best.pth \
    --video_a    /data/cam_a.mp4 \
    --video_b    /data/cam_b.mp4 \
    --output     stitched_output.mp4 \
    --img_size   512
```

### Python API

```python
from fusion.stitcher import PanoramicStitcher
import cv2

stitcher = PanoramicStitcher(
    checkpoint_path="checkpoints/stage2_best.pth",
    img_size=512,
)

cap_a = cv2.VideoCapture("cam_a.mp4")
cap_b = cv2.VideoCapture("cam_b.mp4")

while True:
    ret_a, fa = cap_a.read()
    ret_b, fb = cap_b.read()
    if not ret_a or not ret_b:
        break

    panorama, info = stitcher.stitch(fa, fb)
    cv2.imshow("Panorama", panorama)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Three-camera setup (Section 4.2.2)

```python
panorama = stitcher.stitch_multi([frame_a, frame_b, frame_c])
```

---

## Visualisation

Generate qualitative comparison figures:

```bash
python visualize.py \
    --checkpoint  checkpoints/stage2_best.pth \
    --img_a       samples/cam_a.jpg \
    --img_b       samples/cam_b.jpg \
    --output_dir  vis_output/
```

Outputs:
- `registration_quality.png` — input | target | warped | diff heatmap
- `seam_diagnostic.png` — panorama | seam overlay | motion overlay
- `psnr_comparison.png` — grouped bar chart (Table 2 values)
- `ssim_comparison.png` — grouped bar chart (Table 3 values)

---

## Architecture Overview

### Registration Network (Section 3.2)

```
Input: I_A, I_B  (512×512)
         │
    ┌────┴────┐
    │ Siamese │   ResNet-50 + ECA (after Layer0) + CA (after Layer1-3)
    │backbone │
    └────┬────┘
    F_a¹/₈, F_b¹/₈     (128ch, 64×64)
    F_a¹/₁₆, F_b¹/₁₆   (256ch, 32×32)
         │
    GlobalCorrelation(F_a¹/₁₆, F_b¹/₁₆)
         │
    RegressionNet → TensorDLT → H₁
         │
    Warp F_b¹/₈ with H₁
         │
    GlobalCorrelation(F_a¹/₈, warped_F_b¹/₈)
         │
    RegressionNet → TensorDLT → H₂
         │
    Warp I_A, I_B with H₂ → registered pair
```

### Fusion Pipeline (Section 3.3)

```
Frame N
  │
  ├─ First frame?
  │     YES → full registration + seam search over Ω
  │     NO  → use cached H₂
  │
  ├─ MOG2 motion detection → M_Ω
  │
  ├─ Motion near seam?
  │     YES → recompute seam over Ω ∖ M_Ω  (eq. 7)
  │     NO  → use cached seam
  │
  └─ SeamBlender → panoramic frame
```

### Loss Function (Section 3.2.3)

```
L_w = L_sim + λ · L_smooth          (eq. 1,  λ=10)

L_sim   = ‖M ⊙ (H(I_A) − I_B)‖₁   (eq. 2)

L_smooth = L_intra + L_inter        (eq. 3)
  L_intra = Σ ReLU(‖e‖ − η)        (eq. 4)
  L_inter = Σ (1 − cos(e₁, e₂))    (eq. 5)
```

---

## Dependencies

| Package        | Purpose                          |
|----------------|----------------------------------|
| torch          | Deep learning framework          |
| torchvision    | ResNet-50 backbone               |
| opencv-python  | Image processing, MOG2, warping  |
| scikit-image   | PSNR / SSIM metrics              |
| PyMaxflow      | Boykov-Kolmogorov graph cut      |
| matplotlib     | Visualisation                    |
| tqdm           | Training progress bars           |

---

## Citation

```bibtex
@article{zhu2026panoramic,
  title   = {Real-Time Panoramic Surveillance Video Stitching Method
             for Complex Industrial Environments},
  author  = {Zhu, Jiuteng and Guo, Jianyu and Ding, Kailun and
             Wang, Gening and Zhou, Youxuan and Li, Wenhong},
  journal = {Sensors},
  volume  = {26},
  number  = {1},
  pages   = {186},
  year    = {2026},
  doi     = {10.3390/s26010186}
}
```
