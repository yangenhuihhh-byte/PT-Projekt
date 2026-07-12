# PT Projekt – Surface Height Prediction using U-Net

## Overview

This project investigates the prediction of surface height maps from microscope texture images using deep learning.

The goal is to train a convolutional neural network capable of reconstructing the corresponding height map from a single texture image. During the project, several network architectures and dataset generation strategies were explored, and the final implementation is based on a U-Net model.

---

## Objectives

- Generate paired texture–height datasets from Alicona microscope measurements.
- Train a deep learning model for surface height prediction.
- Improve prediction accuracy through optimized patch generation and preprocessing.
- Evaluate the prediction quality by comparing predicted height maps with ground truth.

---

## Dataset

Microscope data were acquired using Alicona measurements.

Dataset preparation includes:

- Patch extraction
- Overlapping patches
- Normalization
- Training / prediction split

Final dataset:

- Patch size: **128 × 128**
- Stride: **192**
- Approximately **17,400 training patches**

---

## Model

Final model:

- U-Net
- PyTorch
- CUDA acceleration

Training settings:

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Optimizer | Adam |
| Loss | L1 Loss |
| Patch size | 128×128 |
| Stride | 192 |

Final training loss:

```
Loss ≈ 0.365
```

---

## Project Structure

```
PT-Projekt
│
├── scripts
│   ├── prepare_dataset.py
│   ├── train.py
│   └── predict.py
│
├── data
│
├── dataset_alicona
│
├── runs
│
├── .gitignore
│
└── README.md
```

---

## Workflow

1. Generate training patches from Alicona measurements.
2. Normalize texture and height data.
3. Train the U-Net model.
4. Save the trained model.
5. Predict height maps from unseen texture images.
6. Compare prediction with ground truth.

---

## Current Progress

Completed:

- Dataset generation
- Overlapping patch extraction
- Data normalization
- U-Net implementation
- Model training
- Prediction visualization

Current best model:

```
Patch size : 128 × 128
Stride     : 192
Epochs     : 100
Loss        : ~0.365
```

The predicted height maps show a significantly improved agreement with the measured ground truth compared with the initial experiments.

---

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

---

## Author

**Enhui Yang**

Technische Universität Berlin

PT Projekt
