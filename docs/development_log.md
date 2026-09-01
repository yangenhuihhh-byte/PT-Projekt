# PT Projekt - Development Log

Author: Enhui Yang

University: Technische Universität Berlin

---

# Project Overview

The objective of this project is to develop a deep learning model capable of reconstructing surface height maps from Alicona microscope texture images.

The development process includes dataset preparation, preprocessing, model design, training, evaluation, optimization, and Git version control.

This document records the major development stages, encountered problems, applied solutions, and obtained results during the project.

---

# Development History
## Stage 1 - Project Initialization

### Objective

Build the complete development environment for the project.

### Tasks

- Create project directory
- Create Python virtual environment
- Install PyTorch
- Install required Python packages
- Verify GPU environment

### Result

The development environment was successfully established and the project structure was created.
## Stage 2 - Alicona Dataset Analysis

### Objective

Understand the structure of the Alicona measurement data and establish paired input-output samples for deep learning.

### Tasks

- Analyze the folder structure of the Alicona dataset
- Identify texture images and height data
- Determine the correspondence between texture.bmp and dem.a3d
- Build the initial data loading pipeline

### Problems

The dataset contains proprietary Alicona data formats that cannot be directly used for neural network training.

### Solution

Develop a custom dataset loader capable of reading both texture images and height data.

### Result

The correspondence between texture images and height maps was successfully established.
## Stage 3 - Parsing Alicona Height Files

### Objective

Convert the original *.a3d files into usable height matrices.

### Problems

The first implementation failed because the number of values stored in the file did not match the expected image resolution.

Typical errors included:

- reshape error
- inconsistent data dimensions

### Analysis

The Alicona *.a3d files contain additional header information before the actual height values.

### Solution

Ignore the header section and only keep the valid height data before reshaping.

### Result

The height map was successfully reconstructed and could be visualized correctly.
## Stage 4 - Dataset Construction

### Objective

Create a PyTorch Dataset for model training.

### Tasks

- Implement SurfaceDataset
- Load texture images
- Load height maps
- Normalize image values
- Return paired samples

### Result

The complete training dataset could now be loaded automatically by the DataLoader.
## Stage 5 - Patch Generation

### Objective

Increase the amount of training data by splitting large microscope images into smaller patches.

### Initial Configuration

Patch size:

256 × 256

Stride:

256

### Problems

Only a limited number of patches were generated.

The model quickly reached its learning limit because of insufficient training samples.

### Result

The first patch dataset was successfully generated and used for model training.
## Stage 6 - First Model Training

### Objective

Train the first neural network for surface reconstruction.

### Initial Network

SimpleCNN

### Training Result

The network converged poorly.

Prediction maps were almost uniform and failed to recover meaningful surface structures.

### Conclusion

A more powerful architecture was required.
## Stage 7 - Migration to U-Net

### Objective

Improve the prediction quality by replacing the initial SimpleCNN with a U-Net architecture.

### Motivation

The SimpleCNN model was unable to recover detailed surface structures and mainly predicted average height values.

### Solution

Replace the original network with a U-Net architecture, which preserves spatial information through skip connections.

### Result

The prediction quality improved significantly, and the model started learning the overall surface geometry.
## Stage 8 - Training Instability

### Objective

Train the U-Net model using the generated dataset.

### Problems

The first training attempt failed.

Observed problems included:

- Extremely large loss values
- NaN during training
- No convergence

Example:

Epoch 1 Loss

10481073440878661468160

Later:

Loss = NaN

### Analysis

The raw height values had a very large numerical range, making optimization unstable.

### Result

The training process became unusable and required further investigation.
## Stage 9 - Height Normalization

### Objective

Improve training stability.

### Solution

Normalize the height map before training using Min-Max normalization.

The normalized values were mapped into the range

[0,1]

### Result

Training immediately became stable.

Typical loss values became

0.46

instead of extremely large values.

The NaN problem disappeared completely.
## Stage 10 - Loss Function Optimization

### Objective

Improve reconstruction quality.

### Original Loss

Mean Squared Error (MSE)

### Modified Loss

L1 Loss

### Motivation

L1 Loss is more robust against local outliers and preserves height discontinuities better than MSE.

### Result

Training became smoother and the prediction quality improved.
## Stage 11 - Increasing the Dataset Size

### Objective

Increase the amount of available training samples.

### Initial Dataset

8526 patches

### Optimization

Introduce overlapping patches.

Modify stride

256

↓

192

### Result

The effective amount of training data increased significantly.

The model generalized better and produced smoother predictions.
## Stage 12 - Hyperparameter Optimization

Several training parameters were evaluated during the project.

Main parameters included:

- Patch size
- Stride
- Number of epochs
- Learning rate
- Loss function

Different combinations were tested to improve the prediction quality.

The best configuration at the current stage is

Patch size

128 × 128

Stride

192

Epochs

100

Loss

L1 Loss
## Stage 13 - Long Training

### Objective

Further improve prediction accuracy.

### Training Configuration

Epochs

100

### Training Result

Final loss

Approximately

0.365

Compared with the initial experiments, the prediction became much closer to the measured height map.
## Stage 14 - Automatic Prediction Saving

### Objective

Automatically save prediction results after inference.

### Previous Workflow

Prediction figures were only displayed using matplotlib.

Manual screenshots were required.

### Improvement

Modify predict.py to automatically save the prediction image into

results/

### Result

Every prediction can now be stored automatically for later comparison and documentation.
## Stage 15 - Git Version Control

### Objective

Introduce version control for the project.

### Completed Tasks

- Install Git
- Initialize local repository
- Configure .gitignore
- Create GitHub repository
- Push source code
- Upload prediction results

### Result

The project is now fully managed using Git and GitHub.
# Current Project Status

Completed

✔ Development environment

✔ Dataset loading

✔ Alicona file parser

✔ Patch generation

✔ U-Net implementation

✔ Stable training

✔ Prediction visualization

✔ Automatic result saving

✔ GitHub repository

Current Best Model

Patch Size

128 × 128

Stride

192

Epochs

100

Loss

≈ 0.365

The current model successfully reconstructs the global surface structure and shows a significant improvement compared with the initial experiments.

Further work will focus on improving local details and reducing remaining prediction errors.

---

# Stage 16 - Post-Midterm Experiment Update

Date: 2026-08-31

## Objective

Extend the project from a single-model demonstration to a complete experiment workflow with clear dataset splits, quantitative model comparison and complete microscope-image prediction.

## Completed Updates

- Reworked Alicona `.al3d` height loading using header metadata such as `DepthImageOffset`.
- Added filtering for invalid height sentinel values before normalization.
- Split data by complete microscope image:
  - Train: 120 complete images, V1-V20
  - Validation: 24 complete images, V21-V24
  - Test: 30 complete images, V25-V29
- Added patch indices for 64, 128 and 256 patch sizes.
- Added model comparison across SimpleCNN, Small U-Net and TransUNet-lite.
- Added `evaluate.py` for patch-level MAE, MSE and RMSE evaluation.
- Added `predict_full_image.py` for complete-image sliding-window prediction.
- Added overlap and Gaussian blending for reducing patch-boundary artifacts.
- Added current-stage report files and report-ready figures.

## Current Best Result

The current best complete-image setting is:

```text
Model: Small U-Net
Patch size: 128
Prediction stride: 64
Blend: Gaussian overlap
```

Average result on 30 complete test microscope images:

```text
Mean MAE_norm  = 0.0846515231
Mean MSE_norm  = 0.0127101013
Mean RMSE_norm = 0.1117579066
```

## Current Status

The project is now in the experiment consolidation stage. The main workflow is complete, and the next work should focus on improving local fine-detail prediction, confirming physical height-unit interpretation and preparing the final project presentation/report.
