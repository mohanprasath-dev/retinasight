# RetinaSight — MATLAB Interoperability Suite

> **SIH 2026 Problem Statement ID: 26038 (MathWorks)**  
> **Topic:** Explainable AI for Diabetic Retinopathy Screening in Rural India  
> **Team:** OnFocus | **Lead Developer:** Mohan Prasath P

---

## 1. Overview
This directory contains the official MathWorks MATLAB pipeline implementation for RetinaSight, demonstrating seamless bi-directional interoperability between our open-source edge deployment stack (PyTorch/ONNX) and the native MathWorks ecosystem.

## 2. Required MathWorks Toolboxes
- **Image Processing Toolbox** (`adapthisteq`, `rgb2gray`, `imbinarize`, `labeloverlay`)
- **Deep Learning Toolbox** (`importONNXNetwork`, `dlarray`, `gradCAM`, `predict`)
- **Medical Imaging Toolbox / Computer Vision Toolbox** (Filtering and geometric ROI)

## 3. How to Run in MATLAB
1. Open MATLAB (R2022b or newer recommended).
2. Set the working directory to `retinasight/matlab/`:
   ```matlab
   cd('path/to/retinasight/matlab');
   ```
3. Run the automated diagnostic pipeline:
   ```matlab
   results = retinasight_pipeline();
   ```
   Or pass a custom fundus photograph and model path:
   ```matlab
   results = retinasight_pipeline('path/to/fundus.png', '../retinasight_resnet50.onnx');
   ```

## 4. Pipeline Execution Stages
1. **Stage 1: Pre-Inference Quality Check:** Evaluates Laplacian variance for motion blur (threshold $\ge 50.0$), mean illumination within retinal FOV ($35.0 - 215.0$), and centering ratio.
2. **Stage 2: Green-Channel Enhancement:** Maximizes hemoglobin optical absorption using CLAHE (`adapthisteq`) and edge-preserving filtering (`imguidedfilter`).
3. **Stage 3: Structure Segmentation:** Isolates blood vessels via morphological bottom-hat transform and detects the Optic Disc center coordinates.
4. **Stage 4: ONNX Network Import:** Directly imports the fine-tuned ResNet50 classifier (`retinasight_resnet50.onnx`) into a native MATLAB DAGNetwork.
5. **Stage 5: Explainability with gradCAM:** Computes gradient-weighted class activation maps (`gradCAM`) focused strictly within the retinal FOV.
