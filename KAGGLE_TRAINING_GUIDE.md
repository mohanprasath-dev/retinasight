# 🚀 RetinaSight — Kaggle GPU Model Training Guide

> **Smart India Hackathon 2026** | **Problem Statement ID:** 26038 (MathWorks) | **Team OnFocus**

This guide shows you how to train our deep learning model on Kaggle's free **NVIDIA Tesla T4 / P100 GPUs** (30 hours/week free GPU).

---

## 📁 Files Provided in Your Repository

1. **Jupyter Notebook (Ready to Upload):**  
   [`kaggle_notebook/retinasight_kaggle_training.ipynb`](file:///d:/SIH2026/retinasight/kaggle_notebook/retinasight_kaggle_training.ipynb)
2. **Python Script (Direct Copy-Paste):**  
   [`kaggle_notebook/retinasight_kaggle_training.py`](file:///d:/SIH2026/retinasight/kaggle_notebook/retinasight_kaggle_training.py)

---

## 🛠️ Step-by-Step Instructions

### Step 1: Open Kaggle & Start a New Notebook

1. Log in to [Kaggle](https://www.kaggle.com).
2. Go to the APTOS competition code page:  
   👉 **[https://www.kaggle.com/competitions/aptos2019-blindness-detection/code](https://www.kaggle.com/competitions/aptos2019-blindness-detection/code)**
3. Click the blue **"New Notebook"** button in the top right.
   _(The 3,662 high-resolution images in `aptos2019-blindness-detection` are automatically mounted at `/kaggle/input/aptos2019-blindness-detection` with 0 download time!)_

---

### Step 2: Enable Free GPU Acceleration

1. Look at the right sidebar menu (**Notebook options**).
2. Click **Accelerator** and select **GPU T4 x 2** (or **GPU P100**).
3. Under **Settings**, make sure **Internet** is toggled to **"Internet On"** (needed once to load torchvision pretrained ResNet-50 backbone weights).

---

### Step 3: Load the Training Code

Choose either **Option A** or **Option B**:

#### Option A (Upload Notebook):

1. In the top Kaggle menu, click **File** -> **Upload Notebook**.
2. Select the file: `d:\SIH2026\retinasight\kaggle_notebook\retinasight_kaggle_training.ipynb`.

#### Option B (Copy & Paste):

1. Open [`kaggle_notebook/retinasight_kaggle_training.py`](file:///d:/SIH2026/retinasight/kaggle_notebook/retinasight_kaggle_training.py).
2. Copy all code.
3. Paste it into the first cell of your Kaggle notebook.

---

### Step 4: Run Training

1. Click **Run All** (or press `Shift + Enter` on the cell).
2. The training executes on the cloud GPU:
   - **Backbone:** ResNet-50 with **Generalized Mean Pooling (GeM)** (from 1st place APTOS solution).
   - **Loss:** `SmoothL1Loss` (continuous ordinal regression).
   - **Optimizer:** AdamW + Cosine Annealing Learning Rate + Mixed Precision (`torch.cuda.amp` fp16).
   - **Evaluation:** Quadratic Weighted Kappa (QWK) optimized via Nelder-Mead simplex search.
3. **Training Duration:** ~12 to 15 minutes for 12 epochs on Kaggle T4 GPU.
4. You will see progress logs like:
   ```text
   Epoch [01/12] (58.2s) | Train Loss: 0.3421 | Val QWK (Optimized): 0.7314
   ...
   Epoch [10/12] (56.8s) | Train Loss: 0.1102 | Val QWK (Optimized): 0.8540
   >>> Best Model Checkpoint Saved!
   ```

---

### Step 5: Download the Output Models

When training completes, look at the right sidebar under **Output** (`/kaggle/working`):

1. You will see:
   - `retinasight_resnet50.onnx` _(89.6 MB production ONNX graph with dynamic batch axis)_
   - `retinasight_resnet50.pth` _(PyTorch checkpoint)_
2. Click the three dots `...` next to `retinasight_resnet50.onnx` and click **Download**.
3. Click the three dots `...` next to `retinasight_resnet50.pth` and click **Download**.

---

### Step 6: Deploy into Your Local Project

1. Move both downloaded files into your local project root directory:
   - `d:\SIH2026\retinasight\retinasight_resnet50.onnx`
   - `d:\SIH2026\retinasight\retinasight_resnet50.pth`
2. Restart your local RetinaSight server:
   - Double-click [`start_retinasight.bat`](file:///d:/SIH2026/retinasight/start_retinasight.bat)
3. Both the FastAPI API (`http://127.0.0.1:8000`) and the Web Dashboard (`http://localhost:5173`) will instantly use the new high-accuracy GPU-trained model!
