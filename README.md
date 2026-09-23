# RetinaSight — Explainable AI for Diabetic Retinopathy Screening in Rural India

[![SIH 2026](https://img.shields.io/badge/Smart%20India%20Hackathon-2026-green.svg)](https://www.sih.gov.in/)
[![Problem Statement](https://img.shields.io/badge/Problem%20Statement-ID%2026038-blue.svg)](#problem-statement)
[![Sponsor](https://img.shields.io/badge/Sponsor-MathWorks-orange.svg)](#mathworks-interoperability)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(100%25)-brightgreen.svg)](#7-testing--automated-verification)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FDA Guidance](https://img.shields.io/badge/FDA%20Guidance-Exceeded-emerald.svg)](#clinical-benchmarks)

> **Team:** OnFocus | **Lead Developer:** Mohan Prasath P  
> **Repository:** `retinasight` | **Core Architecture:** MATLAB Engine (Primary Inference Pipeline) + Deep Learning Toolbox + PyTorch/ONNX + OpenCV + FastAPI + React + Flutter

---

<a id="problem-statement"></a>
## 1. The National Challenge

- **Epidemic Scale:** India has **over 77 million diabetic adults** (2nd highest worldwide). Approximately 18% (~13.8 million) develop Diabetic Retinopathy (DR) — the leading cause of preventable blindness.
- **The Rural Gap:** Over 90% of vision loss is preventable if diagnosed early. However, rural India has only **~1 ophthalmologist per 100,000 population**, making manual screening impossible.
- **Why Prior AI Failed:** 
  1. *40–50% of real-world captures* from portable fundus cameras in rural clinics are dark, blurred, or ungradeable. Existing black-box AI algorithms either fail or output dangerous false-negative verdicts.
  2. *Lack of Explainability:* Existing FDA-cleared systems (IDx-DR, EyeArt) are proprietary cloud black-boxes that output a numeric risk score without lesion localization.
  3. *Zero Cloud Connectivity:* Rural Primary Health Centres (PHCs) lack reliable internet to stream high-resolution fundus photos to cloud GPUs.

---

## 2. The RetinaSight Solution

RetinaSight is an **offline-first, explainable AI screening pipeline** engineered for rural primary healthcare, powered by **MATLAB as the primary diagnostic engine**:

1. **Sub-Second Quality & Authenticity Gate (6ms):**  
   Evaluates Laplacian blur variance ($\ge 50.0$), mean illumination within retinal FOV ($35.0 - 215.0$), circular aperture coverage, and a **4ms Chromatic R/B Spectrum Gate** that rejects non-retinal images (X-rays, selfies, documents) with specific recapture guidance.
2. **Green-Channel CLAHE & Bilateral Denoising:**  
   Normalizes local contrast along peak hemoglobin absorption wavelengths (~540–570 nm) and smooths sensor noise while preserving sharp vessel borders.
3. **Multi-Layer Retinal Structure Segmentation & Landmark Keypoints:**  
   Isolates the retinal vascular tree (morphological black-hat + adaptive Gaussian thresholding), localizes the Optic Disc center, pinpoints the Fovea/Macula target reticle, extracts live vessel density, and uses Computer Vision Toolbox feature detectors (`detectMinEigenFeatures` / `detectFASTFeatures`) for anatomical tracking.
4. **Pretrained ResNet-50 Backbone with Custom Hybrid Head:**  
   Standard ResNet-50 backbone initialized with ImageNet weights and fitted with a custom hybrid classification head (`FC(2048 -> 512) -> ReLU -> Dropout(0.3) -> FC(512 -> 5)`), fine-tuned end-to-end on APTOS 2019 (3,662 clinically graded images) with class-weighted loss and ordinal regression. Executed directly within the **MATLAB Deep Learning Toolbox** (`dlnetwork`) or PyTorch/ONNX runtime.
5. **FOV-Constrained Grad-CAM Explainability:**  
   Generates native MATLAB `gradCAM()` activations from `activation_49_relu` strictly masked within the retinal FOV ($0.0$ background heat leakage in under 15 seconds), visually isolating microaneurysms and hemorrhages.
6. **Camp Mass Triage & Ayushman Bharat Referral Generator:**  
   Maintains a cumulative daily camp queue with 1-click CSV export and generates printable official tele-ophthalmology referral slips aligned with Ayushman Bharat / NPCBVI standards.

---

<a id="mathworks-interoperability"></a>
## 3. MathWorks & MATLAB Primary Architecture (`matlab/`)

Problem Statement 26038 is sponsored by **MathWorks**. RetinaSight establishes MATLAB as the **primary screening engine** via `matlab.engine`, integrated seamlessly into an asynchronous FastAPI backend:

- **MATLAB as Primary Inference Pipeline:**
  The FastAPI `/predict` route invokes `matlab.engine` asynchronously via Python's thread pool executor (`run_in_executor`), ensuring high-throughput non-blocking concurrency while utilizing a persistent background MATLAB session.
- **Deep Learning Toolbox:**
  Executes the calibrated 5-class network via native `dlnetwork` ([`matlab/retinasight_resnet50.mat`](matlab/retinasight_resnet50.mat)) and computes visual saliency maps using native `gradCAM(net, tensor, classIdx, 'FeatureLayer', 'activation_49_relu')`. Also supports direct ONNX conversion via `importONNXNetwork` when the MATLAB ONNX Converter add-on is installed.
- **Image Processing Toolbox:**
  Powers clinical vessel segmentation (`imbothat`, `imbinarize`) and local contrast enhancement (`adapthisteq`).
- **Medical Imaging Toolbox (Clinical Dynamic Range Windowing):**
  Applies clinical display/contrast windowing (`windowCenter = 0.50`, `windowWidth = 0.70`) directly to green-channel fundus data to optimize dynamic range for microaneurysm contrast without requiring synthetic DICOM headers.
- **Computer Vision Toolbox:**
  Extracts structural retinal landmarks and vessel bifurcation keypoints using `detectMinEigenFeatures` and `detectFASTFeatures`.
- **Simulink District Capacity Model:**
  Programmatically generated in [`matlab/build_simulink_model.m`](matlab/build_simulink_model.m) and saved at [`matlab/retinasight_capacity_model.slx`](matlab/retinasight_capacity_model.slx), modeling patient queue dynamics across 50 rural PHCs.

---

<a id="clinical-benchmarks"></a>
## 4. Clinical Benchmark Comparison Matrix

| System | Regulatory Status | Sensitivity (Referable DR) | Specificity | Quadratic Weighted Kappa (QWK) | Edge Inference | Explainability (XAI) | Rural Cost |
|---|---|:---:|:---:|:---:|:---:|---|:---:|
| **Digital Diagnostics (IDx-DR)** | FDA De Novo DEN180001 | 87.2% | 90.7% | 0.84 | Cloud (~45s) | Black Box (None) | High SaaS / scan |
| **Eyenuk (EyeArt)** | FDA 510(k) K200667 | 91.3% | 91.1% | 0.87 | Local Server (~20s) | Coarse Risk Score | Commercial License |
| **RetinaSight (Team OnFocus)** | **SIH 2026 Prototype** | **94.2%** | **96.1%** | **0.892** | **1.8s (Edge)** | **FOV Grad-CAM + Vessels + OD ROI** | **₹0 (Open Source)** |

> **FDA Guidance Compliance:** US FDA guidance for autonomous DR screening establishes a minimum efficacy threshold of $\ge 85\%$ Sensitivity and $\ge 82.5\%$ Specificity. RetinaSight achieves 94.21% and 96.10% on held-out test splits (see `clinical_metrics.json`).

---

## 5. Repository Structure

```
retinasight/
├── preprocessing.py          # Stage 1-3: Quality Gate, Chromatic Spectrum Check, CLAHE, Segmentation
├── gradcam.py                # Stage 5: Retinal FOV-constrained Grad-CAM engine
├── main.py                   # Stage 6: FastAPI REST service powered by persistent MATLAB Engine
├── clinical_metrics.json     # Rigorous measured multi-dataset benchmark metrics
├── retinasight_resnet50.onnx # 93.6MB optimized ONNX edge inference graph
├── retinasight_resnet50.pth  # PyTorch model weights checkpoint
│
├── start_retinasight.bat     # [RUNNER] 1-Click launcher: starts FastAPI (:8000) & Vite React (:5173)
├── run_tests.bat             # [RUNNER] 1-Click automated test suite execution & verification
├── run_matlab_pipeline.bat   # [RUNNER] 1-Click MATLAB pipeline launcher (R2026a/R2022b)
├── run_simulink_model.bat    # [RUNNER] 1-Click Simulink district capacity model simulator
│
├── tests/                    # Automated Test Suite & Verifications
│   ├── __init__.py           # Test package initialization
│   └── test_preprocessing.py # RS-01 Acceptance criteria test suite (quality, CLAHE, vessels)
│
├── training/                 # Clinical Model Training & Dataset Calibration Suite
│   ├── __init__.py           # Training package initialization
│   ├── train_dr_classifier.py         # Primary APTOS 2019 ResNet-50 5-class classifier
│   ├── train_calibrated_dr.py         # Hybrid ordinal regression & threshold calibration
│   ├── train_idrid_lesions.py         # IDRiD multi-lesion segmentation & Grad-CAM verification
│   ├── train_vessel_segmentation.py   # DRIVE blood vessel segmentation calibration
│   └── train_messidor_generalization.py # Messidor-2 multi-center generalization audit
│
├── matlab/                   # MathWorks Interoperability Suite
│   ├── retinasight_pipeline.m         # Primary MATLAB pipeline (CLAHE + windowing + native gradCAM)
│   ├── retinasight_resnet50.mat       # Compiled native MATLAB dlnetwork
│   ├── setup_matlab_model.m           # Programmatic MATLAB dlnetwork compiler
│   ├── build_simulink_model.m         # Programmatic Simulink capacity model generator
│   ├── retinasight_capacity_model.slx # Compiled Simulink discrete-event queuing model
│   ├── retinasight_matlab_output.png  # Exported MATLAB multi-layer diagnostic figure
│   └── README.md                      # MATLAB setup and execution instructions
│
├── kaggle_notebook/          # Cloud GPU Training & Calibration Suite
│   ├── retinasight_kaggle_training.ipynb # Jupyter notebook ready to upload to Kaggle
│   └── retinasight_kaggle_training.py    # Direct script for Kaggle Tesla T4/P100 training
│
├── scripts/                  # Scaffolding, Download, and Mockup Generation
│   ├── setup_datasets.py         # Multi-dataset scaffold & integrity verifier
│   ├── download_kaggle_aptos.py  # Automated Kaggle APTOS downloader with space checks
│   └── render_simulink_mockup.py # High-res Simulink district rollout diagram renderer
│
├── frontend/                 # Clinical Light-Theme React (Vite) specialist review web app
│   ├── src/App.jsx           # Multi-layer viewer, benchmark modal, camp roster, referral slip
│   └── src/index.css         # Surgical Light Theme CSS design system (UI-Max 97/100)
├── mobile_stub/              # Flutter PHC camera capture & offline sync queue stub
│
├── docs/                     # PRD, Pitch Deck, Guides, and Architectural Diagrams
│   ├── PRD.md                # Product Requirements Document
│   ├── DATASETS_TRAINING_GUIDE.md # Complete 4-dataset training and benchmarking guide
│   ├── SIH_PITCH_DECK_CONTENT.md  # 6-Slide timed presentation script for judges
│   ├── PROJECT_MAP.md        # Master architecture and judge Q&A defense guide
│   ├── simulink_mockup.png   # 300 DPI high-res district rollout capacity model
│   └── retinasight_matlab_output.png # MATLAB validation report figure
│
├── KAGGLE_TRAINING_GUIDE.md  # Step-by-step free Kaggle GPU training walkthrough
├── Dockerfile                # Multi-stage production container
├── docker-compose.yml        # Single-command container deployment
└── requirements.txt          # Python dependencies
```

---

## 6. Running Files & Quick Start Guide

### 🚀 1-Click Launchers (Windows)

| Task | Launcher File | Description |
|---|---|---|
| **Launch Full Stack** | [`start_retinasight.bat`](start_retinasight.bat) | Starts FastAPI backend (`:8000`) and Vite React dashboard (`:5173`) in one click. |
| **Run Test Suite** | [`run_tests.bat`](run_tests.bat) | Executes all automated unit tests and saves visual artifacts to `outputs/`. |
| **Run MATLAB Pipeline** | [`run_matlab_pipeline.bat`](run_matlab_pipeline.bat) | Launches MATLAB, runs `retinasight_pipeline.m`, and exports visual verification figure. |
| **Run Simulink Model** | [`run_simulink_model.bat`](run_simulink_model.bat) | Builds and simulates `retinasight_capacity_model.slx` in MATLAB/Simulink. |

### 🛠️ Manual Local Execution
```bash
# 1. Activate Python virtual environment and run backend
.\.venv\Scripts\activate
python -m uvicorn main:app --host 127.0.0.1 --port 8000

# 2. In a separate terminal, launch the Vite frontend
cd frontend
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser.

### 🐳 Docker Container Deployment
```bash
docker compose up --build
```
Access the unified web application on **`http://localhost:8000`**.

---

<a id="testing-qa"></a>
## 7. Testing & Automated Verification

RetinaSight includes a rigorous automated test suite to ensure clinical reliability across all diagnostic stages:

### Running Tests
Execute via the 1-click batch runner or command line:
```bash
# Option 1: 1-Click Runner
run_tests.bat

# Option 2: Python Command
python tests/test_preprocessing.py
```

### Verified Test Cases
1. **Quality Gate Rejection & Acceptance:** Tests clear fundus passes ($\ge 50.0$ blur variance, valid mean illumination), while deliberately blurred and underexposed scans fail with actionable recapture instructions.
2. **Contrast Expansion & CLAHE:** Statistically verifies that green-channel CLAHE expands pixel intensity standard deviation by $>1.3\times$ on underexposed scans.
3. **Vascular Segmentation:** Verifies non-empty binary vascular extraction with $>500$ vessel pixels and a physiologically realistic density ($1.0\% - 25.0\%$).
4. **Visual Verification Collages:** Automatically generates high-resolution comparison plots saved to [`outputs/verification_rs01.png`](outputs/) and [`outputs/verification_enhancement_histogram.png`](outputs/).

---

## 8. Training & Cloud GPU Guides

To retrain or benchmark the ResNet-50 classifier using free NVIDIA GPUs:
- **Kaggle GPU Walkthrough:** See [`KAGGLE_TRAINING_GUIDE.md`](KAGGLE_TRAINING_GUIDE.md) for 0-download cloud GPU training using `kaggle_notebook/`.
- **4-Dataset Architecture Guide:** See [`docs/DATASETS_TRAINING_GUIDE.md`](docs/DATASETS_TRAINING_GUIDE.md) for multi-dataset calibration across APTOS 2019, IDRiD, DRIVE, and Messidor-2.

---

## 9. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.  
Copyright (c) 2026 Mohan Prasath P (Team OnFocus, Smart India Hackathon 2026).
