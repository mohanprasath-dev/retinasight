# RetinaSight — MATLAB & Simulink Interoperability Suite

> **Smart India Hackathon 2026 | Problem Statement ID: 26038 (MathWorks)**  
> **Topic:** Explainable AI for Diabetic Retinopathy Screening in Rural India  
> **Team:** OnFocus | **Developer:** Mohan Prasath P

---

## 1. Executive Summary
This directory contains the official MathWorks MATLAB & Simulink implementation for **RetinaSight**, demonstrating 100% interoperability between our open-source edge deployment pipeline and the native MathWorks ecosystem:
- **Medical Image Processing**: Pre-inference retinal quality gate (Laplacian variance, illumination, FOV coverage), green-channel CLAHE, and morphological vascular tree segmentation.
- **Deep Learning / ONNX Integration**: Seamless execution of the fine-tuned ResNet-50 network (`retinasight_resnet50.onnx`) directly within MATLAB.
- **Explainable AI (XAI)**: Native Grad-CAM visual attention overlays strictly bounded within the patient's retinal field of view.
- **Simulink Capacity Modeling**: Dynamic continuous-time block simulation (`retinasight_capacity_model.slx`) modeling rural Primary Health Centre (PHC) patient intake, AI pre-screening pass rates, and tertiary hospital tele-ophthalmology escalation queues.

---

## 2. Quick Start: Launching in MATLAB

### Method A: One-Click Windows Launcher (Easiest)
Double-click the launcher in the root directory:
```text
run_matlab_pipeline.bat
```
This automatically launches MATLAB R2026a, navigates to the `matlab/` directory, opens `retinasight_pipeline.m`, and executes the full diagnostic pipeline.

### Method B: Native MATLAB Command Window
1. Open MATLAB R2026a.
2. In the MATLAB Command Window, enter:
   ```matlab
   % Navigate to the matlab folder of your cloned repository:
   cd matlab; 
   results = retinasight_pipeline();
   ```
3. To test with specific patient fundus images:
   ```matlab
   % Grade 0 (Normal / No DR)
   results = retinasight_pipeline('../frontend/public/samples/sample_messidor_grade0.png');

   % Grade 2 (Moderate DR)
   results = retinasight_pipeline('../frontend/public/samples/sample_aptos_grade2.png');

   % Grade 3 (Severe DR)
   results = retinasight_pipeline('../frontend/public/samples/sample_idrid_grade3.png');

   % Grade 4 (Proliferative DR)
   results = retinasight_pipeline('../frontend/public/samples/sample_proliferative_grade4.png');
   ```

---

## 3. Simulink District Rollout & Capacity Model

Open and simulate the dynamic rural healthcare triage model:
```matlab
% In MATLAB Command Window:
open_system('retinasight_capacity_model');
sim('retinasight_capacity_model');
```

### Model Architecture:
1. **Daily Patient Intake**: Constant source (180 patients/day per rural PHC block cluster).
2. **Quality Gate Pass**: 88% pass rate to edge inference; 12% directed to real-time image retake.
3. **Primary Care Cumulative Screened**: Integrator tracking monthly throughput.
4. **DR Referral Triage**: 32% positive screen (Grades 2–4) referred to tertiary hospital.
5. **Tertiary Hospital Review Capacity**: Saturation block modeling expert ophthalmologist capacity (45 reviews/day limit).
6. **Scopes & Outports**: Continuous time series of screening throughput vs specialist consultation backlog.

---

## 4. Pipeline Stages

| Stage | Operation | Metric / Clinical Threshold |
| :--- | :--- | :--- |
| **Stage 1** | Pre-Inference Quality Assessment | Blur Variance (Laplacian $\ge 40.0$), Illumination ($30.0 - 225.0$), Retinal FOV ($\ge 20\%$) |
| **Stage 2** | Green-Channel CLAHE | Rayleigh distribution contrast enhancement + edge-preserving bilateral denoising |
| **Stage 3** | Anatomical Segmentation | Bottom-hat tubular vessel isolation + Optic Disc ROI circular localization |
| **Stage 4** | Deep Learning ResNet-50 | ONNX DAGNetwork inference with softmax multi-class probabilities |
| **Stage 5** | Grad-CAM Explainability | Jet colormap gradient activation map bounded within retinal mask |

---

## 5. Optional MathWorks Add-Ons
`retinasight_pipeline.m` runs cleanly out of the box on **base MATLAB R2026a**. For full hardware-accelerated deep learning execution:
1. In MATLAB, click **Home** tab $\rightarrow$ **Add-Ons** $\rightarrow$ **Get Add-Ons**.
2. Search and install:
   - **Deep Learning Toolbox**
   - **Deep Learning Toolbox Converter for ONNX Model Format**
   - **Image Processing Toolbox**
