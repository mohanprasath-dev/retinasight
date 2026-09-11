# RetinaSight

> Explainable AI pipeline for Diabetic Retinopathy screening in rural India — CNN severity grading (ICDR 0-4) with Grad-CAM lesion explainability, built for SIH 2026, PS 26038, Team OnFocus

---

## Overview

Diabetic Retinopathy (DR) is a leading cause of preventable vision impairment and blindness. In rural India, mass screening is hindered by a critical shortage of ophthalmologists (~1 per 100,000 population) and variable-quality fundus imaging from low-cost portable cameras. 

**RetinaSight** provides an end-to-end, explainable screening pipeline:
1. **Automated Quality Gate**: Evaluates blur, illumination, and centering before inference to provide real-time recapture guidance to Primary Health Centre (PHC) staff.
2. **Recoverable Enhancement**: Normalizes contrast using CLAHE and bilateral filtering.
3. **5-Class Severity Classification**: Deep CNN (ResNet50) fine-tuned on the APTOS 2019 dataset to predict ICDR severity (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative DR).
4. **Grad-CAM Lesion Explainability**: Overlays attention heatmaps indicating pathological markers (microaneurysms, hemorrhages, hard/soft exudates) so reviewing specialists can trust and verify triage recommendations in under 30 seconds.

---

## Tech Stack

- **Backend & ML Pipeline**: Python 3.10+, PyTorch, torchvision, OpenCV, ONNX Runtime, NumPy
- **API Service**: FastAPI, Uvicorn
- **Web Interface**: React (Vite), Tailwind CSS
- **Mobile Capture Stub**: Flutter
- **Model Training**: Kaggle Notebooks (NVIDIA GPU environment with direct APTOS 2019 dataset access)

---

## Note on Tech Stack & MATLAB Substitution

> [!NOTE]
> **Deviation from Official PS (ID 26038)**:
> - The official problem statement mentions MATLAB (Deep Learning Toolbox, Simulink, Medical Imaging Toolbox).
> - Due to licensing availability and rural deployment agility, this implementation uses **PyTorch + OpenCV + FastAPI + React** instead.
> - Stage 6 / Simulink deployment planning is provided as an architectural documentation diagram and mathematical capacity model rather than executable `.slx` code.
> - This substitution is disclosed upfront for transparency.

---

## Folder Structure

```
retinasight/
├── preprocessing.py       # Stage 1-3: Quality check, CLAHE enhancement, segmentation stub
├── train_dr_classifier.py # Stage 4: ResNet50 training pipeline (run on Kaggle GPU)
├── gradcam.py              # Stage 5: Grad-CAM heatmap generation & visual overlay
├── main.py                 # Stage 6: FastAPI REST service exposing /predict
├── frontend/               # Stage 7: React (Vite) specialist review web app
├── mobile_stub/            # Stage 8: Flutter camera capture & PHC triage stub
├── docs/                   # Architectural specs, PRD, AGENT context, and rollout diagrams
│   ├── AGENT.md
│   ├── PRD.md
│   ├── retinasight_antigravity_prompts.json
│   └── simulink_mockup.png
├── LICENSE                 # MIT License
├── README.md               # Project documentation
└── .gitignore              # Environment and build exclusion rules
```

---

## Setup & Installation Instructions (Placeholder)

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Git

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/mohanprasath-dev/retinasight.git
cd retinasight

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# Install dependencies (once requirements.txt is generated)
pip install -r requirements.txt

# Run FastAPI backend
uvicorn main:app --reload --port 8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
Copyright (c) 2026 Mohan Prasath P (Team OnFocus).
