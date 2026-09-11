# RetinaSight Mobile Stub — Primary Health Centre (PHC) Capture App

> **SIH 2026 • Problem Statement ID 26038 • Team OnFocus**  
> Explainable AI Diabetic Retinopathy Screening Pipeline for Rural India

---

## Architecture Overview

The `mobile_stub` is a lightweight, cross-platform Flutter application tailored for frontline healthcare workers (ASHA workers, ANMs, ophthalmic assistants) at rural Primary Health Centres (PHCs).

It implements the MVVM layered architecture:
```
mobile_stub/
├── pubspec.yaml                 # Dependencies (http, image_picker, intl)
├── analysis_options.yaml        # Flutter linter configuration
├── android/
│   └── app/src/main/AndroidManifest.xml  # Camera & Internet permissions
└── lib/
    ├── main.dart                # App entrypoint and theme setup
    ├── data/
    │   ├── models/
    │   │   └── prediction_result.dart    # ICDR grading & telemetry model
    │   └── services/
    │       └── screening_api_service.dart # HTTP client + offline queue logger
    └── ui/
        ├── view_models/
        │   └── screening_view_model.dart # State management & camera binding
        └── views/
            └── screening_screen.dart     # Single-screen acquisition interface
```

---

## Features

1. **Hardware Camera Integration**:
   - Captures 45° fundus photos via attached smartphone ophthalmoscope lens (e.g., Remidio / Volk iNview).
   - In-app visual alignment guide: Circular reticle to help center the macula and optic disc.

2. **Quality Gate Rejection Feedback**:
   - Evaluates image quality immediately via the Stage 1 FastAPI preprocessing gate.
   - If blurry ($\sigma^2 < 50.0$) or under-exposed ($I < 35.0$), displays clear guidance: *"Hold camera steady, ensure pupil dilation"*.

3. **Offline Sync Queue Simulation**:
   - Rural areas often experience intermittent connectivity. When offline or unreachable, the app intercepts network exceptions, logs a structured queue record (`[Offline Sync Queue] Queued for sync: Patient ID: ABHA-...`), and caches the request locally.

4. **Inference & Explainability Integration**:
   - Sends multipart image upload to `POST /predict`.
   - Displays ICDR Severity Grade (0–4), confidence %, and Grad-CAM verification status.

---

## How to Run

### Prerequisites
- Flutter SDK (v3.0.0+) installed and configured in PATH.
- Android Studio / Xcode or an active device/emulator.

### Commands
```bash
# Navigate to mobile stub directory
cd mobile_stub

# Fetch dependencies
flutter pub get

# Run on connected Android device / emulator
flutter run
```

*Note: When running on an Android emulator, the app automatically routes requests to `http://10.0.2.2:8000/predict` to communicate with the host machine's FastAPI server.*
