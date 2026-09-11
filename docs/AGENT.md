# AGENT.md — RetinaSight (SIH26038, Team OnFocus)

Project-local Antigravity context. Supplements global `~/.gemini/GEMINI.md`.

## Project
Explainable AI DR (Diabetic Retinopathy) screening pipeline for rural India.
SIH 2026, PS ID 26038, Software edition. Idea PPT already submitted.
This repo = pre-build prototype so the 36hr national round (if selected) is
integration + polish, not cold-start.

## Tech stack (deviates from official PS — note this explicitly in pitch)
- **Official PS asks for MATLAB** (Deep Learning Toolbox, Simulink, Medical Imaging Toolbox).
- **This build uses PyTorch + OpenCV + FastAPI + React** instead — MATLAB not
  available to the team. State this substitution upfront if a judge asks; don't
  imply MATLAB was used.
- Backend: Python 3.10+, FastAPI, PyTorch, OpenCV, onnxruntime
- Frontend: React (Vite)
- Mobile stub: Flutter
- Training: Kaggle Notebooks (free GPU, direct APTOS dataset access — no local download)
- No MATLAB, no Simulink anywhere in this repo. Stage 6 (Simulink) is a
  documentation-only diagram, not code.

## Folder structure
```
retinasight/
  preprocessing.py       # Stage 1-3
  train_dr_classifier.py # Stage 4 — run on Kaggle, not locally
  gradcam.py              # Stage 5
  main.py                 # FastAPI wrapper, Stage 4-5 -> /predict
  frontend/                # React app
  mobile_stub/              # Flutter capture app
  docs/
    simulink_mockup.png
    PRD.md
    AGENT.md
```

## Build order (do not deviate without checking with Mohan)
1. RS-01 Preprocessing
2. RS-02 CNN training script (Kaggle-only, do not run in Antigravity)
3. RS-03 Grad-CAM
4. RS-04 FastAPI wrapper
5. RS-05 React UI
6. RS-06 Flutter stub (lowest priority)
7. RS-07 Simulink mockup diagram (lowest priority)

Full prompt set: `retinasight_antigravity_prompts.json` (RS-01 through RS-07,
JSON format, id/title/instruction/acceptance/note fields).

## Rules
- Screenshot-verify every stage's output before moving to the next. Do not
  trust agent self-report of "done" — this burned Mohan on the ARI project.
  Confirm live, working output before claiming completion.
- Dataset (APTOS 2019) is NOT present in this repo — too large to upload.
  Training happens on Kaggle. Only small sample images (if any) live locally
  for pipeline smoke-testing.
- Flag unknowns as `[NEEDS INPUT: ...]` instead of guessing — never
  fabricate accuracy numbers, dataset stats, or clinical claims not in the
  spec doc or official PS.
- Model escalation: Gemini Flash default → Claude Sonnet after 2 failed
  attempts → Claude Opus reserved for hard single-prompt tasks only
  (e.g. Grad-CAM math, class-imbalance training logic).
- Reference existing skills where relevant: `systematic-debugging`,
  `gstack-qa`, `gstack-review`, `frontend-design`, `webapp-testing`
  (Playwright, for React end-to-end verification).

## Known constraints
- Solo build (Mohan only, OnFocus team — not TaskDrift).
- A working prototype is a bonus at idea/internal stage, not mandatory —
  don't over-invest at the cost of the PPT/pitch quality.

## Repo
- Name: `retinasight`
- License: MIT
- Description: Explainable AI pipeline for Diabetic Retinopathy screening
  in rural India — CNN severity grading (ICDR 0-4) with Grad-CAM lesion
  explainability, built for SIH 2026 (PS 26038, Team OnFocus).
