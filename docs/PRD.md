# PRD — RetinaSight

**Team:** OnFocus | **PS ID:** 26038 | **Event:** Smart India Hackathon 2026
**Status:** Idea round submitted. Pre-building prototype ahead of internal/national round.

## 1. Problem
- India: 77M+ diabetic adults (2nd highest globally, per official PS).
- ~18% develop Diabetic Retinopathy (DR) — leading cause of preventable blindness.
- 90% of DR-related blindness is preventable with early screening (PS-stated).
- ~1 ophthalmologist per 100,000 rural population — manual mass screening infeasible.
- Existing AI tools (IDx-DR, EyeArt, Medios AI): accurate but black-box, and
  degrade on variable-quality images from basic/portable fundus cameras.

## 2. Users
- **PHC health workers** — capture fundus images on basic cameras, need
  immediate accept/reject + recapture feedback.
- **Ophthalmologists / specialists** — review pre-screened, prioritized
  referrals with visual explanation (Grad-CAM), not just a raw score.
- **Health system planners** — need district-scale rollout capacity estimates.

## 3. Scope — MVP (this build)
| # | Stage | Deliverable |
|---|-------|-------------|
| 1 | Quality check | Pass/fail + recapture reason (blur, brightness, centering) |
| 2 | Enhancement | CLAHE + denoise for recoverable low-quality images |
| 3 | Segmentation | Vessel mask (classical CV); microaneurysm/exudate/hemorrhage as stubbed placeholders |
| 4 | DR grading | ResNet50 CNN, ICDR severity 0-4, trained on APTOS 2019 |
| 5 | Explainability | Grad-CAM heatmap overlay, <30s per PS spec |
| 6 | API | FastAPI `/predict` — image in, {severity, confidence, grad_cam_url} out |
| 7 | UI | React upload + result view; Flutter capture stub (mocked sync) |
| 8 | Rollout planning | Static diagram (Simulink math mockup), not executable |

## 4. Explicitly out of scope
- Production database / auth / role-based access
- Full Simulink deployment model (MATLAB not available to team)
- Live offline sync with conflict resolution (mocked only)
- Microaneurysm/exudate/hemorrhage detection beyond placeholder (real model
  needs IDRiD + more time than pre-build window allows)
- Any clinical deployment claims — this is a hackathon prototype, not a
  validated medical device

## 5. Tech stack deviation from official PS
Official PS specifies MATLAB (Deep Learning Toolbox, Simulink, Medical
Imaging Toolbox, Computer Vision Toolbox). This build substitutes:
- MATLAB Deep Learning Toolbox → **PyTorch**
- MATLAB Image/Medical Imaging Toolbox → **OpenCV**
- Simulink → **static diagram/doc**, explicit substitution noted in pitch

Reason: MATLAB/Simulink license not secured for the team (flagged as a
"High severity" risk in the original spec). State this openly if asked —
do not imply MATLAB was used.

## 6. Success metrics (targets from spec, not yet measured)
- Sensitivity ≥ 90%, Specificity ≥ 85% on APTOS/IDRiD test set — **[NEEDS INPUT: not measured yet, depends on Stage 4 training run]**
- Grad-CAM IoU ≥ 0.6 vs IDRiD expert annotations — **[NEEDS INPUT: requires IDRiD, not yet acquired]**
- End-to-end inference < 30 seconds
- Image quality gate rejects <50% of real-world captures — **[NEEDS INPUT: untested on real-world images]**

## 7. Risks (from official PS + spec doc)
| Risk | Severity | Mitigation |
|------|----------|-----------|
| Sub-pixel microaneurysm detection | High | Deferred beyond MVP; stub only |
| Meeting sensitivity/specificity targets | High | Validate on held-out APTOS split; be upfront on real numbers, don't inflate |
| No MATLAB/Simulink license | High | PyTorch/OpenCV substitution, disclosed openly |
| Variable real-world image quality | High | Quality-gate rejects unreadable images before they reach classifier |
| Solo build, limited time before deadline | High | Strict build order (RS-01→RS-07), screenshot-verify each stage, cut RS-06/07 if time runs out |

## 8. Open questions
- **[NEEDS INPUT]** Does internal round require live demo or is PPT + repo link sufficient?

## 9. Repo
- Name: `retinasight`
- License: MIT
- Description: Explainable AI pipeline for Diabetic Retinopathy screening
  in rural India — CNN severity grading (ICDR 0-4) with Grad-CAM lesion
  explainability, built for SIH 2026 (PS 26038, Team OnFocus).
