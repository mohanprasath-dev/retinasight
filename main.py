"""
RetinaSight — FastAPI REST Service (Stage 6)
SIH 2026, PS ID 26038, Team OnFocus

Exposes:
- POST /predict: Accepts multipart fundus image, executes Quality Gate -> CLAHE -> ONNX inference -> Grad-CAM.
- GET /health: Healthcheck and pipeline status.
- Static mount /outputs: Serves generated Grad-CAM heatmap overlays.
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import time
from typing import Dict, List, Optional
import uuid

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as transforms

import gradcam
import preprocessing
import train_dr_classifier


# ------------------------------------------------------------------------------
# 1. Constants & Directory Paths
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
HEATMAPS_DIR = OUTPUTS_DIR / "heatmaps"
HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ONNX_PATH = BASE_DIR / "retinasight_resnet50.onnx"
MODEL_PTH_PATH = BASE_DIR / "retinasight_resnet50.pth"

ICDR_CLASSES = {
	0: "No DR",
	1: "Mild",
	2: "Moderate",
	3: "Severe",
	4: "Proliferative DR",
}

# ImageNet transform for ONNX input
ONNX_TRANSFORM = transforms.Compose([
	transforms.ToPILImage(),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Global ONNX runtime session and PyTorch model
onnx_session: Optional[ort.InferenceSession] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# 2. Application Lifespan
# ------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Initialize ONNX runtime session on startup and perform warm-up."""
	global onnx_session
	print("=" * 70)
	print("Starting RetinaSight FastAPI Service...")
	print(f"Device: {device}")

	if MODEL_ONNX_PATH.exists():
		print(f"[ONNX] Loading inference session from: {MODEL_ONNX_PATH}")
		onnx_session = ort.InferenceSession(str(MODEL_ONNX_PATH), providers=["CPUExecutionProvider"])
		# Warm-up inference
		input_name = onnx_session.get_inputs()[0].name
		dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
		_ = onnx_session.run(None, {input_name: dummy})
		print("[ONNX] Inference session initialized and warmed up successfully.")
	else:
		print(f"[WARNING] ONNX model not found at {MODEL_ONNX_PATH}. Pipeline will load PyTorch fallback.")

	yield
	print("Shutting down RetinaSight FastAPI Service...")


app = FastAPI(
	title="RetinaSight AI Diagnostic API",
	description="Explainable AI Diabetic Retinopathy screening pipeline for rural India (SIH 2026, PS 26038, Team OnFocus)",
	version="1.0.0",
	lifespan=lifespan,
)

# ------------------------------------------------------------------------------
# 3. Middleware & Static Files
# ------------------------------------------------------------------------------
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Mount outputs folder for static heatmap access
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


# ------------------------------------------------------------------------------
# 4. Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
def root():
	return {
		"name": "RetinaSight API",
		"event": "Smart India Hackathon 2026",
		"problem_statement": "PS ID 26038",
		"team": "OnFocus",
		"status": "online",
		"endpoints": {
			"predict": "POST /predict",
			"health": "GET /health",
			"docs": "/docs",
		},
	}


@app.get("/health")
def health_check():
	return {
		"status": "healthy",
		"device": str(device),
		"onnx_model_available": MODEL_ONNX_PATH.exists(),
		"pth_checkpoint_available": MODEL_PTH_PATH.exists(),
		"outputs_directory": str(OUTPUTS_DIR),
	}


@app.post("/predict")
async def predict_retinopathy(file: UploadFile = File(...)):
	"""Full Stage 1-5 Diagnostic Pipeline:
	1. Image Decode: Reads multipart image upload.
	2. Quality Check: Evaluates blur, illumination, and centering.
	   - If FAIL -> Halts and returns {status: 'reject', reason: str, reasons: list}.
	3. Enhancement: Applies green-channel CLAHE + bilateral filtering.
	4. Deep Classification: Runs ONNX Runtime session for ICDR 0-4 severity.
	5. Grad-CAM Explainability: Generates retinal FOV-constrained heatmap overlay.
	6. Serves overlay via /outputs/heatmaps/ and returns full diagnostic JSON.
	"""
	t_start = time.time()

	# 1. Read and decode image
	contents = await file.read()
	if not contents:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file payload uploaded.")

	nparr = np.frombuffer(contents, np.uint8)
	image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
	if image_bgr is None:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Could not decode image. Please upload a valid PNG, JPG, or TIFF retinal capture.",
		)

	# 2. Stage 1: Quality Check Gate
	qc_result = preprocessing.quality_check(image_bgr)
	if not qc_result["passed"]:
		primary_reason = qc_result["reasons"][0] if qc_result["reasons"] else "Image quality below diagnostic threshold."
		return {
			"status": "reject",
			"reason": primary_reason,
			"reasons": qc_result["reasons"],
			"quality_metrics": qc_result["metrics"],
			"processing_time_seconds": round(time.time() - t_start, 3),
		}

	# 3. Stage 2: Recoverable Enhancement
	enhanced_bgr = preprocessing.enhance(image_bgr)

	# 4. Stage 4: Deep Classification via ONNX Runtime
	preprocessed_rgb = train_dr_classifier.apply_retinal_preprocessing(enhanced_bgr, target_size=(256, 256))
	tensor = ONNX_TRANSFORM(preprocessed_rgb).unsqueeze(0)
	tensor_np = tensor.numpy().astype(np.float32)

	global onnx_session
	if onnx_session is not None:
		input_name = onnx_session.get_inputs()[0].name
		ort_outs = onnx_session.run(None, {input_name: tensor_np})
		logits = ort_outs[0][0]
	else:
		# Fallback PyTorch inference
		model = gradcam.load_classifier_model(MODEL_PTH_PATH, device)
		with torch.no_grad():
			out = model(tensor.to(device))
			logits = out[0].cpu().numpy()

	# Compute Softmax probabilities
	exp_logits = np.exp(logits - np.max(logits))
	probs = exp_logits / np.sum(exp_logits)

	severity = int(np.argmax(probs))
	confidence = float(probs[severity])
	severity_label = ICDR_CLASSES.get(severity, f"Class {severity}")

	# 5. Stage 5: Grad-CAM Explainability Generation
	overlay, cam_raw, _, _, _ = gradcam.generate_heatmap(
		image=enhanced_bgr,
		model=MODEL_PTH_PATH if MODEL_PTH_PATH.exists() else MODEL_ONNX_PATH,
		target_layer="layer4",
		target_class=severity,
		alpha=0.5,
		device=device,
	)

	# Save heatmap overlay to static outputs directory
	file_id = f"{uuid.uuid4().hex[:12]}_{int(time.time())}"
	heatmap_filename = f"heatmap_{file_id}.png"
	heatmap_save_path = HEATMAPS_DIR / heatmap_filename
	cv2.imwrite(str(heatmap_save_path), overlay)

	grad_cam_url = f"/outputs/heatmaps/{heatmap_filename}"
	total_time = round(time.time() - t_start, 3)

	return {
		"status": "accept",
		"severity": severity,
		"severity_label": severity_label,
		"confidence": round(confidence, 4),
		"grad_cam_url": grad_cam_url,
		"class_probabilities": {
			ICDR_CLASSES[i]: round(float(probs[i]), 4) for i in range(len(ICDR_CLASSES))
		},
		"quality_metrics": qc_result["metrics"],
		"processing_time_seconds": total_time,
	}


if __name__ == "__main__":
	import uvicorn
	uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
