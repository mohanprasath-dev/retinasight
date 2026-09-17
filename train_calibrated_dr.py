"""
RetinaSight — Balanced Hybrid Ordinal-Classification DR Training & Calibration Pipeline
SIH 2026, PS ID 26038, Team OnFocus

Incorporates winning insights from APTOS 2019 competition (1st place & 5th place solutions):
1. Continuous ordinal regression + multi-class discriminative classification hybrid head.
2. Balanced multi-cohort sampling across APTOS 2019 and IDRiD to eliminate class collapse.
3. Strict alignment with clinical preprocessing (Ben's circular FOV crop + CLAHE).
4. Direct embedding into PyTorch ResNet-50 fc layer for 100% ONNX and Grad-CAM compatibility.
5. Export to production ONNX graph (retinasight_resnet50.onnx).
"""

import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import preprocessing
import train_dr_classifier


ICDR_CLASSES = {
	0: "No DR",
	1: "Mild",
	2: "Moderate",
	3: "Severe",
	4: "Proliferative DR",
}

APTOS_DIR = Path("D:/SIH2026/Datasets/aptos2019")
IDRID_DIR = Path("D:/SIH2026/Datasets/IDRiD/B. Disease Grading")
OUTPUT_PTH = Path("retinasight_resnet50.pth")
OUTPUT_ONNX = Path("retinasight_resnet50.onnx")
CACHE_DIR = Path("cache")


def load_balanced_cohort(samples_per_class: int = 120) -> Tuple[pd.DataFrame, Dict[str, Path]]:
	"""Assemble a balanced multi-cohort dataset across all 5 ICDR severity grades."""
	all_records = []
	id_map = {}

	# 1. Index APTOS 2019
	if APTOS_DIR.exists():
		print(f"[DATASET] Scanning APTOS 2019 at {APTOS_DIR}...")
		csv_file = APTOS_DIR / "train.csv"
		if csv_file.exists():
			df_aptos = pd.read_csv(csv_file)
			for ext in ("*.png", "*.jpg", "*.jpeg"):
				for p in APTOS_DIR.rglob(ext):
					id_map[p.stem] = p

			for _, row in df_aptos.iterrows():
				code = str(row["id_code"])
				if code in id_map:
					all_records.append({"id_code": code, "diagnosis": int(row["diagnosis"]), "source": "APTOS"})

	# 2. Index IDRiD Disease Grading
	if IDRID_DIR.exists():
		print(f"[DATASET] Scanning IDRiD Disease Grading at {IDRID_DIR}...")
		idrid_csv = IDRID_DIR / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"
		idrid_imgs = IDRID_DIR / "1. Original Images" / "a. Training Set"
		if idrid_csv.exists() and idrid_imgs.exists():
			df_idrid = pd.read_csv(idrid_csv)
			for p in idrid_imgs.glob("*.jpg"):
				id_map[p.stem] = p
			for _, row in df_idrid.iterrows():
				code = str(row["Image name"])
				if code in id_map:
					all_records.append({"id_code": code, "diagnosis": int(row["Retinopathy grade"]), "source": "IDRiD"})

	combined_df = pd.DataFrame(all_records)
	print(f"[DATASET] Total indexed images: {len(combined_df)}")
	print(f"[DATASET] Raw distribution:\n{combined_df['diagnosis'].value_counts().sort_index()}")

	# Balanced stratified sampling
	sampled_dfs = []
	for c in range(5):
		sub = combined_df[combined_df["diagnosis"] == c]
		n_sample = min(len(sub), samples_per_class)
		sampled_dfs.append(sub.sample(n=n_sample, random_state=42))

	balanced_df = pd.concat(sampled_dfs).sample(frac=1.0, random_state=42).reset_index(drop=True)
	print(f"\n[BALANCED COHORT] Assembled {len(balanced_df)} samples ({samples_per_class} per class target):")
	print(balanced_df["diagnosis"].value_counts().sort_index())

	return balanced_df, id_map


def extract_or_load_features(
	balanced_df: pd.DataFrame,
	id_map: Dict[str, Path],
	samples_per_class: int,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Extract 2048-dim deep features with clinical preprocessing, or load from cache."""
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	cache_file = CACHE_DIR / f"features_balanced_{samples_per_class}x5.npz"

	if cache_file.exists():
		print(f"[CACHE] Loading cached feature embeddings from: {cache_file}")
		data = np.load(cache_file)
		return data["X"], data["y"]

	print("\n[BACKBONE] Loading ImageNet pretrained ResNet-50 feature extractor...")
	resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
	feat_extractor = nn.Sequential(*list(resnet.children())[:-1])
	feat_extractor.eval()

	tf = transforms.Compose([
		transforms.ToPILImage(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	X, y = [], []
	t0 = time.time()
	total = len(balanced_df)
	print(f"[EXTRACTION] Extracting deep features for {total} images with clinical preprocessing...")

	for i, (_, row) in enumerate(balanced_df.iterrows()):
		img_path = id_map.get(row["id_code"])
		if img_path and img_path.exists():
			img_bgr = cv2.imread(str(img_path))
			if img_bgr is not None:
				# Apply exact clinical preprocessing matching live inference
				enhanced = preprocessing.enhance(img_bgr)
				prep_rgb = train_dr_classifier.apply_retinal_preprocessing(enhanced, target_size=(256, 256))
				tensor = tf(prep_rgb).unsqueeze(0)
				with torch.no_grad():
					feat = feat_extractor(tensor).squeeze().numpy()
				X.append(feat)
				y.append(int(row["diagnosis"]))

		if (i + 1) % 50 == 0 or (i + 1) == total:
			print(f"  -> Extracted {i + 1}/{total} samples ({time.time() - t0:.1f}s)")

	X = np.array(X)
	y = np.array(y)

	np.savez_compressed(cache_file, X=X, y=y)
	print(f"[CACHE] Saved feature embeddings cache to: {cache_file}")
	return X, y


def train_calibrated_model(samples_per_class: int = 120, hybrid_alpha: float = 0.45):
	"""Train hybrid ordinal-classification head and export production ONNX model."""
	print("=" * 75)
	print(" RetinaSight — Training Balanced Hybrid DR Diagnostic Backbone")
	print("=" * 75)

	balanced_df, id_map = load_balanced_cohort(samples_per_class=samples_per_class)
	X, y = extract_or_load_features(balanced_df, id_map, samples_per_class)

	# 80/20 stratified holdout test split
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y
	)
	print(f"[SPLIT] Training samples: {len(X_train)} | Holdout Test samples: {len(X_test)}")

	# 1. Multi-class balanced logistic classification head
	print("\n[OPTIMIZATION 1] Fitting balanced multi-class logistic head...")
	clf = LogisticRegression(class_weight="balanced", C=0.08, max_iter=1000, solver="lbfgs")
	clf.fit(X_train, y_train)

	# 2. Continuous ordinal Ridge regression head (Kaggle 1st-place approach)
	print("[OPTIMIZATION 2] Fitting ordinal continuous regression head...")
	reg = Ridge(alpha=40.0)
	reg.fit(X_train, y_train)

	# Convert ordinal continuous regression into linear 5-class logits
	# Logit_k = 2k(w^T x + b) - k^2
	W_ord = np.zeros((5, 2048), dtype=np.float32)
	b_ord = np.zeros(5, dtype=np.float32)
	for k in range(5):
		W_ord[k] = 2.0 * k * reg.coef_
		b_ord[k] = 2.0 * k * reg.intercept_ - (k ** 2)

	# 3. Blend into hybrid calibrated weight matrix
	W_clf = clf.coef_.astype(np.float32)
	b_clf = clf.intercept_.astype(np.float32)

	scale_clf = float(np.std(X_train @ W_clf.T + b_clf))
	scale_ord = float(np.std(X_train @ W_ord.T + b_ord))

	W_hybrid = hybrid_alpha * (W_clf / scale_clf) + (1.0 - hybrid_alpha) * (W_ord / scale_ord)
	b_hybrid = hybrid_alpha * (b_clf / scale_clf) + (1.0 - hybrid_alpha) * (b_ord / scale_ord)

	# Evaluate on independent holdout test set
	logits_test = X_test @ W_hybrid.T + b_hybrid
	test_preds = np.argmax(logits_test, axis=1)

	logits_train = X_train @ W_hybrid.T + b_hybrid
	train_preds = np.argmax(logits_train, axis=1)

	train_qwk = cohen_kappa_score(y_train, train_preds, weights="quadratic")
	test_qwk = cohen_kappa_score(y_test, test_preds, weights="quadratic")
	test_acc = float(np.mean(test_preds == y_test))

	print("\n" + "=" * 75)
	print(" EVALUATION ON INDEPENDENT HOLDOUT TEST SET")
	print(f" Holdout Test Quadratic Weighted Kappa (QWK): {test_qwk:.4f}")
	print(f" Training Quadratic Weighted Kappa (QWK):     {train_qwk:.4f}")
	print(f" Holdout Test Accuracy:                       {test_acc * 100:.1f}%")
	print("=" * 75)
	print(classification_report(y_test, test_preds, target_names=[ICDR_CLASSES[i] for i in range(5)]))

	# Construct full PyTorch ResNet-50 model with calibrated weights
	print("[PACKAGING] Embedding calibrated weights into PyTorch ResNet-50...")
	production_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
	production_model.fc = nn.Linear(2048, 5)
	production_model.fc.weight.data = torch.from_numpy(W_hybrid).float()
	production_model.fc.bias.data = torch.from_numpy(b_hybrid).float()
	production_model.eval()

	# Save PyTorch checkpoint
	torch.save(production_model.state_dict(), str(OUTPUT_PTH))
	print(f"[CHECKPOINT] Saved calibrated PyTorch model to: {OUTPUT_PTH}")

	# Export to ONNX
	print(f"[ONNX] Exporting calibrated model to: {OUTPUT_ONNX}...")
	dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)
	torch.onnx.export(
		production_model,
		dummy_input,
		str(OUTPUT_ONNX),
		export_params=True,
		opset_version=14,
		do_constant_folding=True,
		input_names=["input"],
		output_names=["logits"],
		dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
		dynamo=False,
	)
	print(f"[ONNX] Export complete! Model size: {OUTPUT_ONNX.stat().st_size / (1024**2):.2f} MB")

	# Verify with ONNX Runtime on real test samples
	import onnxruntime as ort
	session = ort.InferenceSession(str(OUTPUT_ONNX), providers=["CPUExecutionProvider"])
	input_name = session.get_inputs()[0].name

	print("\n[VERIFICATION] Verifying live inference on representative clinical fundus samples:")
	test_cases = [
		("Grade 0 (Normal)", "frontend/public/samples/sample_messidor_grade0.png", 0),
		("Grade 2 (Moderate)", "frontend/public/samples/sample_aptos_grade2.png", 2),
		("Grade 3 (Severe)", "frontend/public/samples/sample_idrid_grade3.png", 3),
		("Grade 4 (Proliferative)", "frontend/public/samples/sample_proliferative_grade4.png", 4),
	]

	tf_verify = transforms.Compose([
		transforms.ToPILImage(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	for label, img_path_str, expected_grade in test_cases:
		p = Path(img_path_str)
		if p.exists():
			img = cv2.imread(str(p))
			enhanced = preprocessing.enhance(img)
			prep = train_dr_classifier.apply_retinal_preprocessing(enhanced, target_size=(256, 256))
			t = tf_verify(prep).unsqueeze(0).numpy()
			out = session.run(None, {input_name: t})[0][0]
			exp_out = np.exp(out - np.max(out))
			probs = exp_out / np.sum(exp_out)
			pred_grade = int(np.argmax(probs))
			print(
				f"  {label:<24} -> Predicted: Grade {pred_grade} ({ICDR_CLASSES[pred_grade]}) | "
				f"Confidence: {probs[pred_grade]*100:.1f}% | "
				f"Probs: {[round(float(pr), 2) for pr in probs]}"
			)

	print("\n[COMPLETE] Calibrated training and production model export successfully finished!")


if __name__ == "__main__":
	n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 120
	train_calibrated_model(samples_per_class=n_samples)
