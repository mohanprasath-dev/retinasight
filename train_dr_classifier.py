"""
RetinaSight — ResNet50 Diabetic Retinopathy Classifier Training Pipeline (Stage 4)
SIH 2026, PS ID 26038, Team OnFocus

==============================================================================
TECH STACK SUBSTITUTION NOTE:
The official PS 26038 specifies MATLAB Deep Learning Toolbox.
This implementation uses PyTorch (TorchVision ResNet50) + ONNX Runtime.
Reason: Eliminates dependency on expensive proprietary MATLAB licenses,
enabling cloud-native training on Kaggle GPUs and lightweight, real-time
inference deployment in rural Primary Health Centres (PHCs).
==============================================================================

Designed to run on Kaggle GPU Notebooks with APTOS 2019 Blindness Detection dataset:
Input Path: /kaggle/input/aptos2019-blindness-detection
Output: Best model checkpoint (.pth) and production-ready ONNX model (.onnx).
"""

import argparse
import io
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms


# ------------------------------------------------------------------------------
# 1. Constants & Clinical Class Definitions
# ------------------------------------------------------------------------------
ICDR_CLASSES = {
	0: "No DR",
	1: "Mild",
	2: "Moderate",
	3: "Severe",
	4: "Proliferative DR",
}
NUM_CLASSES = 5
IMAGE_SIZE = (256, 256)

def resolve_aptos_dir() -> Path:
	candidates = [
		Path("D:/SIH2026/Datasets/aptos2019"),
		Path("D:/SIH2026/Datasets/APTOS"),
		Path("D:/SIH2026/Datasets/aptos"),
		Path("../Datasets/aptos2019"),
		Path("datasets/aptos2019"),
		Path("/kaggle/input/aptos2019-blindness-detection"),
	]
	for c in candidates:
		if c.exists():
			return c
	return Path("datasets/aptos2019")

DEFAULT_APTOS_DIR = resolve_aptos_dir()
KAGGLE_DATASET_DIR = DEFAULT_APTOS_DIR
KAGGLE_TRAIN_CSV = KAGGLE_DATASET_DIR / "train.csv"
KAGGLE_TRAIN_IMAGES = KAGGLE_DATASET_DIR / "train_images"


# ------------------------------------------------------------------------------
# 2. Stage 1-3 Preprocessing Transform (CLAHE + Circular Retinal FOV Crop)
# ------------------------------------------------------------------------------
def apply_retinal_preprocessing(image_bgr: np.ndarray, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
	"""Preprocess retinal fundus image:
	1. Extract green channel with CLAHE for maximal lesion/vessel contrast.
	2. Apply bilateral filtering to denoise sensor artifacts while preserving sharp borders.
	3. Normalize dimensions to target_size.

	Returns:
		Processed RGB image ready for CNN normalization.
	"""
	h, w = image_bgr.shape[:2]

	# Segment retinal circular mask
	gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	_, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if contours:
		largest = max(contours, key=cv2.contourArea)
		(cx, cy), radius = cv2.minEnclosingCircle(largest)
		x1, y1 = max(0, int(cx - radius)), max(0, int(cy - radius))
		x2, y2 = min(w, int(cx + radius)), min(h, int(cy + radius))
		if x2 > x1 and y2 > y1:
			image_bgr = image_bgr[y1:y2, x1:x2]

	# Resize to standardized training resolution
	resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_AREA)

	# CLAHE on green channel
	b, g, r = cv2.split(resized)
	clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
	g_clahe = clahe.apply(g)
	g_denoised = cv2.bilateralFilter(g_clahe, d=7, sigmaColor=50, sigmaSpace=50)

	enhanced_bgr = cv2.merge([b, g_denoised, r])
	enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
	return enhanced_rgb


# ------------------------------------------------------------------------------
# 3. APTOS Dataset Definition
# ------------------------------------------------------------------------------
class AptosDataset(Dataset):
	"""PyTorch Dataset for APTOS 2019 Blindness Detection with clinical preprocessing."""

	def __init__(
		self,
		df: pd.DataFrame,
		image_dir: Path,
		transform: Optional[transforms.Compose] = None,
		is_training: bool = True,
	):
		self.df = df.reset_index(drop=True)
		self.image_dir = Path(image_dir)
		self.transform = transform
		self.is_training = is_training

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
		row = self.df.iloc[idx]
		id_code = str(row.get("id_code", row.get("Image name", row.get("id", ""))))
		label = int(row.get("diagnosis", row.get("Retinopathy grade", 0)))

		# Find image file (handle .png, .jpg, or .tif)
		img_path = None
		for ext in [".png", ".jpg", ".tif", ".jpeg"]:
			cand = self.image_dir / f"{id_code}{ext}"
			if cand.exists():
				img_path = cand
				break
		if img_path is None:
			img_path = self.image_dir / id_code

		if img_path.exists():
			img_bgr = cv2.imread(str(img_path))
		else:
			# Fallback for synthetic/missing images
			img_bgr = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)

		# Apply clinical preprocessing
		img_rgb = apply_retinal_preprocessing(img_bgr, target_size=IMAGE_SIZE)

		# Convert to float PIL/tensor
		if self.transform:
			img_tensor = self.transform(img_rgb)
		else:
			img_tensor = transforms.ToTensor()(img_rgb)

		return img_tensor, label


# ------------------------------------------------------------------------------
# 4. Data Transforms & Augmentations
# ------------------------------------------------------------------------------
def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
	"""Data augmentations for fundus photography:
	Fundus images are rotationally symmetric (no canonical up/down).
	Augment with horizontal/vertical flips, affine rotations, and ImageNet stats.
	"""
	train_transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomVerticalFlip(p=0.5),
		transforms.RandomRotation(degrees=180),
		transforms.ColorJitter(brightness=0.15, contrast=0.15),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	val_transform = transforms.Compose([
		transforms.ToPILImage(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	return train_transform, val_transform


# ------------------------------------------------------------------------------
# 5. ResNet50 Model Architecture
# ------------------------------------------------------------------------------
def build_resnet50_classifier(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
	"""Construct ResNet50 with custom classification head for 5-class ICDR severity."""
	weights = models.ResNet50_Weights.DEFAULT if pretrained else None
	model = models.resnet50(weights=weights)

	# Replace final fully-connected layer with regularized multi-layer head
	in_features = model.fc.in_features
	model.fc = nn.Sequential(
		nn.Dropout(p=0.3),
		nn.Linear(in_features, 512),
		nn.BatchNorm1d(512),
		nn.ReLU(inplace=True),
		nn.Dropout(p=0.2),
		nn.Linear(512, num_classes),
	)
	return model


# ------------------------------------------------------------------------------
# 6. Class Weight Calculation for Imbalance
# ------------------------------------------------------------------------------
def compute_class_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
	"""Compute inverse-frequency class weights to counterbalance severe APTOS class skew:
	w_c = total_samples / (num_classes * count_c)
	"""
	classes, counts = np.unique(labels, return_counts=True)
	count_dict = dict(zip(classes, counts))
	total_samples = len(labels)

	weights = []
	for c in range(num_classes):
		c_count = count_dict.get(c, 1)
		w = total_samples / (num_classes * c_count)
		weights.append(w)

	weight_tensor = torch.tensor(weights, dtype=torch.float32)
	# Normalize so sum(weights) == num_classes
	weight_tensor = weight_tensor / weight_tensor.sum() * num_classes
	return weight_tensor


# ------------------------------------------------------------------------------
# 7. Clinical Validation Metrics (Sensitivity & Specificity per Class)
# ------------------------------------------------------------------------------
def compute_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Union[float, Dict[int, float]]]:
	"""Compute per-class sensitivity (recall), specificity, precision, and QWK score."""
	cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

	sensitivity_per_class = {}
	specificity_per_class = {}

	total_samples = np.sum(cm)
	for i in range(NUM_CLASSES):
		tp = cm[i, i]
		fn = np.sum(cm[i, :]) - tp
		fp = np.sum(cm[:, i]) - tp
		tn = total_samples - (tp + fn + fp)

		sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

		sensitivity_per_class[i] = float(sens)
		specificity_per_class[i] = float(spec)

	# Multi-class Quadratic Weighted Kappa (official clinical standard for DR grading)
	try:
		qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
	except Exception:
		qwk = 0.0

	accuracy = float(np.trace(cm) / total_samples) if total_samples > 0 else 0.0

	return {
		"accuracy": round(accuracy, 4),
		"qwk": round(qwk, 4),
		"sensitivity": {k: round(v, 4) for k, v in sensitivity_per_class.items()},
		"specificity": {k: round(v, 4) for k, v in specificity_per_class.items()},
		"confusion_matrix": cm.tolist(),
	}


# ------------------------------------------------------------------------------
# 8. Training & Evaluation Engine
# ------------------------------------------------------------------------------
def train_one_epoch(
	model: nn.Module,
	dataloader: DataLoader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
) -> Tuple[float, float]:
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for images, labels in dataloader:
		images = images.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * images.size(0)
		_, preds = torch.max(outputs, 1)
		correct += torch.sum(preds == labels).item()
		total += labels.size(0)

	epoch_loss = running_loss / max(total, 1)
	epoch_acc = correct / max(total, 1)
	return epoch_loss, epoch_acc


def evaluate(
	model: nn.Module,
	dataloader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
) -> Tuple[float, Dict[str, Union[float, Dict[int, float]]]]:
	model.eval()
	running_loss = 0.0
	all_preds: List[int] = []
	all_labels: List[int] = []
	total = 0

	with torch.no_grad():
		for images, labels in dataloader:
			images = images.to(device)
			labels = labels.to(device)

			outputs = model(images)
			loss = criterion(outputs, labels)

			running_loss += loss.item() * images.size(0)
			_, preds = torch.max(outputs, 1)

			all_preds.extend(preds.cpu().numpy().tolist())
			all_labels.extend(labels.cpu().numpy().tolist())
			total += labels.size(0)

	val_loss = running_loss / max(total, 1)
	metrics = compute_clinical_metrics(np.array(all_labels), np.array(all_preds))
	return val_loss, metrics


# ------------------------------------------------------------------------------
# 9. ONNX Export & Test Verification
# ------------------------------------------------------------------------------
def export_to_onnx(
	model: nn.Module,
	output_path: Path,
	device: torch.device,
	image_size: Tuple[int, int] = IMAGE_SIZE,
) -> bool:
	"""Export trained PyTorch model to ONNX format with dynamic batch axis.
	Verifies loadability with onnxruntime.
	"""
	model.eval()
	dummy_input = torch.randn(1, 3, image_size[0], image_size[1], device=device)
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	print(f"\n[ONNX] Exporting model to {output_path} ...")
	torch.onnx.export(
		model,
		dummy_input,
		str(output_path),
		export_params=True,
		opset_version=18,
		do_constant_folding=True,
		input_names=["input"],
		output_names=["output"],
		dynamic_axes={
			"input": {0: "batch_size"},
			"output": {0: "batch_size"},
		},
		dynamo=False,
	)
	print(f"[ONNX] Export complete! Model size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")

	# Verify with ONNX Runtime
	print("[ONNX] Verifying export with onnxruntime inference ...")
	try:
		import onnxruntime as ort

		session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
		ort_input_name = session.get_inputs()[0].name
		dummy_np = dummy_input.cpu().numpy().astype(np.float32)
		ort_outputs = session.run(None, {ort_input_name: dummy_np})

		output_logits = ort_outputs[0]
		print(f"[ONNX] Inference verified! Output shape: {output_logits.shape} (Expected: (1, 5))")
		probs = torch.softmax(torch.tensor(output_logits), dim=1).numpy()[0]
		print(f"[ONNX] Test class probabilities: {[round(float(p), 4) for p in probs]}")
		return True
	except Exception as e:
		print(f"[ONNX ERROR] ONNX verification failed: {e}", file=sys.stderr)
		return False


# ------------------------------------------------------------------------------
# 10. Synthetic Dataset Generator for Offline / Local Smoke Testing
# ------------------------------------------------------------------------------
def generate_synthetic_dataset(num_samples: int = 20) -> Tuple[pd.DataFrame, Path]:
	"""Generate a synthetic fundus batch for dry-run smoke testing outside Kaggle."""
	scratch_dir = Path("assets/smoke_test_data")
	images_dir = scratch_dir / "train_images"
	images_dir.mkdir(parents=True, exist_ok=True)

	data = []
	for i in range(num_samples):
		id_code = f"sample_{i:03d}"
		diagnosis = i % NUM_CLASSES  # Evenly cycle through 0-4

		img_path = images_dir / f"{id_code}.png"
		if not img_path.exists():
			# Simple synthetic fundus circle
			canvas = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
			cv2.circle(
				canvas,
				(IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2),
				int(IMAGE_SIZE[0] * 0.44),
				(20 + diagnosis * 5, 80 + diagnosis * 15, 180 - diagnosis * 20),
				-1,
			)
			cv2.imwrite(str(img_path), canvas)

		data.append({"id_code": id_code, "diagnosis": diagnosis})

	df = pd.DataFrame(data)
	csv_path = scratch_dir / "train.csv"
	df.to_csv(csv_path, index=False)
	return df, images_dir


# ------------------------------------------------------------------------------
# 11. Main Training Pipeline
# ------------------------------------------------------------------------------
def main():
	parser = argparse.ArgumentParser(description="RetinaSight Stage 4: ResNet50 DR Classifier Training")
	parser.add_argument("--data-dir", type=str, default=str(KAGGLE_DATASET_DIR), help="Path to APTOS 2019 dataset")
	parser.add_argument("--dataset", type=str, default="auto", choices=["auto", "aptos", "idrid", "synthetic"], help="Dataset selector")
	parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
	parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
	parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
	parser.add_argument("--out-onnx", type=str, default="retinasight_resnet50.onnx", help="Path to export final ONNX model")
	parser.add_argument("--out-pth", type=str, default="retinasight_resnet50.pth", help="Path to save best PyTorch checkpoint")
	parser.add_argument("--smoke-test", action="store_true", help="Run 2-epoch dry run smoke test with synthetic/local samples")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Device] Using device: {device}")

	# Determine data directory & CSV
	data_dir = Path(args.data_dir)
	train_csv_path = data_dir / "train.csv"
	train_images_dir = data_dir / "train_images"

	idrid_csv = Path("D:/SIH2026/Datasets/IDRiD/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")
	idrid_imgs = Path("D:/SIH2026/Datasets/IDRiD/B. Disease Grading/1. Original Images/a. Training Set")

	use_idrid = (args.dataset == "idrid") or (args.dataset == "auto" and not train_csv_path.exists() and idrid_csv.exists())
	use_aptos = (args.dataset == "aptos") or (args.dataset == "auto" and train_csv_path.exists())

	if args.smoke_test or args.dataset == "synthetic":
		print("\n[SMOKE TEST] Synthetic dataset requested.")
		print("[SMOKE TEST] Generating local synthetic fundus test dataset for dry-run verification...")
		df, train_images_dir = generate_synthetic_dataset(num_samples=25)
		epochs = min(args.epochs, 2)
		batch_size = 4
	elif use_aptos:
		print(f"\n[Dataset] Loading APTOS dataset from: {data_dir}")
		df = pd.read_csv(train_csv_path)
		epochs = args.epochs
		batch_size = args.batch_size
	elif use_idrid:
		print(f"\n[Dataset] Loading real Indian Diabetic Retinopathy (IDRiD Disease Grading) cohort from: {idrid_csv}")
		df = pd.read_csv(idrid_csv)
		df["id_code"] = df["Image name"]
		df["diagnosis"] = df["Retinopathy grade"]
		train_images_dir = idrid_imgs
		epochs = args.epochs
		batch_size = args.batch_size
	else:
		print("\n[SMOKE TEST] Dataset not found. Falling back to synthetic mode.")
		df, train_images_dir = generate_synthetic_dataset(num_samples=25)
		epochs = min(args.epochs, 2)
		batch_size = 4

	print(f"[Dataset] Total samples: {len(df)}")
	print(f"[Dataset] Class distribution:\n{df['diagnosis'].value_counts().sort_index()}")

	# Compute class weights for cross entropy
	class_weights = compute_class_weights(df["diagnosis"].values, num_classes=NUM_CLASSES)
	print(f"\n[Loss] Inverse-Frequency Class Weights: {[round(float(w), 3) for w in class_weights]}")
	criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

	# 80/20 Stratified train/val split
	train_df, val_df = train_test_split(
		df,
		test_size=0.20,
		random_state=42,
		stratify=df["diagnosis"],
	)
	print(f"[Split] Train samples: {len(train_df)} | Validation samples: {len(val_df)}")

	train_transform, val_transform = get_data_transforms()
	train_dataset = AptosDataset(train_df, train_images_dir, transform=train_transform, is_training=True)
	val_dataset = AptosDataset(val_df, train_images_dir, transform=val_transform, is_training=False)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

	# Build model
	print("\n[Model] Initializing fine-tuned ResNet50...")
	model = build_resnet50_classifier(num_classes=NUM_CLASSES, pretrained=True).to(device)

	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

	best_qwk = -1.0
	start_time = time.time()

	print(f"\n{'='*75}")
	print(f"Starting Training for {epochs} Epochs")
	print(f"{'='*75}")

	for epoch in range(1, epochs + 1):
		t0 = time.time()
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, metrics = evaluate(model, val_loader, criterion, device)
		scheduler.step()
		elapsed = time.time() - t0

		print(
			f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
			f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}% | "
			f"Val Loss: {val_loss:.4f}, Acc: {metrics['accuracy']*100:.1f}%, QWK: {metrics['qwk']:.4f}"
		)

		print("  Per-Class Sensitivity (Recall):", {ICDR_CLASSES[k]: metrics['sensitivity'][k] for k in range(NUM_CLASSES)})
		print("  Per-Class Specificity:", {ICDR_CLASSES[k]: metrics['specificity'][k] for k in range(NUM_CLASSES)})

		if metrics["qwk"] > best_qwk or epoch == 1:
			best_qwk = metrics["qwk"]
			torch.save(model.state_dict(), args.out_pth)
			print(f"  --> Saved best model checkpoint to {args.out_pth}")

	total_time = time.time() - start_time
	print(f"\n{'='*75}")
	print(f"Training Complete in {total_time/60:.1f} minutes! Best QWK: {best_qwk:.4f}")
	print(f"{'='*75}")

	# Export best model to ONNX
	print(f"\nLoading best checkpoint for ONNX export: {args.out_pth}")
	model.load_state_dict(torch.load(args.out_pth, map_location=device))
	success = export_to_onnx(model, Path(args.out_onnx), device, image_size=IMAGE_SIZE)

	if success:
		print("\n[SUCCESS] RS-02 Training and ONNX Export Verification Complete!")
	else:
		print("\n[FAILURE] ONNX Export or Inference check failed!", file=sys.stderr)
		sys.exit(1)


if __name__ == "__main__":
	main()
