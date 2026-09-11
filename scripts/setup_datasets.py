"""
RetinaSight — Universal Dataset Setup & Scaffold Tool
SIH 2026, PS ID 26038, Team OnFocus

Scaffolds the directory structures and verifies the 4 official datasets:
1. APTOS 2019 Blindness Detection
2. IDRiD (Indian Diabetic Retinopathy Image Dataset)
3. DRIVE (Digital Retinal Images for Vessel Extraction)
4. Messidor-2 Clinical Validation Cohort
"""

import argparse
import os
from pathlib import Path
import sys

DATASET_CONFIG = {
	"aptos2019": {
		"title": "APTOS 2019 Blindness Detection",
		"url": "https://www.kaggle.com/competitions/aptos2019-blindness-detection",
		"dir": "datasets/aptos2019",
		"kaggle_slug": "competitions/aptos2019-blindness-detection",
		"expected_files": ["train.csv", "train_images"],
		"script": "train_dr_classifier.py",
		"purpose": "Primary 5-Class ICDR Severity Classification (ResNet-50)",
	},
	"idrid": {
		"title": "IDRiD (Indian Diabetic Retinopathy Image Dataset)",
		"url": "https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid",
		"dir": "datasets/idrid",
		"expected_files": ["A. Segmentation"],
		"script": "train_idrid_lesions.py",
		"purpose": "Pixel-Level Lesion Ground-Truth & Grad-CAM IoU Alignment",
	},
	"drive": {
		"title": "DRIVE (Digital Retinal Images for Vessel Extraction)",
		"url": "https://drive.grand-challenge.org/",
		"dir": "datasets/drive",
		"expected_files": ["training", "test"],
		"script": "train_vessel_segmentation.py",
		"purpose": "Vascular Tree Segmentation Calibration (Dice >= 0.82)",
	},
	"messidor2": {
		"title": "Messidor-2 Clinical Cohort",
		"url": "https://www.adcis.net/en/third-party/messidor2/",
		"dir": "datasets/messidor2",
		"expected_files": ["images"],
		"script": "train_messidor_generalization.py",
		"purpose": "External Multi-Center Generalization & DME Assessment",
	},
}


def setup_directories(base_dir: Path):
	print("=" * 75)
	print(" RetinaSight — Scaffolding Datasets Directory Structure")
	print("=" * 75)

	for key, cfg in DATASET_CONFIG.items():
		target = base_dir / cfg["dir"]
		target.mkdir(parents=True, exist_ok=True)
		print(f"[CREATED/EXISTS] {target} -> {cfg['title']}")

	print("\nDirectory scaffolding complete.")


def verify_datasets(base_dir: Path):
	print("\n" + "=" * 75)
	print(" RetinaSight — Datasets Status & Verification Audit")
	print("=" * 75)

	all_present = True
	for key, cfg in DATASET_CONFIG.items():
		target = base_dir / cfg["dir"]
		present_count = 0
		for item in cfg["expected_files"]:
			if (target / item).exists():
				present_count += 1

		status_badge = "READY" if present_count == len(cfg["expected_files"]) else "NOT FOUND (Synthetic/Stub Mode Active)"
		if present_count != len(cfg["expected_files"]):
			all_present = False

		print(f"\n* [{key.upper()}] {cfg['title']}")
		print(f"  Status:       {status_badge}")
		print(f"  Purpose:      {cfg['purpose']}")
		print(f"  Directory:    {target}")
		print(f"  Download URL: {cfg['url']}")
		print(f"  Train Script: python {cfg['script']}")

	print("\n" + "-" * 75)
	if not all_present:
		print("NOTE: Even if raw gigabyte-scale datasets are not downloaded locally, all")
		print("training pipelines (train_dr_classifier.py, train_idrid_lesions.py,")
		print("train_vessel_segmentation.py, train_messidor_generalization.py) feature")
		print("automatic synthetic fallbacks and can be executed or dry-run immediately!")
	else:
		print("All 4 core datasets are verified and present on disk.")
	print("=" * 75 + "\n")


def main():
	parser = argparse.ArgumentParser(description="RetinaSight Datasets Scaffolding and Verification")
	parser.add_argument("--verify", action="store_true", help="Audit dataset availability")
	args = parser.parse_args()

	base_dir = Path(__file__).resolve().parent.parent
	setup_directories(base_dir)
	verify_datasets(base_dir)


if __name__ == "__main__":
	main()
