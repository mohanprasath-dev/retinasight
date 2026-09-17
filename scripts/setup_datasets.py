"""
RetinaSight — Universal Dataset Setup & Scaffold Tool
SIH 2026, PS ID 26038, Team OnFocus

Scaffolds the directory structures, inspects downloaded datasets in D:\\SIH2026\\Datasets,
and links them into the workspace with zero data duplication.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

EXTERNAL_DATASETS_DIR = Path("D:/SIH2026/Datasets")

DATASET_CONFIG = {
	"aptos2019": {
		"title": "APTOS 2019 Blindness Detection",
		"url": "https://www.kaggle.com/competitions/aptos2019-blindness-detection",
		"dir": "datasets/aptos2019",
		"external_aliases": ["aptos2019", "APTOS", "aptos"],
		"expected_files": ["train.csv", "train_images"],
		"script": "training/train_dr_classifier.py",
		"purpose": "Primary 5-Class ICDR Severity Classification (ResNet-50)",
	},
	"idrid": {
		"title": "IDRiD (Indian Diabetic Retinopathy Image Dataset)",
		"url": "https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid",
		"dir": "datasets/idrid",
		"external_aliases": ["IDRiD", "idrid"],
		"expected_files": ["A. Segmentation"],
		"script": "training/train_idrid_lesions.py",
		"purpose": "Pixel-Level Lesion Ground-Truth & Grad-CAM IoU Alignment",
	},
	"drive": {
		"title": "DRIVE (Digital Retinal Images for Vessel Extraction)",
		"url": "https://drive.grand-challenge.org/",
		"dir": "datasets/drive",
		"external_aliases": ["DRIVE", "drive"],
		"expected_files": ["datasets", "training", "test", "datasets.zip"],
		"script": "training/train_vessel_segmentation.py",
		"purpose": "Vascular Tree Segmentation Calibration (Dice >= 0.82)",
	},
	"messidor2": {
		"title": "Messidor-2 Clinical Cohort",
		"url": "https://www.adcis.net/en/third-party/messidor2/",
		"dir": "datasets/messidor2",
		"external_aliases": ["Messidor-2", "messidor2", "messidor"],
		"expected_files": ["images", "messidor-2.csv", "messidor_data.csv"],
		"script": "training/train_messidor_generalization.py",
		"purpose": "External Multi-Center Generalization & DME Assessment",
	},
}


def find_external_dataset_path(key: str) -> Path:
	"""Locate dataset folder in D:/SIH2026/Datasets using aliases."""
	if not EXTERNAL_DATASETS_DIR.exists():
		return None
	cfg = DATASET_CONFIG.get(key, {})
	for alias in cfg.get("external_aliases", []):
		candidate = EXTERNAL_DATASETS_DIR / alias
		if candidate.exists():
			return candidate
	return None


def setup_directories(base_dir: Path, link_external: bool = True):
	print("=" * 75)
	print(" RetinaSight — Scaffolding Datasets & Linking External Archives")
	print(f" Source Search Path: {EXTERNAL_DATASETS_DIR}")
	print("=" * 75)

	datasets_root = base_dir / "datasets"
	datasets_root.mkdir(parents=True, exist_ok=True)

	for key, cfg in DATASET_CONFIG.items():
		target = base_dir / cfg["dir"]
		ext_path = find_external_dataset_path(key)

		if ext_path and ext_path.exists():
			print(f"[FOUND EXTERNAL] {cfg['title']} at: {ext_path}")
			# Create directory junction if on Windows and target doesn't exist
			if link_external and not target.exists() and os.name == "nt":
				try:
					cmd = f'cmd /c mklink /J "{target}" "{ext_path}"'
					subprocess.run(cmd, shell=True, capture_output=True, check=True)
					print(f"  -> Linked via Directory Junction: {target} <==> {ext_path}")
				except Exception as e:
					target.mkdir(parents=True, exist_ok=True)
					print(f"  -> Fallback: Created local target folder {target} ({e})")
			elif not target.exists():
				target.mkdir(parents=True, exist_ok=True)
		else:
			target.mkdir(parents=True, exist_ok=True)
			print(f"[LOCAL DIR] {target} -> {cfg['title']}")

	print("\nDirectory setup and junction linking complete.")


def verify_datasets(base_dir: Path):
	print("\n" + "=" * 75)
	print(" RetinaSight — Datasets Status & Verification Audit")
	print("=" * 75)

	all_present = True
	for key, cfg in DATASET_CONFIG.items():
		target = base_dir / cfg["dir"]
		ext_path = find_external_dataset_path(key)

		# Check in local workspace target or external path
		search_dirs = [p for p in [target, ext_path] if p and p.exists()]
		
		found_items = []
		for s_dir in search_dirs:
			for exp in cfg["expected_files"]:
				if (s_dir / exp).exists() and exp not in found_items:
					found_items.append(exp)
			# Also check for zip files
			for z in s_dir.glob("*.zip*"):
				if z.name not in found_items:
					found_items.append(z.name)

		status_badge = "READY / ARCHIVE DETECTED" if found_items else "NOT FOUND (Synthetic Fallback Active)"
		if not found_items:
			all_present = False

		print(f"\n* [{key.upper()}] {cfg['title']}")
		print(f"  Status:       {status_badge}")
		print(f"  Discovered:   {', '.join(found_items) if found_items else 'None'}")
		print(f"  Purpose:      {cfg['purpose']}")
		print(f"  Local Dir:    {target}")
		if ext_path:
			print(f"  External Dir: {ext_path}")
		print(f"  Train Script: python {cfg['script']}")

	print("\n" + "-" * 75)
	print("NOTE: All RetinaSight training scripts (train_dr_classifier.py,")
	print("train_idrid_lesions.py, train_vessel_segmentation.py, train_messidor_generalization.py)")
	print("automatically check D:/SIH2026/Datasets and feature automatic synthetic fallbacks,")
	print("ensuring code executes cleanly whether unzipped or running dry-run benchmarks.")
	print("=" * 75 + "\n")


def main():
	parser = argparse.ArgumentParser(description="RetinaSight Datasets Scaffolding and Verification")
	parser.add_argument("--verify", action="store_true", help="Audit dataset availability")
	parser.add_argument("--no-link", action="store_true", help="Do not create directory junctions")
	args = parser.parse_args()

	base_dir = Path(__file__).resolve().parent.parent
	setup_directories(base_dir, link_external=not args.no_link)
	verify_datasets(base_dir)


if __name__ == "__main__":
	main()
