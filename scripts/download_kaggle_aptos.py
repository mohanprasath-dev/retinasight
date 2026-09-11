"""
RetinaSight — Automated Kaggle APTOS 2019 Downloader & Training Dispatcher
SIH 2026, PS ID 26038, Team OnFocus

Safely authenticates with the Kaggle API, validates available disk space,
downloads the APTOS 2019 Blindness Detection dataset (or optimized 256x256 mirror),
extracts it, and triggers model training.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


APTOS_COMPETITION = "aptos2019-blindness-detection"
APTOS_RESIZED_DATASET = "benjaminwarner/aptos2019-blindness-detection-256x256"


def check_kaggle_auth() -> bool:
	"""Check if Kaggle API is authenticated via ~/.kaggle/kaggle.json or environment."""
	kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
	has_json = kaggle_json.exists()
	has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))

	if has_json or has_env:
		return True

	# Try running kaggle config view
	try:
		res = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True, check=False)
		return res.returncode == 0
	except Exception:
		return False


def get_drive_free_space_gb(path: Path) -> float:
	"""Get free disk space in GB for given directory."""
	try:
		p = path.resolve()
		while not p.exists() and p.parent != p:
			p = p.parent
		total, used, free = shutil.disk_usage(p)
		return free / (1024 ** 3)
	except Exception:
		return 0.0


def download_and_extract(
	dest_dir: Path,
	mode: str = "resized",  # "raw" or "resized"
) -> bool:
	"""Download dataset from Kaggle and extract to dest_dir."""
	dest_dir = Path(dest_dir)
	dest_dir.mkdir(parents=True, exist_ok=True)

	free_gb = get_drive_free_space_gb(dest_dir)
	print(f"[DISK] Destination: {dest_dir} (Free space: {free_gb:.1f} GB)")

	if mode == "raw" and free_gb < 15.0:
		print(f"[WARNING] Raw APTOS competition download requires ~15 GB (zip + extract).")
		print(f"          Current drive has only {free_gb:.1f} GB free.")
		print(f"          Switching to optimized 256x256 dataset (~180 MB) to prevent disk full crash.")
		mode = "resized"

	zip_path = None
	if mode == "raw":
		print(f"\n[DOWNLOAD] Fetching raw APTOS 2019 competition dataset from Kaggle...")
		print(f"           Note: Requires accepting rules at: https://www.kaggle.com/competitions/{APTOS_COMPETITION}/rules")
		cmd = ["kaggle", "competitions", "download", "-c", APTOS_COMPETITION, "-p", str(dest_dir)]
		expected_zip = dest_dir / f"{APTOS_COMPETITION}.zip"
		try:
			subprocess.run(cmd, check=True)
		except subprocess.CalledProcessError as e:
			print(f"[ERROR] Raw download failed (Exit {e.returncode}). Switching to pre-resized mirror...")
			mode = "resized"

	if mode == "resized":
		candidates = [
			"benjaminwarner/aptos2019-blindness-detection-256x256",
			"mariaherrerot/aptos2019-blindness-detection-resized-256x256",
			"tkm2261/aptos2019-blindness-detection-512x-512",
		]
		downloaded = False
		for ds in candidates:
			print(f"\n[DOWNLOAD] Trying optimized mirror: {ds}...")
			cmd = ["kaggle", "datasets", "download", "-d", ds, "-p", str(dest_dir)]
			try:
				subprocess.run(cmd, check=True)
				downloaded = True
				break
			except Exception as e:
				print(f"  -> Candidate {ds} failed ({e}), trying next...")
		if not downloaded:
			print("[ERROR] All mirror downloads failed.")
			return False

	# Find zip file in destination
	zips = list(dest_dir.glob("*.zip"))
	if zips:
		zip_path = zips[0]

	if zip_path and zip_path.exists():
		print(f"[EXTRACT] Extracting {zip_path.name} to {dest_dir}...")
		with zipfile.ZipFile(zip_path, "r") as z:
			z.extractall(dest_dir)
		print("[EXTRACT] Extraction complete.")
		try:
			zip_path.unlink()
			print(f"[CLEANUP] Deleted archive {zip_path.name} to conserve disk space.")
		except Exception:
			pass

		# Normalize directory structure: find train.csv and images
		train_csvs = list(dest_dir.rglob("train.csv"))
		if train_csvs and train_csvs[0] != dest_dir / "train.csv":
			shutil.copy(train_csvs[0], dest_dir / "train.csv")
			print(f"[NORMALIZE] Found and placed train.csv at: {dest_dir / 'train.csv'}")

		return True

	return False


def main():
	parser = argparse.ArgumentParser(description="RetinaSight Kaggle APTOS Downloader & Trainer")
	parser.add_argument("--dest", type=str, default="D:/SIH2026/Datasets/aptos2019", help="Destination folder")
	parser.add_argument("--mode", type=str, default="auto", choices=["auto", "raw", "resized"], help="Download mode")
	parser.add_argument("--train-after", action="store_true", help="Launch training immediately after download")
	parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
	parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
	args = parser.parse_args()

	print("=" * 75)
	print(" RetinaSight — Kaggle APTOS 2019 Integration")
	print("=" * 75)

	if not check_kaggle_auth():
		print("\n[AUTH ERROR] Kaggle API authentication not detected!")
		print("\nTo authenticate:")
		print("  1. Log into your Kaggle account: https://www.kaggle.com/")
		print("  2. Go to Settings: https://www.kaggle.com/settings")
		print("  3. In the 'API' section, click 'Create New Token'. This downloads 'kaggle.json'.")
		print(f"  4. Move 'kaggle.json' to: {Path.home() / '.kaggle' / 'kaggle.json'}")
		print("  5. (Important) Accept competition rules at:")
		print("     https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules")
		print("\nAlternatively, run in terminal: kaggle auth login")
		sys.exit(1)

	dest_path = Path(args.dest)
	free_gb = get_drive_free_space_gb(dest_path)

	mode = args.mode
	if mode == "auto":
		mode = "resized" if free_gb < 15.0 else "raw"

	success = download_and_extract(dest_path, mode=mode)
	if not success:
		sys.exit(1)

	print(f"\n[SUCCESS] Dataset ready at: {dest_path}")

	if args.train_after:
		print("\n" + "=" * 75)
		print(" Launching ResNet-50 Classifier Training on Downloaded Kaggle Data")
		print("=" * 75)
		train_cmd = [
			sys.executable,
			"train_dr_classifier.py",
			"--data-dir",
			str(dest_path),
			"--dataset",
			"aptos",
			"--epochs",
			str(args.epochs),
			"--batch-size",
			str(args.batch_size),
		]
		subprocess.run(train_cmd)


if __name__ == "__main__":
	main()
