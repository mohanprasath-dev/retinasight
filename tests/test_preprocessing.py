"""
RetinaSight — RS-01 Preprocessing Pipeline Verification & Test Suite
SIH 2026, PS ID 26038, Team OnFocus

Tests against acceptance criteria:
1. quality_check() pass/fail on clear vs deliberately blurry test pair + dark image.
2. enhance() visibility & histogram spread improvement on dark image.
3. segment() produces non-empty vessel mask on clear image.
Saves visual verification plots to outputs/.
"""

import os
from pathlib import Path
import sys
from typing import Tuple
import urllib.request
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import preprocessing

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLE_DIR = PROJECT_ROOT / "assets" / "samples"


def create_synthetic_fundus(width: int = 600, height: int = 600) -> np.ndarray:
	"""Generate a realistic synthetic retinal fundus image if network download is unavailable."""
	img = np.zeros((height, width, 3), dtype=np.uint8)
	center = (width // 2, height // 2)
	radius = int(min(width, height) * 0.44)

	# 1. Base retinal background gradient (warm reddish-orange)
	y, x = np.ogrid[:height, :width]
	dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
	mask = dist_from_center <= radius

	# Color gradient: bright orange-red center to deeper red towards periphery
	norm_dist = np.clip(dist_from_center / radius, 0, 1)
	r = np.clip(220 - norm_dist * 80 + np.random.normal(0, 4, (height, width)), 0, 255).astype(np.uint8)
	g = np.clip(110 - norm_dist * 60 + np.random.normal(0, 3, (height, width)), 0, 255).astype(np.uint8)
	b = np.clip(30 - norm_dist * 20 + np.random.normal(0, 2, (height, width)), 0, 255).astype(np.uint8)

	img[:, :, 0] = np.where(mask, b, 0)
	img[:, :, 1] = np.where(mask, g, 0)
	img[:, :, 2] = np.where(mask, r, 0)

	# 2. Optic Disc (bright yellowish-white circle in nasal region)
	od_center = (int(center[0] - radius * 0.45), center[1])
	od_radius = int(radius * 0.16)
	od_dist = np.sqrt((x - od_center[0]) ** 2 + (y - od_center[1]) ** 2)
	od_mask = od_dist <= od_radius

	img[od_mask, 0] = np.clip(120 + np.random.normal(0, 5, np.count_nonzero(od_mask)), 0, 255).astype(np.uint8)
	img[od_mask, 1] = np.clip(210 + np.random.normal(0, 5, np.count_nonzero(od_mask)), 0, 255).astype(np.uint8)
	img[od_mask, 2] = np.clip(245 + np.random.normal(0, 5, np.count_nonzero(od_mask)), 0, 255).astype(np.uint8)

	# 3. Vascular tree (branching dark vessels originating from optic disc)
	vessel_canvas = np.zeros((height, width), dtype=np.uint8)
	branches = [
		[(od_center[0], od_center[1]), (od_center[0] + 60, od_center[1] - 80), (od_center[0] + 160, od_center[1] - 130), (od_center[0] + 280, od_center[1] - 110)],
		[(od_center[0], od_center[1]), (od_center[0] + 70, od_center[1] + 90), (od_center[0] + 170, od_center[1] + 140), (od_center[0] + 290, od_center[1] + 120)],
		[(od_center[0], od_center[1]), (od_center[0] - 40, od_center[1] - 70), (od_center[0] - 80, od_center[1] - 140)],
		[(od_center[0], od_center[1]), (od_center[0] - 40, od_center[1] + 70), (od_center[0] - 80, od_center[1] + 140)],
		[(od_center[0] + 160, od_center[1] - 130), (od_center[0] + 220, od_center[1] - 180), (od_center[0] + 290, od_center[1] - 200)],
		[(od_center[0] + 170, od_center[1] + 140), (od_center[0] + 230, od_center[1] + 190), (od_center[0] + 300, od_center[1] + 210)],
	]

	for path in branches:
		pts = np.array(path, np.int32).reshape((-1, 1, 2))
		cv2.polylines(vessel_canvas, [pts], False, 255, thickness=4)

	# Add secondary finer capillaries
	finer_branches = [
		[(od_center[0] + 100, od_center[1] - 105), (od_center[0] + 140, od_center[1] - 70), (od_center[0] + 180, od_center[1] - 50)],
		[(od_center[0] + 110, od_center[1] + 115), (od_center[0] + 150, od_center[1] + 70), (od_center[0] + 190, od_center[1] + 50)],
		[(od_center[0] + 200, od_center[1] - 135), (od_center[0] + 240, od_center[1] - 100)],
	]
	for path in finer_branches:
		pts = np.array(path, np.int32).reshape((-1, 1, 2))
		cv2.polylines(vessel_canvas, [pts], False, 255, thickness=2)

	# Vessel mask within retina
	vessel_mask = (vessel_canvas > 0) & mask
	# Darken vessels on fundus (high hemoglobin absorption)
	img[vessel_mask, 0] = np.clip(img[vessel_mask, 0].astype(int) - 15, 0, 255).astype(np.uint8)
	img[vessel_mask, 1] = np.clip(img[vessel_mask, 1].astype(int) - 55, 0, 255).astype(np.uint8)
	img[vessel_mask, 2] = np.clip(img[vessel_mask, 2].astype(int) - 75, 0, 255).astype(np.uint8)

	# Gentle overall smoothing to blend natural tissue appearance
	img = cv2.GaussianBlur(img, (3, 3), 0)
	img[~mask] = 0
	return img


def acquire_sample_images() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Acquire clear, blurry, and dark test fundus images."""
	SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
	clear_path_jpg = SAMPLE_DIR / "fundus_clear.jpg"
	clear_path = SAMPLE_DIR / "fundus_clear.png"
	blurry_path = SAMPLE_DIR / "fundus_blurry.png"
	dark_path = SAMPLE_DIR / "fundus_dark.png"

	if clear_path_jpg.exists():
		img = cv2.imread(str(clear_path_jpg))
		cv2.imwrite(str(clear_path), img)
	elif clear_path.exists():
		img = cv2.imread(str(clear_path))
	else:
		img = create_synthetic_fundus(600, 600)
		cv2.imwrite(str(clear_path), img)

	# Deliberately blurry test image (severe defocus / motion blur)
	img_blurry = cv2.GaussianBlur(img, (35, 35), sigmaX=15)
	cv2.imwrite(str(blurry_path), img_blurry)

	# Deliberately dark / underexposed test image (simulated low PHC illumination)
	img_dark = (img.astype(np.float32) * 0.18).astype(np.uint8)
	cv2.imwrite(str(dark_path), img_dark)

	return img, img_blurry, img_dark


def run_tests():
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	print("=" * 70)
	print("RetinaSight RS-01: Preprocessing Pipeline Test Suite")
	print("=" * 70)

	img_clear, img_blurry, img_dark = acquire_sample_images()

	# ---------------------------------------------------------
	# 1. QUALITY CHECK TEST
	# ---------------------------------------------------------
	print("\n--- [Test 1] Quality Check (Blur, Brightness, Centering) ---")
	qc_clear = preprocessing.quality_check(img_clear)
	qc_blurry = preprocessing.quality_check(img_blurry)
	qc_dark = preprocessing.quality_check(img_dark)

	print(f"Clear Image Quality Check:  Passed={qc_clear['passed']} | Metrics: {qc_clear['metrics']}")
	print(f"Blurry Image Quality Check: Passed={qc_blurry['passed']} | Reasons: {qc_blurry['reasons']}")
	print(f"Dark Image Quality Check:   Passed={qc_dark['passed']} | Reasons: {qc_dark['reasons']}")

	assert qc_clear["passed"] is True, f"Expected clear image to pass quality check, failed: {qc_clear['reasons']}"
	assert qc_blurry["passed"] is False, "Expected blurry image to fail quality check"
	assert any("blurry" in r.lower() for r in qc_blurry["reasons"]), f"Expected blur failure reason, got: {qc_blurry['reasons']}"
	assert qc_dark["passed"] is False, "Expected dark image to fail quality check"
	assert any("underexposed" in r.lower() for r in qc_dark["reasons"]), f"Expected underexposed reason, got: {qc_dark['reasons']}"
	print("[PASS] Quality check correctly passes clear image and rejects blurry / dark images!")

	# ---------------------------------------------------------
	# 2. ENHANCEMENT TEST
	# ---------------------------------------------------------
	print("\n--- [Test 2] Enhancement (CLAHE + Bilateral Denoise) ---")
	enhanced_dark = preprocessing.enhance(img_dark)

	mask_dark, _ = preprocessing.get_retina_mask(img_dark)
	raw_dark_pixels = img_dark[:, :, 1][mask_dark > 0]
	enh_dark_pixels = enhanced_dark[:, :, 1][mask_dark > 0]

	raw_std = float(np.std(raw_dark_pixels))
	enh_std = float(np.std(enh_dark_pixels))
	raw_p90_p10 = float(np.percentile(raw_dark_pixels, 90) - np.percentile(raw_dark_pixels, 10))
	enh_p90_p10 = float(np.percentile(enh_dark_pixels, 90) - np.percentile(enh_dark_pixels, 10))

	print(f"Dark Image Green Channel Standard Deviation: Raw={raw_std:.2f} -> Enhanced={enh_std:.2f}")
	print(f"Dark Image Dynamic Range Spread (P90 - P10): Raw={raw_p90_p10:.2f} -> Enhanced={enh_p90_p10:.2f}")

	assert enh_std > raw_std * 1.3, f"Expected enhanced image standard deviation to expand significantly ({enh_std:.2f} vs {raw_std:.2f})"
	assert enh_p90_p10 > raw_p90_p10 * 1.3, f"Expected dynamic range to expand ({enh_p90_p10:.2f} vs {raw_p90_p10:.2f})"
	print("[PASS] Enhancement visibly and quantitatively expands contrast and histogram spread!")

	# ---------------------------------------------------------
	# 3. SEGMENTATION TEST
	# ---------------------------------------------------------
	print("\n--- [Test 3] Segmentation (Vessel & Lesion Stubs) ---")
	seg_results = preprocessing.segment(img_clear)
	vessels = seg_results["vessels"]
	vessel_pixel_count = int(np.count_nonzero(vessels))
	total_retina_pixels = int(np.count_nonzero(preprocessing.get_retina_mask(img_clear)[0]))
	vessel_density = (vessel_pixel_count / total_retina_pixels) * 100 if total_retina_pixels > 0 else 0

	print(f"Vessel Mask: Non-zero pixels={vessel_pixel_count} | Retinal vessel density={vessel_density:.2f}%")
	assert vessel_pixel_count > 500, f"Expected non-empty vessel mask with >500 pixels, got {vessel_pixel_count}"
	assert 1.0 <= vessel_density <= 25.0, f"Expected realistic physiological vessel density (1-25%), got {vessel_density:.2f}%"
	print("[PASS] Vessel segmentation generated non-empty, physiologically valid vessel mask!")

	# ---------------------------------------------------------
	# 4. GENERATE VISUAL VERIFICATION ARTIFACTS
	# ---------------------------------------------------------
	print("\n--- [Artifact Generation] Saving visual verification artifacts ---")
	fig, axes = plt.subplots(2, 3, figsize=(15, 10))

	# Row 1: Quality Check Cases
	axes[0, 0].imshow(cv2.cvtColor(img_clear, cv2.COLOR_BGR2RGB))
	axes[0, 0].set_title(f"Clear Fundus (PASS)\nVar: {qc_clear['metrics']['blur_variance']:.1f}", fontsize=11, color="green")
	axes[0, 0].axis("off")

	axes[0, 1].imshow(cv2.cvtColor(img_blurry, cv2.COLOR_BGR2RGB))
	axes[0, 1].set_title(f"Blurry Capture (FAIL)\nVar: {qc_blurry['metrics']['blur_variance']:.1f}", fontsize=11, color="red")
	axes[0, 1].axis("off")

	axes[0, 2].imshow(cv2.cvtColor(img_dark, cv2.COLOR_BGR2RGB))
	axes[0, 2].set_title(f"Dark Capture (FAIL)\nMean: {qc_dark['metrics']['mean_brightness']:.1f}", fontsize=11, color="red")
	axes[0, 2].axis("off")

	# Row 2: Enhancement & Segmentation
	axes[1, 0].imshow(cv2.cvtColor(enhanced_dark, cv2.COLOR_BGR2RGB))
	axes[1, 0].set_title(f"CLAHE Enhanced Dark Image\nStd: {enh_std:.1f} (was {raw_std:.1f})", fontsize=11, color="blue")
	axes[1, 0].axis("off")

	axes[1, 1].imshow(vessels, cmap="gray")
	axes[1, 1].set_title(f"Segmented Vessels (Binary)\nCount: {vessel_pixel_count} px", fontsize=11)
	axes[1, 1].axis("off")

	# Vessel overlay on clear fundus
	overlay = img_clear.copy()
	overlay[vessels > 0] = [0, 255, 0]  # Green vessel overlay
	blended = cv2.addWeighted(img_clear, 0.65, overlay, 0.35, 0)
	axes[1, 2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
	axes[1, 2].set_title(f"Vessel Overlay on Fundus\nDensity: {vessel_density:.1f}%", fontsize=11)
	axes[1, 2].axis("off")

	plt.tight_layout()
	verification_plot_path = OUTPUT_DIR / "verification_rs01.png"
	plt.savefig(str(verification_plot_path), dpi=150)
	plt.close()
	print(f"[OK] Visual verification collage saved to: {verification_plot_path}")

	# Histogram plot for enhancement verification
	plt.figure(figsize=(10, 4))
	plt.hist(raw_dark_pixels, bins=64, range=(0, 255), color="dimgray", alpha=0.6, label="Raw Dark Image")
	plt.hist(enh_dark_pixels, bins=64, range=(0, 255), color="teal", alpha=0.6, label="CLAHE Enhanced Image")
	plt.title("Histogram Spread Verification (Raw Dark vs CLAHE Enhanced Green Channel)")
	plt.xlabel("Pixel Intensity")
	plt.ylabel("Frequency")
	plt.legend()
	plt.grid(True, linestyle="--", alpha=0.5)
	histogram_plot_path = OUTPUT_DIR / "verification_enhancement_histogram.png"
	plt.savefig(str(histogram_plot_path), dpi=150)
	plt.close()
	print(f"[OK] Enhancement histogram plot saved to: {histogram_plot_path}")

	print("\n" + "=" * 70)
	print("ALL RS-01 ACCEPTANCE CRITERIA PASSED!")
	print("=" * 70)


def test_preprocessing():
	"""Pytest-compatible test discovery function."""
	run_tests()


if __name__ == "__main__":
	run_tests()
