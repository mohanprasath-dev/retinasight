"""
RetinaSight — Image Preprocessing Module (Stage 1-3)
SIH 2026, PS ID 26038, Team OnFocus

Provides retinal fundus image preprocessing:
1. Quality Check: Blur (Laplacian variance), illumination/brightness, and centering detection.
2. Enhancement: Green-channel CLAHE + bilateral filtering for recoverable degraded captures.
3. Segmentation: Classical CV blood vessel segmentation + lesion detection stubs (microaneurysms, exudates, hemorrhages).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


def load_image(image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
	"""Load image from path or return existing numpy array.
	
	Returns BGR image array.
	"""
	if isinstance(image_input, (str, Path)):
		path_str = str(image_input)
		img = cv2.imread(path_str)
		if img is None:
			raise ValueError(f"Could not load image from path: {path_str}")
		return img
	elif isinstance(image_input, np.ndarray):
		return image_input.copy()
	else:
		raise TypeError(f"Unsupported image input type: {type(image_input)}")


def get_retina_mask(image: np.ndarray, threshold: int = 15) -> Tuple[np.ndarray, Optional[Tuple[Tuple[float, float], float]]]:
	"""Segment the circular retinal field of view (FOV) mask from the black background.

	Returns:
		mask: uint8 binary mask (255 for retina, 0 for background)
		circle: ((center_x, center_y), radius) or None if not found
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
	blurred = cv2.GaussianBlur(gray, (9, 9), 0)
	_, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

	# Fill interior holes and clean up edges
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
	closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

	contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		h, w = gray.shape[:2]
		mask = np.ones((h, w), dtype=np.uint8) * 255
		return mask, None

	largest_contour = max(contours, key=cv2.contourArea)
	mask = np.zeros_like(gray, dtype=np.uint8)
	cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

	# Minimum enclosing circle
	(cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
	return mask, ((cx, cy), radius)


def quality_check(
	image: Union[str, Path, np.ndarray],
	blur_threshold: float = 50.0,
	min_brightness: float = 35.0,
	max_brightness: float = 215.0,
	centering_tolerance: float = 0.25,
) -> Dict[str, Union[bool, str, List[str], Dict[str, float]]]:
	"""Evaluate retinal fundus image quality for clinical screening.

	Checks:
	1. Blur: Laplacian variance in the central retinal region. Low values indicate motion or defocus blur.
	2. Brightness: Mean intensity within the retinal FOV mask. Flags underexposure or overexposure.
	3. Centering: FOV circle offset relative to image dimensions. Flags clipped or off-center captures.

	Returns:
		dict with keys:
			'passed' (bool): True if all checks passed.
			'status' (str): 'pass' or 'fail'.
			'reasons' (list[str]): Explanation for failures / recapture guidance.
			'metrics' (dict): Numeric values for blur, brightness, centering offset ratio, and coverage.
	"""
	img = load_image(image)
	h, w = img.shape[:2]
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

	reasons: List[str] = []
	mask, circle = get_retina_mask(img)

	# 1. Blur evaluation on central 60% region (avoids boundary ring edge artifacts)
	crop_h_start, crop_h_end = int(h * 0.20), int(h * 0.80)
	crop_w_start, crop_w_end = int(w * 0.20), int(w * 0.80)
	center_crop = gray[crop_h_start:crop_h_end, crop_w_start:crop_w_end]

	if center_crop.size > 0:
		laplacian_var = float(cv2.Laplacian(center_crop, cv2.CV_64F).var())
	else:
		laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

	if laplacian_var < blur_threshold:
		reasons.append(
			f"Image is too blurry (Laplacian variance: {laplacian_var:.1f} < threshold {blur_threshold:.1f}). Hold camera steady and refocus."
		)

	# 2. Brightness evaluation within retinal mask
	retinal_pixels = gray[mask > 0]
	if len(retinal_pixels) > 0:
		mean_brightness = float(np.mean(retinal_pixels))
	else:
		mean_brightness = float(np.mean(gray))

	if mean_brightness < min_brightness:
		reasons.append(
			f"Image is underexposed / too dark (Mean brightness: {mean_brightness:.1f} < {min_brightness:.1f}). Increase illumination or flash intensity."
		)
	elif mean_brightness > max_brightness:
		reasons.append(
			f"Image is overexposed / washed out (Mean brightness: {mean_brightness:.1f} > {max_brightness:.1f}). Reduce illumination."
		)

	# 3. Centering and FOV coverage evaluation
	total_area = h * w
	retina_area = float(np.count_nonzero(mask))
	fov_coverage = retina_area / total_area if total_area > 0 else 0.0

	if circle is not None:
		(cx, cy), radius = circle
		img_cx, img_cy = w / 2.0, h / 2.0
		center_dist = float(np.sqrt((cx - img_cx) ** 2 + (cy - img_cy) ** 2))
		offset_ratio = center_dist / max(radius, 1.0)
	else:
		offset_ratio = 0.0

	if fov_coverage < 0.25:
		reasons.append(
			f"Retina field of view is severely clipped or obstructed (Coverage: {fov_coverage * 100:.1f}% < 25%). Ensure pupil is aligned."
		)
	elif offset_ratio > centering_tolerance:
		reasons.append(
			f"Retina is off-center (Offset ratio: {offset_ratio:.2f} > tolerance {centering_tolerance:.2f}). Re-align target pupil."
		)

	# 4. Anatomical & Chromatic Spectrum Authenticity Gate
	# Real human fundus images are dominated by hemoglobin/melanin optical absorption (red-orange spectrum)
	if len(img.shape) == 3:
		retina_pixels_bgr = img[mask > 0] if np.count_nonzero(mask) > 0 else img.reshape(-1, 3)
		b_mean = float(np.mean(retina_pixels_bgr[:, 0]))
		g_mean = float(np.mean(retina_pixels_bgr[:, 1]))
		r_mean = float(np.mean(retina_pixels_bgr[:, 2]))
		red_to_blue_ratio = r_mean / max(b_mean, 1.0)

		if red_to_blue_ratio < 1.15 or r_mean < (g_mean * 0.75):
			reasons.append(
				f"Non-retinal image detected: Chromatic signature (R/B ratio: {red_to_blue_ratio:.2f} < 1.15) does not match human fundus optics. Please upload an authentic retinal capture."
			)
	else:
		red_to_blue_ratio = 1.0
		reasons.append(
			"Single-channel grayscale input: Retinal screening requires 3-channel RGB fundus imaging for hemoglobin absorption assessment."
		)

	passed = len(reasons) == 0

	return {
		"passed": passed,
		"status": "pass" if passed else "fail",
		"reasons": reasons,
		"metrics": {
			"blur_variance": round(laplacian_var, 2),
			"mean_brightness": round(mean_brightness, 2),
			"centering_offset_ratio": round(offset_ratio, 3),
			"fov_coverage": round(fov_coverage, 3),
			"red_to_blue_ratio": round(red_to_blue_ratio, 2),
		},
	}


def preprocess_smartphone_capture(
	image: Union[str, Path, np.ndarray],
	suppress_glare: bool = True,
	crop_to_retina: bool = True,
) -> np.ndarray:
	"""Preprocess fundus photographs taken with normal cameras / smartphone clip-on lenses.
	
	Challenges in smartphone / normal camera eye captures:
	1. Corneal Specular Flash Glare: High-intensity white reflections from smartphone LED flash.
	   Solved using Fast Marching Method (Telea inpainting) to eliminate false-positive exudates.
	2. Non-Retinal Surroundings: Eyelids, eyelashes, or iris boundary visible outside pupil.
	   Solved using circular pupil/retinal aperture detection and tight FOV cropping.
	
	Returns:
		Normalized BGR image ready for clinical enhancement and inference.
	"""
	img = load_image(image)
	h, w = img.shape[:2]

	# 1. Specular Flash Glare Detection and Inpainting
	if suppress_glare:
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
		# Specular reflection is saturated in all channels (intensity > 235)
		_, glare_thresh = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
		
		# Find discrete reflection spots (ignore large overexposed fields)
		contours, _ = cv2.findContours(glare_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		glare_mask = np.zeros_like(gray, dtype=np.uint8)
		
		max_glare_area = h * w * 0.06  # Maximum 6% of frame
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if 4 <= area <= max_glare_area:
				cv2.drawContours(glare_mask, [cnt], -1, 255, thickness=cv2.FILLED)
		
		if np.count_nonzero(glare_mask) > 0:
			# Dilate slightly to include specular halo
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
			glare_mask_dilated = cv2.dilate(glare_mask, kernel, iterations=1)
			# Telea inpainting recovers local retinal vascular/pigment context
			img = cv2.inpaint(img, glare_mask_dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

	# 2. Pupil / Retinal Aperture Detection & Circular Cropping
	if crop_to_retina:
		mask, circle = get_retina_mask(img, threshold=12)
		if circle is not None:
			(cx, cy), radius = circle
			if radius > 30:
				x1, y1 = max(0, int(cx - radius)), max(0, int(cy - radius))
				x2, y2 = min(w, int(cx + radius)), min(h, int(cy + radius))
				if (x2 - x1) > 50 and (y2 - y1) > 50:
					cropped = img[y1:y2, x1:x2]
					# Mask outside circular aperture to pure black for uniform CNN input
					crop_h, crop_w = cropped.shape[:2]
					circ_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
					cv2.circle(circ_mask, (int(crop_w / 2), int(crop_h / 2)), int(min(crop_w, crop_h) / 2), 255, -1)
					img = cv2.bitwise_and(cropped, cropped, mask=circ_mask)

	return img


def count_etdrs_lesions(segmentation_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
	"""Compute discrete lesion counts via connected-component analysis for clinical staging.
	
	Aligns with the Early Treatment Diabetic Retinopathy Study (ETDRS) standard:
	- Microaneurysms: discrete capillary micro-dilations (< 125 um)
	- Hard Exudates: lipid deposits with distinct sharp margins
	- Hemorrhages: dot-and-blot or flame intraretinal hemorrhages
	
	Returns:
		dict with discrete integer counts:
			'microaneurysms_count', 'exudates_count', 'hemorrhages_count', 'total_lesions'
	"""
	counts = {}
	for lesion_key in ["microaneurysms", "exudates", "hemorrhages"]:
		mask = segmentation_dict.get(lesion_key)
		if mask is not None and np.count_nonzero(mask) > 0:
			num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
			# Filter out single-pixel noise (area >= 3 px)
			valid_lesions = 0
			for i in range(1, num_labels):
				if stats[i, cv2.CC_STAT_AREA] >= 3:
					valid_lesions += 1
			counts[f"{lesion_key}_count"] = valid_lesions
		else:
			counts[f"{lesion_key}_count"] = 0

	counts["total_lesions"] = sum(counts.values())
	return counts



def enhance(
	image: Union[str, Path, np.ndarray],
	clip_limit: float = 2.5,
	tile_grid_size: Tuple[int, int] = (8, 8),
	bilateral_d: int = 9,
	sigma_color: float = 75.0,
	sigma_space: float = 75.0,
) -> np.ndarray:
	"""Apply CLAHE on green channel and bilateral denoise for recoverable fundus images.

	In fundus photography, the green channel exhibits the highest optical contrast for
	hemoglobin absorption (blood vessels and microaneurysms). This function applies:
	1. CLAHE (Contrast Limited Adaptive Histogram Equalization) on green channel.
	2. Bilateral filtering to smooth sensor noise while preserving sharp vessel borders.
	3. Luminance balance via LAB color space to ensure uniform field illumination.

	Returns:
		Enhanced BGR image as uint8 numpy array.
	"""
	img = load_image(image)

	# Split BGR channels
	b, g, r = cv2.split(img)

	# 1. Apply CLAHE to the green channel
	clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
	g_clahe = clahe.apply(g)

	# 2. Bilateral filter on enhanced green channel for edge-preserving denoising
	g_denoised = cv2.bilateralFilter(g_clahe, d=bilateral_d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

	# Re-merge channels with enhanced green
	b_filtered = cv2.bilateralFilter(b, d=bilateral_d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
	r_filtered = cv2.bilateralFilter(r, d=bilateral_d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
	enhanced_bgr = cv2.merge([b_filtered, g_denoised, r_filtered])

	# 3. Apply subtle illumination leveling across L-channel in LAB space
	lab = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB)
	l_chan, a_chan, b_chan = cv2.split(lab)
	clahe_l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=tile_grid_size)
	l_enhanced = clahe_l.apply(l_chan)
	lab_enhanced = cv2.merge([l_enhanced, a_chan, b_chan])

	result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
	return result


def segment_vessels(image: Union[str, Path, np.ndarray]) -> np.ndarray:
	"""Segment retinal blood vessels using classical computer vision.

	Pipeline:
	1. Extract green channel with CLAHE.
	2. Apply morphological black-hat filtering to isolate dark tubular structures (blood vessels).
	3. Erode retinal mask to exclude aperture edge artifacts.
	4. Apply adaptive Gaussian thresholding and morphological opening.

	Returns:
		Binary mask (uint8, 0 or 255) of blood vessels.
	"""
	img = load_image(image)
	mask, _ = get_retina_mask(img)

	# Green channel has highest vessel contrast
	g = img[:, :, 1] if len(img.shape) == 3 else img

	# Contrast Limited Adaptive Histogram Equalization
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	g_clahe = clahe.apply(g)

	# Black-hat transform extracts structures darker than their surroundings (retinal vessels)
	kernel_bhat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
	bhat = cv2.morphologyEx(g_clahe, cv2.MORPH_BLACKHAT, kernel_bhat)

	# Mask out background, eroding boundary to eliminate outer circular aperture ring
	mask_eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
	bhat_masked = cv2.bitwise_and(bhat, bhat, mask=mask_eroded)

	# Adaptive Gaussian thresholding to detect both major branches and capillaries
	thresh = cv2.adaptiveThreshold(
		bhat_masked,
		255,
		cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY,
		11,
		-3,
	)

	# Morphological opening with small kernel to suppress isolated noise while preserving vessels
	kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
	vessels = cv2.bitwise_and(opened, opened, mask=mask_eroded)

	return vessels


def detect_microaneurysms(image: Union[str, Path, np.ndarray], mask: Optional[np.ndarray] = None) -> np.ndarray:
	"""Placeholder stub: Detect small reddish circular microaneurysms via morphological blob detection.

	Note: Classical placeholder for MVP; deep segmentation model (IDRiD trained) planned for full production.
	"""
	img = load_image(image)
	if mask is None:
		mask, _ = get_retina_mask(img)

	g = img[:, :, 1]
	# Small circular structuring element for sub-pixel/tiny lesions
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)

	_, ma_thresh = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
	mask_eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
	ma_mask = cv2.bitwise_and(ma_thresh, ma_thresh, mask=mask_eroded)

	return ma_mask


def detect_exudates(image: Union[str, Path, np.ndarray], mask: Optional[np.ndarray] = None) -> np.ndarray:
	"""Placeholder stub: Detect bright yellowish hard/soft exudates via luminance thresholding.

	Note: Classical placeholder for MVP; deep segmentation model planned for full production.
	"""
	img = load_image(image)
	if mask is None:
		mask, _ = get_retina_mask(img)

	# Exudates have high intensity in both red and green channels
	g = img[:, :, 1].astype(np.float32)
	r = img[:, :, 2].astype(np.float32)
	bright_response = (g * 0.5 + r * 0.5).astype(np.uint8)

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	bright_clahe = clahe.apply(bright_response)

	# Threshold high intensity regions
	_, ex_thresh = cv2.threshold(bright_clahe, 195, 255, cv2.THRESH_BINARY)

	mask_eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
	exudates_mask = cv2.bitwise_and(ex_thresh, ex_thresh, mask=mask_eroded)

	return exudates_mask


def detect_hemorrhages(image: Union[str, Path, np.ndarray], mask: Optional[np.ndarray] = None) -> np.ndarray:
	"""Placeholder stub: Detect dark red blotch hemorrhages via color thresholding.

	Note: Classical placeholder for MVP; deep segmentation model planned for full production.
	"""
	img = load_image(image)
	if mask is None:
		mask, _ = get_retina_mask(img)

	g = img[:, :, 1]
	# Hemorrhages are larger dark regions (larger kernel than microaneurysms)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
	blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)

	_, hem_thresh = cv2.threshold(blackhat, 25, 255, cv2.THRESH_BINARY)
	mask_eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
	hemorrhages_mask = cv2.bitwise_and(hem_thresh, hem_thresh, mask=mask_eroded)

	return hemorrhages_mask


def segment(image: Union[str, Path, np.ndarray]) -> Dict[str, np.ndarray]:
	"""Stage 3 Segmentation: Segment blood vessels and identify lesion candidates.

	Returns:
		dict containing:
			'vessels': np.ndarray binary mask (uint8, 0 or 255)
			'microaneurysms': np.ndarray binary mask placeholder
			'exudates': np.ndarray binary mask placeholder
			'hemorrhages': np.ndarray binary mask placeholder
	"""
	img = load_image(image)
	mask, _ = get_retina_mask(img)

	vessels = segment_vessels(img)
	microaneurysms = detect_microaneurysms(img, mask)
	exudates = detect_exudates(img, mask)
	hemorrhages = detect_hemorrhages(img, mask)

	return {
		"vessels": vessels,
		"microaneurysms": microaneurysms,
		"exudates": exudates,
		"hemorrhages": hemorrhages,
	}


def locate_optic_disc_and_fovea(
	image: Union[str, Path, np.ndarray],
	mask: Optional[np.ndarray] = None,
) -> Dict[str, Union[Tuple[int, int], int, float]]:
	"""Locate the Optic Disc (brightest vascular convergence) and estimate Fovea position.

	Returns:
		dict with:
			'optic_disc_center': (x, y)
			'optic_disc_radius': r
			'fovea_center': (x, y)
			'fovea_radius': r
	"""
	img = load_image(image)
	h, w = img.shape[:2]
	if mask is None:
		mask, _ = get_retina_mask(img)

	# Optic disc is bright in both green and red channels
	g = img[:, :, 1].astype(np.float32)
	r = img[:, :, 2].astype(np.float32)
	brightness = (g * 0.5 + r * 0.5)

	# Erode mask slightly to avoid perimeter illumination artifacts
	eroded_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
	blurred = cv2.GaussianBlur(brightness, (25, 25), 0)
	blurred_masked = cv2.bitwise_and(blurred, blurred, mask=eroded_mask)

	_, _, _, max_loc = cv2.minMaxLoc(blurred_masked)
	od_x, od_y = max_loc
	od_radius = int(min(h, w) * 0.08)

	# Estimate Fovea: temporally offset by ~2.5 optic disc diameters
	if od_x > w // 2:
		fovea_x = max(int(od_x - 2.5 * od_radius), int(w * 0.25))
	else:
		fovea_x = min(int(od_x + 2.5 * od_radius), int(w * 0.75))
	fovea_y = int(od_y)
	fovea_radius = int(od_radius * 0.6)

	return {
		"optic_disc_center": (od_x, od_y),
		"optic_disc_radius": od_radius,
		"fovea_center": (fovea_x, fovea_y),
		"fovea_radius": fovea_radius,
	}


def render_vessel_overlay(
	image: Union[str, Path, np.ndarray],
	vessels_mask: np.ndarray,
	alpha: float = 0.45,
) -> np.ndarray:
	"""Render a clinical vascular angiographic tree overlay in electric cyan."""
	img = load_image(image)
	overlay = img.copy()

	# Electric cyan color for blood vessels [B=255, G=220, R=0]
	cyan = np.array([255, 220, 0], dtype=np.uint8)
	colored_vessels = np.zeros_like(img)
	colored_vessels[vessels_mask > 0] = cyan

	# Alpha blend only on vessel pixels
	vessel_indices = vessels_mask > 0
	overlay[vessel_indices] = cv2.addWeighted(
		img[vessel_indices], 1.0 - alpha, colored_vessels[vessel_indices], alpha, 0
	)
	return overlay


def render_anatomy_overlay(
	image: Union[str, Path, np.ndarray],
	anatomy_info: Dict[str, Union[Tuple[int, int], int]],
) -> np.ndarray:
	"""Draw surgical markers for Optic Disc (yellow ring) and Fovea (cyan target)."""
	img = load_image(image).copy()
	od_c = anatomy_info["optic_disc_center"]
	od_r = anatomy_info["optic_disc_radius"]
	fovea_c = anatomy_info["fovea_center"]
	fovea_r = anatomy_info["fovea_radius"]

	# Draw Optic Disc target ring (amber/yellow)
	cv2.circle(img, od_c, od_r, (0, 215, 255), 2, cv2.LINE_AA)
	cv2.circle(img, od_c, 3, (0, 215, 255), -1, cv2.LINE_AA)
	cv2.putText(
		img, "OPTIC DISC", (od_c[0] - 40, od_c[1] - od_r - 8),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA
	)

	# Draw Fovea/Macula target reticle (cyan)
	cv2.circle(img, fovea_c, fovea_r, (255, 220, 0), 2, cv2.LINE_AA)
	cv2.drawMarker(img, fovea_c, (255, 220, 0), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
	cv2.putText(
		img, "FOVEA / MACULA", (fovea_c[0] - 50, fovea_c[1] + fovea_r + 16),
		cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1, cv2.LINE_AA
	)

	return img


def render_composite_overlay(
	image: Union[str, Path, np.ndarray],
	heatmap_overlay: np.ndarray,
	vessels_mask: np.ndarray,
	anatomy_info: Dict[str, Union[Tuple[int, int], int]],
) -> np.ndarray:
	"""Merge Grad-CAM lesions, vascular tree, and anatomical landmarks into one multi-structure view."""
	# Start from heatmap overlay
	composite = heatmap_overlay.copy()

	# Blend vessels in electric cyan
	cyan = np.array([255, 220, 0], dtype=np.uint8)
	vessel_indices = vessels_mask > 0
	composite[vessel_indices] = cv2.addWeighted(
		composite[vessel_indices], 0.65, np.tile(cyan, (np.count_nonzero(vessel_indices), 1)), 0.35, 0
	)

	# Overlay anatomical targets
	composite = render_anatomy_overlay(composite, anatomy_info)
	return composite

