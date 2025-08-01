"""
Image Capture and Preprocessing for Celestial Navigation
Handles multiple image averaging and enhancement for better star detection
"""

import cv2
import numpy as np
from typing import List


def process_multiple_images(images: List[np.ndarray]) -> np.ndarray:
    """
    Process multiple images to reduce noise and improve star detection

    Multi-exposure averaging significantly improves precision by:
    - Reducing random sensor noise
    - Enhancing faint stars
    - Improving star centroid accuracy

    Args:
        images: List of OpenCV images (BGR format)

    Returns:
        Single averaged and enhanced image
    """

    if not images:
        raise ValueError("No images provided")

    if len(images) == 1:
        return enhance_single_image(images[0])

    print(f"📸 Processing {len(images)} images for averaging...")

    # Convert all images to same size (use smallest dimensions)
    min_height = min(img.shape[0] for img in images)
    min_width = min(img.shape[1] for img in images)

    # Resize all images to same dimensions
    resized_images = []
    for img in images:
        if img.shape[:2] != (min_height, min_width):
            resized = cv2.resize(img, (min_width, min_height))
            resized_images.append(resized)
        else:
            resized_images.append(img)

    # Convert to float32 for averaging
    float_images = [img.astype(np.float32) for img in resized_images]

    # Average all images
    averaged = np.mean(float_images, axis=0)

    # Convert back to uint8
    averaged_uint8 = np.clip(averaged, 0, 255).astype(np.uint8)

    # Apply enhancement to averaged image
    enhanced = enhance_single_image(averaged_uint8)

    print("✅ Image averaging completed")
    return enhanced


def enhance_single_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance single image for better star visibility

    Enhancement steps:
    1. Convert to grayscale for star detection
    2. Reduce noise while preserving stars
    3. Enhance contrast for faint stars
    4. Apply histogram equalization

    Args:
        image: Single OpenCV image (BGR format)

    Returns:
        Enhanced grayscale image optimized for star detection
    """

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Step 1: Gaussian blur to reduce noise (small kernel to preserve stars)
    # Stars are point sources, so small blur helps with noise but keeps stars sharp
    denoised = cv2.GaussianBlur(gray, (3, 3), 0.8)

    # Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Enhances local contrast, making faint stars more visible
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Step 3: Gamma correction to bring out faint stars
    # Gamma < 1 brightens darker regions where faint stars might be
    gamma = 0.7
    gamma_corrected = np.power(enhanced / 255.0, gamma) * 255.0
    gamma_corrected = np.clip(gamma_corrected, 0, 255).astype(np.uint8)

    print("🔧 Image enhancement applied: denoising + CLAHE + gamma correction")
    return gamma_corrected


def assess_image_quality(image: np.ndarray) -> dict:
    """
    Assess image quality for celestial navigation

    Quality metrics:
    - Overall brightness (should not be too bright - light pollution)
    - Contrast (higher contrast helps star detection)
    - Noise level (lower is better)
    - Sharp edges (indicates good focus)

    Args:
        image: Grayscale image

    Returns:
        Dictionary with quality metrics
    """

    # Calculate basic statistics
    mean_brightness = np.mean(image)
    std_dev = np.std(image)

    # Calculate contrast using standard deviation
    contrast_score = std_dev / 255.0

    # Calculate noise level using Laplacian variance
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpness_score = laplacian.var()

    # Light pollution detection (too bright overall is bad)
    light_pollution_score = min(1.0, max(0.0, (100 - mean_brightness) / 100))

    # Overall quality score (0-10 scale)
    overall_score = (
            contrast_score * 3 +  # Contrast is most important
            (sharpness_score / 1000) * 2 +  # Sharpness helps star detection
            light_pollution_score * 5  # Low light pollution is crucial
    )
    overall_score = min(10.0, max(0.0, overall_score))

    return {
        'overall_quality': round(overall_score, 1),
        'mean_brightness': round(mean_brightness, 1),
        'contrast': round(contrast_score, 3),
        'sharpness': round(sharpness_score, 1),
        'light_pollution': round(light_pollution_score, 3),
        'suitable_for_navigation': overall_score >= 5.0
    }