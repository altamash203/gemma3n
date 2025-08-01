"""
Star and Horizon Detection using OpenCV
Detects bright stars as point sources and horizon line if visible
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import math


def detect_stars_and_horizon(image: np.ndarray) -> Dict:
    """
    Main detection function - finds stars and horizon in image

    Detection strategy:
    1. Find bright point sources (stars) using blob detection
    2. Attempt to detect horizon line if visible
    3. Calculate star positions and brightness
    4. Filter out noise and artifacts

    Args:
        image: Enhanced grayscale image from capture.py

    Returns:
        Dictionary containing star positions, brightness, and horizon info
    """

    print("🔍 Starting star and horizon detection...")

    # Detect stars first
    stars = detect_stars(image)

    # Attempt horizon detection
    horizon_results = detect_horizon(image)

    # Calculate angular relationships between stars
    star_angles = calculate_star_angles(stars, image.shape)

    results = {
        'stars': stars,
        'star_count': len(stars),
        'star_angles': star_angles,
        'horizon_detected': horizon_results['detected'],
        'horizon_angle': horizon_results.get('angle'),
        'image_height': image.shape[0],
        'image_width': image.shape[1]
    }

    print(f"✅ Detection complete: {len(stars)} stars found, horizon: {horizon_results['detected']}")
    return results


def detect_stars(image: np.ndarray) -> List[Dict]:
    """
    Detect bright stars using blob detection

    Stars appear as bright point sources with these characteristics:
    - Small circular/slightly elliptical shape
    - Higher brightness than surroundings
    - Relatively isolated (not part of larger structures)

    Args:
        image: Grayscale image

    Returns:
        List of star dictionaries with position and brightness
    """

    # Set up blob detector parameters for star detection
    params = cv2.SimpleBlobDetector_Params()

    # Filter by brightness/area
    params.filterByArea = True
    params.minArea = 5  # Minimum star size in pixels
    params.maxArea = 500  # Maximum to filter out large bright objects

    # Filter by circularity (stars should be roughly circular)
    params.filterByCircularity = True
    params.minCircularity = 0.3  # Allow some elongation due to atmosphere

    # Filter by convexity (stars should be convex shapes)
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by inertia (roundness)
    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)

    # Apply threshold to find bright objects
    # Use adaptive threshold to handle varying brightness across image
    threshold_image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -3
    )

    # Detect blobs
    keypoints = detector.detect(threshold_image)

    # Convert keypoints to star data with brightness calculation
    stars = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])

        # Calculate star brightness in local region
        brightness = calculate_star_brightness(image, x, y)

        # Only keep bright enough objects (filter out noise)
        if brightness > 100:  # Minimum brightness threshold
            stars.append({
                'x': x,
                'y': y,
                'brightness': brightness,
                'size': kp.size
            })

    # Sort stars by brightness (brightest first)
    stars.sort(key=lambda s: s['brightness'], reverse=True)

    # Keep only the brightest stars (practical limit for navigation)
    max_stars = 10  # Most bright stars visible to naked eye
    stars = stars[:max_stars]

    print(f"⭐ Detected {len(stars)} bright stars")
    return stars


def calculate_star_brightness(image: np.ndarray, x: int, y: int, radius: int = 3) -> float:
    """
    Calculate star brightness in local region

    Method: Compare star pixel intensity to local background

    Args:
        image: Grayscale image
        x, y: Star center coordinates
        radius: Radius for brightness calculation

    Returns:
        Brightness value (higher = brighter star)
    """

    h, w = image.shape

    # Ensure coordinates are within image bounds
    x = max(radius, min(w - radius - 1, x))
    y = max(radius, min(h - radius - 1, y))

    # Extract small region around star
    region = image[y - radius:y + radius + 1, x - radius:x + radius + 1]

    # Star brightness is the maximum value in the region
    star_brightness = np.max(region)

    # Background is the median of edge pixels (to avoid including the star)
    edge_pixels = np.concatenate([
        region[0, :],  # Top edge
        region[-1, :],  # Bottom edge
        region[:, 0],  # Left edge
        region[:, -1]  # Right edge
    ])
    background = np.median(edge_pixels)

    # Return contrast (star brightness above background)
    return float(star_brightness - background)


def detect_horizon(image: np.ndarray) -> Dict:
    """
    Attempt to detect horizon line if visible

    Horizon detection helps improve accuracy by providing true vertical reference
    instead of relying on device accelerometer

    Args:
        image: Grayscale image

    Returns:
        Dictionary with horizon detection results
    """

    # Apply edge detection to find strong horizontal lines
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Use Hough Line Transform to find straight lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        return {'detected': False}

    # Look for near-horizontal lines (horizon should be mostly horizontal)
    horizon_candidates = []

    for line in lines:
        rho, theta = line[0]

        # Convert to degrees for easier interpretation
        angle_deg = np.degrees(theta)

        # Horizon should be close to horizontal (0° or 180°)
        # Allow some tolerance for camera tilt
        if (abs(angle_deg) < 15) or (abs(angle_deg - 180) < 15):
            horizon_candidates.append({
                'rho': rho,
                'theta': theta,
                'angle_deg': angle_deg
            })

    if not horizon_candidates:
        return {'detected': False}

    # Take the strongest horizontal line as horizon
    # (Hough lines are sorted by strength)
    horizon = horizon_candidates[0]

    print(f"🌅 Horizon detected at angle: {horizon['angle_deg']:.1f}°")

    return {
        'detected': True,
        'rho': horizon['rho'],
        'theta': horizon['theta'],
        'angle': horizon['angle_deg']
    }


def calculate_star_angles(stars: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
    """
    Calculate angular distances between all pairs of detected stars

    These angular relationships form unique patterns that help identify
    which stars we're looking at by comparing to star catalog

    Args:
        stars: List of detected stars with x,y coordinates
        image_shape: (height, width) of image

    Returns:
        List of angular relationships between star pairs
    """

    if len(stars) < 2:
        return []

    angles = []
    height, width = image_shape

    # Calculate pixel-to-degree conversion
    # Typical phone camera has ~60-70° field of view
    horizontal_fov = 65.0  # degrees (conservative estimate)
    vertical_fov = horizontal_fov * height / width

    pixels_per_degree_x = width / horizontal_fov
    pixels_per_degree_y = height / vertical_fov

    # Calculate angles between all star pairs
    for i in range(len(stars)):
        for j in range(i + 1, len(stars)):
            star1, star2 = stars[i], stars[j]

            # Pixel distance
            dx = star2['x'] - star1['x']
            dy = star2['y'] - star1['y']
            pixel_distance = math.sqrt(dx * dx + dy * dy)

            # Convert to angular distance (rough approximation)
            # More precise calculation would account for lens distortion
            angular_distance = pixel_distance / ((pixels_per_degree_x + pixels_per_degree_y) / 2)

            angles.append({
                'star1_index': i,
                'star2_index': j,
                'angular_distance': round(angular_distance, 2),
                'pixel_distance': round(pixel_distance, 1),
                'brightness_ratio': round(star1['brightness'] / star2['brightness'], 2)
            })

    # Sort by angular distance for easier pattern matching
    angles.sort(key=lambda a: a['angular_distance'])

    print(f"📐 Calculated {len(angles)} angular relationships")
    return angles