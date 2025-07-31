import cv2
import numpy as np
import math


def detect_three_brightest_stars(image_path):
    """
    Detect 3 brightest blobs (stars) and return pixel coordinates
    """
    # Read image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance image for star detection
    # Apply CLAHE for better contrast in low light
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Reduce noise
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Alternative approach using contour detection for stars
    # Apply threshold to find bright objects
    _, thresh = cv2.threshold(denoised, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area and shape
    star_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 10 < area < 500:  # Filter by size
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            # Check if roughly circular (aspect ratio close to 1)
            if 0.7 < aspect_ratio < 1.3:
                # Calculate centroid
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']

                    # Use area as brightness measure
                    brightness = area / 500.0  # Normalize

                    star_candidates.append({
                        'x': cx,
                        'y': cy,
                        'brightness': brightness,
                        'area': area
                    })

    # Sort by brightness and get top candidates
    star_candidates.sort(key=lambda x: x['brightness'], reverse=True)
    keypoints = star_candidates

    # Get top 3 brightest stars
    top_3_stars = keypoints[:3]

    # Extract coordinates
    star_pixels = []
    for i, star in enumerate(top_3_stars):
        star_pixels.append({
            'id': f'Star_{i + 1}',
            'x': star['x'],
            'y': star['y'],
            'brightness': star['brightness'],
            'area': star['area']
        })

    return star_pixels, image


def calculate_pixel_distance(star1, star2):
    """
    Calculate pixel distance between two stars
    """
    dx = star2['x'] - star1['x']
    dy = star2['y'] - star1['y']
    return math.sqrt(dx * dx + dy * dy)


def pixel_to_angle(pixel_distance, image_width, fov_degrees=60):
    """
    Convert pixel distance to angle using field of view
    Assumes typical smartphone camera FOV of 60 degrees
    """
    angle_per_pixel = fov_degrees / image_width
    angle = pixel_distance * angle_per_pixel
    return angle


def calculate_star_triangle_angles(stars, image_width):
    """
    Calculate the three angles of triangle formed by 3 stars
    Returns angles in degrees
    """
    if len(stars) < 3:
        return None

    # Get pixel distances between stars
    dist_12 = calculate_pixel_distance(stars[0], stars[1])  # Star1 to Star2
    dist_23 = calculate_pixel_distance(stars[1], stars[2])  # Star2 to Star3
    dist_13 = calculate_pixel_distance(stars[0], stars[2])  # Star1 to Star3

    # Convert pixel distances to angles
    angle_12 = pixel_to_angle(dist_12, image_width)
    angle_23 = pixel_to_angle(dist_23, image_width)
    angle_13 = pixel_to_angle(dist_13, image_width)

    return {
        'star1_star2_angle': round(angle_12, 2),
        'star2_star3_angle': round(angle_23, 2),
        'star1_star3_angle': round(angle_13, 2),
        'triangle_pattern': [round(angle_12, 2), round(angle_23, 2), round(angle_13, 2)]
    }


def process_star_image(image_path):
    """
    Main function: Process image and return star coordinates + angles
    """
    print(f"Processing image: {image_path}")

    # Step 1: Detect 3 brightest stars
    stars, original_image = detect_three_brightest_stars(image_path)

    if len(stars) < 3:
        return {
            'success': False,
            'message': f'Only {len(stars)} stars detected. Need at least 3.',
            'stars_found': len(stars)
        }

    # Step 2: Calculate angles between stars
    image_width = original_image.shape[1]
    angles = calculate_star_triangle_angles(stars, image_width)

    # Return results
    return {
        'success': True,
        'stars_detected': len(stars),
        'star_coordinates': stars,
        'triangle_angles': angles
    }


# Example usage
if __name__ == "__main__":
    # Test with your star image
    result = process_star_image("star_image.jpg")

    if result['success']:
        print("✅ Star detection successful!")
        print(f"Stars found: {result['stars_detected']}")

        print("\n📍 Star Coordinates:")
        for star in result['star_coordinates']:
            print(f"  {star['id']}: ({star['x']:.1f}, {star['y']:.1f}) brightness: {star['brightness']:.3f}")

        print(f"\n📐 Triangle Angles:")
        angles = result['triangle_angles']
        print(f"  Star1-Star2: {angles['star1_star2_angle']}°")
        print(f"  Star2-Star3: {angles['star2_star3_angle']}°")
        print(f"  Star1-Star3: {angles['star1_star3_angle']}°")
        print(f"  Pattern: {angles['triangle_pattern']}")

    else:
        print(f"❌ {result['message']}")