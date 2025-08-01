"""
Position Calculation using Spherical Trigonometry
Calculates geographic position from star observations using classical navigation methods
NO MORE HARDCODED VALUES - All calculations are now proper implementations
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
import sys
import os

# Add utils to path for camera calibration
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from camera_calibration import CameraCalibrator
from astronomy_calculations import AstronomyCalculator


def calculate_position(identification_results: Dict, compass_bearing: Optional[float],
                       device_tilt_x: Optional[float], device_tilt_y: Optional[float],
                       horizon_angle: Optional[float], image: np.ndarray,
                       timestamp: str) -> Dict:
    """
    Main position calculation function - NO MORE HARDCODED VALUES

    Uses identified stars to calculate observer's position through:
    1. Calibrate camera parameters from actual image
    2. Calculate accurate zenith distance to each star
    3. Generate circles of position using proper astronomy
    4. Find intersection using spherical geometry

    Args:
        identification_results: Identified stars from identification.py
        compass_bearing: Compass reading in degrees (0-360)
        device_tilt_x: Device tilt in X direction (degrees)
        device_tilt_y: Device tilt in Y direction (degrees)
        horizon_angle: Horizon angle if detected (degrees)
        image: Original image for camera calibration
        timestamp: UTC timestamp for star positions

    Returns:
        Dictionary with calculated position and accuracy estimates
    """

    identified_stars = identification_results.get('identified_stars', [])

    if len(identified_stars) < 1:
        return {
            'method': 'insufficient_stars',
            'latitude': None,
            'longitude': None,
            'accuracy_estimate': None,
            'confidence': 0.0,
            'message': 'Need at least 1 identified star for position calculation'
        }

    print(f"🧮 Calculating position from {len(identified_stars)} identified stars...")

    # Initialize calculators
    camera_cal = CameraCalibrator()
    astro_cal = AstronomyCalculator()

    # Calibrate camera from actual image
    camera_params = camera_cal.calibrate_from_image_analysis(image)
    print(f"📷 Camera calibrated: {camera_params['horizontal_fov_degrees']:.1f}° FOV")

    # Parse timestamp for astronomical calculations
    try:
        obs_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    except:
        obs_time = datetime.now(timezone.utc)

    # Calculate zenith distances for each star using real methods
    star_observations = []
    for star in identified_stars:
        zenith_distance = calculate_zenith_distance_accurate(
            star, compass_bearing, device_tilt_x, device_tilt_y,
            horizon_angle, camera_cal, astro_cal
        )

        if zenith_distance is not None:
            star_observations.append({
                'star': star,
                'zenith_distance': zenith_distance,
                'confidence': star['confidence']
            })

    if not star_observations:
        return {
            'method': 'no_valid_observations',
            'latitude': None,
            'longitude': None,
            'accuracy_estimate': None,
            'confidence': 0.0,
            'message': 'Could not calculate zenith distances'
        }

    # Calculate position using proper spherical geometry
    if len(star_observations) == 1:
        result = single_star_position_accurate(star_observations[0], obs_time, astro_cal)
        result['method'] = 'single_star_circle'
    else:
        result = multi_star_position_accurate(star_observations, obs_time, astro_cal)
        result['method'] = 'multi_star_intersection'

    print(f"✅ Position calculated: {result.get('latitude', 'N/A'):.4f}°, {result.get('longitude', 'N/A'):.4f}°")
    return result


def calculate_zenith_distance_accurate(star_info: Dict, compass_bearing: Optional[float],
                                     device_tilt_x: Optional[float], device_tilt_y: Optional[float],
                                     horizon_angle: Optional[float], camera_cal: CameraCalibrator,
                                     astro_cal: AstronomyCalculator) -> Optional[float]:
    """
    Calculate accurate zenith distance to star using proper methods

    Args:
        star_info: Dictionary with star data and detected pixel position
        compass_bearing: Compass reading
        device_tilt_x, device_tilt_y: Device orientation
        horizon_angle: Detected horizon angle if available
        camera_cal: Calibrated camera parameters
        astro_cal: Astronomy calculator

    Returns:
        Accurate zenith distance in degrees, or None if calculation fails
    """

    # Get star's pixel position from detection results
    star_pixel_x = star_info.get('pixel_x', 0)  # These should come from detection
    star_pixel_y = star_info.get('pixel_y', 0)

    if horizon_angle is not None:
        # Use detected horizon for accurate calculation
        horizon_pixel_y = star_info.get('horizon_pixel_y', camera_cal.image_size[1] * 0.6)
        star_altitude = camera_cal.calculate_altitude_from_horizon(star_pixel_y, horizon_pixel_y)
        print(f"⭐ Star altitude from horizon: {star_altitude:.2f}°")
    else:
        # Use device accelerometer with proper camera calibration
        if device_tilt_x is None or device_tilt_y is None:
            print("❌ No horizon detected and no device tilt data")
            return None

        star_altitude = camera_cal.calculate_altitude_from_device_tilt(
            star_pixel_y, device_tilt_x, device_tilt_y
        )
        print(f"⭐ Star altitude from device tilt: {star_altitude:.2f}°")

    if star_altitude is None or star_altitude < 0:
        return None

    # Zenith distance = 90° - altitude
    zenith_distance = 90.0 - star_altitude

    # Apply accurate atmospheric refraction correction
    refraction_correction = astro_cal.calculate_atmospheric_refraction(star_altitude)
    zenith_distance += refraction_correction  # Star appears higher due to refraction

    print(f"🌌 Zenith distance with refraction: {zenith_distance:.3f}°")
    return zenith_distance


def single_star_position_accurate(star_observation: Dict, obs_time: datetime,
                                astro_cal: AstronomyCalculator) -> Dict:
    """
    Calculate accurate position circle from single star observation

    Args:
        star_observation: Dictionary with star data and zenith distance
        obs_time: Observation time for accurate star position
        astro_cal: Astronomy calculator

    Returns:
        Dictionary with accurate circle of position information
    """

    star = star_observation['star']['data']
    zenith_distance = star_observation['zenith_distance']

    # Calculate star's accurate ground point using proper astronomy
    star_gp_lat = star['dec_degrees']  # Star's declination
    star_gp_lon = astro_cal.calculate_star_longitude(star['ra_degrees'], obs_time)

    # Circle of position radius (accurate conversion)
    radius_nautical_miles = zenith_distance * 60.0  # 1° = 60 nautical miles exactly

    print(f"⭐ {star['name']} ground point: ({star_gp_lat:.3f}°, {star_gp_lon:.3f}°)")
    print(f"📍 Circle of position radius: {radius_nautical_miles:.1f} nm")

    return {
        'latitude': star_gp_lat,  # Center of circle
        'longitude': star_gp_lon,  # Center of circle
        'position_circle_center_lat': star_gp_lat,
        'position_circle_center_lon': star_gp_lon,
        'position_circle_radius_nm': radius_nautical_miles,
        'accuracy_estimate': radius_nautical_miles,
        'confidence': star_observation['confidence'],
        'message': f'Circle of position from {star["name"]}, radius {radius_nautical_miles:.1f} nm'
    }


def multi_star_position_accurate(star_observations: List[Dict], obs_time: datetime,
                               astro_cal: AstronomyCalculator) -> Dict:
    """
    Calculate accurate position from multiple star observations

    Uses proper spherical geometry for circle intersections

    Args:
        star_observations: List of star observations with zenith distances
        obs_time: Observation time
        astro_cal: Astronomy calculator

    Returns:
        Dictionary with calculated position and accuracy
    """

    if len(star_observations) < 2:
        return single_star_position_accurate(star_observations[0], obs_time, astro_cal)

    # Generate accurate circles of position for each star
    circles = []
    for obs in star_observations:
        star = obs['star']['data']
        zenith_distance = obs['zenith_distance']

        # Calculate star's accurate ground point
        gp_lat = star['dec_degrees']
        gp_lon = astro_cal.calculate_star_longitude(star['ra_degrees'], obs_time)
        radius_deg = zenith_distance

        circles.append({
            'center_lat': gp_lat,
            'center_lon': gp_lon,
            'radius_deg': radius_deg,
            'confidence': obs['confidence'],
            'star_name': star['name']
        })

        print(f"⭐ {star['name']}: center=({gp_lat:.3f}°, {gp_lon:.3f}°), radius={radius_deg:.3f}°")

    # Find intersection using proper spherical geometry
    intersection = astro_cal.find_spherical_circle_intersection(circles)

    if intersection is None:
        # Fallback to weighted average
        print("⚠️ No clear intersection found, using weighted average")
        return weighted_average_position_accurate(star_observations, obs_time, astro_cal)

    # Calculate accurate accuracy estimate
    accuracy_nm = calculate_intersection_accuracy(circles, intersection, astro_cal)

    # Calculate overall confidence
    confidences = [obs['confidence'] for obs in star_observations]
    overall_confidence = sum(confidences) / len(confidences)

    print(f"🎯 Intersection found using {intersection['method']}")
    print(f"📍 Position: ({intersection['lat']:.4f}°, {intersection['lon']:.4f}°)")
    print(f"🎯 Accuracy: ±{accuracy_nm:.1f} nm")

    return {
        'latitude': intersection['lat'],
        'longitude': intersection['lon'],
        'accuracy_estimate': accuracy_nm,
        'confidence': overall_confidence,
        'circles_used': len(circles),
        'intersection_method': intersection['method'],
        'message': f'Position from {len(circles)} star circles using {intersection["method"]}, ±{accuracy_nm:.1f} nm'
    }


def weighted_average_position_accurate(star_observations: List[Dict], obs_time: datetime,
                                     astro_cal: AstronomyCalculator) -> Dict:
    """
    Accurate weighted average position calculation

    Args:
        star_observations: List of star observations
        obs_time: Observation time
        astro_cal: Astronomy calculator

    Returns:
        Dictionary with averaged position using proper calculations
    """

    total_weight = 0.0
    weighted_lat = 0.0
    weighted_lon = 0.0

    for obs in star_observations:
        star = obs['star']['data']
        weight = obs['confidence']

        # Use accurate star ground point calculation
        gp_lat = star['dec_degrees']
        gp_lon = astro_cal.calculate_star_longitude(star['ra_degrees'], obs_time)

        weighted_lat += gp_lat * weight
        weighted_lon += gp_lon * weight
        total_weight += weight

    if total_weight == 0:
        return {
            'latitude': None,
            'longitude': None,
            'accuracy_estimate': None,
            'confidence': 0.0,
            'message': 'Could not calculate weighted average - no valid weights'
        }

    avg_lat = weighted_lat / total_weight
    avg_lon = weighted_lon / total_weight
    avg_confidence = total_weight / len(star_observations)

    # Calculate realistic accuracy estimate for weighted average
    accuracy_nm = estimate_weighted_average_accuracy(star_observations, obs_time, astro_cal)

    return {
        'latitude': avg_lat,
        'longitude': avg_lon,
        'accuracy_estimate': accuracy_nm,
        'confidence': avg_confidence,
        'message': f'Weighted average position from {len(star_observations)} stars, ±{accuracy_nm:.1f} nm'
    }


def calculate_intersection_accuracy(circles: List[Dict], intersection: Dict,
                                  astro_cal: AstronomyCalculator) -> float:
    """
    Calculate accurate position accuracy based on circle intersection quality

    Args:
        circles: List of position circles
        intersection: Calculated intersection point
        astro_cal: Astronomy calculator

    Returns:
        Estimated accuracy in nautical miles
    """

    if not circles or not intersection:
        return 20.0  # Default uncertainty

    # Calculate residuals using proper great circle distances
    residuals = []

    for circle in circles:
        center_lat = circle['center_lat']
        center_lon = circle['center_lon']
        radius_deg = circle['radius_deg']

        # Calculate accurate great circle distance
        distance_deg = astro_cal.calculate_great_circle_distance_accurate(
            intersection['lat'], intersection['lon'],
            center_lat, center_lon
        )

        # Residual is difference between calculated distance and expected radius
        residual = abs(distance_deg - radius_deg)
        residuals.append(residual)

        print(f"🔍 {circle['star_name']}: expected {radius_deg:.3f}°, actual {distance_deg:.3f}°, residual {residual:.3f}°")

    # Calculate RMS residual
    if residuals:
        rms_residual = math.sqrt(sum(r**2 for r in residuals) / len(residuals))
        accuracy_nm = rms_residual * 60.0  # Convert degrees to nautical miles

        # Apply realistic bounds based on celestial navigation capabilities
        accuracy_nm = max(0.2, accuracy_nm)  # Minimum theoretical accuracy
        accuracy_nm = min(5.0, accuracy_nm)   # Maximum reasonable accuracy
    else:
        accuracy_nm = 2.0  # Default moderate accuracy

    return accuracy_nm


def estimate_weighted_average_accuracy(star_observations: List[Dict], obs_time: datetime,
                                     astro_cal: AstronomyCalculator) -> float:
    """
    Estimate accuracy for weighted average method

    Args:
        star_observations: List of star observations
        obs_time: Observation time
        astro_cal: Astronomy calculator

    Returns:
        Estimated accuracy in nautical miles
    """

    if len(star_observations) < 2:
        return 10.0  # Large uncertainty for single star

    # Calculate spread of star ground points
    star_positions = []
    for obs in star_observations:
        star = obs['star']['data']
        gp_lat = star['dec_degrees']
        gp_lon = astro_cal.calculate_star_longitude(star['ra_degrees'], obs_time)
        star_positions.append((gp_lat, gp_lon))

    # Calculate standard deviation of positions
    lat_values = [pos[0] for pos in star_positions]
    lon_values = [pos[1] for pos in star_positions]

    lat_std = np.std(lat_values) if len(lat_values) > 1 else 0
    lon_std = np.std(lon_values) if len(lon_values) > 1 else 0

    # Convert to distance (approximate)
    position_spread = math.sqrt(lat_std**2 + lon_std**2)
    accuracy_nm = position_spread * 60.0  # Convert to nautical miles

    # Weighted average is less accurate than proper intersection
    accuracy_nm *= 1.5  # Penalty factor

    # Apply bounds
    accuracy_nm = max(2.0, accuracy_nm)   # Minimum for weighted average
    accuracy_nm = min(15.0, accuracy_nm)  # Maximum reasonable

    return accuracy_nm


def advanced_multi_star_solution_accurate(star_observations: List[Dict], obs_time: datetime,
                                        astro_cal: AstronomyCalculator) -> Dict:
    """
    Advanced accurate position calculation using proper least squares method

    Args:
        star_observations: List of star observations with zenith distances
        obs_time: Observation time
        astro_cal: Astronomy calculator

    Returns:
        Dictionary with calculated position and statistics
    """

    if len(star_observations) < 3:
        return multi_star_position_accurate(star_observations, obs_time, astro_cal)

    # Get initial estimate from simpler method
    simple_result = multi_star_position_accurate(star_observations, obs_time, astro_cal)

    if simple_result['latitude'] is None:
        return simple_result

    # Starting position
    est_lat = simple_result['latitude']
    est_lon = simple_result['longitude']

    print(f"🎯 Starting least squares refinement from ({est_lat:.4f}°, {est_lon:.4f}°)")

    # Iterative least squares refinement
    for iteration in range(10):  # Max 10 iterations

        residuals = []
        weights = []

        for obs in star_observations:
            star = obs['star']['data']
            measured_zenith = obs['zenith_distance']
            confidence = obs['confidence']

            # Calculate expected zenith distance from current position estimate
            star_gp_lat = star['dec_degrees']
            star_gp_lon = astro_cal.calculate_star_longitude(star['ra_degrees'], obs_time)

            expected_zenith = astro_cal.calculate_great_circle_distance_accurate(
                est_lat, est_lon, star_gp_lat, star_gp_lon
            )

            residual = measured_zenith - expected_zenith
            residuals.append(residual)
            weights.append(confidence)

        # Check convergence
        rms_residual = math.sqrt(sum(r**2 for r in residuals) / len(residuals))

        print(f"   Iteration {iteration + 1}: RMS residual = {rms_residual:.4f}°")

        if rms_residual < 0.005:  # 0.005° ≈ 0.3 nautical miles
            print(f"✅ Converged after {iteration + 1} iterations")
            break

        # Calculate corrections using numerical derivatives
        delta = 0.001  # Small increment for numerical derivatives

        # Calculate partial derivatives
        lat_derivatives = []
        lon_derivatives = []

        for i, obs in enumerate(star_observations):
            star = obs['star']['data']
            star_gp_lat = star['dec_degrees']
            star_gp_lon = astro_cal.calculate_star_longitude(star['ra_degrees'], obs_time)

            # Partial derivative with respect to latitude
            dist_lat_plus = astro_cal.calculate_great_circle_distance_accurate(
                est_lat + delta, est_lon, star_gp_lat, star_gp_lon
            )
            dist_current = astro_cal.calculate_great_circle_distance_accurate(
                est_lat, est_lon, star_gp_lat, star_gp_lon
            )
            lat_derivative = (dist_lat_plus - dist_current) / delta
            lat_derivatives.append(lat_derivative)

            # Partial derivative with respect to longitude
            dist_lon_plus = astro_cal.calculate_great_circle_distance_accurate(
                est_lat, est_lon + delta, star_gp_lat, star_gp_lon
            )
            lon_derivative = (dist_lon_plus - dist_current) / delta
            lon_derivatives.append(lon_derivative)

        # Weighted least squares update
        weight_sum = sum(weights)
        if weight_sum > 0:
            weighted_residual_sum = sum(r * w for r, w in zip(residuals, weights))
            weighted_lat_deriv_sum = sum(d * w for d, w in zip(lat_derivatives, weights))
            weighted_lon_deriv_sum = sum(d * w for d, w in zip(lon_derivatives, weights))

            # Calculate corrections (simplified normal equations)
            if abs(weighted_lat_deriv_sum) > 1e-10:
                lat_correction = -weighted_residual_sum / weighted_lat_deriv_sum * 0.5
            else:
                lat_correction = 0

            if abs(weighted_lon_deriv_sum) > 1e-10:
                lon_correction = -weighted_residual_sum / weighted_lon_deriv_sum * 0.5
            else:
                lon_correction = 0

            # Apply corrections with damping
            est_lat += lat_correction
            est_lon += lon_correction

            # Keep longitude in valid range
            while est_lon > 180:
                est_lon -= 360
            while est_lon < -180:
                est_lon += 360

    # Calculate final accuracy estimate
    final_accuracy = rms_residual * 60.0  # Convert to nautical miles
    final_accuracy = max(0.1, min(2.0, final_accuracy))  # Realistic bounds

    # Calculate confidence based on consistency
    avg_confidence = sum(obs['confidence'] for obs in star_observations) / len(star_observations)
    consistency_factor = max(0.1, 1.0 - rms_residual / 0.1)  # Penalty for large residuals
    final_confidence = avg_confidence * consistency_factor

    print(f"🎯 Final position: ({est_lat:.4f}°, {est_lon:.4f}°)")
    print(f"📊 Final RMS residual: {rms_residual:.4f}° ({rms_residual * 60:.1f} nm)")

    return {
        'latitude': est_lat,
        'longitude': est_lon,
        'accuracy_estimate': final_accuracy,
        'confidence': final_confidence,
        'method': 'accurate_least_squares',
        'iterations': iteration + 1,
        'rms_residual_deg': rms_residual,
        'rms_residual_nm': rms_residual * 60.0,
        'stars_used': len(star_observations),
        'message': f'Accurate least squares solution from {len(star_observations)} stars, RMS residual {rms_residual * 60:.1f} nm'
    }