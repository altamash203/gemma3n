"""
Sun Position Calculator for Celestial Navigation
Calculates sun's position for daytime navigation using sun sights
"""

import math
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional


def calculate_sun_position(timestamp: str, latitude_estimate: Optional[float] = None,
                           longitude_estimate: Optional[float] = None) -> Dict:
    """
    Calculate sun's current position (Right Ascension and Declination)

    Sun navigation is often more accurate than star navigation because:
    - Sun is much brighter and easier to see
    - Only need one object (no star identification required)
    - Works in daytime when stars not visible

    Args:
        timestamp: UTC timestamp in ISO format
        latitude_estimate: Rough observer latitude (for visibility check)
        longitude_estimate: Rough observer longitude (for visibility check)

    Returns:
        Dictionary with sun's celestial coordinates and ground point
    """

    print("☀️ Calculating sun position for navigation...")

    # Parse timestamp
    try:
        obs_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    except:
        obs_time = datetime.now(timezone.utc)

    # Calculate sun's position using simplified astronomical formulas
    sun_ra, sun_dec = calculate_sun_ra_dec(obs_time)

    # Calculate sun's ground point (geographical position directly below sun)
    sun_gp_lat, sun_gp_lon = calculate_sun_ground_point(obs_time, sun_ra, sun_dec)

    # Check if sun is visible from estimated location
    sun_visible = True
    sun_altitude = None
    sun_azimuth = None

    if latitude_estimate is not None and longitude_estimate is not None:
        sun_altitude, sun_azimuth = calculate_sun_altitude_azimuth(
            obs_time, latitude_estimate, longitude_estimate, sun_ra, sun_dec
        )
        sun_visible = sun_altitude > 0  # Sun is above horizon

    result = {
        'timestamp': timestamp,
        'sun_ra_degrees': sun_ra,
        'sun_dec_degrees': sun_dec,
        'sun_gp_latitude': sun_gp_lat,
        'sun_gp_longitude': sun_gp_lon,
        'sun_visible': sun_visible,
        'sun_altitude': sun_altitude,
        'sun_azimuth': sun_azimuth,
        'method': 'simplified_solar_calculation'
    }

    print(f"☀️ Sun position: RA={sun_ra:.1f}°, Dec={sun_dec:.1f}°, GP=({sun_gp_lat:.1f}°, {sun_gp_lon:.1f}°)")
    return result


def calculate_sun_ra_dec(obs_time: datetime) -> Tuple[float, float]:
    """
    Calculate sun's Right Ascension and Declination using simplified formulas

    This is a simplified calculation suitable for navigation accuracy.
    For higher precision, use professional astronomical libraries like Skyfield.

    Args:
        obs_time: Observation time (UTC)

    Returns:
        Tuple of (Right Ascension in degrees, Declination in degrees)
    """

    # Days since J2000.0 epoch (January 1, 2000, 12:00 UTC)
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    delta = obs_time - j2000
    days_since_j2000 = delta.total_seconds() / 86400.0  # 86400 seconds per day

    # Mean longitude of sun (simplified)
    # Sun moves approximately 360° per year = 0.9856° per day
    mean_longitude = (280.460 + 0.9856474 * days_since_j2000) % 360.0

    # Mean anomaly (simplified)
    mean_anomaly = (357.528 + 0.9856003 * days_since_j2000) % 360.0
    mean_anomaly_rad = math.radians(mean_anomaly)

    # Ecliptic longitude (apply equation of center - simplified)
    equation_of_center = 1.915 * math.sin(mean_anomaly_rad) + 0.020 * math.sin(2 * mean_anomaly_rad)
    ecliptic_longitude = mean_longitude + equation_of_center
    ecliptic_longitude_rad = math.radians(ecliptic_longitude)

    # Obliquity of ecliptic (tilt of Earth's axis)
    # Changes slowly over time, simplified formula
    obliquity = 23.439 - 0.0000004 * days_since_j2000
    obliquity_rad = math.radians(obliquity)

    # Convert ecliptic coordinates to equatorial coordinates
    # Right Ascension
    ra_rad = math.atan2(
        math.cos(obliquity_rad) * math.sin(ecliptic_longitude_rad),
        math.cos(ecliptic_longitude_rad)
    )
    ra_degrees = math.degrees(ra_rad)
    if ra_degrees < 0:
        ra_degrees += 360.0

    # Declination
    dec_rad = math.asin(math.sin(obliquity_rad) * math.sin(ecliptic_longitude_rad))
    dec_degrees = math.degrees(dec_rad)

    return ra_degrees, dec_degrees


def calculate_sun_ground_point(obs_time: datetime, sun_ra: float, sun_dec: float) -> Tuple[float, float]:
    """
    Calculate sun's ground point (geographical position directly below sun)

    The ground point is where the sun is at zenith (directly overhead).
    Latitude = Sun's declination
    Longitude = depends on Greenwich Hour Angle

    Args:
        obs_time: Observation time (UTC)
        sun_ra: Sun's Right Ascension (degrees)
        sun_dec: Sun's Declination (degrees)

    Returns:
        Tuple of (ground point latitude, ground point longitude)
    """

    # Ground point latitude is simply the sun's declination
    gp_latitude = sun_dec

    # Ground point longitude requires Greenwich Hour Angle calculation
    gha = calculate_greenwich_hour_angle(obs_time, sun_ra)

    # Greenwich Hour Angle is measured westward from Greenwich
    # Convert to longitude (eastward from Greenwich)
    gp_longitude = -gha  # Negative because GHA is westward

    # Normalize longitude to [-180, 180] range
    while gp_longitude > 180:
        gp_longitude -= 360
    while gp_longitude < -180:
        gp_longitude += 360

    return gp_latitude, gp_longitude


def calculate_greenwich_hour_angle(obs_time: datetime, object_ra: float) -> float:
    """
    Calculate Greenwich Hour Angle for any celestial object

    GHA = Greenwich Mean Sidereal Time - Right Ascension

    Args:
        obs_time: Observation time (UTC)
        object_ra: Object's Right Ascension (degrees)

    Returns:
        Greenwich Hour Angle in degrees
    """

    # Calculate Greenwich Mean Sidereal Time (simplified)
    gmst = calculate_greenwich_mean_sidereal_time(obs_time)

    # Greenwich Hour Angle = GMST - RA
    gha = gmst - object_ra

    # Normalize to [0, 360) range
    while gha < 0:
        gha += 360
    while gha >= 360:
        gha -= 360

    return gha


def calculate_greenwich_mean_sidereal_time(obs_time: datetime) -> float:
    """
    Calculate Greenwich Mean Sidereal Time (simplified version)

    Sidereal time tracks Earth's rotation relative to stars.
    This is a simplified calculation - professional navigation uses
    more precise algorithms accounting for nutation and precession.

    Args:
        obs_time: Observation time (UTC)

    Returns:
        Greenwich Mean Sidereal Time in degrees
    """

    # Days since J2000.0 epoch
    j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    delta = obs_time - j2000
    days_since_j2000 = delta.total_seconds() / 86400.0

    # Hours since midnight UTC
    hours_since_midnight = obs_time.hour + obs_time.minute / 60.0 + obs_time.second / 3600.0

    # GMST at 0h UT (midnight) - simplified formula
    gmst_0h = 280.46061837 + 360.98564736629 * days_since_j2000

    # Add time since midnight (sidereal rate is ~1.002737909 times solar rate)
    gmst = gmst_0h + hours_since_midnight * 15.04107 * 1.002737909

    # Normalize to [0, 360) range
    gmst = gmst % 360.0

    return gmst


def calculate_sun_altitude_azimuth(obs_time: datetime, observer_lat: float, observer_lon: float,
                                   sun_ra: float, sun_dec: float) -> Tuple[float, float]:
    """
    Calculate sun's altitude and azimuth as seen from observer's location

    Args:
        obs_time: Observation time (UTC)
        observer_lat: Observer's latitude (degrees)
        observer_lon: Observer's longitude (degrees)
        sun_ra: Sun's Right Ascension (degrees)
        sun_dec: Sun's Declination (degrees)

    Returns:
        Tuple of (altitude in degrees, azimuth in degrees)
    """

    # Calculate Local Hour Angle
    gha = calculate_greenwich_hour_angle(obs_time, sun_ra)
    local_hour_angle = gha + observer_lon

    # Convert to radians for trigonometry
    lat_rad = math.radians(observer_lat)
    dec_rad = math.radians(sun_dec)
    lha_rad = math.radians(local_hour_angle)

    # Calculate altitude using spherical trigonometry
    altitude_rad = math.asin(
        math.sin(lat_rad) * math.sin(dec_rad) +
        math.cos(lat_rad) * math.cos(dec_rad) * math.cos(lha_rad)
    )
    altitude = math.degrees(altitude_rad)

    # Calculate azimuth
    azimuth_rad = math.atan2(
        -math.sin(lha_rad),
        math.tan(dec_rad) * math.cos(lat_rad) - math.sin(lat_rad) * math.cos(lha_rad)
    )
    azimuth = math.degrees(azimuth_rad)

    # Normalize azimuth to [0, 360) range
    if azimuth < 0:
        azimuth += 360

    return altitude, azimuth


def calculate_sun_zenith_distance_from_image(sun_detection_data: Dict, horizon_angle: Optional[float],
                                             device_tilt_x: Optional[float], device_tilt_y: Optional[float]) -> \
Optional[float]:
    """
    Calculate zenith distance to sun from image detection data

    Similar to star zenith distance calculation but optimized for sun's brightness
    Sun appears as bright disk rather than point source

    Args:
        sun_detection_data: Sun detection results (center position, brightness)
        horizon_angle: Detected horizon angle if available
        device_tilt_x, device_tilt_y: Device orientation from accelerometer

    Returns:
        Zenith distance to sun in degrees, or None if calculation fails
    """

    if not sun_detection_data:
        return None

    # Calculate sun's altitude above horizon
    if horizon_angle is not None:
        # Use detected horizon for more accurate calculation
        sun_altitude = calculate_sun_altitude_from_horizon(sun_detection_data, horizon_angle)
    else:
        # Use device accelerometer as fallback
        sun_altitude = calculate_sun_altitude_from_tilt(sun_detection_data, device_tilt_x, device_tilt_y)

    if sun_altitude is None:
        return None

    # Zenith distance = 90° - altitude
    zenith_distance = 90.0 - sun_altitude

    # Apply atmospheric refraction correction (same as for stars)
    if sun_altitude > 0:
        # Sun refraction is similar to star refraction but slightly different due to sun's disk
        if sun_altitude > 15:
            refraction_arcmin = 0.97 / math.tan(math.radians(sun_altitude))
        else:
            refraction_arcmin = 1.02 / math.tan(math.radians(sun_altitude + 10.3 / (sun_altitude + 5.11)))

        refraction_deg = refraction_arcmin / 60.0
        zenith_distance += refraction_deg  # Sun appears higher due to refraction

    return zenith_distance


def calculate_sun_altitude_from_horizon(sun_data: Dict, horizon_angle: float) -> Optional[float]:
    """
    Calculate sun altitude using detected horizon line

    Args:
        sun_data: Sun detection data with position
        horizon_angle: Angle of detected horizon

    Returns:
        Sun altitude in degrees above horizon
    """

    # This would use the sun's pixel position relative to horizon line
    # and convert to angular altitude using camera field of view

    # Placeholder - should be replaced with actual pixel-to-angle conversion
    # that accounts for sun's position in image and horizon line position
    estimated_altitude = 45.0  # Placeholder value

    return estimated_altitude


def calculate_sun_altitude_from_tilt(sun_data: Dict, device_tilt_x: Optional[float],
                                     device_tilt_y: Optional[float]) -> Optional[float]:
    """
    Calculate sun altitude using device accelerometer data

    Args:
        sun_data: Sun detection data
        device_tilt_x, device_tilt_y: Device tilt readings

    Returns:
        Sun altitude in degrees
    """

    if device_tilt_x is None or device_tilt_y is None:
        return None

    # Calculate device orientation relative to zenith
    device_zenith_angle = math.sqrt(device_tilt_x ** 2 + device_tilt_y ** 2)

    # Placeholder calculation - should use sun's actual pixel position
    # and camera field of view parameters
    estimated_altitude = 90.0 - device_zenith_angle - 15.0  # Rough estimate

    # Ensure altitude is reasonable
    estimated_altitude = max(0.0, min(90.0, estimated_altitude))

    return estimated_altitude


def sun_sight_navigation(sun_zenith_distance: float, sun_position: Dict) -> Dict:
    """
    Calculate observer position using sun sight (single observation)

    Sun navigation advantage: only need one observation (no star identification)
    Disadvantage: gives circle of position, need multiple observations for exact fix

    Args:
        sun_zenith_distance: Measured zenith distance to sun
        sun_position: Sun's calculated celestial position

    Returns:
        Dictionary with position circle information
    """

    # Sun's ground point (where sun is directly overhead)
    sun_gp_lat = sun_position['sun_gp_latitude']
    sun_gp_lon = sun_position['sun_gp_longitude']

    # Circle of position: all points where sun would have this zenith distance
    radius_nautical_miles = sun_zenith_distance * 60.0  # 1° = 60 nautical miles

    return {
        'method': 'sun_sight',
        'position_type': 'circle',
        'center_latitude': sun_gp_lat,
        'center_longitude': sun_gp_lon,
        'radius_nautical_miles': radius_nautical_miles,
        'zenith_distance': sun_zenith_distance,
        'timestamp': sun_position['timestamp'],
        'accuracy_estimate': radius_nautical_miles,
        'message': f'Sun circle of position, radius {radius_nautical_miles:.1f} nm'
    }


def running_fix_with_sun(sun_observations: List[Dict]) -> Dict:
    """
    Calculate running fix using multiple sun observations taken at different times

    Running fix technique:
    1. Take sun sight at time T1
    2. Wait 2-4 hours for sun to move significantly
    3. Take second sun sight at time T2
    4. Advance first circle of position to time T2 using estimated course/speed
    5. Intersection gives position fix

    Args:
        sun_observations: List of sun sights with timestamps and positions

    Returns:
        Dictionary with calculated position from running fix
    """

    if len(sun_observations) < 2:
        return {
            'method': 'insufficient_observations',
            'message': 'Need at least 2 sun observations for running fix'
        }

    # Sort observations by time
    sun_observations.sort(key=lambda obs: obs['timestamp'])

    first_obs = sun_observations[0]
    last_obs = sun_observations[-1]

    # Calculate time difference
    time1 = datetime.fromisoformat(first_obs['timestamp'].replace('Z', '+00:00'))
    time2 = datetime.fromisoformat(last_obs['timestamp'].replace('Z', '+00:00'))
    time_diff_hours = (time2 - time1).total_seconds() / 3600.0

    if time_diff_hours < 1.0:
        return {
            'method': 'insufficient_time_separation',
            'message': f'Only {time_diff_hours:.1f} hours between observations, need 2+ hours'
        }

    # For simplified calculation, find intersection of two circles
    # Real running fix would advance first circle based on estimated course/speed

    # Use centers of circles as rough position estimates
    lat1, lon1 = first_obs['center_latitude'], first_obs['center_longitude']
    lat2, lon2 = last_obs['center_latitude'], last_obs['center_longitude']

    # Simple intersection calculation (should be replaced with proper spherical geometry)
    intersection_lat = (lat1 + lat2) / 2.0
    intersection_lon = (lon1 + lon2) / 2.0

    # Estimate accuracy based on circle sizes and separation
    avg_radius = (first_obs['radius_nautical_miles'] + last_obs['radius_nautical_miles']) / 2.0
    accuracy_estimate = avg_radius * 0.7  # Running fix typically more accurate than single sight

    return {
        'method': 'sun_running_fix',
        'latitude': intersection_lat,
        'longitude': intersection_lon,
        'accuracy_estimate': accuracy_estimate,
        'time_separation_hours': time_diff_hours,
        'observations_used': len(sun_observations),
        'message': f'Running fix from {len(sun_observations)} sun sights over {time_diff_hours:.1f} hours'
    }