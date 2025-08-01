"""
Star Pattern Identification and Matching
Identifies stars by comparing angular patterns to star catalog
"""

import json
import math
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np


def identify_stars_from_pattern(detection_results: Dict, timestamp: str,
                                lat_estimate: Optional[float] = None,
                                lon_estimate: Optional[float] = None) -> Dict:
    """
    Main star identification function using angular pattern matching

    Process:
    1. Load star catalog and filter by time/location visibility
    2. Generate theoretical angular patterns for visible stars
    3. Match detected pattern against catalog patterns
    4. Return most likely star identifications

    Args:
        detection_results: Results from detection.py containing stars and angles
        timestamp: UTC timestamp string (ISO format)
        lat_estimate: Rough latitude for filtering visible stars
        lon_estimate: Rough longitude for filtering visible stars

    Returns:
        Dictionary with identified stars and confidence scores
    """

    stars = detection_results['stars']
    star_angles = detection_results['star_angles']

    if len(stars) < 2:
        return {
            'method': 'insufficient_data',
            'identified_stars': [],
            'confidence': 0.0,
            'message': 'Need at least 2 stars for identification'
        }

    print(f"🔍 Identifying {len(stars)} detected stars...")

    # Load and filter star catalog
    visible_stars = get_visible_stars(timestamp, lat_estimate, lon_estimate)

    if len(visible_stars) < 2:
        return {
            'method': 'no_visible_stars',
            'identified_stars': [],
            'confidence': 0.0,
            'message': 'No bright stars visible at this time/location'
        }

    # Generate theoretical patterns for visible stars
    theoretical_patterns = generate_theoretical_patterns(visible_stars)

    # Match detected pattern to theoretical patterns
    best_match = match_star_patterns(star_angles, theoretical_patterns, stars)

    if best_match['confidence'] > 0.3:  # Minimum confidence threshold
        print(f"✅ Stars identified with {best_match['confidence']:.1f} confidence")
        return best_match
    else:
        return {
            'method': 'no_match',
            'identified_stars': [],
            'confidence': best_match['confidence'],
            'message': 'Could not match detected pattern to known stars'
        }


def get_visible_stars(timestamp: str, lat_estimate: Optional[float],
                      lon_estimate: Optional[float]) -> List[Dict]:
    """
    Filter star catalog to find stars visible at given time and location

    Visibility criteria:
    1. Star is above horizon (altitude > 0°)
    2. Star is bright enough (magnitude < 1.5)
    3. Time of year is appropriate for star's season

    Args:
        timestamp: UTC timestamp
        lat_estimate: Observer latitude
        lon_estimate: Observer longitude

    Returns:
        List of visible stars from catalog
    """

    try:
        with open('data/star_catalog.json', 'r') as f:
            catalog = json.load(f)
        bright_stars = catalog['bright_stars']
    except:
        # Fallback to hardcoded essential navigation stars if file not found
        bright_stars = get_essential_navigation_stars()

    # Parse timestamp
    try:
        obs_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    except:
        obs_time = datetime.utcnow()

    visible_stars = []

    for star in bright_stars:
        # Basic visibility check based on season/time
        if is_star_visible_by_season(star, obs_time):
            # If we have location, do more precise visibility calculation
            if lat_estimate is not None:
                if is_star_above_horizon(star, obs_time, lat_estimate, lon_estimate or 0):
                    visible_stars.append(star)
            else:
                # Without precise location, include if seasonally visible
                visible_stars.append(star)

    # Sort by brightness (magnitude - lower is brighter)
    visible_stars.sort(key=lambda s: s['magnitude'])

    # Limit to brightest stars to reduce computation
    max_stars = 15
    visible_stars = visible_stars[:max_stars]

    print(f"⭐ {len(visible_stars)} stars visible at this time")
    return visible_stars


def is_star_visible_by_season(star: Dict, obs_time: datetime) -> bool:
    """
    Check if star is visible based on time of year (rough approximation)

    Args:
        star: Star dictionary with RA/DEC
        obs_time: Observation time

    Returns:
        True if star is likely visible
    """

    # Get month number (1-12)
    month = obs_time.month

    # Star's Right Ascension in hours
    star_ra_hours = star['ra_hours']

    # Stars are best visible when opposite the Sun
    # Sun's RA changes ~2 hours per month
    sun_ra_hours = (month - 3) * 2  # Rough approximation (March = 0 hours)
    sun_ra_hours = sun_ra_hours % 24

    # Star is visible if it's roughly opposite the Sun (±6 hours window)
    ra_diff = abs(star_ra_hours - sun_ra_hours)
    if ra_diff > 12:
        ra_diff = 24 - ra_diff

    # Visible if star is 6-18 hours away from Sun's position
    return 6 <= ra_diff <= 18


def is_star_above_horizon(star: Dict, obs_time: datetime,
                          lat: float, lon: float) -> bool:
    """
    Calculate if star is above horizon at given time/location

    Simplified calculation - more precise version would use proper
    sidereal time and atmospheric refraction corrections

    Args:
        star: Star with RA/DEC coordinates
        obs_time: Observation time
        lat: Observer latitude
        lon: Observer longitude

    Returns:
        True if star is above horizon
    """

    # This is a simplified calculation
    # For production, use proper astronomical libraries like PyEphem or Skyfield

    # Convert to radians
    lat_rad = math.radians(lat)
    dec_rad = math.radians(star['dec_degrees'])

    # Hour angle calculation (simplified)
    # In reality, need proper sidereal time calculation
    hours_since_midnight = obs_time.hour + obs_time.minute / 60.0
    local_sidereal_time = hours_since_midnight + lon / 15.0  # Rough approximation
    hour_angle = local_sidereal_time - star['ra_hours']
    hour_angle_rad = math.radians(hour_angle * 15)

    # Calculate altitude using spherical trigonometry
    altitude_rad = math.asin(
        math.sin(lat_rad) * math.sin(dec_rad) +
        math.cos(lat_rad) * math.cos(dec_rad) * math.cos(hour_angle_rad)
    )

    altitude_deg = math.degrees(altitude_rad)

    # Star is visible if altitude > 10° (avoid horizon effects)
    return altitude_deg > 10.0


def generate_theoretical_patterns(visible_stars: List[Dict]) -> List[Dict]:
    """
    Generate theoretical angular patterns between visible stars

    For each combination of 2-3 stars, calculate expected angular distances
    These patterns will be matched against detected patterns

    Args:
        visible_stars: List of stars visible at observation time

    Returns:
        List of theoretical star patterns with angular relationships
    """

    patterns = []

    # Generate patterns for all pairs of stars
    for i in range(len(visible_stars)):
        for j in range(i + 1, len(visible_stars)):
            star1, star2 = visible_stars[i], visible_stars[j]

            # Calculate angular separation using spherical trigonometry
            angular_distance = calculate_angular_separation(
                star1['ra_degrees'], star1['dec_degrees'],
                star2['ra_degrees'], star2['dec_degrees']
            )

            patterns.append({
                'star1': star1['name'],
                'star2': star2['name'],
                'star1_data': star1,
                'star2_data': star2,
                'angular_distance': angular_distance,
                'brightness_ratio': 10 ** ((star2['magnitude'] - star1['magnitude']) / 2.5)
            })

    return patterns


def calculate_angular_separation(ra1: float, dec1: float,
                                 ra2: float, dec2: float) -> float:
    """
    Calculate angular separation between two celestial objects

    Uses spherical law of cosines for celestial sphere

    Args:
        ra1, dec1: Right Ascension and Declination of first object (degrees)
        ra2, dec2: Right Ascension and Declination of second object (degrees)

    Returns:
        Angular separation in degrees
    """

    # Convert to radians
    ra1_rad = math.radians(ra1)
    dec1_rad = math.radians(dec1)
    ra2_rad = math.radians(ra2)
    dec2_rad = math.radians(dec2)

    # Spherical law of cosines
    cos_separation = (
            math.sin(dec1_rad) * math.sin(dec2_rad) +
            math.cos(dec1_rad) * math.cos(dec2_rad) * math.cos(ra1_rad - ra2_rad)
    )

    # Avoid domain errors
    cos_separation = max(-1.0, min(1.0, cos_separation))

    separation_rad = math.acos(cos_separation)
    separation_deg = math.degrees(separation_rad)

    return separation_deg


def match_star_patterns(detected_angles: List[Dict], theoretical_patterns: List[Dict],
                        detected_stars: List[Dict]) -> Dict:
    """
    Match detected angular patterns to theoretical star patterns

    Finds best match by comparing angular distances with tolerance for
    measurement errors and lens distortion

    Args:
        detected_angles: Angular relationships from detection
        theoretical_patterns: Expected patterns from star catalog
        detected_stars: Raw star detection data

    Returns:
        Best matching star identification with confidence score
    """

    if not detected_angles or not theoretical_patterns:
        return {'method': 'no_data', 'identified_stars': [], 'confidence': 0.0}

    best_matches = []
    tolerance = 3.0  # degrees - tolerance for measurement errors

    # Try to match each detected angle to theoretical patterns
    for detected in detected_angles:
        detected_angle = detected['angular_distance']

        for theoretical in theoretical_patterns:
            theoretical_angle = theoretical['angular_distance']

            # Check if angles match within tolerance
            angle_diff = abs(detected_angle - theoretical_angle)

            if angle_diff <= tolerance:
                # Calculate confidence based on how close the match is
                confidence = 1.0 - (angle_diff / tolerance)

                # Bonus confidence for brightness ratio match
                if 'brightness_ratio' in detected and 'brightness_ratio' in theoretical:
                    brightness_match = 1.0 - abs(
                        math.log(detected['brightness_ratio']) -
                        math.log(theoretical['brightness_ratio'])
                    ) / 2.0
                    brightness_match = max(0.0, min(1.0, brightness_match))
                    confidence = (confidence + brightness_match) / 2.0

                best_matches.append({
                    'detected': detected,
                    'theoretical': theoretical,
                    'confidence': confidence,
                    'angle_error': angle_diff
                })

    if not best_matches:
        return {'method': 'no_pattern_match', 'identified_stars': [], 'confidence': 0.0}

    # Sort by confidence and take best matches
    best_matches.sort(key=lambda m: m['confidence'], reverse=True)

    # Extract identified stars from best matches
    identified_stars = []
    overall_confidence = 0.0

    # Take top matches and extract unique star identifications
    seen_stars = set()
    for match in best_matches[:5]:  # Top 5 matches
        star1_name = match['theoretical']['star1']
        star2_name = match['theoretical']['star2']

        if star1_name not in seen_stars:
            identified_stars.append({
                'name': star1_name,
                'data': match['theoretical']['star1_data'],
                'detected_index': match['detected']['star1_index'],
                'confidence': match['confidence']
            })
            seen_stars.add(star1_name)

        if star2_name not in seen_stars:
            identified_stars.append({
                'name': star2_name,
                'data': match['theoretical']['star2_data'],
                'detected_index': match['detected']['star2_index'],
                'confidence': match['confidence']
            })
            seen_stars.add(star2_name)

        overall_confidence += match['confidence']

    # Average confidence across matches
    if best_matches:
        overall_confidence = overall_confidence / len(best_matches[:5])

    return {
        'method': 'pattern_matching',
        'identified_stars': identified_stars,
        'confidence': round(overall_confidence, 3),
        'total_matches': len(best_matches),
        'best_angle_error': round(best_matches[0]['angle_error'], 2) if best_matches else 0
    }


def get_essential_navigation_stars() -> List[Dict]:
    """
    Fallback list of essential navigation stars if catalog file not available

    Returns:
        List of most important bright stars for navigation
    """

    return [
        {'name': 'Sirius', 'ra_hours': 6.752, 'ra_degrees': 101.287, 'dec_degrees': -16.716, 'magnitude': -1.44},
        {'name': 'Canopus', 'ra_hours': 6.4, 'ra_degrees': 95.988, 'dec_degrees': -52.696, 'magnitude': -0.62},
        {'name': 'Arcturus', 'ra_hours': 14.261, 'ra_degrees': 213.915, 'dec_degrees': 19.182, 'magnitude': -0.05},
        {'name': 'Vega', 'ra_hours': 18.615, 'ra_degrees': 279.234, 'dec_degrees': 38.784, 'magnitude': 0.03},
        {'name': 'Capella', 'ra_hours': 5.278, 'ra_degrees': 79.172, 'dec_degrees': 45.998, 'magnitude': 0.08},
        {'name': 'Rigel', 'ra_hours': 5.242, 'ra_degrees': 78.634, 'dec_degrees': -8.202, 'magnitude': 0.18},
        {'name': 'Procyon', 'ra_hours': 7.655, 'ra_degrees': 114.825, 'dec_degrees': 5.225, 'magnitude': 0.34},
        {'name': 'Betelgeuse', 'ra_hours': 5.919, 'ra_degrees': 88.793, 'dec_degrees': 7.407, 'magnitude': 0.45}
    ]