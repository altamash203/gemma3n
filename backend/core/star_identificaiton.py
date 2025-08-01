import json
import math
from requests import get
from datetime import datetime
import csv


def download_bright_star_catalog():
    """
    Download and create a bright star catalog for navigation
    Using Hipparcos data - brightest 300 stars
    """
    # This is a simplified catalog - in practice you'd download from:
    # https://github.com/astronexus/HYG-Database
    # or http://tdc-www.harvard.edu/catalogs/bsc5.html

    bright_stars = [
        # Format: [Name, RA_hours, Dec_degrees, Magnitude]
        ["Sirius", 6.75, -16.72, -1.46],
        ["Canopus", 6.40, -52.70, -0.74],
        ["Arcturus", 14.26, 19.18, -0.05],
        ["Vega", 18.62, 38.78, 0.03],
        ["Capella", 5.28, 45.99, 0.08],
        ["Rigel", 5.24, -8.20, 0.13],
        ["Procyon", 7.65, 5.23, 0.34],
        ["Betelgeuse", 5.92, 7.41, 0.50],
        ["Achernar", 1.63, -57.24, 0.46],
        ["Altair", 19.85, 8.87, 0.77],
        ["Aldebaran", 4.60, 16.51, 0.85],
        ["Antares", 16.49, -26.30, 1.09],
        ["Spica", 13.42, -11.16, 1.04],
        ["Pollux", 7.76, 27.99, 1.14],
        ["Fomalhaut", 22.96, -29.62, 1.16],
        ["Deneb", 20.69, 45.28, 1.25],
        ["Regulus", 10.14, 11.97, 1.35],
        ["Adhara", 6.98, -28.97, 1.50],
        ["Castor", 7.58, 31.89, 1.57],
        ["Gacrux", 12.52, -57.11, 1.63],
        # Add more stars as needed...
        ["Polaris", 2.53, 89.26, 1.98],  # North Star
        ["Bellatrix", 5.42, 6.35, 1.64],
        ["Elnath", 5.44, 28.61, 1.68],
        ["Miaplacidus", 9.22, -69.72, 1.68],
        ["Alnilam", 5.60, -1.20, 1.70],
    ]

    # Convert to proper format
    catalog = []
    for star_data in bright_stars:
        name, ra_hours, dec_deg, magnitude = star_data
        catalog.append({
            "name": name,
            "ra": ra_hours * 15.0,  # Convert hours to degrees
            "dec": dec_deg,
            "magnitude": magnitude,
            "id": len(catalog) + 1
        })

    # Save to JSON file
    with open('bright_star_catalog.json', 'w') as f:
        json.dump(catalog, f, indent=2)

    print(f"✅ Created catalog with {len(catalog)} bright stars")
    return catalog


def load_star_catalog():
    """Load star catalog from file"""
    try:
        with open('bright_star_catalog.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("📥 Downloading star catalog...")
        return download_bright_star_catalog()


def calculate_star_position(ra, dec, timestamp, observer_lat=40.0, observer_lon=-74.0):
    """
    Calculate where a star should appear in the sky at given time/location
    Returns altitude and azimuth
    """
    # Simplified calculation - in practice use pyephem or skyfield
    # This is a basic approximation for demonstration

    # Convert timestamp to Julian Day (simplified)
    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    julian_day = 2440588 + (dt.timestamp() / 86400.0)

    # Calculate Local Sidereal Time (LST)
    # Simplified formula
    lst = (280.16 + 360.9856235 * (julian_day - 2451545.0)) % 360
    lst += observer_lon
    lst = lst % 360

    # Convert RA/Dec to Alt/Az (simplified spherical astronomy)
    hour_angle = (lst - ra) % 360
    if hour_angle > 180:
        hour_angle -= 360

    # Convert to radians
    ha_rad = math.radians(hour_angle)
    dec_rad = math.radians(dec)
    lat_rad = math.radians(observer_lat)

    # Calculate altitude
    sin_alt = (math.sin(dec_rad) * math.sin(lat_rad) +
               math.cos(dec_rad) * math.cos(lat_rad) * math.cos(ha_rad))
    altitude = math.degrees(math.asin(max(-1, min(1, sin_alt))))

    # Calculate azimuth
    cos_az = ((math.sin(dec_rad) - math.sin(lat_rad) * math.sin(math.radians(altitude))) /
              (math.cos(lat_rad) * math.cos(math.radians(altitude))))
    azimuth = math.degrees(math.acos(max(-1, min(1, cos_az))))

    if math.sin(ha_rad) > 0:
        azimuth = 360 - azimuth

    return altitude, azimuth


def get_visible_stars(timestamp, observer_lat=40.0, observer_lon=-74.0, min_altitude=10):
    """
    Get list of stars visible above horizon at given time/location
    """
    catalog = load_star_catalog()
    visible_stars = []

    for star in catalog:
        altitude, azimuth = calculate_star_position(
            star['ra'], star['dec'], timestamp, observer_lat, observer_lon
        )

        # Only include stars above horizon
        if altitude > min_altitude:
            star_copy = star.copy()
            star_copy['current_altitude'] = round(altitude, 2)
            star_copy['current_azimuth'] = round(azimuth, 2)
            visible_stars.append(star_copy)

    # Sort by magnitude (brightest first)
    visible_stars.sort(key=lambda x: x['magnitude'])
    return visible_stars


def calculate_angular_distance(alt1, az1, alt2, az2):
    """
    Calculate angular distance between two celestial points
    """
    # Convert to radians
    alt1_r, az1_r = math.radians(alt1), math.radians(az1)
    alt2_r, az2_r = math.radians(alt2), math.radians(az2)

    # Spherical law of cosines
    cos_dist = (math.sin(alt1_r) * math.sin(alt2_r) +
                math.cos(alt1_r) * math.cos(alt2_r) *
                math.cos(az2_r - az1_r))

    return math.degrees(math.acos(max(-1, min(1, cos_dist))))


def find_star_triangle_matches(detected_angles, timestamp, observer_lat=40.0, observer_lon=-74.0):
    """
    Find 3-star combinations that match detected triangle pattern
    """
    visible_stars = get_visible_stars(timestamp, observer_lat, observer_lon)

    # Get brightest visible stars (limit search space)
    candidates = visible_stars[:15]  # Top 15 brightest

    matches = []
    tolerance = 3.0  # ±3 degrees tolerance

    # Check all combinations of 3 stars
    from itertools import combinations
    for star_combo in combinations(candidates, 3):
        star1, star2, star3 = star_combo

        # Calculate theoretical angles between these stars
        angle_12 = calculate_angular_distance(
            star1['current_altitude'], star1['current_azimuth'],
            star2['current_altitude'], star2['current_azimuth']
        )
        angle_23 = calculate_angular_distance(
            star2['current_altitude'], star2['current_azimuth'],
            star3['current_altitude'], star3['current_azimuth']
        )
        angle_13 = calculate_angular_distance(
            star1['current_altitude'], star1['current_azimuth'],
            star3['current_altitude'], star3['current_azimuth']
        )

        theoretical_pattern = sorted([angle_12, angle_23, angle_13])
        detected_pattern = sorted(detected_angles)

        # Check if patterns match within tolerance
        differences = [abs(t - d) for t, d in zip(theoretical_pattern, detected_pattern)]

        if all(diff <= tolerance for diff in differences):
            avg_error = sum(differences) / len(differences)
            matches.append({
                'stars': [star1['name'], star2['name'], star3['name']],
                'star_data': star_combo,
                'theoretical_angles': theoretical_pattern,
                'error': round(avg_error, 2),
                'confidence': round(1.0 - (avg_error / tolerance), 2)
            })

    # Sort by confidence (lowest error first)
    matches.sort(key=lambda x: x['error'])
    return matches


def identify_stars_from_triangle(detected_angles, timestamp, observer_lat=40.0, observer_lon=-74.0):
    """
    Main function: Identify stars from detected triangle pattern
    """
    print(f"🔍 Identifying stars from pattern: {detected_angles}")
    print(f"📅 Time: {timestamp}")
    print(f"📍 Location: {observer_lat}°N, {observer_lon}°W")

    # Find matching star combinations
    matches = find_star_triangle_matches(detected_angles, timestamp, observer_lat, observer_lon)

    if not matches:
        return {
            'success': False,
            'message': 'No star patterns match detected angles',
            'visible_stars_count': len(get_visible_stars(timestamp, observer_lat, observer_lon))
        }

    # Return best match
    best_match = matches[0]

    return {
        'success': True,
        'identified_stars': best_match['stars'],
        'confidence': best_match['confidence'],
        'error_degrees': best_match['error'],
        'star_positions': best_match['star_data'],
        'alternative_matches': matches[1:3]  # Top 2 alternatives
    }


# Example usage
if __name__ == "__main__":
    # Test with sample data
    detected_triangle = [25.4, 31.8, 18.9]  # From your image processing
    current_time = "2024-07-30T21:30:00Z"
    latitude = 40.7589  # New York
    longitude = -73.9851

    result = identify_stars_from_triangle(detected_triangle, current_time, latitude, longitude)

    if result['success']:
        print("\n✅ Star identification successful!")
        print(f"🌟 Identified stars: {', '.join(result['identified_stars'])}")
        print(f"🎯 Confidence: {result['confidence']}")
        print(f"📏 Error: ±{result['error_degrees']}°")

        print("\n📊 Star positions:")
        for star in result['star_positions']:
            print(f"  {star['name']}: Alt={star['current_altitude']}°, Az={star['current_azimuth']}°")

    else:
        print(f"❌ {result['message']}")