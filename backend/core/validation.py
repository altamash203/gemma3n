"""
Result Validation and Quality Assessment
Validates calculated position and provides confidence assessment
"""

import math
from typing import Dict, List, Optional


def validate_results(position_results: Dict, detection_results: Dict) -> Dict:
    """
    Main validation function - assesses quality and reliability of navigation results

    Validation checks:
    1. Position reasonableness (on Earth, not in impossible locations)
    2. Accuracy estimates within acceptable bounds
    3. Consistency between different measurements
    4. Data quality indicators

    Args:
        position_results: Results from calculation.py
        detection_results: Original detection data for cross-validation

    Returns:
        Dictionary with validated results and recommendations
    """

    print("🔍 Validating navigation results...")

    # Extract key values
    latitude = position_results.get('latitude')
    longitude = position_results.get('longitude')
    accuracy = position_results.get('accuracy_estimate')
    confidence = position_results.get('confidence', 0.0)
    method = position_results.get('method', 'unknown')

    # Perform validation checks
    validation_results = {
        'position_valid': validate_position_coordinates(latitude, longitude),
        'accuracy_acceptable': validate_accuracy_estimate(accuracy),
        'confidence_reasonable': validate_confidence_level(confidence),
        'data_quality': assess_data_quality(detection_results),
        'method_appropriate': validate_method_choice(method, detection_results)
    }

    # Calculate overall reliability
    overall_reliability = calculate_overall_reliability(validation_results, confidence)

    # Generate recommendations
    recommendations = generate_recommendations(validation_results, position_results, detection_results)

    # Compile final results
    final_results = {
        'latitude': latitude,
        'longitude': longitude,
        'accuracy_estimate': accuracy,
        'confidence': confidence,
        'method': method,
        'reliability': overall_reliability,
        'validation': validation_results,
        'recommendations': recommendations,
        'suitable_for_navigation': overall_reliability >= 0.6
    }

    print(f"✅ Validation complete - Reliability: {overall_reliability:.1f}")
    return final_results


def validate_position_coordinates(latitude: Optional[float], longitude: Optional[float]) -> Dict:
    """
    Check if calculated position coordinates are valid and reasonable

    Args:
        latitude: Calculated latitude
        longitude: Calculated longitude

    Returns:
        Dictionary with coordinate validation results
    """

    if latitude is None or longitude is None:
        return {
            'valid': False,
            'reason': 'Missing coordinates',
            'score': 0.0
        }

    # Check coordinate bounds
    if not (-90.0 <= latitude <= 90.0):
        return {
            'valid': False,
            'reason': f'Latitude {latitude:.2f}° out of bounds [-90, 90]',
            'score': 0.0
        }

    if not (-180.0 <= longitude <= 180.0):
        return {
            'valid': False,
            'reason': f'Longitude {longitude:.2f}° out of bounds [-180, 180]',
            'score': 0.0
        }

    # Check for obviously impossible locations
    impossible_locations = []

    # Check if position is in the middle of major continents (unlikely for maritime navigation)
    # This is a simplified check - real implementation could use coastline databases

    # Check if position is at exactly 0,0 (likely calculation error)
    if abs(latitude) < 0.001 and abs(longitude) < 0.001:
        impossible_locations.append("Position at 0°,0° suggests calculation error")

    # Check for extreme polar positions (unlikely for casual navigation)
    if abs(latitude) > 80.0:
        impossible_locations.append(f"Extreme polar latitude {latitude:.1f}° unusual for navigation")

    if impossible_locations:
        return {
            'valid': True,  # Technically valid coordinates
            'warnings': impossible_locations,
            'score': 0.7
        }

    return {
        'valid': True,
        'reason': 'Coordinates within acceptable bounds',
        'score': 1.0
    }


def validate_accuracy_estimate(accuracy: Optional[float]) -> Dict:
    """
    Validate accuracy estimate is reasonable for celestial navigation

    Args:
        accuracy: Estimated accuracy in nautical miles

    Returns:
        Dictionary with accuracy validation results
    """

    if accuracy is None:
        return {
            'acceptable': False,
            'reason': 'No accuracy estimate provided',
            'score': 0.0
        }

    # Typical celestial navigation accuracy ranges
    excellent_accuracy = 0.5  # nautical miles
    good_accuracy = 2.0
    acceptable_accuracy = 5.0
    poor_accuracy = 10.0

    if accuracy <= excellent_accuracy:
        return {
            'acceptable': True,
            'quality': 'excellent',
            'reason': f'Excellent accuracy: ±{accuracy:.1f} nm',
            'score': 1.0
        }
    elif accuracy <= good_accuracy:
        return {
            'acceptable': True,
            'quality': 'good',
            'reason': f'Good accuracy: ±{accuracy:.1f} nm',
            'score': 0.8
        }
    elif accuracy <= acceptable_accuracy:
        return {
            'acceptable': True,
            'quality': 'acceptable',
            'reason': f'Acceptable accuracy: ±{accuracy:.1f} nm',
            'score': 0.6
        }
    elif accuracy <= poor_accuracy:
        return {
            'acceptable': True,
            'quality': 'poor',
            'reason': f'Poor accuracy: ±{accuracy:.1f} nm',
            'score': 0.3
        }
    else:
        return {
            'acceptable': False,
            'quality': 'unacceptable',
            'reason': f'Unacceptable accuracy: ±{accuracy:.1f} nm (>10 nm)',
            'score': 0.1
        }


def validate_confidence_level(confidence: float) -> Dict:
    """
    Check if confidence level is reasonable and consistent

    Args:
        confidence: Confidence score (0.0 to 1.0)

    Returns:
        Dictionary with confidence validation results
    """

    if not (0.0 <= confidence <= 1.0):
        return {
            'reasonable': False,
            'reason': f'Confidence {confidence:.2f} out of range [0.0, 1.0]',
            'score': 0.0
        }

    if confidence >= 0.8:
        return {
            'reasonable': True,
            'level': 'high',
            'reason': f'High confidence: {confidence:.2f}',
            'score': 1.0
        }
    elif confidence >= 0.6:
        return {
            'reasonable': True,
            'level': 'medium',
            'reason': f'Medium confidence: {confidence:.2f}',
            'score': 0.8
        }
    elif confidence >= 0.3:
        return {
            'reasonable': True,
            'level': 'low',
            'reason': f'Low confidence: {confidence:.2f}',
            'score': 0.5
        }
    else:
        return {
            'reasonable': False,
            'level': 'very_low',
            'reason': f'Very low confidence: {confidence:.2f}',
            'score': 0.2
        }


def assess_data_quality(detection_results: Dict) -> Dict:
    """
    Assess quality of input data used for navigation

    Args:
        detection_results: Original detection data

    Returns:
        Dictionary with data quality assessment
    """

    star_count = detection_results.get('star_count', 0)
    horizon_detected = detection_results.get('horizon_detected', False)

    quality_factors = []
    total_score = 0.0
    max_score = 0.0

    # Star count assessment
    if star_count >= 3:
        quality_factors.append("Excellent star count (3+ stars)")
        total_score += 1.0
    elif star_count >= 2:
        quality_factors.append("Good star count (2 stars)")
        total_score += 0.8
    elif star_count >= 1:
        quality_factors.append("Minimal star count (1 star)")
        total_score += 0.4
    else:
        quality_factors.append("Insufficient stars detected")
        total_score += 0.0
    max_score += 1.0

    # Horizon detection assessment
    if horizon_detected:
        quality_factors.append("Horizon detected (improves accuracy)")
        total_score += 0.5
    else:
        quality_factors.append("No horizon detected (using sensors)")
        total_score += 0.2
    max_score += 0.5

    # Calculate overall data quality score
    if max_score > 0:
        quality_score = total_score / max_score
    else:
        quality_score = 0.0

    return {
        'score': quality_score,
        'factors': quality_factors,
        'star_count': star_count,
        'horizon_available': horizon_detected
    }


def validate_method_choice(method: str, detection_results: Dict) -> Dict:
    """
    Validate that the chosen calculation method was appropriate for the data

    Args:
        method: Method used for calculation
        detection_results: Detection data that influenced method choice

    Returns:
        Dictionary with method validation results
    """

    star_count = detection_results.get('star_count', 0)
    horizon_detected = detection_results.get('horizon_detected', False)

    # Define method appropriateness rules
    method_rules = {
        'multi_star_intersection': {
            'min_stars': 2,
            'description': 'Multiple star intersection method',
            'accuracy': 'high'
        },
        'single_star_circle': {
            'min_stars': 1,
            'description': 'Single star circle of position',
            'accuracy': 'low'
        },
        'triangular': {
            'min_stars': 3,
            'description': 'Triangular star pattern method',
            'accuracy': 'very_high'
        },
        'zenith': {
            'min_stars': 1,
            'description': 'Zenith distance method',
            'accuracy': 'medium'
        }
    }

    if method not in method_rules:
        return {
            'appropriate': False,
            'reason': f'Unknown method: {method}',
            'score': 0.0
        }

    rule = method_rules[method]

    # Check if method requirements are met
    if star_count < rule['min_stars']:
        return {
            'appropriate': False,
            'reason': f'{rule["description"]} requires {rule["min_stars"]} stars, only {star_count} available',
            'score': 0.2
        }

    # Method is appropriate
    return {
        'appropriate': True,
        'reason': f'{rule["description"]} appropriate for {star_count} stars',
        'expected_accuracy': rule['accuracy'],
        'score': 1.0
    }


def calculate_overall_reliability(validation_results: Dict, confidence: float) -> float:
    """
    Calculate overall reliability score from all validation checks

    Args:
        validation_results: Results from all validation checks
        confidence: Original confidence score

    Returns:
        Overall reliability score (0.0 to 1.0)
    """

    # Weight different validation aspects
    weights = {
        'position_valid': 0.3,
        'accuracy_acceptable': 0.2,
        'confidence_reasonable': 0.2,
        'data_quality': 0.2,
        'method_appropriate': 0.1
    }

    total_score = 0.0
    total_weight = 0.0

    for aspect, weight in weights.items():
        if aspect in validation_results:
            score = validation_results[aspect].get('score', 0.0)
            total_score += score * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    # Base reliability from validation
    base_reliability = total_score / total_weight

    # Adjust based on original confidence
    # High original confidence should boost reliability
    # Low original confidence should reduce it
    confidence_factor = (confidence + 1.0) / 2.0  # Scale to [0.5, 1.0]

    overall_reliability = base_reliability * confidence_factor

    # Ensure result is in [0.0, 1.0] range
    return max(0.0, min(1.0, overall_reliability))


def generate_recommendations(validation_results: Dict, position_results: Dict, detection_results: Dict) -> List[str]:
    """
    Generate actionable recommendations based on validation results

    Args:
        validation_results: Results from validation checks
        position_results: Position calculation results
        detection_results: Original detection data

    Returns:
        List of recommendation strings
    """

    recommendations = []

    # Position-related recommendations
    if not validation_results.get('position_valid', {}).get('valid', True):
        recommendations.append("⚠️ Position coordinates appear invalid - check calculation")

    # Accuracy-related recommendations
    accuracy_result = validation_results.get('accuracy_acceptable', {})
    if not accuracy_result.get('acceptable', True):
        recommendations.append("📍 Low accuracy detected - take additional readings for better precision")
    elif accuracy_result.get('quality') == 'poor':
        recommendations.append("📍 Consider retaking photos in better conditions for improved accuracy")

    # Confidence-related recommendations
    confidence_result = validation_results.get('confidence_reasonable', {})
    if confidence_result.get('level') == 'low':
        recommendations.append("🔍 Low confidence - verify star identifications and sensor readings")
    elif confidence_result.get('level') == 'very_low':
        recommendations.append("❌ Very low confidence - recommend not using for navigation")

    # Data quality recommendations
    data_quality = validation_results.get('data_quality', {})
    star_count = data_quality.get('star_count', 0)

    if star_count < 2:
        recommendations.append("⭐ Try to capture more stars for better triangulation")

    if not data_quality.get('horizon_available', False):
        recommendations.append("🌅 Look for clear horizon in photos to improve accuracy")

    # Method-specific recommendations
    method_result = validation_results.get('method_appropriate', {})
    if not method_result.get('appropriate', True):
        recommendations.append("⚙️ Consider different observation method based on available data")

    # General recommendations
    overall_reliability = calculate_overall_reliability(validation_results, position_results.get('confidence', 0.0))

    if overall_reliability >= 0.8:
        recommendations.append("✅ Results appear reliable for navigation use")
    elif overall_reliability >= 0.6:
        recommendations.append("⚠️ Results moderately reliable - consider taking backup reading")
    elif overall_reliability >= 0.4:
        recommendations.append("🔄 Results uncertain - recommend taking another reading")
    else:
        recommendations.append("❌ Results unreliable - do not use for critical navigation")

    # If no specific recommendations, add general advice
    if len(recommendations) == 0:
        recommendations.append("📸 Consider taking additional readings to verify position")

    return recommendations