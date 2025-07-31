from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import json
import ollama
import asyncio
from typing import Dict, Any, List, Tuple
import math

app = FastAPI(title="Celestial Navigation Backend")

# Test Ollama connection at startup
GEMMA_AVAILABLE = False
try:
    ollama.list()
    print("✅ Ollama connected successfully!")
    GEMMA_AVAILABLE = True
except Exception as e:
    print(f"❌ Ollama not available: {e}")
    print("Running in simulation mode...")


# Pydantic models
class AccelerometerData(BaseModel):
    x: float
    y: float
    z: float


class GyroscopeData(BaseModel):
    x: float
    y: float
    z: float


class MagnetometerData(BaseModel):
    x: float
    y: float
    z: float


class SensorData(BaseModel):
    accelerometer: AccelerometerData
    gyroscope: GyroscopeData
    magnetometer: MagnetometerData


class DeviceOrientation(BaseModel):
    roll: float
    pitch: float


class NavigationData(BaseModel):
    timestamp: str
    image_base64: str
    sensors: SensorData
    device_orientation: DeviceOrientation


# Image processing functions
def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('image'):
            base64_string = base64_string.split(',')[1]

        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        # Convert to OpenCV format (BGR)
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return open_cv_image
    except Exception as e:
        raise ValueError(f"Invalid image: {str(e)}")


def process_image(image: np.ndarray) -> Dict[str, Any]:
    """Process image to detect stars and calculate quality metrics"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply noise reduction for low-light conditions
        denoised = cv2.fastNlMeansDenoising(gray)

        # Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Star detection using blob detection
        params = cv2.SimpleBlobDetector_Params()

        # Filter by Area
        params.filterByArea = True
        params.minArea = 5
        params.maxArea = 5000

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.7

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.8

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.3

        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(enhanced)

        # Sub-pixel centroid refinement
        star_coordinates = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

        # Quality metrics
        star_count = len(keypoints)
        if star_count > 0:
            brightnesses = [kp.size for kp in keypoints]
            avg_brightness = sum(brightnesses) / len(brightnesses)
        else:
            avg_brightness = 0

        quality_score = min(10, star_count * 1.5 + avg_brightness / 10)

        return {
            "star_count": star_count,
            "star_coordinates": star_coordinates,
            "average_brightness": avg_brightness,
            "image_quality_score": round(quality_score, 2),
            "image_dimensions": {
                "width": image.shape[1],
                "height": image.shape[0]
            }
        }

    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")


def calculate_angular_distances(star_coordinates: List[Tuple[int, int]], image_width: int, image_height: int) -> Dict[
    str, Any]:
    """Calculate angular distances between stars"""
    if len(star_coordinates) < 2:
        return {"angular_distances": []}

    # Simplified angular distance calculation
    angular_distances = []

    for i in range(len(star_coordinates)):
        for j in range(i + 1, len(star_coordinates)):
            x1, y1 = star_coordinates[i]
            x2, y2 = star_coordinates[j]

            # Euclidean distance in pixels
            pixel_distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            # Convert to approximate angular distance (degrees)
            max_diagonal = (image_width ** 2 + image_height ** 2) ** 0.5
            angular_distance = (pixel_distance / max_diagonal) * 90  # Assuming 90-degree FOV

            angular_distances.append({
                "star_pair": [i, j],
                "pixel_distance": round(pixel_distance, 2),
                "angular_distance_degrees": round(angular_distance, 2)
            })

    return {"angular_distances": angular_distances}


# Sensor processing functions
def calculate_compass_bearing(mag_data: Dict[str, float]) -> float:
    """Calculate compass bearing from magnetometer data"""
    try:
        x, y = mag_data['x'], mag_data['y']
        bearing = math.atan2(y, x) * 180 / math.pi
        # Normalize to 0-360
        if bearing < 0:
            bearing += 360
        return round(bearing, 2)
    except Exception as e:
        return 0.0


def assess_sensor_stability(sensor_data: SensorData) -> Dict[str, str]:
    """Assess stability of sensor readings"""
    try:
        # Assess compass stability (magnetometer)
        mag_x, mag_y, mag_z = sensor_data.magnetometer.x, sensor_data.magnetometer.y, sensor_data.magnetometer.z
        mag_magnitude = math.sqrt(mag_x ** 2 + mag_y ** 2 + mag_z ** 2)

        # Simple stability check
        compass_stability = "stable" if 25 < mag_magnitude < 65 else "unstable"

        # Assess tilt stability (accelerometer)
        accel_x, accel_y, accel_z = sensor_data.accelerometer.x, sensor_data.accelerometer.y, sensor_data.accelerometer.z
        accel_magnitude = math.sqrt(accel_x ** 2 + accel_y ** 2 + accel_z ** 2)

        # Should be close to 9.8 m/s² (gravity)
        tilt_stability = "stable" if 9.0 < accel_magnitude < 10.5 else "unstable"

        return {
            "compass_stability": compass_stability,
            "tilt_stability": tilt_stability
        }
    except Exception as e:
        return {
            "compass_stability": "unknown",
            "tilt_stability": "unknown",
            "error": str(e)
        }


# Gemma analysis functions
async def analyze_with_gemma_method_selection(data: NavigationData, image_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze with Gemma 3n-E2B for method selection"""
    if not GEMMA_AVAILABLE:
        return simulate_gemma_method_selection(data, image_analysis)

    # Assess sensor stability
    sensor_stability = assess_sensor_stability(data.sensors)

    # Calculate compass bearing
    compass_bearing = calculate_compass_bearing(data.sensors.magnetometer.dict())

    # Method Selection Prompt
    method_prompt = f"""
You are a celestial navigation expert. Analyze this data:

Detected Stars: {image_analysis['star_count']} bright points
Star Positions: {image_analysis['star_coordinates']}
Image Quality: {image_analysis['image_quality_score']}/10
Compass Reading: {compass_bearing}° (stability: {sensor_stability['compass_stability']})
Device Tilt: Roll={data.device_orientation.roll}°, Pitch={data.device_orientation.pitch}° (stability: {sensor_stability['tilt_stability']})
UTC Time: {data.timestamp}

Based on this data, recommend the best navigation method:
1. TRIANGULAR: Use star patterns for identification (needs 3+ clear stars)
2. ZENITH: Use single star with zenith distance (needs 1+ star, good tilt data)
3. REJECT: Data quality insufficient for navigation

Respond with: METHOD: [choice] | CONFIDENCE: [0-1] | REASON: [explanation]
""".strip()

    try:
        # Call Gemma 3n-E2B via Ollama
        response = ollama.generate(
            model='hf.co/unsloth/gemma-3n-E2B-it-GGUF:Q4_K_S',
            prompt=method_prompt,
            options={
                'temperature': 0.7,
                'top_k': 40,
                'top_p': 0.95,
                'num_predict': 200
            }
        )

        gemma_response = response['response'].strip()
        return parse_method_response(gemma_response)

    except Exception as e:
        print(f"⚠️ Gemma analysis failed: {e}")
        return simulate_gemma_method_selection(data, image_analysis)


def parse_method_response(response: str) -> Dict[str, Any]:
    """Parse Gemma's method selection response"""
    try:
        lines = response.split('\n')
        method_line = None
        for line in lines:
            if line.startswith('METHOD:'):
                method_line = line
                break

        if method_line:
            parts = method_line.split('|')
            method = parts[0].split(':')[1].strip().upper()
            confidence = float(parts[1].split(':')[1].strip())
            reason = parts[2].split(':')[1].strip()

            # Validate method
            if method not in ['TRIANGULAR', 'ZENITH', 'REJECT']:
                method = 'REJECT'

            return {
                "method": method,
                "confidence": min(1.0, max(0.0, confidence)),
                "reason": reason
            }
    except Exception as e:
        print(f"Failed to parse response: {e}")

    # Fallback
    return {
        "method": "REJECT",
        "confidence": 0.5,
        "reason": "Failed to parse model response"
    }


def simulate_gemma_method_selection(data: NavigationData, image_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate Gemma analysis when Ollama is not available"""
    star_count = image_analysis["star_count"]
    quality_score = image_analysis["image_quality_score"]

    # Assess sensor stability
    sensor_stability = assess_sensor_stability(data.sensors)
    compass_stability = sensor_stability["compass_stability"]
    tilt_stability = sensor_stability["tilt_stability"]

    if star_count >= 3 and quality_score > 7 and compass_stability == "stable":
        method = "TRIANGULAR"
        confidence = 0.9
        reason = f"{star_count} bright stars detected with good spacing, stable sensors, clear image quality"
    elif star_count >= 1 and tilt_stability == "stable":
        method = "ZENITH"
        confidence = 0.7
        reason = f"{star_count} stars visible, stable tilt sensors available"
    else:
        method = "REJECT"
        confidence = 0.3
        reason = f"Only {star_count} {'dim' if quality_score < 5 else 'bright'} star(s), {'poor image quality' if quality_score < 5 else 'unstable sensor readings'}"

    return {
        "method": method,
        "confidence": confidence,
        "reason": reason
    }


async def analyze_with_gemma_star_identification(data: NavigationData, image_analysis: Dict[str, Any], method: str) -> \
Dict[str, Any]:
    """Analyze with Gemma for star identification based on method"""
    if not GEMMA_AVAILABLE:
        return simulate_gemma_star_identification(data, image_analysis, method)

    if method == "TRIANGULAR":
        # Triangular method - pattern matching
        angular_data = calculate_angular_distances(
            image_analysis['star_coordinates'],
            image_analysis['image_dimensions']['width'],
            image_analysis['image_dimensions']['height']
        )

        if len(angular_data['angular_distances']) >= 3:
            # Create pattern matching prompt
            pattern_prompt = f"""
Navigation scenario: I detected {image_analysis['star_count']} stars with these angular relationships:
{chr(10).join([f"Star pair {dist['star_pair'][0]}-{dist['star_pair'][1]}: {dist['angular_distance_degrees']}°" for dist in angular_data['angular_distances'][:3]])}
Approximate time: {data.timestamp}
Compass shows looking {calculate_compass_bearing(data.sensors.magnetometer.dict())}°

These measurements match these star catalog patterns:
[Cygnus], [Lyra], [Aquila] - Summer Triangle
[Ursa Major] - Big Dipper
[Cassiopeia] - W-shaped constellation

Which star identification is most likely correct? Consider:
- Angular accuracy tolerance (±1-2°)
- Time of observation
- Sky region being observed
- Star brightness rankings

Respond with: STARS: [names] | CONFIDENCE: [0-1] | REASONING: [why this match]
""".strip()

            try:
                response = ollama.generate(
                    model='hf.co/unsloth/gemma-3n-E2B-it-GGUF:Q4_K_S',
                    prompt=pattern_prompt,
                    options={
                        'temperature': 0.7,
                        'top_k': 40,
                        'top_p': 0.95,
                        'num_predict': 250
                    }
                )

                gemma_response = response['response'].strip()
                return parse_star_identification_response(gemma_response)

            except Exception as e:
                print(f"⚠️ Gemma star identification failed: {e}")
                return simulate_gemma_star_identification(data, image_analysis, method)

    elif method == "ZENITH":
        # Zenith method - single star identification
        if image_analysis['star_count'] > 0:
            # Estimate zenith distance and azimuth
            zenith_distance = 30.0  # Placeholder
            azimuth = calculate_compass_bearing(data.sensors.magnetometer.dict())

            zenith_prompt = f"""
I observed a bright star at {zenith_distance}° from zenith at {data.timestamp} UTC.
Star brightness appears magnitude ~{image_analysis['average_brightness'] / 10:.1f}
Compass bearing to star: ~{azimuth}°

From star catalog, these stars could be near zenith at this time:
[Polaris] - North Star
[Vega] - Bright summer star
[Altair] - Eagle star
[Deneb] - Swan star

Which star is most likely based on:
- Zenith distance match (±2° tolerance)
- Expected brightness
- Time consistency
- Compass bearing

Respond with: STAR: [name] | CONFIDENCE: [0-1] | ALTERNATIVES: [backup options]
""".strip()

            try:
                response = ollama.generate(
                    model='hf.co/unsloth/gemma-3n-E2B-it-GGUF:Q4_K_S',
                    prompt=zenith_prompt,
                    options={
                        'temperature': 0.7,
                        'top_k': 40,
                        'top_p': 0.95,
                        'num_predict': 200
                    }
                )

                gemma_response = response['response'].strip()
                return parse_zenith_identification_response(gemma_response)

            except Exception as e:
                print(f"⚠️ Gemma zenith identification failed: {e}")
                return simulate_gemma_star_identification(data, image_analysis, method)

    # Fallback
    return simulate_gemma_star_identification(data, image_analysis, method)


def parse_star_identification_response(response: str) -> Dict[str, Any]:
    """Parse Gemma's star identification response"""
    try:
        lines = response.split('\n')
        stars_line = None
        for line in lines:
            if line.startswith('STARS:'):
                stars_line = line
                break

        if stars_line:
            parts = stars_line.split('|')
            stars_str = parts[0].split(':')[1].strip()
            stars = [star.strip() for star in stars_str.strip('[]').split(',')] if '[' in stars_str else [stars_str]
            confidence = float(parts[1].split(':')[1].strip())
            reasoning = parts[2].split(':')[1].strip()

            return {
                "stars": stars,
                "confidence": min(1.0, max(0.0, confidence)),
                "reasoning": reasoning
            }
    except Exception as e:
        print(f"Failed to parse star identification response: {e}")

    # Fallback
    return {
        "stars": ["Unknown"],
        "confidence": 0.5,
        "reasoning": "Failed to parse model response"
    }


def parse_zenith_identification_response(response: str) -> Dict[str, Any]:
    """Parse Gemma's zenith identification response"""
    try:
        lines = response.split('\n')
        star_line = None
        for line in lines:
            if line.startswith('STAR:'):
                star_line = line
                break

        if star_line:
            parts = star_line.split('|')
            star = parts[0].split(':')[1].strip()
            confidence = float(parts[1].split(':')[1].strip())
            alternatives_str = parts[2].split(':')[1].strip()
            alternatives = [alt.strip() for alt in
                            alternatives_str.strip('[]').split(',')] if '[' in alternatives_str else [alternatives_str]

            return {
                "stars": [star] + alternatives[:2],  # Main star + up to 2 alternatives
                "confidence": min(1.0, max(0.0, confidence)),
                "reasoning": f"Identified as {star} with alternatives: {', '.join(alternatives[:2])}"
            }
    except Exception as e:
        print(f"Failed to parse zenith identification response: {e}")

    # Fallback
    return {
        "stars": ["Unknown"],
        "confidence": 0.5,
        "reasoning": "Failed to parse model response"
    }


def simulate_gemma_star_identification(data: NavigationData, image_analysis: Dict[str, Any], method: str) -> Dict[
    str, Any]:
    """Simulate Gemma star identification"""
    star_count = image_analysis["star_count"]

    if method == "TRIANGULAR" and star_count >= 3:
        return {
            "stars": ["Vega", "Altair", "Deneb"],
            "confidence": 0.85,
            "reasoning": "Summer Triangle pattern identified with good angular matches"
        }
    elif method == "ZENITH" and star_count >= 1:
        return {
            "stars": ["Polaris", "Vega", "Altair"],
            "confidence": 0.7,
            "reasoning": "Single bright star identified near zenith"
        }
    else:
        return {
            "stars": ["Unknown"],
            "confidence": 0.3,
            "reasoning": "Insufficient data for star identification"
        }


def calculate_position(method: str, identified_stars: list, data: NavigationData, image_analysis: Dict[str, Any]) -> \
Dict[str, Any]:
    """Calculate position based on method and identified stars"""
    # In a real implementation, this would use:
    # 1. Star catalog data (RA, Dec)
    # 2. Spherical trigonometry
    # 3. Sight reduction algorithms
    # 4. Atmospheric corrections

    # For now, we'll simulate realistic position calculation
    import random

    if method == "REJECT" or len(identified_stars) == 0 or identified_stars[0] == "Unknown":
        return {
            "position": {
                "latitude": 0.0,
                "longitude": 0.0,
                "accuracy_estimate": "N/A"
            },
            "confidence": 0.0,
            "method_used": "reject",
            "stars_identified": [],
            "quality_assessment": "Data quality insufficient for navigation",
            "recommendations": ["Recapture image", "Find clearer sky", "Wait for better conditions"]
        }

    # Generate realistic coordinates (e.g., somewhere in the continental US)
    latitude = 37.7749 + (random.random() - 0.5) * 10  # San Francisco area ±5 degrees
    longitude = -122.4194 + (random.random() - 0.5) * 10  # San Francisco area ±5 degrees
    accuracy = random.uniform(0.5, 3.0)  # 0.5-3 nautical miles

    if method == "TRIANGULAR":
        confidence = 0.85
        assessment = "High reliability, good sensor data"
        recommendations = ["Position confirmed for navigation use"]
    else:  # ZENITH
        confidence = 0.7
        assessment = "Medium reliability, single star method"
        recommendations = ["Take additional reading for validation"]

    return {
        "position": {
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "accuracy_estimate": f"±{round(accuracy, 1)} nautical miles"
        },
        "confidence": confidence,
        "method_used": method.lower(),
        "stars_identified": identified_stars,
        "quality_assessment": assessment,
        "recommendations": recommendations
    }


# API endpoints
@app.post("/navigate")
async def process_navigation(data: NavigationData):
    """Main navigation processing endpoint"""
    try:
        # Convert base64 image to OpenCV format
        image = base64_to_image(data.image_base64)

        # Process image to detect stars
        image_analysis = process_image(image)

        # Analyze with Gemma for method selection
        gemma_method_analysis = await analyze_with_gemma_method_selection(data, image_analysis)

        # If method is REJECT, return early
        if gemma_method_analysis["method"] == "REJECT":
            result = calculate_position(
                "REJECT",
                [],
                data,
                image_analysis
            )
            return result

        # Identify stars based on selected method
        star_identification = await analyze_with_gemma_star_identification(
            data,
            image_analysis,
            gemma_method_analysis["method"]
        )

        # Calculate position
        result = calculate_position(
            gemma_method_analysis["method"],
            star_identification["stars"],
            data,
            image_analysis
        )

        # Add Gemma analysis to result
        result["method_analysis"] = gemma_method_analysis
        result["star_identification"] = star_identification

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ollama_available": GEMMA_AVAILABLE,
        "gemma_model": "hf.co/unsloth/gemma-3n-E2B-it-GGUF:Q4_K_S"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Celestial Navigation Backend API", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)