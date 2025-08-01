"""
Celestial Navigation API Server
Emergency navigation when GPS fails
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import List
import base64
import cv2
import numpy as np

from core.capture import process_multiple_images
from core.detection import detect_stars_and_horizon
from core.identification import identify_stars_from_pattern
from core.calculation import calculate_position
from core.validation import validate_results

app = FastAPI(title="Celestial Navigation API", version="1.0.0")

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process")
async def process_celestial_navigation(
        images: List[UploadFile] = File(...),
        timestamp: str = Form(...),
        latitude_estimate: float = Form(None),  # Optional rough location
        longitude_estimate: float = Form(None),
        compass_bearing: float = Form(None),
        device_tilt_x: float = Form(None),
        device_tilt_y: float = Form(None)
):
    """
    Main endpoint for celestial navigation processing

    Args:
        images: List of image files (5-10 recommended for accuracy)
        timestamp: UTC timestamp in ISO format
        latitude_estimate: Rough latitude for star catalog filtering
        longitude_estimate: Rough longitude for star catalog filtering
        compass_bearing: Compass reading in degrees (0-360)
        device_tilt_x: Device tilt in X axis (degrees)
        device_tilt_y: Device tilt in Y axis (degrees)

    Returns:
        JSON with calculated position, confidence, and method used
    """

    try:
        # Convert uploaded images to OpenCV format
        cv_images = []
        for image_file in images:
            contents = await image_file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv_images.append(img)

        # STEP 1: Process multiple images for noise reduction
        averaged_image = process_multiple_images(cv_images)

        # STEP 2: Detect stars and horizon in averaged image
        detection_results = detect_stars_and_horizon(averaged_image)

        # STEP 3: Identify stars from angular patterns
        identification_results = identify_stars_from_pattern(
            detection_results,
            timestamp,
            latitude_estimate,
            longitude_estimate
        )

        # STEP 4: Calculate position using spherical trigonometry
        position_results = calculate_position(
            identification_results,
            compass_bearing,
            device_tilt_x,
            device_tilt_y,
            detection_results.get('horizon_angle')
        )

        # STEP 5: Validate and assess quality of results
        final_results = validate_results(position_results, detection_results)

        return {
            "success": True,
            "position": {
                "latitude": final_results['latitude'],
                "longitude": final_results['longitude'],
                "accuracy_nautical_miles": final_results['accuracy_estimate']
            },
            "confidence": final_results['confidence'],
            "method_used": final_results['method'],
            "stars_identified": final_results['stars'],
            "recommendations": final_results['recommendations'],
            "timestamp": timestamp
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "recommendations": ["Check image quality", "Ensure clear sky view", "Try again with better conditions"]
        }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Celestial Navigation API running"}


@app.get("/bright_stars")
async def get_bright_stars():
    """Get list of bright stars used for navigation"""
    try:
        with open('data/star_catalog.json', 'r') as f:
            catalog = json.load(f)
        return {"stars": catalog['bright_stars']}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    print("🌟 Starting Celestial Navigation API")
    print("📍 Emergency navigation when GPS fails")
    uvicorn.run(app, host="0.0.0.0", port=8000)