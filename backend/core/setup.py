#!/usr/bin/env python3
"""
Celestial Navigation API Setup Script
By Allah's will - bismillah!

Quick setup and testing script for the celestial navigation system
"""

import os
import sys
import json
import subprocess
from pathlib import Path


def create_directory_structure():
    """Create the required directory structure"""

    directories = [
        'core',
        'ai',
        'models',
        'data',
        'utils',
        'tests'
    ]

    print("📁 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

        # Create __init__.py files for Python packages
        if directory in ['core', 'ai', 'utils']:
            init_file = Path(directory) / '__init__.py'
            init_file.touch()

    print("✅ Directory structure created")


def install_dependencies():
    """Install Python dependencies"""

    print("📦 Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                       check=True)
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        print("💡 Try running: pip install -r requirements.txt")


def create_sample_star_catalog():
    """Create a basic star catalog file"""

    print("⭐ Creating sample star catalog...")

    # Essential navigation stars (simplified)
    star_catalog = {
        "catalog_info": {
            "title": "Essential Navigation Stars",
            "description": "Brightest stars for emergency celestial navigation",
            "epoch": "J2000.0",
            "source": "Simplified for hackathon"
        },
        "bright_stars": [
            {
                "name": "Sirius",
                "constellation": "Canis Major",
                "magnitude": -1.44,
                "ra_hours": 6.752,
                "ra_degrees": 101.287,
                "dec_degrees": -16.716,
                "color": "blue-white"
            },
            {
                "name": "Canopus",
                "constellation": "Carina",
                "magnitude": -0.62,
                "ra_hours": 6.4,
                "ra_degrees": 95.988,
                "dec_degrees": -52.696,
                "color": "white"
            },
            {
                "name": "Arcturus",
                "constellation": "Boötes",
                "magnitude": -0.05,
                "ra_hours": 14.261,
                "ra_degrees": 213.915,
                "dec_degrees": 19.182,
                "color": "orange"
            },
            {
                "name": "Vega",
                "constellation": "Lyra",
                "magnitude": 0.03,
                "ra_hours": 18.615,
                "ra_degrees": 279.234,
                "dec_degrees": 38.784,
                "color": "blue-white"
            },
            {
                "name": "Capella",
                "constellation": "Auriga",
                "magnitude": 0.08,
                "ra_hours": 5.278,
                "ra_degrees": 79.172,
                "dec_degrees": 45.998,
                "color": "yellow"
            },
            {
                "name": "Rigel",
                "constellation": "Orion",
                "magnitude": 0.18,
                "ra_hours": 5.242,
                "ra_degrees": 78.634,
                "dec_degrees": -8.202,
                "color": "blue-white"
            }
        ]
    }

    catalog_path = Path('data/star_catalog.json')
    with open(catalog_path, 'w') as f:
        json.dump(star_catalog, f, indent=2)

    print(f"✅ Star catalog created: {catalog_path}")


def test_basic_functionality():
    """Test basic system functionality"""

    print("🧪 Testing basic functionality...")

    try:
        # Test imports
        from core.capture import process_multiple_images
        from core.detection import detect_stars_and_horizon
        from core.identification import identify_stars_from_pattern
        from core.calculation import calculate_position
        from core.validation import validate_results
        from core.sun_calculator import calculate_sun_position

        print("✅ All core modules imported successfully")

        # Test star catalog loading
        catalog_path = Path('data/star_catalog.json')
        if catalog_path.exists():
            with open(catalog_path, 'r') as f:
                catalog = json.load(f)
            print(f"✅ Star catalog loaded: {len(catalog['bright_stars'])} stars")

        print("🎉 Basic functionality test passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("💡 Check that all files are in correct locations")


def show_next_steps():
    """Display next steps for development"""

    print("\n" + "=" * 50)
    print("🌟 CELESTIAL NAVIGATION API SETUP COMPLETE!")
    print("=" * 50)

    print("\n📋 NEXT STEPS:")
    print("1. Download Ollama and Gemma model:")
    print("   • Install Ollama from: https://ollama.ai/download")
    print("   • Run: ollama pull gemma2:2b-instruct-q4_0")
    print("   • Start server: ollama serve")

    print("\n2. Get better star catalog data:")
    print("   • Hipparcos Catalog: https://heasarc.gsfc.nasa.gov/W3Browse/all/hipparcos.html")
    print("   • Yale Bright Star Catalog: http://tdc-www.harvard.edu/catalogs/bsc5.html")
    print("   • Simplified JSON: https://github.com/astronexus/HYG-Database")

    print("\n3. Test the API:")
    print("   • Run: python main.py")
    print("   • Visit: http://localhost:8000/docs")
    print("   • Test endpoint: http://localhost:8000/health")

    print("\n4. Take test photos:")
    print("   • Clear night sky with 2-3 bright stars")
    print("   • Include horizon if possible")
    print("   • Multiple photos for averaging")

    print("\n5. Connect Flutter app:")
    print("   • POST images to: http://localhost:8000/process")
    print("   • Include timestamp and sensor data")

    print("\n🚀 By Allah's will - ready for hackathon!")
    print("✨ Emergency navigation when GPS fails!")


def main():
    """Main setup function"""

    print("🌟 CELESTIAL NAVIGATION API SETUP")
    print("🎯 Emergency Navigation When GPS Fails")
    print("🚀 By Allah's will - bismillah!\n")

    # Run setup steps
    create_directory_structure()
    create_sample_star_catalog()

    # Check if requirements.txt exists
    if Path('requirements.txt').exists():
        install_dependencies()
    else:
        print("⚠️ requirements.txt not found - skipping dependency installation")

    test_basic_functionality()
    show_next_steps()


if __name__ == "__main__":
    main()