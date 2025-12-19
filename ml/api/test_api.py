"""
Test script for Coconut Health Monitor API
Run with: python test_api.py

Make sure the API is running first: python run_api.py
"""

import requests
import os
import sys
from pathlib import Path

# API Base URL (use 5001 if backend is on 5000)
API_URL = "http://localhost:5001"

def test_home():
    """Test home endpoint"""
    print("\n1. Testing / endpoint...")
    response = requests.get(f"{API_URL}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def test_health():
    """Test health endpoint"""
    print("\n2. Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200

def test_model_info():
    """Test model-info endpoint"""
    print("\n3. Testing /model-info endpoint...")
    response = requests.get(f"{API_URL}/model-info")
    print(f"   Status: {response.status_code}")
    data = response.json()
    print(f"   Model: {data.get('model_name')}")
    print(f"   Accuracy: {data.get('test_accuracy', 0):.2%}")
    print(f"   Classes: {data.get('classes')}")
    return response.status_code == 200

def test_predict(image_path):
    """Test predict endpoint with an image"""
    print(f"\n4. Testing /predict endpoint with: {image_path}")

    if not os.path.exists(image_path):
        print(f"   ERROR: Image not found at {image_path}")
        return False

    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(f"{API_URL}/predict", files=files)

    print(f"   Status: {response.status_code}")
    data = response.json()

    if data.get('success'):
        pred = data['prediction']
        print(f"   Prediction: {pred['label']}")
        print(f"   Confidence: {pred['confidence']:.2%}")
        print(f"   Infected: {pred['is_infected']}")
        print(f"   Probabilities: {data['probabilities']}")
    else:
        print(f"   Error: {data.get('error')}")

    return response.status_code == 200

def main():
    print("=" * 60)
    print("  COCONUT HEALTH MONITOR API - TEST SUITE")
    print("=" * 60)

    # Test basic endpoints
    results = []
    results.append(("Home", test_home()))
    results.append(("Health", test_health()))
    results.append(("Model Info", test_model_info()))

    # Test prediction with a sample image
    # Look for test images in the data folder
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'pest'

    mite_path = data_path / 'coconut_mite'
    healthy_path = data_path / 'healthy'

    # Find a test image
    test_images = []

    if mite_path.exists():
        mite_images = list(mite_path.glob('*.jpg')) + list(mite_path.glob('*.JPG'))
        if mite_images:
            test_images.append(('Mite Image', str(mite_images[0])))

    if healthy_path.exists():
        healthy_images = list(healthy_path.glob('*.jpg')) + list(healthy_path.glob('*.JPG'))
        if healthy_images:
            test_images.append(('Healthy Image', str(healthy_images[0])))

    # Test predictions
    for name, img_path in test_images:
        results.append((f"Predict ({name})", test_predict(img_path)))

    # Summary
    print("\n" + "=" * 60)
    print("  TEST RESULTS SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed_count}/{len(results)} tests passed")

if __name__ == '__main__':
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to API server!")
        print("Make sure the API is running: python run_api.py")
        sys.exit(1)
