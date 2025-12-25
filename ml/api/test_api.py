"""
Test script for Coconut Health Monitor API v3.0
Run with: python test_api.py

Tests the v10 mite model (3-class) with threshold adjustment
"""

import requests
import os
import sys
from pathlib import Path

# API Base URL
API_URL = "http://localhost:5001"

# Test data path (v4_clean dataset)
BASE_DIR = Path(__file__).parent.parent
TEST_DIR = BASE_DIR / 'data' / 'raw' / 'pest_mite' / 'dataset_v4_clean' / 'test'

def test_home():
    """Test home endpoint"""
    print("\n[1] Testing / endpoint...")
    response = requests.get(f"{API_URL}/")
    print(f"    Status: {response.status_code}")
    data = response.json()
    print(f"    Version: {data.get('version')}")
    print(f"    Mite Model: {data.get('mite_model')}")
    return response.status_code == 200

def test_models():
    """Test models endpoint"""
    print("\n[2] Testing /models endpoint...")
    response = requests.get(f"{API_URL}/models")
    print(f"    Status: {response.status_code}")
    data = response.json()
    if 'mite' in data:
        print(f"    Mite Model: {data['mite']['version']}")
        print(f"    Accuracy: {data['mite']['accuracy']*100:.2f}%")
        print(f"    Mite Recall: {data['mite']['mite_recall']*100:.0f}%")
        print(f"    Threshold: {data['mite']['threshold']} (boost: {data['mite']['boost_factor']}x)")
    return response.status_code == 200

def test_predict_mite(image_path, expected_class):
    """Test mite prediction"""
    if not os.path.exists(image_path):
        print(f"    ERROR: Image not found at {image_path}")
        return False, None

    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        response = requests.post(f"{API_URL}/predict/mite", files=files)

    if response.status_code != 200:
        print(f"    ERROR: Status {response.status_code}")
        return False, None

    data = response.json()
    pred = data['prediction']
    probs = data['probabilities']

    predicted = pred['class']
    correct = predicted == expected_class

    print(f"    Expected: {expected_class}")
    print(f"    Predicted: {predicted} {'OK' if correct else 'WRONG'}")
    print(f"    Probabilities: mite={probs['coconut_mite']*100:.1f}%, healthy={probs['healthy']*100:.1f}%, not_coco={probs['not_coconut']*100:.1f}%")

    return correct, data

def main():
    print("=" * 60)
    print("  COCONUT HEALTH MONITOR API v3.0 - TEST")
    print("=" * 60)

    results = []

    # Test endpoints
    results.append(("Home Endpoint", test_home()))
    results.append(("Models Endpoint", test_models()))

    # Test mite predictions
    print("\n[3] Testing MITE image prediction...")
    mite_img = TEST_DIR / 'coconut_mite' / '0001.png'
    correct, _ = test_predict_mite(str(mite_img), 'coconut_mite')
    results.append(("Mite Image", correct))

    print("\n[4] Testing HEALTHY image prediction...")
    healthy_img = TEST_DIR / 'healthy' / '0001.png'
    correct, _ = test_predict_mite(str(healthy_img), 'healthy')
    results.append(("Healthy Image", correct))

    print("\n[5] Testing NOT_COCONUT image prediction...")
    not_coco_img = TEST_DIR / 'not_coconut' / '0002.jpg'
    correct, _ = test_predict_mite(str(not_coco_img), 'not_coconut')
    results.append(("Not Coconut Image", correct))

    # Summary
    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    passed_count = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed_count}/{len(results)} tests passed")
    print("=" * 60)

if __name__ == '__main__':
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Cannot connect to API!")
        print("Make sure API is running: python app.py")
        sys.exit(1)
