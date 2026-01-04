"""Test script for Leaf Health API endpoint"""

import requests
import os

# API endpoint
API_URL = "http://127.0.0.1:5001/predict/leaf-health"

# Test image path
test_image_path = "data/raw/leaf_health/dataset/test/healthy/1.jpg"

if not os.path.exists(test_image_path):
    print(f"Error: Test image not found at {test_image_path}")
    exit(1)

print("="*70)
print("TESTING LEAF HEALTH API ENDPOINT")
print("="*70)
print(f"\nAPI URL: {API_URL}")
print(f"Test Image: {test_image_path}")
print("\nSending request...")

# Send request
with open(test_image_path, 'rb') as image_file:
    files = {'image': image_file}
    response = requests.post(API_URL, files=files)

print(f"\nStatus Code: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print("\n" + "="*70)
    print("RESPONSE:")
    print("="*70)

    if result.get('success'):
        print(f"\n✓ Success: True")
        print(f"\nPrediction: {result['prediction'].upper()}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        print(f"\nProbabilities:")
        print(f"  Healthy:   {result['probabilities']['healthy']*100:.2f}%")
        print(f"  Unhealthy: {result['probabilities']['unhealthy']*100:.2f}%")
        print(f"\nIs Healthy: {result['is_healthy']}")
        print(f"\nMessage: {result['message']}")
        print(f"\nRecommendation: {result['recommendation']}")
        print(f"\nModel Info:")
        print(f"  Version: {result['model_info']['version']}")
        print(f"  Accuracy: {result['model_info']['accuracy']}")
        print("="*70)
        print("\n✓ API TEST PASSED!")
    else:
        print(f"\n✗ Error: {result.get('error')}")
else:
    print(f"\n✗ Error: {response.text}")

print("="*70)
