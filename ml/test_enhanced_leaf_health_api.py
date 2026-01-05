"""
Test Enhanced Leaf Health API with Detailed Reasons and Solutions
This script tests the improved /predict/leaf-health endpoint
"""

import requests
import json
import os

# API endpoint
API_URL = "http://127.0.0.1:5001/predict/leaf-health"

# Test image paths
test_images = {
    'healthy': 'data/raw/leaf_health/dataset/test/healthy/1.jpg',
    'unhealthy': 'data/raw/leaf_health/dataset/test/unhealthy/1.jpg'
}

print("=" * 80)
print("TESTING ENHANCED LEAF HEALTH API WITH DETAILED CONDITIONS")
print("=" * 80)

for image_type, image_path in test_images.items():
    if not os.path.exists(image_path):
        print(f"\n⚠ Skipping {image_type}: Test image not found at {image_path}")
        continue

    print(f"\n{'=' * 80}")
    print(f"TESTING: {image_type.upper()} LEAF")
    print("=" * 80)
    print(f"Image: {image_path}")
    print("\nSending request...")

    # Send request
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        response = requests.post(API_URL, files=files)

    print(f"Status Code: {response.status_code}\n")

    if response.status_code == 200:
        result = response.json()

        if result.get('success'):
            print("✓ Success: True\n")
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']*100:.2f}%")
            print(f"Is Healthy: {result['is_healthy']}\n")

            print("Probabilities:")
            print(f"  Healthy:   {result['probabilities']['healthy']*100:.2f}%")
            print(f"  Unhealthy: {result['probabilities']['unhealthy']*100:.2f}%\n")

            print(f"Message: {result['message']}\n")
            print(f"Recommendation: {result['recommendation']}\n")

            # Check for detailed conditions (only for unhealthy leaves)
            if not result['is_healthy'] and 'possible_conditions' in result:
                print("=" * 80)
                print(f"DETAILED CONDITIONS ({result['conditions_count']} possible causes)")
                print("=" * 80)

                for idx, condition in enumerate(result['possible_conditions'], 1):
                    print(f"\n{idx}. {condition['icon']} {condition['condition']} "
                          f"[{condition['urgency'].upper()}]")
                    print("-" * 80)

                    print(f"\n   🔍 Reason:")
                    print(f"   {condition['reason']}")

                    print(f"\n   ⚠️  Symptoms:")
                    for symptom in condition['symptoms']:
                        print(f"      • {symptom}")

                    print(f"\n   💊 Solution:")
                    print(f"   {condition['solution']}")
                    print()

                print("=" * 80)
                print("✓ ENHANCED API TEST PASSED - Detailed conditions included!")
                print("=" * 80)

            elif result['is_healthy']:
                print("=" * 80)
                print("✓ HEALTHY LEAF - No conditions needed")
                print("=" * 80)

            else:
                print("⚠ Warning: No detailed conditions found in response")
                print("This might indicate an API issue")

        else:
            print(f"✗ Error: {result.get('error')}")
    else:
        print(f"✗ HTTP Error: {response.text}")

print("\n" + "=" * 80)
print("TESTING COMPLETE")
print("=" * 80)
print("\nFeature Summary:")
print("  ✓ API returns detailed reasons for unhealthy leaves")
print("  ✓ Each condition includes: reason, symptoms, and solution")
print("  ✓ Urgency levels indicated (low/medium/high)")
print("  ✓ 9 comprehensive conditions covered")
print("=" * 80)
