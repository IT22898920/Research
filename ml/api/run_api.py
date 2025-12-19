"""
Quick script to run the Coconut Health Monitor API
Run with: python run_api.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, load_model

if __name__ == '__main__':
    print("=" * 60)
    print("  COCONUT HEALTH MONITOR - PEST DETECTION API")
    print("=" * 60)
    print()

    # Load the trained model
    load_model()

    print()
    print("API Endpoints:")
    print("-" * 40)
    print("  GET  /           - API information")
    print("  GET  /health     - Health check")
    print("  GET  /model-info - Model information")
    print("  POST /predict    - Single image prediction")
    print("  POST /predict/batch - Multiple images prediction")
    print("-" * 40)
    print()
    print("Starting server on http://localhost:5000")
    print("Press Ctrl+C to stop")
    print()

    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
