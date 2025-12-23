"""
Quick script to run the Coconut Health Monitor API
Run with: python run_api.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, load_models

if __name__ == '__main__':
    print("=" * 60)
    print("  COCONUT HEALTH MONITOR - PEST DETECTION API v2.0")
    print("=" * 60)
    print()

    # Load all trained models
    load_models()

    print()
    print("API Endpoints:")
    print("-" * 40)
    print("  GET  /                  - API information")
    print("  GET  /health            - Health check")
    print("  GET  /models            - List all models")
    print("  POST /predict/mite      - Mite detection")
    print("  POST /predict/caterpillar - Caterpillar detection")
    print("  POST /predict/all       - All pest detection")
    print("  POST /predict           - Legacy (mite)")
    print("-" * 40)
    print()
    print("Starting ML server on http://localhost:5001")
    print("(Auth backend runs on port 5000)")
    print("Press Ctrl+C to stop")
    print()

    # Run the Flask app on port 5001 (port 5000 is for Node.js auth backend)
    app.run(host='0.0.0.0', port=5001, debug=True)
