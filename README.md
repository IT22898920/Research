# Coconut Health Monitor

AI-Powered Drone-Based System for Comprehensive Coconut Tree Health Monitoring and Yield Prediction.

## Project Overview

This mobile application is part of a research project that uses drone technology and artificial intelligence to monitor coconut tree health and predict yields. The app serves as the user interface for farmers and agricultural professionals to access monitoring data and insights.

## Tech Stack

### Mobile App
- **Frontend:** React Native 0.82.1
- **Navigation:** React Navigation 7.x
- **Backend:** Node.js + Express (planned)
- **Database:** MongoDB (planned)
- **Language:** JavaScript/TypeScript

### Machine Learning
- **Framework:** TensorFlow 2.20.0
- **Model:** EfficientNetB0 (Transfer Learning)
- **API:** Flask 3.1.2
- **Language:** Python 3.13

## Features

### Current Features
- User Authentication UI (Login/Signup screens)
- Navigation between screens
- Responsive design for Android devices
- Google OAuth authentication
- **Coconut Mite Detection Model** (95.62% accuracy)
- Flask REST API for pest detection
- TFLite model for mobile deployment

### Planned Features
- MongoDB integration for user authentication
- Dashboard with coconut tree health data
- White Fly detection model
- Coconut Caterpillar detection model
- Drone image analysis results
- Yield prediction visualization
- Farm management tools
- Push notifications for health alerts

## Project Structure

```
CoconutHealthMonitor/
├── src/
│   ├── screens/
│   │   ├── LoginScreen.js      # User login interface
│   │   └── SignupScreen.js     # User registration interface
│   ├── services/
│   │   └── pestDetectionApi.js # API client for pest detection
│   ├── components/             # Reusable UI components
│   └── navigation/             # Navigation configuration
├── Research/
│   └── ml/
│       ├── notebooks/          # Jupyter notebooks for training
│       │   ├── 01_coconut_mite_exploration.ipynb
│       │   ├── 02_coconut_mite_training_local.ipynb
│       │   └── 02_coconut_mite_training_colab.ipynb
│       ├── models/
│       │   └── coconut_mite/   # Trained model files
│       │       ├── coconut_mite_model.keras
│       │       ├── coconut_mite_model.h5
│       │       ├── coconut_mite_model.tflite
│       │       └── model_info.json
│       ├── api/
│       │   ├── app.py          # Flask API server
│       │   ├── run_api.py      # API startup script
│       │   └── test_api.py     # API tests
│       └── data/raw/pest/      # Training images (not in git)
├── android/                    # Android native code
├── ios/                        # iOS native code (not configured)
├── App.tsx                     # Main application entry point
├── package.json                # Project dependencies
└── README.md                   # This file
```

## Prerequisites

- Node.js >= 20.19.4
- JDK 17
- Android Studio with:
  - Android SDK
  - Android SDK Platform (API 36)
  - Android Virtual Device (Emulator)
  - NDK 27.1.12297006
  - CMake 3.22.1
- Environment Variables:
  - `ANDROID_HOME` = `C:\Users\<username>\AppData\Local\Android\Sdk`
  - Add to PATH: `%ANDROID_HOME%\platform-tools`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CoconutHealthMonitor
```

2. Install dependencies:
```bash
npm install
```

3. Create autolinking configuration (if needed):
```bash
mkdir -p android/build/generated/autolinking
```

## Running the App

1. Start Android Emulator from Android Studio Device Manager

2. Start Metro bundler:
```bash
node node_modules\@react-native-community\cli\build\bin.js start
```

3. In a new terminal, build and run:
```bash
node node_modules\@react-native-community\cli\build\bin.js run-android
```

## Machine Learning

### Trained Models

| Model | Accuracy | Status |
|-------|----------|--------|
| Coconut Mite Detection | 95.62% | Trained |
| White Fly Detection | - | Pending |
| Coconut Caterpillar Detection | - | Pending |

### Running the ML API

```powershell
# Navigate to API folder
cd Research/ml/api

# Start the Flask API server
python run_api.py
```

API will be available at `http://localhost:5000`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/model-info` | Model details |
| POST | `/predict` | Predict pest from image |

### Training New Models

```powershell
# Navigate to ML folder
cd Research/ml

# Start Jupyter Lab
python -m jupyterlab

# Open the training notebook and run all cells
```

## Development Notes

### Known Issues
- Standard `npx react-native` commands may not work on Windows; use the full node path instead
- Autolinking requires manual configuration file (see CLAUDE.md for details)
- First build takes 5-10 minutes due to native compilation

### Build Commands (Windows PowerShell)
```powershell
# Start Metro
node node_modules\@react-native-community\cli\build\bin.js start

# Run on Android
node node_modules\@react-native-community\cli\build\bin.js run-android

# Clean build
cd android
.\gradlew clean
cd ..
```

## Team

SLIIT Research Project Team

## License

This project is part of academic research at SLIIT.
