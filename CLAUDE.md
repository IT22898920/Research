# CLAUDE.md - Project Context for AI Assistants

This file provides context for AI assistants (like Claude) working on this codebase.

## Project Summary

**Coconut Health Monitor** - A React Native mobile application for an AI-powered drone-based coconut tree health monitoring and yield prediction system. This is a SLIIT research project.

## Tech Stack

### Mobile App
- React Native 0.82.1
- React Navigation 7.x (native-stack)
- react-native-screens 4.18.0
- react-native-safe-area-context 5.6.2
- @react-native-firebase/app & auth
- @react-native-google-signin/google-signin
- Target: Android (API 36)

### Machine Learning
- Python 3.13
- TensorFlow 2.20.0
- EfficientNetB0 (Transfer Learning) - Coconut Mite model
- MobileNetV2 (Transfer Learning) - Coconut Caterpillar model
- Flask 3.1.2 (API Server)
- Pillow, NumPy, Matplotlib, Seaborn, Scikit-learn

## Package & Firebase Info

- **Package Name:** `com.coconuthealthmonitorNew`
- **Firebase Project:** `coconut-app-cd914`
- **Google Cloud Project:** `coconut-app-cd914`
- **SHA-1 Fingerprint:** `5E:8F:16:06:2E:A3:CD:2C:4A:0D:54:78:76:BA:A6:F3:8C:AB:F6:25`

## Project Location

- **Development Path:** `D:\Projects\CoconutHealthMonitor`
- **Original Path:** `D:\SLIIT\Reaserch Project\R2\CoconutHealthMonitor` (moved due to spaces in path causing build issues)

## Key Files

### App Entry Point
- `App.tsx` - Main application with navigation setup

### Screens
- `src/screens/LoginScreen.js` - User login UI
- `src/screens/SignupScreen.js` - User registration UI

### Android Configuration
- `android/settings.gradle` - Module configuration (autolinking disabled, manual linking used)
- `android/app/build.gradle` - App build configuration
- `android/build/generated/autolinking/autolinking.json` - Manual autolinking config (required!)

### ML & API Files
- `Research/ml/notebooks/` - Jupyter notebooks for model training
  - `02_coconut_caterpillar_training.ipynb` - Caterpillar model training
  - `03_coconut_mite_training.ipynb` - Mite model training (v4-v5)
  - `04_coconut_mite_proper_training.ipynb` - Mite model proper training
  - `05_mite_model_results.ipynb` - Mite v6 results analysis
  - `06_mite_v7_results.ipynb` - Mite v7 (anti-overfit) results analysis
- `Research/ml/models/coconut_mite_v7/` - Latest Mite detection model (82.54% accuracy, anti-overfit)
- `Research/ml/models/coconut_caterpillar/` - Caterpillar detection model (98.91% accuracy)
- `Research/ml/api/app.py` - Flask API for model serving
- `src/services/pestDetectionApi.js` - React Native API client

## Build Commands (Windows PowerShell)

```powershell
# Navigate to project
cd D:\Projects\CoconutHealthMonitor

# Start Metro bundler (Terminal 1)
node node_modules\@react-native-community\cli\build\bin.js start

# Build and run on Android (Terminal 2)
node node_modules\@react-native-community\cli\build\bin.js run-android

# Clean build
cd android && .\gradlew clean && cd ..

# Check connected devices
adb devices
```

## ML Commands (Windows PowerShell)

```powershell
# Navigate to ML folder
cd "D:\SLIIT\Reaserch Project\CoconutHealthMonitor\Research\ml"

# Run Jupyter Lab
python -m jupyterlab

# Run Flask API
cd api
python run_api.py

# Test API
python test_api.py
```

## Trained Models

### Coconut Mite Detection Model (v7 - Latest)
- **Model:** EfficientNetB0 (Transfer Learning) with Anti-Overfit measures
- **Version:** v7_anti_overfit
- **Test Accuracy:** 82.54%
- **Test F1 Score:** 82.47%
- **Train-Val Gap:** 6.2% (reduced overfitting)
- **Optimal Threshold:** 0.60
- **Input Size:** 224x224x3
- **Classes:** coconut_mite, healthy
- **Anti-Overfit Changes:**
  - Dropout: 0.6, L2 Regularization: 0.02
  - Label Smoothing: 0.1
  - Dense layer: 32 units
  - Stronger augmentation, Earlier early stopping
- **Per-Class Metrics:**
  - coconut_mite: P=0.83, R=0.84, F1=0.84
  - healthy: P=0.82, R=0.81, F1=0.81
- **Files:**
  - `models/coconut_mite_v7/best_model.keras`
  - `models/coconut_mite_v7/model_info.json`
- **API Endpoint:** `/predict/mite`

### Model Version History (Coconut Mite)
| Version | Accuracy | F1 Score | Train-Val Gap | Notes |
|---------|----------|----------|---------------|-------|
| v4 | ~90% | - | High | Initial training |
| v5 | ~92% | - | High | Improved augmentation |
| v6 | ~85% | - | Medium | Reduced overfitting |
| v7 | 82.54% | 82.47% | 6.2% | Best generalization |

### Coconut Caterpillar Detection Model
- **Model:** MobileNetV2 (Transfer Learning)
- **Accuracy:** 98.91%
- **Input Size:** 224x224x3
- **Classes:** caterpillar, healthy
- **Optimal Threshold:** 0.20 (for balanced P/R/F1)
- **Dataset:** 9,108 images (8,925 train + 91 val + 92 test)
- **Training Time:** 64.4 minutes (24 epochs, early stopped at epoch 14)
- **Per-Class Metrics:**
  - caterpillar: P=0.98, R=1.00, F1=0.99
  - healthy: P=1.00, R=0.98, F1=0.99
- **Files:**
  - `models/coconut_caterpillar/caterpillar_model.keras`
  - `models/coconut_caterpillar/model_info.json`
  - `models/coconut_caterpillar/TRAINING_SUMMARY.txt`
- **API Endpoint:** `/predict/caterpillar`
- **Uthpala Miss Requirements:** ALL PASSED

### Pending Models
- White Fly Detection (not trained yet)

## Important Notes

### Why standard commands don't work
Standard `npx react-native run-android` fails on Windows with "react-native is not recognized" error. Use the full node path instead:
```powershell
node node_modules\@react-native-community\cli\build\bin.js run-android
```

### Autolinking Configuration
The project uses manual autolinking because `autolinkLibrariesFromCommand()` fails on Windows. The autolinking.json file must exist at:
```
android/build/generated/autolinking/autolinking.json
```

If build fails with autolinking errors, recreate this file with the proper JSON structure containing dependencies and project info.

### Required Environment Variables
```powershell
$env:ANDROID_HOME = "C:\Users\DELL\AppData\Local\Android\Sdk"
$env:Path += ";$env:ANDROID_HOME\platform-tools"
```

### settings.gradle Configuration
The project uses manual module linking instead of autolinking:
```gradle
include ':react-native-screens'
project(':react-native-screens').projectDir = new File(rootProject.projectDir, '../node_modules/react-native-screens/android')

include ':react-native-safe-area-context'
project(':react-native-safe-area-context').projectDir = new File(rootProject.projectDir, '../node_modules/react-native-safe-area-context/android')
```

## Current State

### Completed
- Project setup with React Native 0.82.1
- Login screen UI
- Signup screen UI
- Navigation between screens
- Android build configuration
- Firebase & Google Sign-In integration
- Google OAuth authentication working
- ML folder structure setup
- Coconut Mite detection model v7 trained (82.54% accuracy, anti-overfit optimized)
- Coconut Caterpillar detection model trained (98.91% accuracy)
- Flask API for model serving (supports both mite and caterpillar)
- React Native API client service
- Multiple mite model iterations (v4-v7) to reduce overfitting

### Next Steps (Planned)
1. Dashboard screen after login
2. MongoDB backend integration
3. Train White Fly detection model
4. Integrate pest detection with mobile app
5. Drone data visualization
6. Health monitoring features
7. Yield prediction display

## Troubleshooting

### Build Fails with Autolinking Error
1. Create directory: `mkdir -p android/build/generated/autolinking`
2. Create autolinking.json with proper structure
3. Run build again

### Emulator Not Detected
```powershell
adb devices  # Should show connected emulator
```

### Metro Permission Error
Run PowerShell as Administrator or use:
```powershell
node node_modules\@react-native-community\cli\build\bin.js start --reset-cache
```

### NDK Issues
If NDK installation is corrupted:
```powershell
Remove-Item -Recurse -Force "C:\Users\DELL\AppData\Local\Android\Sdk\ndk\27.1.12297006"
# Then rebuild - NDK will be reinstalled automatically
```

### Google Sign-In DEVELOPER_ERROR
If Google Sign-In fails with DEVELOPER_ERROR:
1. Verify SHA-1 fingerprint: `cd android && .\gradlew signingReport`
2. Check Google Cloud Console has Android OAuth client with correct package name & SHA-1
3. Ensure `google-services.json` has matching `client_type: 1` entry
4. Web Client ID in `src/config/googleAuth.js` must match Google Cloud Console

### Port 8081 Already in Use
```powershell
netstat -ano | findstr :8081
taskkill /PID <PID_NUMBER> /F
```
