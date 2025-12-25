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
- EfficientNetB0 (Transfer Learning) - Coconut Mite model v10
- MobileNetV2 (Transfer Learning) - Coconut Caterpillar model v2
- Flask 3.1.2 (API Server v4.0)
- Focal Loss for handling class imbalance
- Pillow, NumPy, Matplotlib, Seaborn, Scikit-learn

## Package & Firebase Info

- **Package Name:** `com.coconuthealthmonitorNew`
- **Firebase Project:** `coconut-app-cd914`
- **Google Cloud Project:** `coconut-app-cd914`
- **SHA-1 Fingerprint:** `5E:8F:16:06:2E:A3:CD:2C:4A:0D:54:78:76:BA:A6:F3:8C:AB:F6:25`

## Project Location

- **Development Path:** `D:\SLIIT\Reaserch Project\CoconutHealthMonitor\Research`

## Key Files

### App Entry Point
- `App.tsx` - Main application with navigation setup

### Screens
- `src/screens/LoginScreen.js` - User login UI
- `src/screens/SignupScreen.js` - User registration UI
- `src/screens/PestDetectionScreen.js` - Pest detection with All Pests, Mite, Caterpillar options

### Services
- `src/services/pestDetectionApi.js` - React Native API client (v4.0)

### Android Configuration
- `android/settings.gradle` - Module configuration (autolinking disabled, manual linking used)
- `android/app/build.gradle` - App build configuration
- `android/build/generated/autolinking/autolinking.json` - Manual autolinking config (required!)

### ML & API Files
- `ml/api/app.py` - Flask API v4.0 for model serving
- `ml/models/coconut_mite_v10/` - Latest Mite detection model (3-class)
- `ml/models/coconut_caterpillar_v2/` - Latest Caterpillar detection model (3-class)
- `ml/notebooks/` - Jupyter notebooks for model training

## Build Commands (Windows PowerShell)

```powershell
# Navigate to project
cd "D:\SLIIT\Reaserch Project\CoconutHealthMonitor\Research"

# Start Metro bundler (Terminal 1)
node node_modules\@react-native-community\cli\build\bin.js start

# Build and run on Android (Terminal 2)
node node_modules\@react-native-community\cli\build\bin.js run-android

# Clear Metro cache and restart
npx react-native start --reset-cache

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
python app.py

# Test API
python test_api.py
```

## Trained Models

### Coconut Mite Detection Model (v10 - Latest)
- **Model:** EfficientNetB0 (Transfer Learning) with Focal Loss
- **Version:** v10 (3-class)
- **Test Accuracy:** 91.44%
- **Classes:** `coconut_mite`, `healthy`, `not_coconut`
- **Input Size:** 224x224x3
- **Features:**
  - Focal Loss (gamma=2.0) for class imbalance
  - Mite boost factor for improved recall
  - Non-coconut image detection
- **Per-Class Metrics:**
  - coconut_mite: 79% recall
  - healthy: High precision
  - not_coconut: Rejects non-coconut images
- **Files:**
  - `models/coconut_mite_v10/best_model.keras`
  - `models/coconut_mite_v10/model_info.json`
- **API Endpoint:** `/predict/mite`

### Coconut Caterpillar Detection Model (v2 - Latest)
- **Model:** MobileNetV2 (Transfer Learning) with Focal Loss
- **Version:** v2 (3-class)
- **Test Accuracy:** 97.47%
- **Macro F1 Score:** 96.30%
- **Classes:** `caterpillar`, `healthy`, `not_coconut`
- **Input Size:** 224x224x3
- **Per-Class Metrics:**
  - caterpillar: 91.49% recall
  - healthy: 100% recall
  - not_coconut: 98.38% recall
- **Files:**
  - `models/coconut_caterpillar_v2/best_model.keras`
  - `models/coconut_caterpillar_v2/model_info.json`
- **API Endpoint:** `/predict/caterpillar`

### Model Version History

#### Coconut Mite
| Version | Accuracy | Classes | Notes |
|---------|----------|---------|-------|
| v4-v7 | 82-92% | 2 | Binary classification |
| v8-v9 | ~88% | 3 | Added not_coconut |
| **v10** | **91.44%** | **3** | **Current - Focal Loss, optimized** |

#### Coconut Caterpillar
| Version | Accuracy | Classes | Notes |
|---------|----------|---------|-------|
| v1 | 98.91% | 2 | Binary classification |
| **v2** | **97.47%** | **3** | **Current - Added not_coconut** |

### Pending Models
- White Fly Detection (not trained yet)

## API Endpoints (v4.0)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check with model status |
| `/models` | GET | List all loaded models |
| `/predict/mite` | POST | Mite detection (3-class) |
| `/predict/caterpillar` | POST | Caterpillar detection (3-class) |
| `/predict/all` | POST | All pests detection with smart combined logic |
| `/predict` | POST | Legacy endpoint (redirects to mite) |

## Smart Combined Logic (v7)

The `/predict/all` endpoint uses intelligent decision logic:

```
Valid Detection = Model predicts (healthy OR pest) WITH >40% confidence

Rejection Rules:
- If ANY model confidently detects a valid coconut class (>40%) → Accept image
- Only reject if NO model finds a valid detection (all say not_coconut with low confidence)
```

**Examples:**
| Scenario | Mite Result | Caterpillar Result | Final Decision |
|----------|-------------|-------------------|----------------|
| Healthy leaf | not_coconut 99% | healthy 98% | ✅ Healthy (trust caterpillar) |
| Mite infection | coconut_mite 55% | not_coconut 65% | ✅ Mite Infected (trust mite) |
| Garden scene | coconut_mite 39% | not_coconut 99% | ❌ Not valid (<40% confidence) |

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

### Required Environment Variables
```powershell
$env:ANDROID_HOME = "C:\Users\DELL\AppData\Local\Android\Sdk"
$env:Path += ";$env:ANDROID_HOME\platform-tools"
```

## Current State

### Completed
- Project setup with React Native 0.82.1
- Login/Signup screens with Firebase Auth
- Google OAuth authentication
- Pest Detection Screen with three options (All Pests, Mite, Caterpillar)
- Coconut Mite detection model v10 (91.44% accuracy, 3-class)
- Coconut Caterpillar detection model v2 (97.47% accuracy, 3-class)
- Flask API v4.0 with smart combined logic
- Non-coconut image rejection
- React Native API client service v4.0
- Mobile app fully integrated with ML API

### Next Steps (Planned)
1. Dashboard screen after login
2. MongoDB backend integration
3. Train White Fly detection model
4. Drone data visualization
5. Health monitoring features
6. Yield prediction display

## Troubleshooting

### Flask Server Multiple Instances
Kill all Python processes before restarting:
```powershell
tasklist | findstr -i python
taskkill /PID <PID> /F
```

### Metro Cache Issues
```powershell
npx react-native start --reset-cache
```

### Build Fails with Autolinking Error
1. Create directory: `mkdir -p android/build/generated/autolinking`
2. Create autolinking.json with proper structure
3. Run build again

### Google Sign-In DEVELOPER_ERROR
1. Verify SHA-1 fingerprint: `cd android && .\gradlew signingReport`
2. Check Google Cloud Console has Android OAuth client with correct package name & SHA-1
3. Ensure `google-services.json` has matching `client_type: 1` entry

### Port 8081 Already in Use
```powershell
netstat -ano | findstr :8081
taskkill /PID <PID_NUMBER> /F
```
