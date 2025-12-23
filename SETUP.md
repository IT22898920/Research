# Project Setup Guide

This guide helps team members set up the project after cloning from Git.

## Prerequisites

Make sure you have these installed:
- Node.js >= 20
- Python 3.13
- Android Studio (with Android SDK)
- Git

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd CoconutHealthMonitor/Research
```

## Step 2: Get Required Files from Team Lead

These files are NOT in Git (for security). Get them from your team lead:

### Required Files:

| File | Location | Description |
|------|----------|-------------|
| `.env` | Root folder | Environment variables |
| `google-services.json` | `android/app/` | Firebase config |
| `best_model.keras` | `ml/models/coconut_mite_v7/` | Mite detection model |
| `caterpillar_model.keras` | `ml/models/coconut_caterpillar/` | Caterpillar detection model |

### Create `.env` file in root folder:

```env
# Firebase Configuration
FIREBASE_PROJECT_ID=coconut-app-cd914
FIREBASE_PROJECT_NUMBER=<get-from-team-lead>
FIREBASE_API_KEY=<get-from-team-lead>
FIREBASE_APP_ID=<get-from-team-lead>

# Google OAuth
GOOGLE_WEB_CLIENT_ID=<get-from-team-lead>.apps.googleusercontent.com
GOOGLE_ANDROID_CLIENT_ID=<get-from-team-lead>.apps.googleusercontent.com

# App Configuration
APP_PACKAGE_NAME=com.coconuthealthmonitorNew
DEBUG_SHA1=<your-debug-sha1>

# ML API (local development)
API_BASE_URL=http://10.0.2.2:5000
```

## Step 3: Install Dependencies

### React Native (Mobile App)
```powershell
npm install
```

### Python (ML)
```powershell
cd ml
pip install -r requirements.txt
```

## Step 4: Android Setup

### Get your SHA-1 fingerprint:
```powershell
cd android
.\gradlew signingReport
```

Copy the SHA-1 and:
1. Add it to your `.env` file
2. Send it to team lead to add to Firebase Console

### Create autolinking.json (if build fails):
```powershell
mkdir -p android/build/generated/autolinking
```

Then create `android/build/generated/autolinking/autolinking.json`:
```json
{
  "reactNativeVersion": "0.82.1",
  "dependencies": {},
  "project": {
    "android": {
      "sourceDir": "./android",
      "appName": "app",
      "packageName": "com.coconuthealthmonitorNew"
    }
  }
}
```

## Step 5: Run the Project

### Terminal 1 - Start Metro Bundler:
```powershell
node node_modules\@react-native-community\cli\build\bin.js start
```

### Terminal 2 - Run on Android:
```powershell
node node_modules\@react-native-community\cli\build\bin.js run-android
```

### Terminal 3 - Run ML API (optional):
```powershell
cd ml/api
python run_api.py
```

## Troubleshooting

### Build fails with autolinking error
Create the `autolinking.json` file as shown in Step 4.

### Metro Permission Error
Run PowerShell as Administrator.

### Google Sign-In DEVELOPER_ERROR
1. Check SHA-1 is added to Firebase Console
2. Verify `google-services.json` is in `android/app/`
3. Make sure Web Client ID in `.env` matches Google Cloud Console

### Port 8081 already in use
```powershell
netstat -ano | findstr :8081
taskkill /PID <PID_NUMBER> /F
```

### ML API not working
1. Check model files exist in `ml/models/`
2. Verify Python dependencies installed
3. Check port 5000 is available

## Project Structure

```
Research/
├── .env                    # Environment variables (create this)
├── android/
│   └── app/
│       └── google-services.json  # Firebase config (get from team lead)
├── ml/
│   ├── api/                # Flask API server
│   ├── models/
│   │   ├── coconut_mite_v7/
│   │   │   └── best_model.keras  # Get from team lead
│   │   └── coconut_caterpillar/
│   │       └── caterpillar_model.keras  # Get from team lead
│   └── notebooks/          # Jupyter notebooks
├── src/                    # React Native source
└── App.tsx                 # App entry point
```

## Need Help?

Contact your team lead for:
- Missing configuration files
- Firebase credentials
- Model files
- Any setup issues

---

Last updated: December 2024
