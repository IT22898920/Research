# Coconut Health Monitor

AI-Powered Drone-Based System for Comprehensive Coconut Tree Health Monitoring and Yield Prediction.

## Project Overview

This mobile application is part of a research project that uses drone technology and artificial intelligence to monitor coconut tree health and predict yields. The app serves as the user interface for farmers and agricultural professionals to access monitoring data and insights.

## Tech Stack

- **Frontend:** React Native 0.82.1
- **Navigation:** React Navigation 7.x
- **Backend:** Node.js + Express (planned)
- **Database:** MongoDB (planned)
- **Language:** JavaScript/TypeScript

## Features

### Current Features
- User Authentication UI (Login/Signup screens)
- Navigation between screens
- Responsive design for Android devices

### Planned Features
- MongoDB integration for user authentication
- Dashboard with coconut tree health data
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
│   ├── components/             # Reusable UI components
│   └── navigation/             # Navigation configuration
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
