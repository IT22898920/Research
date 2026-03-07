# 🌳 Branch Health Detection - Mobile App Integration Guide

## 🎉 Integration Complete!

Your Branch Health Detection system is now **fully integrated** into the mobile app!

---

## 📱 What Was Added

### 1. **New Screen** - `BranchHealthScreen.js`
✅ Complete UI for branch health detection
✅ Camera & Gallery integration
✅ Real-time analysis with loading states
✅ Beautiful results display
✅ Unhealthy percentage visualization
✅ API status indicator

### 2. **Navigation** - Updated `App.tsx`
✅ Added BranchHealth route
✅ Connected to navigation stack

### 3. **Dashboard** - Updated `DashboardScreen.js`
✅ Added "Branch Health Monitor" button
✅ Icon: 🌳
✅ Positioned after Leaf Health

### 4. **API Service** - Already Updated
✅ `detectBranchHealth()` function ready
✅ Full API integration

---

## 🚀 How to Use (User Journey)

### Step 1: Open App & Login
```
1. Launch app
2. Login with credentials
3. You'll see the Dashboard
```

### Step 2: Navigate to Branch Health
```
Dashboard
  ↓
Click "🌳 Branch Health Monitor"
  ↓
BranchHealthScreen opens
```

### Step 3: Capture/Select Image
```
Two options:
📷 "Take Photo" - Opens camera
   OR
🖼️ "Choose from Gallery" - Opens gallery
```

### Step 4: Analyze
```
1. Image preview appears
2. Click "🔍 Analyze Branch"
3. Wait ~2-5 seconds (loading animation)
4. Results appear!
```

### Step 5: View Results
```
Results Display:
✅/⚠️ Health Status (Healthy/Unhealthy)
📊 Confidence Percentage
📈 Unhealthy Percentage (if unhealthy)
📋 Analysis Message
💡 Recommendation
📊 Probabilities (bar charts)
🤖 Model Information
```

---

## 📊 What Users See

### Healthy Branch Result:
```
┌─────────────────────────────────────┐
│ ✅ Healthy Branch                   │
│ 99.8% confident                     │
├─────────────────────────────────────┤
│ 📋 Analysis:                        │
│ Branch appears to be very healthy!  │
├─────────────────────────────────────┤
│ 💡 Recommendation:                  │
│ Continue regular monitoring and     │
│ maintain good care practices.       │
├─────────────────────────────────────┤
│ 📊 Detection Probabilities          │
│ ✅ Healthy:   [████████] 99.8%     │
│ ⚠️ Unhealthy: [░░░░░░░░]  0.2%     │
└─────────────────────────────────────┘
```

### Unhealthy Branch Result:
```
┌─────────────────────────────────────┐
│ ⚠️ Unhealthy Branch                 │
│ 98.5% confident                     │
├─────────────────────────────────────┤
│ Unhealthy Percentage:               │
│ [████████████████████░░░░] 85%     │
├─────────────────────────────────────┤
│ 📋 Analysis:                        │
│ Branch shows signs of being         │
│ unhealthy (85% unhealthy).          │
├─────────────────────────────────────┤
│ 💡 Recommendation:                  │
│ Inspect the branch for pest damage, │
│ disease, or nutrient deficiencies.  │
│ Consider pruning if severely        │
│ damaged.                            │
├─────────────────────────────────────┤
│ 📊 Detection Probabilities          │
│ ✅ Healthy:   [░░░░░░░░]  1.5%     │
│ ⚠️ Unhealthy: [████████] 98.5%     │
└─────────────────────────────────────┘
```

---

## 🎨 UI Features

### 1. **Beautiful Design**
- ✅ Clean, modern interface
- ✅ Coconut green color scheme (#2E7D32)
- ✅ Card-based layout
- ✅ Smooth animations

### 2. **Visual Feedback**
- 🟢 Green for healthy
- 🔴 Red for unhealthy
- 🟡 Yellow for warnings
- ⚪ Gray for neutral

### 3. **Progress Bars**
- Unhealthy percentage bar
- Color-coded based on severity:
  - 🔴 Red: >70% unhealthy (severe)
  - 🟠 Orange: 40-70% (moderate)
  - 🟡 Yellow: <40% (mild)

### 4. **Interactive Elements**
- ✓ Tap to take photo
- ✓ Tap to choose from gallery
- ✓ Clear button to reset
- ✓ Analyze button with loading state

---

## 🔧 Technical Details

### API Integration
```javascript
// Function call
const result = await detectBranchHealth(imageUri);

// Response structure
{
  success: true,
  detectionType: 'branch_health',
  prediction: 'healthy' | 'unhealthy',
  confidence: 0.998,
  probabilities: {
    healthy: 0.998,
    unhealthy: 0.002
  },
  unhealthyPercentage: 0,
  isHealthy: true,
  message: "Branch appears to be very healthy!",
  recommendation: "Continue regular monitoring...",
  modelInfo: {
    version: 'v1',
    accuracy: '99.63%'
  }
}
```

### Image Picker
```javascript
// Uses react-native-image-picker
- Quality: 0.8 (80% compression)
- Format: JPEG
- Max file size: ~2-5 MB
```

### API Connection
```javascript
// Default: Android Emulator
const API_BASE_URL = 'http://10.0.2.2:5001';

// For real device, change to computer IP:
const API_BASE_URL = 'http://192.168.x.x:5001';
```

---

## 🧪 Testing Steps

### Prerequisites:
```bash
1. ✅ Flask API running (port 5001)
2. ✅ Branch Health model trained
3. ✅ Model files in correct location
4. ✅ React Native app built
```

### Test Flow:

#### 1. Start Flask API
```bash
cd ml/api
python app.py

# Should see:
# [5] Loading Branch Health model (v1 - 2-class)...
#     Status: LOADED ✓
```

#### 2. Build & Run React Native App
```bash
# Terminal 1: Metro Bundler
cd D:\SLIIT\Reaserch Project\CoconutHealthMonitor\Research
node node_modules\@react-native-community\cli\build\bin.js start

# Terminal 2: Run on Android
node node_modules\@react-native-community\cli\build\bin.js run-android
```

#### 3. Test on Emulator/Device
```
1. Open app
2. Login
3. Click "🌳 Branch Health Monitor"
4. Take/Choose test image
5. Click "Analyze Branch"
6. Verify results display correctly
```

#### 4. Test Different Scenarios

**Test 1: Healthy Branch**
- Upload healthy branch image
- Expected: Green card, "Healthy Branch", 0% unhealthy

**Test 2: Unhealthy Branch**
- Upload unhealthy branch image
- Expected: Red card, "Unhealthy Branch", >50% unhealthy

**Test 3: Non-Branch Image**
- Upload random image (e.g., person, car)
- Expected: Works (model will classify as healthy/unhealthy)
  Note: Model doesn't reject non-coconut (unlike pest models)

**Test 4: API Offline**
- Stop Flask server
- Try to analyze
- Expected: "API Offline" alert

---

## 📱 Screen Flow Diagram

```
App Launch
    ↓
Login Screen
    ↓
Dashboard
    ↓
┌───────────────────────────────────┐
│  🌳 Branch Health Monitor         │ ← Click here
└───────────────────────────────────┘
    ↓
BranchHealthScreen
    ↓
┌─────────────────┬─────────────────┐
│  📷 Take Photo  │  🖼️ Gallery    │
└─────────────────┴─────────────────┘
    ↓
Image Selected
    ↓
┌──────────────────────────────────┐
│  🔍 Analyze Branch               │ ← Click to analyze
└──────────────────────────────────┘
    ↓
Loading... (2-5 seconds)
    ↓
Results Display
    ↓
┌──────────────────────────────────┐
│  ✅ Status Card                  │
│  📊 Probabilities                │
│  🤖 Model Info                   │
│  🔄 Analyze Another              │
└──────────────────────────────────┘
```

---

## 🎯 Features Comparison

| Feature | Leaf Health | Branch Health |
|---------|-------------|---------------|
| **Accuracy** | 93.70% | 99.63% ⭐ |
| **Classes** | 2 (healthy/unhealthy) | 2 (healthy/unhealthy) |
| **Special Output** | 9 detailed conditions | Unhealthy percentage |
| **Use Case** | Leaf yellowing detection | Branch damage detection |
| **Response Time** | ~2-3 seconds | ~2-3 seconds |
| **Icon** | 🌿 | 🌳 |

---

## 🔧 Customization Options

### Change API URL (for real device):
```javascript
// In BranchHealthScreen.js, line 16:
const API_BASE_URL = 'http://192.168.x.x:5001';
// Replace x.x with your computer's IP
```

### Change Colors:
```javascript
// Healthy color (green):
'#2E7D32' → Your color

// Unhealthy color (red):
'#F44336' → Your color
```

### Adjust Confidence Threshold:
```javascript
// Currently displays confidence as-is
// To filter low-confidence results:
if (result.confidence < 0.8) {
  Alert.alert('Low Confidence', 'Result may be unreliable');
}
```

---

## 🐛 Troubleshooting

### Issue 1: "API Offline" Error
**Solution:**
```bash
1. Check if Flask server is running
2. Verify URL matches (emulator vs device)
3. Check firewall settings
4. Try: curl http://10.0.2.2:5001/health
```

### Issue 2: Image Not Uploading
**Solution:**
```bash
1. Check permissions in AndroidManifest.xml:
   - CAMERA permission
   - READ_EXTERNAL_STORAGE
2. Request permissions at runtime
3. Verify react-native-image-picker is installed
```

### Issue 3: Results Not Displaying
**Solution:**
```bash
1. Check console logs (Metro bundler)
2. Verify API response structure
3. Check network tab in Chrome DevTools
4. Test API directly with Postman
```

### Issue 4: App Crashes on Image Select
**Solution:**
```bash
1. Rebuild app: cd android && ./gradlew clean
2. Clear Metro cache: npx react-native start --reset-cache
3. Check react-native-image-picker version compatibility
```

---

## 📊 Performance Metrics

### Expected Performance:
- **Image Upload:** <1 second
- **API Request:** 2-5 seconds
- **Result Display:** Instant
- **Total Time:** 3-6 seconds

### Optimization Tips:
1. Compress images before upload (already 80%)
2. Cache model in Flask API (already done)
3. Use loading indicators (already done)
4. Handle errors gracefully (already done)

---

## 🎓 User Training Guide (සිංහල)

### භාවිතා කරන ආකාරය:

#### 1. App එක Open කරන්න
```
Login → Dashboard → "🌳 Branch Health Monitor" click කරන්න
```

#### 2. Photo එකක් ගන්න
```
"📷 Take Photo" - Camera එක open වෙයි
     හෝ
"🖼️ Choose from Gallery" - Gallery එක open වෙයි
```

#### 3. Analyze කරන්න
```
Photo එක select කරපු පස්සේ:
"🔍 Analyze Branch" button එක click කරන්න
```

#### 4. Results බලන්න
```
2-5 seconds වලින් results පෙන්වයි:
- Branch එක healthy ද unhealthy ද
- Confidence percentage එක
- Unhealthy percentage එක (unhealthy නම්)
- මොනවද කරන්න ඕන කියලා recommendation එක
```

#### 5. තවත් photo එකක් check කරන්න
```
"🔄 Analyze Another" click කරලා නැවත කරන්න
```

---

## 🎉 Success Criteria

Your integration is successful if:

✅ Dashboard shows "🌳 Branch Health Monitor" button
✅ Clicking button opens BranchHealthScreen
✅ Camera/Gallery opens successfully
✅ Image displays in preview
✅ "Analyze Branch" button works
✅ Loading animation shows during analysis
✅ Results display with all information
✅ Confidence shows correctly
✅ Unhealthy percentage shows for unhealthy branches
✅ "Analyze Another" resets screen
✅ API status indicator works

---

## 📞 Support & Next Steps

### Need Help?
1. Check console logs in Metro bundler
2. Verify Flask API is running
3. Test API endpoint directly
4. Review error messages

### Future Enhancements:
- 📸 Save results to database
- 📊 View history of branch analyses
- 📈 Track branch health over time
- 🔔 Notifications for unhealthy branches
- 📍 GPS location tagging
- 🗺️ Map view of analyzed branches

---

## 🎯 Summary

**✅ Complete Mobile Integration:**
- New screen created
- Navigation configured
- Dashboard updated
- API connected
- Beautiful UI designed
- Error handling implemented
- Loading states added
- Results display optimized

**🚀 Ready for Production:**
- 99.63% accuracy model
- Fast response time (2-5s)
- User-friendly interface
- Comprehensive error handling

**📱 User Experience:**
- Simple 5-step process
- Clear visual feedback
- Actionable recommendations
- Professional design

---

**Status: ✅ COMPLETE & READY TO USE!**

**Created:** January 4, 2026
**Version:** Mobile App v1.0 with Branch Health Detection
**Model:** Branch Health v1 (99.63% accuracy)

🎉 **Congratulations! Your Branch Health Detection system is now live in the mobile app!** 🎉
