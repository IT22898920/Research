/**
 * Add Tree Screen
 * Capture coconut tree GPS location and add to plantation
 * Supports GPS + Manual location selection
 */

import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  TextInput,
  ScrollView,
  ActivityIndicator,
  Platform,
  Modal,
  PermissionsAndroid,
} from 'react-native';
import Geolocation from '@react-native-community/geolocation';
import {treeAPI} from '../services/treeApi';

// Sri Lanka coconut growing regions with coordinates
const SRI_LANKA_LOCATIONS = [
  { id: '1', name: 'Kurunegala', district: 'Coconut Triangle', lat: 7.4863, lng: 80.3647 },
  { id: '2', name: 'Puttalam', district: 'Coconut Triangle', lat: 8.0362, lng: 79.8283 },
  { id: '3', name: 'Gampaha', district: 'Coconut Triangle', lat: 7.0840, lng: 80.0098 },
  { id: '4', name: 'Colombo', district: 'Western', lat: 6.9271, lng: 79.8612 },
  { id: '5', name: 'Negombo', district: 'Western', lat: 7.2008, lng: 79.8358 },
  { id: '6', name: 'Kalutara', district: 'Western', lat: 6.5854, lng: 79.9607 },
  { id: '7', name: 'Galle', district: 'Southern', lat: 6.0535, lng: 80.2210 },
  { id: '8', name: 'Matara', district: 'Southern', lat: 5.9549, lng: 80.5550 },
  { id: '9', name: 'Kandy', district: 'Central', lat: 7.2906, lng: 80.6337 },
  { id: '10', name: 'Anuradhapura', district: 'North Central', lat: 8.3114, lng: 80.4037 },
];

export default function AddTreeScreen({navigation, route}) {
  const plantationId = route.params?.plantationId;
  const plantationName = route.params?.plantationName || 'My Plantation';

  const [latitude, setLatitude] = useState(null);
  const [longitude, setLongitude] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [locationName, setLocationName] = useState('');
  const [gettingLocation, setGettingLocation] = useState(false);
  const [saving, setSaving] = useState(false);
  const [treeLabel, setTreeLabel] = useState('');
  const [notes, setNotes] = useState('');
  const [showLocationPicker, setShowLocationPicker] = useState(false);
  const [hasPermission, setHasPermission] = useState(false);

  useEffect(() => {
    checkPermission();
  }, []);

  const checkPermission = async () => {
    if (Platform.OS === 'android') {
      try {
        const granted = await PermissionsAndroid.check(
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION
        );
        setHasPermission(granted);
      } catch (err) {
        console.log('Permission check error:', err);
      }
    } else {
      setHasPermission(true);
    }
  };

  const requestPermission = async () => {
    if (Platform.OS === 'android') {
      try {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
          {
            title: 'Location Permission',
            message: 'This app needs access to your location to mark tree positions.',
            buttonNeutral: 'Ask Me Later',
            buttonNegative: 'Cancel',
            buttonPositive: 'OK',
          },
        );
        const isGranted = granted === PermissionsAndroid.RESULTS.GRANTED;
        setHasPermission(isGranted);
        return isGranted;
      } catch (err) {
        console.warn('Permission request error:', err);
        return false;
      }
    }
    return true;
  };

  const getCurrentLocation = async () => {
    try {
      // Check/request permission first
      let permission = hasPermission;
      if (!permission) {
        permission = await requestPermission();
      }

      if (!permission) {
        Alert.alert(
          'Permission Required',
          'Location permission is needed. You can also select a location manually.',
          [
            { text: 'Select Manually', onPress: () => setShowLocationPicker(true) },
            { text: 'OK' }
          ]
        );
        return;
      }

      setGettingLocation(true);

      // Use watchPosition for better accuracy - collect readings and pick the best one
      let bestReading = null;
      let readingCount = 0;
      const maxReadings = 5;
      const targetAccuracy = 10; // meters

      const watchId = Geolocation.watchPosition(
        (position) => {
          const { latitude: lat, longitude: lng, accuracy: acc } = position.coords;
          readingCount++;
          console.log(`GPS Reading ${readingCount}: accuracy=${acc?.toFixed(1)}m`);

          // Keep the reading with best accuracy
          if (!bestReading || (acc && acc < bestReading.accuracy)) {
            bestReading = { lat, lng, accuracy: acc };
          }

          // Stop if we got good enough accuracy or max readings
          if ((acc && acc <= targetAccuracy) || readingCount >= maxReadings) {
            Geolocation.clearWatch(watchId);

            setLatitude(bestReading.lat);
            setLongitude(bestReading.lng);
            setAccuracy(bestReading.accuracy);
            setLocationName('Current Location (GPS)');
            setGettingLocation(false);

            Alert.alert(
              'Location Captured! ✅',
              `Lat: ${bestReading.lat.toFixed(6)}\nLng: ${bestReading.lng.toFixed(6)}\nAccuracy: ${bestReading.accuracy ? bestReading.accuracy.toFixed(1) + 'm' : 'N/A'}\n(Best of ${readingCount} readings)`
            );
          }
        },
        (error) => {
          Geolocation.clearWatch(watchId);
          setGettingLocation(false);
          console.log('GPS Error:', error.code, error.message);

          // If we have a reading, use it even if not ideal
          if (bestReading) {
            setLatitude(bestReading.lat);
            setLongitude(bestReading.lng);
            setAccuracy(bestReading.accuracy);
            setLocationName('Current Location (GPS)');
            Alert.alert(
              'Location Captured ⚠️',
              `Using best available reading:\nLat: ${bestReading.lat.toFixed(6)}\nLng: ${bestReading.lng.toFixed(6)}\nAccuracy: ${bestReading.accuracy?.toFixed(1)}m`
            );
            return;
          }

          let errorMsg = 'Could not get your location. ';
          switch (error.code) {
            case 1:
              errorMsg += 'Permission denied.';
              break;
            case 2:
              errorMsg += 'Location unavailable. Try again or select manually.';
              break;
            case 3:
              errorMsg += 'Request timed out. Try again.';
              break;
            default:
              errorMsg += error.message;
          }

          Alert.alert(
            'GPS Error',
            errorMsg,
            [
              { text: 'Try Again', onPress: getCurrentLocation },
              { text: 'Select Manually', onPress: () => setShowLocationPicker(true) },
            ]
          );
        },
        {
          enableHighAccuracy: true,
          distanceFilter: 0,
          interval: 1000,
          fastestInterval: 500,
        }
      );

      // Timeout after 10 seconds
      setTimeout(() => {
        if (gettingLocation) {
          Geolocation.clearWatch(watchId);
          if (bestReading) {
            setLatitude(bestReading.lat);
            setLongitude(bestReading.lng);
            setAccuracy(bestReading.accuracy);
            setLocationName('Current Location (GPS)');
            setGettingLocation(false);
            Alert.alert(
              'Location Captured ✅',
              `Lat: ${bestReading.lat.toFixed(6)}\nLng: ${bestReading.lng.toFixed(6)}\nAccuracy: ${bestReading.accuracy?.toFixed(1)}m`
            );
          } else {
            setGettingLocation(false);
            Alert.alert(
              'Location Timeout ⏱️',
              'Could not get GPS location.\n\n📱 On Emulator: Click "..." → Location → Set coordinates\n\n📍 Or select location manually below.',
              [
                { text: 'Select Manually', onPress: () => setShowLocationPicker(true) },
                { text: 'Try Again', onPress: getCurrentLocation },
              ]
            );
          }
        }
      }, 10000);
    } catch (err) {
      setGettingLocation(false);
      console.log('Get location error:', err);
      Alert.alert(
        'Error',
        'Something went wrong. Please try selecting location manually.',
        [{ text: 'Select Manually', onPress: () => setShowLocationPicker(true) }]
      );
    }
  };

  const selectManualLocation = (location) => {
    setLatitude(location.lat);
    setLongitude(location.lng);
    setLocationName(location.name);
    setAccuracy(null);
    setShowLocationPicker(false);
  };

  const clearLocation = () => {
    setLatitude(null);
    setLongitude(null);
    setLocationName('');
    setAccuracy(null);
  };

  const saveTree = async () => {
    if (!latitude || !longitude) {
      Alert.alert('Error', 'Please capture or select a location first.');
      return;
    }

    if (!treeLabel.trim()) {
      Alert.alert('Error', 'Please enter a tree label/number.');
      return;
    }

    setSaving(true);
    try {
      const treeData = {
        plantationId: plantationId || 'default',
        plantationName: plantationName,
        label: treeLabel.trim(),
        latitude: latitude,
        longitude: longitude,
        locationName: locationName,
        accuracy: accuracy,
        notes: notes.trim(),
        lastHealthStatus: null,
      };

      await treeAPI.addTree(treeData);

      Alert.alert(
        'Success! 🎉',
        `Tree "${treeLabel}" saved successfully!`,
        [
          {
            text: 'Add Another',
            onPress: () => {
              setTreeLabel('');
              setNotes('');
              clearLocation();
            },
          },
          {
            text: 'View All Trees',
            onPress: () => navigation.navigate('PlantationMap', {plantationId}),
          },
        ],
      );
    } catch (error) {
      Alert.alert('Error', 'Failed to save tree. Please try again.');
      console.error('Save tree error:', error);
    } finally {
      setSaving(false);
    }
  };

  const getAccuracyColor = () => {
    if (!accuracy) return '#888';
    if (accuracy <= 5) return '#4CAF50';
    if (accuracy <= 10) return '#8BC34A';
    if (accuracy <= 20) return '#FFC107';
    return '#F44336';
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Add Tree Location</Text>
        <View style={styles.placeholder} />
      </View>

      <ScrollView style={styles.content} contentContainerStyle={styles.contentContainer}>
        {/* Location Section */}
        <View style={styles.section}>
          <Text style={styles.sectionIcon}>📍</Text>
          <Text style={styles.sectionTitle}>Tree Location</Text>

          {/* GPS Button */}
          <TouchableOpacity
            style={[styles.gpsButton, gettingLocation && styles.gpsButtonDisabled]}
            onPress={getCurrentLocation}
            disabled={gettingLocation}
          >
            {gettingLocation ? (
              <>
                <ActivityIndicator color="#fff" size="small" />
                <Text style={styles.gpsButtonText}>Getting Location...</Text>
              </>
            ) : (
              <>
                <Text style={styles.gpsButtonIcon}>📡</Text>
                <Text style={styles.gpsButtonText}>Get Current Location (GPS)</Text>
              </>
            )}
          </TouchableOpacity>

          {/* OR Divider */}
          <View style={styles.divider}>
            <View style={styles.dividerLine} />
            <Text style={styles.dividerText}>OR</Text>
            <View style={styles.dividerLine} />
          </View>

          {/* Manual Selection Button */}
          <TouchableOpacity
            style={styles.manualButton}
            onPress={() => setShowLocationPicker(true)}
          >
            <Text style={styles.manualButtonIcon}>🗺️</Text>
            <Text style={styles.manualButtonText}>Select Location Manually</Text>
          </TouchableOpacity>

          {/* Selected Location Display */}
          {latitude && longitude && (
            <View style={styles.locationDisplay}>
              <View style={styles.locationHeader}>
                <Text style={styles.locationTitle}>✅ Location Set</Text>
                <TouchableOpacity onPress={clearLocation}>
                  <Text style={styles.clearButton}>Clear</Text>
                </TouchableOpacity>
              </View>
              <Text style={styles.locationNameText}>{locationName}</Text>
              <Text style={styles.coordsText}>
                Lat: {latitude.toFixed(6)} | Lng: {longitude.toFixed(6)}
              </Text>
              {accuracy && (
                <View style={[styles.accuracyBadge, { backgroundColor: getAccuracyColor() }]}>
                  <Text style={styles.accuracyText}>Accuracy: {accuracy.toFixed(1)}m</Text>
                </View>
              )}
            </View>
          )}
        </View>

        {/* Tree Details Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>🌴 Tree Details</Text>

          <Text style={styles.label}>Tree Label/Number *</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g., Tree 001, A1, North-1"
            value={treeLabel}
            onChangeText={setTreeLabel}
            maxLength={50}
          />

          <Text style={styles.label}>Notes (Optional)</Text>
          <TextInput
            style={[styles.input, styles.notesInput]}
            placeholder="Any additional notes about this tree..."
            value={notes}
            onChangeText={setNotes}
            multiline
            numberOfLines={3}
            maxLength={500}
          />
        </View>

        {/* Save Button */}
        <TouchableOpacity
          style={[
            styles.saveButton,
            (!latitude || !longitude || !treeLabel || saving) && styles.saveButtonDisabled
          ]}
          onPress={saveTree}
          disabled={!latitude || !longitude || !treeLabel || saving}
        >
          {saving ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.saveButtonText}>💾 Save Tree Location</Text>
          )}
        </TouchableOpacity>

        {/* View Trees Button */}
        <TouchableOpacity
          style={styles.viewTreesButton}
          onPress={() => navigation.navigate('PlantationMap', {plantationId})}
        >
          <Text style={styles.viewTreesButtonText}>🌴 View All Trees</Text>
        </TouchableOpacity>
      </ScrollView>

      {/* Location Picker Modal */}
      <Modal
        visible={showLocationPicker}
        animationType="slide"
        onRequestClose={() => setShowLocationPicker(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Select Location</Text>
            <TouchableOpacity onPress={() => setShowLocationPicker(false)}>
              <Text style={styles.modalClose}>✕</Text>
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.locationList}>
            {SRI_LANKA_LOCATIONS.map((location) => (
              <TouchableOpacity
                key={location.id}
                style={styles.locationItem}
                onPress={() => selectManualLocation(location)}
              >
                <Text style={styles.locationItemIcon}>📍</Text>
                <View style={styles.locationItemInfo}>
                  <Text style={styles.locationItemName}>{location.name}</Text>
                  <Text style={styles.locationItemDistrict}>{location.district}</Text>
                </View>
                <Text style={styles.locationItemArrow}>→</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 15,
    paddingTop: 50,
    paddingBottom: 15,
    backgroundColor: '#2e7d32',
  },
  backButton: {
    padding: 5,
  },
  backButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '500',
  },
  headerTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  placeholder: {
    width: 50,
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: 20,
    paddingBottom: 40,
  },
  section: {
    backgroundColor: '#fff',
    borderRadius: 15,
    padding: 20,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  sectionIcon: {
    fontSize: 40,
    textAlign: 'center',
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
    textAlign: 'center',
  },
  gpsButton: {
    backgroundColor: '#1976D2',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
    gap: 10,
  },
  gpsButtonDisabled: {
    backgroundColor: '#90CAF9',
  },
  gpsButtonIcon: {
    fontSize: 24,
  },
  gpsButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 15,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: '#e0e0e0',
  },
  dividerText: {
    marginHorizontal: 15,
    color: '#888',
    fontWeight: '500',
  },
  manualButton: {
    backgroundColor: '#fff',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#2e7d32',
    gap: 10,
  },
  manualButtonIcon: {
    fontSize: 24,
  },
  manualButtonText: {
    color: '#2e7d32',
    fontSize: 16,
    fontWeight: 'bold',
  },
  locationDisplay: {
    backgroundColor: '#e8f5e9',
    borderRadius: 12,
    padding: 15,
    marginTop: 15,
  },
  locationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  locationTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2e7d32',
  },
  clearButton: {
    color: '#d32f2f',
    fontWeight: '600',
  },
  locationNameText: {
    fontSize: 14,
    color: '#333',
    marginBottom: 4,
  },
  coordsText: {
    fontSize: 12,
    color: '#666',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  accuracyBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 10,
    marginTop: 8,
  },
  accuracyText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  label: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  input: {
    backgroundColor: '#f8f9fa',
    borderRadius: 10,
    padding: 15,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    marginBottom: 15,
  },
  notesInput: {
    height: 100,
    textAlignVertical: 'top',
  },
  saveButton: {
    backgroundColor: '#2e7d32',
    padding: 18,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 15,
  },
  saveButtonDisabled: {
    backgroundColor: '#a5d6a7',
  },
  saveButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  viewTreesButton: {
    backgroundColor: '#fff',
    padding: 18,
    borderRadius: 10,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#2e7d32',
  },
  viewTreesButtonText: {
    color: '#2e7d32',
    fontSize: 16,
    fontWeight: 'bold',
  },
  // Modal Styles
  modalContainer: {
    flex: 1,
    backgroundColor: '#fff',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingTop: 50,
    backgroundColor: '#2e7d32',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  modalClose: {
    fontSize: 24,
    color: '#fff',
    padding: 5,
  },
  locationList: {
    flex: 1,
  },
  locationItem: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 18,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  locationItemIcon: {
    fontSize: 28,
    marginRight: 15,
  },
  locationItemInfo: {
    flex: 1,
  },
  locationItemName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  locationItemDistrict: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
  },
  locationItemArrow: {
    fontSize: 20,
    color: '#ccc',
  },
});
