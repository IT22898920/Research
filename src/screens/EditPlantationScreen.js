import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  ScrollView,
  Alert,
  ActivityIndicator,
  Modal,
  Platform,
  PermissionsAndroid,
} from 'react-native';
import Geolocation from '@react-native-community/geolocation';
import {plantationAPI} from '../services/plantationApi';

// Sri Lankan districts for manual selection
const SRI_LANKAN_DISTRICTS = [
  {name: 'Kurunegala', province: 'North Western'},
  {name: 'Puttalam', province: 'North Western'},
  {name: 'Gampaha', province: 'Western'},
  {name: 'Colombo', province: 'Western'},
  {name: 'Kalutara', province: 'Western'},
  {name: 'Galle', province: 'Southern'},
  {name: 'Matara', province: 'Southern'},
  {name: 'Hambantota', province: 'Southern'},
  {name: 'Kegalle', province: 'Sabaragamuwa'},
  {name: 'Ratnapura', province: 'Sabaragamuwa'},
  {name: 'Kandy', province: 'Central'},
  {name: 'Matale', province: 'Central'},
  {name: 'Anuradhapura', province: 'North Central'},
  {name: 'Polonnaruwa', province: 'North Central'},
  {name: 'Batticaloa', province: 'Eastern'},
  {name: 'Ampara', province: 'Eastern'},
  {name: 'Trincomalee', province: 'Eastern'},
  {name: 'Jaffna', province: 'Northern'},
];

export default function EditPlantationScreen({navigation, route}) {
  const {plantation} = route.params;

  const [name, setName] = useState(plantation.name || '');
  const [description, setDescription] = useState(plantation.description || '');
  const [area, setArea] = useState(plantation.area?.toString() || '');
  const [latitude, setLatitude] = useState(
    plantation.location?.coordinates?.[1] || null,
  );
  const [longitude, setLongitude] = useState(
    plantation.location?.coordinates?.[0] || null,
  );
  const [district, setDistrict] = useState(plantation.location?.district || '');
  const [province, setProvince] = useState(plantation.location?.province || '');
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [gettingLocation, setGettingLocation] = useState(false);
  const [showDistrictPicker, setShowDistrictPicker] = useState(false);

  const checkLocationPermission = async () => {
    if (Platform.OS === 'android') {
      try {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
        );
        return granted === PermissionsAndroid.RESULTS.GRANTED;
      } catch (err) {
        console.warn(err);
        return false;
      }
    }
    return true;
  };

  const getCurrentLocation = async () => {
    setGettingLocation(true);

    try {
      const hasPermission = await checkLocationPermission();
      if (!hasPermission) {
        Alert.alert('Permission Denied', 'Location permission is required');
        setGettingLocation(false);
        return;
      }

      Geolocation.getCurrentPosition(
        (position) => {
          setLatitude(position.coords.latitude);
          setLongitude(position.coords.longitude);
          setGettingLocation(false);
          Alert.alert('Success', 'Location updated successfully!');
        },
        (error) => {
          console.error('Location error:', error);
          setGettingLocation(false);
          Alert.alert(
            'Location Error',
            'Could not get your location. Please select district manually.',
          );
        },
        {
          enableHighAccuracy: true,
          timeout: 15000,
          maximumAge: 10000,
        },
      );
    } catch (error) {
      console.error('Location error:', error);
      setGettingLocation(false);
      Alert.alert('Error', 'Failed to get location');
    }
  };

  const selectDistrict = (selectedDistrict) => {
    setDistrict(selectedDistrict.name);
    setProvince(selectedDistrict.province);
    setShowDistrictPicker(false);
  };

  const handleSave = async () => {
    if (!name.trim()) {
      Alert.alert('Error', 'Please enter a plantation name');
      return;
    }

    setSaving(true);

    try {
      const updateData = {
        name: name.trim(),
        description: description.trim() || undefined,
        area: area ? parseFloat(area) : undefined,
        location: {
          coordinates: latitude && longitude ? [longitude, latitude] : [0, 0],
          district: district || undefined,
          province: province || undefined,
        },
      };

      await plantationAPI.update(plantation._id, updateData);

      Alert.alert('Success', 'Plantation updated successfully!', [
        {
          text: 'OK',
          onPress: () => navigation.goBack(),
        },
      ]);
    } catch (error) {
      console.error('Save error:', error);
      Alert.alert('Error', error.message || 'Failed to update plantation');
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = () => {
    Alert.alert(
      'Delete Plantation',
      `Are you sure you want to delete "${plantation.name}"?\n\nThis will also delete all ${plantation.stats?.totalTrees || 0} trees in this plantation. This action cannot be undone.`,
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            setDeleting(true);
            try {
              await plantationAPI.delete(plantation._id);
              Alert.alert('Success', 'Plantation deleted successfully', [
                {
                  text: 'OK',
                  onPress: () => navigation.navigate('PlantationList'),
                },
              ]);
            } catch (error) {
              Alert.alert('Error', 'Failed to delete plantation');
            } finally {
              setDeleting(false);
            }
          },
        },
      ],
    );
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}>
          <Text style={styles.backButtonText}>← Cancel</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Edit Plantation</Text>
        <View style={styles.headerRight} />
      </View>

      <ScrollView style={styles.content} contentContainerStyle={styles.contentContainer}>
        {/* Name Input */}
        <View style={styles.section}>
          <Text style={styles.sectionIcon}>🌴</Text>
          <Text style={styles.sectionTitle}>Plantation Name *</Text>
          <TextInput
            style={styles.input}
            placeholder="Enter plantation name"
            value={name}
            onChangeText={setName}
            maxLength={100}
          
            placeholderTextColor="#999"
          />
        </View>

        {/* Description Input */}
        <View style={styles.section}>
          <Text style={styles.sectionIcon}>📝</Text>
          <Text style={styles.sectionTitle}>Description</Text>
          <TextInput
            style={[styles.input, styles.textArea]}
            placeholder="Enter description (optional)"
            value={description}
            onChangeText={setDescription}
            multiline
            numberOfLines={3}
            maxLength={500}
          
            placeholderTextColor="#999"
          />
        </View>

        {/* Area Input */}
        <View style={styles.section}>
          <Text style={styles.sectionIcon}>📐</Text>
          <Text style={styles.sectionTitle}>Area (Hectares)</Text>
          <TextInput
            style={styles.input}
            placeholder="Enter area in hectares (optional)"
            value={area}
            onChangeText={setArea}
            keyboardType="decimal-pad"
          
            placeholderTextColor="#999"
          />
        </View>

        {/* Location Section */}
        <View style={styles.section}>
          <Text style={styles.sectionIcon}>📍</Text>
          <Text style={styles.sectionTitle}>Location</Text>

          {/* GPS Button */}
          <TouchableOpacity
            style={styles.locationButton}
            onPress={getCurrentLocation}
            disabled={gettingLocation}>
            {gettingLocation ? (
              <ActivityIndicator color="#2e7d32" size="small" />
            ) : (
              <>
                <Text style={styles.locationButtonIcon}>📡</Text>
                <Text style={styles.locationButtonText}>Update GPS Location</Text>
              </>
            )}
          </TouchableOpacity>

          {/* GPS Coordinates Display */}
          {latitude && longitude && (
            <View style={styles.coordinatesBox}>
              <Text style={styles.coordinatesLabel}>GPS Coordinates:</Text>
              <Text style={styles.coordinatesText}>
                {latitude.toFixed(6)}, {longitude.toFixed(6)}
              </Text>
            </View>
          )}

          {/* District Selector */}
          <TouchableOpacity
            style={styles.districtButton}
            onPress={() => setShowDistrictPicker(true)}>
            <Text style={styles.districtButtonIcon}>🗺️</Text>
            <Text style={styles.districtButtonText}>
              {district ? `${district}, ${province}` : 'Select District'}
            </Text>
            <Text style={styles.districtButtonArrow}>▼</Text>
          </TouchableOpacity>
        </View>

        {/* Stats Section (Read Only) */}
        <View style={styles.section}>
          <Text style={styles.sectionIcon}>📊</Text>
          <Text style={styles.sectionTitle}>Current Statistics</Text>
          <View style={styles.statsGrid}>
            <View style={styles.statBox}>
              <Text style={styles.statNumber}>{plantation.stats?.totalTrees || 0}</Text>
              <Text style={styles.statLabel}>Total Trees</Text>
            </View>
            <View style={styles.statBox}>
              <Text style={[styles.statNumber, {color: '#4CAF50'}]}>
                {plantation.stats?.healthyTrees || 0}
              </Text>
              <Text style={styles.statLabel}>Healthy</Text>
            </View>
            <View style={styles.statBox}>
              <Text style={[styles.statNumber, {color: '#FF9800'}]}>
                {plantation.stats?.unhealthyTrees || 0}
              </Text>
              <Text style={styles.statLabel}>Unhealthy</Text>
            </View>
            <View style={styles.statBox}>
              <Text style={[styles.statNumber, {color: '#F44336'}]}>
                {plantation.stats?.infectedTrees || 0}
              </Text>
              <Text style={styles.statLabel}>Infected</Text>
            </View>
          </View>
        </View>

        {/* Save Button */}
        <TouchableOpacity
          style={[styles.saveButton, (!name.trim() || saving) && styles.saveButtonDisabled]}
          onPress={handleSave}
          disabled={!name.trim() || saving}>
          {saving ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.saveButtonText}>💾 Save Changes</Text>
          )}
        </TouchableOpacity>

        {/* Delete Button */}
        <TouchableOpacity
          style={styles.deleteButton}
          onPress={handleDelete}
          disabled={deleting}>
          {deleting ? (
            <ActivityIndicator color="#d32f2f" />
          ) : (
            <Text style={styles.deleteButtonText}>🗑️ Delete Plantation</Text>
          )}
        </TouchableOpacity>
      </ScrollView>

      {/* District Picker Modal */}
      <Modal
        visible={showDistrictPicker}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowDistrictPicker(false)}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Select District</Text>
              <TouchableOpacity onPress={() => setShowDistrictPicker(false)}>
                <Text style={styles.modalClose}>✕</Text>
              </TouchableOpacity>
            </View>
            <ScrollView style={styles.districtList}>
              {SRI_LANKAN_DISTRICTS.map((item, index) => (
                <TouchableOpacity
                  key={index}
                  style={styles.districtItem}
                  onPress={() => selectDistrict(item)}>
                  <Text style={styles.districtName}>{item.name}</Text>
                  <Text style={styles.districtProvince}>{item.province} Province</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
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
  headerRight: {
    width: 70,
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
    borderRadius: 12,
    padding: 15,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  sectionIcon: {
    fontSize: 24,
    marginBottom: 5,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  input: {
    backgroundColor: '#f8f9fa',
    borderRadius: 10,
    padding: 15,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    color: '#000',
  },
  textArea: {
    minHeight: 80,
    textAlignVertical: 'top',
  },
  locationButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#e8f5e9',
    padding: 15,
    borderRadius: 10,
    marginBottom: 10,
  },
  locationButtonIcon: {
    fontSize: 20,
    marginRight: 10,
  },
  locationButtonText: {
    color: '#2e7d32',
    fontSize: 16,
    fontWeight: '600',
  },
  coordinatesBox: {
    backgroundColor: '#f0f7f0',
    padding: 12,
    borderRadius: 8,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#c8e6c9',
  },
  coordinatesLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 4,
  },
  coordinatesText: {
    fontSize: 14,
    color: '#2e7d32',
    fontWeight: '600',
  },
  districtButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f8f9fa',
    padding: 15,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  districtButtonIcon: {
    fontSize: 20,
    marginRight: 10,
  },
  districtButtonText: {
    flex: 1,
    fontSize: 16,
    color: '#333',
  },
  districtButtonArrow: {
    fontSize: 12,
    color: '#666',
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  statBox: {
    width: '50%',
    padding: 10,
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  statLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  saveButton: {
    backgroundColor: '#2e7d32',
    padding: 18,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 10,
  },
  saveButtonDisabled: {
    backgroundColor: '#a5d6a7',
  },
  saveButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  deleteButton: {
    backgroundColor: '#fff',
    padding: 18,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 10,
    borderWidth: 2,
    borderColor: '#d32f2f',
  },
  deleteButtonText: {
    color: '#d32f2f',
    fontSize: 16,
    fontWeight: 'bold',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '70%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  modalClose: {
    fontSize: 24,
    color: '#666',
    padding: 5,
  },
  districtList: {
    padding: 10,
  },
  districtItem: {
    padding: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  districtName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  districtProvince: {
    fontSize: 13,
    color: '#666',
    marginTop: 2,
  },
});
