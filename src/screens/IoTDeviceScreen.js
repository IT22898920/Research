/**
 * IoT Device Management Screen
 * Register, manage, and monitor ESP32 GPS devices
 */

import React, {useState, useEffect, useCallback} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  TextInput,
  ScrollView,
  ActivityIndicator,
  Modal,
  FlatList,
  Clipboard,
  Platform,
  RefreshControl,
} from 'react-native';
import {iotAPI} from '../services/iotApi';
import {plantationAPI} from '../services/plantationApi';

export default function IoTDeviceScreen({navigation}) {
  const [devices, setDevices] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  // Register modal
  const [showRegister, setShowRegister] = useState(false);
  const [newDeviceId, setNewDeviceId] = useState('');
  const [newDeviceName, setNewDeviceName] = useState('');
  const [selectedPlantation, setSelectedPlantation] = useState(null);
  const [plantations, setPlantations] = useState([]);
  const [registering, setRegistering] = useState(false);
  const [showPlantationPicker, setShowPlantationPicker] = useState(false);

  // Device key display
  const [showDeviceKey, setShowDeviceKey] = useState(false);
  const [registeredDevice, setRegisteredDevice] = useState(null);

  // Device details
  const [showDetails, setShowDetails] = useState(false);
  const [selectedDevice, setSelectedDevice] = useState(null);
  const [deviceTrees, setDeviceTrees] = useState([]);
  const [loadingTrees, setLoadingTrees] = useState(false);

  useEffect(() => {
    loadDevices();
  }, []);

  const loadDevices = async () => {
    try {
      const data = await iotAPI.getMyDevices();
      setDevices(data);
    } catch (error) {
      console.error('Load devices error:', error);
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadDevices();
    setRefreshing(false);
  }, []);

  const loadPlantations = async () => {
    try {
      const data = await plantationAPI.getMyPlantations();
      setPlantations(data);
    } catch (error) {
      console.error('Load plantations error:', error);
    }
  };

  const openRegisterModal = () => {
    setNewDeviceId('');
    setNewDeviceName('');
    setSelectedPlantation(null);
    loadPlantations();
    setShowRegister(true);
  };

  const registerDevice = async () => {
    if (!newDeviceId.trim()) {
      Alert.alert('Error', 'Please enter a Device ID');
      return;
    }

    setRegistering(true);
    try {
      const device = await iotAPI.registerDevice({
        deviceId: newDeviceId.trim(),
        name: newDeviceName.trim() || undefined,
        defaultPlantation: selectedPlantation?._id || undefined,
      });

      setRegisteredDevice(device);
      setShowRegister(false);
      setShowDeviceKey(true);
      loadDevices();
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to register device');
    } finally {
      setRegistering(false);
    }
  };

  const copyToClipboard = (text, label) => {
    if (Clipboard.setString) {
      Clipboard.setString(text);
      Alert.alert('Copied!', `${label} copied to clipboard`);
    }
  };

  const viewDeviceDetails = async (device) => {
    setSelectedDevice(device);
    setShowDetails(true);
    setLoadingTrees(true);

    try {
      const result = await iotAPI.getDeviceLocations(device._id);
      setDeviceTrees(result.trees);
    } catch (error) {
      console.error('Load device trees error:', error);
    } finally {
      setLoadingTrees(false);
    }
  };

  const deleteDevice = (device) => {
    Alert.alert(
      'Delete Device',
      `Are you sure you want to delete "${device.name}"?\n\nThis will NOT delete trees captured by this device.`,
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await iotAPI.deleteDevice(device._id);
              Alert.alert('Deleted', 'Device removed successfully');
              loadDevices();
              setShowDetails(false);
            } catch (error) {
              Alert.alert('Error', 'Failed to delete device');
            }
          },
        },
      ]
    );
  };

  const getTimeSince = (date) => {
    if (!date) return 'Never';
    const seconds = Math.floor((Date.now() - new Date(date)) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  const renderDevice = ({item}) => (
    <TouchableOpacity
      style={styles.deviceCard}
      onPress={() => viewDeviceDetails(item)}
    >
      <View style={styles.deviceHeader}>
        <View style={styles.deviceIconContainer}>
          <Text style={styles.deviceIcon}>📡</Text>
        </View>
        <View style={styles.deviceInfo}>
          <Text style={styles.deviceName}>{item.name}</Text>
          <Text style={styles.deviceIdText}>ID: {item.deviceId}</Text>
        </View>
        <View
          style={[
            styles.statusDot,
            {backgroundColor: item.isActive ? '#4CAF50' : '#999'},
          ]}
        />
      </View>

      <View style={styles.deviceStats}>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{item.totalReadings || 0}</Text>
          <Text style={styles.statLabel}>Readings</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>{getTimeSince(item.lastSeenAt)}</Text>
          <Text style={styles.statLabel}>Last Seen</Text>
        </View>
        <View style={styles.statItem}>
          <Text style={styles.statValue}>
            {item.defaultPlantation?.name || 'None'}
          </Text>
          <Text style={styles.statLabel}>Plantation</Text>
        </View>
      </View>
    </TouchableOpacity>
  );

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#2e7d32" />
        <Text style={styles.loadingText}>Loading devices...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          style={styles.backButton}
        >
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>IoT Devices</Text>
        <TouchableOpacity onPress={openRegisterModal} style={styles.addButton}>
          <Text style={styles.addButtonText}>+ Add</Text>
        </TouchableOpacity>
      </View>

      {/* Info Banner */}
      <View style={styles.infoBanner}>
        <Text style={styles.infoBannerIcon}>📡</Text>
        <Text style={styles.infoBannerText}>
          Connect ESP32 GPS devices for accurate tree location mapping (~2m
          accuracy)
        </Text>
      </View>

      {/* Device List */}
      {devices.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyIcon}>🔌</Text>
          <Text style={styles.emptyTitle}>No Devices Registered</Text>
          <Text style={styles.emptyText}>
            Register your ESP32 GPS device to start capturing accurate tree
            locations.
          </Text>
          <TouchableOpacity
            style={styles.emptyButton}
            onPress={openRegisterModal}
          >
            <Text style={styles.emptyButtonText}>+ Register Device</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <FlatList
          data={devices}
          renderItem={renderDevice}
          keyExtractor={(item) => item._id}
          contentContainerStyle={styles.listContainer}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={onRefresh}
              colors={['#2e7d32']}
            />
          }
        />
      )}

      {/* Register Device Modal */}
      <Modal
        visible={showRegister}
        animationType="slide"
        onRequestClose={() => setShowRegister(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <TouchableOpacity onPress={() => setShowRegister(false)}>
              <Text style={styles.modalClose}>✕</Text>
            </TouchableOpacity>
            <Text style={styles.modalTitle}>Register Device</Text>
            <View style={{width: 30}} />
          </View>

          <ScrollView style={styles.modalContent}>
            <Text style={styles.fieldLabel}>Device ID *</Text>
            <TextInput
              style={styles.input}
              placeholder="e.g., ESP32_001"
              value={newDeviceId}
              onChangeText={setNewDeviceId}
              maxLength={50}
              autoCapitalize="none"
            />
            <Text style={styles.fieldHint}>
              Unique identifier for your ESP32 device
            </Text>

            <Text style={styles.fieldLabel}>Device Name</Text>
            <TextInput
              style={styles.input}
              placeholder="e.g., GPS Device - Kurunegala"
              value={newDeviceName}
              onChangeText={setNewDeviceName}
              maxLength={100}
            />

            <Text style={styles.fieldLabel}>Default Plantation</Text>
            <TouchableOpacity
              style={styles.plantationSelector}
              onPress={() => setShowPlantationPicker(true)}
            >
              <Text
                style={
                  selectedPlantation
                    ? styles.plantationSelected
                    : styles.plantationPlaceholder
                }
              >
                {selectedPlantation?.name || 'Select a plantation (optional)'}
              </Text>
              <Text style={styles.selectorArrow}>▼</Text>
            </TouchableOpacity>
            <Text style={styles.fieldHint}>
              Trees captured by this device will be added to this plantation
            </Text>

            <TouchableOpacity
              style={[
                styles.registerButton,
                (!newDeviceId.trim() || registering) &&
                  styles.registerButtonDisabled,
              ]}
              onPress={registerDevice}
              disabled={!newDeviceId.trim() || registering}
            >
              {registering ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.registerButtonText}>
                  Register Device
                </Text>
              )}
            </TouchableOpacity>
          </ScrollView>

          {/* Plantation Picker */}
          <Modal
            visible={showPlantationPicker}
            animationType="slide"
            transparent
            onRequestClose={() => setShowPlantationPicker(false)}
          >
            <View style={styles.pickerOverlay}>
              <View style={styles.pickerContainer}>
                <View style={styles.pickerHeader}>
                  <Text style={styles.pickerTitle}>Select Plantation</Text>
                  <TouchableOpacity
                    onPress={() => setShowPlantationPicker(false)}
                  >
                    <Text style={styles.pickerClose}>✕</Text>
                  </TouchableOpacity>
                </View>

                <TouchableOpacity
                  style={styles.pickerItem}
                  onPress={() => {
                    setSelectedPlantation(null);
                    setShowPlantationPicker(false);
                  }}
                >
                  <Text style={styles.pickerItemText}>None (set later)</Text>
                </TouchableOpacity>

                {plantations.map((p) => (
                  <TouchableOpacity
                    key={p._id}
                    style={styles.pickerItem}
                    onPress={() => {
                      setSelectedPlantation(p);
                      setShowPlantationPicker(false);
                    }}
                  >
                    <Text style={styles.pickerItemText}>{p.name}</Text>
                    <Text style={styles.pickerItemSub}>
                      {p.stats?.totalTrees || 0} trees
                    </Text>
                  </TouchableOpacity>
                ))}
              </View>
            </View>
          </Modal>
        </View>
      </Modal>

      {/* Device Key Modal (shown after registration) */}
      <Modal
        visible={showDeviceKey}
        animationType="fade"
        transparent
        onRequestClose={() => setShowDeviceKey(false)}
      >
        <View style={styles.keyOverlay}>
          <View style={styles.keyContainer}>
            <Text style={styles.keyTitle}>Device Registered!</Text>
            <Text style={styles.keyWarning}>
              Save this device key — it won't be shown again!
            </Text>

            <View style={styles.keySection}>
              <Text style={styles.keyLabel}>Device ID</Text>
              <TouchableOpacity
                style={styles.keyValue}
                onPress={() =>
                  copyToClipboard(registeredDevice?.deviceId, 'Device ID')
                }
              >
                <Text style={styles.keyValueText}>
                  {registeredDevice?.deviceId}
                </Text>
                <Text style={styles.copyIcon}>📋</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.keySection}>
              <Text style={styles.keyLabel}>Device Key (Secret)</Text>
              <TouchableOpacity
                style={styles.keyValue}
                onPress={() =>
                  copyToClipboard(registeredDevice?.deviceKey, 'Device Key')
                }
              >
                <Text style={[styles.keyValueText, styles.keySecret]}>
                  {registeredDevice?.deviceKey}
                </Text>
                <Text style={styles.copyIcon}>📋</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.keyInstructions}>
              <Text style={styles.keyInstructionsTitle}>
                ESP32 Firmware Configuration:
              </Text>
              <Text style={styles.keyCode}>
                {`#define DEVICE_ID "${registeredDevice?.deviceId}"\n#define DEVICE_KEY "${registeredDevice?.deviceKey?.substring(0, 20)}..."`}
              </Text>
            </View>

            <TouchableOpacity
              style={styles.keyDoneButton}
              onPress={() => setShowDeviceKey(false)}
            >
              <Text style={styles.keyDoneButtonText}>
                I've Saved the Key
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Device Details Modal */}
      <Modal
        visible={showDetails}
        animationType="slide"
        onRequestClose={() => setShowDetails(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <TouchableOpacity onPress={() => setShowDetails(false)}>
              <Text style={styles.modalClose}>✕</Text>
            </TouchableOpacity>
            <Text style={styles.modalTitle}>
              {selectedDevice?.name || 'Device Details'}
            </Text>
            <TouchableOpacity onPress={() => deleteDevice(selectedDevice)}>
              <Text style={styles.deleteText}>Delete</Text>
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.modalContent}>
            {/* Device Info */}
            <View style={styles.detailSection}>
              <Text style={styles.detailSectionTitle}>Device Info</Text>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Device ID</Text>
                <Text style={styles.detailValue}>
                  {selectedDevice?.deviceId}
                </Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Status</Text>
                <Text
                  style={[
                    styles.detailValue,
                    {
                      color: selectedDevice?.isActive ? '#4CAF50' : '#F44336',
                    },
                  ]}
                >
                  {selectedDevice?.isActive ? 'Active' : 'Inactive'}
                </Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Total Readings</Text>
                <Text style={styles.detailValue}>
                  {selectedDevice?.totalReadings || 0}
                </Text>
              </View>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Last Seen</Text>
                <Text style={styles.detailValue}>
                  {selectedDevice?.lastSeenAt
                    ? new Date(selectedDevice.lastSeenAt).toLocaleString()
                    : 'Never'}
                </Text>
              </View>
              {selectedDevice?.lastLocation?.latitude && (
                <View style={styles.detailRow}>
                  <Text style={styles.detailLabel}>Last Location</Text>
                  <Text style={styles.detailValue}>
                    {selectedDevice.lastLocation.latitude.toFixed(6)},{' '}
                    {selectedDevice.lastLocation.longitude.toFixed(6)}
                  </Text>
                </View>
              )}
            </View>

            {/* Trees captured by this device */}
            <View style={styles.detailSection}>
              <Text style={styles.detailSectionTitle}>
                Captured Trees ({deviceTrees.length})
              </Text>
              {loadingTrees ? (
                <ActivityIndicator
                  color="#2e7d32"
                  style={{marginVertical: 20}}
                />
              ) : deviceTrees.length === 0 ? (
                <Text style={styles.noTreesText}>
                  No trees captured yet. Use the device to start marking tree
                  locations.
                </Text>
              ) : (
                deviceTrees.map((tree) => (
                  <View key={tree._id} style={styles.treeItem}>
                    <Text style={styles.treeLabel}>{tree.label}</Text>
                    <Text style={styles.treeCoords}>
                      {tree.latitude?.toFixed(6)}, {tree.longitude?.toFixed(6)}
                    </Text>
                    <Text style={styles.treeDate}>
                      {new Date(tree.createdAt).toLocaleDateString()}
                    </Text>
                  </View>
                ))
              )}
            </View>
          </ScrollView>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: '#f5f5f5'},
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  loadingText: {marginTop: 10, color: '#666'},

  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 15,
    paddingTop: 50,
    paddingBottom: 15,
    backgroundColor: '#2e7d32',
  },
  backButton: {padding: 5},
  backButtonText: {color: '#fff', fontSize: 16, fontWeight: '500'},
  headerTitle: {color: '#fff', fontSize: 18, fontWeight: 'bold'},
  addButton: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 15,
  },
  addButtonText: {color: '#fff', fontWeight: 'bold'},

  // Info Banner
  infoBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#e8f5e9',
    padding: 12,
    marginHorizontal: 15,
    marginTop: 15,
    borderRadius: 10,
    gap: 10,
  },
  infoBannerIcon: {fontSize: 24},
  infoBannerText: {flex: 1, color: '#2e7d32', fontSize: 13},

  // Empty State
  emptyContainer: {flex: 1, justifyContent: 'center', alignItems: 'center', padding: 40},
  emptyIcon: {fontSize: 60, marginBottom: 15},
  emptyTitle: {fontSize: 20, fontWeight: 'bold', color: '#333', marginBottom: 10},
  emptyText: {fontSize: 14, color: '#666', textAlign: 'center', marginBottom: 20},
  emptyButton: {
    backgroundColor: '#2e7d32',
    paddingHorizontal: 25,
    paddingVertical: 12,
    borderRadius: 25,
  },
  emptyButtonText: {color: '#fff', fontWeight: 'bold', fontSize: 16},

  // Device List
  listContainer: {padding: 15},
  deviceCard: {
    backgroundColor: '#fff',
    borderRadius: 15,
    padding: 15,
    marginBottom: 12,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  deviceHeader: {flexDirection: 'row', alignItems: 'center', marginBottom: 12},
  deviceIconContainer: {
    width: 45,
    height: 45,
    borderRadius: 22,
    backgroundColor: '#e8f5e9',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  deviceIcon: {fontSize: 22},
  deviceInfo: {flex: 1},
  deviceName: {fontSize: 16, fontWeight: 'bold', color: '#333'},
  deviceIdText: {fontSize: 12, color: '#888', marginTop: 2},
  statusDot: {width: 10, height: 10, borderRadius: 5},

  deviceStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
    paddingTop: 12,
  },
  statItem: {alignItems: 'center'},
  statValue: {fontSize: 14, fontWeight: 'bold', color: '#333'},
  statLabel: {fontSize: 11, color: '#888', marginTop: 2},

  // Modal
  modalContainer: {flex: 1, backgroundColor: '#fff'},
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingTop: Platform.OS === 'ios' ? 50 : 40,
    backgroundColor: '#2e7d32',
  },
  modalTitle: {fontSize: 18, fontWeight: 'bold', color: '#fff'},
  modalClose: {fontSize: 24, color: '#fff', padding: 5},
  modalContent: {flex: 1, padding: 20},
  deleteText: {color: '#ffcdd2', fontWeight: 'bold'},

  // Form
  fieldLabel: {fontSize: 14, fontWeight: 'bold', color: '#333', marginBottom: 8, marginTop: 15},
  fieldHint: {fontSize: 12, color: '#888', marginTop: -5, marginBottom: 10},
  input: {
    backgroundColor: '#f8f9fa',
    borderRadius: 10,
    padding: 15,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    marginBottom: 10,
  },
  plantationSelector: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#f8f9fa',
    borderRadius: 10,
    padding: 15,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    marginBottom: 10,
  },
  plantationPlaceholder: {color: '#999', fontSize: 16},
  plantationSelected: {color: '#333', fontSize: 16},
  selectorArrow: {color: '#999', fontSize: 12},

  registerButton: {
    backgroundColor: '#2e7d32',
    padding: 18,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 25,
  },
  registerButtonDisabled: {backgroundColor: '#a5d6a7'},
  registerButtonText: {color: '#fff', fontSize: 16, fontWeight: 'bold'},

  // Plantation Picker
  pickerOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  pickerContainer: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '60%',
  },
  pickerHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  pickerTitle: {fontSize: 18, fontWeight: 'bold', color: '#333'},
  pickerClose: {fontSize: 20, color: '#666'},
  pickerItem: {
    padding: 18,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  pickerItemText: {fontSize: 16, color: '#333'},
  pickerItemSub: {fontSize: 12, color: '#888', marginTop: 2},

  // Device Key Modal
  keyOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'center',
    padding: 20,
  },
  keyContainer: {backgroundColor: '#fff', borderRadius: 20, padding: 25},
  keyTitle: {fontSize: 22, fontWeight: 'bold', color: '#2e7d32', textAlign: 'center', marginBottom: 8},
  keyWarning: {
    fontSize: 14,
    color: '#d32f2f',
    textAlign: 'center',
    marginBottom: 20,
    fontWeight: '600',
  },
  keySection: {marginBottom: 15},
  keyLabel: {fontSize: 12, color: '#888', marginBottom: 5, fontWeight: '600'},
  keyValue: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  keyValueText: {
    flex: 1,
    fontSize: 14,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    color: '#333',
  },
  keySecret: {fontSize: 11},
  copyIcon: {fontSize: 18, marginLeft: 8},
  keyInstructions: {
    backgroundColor: '#1a1a2e',
    borderRadius: 10,
    padding: 15,
    marginTop: 10,
    marginBottom: 20,
  },
  keyInstructionsTitle: {color: '#aaa', fontSize: 12, marginBottom: 8},
  keyCode: {
    color: '#4ade80',
    fontSize: 12,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  keyDoneButton: {
    backgroundColor: '#2e7d32',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  keyDoneButtonText: {color: '#fff', fontSize: 16, fontWeight: 'bold'},

  // Detail Section
  detailSection: {
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    padding: 15,
    marginBottom: 15,
  },
  detailSectionTitle: {fontSize: 16, fontWeight: 'bold', color: '#333', marginBottom: 12},
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  detailLabel: {fontSize: 14, color: '#666'},
  detailValue: {fontSize: 14, fontWeight: '600', color: '#333'},
  noTreesText: {color: '#888', fontSize: 14, textAlign: 'center', padding: 20},

  // Tree items
  treeItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  treeLabel: {flex: 1, fontSize: 14, fontWeight: '600', color: '#333'},
  treeCoords: {
    fontSize: 11,
    color: '#888',
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    marginRight: 10,
  },
  treeDate: {fontSize: 11, color: '#aaa'},
});
