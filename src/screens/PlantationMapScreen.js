/**
 * Plantation Map Screen
 * Display all coconut trees with health status on a real map
 */

import React, {useState, useCallback, useRef, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  FlatList,
  Modal,
  ScrollView,
  Platform,
  PermissionsAndroid,
} from 'react-native';
import MapView, {Marker, Circle, PROVIDER_GOOGLE} from 'react-native-maps';
import Geolocation from '@react-native-community/geolocation';
import {treeAPI} from '../services/treeApi';
import {useFocusEffect} from '@react-navigation/native';


export default function PlantationMapScreen({navigation, route}) {
  const plantationId = route.params?.plantationId;
  const mapRef = useRef(null);

  const [trees, setTrees] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedTree, setSelectedTree] = useState(null);
  const [showTreeModal, setShowTreeModal] = useState(false);
  const [stats, setStats] = useState(null);
  const [viewMode, setViewMode] = useState('map'); // 'map' or 'list' - default to map
  const [mapReady, setMapReady] = useState(false);
  const [mapType, setMapType] = useState('standard'); // 'standard' or 'satellite'

  // Custom stable current location with AUTO-STABILIZATION
  const [currentLocation, setCurrentLocation] = useState(null);
  const [locationAccuracy, setLocationAccuracy] = useState(null);
  const [stabilityStatus, setStabilityStatus] = useState('searching'); // 'searching', 'stabilizing', 'stable'
  const [stabilityProgress, setStabilityProgress] = useState(0); // 0-100%
  const watchIdRef = useRef(null);
  const locationHistoryRef = useRef([]); // Store last N readings for averaging
  const stableLocationRef = useRef(null); // Store the stable location once achieved
  const MAX_HISTORY = 5; // Number of readings to average
  const STABILITY_THRESHOLD = 5; // meters - if all readings within this, consider stable

  // Load trees when screen is focused
  useFocusEffect(
    useCallback(() => {
      loadTrees();
    }, [plantationId])
  );

  // Start/stop location tracking based on view mode
  useEffect(() => {
    if (viewMode === 'map') {
      startLocationTracking();
    }
    return () => {
      stopLocationTracking();
    };
  }, [viewMode]);

  // Calculate distance between two coordinates (in meters)
  const getDistance = (lat1, lon1, lat2, lon2) => {
    const R = 6371000; // Earth's radius in meters
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon/2) * Math.sin(dLon/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    return R * c;
  };

  // Calculate averaged/smoothed position from history
  const getSmoothedLocation = (history) => {
    if (history.length === 0) return null;

    const sumLat = history.reduce((sum, loc) => sum + loc.latitude, 0);
    const sumLng = history.reduce((sum, loc) => sum + loc.longitude, 0);
    const avgAccuracy = history.reduce((sum, loc) => sum + loc.accuracy, 0) / history.length;

    return {
      latitude: sumLat / history.length,
      longitude: sumLng / history.length,
      accuracy: avgAccuracy,
    };
  };

  // Calculate max spread between all readings in history
  const calculateSpread = (history) => {
    if (history.length < 2) return 0;

    let maxDistance = 0;
    for (let i = 0; i < history.length; i++) {
      for (let j = i + 1; j < history.length; j++) {
        const dist = getDistance(
          history[i].latitude, history[i].longitude,
          history[j].latitude, history[j].longitude
        );
        if (dist > maxDistance) maxDistance = dist;
      }
    }
    return maxDistance;
  };

  // Check if location is stable based on spread of readings
  const checkStability = (history) => {
    if (history.length < MAX_HISTORY) {
      // Still collecting readings
      setStabilityStatus('searching');
      setStabilityProgress(Math.round((history.length / MAX_HISTORY) * 50)); // 0-50%
      return false;
    }

    const spread = calculateSpread(history);

    if (spread <= STABILITY_THRESHOLD) {
      // All readings within threshold - STABLE!
      setStabilityStatus('stable');
      setStabilityProgress(100);
      return true;
    } else if (spread <= STABILITY_THRESHOLD * 3) {
      // Getting closer - stabilizing
      setStabilityStatus('stabilizing');
      const progress = 50 + Math.round((1 - (spread - STABILITY_THRESHOLD) / (STABILITY_THRESHOLD * 2)) * 50);
      setStabilityProgress(Math.min(95, Math.max(50, progress)));
      return false;
    } else {
      // Still too spread out
      setStabilityStatus('searching');
      setStabilityProgress(40);
      return false;
    }
  };

  // Start watching location with smoothing
  const startLocationTracking = async () => {
    try {
      // Request permission on Android
      if (Platform.OS === 'android') {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION
        );
        if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
          console.log('Location permission denied');
          return;
        }
      }

      watchIdRef.current = Geolocation.watchPosition(
        (position) => {
          // If already stable, don't update (auto-locked)
          if (stableLocationRef.current) {
            return;
          }

          const { latitude, longitude, accuracy } = position.coords;

          // Only accept readings with reasonable accuracy (< 30m)
          if (accuracy > 30) {
            console.log(`Ignoring poor accuracy: ${accuracy.toFixed(1)}m`);
            return;
          }

          const newReading = { latitude, longitude, accuracy };

          // Filter out outliers - if current history exists, check distance
          if (locationHistoryRef.current.length > 0) {
            const avgLocation = getSmoothedLocation(locationHistoryRef.current);
            const distance = getDistance(
              avgLocation.latitude,
              avgLocation.longitude,
              latitude,
              longitude
            );

            // Ignore readings that are too far from average (likely GPS jump)
            if (distance > 30) {
              console.log(`Ignoring outlier: ${distance.toFixed(1)}m from average`);
              return;
            }
          }

          // Add to history
          locationHistoryRef.current.push(newReading);

          // Keep only last N readings
          if (locationHistoryRef.current.length > MAX_HISTORY) {
            locationHistoryRef.current.shift();
          }

          // Calculate smoothed position
          const smoothed = getSmoothedLocation(locationHistoryRef.current);

          if (smoothed) {
            setCurrentLocation({
              latitude: smoothed.latitude,
              longitude: smoothed.longitude,
            });
            setLocationAccuracy(smoothed.accuracy);

            // Check if we've achieved stability
            const isStable = checkStability(locationHistoryRef.current);
            if (isStable) {
              // AUTO-LOCK: Store the stable location
              stableLocationRef.current = {
                latitude: smoothed.latitude,
                longitude: smoothed.longitude,
                accuracy: smoothed.accuracy,
              };
              console.log('GPS STABLE! Auto-locked at:', smoothed);
            }
          }
        },
        (error) => {
          console.log('Location error:', error);
        },
        {
          enableHighAccuracy: true,
          distanceFilter: 3, // Update when moved 3+ meters
          interval: 3000, // Update every 3 seconds
          fastestInterval: 1000,
        }
      );
    } catch (err) {
      console.log('Start location tracking error:', err);
    }
  };

  // Stop watching location
  const stopLocationTracking = () => {
    if (watchIdRef.current !== null) {
      Geolocation.clearWatch(watchIdRef.current);
      watchIdRef.current = null;
    }
  };

  // Center map on current location
  const centerOnCurrentLocation = () => {
    if (currentLocation && mapRef.current) {
      mapRef.current.animateToRegion({
        latitude: currentLocation.latitude,
        longitude: currentLocation.longitude,
        latitudeDelta: 0.005,
        longitudeDelta: 0.005,
      }, 500);
    } else {
      Alert.alert('Location', 'Getting your location... Please wait.');
    }
  };

  // Recalibrate - restart the stabilization process
  const recalibrateLocation = () => {
    // Clear stable location and history
    stableLocationRef.current = null;
    locationHistoryRef.current = [];
    setStabilityStatus('searching');
    setStabilityProgress(0);
    Alert.alert('🔄 Recalibrating', 'Finding your location again. Please wait...');
  };

  const loadTrees = async () => {
    setLoading(true);
    try {
      let treesData;
      if (plantationId) {
        treesData = await treeAPI.getTreesByPlantation(plantationId);
      } else {
        treesData = await treeAPI.getAllTrees();
      }
      setTrees(treesData);

      // Calculate stats
      const treeStats = await treeAPI.getTreeStats(plantationId);
      setStats(treeStats);
    } catch (error) {
      console.error('Error loading trees:', error);
      Alert.alert('Error', 'Failed to load trees.');
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (healthStatus) => {
    switch (healthStatus) {
      case 'healthy':
        return '#4CAF50';
      case 'unhealthy':
        return '#FF9800';
      case 'infected':
        return '#F44336';
      default:
        return '#9E9E9E';
    }
  };

  const getStatusText = (healthStatus) => {
    switch (healthStatus) {
      case 'healthy':
        return 'Healthy';
      case 'unhealthy':
        return 'Unhealthy';
      case 'infected':
        return 'Infected';
      default:
        return 'Not Scanned';
    }
  };

  const getStatusIcon = (healthStatus) => {
    switch (healthStatus) {
      case 'healthy':
        return '✅';
      case 'unhealthy':
        return '⚠️';
      case 'infected':
        return '🔴';
      default:
        return '❓';
    }
  };

  const handleTreePress = (tree) => {
    setSelectedTree(tree);
    setShowTreeModal(true);
  };

  const handleScanTree = () => {
    setShowTreeModal(false);
    navigation.navigate('PestDetection', {treeId: selectedTree.id, treeLabel: selectedTree.label});
  };

  const handleDeleteTree = () => {
    Alert.alert(
      'Delete Tree',
      `Are you sure you want to delete "${selectedTree.label}"?`,
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await treeAPI.deleteTree(selectedTree.id);
              setShowTreeModal(false);
              loadTrees();
              Alert.alert('Success', 'Tree deleted successfully.');
            } catch (error) {
              Alert.alert('Error', 'Failed to delete tree.');
            }
          },
        },
      ],
    );
  };

  // Fit map to show all tree markers
  const fitAllMarkers = () => {
    if (mapRef.current && trees.length > 0) {
      const coordinates = trees.map(tree => ({
        latitude: tree.latitude,
        longitude: tree.longitude,
      }));
      mapRef.current.fitToCoordinates(coordinates, {
        edgePadding: {top: 100, right: 50, bottom: 100, left: 50},
        animated: true,
      });
    }
  };

  // Handle map ready
  const onMapReady = () => {
    setMapReady(true);
    // Fit markers after a short delay to ensure map is fully loaded
    setTimeout(() => {
      fitAllMarkers();
    }, 500);
  };

  // Get marker color based on health status
  const getMarkerColor = (healthStatus) => {
    switch (healthStatus) {
      case 'healthy':
        return '#4CAF50';
      case 'unhealthy':
        return '#FF9800';
      case 'infected':
        return '#F44336';
      default:
        return '#9E9E9E';
    }
  };

  // Handle marker drag to update tree location
  const handleMarkerDrag = (tree, newCoordinate) => {
    Alert.alert(
      'Update Location? 📍',
      `Move "${tree.label}" to new position?\n\nNew Lat: ${newCoordinate.latitude.toFixed(6)}\nNew Lng: ${newCoordinate.longitude.toFixed(6)}`,
      [
        {
          text: 'Cancel',
          style: 'cancel',
          onPress: () => loadTrees(), // Reload to reset marker position
        },
        {
          text: 'Update',
          onPress: async () => {
            try {
              await treeAPI.updateTree(tree.id, {
                latitude: newCoordinate.latitude,
                longitude: newCoordinate.longitude,
                accuracy: null, // Clear accuracy since manually adjusted
              });
              Alert.alert('Success! ✅', 'Tree location updated.');
              loadTrees();
            } catch (error) {
              Alert.alert('Error', 'Failed to update location.');
              loadTrees();
            }
          },
        },
      ],
    );
  };

  const renderTreeItem = ({item}) => (
    <TouchableOpacity
      style={styles.treeCard}
      onPress={() => handleTreePress(item)}
    >
      <View style={styles.treeIconContainer}>
        <Text style={styles.treeIcon}>🌴</Text>
        <View style={[styles.statusDot, {backgroundColor: getStatusColor(item.lastHealthStatus)}]} />
      </View>
      <View style={styles.treeInfo}>
        <Text style={styles.treeLabel}>{item.label}</Text>
        <Text style={styles.treeCoords}>
          📍 {item.latitude.toFixed(5)}, {item.longitude.toFixed(5)}
        </Text>
        <View style={styles.treeStatusRow}>
          <Text style={styles.statusIcon}>{getStatusIcon(item.lastHealthStatus)}</Text>
          <Text style={[styles.treeStatus, {color: getStatusColor(item.lastHealthStatus)}]}>
            {getStatusText(item.lastHealthStatus)}
          </Text>
        </View>
      </View>
      <Text style={styles.treeArrow}>→</Text>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>My Plantation</Text>
        <TouchableOpacity
          style={styles.addButton}
          onPress={() => navigation.navigate('AddTree', {plantationId})}
        >
          <Text style={styles.addButtonText}>+ Add</Text>
        </TouchableOpacity>
      </View>

      {/* View Mode Toggle */}
      <View style={styles.viewToggle}>
        <TouchableOpacity
          style={[styles.toggleButton, viewMode === 'map' && styles.toggleButtonActive]}
          onPress={() => setViewMode('map')}
        >
          <Text style={[styles.toggleText, viewMode === 'map' && styles.toggleTextActive]}>
            🗺️ Map
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.toggleButton, viewMode === 'list' && styles.toggleButtonActive]}
          onPress={() => setViewMode('list')}
        >
          <Text style={[styles.toggleText, viewMode === 'list' && styles.toggleTextActive]}>
            📋 List
          </Text>
        </TouchableOpacity>
      </View>

      {/* Stats Bar - Show for list view */}
      {stats && viewMode === 'list' && (
        <View style={styles.statsBar}>
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>{stats.total}</Text>
            <Text style={styles.statLabel}>Total</Text>
          </View>
          <View style={[styles.statItem, styles.statBorder]}>
            <Text style={[styles.statNumber, {color: '#4CAF50'}]}>{stats.healthy}</Text>
            <Text style={styles.statLabel}>Healthy</Text>
          </View>
          <View style={[styles.statItem, styles.statBorder]}>
            <Text style={[styles.statNumber, {color: '#FF9800'}]}>{stats.unhealthy}</Text>
            <Text style={styles.statLabel}>Unhealthy</Text>
          </View>
          <View style={[styles.statItem, styles.statBorder]}>
            <Text style={[styles.statNumber, {color: '#F44336'}]}>{stats.infected}</Text>
            <Text style={styles.statLabel}>Infected</Text>
          </View>
          <View style={[styles.statItem, styles.statBorder]}>
            <Text style={[styles.statNumber, {color: '#9E9E9E'}]}>{stats.notScanned}</Text>
            <Text style={styles.statLabel}>Pending</Text>
          </View>
        </View>
      )}

      {/* Legend - Show for list view only */}
      {viewMode === 'list' && (
        <View style={styles.legend}>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, {backgroundColor: '#4CAF50'}]} />
            <Text style={styles.legendText}>Healthy</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, {backgroundColor: '#FF9800'}]} />
            <Text style={styles.legendText}>Unhealthy</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, {backgroundColor: '#F44336'}]} />
            <Text style={styles.legendText}>Infected</Text>
          </View>
          <View style={styles.legendItem}>
            <View style={[styles.legendDot, {backgroundColor: '#9E9E9E'}]} />
            <Text style={styles.legendText}>Not Scanned</Text>
          </View>
        </View>
      )}

      {/* Content Area */}
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2e7d32" />
          <Text style={styles.loadingText}>Loading trees...</Text>
        </View>
      ) : trees.length === 0 ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateIcon}>🌴</Text>
          <Text style={styles.emptyStateTitle}>No Trees Added Yet</Text>
          <Text style={styles.emptyStateText}>
            Start by adding your first coconut tree location.
          </Text>
          <TouchableOpacity
            style={styles.emptyStateButton}
            onPress={() => navigation.navigate('AddTree', {plantationId})}
          >
            <Text style={styles.emptyStateButtonText}>+ Add First Tree</Text>
          </TouchableOpacity>
        </View>
      ) : viewMode === 'map' ? (
        /* Map View with Google Maps */
        <View style={styles.mapContainer}>
          <MapView
            ref={mapRef}
            style={styles.map}
            provider={PROVIDER_GOOGLE}
            mapType={mapType}
            onMapReady={onMapReady}
            showsUserLocation={false}
            showsMyLocationButton={false}
            initialRegion={{
              latitude: trees.length > 0 ? trees[0].latitude : 7.8731,
              longitude: trees.length > 0 ? trees[0].longitude : 80.7718,
              latitudeDelta: 0.01,
              longitudeDelta: 0.01,
            }}
          >
            {/* Custom Current Location Marker with Accuracy Circle */}
            {currentLocation && (
              <>
                {/* Accuracy Circle for current location */}
                {locationAccuracy && (
                  <Circle
                    center={currentLocation}
                    radius={locationAccuracy}
                    fillColor="rgba(33, 150, 243, 0.15)"
                    strokeColor="rgba(33, 150, 243, 0.5)"
                    strokeWidth={2}
                  />
                )}
                {/* Current Location Marker */}
                <Marker
                  coordinate={currentLocation}
                  anchor={{x: 0.5, y: 0.5}}
                >
                  <View style={styles.currentLocationMarker}>
                    <View style={styles.currentLocationDot} />
                  </View>
                </Marker>
              </>
            )}

            {/* Tree Markers */}
            {trees.map((tree) => (
              <React.Fragment key={tree.id}>
                {/* Accuracy Circle for tree */}
                {tree.accuracy && tree.accuracy > 0 && (
                  <Circle
                    center={{
                      latitude: tree.latitude,
                      longitude: tree.longitude,
                    }}
                    radius={tree.accuracy}
                    fillColor="rgba(46, 125, 50, 0.15)"
                    strokeColor="rgba(46, 125, 50, 0.5)"
                    strokeWidth={1}
                  />
                )}
                {/* Tree Marker */}
                <Marker
                  coordinate={{
                    latitude: tree.latitude,
                    longitude: tree.longitude,
                  }}
                  title={tree.label}
                  description={`Status: ${getStatusText(tree.lastHealthStatus)}`}
                  onPress={() => handleTreePress(tree)}
                  pinColor={getMarkerColor(tree.lastHealthStatus)}
                  draggable={true}
                  onDragEnd={(e) => handleMarkerDrag(tree, e.nativeEvent.coordinate)}
                />
              </React.Fragment>
            ))}
          </MapView>

          {/* Floating Legend */}
          <View style={styles.floatingLegend}>
            <View style={styles.legendRow}>
              <View style={[styles.legendDot, {backgroundColor: '#4CAF50'}]} />
              <Text style={styles.legendSmallText}>Healthy</Text>
            </View>
            <View style={styles.legendRow}>
              <View style={[styles.legendDot, {backgroundColor: '#FF9800'}]} />
              <Text style={styles.legendSmallText}>Unhealthy</Text>
            </View>
            <View style={styles.legendRow}>
              <View style={[styles.legendDot, {backgroundColor: '#F44336'}]} />
              <Text style={styles.legendSmallText}>Infected</Text>
            </View>
            <View style={styles.legendRow}>
              <View style={[styles.legendDot, {backgroundColor: '#9E9E9E'}]} />
              <Text style={styles.legendSmallText}>Not Scanned</Text>
            </View>
          </View>

          {/* Map Type Toggle */}
          <TouchableOpacity
            style={styles.mapTypeButton}
            onPress={() => setMapType(mapType === 'standard' ? 'satellite' : 'standard')}
          >
            <Text style={styles.mapTypeButtonText}>
              {mapType === 'standard' ? '🛰️ Satellite' : '🗺️ Map'}
            </Text>
          </TouchableOpacity>

          {/* Fit All Button */}
          <TouchableOpacity style={styles.fitAllButton} onPress={fitAllMarkers}>
            <Text style={styles.fitAllButtonText}>📍 Fit All</Text>
          </TouchableOpacity>

          {/* Center on Me Button */}
          <TouchableOpacity style={styles.centerOnMeButton} onPress={centerOnCurrentLocation}>
            <Text style={styles.centerOnMeButtonText}>📌 My Location</Text>
          </TouchableOpacity>

          {/* Stability Status & Recalibrate Button */}
          <View style={styles.stabilityContainer}>
            {/* Status Indicator */}
            <View style={[
              styles.stabilityIndicator,
              stabilityStatus === 'stable' && styles.stabilityStable,
              stabilityStatus === 'stabilizing' && styles.stabilitySearching,
              stabilityStatus === 'searching' && styles.stabilitySearching,
            ]}>
              <Text style={styles.stabilityIcon}>
                {stabilityStatus === 'stable' ? '✅' : stabilityStatus === 'stabilizing' ? '🔄' : '📡'}
              </Text>
              <Text style={styles.stabilityText}>
                {stabilityStatus === 'stable'
                  ? 'Stable'
                  : stabilityStatus === 'stabilizing'
                    ? `Stabilizing ${stabilityProgress}%`
                    : `Searching ${stabilityProgress}%`}
              </Text>
            </View>

            {/* Recalibrate Button - only show when stable */}
            {stabilityStatus === 'stable' && (
              <TouchableOpacity style={styles.recalibrateButton} onPress={recalibrateLocation}>
                <Text style={styles.recalibrateText}>🔄 Recalibrate</Text>
              </TouchableOpacity>
            )}
          </View>

          {/* Location Accuracy Info */}
          {currentLocation && locationAccuracy && (
            <View style={styles.accuracyInfo}>
              <Text style={styles.accuracyInfoText}>
                GPS: ±{locationAccuracy.toFixed(0)}m
              </Text>
            </View>
          )}

          {/* Tree Count Badge */}
          <View style={styles.treeCountBadge}>
            <Text style={styles.treeCountText}>🌴 {trees.length} trees</Text>
          </View>

          {/* Drag Hint */}
          <View style={styles.dragHint}>
            <Text style={styles.dragHintText}>💡 Long press & drag markers to adjust position</Text>
          </View>
        </View>
      ) : (
        /* List View */
        <FlatList
          data={trees}
          renderItem={renderTreeItem}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listContainer}
          refreshing={loading}
          onRefresh={loadTrees}
        />
      )}

      {/* Tree Detail Modal */}
      <Modal
        visible={showTreeModal}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setShowTreeModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>{selectedTree?.label}</Text>
              <TouchableOpacity onPress={() => setShowTreeModal(false)}>
                <Text style={styles.modalClose}>✕</Text>
              </TouchableOpacity>
            </View>

            <ScrollView style={styles.modalBody}>
              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Health Status:</Text>
                <View style={[
                  styles.statusBadge,
                  {backgroundColor: getStatusColor(selectedTree?.lastHealthStatus)}
                ]}>
                  <Text style={styles.statusBadgeText}>
                    {getStatusText(selectedTree?.lastHealthStatus)}
                  </Text>
                </View>
              </View>

              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Latitude:</Text>
                <Text style={styles.detailValue}>{selectedTree?.latitude.toFixed(6)}</Text>
              </View>

              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Longitude:</Text>
                <Text style={styles.detailValue}>{selectedTree?.longitude.toFixed(6)}</Text>
              </View>

              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>GPS Accuracy:</Text>
                <Text style={styles.detailValue}>
                  {selectedTree?.accuracy ? `${selectedTree.accuracy.toFixed(1)}m` : 'N/A'}
                </Text>
              </View>

              <View style={styles.detailRow}>
                <Text style={styles.detailLabel}>Added On:</Text>
                <Text style={styles.detailValue}>
                  {selectedTree?.createdAt
                    ? new Date(selectedTree.createdAt).toLocaleDateString()
                    : 'N/A'}
                </Text>
              </View>

              {selectedTree?.notes && (
                <View style={styles.notesSection}>
                  <Text style={styles.detailLabel}>Notes:</Text>
                  <Text style={styles.notesText}>{selectedTree.notes}</Text>
                </View>
              )}

              {/* Health History */}
              {selectedTree?.healthHistory?.length > 0 && (
                <View style={styles.historySection}>
                  <Text style={styles.historySectionTitle}>Health History</Text>
                  {selectedTree.healthHistory.slice(-5).reverse().map((scan, index) => (
                    <View key={scan.id || index} style={styles.historyItem}>
                      <Text style={styles.historyDate}>
                        {new Date(scan.scannedAt).toLocaleDateString()}
                      </Text>
                      <Text style={[
                        styles.historyStatus,
                        {color: getStatusColor(scan.status)}
                      ]}>
                        {getStatusText(scan.status)}
                      </Text>
                    </View>
                  ))}
                </View>
              )}
            </ScrollView>

            <View style={styles.modalActions}>
              <TouchableOpacity
                style={styles.scanButton}
                onPress={handleScanTree}
              >
                <Text style={styles.scanButtonText}>🔬 Scan for Diseases</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.deleteButton}
                onPress={handleDeleteTree}
              >
                <Text style={styles.deleteButtonText}>🗑️ Delete Tree</Text>
              </TouchableOpacity>
            </View>
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
  viewToggle: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    padding: 10,
    justifyContent: 'center',
    gap: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  toggleButton: {
    paddingHorizontal: 25,
    paddingVertical: 10,
    borderRadius: 20,
    backgroundColor: '#f0f0f0',
  },
  toggleButtonActive: {
    backgroundColor: '#2e7d32',
  },
  toggleText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
  },
  toggleTextActive: {
    color: '#fff',
  },
  mapContainer: {
    flex: 1,
    position: 'relative',
  },
  map: {
    flex: 1,
    width: '100%',
    height: '100%',
  },
  floatingLegend: {
    position: 'absolute',
    top: 10,
    left: 10,
    backgroundColor: 'rgba(255,255,255,0.95)',
    borderRadius: 10,
    padding: 10,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 5,
  },
  legendRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 5,
  },
  legendSmallText: {
    fontSize: 11,
    color: '#333',
    marginLeft: 6,
  },
  mapTypeButton: {
    position: 'absolute',
    top: 10,
    right: 10,
    backgroundColor: '#673AB7',
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 5,
  },
  mapTypeButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 13,
  },
  fitAllButton: {
    position: 'absolute',
    top: 55,
    right: 10,
    backgroundColor: '#2e7d32',
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 5,
  },
  fitAllButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 13,
  },
  centerOnMeButton: {
    position: 'absolute',
    top: 100,
    right: 10,
    backgroundColor: '#1976D2',
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 5,
  },
  centerOnMeButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 13,
  },
  stabilityContainer: {
    position: 'absolute',
    top: 145,
    right: 10,
    alignItems: 'flex-end',
  },
  stabilityIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FF9800',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 20,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 5,
  },
  stabilityStable: {
    backgroundColor: '#4CAF50',
  },
  stabilitySearching: {
    backgroundColor: '#FF9800',
  },
  stabilityIcon: {
    fontSize: 14,
    marginRight: 5,
  },
  stabilityText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 12,
  },
  recalibrateButton: {
    marginTop: 8,
    backgroundColor: '#1976D2',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.2,
    shadowRadius: 4,
    elevation: 5,
  },
  recalibrateText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 11,
  },
  accuracyInfo: {
    position: 'absolute',
    top: 225,
    right: 10,
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 10,
  },
  accuracyInfoText: {
    color: '#fff',
    fontSize: 11,
  },
  currentLocationMarker: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: 'rgba(33, 150, 243, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#2196F3',
  },
  currentLocationDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#2196F3',
  },
  treeCountBadge: {
    position: 'absolute',
    bottom: 55,
    alignSelf: 'center',
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
  },
  treeCountText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
  },
  dragHint: {
    position: 'absolute',
    bottom: 15,
    alignSelf: 'center',
    backgroundColor: 'rgba(103, 58, 183, 0.9)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 15,
  },
  dragHintText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '500',
  },
  mapPlaceholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#e8f5e9',
  },
  mapPlaceholderIcon: {
    fontSize: 80,
    marginBottom: 20,
  },
  mapPlaceholderText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  mapPlaceholderSubtext: {
    fontSize: 14,
    color: '#666',
    marginBottom: 20,
  },
  switchToListButton: {
    backgroundColor: '#2e7d32',
    paddingHorizontal: 25,
    paddingVertical: 12,
    borderRadius: 25,
  },
  switchToListButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
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
  addButton: {
    backgroundColor: 'rgba(255,255,255,0.2)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
  },
  addButtonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  statsBar: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statBorder: {
    borderLeftWidth: 1,
    borderLeftColor: '#e0e0e0',
  },
  statNumber: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  statLabel: {
    fontSize: 10,
    color: '#666',
    marginTop: 2,
  },
  legend: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    backgroundColor: '#fff',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  legendDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 5,
  },
  legendText: {
    fontSize: 11,
    color: '#666',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 10,
    color: '#666',
  },
  listContainer: {
    padding: 15,
    paddingBottom: 30,
  },
  treeCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  treeIconContainer: {
    position: 'relative',
    marginRight: 15,
  },
  treeIcon: {
    fontSize: 40,
  },
  statusDot: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 14,
    height: 14,
    borderRadius: 7,
    borderWidth: 2,
    borderColor: '#fff',
  },
  treeInfo: {
    flex: 1,
  },
  treeLabel: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  treeCoords: {
    fontSize: 12,
    color: '#666',
    marginBottom: 4,
  },
  treeStatusRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusIcon: {
    fontSize: 14,
    marginRight: 5,
  },
  treeStatus: {
    fontSize: 13,
    fontWeight: '600',
  },
  treeArrow: {
    fontSize: 20,
    color: '#ccc',
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  emptyStateIcon: {
    fontSize: 80,
    marginBottom: 20,
  },
  emptyStateTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  emptyStateText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 25,
  },
  emptyStateButton: {
    backgroundColor: '#2e7d32',
    paddingHorizontal: 30,
    paddingVertical: 15,
    borderRadius: 25,
  },
  emptyStateButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 16,
  },
  // Modal Styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#fff',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    maxHeight: '80%',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
  },
  modalClose: {
    fontSize: 24,
    color: '#666',
    padding: 5,
  },
  modalBody: {
    padding: 20,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  detailLabel: {
    fontSize: 14,
    color: '#666',
  },
  detailValue: {
    fontSize: 14,
    color: '#333',
    fontWeight: '500',
  },
  statusBadge: {
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 12,
  },
  statusBadgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  notesSection: {
    marginTop: 10,
    paddingTop: 15,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  notesText: {
    fontSize: 14,
    color: '#333',
    marginTop: 8,
    lineHeight: 20,
  },
  historySection: {
    marginTop: 15,
    paddingTop: 15,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  historySectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  historyItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  historyDate: {
    fontSize: 13,
    color: '#666',
  },
  historyStatus: {
    fontSize: 13,
    fontWeight: '500',
  },
  modalActions: {
    padding: 20,
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
    gap: 10,
  },
  scanButton: {
    backgroundColor: '#2e7d32',
    padding: 16,
    borderRadius: 10,
    alignItems: 'center',
  },
  scanButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  deleteButton: {
    backgroundColor: '#ffebee',
    padding: 16,
    borderRadius: 10,
    alignItems: 'center',
  },
  deleteButtonText: {
    color: '#d32f2f',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
