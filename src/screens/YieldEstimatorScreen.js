/**
 * Yield Estimator Screen
 * Estimates coconut yield by counting individual nuts on the tree
 * Model: YOLOv8 TFLite - Coconut Yield Estimator (Nuts per Tree)
 * Accepts 1 or 2 images from opposite sides of tree
 */

import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
  SafeAreaView,
} from 'react-native';
import {launchCamera, launchImageLibrary} from 'react-native-image-picker';
import {detectYield} from '../services/pestDetectionApi';
import {treeAPI} from '../services/treeApi';

// For Real Device via USB (adb reverse)
const API_BASE_URL = 'http://localhost:5001';

export default function YieldEstimatorScreen({navigation, route}) {
  // Get tree info from navigation params (if scanning a specific tree)
  const treeId = route?.params?.treeId;
  const treeLabel = route?.params?.treeLabel;

  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    checkApi();
  }, []);

  const checkApi = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      setApiStatus(data.status === 'healthy' ? 'online' : 'offline');
    } catch (error) {
      setApiStatus('offline');
    }
  };

  const imagePickerOptions = {
    mediaType: 'photo',
    quality: 0.8,
    maxWidth: 1280,
    maxHeight: 1280,
  };

  const selectImage = (imageNumber) => {
    Alert.alert(
      'Select Image',
      'Choose image source',
      [
        {text: 'Cancel', style: 'cancel'},
        {text: 'Camera', onPress: () => openCamera(imageNumber)},
        {text: 'Gallery', onPress: () => openGallery(imageNumber)},
      ],
    );
  };

  const openCamera = (imageNumber) => {
    launchCamera(imagePickerOptions, (response) => {
      handleImageResponse(response, imageNumber);
    });
  };

  const openGallery = (imageNumber) => {
    launchImageLibrary(imagePickerOptions, (response) => {
      handleImageResponse(response, imageNumber);
    });
  };

  const handleImageResponse = (response, imageNumber) => {
    if (response.didCancel) return;
    if (response.errorCode) {
      Alert.alert('Error', response.errorMessage || 'Failed to select image');
      return;
    }
    if (response.assets && response.assets[0]) {
      if (imageNumber === 1) {
        setImage1(response.assets[0]);
      } else {
        setImage2(response.assets[0]);
      }
      setResult(null);
    }
  };

  const clearImages = () => {
    setImage1(null);
    setImage2(null);
    setResult(null);
  };

  // Save scan result to tree record
  const saveToTreeRecord = async (yieldResult) => {
    if (!treeId) return;

    try {
      // Save nut count to tree's scan results
      await treeAPI.updateScanResults(treeId, {
        nutCount: yieldResult.estimatedTotal || yieldResult.totalNutCount,
      });

      // Also add to health history
      const treeScanData = {
        status: yieldResult.yieldCategory === 'none' ? 'unknown' :
                yieldResult.yieldCategory === 'low' ? 'unhealthy' : 'healthy',
        scanType: 'yield_estimation',
        details: {
          totalNutCount: yieldResult.totalNutCount,
          estimatedTotal: yieldResult.estimatedTotal,
          yieldCategory: yieldResult.yieldCategory,
          message: yieldResult.message,
          recommendation: yieldResult.recommendation,
        },
      };

      await treeAPI.addHealthScan(treeId, treeScanData);
      console.log('Yield scan saved to tree record:', treeId);
    } catch (error) {
      console.error('Error saving to tree record:', error);
    }
  };

  const analyzeImages = async () => {
    if (!image1) {
      Alert.alert('Error', 'Please select at least one image');
      return;
    }

    if (apiStatus !== 'online') {
      Alert.alert(
        'API Offline',
        'The ML API server is offline. Please make sure the Flask server is running.',
      );
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const yieldResult = await detectYield(
        image1.uri,
        image2?.uri || null
      );

      if (yieldResult.success) {
        setResult(yieldResult);
        // Save to tree record if treeId exists
        await saveToTreeRecord(yieldResult);
      } else {
        Alert.alert('Error', yieldResult.error || 'Failed to analyze images');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to connect to server');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const getYieldCategoryColor = (category) => {
    switch (category) {
      case 'excellent': return '#4CAF50';
      case 'good': return '#8BC34A';
      case 'moderate': return '#FFC107';
      case 'low': return '#FF9800';
      case 'none': return '#F44336';
      default: return '#9E9E9E';
    }
  };

  const getYieldCategoryIcon = (category) => {
    switch (category) {
      case 'excellent': return '🌟';
      case 'good': return '✅';
      case 'moderate': return '📊';
      case 'low': return '⚠️';
      case 'none': return '❌';
      default: return '❓';
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          style={styles.backButton}>
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Yield Estimator</Text>
        <View
          style={[
            styles.apiStatusBadge,
            apiStatus === 'online' ? styles.apiOnline : styles.apiOffline,
          ]}>
          <Text style={styles.apiStatusText}>
            {apiStatus === 'online' ? 'API Online' : 'API Offline'}
          </Text>
        </View>
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Tree Info (if scanning specific tree) */}
        {treeLabel && (
          <View style={styles.treeInfoCard}>
            <Text style={styles.treeInfoIcon}>🌴</Text>
            <Text style={styles.treeInfoText}>Scanning: {treeLabel}</Text>
          </View>
        )}

        {/* Instructions */}
        <View style={styles.instructionCard}>
          <Text style={styles.instructionTitle}>📷 How to Use</Text>
          <Text style={styles.instructionText}>
            1. Take a photo showing coconuts on the tree{'\n'}
            2. For better accuracy, add a second photo from the opposite side{'\n'}
            3. Tap "Estimate Yield" to count nuts
          </Text>
        </View>

        {/* Image Selection */}
        <View style={styles.imageSection}>
          <Text style={styles.sectionTitle}>Select Images</Text>

          {/* Image 1 */}
          <TouchableOpacity
            style={styles.imageContainer}
            onPress={() => selectImage(1)}>
            {image1 ? (
              <Image source={{uri: image1.uri}} style={styles.imagePreview} />
            ) : (
              <View style={styles.imagePlaceholder}>
                <Text style={styles.placeholderIcon}>📷</Text>
                <Text style={styles.placeholderText}>Image 1 (Required)</Text>
                <Text style={styles.placeholderHint}>Tap to select</Text>
              </View>
            )}
          </TouchableOpacity>

          {/* Image 2 */}
          <TouchableOpacity
            style={styles.imageContainer}
            onPress={() => selectImage(2)}>
            {image2 ? (
              <Image source={{uri: image2.uri}} style={styles.imagePreview} />
            ) : (
              <View style={[styles.imagePlaceholder, styles.optionalImage]}>
                <Text style={styles.placeholderIcon}>📷</Text>
                <Text style={styles.placeholderText}>Image 2 (Optional)</Text>
                <Text style={styles.placeholderHint}>Opposite side of tree</Text>
              </View>
            )}
          </TouchableOpacity>
        </View>

        {/* Action Buttons */}
        <View style={styles.buttonRow}>
          {(image1 || image2) && (
            <TouchableOpacity style={styles.clearButton} onPress={clearImages}>
              <Text style={styles.clearButtonText}>🗑️ Clear</Text>
            </TouchableOpacity>
          )}

          <TouchableOpacity
            style={[
              styles.analyzeButton,
              (!image1 || loading) && styles.buttonDisabled,
            ]}
            onPress={analyzeImages}
            disabled={!image1 || loading}>
            {loading ? (
              <View style={styles.loadingContainer}>
                <ActivityIndicator color="#fff" size="small" />
                <Text style={styles.buttonText}>Counting...</Text>
              </View>
            ) : (
              <Text style={styles.buttonText}>🥥 Estimate Yield</Text>
            )}
          </TouchableOpacity>
        </View>

        {/* Results */}
        {result && (
          <View style={styles.resultContainer}>
            {/* Yield Category Header */}
            <View style={[
              styles.categoryHeader,
              {backgroundColor: getYieldCategoryColor(result.yieldCategory)}
            ]}>
              <Text style={styles.categoryIcon}>
                {getYieldCategoryIcon(result.yieldCategory)}
              </Text>
              <Text style={styles.categoryText}>
                {result.yieldCategory.toUpperCase()} YIELD
              </Text>
            </View>

            {/* Nut Count */}
            <View style={styles.countContainer}>
              <View style={styles.countBox}>
                <Text style={styles.countNumber}>{result.totalNutCount}</Text>
                <Text style={styles.countLabel}>Nuts Detected</Text>
              </View>
              <View style={styles.countDivider} />
              <View style={styles.countBox}>
                <Text style={styles.countNumber}>{result.estimatedTotal}</Text>
                <Text style={styles.countLabel}>Estimated Total</Text>
              </View>
            </View>

            {/* Estimation Note */}
            <View style={styles.noteContainer}>
              <Text style={styles.noteText}>ℹ️ {result.estimationNote}</Text>
            </View>

            {/* Message */}
            <View style={styles.messageContainer}>
              <Text style={styles.messageTitle}>Assessment</Text>
              <Text style={styles.messageText}>{result.message}</Text>
            </View>

            {/* Recommendation */}
            <View style={styles.recommendationContainer}>
              <Text style={styles.recommendationTitle}>💡 Recommendation</Text>
              <Text style={styles.recommendationText}>{result.recommendation}</Text>
            </View>

            {/* Per-Image Results */}
            {result.results && result.results.length > 0 && (
              <View style={styles.detailsContainer}>
                <Text style={styles.detailsTitle}>Detection Details</Text>
                {result.results.map((imgResult, index) => (
                  <View key={index} style={styles.imageResultRow}>
                    <Text style={styles.imageResultLabel}>
                      📷 Image {index + 1}:
                    </Text>
                    <Text style={styles.imageResultValue}>
                      {imgResult.nut_count} nuts ({(imgResult.average_confidence * 100).toFixed(0)}% conf)
                    </Text>
                  </View>
                ))}
              </View>
            )}

            {/* Model Info */}
            <View style={styles.modelInfoContainer}>
              <Text style={styles.modelInfoText}>
                Model: {result.modelInfo?.name} • {result.modelInfo?.format}
              </Text>
            </View>

            {/* Scan Another */}
            <TouchableOpacity style={styles.scanAnotherButton} onPress={clearImages}>
              <Text style={styles.scanAnotherText}>🔄 Scan Another Tree</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Info Section */}
        <View style={styles.infoSection}>
          <Text style={styles.infoTitle}>About Yield Estimation</Text>
          <Text style={styles.infoText}>
            This AI-powered tool counts individual coconut nuts visible on the tree
            to estimate yield. For best results:
          </Text>
          <View style={styles.infoList}>
            <Text style={styles.infoItem}>• Capture clear images of coconut bunches</Text>
            <Text style={styles.infoItem}>• Include both sides of the tree if possible</Text>
            <Text style={styles.infoItem}>• Ensure good lighting conditions</Text>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
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
    backgroundColor: '#795548',
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
  apiStatusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 12,
  },
  apiOnline: {
    backgroundColor: 'rgba(255,255,255,0.2)',
  },
  apiOffline: {
    backgroundColor: '#F44336',
  },
  apiStatusText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '600',
  },
  scrollContent: {
    padding: 15,
    paddingBottom: 30,
  },
  treeInfoCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#EFEBE9',
    padding: 12,
    borderRadius: 10,
    marginBottom: 15,
  },
  treeInfoIcon: {
    fontSize: 24,
    marginRight: 10,
  },
  treeInfoText: {
    fontSize: 14,
    color: '#5D4037',
    fontWeight: '600',
  },
  instructionCard: {
    backgroundColor: '#FFF8E1',
    padding: 15,
    borderRadius: 12,
    marginBottom: 15,
  },
  instructionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#F57C00',
    marginBottom: 8,
  },
  instructionText: {
    fontSize: 13,
    color: '#333',
    lineHeight: 20,
  },
  imageSection: {
    marginBottom: 15,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  imageContainer: {
    marginBottom: 10,
  },
  imagePreview: {
    width: '100%',
    height: 180,
    borderRadius: 12,
  },
  imagePlaceholder: {
    width: '100%',
    height: 120,
    backgroundColor: '#fff',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: '#795548',
    borderStyle: 'dashed',
    justifyContent: 'center',
    alignItems: 'center',
  },
  optionalImage: {
    borderColor: '#BDBDBD',
    height: 100,
  },
  placeholderIcon: {
    fontSize: 30,
    marginBottom: 5,
  },
  placeholderText: {
    fontSize: 14,
    color: '#666',
    fontWeight: '600',
  },
  placeholderHint: {
    fontSize: 12,
    color: '#999',
    marginTop: 3,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 15,
  },
  clearButton: {
    backgroundColor: '#FFEBEE',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderRadius: 10,
  },
  clearButtonText: {
    color: '#D32F2F',
    fontWeight: 'bold',
  },
  analyzeButton: {
    flex: 1,
    backgroundColor: '#795548',
    paddingVertical: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  buttonDisabled: {
    backgroundColor: '#BCAAA4',
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  resultContainer: {
    backgroundColor: '#fff',
    borderRadius: 15,
    overflow: 'hidden',
    marginBottom: 15,
  },
  categoryHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    gap: 10,
  },
  categoryIcon: {
    fontSize: 28,
  },
  categoryText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
  },
  countContainer: {
    flexDirection: 'row',
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#f0f0f0',
  },
  countBox: {
    flex: 1,
    alignItems: 'center',
  },
  countDivider: {
    width: 1,
    backgroundColor: '#e0e0e0',
  },
  countNumber: {
    fontSize: 40,
    fontWeight: 'bold',
    color: '#795548',
  },
  countLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 5,
  },
  noteContainer: {
    backgroundColor: '#F5F5F5',
    padding: 10,
    marginHorizontal: 15,
    marginTop: 10,
    borderRadius: 8,
  },
  noteText: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
  messageContainer: {
    padding: 15,
  },
  messageTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 5,
  },
  messageText: {
    fontSize: 14,
    color: '#555',
    lineHeight: 20,
  },
  recommendationContainer: {
    backgroundColor: '#FFF8E1',
    padding: 15,
    marginHorizontal: 15,
    borderRadius: 10,
  },
  recommendationTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#F57C00',
    marginBottom: 5,
  },
  recommendationText: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  detailsContainer: {
    padding: 15,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
    marginTop: 10,
  },
  detailsTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  imageResultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 5,
  },
  imageResultLabel: {
    fontSize: 13,
    color: '#666',
  },
  imageResultValue: {
    fontSize: 13,
    fontWeight: '600',
    color: '#795548',
  },
  modelInfoContainer: {
    alignItems: 'center',
    padding: 10,
    borderTopWidth: 1,
    borderTopColor: '#f0f0f0',
  },
  modelInfoText: {
    fontSize: 11,
    color: '#999',
  },
  scanAnotherButton: {
    backgroundColor: '#795548',
    margin: 15,
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  scanAnotherText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  infoSection: {
    backgroundColor: '#fff',
    borderRadius: 15,
    padding: 20,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  infoText: {
    fontSize: 13,
    color: '#666',
    lineHeight: 20,
    marginBottom: 10,
  },
  infoList: {
    marginTop: 5,
  },
  infoItem: {
    fontSize: 13,
    color: '#555',
    marginBottom: 5,
  },
});
