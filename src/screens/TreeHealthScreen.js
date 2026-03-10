/**
 * Tree Health Screen
 * Analyzes overall coconut tree health using coconut_tree_health_v1 model
 * Model: MobileNetV2, 2-class (healthy/unhealthy), 99.72% accuracy
 */

import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
  ScrollView,
  SafeAreaView,
} from 'react-native';
import {launchImageLibrary, launchCamera} from 'react-native-image-picker';
import {treeAPI} from '../services/treeApi';

// For Real Device via USB (adb reverse)
const API_BASE_URL = 'http://localhost:5001';

export default function TreeHealthScreen({navigation, route}) {
  // Get tree info from navigation params (if scanning a specific tree)
  const treeId = route?.params?.treeId;
  const treeLabel = route?.params?.treeLabel;

  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    checkApi();
  }, []);

  // Save scan result to tree record (for map integration)
  const saveToTreeRecord = async (scanResult) => {
    if (!treeId) return;

    try {
      let status = scanResult.is_healthy ? 'healthy' : 'unhealthy';

      // Save detected issues
      const detectedIssues = [];
      if (!scanResult.is_healthy) {
        detectedIssues.push(`Unhealthy tree (${scanResult.unhealthy_percentage || 0}% affected)`);
      }

      await treeAPI.updateScanResults(treeId, {
        detectedIssues: detectedIssues,
        healthStatus: status,
      });

      // Also add to health history
      const treeScanData = {
        status: status,
        scanType: 'tree_health',
        details: {
          prediction: scanResult.prediction,
          confidence: scanResult.confidence,
          probabilities: scanResult.probabilities,
          unhealthyPercentage: scanResult.unhealthy_percentage,
          message: scanResult.message,
          recommendation: scanResult.recommendation,
        },
      };

      await treeAPI.addHealthScan(treeId, treeScanData);
      console.log('Tree health scan saved to tree record:', treeId);
    } catch (error) {
      console.error('Error saving to tree record:', error);
    }
  };

  const checkApi = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      setApiStatus(data.status === 'healthy' ? 'online' : 'offline');
    } catch (error) {
      setApiStatus('offline');
    }
  };

  const handleTakePhoto = () => {
    launchCamera(
      {
        mediaType: 'photo',
        quality: 0.8,
        saveToPhotos: false,
      },
      response => {
        if (response.didCancel) {
          console.log('User cancelled camera');
        } else if (response.errorCode) {
          Alert.alert('Error', 'Failed to capture image');
        } else if (response.assets && response.assets[0]) {
          setSelectedImage(response.assets[0]);
          setResult(null);
        }
      },
    );
  };

  const handleChooseFromGallery = () => {
    launchImageLibrary(
      {
        mediaType: 'photo',
        quality: 0.8,
      },
      response => {
        if (response.didCancel) {
          console.log('User cancelled image picker');
        } else if (response.errorCode) {
          Alert.alert('Error', 'Failed to pick image');
        } else if (response.assets && response.assets[0]) {
          setSelectedImage(response.assets[0]);
          setResult(null);
        }
      },
    );
  };

  const handleAnalyze = async () => {
    if (!selectedImage) {
      Alert.alert('No Image', 'Please select or capture an image first');
      return;
    }

    if (apiStatus !== 'online') {
      Alert.alert(
        'API Offline',
        'The ML API server is offline. Please make sure the Flask server is running.',
      );
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('image', {
        uri: selectedImage.uri,
        type: selectedImage.type || 'image/jpeg',
        name: selectedImage.fileName || 'image.jpg',
      });

      const response = await fetch(`${API_BASE_URL}/predict/tree-health`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
        // Save to tree record if treeId exists
        await saveToTreeRecord(data);
      } else {
        Alert.alert('Error', data.error || 'Analysis failed');
      }
    } catch (error) {
      console.error('API Error:', error);
      Alert.alert('Error', 'Failed to analyze image. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setResult(null);
  };

  const getStatusStyle = () => {
    if (!result) return {};
    return result.is_healthy
      ? {backgroundColor: '#E8F5E9', borderColor: '#4CAF50'}
      : {backgroundColor: '#FFEBEE', borderColor: '#F44336'};
  };

  const getStatusTextStyle = () => {
    if (!result) return {};
    return result.is_healthy ? {color: '#2E7D32'} : {color: '#C62828'};
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
        <Text style={styles.headerTitle}>Tree Health</Text>
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

        {/* Image Preview */}
        <View style={styles.imageSection}>
          {selectedImage ? (
            <View style={styles.imagePreviewContainer}>
              <Image
                source={{uri: selectedImage.uri}}
                style={styles.imagePreview}
                resizeMode="cover"
              />
              <TouchableOpacity
                style={styles.removeImageButton}
                onPress={handleReset}>
                <Text style={styles.removeImageText}>✕</Text>
              </TouchableOpacity>
            </View>
          ) : (
            <View style={styles.imagePlaceholder}>
              <Text style={styles.imagePlaceholderIcon}>🌴</Text>
              <Text style={styles.imagePlaceholderText}>
                Take or select a photo of a coconut tree
              </Text>
              <Text style={styles.imagePlaceholderHint}>
                For best results, capture the full tree or trunk
              </Text>
            </View>
          )}
        </View>

        {/* Image Selection Buttons */}
        {!selectedImage && (
          <View style={styles.buttonRow}>
            <TouchableOpacity
              style={[styles.selectButton, styles.cameraButton]}
              onPress={handleTakePhoto}>
              <Text style={styles.selectButtonIcon}>📷</Text>
              <Text style={styles.selectButtonText}>Take Photo</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.selectButton, styles.galleryButton]}
              onPress={handleChooseFromGallery}>
              <Text style={styles.selectButtonIcon}>🖼️</Text>
              <Text style={styles.selectButtonText}>Gallery</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Analyze Button */}
        {selectedImage && !result && (
          <TouchableOpacity
            style={[
              styles.analyzeButton,
              isAnalyzing && styles.analyzeButtonDisabled,
            ]}
            onPress={handleAnalyze}
            disabled={isAnalyzing}>
            {isAnalyzing ? (
              <View style={styles.analyzingContainer}>
                <ActivityIndicator color="#fff" size="small" />
                <Text style={styles.analyzeButtonText}>Analyzing...</Text>
              </View>
            ) : (
              <Text style={styles.analyzeButtonText}>🔬 Analyze Tree Health</Text>
            )}
          </TouchableOpacity>
        )}

        {/* Results Section */}
        {result && (
          <View style={[styles.resultContainer, getStatusStyle()]}>
            {/* Status Header */}
            <View style={styles.resultHeader}>
              <Text style={styles.resultIcon}>
                {result.is_healthy ? '✅' : '⚠️'}
              </Text>
              <Text style={[styles.resultTitle, getStatusTextStyle()]}>
                {result.label}
              </Text>
            </View>

            {/* Confidence */}
            <View style={styles.confidenceContainer}>
              <Text style={styles.confidenceLabel}>Confidence:</Text>
              <View style={styles.confidenceBar}>
                <View
                  style={[
                    styles.confidenceFill,
                    {
                      width: `${(result.confidence * 100).toFixed(0)}%`,
                      backgroundColor: result.is_healthy ? '#4CAF50' : '#F44336',
                    },
                  ]}
                />
              </View>
              <Text style={styles.confidenceValue}>
                {(result.confidence * 100).toFixed(1)}%
              </Text>
            </View>

            {/* Unhealthy Percentage (if unhealthy) */}
            {!result.is_healthy && (
              <View style={styles.unhealthyPercentageContainer}>
                <Text style={styles.unhealthyPercentageLabel}>
                  Health Issues Detected:
                </Text>
                <Text style={styles.unhealthyPercentageValue}>
                  {result.unhealthy_percentage.toFixed(1)}%
                </Text>
              </View>
            )}

            {/* Message */}
            <View style={styles.messageContainer}>
              <Text style={styles.messageTitle}>Assessment:</Text>
              <Text style={styles.messageText}>{result.message}</Text>
            </View>

            {/* Recommendation */}
            <View style={styles.recommendationContainer}>
              <Text style={styles.recommendationTitle}>💡 Recommendation:</Text>
              <Text style={styles.recommendationText}>
                {result.recommendation}
              </Text>
            </View>

            {/* Probabilities */}
            <View style={styles.probabilitiesContainer}>
              <Text style={styles.probabilitiesTitle}>Detailed Analysis:</Text>
              <View style={styles.probabilityRow}>
                <Text style={styles.probabilityLabel}>Healthy:</Text>
                <View style={styles.probabilityBarContainer}>
                  <View
                    style={[
                      styles.probabilityBar,
                      {
                        width: `${(result.probabilities.healthy * 100).toFixed(0)}%`,
                        backgroundColor: '#4CAF50',
                      },
                    ]}
                  />
                </View>
                <Text style={styles.probabilityValue}>
                  {(result.probabilities.healthy * 100).toFixed(1)}%
                </Text>
              </View>
              <View style={styles.probabilityRow}>
                <Text style={styles.probabilityLabel}>Unhealthy:</Text>
                <View style={styles.probabilityBarContainer}>
                  <View
                    style={[
                      styles.probabilityBar,
                      {
                        width: `${(result.probabilities.unhealthy * 100).toFixed(0)}%`,
                        backgroundColor: '#F44336',
                      },
                    ]}
                  />
                </View>
                <Text style={styles.probabilityValue}>
                  {(result.probabilities.unhealthy * 100).toFixed(1)}%
                </Text>
              </View>
            </View>

            {/* Model Info */}
            <View style={styles.modelInfoContainer}>
              <Text style={styles.modelInfoText}>
                Model: {result.model_info?.name} ({result.model_info?.version}) •
                Accuracy: {result.model_info?.accuracy}
              </Text>
            </View>

            {/* Scan Another Button */}
            <TouchableOpacity
              style={styles.scanAnotherButton}
              onPress={handleReset}>
              <Text style={styles.scanAnotherButtonText}>🔄 Scan Another Tree</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Info Section */}
        <View style={styles.infoSection}>
          <Text style={styles.infoTitle}>About This Feature</Text>
          <Text style={styles.infoText}>
            This AI-powered tool analyzes images of coconut trees to determine
            their overall health status. The model can detect:
          </Text>
          <View style={styles.infoList}>
            <Text style={styles.infoItem}>✅ Healthy coconut trees</Text>
            <Text style={styles.infoItem}>⚠️ Unhealthy trees (diseases, pests, nutrient deficiency)</Text>
          </View>
          <Text style={styles.infoAccuracy}>
            Model Accuracy: 99.72% • Classes: healthy, unhealthy
          </Text>
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
    paddingVertical: 15,
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
    backgroundColor: '#E8F5E9',
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
    color: '#2E7D32',
    fontWeight: '600',
  },
  imageSection: {
    marginBottom: 15,
  },
  imagePreviewContainer: {
    position: 'relative',
    borderRadius: 15,
    overflow: 'hidden',
  },
  imagePreview: {
    width: '100%',
    height: 300,
    borderRadius: 15,
  },
  removeImageButton: {
    position: 'absolute',
    top: 10,
    right: 10,
    backgroundColor: 'rgba(0,0,0,0.6)',
    width: 30,
    height: 30,
    borderRadius: 15,
    justifyContent: 'center',
    alignItems: 'center',
  },
  removeImageText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  imagePlaceholder: {
    backgroundColor: '#fff',
    borderRadius: 15,
    height: 250,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#e0e0e0',
    borderStyle: 'dashed',
  },
  imagePlaceholderIcon: {
    fontSize: 60,
    marginBottom: 15,
  },
  imagePlaceholderText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
  },
  imagePlaceholderHint: {
    fontSize: 12,
    color: '#999',
    marginTop: 8,
    textAlign: 'center',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 15,
  },
  selectButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 12,
    gap: 8,
  },
  cameraButton: {
    backgroundColor: '#2e7d32',
  },
  galleryButton: {
    backgroundColor: '#1976D2',
  },
  selectButtonIcon: {
    fontSize: 20,
  },
  selectButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
  analyzeButton: {
    backgroundColor: '#2e7d32',
    padding: 18,
    borderRadius: 12,
    alignItems: 'center',
    marginBottom: 15,
  },
  analyzeButtonDisabled: {
    backgroundColor: '#81C784',
  },
  analyzingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  analyzeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  resultContainer: {
    backgroundColor: '#fff',
    borderRadius: 15,
    padding: 20,
    marginBottom: 15,
    borderWidth: 2,
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  resultIcon: {
    fontSize: 40,
    marginRight: 15,
  },
  resultTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    flex: 1,
  },
  confidenceContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  confidenceLabel: {
    fontSize: 14,
    color: '#666',
    width: 85,
  },
  confidenceBar: {
    flex: 1,
    height: 10,
    backgroundColor: '#E0E0E0',
    borderRadius: 5,
    marginHorizontal: 10,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 5,
  },
  confidenceValue: {
    fontSize: 14,
    fontWeight: 'bold',
    width: 50,
    textAlign: 'right',
  },
  unhealthyPercentageContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#FFEBEE',
    padding: 12,
    borderRadius: 8,
    marginBottom: 15,
  },
  unhealthyPercentageLabel: {
    fontSize: 14,
    color: '#C62828',
    fontWeight: '500',
  },
  unhealthyPercentageValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#C62828',
  },
  messageContainer: {
    marginBottom: 15,
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
    borderRadius: 10,
    marginBottom: 15,
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
  probabilitiesContainer: {
    backgroundColor: '#F5F5F5',
    padding: 15,
    borderRadius: 10,
    marginBottom: 15,
  },
  probabilitiesTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  probabilityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  probabilityLabel: {
    fontSize: 13,
    color: '#666',
    width: 75,
  },
  probabilityBarContainer: {
    flex: 1,
    height: 8,
    backgroundColor: '#E0E0E0',
    borderRadius: 4,
    marginHorizontal: 8,
    overflow: 'hidden',
  },
  probabilityBar: {
    height: '100%',
    borderRadius: 4,
  },
  probabilityValue: {
    fontSize: 13,
    fontWeight: '600',
    width: 45,
    textAlign: 'right',
  },
  modelInfoContainer: {
    alignItems: 'center',
    marginBottom: 15,
  },
  modelInfoText: {
    fontSize: 11,
    color: '#999',
  },
  scanAnotherButton: {
    backgroundColor: '#2e7d32',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  scanAnotherButtonText: {
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
    marginBottom: 10,
  },
  infoItem: {
    fontSize: 13,
    color: '#555',
    marginBottom: 5,
  },
  infoAccuracy: {
    fontSize: 12,
    color: '#2e7d32',
    fontWeight: '600',
    textAlign: 'center',
    marginTop: 10,
  },
});
