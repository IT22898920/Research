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
import {detectBranchHealth} from '../services/pestDetectionApi';

export default function BranchHealthScreen({navigation}) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    checkApi();
  }, []);

  const checkApi = async () => {
    try {
      const response = await fetch('http://10.0.2.2:5001/health');
      const data = await response.json();
      setApiStatus(
        data.status === 'healthy' && data.models.branch_health
          ? 'online'
          : 'offline',
      );
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
      const apiResult = await detectBranchHealth(selectedImage.uri);

      if (apiResult.success) {
        setResult(apiResult);
      } else {
        Alert.alert('Error', apiResult.error || 'Failed to analyze image');
      }
    } catch (error) {
      console.error('Analysis error:', error);
      Alert.alert('Error', 'Failed to connect to ML API server');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setResult(null);
  };

  const getHealthColor = () => {
    if (!result) return '#757575';
    return result.isHealthy ? '#4CAF50' : '#F44336';
  };

  const getHealthIcon = () => {
    if (!result) return '🌿';
    return result.isHealthy ? '✅' : '⚠️';
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>🌳 Branch Health Monitor</Text>
          <Text style={styles.subtitle}>
            Check if your coconut tree branch is healthy or unhealthy
          </Text>
          <View style={styles.apiStatusContainer}>
            <View
              style={[
                styles.statusDot,
                {
                  backgroundColor:
                    apiStatus === 'online'
                      ? '#4CAF50'
                      : apiStatus === 'offline'
                      ? '#F44336'
                      : '#FFC107',
                },
              ]}
            />
            <Text style={styles.apiStatusText}>
              API: {apiStatus.toUpperCase()}
            </Text>
          </View>
        </View>

        {/* Image Selection Buttons */}
        {!selectedImage && (
          <View style={styles.actionContainer}>
            <TouchableOpacity
              style={styles.actionButton}
              onPress={handleTakePhoto}>
              <Text style={styles.actionButtonIcon}>📷</Text>
              <Text style={styles.actionButtonText}>Take Photo</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.actionButton}
              onPress={handleChooseFromGallery}>
              <Text style={styles.actionButtonIcon}>🖼️</Text>
              <Text style={styles.actionButtonText}>Choose from Gallery</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Selected Image Preview */}
        {selectedImage && (
          <View style={styles.imageContainer}>
            <Image source={{uri: selectedImage.uri}} style={styles.image} />
            <TouchableOpacity style={styles.resetButton} onPress={handleReset}>
              <Text style={styles.resetButtonText}>✕ Clear</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Analyze Button */}
        {selectedImage && !result && (
          <TouchableOpacity
            style={[
              styles.analyzeButton,
              apiStatus !== 'online' && styles.analyzeButtonDisabled,
            ]}
            onPress={handleAnalyze}
            disabled={isAnalyzing || apiStatus !== 'online'}>
            {isAnalyzing ? (
              <ActivityIndicator color="#FFF" size="small" />
            ) : (
              <>
                <Text style={styles.analyzeButtonIcon}>🔍</Text>
                <Text style={styles.analyzeButtonText}>Analyze Branch</Text>
              </>
            )}
          </TouchableOpacity>
        )}

        {/* Results */}
        {result && (
          <View style={styles.resultsContainer}>
            {/* Status Card */}
            <View
              style={[
                styles.statusCard,
                {borderLeftColor: getHealthColor(), borderLeftWidth: 6},
              ]}>
              <View style={styles.statusHeader}>
                <Text style={styles.statusIcon}>{getHealthIcon()}</Text>
                <View style={styles.statusTextContainer}>
                  <Text style={styles.statusTitle}>
                    {result.isHealthy ? 'Healthy Branch' : 'Unhealthy Branch'}
                  </Text>
                  <Text style={styles.statusSubtitle}>
                    {(result.confidence * 100).toFixed(1)}% confident
                  </Text>
                </View>
              </View>

              {/* Unhealthy Percentage */}
              {!result.isHealthy && result.unhealthyPercentage > 0 && (
                <View style={styles.percentageContainer}>
                  <Text style={styles.percentageLabel}>
                    Unhealthy Percentage:
                  </Text>
                  <View style={styles.percentageBar}>
                    <View
                      style={[
                        styles.percentageFill,
                        {
                          width: `${result.unhealthyPercentage}%`,
                          backgroundColor:
                            result.unhealthyPercentage > 70
                              ? '#F44336'
                              : result.unhealthyPercentage > 40
                              ? '#FF9800'
                              : '#FFC107',
                        },
                      ]}
                    />
                  </View>
                  <Text style={styles.percentageValue}>
                    {result.unhealthyPercentage}%
                  </Text>
                </View>
              )}

              {/* Message */}
              <View style={styles.messageContainer}>
                <Text style={styles.messageTitle}>📋 Analysis:</Text>
                <Text style={styles.messageText}>{result.message}</Text>
              </View>

              {/* Recommendation */}
              <View style={styles.recommendationContainer}>
                <Text style={styles.recommendationTitle}>
                  💡 Recommendation:
                </Text>
                <Text style={styles.recommendationText}>
                  {result.recommendation}
                </Text>
              </View>
            </View>

            {/* Probabilities Card */}
            <View style={styles.probabilitiesCard}>
              <Text style={styles.probabilitiesTitle}>
                📊 Detection Probabilities
              </Text>

              <View style={styles.probabilityRow}>
                <Text style={styles.probabilityLabel}>✅ Healthy:</Text>
                <View style={styles.probabilityBarContainer}>
                  <View
                    style={[
                      styles.probabilityBar,
                      {
                        width: `${result.probabilities.healthy * 100}%`,
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
                <Text style={styles.probabilityLabel}>⚠️ Unhealthy:</Text>
                <View style={styles.probabilityBarContainer}>
                  <View
                    style={[
                      styles.probabilityBar,
                      {
                        width: `${result.probabilities.unhealthy * 100}%`,
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
            <View style={styles.modelInfoCard}>
              <Text style={styles.modelInfoTitle}>🤖 Model Information</Text>
              <Text style={styles.modelInfoText}>
                Version: {result.modelInfo?.version || 'v1'}
              </Text>
              <Text style={styles.modelInfoText}>
                Accuracy: {result.modelInfo?.accuracy || '99.63%'}
              </Text>
              <Text style={styles.modelInfoText}>
                Architecture: MobileNetV2 with Focal Loss
              </Text>
            </View>

            {/* Action Buttons */}
            <View style={styles.actionButtonsContainer}>
              <TouchableOpacity
                style={styles.secondaryButton}
                onPress={handleReset}>
                <Text style={styles.secondaryButtonText}>
                  🔄 Analyze Another
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Info Section */}
        {!selectedImage && !result && (
          <View style={styles.infoSection}>
            <Text style={styles.infoTitle}>ℹ️ How to Use</Text>
            <Text style={styles.infoText}>
              1. Take a photo or choose from gallery
            </Text>
            <Text style={styles.infoText}>
              2. Make sure the branch is clearly visible
            </Text>
            <Text style={styles.infoText}>
              3. Tap "Analyze Branch" to get results
            </Text>
            <Text style={styles.infoText}>
              4. View health status and recommendations
            </Text>

            <View style={styles.tipsContainer}>
              <Text style={styles.tipsTitle}>💡 Tips for Best Results:</Text>
              <Text style={styles.tipText}>
                • Take photos in good lighting
              </Text>
              <Text style={styles.tipText}>
                • Focus on a single branch
              </Text>
              <Text style={styles.tipText}>• Avoid blurry images</Text>
              <Text style={styles.tipText}>
                • Capture the full branch if possible
              </Text>
            </View>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  scrollContent: {
    padding: 16,
  },
  header: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 12,
    marginBottom: 16,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#2E7D32',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    marginBottom: 12,
  },
  apiStatusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  statusDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  apiStatusText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#666',
  },
  actionContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  actionButton: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 12,
    alignItems: 'center',
    marginHorizontal: 4,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  actionButtonIcon: {
    fontSize: 40,
    marginBottom: 8,
  },
  actionButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  imageContainer: {
    position: 'relative',
    marginBottom: 16,
  },
  image: {
    width: '100%',
    height: 300,
    borderRadius: 12,
    resizeMode: 'cover',
  },
  resetButton: {
    position: 'absolute',
    top: 12,
    right: 12,
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
  },
  resetButtonText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 14,
  },
  analyzeButton: {
    backgroundColor: '#2E7D32',
    padding: 16,
    borderRadius: 12,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
    elevation: 3,
  },
  analyzeButtonDisabled: {
    backgroundColor: '#BDBDBD',
  },
  analyzeButtonIcon: {
    fontSize: 20,
    marginRight: 8,
  },
  analyzeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  resultsContainer: {
    marginTop: 8,
  },
  statusCard: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 12,
    marginBottom: 16,
    elevation: 2,
  },
  statusHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  statusIcon: {
    fontSize: 48,
    marginRight: 16,
  },
  statusTextContainer: {
    flex: 1,
  },
  statusTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  statusSubtitle: {
    fontSize: 14,
    color: '#666',
  },
  percentageContainer: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#E0E0E0',
  },
  percentageLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
    marginBottom: 8,
  },
  percentageBar: {
    height: 24,
    backgroundColor: '#E0E0E0',
    borderRadius: 12,
    overflow: 'hidden',
    marginBottom: 8,
  },
  percentageFill: {
    height: '100%',
    borderRadius: 12,
  },
  percentageValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#F44336',
    textAlign: 'center',
  },
  messageContainer: {
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#E0E0E0',
  },
  messageTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  messageText: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
  recommendationContainer: {
    marginTop: 16,
    padding: 12,
    backgroundColor: '#E8F5E9',
    borderRadius: 8,
  },
  recommendationTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2E7D32',
    marginBottom: 8,
  },
  recommendationText: {
    fontSize: 14,
    color: '#1B5E20',
    lineHeight: 20,
  },
  probabilitiesCard: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 12,
    marginBottom: 16,
    elevation: 2,
  },
  probabilitiesTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
  },
  probabilityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  probabilityLabel: {
    width: 100,
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
  },
  probabilityBarContainer: {
    flex: 1,
    height: 20,
    backgroundColor: '#E0E0E0',
    borderRadius: 10,
    overflow: 'hidden',
    marginHorizontal: 8,
  },
  probabilityBar: {
    height: '100%',
    borderRadius: 10,
  },
  probabilityValue: {
    width: 60,
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'right',
  },
  modelInfoCard: {
    backgroundColor: '#F5F5F5',
    padding: 16,
    borderRadius: 12,
    marginBottom: 16,
  },
  modelInfoTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#666',
    marginBottom: 8,
  },
  modelInfoText: {
    fontSize: 12,
    color: '#666',
    marginBottom: 4,
  },
  actionButtonsContainer: {
    marginTop: 8,
  },
  secondaryButton: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#2E7D32',
  },
  secondaryButtonText: {
    color: '#2E7D32',
    fontSize: 16,
    fontWeight: 'bold',
  },
  infoSection: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 12,
    marginTop: 8,
  },
  infoTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
  },
  infoText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 8,
    paddingLeft: 8,
  },
  tipsContainer: {
    marginTop: 20,
    padding: 16,
    backgroundColor: '#E3F2FD',
    borderRadius: 8,
  },
  tipsTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#1976D2',
    marginBottom: 12,
  },
  tipText: {
    fontSize: 13,
    color: '#1565C0',
    marginBottom: 6,
    paddingLeft: 8,
  },
});
