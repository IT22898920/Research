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
  Modal,
  TextInput,
} from 'react-native';
import {launchImageLibrary, launchCamera} from 'react-native-image-picker';
import {detectBranchHealth} from '../services/pestDetectionApi';
import {ML_API_URL, fetchWithTimeout} from '../config/apiConfig';
import {treeAPI} from '../services/treeApi';
import {
  getHealthTreatmentRecommendations,
  isApiKeyConfigured,
  setApiKey,
} from '../services/treatmentApi';

export default function BranchHealthScreen({navigation, route}) {
  // Get tree info from navigation params (if scanning a specific tree)
  const treeId = route?.params?.treeId;
  const treeLabel = route?.params?.treeLabel;

  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  // Treatment states
  const [treatment, setTreatment] = useState(null);
  const [isLoadingTreatment, setIsLoadingTreatment] = useState(false);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [hasApiKey, setHasApiKey] = useState(false);

  useEffect(() => {
    checkApi();
    checkApiKeyStatus();
  }, []);

  const checkApiKeyStatus = async () => {
    const configured = await isApiKeyConfigured();
    setHasApiKey(configured);
  };

  // Get treatment recommendations
  const handleGetTreatment = async () => {
    if (!result || result.isHealthy) return;

    if (!hasApiKey) {
      setShowApiKeyModal(true);
      return;
    }

    setIsLoadingTreatment(true);
    try {
      const severity = result.unhealthyPercentage > 60 ? 'severe' : result.unhealthyPercentage > 30 ? 'moderate' : 'mild';
      const response = await getHealthTreatmentRecommendations({
        conditionType: 'unhealthy_branch',
        severity,
        confidence: result.confidence,
        additionalInfo: {
          unhealthyPercentage: result.unhealthyPercentage,
        },
        language: 'en',
      });

      if (response.success) {
        setTreatment(response.data);
      } else if (response.fallback) {
        setTreatment(response.fallback);
      } else {
        Alert.alert('Error', response.error || 'Failed to get treatment recommendations');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to get treatment recommendations');
    } finally {
      setIsLoadingTreatment(false);
    }
  };

  // Save API key
  const handleSaveApiKey = async () => {
    if (apiKeyInput.trim()) {
      await setApiKey(apiKeyInput.trim());
      setHasApiKey(true);
      setShowApiKeyModal(false);
      setApiKeyInput('');
      handleGetTreatment();
    }
  };

  // Save scan result to tree record (for map integration)
  const saveToTreeRecord = async (scanResult) => {
    if (!treeId) return;

    try {
      // Determine health status
      let status = scanResult.isHealthy ? 'healthy' : 'unhealthy';

      // Save detected issues
      const detectedIssues = [];
      if (!scanResult.isHealthy) {
        detectedIssues.push(`Unhealthy branch (${scanResult.unhealthyPercentage || 0}% damaged)`);
      }

      await treeAPI.updateScanResults(treeId, {
        detectedIssues: detectedIssues,
        healthStatus: status,
      });

      // Also add to health history
      const treeScanData = {
        status: status,
        scanType: 'branch_health',
        details: {
          prediction: scanResult.prediction,
          confidence: scanResult.confidence,
          probabilities: scanResult.probabilities,
          unhealthyPercentage: scanResult.unhealthyPercentage,
          message: scanResult.message,
          recommendation: scanResult.recommendation,
        },
      };

      await treeAPI.addHealthScan(treeId, treeScanData);
      console.log('Branch health scan saved to tree record:', treeId);
    } catch (error) {
      console.error('Error saving to tree record:', error);
    }
  };

  const checkApi = async () => {
    try {
      // Uses centralized config
      const response = await fetchWithTimeout(`${ML_API_URL}/health`, {}, 30000, 2);
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
      const apiResult = await detectBranchHealth(selectedImage.uri);

      if (apiResult.success) {
        setResult(apiResult);
        // Save to tree record if scanning a specific tree
        if (treeId) {
          saveToTreeRecord(apiResult);
        }
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
    setTreatment(null);
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
        {/* Tree Info Banner - Show when scanning a specific tree */}
        {treeId && (
          <View style={styles.treeBanner}>
            <Text style={styles.treeBannerIcon}>🌴</Text>
            <View style={styles.treeBannerInfo}>
              <Text style={styles.treeBannerLabel}>Scanning Tree:</Text>
              <Text style={styles.treeBannerTitle}>{treeLabel || 'Unknown Tree'}</Text>
            </View>
          </View>
        )}

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

            {/* Get Treatment Button - Only for unhealthy */}
            {!result.isHealthy && !treatment && (
              <TouchableOpacity
                style={[styles.treatmentButton, isLoadingTreatment && styles.treatmentButtonDisabled]}
                onPress={handleGetTreatment}
                disabled={isLoadingTreatment}>
                {isLoadingTreatment ? (
                  <View style={styles.loadingRow}>
                    <ActivityIndicator color="#FFF" size="small" />
                    <Text style={styles.treatmentButtonText}>Getting Treatment Plan...</Text>
                  </View>
                ) : (
                  <Text style={styles.treatmentButtonText}>💊 Get AI Treatment Plan</Text>
                )}
              </TouchableOpacity>
            )}

            {/* Treatment Results */}
            {treatment && (
              <View style={styles.treatmentContainer}>
                <Text style={styles.treatmentTitle}>💊 AI Treatment Plan</Text>

                <View style={styles.treatmentCard}>
                  <Text style={styles.treatmentCardTitle}>📋 Summary</Text>
                  <Text style={styles.treatmentText}>{treatment.summary}</Text>
                  <View style={[styles.urgencyBadgeLarge, {
                    backgroundColor: treatment.urgency === 'critical' ? '#d32f2f' :
                      treatment.urgency === 'high' ? '#f44336' :
                      treatment.urgency === 'medium' ? '#ff9800' : '#4caf50'
                  }]}>
                    <Text style={styles.urgencyTextLarge}>
                      Urgency: {treatment.urgency?.toUpperCase()}
                    </Text>
                  </View>
                </View>

                {treatment.diagnosis && (
                  <View style={styles.treatmentCard}>
                    <Text style={styles.treatmentCardTitle}>🔍 Diagnosis</Text>
                    <Text style={styles.treatmentText}>{treatment.diagnosis}</Text>
                  </View>
                )}

                {treatment.treatments && treatment.treatments.length > 0 && (
                  <View style={styles.treatmentCard}>
                    <Text style={styles.treatmentCardTitle}>💉 Treatments</Text>
                    {treatment.treatments.map((t, idx) => (
                      <View key={idx} style={styles.treatmentItem}>
                        <View style={styles.treatmentItemHeader}>
                          <Text style={styles.treatmentItemName}>{t.name}</Text>
                          <View style={[styles.treatmentTypeBadge, {
                            backgroundColor: t.type === 'chemical' ? '#e57373' :
                              t.type === 'organic' ? '#81c784' : '#ffb74d'
                          }]}>
                            <Text style={styles.treatmentTypeText}>{t.type}</Text>
                          </View>
                        </View>
                        <Text style={styles.treatmentItemDesc}>{t.description}</Text>
                        <Text style={styles.treatmentDetail}>📏 {t.dosage}</Text>
                        <Text style={styles.treatmentDetail}>🔄 {t.frequency}</Text>
                        <Text style={styles.treatmentDetail}>💰 {t.cost_estimate}</Text>
                      </View>
                    ))}
                  </View>
                )}

                {treatment.preventive_measures && (
                  <View style={styles.treatmentCard}>
                    <Text style={styles.treatmentCardTitle}>🛡️ Prevention</Text>
                    {treatment.preventive_measures.map((item, idx) => (
                      <Text key={idx} style={styles.listItem}>• {item}</Text>
                    ))}
                  </View>
                )}

                <View style={styles.treatmentCard}>
                  <Text style={styles.treatmentCardTitle}>📅 Expected Recovery</Text>
                  <Text style={styles.treatmentText}>{treatment.expected_recovery}</Text>
                </View>
              </View>
            )}

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

      {/* API Key Modal */}
      <Modal
        visible={showApiKeyModal}
        transparent
        animationType="slide"
        onRequestClose={() => setShowApiKeyModal(false)}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>🔑 Enter Groq API Key</Text>
            <Text style={styles.modalSubtitle}>
              Get your free API key from groq.com for AI treatment recommendations
            </Text>
            <TextInput
              style={styles.apiKeyInput}
              placeholder="gsk_xxxxxxxxxxxx"
              value={apiKeyInput}
              onChangeText={setApiKeyInput}
              autoCapitalize="none"
              secureTextEntry
            
            placeholderTextColor="#999"
          />
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={styles.modalCancelButton}
                onPress={() => setShowApiKeyModal(false)}>
                <Text style={styles.modalCancelText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.modalSaveButton}
                onPress={handleSaveApiKey}>
                <Text style={styles.modalSaveText}>Save</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
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
  treeBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#e8f5e9',
    padding: 12,
    marginBottom: 16,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#4CAF50',
  },
  treeBannerIcon: {
    fontSize: 28,
    marginRight: 12,
  },
  treeBannerInfo: {
    flex: 1,
  },
  treeBannerLabel: {
    fontSize: 11,
    color: '#666',
  },
  treeBannerTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2e7d32',
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
  // Treatment styles
  treatmentButton: {
    backgroundColor: '#7B1FA2',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginVertical: 16,
  },
  treatmentButtonDisabled: {
    backgroundColor: '#CE93D8',
  },
  treatmentButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  treatmentContainer: {
    marginTop: 16,
  },
  treatmentTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#7B1FA2',
    marginBottom: 16,
    textAlign: 'center',
  },
  treatmentCard: {
    backgroundColor: '#FFF',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    elevation: 2,
  },
  treatmentCardTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
  },
  treatmentText: {
    fontSize: 14,
    color: '#555',
    lineHeight: 20,
  },
  urgencyBadgeLarge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    marginTop: 8,
  },
  urgencyTextLarge: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: 'bold',
  },
  treatmentItem: {
    backgroundColor: '#F5F5F5',
    borderRadius: 8,
    padding: 12,
    marginBottom: 10,
  },
  treatmentItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  treatmentItemName: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#333',
    flex: 1,
  },
  treatmentTypeBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 10,
  },
  treatmentTypeText: {
    color: '#FFF',
    fontSize: 10,
    fontWeight: 'bold',
    textTransform: 'uppercase',
  },
  treatmentItemDesc: {
    fontSize: 13,
    color: '#555',
    marginBottom: 8,
  },
  treatmentDetail: {
    fontSize: 12,
    color: '#666',
    marginBottom: 4,
  },
  listItem: {
    fontSize: 13,
    color: '#555',
    marginBottom: 6,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: '#FFF',
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 400,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 8,
    textAlign: 'center',
  },
  modalSubtitle: {
    fontSize: 13,
    color: '#666',
    marginBottom: 16,
    textAlign: 'center',
  },
  apiKeyInput: {
    backgroundColor: '#F5F5F5',
    borderRadius: 8,
    padding: 12,
    fontSize: 14,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#E0E0E0',
  },
  modalButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  modalCancelButton: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#F5F5F5',
    alignItems: 'center',
  },
  modalCancelText: {
    color: '#666',
    fontWeight: '600',
  },
  modalSaveButton: {
    flex: 1,
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#7B1FA2',
    alignItems: 'center',
  },
  modalSaveText: {
    color: '#FFF',
    fontWeight: '600',
  },
});
