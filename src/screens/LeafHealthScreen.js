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
  Animated,
  Modal,
  TextInput,
} from 'react-native';
import {launchImageLibrary, launchCamera} from 'react-native-image-picker';
import {treeAPI} from '../services/treeApi';
import {
  getHealthTreatmentRecommendations,
  isApiKeyConfigured,
  setApiKey,
} from '../services/treatmentApi';

// Import centralized config - CHANGE IP IN ../config/apiConfig.js ONLY!
import {ML_API_URL, fetchWithTimeout} from '../config/apiConfig';
const API_BASE_URL = ML_API_URL;

// Condition Card Component - Always Expanded with All Details
const ConditionCard = ({condition, index}) => {
  const urgencyColors = {
    high: '#FF5252',
    medium: '#FFA726',
    low: '#66BB6A',
  };

  return (
    <View style={styles.conditionCard}>
      {/* Header */}
      <View style={styles.conditionHeader}>
        <View style={styles.conditionHeaderLeft}>
          <Text style={styles.conditionIcon}>{condition.icon}</Text>
          <View style={styles.conditionTitleContainer}>
            <Text style={styles.conditionTitle}>{condition.condition}</Text>
            <View
              style={[
                styles.urgencyBadge,
                {backgroundColor: urgencyColors[condition.urgency]},
              ]}>
              <Text style={styles.urgencyText}>
                {condition.urgency.toUpperCase()}
              </Text>
            </View>
          </View>
        </View>
      </View>

      {/* Content - Always Visible */}
      <View style={styles.conditionContent}>
        {/* Reason */}
        <View style={styles.conditionSection}>
          <Text style={styles.sectionTitle}>🔍 Reason:</Text>
          <Text style={styles.sectionText}>{condition.reason}</Text>
        </View>

        {/* Symptoms */}
        <View style={styles.conditionSection}>
          <Text style={styles.sectionTitle}>⚠️ Symptoms:</Text>
          {condition.symptoms.map((symptom, idx) => (
            <Text key={idx} style={styles.symptomText}>
              • {symptom}
            </Text>
          ))}
        </View>

        {/* Solution */}
        <View style={styles.conditionSection}>
          <Text style={styles.sectionTitle}>💊 Solution:</Text>
          <Text style={styles.solutionText}>{condition.solution}</Text>
        </View>
      </View>
    </View>
  );
};

export default function LeafHealthScreen({navigation, route}) {
  // Get tree info from navigation params (if scanning a specific tree)
  const treeId = route?.params?.treeId;
  const treeLabel = route?.params?.treeLabel;

  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');
  const [imageMode, setImageMode] = useState('phone'); // 'phone' or 'drone'

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
    if (!result || result.is_healthy) return;

    if (!hasApiKey) {
      setShowApiKeyModal(true);
      return;
    }

    setIsLoadingTreatment(true);
    try {
      const severity = result.confidence > 0.8 ? 'severe' : result.confidence > 0.5 ? 'moderate' : 'mild';
      const response = await getHealthTreatmentRecommendations({
        conditionType: 'unhealthy_leaf',
        severity,
        confidence: result.confidence,
        additionalInfo: {
          possibleConditions: result.possible_conditions?.slice(0, 3),
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
      let status = scanResult.is_healthy ? 'healthy' : 'unhealthy';

      // Save detected issues
      const detectedIssues = [];
      if (!scanResult.is_healthy) {
        detectedIssues.push('Unhealthy leaf detected');
        // Add possible conditions as issues
        if (scanResult.possible_conditions && scanResult.possible_conditions.length > 0) {
          scanResult.possible_conditions.slice(0, 3).forEach(condition => {
            detectedIssues.push(condition.condition);
          });
        }
      }

      await treeAPI.updateScanResults(treeId, {
        detectedIssues: detectedIssues,
        healthStatus: status,
      });

      // Also add to health history
      const treeScanData = {
        status: status,
        scanType: 'leaf_health',
        details: {
          prediction: scanResult.prediction,
          confidence: scanResult.confidence,
          probabilities: scanResult.probabilities,
          message: scanResult.message,
          recommendation: scanResult.recommendation,
          possibleConditions: scanResult.possible_conditions,
        },
      };

      await treeAPI.addHealthScan(treeId, treeScanData);
      console.log('Leaf health scan saved to tree record:', treeId);
    } catch (error) {
      console.error('Error saving to tree record:', error);
    }
  };

  const checkApi = async () => {
    try {
      const response = await fetchWithTimeout(`${API_BASE_URL}/health`, {}, 30000, 2);
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
      formData.append('mode', imageMode);

      const response = await fetchWithTimeout(`${API_BASE_URL}/predict/leaf-health`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }, 90000, 1);

      const data = await response.json();

      if (data.success) {
        setResult(data);
        // Save to tree record if scanning a specific tree
        if (treeId) {
          saveToTreeRecord(data);
        }
      } else {
        Alert.alert('Error', data.error || 'Failed to analyze image');
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

  return (
    <SafeAreaView style={styles.container}>
      {/* Top Header Bar */}
      <View style={styles.headerBar}>
        <TouchableOpacity
          onPress={() => navigation.goBack()}
          style={styles.backButton}
          hitSlop={{top: 15, bottom: 15, left: 15, right: 15}}>
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerBarTitle}>Leaf Health</Text>
        <View
          style={[
            styles.apiStatusBadge,
            apiStatus === 'online' ? styles.apiOnline : styles.apiOffline,
          ]}>
          <Text style={styles.apiStatusBadgeText}>
            {apiStatus === 'online' ? 'API Online' : 'API Offline'}
          </Text>
        </View>
      </View>

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

        {/* Info Card */}
        <View style={styles.infoCard}>
          <Text style={styles.infoIcon}>🍃</Text>
          <Text style={styles.infoTitle}>Leaf Health Monitor</Text>
          <Text style={styles.infoSubtitle}>
            Check if your coconut leaf is healthy or unhealthy
          </Text>
        </View>

        {/* Image Mode Selection */}
        <View style={styles.modeSelectionContainer}>
          <Text style={styles.modeSelectionTitle}>Select Image Source</Text>
          <View style={styles.modeButtonsRow}>
            <TouchableOpacity
              style={[
                styles.modeButton,
                imageMode === 'phone' && styles.modeButtonActive,
              ]}
              onPress={() => {
                setImageMode('phone');
                setResult(null);
              }}>
              <Text style={styles.modeButtonIcon}>📱</Text>
              <Text
                style={[
                  styles.modeButtonText,
                  imageMode === 'phone' && styles.modeButtonTextActive,
                ]}>
                Phone Camera
              </Text>
              <Text style={styles.modeButtonHint}>For close-up photos</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.modeButton,
                imageMode === 'drone' && styles.modeButtonActive,
              ]}
              onPress={() => {
                setImageMode('drone');
                setResult(null);
              }}>
              <Text style={styles.modeButtonIcon}>🚁</Text>
              <Text
                style={[
                  styles.modeButtonText,
                  imageMode === 'drone' && styles.modeButtonTextActive,
                ]}>
                Drone Camera
              </Text>
              <Text style={styles.modeButtonHint}>For aerial photos</Text>
            </TouchableOpacity>
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
              isAnalyzing && styles.analyzeButtonDisabled,
            ]}
            onPress={handleAnalyze}
            disabled={isAnalyzing}>
            {isAnalyzing ? (
              <View style={styles.analyzingContainer}>
                <ActivityIndicator color="#FFF" />
                <Text style={styles.analyzeButtonText}>Analyzing...</Text>
              </View>
            ) : (
              <Text style={styles.analyzeButtonText}>🔍 Analyze Leaf</Text>
            )}
          </TouchableOpacity>
        )}

        {/* Results */}
        {result && (
          <View style={styles.resultsContainer}>
            <Text style={styles.resultsTitle}>Analysis Results</Text>

            {/* Health Status */}
            <View
              style={[
                styles.statusCard,
                {
                  backgroundColor: result.is_healthy
                    ? '#E8F5E9'
                    : '#FFEBEE',
                  borderColor: result.is_healthy ? '#4CAF50' : '#F44336',
                },
              ]}>
              <Text style={styles.statusIcon}>
                {result.is_healthy ? '✓' : '⚠'}
              </Text>
              <Text
                style={[
                  styles.statusText,
                  {color: result.is_healthy ? '#2E7D32' : '#C62828'},
                ]}>
                {result.prediction.toUpperCase()}
              </Text>
              <Text style={styles.confidenceText}>
                Confidence: {(result.confidence * 100).toFixed(1)}%
              </Text>
            </View>

            {/* Probabilities */}
            <View style={styles.probabilitiesCard}>
              <Text style={styles.cardTitle}>Detailed Probabilities</Text>

              <View style={styles.probabilityRow}>
                <Text style={styles.probabilityLabel}>Healthy:</Text>
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
                <Text style={styles.probabilityLabel}>Unhealthy:</Text>
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

            {/* Message */}
            <View style={styles.messageCard}>
              <Text style={styles.messageIcon}>💬</Text>
              <Text style={styles.messageText}>{result.message}</Text>
            </View>

            {/* Recommendation */}
            <View style={styles.recommendationCard}>
              <Text style={styles.recommendationTitle}>
                💡 Recommendation
              </Text>
              <Text style={styles.recommendationText}>
                {result.recommendation}
              </Text>
            </View>

            {/* Possible Conditions (only for unhealthy leaves) */}
            {!result.is_healthy && result.possible_conditions && (
              <View style={styles.possibleConditionsContainer}>
                <Text style={styles.possibleConditionsTitle}>
                  🌴 Unhealthy Coconut Tree Solutions ({result.conditions_count}
                  )
                </Text>
                <Text style={styles.possibleConditionsSubtitle}>
                  All possible causes with detailed reasons, symptoms, and
                  treatment solutions
                </Text>

                {result.possible_conditions.map((condition, index) => (
                  <ConditionCard
                    key={index}
                    condition={condition}
                    index={index}
                  />
                ))}

                <View style={styles.noteCard}>
                  <Text style={styles.noteIcon}>ℹ️</Text>
                  <Text style={styles.noteText}>
                    Important: Compare the symptoms above with your coconut
                    tree's actual condition to identify the correct cause.
                    Consult an agricultural expert for accurate diagnosis and
                    treatment.
                  </Text>
                </View>
              </View>
            )}

            {/* Get Treatment Button - Only for unhealthy */}
            {!result.is_healthy && !treatment && (
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

                {/* Summary */}
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

                {/* Diagnosis */}
                {treatment.diagnosis && (
                  <View style={styles.treatmentCard}>
                    <Text style={styles.treatmentCardTitle}>🔍 Diagnosis</Text>
                    <Text style={styles.treatmentText}>{treatment.diagnosis}</Text>
                  </View>
                )}

                {/* Treatments */}
                {treatment.treatments && treatment.treatments.length > 0 && (
                  <View style={styles.treatmentCard}>
                    <Text style={styles.treatmentCardTitle}>💉 Recommended Treatments</Text>
                    {treatment.treatments.map((t, idx) => (
                      <View key={idx} style={styles.treatmentItem}>
                        <View style={styles.treatmentItemHeader}>
                          <Text style={styles.treatmentItemName}>{t.name}</Text>
                          <View style={[styles.treatmentTypeBadge, {
                            backgroundColor: t.type === 'chemical' ? '#e57373' :
                              t.type === 'organic' ? '#81c784' :
                              t.type === 'nutritional' ? '#64b5f6' : '#ffb74d'
                          }]}>
                            <Text style={styles.treatmentTypeText}>{t.type}</Text>
                          </View>
                        </View>
                        <Text style={styles.treatmentItemDesc}>{t.description}</Text>
                        <View style={styles.treatmentDetails}>
                          <Text style={styles.treatmentDetail}>📏 {t.dosage}</Text>
                          <Text style={styles.treatmentDetail}>🔄 {t.frequency}</Text>
                          <Text style={styles.treatmentDetail}>⏱️ {t.duration}</Text>
                          <Text style={styles.treatmentDetail}>💰 {t.cost_estimate}</Text>
                        </View>
                      </View>
                    ))}
                  </View>
                )}

                {/* Nutritional Recommendations */}
                {treatment.nutritional_recommendations && treatment.nutritional_recommendations.length > 0 && (
                  <View style={styles.treatmentCard}>
                    <Text style={styles.treatmentCardTitle}>🌱 Nutritional Recommendations</Text>
                    {treatment.nutritional_recommendations.map((item, idx) => (
                      <Text key={idx} style={styles.listItem}>• {item}</Text>
                    ))}
                  </View>
                )}

                {/* Preventive Measures */}
                {treatment.preventive_measures && treatment.preventive_measures.length > 0 && (
                  <View style={styles.treatmentCard}>
                    <Text style={styles.treatmentCardTitle}>🛡️ Preventive Measures</Text>
                    {treatment.preventive_measures.map((item, idx) => (
                      <Text key={idx} style={styles.listItem}>• {item}</Text>
                    ))}
                  </View>
                )}

                {/* Safety Precautions */}
                {treatment.safety_precautions && treatment.safety_precautions.length > 0 && (
                  <View style={styles.treatmentCard}>
                    <Text style={styles.treatmentCardTitle}>⚠️ Safety Precautions</Text>
                    {treatment.safety_precautions.map((item, idx) => (
                      <Text key={idx} style={styles.listItem}>• {item}</Text>
                    ))}
                  </View>
                )}

                {/* Recovery & Expert */}
                <View style={styles.treatmentCard}>
                  <Text style={styles.treatmentCardTitle}>📅 Recovery Timeline</Text>
                  <Text style={styles.treatmentText}>{treatment.expected_recovery}</Text>
                </View>

                <View style={styles.treatmentCard}>
                  <Text style={styles.treatmentCardTitle}>👨‍🌾 When to Seek Expert</Text>
                  <Text style={styles.treatmentText}>{treatment.when_to_seek_expert}</Text>
                </View>
              </View>
            )}

            {/* Model Info */}
            <View style={styles.modelInfoCard}>
              <Text style={styles.modelInfoText}>
                Model: {result.model_info.version} | Accuracy:{' '}
                {result.model_info.accuracy}
              </Text>
            </View>

            {/* Analyze Another Button */}
            <TouchableOpacity
              style={styles.analyzeAnotherButton}
              onPress={handleReset}>
              <Text style={styles.analyzeAnotherButtonText}>
                Analyze Another Leaf
              </Text>
            </TouchableOpacity>
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
              Get your free API key from groq.com to enable AI-powered treatment recommendations
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
                <Text style={styles.modalSaveText}>Save & Continue</Text>
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
    backgroundColor: '#f0f4f0',
  },
  headerBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 15,
    paddingVertical: 15,
    backgroundColor: '#2e7d32',
  },
  backButton: {
    padding: 10,
    marginLeft: -5,
  },
  backButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '500',
  },
  headerBarTitle: {
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
    backgroundColor: 'rgba(255, 255, 255, 0.25)',
  },
  apiOffline: {
    backgroundColor: '#c62828',
  },
  apiStatusBadgeText: {
    color: '#fff',
    fontSize: 11,
    fontWeight: '600',
  },
  scrollContent: {
    padding: 16,
  },
  infoCard: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    marginBottom: 20,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 8,
  },
  infoIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  infoTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#2e7d32',
    marginBottom: 8,
  },
  infoSubtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
  },
  modeSelectionContainer: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 16,
    marginBottom: 20,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 8,
  },
  modeSelectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 12,
    textAlign: 'center',
  },
  modeButtonsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 12,
  },
  modeButton: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#e0e0e0',
  },
  modeButtonActive: {
    backgroundColor: '#e8f5e9',
    borderColor: '#2e7d32',
  },
  modeButtonIcon: {
    fontSize: 32,
    marginBottom: 8,
  },
  modeButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
  },
  modeButtonTextActive: {
    color: '#2e7d32',
  },
  modeButtonHint: {
    fontSize: 11,
    color: '#999',
    marginTop: 4,
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
    alignItems: 'center',
    marginBottom: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#2E7D32',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 12,
  },
  apiStatusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFF',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  apiStatusText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#333',
  },
  actionContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 24,
    gap: 12,
  },
  actionButton: {
    backgroundColor: '#FFF',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    flex: 1,
    elevation: 4,
    shadowColor: '#2e7d32',
    shadowOffset: {width: 0, height: 4},
    shadowOpacity: 0.15,
    shadowRadius: 8,
    borderWidth: 2,
    borderColor: '#e8f5e9',
  },
  actionButtonIcon: {
    fontSize: 52,
    marginBottom: 12,
  },
  actionButtonText: {
    fontSize: 14,
    fontWeight: '700',
    color: '#2e7d32',
    textAlign: 'center',
  },
  imageContainer: {
    alignItems: 'center',
    marginBottom: 24,
  },
  image: {
    width: '100%',
    height: 300,
    borderRadius: 12,
    resizeMode: 'cover',
  },
  resetButton: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 20,
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  resetButtonText: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: '600',
  },
  analyzeButton: {
    backgroundColor: '#4CAF50',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginBottom: 24,
  },
  analyzeButtonDisabled: {
    backgroundColor: '#A5D6A7',
  },
  analyzingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  analyzeButtonText: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 8,
  },
  resultsContainer: {
    marginTop: 8,
  },
  resultsTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 16,
    textAlign: 'center',
  },
  statusCard: {
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    marginBottom: 16,
    borderWidth: 2,
  },
  statusIcon: {
    fontSize: 48,
    marginBottom: 8,
  },
  statusText: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  confidenceText: {
    fontSize: 16,
    color: '#666',
  },
  probabilitiesCard: {
    backgroundColor: '#FFF',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
    marginBottom: 12,
  },
  probabilityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  probabilityLabel: {
    fontSize: 14,
    color: '#666',
    width: 80,
  },
  probabilityBarContainer: {
    flex: 1,
    height: 20,
    backgroundColor: '#E0E0E0',
    borderRadius: 10,
    marginHorizontal: 8,
    overflow: 'hidden',
  },
  probabilityBar: {
    height: '100%',
    borderRadius: 10,
  },
  probabilityValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    width: 50,
    textAlign: 'right',
  },
  messageCard: {
    backgroundColor: '#E3F2FD',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    flexDirection: 'row',
    alignItems: 'center',
  },
  messageIcon: {
    fontSize: 24,
    marginRight: 12,
  },
  messageText: {
    flex: 1,
    fontSize: 14,
    color: '#1976D2',
  },
  recommendationCard: {
    backgroundColor: '#FFF9C4',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
  },
  recommendationTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#F57C00',
    marginBottom: 8,
  },
  recommendationText: {
    fontSize: 14,
    color: '#666',
    lineHeight: 20,
  },
  modelInfoCard: {
    backgroundColor: '#F5F5F5',
    borderRadius: 8,
    padding: 12,
    marginBottom: 16,
  },
  modelInfoText: {
    fontSize: 12,
    color: '#999',
    textAlign: 'center',
  },
  analyzeAnotherButton: {
    backgroundColor: '#2196F3',
    borderRadius: 12,
    padding: 14,
    alignItems: 'center',
  },
  analyzeAnotherButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
  // Possible Conditions Styles
  possibleConditionsContainer: {
    marginTop: 8,
  },
  possibleConditionsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 4,
  },
  possibleConditionsSubtitle: {
    fontSize: 12,
    color: '#666',
    marginBottom: 12,
    fontStyle: 'italic',
  },
  conditionCard: {
    backgroundColor: '#FFF',
    borderRadius: 12,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#E0E0E0',
    overflow: 'hidden',
  },
  conditionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 14,
    backgroundColor: '#F8F8F8',
    borderBottomWidth: 2,
    borderBottomColor: '#E0E0E0',
  },
  conditionHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  conditionIcon: {
    fontSize: 28,
    marginRight: 12,
  },
  conditionTitleContainer: {
    flex: 1,
  },
  conditionTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  urgencyBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  urgencyText: {
    fontSize: 9,
    fontWeight: 'bold',
    color: '#FFF',
  },
  conditionContent: {
    padding: 16,
    backgroundColor: '#FFFFFF',
  },
  conditionSection: {
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 13,
    fontWeight: '600',
    color: '#333',
    marginBottom: 6,
  },
  sectionText: {
    fontSize: 13,
    color: '#555',
    lineHeight: 19,
  },
  symptomText: {
    fontSize: 12,
    color: '#666',
    lineHeight: 18,
    marginLeft: 8,
    marginBottom: 2,
  },
  solutionText: {
    fontSize: 13,
    color: '#2E7D32',
    lineHeight: 19,
    fontWeight: '500',
  },
  noteCard: {
    backgroundColor: '#E3F2FD',
    borderRadius: 8,
    padding: 12,
    marginTop: 8,
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  noteIcon: {
    fontSize: 18,
    marginRight: 8,
    marginTop: 2,
  },
  noteText: {
    flex: 1,
    fontSize: 11,
    color: '#1976D2',
    lineHeight: 16,
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
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
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
  treatmentDetails: {
    borderTopWidth: 1,
    borderTopColor: '#E0E0E0',
    paddingTop: 8,
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
    lineHeight: 18,
  },
  // Modal styles
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
