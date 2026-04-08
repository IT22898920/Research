import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
  ActivityIndicator,
  Alert,
  Modal,
  TextInput,
} from 'react-native';
import {launchCamera, launchImageLibrary} from 'react-native-image-picker';
import {useLanguage} from '../context/LanguageContext';
import {detectDisease, detectLeafDieback, checkApiHealth} from '../services/pestDetectionApi';
import {scanAPI} from '../services/scanApi';
import {treeAPI} from '../services/treeApi';
import {
  getHealthTreatmentRecommendations,
  isApiKeyConfigured,
  setApiKey,
} from '../services/treatmentApi';

// Disease detection types
const DISEASE_DETECTION_TYPES = {
  BABY_COCONUT: 'baby_coconut',
  OTHER_COCONUT: 'other_coconut',
};

export default function DiseaseDetectionScreen({navigation, route}) {
  const {t} = useLanguage();

  // Get tree info from navigation params (if scanning a specific tree)
  const treeId = route?.params?.treeId;
  const treeLabel = route?.params?.treeLabel;

  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');
  const [selectedType, setSelectedType] = useState(DISEASE_DETECTION_TYPES.BABY_COCONUT);

  // Treatment states
  const [treatmentData, setTreatmentData] = useState(null);
  const [isLoadingTreatment, setIsLoadingTreatment] = useState(false);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [hasApiKey, setHasApiKey] = useState(false);

  useEffect(() => {
    checkApi();
    checkApiKeyStatus();
  }, []);

  const checkApi = async () => {
    const health = await checkApiHealth();
    setApiStatus(health.success ? 'online' : 'offline');
  };

  const checkApiKeyStatus = async () => {
    const configured = await isApiKeyConfigured();
    setHasApiKey(configured);
  };

  const handleGetTreatment = async () => {
    if (!result?.prediction) return;

    // Determine condition type based on detection
    let conditionType = 'leaf_spot'; // default
    if (selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT) {
      conditionType = 'leaf_dieback';
    } else {
      if (result.prediction.is_leaf_rot) {
        conditionType = 'leaf_rot';
      } else if (result.prediction.is_leaf_spot) {
        conditionType = 'leaf_spot';
      }
    }

    // Determine severity based on confidence
    let severity = 'moderate';
    const confidence = result.prediction.confidence || 0;
    if (confidence > 0.8) {
      severity = 'severe';
    } else if (confidence < 0.5) {
      severity = 'mild';
    }

    setIsLoadingTreatment(true);

    try {
      const response = await getHealthTreatmentRecommendations({
        conditionType,
        severity,
        confidence,
        language: 'en',
      });

      if (response.success) {
        setTreatmentData({...response.data, source: 'ai'});
      } else if (response.fallback) {
        setTreatmentData({...response.fallback, source: 'fallback'});
        if (response.error === 'API key not configured') {
          setShowApiKeyModal(true);
        }
      } else {
        Alert.alert('Error', response.error || 'Failed to get treatment recommendations');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to get treatment recommendations');
    } finally {
      setIsLoadingTreatment(false);
    }
  };

  const handleSaveApiKey = async () => {
    if (apiKeyInput.trim()) {
      await setApiKey(apiKeyInput.trim());
      setHasApiKey(true);
      setShowApiKeyModal(false);
      setApiKeyInput('');
      // Retry getting treatment
      handleGetTreatment();
    }
  };

  const handleTakePhoto = () => {
    launchCamera(
      {
        mediaType: 'photo',
        quality: 0.8,
        maxWidth: 1024,
        maxHeight: 1024,
        includeBase64: true,
      },
      handleImageResponse,
    );
  };

  const handleChooseFromGallery = () => {
    launchImageLibrary(
      {
        mediaType: 'photo',
        quality: 0.8,
        maxWidth: 1024,
        maxHeight: 1024,
        includeBase64: true,
      },
      handleImageResponse,
    );
  };

  const handleImageResponse = response => {
    if (response.didCancel) return;
    if (response.errorCode) {
      Alert.alert('Error', response.errorMessage);
      return;
    }
    if (response.assets && response.assets[0]) {
      setSelectedImage(response.assets[0]);
      setResult(null);
    }
  };

  const selectImage = () => {
    Alert.alert(
      t('diseaseDetection.selectImage'),
      '',
      [
        {text: t('diseaseDetection.takePhoto'), onPress: handleTakePhoto},
        {text: t('diseaseDetection.chooseGallery'), onPress: handleChooseFromGallery},
        {text: t('common.cancel'), style: 'cancel'},
      ]
    );
  };

  const analyzeImage = async () => {
    if (!selectedImage) {
      Alert.alert('No Image', 'Please select an image first');
      return;
    }

    if (apiStatus !== 'online') {
      Alert.alert('API Offline', 'ML API is not available. Please check if the server is running.');
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    try {
      let response;
      if (selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT) {
        response = await detectLeafDieback(selectedImage.uri);
      } else {
        response = await detectDisease(selectedImage.uri);
      }

      if (response.success) {
        setResult({...response, detectionType: selectedType});

        // Save scan to database
        try {
          // Get base64 from image picker response
          let imageBase64 = null;
          if (selectedImage.base64) {
            imageBase64 = `data:image/jpeg;base64,${selectedImage.base64}`;
          }

          // Determine pest type based on detection
          let pestType = 'disease';
          if (selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT) {
            pestType = response.prediction?.is_leaf_dieback ? 'leaf_dieback' : 'leaf_dieback_healthy';
          } else {
            if (response.prediction?.is_leaf_rot) {
              pestType = 'leaf_rot';
            } else if (response.prediction?.is_leaf_spot) {
              pestType = 'leaf_spot';
            } else {
              pestType = 'leaf_disease_healthy';
            }
          }

          const scanData = {
            scanType: 'disease',
            pestType: pestType,
            detectionType: selectedType,
            isInfected: response.prediction?.is_diseased || response.prediction?.is_leaf_dieback || false,
            isValidImage: response.prediction?.is_valid_image !== false,
            confidence: response.prediction?.confidence || 0,
            pestsDetected: response.prediction?.is_diseased ? [pestType] : [],
            probabilities: response.probabilities || {},
            modelVersion: response.model_version || 'v4',
          };

          await scanAPI.saveScan(scanData, imageBase64);
          console.log('Scan saved successfully');

          // Save to tree record if scanning a specific tree
          if (treeId) {
            const status = scanData.isInfected ? 'infected' : 'healthy';

            // Save detected issues
            const detectedIssues = [];
            if (scanData.isInfected && pestType) {
              detectedIssues.push(`${pestType} disease detected`);
            }

            await treeAPI.updateScanResults(treeId, {
              detectedIssues: detectedIssues,
              healthStatus: status,
            });

            // Also add to health history
            const treeScanData = {
              status: status,
              scanType: 'disease',
              details: {
                detectionType: selectedType,
                pestType: pestType,
                confidence: scanData.confidence,
                probabilities: scanData.probabilities,
                prediction: response.prediction,
              },
            };
            await treeAPI.addHealthScan(treeId, treeScanData);
            console.log('Disease scan saved to tree record:', treeId);
          }
        } catch (saveErr) {
          console.log('Could not save scan:', saveErr.message);
          // Don't show error to user - scan result is still shown
        }
      } else {
        Alert.alert('Error', response.error || 'Failed to analyze image');
      }
    } catch (error) {
      Alert.alert('Error', error.message || 'Failed to connect to server');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetScan = () => {
    setSelectedImage(null);
    setResult(null);
    setTreatmentData(null);
  };

  const getStatusColor = status => {
    switch (status) {
      case 'diseased':
        return '#d32f2f';
      case 'healthy':
        return '#2e7d32';
      case 'invalid':
        return '#ff9800';
      default:
        return '#666';
    }
  };

  const getStatusIcon = status => {
    switch (status) {
      case 'diseased':
        return selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT ? '🥀' : '🦠';
      case 'healthy':
        return selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT ? '🌱' : '✅';
      case 'invalid':
        return '❌';
      default:
        return '❓';
    }
  };

  const getRecommendation = prediction => {
    if (selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT) {
      if (prediction.is_leaf_dieback) {
        return t('babyCoconut.leafDiebackDesc');
      } else if (prediction.is_healthy) {
        return t('babyCoconut.healthyDesc');
      }
    } else {
      if (prediction.is_leaf_rot) {
        return t('diseaseDetection.leafRotDesc');
      } else if (prediction.is_leaf_spot) {
        return t('diseaseDetection.leafSpotDesc');
      } else if (prediction.is_healthy) {
        return t('diseaseDetection.healthyDesc');
      }
    }
    return '';
  };

  const getThemeColor = () => {
    return selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT ? '#8B4513' : '#2e7d32';
  };

  const renderTypeSelector = () => (
    <View style={styles.typeContainer}>
      <Text style={styles.typeTitle}>{t('leafDisease.selectType')}:</Text>
      <View style={styles.typeButtons}>
        <TouchableOpacity
          style={[
            styles.typeButton,
            selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT && styles.typeButtonActiveBaby,
          ]}
          onPress={() => {
            setSelectedType(DISEASE_DETECTION_TYPES.BABY_COCONUT);
            setResult(null);
          }}>
          <Text style={styles.typeIcon}>🌴</Text>
          <Text
            style={[
              styles.typeText,
              selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT && styles.typeTextActiveBaby,
            ]}>
            {t('babyCoconut.title')}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.typeButton,
            selectedType === DISEASE_DETECTION_TYPES.OTHER_COCONUT && styles.typeButtonActiveOther,
          ]}
          onPress={() => {
            setSelectedType(DISEASE_DETECTION_TYPES.OTHER_COCONUT);
            setResult(null);
          }}>
          <Text style={styles.typeIcon}>🍃</Text>
          <Text
            style={[
              styles.typeText,
              selectedType === DISEASE_DETECTION_TYPES.OTHER_COCONUT && styles.typeTextActiveOther,
            ]}>
            {t('leafDisease.otherCoconut')}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderProbabilities = () => {
    if (!result?.probabilities) return null;

    const probs = result.probabilities;

    if (selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT) {
      return (
        <View style={styles.probabilitiesBox}>
          <Text style={styles.probabilitiesTitle}>{t('babyCoconut.probabilities')}</Text>
          <View style={styles.probabilityRow}>
            <Text style={styles.probabilityLabel}>{t('babyCoconut.leafDieback')}:</Text>
            <View style={styles.probabilityBarContainer}>
              <View
                style={[
                  styles.probabilityBar,
                  {
                    width: `${(probs.leaf_die_back || 0) * 100}%`,
                    backgroundColor: '#d32f2f',
                  },
                ]}
              />
            </View>
            <Text style={styles.probabilityValue}>
              {((probs.leaf_die_back || 0) * 100).toFixed(1)}%
            </Text>
          </View>
          <View style={styles.probabilityRow}>
            <Text style={styles.probabilityLabel}>{t('babyCoconut.healthy')}:</Text>
            <View style={styles.probabilityBarContainer}>
              <View
                style={[
                  styles.probabilityBar,
                  {
                    width: `${(probs.healthy || 0) * 100}%`,
                    backgroundColor: '#2e7d32',
                  },
                ]}
              />
            </View>
            <Text style={styles.probabilityValue}>
              {((probs.healthy || 0) * 100).toFixed(1)}%
            </Text>
          </View>
        </View>
      );
    } else {
      return (
        <View style={styles.probabilitiesBox}>
          <Text style={styles.probabilitiesTitle}>{t('diseaseDetection.probabilities')}</Text>
          <View style={styles.probabilityRow}>
            <Text style={styles.probabilityLabel}>{t('diseaseDetection.leafRot')}:</Text>
            <View style={styles.probabilityBarContainer}>
              <View
                style={[
                  styles.probabilityBar,
                  {
                    width: `${(probs.leaf_rot || 0) * 100}%`,
                    backgroundColor: '#d32f2f',
                  },
                ]}
              />
            </View>
            <Text style={styles.probabilityValue}>
              {((probs.leaf_rot || 0) * 100).toFixed(1)}%
            </Text>
          </View>
          <View style={styles.probabilityRow}>
            <Text style={styles.probabilityLabel}>{t('diseaseDetection.leafSpot')}:</Text>
            <View style={styles.probabilityBarContainer}>
              <View
                style={[
                  styles.probabilityBar,
                  {
                    width: `${(probs.leaf_spot || 0) * 100}%`,
                    backgroundColor: '#ff9800',
                  },
                ]}
              />
            </View>
            <Text style={styles.probabilityValue}>
              {((probs.leaf_spot || 0) * 100).toFixed(1)}%
            </Text>
          </View>
          <View style={styles.probabilityRow}>
            <Text style={styles.probabilityLabel}>{t('diseaseDetection.healthy')}:</Text>
            <View style={styles.probabilityBarContainer}>
              <View
                style={[
                  styles.probabilityBar,
                  {
                    width: `${(probs.healthy || 0) * 100}%`,
                    backgroundColor: '#2e7d32',
                  },
                ]}
              />
            </View>
            <Text style={styles.probabilityValue}>
              {((probs.healthy || 0) * 100).toFixed(1)}%
            </Text>
          </View>
        </View>
      );
    }
  };

  const renderResult = () => {
    if (!result || !result.prediction) return null;

    return (
      <View style={styles.resultContainer}>
        <View
          style={[
            styles.resultHeader,
            {backgroundColor: getStatusColor(result.prediction.status)},
          ]}>
          <Text style={styles.resultIcon}>
            {getStatusIcon(result.prediction.status)}
          </Text>
          <Text style={styles.resultLabel}>{result.prediction.label}</Text>
          <Text style={styles.resultConfidence}>
            {(result.prediction.confidence * 100).toFixed(1)}%
          </Text>
        </View>

        {/* Message */}
        <View style={styles.messageBox}>
          <Text style={styles.messageText}>{result.prediction.message}</Text>
        </View>

        {/* Recommendation */}
        {result.prediction.is_valid_image && (
          <View style={[styles.recommendationBox, {borderLeftColor: getThemeColor()}]}>
            <Text style={[styles.recommendationTitle, {color: getThemeColor()}]}>
              {selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT
                ? t('babyCoconut.recommendation')
                : t('diseaseDetection.recommendation')}
            </Text>
            <Text style={styles.recommendationText}>
              {getRecommendation(result.prediction)}
            </Text>
          </View>
        )}

        {/* Probabilities */}
        {renderProbabilities()}

        {/* Treatment Button - only show for diseased results */}
        {result.prediction.is_diseased || result.prediction.is_leaf_dieback ? (
          !treatmentData ? (
            <TouchableOpacity
              style={[styles.treatmentButton, {backgroundColor: getThemeColor()}]}
              onPress={handleGetTreatment}
              disabled={isLoadingTreatment}>
              {isLoadingTreatment ? (
                <ActivityIndicator color="#fff" size="small" />
              ) : (
                <>
                  <Text style={styles.treatmentButtonIcon}>💊</Text>
                  <Text style={styles.treatmentButtonText}>Get AI Treatment Plan</Text>
                </>
              )}
            </TouchableOpacity>
          ) : (
            <View style={styles.treatmentContainer}>
              <View style={styles.treatmentHeader}>
                <Text style={styles.treatmentTitle}>💊 Treatment Plan</Text>
                {treatmentData.source === 'fallback' && (
                  <View style={styles.fallbackBadge}>
                    <Text style={styles.fallbackBadgeText}>Offline</Text>
                  </View>
                )}
              </View>

              {/* Summary */}
              <View style={styles.treatmentSection}>
                <Text style={styles.treatmentSectionTitle}>Summary</Text>
                <Text style={styles.treatmentText}>{treatmentData.summary}</Text>
              </View>

              {/* Urgency */}
              <View style={[styles.urgencyBadge, {
                backgroundColor: treatmentData.urgency === 'critical' ? '#d32f2f' :
                  treatmentData.urgency === 'high' ? '#f57c00' :
                  treatmentData.urgency === 'medium' ? '#fbc02d' : '#4caf50'
              }]}>
                <Text style={styles.urgencyText}>
                  Urgency: {treatmentData.urgency?.toUpperCase()}
                </Text>
              </View>

              {/* Diagnosis */}
              {treatmentData.diagnosis && (
                <View style={styles.treatmentSection}>
                  <Text style={styles.treatmentSectionTitle}>🔍 Diagnosis</Text>
                  <Text style={styles.treatmentText}>{treatmentData.diagnosis}</Text>
                </View>
              )}

              {/* Treatments */}
              {treatmentData.treatments?.length > 0 && (
                <View style={styles.treatmentSection}>
                  <Text style={styles.treatmentSectionTitle}>🧪 Treatments</Text>
                  {treatmentData.treatments.map((treatment, index) => (
                    <View key={index} style={styles.treatmentCard}>
                      <View style={styles.treatmentCardHeader}>
                        <Text style={styles.treatmentCardTitle}>{treatment.name}</Text>
                        <View style={[styles.treatmentTypeBadge, {
                          backgroundColor: treatment.type === 'chemical' ? '#e3f2fd' :
                            treatment.type === 'organic' ? '#e8f5e9' :
                            treatment.type === 'nutritional' ? '#fff3e0' : '#f3e5f5'
                        }]}>
                          <Text style={[styles.treatmentTypeText, {
                            color: treatment.type === 'chemical' ? '#1565c0' :
                              treatment.type === 'organic' ? '#2e7d32' :
                              treatment.type === 'nutritional' ? '#ef6c00' : '#7b1fa2'
                          }]}>{treatment.type}</Text>
                        </View>
                      </View>
                      <Text style={styles.treatmentCardDesc}>{treatment.description}</Text>
                      <View style={styles.treatmentDetails}>
                        <Text style={styles.treatmentDetailItem}>💧 {treatment.dosage}</Text>
                        <Text style={styles.treatmentDetailItem}>🔄 {treatment.frequency}</Text>
                        <Text style={styles.treatmentDetailItem}>⏱️ {treatment.duration}</Text>
                        {treatment.cost_estimate && (
                          <Text style={styles.treatmentDetailItem}>💰 {treatment.cost_estimate}</Text>
                        )}
                      </View>
                    </View>
                  ))}
                </View>
              )}

              {/* Nutritional Recommendations */}
              {treatmentData.nutritional_recommendations?.length > 0 && (
                <View style={styles.treatmentSection}>
                  <Text style={styles.treatmentSectionTitle}>🌱 Nutrition</Text>
                  {treatmentData.nutritional_recommendations.map((rec, index) => (
                    <Text key={index} style={styles.listItem}>• {rec}</Text>
                  ))}
                </View>
              )}

              {/* Preventive Measures */}
              {treatmentData.preventive_measures?.length > 0 && (
                <View style={styles.treatmentSection}>
                  <Text style={styles.treatmentSectionTitle}>🛡️ Prevention</Text>
                  {treatmentData.preventive_measures.map((measure, index) => (
                    <Text key={index} style={styles.listItem}>• {measure}</Text>
                  ))}
                </View>
              )}

              {/* Safety */}
              {treatmentData.safety_precautions?.length > 0 && (
                <View style={styles.treatmentSection}>
                  <Text style={styles.treatmentSectionTitle}>⚠️ Safety</Text>
                  {treatmentData.safety_precautions.map((precaution, index) => (
                    <Text key={index} style={styles.listItem}>• {precaution}</Text>
                  ))}
                </View>
              )}

              {/* Recovery */}
              {treatmentData.expected_recovery && (
                <View style={styles.treatmentSection}>
                  <Text style={styles.treatmentSectionTitle}>📅 Expected Recovery</Text>
                  <Text style={styles.treatmentText}>{treatmentData.expected_recovery}</Text>
                </View>
              )}

              {/* When to Seek Expert */}
              {treatmentData.when_to_seek_expert && (
                <View style={[styles.treatmentSection, styles.expertSection]}>
                  <Text style={styles.treatmentSectionTitle}>👨‍🌾 When to Seek Expert</Text>
                  <Text style={styles.treatmentText}>{treatmentData.when_to_seek_expert}</Text>
                </View>
              )}
            </View>
          )
        ) : null}

        {/* Scan Again Button */}
        <TouchableOpacity
          style={[styles.resetButton, {backgroundColor: getThemeColor()}]}
          onPress={resetScan}>
          <Text style={styles.resetButtonText}>
            🔄 {selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT
              ? t('babyCoconut.scanAgain')
              : t('diseaseDetection.scanAgain')}
          </Text>
        </TouchableOpacity>
      </View>
    );
  };

  const renderInfoSection = () => {
    if (selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT) {
      return (
        <View style={[styles.infoContainer, {backgroundColor: '#FFF8DC'}]}>
          <Text style={[styles.infoTitle, {color: '#8B4513'}]}>
            {t('babyCoconut.aboutFeature')}
          </Text>
          <Text style={styles.infoText}>
            {t('babyCoconut.aboutDescription')}{'\n'}
            • {t('babyCoconut.detectsLeafDieback')}{'\n'}
            • {t('babyCoconut.forYoungTrees')}{'\n'}
            • {t('babyCoconut.detectsNonCoconut')}
          </Text>
        </View>
      );
    } else {
      return (
        <View style={[styles.infoContainer, {backgroundColor: '#e8f5e9'}]}>
          <Text style={[styles.infoTitle, {color: '#2e7d32'}]}>
            {t('diseaseDetection.aboutFeature')}
          </Text>
          <Text style={styles.infoText}>
            {t('diseaseDetection.aboutDescription')}{'\n'}
            • {t('diseaseDetection.detectsLeafRot')}{'\n'}
            • {t('diseaseDetection.detectsLeafSpot')}{'\n'}
            • {t('diseaseDetection.detectsNonCoconut')}
          </Text>
        </View>
      );
    }
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>← {t('common.back')}</Text>
        </TouchableOpacity>
        <Text style={styles.title}>{t('leafDisease.title')}</Text>
        <View
          style={[
            styles.statusBadge,
            {backgroundColor: apiStatus === 'online' ? '#4caf50' : '#f44336'},
          ]}>
          <Text style={styles.statusText}>
            {apiStatus === 'checking' ? '...' : apiStatus === 'online' ? 'Online' : 'Offline'}
          </Text>
        </View>
      </View>

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

      {/* Type Selector */}
      {renderTypeSelector()}

      {/* Image Preview */}
      <View style={styles.imageContainer}>
        {selectedImage ? (
          <Image source={{uri: selectedImage.uri}} style={styles.previewImage} />
        ) : (
          <View style={styles.placeholderContainer}>
            <Text style={styles.placeholderIcon}>
              {selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT ? '🌴' : '🍃'}
            </Text>
            <Text style={styles.placeholderText}>
              {selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT
                ? t('babyCoconut.selectImage')
                : t('diseaseDetection.selectImage')}
            </Text>
          </View>
        )}
      </View>

      {/* Action Buttons */}
      {!result && (
        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={[styles.selectButton, {borderColor: getThemeColor()}]}
            onPress={selectImage}>
            <Text style={[styles.selectButtonText, {color: getThemeColor()}]}>
              📷 {t('diseaseDetection.selectImage')}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[
              styles.analyzeButton,
              {backgroundColor: getThemeColor()},
              (!selectedImage || isAnalyzing) && styles.buttonDisabled
            ]}
            onPress={analyzeImage}
            disabled={!selectedImage || isAnalyzing}>
            {isAnalyzing ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.analyzeButtonText}>
                🔬 {selectedType === DISEASE_DETECTION_TYPES.BABY_COCONUT
                  ? t('babyCoconut.analyze')
                  : t('diseaseDetection.analyzing').replace('...', '')}
              </Text>
            )}
          </TouchableOpacity>
        </View>
      )}

      {/* Results */}
      {renderResult()}

      {/* Info Section */}
      {!result && renderInfoSection()}

      {/* API Key Modal */}
      <Modal
        visible={showApiKeyModal}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setShowApiKeyModal(false)}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>🔑 Configure AI Treatment</Text>
            <Text style={styles.modalDescription}>
              Enter your Groq API key to get AI-powered treatment recommendations.
              Get a free key at console.groq.com
            </Text>
            <TextInput
              style={styles.apiKeyInput}
              placeholder="Enter Groq API Key"
              value={apiKeyInput}
              onChangeText={setApiKeyInput}
              secureTextEntry={true}
              autoCapitalize="none"
            />
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={styles.modalCancelButton}
                onPress={() => setShowApiKeyModal(false)}>
                <Text style={styles.modalCancelText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalSaveButton, {backgroundColor: getThemeColor()}]}
                onPress={handleSaveApiKey}>
                <Text style={styles.modalSaveText}>Save</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </ScrollView>
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
    padding: 15,
    paddingTop: 50,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  backButton: {
    fontSize: 16,
    color: '#2e7d32',
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  treeBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#e8f5e9',
    padding: 12,
    marginHorizontal: 15,
    marginTop: 10,
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
  typeContainer: {
    backgroundColor: '#fff',
    padding: 15,
    marginBottom: 10,
  },
  typeTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  typeButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  typeButton: {
    flex: 1,
    alignItems: 'center',
    padding: 15,
    marginHorizontal: 5,
    backgroundColor: '#f5f5f5',
    borderRadius: 10,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  typeButtonActiveBaby: {
    backgroundColor: '#FFF8DC',
    borderColor: '#8B4513',
  },
  typeButtonActiveOther: {
    backgroundColor: '#e8f5e9',
    borderColor: '#2e7d32',
  },
  typeIcon: {
    fontSize: 30,
    marginBottom: 8,
  },
  typeText: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
  },
  typeTextActiveBaby: {
    color: '#8B4513',
    fontWeight: 'bold',
  },
  typeTextActiveOther: {
    color: '#2e7d32',
    fontWeight: 'bold',
  },
  imageContainer: {
    margin: 15,
    height: 250,
    backgroundColor: '#fff',
    borderRadius: 15,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  previewImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  placeholderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f9f9f9',
  },
  placeholderIcon: {
    fontSize: 60,
    marginBottom: 10,
  },
  placeholderText: {
    color: '#999',
    fontSize: 14,
  },
  buttonContainer: {
    paddingHorizontal: 15,
    gap: 10,
  },
  selectButton: {
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    borderWidth: 2,
  },
  selectButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  analyzeButton: {
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  analyzeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  resultContainer: {
    margin: 15,
    backgroundColor: '#fff',
    borderRadius: 15,
    overflow: 'hidden',
  },
  resultHeader: {
    padding: 20,
    alignItems: 'center',
  },
  resultIcon: {
    fontSize: 40,
    marginBottom: 10,
  },
  resultLabel: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'center',
  },
  resultConfidence: {
    fontSize: 16,
    color: 'rgba(255,255,255,0.9)',
    marginTop: 5,
  },
  messageBox: {
    padding: 15,
    backgroundColor: '#f5f5f5',
  },
  messageText: {
    fontSize: 14,
    color: '#333',
    textAlign: 'center',
  },
  recommendationBox: {
    padding: 15,
    borderTopWidth: 1,
    borderTopColor: '#eee',
    borderLeftWidth: 4,
    marginHorizontal: 10,
    marginTop: 10,
    backgroundColor: '#fafafa',
    borderRadius: 8,
  },
  recommendationTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  recommendationText: {
    fontSize: 14,
    color: '#333',
    lineHeight: 22,
  },
  probabilitiesBox: {
    padding: 15,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  probabilitiesTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  probabilityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  probabilityLabel: {
    width: 90,
    fontSize: 12,
    color: '#666',
  },
  probabilityBarContainer: {
    flex: 1,
    height: 10,
    backgroundColor: '#eee',
    borderRadius: 5,
    marginHorizontal: 10,
    overflow: 'hidden',
  },
  probabilityBar: {
    height: '100%',
    borderRadius: 5,
  },
  probabilityValue: {
    width: 50,
    fontSize: 12,
    color: '#333',
    textAlign: 'right',
  },
  resetButton: {
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    margin: 15,
  },
  resetButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  infoContainer: {
    margin: 15,
    padding: 15,
    borderRadius: 10,
    marginBottom: 30,
  },
  infoTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  infoText: {
    fontSize: 13,
    color: '#333',
    lineHeight: 20,
  },
  // Treatment styles
  treatmentButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 10,
    margin: 15,
    marginBottom: 0,
  },
  treatmentButtonIcon: {
    fontSize: 20,
    marginRight: 8,
  },
  treatmentButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  treatmentContainer: {
    margin: 15,
    marginBottom: 0,
    padding: 15,
    backgroundColor: '#f8f9fa',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  treatmentHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  treatmentTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  fallbackBadge: {
    backgroundColor: '#ff9800',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  fallbackBadgeText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  treatmentSection: {
    marginBottom: 15,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  treatmentSectionTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#555',
    marginBottom: 8,
  },
  treatmentText: {
    fontSize: 14,
    color: '#333',
    lineHeight: 20,
  },
  urgencyBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 15,
    marginBottom: 15,
  },
  urgencyText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  treatmentCard: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 12,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  treatmentCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  treatmentCardTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    flex: 1,
  },
  treatmentTypeBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 10,
  },
  treatmentTypeText: {
    fontSize: 10,
    fontWeight: 'bold',
  },
  treatmentCardDesc: {
    fontSize: 13,
    color: '#555',
    marginBottom: 8,
  },
  treatmentDetails: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  treatmentDetailItem: {
    fontSize: 11,
    color: '#666',
    backgroundColor: '#f5f5f5',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 5,
  },
  listItem: {
    fontSize: 13,
    color: '#333',
    lineHeight: 22,
    marginLeft: 5,
  },
  expertSection: {
    backgroundColor: '#fff3e0',
    padding: 12,
    borderRadius: 8,
    borderBottomWidth: 0,
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
    backgroundColor: '#fff',
    borderRadius: 15,
    padding: 20,
    width: '100%',
    maxWidth: 400,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
    textAlign: 'center',
  },
  modalDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 15,
    textAlign: 'center',
    lineHeight: 20,
  },
  apiKeyInput: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 10,
    padding: 12,
    fontSize: 14,
    marginBottom: 15,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 10,
  },
  modalCancelButton: {
    flex: 1,
    padding: 12,
    borderRadius: 10,
    backgroundColor: '#f5f5f5',
    alignItems: 'center',
  },
  modalCancelText: {
    color: '#666',
    fontSize: 14,
    fontWeight: 'bold',
  },
  modalSaveButton: {
    flex: 1,
    padding: 12,
    borderRadius: 10,
    alignItems: 'center',
  },
  modalSaveText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
});
