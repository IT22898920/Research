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
  Modal,
  TextInput,
} from 'react-native';
import {launchImageLibrary, launchCamera} from 'react-native-image-picker';
import {
  checkApiHealth,
  detectMite,
  detectCaterpillar,
  detectWhiteFly,
  detectAllPests,
  PEST_TYPES,
} from '../services/pestDetectionApi';
import {
  getTreatmentRecommendations,
  setApiKey,
  isApiKeyConfigured,
} from '../services/treatmentApi';
import {scanAPI} from '../services/scanApi';
import {showPestDetectionNotification} from '../services/notificationService';
import {useLanguage} from '../context/LanguageContext';

export default function PestDetectionScreen({navigation}) {
  const {t, currentLanguage} = useLanguage();
  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');
  const [selectedPestType, setSelectedPestType] = useState(PEST_TYPES.ALL);

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

  const checkApi = async () => {
    const health = await checkApiHealth();
    setApiStatus(health.success ? 'online' : 'offline');
  };

  const checkApiKeyStatus = async () => {
    const configured = await isApiKeyConfigured();
    setHasApiKey(configured);
  };

  // Save scan result to database (non-blocking)
  const saveScanToDatabase = async (scanResult, scanType) => {
    try {
      let scanData;

      if (scanType === PEST_TYPES.ALL) {
        const miteResult = scanResult.results?.mite;
        const caterpillarResult = scanResult.results?.caterpillar;
        const whiteFlyResult = scanResult.results?.white_fly;
        const isInfected = !scanResult.summary?.is_healthy;

        // Get severity from the highest confidence infected pest
        let severity = null;
        if (miteResult?.is_infected) {
          const sev = getSeverity(miteResult.confidence, true);
          if (sev) severity = {level: sev.level, percent: sev.percent};
        } else if (caterpillarResult?.is_infected) {
          const sev = getSeverity(caterpillarResult.confidence, true);
          if (sev) severity = {level: sev.level, percent: sev.percent};
        } else if (whiteFlyResult?.is_infected) {
          const sev = getSeverity(whiteFlyResult.confidence, true);
          if (sev) severity = {level: sev.level, percent: sev.percent};
        }

        scanData = {
          scanType: 'all',
          isInfected: isInfected,
          isValidImage: scanResult.summary?.is_valid_image !== false,
          results: {
            mite: miteResult
              ? {
                  detected: miteResult.is_infected || false,
                  confidence: miteResult.confidence || 0,
                  class: miteResult.class || miteResult.label,
                }
              : null,
            caterpillar: caterpillarResult
              ? {
                  detected: caterpillarResult.is_infected || false,
                  confidence: caterpillarResult.confidence || 0,
                  class: caterpillarResult.class || caterpillarResult.label,
                }
              : null,
            white_fly: whiteFlyResult
              ? {
                  detected: whiteFlyResult.is_infected || false,
                  confidence: whiteFlyResult.confidence || 0,
                  class: whiteFlyResult.class || whiteFlyResult.label,
                }
              : null,
          },
          pestsDetected: scanResult.summary?.pests_detected || [],
          severity: severity,
        };
      } else {
        const prediction = scanResult.prediction;
        const isInfected = prediction?.is_infected || false;
        const severity = getSeverity(prediction?.confidence || 0, isInfected);

        // Map scan type to result key and pest name
        const scanTypeMap = {
          [PEST_TYPES.MITE]: {key: 'mite', pest: 'coconut_mite'},
          [PEST_TYPES.CATERPILLAR]: {key: 'caterpillar', pest: 'caterpillar'},
          [PEST_TYPES.WHITE_FLY]: {key: 'white_fly', pest: 'white_fly'},
        };
        const typeInfo = scanTypeMap[scanType] || {key: 'mite', pest: 'coconut_mite'};

        scanData = {
          scanType: typeInfo.key,
          isInfected: isInfected,
          isValidImage: prediction?.is_valid_image !== false,
          results: {
            [typeInfo.key]: {
              detected: isInfected,
              confidence: prediction?.confidence || 0,
              class: prediction?.class || prediction?.label,
            },
          },
          pestsDetected: isInfected ? [typeInfo.pest] : [],
          severity: severity
            ? {level: severity.level, percent: severity.percent}
            : null,
        };
      }

      // Get base64 image if available for Cloudinary upload
      const imageBase64 = selectedImage?.base64 || null;
      const savedScan = await scanAPI.saveScan(scanData, imageBase64);
      console.log('Scan saved to database successfully');

      // Show notification if pest was detected
      if (scanData.isInfected && scanData.pestsDetected?.length > 0) {
        let pestName = 'Pest';
        if (scanData.pestsDetected.includes('coconut_mite')) {
          pestName = 'Coconut Mite';
        } else if (scanData.pestsDetected.includes('caterpillar')) {
          pestName = 'Caterpillar';
        } else if (scanData.pestsDetected.includes('white_fly')) {
          pestName = 'White Fly';
        }
        const severity = scanData.severity?.level || 'moderate';
        showPestDetectionNotification(pestName, severity, savedScan?.data?._id);
      }
    } catch (error) {
      console.error('Error saving scan to database:', error);
      // Don't show error to user - scan saving is secondary
    }
  };

  const handleSaveApiKey = async () => {
    if (apiKeyInput.trim()) {
      await setApiKey(apiKeyInput.trim());
      setHasApiKey(true);
      setShowApiKeyModal(false);
      setApiKeyInput('');
      Alert.alert(t('common.success'), 'API Key saved successfully');
    }
  };

  const fetchTreatment = async (pestType, severity, confidence) => {
    setIsLoadingTreatment(true);
    setTreatment(null);

    try {
      const response = await getTreatmentRecommendations({
        pestType: pestType,
        severity: severity?.level || 'moderate',
        confidence: confidence || 0.7,
        language: currentLanguage,
      });

      if (response.success) {
        setTreatment({...response.data, source: 'ai'});
      } else if (response.fallback) {
        setTreatment({...response.fallback, source: 'fallback'});
      } else {
        Alert.alert(t('common.error'), response.error || 'Failed to get treatment');
      }
    } catch (error) {
      console.error('Treatment fetch error:', error);
      Alert.alert(t('common.error'), error.message);
    } finally {
      setIsLoadingTreatment(false);
    }
  };

  const selectImage = () => {
    Alert.alert(t('pestDetection.selectImage'), t('pestDetection.selectImage'), [
      {text: t('pestDetection.takePhoto'), onPress: () => openCamera()},
      {text: t('pestDetection.chooseGallery'), onPress: () => openGallery()},
      {text: t('common.cancel'), style: 'cancel'},
    ]);
  };

  const openCamera = () => {
    launchCamera(
      {mediaType: 'photo', quality: 0.8, maxWidth: 1024, maxHeight: 1024, includeBase64: true},
      handleImageResponse,
    );
  };

  const openGallery = () => {
    launchImageLibrary(
      {mediaType: 'photo', quality: 0.8, maxWidth: 1024, maxHeight: 1024, includeBase64: true},
      handleImageResponse,
    );
  };

  const handleImageResponse = response => {
    if (response.didCancel) return;
    if (response.errorCode) {
      Alert.alert('Error', response.errorMessage || 'Failed to select image');
      return;
    }
    if (response.assets && response.assets[0]) {
      setSelectedImage(response.assets[0]);
      setResult(null);
      setTreatment(null);  // Clear previous treatment when new image selected
    }
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
      switch (selectedPestType) {
        case PEST_TYPES.MITE:
          response = await detectMite(selectedImage.uri);
          break;
        case PEST_TYPES.CATERPILLAR:
          response = await detectCaterpillar(selectedImage.uri);
          break;
        case PEST_TYPES.WHITE_FLY:
          response = await detectWhiteFly(selectedImage.uri);
          break;
        case PEST_TYPES.ALL:
        default:
          response = await detectAllPests(selectedImage.uri);
          break;
      }

      if (response.success) {
        setResult({...response, pestType: selectedPestType});
        // Save scan to database (non-blocking)
        saveScanToDatabase(response, selectedPestType);
      } else {
        Alert.alert('Analysis Failed', response.error || 'Could not analyze image');
      }
    } catch (error) {
      Alert.alert('Error', error.message || 'Something went wrong');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getResultColor = () => {
    if (!result) return '#666';
    if (selectedPestType === PEST_TYPES.ALL) {
      if (!result.summary?.is_valid_image) return '#ff9800'; // Orange for invalid
      return result.summary?.is_healthy ? '#2e7d32' : '#d32f2f';
    }
    if (!result.prediction?.is_valid_image) return '#ff9800'; // Orange for invalid
    return result.prediction?.is_infected ? '#d32f2f' : '#2e7d32';
  };

  const getResultIcon = () => {
    if (!result) return '🔍';
    if (selectedPestType === PEST_TYPES.ALL) {
      if (!result.summary?.is_valid_image) return '❓'; // Not a coconut
      return result.summary?.is_healthy ? '✅' : '🐛';
    }
    if (!result.prediction?.is_valid_image) return '❓'; // Not a coconut
    return result.prediction?.is_infected ? '🐛' : '✅';
  };

  // Severity calculation based on confidence score
  const getSeverity = (confidence, isInfected) => {
    if (!isInfected) return null;

    const confidencePercent = confidence * 100;

    if (confidencePercent < 70) {
      return {
        level: 'mild',
        label: t('pestDetection.severityMild'),
        description: t('pestDetection.severityMildDesc'),
        color: '#4caf50', // Green
        icon: '🟢',
        percent: Math.round((confidencePercent - 50) / 20 * 30), // 0-30%
      };
    } else if (confidencePercent < 85) {
      return {
        level: 'moderate',
        label: t('pestDetection.severityModerate'),
        description: t('pestDetection.severityModerateDesc'),
        color: '#ff9800', // Orange
        icon: '🟡',
        percent: Math.round(30 + (confidencePercent - 70) / 15 * 30), // 30-60%
      };
    } else {
      return {
        level: 'severe',
        label: t('pestDetection.severitySevere'),
        description: t('pestDetection.severitySevereDesc'),
        color: '#f44336', // Red
        icon: '🔴',
        percent: Math.round(60 + (confidencePercent - 85) / 15 * 40), // 60-100%
      };
    }
  };

  // Render severity indicator component
  const renderSeverityIndicator = (severity) => {
    if (!severity) return null;

    return (
      <View style={styles.severityContainer}>
        <View style={styles.severityHeader}>
          <Text style={styles.severityLabel}>{t('pestDetection.severity')}:</Text>
          <View style={[styles.severityBadge, {backgroundColor: severity.color}]}>
            <Text style={styles.severityBadgeText}>{severity.icon} {severity.label}</Text>
          </View>
        </View>
        <View style={styles.severityBarContainer}>
          <View style={styles.severityBarBackground}>
            <View
              style={[
                styles.severityBarFill,
                {width: `${severity.percent}%`, backgroundColor: severity.color}
              ]}
            />
          </View>
          <Text style={styles.severityPercent}>{severity.percent}%</Text>
        </View>
        <Text style={[styles.severityDescription, {color: severity.color}]}>
          {severity.description}
        </Text>
      </View>
    );
  };

  const renderPestTypeSelector = () => (
    <View style={styles.pestTypeContainer}>
      <Text style={styles.pestTypeTitle}>{t('pestDetection.title')}:</Text>
      <View style={styles.pestTypeButtons}>
        <TouchableOpacity
          style={[
            styles.pestTypeButton,
            selectedPestType === PEST_TYPES.ALL && styles.pestTypeButtonActive,
          ]}
          onPress={() => setSelectedPestType(PEST_TYPES.ALL)}>
          <Text style={styles.pestTypeIcon}>🔬</Text>
          <Text
            style={[
              styles.pestTypeText,
              selectedPestType === PEST_TYPES.ALL && styles.pestTypeTextActive,
            ]}>
            {t('pestDetection.detectAll')}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.pestTypeButton,
            selectedPestType === PEST_TYPES.MITE && styles.pestTypeButtonActive,
          ]}
          onPress={() => setSelectedPestType(PEST_TYPES.MITE)}>
          <Text style={styles.pestTypeIcon}>🕷️</Text>
          <Text
            style={[
              styles.pestTypeText,
              selectedPestType === PEST_TYPES.MITE && styles.pestTypeTextActive,
            ]}>
            {t('pestDetection.coconutMite')}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.pestTypeButton,
            selectedPestType === PEST_TYPES.CATERPILLAR && styles.pestTypeButtonActive,
          ]}
          onPress={() => setSelectedPestType(PEST_TYPES.CATERPILLAR)}>
          <Text style={styles.pestTypeIcon}>🐛</Text>
          <Text
            style={[
              styles.pestTypeText,
              selectedPestType === PEST_TYPES.CATERPILLAR && styles.pestTypeTextActive,
            ]}>
            {t('pestDetection.caterpillar')}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[
            styles.pestTypeButton,
            selectedPestType === PEST_TYPES.WHITE_FLY && styles.pestTypeButtonActive,
          ]}
          onPress={() => setSelectedPestType(PEST_TYPES.WHITE_FLY)}>
          <Text style={styles.pestTypeIcon}>🦟</Text>
          <Text
            style={[
              styles.pestTypeText,
              selectedPestType === PEST_TYPES.WHITE_FLY && styles.pestTypeTextActive,
            ]}>
            {t('pestDetection.whiteFly') || 'White Fly'}
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderSingleResult = () => {
    if (!result?.prediction) return null;

    const isValidImage = result.prediction?.is_valid_image !== false;

    // Handle non-coconut images
    if (!isValidImage) {
      return (
        <View style={[styles.resultContainer, {borderColor: '#ff9800'}]}>
          <Text style={styles.resultIcon}>❓</Text>
          <Text style={[styles.resultTitle, {color: '#ff9800'}]}>
            {result.prediction?.label || t('pestDetection.notCoconut')}
          </Text>
          <Text style={styles.confidenceText}>
            {t('pestDetection.confidence')}: {((result.prediction?.confidence || 0) * 100).toFixed(1)}%
          </Text>

          <View style={styles.invalidImageBox}>
            <Text style={styles.invalidImageTitle}>{t('pestDetection.imageNotRecognized')}</Text>
            <Text style={styles.invalidImageText}>
              {result.prediction?.message || t('pestDetection.uploadClearImage')}
            </Text>
          </View>

          {result.probabilities && (
            <View style={styles.probabilitiesContainer}>
              <Text style={styles.probabilitiesTitle}>{t('pestDetection.probabilities')}:</Text>
              {Object.entries(result.probabilities).map(([label, prob]) => (
                <View key={label} style={styles.probabilityRow}>
                  <Text style={styles.probabilityLabel}>{label}:</Text>
                  <View style={styles.probabilityBarContainer}>
                    <View style={[styles.probabilityBar, {width: `${(prob || 0) * 100}%`, backgroundColor: label === 'not_coconut' ? '#ff9800' : '#2e7d32'}]} />
                  </View>
                  <Text style={styles.probabilityValue}>{((prob || 0) * 100).toFixed(1)}%</Text>
                </View>
              ))}
            </View>
          )}
        </View>
      );
    }

    const severity = getSeverity(result.prediction?.confidence || 0, result.prediction?.is_infected);

    return (
      <View style={[styles.resultContainer, {borderColor: getResultColor()}]}>
        <Text style={styles.resultIcon}>{getResultIcon()}</Text>
        <Text style={[styles.resultTitle, {color: getResultColor()}]}>
          {result.prediction?.label || (result.prediction?.is_infected ? t('pestDetection.infected') : t('pestDetection.healthy'))}
        </Text>
        <Text style={styles.confidenceText}>
          {t('pestDetection.confidence')}: {((result.prediction?.confidence || 0) * 100).toFixed(1)}%
        </Text>

        {/* Severity Indicator - Only show when infected */}
        {renderSeverityIndicator(severity)}

        {result.probabilities && (
          <View style={styles.probabilitiesContainer}>
            <Text style={styles.probabilitiesTitle}>{t('pestDetection.probabilities')}:</Text>
            {Object.entries(result.probabilities).map(([label, prob]) => (
              <View key={label} style={styles.probabilityRow}>
                <Text style={styles.probabilityLabel}>{label}:</Text>
                <View style={styles.probabilityBarContainer}>
                  <View style={[styles.probabilityBar, {width: `${(prob || 0) * 100}%`}]} />
                </View>
                <Text style={styles.probabilityValue}>{((prob || 0) * 100).toFixed(1)}%</Text>
              </View>
            ))}
          </View>
        )}

        {result.prediction?.is_infected && (
          <View style={styles.warningBox}>
            <Text style={styles.warningTitle}>⚠️ {t('pestDetection.actionRequired')}</Text>
            <Text style={styles.warningText}>
              {result.pestType === PEST_TYPES.MITE
                ? t('pestDetection.coconutMite') + ' - ' + t('pestDetection.detected')
                : result.pestType === PEST_TYPES.CATERPILLAR
                ? t('pestDetection.caterpillar') + ' - ' + t('pestDetection.detected')
                : (t('pestDetection.whiteFly') || 'White Fly') + ' - ' + t('pestDetection.detected')}
            </Text>
          </View>
        )}
      </View>
    );
  };

  const renderAllPestsResult = () => {
    if (!result?.results && !result?.summary) return null;

    const isValidImage = result.summary?.is_valid_image !== false;

    // Handle non-coconut images
    if (!isValidImage) {
      return (
        <View style={[styles.resultContainer, {borderColor: '#ff9800'}]}>
          <Text style={styles.resultIcon}>❓</Text>
          <Text style={[styles.resultTitle, {color: '#ff9800'}]}>
            {result.summary?.label || t('pestDetection.notCoconut')}
          </Text>

          <View style={styles.invalidImageBox}>
            <Text style={styles.invalidImageTitle}>{t('pestDetection.imageNotRecognized')}</Text>
            <Text style={styles.invalidImageText}>
              {result.summary?.message || t('pestDetection.uploadClearImage')}
            </Text>
          </View>

          {result.summary?.recommendation && (
            <View style={styles.recommendationBox}>
              <Text style={styles.recommendationTitle}>{t('pestDetection.recommendation')}</Text>
              <Text style={styles.recommendationText}>
                {result.summary.recommendation}
              </Text>
            </View>
          )}
        </View>
      );
    }

    return (
      <View style={[styles.resultContainer, {borderColor: getResultColor()}]}>
        <Text style={styles.resultIcon}>{getResultIcon()}</Text>
        <Text style={[styles.resultTitle, {color: getResultColor()}]}>
          {result.summary?.label || (result.summary?.is_healthy ? t('pestDetection.healthy') : t('pestDetection.infected'))}
        </Text>

        {/* Display API message if available */}
        {result.summary?.message && (
          <Text style={styles.summaryMessage}>{result.summary.message}</Text>
        )}

        {/* Individual Results */}
        <View style={styles.allResultsContainer}>
          {/* Mite Result */}
          {result.results?.mite && (
            <View style={styles.pestResultCard}>
              <Text style={styles.pestResultIcon}>🕷️</Text>
              <Text style={styles.pestResultName}>{t('pestDetection.coconutMite')}</Text>
              <Text
                style={[
                  styles.pestResultStatus,
                  {color: result.results.mite.class === 'not_coconut' ? '#ff9800' : result.results.mite.is_infected ? '#d32f2f' : '#2e7d32'},
                ]}>
                {result.results.mite.class === 'not_coconut' ? 'N/A' : result.results.mite.is_infected ? t('pestDetection.detected') : t('pestDetection.notFound')}
              </Text>
              <Text style={styles.pestResultConfidence}>
                {((result.results.mite.confidence || 0) * 100).toFixed(1)}%
              </Text>
              {/* Mite Severity */}
              {result.results.mite.is_infected && (
                <View style={[styles.miniSeverityBadge, {backgroundColor: getSeverity(result.results.mite.confidence, true)?.color}]}>
                  <Text style={styles.miniSeverityText}>
                    {getSeverity(result.results.mite.confidence, true)?.icon} {getSeverity(result.results.mite.confidence, true)?.label}
                  </Text>
                </View>
              )}
            </View>
          )}

          {/* Caterpillar Result */}
          {result.results?.caterpillar && (
            <View style={styles.pestResultCard}>
              <Text style={styles.pestResultIcon}>🐛</Text>
              <Text style={styles.pestResultName}>{t('pestDetection.caterpillar')}</Text>
              <Text
                style={[
                  styles.pestResultStatus,
                  {color: result.results.caterpillar.class === 'not_coconut' ? '#ff9800' : result.results.caterpillar.is_infected ? '#d32f2f' : '#2e7d32'},
                ]}>
                {result.results.caterpillar.class === 'not_coconut' ? 'N/A' : result.results.caterpillar.is_infected ? t('pestDetection.detected') : t('pestDetection.notFound')}
              </Text>
              <Text style={styles.pestResultConfidence}>
                {((result.results.caterpillar.confidence || 0) * 100).toFixed(1)}%
              </Text>
              {/* Caterpillar Severity */}
              {result.results.caterpillar.is_infected && (
                <View style={[styles.miniSeverityBadge, {backgroundColor: getSeverity(result.results.caterpillar.confidence, true)?.color}]}>
                  <Text style={styles.miniSeverityText}>
                    {getSeverity(result.results.caterpillar.confidence, true)?.icon} {getSeverity(result.results.caterpillar.confidence, true)?.label}
                  </Text>
                </View>
              )}
            </View>
          )}

          {/* White Fly Result */}
          {result.results?.white_fly && (
            <View style={styles.pestResultCard}>
              <Text style={styles.pestResultIcon}>🦟</Text>
              <Text style={styles.pestResultName}>{t('pestDetection.whiteFly') || 'White Fly'}</Text>
              <Text
                style={[
                  styles.pestResultStatus,
                  {color: result.results.white_fly.class === 'not_coconut' ? '#ff9800' : result.results.white_fly.is_infected ? '#d32f2f' : '#2e7d32'},
                ]}>
                {result.results.white_fly.class === 'not_coconut' ? 'N/A' : result.results.white_fly.is_infected ? t('pestDetection.detected') : t('pestDetection.notFound')}
              </Text>
              <Text style={styles.pestResultConfidence}>
                {((result.results.white_fly.confidence || 0) * 100).toFixed(1)}%
              </Text>
              {/* White Fly Severity */}
              {result.results.white_fly.is_infected && (
                <View style={[styles.miniSeverityBadge, {backgroundColor: getSeverity(result.results.white_fly.confidence, true)?.color}]}>
                  <Text style={styles.miniSeverityText}>
                    {getSeverity(result.results.white_fly.confidence, true)?.icon} {getSeverity(result.results.white_fly.confidence, true)?.label}
                  </Text>
                </View>
              )}
            </View>
          )}
        </View>

        {/* Overall Severity Summary */}
        {!result.summary?.is_healthy && (
          <View style={styles.overallSeverityContainer}>
            <Text style={styles.overallSeverityTitle}>{t('pestDetection.severity')} {t('pestDetection.result')}</Text>
            {result.results?.mite?.is_infected && renderSeverityIndicator(getSeverity(result.results.mite.confidence, true))}
            {result.results?.caterpillar?.is_infected && renderSeverityIndicator(getSeverity(result.results.caterpillar.confidence, true))}
            {result.results?.white_fly?.is_infected && renderSeverityIndicator(getSeverity(result.results.white_fly.confidence, true))}
          </View>
        )}

        {/* Recommendation from API */}
        {result.summary?.recommendation && (
          <View style={[styles.recommendationBox, !result.summary?.is_healthy && styles.warningRecommendation]}>
            <Text style={[styles.recommendationTitle, !result.summary?.is_healthy && styles.warningRecommendationTitle]}>
              {result.summary?.is_healthy ? '✓ ' + t('pestDetection.recommendation') : '⚠️ ' + t('pestDetection.actionRequired')}
            </Text>
            <Text style={[styles.recommendationText, !result.summary?.is_healthy && styles.warningRecommendationText]}>
              {result.summary.recommendation}
            </Text>
          </View>
        )}

        {/* Pests detected list */}
        {result.summary?.pests_detected && result.summary.pests_detected.length > 0 && (
          <View style={styles.pestsDetectedBox}>
            <Text style={styles.pestsDetectedTitle}>{t('pestDetection.pestsDetected')}:</Text>
            <Text style={styles.pestsDetectedText}>
              {result.summary.pests_detected.join(', ')}
            </Text>
          </View>
        )}
      </View>
    );
  };

  // Get urgency color
  const getUrgencyColor = (urgency) => {
    switch (urgency) {
      case 'low': return '#4caf50';
      case 'medium': return '#ff9800';
      case 'high': return '#f44336';
      case 'critical': return '#9c27b0';
      default: return '#666';
    }
  };

  // Get urgency label
  const getUrgencyLabel = (urgency) => {
    switch (urgency) {
      case 'low': return t('pestDetection.urgencyLow');
      case 'medium': return t('pestDetection.urgencyMedium');
      case 'high': return t('pestDetection.urgencyHigh');
      case 'critical': return t('pestDetection.urgencyCritical');
      default: return urgency;
    }
  };

  // Render treatment button
  const renderTreatmentButton = () => {
    // Only show if pest is detected
    const isInfected = selectedPestType === PEST_TYPES.ALL
      ? !result?.summary?.is_healthy
      : result?.prediction?.is_infected;

    if (!result || !isInfected) return null;

    const getPestTypeForTreatment = () => {
      if (selectedPestType === PEST_TYPES.MITE) return 'coconut_mite';
      if (selectedPestType === PEST_TYPES.CATERPILLAR) return 'caterpillar';
      if (selectedPestType === PEST_TYPES.WHITE_FLY) return 'white_fly';
      // For ALL, pick the first detected pest
      if (result?.results?.mite?.is_infected) return 'coconut_mite';
      if (result?.results?.caterpillar?.is_infected) return 'caterpillar';
      if (result?.results?.white_fly?.is_infected) return 'white_fly';
      return 'coconut_mite';
    };

    const getConfidenceForTreatment = () => {
      if (selectedPestType !== PEST_TYPES.ALL) {
        return result?.prediction?.confidence || 0.7;
      }
      if (result?.results?.mite?.is_infected) return result.results.mite.confidence;
      if (result?.results?.caterpillar?.is_infected) return result.results.caterpillar.confidence;
      if (result?.results?.white_fly?.is_infected) return result.results.white_fly.confidence;
      return 0.7;
    };

    const severity = getSeverity(getConfidenceForTreatment(), true);

    return (
      <TouchableOpacity
        style={[styles.treatmentButton, isLoadingTreatment && styles.buttonDisabled]}
        onPress={() => fetchTreatment(getPestTypeForTreatment(), severity, getConfidenceForTreatment())}
        disabled={isLoadingTreatment}>
        {isLoadingTreatment ? (
          <View style={styles.treatmentButtonContent}>
            <ActivityIndicator color="#fff" size="small" />
            <Text style={styles.treatmentButtonText}>{t('pestDetection.loadingTreatment')}</Text>
          </View>
        ) : (
          <View style={styles.treatmentButtonContent}>
            <Text style={styles.treatmentButtonIcon}>💊</Text>
            <Text style={styles.treatmentButtonText}>{t('pestDetection.getTreatment')}</Text>
          </View>
        )}
      </TouchableOpacity>
    );
  };

  // Render treatment results
  const renderTreatmentResults = () => {
    if (!treatment) return null;

    return (
      <View style={styles.treatmentContainer}>
        <View style={styles.treatmentHeader}>
          <Text style={styles.treatmentTitle}>💊 {t('pestDetection.treatmentPlan')}</Text>
          {treatment.source === 'ai' ? (
            <View style={styles.aiPoweredBadge}>
              <Text style={styles.aiPoweredText}>🤖 {t('pestDetection.poweredByAI')}</Text>
            </View>
          ) : (
            <View style={styles.fallbackBadge}>
              <Text style={styles.fallbackText}>📋 {t('pestDetection.usingFallback')}</Text>
            </View>
          )}
        </View>

        {/* Summary */}
        {treatment.summary && (
          <Text style={styles.treatmentSummary}>{treatment.summary}</Text>
        )}

        {/* Urgency */}
        {treatment.urgency && (
          <View style={[styles.urgencyBadge, {backgroundColor: getUrgencyColor(treatment.urgency)}]}>
            <Text style={styles.urgencyText}>{getUrgencyLabel(treatment.urgency)}</Text>
          </View>
        )}

        {/* Treatments */}
        {treatment.treatments && treatment.treatments.length > 0 && (
          <View style={styles.treatmentsSection}>
            {treatment.treatments.map((item, index) => (
              <View key={index} style={styles.treatmentCard}>
                <View style={styles.treatmentCardHeader}>
                  <Text style={styles.treatmentCardIcon}>
                    {item.type === 'chemical' ? '🧪' : item.type === 'organic' ? '🌿' : '🌾'}
                  </Text>
                  <View style={styles.treatmentCardTitleContainer}>
                    <Text style={styles.treatmentCardType}>
                      {item.type === 'chemical' ? t('pestDetection.chemicalTreatment') :
                       item.type === 'organic' ? t('pestDetection.organicTreatment') :
                       t('pestDetection.culturalTreatment')}
                    </Text>
                    <Text style={styles.treatmentCardName}>{item.name}</Text>
                  </View>
                </View>
                {item.description && (
                  <Text style={styles.treatmentCardDesc}>{item.description}</Text>
                )}
                <View style={styles.treatmentCardDetails}>
                  {item.dosage && (
                    <View style={styles.treatmentDetail}>
                      <Text style={styles.treatmentDetailLabel}>{t('pestDetection.dosage')}:</Text>
                      <Text style={styles.treatmentDetailValue}>{item.dosage}</Text>
                    </View>
                  )}
                  {item.frequency && (
                    <View style={styles.treatmentDetail}>
                      <Text style={styles.treatmentDetailLabel}>{t('pestDetection.frequency')}:</Text>
                      <Text style={styles.treatmentDetailValue}>{item.frequency}</Text>
                    </View>
                  )}
                  {item.duration && (
                    <View style={styles.treatmentDetail}>
                      <Text style={styles.treatmentDetailLabel}>{t('pestDetection.duration')}:</Text>
                      <Text style={styles.treatmentDetailValue}>{item.duration}</Text>
                    </View>
                  )}
                  {item.cost_estimate && (
                    <View style={styles.treatmentDetail}>
                      <Text style={styles.treatmentDetailLabel}>{t('pestDetection.estimatedCost')}:</Text>
                      <Text style={styles.treatmentDetailValue}>{item.cost_estimate}</Text>
                    </View>
                  )}
                </View>
              </View>
            ))}
          </View>
        )}

        {/* Preventive Measures */}
        {treatment.preventive_measures && treatment.preventive_measures.length > 0 && (
          <View style={styles.preventiveSection}>
            <Text style={styles.sectionTitle}>🛡️ {t('pestDetection.preventiveMeasures')}</Text>
            {treatment.preventive_measures.map((measure, index) => (
              <View key={index} style={styles.listItem}>
                <Text style={styles.listBullet}>•</Text>
                <Text style={styles.listText}>{measure}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Safety Precautions */}
        {treatment.safety_precautions && treatment.safety_precautions.length > 0 && (
          <View style={styles.safetySection}>
            <Text style={styles.sectionTitle}>⚠️ {t('pestDetection.safetyPrecautions')}</Text>
            {treatment.safety_precautions.map((precaution, index) => (
              <View key={index} style={styles.listItem}>
                <Text style={styles.listBullet}>•</Text>
                <Text style={styles.listText}>{precaution}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Recovery & Expert */}
        <View style={styles.recoverySection}>
          {treatment.expected_recovery && (
            <View style={styles.recoveryItem}>
              <Text style={styles.recoveryLabel}>⏱️ {t('pestDetection.expectedRecovery')}:</Text>
              <Text style={styles.recoveryValue}>{treatment.expected_recovery}</Text>
            </View>
          )}
          {treatment.when_to_seek_expert && (
            <View style={styles.recoveryItem}>
              <Text style={styles.recoveryLabel}>👨‍⚕️ {t('pestDetection.whenToSeekExpert')}:</Text>
              <Text style={styles.recoveryValue}>{treatment.when_to_seek_expert}</Text>
            </View>
          )}
        </View>
      </View>
    );
  };

  // Render API Key Modal
  const renderApiKeyModal = () => (
    <Modal
      visible={showApiKeyModal}
      transparent
      animationType="fade"
      onRequestClose={() => setShowApiKeyModal(false)}>
      <View style={styles.modalOverlay}>
        <View style={styles.modalContent}>
          <Text style={styles.modalTitle}>{t('pestDetection.apiKeyRequired')}</Text>
          <Text style={styles.modalDescription}>{t('pestDetection.apiKeyDescription')}</Text>
          <TextInput
            style={styles.apiKeyInput}
            placeholder={t('pestDetection.apiKeyPlaceholder')}
            placeholderTextColor="#999"
            value={apiKeyInput}
            onChangeText={setApiKeyInput}
            secureTextEntry
            autoCapitalize="none"
          />
          <View style={styles.modalButtons}>
            <TouchableOpacity
              style={styles.modalCancelButton}
              onPress={() => setShowApiKeyModal(false)}>
              <Text style={styles.modalCancelText}>{t('common.cancel')}</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.modalSaveButton}
              onPress={handleSaveApiKey}>
              <Text style={styles.modalSaveText}>{t('pestDetection.saveApiKey')}</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    </Modal>
  );

  return (
    <ScrollView style={styles.container}>
      {/* API Key Modal */}
      {renderApiKeyModal()}

      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>← {t('common.back')}</Text>
        </TouchableOpacity>
        <Text style={styles.title}>{t('pestDetection.title')}</Text>
        <View
          style={[
            styles.statusBadge,
            {backgroundColor: apiStatus === 'online' ? '#4caf50' : '#f44336'},
          ]}>
          <Text style={styles.statusText}>
            {apiStatus === 'checking' ? '...' : apiStatus === 'online' ? 'API Online' : 'API Offline'}
          </Text>
        </View>
      </View>

      {/* Pest Type Selector */}
      {renderPestTypeSelector()}

      {/* Image Preview */}
      <View style={styles.imageContainer}>
        {selectedImage ? (
          <Image source={{uri: selectedImage.uri}} style={styles.previewImage} />
        ) : (
          <View style={styles.placeholderContainer}>
            <Text style={styles.placeholderIcon}>🌴</Text>
            <Text style={styles.placeholderText}>{t('pestDetection.selectImage')}</Text>
          </View>
        )}
      </View>

      {/* Action Buttons */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.selectButton} onPress={selectImage}>
          <Text style={styles.selectButtonText}>📷 {t('pestDetection.selectImage')}</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.analyzeButton, (!selectedImage || isAnalyzing) && styles.buttonDisabled]}
          onPress={analyzeImage}
          disabled={!selectedImage || isAnalyzing}>
          {isAnalyzing ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.analyzeButtonText}>🔬 {t('pestDetection.detectAll')}</Text>
          )}
        </TouchableOpacity>
      </View>

      {/* Results */}
      {result && (selectedPestType === PEST_TYPES.ALL ? renderAllPestsResult() : renderSingleResult())}

      {/* Treatment Button */}
      {renderTreatmentButton()}

      {/* Treatment Results */}
      {renderTreatmentResults()}

      {/* Info Section */}
      <View style={styles.infoContainer}>
        <Text style={styles.infoTitle}>{t('pestDetection.aboutFeature')}</Text>
        <Text style={styles.infoText}>
          {t('pestDetection.aboutDescription')}{'\n'}
          • {t('pestDetection.miteAccuracy')}{'\n'}
          • {t('pestDetection.caterpillarAccuracy')}{'\n'}
          • {t('pestDetection.whiteFlyAccuracy') || 'White Fly: 98.06% accuracy'}{'\n'}
          • {t('pestDetection.detectsNonCoconut')}
        </Text>
      </View>
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
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  statusText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  pestTypeContainer: {
    backgroundColor: '#fff',
    padding: 15,
    marginBottom: 10,
  },
  pestTypeTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  pestTypeButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  pestTypeButton: {
    flex: 1,
    alignItems: 'center',
    padding: 10,
    marginHorizontal: 5,
    backgroundColor: '#f5f5f5',
    borderRadius: 10,
    borderWidth: 2,
    borderColor: 'transparent',
  },
  pestTypeButtonActive: {
    backgroundColor: '#e8f5e9',
    borderColor: '#2e7d32',
  },
  pestTypeIcon: {
    fontSize: 24,
    marginBottom: 5,
  },
  pestTypeText: {
    fontSize: 12,
    color: '#666',
  },
  pestTypeTextActive: {
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
    borderColor: '#2e7d32',
  },
  selectButtonText: {
    color: '#2e7d32',
    fontSize: 16,
    fontWeight: 'bold',
  },
  analyzeButton: {
    backgroundColor: '#2e7d32',
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
    padding: 20,
    backgroundColor: '#fff',
    borderRadius: 15,
    borderWidth: 2,
    alignItems: 'center',
  },
  resultIcon: {
    fontSize: 50,
    marginBottom: 10,
  },
  resultTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  confidenceText: {
    fontSize: 16,
    color: '#666',
    marginBottom: 15,
  },
  probabilitiesContainer: {
    width: '100%',
    marginTop: 10,
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
    width: 100,
    fontSize: 12,
    color: '#666',
  },
  probabilityBarContainer: {
    flex: 1,
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    marginHorizontal: 10,
  },
  probabilityBar: {
    height: '100%',
    backgroundColor: '#2e7d32',
    borderRadius: 4,
  },
  probabilityValue: {
    width: 50,
    fontSize: 12,
    color: '#333',
    textAlign: 'right',
  },
  allResultsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
    marginTop: 15,
  },
  pestResultCard: {
    alignItems: 'center',
    padding: 15,
    backgroundColor: '#f9f9f9',
    borderRadius: 10,
    minWidth: 120,
  },
  pestResultIcon: {
    fontSize: 30,
    marginBottom: 5,
  },
  pestResultName: {
    fontSize: 12,
    color: '#666',
    marginBottom: 5,
  },
  pestResultStatus: {
    fontSize: 14,
    fontWeight: 'bold',
  },
  pestResultConfidence: {
    fontSize: 11,
    color: '#999',
    marginTop: 3,
  },
  warningBox: {
    marginTop: 15,
    padding: 15,
    backgroundColor: '#fff3e0',
    borderRadius: 10,
    width: '100%',
  },
  warningTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#e65100',
    marginBottom: 5,
  },
  warningText: {
    fontSize: 13,
    color: '#bf360c',
  },
  infoContainer: {
    margin: 15,
    padding: 15,
    backgroundColor: '#e3f2fd',
    borderRadius: 10,
    marginBottom: 30,
  },
  infoTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#1565c0',
    marginBottom: 5,
  },
  infoText: {
    fontSize: 13,
    color: '#1976d2',
    lineHeight: 20,
  },
  // New styles for invalid image handling
  invalidImageBox: {
    marginTop: 15,
    padding: 15,
    backgroundColor: '#fff8e1',
    borderRadius: 10,
    width: '100%',
    borderWidth: 1,
    borderColor: '#ffcc02',
  },
  invalidImageTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#f57c00',
    marginBottom: 5,
  },
  invalidImageText: {
    fontSize: 13,
    color: '#ef6c00',
    lineHeight: 18,
  },
  // Recommendation box styles
  recommendationBox: {
    marginTop: 15,
    padding: 15,
    backgroundColor: '#e8f5e9',
    borderRadius: 10,
    width: '100%',
  },
  recommendationTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#2e7d32',
    marginBottom: 5,
  },
  recommendationText: {
    fontSize: 13,
    color: '#1b5e20',
    lineHeight: 18,
  },
  warningRecommendation: {
    backgroundColor: '#fff3e0',
  },
  warningRecommendationTitle: {
    color: '#e65100',
  },
  warningRecommendationText: {
    color: '#bf360c',
  },
  // Summary message style
  summaryMessage: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 10,
  },
  // Pests detected box
  pestsDetectedBox: {
    marginTop: 10,
    padding: 10,
    backgroundColor: '#ffebee',
    borderRadius: 8,
    width: '100%',
    flexDirection: 'row',
    alignItems: 'center',
  },
  pestsDetectedTitle: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#c62828',
    marginRight: 5,
  },
  pestsDetectedText: {
    fontSize: 12,
    color: '#d32f2f',
    flex: 1,
  },
  // Severity indicator styles
  severityContainer: {
    width: '100%',
    marginTop: 15,
    padding: 15,
    backgroundColor: '#fafafa',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  severityHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  severityLabel: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
  },
  severityBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  severityBadgeText: {
    color: '#fff',
    fontSize: 13,
    fontWeight: 'bold',
  },
  severityBarContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  severityBarBackground: {
    flex: 1,
    height: 10,
    backgroundColor: '#e0e0e0',
    borderRadius: 5,
    overflow: 'hidden',
  },
  severityBarFill: {
    height: '100%',
    borderRadius: 5,
  },
  severityPercent: {
    marginLeft: 10,
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    minWidth: 45,
  },
  severityDescription: {
    fontSize: 12,
    fontStyle: 'italic',
    marginTop: 5,
  },
  // Mini severity badge for pest cards
  miniSeverityBadge: {
    marginTop: 8,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  miniSeverityText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  // Overall severity container
  overallSeverityContainer: {
    width: '100%',
    marginTop: 15,
    padding: 10,
    backgroundColor: '#f5f5f5',
    borderRadius: 10,
  },
  overallSeverityTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
    textAlign: 'center',
  },
  // Treatment styles
  treatmentButton: {
    margin: 15,
    padding: 15,
    backgroundColor: '#1565c0',
    borderRadius: 10,
    alignItems: 'center',
  },
  treatmentButtonContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  treatmentButtonIcon: {
    fontSize: 20,
    marginRight: 10,
  },
  treatmentButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  treatmentContainer: {
    margin: 15,
    padding: 15,
    backgroundColor: '#fff',
    borderRadius: 15,
    borderWidth: 2,
    borderColor: '#1565c0',
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
    color: '#1565c0',
  },
  aiPoweredBadge: {
    backgroundColor: '#e3f2fd',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 15,
  },
  aiPoweredText: {
    fontSize: 10,
    color: '#1565c0',
    fontWeight: 'bold',
  },
  fallbackBadge: {
    backgroundColor: '#fff3e0',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 15,
  },
  fallbackText: {
    fontSize: 10,
    color: '#ff9800',
    fontWeight: 'bold',
  },
  treatmentSummary: {
    fontSize: 14,
    color: '#333',
    marginBottom: 15,
    lineHeight: 20,
  },
  urgencyBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    marginBottom: 15,
  },
  urgencyText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  treatmentsSection: {
    marginBottom: 15,
  },
  treatmentCard: {
    backgroundColor: '#f5f5f5',
    borderRadius: 10,
    padding: 15,
    marginBottom: 10,
  },
  treatmentCardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  treatmentCardIcon: {
    fontSize: 30,
    marginRight: 10,
  },
  treatmentCardTitleContainer: {
    flex: 1,
  },
  treatmentCardType: {
    fontSize: 11,
    color: '#666',
    textTransform: 'uppercase',
  },
  treatmentCardName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
  },
  treatmentCardDesc: {
    fontSize: 13,
    color: '#666',
    marginBottom: 10,
    lineHeight: 18,
  },
  treatmentCardDetails: {
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 10,
  },
  treatmentDetail: {
    flexDirection: 'row',
    marginBottom: 5,
  },
  treatmentDetailLabel: {
    fontSize: 12,
    color: '#666',
    width: 100,
  },
  treatmentDetailValue: {
    fontSize: 12,
    color: '#333',
    flex: 1,
    fontWeight: '500',
  },
  preventiveSection: {
    backgroundColor: '#e8f5e9',
    borderRadius: 10,
    padding: 15,
    marginBottom: 15,
  },
  safetySection: {
    backgroundColor: '#fff3e0',
    borderRadius: 10,
    padding: 15,
    marginBottom: 15,
  },
  sectionTitle: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  listItem: {
    flexDirection: 'row',
    marginBottom: 5,
  },
  listBullet: {
    fontSize: 12,
    color: '#666',
    marginRight: 8,
  },
  listText: {
    fontSize: 12,
    color: '#333',
    flex: 1,
    lineHeight: 18,
  },
  recoverySection: {
    backgroundColor: '#f5f5f5',
    borderRadius: 10,
    padding: 15,
  },
  recoveryItem: {
    marginBottom: 10,
  },
  recoveryLabel: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 3,
  },
  recoveryValue: {
    fontSize: 12,
    color: '#666',
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
    backgroundColor: '#fff',
    borderRadius: 15,
    padding: 20,
    width: '100%',
    maxWidth: 350,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  modalDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 15,
    lineHeight: 20,
  },
  apiKeyInput: {
    backgroundColor: '#f5f5f5',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 10,
    padding: 15,
    fontSize: 14,
    color: '#333',
    marginBottom: 15,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  modalCancelButton: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    marginRight: 10,
  },
  modalCancelText: {
    color: '#666',
    fontSize: 14,
  },
  modalSaveButton: {
    backgroundColor: '#1565c0',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 8,
  },
  modalSaveText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
});
