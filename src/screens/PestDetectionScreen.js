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
} from 'react-native';
import {launchImageLibrary, launchCamera} from 'react-native-image-picker';
import {
  checkApiHealth,
  detectMite,
  detectCaterpillar,
  detectAllPests,
  PEST_TYPES,
} from '../services/pestDetectionApi';

export default function PestDetectionScreen({navigation}) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');
  const [selectedPestType, setSelectedPestType] = useState(PEST_TYPES.ALL);

  useEffect(() => {
    checkApi();
  }, []);

  const checkApi = async () => {
    const health = await checkApiHealth();
    setApiStatus(health.success ? 'online' : 'offline');
  };

  const selectImage = () => {
    Alert.alert('Select Image', 'Choose how to select an image', [
      {text: 'Camera', onPress: () => openCamera()},
      {text: 'Gallery', onPress: () => openGallery()},
      {text: 'Cancel', style: 'cancel'},
    ]);
  };

  const openCamera = () => {
    launchCamera(
      {mediaType: 'photo', quality: 0.8, maxWidth: 1024, maxHeight: 1024},
      handleImageResponse,
    );
  };

  const openGallery = () => {
    launchImageLibrary(
      {mediaType: 'photo', quality: 0.8, maxWidth: 1024, maxHeight: 1024},
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
        case PEST_TYPES.ALL:
        default:
          response = await detectAllPests(selectedImage.uri);
          break;
      }

      if (response.success) {
        setResult({...response, pestType: selectedPestType});
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
      return result.summary?.is_healthy ? '#2e7d32' : '#d32f2f';
    }
    return result.prediction?.is_infected ? '#d32f2f' : '#2e7d32';
  };

  const getResultIcon = () => {
    if (!result) return '🔍';
    if (selectedPestType === PEST_TYPES.ALL) {
      return result.summary?.is_healthy ? '✅' : '🐛';
    }
    return result.prediction?.is_infected ? '🐛' : '✅';
  };

  const renderPestTypeSelector = () => (
    <View style={styles.pestTypeContainer}>
      <Text style={styles.pestTypeTitle}>Select Detection Type:</Text>
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
            All Pests
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
            Mite
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
            Caterpillar
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );

  const renderSingleResult = () => {
    if (!result?.prediction) return null;

    return (
      <View style={[styles.resultContainer, {borderColor: getResultColor()}]}>
        <Text style={styles.resultIcon}>{getResultIcon()}</Text>
        <Text style={[styles.resultTitle, {color: getResultColor()}]}>
          {result.prediction?.label || (result.prediction?.is_infected ? 'Pest Detected' : 'Healthy')}
        </Text>
        <Text style={styles.confidenceText}>
          Confidence: {((result.prediction?.confidence || 0) * 100).toFixed(1)}%
        </Text>

        {result.probabilities && (
          <View style={styles.probabilitiesContainer}>
            <Text style={styles.probabilitiesTitle}>Probabilities:</Text>
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
            <Text style={styles.warningTitle}>⚠️ Action Required</Text>
            <Text style={styles.warningText}>
              {result.pestType === PEST_TYPES.MITE
                ? 'Coconut mite infection detected. Consider applying appropriate pesticide treatment.'
                : 'Caterpillar damage detected. Consider applying appropriate pest control measures.'}
            </Text>
          </View>
        )}
      </View>
    );
  };

  const renderAllPestsResult = () => {
    if (!result?.results) return null;

    return (
      <View style={[styles.resultContainer, {borderColor: getResultColor()}]}>
        <Text style={styles.resultIcon}>{getResultIcon()}</Text>
        <Text style={[styles.resultTitle, {color: getResultColor()}]}>
          {result.summary?.label || (result.summary?.is_healthy ? 'Healthy' : 'Pest Detected')}
        </Text>

      {/* Individual Results */}
      <View style={styles.allResultsContainer}>
        {/* Mite Result */}
        {result.results?.mite && (
          <View style={styles.pestResultCard}>
            <Text style={styles.pestResultIcon}>🕷️</Text>
            <Text style={styles.pestResultName}>Coconut Mite</Text>
            <Text
              style={[
                styles.pestResultStatus,
                {color: result.results.mite.is_infected ? '#d32f2f' : '#2e7d32'},
              ]}>
              {result.results.mite.is_infected ? 'DETECTED' : 'Not Found'}
            </Text>
            <Text style={styles.pestResultConfidence}>
              {((result.results.mite.confidence || 0) * 100).toFixed(1)}%
            </Text>
          </View>
        )}

        {/* Caterpillar Result */}
        {result.results?.caterpillar && (
          <View style={styles.pestResultCard}>
            <Text style={styles.pestResultIcon}>🐛</Text>
            <Text style={styles.pestResultName}>Caterpillar</Text>
            <Text
              style={[
                styles.pestResultStatus,
                {color: result.results.caterpillar.is_infected ? '#d32f2f' : '#2e7d32'},
              ]}>
              {result.results.caterpillar.is_infected ? 'DETECTED' : 'Not Found'}
            </Text>
            <Text style={styles.pestResultConfidence}>
              {((result.results.caterpillar.confidence || 0) * 100).toFixed(1)}%
            </Text>
          </View>
        )}
      </View>

      {/* Warning if pests detected */}
      {result.summary && !result.summary.is_healthy && (
        <View style={styles.warningBox}>
          <Text style={styles.warningTitle}>⚠️ Action Required</Text>
          <Text style={styles.warningText}>
            Detected: {result.summary.pests_detected?.join(', ') || 'Unknown pest'}.
            Consider applying appropriate pest control measures.
          </Text>
        </View>
      )}
      </View>
    );
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Pest Detection</Text>
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
            <Text style={styles.placeholderText}>Select a coconut image to analyze</Text>
          </View>
        )}
      </View>

      {/* Action Buttons */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.selectButton} onPress={selectImage}>
          <Text style={styles.selectButtonText}>📷 Select Image</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.analyzeButton, (!selectedImage || isAnalyzing) && styles.buttonDisabled]}
          onPress={analyzeImage}
          disabled={!selectedImage || isAnalyzing}>
          {isAnalyzing ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.analyzeButtonText}>🔬 Analyze for Pests</Text>
          )}
        </TouchableOpacity>
      </View>

      {/* Results */}
      {result && (selectedPestType === PEST_TYPES.ALL ? renderAllPestsResult() : renderSingleResult())}

      {/* Info Section */}
      <View style={styles.infoContainer}>
        <Text style={styles.infoTitle}>About This Feature</Text>
        <Text style={styles.infoText}>
          AI-powered pest detection using MobileNetV2 and EfficientNetB0 models.{'\n'}
          • Mite Detection: 95%+ accuracy{'\n'}
          • Caterpillar Detection: 99%+ accuracy
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
});
