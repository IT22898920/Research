import React, {useState} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Image,
  ScrollView,
  ActivityIndicator,
  Alert,
} from 'react-native';
import {launchCamera, launchImageLibrary} from 'react-native-image-picker';
import {useLanguage} from '../context/LanguageContext';
import {detectDisease} from '../services/pestDetectionApi';

export default function DiseaseDetectionScreen({navigation}) {
  const {t} = useLanguage();
  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  const handleTakePhoto = () => {
    launchCamera(
      {
        mediaType: 'photo',
        quality: 0.8,
        maxWidth: 1024,
        maxHeight: 1024,
      },
      response => {
        if (response.didCancel) return;
        if (response.errorCode) {
          Alert.alert('Error', response.errorMessage);
          return;
        }
        if (response.assets && response.assets[0]) {
          setSelectedImage(response.assets[0].uri);
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
        maxWidth: 1024,
        maxHeight: 1024,
      },
      response => {
        if (response.didCancel) return;
        if (response.errorCode) {
          Alert.alert('Error', response.errorMessage);
          return;
        }
        if (response.assets && response.assets[0]) {
          setSelectedImage(response.assets[0].uri);
          setResult(null);
        }
      },
    );
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    setIsAnalyzing(true);
    try {
      const response = await detectDisease(selectedImage);

      if (response.success) {
        setResult(response);
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
        return '🦠';
      case 'healthy':
        return '✅';
      case 'invalid':
        return '❌';
      default:
        return '❓';
    }
  };

  const getRecommendation = prediction => {
    if (prediction.is_leaf_rot) {
      return t('diseaseDetection.leafRotDesc');
    } else if (prediction.is_leaf_spot) {
      return t('diseaseDetection.leafSpotDesc');
    } else if (prediction.is_healthy) {
      return t('diseaseDetection.healthyDesc');
    }
    return '';
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}>
          <Text style={styles.backButtonText}>← {t('common.back')}</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>{t('diseaseDetection.title')}</Text>
      </View>

      {/* Image Selection Area */}
      {!selectedImage ? (
        <View style={styles.selectionContainer}>
          <Text style={styles.selectionIcon}>🍃</Text>
          <Text style={styles.selectionTitle}>
            {t('diseaseDetection.selectImage')}
          </Text>
          <Text style={styles.selectionSubtitle}>
            {t('diseaseDetection.aboutDescription')}
          </Text>

          <TouchableOpacity
            style={styles.actionButton}
            onPress={handleTakePhoto}>
            <Text style={styles.actionButtonIcon}>📷</Text>
            <Text style={styles.actionButtonText}>
              {t('diseaseDetection.takePhoto')}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, styles.secondaryButton]}
            onPress={handleChooseFromGallery}>
            <Text style={styles.actionButtonIcon}>🖼️</Text>
            <Text style={styles.actionButtonText}>
              {t('diseaseDetection.chooseGallery')}
            </Text>
          </TouchableOpacity>

          {/* About Section */}
          <View style={styles.aboutSection}>
            <Text style={styles.aboutTitle}>
              {t('diseaseDetection.aboutFeature')}
            </Text>
            <Text style={styles.aboutItem}>
              • {t('diseaseDetection.detectsLeafRot')}
            </Text>
            <Text style={styles.aboutItem}>
              • {t('diseaseDetection.detectsLeafSpot')}
            </Text>
            <Text style={styles.aboutItem}>
              • {t('diseaseDetection.detectsNonCoconut')}
            </Text>
          </View>
        </View>
      ) : (
        <View style={styles.analysisContainer}>
          {/* Selected Image */}
          <Image source={{uri: selectedImage}} style={styles.selectedImage} />

          {/* Analyze Button or Loading */}
          {!result && !isAnalyzing && (
            <TouchableOpacity
              style={styles.analyzeButton}
              onPress={analyzeImage}>
              <Text style={styles.analyzeButtonText}>
                🔬 {t('diseaseDetection.analyzing').replace('...', '')}
              </Text>
            </TouchableOpacity>
          )}

          {isAnalyzing && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#2e7d32" />
              <Text style={styles.loadingText}>
                {t('diseaseDetection.analyzing')}
              </Text>
            </View>
          )}

          {/* Results */}
          {result && result.prediction && (
            <View style={styles.resultContainer}>
              <View
                style={[
                  styles.resultHeader,
                  {
                    backgroundColor: getStatusColor(result.prediction.status),
                  },
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
                <View style={styles.recommendationBox}>
                  <Text style={styles.recommendationTitle}>
                    {t('diseaseDetection.recommendation')}
                  </Text>
                  <Text style={styles.recommendationText}>
                    {getRecommendation(result.prediction)}
                  </Text>
                </View>
              )}

              {/* Probabilities */}
              {result.probabilities && (
                <View style={styles.probabilitiesBox}>
                  <Text style={styles.probabilitiesTitle}>
                    {t('diseaseDetection.probabilities')}
                  </Text>
                  <View style={styles.probabilityRow}>
                    <Text style={styles.probabilityLabel}>
                      {t('diseaseDetection.leafRot')}:
                    </Text>
                    <View style={styles.probabilityBarContainer}>
                      <View
                        style={[
                          styles.probabilityBar,
                          {
                            width: `${result.probabilities.leaf_rot * 100}%`,
                            backgroundColor: '#d32f2f',
                          },
                        ]}
                      />
                    </View>
                    <Text style={styles.probabilityValue}>
                      {(result.probabilities.leaf_rot * 100).toFixed(1)}%
                    </Text>
                  </View>
                  <View style={styles.probabilityRow}>
                    <Text style={styles.probabilityLabel}>
                      {t('diseaseDetection.leafSpot')}:
                    </Text>
                    <View style={styles.probabilityBarContainer}>
                      <View
                        style={[
                          styles.probabilityBar,
                          {
                            width: `${result.probabilities.leaf_spot * 100}%`,
                            backgroundColor: '#ff9800',
                          },
                        ]}
                      />
                    </View>
                    <Text style={styles.probabilityValue}>
                      {(result.probabilities.leaf_spot * 100).toFixed(1)}%
                    </Text>
                  </View>
                  <View style={styles.probabilityRow}>
                    <Text style={styles.probabilityLabel}>
                      {t('diseaseDetection.healthy')}:
                    </Text>
                    <View style={styles.probabilityBarContainer}>
                      <View
                        style={[
                          styles.probabilityBar,
                          {
                            width: `${result.probabilities.healthy * 100}%`,
                            backgroundColor: '#2e7d32',
                          },
                        ]}
                      />
                    </View>
                    <Text style={styles.probabilityValue}>
                      {(result.probabilities.healthy * 100).toFixed(1)}%
                    </Text>
                  </View>
                </View>
              )}

              {/* Scan Again Button */}
              <TouchableOpacity style={styles.resetButton} onPress={resetScan}>
                <Text style={styles.resetButtonText}>
                  🔄 {t('diseaseDetection.scanAgain')}
                </Text>
              </TouchableOpacity>
            </View>
          )}

          {/* Reset button when no result yet */}
          {!result && !isAnalyzing && (
            <TouchableOpacity
              style={[styles.resetButton, styles.secondaryButton]}
              onPress={resetScan}>
              <Text style={styles.resetButtonText}>
                ← {t('common.back')}
              </Text>
            </TouchableOpacity>
          )}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    backgroundColor: '#2e7d32',
    padding: 20,
    paddingTop: 50,
    flexDirection: 'row',
    alignItems: 'center',
  },
  backButton: {
    marginRight: 15,
  },
  backButtonText: {
    color: '#fff',
    fontSize: 16,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
  },
  selectionContainer: {
    padding: 20,
    alignItems: 'center',
  },
  selectionIcon: {
    fontSize: 80,
    marginBottom: 20,
  },
  selectionTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
    textAlign: 'center',
  },
  selectionSubtitle: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 30,
    paddingHorizontal: 20,
  },
  actionButton: {
    backgroundColor: '#2e7d32',
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 10,
    width: '100%',
    marginBottom: 15,
  },
  secondaryButton: {
    backgroundColor: '#fff',
    borderWidth: 2,
    borderColor: '#2e7d32',
  },
  actionButtonIcon: {
    fontSize: 24,
    marginRight: 10,
  },
  actionButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
  },
  aboutSection: {
    backgroundColor: '#e8f5e9',
    padding: 20,
    borderRadius: 10,
    width: '100%',
    marginTop: 20,
  },
  aboutTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2e7d32',
    marginBottom: 10,
  },
  aboutItem: {
    fontSize: 14,
    color: '#333',
    marginBottom: 5,
  },
  analysisContainer: {
    padding: 20,
  },
  selectedImage: {
    width: '100%',
    height: 300,
    borderRadius: 15,
    marginBottom: 20,
  },
  analyzeButton: {
    backgroundColor: '#2e7d32',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginBottom: 15,
  },
  analyzeButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  loadingContainer: {
    alignItems: 'center',
    padding: 30,
  },
  loadingText: {
    marginTop: 15,
    fontSize: 16,
    color: '#666',
  },
  resultContainer: {
    backgroundColor: '#fff',
    borderRadius: 15,
    overflow: 'hidden',
    marginBottom: 20,
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
  },
  recommendationTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#2e7d32',
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
    width: 80,
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
    backgroundColor: '#2e7d32',
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 15,
  },
  resetButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
