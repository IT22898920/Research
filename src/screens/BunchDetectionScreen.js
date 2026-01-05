/**
 * Bunch Detection Screen
 * Detects coconut bunches from drone images for yield prediction
 * Accepts 2 images from opposite sides of the tree
 */

import React, {useState} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  Alert,
} from 'react-native';
import {launchCamera, launchImageLibrary} from 'react-native-image-picker';
import {useLanguage} from '../context/LanguageContext';
import {detectBunches} from '../services/pestDetectionApi';

export default function BunchDetectionScreen({navigation}) {
  const {t} = useLanguage();

  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const imagePickerOptions = {
    mediaType: 'photo',
    quality: 0.8,
    maxWidth: 1024,
    maxHeight: 1024,
  };

  const selectImage = (imageNumber) => {
    Alert.alert(
      t('bunchDetection.selectImage'),
      t('bunchDetection.chooseSource'),
      [
        {
          text: t('common.cancel'),
          style: 'cancel',
        },
        {
          text: t('bunchDetection.camera'),
          onPress: () => openCamera(imageNumber),
        },
        {
          text: t('bunchDetection.gallery'),
          onPress: () => openGallery(imageNumber),
        },
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
      setResult(null); // Clear previous result
    }
  };

  const clearImages = () => {
    setImage1(null);
    setImage2(null);
    setResult(null);
  };

  const analyzeImages = async () => {
    if (!image1) {
      Alert.alert(t('common.error'), t('bunchDetection.selectAtLeastOne'));
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const response = await detectBunches(
        image1.uri,
        image2 ? image2.uri : null
      );

      if (response.success) {
        setResult(response);
      } else {
        Alert.alert(t('common.error'), response.error || t('bunchDetection.analysisError'));
      }
    } catch (error) {
      console.error('Bunch detection error:', error);
      Alert.alert(t('common.error'), error.message || t('bunchDetection.analysisError'));
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.7) return '#4caf50';
    if (confidence >= 0.5) return '#ff9800';
    return '#f44336';
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>← {t('common.back')}</Text>
        </TouchableOpacity>
        <Text style={styles.title}>{t('bunchDetection.title')}</Text>
        <View style={{width: 50}} />
      </View>

      {/* Instruction Card */}
      <View style={styles.instructionCard}>
        <Text style={styles.instructionIcon}>📸</Text>
        <Text style={styles.instructionTitle}>{t('bunchDetection.howToUse')}</Text>
        <Text style={styles.instructionText}>
          {t('bunchDetection.instruction')}
        </Text>
        <View style={styles.tipContainer}>
          <Text style={styles.tipIcon}>💡</Text>
          <Text style={styles.tipText}>{t('bunchDetection.tip')}</Text>
        </View>
      </View>

      {/* Image Upload Section */}
      <View style={styles.imagesSection}>
        <Text style={styles.sectionTitle}>{t('bunchDetection.uploadImages')}</Text>

        <View style={styles.imagesRow}>
          {/* Image 1 */}
          <TouchableOpacity
            style={styles.imageCard}
            onPress={() => selectImage(1)}
            disabled={loading}>
            {image1 ? (
              <Image source={{uri: image1.uri}} style={styles.selectedImage} />
            ) : (
              <View style={styles.placeholderContainer}>
                <Text style={styles.placeholderIcon}>📷</Text>
                <Text style={styles.placeholderText}>{t('bunchDetection.image1')}</Text>
                <Text style={styles.placeholderSubtext}>{t('bunchDetection.side1')}</Text>
              </View>
            )}
          </TouchableOpacity>

          {/* Image 2 */}
          <TouchableOpacity
            style={styles.imageCard}
            onPress={() => selectImage(2)}
            disabled={loading}>
            {image2 ? (
              <Image source={{uri: image2.uri}} style={styles.selectedImage} />
            ) : (
              <View style={styles.placeholderContainer}>
                <Text style={styles.placeholderIcon}>📷</Text>
                <Text style={styles.placeholderText}>{t('bunchDetection.image2')}</Text>
                <Text style={styles.placeholderSubtext}>{t('bunchDetection.side2')}</Text>
              </View>
            )}
          </TouchableOpacity>
        </View>

        {/* Action Buttons */}
        <View style={styles.actionButtons}>
          {(image1 || image2) && (
            <TouchableOpacity
              style={styles.clearButton}
              onPress={clearImages}
              disabled={loading}>
              <Text style={styles.clearButtonText}>{t('common.clear')}</Text>
            </TouchableOpacity>
          )}

          <TouchableOpacity
            style={[styles.analyzeButton, !image1 && styles.disabledButton]}
            onPress={analyzeImages}
            disabled={!image1 || loading}>
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <Text style={styles.analyzeButtonText}>{t('bunchDetection.analyze')}</Text>
            )}
          </TouchableOpacity>
        </View>
      </View>

      {/* Results Section */}
      {result && (
        <View style={styles.resultSection}>
          <Text style={styles.sectionTitle}>{t('bunchDetection.results')}</Text>

          {/* Total Count Card */}
          <View style={styles.totalCountCard}>
            <Text style={styles.totalCountIcon}>🥥</Text>
            <View style={styles.totalCountContent}>
              <Text style={styles.totalCountLabel}>{t('bunchDetection.totalBunches')}</Text>
              <Text style={styles.totalCountValue}>{result.totalBunchCount}</Text>
            </View>
          </View>

          {/* Confidence */}
          {result.averageConfidence > 0 && (
            <View style={styles.confidenceCard}>
              <Text style={styles.confidenceLabel}>{t('bunchDetection.avgConfidence')}</Text>
              <View style={styles.confidenceBar}>
                <View
                  style={[
                    styles.confidenceFill,
                    {
                      width: `${result.averageConfidence * 100}%`,
                      backgroundColor: getConfidenceColor(result.averageConfidence),
                    },
                  ]}
                />
              </View>
              <Text style={styles.confidenceText}>
                {(result.averageConfidence * 100).toFixed(1)}%
              </Text>
            </View>
          )}

          {/* Per-Image Results */}
          <View style={styles.perImageResults}>
            {result.results.image1 && !result.results.image1.error && (
              <View style={styles.perImageCard}>
                <Text style={styles.perImageTitle}>{t('bunchDetection.image1Result')}</Text>
                <Text style={styles.perImageCount}>
                  {result.results.image1.bunch_count} {t('bunchDetection.bunchesDetected')}
                </Text>
              </View>
            )}

            {result.results.image2 && !result.results.image2.error && (
              <View style={styles.perImageCard}>
                <Text style={styles.perImageTitle}>{t('bunchDetection.image2Result')}</Text>
                <Text style={styles.perImageCount}>
                  {result.results.image2.bunch_count} {t('bunchDetection.bunchesDetected')}
                </Text>
              </View>
            )}
          </View>

          {/* Message */}
          <View style={styles.messageCard}>
            <Text style={styles.messageText}>{result.message}</Text>
            <Text style={styles.recommendationText}>{result.recommendation}</Text>
          </View>

          {/* Yield Estimation */}
          {result.totalBunchCount > 0 && (
            <View style={styles.yieldCard}>
              <Text style={styles.yieldTitle}>{t('bunchDetection.yieldEstimate')}</Text>
              <Text style={styles.yieldText}>
                {t('bunchDetection.estimatedYield', {
                  min: result.totalBunchCount * 8,
                  max: result.totalBunchCount * 15,
                })}
              </Text>
              <Text style={styles.yieldNote}>{t('bunchDetection.yieldNote')}</Text>
            </View>
          )}
        </View>
      )}

      <View style={{height: 40}} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  contentContainer: {
    paddingBottom: 40,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingTop: 50,
    backgroundColor: '#2e7d32',
  },
  backButton: {
    color: '#fff',
    fontSize: 16,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
  },
  instructionCard: {
    backgroundColor: '#fff',
    margin: 20,
    padding: 20,
    borderRadius: 15,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  instructionIcon: {
    fontSize: 40,
    marginBottom: 10,
  },
  instructionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  instructionText: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    lineHeight: 22,
  },
  tipContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#fff3e0',
    padding: 12,
    borderRadius: 10,
    marginTop: 15,
  },
  tipIcon: {
    fontSize: 20,
    marginRight: 10,
  },
  tipText: {
    flex: 1,
    fontSize: 13,
    color: '#e65100',
    fontStyle: 'italic',
  },
  imagesSection: {
    marginHorizontal: 20,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  imagesRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  imageCard: {
    width: '48%',
    aspectRatio: 1,
    backgroundColor: '#fff',
    borderRadius: 15,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  selectedImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'cover',
  },
  placeholderContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 15,
  },
  placeholderIcon: {
    fontSize: 40,
    marginBottom: 10,
  },
  placeholderText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#666',
    marginBottom: 5,
  },
  placeholderSubtext: {
    fontSize: 12,
    color: '#999',
    textAlign: 'center',
  },
  actionButtons: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 20,
    gap: 15,
  },
  clearButton: {
    paddingVertical: 15,
    paddingHorizontal: 30,
    borderRadius: 25,
    borderWidth: 2,
    borderColor: '#f44336',
  },
  clearButtonText: {
    color: '#f44336',
    fontSize: 16,
    fontWeight: '600',
  },
  analyzeButton: {
    backgroundColor: '#2e7d32',
    paddingVertical: 15,
    paddingHorizontal: 40,
    borderRadius: 25,
    minWidth: 150,
    alignItems: 'center',
  },
  analyzeButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  disabledButton: {
    backgroundColor: '#ccc',
  },
  resultSection: {
    marginHorizontal: 20,
  },
  totalCountCard: {
    backgroundColor: '#e8f5e9',
    padding: 25,
    borderRadius: 15,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  totalCountIcon: {
    fontSize: 50,
    marginRight: 20,
  },
  totalCountContent: {
    flex: 1,
  },
  totalCountLabel: {
    fontSize: 14,
    color: '#2e7d32',
    marginBottom: 5,
  },
  totalCountValue: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#1b5e20',
  },
  confidenceCard: {
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 15,
    marginBottom: 15,
  },
  confidenceLabel: {
    fontSize: 14,
    color: '#666',
    marginBottom: 10,
  },
  confidenceBar: {
    height: 10,
    backgroundColor: '#e0e0e0',
    borderRadius: 5,
    overflow: 'hidden',
    marginBottom: 5,
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 5,
  },
  confidenceText: {
    fontSize: 14,
    color: '#333',
    fontWeight: '600',
    textAlign: 'right',
  },
  perImageResults: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 15,
  },
  perImageCard: {
    width: '48%',
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 15,
    alignItems: 'center',
  },
  perImageTitle: {
    fontSize: 12,
    color: '#666',
    marginBottom: 5,
  },
  perImageCount: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  messageCard: {
    backgroundColor: '#fff',
    padding: 15,
    borderRadius: 15,
    marginBottom: 15,
  },
  messageText: {
    fontSize: 14,
    color: '#333',
    marginBottom: 10,
  },
  recommendationText: {
    fontSize: 13,
    color: '#666',
    fontStyle: 'italic',
  },
  yieldCard: {
    backgroundColor: '#fff8e1',
    padding: 20,
    borderRadius: 15,
    alignItems: 'center',
  },
  yieldTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#f57c00',
    marginBottom: 10,
  },
  yieldText: {
    fontSize: 18,
    color: '#e65100',
    fontWeight: '600',
    marginBottom: 5,
  },
  yieldNote: {
    fontSize: 12,
    color: '#999',
    textAlign: 'center',
  },
});
