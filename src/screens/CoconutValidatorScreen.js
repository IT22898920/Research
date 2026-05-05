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
} from 'react-native';
import {launchImageLibrary, launchCamera} from 'react-native-image-picker';
import {ML_API_URL, fetchWithTimeout} from '../config/apiConfig';

const API_BASE_URL = ML_API_URL;

export default function CoconutValidatorScreen({navigation}) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    checkApi();
  }, []);

  const checkApi = async () => {
    try {
      const response = await fetchWithTimeout(`${API_BASE_URL}/health`, {}, 30000, 2);
      const data = await response.json();
      setApiStatus(data.models?.validator ? 'online' : 'offline');
    } catch (error) {
      setApiStatus('offline');
    }
  };

  const handleTakePhoto = () => {
    launchCamera({mediaType: 'photo', quality: 0.8, saveToPhotos: false}, response => {
      if (response.assets && response.assets[0]) {
        setSelectedImage(response.assets[0]);
        setResult(null);
      }
    });
  };

  const handleChooseFromGallery = () => {
    launchImageLibrary({mediaType: 'photo', quality: 0.8}, response => {
      if (response.assets && response.assets[0]) {
        setSelectedImage(response.assets[0]);
        setResult(null);
      }
    });
  };

  const handleValidate = async () => {
    if (!selectedImage) {
      Alert.alert('No Image', 'Please select or capture an image first');
      return;
    }

    setIsAnalyzing(true);
    setResult(null);

    const buildFormData = () => {
      const fd = new FormData();
      fd.append('image', {
        uri: selectedImage.uri,
        type: selectedImage.type || 'image/jpeg',
        name: selectedImage.fileName || 'image.jpg',
      });
      return fd;
    };

    try {
      // Step 1: Validate if image is a coconut
      const validateRes = await fetchWithTimeout(
        `${API_BASE_URL}/predict/validate`,
        {method: 'POST', body: buildFormData(), headers: {'Content-Type': 'multipart/form-data'}},
        60000,
        1,
      );
      const validateData = await validateRes.json();

      if (!validateData.success) {
        Alert.alert('Error', validateData.error || 'Failed to validate image');
        setIsAnalyzing(false);
        return;
      }

      // If not a coconut, just show validate result
      if (!validateData.is_coconut) {
        setResult({validate: validateData, mite: null});
        setIsAnalyzing(false);
        return;
      }

      // Step 2: Check for mite using v12 model
      try {
        const miteRes = await fetchWithTimeout(
          `${API_BASE_URL}/predict/mite_v12`,
          {method: 'POST', body: buildFormData(), headers: {'Content-Type': 'multipart/form-data'}},
          60000,
          1,
        );
        const miteData = await miteRes.json();
        setResult({validate: validateData, mite: miteData.success ? miteData : null});
      } catch (e) {
        setResult({validate: validateData, mite: null});
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to connect to ML API server');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Coconut Validator</Text>
        <View style={[styles.statusBadge, apiStatus === 'online' ? styles.statusOnline : styles.statusOffline]}>
          <Text style={styles.statusText}>{apiStatus === 'online' ? 'API Online' : 'API Offline'}</Text>
        </View>
      </View>

      <View style={styles.infoCard}>
        <Text style={styles.infoTitle}>✅ Coconut Validator</Text>
        <Text style={styles.infoText}>
          Verify if your image is actually a coconut using our binary classifier.
        </Text>
        <Text style={styles.infoStat}>📊 Test Accuracy: 100%</Text>
        <Text style={styles.infoStat}>🧠 Model: MobileNetV2 (Transfer Learning)</Text>
      </View>

      <View style={styles.imageSourceCard}>
        <Text style={styles.cardTitle}>Select Image</Text>
        <View style={styles.sourceRow}>
          <TouchableOpacity style={styles.sourceButton} onPress={handleTakePhoto}>
            <Text style={styles.sourceIcon}>📷</Text>
            <Text style={styles.sourceText}>Camera</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.sourceButton} onPress={handleChooseFromGallery}>
            <Text style={styles.sourceIcon}>🖼️</Text>
            <Text style={styles.sourceText}>Gallery</Text>
          </TouchableOpacity>
        </View>
      </View>

      {selectedImage && (
        <View style={styles.imagePreview}>
          <Image source={{uri: selectedImage.uri}} style={styles.previewImage} />
        </View>
      )}

      <TouchableOpacity
        style={[styles.analyzeButton, (!selectedImage || isAnalyzing) && styles.disabledButton]}
        onPress={handleValidate}
        disabled={!selectedImage || isAnalyzing}>
        {isAnalyzing ? (
          <ActivityIndicator color="white" />
        ) : (
          <Text style={styles.analyzeButtonText}>🔍 Validate Image</Text>
        )}
      </TouchableOpacity>

      {result && result.validate && (
        <View style={[styles.resultCard, result.validate.is_coconut ? styles.successCard : styles.errorCard]}>
          <Text style={styles.resultIcon}>{result.validate.is_coconut ? '✅' : '❌'}</Text>
          <Text style={styles.resultTitle}>
            {result.validate.is_coconut ? 'COCONUT DETECTED' : 'NOT A COCONUT'}
          </Text>
          <Text style={styles.resultMessage}>{result.validate.message}</Text>

          <View style={styles.confidenceBox}>
            <Text style={styles.confidenceLabel}>Confidence:</Text>
            <Text style={styles.confidenceValue}>
              {(result.validate.confidence * 100).toFixed(1)}%
            </Text>
          </View>

          <View style={styles.probsBox}>
            <Text style={styles.probsTitle}>Validator Probabilities:</Text>
            <View style={styles.probRow}>
              <Text style={styles.probLabel}>🥥 Coconut:</Text>
              <Text style={styles.probValue}>
                {(result.validate.probabilities.coconut * 100).toFixed(2)}%
              </Text>
            </View>
            <View style={styles.probRow}>
              <Text style={styles.probLabel}>❌ Not Coconut:</Text>
              <Text style={styles.probValue}>
                {(result.validate.probabilities.not_coconut * 100).toFixed(2)}%
              </Text>
            </View>
          </View>
        </View>
      )}

      {result && result.mite && (
        <View style={[styles.resultCard, result.mite.prediction.is_infected ? styles.errorCard : styles.successCard]}>
          <Text style={styles.resultIcon}>{result.mite.prediction.is_infected ? '🐛' : '✅'}</Text>
          <Text style={styles.resultTitle}>
            COCONUT MITE CHECK (v12 - 97.44%)
          </Text>
          <Text style={styles.resultMessage}>{result.mite.prediction.message}</Text>

          <View style={styles.confidenceBox}>
            <Text style={styles.confidenceLabel}>Confidence:</Text>
            <Text style={styles.confidenceValue}>
              {(result.mite.prediction.confidence * 100).toFixed(1)}%
            </Text>
          </View>

          <View style={styles.probsBox}>
            <Text style={styles.probsTitle}>Mite Probabilities:</Text>
            <View style={styles.probRow}>
              <Text style={styles.probLabel}>✅ Healthy:</Text>
              <Text style={styles.probValue}>
                {(result.mite.probabilities.healthy * 100).toFixed(2)}%
              </Text>
            </View>
            <View style={styles.probRow}>
              <Text style={styles.probLabel}>🐛 Mite:</Text>
              <Text style={styles.probValue}>
                {(result.mite.probabilities.mite * 100).toFixed(2)}%
              </Text>
            </View>
          </View>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: '#f5f5f5'},
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    backgroundColor: '#2E7D32',
  },
  backButton: {color: 'white', fontSize: 16},
  title: {color: 'white', fontSize: 18, fontWeight: 'bold'},
  statusBadge: {paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12},
  statusOnline: {backgroundColor: '#4CAF50'},
  statusOffline: {backgroundColor: '#f44336'},
  statusText: {color: 'white', fontSize: 12, fontWeight: 'bold'},
  infoCard: {
    backgroundColor: '#E8F5E9',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    borderLeftWidth: 4,
    borderLeftColor: '#2E7D32',
  },
  infoTitle: {fontSize: 16, fontWeight: 'bold', color: '#2E7D32', marginBottom: 8},
  infoText: {fontSize: 14, color: '#555', marginBottom: 8},
  infoStat: {fontSize: 13, color: '#2E7D32', fontWeight: '600', marginTop: 4},
  imageSourceCard: {
    backgroundColor: 'white',
    margin: 16,
    marginTop: 0,
    padding: 16,
    borderRadius: 12,
  },
  cardTitle: {fontSize: 16, fontWeight: 'bold', marginBottom: 12, color: '#333'},
  sourceRow: {flexDirection: 'row', justifyContent: 'space-between'},
  sourceButton: {
    flex: 1,
    backgroundColor: '#E8F5E9',
    padding: 16,
    margin: 4,
    borderRadius: 8,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#2E7D32',
  },
  sourceIcon: {fontSize: 32},
  sourceText: {marginTop: 4, color: '#2E7D32', fontWeight: '600'},
  imagePreview: {
    backgroundColor: 'white',
    margin: 16,
    marginTop: 0,
    padding: 8,
    borderRadius: 12,
  },
  previewImage: {width: '100%', height: 200, borderRadius: 8, resizeMode: 'cover'},
  analyzeButton: {
    backgroundColor: '#2E7D32',
    margin: 16,
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  disabledButton: {backgroundColor: '#999'},
  analyzeButtonText: {color: 'white', fontSize: 16, fontWeight: 'bold'},
  resultCard: {
    margin: 16,
    padding: 20,
    borderRadius: 12,
    alignItems: 'center',
  },
  successCard: {backgroundColor: '#C8E6C9', borderWidth: 2, borderColor: '#2E7D32'},
  errorCard: {backgroundColor: '#FFCDD2', borderWidth: 2, borderColor: '#c62828'},
  resultIcon: {fontSize: 48, marginBottom: 8},
  resultTitle: {fontSize: 18, fontWeight: 'bold', marginBottom: 8, color: '#333'},
  resultMessage: {fontSize: 14, textAlign: 'center', marginBottom: 16, color: '#555'},
  confidenceBox: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'white',
    padding: 12,
    borderRadius: 8,
    width: '100%',
    marginBottom: 12,
  },
  confidenceLabel: {fontSize: 14, fontWeight: '600', color: '#555'},
  confidenceValue: {fontSize: 18, fontWeight: 'bold', color: '#2E7D32'},
  probsBox: {backgroundColor: 'white', padding: 12, borderRadius: 8, width: '100%'},
  probsTitle: {fontSize: 14, fontWeight: 'bold', marginBottom: 8, color: '#333'},
  probRow: {flexDirection: 'row', justifyContent: 'space-between', marginVertical: 4},
  probLabel: {fontSize: 13, color: '#555'},
  probValue: {fontSize: 13, fontWeight: 'bold', color: '#333'},
});
