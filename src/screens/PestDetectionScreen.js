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
import {predictPest, checkApiHealth} from '../services/pestDetectionApi';

export default function PestDetectionScreen({navigation}) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    checkApi();
  }, []);

  const checkApi = async () => {
    const health = await checkApiHealth();
    setApiStatus(health.success ? 'online' : 'offline');
  };

  const selectImage = () => {
    Alert.alert(
      'Select Image',
      'Choose how to select an image',
      [
        {
          text: 'Camera',
          onPress: () => openCamera(),
        },
        {
          text: 'Gallery',
          onPress: () => openGallery(),
        },
        {
          text: 'Cancel',
          style: 'cancel',
        },
      ],
    );
  };

  const openCamera = () => {
    launchCamera(
      {
        mediaType: 'photo',
        quality: 0.8,
        maxWidth: 1024,
        maxHeight: 1024,
      },
      handleImageResponse,
    );
  };

  const openGallery = () => {
    launchImageLibrary(
      {
        mediaType: 'photo',
        quality: 0.8,
        maxWidth: 1024,
        maxHeight: 1024,
      },
      handleImageResponse,
    );
  };

  const handleImageResponse = (response) => {
    if (response.didCancel) {
      return;
    }
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
      const response = await predictPest(selectedImage.uri);

      if (response.success) {
        setResult(response);
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
    return result.prediction.is_infected ? '#d32f2f' : '#2e7d32';
  };

  const getResultIcon = () => {
    if (!result) return '🔍';
    return result.prediction.is_infected ? '🐛' : '✅';
  };

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Pest Detection</Text>
        <View style={[styles.statusBadge, {backgroundColor: apiStatus === 'online' ? '#4caf50' : '#f44336'}]}>
          <Text style={styles.statusText}>
            {apiStatus === 'checking' ? '...' : apiStatus === 'online' ? 'API Online' : 'API Offline'}
          </Text>
        </View>
      </View>

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
      {result && (
        <View style={[styles.resultContainer, {borderColor: getResultColor()}]}>
          <Text style={styles.resultIcon}>{getResultIcon()}</Text>
          <Text style={[styles.resultTitle, {color: getResultColor()}]}>
            {result.prediction.label}
          </Text>
          <Text style={styles.confidenceText}>
            Confidence: {(result.prediction.confidence * 100).toFixed(1)}%
          </Text>

          <View style={styles.probabilitiesContainer}>
            <Text style={styles.probabilitiesTitle}>Probabilities:</Text>
            {Object.entries(result.probabilities).map(([label, prob]) => (
              <View key={label} style={styles.probabilityRow}>
                <Text style={styles.probabilityLabel}>{label}:</Text>
                <View style={styles.probabilityBarContainer}>
                  <View style={[styles.probabilityBar, {width: `${prob * 100}%`}]} />
                </View>
                <Text style={styles.probabilityValue}>{(prob * 100).toFixed(1)}%</Text>
              </View>
            ))}
          </View>

          {result.prediction.is_infected && (
            <View style={styles.warningBox}>
              <Text style={styles.warningTitle}>⚠️ Action Required</Text>
              <Text style={styles.warningText}>
                Coconut mite infection detected. Consider applying appropriate pesticide treatment.
              </Text>
            </View>
          )}
        </View>
      )}

      {/* Info Section */}
      <View style={styles.infoContainer}>
        <Text style={styles.infoTitle}>About This Feature</Text>
        <Text style={styles.infoText}>
          This AI-powered detector uses EfficientNetB0 model trained on coconut images to detect
          coconut mite infections with 95%+ accuracy.
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
  imageContainer: {
    margin: 15,
    height: 300,
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
    lineHeight: 18,
  },
});
