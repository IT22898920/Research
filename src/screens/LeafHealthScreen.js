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
} from 'react-native';
import {launchImageLibrary, launchCamera} from 'react-native-image-picker';

const API_BASE_URL = 'http://10.0.2.2:5001'; // Android emulator localhost

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

export default function LeafHealthScreen({navigation}) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    checkApi();
  }, []);

  const checkApi = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
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

      const response = await fetch(`${API_BASE_URL}/predict/leaf-health`, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = await response.json();

      if (data.success) {
        setResult(data);
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
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Leaf Health Monitor</Text>
          <Text style={styles.subtitle}>
            Check if your coconut leaf is healthy or unhealthy
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
    justifyContent: 'space-around',
    marginBottom: 24,
  },
  actionButton: {
    backgroundColor: '#FFF',
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    width: '45%',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  actionButtonIcon: {
    fontSize: 48,
    marginBottom: 8,
  },
  actionButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
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
});
