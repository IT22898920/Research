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
} from 'react-native';
import {launchImageLibrary, launchCamera} from 'react-native-image-picker';
import {detectTreeHealth} from '../services/pestDetectionApi';
import {saveTreeHealthRecord} from '../services/treeHealthHistoryService';

export default function TreeHealthScreen({navigation}) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    checkApi();
  }, []);

  const checkApi = async () => {
    try {
      const response = await fetch('http://10.0.2.2:5001/health');
      const data = await response.json();
      setApiStatus(
        data.status === 'healthy' && data.models.tree_health
          ? 'online'
          : 'offline',
      );
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
      const apiResult = await detectTreeHealth(selectedImage.uri);

      if (apiResult.success) {
        setResult(apiResult);
        await saveTreeHealthRecord(apiResult, selectedImage.uri);
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
  };

  const getHealthColor = () => {
    if (!result) return '#757575';
    return result.isHealthy ? '#4CAF50' : '#F44336';
  };

  const getHealthIcon = () => {
    if (!result) return '🌴';
    return result.isHealthy ? '✅' : '⚠️';
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerTopRow}>
            <Text style={styles.title}>🌴 Tree Health Monitor</Text>
            <TouchableOpacity
              style={styles.historyBtn}
              onPress={() => navigation.navigate('TreeHealthHistory')}>
              <Text style={styles.historyBtnText}>📋 History</Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.subtitle}>
            Check if your coconut tree is healthy or unhealthy
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
                <Text style={styles.analyzeButtonText}>Analyze Tree</Text>
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
                    {result.isHealthy ? 'Healthy Tree' : 'Unhealthy Tree'}
                  </Text>
                  <Text style={styles.statusSubtitle}>
                    {(result.confidence * 100).toFixed(1)}% confident
                  </Text>
                </View>
              </View>

              {/* Unhealthy Indicator */}
              {!result.isHealthy && (
                <View style={styles.unhealthyIndicatorRow}>
                  <View style={styles.redDot} />
                  <Text style={styles.unhealthyIndicatorText}>Tree is unhealthy</Text>
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


            {/* Meditations & Solutions - Only for Unhealthy */}
            {!result.isHealthy && result.severityInfo && (
              <View style={styles.meditationsContainer}>

                {/* Severity Badge */}
                <View style={[styles.severityBadge, {backgroundColor: result.severityInfo.urgencyColor + '20', borderColor: result.severityInfo.urgencyColor}]}>
                  <View style={[styles.urgencyDot, {backgroundColor: result.severityInfo.urgencyColor}]} />
                  <Text style={[styles.severityLabel, {color: result.severityInfo.urgencyColor}]}>
                    {result.severityInfo.severity} — {result.severityInfo.urgency} Priority
                  </Text>
                </View>

                <Text style={styles.severityDescription}>{result.severityInfo.severityDescription}</Text>

                {/* Possible Conditions */}
                <Text style={styles.sectionTitle}>🔍 Possible Conditions</Text>
                {result.severityInfo.possibleConditions.map((cond, idx) => (
                  <View key={idx} style={styles.conditionCard}>
                    <View style={styles.conditionHeader}>
                      <Text style={styles.conditionIcon}>{cond.icon}</Text>
                      <View style={styles.conditionTitleRow}>
                        <Text style={styles.conditionName}>{cond.name}</Text>
                        {cond.urgency && (
                          <View style={[styles.urgencyBadge, {
                            backgroundColor:
                              cond.urgency === 'high' ? '#FF5252' :
                              cond.urgency === 'medium' ? '#FFA726' : '#66BB6A'
                          }]}>
                            <Text style={styles.urgencyText}>{cond.urgency.toUpperCase()}</Text>
                          </View>
                        )}
                      </View>
                    </View>
                    <View style={styles.conditionBody}>
                      <Text style={styles.conditionSectionLabel}>🔍 Reason:</Text>
                      <Text style={styles.conditionSectionText}>{cond.reason}</Text>
                      <Text style={styles.conditionSectionLabel}>⚠️ Symptoms:</Text>
                      {cond.symptoms && cond.symptoms.map((s, si) => (
                        <Text key={si} style={styles.conditionSymptom}>• {s}</Text>
                      ))}
                      <Text style={styles.conditionSectionLabel}>💊 Solution:</Text>
                      <Text style={styles.conditionSolutionText}>{cond.solution}</Text>
                    </View>
                  </View>
                ))}

                {/* Immediate Actions */}
                <Text style={styles.sectionTitle}>⚡ Immediate Actions</Text>
                <View style={styles.actionsCard}>
                  {result.severityInfo.immediateActions.map((action, idx) => (
                    <View key={idx} style={styles.actionItem}>
                      <View style={[styles.actionBullet, {backgroundColor: result.severityInfo.urgencyColor}]} />
                      <Text style={styles.actionText}>{action}</Text>
                    </View>
                  ))}
                </View>

                {/* Treatment Steps */}
                <Text style={styles.sectionTitle}>💊 Treatment Steps</Text>
                {result.severityInfo.treatmentSteps.map((step, idx) => (
                  <View key={idx} style={styles.treatmentCard}>
                    <View style={[styles.stepNumberBadge, {backgroundColor: result.severityInfo.urgencyColor}]}>
                      <Text style={styles.stepNumber}>{step.step}</Text>
                    </View>
                    <View style={styles.treatmentTextContainer}>
                      <Text style={styles.treatmentTitle}>{step.title}</Text>
                      <Text style={styles.treatmentDetail}>{step.detail}</Text>
                    </View>
                  </View>
                ))}

                {/* Preventive Measures */}
                <Text style={styles.sectionTitle}>🛡️ Preventive Measures</Text>
                <View style={styles.preventiveCard}>
                  {result.severityInfo.preventiveMeasures.map((measure, idx) => (
                    <View key={idx} style={styles.preventiveItem}>
                      <Text style={styles.preventiveIcon}>✓</Text>
                      <Text style={styles.preventiveText}>{measure}</Text>
                    </View>
                  ))}
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
                Accuracy: {result.modelInfo?.accuracy || '99.72%'}
              </Text>
              <Text style={styles.modelInfoText}>
                Macro F1: {result.modelInfo?.macro_f1 || '99.09%'}
              </Text>
              <Text style={styles.modelInfoText}>
                Architecture: MobileNetV2 with Focal Loss + Class Weights
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
              2. Make sure the coconut tree is clearly visible
            </Text>
            <Text style={styles.infoText}>
              3. Tap "Analyze Tree" to get results
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
                • Capture the full tree if possible
              </Text>
              <Text style={styles.tipText}>• Avoid blurry images</Text>
              <Text style={styles.tipText}>
                • Include both trunk and leaves
              </Text>
            </View>
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
  headerTopRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  historyBtn: {
    backgroundColor: '#E8F5E9',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2E7D32',
  },
  historyBtnText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#2E7D32',
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#2E7D32',
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
  unhealthyIndicatorRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 14,
    paddingTop: 14,
    borderTopWidth: 1,
    borderTopColor: '#E0E0E0',
  },
  redDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#F44336',
    marginRight: 8,
  },
  unhealthyIndicatorText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#F44336',
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

  // Meditations & Solutions styles
  meditationsContainer: {
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
  severityBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1.5,
    marginBottom: 12,
  },
  urgencyDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  severityLabel: {
    fontSize: 15,
    fontWeight: 'bold',
  },
  severityDescription: {
    fontSize: 13,
    color: '#555',
    lineHeight: 20,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 15,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
    marginTop: 4,
  },
  conditionCard: {
    backgroundColor: '#FFF8E1',
    borderRadius: 8,
    padding: 12,
    marginBottom: 10,
  },
  conditionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  conditionIcon: {
    fontSize: 24,
    marginRight: 10,
  },
  conditionTitleRow: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 6,
  },
  conditionName: {
    fontSize: 14,
    fontWeight: '700',
    color: '#333',
  },
  urgencyBadge: {
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  urgencyText: {
    fontSize: 10,
    fontWeight: 'bold',
    color: '#fff',
  },
  conditionBody: {
    paddingLeft: 4,
  },
  conditionSectionLabel: {
    fontSize: 13,
    fontWeight: '600',
    color: '#444',
    marginTop: 6,
    marginBottom: 3,
  },
  conditionSectionText: {
    fontSize: 12,
    color: '#555',
    lineHeight: 18,
    marginBottom: 4,
  },
  conditionSymptom: {
    fontSize: 12,
    color: '#666',
    lineHeight: 18,
    paddingLeft: 4,
  },
  conditionSolutionText: {
    fontSize: 12,
    color: '#1B5E20',
    lineHeight: 18,
    backgroundColor: '#E8F5E9',
    borderRadius: 6,
    padding: 8,
    marginTop: 4,
  },
  actionsCard: {
    backgroundColor: '#FFF3E0',
    borderRadius: 8,
    padding: 14,
    marginBottom: 16,
  },
  actionItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  actionBullet: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginTop: 5,
    marginRight: 10,
  },
  actionText: {
    flex: 1,
    fontSize: 13,
    color: '#444',
    lineHeight: 19,
  },
  treatmentCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#F3E5F5',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  stepNumberBadge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
    marginTop: 2,
  },
  stepNumber: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 13,
  },
  treatmentTextContainer: {
    flex: 1,
  },
  treatmentTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 4,
  },
  treatmentDetail: {
    fontSize: 12,
    color: '#555',
    lineHeight: 18,
  },
  preventiveCard: {
    backgroundColor: '#E8F5E9',
    borderRadius: 8,
    padding: 14,
    marginBottom: 8,
  },
  preventiveItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 8,
  },
  preventiveIcon: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#2E7D32',
    marginRight: 10,
    marginTop: 1,
  },
  preventiveText: {
    flex: 1,
    fontSize: 13,
    color: '#1B5E20',
    lineHeight: 19,
  },
});
