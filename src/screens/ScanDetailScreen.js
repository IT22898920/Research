import React, {useState, useEffect} from 'react';
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
import {useLanguage} from '../context/LanguageContext';
import {scanAPI} from '../services/scanApi';

export default function ScanDetailScreen({navigation, route}) {
  const {t} = useLanguage();
  const {scanId, scanData} = route.params || {};

  const [scan, setScan] = useState(scanData || null);
  const [loading, setLoading] = useState(!scanData);

  useEffect(() => {
    if (!scanData && scanId) {
      loadScanDetails();
    }
  }, [scanId]);

  const loadScanDetails = async () => {
    try {
      const response = await scanAPI.getScan(scanId);
      setScan(response.data);
    } catch (error) {
      Alert.alert(t('common.error'), 'Failed to load scan details');
      navigation.goBack();
    } finally {
      setLoading(false);
    }
  };

  const formatDate = dateString => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getSeverityColor = level => {
    switch (level) {
      case 'mild':
        return '#4caf50';
      case 'moderate':
        return '#ff9800';
      case 'severe':
        return '#f44336';
      default:
        return '#666';
    }
  };

  const getConfidenceColor = confidence => {
    if (confidence >= 0.8) return '#4caf50';
    if (confidence >= 0.5) return '#ff9800';
    return '#f44336';
  };

  const deleteScan = () => {
    Alert.alert(
      t('common.delete'),
      'Are you sure you want to delete this scan?',
      [
        {text: t('common.cancel'), style: 'cancel'},
        {
          text: t('common.delete'),
          style: 'destructive',
          onPress: async () => {
            try {
              console.log('Deleting scan:', scan._id);
              await scanAPI.deleteScan(scan._id);
              navigation.goBack();
            } catch (error) {
              console.error('Delete error:', error);
              Alert.alert(t('common.error'), error.message || 'Failed to delete scan');
            }
          },
        },
      ],
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#2e7d32" />
        <Text style={styles.loadingText}>{t('common.loading')}</Text>
      </View>
    );
  }

  if (!scan) {
    return (
      <View style={styles.loadingContainer}>
        <Text style={styles.errorText}>Scan not found</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>← {t('common.back')}</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Scan Details</Text>
        <TouchableOpacity onPress={deleteScan}>
          <Text style={styles.deleteButton}>Delete</Text>
        </TouchableOpacity>
      </View>

      {/* Image */}
      {scan.imageUrl ? (
        <View style={styles.imageContainer}>
          <Image source={{uri: scan.imageUrl}} style={styles.scanImage} />
        </View>
      ) : (
        <View style={styles.noImageContainer}>
          <Text style={styles.noImageIcon}>🌴</Text>
          <Text style={styles.noImageText}>No image saved</Text>
        </View>
      )}

      {/* Result Status */}
      <View
        style={[
          styles.statusCard,
          {
            backgroundColor: !scan.isValidImage
              ? '#fff3e0'
              : scan.isInfected
              ? '#ffebee'
              : '#e8f5e9',
          },
        ]}>
        <Text style={styles.statusIcon}>
          {!scan.isValidImage ? '❓' : scan.isInfected ? '🐛' : '✅'}
        </Text>
        <Text
          style={[
            styles.statusText,
            {
              color: !scan.isValidImage
                ? '#e65100'
                : scan.isInfected
                ? '#c62828'
                : '#2e7d32',
            },
          ]}>
          {!scan.isValidImage
            ? 'Invalid Image'
            : scan.isInfected
            ? 'Pest Detected'
            : t('pestDetection.healthy')}
        </Text>
      </View>

      {/* Scan Info */}
      <View style={styles.infoCard}>
        <Text style={styles.cardTitle}>Scan Information</Text>

        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Scan Type</Text>
          <Text style={styles.infoValue}>{scan.scanType.toUpperCase()}</Text>
        </View>

        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Date</Text>
          <Text style={styles.infoValue}>{formatDate(scan.createdAt)}</Text>
        </View>

        {scan.deviceInfo && (
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Device</Text>
            <Text style={styles.infoValue}>
              {scan.deviceInfo.platform} - {scan.deviceInfo.model}
            </Text>
          </View>
        )}
      </View>

      {/* Severity */}
      {scan.severity?.level && (
        <View style={styles.infoCard}>
          <Text style={styles.cardTitle}>Severity</Text>
          <View style={styles.severityContainer}>
            <View
              style={[
                styles.severityBadge,
                {backgroundColor: getSeverityColor(scan.severity.level)},
              ]}>
              <Text style={styles.severityBadgeText}>
                {scan.severity.level.toUpperCase()}
              </Text>
            </View>
            {scan.severity.percent && (
              <Text style={styles.severityPercent}>
                {scan.severity.percent}% confidence
              </Text>
            )}
          </View>
        </View>
      )}

      {/* Detection Results */}
      {scan.results && (
        <View style={styles.infoCard}>
          <Text style={styles.cardTitle}>Detection Results</Text>

          {scan.results.mite && (
            <View style={styles.resultItem}>
              <View style={styles.resultHeader}>
                <Text style={styles.resultIcon}>🕷️</Text>
                <Text style={styles.resultName}>{t('pestDetection.coconutMite')}</Text>
              </View>
              <View style={styles.resultDetails}>
                <Text
                  style={[
                    styles.resultStatus,
                    {color: scan.results.mite.detected ? '#f44336' : '#4caf50'},
                  ]}>
                  {scan.results.mite.detected ? 'DETECTED' : 'Not Detected'}
                </Text>
                {scan.results.mite.confidence > 0 && (
                  <View style={styles.confidenceBar}>
                    <View
                      style={[
                        styles.confidenceFill,
                        {
                          width: `${scan.results.mite.confidence * 100}%`,
                          backgroundColor: getConfidenceColor(
                            scan.results.mite.confidence,
                          ),
                        },
                      ]}
                    />
                  </View>
                )}
                <Text style={styles.confidenceText}>
                  {(scan.results.mite.confidence * 100).toFixed(1)}% confidence
                </Text>
              </View>
            </View>
          )}

          {scan.results.caterpillar && (
            <View style={styles.resultItem}>
              <View style={styles.resultHeader}>
                <Text style={styles.resultIcon}>🐛</Text>
                <Text style={styles.resultName}>{t('pestDetection.caterpillar')}</Text>
              </View>
              <View style={styles.resultDetails}>
                <Text
                  style={[
                    styles.resultStatus,
                    {
                      color: scan.results.caterpillar.detected
                        ? '#f44336'
                        : '#4caf50',
                    },
                  ]}>
                  {scan.results.caterpillar.detected ? 'DETECTED' : 'Not Detected'}
                </Text>
                {scan.results.caterpillar.confidence > 0 && (
                  <View style={styles.confidenceBar}>
                    <View
                      style={[
                        styles.confidenceFill,
                        {
                          width: `${scan.results.caterpillar.confidence * 100}%`,
                          backgroundColor: getConfidenceColor(
                            scan.results.caterpillar.confidence,
                          ),
                        },
                      ]}
                    />
                  </View>
                )}
                <Text style={styles.confidenceText}>
                  {(scan.results.caterpillar.confidence * 100).toFixed(1)}% confidence
                </Text>
              </View>
            </View>
          )}
        </View>
      )}

      {/* Pests Detected */}
      {scan.pestsDetected && scan.pestsDetected.length > 0 && (
        <View style={styles.infoCard}>
          <Text style={styles.cardTitle}>Pests Detected</Text>
          <View style={styles.pestsContainer}>
            {scan.pestsDetected.map((pest, index) => (
              <View key={index} style={styles.pestTag}>
                <Text style={styles.pestTagText}>
                  {pest === 'coconut_mite'
                    ? t('pestDetection.coconutMite')
                    : t('pestDetection.caterpillar')}
                </Text>
              </View>
            ))}
          </View>
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  },
  loadingText: {
    color: '#666',
    marginTop: 10,
  },
  errorText: {
    color: '#f44336',
    fontSize: 16,
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
  deleteButton: {
    color: '#ffcdd2',
    fontSize: 14,
  },
  imageContainer: {
    margin: 20,
    borderRadius: 15,
    overflow: 'hidden',
    backgroundColor: '#fff',
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  scanImage: {
    width: '100%',
    height: 250,
    resizeMode: 'cover',
  },
  noImageContainer: {
    margin: 20,
    padding: 40,
    borderRadius: 15,
    backgroundColor: '#fff',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  noImageIcon: {
    fontSize: 50,
    marginBottom: 10,
  },
  noImageText: {
    color: '#999',
    fontSize: 14,
  },
  statusCard: {
    marginHorizontal: 20,
    marginBottom: 15,
    padding: 20,
    borderRadius: 15,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
  },
  statusIcon: {
    fontSize: 40,
    marginRight: 15,
  },
  statusText: {
    fontSize: 22,
    fontWeight: 'bold',
  },
  infoCard: {
    backgroundColor: '#fff',
    marginHorizontal: 20,
    marginBottom: 15,
    padding: 20,
    borderRadius: 15,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 1},
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  cardTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 15,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  infoLabel: {
    color: '#888',
    fontSize: 14,
  },
  infoValue: {
    color: '#333',
    fontSize: 14,
    fontWeight: '500',
  },
  severityContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  severityBadge: {
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
  },
  severityBadgeText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
  },
  severityPercent: {
    color: '#666',
    marginLeft: 15,
    fontSize: 14,
  },
  resultItem: {
    marginBottom: 20,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  resultIcon: {
    fontSize: 24,
    marginRight: 10,
  },
  resultName: {
    color: '#333',
    fontSize: 16,
    fontWeight: '600',
  },
  resultDetails: {
    marginLeft: 34,
  },
  resultStatus: {
    fontSize: 14,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  confidenceBar: {
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    marginBottom: 5,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    borderRadius: 4,
  },
  confidenceText: {
    color: '#888',
    fontSize: 12,
  },
  pestsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  pestTag: {
    backgroundColor: '#f44336',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 10,
    marginBottom: 10,
  },
  pestTagText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 14,
  },
});
