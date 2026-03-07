import React, {useState, useEffect, useCallback} from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  ActivityIndicator,
  RefreshControl,
  Alert,
  Image,
} from 'react-native';
import {useLanguage} from '../context/LanguageContext';
import {scanAPI} from '../services/scanApi';

// Disease types to filter
const DISEASE_TYPES = ['leaf_dieback', 'leaf_rot', 'leaf_spot', 'disease', 'Leaf Rot', 'Leaf_Spot'];

export default function LeafDiseaseHistoryScreen({navigation}) {
  const {t} = useLanguage();

  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [filter, setFilter] = useState('all'); // 'all', 'diseased', 'healthy'

  const loadScans = async () => {
    setLoading(true);
    try {
      const response = await scanAPI.getMyScans({limit: 100});
      console.log('API Response:', JSON.stringify(response, null, 2));

      // Try different response structures
      let allScans = [];
      if (response?.data?.scans && Array.isArray(response.data.scans)) {
        allScans = response.data.scans;
      } else if (response?.scans && Array.isArray(response.scans)) {
        allScans = response.scans;
      } else if (response?.data && Array.isArray(response.data)) {
        allScans = response.data;
      } else if (Array.isArray(response)) {
        allScans = response;
      }

      console.log('All scans count:', allScans.length);

      if (allScans.length > 0) {
        // Filter only leaf disease scans (scanType === 'disease')
        let filteredScans = allScans.filter(scan => {
          return scan.scanType === 'disease';
        });

        console.log('Filtered disease scans:', filteredScans.length);

        // Apply status filter
        if (filter === 'diseased') {
          filteredScans = filteredScans.filter(s => s.isInfected);
        } else if (filter === 'healthy') {
          filteredScans = filteredScans.filter(s => !s.isInfected && s.isValidImage !== false);
        }

        setScans(filteredScans);
      } else {
        setScans([]);
      }
    } catch (error) {
      console.error('Load scans error:', error);
      setScans([]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadScans();
  }, [filter]);

  const onRefresh = () => {
    setRefreshing(true);
    loadScans();
  };

  const deleteScan = async scanId => {
    Alert.alert(t('common.delete'), t('leafDisease.deleteConfirm'), [
      {text: t('common.cancel'), style: 'cancel'},
      {
        text: t('common.delete'),
        style: 'destructive',
        onPress: async () => {
          try {
            await scanAPI.deleteScan(scanId);
            setScans(prev => prev.filter(s => s._id !== scanId));
          } catch (error) {
            Alert.alert(t('common.error'), 'Failed to delete scan');
          }
        },
      },
    ]);
  };

  const formatDate = dateString => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getScanIcon = scan => {
    if (!scan.isValidImage) return '❌';
    if (scan.isInfected) return '🦠';
    return '✅';
  };

  const getScanLabel = scan => {
    if (!scan.isValidImage) return t('diseaseDetection.notCoconut');
    if (!scan.isInfected) return t('diseaseDetection.healthy');

    // Check for specific disease types
    const pestType = scan.pestType || scan.detectionType || '';
    if (pestType.includes('dieback') || pestType.includes('leaf_die_back')) {
      return t('babyCoconut.leafDieback');
    }
    if (pestType.includes('rot') || pestType === 'Leaf Rot') {
      return t('diseaseDetection.leafRot');
    }
    if (pestType.includes('spot') || pestType === 'Leaf_Spot') {
      return t('diseaseDetection.leafSpot');
    }
    return t('diseaseDetection.diseased');
  };

  const getScanTypeLabel = scan => {
    const pestType = scan.pestType || scan.detectionType || scan.scanType || '';
    if (pestType.includes('dieback') || pestType.includes('leaf_die_back')) {
      return t('babyCoconut.title');
    }
    return t('leafDisease.otherCoconut');
  };

  const renderScanItem = ({item}) => (
    <TouchableOpacity
      style={styles.scanCard}
      onPress={() => navigation.navigate('ScanDetail', {scanId: item._id, scanData: item})}
      onLongPress={() => deleteScan(item._id)}>
      {item.imageUrl ? (
        <Image source={{uri: item.imageUrl}} style={styles.scanImage} />
      ) : (
        <View style={[styles.scanImage, styles.scanImagePlaceholder]}>
          <Text style={styles.scanIconText}>{getScanIcon(item)}</Text>
        </View>
      )}
      <View style={styles.scanContent}>
        <Text style={styles.scanType}>{getScanTypeLabel(item)}</Text>
        <Text
          style={[
            styles.scanResult,
            {color: item.isInfected ? '#d32f2f' : '#2e7d32'},
          ]}>
          {getScanLabel(item)}
        </Text>
        <Text style={styles.scanDate}>{formatDate(item.createdAt)}</Text>
      </View>
      <View style={[
        styles.statusBadge,
        {backgroundColor: item.isInfected ? '#ffebee' : '#e8f5e9'}
      ]}>
        <Text style={styles.statusIcon}>{getScanIcon(item)}</Text>
      </View>
      <Text style={styles.arrowIcon}>›</Text>
    </TouchableOpacity>
  );

  const renderFilterBar = () => (
    <View style={styles.filterBar}>
      {['all', 'diseased', 'healthy'].map(f => (
        <TouchableOpacity
          key={f}
          style={[styles.filterButton, filter === f && styles.filterButtonActive]}
          onPress={() => setFilter(f)}>
          <Text
            style={[
              styles.filterText,
              filter === f && styles.filterTextActive,
            ]}>
            {f === 'all'
              ? t('common.all')
              : f === 'diseased'
              ? t('diseaseDetection.diseased')
              : t('diseaseDetection.healthy')}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  const renderEmpty = () => (
    <View style={styles.emptyContainer}>
      <Text style={styles.emptyIcon}>🍃</Text>
      <Text style={styles.emptyText}>{t('common.noData')}</Text>
      <Text style={styles.emptySubtext}>{t('leafDisease.noScansYet')}</Text>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#8B4513" />
        <Text style={styles.loadingText}>{t('common.loading')}</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>← {t('common.back')}</Text>
        </TouchableOpacity>
        <Text style={styles.title}>{t('leafDisease.historyTitle')}</Text>
        <View style={styles.countBadge}>
          <Text style={styles.countText}>{scans.length}</Text>
        </View>
      </View>

      {renderFilterBar()}

      <FlatList
        data={scans}
        renderItem={renderScanItem}
        keyExtractor={item => item._id}
        contentContainerStyle={styles.listContent}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            colors={['#8B4513']}
            tintColor="#8B4513"
          />
        }
        ListEmptyComponent={renderEmpty}
      />
    </View>
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
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 15,
    paddingTop: 50,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  backButton: {
    color: '#8B4513',
    fontSize: 16,
  },
  title: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  countBadge: {
    backgroundColor: '#8B4513',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  countText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  filterBar: {
    flexDirection: 'row',
    paddingHorizontal: 15,
    paddingVertical: 12,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 10,
    backgroundColor: '#f0f0f0',
  },
  filterButtonActive: {
    backgroundColor: '#8B4513',
  },
  filterText: {
    color: '#666',
    fontSize: 14,
  },
  filterTextActive: {
    color: '#fff',
    fontWeight: 'bold',
  },
  listContent: {
    padding: 15,
  },
  scanCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 12,
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 1},
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  scanImage: {
    width: 60,
    height: 60,
    borderRadius: 10,
    marginRight: 12,
  },
  scanImagePlaceholder: {
    backgroundColor: '#f5f5f5',
    justifyContent: 'center',
    alignItems: 'center',
  },
  scanIconText: {
    fontSize: 24,
  },
  scanContent: {
    flex: 1,
  },
  scanType: {
    fontSize: 11,
    color: '#8B4513',
    fontWeight: '600',
    marginBottom: 2,
  },
  scanResult: {
    fontSize: 15,
    fontWeight: 'bold',
    marginBottom: 2,
  },
  scanDate: {
    fontSize: 12,
    color: '#999',
  },
  statusBadge: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 5,
  },
  statusIcon: {
    fontSize: 16,
  },
  arrowIcon: {
    color: '#ccc',
    fontSize: 24,
  },
  emptyContainer: {
    alignItems: 'center',
    padding: 50,
  },
  emptyIcon: {
    fontSize: 60,
    marginBottom: 15,
  },
  emptyText: {
    fontSize: 18,
    color: '#333',
    fontWeight: 'bold',
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    marginTop: 5,
    textAlign: 'center',
  },
});
