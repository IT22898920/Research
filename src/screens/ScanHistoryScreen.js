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
} from 'react-native';
import {useLanguage} from '../context/LanguageContext';
import {scanAPI} from '../services/scanApi';

export default function ScanHistoryScreen({navigation}) {
  const {t} = useLanguage();

  const [scans, setScans] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [pagination, setPagination] = useState({
    currentPage: 1,
    hasMore: true,
  });
  const [filter, setFilter] = useState('all'); // 'all', 'infected', 'healthy'

  const loadScans = useCallback(
    async (page = 1, refresh = false) => {
      try {
        const params = {
          page,
          limit: 20,
        };

        if (filter === 'infected') params.isInfected = 'true';
        if (filter === 'healthy') params.isInfected = 'false';

        const response = await scanAPI.getMyScans(params);

        if (refresh || page === 1) {
          setScans(response.data.scans);
        } else {
          setScans(prev => [...prev, ...response.data.scans]);
        }

        setPagination({
          currentPage: response.data.pagination.currentPage,
          hasMore: response.data.pagination.hasMore,
        });
      } catch (error) {
        console.error('Load scans error:', error);
        Alert.alert(t('common.error'), 'Failed to load scan history');
      } finally {
        setLoading(false);
        setRefreshing(false);
        setLoadingMore(false);
      }
    },
    [filter, t],
  );

  useEffect(() => {
    setLoading(true);
    loadScans(1, true);
  }, [filter]);

  const onRefresh = () => {
    setRefreshing(true);
    loadScans(1, true);
  };

  const loadMore = () => {
    if (pagination.hasMore && !loadingMore) {
      setLoadingMore(true);
      loadScans(pagination.currentPage + 1);
    }
  };

  const deleteScan = async scanId => {
    Alert.alert(t('common.delete'), 'Are you sure you want to delete this scan?', [
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
    if (!scan.isValidImage) return '❓';
    return scan.isInfected ? '🐛' : '✅';
  };

  const getScanLabel = scan => {
    if (!scan.isValidImage) return 'Invalid Image';
    if (!scan.isInfected) return t('pestDetection.healthy');
    return scan.pestsDetected?.length > 0
      ? scan.pestsDetected
          .map(p =>
            p === 'coconut_mite'
              ? t('pestDetection.coconutMite')
              : t('pestDetection.caterpillar'),
          )
          .join(', ')
      : t('pestDetection.infected');
  };

  const renderScanItem = ({item}) => (
    <TouchableOpacity
      style={styles.scanCard}
      onPress={() => navigation.navigate('ScanDetail', {scanId: item._id, scanData: item})}
      onLongPress={() => deleteScan(item._id)}>
      <View style={styles.scanIcon}>
        <Text style={styles.scanIconText}>{getScanIcon(item)}</Text>
      </View>
      <View style={styles.scanContent}>
        <Text style={styles.scanType}>{item.scanType.toUpperCase()} Scan</Text>
        <Text
          style={[
            styles.scanResult,
            {color: item.isInfected ? '#f44336' : '#4caf50'},
          ]}>
          {getScanLabel(item)}
        </Text>
        <Text style={styles.scanDate}>{formatDate(item.createdAt)}</Text>
      </View>
      {item.severity?.level && (
        <View
          style={[
            styles.severityBadge,
            {
              backgroundColor:
                item.severity.level === 'mild'
                  ? '#4caf50'
                  : item.severity.level === 'moderate'
                  ? '#ff9800'
                  : '#f44336',
            },
          ]}>
          <Text style={styles.severityText}>
            {item.severity.level.toUpperCase()}
          </Text>
        </View>
      )}
      <Text style={styles.arrowIcon}>›</Text>
    </TouchableOpacity>
  );

  const renderFilterBar = () => (
    <View style={styles.filterBar}>
      {['all', 'infected', 'healthy'].map(f => (
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
              ? 'All'
              : f === 'infected'
              ? t('pestDetection.infected')
              : t('pestDetection.healthy')}
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  const renderEmpty = () => (
    <View style={styles.emptyContainer}>
      <Text style={styles.emptyIcon}>📋</Text>
      <Text style={styles.emptyText}>{t('common.noData')}</Text>
      <Text style={styles.emptySubtext}>Your scan history will appear here</Text>
    </View>
  );

  const renderFooter = () => {
    if (!loadingMore) return null;
    return (
      <View style={styles.footer}>
        <ActivityIndicator size="small" color="#2e7d32" />
      </View>
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

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>← {t('common.back')}</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Scan History</Text>
        <TouchableOpacity
          onPress={() => navigation.navigate('Analytics', {isAdmin: false})}>
          <Text style={styles.statsButton}>📊</Text>
        </TouchableOpacity>
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
            colors={['#2e7d32']}
            tintColor="#2e7d32"
          />
        }
        onEndReached={loadMore}
        onEndReachedThreshold={0.5}
        ListEmptyComponent={renderEmpty}
        ListFooterComponent={renderFooter}
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
  statsButton: {
    fontSize: 24,
  },
  filterBar: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    paddingVertical: 15,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  filterButton: {
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
    marginRight: 10,
    backgroundColor: '#f0f0f0',
  },
  filterButtonActive: {
    backgroundColor: '#2e7d32',
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
    padding: 20,
    paddingTop: 10,
  },
  scanCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
    marginBottom: 10,
    flexDirection: 'row',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 1},
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  scanIcon: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: '#e8f5e9',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
  },
  scanIconText: {
    fontSize: 24,
  },
  scanContent: {
    flex: 1,
  },
  scanType: {
    fontSize: 12,
    color: '#888',
  },
  scanResult: {
    fontSize: 16,
    fontWeight: 'bold',
    marginVertical: 2,
  },
  scanDate: {
    fontSize: 12,
    color: '#999',
  },
  severityBadge: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 12,
  },
  severityText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: 'bold',
  },
  arrowIcon: {
    color: '#ccc',
    fontSize: 24,
    marginLeft: 10,
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
    color: '#888',
    marginTop: 5,
  },
  footer: {
    padding: 20,
    alignItems: 'center',
  },
});
