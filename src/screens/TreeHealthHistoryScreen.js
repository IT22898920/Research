import React, {useState, useCallback} from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Alert,
  ActivityIndicator,
  Dimensions,
  SafeAreaView,
} from 'react-native';
import {useFocusEffect} from '@react-navigation/native';
import {LineChart} from 'react-native-chart-kit';
import {
  getTreeHealthHistory,
  deleteTreeHealthRecord,
  clearTreeHealthHistory,
  getTreeHealthTrend,
} from '../services/treeHealthHistoryService';

const SCREEN_WIDTH = Dimensions.get('window').width;

export default function TreeHealthHistoryScreen({navigation}) {
  const [history, setHistory] = useState([]);
  const [trendData, setTrendData] = useState([]);
  const [loading, setLoading] = useState(true);

  useFocusEffect(
    useCallback(() => {
      loadData();
    }, []),
  );

  const loadData = async () => {
    setLoading(true);
    const [h, t] = await Promise.all([
      getTreeHealthHistory(),
      getTreeHealthTrend(10),
    ]);
    setHistory(h);
    setTrendData(t);
    setLoading(false);
  };

  const handleDelete = (id) => {
    Alert.alert('Delete Record', 'Remove this record from history?', [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          await deleteTreeHealthRecord(id);
          loadData();
        },
      },
    ]);
  };

  const handleClearAll = () => {
    Alert.alert('Clear All History', 'This will delete all tree health records. Continue?', [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Clear All',
        style: 'destructive',
        onPress: async () => {
          await clearTreeHealthHistory();
          loadData();
        },
      },
    ]);
  };

  const formatDate = (isoString) => {
    const d = new Date(isoString);
    return `${d.getDate()}/${d.getMonth() + 1}/${d.getFullYear()} ${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`;
  };

  const getStats = () => {
    if (!history.length) return null;
    const total = history.length;
    const unhealthyCount = history.filter(r => !r.isHealthy).length;
    const avgPct = Math.round(history.reduce((s, r) => s + r.unhealthyPercentage, 0) / total);
    const latest = history[0];
    const prev = history[1];
    let trend = null;
    if (latest && prev) {
      const diff = latest.unhealthyPercentage - prev.unhealthyPercentage;
      trend = diff > 0 ? `↑ ${diff}% worse` : diff < 0 ? `↓ ${Math.abs(diff)}% better` : '→ No change';
    }
    return {total, unhealthyCount, avgPct, trend};
  };

  const stats = getStats();

  const chartDataPoints =
    trendData.length >= 2
      ? trendData.map(r => r.unhealthyPercentage)
      : null;

  const chartLabels =
    trendData.length >= 2
      ? trendData.map(r => {
          const d = new Date(r.timestamp);
          return `${d.getDate()}/${d.getMonth() + 1}`;
        })
      : null;

  const renderItem = ({item}) => {
    const color = item.isHealthy
      ? '#4CAF50'
      : item.unhealthyPercentage > 80
      ? '#B71C1C'
      : item.unhealthyPercentage > 60
      ? '#F44336'
      : item.unhealthyPercentage > 30
      ? '#FF9800'
      : '#FFC107';

    return (
      <View style={[styles.recordCard, {borderLeftColor: color}]}>
        <View style={styles.recordHeader}>
          <View style={styles.recordLeft}>
            <Text style={styles.recordIcon}>
              {item.isHealthy ? '✅' : '⚠️'}
            </Text>
            <View>
              <Text style={[styles.recordStatus, {color}]}>
                {item.isHealthy ? 'Healthy' : `Unhealthy — ${item.unhealthyPercentage}%`}
              </Text>
              {!item.isHealthy && item.severity && (
                <View style={[styles.severityTag, {backgroundColor: color + '20', borderColor: color}]}>
                  <Text style={[styles.severityTagText, {color}]}>
                    {item.severity} · {item.urgency}
                  </Text>
                </View>
              )}
              <Text style={styles.recordDate}>{formatDate(item.timestamp)}</Text>
            </View>
          </View>
          <View style={styles.recordRight}>
            <Text style={styles.confidenceText}>
              {(item.confidence * 100).toFixed(0)}%
            </Text>
            <Text style={styles.confidenceLabel}>confident</Text>
            <TouchableOpacity
              style={styles.deleteBtn}
              onPress={() => handleDelete(item.id)}>
              <Text style={styles.deleteBtnText}>✕</Text>
            </TouchableOpacity>
          </View>
        </View>
      </View>
    );
  };

  if (loading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#2E7D32" />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
          <Text style={styles.backBtnText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Tree Health History</Text>
        {history.length > 0 && (
          <TouchableOpacity onPress={handleClearAll}>
            <Text style={styles.clearAllText}>Clear All</Text>
          </TouchableOpacity>
        )}
      </View>

      {history.length === 0 ? (
        <View style={styles.emptyContainer}>
          <Text style={styles.emptyIcon}>🌴</Text>
          <Text style={styles.emptyTitle}>No History Yet</Text>
          <Text style={styles.emptySubtitle}>
            Analyze a coconut tree to start tracking its health over time.
          </Text>
          <TouchableOpacity
            style={styles.analyzeNowBtn}
            onPress={() => navigation.navigate('TreeHealth')}>
            <Text style={styles.analyzeNowText}>Analyze a Tree Now</Text>
          </TouchableOpacity>
        </View>
      ) : (
        <FlatList
          data={history}
          keyExtractor={item => item.id}
          renderItem={renderItem}
          ListHeaderComponent={() => (
            <>
              {/* Stats Summary */}
              {stats && (
                <View style={styles.statsContainer}>
                  <View style={styles.statBox}>
                    <Text style={styles.statValue}>{stats.total}</Text>
                    <Text style={styles.statLabel}>Total Scans</Text>
                  </View>
                  <View style={[styles.statBox, styles.statBoxMiddle]}>
                    <Text style={[styles.statValue, {color: '#F44336'}]}>
                      {stats.unhealthyCount}
                    </Text>
                    <Text style={styles.statLabel}>Unhealthy</Text>
                  </View>
                  <View style={styles.statBox}>
                    <Text style={[styles.statValue, {color: '#FF9800'}]}>
                      {stats.avgPct}%
                    </Text>
                    <Text style={styles.statLabel}>Avg Unhealthy</Text>
                  </View>
                </View>
              )}

              {/* Trend Graph */}
              {chartDataPoints && (
                <View style={styles.chartCard}>
                  <Text style={styles.chartTitle}>📈 Unhealthy % Trend</Text>
                  {stats?.trend && (
                    <Text style={[
                      styles.trendText,
                      {color: stats.trend.includes('worse') ? '#F44336' : stats.trend.includes('better') ? '#4CAF50' : '#888'},
                    ]}>
                      {stats.trend} (latest vs previous)
                    </Text>
                  )}
                  <LineChart
                    data={{
                      labels: chartLabels,
                      datasets: [{
                        data: chartDataPoints,
                        color: (opacity = 1) => `rgba(244, 67, 54, ${opacity})`,
                        strokeWidth: 2,
                      }],
                    }}
                    width={SCREEN_WIDTH - 48}
                    height={180}
                    yAxisSuffix="%"
                    yAxisInterval={1}
                    fromZero
                    chartConfig={{
                      backgroundColor: '#fff',
                      backgroundGradientFrom: '#fff',
                      backgroundGradientTo: '#fff',
                      decimalPlaces: 0,
                      color: (opacity = 1) => `rgba(244, 67, 54, ${opacity})`,
                      labelColor: () => '#666',
                      propsForDots: {r: '4', strokeWidth: '2', stroke: '#B71C1C'},
                      propsForBackgroundLines: {stroke: '#E0E0E0'},
                    }}
                    bezier
                    style={styles.chart}
                  />
                  <Text style={styles.chartCaption}>
                    Last {trendData.length} scans (oldest → latest)
                  </Text>
                </View>
              )}

              <Text style={styles.historyListTitle}>📋 All Records</Text>
            </>
          )}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: '#F5F5F5'},
  loadingContainer: {flex: 1, justifyContent: 'center', alignItems: 'center'},
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#fff',
    paddingHorizontal: 16,
    paddingVertical: 14,
    elevation: 2,
  },
  backBtn: {paddingRight: 8},
  backBtnText: {fontSize: 15, color: '#2E7D32', fontWeight: '600'},
  title: {fontSize: 17, fontWeight: 'bold', color: '#222'},
  clearAllText: {fontSize: 13, color: '#F44336', fontWeight: '600'},
  listContent: {padding: 16, paddingBottom: 40},

  // Stats
  statsContainer: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 12,
    marginBottom: 16,
    elevation: 2,
    overflow: 'hidden',
  },
  statBox: {flex: 1, alignItems: 'center', paddingVertical: 16},
  statBoxMiddle: {
    borderLeftWidth: 1,
    borderRightWidth: 1,
    borderColor: '#F0F0F0',
  },
  statValue: {fontSize: 24, fontWeight: 'bold', color: '#333'},
  statLabel: {fontSize: 12, color: '#888', marginTop: 4},

  // Chart
  chartCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 16,
    elevation: 2,
  },
  chartTitle: {fontSize: 15, fontWeight: 'bold', color: '#333', marginBottom: 4},
  trendText: {fontSize: 12, fontWeight: '600', marginBottom: 12},
  chart: {borderRadius: 8, marginTop: 4},
  chartCaption: {fontSize: 11, color: '#999', textAlign: 'center', marginTop: 8},

  // List
  historyListTitle: {fontSize: 15, fontWeight: 'bold', color: '#333', marginBottom: 10},
  recordCard: {
    backgroundColor: '#fff',
    borderRadius: 10,
    padding: 14,
    marginBottom: 10,
    borderLeftWidth: 4,
    elevation: 1,
  },
  recordHeader: {flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between'},
  recordLeft: {flexDirection: 'row', alignItems: 'flex-start', flex: 1},
  recordIcon: {fontSize: 22, marginRight: 10, marginTop: 2},
  recordStatus: {fontSize: 14, fontWeight: 'bold'},
  severityTag: {
    alignSelf: 'flex-start',
    marginTop: 4,
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
    borderWidth: 1,
  },
  severityTagText: {fontSize: 11, fontWeight: '600'},
  recordDate: {fontSize: 11, color: '#999', marginTop: 4},
  recordRight: {alignItems: 'center'},
  confidenceText: {fontSize: 18, fontWeight: 'bold', color: '#333'},
  confidenceLabel: {fontSize: 10, color: '#999'},
  deleteBtn: {marginTop: 8, padding: 4},
  deleteBtnText: {fontSize: 14, color: '#BDBDBD'},

  // Empty
  emptyContainer: {flex: 1, alignItems: 'center', justifyContent: 'center', padding: 32},
  emptyIcon: {fontSize: 60, marginBottom: 16},
  emptyTitle: {fontSize: 20, fontWeight: 'bold', color: '#333', marginBottom: 8},
  emptySubtitle: {fontSize: 14, color: '#888', textAlign: 'center', lineHeight: 22, marginBottom: 24},
  analyzeNowBtn: {
    backgroundColor: '#2E7D32',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 10,
  },
  analyzeNowText: {color: '#fff', fontWeight: 'bold', fontSize: 15},
});
