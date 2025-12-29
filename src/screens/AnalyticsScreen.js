import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  ActivityIndicator,
  Dimensions,
  RefreshControl,
} from 'react-native';
import {LineChart, BarChart, PieChart, ProgressChart} from 'react-native-chart-kit';
import {useLanguage} from '../context/LanguageContext';
import {scanAPI} from '../services/scanApi';

const screenWidth = Dimensions.get('window').width;

// Chart configuration
const chartConfig = {
  backgroundColor: '#16213e',
  backgroundGradientFrom: '#16213e',
  backgroundGradientTo: '#1a1a2e',
  decimalPlaces: 0,
  color: (opacity = 1) => `rgba(233, 69, 96, ${opacity})`,
  labelColor: (opacity = 1) => `rgba(170, 170, 170, ${opacity})`,
  style: {
    borderRadius: 16,
  },
  propsForDots: {
    r: '6',
    strokeWidth: '2',
    stroke: '#e94560',
  },
  barPercentage: 0.6,
};

// Simple Stat Card Component (no hooks to avoid order issues)
const StatCard = ({number, label, color, icon}) => (
  <View style={styles.statCard}>
    <Text style={styles.statIcon}>{icon}</Text>
    <Text style={[styles.statNumber, {color: color}]}>{number}</Text>
    <Text style={styles.statLabel}>{label}</Text>
  </View>
);

// Health Score Ring Component
const HealthScoreRing = ({score}) => {
  const getScoreColor = () => {
    if (score >= 80) return '#4caf50';
    if (score >= 60) return '#ff9800';
    return '#f44336';
  };

  const getScoreLabel = () => {
    if (score >= 80) return 'Excellent';
    if (score >= 60) return 'Good';
    if (score >= 40) return 'Fair';
    return 'Poor';
  };

  return (
    <View style={styles.healthScoreContainer}>
      <Text style={styles.sectionTitle}>Health Score</Text>
      <View style={styles.healthScoreRing}>
        <ProgressChart
          data={{
            data: [score / 100],
          }}
          width={160}
          height={160}
          strokeWidth={12}
          radius={60}
          chartConfig={{
            ...chartConfig,
            color: (opacity = 1) => {
              const color = getScoreColor();
              return color;
            },
          }}
          hideLegend
          style={{alignSelf: 'center'}}
        />
        <View style={styles.healthScoreCenter}>
          <Text style={[styles.healthScoreValue, {color: getScoreColor()}]}>
            {score}%
          </Text>
          <Text style={styles.healthScoreLabel}>{getScoreLabel()}</Text>
        </View>
      </View>
      <Text style={styles.healthScoreDesc}>
        Based on your recent scan results
      </Text>
    </View>
  );
};

export default function AnalyticsScreen({navigation, route}) {
  const {t} = useLanguage();
  const isAdmin = route.params?.isAdmin || false;

  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [period, setPeriod] = useState(30);
  const [stats, setStats] = useState(null);
  const [trends, setTrends] = useState([]);
  const [distribution, setDistribution] = useState(null);
  const [recentScans, setRecentScans] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadData();
  }, [period]);

  const loadData = async () => {
    try {
      setError(null);

      if (isAdmin) {
        const [analyticsRes, trendsRes, distRes] = await Promise.all([
          scanAPI.getAnalytics(period),
          scanAPI.getTrends(period),
          scanAPI.getPestDistribution(),
        ]);

        setStats(analyticsRes.data.overview);
        setTrends(trendsRes.data);
        setDistribution(distRes.data);
      } else {
        const [statsRes, scansRes] = await Promise.all([
          scanAPI.getMyStats(),
          scanAPI.getMyScans({limit: 100}),
        ]);

        setStats(statsRes.data);
        setRecentScans(scansRes.data.scans?.slice(0, 5) || []);
        processScansTrends(scansRes.data.scans || []);
        processDistribution(scansRes.data.scans || []);
      }
    } catch (err) {
      console.error('Load data error:', err);
      setError(err.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const processScansTrends = scans => {
    if (!scans || scans.length === 0) {
      setTrends([]);
      return;
    }

    const grouped = scans.reduce((acc, scan) => {
      const date = scan.createdAt?.split('T')[0];
      if (!date) return acc;
      if (!acc[date]) {
        acc[date] = {totalScans: 0, infectedScans: 0};
      }
      acc[date].totalScans++;
      if (scan.isInfected) acc[date].infectedScans++;
      return acc;
    }, {});

    const trendData = Object.entries(grouped)
      .map(([date, data]) => ({
        date,
        ...data,
        infectionRate:
          data.totalScans > 0 ? (data.infectedScans / data.totalScans) * 100 : 0,
      }))
      .sort((a, b) => a.date.localeCompare(b.date))
      .slice(-period);

    setTrends(trendData);
  };

  const processDistribution = scans => {
    if (!scans || scans.length === 0) {
      setDistribution(null);
      return;
    }

    let miteCount = 0;
    let caterpillarCount = 0;
    let healthyCount = 0;

    scans.forEach(scan => {
      if (!scan.isInfected) {
        healthyCount++;
      } else {
        if (scan.pestsDetected?.includes('coconut_mite')) miteCount++;
        if (scan.pestsDetected?.includes('caterpillar')) caterpillarCount++;
      }
    });

    setDistribution({
      miteOnly: miteCount,
      caterpillarOnly: caterpillarCount,
      healthy: healthyCount,
      total: scans.length,
    });
  };

  const onRefresh = () => {
    setRefreshing(true);
    loadData();
  };

  const calculateHealthScore = () => {
    if (!stats || stats.totalScans === 0) return 100;
    const healthyRate = ((stats.healthyScans || 0) / stats.totalScans) * 100;
    return Math.round(healthyRate);
  };

  const renderPeriodSelector = () => (
    <View style={styles.periodSelector}>
      {[7, 30, 90].map(days => (
        <TouchableOpacity
          key={days}
          style={[
            styles.periodButton,
            period === days && styles.periodButtonActive,
          ]}
          onPress={() => setPeriod(days)}>
          <Text
            style={[
              styles.periodButtonText,
              period === days && styles.periodButtonTextActive,
            ]}>
            {days}D
          </Text>
        </TouchableOpacity>
      ))}
    </View>
  );

  const renderStatsCards = () => (
    <View style={styles.statsContainer}>
      <View style={styles.statsRow}>
        <StatCard
          number={stats?.totalScans || 0}
          label={t('dashboard.totalScans')}
          color="#e94560"
          icon="📊"
        />
        <StatCard
          number={stats?.infectedScans || 0}
          label={t('dashboard.infectedTrees')}
          color="#f44336"
          icon="🐛"
        />
        <StatCard
          number={stats?.healthyScans || 0}
          label={t('dashboard.healthyTrees')}
          color="#4caf50"
          icon="✅"
        />
      </View>
    </View>
  );

  const renderInfectionRateCard = () => (
    <View style={styles.rateCard}>
      <View style={styles.rateHeader}>
        <Text style={styles.rateLabel}>Infection Rate</Text>
        <View
          style={[
            styles.rateBadge,
            {
              backgroundColor:
                parseFloat(stats?.infectionRate || 0) > 50
                  ? '#f44336'
                  : parseFloat(stats?.infectionRate || 0) > 25
                  ? '#ff9800'
                  : '#4caf50',
            },
          ]}>
          <Text style={styles.rateBadgeText}>
            {parseFloat(stats?.infectionRate || 0) > 50
              ? 'HIGH'
              : parseFloat(stats?.infectionRate || 0) > 25
              ? 'MEDIUM'
              : 'LOW'}
          </Text>
        </View>
      </View>
      <Text style={styles.rateValue}>
        {parseFloat(stats?.infectionRate || 0).toFixed(1)}%
      </Text>
      <View style={styles.rateBar}>
        <View
          style={[
            styles.rateBarFill,
            {
              width: `${Math.min(parseFloat(stats?.infectionRate || 0), 100)}%`,
              backgroundColor:
                parseFloat(stats?.infectionRate || 0) > 50
                  ? '#f44336'
                  : parseFloat(stats?.infectionRate || 0) > 25
                  ? '#ff9800'
                  : '#4caf50',
            },
          ]}
        />
      </View>
    </View>
  );

  const renderTrendChart = () => {
    if (!trends || trends.length < 2) {
      return (
        <View style={styles.chartCard}>
          <Text style={styles.chartTitle}>Scan Activity</Text>
          <View style={styles.chartPlaceholder}>
            <Text style={styles.placeholderIcon}>📈</Text>
            <Text style={styles.placeholderText}>
              Not enough data for trends
            </Text>
            <Text style={styles.placeholderSubtext}>
              Perform more scans to see trends
            </Text>
          </View>
        </View>
      );
    }

    const chartTrends = trends.slice(-7);

    const lineData = {
      labels: chartTrends.map(t => {
        const date = new Date(t.date);
        return `${date.getMonth() + 1}/${date.getDate()}`;
      }),
      datasets: [
        {
          data: chartTrends.map(t => t.totalScans || 0),
          color: (opacity = 1) => `rgba(76, 175, 80, ${opacity})`,
          strokeWidth: 2,
        },
        {
          data: chartTrends.map(t => t.infectedScans || 0),
          color: (opacity = 1) => `rgba(244, 67, 54, ${opacity})`,
          strokeWidth: 2,
        },
      ],
      legend: ['Total', 'Infected'],
    };

    return (
      <View style={styles.chartCard}>
        <Text style={styles.chartTitle}>Scan Activity Trend</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          <LineChart
            data={lineData}
            width={Math.max(screenWidth - 40, chartTrends.length * 60)}
            height={220}
            chartConfig={{
              ...chartConfig,
              color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
            }}
            bezier
            style={styles.chart}
            fromZero
          />
        </ScrollView>
      </View>
    );
  };

  const renderBarChart = () => {
    if (!trends || trends.length < 2) return null;

    const chartTrends = trends.slice(-7);

    const barData = {
      labels: chartTrends.map(t => {
        const date = new Date(t.date);
        return `${date.getDate()}`;
      }),
      datasets: [
        {
          data: chartTrends.map(t => Math.round(t.infectionRate || 0)),
        },
      ],
    };

    return (
      <View style={styles.chartCard}>
        <Text style={styles.chartTitle}>Daily Infection Rate (%)</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          <BarChart
            data={barData}
            width={Math.max(screenWidth - 40, chartTrends.length * 50)}
            height={200}
            chartConfig={{
              ...chartConfig,
              color: (opacity = 1) => `rgba(233, 69, 96, ${opacity})`,
            }}
            style={styles.chart}
            fromZero
            showValuesOnTopOfBars
          />
        </ScrollView>
      </View>
    );
  };

  const renderPestDistribution = () => {
    if (!distribution || distribution.total === 0) {
      return null;
    }

    const pieData = [];

    if (distribution.healthy > 0) {
      pieData.push({
        name: 'Healthy',
        count: distribution.healthy,
        color: '#4caf50',
        legendFontColor: '#aaa',
        legendFontSize: 12,
      });
    }

    if (distribution.miteOnly > 0) {
      pieData.push({
        name: t('pestDetection.coconutMite'),
        count: distribution.miteOnly,
        color: '#e94560',
        legendFontColor: '#aaa',
        legendFontSize: 12,
      });
    }

    if (distribution.caterpillarOnly > 0) {
      pieData.push({
        name: t('pestDetection.caterpillar'),
        count: distribution.caterpillarOnly,
        color: '#ff9800',
        legendFontColor: '#aaa',
        legendFontSize: 12,
      });
    }

    if (pieData.length === 0) return null;

    return (
      <View style={styles.chartCard}>
        <Text style={styles.chartTitle}>Scan Distribution</Text>
        <PieChart
          data={pieData}
          width={screenWidth - 40}
          height={200}
          chartConfig={chartConfig}
          accessor="count"
          backgroundColor="transparent"
          paddingLeft="15"
          absolute
        />
      </View>
    );
  };

  const renderRecentScans = () => {
    if (!recentScans || recentScans.length === 0) return null;

    return (
      <View style={styles.recentScansCard}>
        <View style={styles.recentScansHeader}>
          <Text style={styles.chartTitle}>Recent Scans</Text>
          <TouchableOpacity onPress={() => navigation.navigate('ScanHistory')}>
            <Text style={styles.viewAllText}>View All →</Text>
          </TouchableOpacity>
        </View>
        {recentScans.map((scan, index) => (
          <TouchableOpacity
            key={scan._id || index}
            style={styles.recentScanItem}
            onPress={() =>
              navigation.navigate('ScanDetail', {
                scanId: scan._id,
                scanData: scan,
              })
            }>
            <View style={styles.recentScanIcon}>
              <Text style={{fontSize: 20}}>
                {scan.isInfected ? '🐛' : '✅'}
              </Text>
            </View>
            <View style={styles.recentScanInfo}>
              <Text style={styles.recentScanType}>
                {scan.scanType?.toUpperCase()} Scan
              </Text>
              <Text
                style={[
                  styles.recentScanResult,
                  {color: scan.isInfected ? '#f44336' : '#4caf50'},
                ]}>
                {scan.isInfected ? 'Infected' : 'Healthy'}
              </Text>
            </View>
            <Text style={styles.recentScanDate}>
              {new Date(scan.createdAt).toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
              })}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    );
  };

  const renderQuickStats = () => (
    <View style={styles.quickStatsRow}>
      <View style={styles.quickStatItem}>
        <Text style={styles.quickStatValue}>
          {stats?.miteDetections || 0}
        </Text>
        <Text style={styles.quickStatLabel}>Mite Cases</Text>
      </View>
      <View style={styles.quickStatDivider} />
      <View style={styles.quickStatItem}>
        <Text style={styles.quickStatValue}>
          {stats?.caterpillarDetections || 0}
        </Text>
        <Text style={styles.quickStatLabel}>Caterpillar Cases</Text>
      </View>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#e94560" />
        <Text style={styles.loadingText}>{t('common.loading')}</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          colors={['#e94560']}
          tintColor="#e94560"
        />
      }>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()}>
          <Text style={styles.backButton}>← {t('common.back')}</Text>
        </TouchableOpacity>
        <Text style={styles.title}>
          {isAdmin ? 'System Analytics' : 'My Analytics'}
        </Text>
        {renderPeriodSelector()}
      </View>

      {error && (
        <View style={styles.errorCard}>
          <Text style={styles.errorText}>{error}</Text>
          <TouchableOpacity onPress={loadData}>
            <Text style={styles.retryText}>{t('common.retry')}</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Stats Cards */}
      {renderStatsCards()}

      {/* Health Score */}
      {!isAdmin && stats && stats.totalScans > 0 && (
        <HealthScoreRing score={calculateHealthScore()} />
      )}

      {/* Infection Rate */}
      {stats && renderInfectionRateCard()}

      {/* Quick Stats */}
      {stats && renderQuickStats()}

      {/* Charts */}
      {renderTrendChart()}
      {renderBarChart()}
      {renderPestDistribution()}

      {/* Recent Scans */}
      {!isAdmin && renderRecentScans()}

      <View style={{height: 40}} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1a1a2e',
  },
  loadingText: {
    color: '#aaa',
    marginTop: 10,
  },
  header: {
    padding: 20,
    paddingTop: 50,
    backgroundColor: '#16213e',
  },
  backButton: {
    color: '#e94560',
    fontSize: 16,
    marginBottom: 10,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 15,
  },
  periodSelector: {
    flexDirection: 'row',
    backgroundColor: '#1a1a2e',
    borderRadius: 25,
    padding: 4,
    alignSelf: 'flex-start',
  },
  periodButton: {
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 20,
  },
  periodButtonActive: {
    backgroundColor: '#e94560',
  },
  periodButtonText: {
    color: '#aaa',
    fontSize: 14,
    fontWeight: '600',
  },
  periodButtonTextActive: {
    color: '#fff',
  },
  statsContainer: {
    padding: 20,
    paddingBottom: 10,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  statCard: {
    backgroundColor: '#16213e',
    borderRadius: 15,
    padding: 15,
    alignItems: 'center',
    flex: 1,
    marginHorizontal: 5,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  statIcon: {
    fontSize: 24,
    marginBottom: 5,
  },
  statNumber: {
    fontSize: 28,
    fontWeight: 'bold',
  },
  statLabel: {
    fontSize: 11,
    color: '#aaa',
    marginTop: 5,
    textAlign: 'center',
  },
  healthScoreContainer: {
    backgroundColor: '#16213e',
    marginHorizontal: 20,
    marginBottom: 15,
    borderRadius: 15,
    padding: 20,
    alignItems: 'center',
  },
  healthScoreRing: {
    position: 'relative',
    alignItems: 'center',
    justifyContent: 'center',
  },
  healthScoreCenter: {
    position: 'absolute',
    alignItems: 'center',
  },
  healthScoreValue: {
    fontSize: 32,
    fontWeight: 'bold',
  },
  healthScoreLabel: {
    fontSize: 14,
    color: '#aaa',
  },
  healthScoreDesc: {
    fontSize: 12,
    color: '#666',
    marginTop: 10,
  },
  rateCard: {
    backgroundColor: '#16213e',
    borderRadius: 15,
    padding: 20,
    marginHorizontal: 20,
    marginBottom: 15,
  },
  rateHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  rateLabel: {
    fontSize: 16,
    color: '#aaa',
    fontWeight: '600',
  },
  rateBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  rateBadgeText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  rateValue: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#e94560',
    marginBottom: 10,
  },
  rateBar: {
    height: 10,
    backgroundColor: '#0f3460',
    borderRadius: 5,
    overflow: 'hidden',
  },
  rateBarFill: {
    height: '100%',
    borderRadius: 5,
  },
  quickStatsRow: {
    flexDirection: 'row',
    backgroundColor: '#16213e',
    marginHorizontal: 20,
    marginBottom: 15,
    borderRadius: 15,
    padding: 15,
    alignItems: 'center',
  },
  quickStatItem: {
    flex: 1,
    alignItems: 'center',
  },
  quickStatDivider: {
    width: 1,
    height: 40,
    backgroundColor: '#0f3460',
  },
  quickStatValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
  },
  quickStatLabel: {
    fontSize: 12,
    color: '#aaa',
    marginTop: 5,
  },
  chartCard: {
    backgroundColor: '#16213e',
    borderRadius: 15,
    padding: 15,
    marginHorizontal: 20,
    marginBottom: 15,
  },
  chartTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 15,
  },
  chart: {
    borderRadius: 10,
  },
  chartPlaceholder: {
    padding: 40,
    alignItems: 'center',
  },
  placeholderIcon: {
    fontSize: 40,
    marginBottom: 10,
  },
  placeholderText: {
    color: '#aaa',
    fontSize: 14,
  },
  placeholderSubtext: {
    color: '#666',
    fontSize: 12,
    marginTop: 5,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 15,
    alignSelf: 'flex-start',
  },
  recentScansCard: {
    backgroundColor: '#16213e',
    borderRadius: 15,
    padding: 15,
    marginHorizontal: 20,
    marginBottom: 15,
  },
  recentScansHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  viewAllText: {
    color: '#e94560',
    fontSize: 14,
  },
  recentScanItem: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#0f3460',
  },
  recentScanIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#0f3460',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  recentScanInfo: {
    flex: 1,
  },
  recentScanType: {
    fontSize: 12,
    color: '#aaa',
  },
  recentScanResult: {
    fontSize: 14,
    fontWeight: '600',
  },
  recentScanDate: {
    fontSize: 12,
    color: '#666',
  },
  errorCard: {
    backgroundColor: '#3d1515',
    borderRadius: 10,
    padding: 15,
    marginHorizontal: 20,
    marginTop: 10,
    alignItems: 'center',
  },
  errorText: {
    color: '#ff6b6b',
    marginBottom: 10,
  },
  retryText: {
    color: '#e94560',
    fontWeight: 'bold',
  },
});
