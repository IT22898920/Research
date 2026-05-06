import React, {useState, useEffect, useCallback} from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  RefreshControl,
  Alert,
  ActivityIndicator,
} from 'react-native';
import {useFocusEffect} from '@react-navigation/native';
import {plantationAPI} from '../services/plantationApi';
import {treeAPI} from '../services/treeApi';

export default function PlantationListScreen({navigation}) {
  const [plantations, setPlantations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [analytics, setAnalytics] = useState(null);

  // Load plantations when screen is focused
  useFocusEffect(
    useCallback(() => {
      loadPlantations();
    }, []),
  );

  const loadPlantations = async () => {
    try {
      const [plantationsRes, analyticsRes] = await Promise.all([
        plantationAPI.getAll(),
        plantationAPI.getAnalytics(),
      ]);

      const plantationsList = plantationsRes.data || [];

      // Fetch detailed tree stats for each plantation
      const plantationsWithDetails = await Promise.all(
        plantationsList.map(async (plant) => {
          try {
            const treesRes = await treeAPI.getTreesByPlantation(plant._id);
            const trees = treesRes.trees || [];
            const detailedStats = computeDetailedStats(trees);
            return {...plant, detailedStats};
          } catch (e) {
            return {...plant, detailedStats: null};
          }
        }),
      );

      setPlantations(plantationsWithDetails);
      setAnalytics(analyticsRes.data || null);
    } catch (error) {
      console.error('Error loading plantations:', error);
      Alert.alert('Error', 'Failed to load plantations');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  // Compute detailed stats from trees (pest/disease counts, yield, etc.)
  const computeDetailedStats = (trees) => {
    const stats = {
      pests: {mite: 0, caterpillar: 0, white_fly: 0, total_with_pests: 0},
      diseases: {leaf_rot: 0, leaf_spot: 0, leaf_dieback: 0, total_with_disease: 0},
      health: {nitrogen: 0, potassium: 0, magnesium: 0, water_stress: 0, iron: 0, total_unhealthy_leaves: 0},
      yield: {totalNuts: 0, totalBunches: 0, scannedTrees: 0, projected7Months: 0},
      branchHealth: {unhealthy: 0},
    };

    trees.forEach((tree) => {
      // Combine ALL detected issues from current detectedIssues + entire healthHistory
      const allIssues = new Set();

      // Add current detected issues
      (tree.detectedIssues || []).forEach((i) => allIssues.add(i.toLowerCase()));

      // Add issues from health history scans
      (tree.healthHistory || []).forEach((scan) => {
        // Check scanType + status to detect issues
        const scanType = (scan.scanType || '').toLowerCase();
        const status = (scan.status || '').toLowerCase();
        const detailsText = JSON.stringify(scan.details || {}).toLowerCase();

        // Pests detected (status infected for pest scan types)
        if (status === 'infected' || status === 'unhealthy') {
          if (scanType.includes('mite') || detailsText.includes('mite')) allIssues.add('mite_history');
          if (scanType.includes('caterpillar') || detailsText.includes('caterpillar')) allIssues.add('caterpillar_history');
          if (scanType.includes('white') || detailsText.includes('white_fly') || detailsText.includes('white fly')) allIssues.add('white_fly_history');
          if (scanType === 'disease' || detailsText.includes('leaf rot') || detailsText.includes('leaf_rot')) {
            if (detailsText.includes('rot')) allIssues.add('leaf_rot_history');
            if (detailsText.includes('spot')) allIssues.add('leaf_spot_history');
          }
          if (scanType === 'leaf_dieback' || detailsText.includes('die_back')) allIssues.add('leaf_dieback_history');
          if (scanType === 'leaf_health') {
            if (detailsText.includes('nitrogen')) allIssues.add('nitrogen_history');
            if (detailsText.includes('potassium')) allIssues.add('potassium_history');
            if (detailsText.includes('magnesium')) allIssues.add('magnesium_history');
            if (detailsText.includes('water')) allIssues.add('water_stress_history');
            if (detailsText.includes('iron')) allIssues.add('iron_history');
            allIssues.add('unhealthy_leaf_history');
          }
          if (scanType === 'branch_health') allIssues.add('unhealthy_branch_history');
        }
      });

      const issuesText = Array.from(allIssues).join(' ');

      // Pests (each tree counted once per pest type)
      let hasPest = false;
      if (issuesText.includes('mite')) {stats.pests.mite++; hasPest = true;}
      if (issuesText.includes('caterpillar')) {stats.pests.caterpillar++; hasPest = true;}
      if (issuesText.includes('white_fly') || issuesText.includes('white fly')) {stats.pests.white_fly++; hasPest = true;}
      if (hasPest) stats.pests.total_with_pests++;

      // Diseases
      let hasDisease = false;
      if (issuesText.includes('leaf_rot') || issuesText.includes('leaf rot')) {stats.diseases.leaf_rot++; hasDisease = true;}
      if (issuesText.includes('leaf_spot') || issuesText.includes('leaf spot')) {stats.diseases.leaf_spot++; hasDisease = true;}
      if (issuesText.includes('dieback') || issuesText.includes('die_back')) {stats.diseases.leaf_dieback++; hasDisease = true;}
      if (hasDisease) stats.diseases.total_with_disease++;

      // Leaf health issues
      let hasHealthIssue = false;
      if (issuesText.includes('nitrogen')) {stats.health.nitrogen++; hasHealthIssue = true;}
      if (issuesText.includes('potassium')) {stats.health.potassium++; hasHealthIssue = true;}
      if (issuesText.includes('magnesium')) {stats.health.magnesium++; hasHealthIssue = true;}
      if (issuesText.includes('water')) {stats.health.water_stress++; hasHealthIssue = true;}
      if (issuesText.includes('iron')) {stats.health.iron++; hasHealthIssue = true;}
      if (issuesText.includes('unhealthy_leaf')) hasHealthIssue = true;
      if (hasHealthIssue) stats.health.total_unhealthy_leaves++;

      // Branch health
      if (issuesText.includes('unhealthy_branch')) stats.branchHealth.unhealthy++;

      // Yield
      if (tree.nutCount) {
        stats.yield.totalNuts += tree.nutCount;
        stats.yield.scannedTrees++;
      }
      if (tree.bunchCount) {
        stats.yield.totalBunches += tree.bunchCount;
      }
    });

    // 7-month yield = direct sum of nut counts from scanned trees (no multiplication)
    stats.yield.projected7Months = stats.yield.totalNuts;

    return stats;
  };

  const handleRefresh = () => {
    setRefreshing(true);
    loadPlantations();
  };

  const handleDeletePlantation = (plantation) => {
    Alert.alert(
      'Delete Plantation',
      `Are you sure you want to delete "${plantation.name}"?\n\nThis will also delete all ${plantation.stats?.totalTrees || 0} trees in this plantation.`,
      [
        {text: 'Cancel', style: 'cancel'},
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await plantationAPI.delete(plantation._id);
              loadPlantations();
              Alert.alert('Success', 'Plantation deleted successfully');
            } catch (error) {
              Alert.alert('Error', 'Failed to delete plantation');
            }
          },
        },
      ],
    );
  };

  const getHealthColor = (status) => {
    if (status === 'healthy') return '#4CAF50';
    if (status === 'unhealthy') return '#FF9800';
    if (status === 'infected') return '#F44336';
    return '#9E9E9E';
  };

  const renderPlantationCard = ({item}) => {
    const stats = item.stats || {};
    const totalTrees = stats.totalTrees || 0;
    const healthyPercent = totalTrees > 0 ? Math.round((stats.healthyTrees / totalTrees) * 100) : 0;

    return (
      <TouchableOpacity
        style={styles.card}
        onPress={() =>
          navigation.navigate('PlantationMap', {
            plantationId: item._id,
            plantationName: item.name,
          })
        }>
        {/* Card Header */}
        <View style={styles.cardHeader}>
          <View style={styles.cardTitleRow}>
            <Text style={styles.cardIcon}>🌴</Text>
            <Text style={styles.cardTitle}>{item.name}</Text>
          </View>
          <View style={styles.cardActions}>
            <TouchableOpacity
              style={styles.actionButton}
              onPress={() =>
                navigation.navigate('EditPlantation', {plantation: item})
              }>
              <Text style={styles.actionIcon}>✏️</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.actionButton}
              onPress={() => handleDeletePlantation(item)}>
              <Text style={styles.actionIcon}>🗑️</Text>
            </TouchableOpacity>
          </View>
        </View>

        {/* Description */}
        {item.description && (
          <Text style={styles.cardDescription} numberOfLines={2}>
            {item.description}
          </Text>
        )}

        {/* Stats */}
        <View style={styles.statsContainer}>
          <View style={styles.statItem}>
            <Text style={styles.statNumber}>{totalTrees}</Text>
            <Text style={styles.statLabel}>Trees</Text>
          </View>
          <View style={[styles.statItem, {borderLeftWidth: 1, borderLeftColor: '#eee'}]}>
            <Text style={[styles.statNumber, {color: '#4CAF50'}]}>
              {stats.healthyTrees || 0}
            </Text>
            <Text style={styles.statLabel}>Healthy</Text>
          </View>
          <View style={[styles.statItem, {borderLeftWidth: 1, borderLeftColor: '#eee'}]}>
            <Text style={[styles.statNumber, {color: '#FF9800'}]}>
              {stats.unhealthyTrees || 0}
            </Text>
            <Text style={styles.statLabel}>Unhealthy</Text>
          </View>
          <View style={[styles.statItem, {borderLeftWidth: 1, borderLeftColor: '#eee'}]}>
            <Text style={[styles.statNumber, {color: '#F44336'}]}>
              {stats.infectedTrees || 0}
            </Text>
            <Text style={styles.statLabel}>Infected</Text>
          </View>
        </View>

        {/* Health Bar */}
        {totalTrees > 0 && (
          <View style={styles.healthBarContainer}>
            <View style={styles.healthBar}>
              <View style={[styles.healthBarFill, {width: `${(stats.healthyTrees / totalTrees) * 100}%`, backgroundColor: '#4CAF50'}]} />
              <View style={[styles.healthBarFill, {width: `${(stats.unhealthyTrees / totalTrees) * 100}%`, backgroundColor: '#FF9800'}]} />
              <View style={[styles.healthBarFill, {width: `${(stats.infectedTrees / totalTrees) * 100}%`, backgroundColor: '#F44336'}]} />
              <View style={[styles.healthBarFill, {width: `${(stats.notScannedTrees / totalTrees) * 100}%`, backgroundColor: '#9E9E9E'}]} />
            </View>
            <Text style={styles.healthPercent}>{healthyPercent}% Healthy</Text>
          </View>
        )}

        {/* Detailed Stats Summary */}
        {item.detailedStats && totalTrees > 0 && (
          <View style={styles.detailedStatsContainer}>
            {/* Yield Forecast */}
            {item.detailedStats.yield.projected7Months > 0 && (
              <View style={styles.summaryCard}>
                <Text style={styles.summaryHeader}>🥥 7-Month Harvest Forecast</Text>
                <Text style={styles.yieldNumber}>{item.detailedStats.yield.projected7Months.toLocaleString()} nuts</Text>
                <Text style={styles.summarySubtext}>
                  Sum of young nuts on {item.detailedStats.yield.scannedTrees} tree(s) — these will mature for harvest in 7 months
                </Text>
                <Text style={styles.summarySubtext}>
                  🌴 {item.detailedStats.yield.totalBunches} bunches counted
                </Text>
              </View>
            )}

            {/* Pest Detection Summary */}
            {item.detailedStats.pests.total_with_pests > 0 && (
              <View style={[styles.summaryCard, {backgroundColor: '#FFEBEE'}]}>
                <Text style={[styles.summaryHeader, {color: '#C62828'}]}>🐛 Pest Detection ({item.detailedStats.pests.total_with_pests} trees)</Text>
                <View style={styles.summaryRow}>
                  {item.detailedStats.pests.mite > 0 && <Text style={styles.summaryItem}>• Mite: <Text style={styles.summaryCount}>{item.detailedStats.pests.mite}</Text></Text>}
                  {item.detailedStats.pests.caterpillar > 0 && <Text style={styles.summaryItem}>• Caterpillar: <Text style={styles.summaryCount}>{item.detailedStats.pests.caterpillar}</Text></Text>}
                  {item.detailedStats.pests.white_fly > 0 && <Text style={styles.summaryItem}>• White Fly: <Text style={styles.summaryCount}>{item.detailedStats.pests.white_fly}</Text></Text>}
                </View>
              </View>
            )}

            {/* Disease Detection Summary */}
            {item.detailedStats.diseases.total_with_disease > 0 && (
              <View style={[styles.summaryCard, {backgroundColor: '#FFF3E0'}]}>
                <Text style={[styles.summaryHeader, {color: '#E65100'}]}>🍃 Disease Detection ({item.detailedStats.diseases.total_with_disease} trees)</Text>
                <View style={styles.summaryRow}>
                  {item.detailedStats.diseases.leaf_rot > 0 && <Text style={styles.summaryItem}>• Leaf Rot: <Text style={styles.summaryCount}>{item.detailedStats.diseases.leaf_rot}</Text></Text>}
                  {item.detailedStats.diseases.leaf_spot > 0 && <Text style={styles.summaryItem}>• Leaf Spot: <Text style={styles.summaryCount}>{item.detailedStats.diseases.leaf_spot}</Text></Text>}
                  {item.detailedStats.diseases.leaf_dieback > 0 && <Text style={styles.summaryItem}>• Leaf Dieback: <Text style={styles.summaryCount}>{item.detailedStats.diseases.leaf_dieback}</Text></Text>}
                </View>
              </View>
            )}

            {/* Leaf Health Issues */}
            {item.detailedStats.health.total_unhealthy_leaves > 0 && (
              <View style={[styles.summaryCard, {backgroundColor: '#FFFDE7'}]}>
                <Text style={[styles.summaryHeader, {color: '#F57F17'}]}>🌿 Leaf Health Issues ({item.detailedStats.health.total_unhealthy_leaves} trees)</Text>
                <View style={styles.summaryRow}>
                  {item.detailedStats.health.nitrogen > 0 && <Text style={styles.summaryItem}>• N Deficiency: <Text style={styles.summaryCount}>{item.detailedStats.health.nitrogen}</Text></Text>}
                  {item.detailedStats.health.potassium > 0 && <Text style={styles.summaryItem}>• K Deficiency: <Text style={styles.summaryCount}>{item.detailedStats.health.potassium}</Text></Text>}
                  {item.detailedStats.health.magnesium > 0 && <Text style={styles.summaryItem}>• Mg Deficiency: <Text style={styles.summaryCount}>{item.detailedStats.health.magnesium}</Text></Text>}
                  {item.detailedStats.health.water_stress > 0 && <Text style={styles.summaryItem}>• Water Stress: <Text style={styles.summaryCount}>{item.detailedStats.health.water_stress}</Text></Text>}
                  {item.detailedStats.health.iron > 0 && <Text style={styles.summaryItem}>• Iron Deficiency: <Text style={styles.summaryCount}>{item.detailedStats.health.iron}</Text></Text>}
                </View>
              </View>
            )}

            {/* Branch Health */}
            {item.detailedStats.branchHealth.unhealthy > 0 && (
              <View style={[styles.summaryCard, {backgroundColor: '#FFEBEE'}]}>
                <Text style={[styles.summaryHeader, {color: '#C62828'}]}>🌳 Branch Issues</Text>
                <Text style={styles.summaryItem}>• Unhealthy Branches: <Text style={styles.summaryCount}>{item.detailedStats.branchHealth.unhealthy} trees</Text></Text>
              </View>
            )}
          </View>
        )}

        {/* View Map Button */}
        <TouchableOpacity
          style={styles.viewMapButton}
          onPress={() =>
            navigation.navigate('PlantationMap', {
              plantationId: item._id,
              plantationName: item.name,
            })
          }>
          <Text style={styles.viewMapText}>🗺️ View on Map</Text>
        </TouchableOpacity>
      </TouchableOpacity>
    );
  };

  const renderEmptyState = () => (
    <View style={styles.emptyState}>
      <Text style={styles.emptyIcon}>🌱</Text>
      <Text style={styles.emptyTitle}>No Plantations Yet</Text>
      <Text style={styles.emptyDescription}>
        Create your first plantation to start tracking your coconut trees
      </Text>
      <TouchableOpacity
        style={styles.createButton}
        onPress={() => navigation.navigate('AddPlantation')}>
        <Text style={styles.createButtonText}>+ Create Plantation</Text>
      </TouchableOpacity>
    </View>
  );

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#2e7d32" />
        <Text style={styles.loadingText}>Loading plantations...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => navigation.goBack()}>
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>My Plantations</Text>
        <View style={styles.headerRight} />
      </View>

      {/* Overall Stats */}
      {analytics && (
        <View style={styles.overallStats}>
          <View style={styles.overallStatItem}>
            <Text style={styles.overallStatNumber}>
              {analytics.totalPlantations || 0}
            </Text>
            <Text style={styles.overallStatLabel}>Plantations</Text>
          </View>
          <View style={styles.overallStatItem}>
            <Text style={styles.overallStatNumber}>
              {analytics.treeStats?.totalTrees || 0}
            </Text>
            <Text style={styles.overallStatLabel}>Total Trees</Text>
          </View>
          <View style={styles.overallStatItem}>
            <Text style={[styles.overallStatNumber, {color: '#4CAF50'}]}>
              {analytics.treeStats?.healthyTrees || 0}
            </Text>
            <Text style={styles.overallStatLabel}>Healthy</Text>
          </View>
        </View>
      )}

      {/* Plantation List */}
      <FlatList
        data={plantations}
        renderItem={renderPlantationCard}
        keyExtractor={(item) => item._id}
        contentContainerStyle={styles.listContainer}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={handleRefresh}
            colors={['#2e7d32']}
          />
        }
        ListEmptyComponent={renderEmptyState}
      />

      {/* FAB - Add Plantation */}
      <TouchableOpacity
        style={styles.fab}
        onPress={() => navigation.navigate('AddPlantation')}>
        <Text style={styles.fabText}>+</Text>
      </TouchableOpacity>
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
    marginTop: 10,
    fontSize: 14,
    color: '#666',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 15,
    paddingTop: 50,
    paddingBottom: 15,
    backgroundColor: '#2e7d32',
  },
  backButton: {
    padding: 5,
  },
  backButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '500',
  },
  headerTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  headerRight: {
    width: 60,
  },
  overallStats: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    paddingVertical: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  overallStatItem: {
    flex: 1,
    alignItems: 'center',
  },
  overallStatNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#333',
  },
  overallStatLabel: {
    fontSize: 12,
    color: '#666',
    marginTop: 2,
  },
  listContainer: {
    padding: 15,
    paddingBottom: 80,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 15,
    marginBottom: 15,
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 2},
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  cardTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  cardIcon: {
    fontSize: 24,
    marginRight: 10,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    flex: 1,
  },
  cardActions: {
    flexDirection: 'row',
  },
  actionButton: {
    padding: 8,
    marginLeft: 5,
  },
  actionIcon: {
    fontSize: 18,
  },
  cardDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 12,
    lineHeight: 20,
  },
  statsContainer: {
    flexDirection: 'row',
    backgroundColor: '#f8f9fa',
    borderRadius: 8,
    padding: 10,
    marginBottom: 12,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: 5,
  },
  statNumber: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  statLabel: {
    fontSize: 11,
    color: '#666',
    marginTop: 2,
  },
  healthBarContainer: {
    marginBottom: 12,
  },
  healthBar: {
    flexDirection: 'row',
    height: 8,
    backgroundColor: '#eee',
    borderRadius: 4,
    overflow: 'hidden',
  },
  healthBarFill: {
    height: '100%',
  },
  healthPercent: {
    fontSize: 12,
    color: '#666',
    textAlign: 'right',
    marginTop: 4,
  },
  detailedStatsContainer: {
    marginTop: 4,
    marginBottom: 8,
  },
  summaryCard: {
    backgroundColor: '#E8F5E9',
    borderRadius: 8,
    padding: 10,
    marginBottom: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#2E7D32',
  },
  summaryHeader: {
    fontSize: 13,
    fontWeight: 'bold',
    color: '#2E7D32',
    marginBottom: 4,
  },
  yieldNumber: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#2E7D32',
    marginVertical: 2,
  },
  summarySubtext: {
    fontSize: 11,
    color: '#666',
    fontStyle: 'italic',
  },
  summaryRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  summaryItem: {
    fontSize: 12,
    color: '#444',
    marginRight: 8,
    marginVertical: 1,
  },
  summaryCount: {
    fontWeight: 'bold',
    color: '#000',
  },
  viewMapButton: {
    backgroundColor: '#e8f5e9',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  viewMapText: {
    color: '#2e7d32',
    fontSize: 14,
    fontWeight: '600',
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
  },
  emptyIcon: {
    fontSize: 60,
    marginBottom: 15,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#333',
    marginBottom: 10,
  },
  emptyDescription: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    paddingHorizontal: 40,
    marginBottom: 20,
  },
  createButton: {
    backgroundColor: '#2e7d32',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 25,
  },
  createButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  fab: {
    position: 'absolute',
    right: 20,
    bottom: 20,
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#2e7d32',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {width: 0, height: 4},
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
  fabText: {
    color: '#fff',
    fontSize: 32,
    fontWeight: '300',
    marginTop: -2,
  },
});
