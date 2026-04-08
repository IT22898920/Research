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

      setPlantations(plantationsRes.data || []);
      setAnalytics(analyticsRes.data || null);
    } catch (error) {
      console.error('Error loading plantations:', error);
      Alert.alert('Error', 'Failed to load plantations');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
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
              <View
                style={[
                  styles.healthBarFill,
                  {
                    width: `${(stats.healthyTrees / totalTrees) * 100}%`,
                    backgroundColor: '#4CAF50',
                  },
                ]}
              />
              <View
                style={[
                  styles.healthBarFill,
                  {
                    width: `${(stats.unhealthyTrees / totalTrees) * 100}%`,
                    backgroundColor: '#FF9800',
                  },
                ]}
              />
              <View
                style={[
                  styles.healthBarFill,
                  {
                    width: `${(stats.infectedTrees / totalTrees) * 100}%`,
                    backgroundColor: '#F44336',
                  },
                ]}
              />
              <View
                style={[
                  styles.healthBarFill,
                  {
                    width: `${(stats.notScannedTrees / totalTrees) * 100}%`,
                    backgroundColor: '#9E9E9E',
                  },
                ]}
              />
            </View>
            <Text style={styles.healthPercent}>{healthyPercent}% Healthy</Text>
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
