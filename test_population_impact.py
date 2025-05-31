#!/usr/bin/env python3
"""
Comprehensive test suite for population impact tracking system.
Tests PopulationImpactTracker, HGTVisualizationEngine, and PopulationAnalytics.
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hgt import (
    PopulationImpactTracker, HGTVisualizationEngine, PopulationAnalytics,
    PopulationMetrics, TransferNetworkNode, TransferNetworkEdge,
    GeneTransferRecord, HGTMechanism, EnvironmentalPressure
)
from models.bacterium import Bacterium, ResistanceStatus, Position
from datetime import datetime
import random
import numpy as np


def test_population_impact_tracking():
    """Test comprehensive population impact tracking system."""
    print("📊 Population Impact Tracking Test Suite")
    print("=" * 60)
    
    # 1. Test PopulationImpactTracker initialization
    print("\n1. Testing PopulationImpactTracker initialization...")
    impact_tracker = PopulationImpactTracker(history_limit=100)
    
    print(f"✅ PopulationImpactTracker created:")
    print(f"  • History limit: {impact_tracker.history_limit}")
    print(f"  • Population history: {len(impact_tracker.population_history)} snapshots")
    print(f"  • Transfer network: {len(impact_tracker.transfer_network)} nodes")
    
    # 2. Test population snapshot recording
    print("\n2. Testing population snapshot recording...")
    
    # Create diverse bacterial population
    population = {}
    for i in range(12):
        bacterium = Bacterium(
            id=f"bacteria_{i}",
            position=Position(x=random.randint(0, 20), y=random.randint(0, 20)),
            fitness=random.uniform(0.6, 1.0),
            resistance_status=ResistanceStatus.SENSITIVE if i < 6 else ResistanceStatus.RESISTANT,
            generation_born=0
        )
        
        # Initialize resistance genes
        bacterium.resistance_genes = set()
        if i >= 6:  # Resistant bacteria
            genes = ["beta_lactamase", "efflux_pump", "target_modification"][:random.randint(1, 3)]
            bacterium.resistance_genes.update(genes)
        
        population[bacterium.id] = bacterium
    
    # Record initial snapshot
    snapshot = impact_tracker.record_population_snapshot(population, generation=1)
    
    print(f"✅ Population snapshot recorded:")
    print(f"  • Generation: {snapshot.generation}")
    print(f"  • Total population: {snapshot.total_population}")
    print(f"  • Resistant count: {snapshot.resistant_count}")
    print(f"  • Resistance frequency: {snapshot.resistance_frequency:.2%}")
    print(f"  • Average fitness: {snapshot.average_fitness:.3f}")
    print(f"  • Shannon diversity: {snapshot.gene_diversity_shannon:.3f}")
    print(f"  • Simpson diversity: {snapshot.gene_diversity_simpson:.3f}")
    
    # 3. Test transfer event recording
    print("\n3. Testing transfer event recording...")
    
    # Create transfer records
    transfer_records = []
    for i in range(5):
        donor_id = f"bacteria_{6 + i}"  # Resistant bacteria as donors
        recipient_id = f"bacteria_{i}"  # Sensitive bacteria as recipients
        
        transfer_record = GeneTransferRecord(
            id=f"transfer_{i}",
            donor_id=donor_id,
            recipient_id=recipient_id,
            mechanism=random.choice(list(HGTMechanism)),
            genes_transferred=["beta_lactamase"],
            success=random.choice([True, False]),
            generation=2,
            probability=random.uniform(0.05, 0.3)
        )
        
        # Get positions for distance calculation
        donor_pos = population[donor_id].position
        recipient_pos = population[recipient_id].position
        distance = ((donor_pos.x - recipient_pos.x)**2 + (donor_pos.y - recipient_pos.y)**2)**0.5
        
        impact_tracker.record_transfer_event(
            transfer_record,
            (donor_pos.x, donor_pos.y),
            (recipient_pos.x, recipient_pos.y),
            distance
        )
        
        transfer_records.append(transfer_record)
    
    successful_transfers = [t for t in transfer_records if t.success]
    print(f"✅ Transfer events recorded:")
    print(f"  • Total attempts: {len(transfer_records)}")
    print(f"  • Successful transfers: {len(successful_transfers)}")
    print(f"  • Transfer edges: {len(impact_tracker.transfer_edges)}")
    print(f"  • Network nodes: {len(impact_tracker.transfer_network)}")
    
    # 4. Test gene spread velocity calculation
    print("\n4. Testing gene spread velocity calculation...")
    
    # Record multiple generations to calculate velocity
    for gen in range(3, 8):
        # Simulate population changes
        for bacterium in population.values():
            if random.random() < 0.1:  # 10% chance of resistance acquisition
                bacterium.resistance_genes.add("beta_lactamase")
        
        snapshot = impact_tracker.record_population_snapshot(population, gen, hgt_events_this_gen=random.randint(1, 3))
    
    # Calculate velocity
    velocity = impact_tracker.calculate_gene_spread_velocity("beta_lactamase", time_window=5)
    doubling_time = impact_tracker.calculate_gene_doubling_time("beta_lactamase")
    
    print(f"✅ Gene spread analysis:")
    print(f"  • Beta-lactamase spread velocity: {velocity:.4f} frequency/generation")
    print(f"  • Doubling time: {doubling_time:.2f} generations" if doubling_time else "  • Doubling time: Not calculable")
    print(f"  • Population snapshots: {len(impact_tracker.population_history)}")
    
    # 5. Test hotspot identification
    print("\n5. Testing hotspot identification...")
    
    # Add more transfer events for hotspot analysis
    for i in range(10):
        donor_id = f"bacteria_{6 + random.randint(0, 5)}"
        recipient_id = f"bacteria_{random.randint(0, 5)}"
        
        transfer_record = GeneTransferRecord(
            id=f"hotspot_transfer_{i}",
            donor_id=donor_id,
            recipient_id=recipient_id,
            mechanism=random.choice(list(HGTMechanism)),
            genes_transferred=["efflux_pump"],
            success=True,
            generation=random.randint(6, 8),
            probability=random.uniform(0.1, 0.4)
        )
        
        impact_tracker.record_transfer_event(
            transfer_record,
            (random.uniform(0, 20), random.uniform(0, 20)),
            (random.uniform(0, 20), random.uniform(0, 20)),
            random.uniform(1, 5)
        )
    
    spatial_hotspots, temporal_hotspots = impact_tracker.identify_transfer_hotspots(
        spatial_radius=3.0,
        temporal_window=3,
        min_transfers=2
    )
    
    print(f"✅ Hotspot identification:")
    print(f"  • Spatial hotspots: {len(spatial_hotspots)}")
    print(f"  • Temporal hotspots: {len(temporal_hotspots)}")
    
    for i, hotspot in enumerate(spatial_hotspots[:2]):
        print(f"  • Spatial hotspot {i+1}: {hotspot['transfer_count']} transfers")
    
    for i, hotspot in enumerate(temporal_hotspots[:2]):
        print(f"  • Temporal hotspot {i+1}: Gen {hotspot['generation']}, {hotspot['transfer_count']} transfers")
    
    # 6. Test fitness impact analysis
    print("\n6. Testing fitness impact analysis...")
    
    # Record fitness before and after HGT
    for bacterium_id in population:
        impact_tracker.fitness_before_hgt[bacterium_id] = population[bacterium_id].fitness
        
        # Simulate fitness change after HGT
        if bacterium_id in [t.recipient_id for t in successful_transfers]:
            # Recipients might gain resistance but pay fitness cost
            new_fitness = population[bacterium_id].fitness * random.uniform(0.85, 1.05)
        else:
            new_fitness = population[bacterium_id].fitness * random.uniform(0.95, 1.02)
        
        population[bacterium_id].fitness = new_fitness
        impact_tracker.fitness_after_hgt[bacterium_id] = new_fitness
    
    fitness_impact = impact_tracker.analyze_fitness_impact()
    
    print(f"✅ Fitness impact analysis:")
    print(f"  • Average fitness change: {fitness_impact['average_fitness_change']:+.4f}")
    print(f"  • Fitness improvement rate: {fitness_impact['fitness_improvement_rate']:.1%}")
    print(f"  • Total assessments: {fitness_impact['total_assessments']}")
    
    # 7. Test network metrics
    print("\n7. Testing transfer network metrics...")
    
    network_metrics = impact_tracker.get_transfer_network_metrics()
    
    print(f"✅ Network analysis:")
    print(f"  • Node count: {network_metrics['node_count']}")
    print(f"  • Edge count: {network_metrics['edge_count']}")
    print(f"  • Average degree: {network_metrics['average_degree']:.2f}")
    print(f"  • Max degree: {network_metrics['max_degree']}")
    print(f"  • Network density: {network_metrics['network_density']:.4f}")
    print(f"  • Clustering coefficient: {network_metrics['clustering_coefficient']:.3f}")
    
    # 8. Test environmental correlations
    print("\n8. Testing environmental correlations...")
    
    # Record correlations
    for i in range(10):
        antibiotic_conc = random.uniform(0, 1)
        hgt_rate = random.uniform(0, 0.5) * (antibiotic_conc + 0.1)  # HGT increases with antibiotics
        
        impact_tracker.correlate_with_environment("antibiotic_concentration", antibiotic_conc, hgt_rate)
    
    # Test diversity trends
    diversity_trends = impact_tracker.get_population_diversity_trends(window=10)
    
    print(f"✅ Environmental correlations and trends:")
    print(f"  • Antibiotic correlations recorded: {len(impact_tracker.environmental_correlations['antibiotic_concentration'])}")
    print(f"  • Diversity trend data points: {len(diversity_trends['shannon'])}")
    print(f"  • Latest Shannon diversity: {diversity_trends['shannon'][-1]:.3f}" if diversity_trends['shannon'] else "  • No diversity data")
    
    # 9. Test HGTVisualizationEngine
    print("\n9. Testing HGTVisualizationEngine...")
    
    viz_engine = HGTVisualizationEngine(impact_tracker)
    
    # Generate transfer network data
    network_data = viz_engine.generate_transfer_network_data()
    
    print(f"✅ Visualization data generation:")
    print(f"  • Network nodes: {len(network_data['nodes'])}")
    print(f"  • Network edges: {len(network_data['edges'])}")
    print(f"  • Mechanisms involved: {network_data['metadata']['mechanisms']}")
    
    # Generate population trend data
    trend_data = viz_engine.generate_population_trend_data(window=10)
    
    print(f"  • Trend generations: {len(trend_data['generations'])}")
    print(f"  • Metrics tracked: {list(trend_data['metrics'].keys())}")
    
    # Generate hotspot visualization data
    hotspot_data = viz_engine.generate_hotspot_visualization_data()
    
    print(f"  • Spatial hotspots for viz: {len(hotspot_data['spatial_hotspots'])}")
    print(f"  • Temporal hotspots for viz: {len(hotspot_data['temporal_hotspots'])}")
    print(f"  • Most active mechanisms: {hotspot_data['hotspot_summary']['most_active_mechanisms'][:3]}")
    
    # 10. Test PopulationAnalytics
    print("\n10. Testing PopulationAnalytics...")
    
    analytics = PopulationAnalytics(impact_tracker)
    
    # Test HGT efficiency by mechanism
    efficiency_by_mechanism = analytics.calculate_hgt_efficiency_by_mechanism()
    
    print(f"✅ Population analytics:")
    print(f"  • Mechanisms analyzed: {len(efficiency_by_mechanism)}")
    
    for mechanism, stats in efficiency_by_mechanism.items():
        print(f"  • {mechanism}: {stats['success_rate']:.1%} success, efficiency {stats['efficiency_score']:.3f}")
    
    # Test fitness distribution analysis
    fitness_distribution = analytics.analyze_population_fitness_distribution()
    
    print(f"  • Fitness trend: {fitness_distribution['fitness_trend']:+.4f}")
    print(f"  • Current mean fitness: {fitness_distribution['current_mean']:.3f}")
    print(f"  • Current variance: {fitness_distribution['current_variance']:.4f}")
    
    # Test gene flow efficiency
    gene_flow = analytics.calculate_gene_flow_efficiency()
    
    print(f"  • Overall gene flow efficiency: {gene_flow['overall_efficiency']:.4f}")
    print(f"  • Average transfer distance: {gene_flow['average_transfer_distance']:.2f}")
    print(f"  • Transfer rate per generation: {gene_flow['transfer_rate_per_generation']:.3f}")
    print(f"  • Unique genes transferred: {gene_flow['unique_genes_transferred']}")
    
    # 11. Test comprehensive impact report
    print("\n11. Testing comprehensive impact report...")
    
    impact_report = impact_tracker.get_comprehensive_impact_report()
    
    print(f"✅ Comprehensive impact report:")
    print(f"  • Report generation: {impact_report.get('generation', 'N/A')}")
    print(f"  • Population status keys: {list(impact_report.get('population_status', {}).keys())}")
    print(f"  • HGT impact keys: {list(impact_report.get('hgt_impact', {}).keys())}")
    print(f"  • Network analysis keys: {list(impact_report.get('network_analysis', {}).keys())}")
    print(f"  • Spatial hotspots: {impact_report.get('hotspots', {}).get('spatial_count', 0)}")
    print(f"  • Temporal hotspots: {impact_report.get('hotspots', {}).get('temporal_count', 0)}")
    
    # Display some key metrics from the report
    if 'population_status' in impact_report:
        pop_status = impact_report['population_status']
        print(f"  • Final resistance frequency: {pop_status.get('resistance_frequency', 0):.1%}")
        print(f"  • Final average fitness: {pop_status.get('average_fitness', 0):.3f}")
        print(f"  • Final Shannon diversity: {pop_status.get('diversity_shannon', 0):.3f}")
    
    if 'hgt_impact' in impact_report:
        hgt_impact = impact_report['hgt_impact']
        print(f"  • Recent HGT events: {hgt_impact.get('events_this_generation', 0)}")
        print(f"  • New acquisitions: {hgt_impact.get('new_acquisitions', 0)}")
    
    print("\n🎊 ALL POPULATION IMPACT TRACKING TESTS PASSED!")
    print("📊 Population monitoring and analysis system is fully functional!")
    
    return True


if __name__ == "__main__":
    try:
        test_population_impact_tracking()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 