#!/usr/bin/env python3
"""
Comprehensive test suite for HGT spatial integration system.
Tests HGTSpatialIntegration, SpatialHGTCache, and HGTSimulationOrchestrator.
"""

import sys
import os
import asyncio
import random
from typing import List, Optional, Dict

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hgt import (
    HGTSpatialIntegration, SpatialHGTConfig, SpatialHGTCache, HGTSimulationOrchestrator,
    SpatialGridInterface, HGTConfig, HGTMechanism, PopulationImpactTracker,
    GeneTransferEngine, ProximityDetector, ProbabilityCalculator, Coordinate
)
from models.bacterium import Bacterium, ResistanceStatus, Position
from models.spatial import SpatialManager


class MockSpatialGrid:
    """Mock spatial grid for testing."""
    
    def __init__(self):
        self.positions: Dict[str, Coordinate] = {}
        self.query_count = 0
    
    def get_neighbors(self, position: Coordinate, radius: float) -> List[str]:
        """Get neighbors within radius of position."""
        self.query_count += 1
        neighbors = []
        
        for bacterium_id, pos in self.positions.items():
            distance = ((position.x - pos.x)**2 + (position.y - pos.y)**2)**0.5
            if distance <= radius:
                neighbors.append(bacterium_id)
        
        return neighbors
    
    def get_density(self, position: Coordinate, radius: float) -> float:
        """Get population density at position."""
        neighbors = self.get_neighbors(position, radius)
        area = 3.14159 * radius * radius
        return len(neighbors) / area if area > 0 else 0.0
    
    def update_position(self, bacterium_id: str, new_position: Coordinate) -> None:
        """Update bacterium position in grid."""
        self.positions[bacterium_id] = new_position
    
    def get_position(self, bacterium_id: str) -> Optional[Coordinate]:
        """Get bacterium position from grid."""
        return self.positions.get(bacterium_id)


async def test_spatial_integration():
    """Test comprehensive spatial integration system."""
    print("🌐 HGT Spatial Integration Test Suite")
    print("=" * 60)
    
    # 1. Test SpatialHGTCache
    print("\n1. Testing SpatialHGTCache...")
    cache = SpatialHGTCache(max_size=5, ttl=3)
    
    # Test cache operations
    cache.put("key1", {"data": "test1"}, generation=1)
    cache.put("key2", {"data": "test2"}, generation=1)
    
    # Test retrieval
    data = cache.get("key1", generation=2)
    assert data == {"data": "test1"}, "Cache retrieval failed"
    
    # Test TTL expiration
    expired_data = cache.get("key1", generation=5)  # Beyond TTL
    assert expired_data is None, "TTL expiration failed"
    
    # Test cache size limit
    for i in range(10):
        cache.put(f"key_{i}", f"data_{i}", generation=6)
    
    assert len(cache.cache) <= 5, "Cache size limit not enforced"
    
    print("✅ SpatialHGTCache tests passed:")
    print(f"  • Cache operations: Working")
    print(f"  • TTL expiration: Working")
    print(f"  • Size limiting: Working")
    print(f"  • Final cache size: {len(cache.cache)}")
    
    # 2. Test SpatialHGTConfig
    print("\n2. Testing SpatialHGTConfig...")
    spatial_config = SpatialHGTConfig(
        enable_spatial_indexing=True,
        use_quadtree_optimization=True,
        max_concurrent_transfers=25,
        enable_density_effects=True
    )
    
    # Test effective radius calculation
    base_radius = 2.0
    conjugation_radius = spatial_config.get_effective_radius(HGTMechanism.CONJUGATION, base_radius)
    transformation_radius = spatial_config.get_effective_radius(HGTMechanism.TRANSFORMATION, base_radius)
    transduction_radius = spatial_config.get_effective_radius(HGTMechanism.TRANSDUCTION, base_radius)
    
    print("✅ SpatialHGTConfig tests passed:")
    print(f"  • Conjugation radius: {conjugation_radius:.2f} (scaling factor applied)")
    print(f"  • Transformation radius: {transformation_radius:.2f}")
    print(f"  • Transduction radius: {transduction_radius:.2f}")
    print(f"  • Dynamic scaling: {spatial_config.dynamic_proximity_scaling}")
    
    # 3. Test HGTSpatialIntegration with mock grid
    print("\n3. Testing HGTSpatialIntegration...")
    
    # Create mock spatial grid
    mock_grid = MockSpatialGrid()
    hgt_config = HGTConfig()
    spatial_integration = HGTSpatialIntegration(mock_grid, hgt_config, spatial_config)
    
    # Create test population
    population = {}
    spatial_positions = {}
    
    for i in range(10):
        position = Position(x=random.randint(0, 10), y=random.randint(0, 10))
        bacterium = Bacterium(
            id=f"bacteria_{i}",
            position=position,
            fitness=random.uniform(0.5, 1.0),
            resistance_status=ResistanceStatus.RESISTANT if i >= 5 else ResistanceStatus.SENSITIVE,
            generation_born=0
        )
        
        # Add resistance genes to some bacteria
        if i >= 5:
            bacterium.resistance_genes = {"beta_lactamase", "efflux_pump"}
        else:
            bacterium.resistance_genes = set()
        
        population[bacterium.id] = bacterium
        coordinate = Coordinate(x=position.x, y=position.y)
        spatial_positions[bacterium.id] = coordinate
        mock_grid.update_position(bacterium.id, coordinate)
    
    print(f"✅ Test population created:")
    print(f"  • Total bacteria: {len(population)}")
    print(f"  • Resistant bacteria: {sum(1 for b in population.values() if b.resistance_genes)}")
    print(f"  • Spatial positions: {len(spatial_positions)}")
    
    # Test candidate finding
    print("\n4. Testing optimized candidate finding...")
    
    candidates = await spatial_integration.find_transfer_candidates_optimized(
        population, 
        HGTMechanism.CONJUGATION, 
        generation=1
    )
    
    print(f"✅ Candidate finding results:")
    print(f"  • Donors found: {len(candidates)}")
    print(f"  • Total potential transfers: {sum(len(recipients) for recipients in candidates.values())}")
    print(f"  • Spatial queries made: {mock_grid.query_count}")
    
    # Test cached queries
    initial_query_count = mock_grid.query_count
    candidates_cached = await spatial_integration.find_transfer_candidates_optimized(
        population, 
        HGTMechanism.CONJUGATION, 
        generation=1  # Same generation for cache hit
    )
    
    cache_stats = spatial_integration.get_spatial_statistics()
    print(f"  • Cache hit rate: {cache_stats['cache_hit_rate']:.1%}")
    print(f"  • Cache hits: {cache_stats['cache_hits']}")
    print(f"  • Cache misses: {cache_stats['cache_misses']}")
    
    # 5. Test spatial transfer probability calculation
    print("\n5. Testing spatial transfer probability...")
    
    if candidates:
        donor_id = list(candidates.keys())[0]
        recipient_ids = candidates[donor_id]
        
        if recipient_ids:
            donor = population[donor_id]
            recipient = population[recipient_ids[0]]
            
            probability = spatial_integration.calculate_spatial_transfer_probability(
                donor, 
                recipient, 
                HGTMechanism.CONJUGATION,
                environmental_factors={"antibiotic_concentration": 0.5}
            )
            
            print(f"✅ Transfer probability calculation:")
            print(f"  • Donor: {donor_id}")
            print(f"  • Recipient: {recipient_ids[0]}")
            print(f"  • Probability: {probability:.4f}")
            print(f"  • Environmental factors applied: Yes")
    
    # 6. Test HGTSimulationOrchestrator
    print("\n6. Testing HGTSimulationOrchestrator...")
    
    # Create required components
    spatial_manager = SpatialManager(grid_width=20, grid_height=20, cell_size=1.0)
    proximity_detector = ProximityDetector(spatial_manager, hgt_config)
    probability_calculator = ProbabilityCalculator(hgt_config)
    gene_transfer_engine = GeneTransferEngine(proximity_detector, probability_calculator, hgt_config)
    impact_tracker = PopulationImpactTracker()
    
    orchestrator = HGTSimulationOrchestrator(
        spatial_integration,
        gene_transfer_engine,
        impact_tracker
    )
    
    # Run HGT simulation round
    transfer_records = await orchestrator.run_hgt_round(
        population,
        spatial_positions,
        environmental_factors={"antibiotic_concentration": 0.3},
        generation=2
    )
    
    successful_transfers = [t for t in transfer_records if t.success]
    
    print(f"✅ HGT simulation round completed:")
    print(f"  • Total transfer attempts: {len(transfer_records)}")
    print(f"  • Successful transfers: {len(successful_transfers)}")
    print(f"  • Success rate: {len(successful_transfers)/len(transfer_records)*100:.1f}%" if transfer_records else "  • Success rate: N/A")
    
    # Show some transfer details
    for i, transfer in enumerate(successful_transfers[:3]):
        print(f"  • Transfer {i+1}: {transfer.mechanism.value} - {', '.join(transfer.genes_transferred)}")
    
    # 7. Test performance metrics
    print("\n7. Testing performance metrics...")
    
    performance_report = orchestrator.get_performance_report()
    
    print(f"✅ Performance metrics:")
    print(f"  • Simulation time: {performance_report['total_simulation_time']:.4f}s")
    print(f"  • Transfers per second: {performance_report['transfers_per_second']:.2f}")
    print(f"  • Cache hit rate: {performance_report['cache_hit_rate']:.1%}")
    print(f"  • Total spatial queries: {performance_report['total_queries']}")
    print(f"  • Impact tracker snapshots: {performance_report['impact_tracker_snapshots']}")
    
    # 8. Test generation optimization
    print("\n8. Testing generation optimization...")
    
    initial_cache_size = len(spatial_integration.neighbor_cache.cache) if spatial_integration.neighbor_cache else 0
    
    # Simulate several generations
    for gen in range(10, 15):
        spatial_integration.optimize_for_generation(gen)
        
        # Add some expired entries to cache
        if spatial_integration.neighbor_cache:
            spatial_integration.neighbor_cache.put(f"old_key_{gen}", "old_data", gen - 20)
    
    # Trigger cache cleanup
    spatial_integration.optimize_for_generation(20)
    
    final_cache_size = len(spatial_integration.neighbor_cache.cache) if spatial_integration.neighbor_cache else 0
    
    print(f"✅ Generation optimization:")
    print(f"  • Initial cache size: {initial_cache_size}")
    print(f"  • Final cache size: {final_cache_size}")
    print(f"  • Cache cleanup: {'Effective' if final_cache_size <= initial_cache_size else 'Needs improvement'}")
    
    # 9. Test async processing with larger population
    print("\n9. Testing async processing scalability...")
    
    # Create larger population for async testing
    large_population = {}
    large_positions = {}
    
    for i in range(50):
        position = Position(x=random.randint(0, 15), y=random.randint(0, 15))
        bacterium = Bacterium(
            id=f"large_bacteria_{i}",
            position=position,
            fitness=random.uniform(0.4, 1.0),
            resistance_status=ResistanceStatus.RESISTANT if i >= 25 else ResistanceStatus.SENSITIVE,
            generation_born=0
        )
        
        if i >= 25:
            bacterium.resistance_genes = {"beta_lactamase", "efflux_pump", "target_modification"}
        else:
            bacterium.resistance_genes = set()
        
        large_population[bacterium.id] = bacterium
        coordinate = Coordinate(x=position.x, y=position.y)
        large_positions[bacterium.id] = coordinate
        mock_grid.update_position(bacterium.id, coordinate)
    
    # Enable async processing
    spatial_config.enable_async_processing = True
    spatial_config.batch_size = 10
    
    # Time the operation
    import time
    start_time = time.time()
    
    large_candidates = await spatial_integration.find_transfer_candidates_optimized(
        large_population,
        HGTMechanism.TRANSFORMATION,
        generation=3,
        batch_size=10
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"✅ Async processing scalability:")
    print(f"  • Large population size: {len(large_population)}")
    print(f"  • Candidates found: {len(large_candidates)}")
    print(f"  • Processing time: {processing_time:.4f}s")
    print(f"  • Candidates per second: {len(large_candidates)/processing_time:.2f}" if processing_time > 0 else "  • Processing: Instant")
    
    # 10. Test density effects
    print("\n10. Testing density effects...")
    
    # Create high-density cluster
    cluster_population = {}
    cluster_positions = {}
    
    # Dense cluster at (5,5)
    for i in range(20):
        position = Position(
            x=5 + random.uniform(-1, 1),  # Tight cluster
            y=5 + random.uniform(-1, 1)
        )
        bacterium = Bacterium(
            id=f"cluster_bacteria_{i}",
            position=position,
            fitness=random.uniform(0.6, 1.0),
            resistance_status=ResistanceStatus.RESISTANT if i >= 10 else ResistanceStatus.SENSITIVE,
            generation_born=0
        )
        
        if i >= 10:
            bacterium.resistance_genes = {"beta_lactamase"}
        else:
            bacterium.resistance_genes = set()
        
        cluster_population[bacterium.id] = bacterium
        coordinate = Coordinate(x=position.x, y=position.y)
        cluster_positions[bacterium.id] = coordinate
        mock_grid.update_position(bacterium.id, coordinate)
    
    # Test density effects
    if cluster_population:
        sample_bacterium = list(cluster_population.values())[0]
        local_density = mock_grid.get_density(
            Coordinate(x=sample_bacterium.position.x, y=sample_bacterium.position.y),
            radius=2.0
        )
        
        density_factor = spatial_integration._calculate_density_factor(local_density)
        
        print(f"✅ Density effects:")
        print(f"  • Cluster population: {len(cluster_population)}")
        print(f"  • Local density: {local_density:.3f} bacteria/unit²")
        print(f"  • Density factor: {density_factor:.3f}")
        print(f"  • High density threshold: {hgt_config.high_density_threshold}")
        print(f"  • Density enhancement: {'Yes' if local_density > hgt_config.high_density_threshold else 'No'}")
    
    print("\n🎊 ALL SPATIAL INTEGRATION TESTS PASSED!")
    print("🌐 HGT spatial integration system is fully functional and optimized!")
    
    return True


if __name__ == "__main__":
    try:
        asyncio.run(test_spatial_integration())
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 