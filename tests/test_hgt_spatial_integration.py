import pytest
import asyncio
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Optional
from dataclasses import dataclass

# Import the necessary modules and classes
from models.bacterium import Bacterium, ResistanceStatus
from models.hgt import (
    HGTMechanism, 
    HGTConfig, 
    SpatialHGTConfig,
    SpatialHGTCache,
    HGTSpatialIntegration,
    HGTSimulationOrchestrator,
    Coordinate
)

# Mock spatial grid interface
class MockSpatialGrid:
    """Mock spatial grid for testing."""
    
    def __init__(self):
        self.bacteria_positions = {}
        self.densities = {}
    
    def get_neighbors(self, position: Coordinate, radius: float) -> List[str]:
        """Return mock neighbors based on position and radius."""
        neighbors = []
        for bacterium_id, pos in self.bacteria_positions.items():
            if pos:
                distance = ((position.x - pos.x) ** 2 + (position.y - pos.y) ** 2) ** 0.5
                if 0 < distance <= radius:  # Exclude self
                    neighbors.append(bacterium_id)
        return neighbors
    
    def get_density(self, position: Coordinate, radius: float) -> float:
        """Return mock density at position."""
        return self.densities.get(f"{position.x}_{position.y}", 1.0)
    
    def add_bacterium(self, bacterium_id: str, position: Coordinate):
        """Add bacterium to mock grid."""
        self.bacteria_positions[bacterium_id] = position
    
    def set_density(self, position: Coordinate, density: float):
        """Set density at position."""
        self.densities[f"{position.x}_{position.y}"] = density


def create_test_bacterium(
    bacterium_id: str, 
    x: float, 
    y: float, 
    fitness: float = 0.8,
    resistance_genes: Optional[List[str]] = None
) -> Bacterium:
    """Create a test bacterium with position."""
    bacterium = Bacterium(
        id=bacterium_id,
        species="test_species",
        resistance_status=ResistanceStatus.SENSITIVE
    )
    bacterium.fitness = fitness
    bacterium.position = Coordinate(x=x, y=y)
    bacterium.resistance_genes = resistance_genes or ["beta_lactamase"]
    return bacterium


class TestSpatialHGTCache:
    """Test spatial HGT cache functionality."""
    
    def test_cache_creation(self):
        """Test cache initialization."""
        cache = SpatialHGTCache(max_size=100, ttl=5)
        assert cache.max_size == 100
        assert cache.ttl == 5
        assert len(cache.cache) == 0
    
    def test_cache_put_and_get(self):
        """Test basic cache operations."""
        cache = SpatialHGTCache(max_size=10, ttl=5)
        
        # Put data
        cache.put("test_key", ["recipient1", "recipient2"], current_generation=1)
        
        # Get data
        result = cache.get("test_key", current_generation=1)
        assert result == ["recipient1", "recipient2"]
        
        # Non-existent key
        assert cache.get("nonexistent", current_generation=1) is None
    
    def test_cache_ttl_expiry(self):
        """Test cache TTL expiry."""
        cache = SpatialHGTCache(max_size=10, ttl=2)
        
        # Put data in generation 1
        cache.put("test_key", ["recipient1"], current_generation=1)
        
        # Should be available in generation 2
        assert cache.get("test_key", current_generation=2) == ["recipient1"]
        
        # Should be available in generation 3 (within TTL)
        assert cache.get("test_key", current_generation=3) == ["recipient1"]
        
        # Should be expired in generation 4 (beyond TTL)
        assert cache.get("test_key", current_generation=4) is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = SpatialHGTCache(max_size=2, ttl=10)
        
        # Fill cache
        cache.put("key1", ["data1"], current_generation=1)
        cache.put("key2", ["data2"], current_generation=1)
        
        # Access key1 at a later time to make it more recent
        cache.get("key1", current_generation=2)
        
        # Add key3, should evict key2 (least recently used)
        cache.put("key3", ["data3"], current_generation=3)
        
        assert cache.get("key1", current_generation=3) == ["data1"]
        assert cache.get("key2", current_generation=3) is None  # Evicted
        assert cache.get("key3", current_generation=3) == ["data3"]


class TestHGTSpatialIntegration:
    """Test HGT spatial integration functionality."""
    
    @pytest.fixture
    def spatial_grid(self):
        """Create mock spatial grid."""
        return MockSpatialGrid()
    
    @pytest.fixture
    def hgt_config(self):
        """Create HGT configuration."""
        return HGTConfig()
    
    @pytest.fixture
    def spatial_config(self):
        """Create spatial HGT configuration."""
        return SpatialHGTConfig()
    
    @pytest.fixture
    def spatial_integration(self, spatial_grid, hgt_config, spatial_config):
        """Create spatial integration instance."""
        return HGTSpatialIntegration(spatial_grid, hgt_config, spatial_config)
    
    def test_spatial_integration_creation(self, spatial_integration):
        """Test spatial integration initialization."""
        assert spatial_integration.spatial_grid is not None
        assert spatial_integration.hgt_config is not None
        assert spatial_integration.spatial_config is not None
        assert spatial_integration.neighbor_cache is not None
        assert spatial_integration.query_stats["spatial_queries"] == 0
    
    @pytest.mark.asyncio
    async def test_find_transfer_candidates(self, spatial_integration, spatial_grid):
        """Test finding transfer candidates."""
        # Create test population
        donor = create_test_bacterium("donor1", 5.0, 5.0, fitness=0.8)
        recipient1 = create_test_bacterium("recip1", 6.0, 5.0, fitness=0.7)  # Close
        recipient2 = create_test_bacterium("recip2", 10.0, 10.0, fitness=0.6)  # Far
        
        population = {
            "donor1": donor,
            "recip1": recipient1,
            "recip2": recipient2
        }
        
        # Add to spatial grid
        spatial_grid.add_bacterium("donor1", Coordinate(5.0, 5.0))
        spatial_grid.add_bacterium("recip1", Coordinate(6.0, 5.0))
        spatial_grid.add_bacterium("recip2", Coordinate(10.0, 10.0))
        
        # Find candidates for conjugation (short range)
        candidates = await spatial_integration.find_transfer_candidates_optimized(
            population, HGTMechanism.CONJUGATION, current_generation=1
        )
        
        # Should find donor1 with recip1 as candidate
        assert "donor1" in candidates
        assert "recip1" in candidates["donor1"]
        assert "recip2" not in candidates.get("donor1", [])
    
    def test_spatial_transfer_probability(self, spatial_integration, spatial_grid):
        """Test spatial transfer probability calculation."""
        # Create close bacteria
        donor = create_test_bacterium("donor1", 5.0, 5.0)
        recipient = create_test_bacterium("recip1", 6.0, 5.0)
        
        # Set density
        spatial_grid.set_density(Coordinate(5.0, 5.0), 2.0)
        
        # Calculate probability
        prob = spatial_integration.calculate_spatial_transfer_probability(
            donor, recipient, HGTMechanism.CONJUGATION
        )
        
        assert 0.0 <= prob <= 1.0
        assert prob > 0  # Should have some probability for close bacteria
    
    def test_spatial_transfer_probability_distance_decay(self, spatial_integration):
        """Test distance decay in transfer probability."""
        # Create bacteria at different distances
        donor = create_test_bacterium("donor1", 0.0, 0.0)
        close_recipient = create_test_bacterium("close", 1.0, 0.0)
        far_recipient = create_test_bacterium("far", 5.0, 0.0)
        
        # Calculate probabilities
        close_prob = spatial_integration.calculate_spatial_transfer_probability(
            donor, close_recipient, HGTMechanism.CONJUGATION
        )
        far_prob = spatial_integration.calculate_spatial_transfer_probability(
            donor, far_recipient, HGTMechanism.CONJUGATION
        )
        
        # Close bacteria should have higher probability
        assert close_prob > far_prob
    
    def test_cache_integration(self, spatial_integration, spatial_grid):
        """Test cache integration in spatial queries."""
        # Create test population
        donor = create_test_bacterium("donor1", 5.0, 5.0)
        recipient = create_test_bacterium("recip1", 6.0, 5.0)
        
        population = {"donor1": donor, "recip1": recipient}
        spatial_grid.add_bacterium("donor1", Coordinate(5.0, 5.0))
        spatial_grid.add_bacterium("recip1", Coordinate(6.0, 5.0))
        
        # First query - should miss cache
        asyncio.run(spatial_integration.find_transfer_candidates_optimized(
            population, HGTMechanism.CONJUGATION, current_generation=1
        ))
        
        stats1 = spatial_integration.get_spatial_statistics()
        initial_misses = stats1["cache_misses"]
        
        # Second query - should hit cache
        asyncio.run(spatial_integration.find_transfer_candidates_optimized(
            population, HGTMechanism.CONJUGATION, current_generation=1
        ))
        
        stats2 = spatial_integration.get_spatial_statistics()
        assert stats2["cache_hits"] > stats1["cache_hits"]
    
    def test_density_effects(self, spatial_integration, spatial_grid):
        """Test density effects on transfer candidates."""
        # Create test population
        donor = create_test_bacterium("donor1", 5.0, 5.0)
        recipients = [
            create_test_bacterium(f"recip{i}", 5.0 + i*0.5, 5.0) 
            for i in range(1, 6)
        ]
        
        population = {"donor1": donor}
        population.update({f"recip{i}": recipients[i-1] for i in range(1, 6)})
        
        # Add to spatial grid
        spatial_grid.add_bacterium("donor1", Coordinate(5.0, 5.0))
        for i, recipient in enumerate(recipients, 1):
            spatial_grid.add_bacterium(f"recip{i}", Coordinate(5.0 + i*0.5, 5.0))
        
        # Set high density
        spatial_grid.set_density(Coordinate(5.0, 5.0), 5.0)
        
        # Enable density effects
        spatial_integration.spatial_config.enable_density_effects = True
        
        candidates = asyncio.run(spatial_integration.find_transfer_candidates_optimized(
            population, HGTMechanism.CONJUGATION, current_generation=1
        ))
        
        # Should find candidates with density effects
        assert "donor1" in candidates
        assert len(candidates["donor1"]) > 0


class TestHGTSimulationOrchestrator:
    """Test HGT simulation orchestrator."""
    
    @pytest.fixture
    def spatial_grid(self):
        """Create mock spatial grid."""
        return MockSpatialGrid()
    
    @pytest.fixture
    def orchestrator(self, spatial_grid):
        """Create orchestrator instance."""
        hgt_config = HGTConfig()
        spatial_config = SpatialHGTConfig()
        return HGTSimulationOrchestrator(spatial_grid, hgt_config, spatial_config)
    
    def test_orchestrator_creation(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.spatial_integration is not None
        assert orchestrator.hgt_config is not None
        assert orchestrator.spatial_config is not None
    
    @pytest.mark.asyncio
    async def test_run_hgt_round(self, orchestrator, spatial_grid):
        """Test running a complete HGT round."""
        # Create test population
        population = {}
        for i in range(10):
            bacterium = create_test_bacterium(f"bact_{i}", i*2.0, i*2.0)
            population[f"bact_{i}"] = bacterium
            spatial_grid.add_bacterium(f"bact_{i}", Coordinate(i*2.0, i*2.0))
        
        # Run HGT round
        results = await orchestrator.run_hgt_round(population, generation=1)
        
        # Verify results structure
        assert "total_transfers" in results
        assert "successful_transfers" in results
        assert "failed_transfers" in results
        assert "transfers_by_mechanism" in results
        assert "spatial_statistics" in results
        
        # Should have attempted some transfers
        assert results["total_transfers"] >= 0
    
    def test_performance_optimization(self, orchestrator):
        """Test performance optimization features."""
        # Test generation optimization
        orchestrator.optimize_for_generation(generation=10)
        
        # Should complete without errors
        stats = orchestrator.get_performance_statistics()
        assert "spatial_queries" in stats
        assert "cache_hit_rate" in stats


class TestSpatialIntegrationIntegration:
    """Integration tests for spatial HGT system."""
    
    @pytest.mark.asyncio
    async def test_full_spatial_hgt_workflow(self):
        """Test complete spatial HGT workflow."""
        # Create spatial grid and integration
        spatial_grid = MockSpatialGrid()
        hgt_config = HGTConfig()
        spatial_config = SpatialHGTConfig()
        integration = HGTSpatialIntegration(spatial_grid, hgt_config, spatial_config)
        
        # Create diverse population
        population = {}
        positions = [
            (5.0, 5.0), (6.0, 5.0), (7.0, 5.0),  # Close cluster
            (15.0, 15.0), (16.0, 15.0),           # Distant cluster
            (10.0, 10.0)                          # Isolated
        ]
        
        for i, (x, y) in enumerate(positions):
            bacterium = create_test_bacterium(f"bact_{i}", x, y, fitness=0.7)
            population[f"bact_{i}"] = bacterium
            spatial_grid.add_bacterium(f"bact_{i}", Coordinate(x, y))
        
        # Set varying densities
        spatial_grid.set_density(Coordinate(5.0, 5.0), 3.0)  # High density cluster
        spatial_grid.set_density(Coordinate(15.0, 15.0), 2.0)  # Medium density
        spatial_grid.set_density(Coordinate(10.0, 10.0), 0.5)  # Low density
        
        # Test all HGT mechanisms
        mechanisms = [HGTMechanism.CONJUGATION, HGTMechanism.TRANSFORMATION, HGTMechanism.TRANSDUCTION]
        
        for mechanism in mechanisms:
            candidates = await integration.find_transfer_candidates_optimized(
                population, mechanism, current_generation=1
            )
            
            # Should find some candidates
            assert isinstance(candidates, dict)
            
            # High density areas should have more transfers
            cluster_donors = [key for key in candidates.keys() if key in ["bact_0", "bact_1", "bact_2"]]
            isolated_donors = [key for key in candidates.keys() if key == "bact_5"]
            
            # Verify spatial effects (clusters should be more active)
            if cluster_donors and isolated_donors:
                cluster_recipients = sum(len(candidates[d]) for d in cluster_donors)
                isolated_recipients = sum(len(candidates[d]) for d in isolated_donors)
                assert cluster_recipients >= isolated_recipients
        
        # Test performance monitoring
        stats = integration.get_spatial_statistics()
        assert stats["spatial_queries"] > 0
        assert stats["total_queries"] > 0
    
    def test_environmental_factor_integration(self):
        """Test environmental factor integration in spatial calculations."""
        spatial_grid = MockSpatialGrid()
        hgt_config = HGTConfig()
        spatial_config = SpatialHGTConfig()
        integration = HGTSpatialIntegration(spatial_grid, hgt_config, spatial_config)
        
        donor = create_test_bacterium("donor", 5.0, 5.0)
        recipient = create_test_bacterium("recipient", 6.0, 5.0)
        
        # Test without environmental factors
        base_prob = integration.calculate_spatial_transfer_probability(
            donor, recipient, HGTMechanism.CONJUGATION
        )
        
        # Test with antibiotic stress
        stress_prob = integration.calculate_spatial_transfer_probability(
            donor, recipient, HGTMechanism.CONJUGATION,
            environmental_factors={"antibiotic_concentration": 0.5}
        )
        
        # Stress should increase transfer probability
        assert stress_prob > base_prob 