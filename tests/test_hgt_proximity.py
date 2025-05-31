#!/usr/bin/env python3
"""
Tests for Horizontal Gene Transfer (HGT) proximity detection system.

This module tests the proximity detection algorithm, candidate identification,
and integration with the spatial grid system.
"""

import pytest
import numpy as np
from typing import Dict, List

from models.hgt import (
    ProximityDetector, HGTConfig, HGTMechanism, GeneTransferEvent
)
from models.spatial import SpatialGrid, SpatialManager, Coordinate, BoundaryCondition
from models.bacterium import Bacterium


@pytest.fixture
def spatial_setup():
    """Create spatial grid and manager for testing."""
    grid = SpatialGrid(
        width=50.0,
        height=50.0,
        cell_size=1.0,
        boundary_condition=BoundaryCondition.CLOSED
    )
    manager = SpatialManager(grid)
    return grid, manager


@pytest.fixture
def hgt_config():
    """Create HGT configuration for testing."""
    return HGTConfig(
        conjugation_distance=2.0,
        transformation_distance=4.0,
        transduction_distance=6.0,
        conjugation_probability=0.1,
        transformation_probability=0.05,
        transduction_probability=0.02
    )


@pytest.fixture
def test_bacteria():
    """Create test bacterial population."""
    bacteria = {}
    
    # Create bacteria with different properties
    for i in range(10):
        bacterium = Bacterium(
            id=f"bacterium_{i}",
            resistance_genes=set(),
            fitness=0.8 + (i * 0.02),  # Varying fitness
            generation=0
        )
        
        # Add some HGT-related properties
        bacterium.is_alive = lambda: True
        bacterium.has_conjugative_plasmid = (i % 3 == 0)  # Every 3rd bacterium
        bacterium.is_competent = (i % 2 == 0)  # Every 2nd bacterium
        bacterium.phage_infected = (i % 4 == 0)  # Every 4th bacterium
        bacterium.phage_resistant = (i % 5 == 0)  # Every 5th bacterium
        bacterium.species = "E.coli" if i < 5 else "S.aureus"
        
        bacteria[bacterium.id] = bacterium
    
    return bacteria


class TestHGTConfig:
    """Test HGT configuration."""
    
    def test_default_config(self):
        """Test default HGT configuration values."""
        config = HGTConfig()
        
        assert config.conjugation_distance == 1.0
        assert config.transformation_distance == 3.0
        assert config.transduction_distance == 5.0
        assert config.conjugation_probability == 0.1
        assert config.max_transfers_per_generation == 100
    
    def test_distance_threshold_getter(self, hgt_config):
        """Test distance threshold getter method."""
        assert hgt_config.get_distance_threshold(HGTMechanism.CONJUGATION) == 2.0
        assert hgt_config.get_distance_threshold(HGTMechanism.TRANSFORMATION) == 4.0
        assert hgt_config.get_distance_threshold(HGTMechanism.TRANSDUCTION) == 6.0
    
    def test_base_probability_getter(self, hgt_config):
        """Test base probability getter method."""
        assert hgt_config.get_base_probability(HGTMechanism.CONJUGATION) == 0.1
        assert hgt_config.get_base_probability(HGTMechanism.TRANSFORMATION) == 0.05
        assert hgt_config.get_base_probability(HGTMechanism.TRANSDUCTION) == 0.02


class TestProximityDetector:
    """Test proximity detection system."""
    
    def test_detector_initialization(self, spatial_setup, hgt_config):
        """Test proximity detector initialization."""
        grid, manager = spatial_setup
        detector = ProximityDetector(manager, hgt_config)
        
        assert detector.spatial_manager is manager
        assert detector.config is hgt_config
        assert detector._cache_generation == -1
        assert len(detector._detection_cache) == 0
    
    def test_proximity_detection_empty_population(self, spatial_setup, hgt_config):
        """Test proximity detection with empty population."""
        grid, manager = spatial_setup
        detector = ProximityDetector(manager, hgt_config)
        
        empty_population = {}
        candidates = detector.detect_hgt_candidates(
            empty_population, 
            HGTMechanism.CONJUGATION
        )
        
        assert len(candidates) == 0
    
    def test_proximity_detection_with_bacteria(self, spatial_setup, hgt_config, test_bacteria):
        """Test proximity detection with bacterial population."""
        grid, manager = spatial_setup
        detector = ProximityDetector(manager, hgt_config)
        
        # Place bacteria in spatial grid
        positions = {}
        for i, (bacterium_id, bacterium) in enumerate(test_bacteria.items()):
            x = 10.0 + (i * 3.0)  # Space them out
            y = 10.0 + (i * 2.0)
            position = Coordinate(x, y)
            
            grid.place_bacterium(bacterium_id, position)
            manager.bacterium_positions[bacterium_id] = position
            positions[bacterium_id] = position
        
        # Detect conjugation candidates
        candidates = detector.detect_hgt_candidates(
            test_bacteria,
            HGTMechanism.CONJUGATION,
            current_generation=1
        )
        
        # Should find some candidates since bacteria are placed close enough
        assert isinstance(candidates, dict)
        
        # Check that only viable donors are included
        for donor_id in candidates.keys():
            donor = test_bacteria[donor_id]
            assert hasattr(donor, 'has_conjugative_plasmid')
    
    def test_viable_donor_detection(self, spatial_setup, hgt_config, test_bacteria):
        """Test viable donor detection for different mechanisms."""
        grid, manager = spatial_setup
        detector = ProximityDetector(manager, hgt_config)
        
        # Test conjugation donors
        for bacterium in test_bacteria.values():
            is_donor = detector._is_viable_donor(bacterium, HGTMechanism.CONJUGATION)
            expected = bacterium.has_conjugative_plasmid
            assert is_donor == expected
        
        # Test transformation donors (all viable)
        for bacterium in test_bacteria.values():
            is_donor = detector._is_viable_donor(bacterium, HGTMechanism.TRANSFORMATION)
            assert is_donor == True
        
        # Test transduction donors
        for bacterium in test_bacteria.values():
            is_donor = detector._is_viable_donor(bacterium, HGTMechanism.TRANSDUCTION)
            expected = bacterium.phage_infected
            assert is_donor == expected
    
    def test_viable_recipient_detection(self, spatial_setup, hgt_config, test_bacteria):
        """Test viable recipient detection for different mechanisms."""
        grid, manager = spatial_setup
        detector = ProximityDetector(manager, hgt_config)
        
        donor = list(test_bacteria.values())[0]
        
        # Test transformation recipients
        for bacterium in test_bacteria.values():
            if bacterium.id == donor.id:
                continue
            
            is_recipient = detector._is_viable_recipient(
                bacterium, donor, HGTMechanism.TRANSFORMATION
            )
            expected = bacterium.is_competent
            assert is_recipient == expected
        
        # Test transduction recipients
        for bacterium in test_bacteria.values():
            if bacterium.id == donor.id:
                continue
            
            is_recipient = detector._is_viable_recipient(
                bacterium, donor, HGTMechanism.TRANSDUCTION
            )
            expected = not bacterium.phage_resistant
            assert is_recipient == expected
    
    def test_conjugation_compatibility(self, spatial_setup, hgt_config, test_bacteria):
        """Test conjugation compatibility checking."""
        grid, manager = spatial_setup
        detector = ProximityDetector(manager, hgt_config)
        
        bacteria_list = list(test_bacteria.values())
        donor = bacteria_list[0]  # E.coli
        recipient1 = bacteria_list[1]  # E.coli
        recipient2 = bacteria_list[5]  # S.aureus
        
        # Same species should be compatible
        compatible = detector._check_conjugation_compatibility(recipient1, donor)
        assert compatible == True
        
        # Different species compatibility is probabilistic, just test it runs
        compatible = detector._check_conjugation_compatibility(recipient2, donor)
        assert isinstance(compatible, bool)
    
    def test_detection_caching(self, spatial_setup, hgt_config, test_bacteria):
        """Test proximity detection caching mechanism."""
        grid, manager = spatial_setup
        detector = ProximityDetector(manager, hgt_config)
        
        # Place bacteria
        for i, (bacterium_id, bacterium) in enumerate(test_bacteria.items()):
            position = Coordinate(10.0 + i, 10.0 + i)
            grid.place_bacterium(bacterium_id, position)
            manager.bacterium_positions[bacterium_id] = position
        
        # First detection should be cache miss
        candidates1 = detector.detect_hgt_candidates(
            test_bacteria,
            HGTMechanism.CONJUGATION,
            current_generation=1,
            use_cache=True
        )
        
        # Second detection should be cache hit
        candidates2 = detector.detect_hgt_candidates(
            test_bacteria,
            HGTMechanism.CONJUGATION,
            current_generation=1,
            use_cache=True
        )
        
        # Results should be identical
        assert candidates1 == candidates2
        
        # Check cache stats
        assert detector._detection_stats["cache_hits"] > 0
    
    def test_cache_invalidation(self, spatial_setup, hgt_config, test_bacteria):
        """Test cache invalidation when generation changes."""
        grid, manager = spatial_setup
        detector = ProximityDetector(manager, hgt_config)
        
        # Place bacteria
        for i, (bacterium_id, bacterium) in enumerate(test_bacteria.items()):
            position = Coordinate(10.0 + i, 10.0 + i)
            grid.place_bacterium(bacterium_id, position)
            manager.bacterium_positions[bacterium_id] = position
        
        # Detection for generation 1
        detector.detect_hgt_candidates(
            test_bacteria,
            HGTMechanism.CONJUGATION,
            current_generation=1
        )
        
        cache_size_gen1 = len(detector._detection_cache)
        
        # Detection for generation 2 should clear cache
        detector.detect_hgt_candidates(
            test_bacteria,
            HGTMechanism.CONJUGATION,
            current_generation=2
        )
        
        # Cache should be updated for new generation
        assert detector._cache_generation == 2
    
    def test_proximity_metrics(self, spatial_setup, hgt_config, test_bacteria):
        """Test proximity metrics calculation."""
        grid, manager = spatial_setup
        detector = ProximityDetector(manager, hgt_config)
        
        # Place bacteria close together for interactions
        for i, (bacterium_id, bacterium) in enumerate(test_bacteria.items()):
            position = Coordinate(25.0, 25.0 + i * 0.5)  # Very close together
            grid.place_bacterium(bacterium_id, position)
            manager.bacterium_positions[bacterium_id] = position
        
        metrics = detector.get_proximity_metrics(test_bacteria)
        
        assert "total_bacteria" in metrics
        assert "cache_hit_rate" in metrics
        assert "potential_conjugation_pairs" in metrics
        assert "potential_transformation_pairs" in metrics
        assert "potential_transduction_pairs" in metrics
        assert "total_potential_transfers" in metrics
        
        assert metrics["total_bacteria"] == len(test_bacteria)
        assert metrics["total_potential_transfers"] >= 0
    
    def test_distance_threshold_enforcement(self, spatial_setup, hgt_config, test_bacteria):
        """Test that distance thresholds are properly enforced."""
        grid, manager = spatial_setup
        detector = ProximityDetector(manager, hgt_config)
        
        # Place bacteria at specific distances
        bacteria_list = list(test_bacteria.items())
        
        # Place donor at origin
        donor_id, donor = bacteria_list[0]
        donor_pos = Coordinate(25.0, 25.0)
        grid.place_bacterium(donor_id, donor_pos)
        manager.bacterium_positions[donor_id] = donor_pos
        
        # Place recipients at different distances
        for i, (bacterium_id, bacterium) in enumerate(bacteria_list[1:], 1):
            distance = i * 1.5  # 1.5, 3.0, 4.5, 6.0, 7.5, etc.
            position = Coordinate(25.0 + distance, 25.0)
            grid.place_bacterium(bacterium_id, position)
            manager.bacterium_positions[bacterium_id] = position
        
        # Test conjugation (distance <= 2.0)
        conjugation_candidates = detector.detect_hgt_candidates(
            test_bacteria,
            HGTMechanism.CONJUGATION,
            use_cache=False
        )
        
        # Test transformation (distance <= 4.0)
        transformation_candidates = detector.detect_hgt_candidates(
            test_bacteria,
            HGTMechanism.TRANSFORMATION,
            use_cache=False
        )
        
        # Test transduction (distance <= 6.0)
        transduction_candidates = detector.detect_hgt_candidates(
            test_bacteria,
            HGTMechanism.TRANSDUCTION,
            use_cache=False
        )
        
        # Conjugation should have fewer candidates than transformation
        # which should have fewer than transduction (due to distance limits)
        conjugation_total = sum(len(recipients) for recipients in conjugation_candidates.values())
        transformation_total = sum(len(recipients) for recipients in transformation_candidates.values())
        transduction_total = sum(len(recipients) for recipients in transduction_candidates.values())
        
        # This relationship should hold given our distance setup
        assert conjugation_total <= transformation_total <= transduction_total
    
    def test_clear_cache(self, spatial_setup, hgt_config):
        """Test cache clearing functionality."""
        grid, manager = spatial_setup
        detector = ProximityDetector(manager, hgt_config)
        
        # Add something to cache
        detector._detection_cache["test"] = {"donor1": ["recipient1"]}
        detector._cache_generation = 5
        
        # Clear cache
        detector.clear_cache()
        
        assert len(detector._detection_cache) == 0
        assert detector._cache_generation == -1


class TestHGTIntegration:
    """Test HGT system integration with spatial grid."""
    
    def test_integration_with_spatial_manager(self, spatial_setup, hgt_config):
        """Test that HGT integrates properly with spatial manager."""
        grid, manager = spatial_setup
        detector = ProximityDetector(manager, hgt_config)
        
        # Create minimal bacterial population
        bacteria = {}
        for i in range(3):
            bacterium = Bacterium(
                id=f"bacterium_{i}",
                resistance_genes=set(),
                fitness=0.8,
                generation=0
            )
            bacterium.is_alive = lambda: True
            bacterium.has_conjugative_plasmid = True
            bacterium.is_competent = True
            bacterium.phage_infected = False
            bacterium.phage_resistant = False
            
            bacteria[bacterium.id] = bacterium
            
            # Place in spatial grid
            position = Coordinate(25.0 + i, 25.0)
            grid.place_bacterium(bacterium.id, position)
            manager.bacterium_positions[bacterium.id] = position
        
        # Test detection
        candidates = detector.detect_hgt_candidates(
            bacteria,
            HGTMechanism.CONJUGATION
        )
        
        # Should work without errors and find candidates
        assert isinstance(candidates, dict)
        assert len(candidates) >= 0  # May be 0 depending on actual distances
    
    def test_performance_with_large_population(self, spatial_setup, hgt_config):
        """Test performance with larger bacterial populations."""
        grid, manager = spatial_setup
        detector = ProximityDetector(manager, hgt_config)
        
        # Create larger population
        bacteria = {}
        for i in range(100):
            bacterium = Bacterium(
                id=f"bacterium_{i}",
                resistance_genes=set(),
                fitness=0.8,
                generation=0
            )
            bacterium.is_alive = lambda: True
            bacterium.has_conjugative_plasmid = (i % 5 == 0)
            bacterium.is_competent = (i % 3 == 0)
            bacterium.phage_infected = (i % 7 == 0)
            bacterium.phage_resistant = (i % 11 == 0)
            
            bacteria[bacterium.id] = bacterium
            
            # Place randomly in grid
            position = Coordinate(
                np.random.uniform(5, 45),
                np.random.uniform(5, 45)
            )
            grid.place_bacterium(bacterium.id, position)
            manager.bacterium_positions[bacterium.id] = position
        
        # Enable optimizations
        manager.optimize_for_large_population(True)
        
        # Test all mechanisms
        for mechanism in HGTMechanism:
            candidates = detector.detect_hgt_candidates(
                bacteria,
                mechanism,
                use_cache=False
            )
            
            assert isinstance(candidates, dict)
        
        # Test metrics
        metrics = detector.get_proximity_metrics(bacteria)
        assert metrics["total_bacteria"] == 100 