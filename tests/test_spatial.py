"""
Tests for the spatial grid system.
"""

import pytest
import numpy as np
import math
from models.spatial import (
    Coordinate, GridCell, SpatialGrid, SpatialManager, BoundaryCondition
)


class TestCoordinate:
    """Test coordinate functionality."""
    
    def test_coordinate_creation(self):
        """Test basic coordinate creation."""
        coord = Coordinate(10.5, 20.3)
        assert coord.x == 10.5
        assert coord.y == 20.3
    
    def test_distance_calculation(self):
        """Test distance calculation between coordinates."""
        coord1 = Coordinate(0, 0)
        coord2 = Coordinate(3, 4)
        
        # Euclidean distance
        assert coord1.distance_to(coord2) == 5.0
        assert coord2.distance_to(coord1) == 5.0
        
        # Manhattan distance
        assert coord1.manhattan_distance_to(coord2) == 7.0
        assert coord2.manhattan_distance_to(coord1) == 7.0
    
    def test_within_distance(self):
        """Test within distance checking."""
        coord1 = Coordinate(0, 0)
        coord2 = Coordinate(3, 4)
        
        assert coord1.is_within_distance(coord2, 5.0)
        assert coord1.is_within_distance(coord2, 5.1)
        assert not coord1.is_within_distance(coord2, 4.9)
    
    def test_coordinate_equality(self):
        """Test coordinate equality and hashing."""
        coord1 = Coordinate(1.0, 2.0)
        coord2 = Coordinate(1.0, 2.0)
        coord3 = Coordinate(1.01, 2.0)  # More clearly different
        
        assert coord1 == coord2
        assert coord1 != coord3
        assert hash(coord1) == hash(coord2)


class TestGridCell:
    """Test grid cell functionality."""
    
    def test_cell_creation(self):
        """Test basic cell creation."""
        coord = Coordinate(5, 5)
        cell = GridCell(coordinate=coord)
        
        assert cell.coordinate == coord
        assert cell.get_bacteria_count() == 0
        assert cell.antibiotic_concentration == 0.0
        assert cell.nutrient_concentration == 1.0
    
    def test_bacteria_management(self):
        """Test adding and removing bacteria from cells."""
        coord = Coordinate(5, 5)
        cell = GridCell(coordinate=coord)
        
        # Add bacteria
        cell.add_bacterium("bact_1")
        cell.add_bacterium("bact_2")
        assert cell.get_bacteria_count() == 2
        assert "bact_1" in cell.bacteria_ids
        assert "bact_2" in cell.bacteria_ids
        
        # Remove bacterium
        cell.remove_bacterium("bact_1")
        assert cell.get_bacteria_count() == 1
        assert "bact_1" not in cell.bacteria_ids
        assert "bact_2" in cell.bacteria_ids
        
        # Remove non-existent bacterium (should not error)
        cell.remove_bacterium("nonexistent")
        assert cell.get_bacteria_count() == 1
    
    def test_overcrowding(self):
        """Test overcrowding detection."""
        coord = Coordinate(5, 5)
        cell = GridCell(coordinate=coord)
        
        # Add bacteria up to capacity
        for i in range(10):
            cell.add_bacterium(f"bact_{i}")
        
        assert not cell.is_overcrowded(max_capacity=10)
        
        # Add one more to exceed capacity
        cell.add_bacterium("bact_10")
        assert cell.is_overcrowded(max_capacity=10)


class TestSpatialGrid:
    """Test spatial grid functionality."""
    
    def test_grid_initialization(self):
        """Test basic grid initialization."""
        grid = SpatialGrid(width=50, height=50, cell_size=5)
        
        assert grid.width == 50
        assert grid.height == 50
        assert grid.cell_size == 5
        assert grid.grid_width == 10  # 50 / 5
        assert grid.grid_height == 10
        assert len(grid.cells) == 100  # 10 * 10
    
    def test_coordinate_grid_conversion(self):
        """Test conversion between coordinates and grid indices."""
        grid = SpatialGrid(width=50, height=50, cell_size=5)
        
        # Test coordinate to grid index
        coord = Coordinate(12.3, 7.8)
        grid_index = grid._coordinate_to_grid_index(coord)
        assert grid_index == (2, 1)  # 12.3/5 = 2.46 -> 2, 7.8/5 = 1.56 -> 1
        
        # Test grid index to coordinate
        back_coord = grid._grid_index_to_coordinate(2, 1)
        assert back_coord == Coordinate(12.5, 7.5)  # Cell centers
    
    def test_boundary_conditions(self):
        """Test different boundary conditions."""
        # Closed boundaries
        grid_closed = SpatialGrid(width=10, height=10, cell_size=1, 
                                 boundary_condition=BoundaryCondition.CLOSED)
        
        assert grid_closed.is_valid_position(Coordinate(5, 5))
        assert grid_closed.is_valid_position(Coordinate(0, 0))
        assert grid_closed.is_valid_position(Coordinate(10, 10))
        assert not grid_closed.is_valid_position(Coordinate(-1, 5))
        assert not grid_closed.is_valid_position(Coordinate(5, 11))
        
        # Periodic boundaries
        grid_periodic = SpatialGrid(width=10, height=10, cell_size=1,
                                   boundary_condition=BoundaryCondition.PERIODIC)
        
        assert grid_periodic.is_valid_position(Coordinate(-1, 5))  # All positions valid
        assert grid_periodic.is_valid_position(Coordinate(15, 5))
    
    def test_bacterium_placement(self):
        """Test placing bacteria on the grid."""
        grid = SpatialGrid(width=10, height=10, cell_size=1)
        
        # Place bacterium
        position = Coordinate(3.5, 7.2)
        result = grid.place_bacterium("bact_1", position)
        
        assert result is True
        assert grid.get_bacterium_position("bact_1") == position
        
        # Check if bacterium is in correct cell
        grid_index = grid._coordinate_to_grid_index(position)
        cell = grid.cells[grid_index]
        assert "bact_1" in cell.bacteria_ids
    
    def test_bacterium_movement(self):
        """Test moving bacteria on the grid."""
        grid = SpatialGrid(width=10, height=10, cell_size=1)
        
        # Place bacterium
        initial_pos = Coordinate(3, 4)
        grid.place_bacterium("bact_1", initial_pos)
        
        # Move bacterium
        new_pos = Coordinate(7, 8)
        result = grid.move_bacterium("bact_1", new_pos)
        
        assert result is True
        assert grid.get_bacterium_position("bact_1") == new_pos
        
        # Check old cell is empty
        old_grid_index = grid._coordinate_to_grid_index(initial_pos)
        old_cell = grid.cells[old_grid_index]
        assert "bact_1" not in old_cell.bacteria_ids
        
        # Check new cell contains bacterium
        new_grid_index = grid._coordinate_to_grid_index(new_pos)
        new_cell = grid.cells[new_grid_index]
        assert "bact_1" in new_cell.bacteria_ids
    
    def test_bacterium_removal(self):
        """Test removing bacteria from the grid."""
        grid = SpatialGrid(width=10, height=10, cell_size=1)
        
        # Place bacterium
        position = Coordinate(5, 5)
        grid.place_bacterium("bact_1", position)
        
        # Remove bacterium
        result = grid.remove_bacterium("bact_1")
        
        assert result is True
        assert grid.get_bacterium_position("bact_1") is None
        
        # Check cell is empty
        grid_index = grid._coordinate_to_grid_index(position)
        cell = grid.cells[grid_index]
        assert "bact_1" not in cell.bacteria_ids
        
        # Try to remove non-existent bacterium
        result2 = grid.remove_bacterium("nonexistent")
        assert result2 is False
    
    def test_neighbor_detection(self):
        """Test finding neighboring bacteria."""
        grid = SpatialGrid(width=20, height=20, cell_size=1)
        
        # Place central bacterium
        center_pos = Coordinate(10, 10)
        grid.place_bacterium("center", center_pos)
        
        # Place neighbors at various distances
        positions = [
            ("close_1", Coordinate(11, 10)),     # Distance 1
            ("close_2", Coordinate(10, 11)),     # Distance 1
            ("medium", Coordinate(12, 12)),      # Distance ~2.83
            ("far", Coordinate(15, 15)),         # Distance ~7.07
        ]
        
        for bact_id, pos in positions:
            grid.place_bacterium(bact_id, pos)
        
        # Test different search radii
        neighbors_r1 = grid.get_neighbors("center", radius=1.5)
        assert len(neighbors_r1) == 2
        assert "close_1" in neighbors_r1
        assert "close_2" in neighbors_r1
        
        neighbors_r3 = grid.get_neighbors("center", radius=3.0)
        assert len(neighbors_r3) == 3
        assert "medium" in neighbors_r3
        
        neighbors_r8 = grid.get_neighbors("center", radius=8.0)
        assert len(neighbors_r8) == 4
        assert "far" in neighbors_r8
        
        # Test including self
        neighbors_with_self = grid.get_neighbors("center", radius=1.5, include_self=True)
        assert len(neighbors_with_self) == 3
        assert "center" in neighbors_with_self
    
    def test_antibiotic_zones(self):
        """Test antibiotic zone functionality."""
        grid = SpatialGrid(width=20, height=20, cell_size=1)
        
        # Add antibiotic zone
        center = Coordinate(10, 10)
        radius = 5.0
        concentration = 0.8
        
        grid.add_antibiotic_zone(center, radius, concentration)
        
        # Test concentrations at various positions
        # Center should have maximum concentration
        center_conc = grid.get_antibiotic_concentration(center)
        assert center_conc == concentration
        
        # Mid-range should have moderate concentration
        mid_pos = Coordinate(12, 10)  # Distance = 2 (well within zone)
        mid_conc = grid.get_antibiotic_concentration(mid_pos)
        assert 0 < mid_conc < concentration
        
        # Edge should have very low concentration
        edge_pos = Coordinate(14.5, 10)  # Distance = 4.5 (near edge)
        edge_conc = grid.get_antibiotic_concentration(edge_pos)
        assert 0 < edge_conc < mid_conc
        
        # Outside zone should have no concentration
        outside_pos = Coordinate(16, 10)  # Distance = 6 (outside)
        outside_conc = grid.get_antibiotic_concentration(outside_pos)
        assert outside_conc == 0.0
    
    def test_local_density(self):
        """Test local density calculation."""
        grid = SpatialGrid(width=20, height=20, cell_size=1)
        
        # Place bacteria in a cluster
        center = Coordinate(10, 10)
        positions = [
            Coordinate(10, 10),
            Coordinate(11, 10),
            Coordinate(10, 11),
            Coordinate(9, 10),
            Coordinate(10, 9),
        ]
        
        for i, pos in enumerate(positions):
            grid.place_bacterium(f"bact_{i}", pos)
        
        # Calculate density
        density = grid.get_local_density(center, radius=2.0)
        expected_area = math.pi * 2.0**2
        expected_density = 5 / expected_area
        
        assert abs(density - expected_density) < 0.01
    
    def test_grid_statistics(self):
        """Test grid statistics calculation."""
        grid = SpatialGrid(width=10, height=10, cell_size=1)
        
        # Add some bacteria
        positions = [
            Coordinate(1, 1),
            Coordinate(2, 2),
            Coordinate(3, 3),
        ]
        
        for i, pos in enumerate(positions):
            grid.place_bacterium(f"bact_{i}", pos)
        
        # Add antibiotic zone
        grid.add_antibiotic_zone(Coordinate(5, 5), 2.0, 0.5)
        
        stats = grid.get_grid_statistics()
        
        assert stats['total_bacteria'] == 3
        assert stats['occupied_cells'] == 3
        assert stats['total_cells'] == 100
        assert stats['occupancy_rate'] == 0.03
        assert stats['antibiotic_coverage'] > 0
        assert stats['grid_dimensions'] == (10, 10)
        assert stats['physical_dimensions'] == (10, 10)
    
    def test_grid_clear(self):
        """Test clearing the grid."""
        grid = SpatialGrid(width=10, height=10, cell_size=1)
        
        # Add bacteria and antibiotic zones
        grid.place_bacterium("bact_1", Coordinate(5, 5))
        grid.add_antibiotic_zone(Coordinate(5, 5), 2.0, 0.5)
        
        # Clear grid
        grid.clear()
        
        # Verify everything is cleared
        assert len(grid.bacterial_positions) == 0
        assert len(grid.antibiotic_zones) == 0
        
        for cell in grid.cells.values():
            assert len(cell.bacteria_ids) == 0
            assert cell.antibiotic_concentration == 0.0


class TestSpatialManager:
    """Test spatial manager functionality."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        grid = SpatialGrid(width=10, height=10, cell_size=1)
        manager = SpatialManager(grid)
        
        assert manager.grid == grid
    
    def test_random_population_initialization(self):
        """Test random population placement."""
        grid = SpatialGrid(width=10, height=10, cell_size=1)
        manager = SpatialManager(grid)
        
        bacterium_ids = [f"bact_{i}" for i in range(5)]
        positions = manager.initialize_random_population(5, bacterium_ids)
        
        assert len(positions) == 5
        
        # Check all bacteria are placed on grid
        for bact_id in bacterium_ids:
            assert bact_id in positions
            assert grid.get_bacterium_position(bact_id) == positions[bact_id]
    
    def test_hgt_candidates(self):
        """Test HGT candidate calculation."""
        grid = SpatialGrid(width=10, height=10, cell_size=1)
        manager = SpatialManager(grid)
        
        # Place donor and potential recipients
        grid.place_bacterium("donor", Coordinate(5, 5))
        grid.place_bacterium("close", Coordinate(6, 5))    # Distance 1
        grid.place_bacterium("medium", Coordinate(7, 7))   # Distance ~2.83
        grid.place_bacterium("far", Coordinate(1, 1))      # Distance ~5.66
        
        # Test HGT candidates within radius 2
        candidates = manager.calculate_hgt_candidates("donor", hgt_radius=2.0)
        
        assert "close" in candidates
        assert "donor" not in candidates  # Should not include self
        assert "far" not in candidates    # Too far
    
    def test_bacterial_movement_simulation(self):
        """Test bacterial movement simulation."""
        grid = SpatialGrid(width=20, height=20, cell_size=1)
        manager = SpatialManager(grid)
        
        # Place bacterium
        initial_pos = Coordinate(10, 10)
        grid.place_bacterium("bact_1", initial_pos)
        
        # Simulate movement
        new_pos = manager.simulate_bacterial_movement("bact_1", movement_radius=1.0)
        
        assert new_pos is not None
        assert initial_pos.distance_to(new_pos) <= 1.0
        assert grid.get_bacterium_position("bact_1") == new_pos
        
        # Test movement of non-existent bacterium
        result = manager.simulate_bacterial_movement("nonexistent", movement_radius=1.0)
        assert result is None
    
    def test_environmental_pressure(self):
        """Test environmental pressure calculation."""
        grid = SpatialGrid(width=20, height=20, cell_size=1)
        manager = SpatialManager(grid)
        
        # Place bacterium
        pos = Coordinate(10, 10)
        grid.place_bacterium("bact_1", pos)
        
        # Add environmental factors
        grid.add_antibiotic_zone(pos, 5.0, 0.6)
        
        # Add some neighbors for crowding
        for i in range(3):
            grid.place_bacterium(f"neighbor_{i}", Coordinate(10 + i*0.5, 10))
        
        pressure = manager.get_environmental_pressure("bact_1")
        
        assert 'antibiotic_concentration' in pressure
        assert 'crowding_pressure' in pressure
        assert 'spatial_stress' in pressure
        assert pressure['antibiotic_concentration'] > 0
        assert pressure['crowding_pressure'] > 0
        
        # Test for non-existent bacterium
        empty_pressure = manager.get_environmental_pressure("nonexistent")
        assert empty_pressure == {}


class TestSpatialIntegration:
    """Integration tests for spatial system."""
    
    def test_full_spatial_workflow(self):
        """Test complete spatial workflow."""
        # Create grid and manager
        grid = SpatialGrid(width=50, height=50, cell_size=2)
        manager = SpatialManager(grid)
        
        # Initialize population
        bacterium_ids = [f"bact_{i}" for i in range(10)]
        positions = manager.initialize_random_population(10, bacterium_ids)
        
        # Add environmental factors
        grid.add_antibiotic_zone(Coordinate(25, 25), 10.0, 0.8)
        
        # Simulate some movements
        for bact_id in bacterium_ids[:5]:
            manager.simulate_bacterial_movement(bact_id, movement_radius=2.0)
        
        # Test HGT calculations
        hgt_candidates = []
        for bact_id in bacterium_ids:
            candidates = manager.calculate_hgt_candidates(bact_id, hgt_radius=5.0)
            hgt_candidates.extend(candidates)
        
        # Get final statistics
        stats = grid.get_grid_statistics()
        
        # Verify workflow completed successfully
        assert stats['total_bacteria'] == 10
        assert len(grid.antibiotic_zones) == 1
        assert isinstance(hgt_candidates, list)
        
        # Test environmental pressures
        pressures = []
        for bact_id in bacterium_ids:
            pressure = manager.get_environmental_pressure(bact_id)
            pressures.append(pressure)
        
        assert len(pressures) == 10
        assert all('spatial_stress' in p for p in pressures)
    
    def test_performance_large_population(self):
        """Test performance with larger populations."""
        grid = SpatialGrid(width=100, height=100, cell_size=5)
        manager = SpatialManager(grid)
        
        # Create larger population
        bacterium_ids = [f"bact_{i}" for i in range(100)]
        positions = manager.initialize_random_population(100, bacterium_ids)
        
        # Test neighbor detection efficiency
        sample_ids = bacterium_ids[:10]
        for bact_id in sample_ids:
            neighbors = grid.get_neighbors(bact_id, radius=10.0)
            # Should complete quickly without hanging
            assert isinstance(neighbors, list)
        
        # Test statistics calculation
        stats = grid.get_grid_statistics()
        assert stats['total_bacteria'] == 100
        
        # Verify all operations completed efficiently
        assert len(positions) == 100 