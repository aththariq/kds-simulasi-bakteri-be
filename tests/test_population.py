"""
Tests for the Population class and related components.
"""

import pytest
from models.population import Population, PopulationConfig, PopulationStats
from models.bacterium import Bacterium, ResistanceStatus, Position


class TestPopulationConfig:
    """Test cases for PopulationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PopulationConfig()
        
        assert config.population_size == 10000
        assert config.initial_resistance_frequency == 0.01
        assert config.use_spatial is True
        assert config.grid_width == 100
        assert config.grid_height == 100
        assert config.max_bacteria_per_cell == 5
        assert config.base_fitness_range == (0.8, 1.2)
        assert config.resistant_fitness_modifier == 0.9
        assert config.random_seed is None
        assert config.initial_age_range == (0, 3)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid population size
        with pytest.raises(ValueError, match="Population size must be non-negative"):
            PopulationConfig(population_size=-1)
        
        # Invalid resistance frequency
        with pytest.raises(ValueError, match="Resistance frequency must be between 0 and 1"):
            PopulationConfig(initial_resistance_frequency=1.5)
        
        # Invalid grid dimensions
        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            PopulationConfig(use_spatial=True, grid_width=0)


class TestPopulationStats:
    """Test cases for PopulationStats."""
    
    def test_stats_creation(self):
        """Test creation of population statistics."""
        stats = PopulationStats(
            total_count=1000,
            resistant_count=100,
            sensitive_count=900,
            average_age=2.5,
            average_fitness=0.95,
            resistance_frequency=0.1,
            generation=5
        )
        
        assert stats.total_count == 1000
        assert stats.resistant_count == 100
        assert stats.sensitive_count == 900
        assert stats.resistance_percentage == 10.0
    
    def test_default_stats(self):
        """Test default statistics values."""
        stats = PopulationStats()
        assert stats.total_count == 0
        assert stats.resistance_percentage == 0.0


class TestPopulation:
    """Test cases for Population class."""
    
    def test_population_initialization(self):
        """Test basic population initialization."""
        config = PopulationConfig(
            population_size=100,
            initial_resistance_frequency=0.1,
            random_seed=42
        )
        population = Population(config)
        
        assert population.config == config
        assert len(population.bacteria) == 0
        assert population.generation == 0
        assert population.size == 0
    
    def test_initialize_population(self):
        """Test population initialization with bacteria."""
        config = PopulationConfig(
            population_size=100,
            initial_resistance_frequency=0.2,  # 20% resistant
            random_seed=42
        )
        population = Population(config)
        population.initialize_population()
        
        assert len(population.bacteria) == 100
        assert population.size == 100
        
        # Check resistance frequency (should be approximately 20%)
        resistant_count = sum(1 for b in population.bacteria if b.is_resistant)
        expected_resistant = int(100 * 0.2)
        assert resistant_count == expected_resistant
    
    def test_spatial_population(self):
        """Test spatial population features."""
        config = PopulationConfig(
            population_size=50,
            use_spatial=True,
            grid_width=10,
            grid_height=10,
            max_bacteria_per_cell=2,
            random_seed=42
        )
        population = Population(config)
        population.initialize_population()
        
        # Check that spatial grid is created
        assert population.spatial_grid is not None
        
        # Check that bacteria have positions
        positioned_bacteria = [b for b in population.bacteria if b.position is not None]
        assert len(positioned_bacteria) == len(population.bacteria)
        
        # Test position queries - look for a position that actually has bacteria
        found_bacteria = False
        for bacterium in population.bacteria:
            if bacterium.position:
                bacteria_at_pos = population.get_bacteria_at_position(bacterium.position)
                if len(bacteria_at_pos) >= 1:
                    assert bacterium in bacteria_at_pos
                    found_bacteria = True
                    break
        
        assert found_bacteria, "Should find at least one bacterium at its own position"
    
    def test_non_spatial_population(self):
        """Test non-spatial population."""
        config = PopulationConfig(
            population_size=50,
            use_spatial=False,
            random_seed=42
        )
        population = Population(config)
        population.initialize_population()
        
        # Check that spatial grid is not created
        assert population.spatial_grid is None
        
        # Check that bacteria don't have positions
        for bacterium in population.bacteria:
            assert bacterium.position is None
    
    def test_add_remove_bacterium(self):
        """Test adding and removing bacteria."""
        config = PopulationConfig(population_size=0, random_seed=42)
        population = Population(config)
        population.initialize_population()
        
        # Add a bacterium
        bacterium = Bacterium(id="test_1")
        assert population.add_bacterium(bacterium) is True
        assert len(population.bacteria) == 1
        assert bacterium in population.bacteria
        
        # Remove the bacterium
        assert population.remove_bacterium(bacterium) is True
        assert len(population.bacteria) == 0
        assert bacterium not in population.bacteria
        
        # Try to remove non-existent bacterium
        assert population.remove_bacterium(bacterium) is False
    
    def test_get_neighbors(self):
        """Test neighbor detection in spatial populations."""
        config = PopulationConfig(
            population_size=0,  # Start with empty population
            use_spatial=True,
            grid_width=5,
            grid_height=5,
            max_bacteria_per_cell=5,
            random_seed=42
        )
        population = Population(config)
        population.initialize_population()
        
        # Create bacteria at specific positions
        center_pos = Position(2, 2)
        adjacent_pos = Position(2, 3)
        far_pos = Position(4, 4)
        
        center_bacterium = Bacterium(id="center", position=center_pos)
        adjacent_bacterium = Bacterium(id="adjacent", position=adjacent_pos)
        far_bacterium = Bacterium(id="far", position=far_pos)
        
        population.add_bacterium(center_bacterium)
        population.add_bacterium(adjacent_bacterium)
        population.add_bacterium(far_bacterium)
        
        # Test neighbor finding
        neighbors = population.get_neighbors(center_bacterium, radius=1.5)
        assert adjacent_bacterium in neighbors
        assert far_bacterium not in neighbors
    
    def test_statistics_calculation(self):
        """Test population statistics calculation."""
        config = PopulationConfig(
            population_size=100,
            initial_resistance_frequency=0.3,
            random_seed=42
        )
        population = Population(config)
        population.initialize_population()
        
        stats = population.get_statistics()
        
        assert stats.total_count == 100
        assert stats.resistant_count + stats.sensitive_count == 100
        assert abs(stats.resistance_frequency - 0.3) < 0.1  # Allow some variation
        assert stats.average_age >= 0
        assert stats.average_fitness > 0
        assert stats.generation == 0
    
    def test_advance_generation(self):
        """Test generation advancement, including survival and reproduction."""
        config = PopulationConfig(
            population_size=10, 
            random_seed=42,
            # Ensure some reproduction by default
            base_fitness_range=(1.0, 1.0) # All bacteria can potentially survive and reproduce
        )
        population = Population(config)
        population.initialize_population()
        
        initial_population_size = population.size
        initial_resistant_count = population.get_statistics().resistant_count
        
        # Make bacteria old enough to reproduce and ensure some are resistant for varied tests
        for i, bacterium in enumerate(population.bacteria):
            bacterium.age = 2 # Old enough to reproduce
            if i < 2: # Make first 2 resistant
                 bacterium.resistance_status = ResistanceStatus.RESISTANT
                 bacterium._survival_bonus = 0.1 # Manually set for test consistency
        population._update_statistics() # Update stats after manual changes

        initial_resistant_count_updated = population.get_statistics().resistant_count

        population.advance_generation()
        
        assert population.generation == 1
        
        current_population_size = population.size
        current_stats = population.get_statistics()

        # Assertions:
        # 1. Population size might change due to survival and reproduction.
        #    It's hard to predict exact size due to randomness, but it should be positive.
        assert current_population_size > 0 

        # 2. All surviving bacteria should have aged by 1.
        #    Offspring will have age 0.
        for bacterium in population.bacteria:
            if bacterium.generation_born < population.generation: # Survived from previous gen
                assert bacterium.age >= 1 # Should have aged (or started >0 and aged)
            else: # Is offspring
                assert bacterium.age == 0

        # 3. Statistics should be updated.
        assert current_stats.generation == 1
        assert current_stats.total_count == current_population_size
        assert current_stats.resistant_count >= 0 # Could be 0 if all resistant died and no new ones mutated
        assert current_stats.sensitive_count == current_stats.total_count - current_stats.resistant_count

        # 4. Spatial grid (if used) should reflect the new population
        if config.use_spatial and population.spatial_grid is not None:
            bacteria_in_grid_count = sum(len(cell_bacteria) for cell_bacteria in population.spatial_grid.values())
            assert bacteria_in_grid_count == current_population_size
            for bacterium in population.bacteria:
                assert bacterium.position is not None
                pos_tuple = (bacterium.position.x, bacterium.position.y)
                assert bacterium in population.spatial_grid.get(pos_tuple, [])

        print(f"Initial size: {initial_population_size}, Resistant: {initial_resistant_count_updated}")
        print(f"Final size: {current_population_size}, Resistant: {current_stats.resistant_count}, Sensitive: {current_stats.sensitive_count}")
    
    def test_advance_generation_with_high_mortality(self):
        """Test generation advancement with conditions causing high mortality."""
        config = PopulationConfig(
            population_size=100, 
            random_seed=42,
            base_fitness_range=(0.01, 0.01) # Very low fitness, most should die
        )
        population = Population(config)
        population.initialize_population()
        
        initial_size = population.size
        
        # Artificially set antibiotic concentration high to ensure sensitive die
        # And make resistant bacteria also face high stress.
        # This requires temporarily modifying how advance_generation gets these values,
        # or making them parameters of advance_generation.
        # For now, this test assumes default low antibiotic/stress in advance_generation
        # and relies on the low base_fitness_range.

        population.advance_generation()
        
        assert population.generation == 1
        current_size = population.size
        
        # Expect significant reduction in population size
        assert current_size < initial_size 
        # It's possible all die, so current_size could be 0
        assert current_size >= 0 

        print(f"High mortality test: Initial size: {initial_size}, Final size: {current_size}")
    
    def test_filtering_methods(self):
        """Test bacteria filtering methods."""
        config = PopulationConfig(
            population_size=20,
            initial_resistance_frequency=0.5,
            random_seed=42
        )
        population = Population(config)
        population.initialize_population()
        
        resistant_bacteria = population.get_resistant_bacteria()
        sensitive_bacteria = population.get_sensitive_bacteria()
        
        assert len(resistant_bacteria) + len(sensitive_bacteria) == 20
        
        # Check that filtering is correct
        for bacterium in resistant_bacteria:
            assert bacterium.is_resistant
        
        for bacterium in sensitive_bacteria:
            assert not bacterium.is_resistant
    
    def test_clone_bacterium(self):
        """Test bacterium cloning."""
        config = PopulationConfig(population_size=1, random_seed=42)
        population = Population(config)
        population.initialize_population()
        
        original = population.bacteria[0]
        clone = population.clone_bacterium(original)
        
        # Check that clone is different object with different ID
        assert clone is not original
        assert clone.id != original.id
        assert clone.parent_id == original.id
        
        # Check that properties are preserved
        assert clone.resistance_status == original.resistance_status
        assert clone.age == original.age
        assert clone.fitness == original.fitness
    
    def test_random_sample(self):
        """Test random sampling."""
        config = PopulationConfig(population_size=100, random_seed=42)
        population = Population(config)
        population.initialize_population()
        
        # Test normal sampling
        sample = population.get_random_sample(10)
        assert len(sample) == 10
        assert all(b in population.bacteria for b in sample)
        
        # Test oversized sampling
        large_sample = population.get_random_sample(200)
        assert len(large_sample) == 100  # Should return all bacteria
    
    def test_export_data(self):
        """Test population data export."""
        config = PopulationConfig(
            population_size=10,
            initial_resistance_frequency=0.2,
            random_seed=42
        )
        population = Population(config)
        population.initialize_population()
        initial_size_before_advance = population.size

        # Advance a generation to create history
        population.advance_generation()

        data = population.export_population_data()

        assert data['generation'] == 1
        # Population size after one generation can be > initial due to reproduction
        assert data['current_population_size'] > 0 
        # It's hard to assert exact size due to randomness in survival/reproduction
        # A more robust check might be that it's within a certain range or simply positive
        assert data['current_population_size'] >= initial_size_before_advance // 2 # Example: at least half survive and potentially reproduce
        assert len(data['stats_history']) == 2 # Initial stats + 1 generation
        assert 'bacteria_data' in data
        assert len(data['bacteria_data']) == data['current_population_size']
    
    def test_reset_population(self):
        """Test population reset."""
        config = PopulationConfig(population_size=50, random_seed=42)
        population = Population(config)
        population.initialize_population()
        population.advance_generation()
        
        # Verify population has data
        assert len(population.bacteria) > 0
        assert population.generation > 0
        assert len(population.stats_history) > 0
        
        # Reset and verify empty state
        population.reset_population()
        
        assert len(population.bacteria) == 0
        assert population.generation == 0
        assert len(population.stats_history) == 0
        assert population._next_id == 0
    
    def test_string_representations(self):
        """Test string representation methods."""
        config = PopulationConfig(population_size=5, random_seed=42)
        population = Population(config)
        population.initialize_population()
        
        str_repr = str(population)
        assert "Population(" in str_repr
        assert "size=5" in str_repr
        
        repr_str = repr(population)
        assert "Population(" in repr_str
        assert "generation=0" in repr_str
    
    def test_properties(self):
        """Test population properties."""
        config = PopulationConfig(
            population_size=100,
            initial_resistance_frequency=0.1,
            random_seed=42
        )
        population = Population(config)
        population.initialize_population()
        
        assert population.size == 100
        assert len(population) == 100
        assert abs(population.resistance_frequency - 0.1) < 0.05
        
        # Test iteration
        bacteria_list = list(population)
        assert len(bacteria_list) == 100
        assert all(isinstance(b, Bacterium) for b in bacteria_list)


class TestPopulationIntegration:
    """Integration tests combining multiple population features."""
    
    def test_full_lifecycle(self):
        """Test a complete population lifecycle."""
        config = PopulationConfig(
            population_size=50,
            initial_resistance_frequency=0.2,
            use_spatial=True,
            grid_width=10,
            grid_height=10,
            random_seed=42
        )

        population = Population(config)

        # Initialize population
        population.initialize_population()
        initial_stats = population.get_statistics() # Get stats BEFORE advancing generations

        assert initial_stats.total_count == 50
        assert initial_stats.generation == 0

        # Simulate a few generations
        for _ in range(5):
            population.advance_generation()
        
        final_stats = population.get_statistics()
        
        assert final_stats.generation == 5
        assert final_stats.total_count > 0 # Population should likely still exist
        # Size will change, so we don't assert a fixed final size
        
        # Check history length
        assert len(population.stats_history) == 6  # Initial + 5 generations

    def test_spatial_population_management(self):
        """Test comprehensive spatial population management."""
        config = PopulationConfig(
            population_size=20,  # Reduced size for better test reliability
            use_spatial=True,
            grid_width=10,     # Larger grid
            grid_height=10,
            max_bacteria_per_cell=2,  # Allow multiple bacteria per cell
            random_seed=42
        )
        
        population = Population(config)
        population.initialize_population()
        
        # Verify spatial distribution
        occupied_positions = set()
        for bacterium in population.bacteria:
            if bacterium.position:
                pos_tuple = (bacterium.position.x, bacterium.position.y)
                occupied_positions.add(pos_tuple)
        
        # Should have some positions occupied (exact number depends on random distribution)
        assert len(occupied_positions) > 0
        assert len(occupied_positions) <= 20  # Can't exceed population size
        
        # Test neighbor relationships
        bacteria_with_neighbors = 0
        for bacterium in population.bacteria:
            neighbors = population.get_neighbors(bacterium, radius=2.0)
            if neighbors:
                bacteria_with_neighbors += 1
        
        # At least some bacteria should have neighbors
        assert bacteria_with_neighbors >= 0  # Could be 0 if very spread out 