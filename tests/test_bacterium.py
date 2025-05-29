"""
Test suite for bacterium models and behaviors.
"""

import pytest
from models.bacterium import Bacterium, ResistanceStatus, Position
from models.mutation import MutationEngine, MutationConfig


class TestBacterium:
    """Test bacterium class functionality."""
    
    def test_bacterium_creation(self):
        """Test basic bacterium creation."""
        bacterium = Bacterium(id="test_1")
        
        assert bacterium.id == "test_1"
        assert bacterium.resistance_status == ResistanceStatus.SENSITIVE
        assert bacterium.age == 0
        assert bacterium.fitness == 1.0
        assert bacterium.position is None
        assert bacterium.generation_born == 0
        assert bacterium.parent_id is None
        assert bacterium._reproduction_attempts == 0
        assert bacterium._survival_bonus == 0.0
    
    def test_resistant_bacterium_creation(self):
        """Test creation of resistant bacterium."""
        bacterium = Bacterium(
            id="test_2",
            resistance_status=ResistanceStatus.RESISTANT,
            fitness=0.9
        )
        
        assert bacterium.is_resistant is True
        assert bacterium._survival_bonus == 0.1  # Resistance bonus applied
    
    def test_age_and_effective_fitness(self):
        """Test aging and effective fitness calculation."""
        bacterium = Bacterium(id="test_3", fitness=1.0)
        
        initial_effective = bacterium.effective_fitness
        assert initial_effective == 1.0
        
        bacterium.age_one_generation()
        assert bacterium.age == 1
        
        aged_effective = bacterium.effective_fitness
        assert aged_effective < initial_effective  # Fitness should decrease with age
    
    def test_survival_probability_with_antibiotics(self):
        """Test survival probability calculation with different antibiotic levels."""
        sensitive = Bacterium(id="sensitive", resistance_status=ResistanceStatus.SENSITIVE)
        resistant = Bacterium(id="resistant", resistance_status=ResistanceStatus.RESISTANT)
        
        # No antibiotics - both should have similar survival (resistant pays cost)
        sensitive_survival_no_ab = sensitive.calculate_survival_probability(0.0)
        resistant_survival_no_ab = resistant.calculate_survival_probability(0.0)
        
        # Resistant should have slightly lower survival due to resistance cost
        assert resistant_survival_no_ab < sensitive_survival_no_ab
        
        # High antibiotics - resistant should survive much better
        sensitive_survival_high_ab = sensitive.calculate_survival_probability(5.0)
        resistant_survival_high_ab = resistant.calculate_survival_probability(5.0)
        
        assert resistant_survival_high_ab > sensitive_survival_high_ab
    
    def test_reproduction(self):
        """Test basic reproduction."""
        parent = Bacterium(id="parent", age=2, fitness=1.0)
        
        offspring = parent.reproduce(
            mutation_rate=0.0,  # No mutations for basic test
            generation=1,
            next_id_generator=lambda: "offspring_1"
        )
        
        assert offspring is not None
        assert offspring.id == "offspring_1"
        assert offspring.parent_id == "parent"
        assert offspring.age == 0
        assert offspring.generation_born == 1
        assert offspring.fitness == parent.fitness
        assert offspring.resistance_status == parent.resistance_status
    
    def test_mutation_during_reproduction(self):
        """Test mutations during reproduction using legacy system."""
        parent = Bacterium(
            id="parent",
            age=1,  # Set age to 1 so bacterium can reproduce
            resistance_status=ResistanceStatus.SENSITIVE,
            fitness=1.0
        )
        
        # Test multiple reproductions to check for mutations
        offspring_list = []
        for i in range(100):  # Try many times to get at least one mutation
            offspring = parent.reproduce(
                mutation_rate=0.5,  # High mutation rate
                generation=1,
                next_id_generator=lambda: f"offspring_{i}"
            )
            if offspring:
                offspring_list.append(offspring)
        
        # Should have some offspring
        assert len(offspring_list) > 0
        
        # Check if any mutations occurred
        resistant_offspring = [o for o in offspring_list if o.is_resistant]
        
        # With 50% mutation rate and 100 attempts, should have some resistant offspring
        assert len(resistant_offspring) > 0
    
    def test_reproduction_with_mutation_engine(self):
        """Test reproduction using the new mutation engine."""
        # Create mutation engine with high mutation rates for testing
        config = MutationConfig(
            point_mutation_rate=0.5,
            fitness_mutation_rate=0.3,
            resistance_mutation_rate=0.4
        )
        mutation_engine = MutationEngine(config)
        
        parent = Bacterium(
            id="parent",
            age=1,  # Set age to 1 so bacterium can reproduce
            resistance_status=ResistanceStatus.SENSITIVE,
            fitness=1.0
        )
        
        # Test reproduction with mutation engine
        offspring = parent.reproduce(
            generation=1,
            next_id_generator=lambda: "offspring_advanced",
            mutation_engine=mutation_engine,
            environmental_factors={'stress': 0.2, 'antibiotic_concentration': 0.1}
        )
        
        assert offspring is not None
        assert offspring.id == "offspring_advanced"
        assert offspring.parent_id == "parent"
        
        # With high mutation rates, offspring might have different properties
        # (This is stochastic, so we just verify the system works)
        assert isinstance(offspring.fitness, float)
        assert offspring.fitness > 0.0
    
    def test_position_and_distance(self):
        """Test position-related functionality."""
        pos1 = Position(0, 0)
        pos2 = Position(3, 4)
        
        distance = pos1.distance_to(pos2)
        assert distance == 5.0  # 3-4-5 triangle
        
        bacterium = Bacterium(id="positioned", position=pos1)
        assert bacterium.position == pos1
    
    def test_spatial_reproduction(self):
        """Test reproduction with spatial positioning."""
        parent = Bacterium(
            id="parent",
            age=1,  # Set age to 1 so bacterium can reproduce
            position=Position(5, 5),
            fitness=1.0
        )
        
        offspring = parent.reproduce(
            mutation_rate=0.0,
            generation=1,
            next_id_generator=lambda: "spatial_offspring"
        )
        
        assert offspring is not None
        assert offspring.position is not None
        
        # Offspring should be placed adjacent to parent
        parent_pos = parent.position
        offspring_pos = offspring.position
        
        distance = parent_pos.distance_to(offspring_pos)
        assert distance <= 1.42  # Maximum distance for adjacent cells (diagonal)


class TestPosition:
    """Test position class."""
    
    def test_position_creation(self):
        """Test position creation."""
        pos = Position(10, 20)
        assert pos.x == 10
        assert pos.y == 20
    
    def test_distance_calculation(self):
        """Test distance calculation between positions."""
        pos1 = Position(0, 0)
        pos2 = Position(0, 5)
        pos3 = Position(5, 0)
        pos4 = Position(3, 4)
        
        assert pos1.distance_to(pos2) == 5.0
        assert pos1.distance_to(pos3) == 5.0
        assert pos1.distance_to(pos4) == 5.0
    
    def test_adjacency(self):
        """Test adjacency detection."""
        center = Position(5, 5)
        
        # Adjacent positions (8-neighborhood)
        adjacent_positions = [
            Position(4, 4), Position(4, 5), Position(4, 6),
            Position(5, 4),                  Position(5, 6),
            Position(6, 4), Position(6, 5), Position(6, 6)
        ]
        
        for pos in adjacent_positions:
            assert center.is_adjacent(pos), f"Position {pos} should be adjacent to {center}"
        
        # Non-adjacent positions
        non_adjacent_positions = [
            Position(3, 3), Position(7, 7), Position(5, 8), Position(2, 5)
        ]
        
        for pos in non_adjacent_positions:
            assert not center.is_adjacent(pos), f"Position {pos} should not be adjacent to {center}"
        
        # Position should not be adjacent to itself
        assert not center.is_adjacent(center) 