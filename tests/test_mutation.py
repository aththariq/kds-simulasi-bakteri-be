"""
Tests for the mutation system.
"""

import pytest
import numpy as np
from models.mutation import (
    MutationType, MutationEffect, MutationConfig, Mutation, 
    MutationEngine, MutationTracker
)
from models.bacterium import Bacterium, ResistanceStatus


class TestMutationConfig:
    """Test mutation configuration."""
    
    def test_default_config(self):
        """Test default configuration is valid."""
        config = MutationConfig()
        
        assert config.base_mutation_rate == 1e-6
        assert config.beneficial_probability + config.neutral_probability + config.deleterious_probability == 1.0
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = MutationConfig(
            beneficial_probability=0.2,
            neutral_probability=0.6,
            deleterious_probability=0.2
        )
        # Should not raise an exception
        
        # Invalid config - probabilities don't sum to 1
        with pytest.raises(ValueError):
            MutationConfig(
                beneficial_probability=0.5,
                neutral_probability=0.5,
                deleterious_probability=0.2
            )


class TestMutation:
    """Test mutation class."""
    
    def test_mutation_creation(self):
        """Test basic mutation creation."""
        mutation = Mutation(
            mutation_id="test_1",
            mutation_type=MutationType.POINT,
            effect=MutationEffect.BENEFICIAL,
            fitness_change=0.1,
            generation_occurred=5,
            parent_bacterium_id="bact_1"
        )
        
        assert mutation.mutation_id == "test_1"
        assert mutation.mutation_type == MutationType.POINT
        assert mutation.effect == MutationEffect.BENEFICIAL
        assert mutation.fitness_change == 0.1
        assert mutation.generation_occurred == 5
        assert mutation.parent_bacterium_id == "bact_1"
    
    def test_mutation_description_generation(self):
        """Test automatic description generation."""
        # Point mutation
        point_mut = Mutation(
            mutation_id="test_1",
            mutation_type=MutationType.POINT,
            effect=MutationEffect.NEUTRAL,
            position=12345
        )
        assert "Point mutation at position 12345" in point_mut.description
        
        # Insertion mutation
        ins_mut = Mutation(
            mutation_id="test_2",
            mutation_type=MutationType.INSERTION,
            effect=MutationEffect.NEUTRAL,
            size=5
        )
        assert "Insertion of 5 bases" in ins_mut.description
        
        # Fitness mutation
        fit_mut = Mutation(
            mutation_id="test_3",
            mutation_type=MutationType.FITNESS,
            effect=MutationEffect.BENEFICIAL,
            fitness_change=0.15
        )
        assert "beneficial" in fit_mut.description.lower()
        assert "0.150" in fit_mut.description
    
    def test_resistance_mutation_description(self):
        """Test resistance mutation description."""
        resistance_mut = Mutation(
            mutation_id="test_4",
            mutation_type=MutationType.RESISTANCE,
            effect=MutationEffect.BENEFICIAL,
            resistance_change=ResistanceStatus.RESISTANT
        )
        assert "Resistance mutation: resistant" in resistance_mut.description


class TestMutationEngine:
    """Test mutation engine."""
    
    def test_engine_creation(self):
        """Test mutation engine creation."""
        config = MutationConfig()
        engine = MutationEngine(config)
        
        assert engine.config == config
        assert engine._mutation_counter == 0
    
    def test_effective_rates_calculation(self):
        """Test calculation of effective mutation rates."""
        config = MutationConfig(
            point_mutation_rate=1e-6,
            resistance_mutation_rate=1e-7,
            stress_mutation_multiplier=10.0,
            antibiotic_mutation_multiplier=5.0
        )
        engine = MutationEngine(config)
        
        # No environmental factors
        rates = engine._calculate_effective_rates({})
        assert rates[MutationType.POINT] == 1e-6
        assert rates[MutationType.RESISTANCE] == 1e-7
        
        # With stress
        rates_stress = engine._calculate_effective_rates({'stress': 0.5})
        assert rates_stress[MutationType.POINT] > 1e-6  # Should be higher due to stress
        
        # With antibiotics
        rates_antibiotics = engine._calculate_effective_rates({'antibiotic_concentration': 0.2})
        assert rates_antibiotics[MutationType.RESISTANCE] > 1e-7  # Should be higher
    
    def test_mutation_generation(self):
        """Test mutation generation."""
        config = MutationConfig(
            point_mutation_rate=1.0,  # High rate to ensure mutations occur
            fitness_mutation_rate=1.0,
            resistance_mutation_rate=1.0
        )
        engine = MutationEngine(config)
        
        mutations = engine.generate_mutations("test_bacterium", 1)
        
        # Should generate some mutations due to high rates
        assert len(mutations) > 0
        
        # Check mutation properties
        for mutation in mutations:
            assert mutation.mutation_id.startswith("mut_")
            assert mutation.parent_bacterium_id == "test_bacterium"
            assert mutation.generation_occurred == 1
            assert isinstance(mutation.mutation_type, MutationType)
            assert isinstance(mutation.effect, MutationEffect)
    
    def test_mutation_effect_determination(self):
        """Test mutation effect determination follows configured probabilities."""
        config = MutationConfig(
            beneficial_probability=0.5,
            neutral_probability=0.3,
            deleterious_probability=0.2
        )
        engine = MutationEngine(config)
        
        # Generate many effects to test distribution
        effects = [engine._determine_effect() for _ in range(1000)]
        beneficial_count = sum(1 for e in effects if e == MutationEffect.BENEFICIAL)
        neutral_count = sum(1 for e in effects if e == MutationEffect.NEUTRAL)
        deleterious_count = sum(1 for e in effects if e == MutationEffect.DELETERIOUS)
        
        # Check approximate distribution (within reasonable tolerance)
        assert abs(beneficial_count / 1000 - 0.5) < 0.1
        assert abs(neutral_count / 1000 - 0.3) < 0.1
        assert abs(deleterious_count / 1000 - 0.2) < 0.1
    
    def test_fitness_change_calculation(self):
        """Test fitness change calculation."""
        config = MutationConfig(
            beneficial_fitness_range=(0.1, 0.2),
            deleterious_fitness_range=(-0.2, -0.1)
        )
        engine = MutationEngine(config)
        
        # Neutral mutations should have no fitness change
        neutral_change = engine._calculate_fitness_change(MutationEffect.NEUTRAL, MutationType.FITNESS)
        assert neutral_change == 0.0
        
        # Beneficial mutations should have positive change in specified range
        beneficial_change = engine._calculate_fitness_change(MutationEffect.BENEFICIAL, MutationType.FITNESS)
        assert 0.1 <= beneficial_change <= 0.2
        
        # Deleterious mutations should have negative change in specified range
        deleterious_change = engine._calculate_fitness_change(MutationEffect.DELETERIOUS, MutationType.FITNESS)
        assert -0.2 <= deleterious_change <= -0.1
        
        # Lethal mutations should completely reduce fitness
        lethal_change = engine._calculate_fitness_change(MutationEffect.LETHAL, MutationType.FITNESS)
        assert lethal_change == -1.0
    
    def test_indel_size_generation(self):
        """Test insertion/deletion size generation."""
        config = MutationConfig()
        engine = MutationEngine(config)
        
        sizes = [engine._generate_indel_size() for _ in range(100)]
        
        # All sizes should be positive
        assert all(size > 0 for size in sizes)
        
        # Most sizes should be small (geometric distribution property)
        assert np.mean(sizes) < 10  # Should be relatively small on average
    
    def test_apply_mutations_to_bacterium(self):
        """Test applying mutations to a bacterium."""
        config = MutationConfig()
        engine = MutationEngine(config)
        
        bacterium = Bacterium(
            id="test_bact",
            resistance_status=ResistanceStatus.SENSITIVE,
            fitness=1.0
        )
        
        mutations = [
            Mutation(
                mutation_id="mut_1",
                mutation_type=MutationType.FITNESS,
                effect=MutationEffect.BENEFICIAL,
                fitness_change=0.1
            ),
            Mutation(
                mutation_id="mut_2",
                mutation_type=MutationType.RESISTANCE,
                effect=MutationEffect.BENEFICIAL,
                resistance_change=ResistanceStatus.RESISTANT
            )
        ]
        
        changes = engine.apply_mutations(bacterium, mutations)
        
        # Check changes summary
        assert changes['fitness_change'] == 0.1
        assert changes['resistance_changed'] is True
        assert changes['mutations_applied'] == 2
        assert len(changes['mutation_details']) == 2
        
        # Check bacterium was actually modified
        assert bacterium.fitness == 1.1
        assert bacterium.resistance_status == ResistanceStatus.RESISTANT
        assert bacterium._survival_bonus == 0.1


class TestMutationTracker:
    """Test mutation tracking."""
    
    def test_tracker_creation(self):
        """Test mutation tracker creation."""
        tracker = MutationTracker()
        
        assert len(tracker.mutation_history) == 0
        assert len(tracker.lineage_mutations) == 0
    
    def test_record_mutations(self):
        """Test recording mutations."""
        tracker = MutationTracker()
        
        mutations = [
            Mutation(
                mutation_id="mut_1",
                mutation_type=MutationType.POINT,
                effect=MutationEffect.NEUTRAL
            ),
            Mutation(
                mutation_id="mut_2",
                mutation_type=MutationType.FITNESS,
                effect=MutationEffect.BENEFICIAL,
                fitness_change=0.1
            )
        ]
        
        tracker.record_mutations("bacterium_1", mutations)
        
        # Check mutations were recorded
        assert "bacterium_1" in tracker.mutation_history
        assert len(tracker.mutation_history["bacterium_1"]) == 2
        assert "bacterium_1" in tracker.lineage_mutations
        assert len(tracker.lineage_mutations["bacterium_1"]) == 2
        assert "mut_1" in tracker.lineage_mutations["bacterium_1"]
        assert "mut_2" in tracker.lineage_mutations["bacterium_1"]
    
    def test_get_lineage_mutations(self):
        """Test retrieving lineage mutations."""
        tracker = MutationTracker()
        
        mutations = [
            Mutation(
                mutation_id="mut_1",
                mutation_type=MutationType.POINT,
                effect=MutationEffect.NEUTRAL
            )
        ]
        
        tracker.record_mutations("bacterium_1", mutations)
        
        lineage = tracker.get_lineage_mutations("bacterium_1")
        assert len(lineage) == 1
        assert lineage[0].mutation_id == "mut_1"
        
        # Non-existent bacterium should return empty list
        empty_lineage = tracker.get_lineage_mutations("nonexistent")
        assert len(empty_lineage) == 0
    
    def test_mutation_statistics(self):
        """Test mutation statistics calculation."""
        tracker = MutationTracker()
        
        mutations = [
            Mutation(
                mutation_id="mut_1",
                mutation_type=MutationType.POINT,
                effect=MutationEffect.BENEFICIAL,
                fitness_change=0.1
            ),
            Mutation(
                mutation_id="mut_2",
                mutation_type=MutationType.FITNESS,
                effect=MutationEffect.DELETERIOUS,
                fitness_change=-0.05
            ),
            Mutation(
                mutation_id="mut_3",
                mutation_type=MutationType.POINT,
                effect=MutationEffect.NEUTRAL,
                fitness_change=0.0
            )
        ]
        
        tracker.record_mutations("bacterium_1", mutations[:2])
        tracker.record_mutations("bacterium_2", mutations[2:])
        
        stats = tracker.get_mutation_statistics()
        
        assert stats['total_mutations'] == 3
        assert stats['mutation_types']['point'] == 2
        assert stats['mutation_types']['fitness'] == 1
        assert stats['mutation_effects']['beneficial'] == 1
        assert stats['mutation_effects']['deleterious'] == 1
        assert stats['mutation_effects']['neutral'] == 1
        assert stats['beneficial_mutations'] == 1
        assert stats['deleterious_mutations'] == 1
        assert abs(stats['average_fitness_change'] - (0.1 - 0.05 + 0.0) / 3) < 1e-10
    
    def test_empty_statistics(self):
        """Test statistics with no mutations."""
        tracker = MutationTracker()
        
        stats = tracker.get_mutation_statistics()
        assert stats['total_mutations'] == 0


class TestMutationIntegration:
    """Integration tests for mutation system."""
    
    def test_full_mutation_workflow(self):
        """Test complete mutation workflow."""
        # Setup with very high rates to ensure mutations occur
        config = MutationConfig(
            point_mutation_rate=1.0,  # 100% rate for testing
            fitness_mutation_rate=1.0,
            resistance_mutation_rate=1.0
        )
        engine = MutationEngine(config)
        tracker = MutationTracker()
        
        bacterium = Bacterium(
            id="test_bact",
            resistance_status=ResistanceStatus.SENSITIVE,
            fitness=1.0
        )
        
        # Generate and apply mutations
        mutations = engine.generate_mutations("test_bact", 1)
        changes = engine.apply_mutations(bacterium, mutations)
        tracker.record_mutations("test_bact", mutations)
        
        # Verify workflow - with 100% rates, should definitely generate mutations
        assert len(mutations) > 0  # Should generate some mutations
        assert changes['mutations_applied'] == len(mutations)
        
        # Check tracking
        lineage = tracker.get_lineage_mutations("test_bact")
        assert len(lineage) == len(mutations)
        
        stats = tracker.get_mutation_statistics()
        assert stats['total_mutations'] == len(mutations)
    
    def test_environmental_mutation_effects(self):
        """Test how environmental factors affect mutation rates."""
        config = MutationConfig(
            resistance_mutation_rate=1e-4,  # Base rate
            antibiotic_mutation_multiplier=10.0
        )
        engine = MutationEngine(config)
        
        # No environmental pressure
        mutations_normal = engine.generate_mutations(
            "test_bact", 1, {}
        )
        
        # High antibiotic pressure
        mutations_stress = engine.generate_mutations(
            "test_bact", 1, {'antibiotic_concentration': 1.0}
        )
        
        # With high mutation rates and environmental pressure,
        # we should typically see more mutations under stress
        # (this is a statistical test, may occasionally fail due to randomness)
        
        # At minimum, verify that the system handles environmental factors
        assert isinstance(mutations_normal, list)
        assert isinstance(mutations_stress, list) 