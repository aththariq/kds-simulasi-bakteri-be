"""
Tests for the selection pressure system.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from models.selection import (
    PressureType, PressureConfig, SelectionResult, SelectionPressure,
    AntimicrobialPressure, ResourcePressure, EnvironmentalPressure,
    SpatialPressure, SelectionEnvironment, CompetitivePressure
)
from models.bacterium import Bacterium, ResistanceStatus


class TestPressureConfig:
    """Test PressureConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PressureConfig(pressure_type=PressureType.ANTIMICROBIAL)
        
        assert config.pressure_type == PressureType.ANTIMICROBIAL
        assert config.intensity == 1.0
        assert config.duration is None
        assert config.time_profile == "constant"
        assert config.enabled is True
        assert config.parameters == {}
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = PressureConfig(
            pressure_type=PressureType.RESOURCE,
            intensity=2.5,
            duration=100,
            time_profile="linear",
            enabled=False,
            parameters={'carrying_capacity': 5000}
        )
        
        assert config.pressure_type == PressureType.RESOURCE
        assert config.intensity == 2.5
        assert config.duration == 100
        assert config.time_profile == "linear"
        assert config.enabled is False
        assert config.parameters['carrying_capacity'] == 5000
    
    def test_validation_negative_intensity(self):
        """Test validation for negative intensity."""
        with pytest.raises(ValueError, match="Pressure intensity must be non-negative"):
            PressureConfig(
                pressure_type=PressureType.ANTIMICROBIAL,
                intensity=-1.0
            )
    
    def test_validation_negative_duration(self):
        """Test validation for negative duration."""
        with pytest.raises(ValueError, match="Duration must be non-negative"):
            PressureConfig(
                pressure_type=PressureType.ANTIMICROBIAL,
                duration=-10
            )


class TestSelectionResult:
    """Test SelectionResult class."""
    
    def test_selection_result_creation(self):
        """Test creating a SelectionResult."""
        result = SelectionResult(
            bacterium_id="test_bact",
            original_fitness=1.0,
            modified_fitness=0.8,
            survival_probability=0.9,
            pressure_effects={'test_effect': 0.2}
        )
        
        assert result.bacterium_id == "test_bact"
        assert result.original_fitness == 1.0
        assert result.modified_fitness == 0.8
        assert result.survival_probability == 0.9
        assert result.pressure_effects['test_effect'] == 0.2
        assert result.selected_for_survival is False
    
    def test_fitness_change_property(self):
        """Test fitness_change property calculation."""
        result = SelectionResult(
            bacterium_id="test_bact",
            original_fitness=1.0,
            modified_fitness=0.7,
            survival_probability=0.8,
            pressure_effects={}
        )
        
        assert abs(result.fitness_change - (-0.3)) < 0.0001  # Use tolerance for floating point comparison


class TestAntimicrobialPressure:
    """Test AntimicrobialPressure class."""
    
    def test_initialization(self):
        """Test AntimicrobialPressure initialization."""
        config = PressureConfig(
            pressure_type=PressureType.ANTIMICROBIAL,
            intensity=2.0,
            parameters={
                'mic_sensitive': 1.0,
                'mic_resistant': 8.0,
                'hill_coefficient': 2.0,
                'max_kill_rate': 0.95
            }
        )
        pressure = AntimicrobialPressure(config)
        
        assert pressure.mic_sensitive == 1.0
        assert pressure.mic_resistant == 8.0
        assert pressure.hill_coefficient == 2.0
        assert pressure.max_kill_rate == 0.95
    
    def test_apply_to_sensitive_bacterium(self):
        """Test antimicrobial pressure on sensitive bacterium."""
        config = PressureConfig(
            pressure_type=PressureType.ANTIMICROBIAL,
            intensity=2.0  # 2x MIC for sensitive
        )
        pressure = AntimicrobialPressure(config)
        
        bacterium = Bacterium(
            id="sensitive_bact",
            resistance_status=ResistanceStatus.SENSITIVE,
            fitness=1.0
        )
        
        result = pressure.apply_to_bacterium(bacterium, {}, 0)
        
        assert result.bacterium_id == "sensitive_bact"
        assert result.original_fitness == 1.0
        assert result.modified_fitness < result.original_fitness  # Fitness reduced
        assert result.survival_probability < 1.0  # Survival probability reduced
        assert 'antimicrobial_concentration' in result.pressure_effects
        assert 'mic_value' in result.pressure_effects
    
    def test_apply_to_resistant_bacterium(self):
        """Test antimicrobial pressure on resistant bacterium."""
        config = PressureConfig(
            pressure_type=PressureType.ANTIMICROBIAL,
            intensity=2.0  # 2x MIC for sensitive, but below MIC for resistant
        )
        pressure = AntimicrobialPressure(config)
        
        bacterium = Bacterium(
            id="resistant_bact",
            resistance_status=ResistanceStatus.RESISTANT,
            fitness=1.0
        )
        
        result = pressure.apply_to_bacterium(bacterium, {}, 0)
        
        # Resistant bacterium should survive better than sensitive
        assert result.survival_probability > 0.5
        assert 'resistant_mic_8.0' in result.pressure_effects
    
    def test_hill_equation_calculation(self):
        """Test Hill equation kill probability calculation."""
        config = PressureConfig(
            pressure_type=PressureType.ANTIMICROBIAL,
            parameters={'max_kill_rate': 0.9, 'hill_coefficient': 2.0}
        )
        pressure = AntimicrobialPressure(config)
        
        # Test at MIC (should be 50% of max kill rate)
        kill_prob = pressure._calculate_kill_probability(1.0, 1.0)
        assert abs(kill_prob - 0.45) < 0.1  # Around 50% of 0.9
        
        # Test at 0 concentration
        kill_prob = pressure._calculate_kill_probability(0.0, 1.0)
        assert kill_prob == 0.0
        
        # Test at high concentration
        kill_prob = pressure._calculate_kill_probability(10.0, 1.0)
        assert kill_prob > 0.8  # Should approach max kill rate
    
    def test_time_profiles(self):
        """Test different time profiles for antimicrobial pressure."""
        # Constant profile
        config = PressureConfig(
            pressure_type=PressureType.ANTIMICROBIAL,
            intensity=2.0,
            time_profile="constant"
        )
        pressure = AntimicrobialPressure(config)
        assert pressure.get_pressure_intensity(0) == 2.0
        assert pressure.get_pressure_intensity(10) == 2.0
        
        # Linear profile
        config.time_profile = "linear"
        config.parameters = {'slope': 0.1}
        pressure = AntimicrobialPressure(config)
        assert pressure.get_pressure_intensity(0) == 2.0
        assert pressure.get_pressure_intensity(10) == 3.0
        
        # Exponential decay profile
        config.time_profile = "exponential"
        config.parameters = {'decay_rate': 0.1}
        pressure = AntimicrobialPressure(config)
        intensity_0 = pressure.get_pressure_intensity(0)
        intensity_10 = pressure.get_pressure_intensity(10)
        assert intensity_10 < intensity_0  # Should decay
        
        # Pulse profile
        config.time_profile = "pulse"
        config.parameters = {'pulse_interval': 24, 'pulse_duration': 6}
        pressure = AntimicrobialPressure(config)
        assert pressure.get_pressure_intensity(3) == 2.0  # Within pulse
        assert pressure.get_pressure_intensity(10) == 0.0  # Outside pulse


class TestResourcePressure:
    """Test ResourcePressure class."""
    
    def test_initialization(self):
        """Test ResourcePressure initialization."""
        config = PressureConfig(
            pressure_type=PressureType.RESOURCE,
            parameters={
                'carrying_capacity': 5000,
                'competition_strength': 1.5,
                'resource_efficiency_resistant': 0.8
            }
        )
        pressure = ResourcePressure(config)
        
        assert pressure.carrying_capacity == 5000
        assert pressure.competition_strength == 1.5
        assert pressure.resource_efficiency_resistant == 0.8
    
    def test_apply_with_low_population(self):
        """Test resource pressure with low population."""
        config = PressureConfig(pressure_type=PressureType.RESOURCE)
        pressure = ResourcePressure(config)
        
        bacterium = Bacterium(id="test_bact", fitness=1.0)
        context = {'total_population': 100, 'local_density': 1.0}
        
        result = pressure.apply_to_bacterium(bacterium, context, 0)
        
        # Low population should have minimal competition
        assert result.modified_fitness >= result.original_fitness * 0.9
        assert result.survival_probability > 0.8
    
    def test_apply_with_high_population(self):
        """Test resource pressure with high population."""
        config = PressureConfig(
            pressure_type=PressureType.RESOURCE,
            intensity=2.0,
            parameters={'carrying_capacity': 1000}
        )
        pressure = ResourcePressure(config)
        
        bacterium = Bacterium(id="test_bact", fitness=1.0)
        context = {'total_population': 2000, 'local_density': 5.0}  # Over capacity
        
        result = pressure.apply_to_bacterium(bacterium, context, 0)
        
        # High population should have strong competition
        assert result.modified_fitness < result.original_fitness
        assert 'competition_factor' in result.pressure_effects
        assert 'local_competition' in result.pressure_effects
    
    def test_resistant_vs_sensitive_resource_efficiency(self):
        """Test resource efficiency difference between resistant and sensitive bacteria."""
        config = PressureConfig(
            pressure_type=PressureType.RESOURCE,
            parameters={'resource_efficiency_resistant': 0.8}
        )
        pressure = ResourcePressure(config)
        
        context = {'total_population': 2000, 'local_density': 3.0}
        
        # Test sensitive bacterium
        sensitive = Bacterium(id="sensitive", resistance_status=ResistanceStatus.SENSITIVE, fitness=1.0)
        sensitive_result = pressure.apply_to_bacterium(sensitive, context, 0)
        
        # Test resistant bacterium
        resistant = Bacterium(id="resistant", resistance_status=ResistanceStatus.RESISTANT, fitness=1.0)
        resistant_result = pressure.apply_to_bacterium(resistant, context, 0)
        
        # Resistant bacteria should have lower resource efficiency (higher fitness reduction)
        assert resistant_result.modified_fitness < sensitive_result.modified_fitness


class TestEnvironmentalPressure:
    """Test EnvironmentalPressure class."""
    
    def test_initialization(self):
        """Test EnvironmentalPressure initialization."""
        config = PressureConfig(
            pressure_type=PressureType.ENVIRONMENTAL,
            parameters={
                'stress_factors': ['temperature', 'ph'],
                'stress_tolerance_resistant': 1.2,
                'baseline_stress': 0.2
            }
        )
        pressure = EnvironmentalPressure(config)
        
        assert pressure.stress_factors == ['temperature', 'ph']
        assert pressure.stress_tolerance_resistant == 1.2
        assert pressure.baseline_stress == 0.2
    
    def test_apply_environmental_stress(self):
        """Test environmental stress application."""
        config = PressureConfig(
            pressure_type=PressureType.ENVIRONMENTAL,
            intensity=1.5
        )
        pressure = EnvironmentalPressure(config)
        
        bacterium = Bacterium(id="test_bact", fitness=1.0)
        
        result = pressure.apply_to_bacterium(bacterium, {}, 0)
        
        assert result.modified_fitness <= result.original_fitness
        assert 'stress_level' in result.pressure_effects
        assert 'effective_stress' in result.pressure_effects
    
    def test_resistant_stress_tolerance(self):
        """Test that resistant bacteria have better stress tolerance."""
        config = PressureConfig(
            pressure_type=PressureType.ENVIRONMENTAL,
            intensity=2.0,
            parameters={'stress_tolerance_resistant': 1.5}
        )
        pressure = EnvironmentalPressure(config)
        
        # Test sensitive bacterium
        sensitive = Bacterium(id="sensitive", resistance_status=ResistanceStatus.SENSITIVE, fitness=1.0)
        sensitive_result = pressure.apply_to_bacterium(sensitive, {}, 0)
        
        # Test resistant bacterium
        resistant = Bacterium(id="resistant", resistance_status=ResistanceStatus.RESISTANT, fitness=1.0)
        resistant_result = pressure.apply_to_bacterium(resistant, {}, 0)
        
        # Resistant bacteria should survive better under environmental stress
        assert resistant_result.survival_probability >= sensitive_result.survival_probability
    
    def test_sine_wave_profile(self):
        """Test sine wave environmental stress profile."""
        config = PressureConfig(
            pressure_type=PressureType.ENVIRONMENTAL,
            intensity=1.0,
            time_profile="sine",
            parameters={'period': 24, 'amplitude': 0.5}
        )
        pressure = EnvironmentalPressure(config)
        
        # Test stress levels at different time points
        stress_0 = pressure.get_pressure_intensity(0)
        stress_6 = pressure.get_pressure_intensity(6)
        stress_12 = pressure.get_pressure_intensity(12)
        
        # Should vary sinusoidally
        assert stress_0 == 1.0  # Base level
        assert stress_6 != stress_0  # Should be different
        assert stress_12 == 1.0  # Back to base after half period


class TestSpatialPressure:
    """Test SpatialPressure class."""
    
    def test_initialization(self):
        """Test SpatialPressure initialization."""
        config = PressureConfig(
            pressure_type=PressureType.SPATIAL,
            parameters={
                'crowding_threshold': 5,
                'dispersal_advantage': 0.2
            }
        )
        pressure = SpatialPressure(config)
        
        assert pressure.crowding_threshold == 5
        assert pressure.dispersal_advantage == 0.2
    
    def test_apply_with_low_density(self):
        """Test spatial pressure with low local density."""
        config = PressureConfig(pressure_type=PressureType.SPATIAL)
        pressure = SpatialPressure(config)
        
        bacterium = Bacterium(id="test_bact", fitness=1.0)
        context = {'local_density': 2, 'neighbor_count': 1}
        
        result = pressure.apply_to_bacterium(bacterium, context, 0)
        
        # Low density should have minimal crowding effect
        assert result.modified_fitness >= result.original_fitness * 0.9
        assert result.survival_probability > 0.9
    
    def test_apply_with_high_density(self):
        """Test spatial pressure with high local density."""
        config = PressureConfig(
            pressure_type=PressureType.SPATIAL,
            intensity=1.5,
            parameters={'crowding_threshold': 3}
        )
        pressure = SpatialPressure(config)
        
        bacterium = Bacterium(id="test_bact", fitness=1.0)
        context = {'local_density': 8, 'neighbor_count': 7}  # High crowding
        
        result = pressure.apply_to_bacterium(bacterium, context, 0)
        
        # High density should reduce fitness and survival
        assert result.modified_fitness < result.original_fitness
        assert result.survival_probability < 1.0
        assert 'crowding_factor' in result.pressure_effects


class TestCompetitivePressure:
    """Test CompetitivePressure class."""
    
    def test_initialization(self):
        """Test CompetitivePressure initialization."""
        config = PressureConfig(
            pressure_type=PressureType.COMPETITIVE,
            parameters={
                'competition_model': 'resistance_dominance',
                'interaction_radius': 2.0,
                'frequency_dependent': True,
                'dominance_factor': 1.5
            }
        )
        pressure = CompetitivePressure(config)
        
        assert pressure.competition_model == 'resistance_dominance'
        assert pressure.interaction_radius == 2.0
        assert pressure.frequency_dependent == True
        assert pressure.dominance_factor == 1.5
    
    def test_apply_with_no_competitors(self):
        """Test competitive pressure with no competitors."""
        config = PressureConfig(pressure_type=PressureType.COMPETITIVE)
        pressure = CompetitivePressure(config)
        
        bacterium = Bacterium(id="test_bact", fitness=1.0)
        context = {'competitors': [], 'total_population': 1}
        
        result = pressure.apply_to_bacterium(bacterium, context, 0)
        
        # No competition should have no effect
        assert result.modified_fitness == result.original_fitness
        assert result.survival_probability == 1.0
        assert result.pressure_effects['competition_strength'] == 0.0
    
    def test_fitness_based_competition(self):
        """Test fitness-based competition model."""
        config = PressureConfig(
            pressure_type=PressureType.COMPETITIVE,
            intensity=1.0,
            parameters={'competition_model': 'fitness_based'}
        )
        pressure = CompetitivePressure(config)
        
        # Create high-fitness bacterium
        high_fitness_bact = Bacterium(id="high_fit", fitness=1.5)
        
        # Create lower-fitness competitors
        competitors = [
            Bacterium(id="comp1", fitness=1.0),
            Bacterium(id="comp2", fitness=0.8)
        ]
        
        context = {'competitors': competitors, 'total_population': 3}
        
        result = pressure.apply_to_bacterium(high_fitness_bact, context, 0)
        
        # High fitness should provide competitive advantage
        assert result.modified_fitness > result.original_fitness
        assert 'competition_strength' in result.pressure_effects
        assert result.pressure_effects['competition_strength'] > 0
    
    def test_resistance_dominance_competition(self):
        """Test resistance dominance competition model."""
        config = PressureConfig(
            pressure_type=PressureType.COMPETITIVE,
            intensity=1.0,
            parameters={
                'competition_model': 'resistance_dominance',
                'dominance_factor': 1.2
            }
        )
        pressure = CompetitivePressure(config)
        
        # Test resistant bacterium in population with few resistant
        resistant_bact = Bacterium(id="resistant", resistance_status=ResistanceStatus.RESISTANT, fitness=1.0)
        
        competitors = [
            Bacterium(id="sens1", resistance_status=ResistanceStatus.SENSITIVE, fitness=1.0),
            Bacterium(id="sens2", resistance_status=ResistanceStatus.SENSITIVE, fitness=1.0),
            Bacterium(id="res1", resistance_status=ResistanceStatus.RESISTANT, fitness=1.0)
        ]
        
        context = {'competitors': competitors, 'total_population': 4}
        
        result = pressure.apply_to_bacterium(resistant_bact, context, 0)
        
        # Resistant should benefit when rare
        assert result.modified_fitness > result.original_fitness
        assert 'competition_strength' in result.pressure_effects
    
    def test_frequency_dependent_selection(self):
        """Test frequency-dependent selection effects."""
        config = PressureConfig(
            pressure_type=PressureType.COMPETITIVE,
            intensity=1.0,
            parameters={'frequency_dependent': True}
        )
        pressure = CompetitivePressure(config)
        
        # Test resistant bacterium in high resistance frequency population
        resistant_bact = Bacterium(id="resistant", resistance_status=ResistanceStatus.RESISTANT, fitness=1.0)
        
        # Many resistant competitors (high frequency)
        competitors = [
            Bacterium(id="res1", resistance_status=ResistanceStatus.RESISTANT, fitness=1.0),
            Bacterium(id="res2", resistance_status=ResistanceStatus.RESISTANT, fitness=1.0),
            Bacterium(id="res3", resistance_status=ResistanceStatus.RESISTANT, fitness=1.0),
            Bacterium(id="sens1", resistance_status=ResistanceStatus.SENSITIVE, fitness=1.0)
        ]
        
        context = {'competitors': competitors, 'total_population': 5}
        
        result = pressure.apply_to_bacterium(resistant_bact, context, 0)
        
        # Should have frequency effect recorded
        assert 'frequency_effect' in result.pressure_effects
        assert result.pressure_effects['frequency_effect'] != 1.0
    
    def test_sensitive_vs_resistant_competition(self):
        """Test competition between sensitive and resistant bacteria."""
        config = PressureConfig(
            pressure_type=PressureType.COMPETITIVE,
            intensity=1.0,
            parameters={'competition_model': 'fitness_based'}
        )
        pressure = CompetitivePressure(config)
        
        # Same fitness but different resistance status
        sensitive_bact = Bacterium(id="sensitive", resistance_status=ResistanceStatus.SENSITIVE, fitness=1.0)
        resistant_bact = Bacterium(id="resistant", resistance_status=ResistanceStatus.RESISTANT, fitness=1.0)
        
        competitors = [resistant_bact]
        context = {'competitors': competitors, 'total_population': 2}
        
        sensitive_result = pressure.apply_to_bacterium(sensitive_bact, context, 0)
        
        competitors = [sensitive_bact]
        resistant_result = pressure.apply_to_bacterium(resistant_bact, context, 0)
        
        # Both should have similar competitive effects given equal fitness
        assert abs(sensitive_result.pressure_effects['competition_strength']) < 0.1
        assert abs(resistant_result.pressure_effects['competition_strength']) < 0.1


class TestSelectionEnvironment:
    """Test SelectionEnvironment class."""
    
    def test_initialization(self):
        """Test SelectionEnvironment initialization."""
        env = SelectionEnvironment()
        
        assert len(env.pressures) == 0
        assert len(env.interaction_effects) == 0
    
    def test_add_remove_pressures(self):
        """Test adding and removing pressures."""
        env = SelectionEnvironment()
        
        # Add pressures
        antimicrobial_config = PressureConfig(pressure_type=PressureType.ANTIMICROBIAL)
        antimicrobial_pressure = AntimicrobialPressure(antimicrobial_config)
        env.add_pressure(antimicrobial_pressure)
        
        resource_config = PressureConfig(pressure_type=PressureType.RESOURCE)
        resource_pressure = ResourcePressure(resource_config)
        env.add_pressure(resource_pressure)
        
        assert len(env.pressures) == 2
        
        # Remove pressure
        removed = env.remove_pressure(PressureType.ANTIMICROBIAL)
        assert removed is True
        assert len(env.pressures) == 1
        assert env.pressures[0].config.pressure_type == PressureType.RESOURCE
    
    def test_apply_selection_no_pressures(self):
        """Test apply_selection with no pressures."""
        env = SelectionEnvironment()
        bacteria = [Bacterium(id="bact1", fitness=1.0)]
        
        results = env.apply_selection(bacteria, {}, 0)
        
        assert len(results) == 1
        assert results[0].modified_fitness == results[0].original_fitness
        assert results[0].survival_probability == 1.0
    
    def test_apply_selection_single_pressure(self):
        """Test apply_selection with single pressure."""
        env = SelectionEnvironment()
        
        config = PressureConfig(pressure_type=PressureType.ANTIMICROBIAL, intensity=2.0)
        pressure = AntimicrobialPressure(config)
        env.add_pressure(pressure)
        
        bacteria = [
            Bacterium(id="sensitive", resistance_status=ResistanceStatus.SENSITIVE, fitness=1.0),
            Bacterium(id="resistant", resistance_status=ResistanceStatus.RESISTANT, fitness=1.0)
        ]
        
        results = env.apply_selection(bacteria, {}, 0)
        
        assert len(results) == 2
        # Resistant bacterium should survive better
        resistant_result = next(r for r in results if r.bacterium_id == "resistant")
        sensitive_result = next(r for r in results if r.bacterium_id == "sensitive")
        assert resistant_result.survival_probability > sensitive_result.survival_probability
    
    def test_apply_selection_multiple_pressures(self):
        """Test apply_selection with multiple pressures."""
        env = SelectionEnvironment()
        
        # Add antimicrobial pressure
        antimicrobial_config = PressureConfig(pressure_type=PressureType.ANTIMICROBIAL, intensity=1.5)
        env.add_pressure(AntimicrobialPressure(antimicrobial_config))
        
        # Add resource pressure
        resource_config = PressureConfig(pressure_type=PressureType.RESOURCE, intensity=1.0)
        env.add_pressure(ResourcePressure(resource_config))
        
        bacteria = [Bacterium(id="test_bact", fitness=1.0)]
        context = {'total_population': 5000}
        
        results = env.apply_selection(bacteria, context, 0)
        
        assert len(results) == 1
        result = results[0]
        
        # Should have effects from both pressures
        assert 'antimicrobial_concentration' in result.pressure_effects
        assert 'competition_factor' in result.pressure_effects
        
        # Combined effect should be stronger than individual pressures
        assert result.modified_fitness < result.original_fitness
    
    def test_get_active_pressures(self):
        """Test getting active pressures."""
        env = SelectionEnvironment()
        
        # Add enabled pressure
        config1 = PressureConfig(pressure_type=PressureType.ANTIMICROBIAL, enabled=True)
        env.add_pressure(AntimicrobialPressure(config1))
        
        # Add disabled pressure
        config2 = PressureConfig(pressure_type=PressureType.RESOURCE, enabled=False)
        env.add_pressure(ResourcePressure(config2))
        
        # Add time-limited pressure
        config3 = PressureConfig(pressure_type=PressureType.ENVIRONMENTAL, duration=10)
        env.add_pressure(EnvironmentalPressure(config3))
        
        # Test at generation 0
        active_pressures = env.get_active_pressures(0)
        assert len(active_pressures) == 2  # Antimicrobial and environmental
        
        # Test at generation 15 (after environmental pressure expires)
        active_pressures = env.get_active_pressures(15)
        assert len(active_pressures) == 1  # Only antimicrobial
    
    def test_pressure_summary(self):
        """Test pressure summary generation."""
        env = SelectionEnvironment()
        
        config = PressureConfig(pressure_type=PressureType.ANTIMICROBIAL, intensity=2.0)
        env.add_pressure(AntimicrobialPressure(config))
        
        summary = env.get_pressure_summary(0)
        
        assert summary['generation'] == 0
        assert summary['active_pressure_count'] == 1
        assert PressureType.ANTIMICROBIAL.value in summary['pressure_types']
        assert summary['pressure_intensities'][PressureType.ANTIMICROBIAL.value] == 2.0
    
    def test_pressure_interaction_logging(self):
        """Test that pressure applications are logged."""
        env = SelectionEnvironment()
        
        config = PressureConfig(pressure_type=PressureType.ANTIMICROBIAL, intensity=1.0)
        pressure = AntimicrobialPressure(config)
        env.add_pressure(pressure)
        
        bacteria = [Bacterium(id="test_bact", fitness=1.0)]
        
        # Apply selection
        env.apply_selection(bacteria, {}, 0)
        
        # Check that pressure logged the application
        assert len(pressure.history) == 1
        log_entry = pressure.history[0]
        assert log_entry['generation'] == 0
        assert log_entry['pressure_type'] == PressureType.ANTIMICROBIAL.value
        assert log_entry['bacteria_affected'] == 1


class TestPressureIntegration:
    """Integration tests for pressure system."""
    
    def test_realistic_antimicrobial_scenario(self):
        """Test realistic antimicrobial treatment scenario."""
        env = SelectionEnvironment()
        
        # Add antimicrobial pressure with pulse dosing
        config = PressureConfig(
            pressure_type=PressureType.ANTIMICROBIAL,
            intensity=4.0,  # 4x MIC for sensitive
            time_profile="pulse",
            parameters={
                'pulse_interval': 24,
                'pulse_duration': 8,
                'mic_sensitive': 1.0,
                'mic_resistant': 8.0
            }
        )
        env.add_pressure(AntimicrobialPressure(config))
        
        # Create mixed population
        bacteria = [
            Bacterium(id="sensitive_1", resistance_status=ResistanceStatus.SENSITIVE, fitness=1.0),
            Bacterium(id="sensitive_2", resistance_status=ResistanceStatus.SENSITIVE, fitness=1.0),
            Bacterium(id="resistant_1", resistance_status=ResistanceStatus.RESISTANT, fitness=0.9),
            Bacterium(id="resistant_2", resistance_status=ResistanceStatus.RESISTANT, fitness=0.9)
        ]
        
        # Test during drug administration (generation 4)
        results_during = env.apply_selection(bacteria, {}, 4)
        
        # Test during drug-free period (generation 12)
        results_free = env.apply_selection(bacteria, {}, 12)
        
        # During drug administration, resistant bacteria should survive better
        resistant_survival_during = np.mean([
            r.survival_probability for r in results_during if 'resistant' in r.bacterium_id
        ])
        sensitive_survival_during = np.mean([
            r.survival_probability for r in results_during if 'sensitive' in r.bacterium_id
        ])
        
        assert resistant_survival_during > sensitive_survival_during
        
        # During drug-free period, survival should be higher for all bacteria
        resistant_survival_free = np.mean([
            r.survival_probability for r in results_free if 'resistant' in r.bacterium_id
        ])
        sensitive_survival_free = np.mean([
            r.survival_probability for r in results_free if 'sensitive' in r.bacterium_id
        ])
        
        assert resistant_survival_free > resistant_survival_during
        assert sensitive_survival_free > sensitive_survival_during
    
    def test_multi_pressure_scenario(self):
        """Test scenario with multiple pressures acting simultaneously."""
        env = SelectionEnvironment()
        
        # Add antimicrobial pressure
        antimicrobial_config = PressureConfig(
            pressure_type=PressureType.ANTIMICROBIAL,
            intensity=2.0
        )
        env.add_pressure(AntimicrobialPressure(antimicrobial_config))
        
        # Add resource competition
        resource_config = PressureConfig(
            pressure_type=PressureType.RESOURCE,
            intensity=1.5,
            parameters={'carrying_capacity': 1000}
        )
        env.add_pressure(ResourcePressure(resource_config))
        
        # Add environmental stress
        environmental_config = PressureConfig(
            pressure_type=PressureType.ENVIRONMENTAL,
            intensity=1.2
        )
        env.add_pressure(EnvironmentalPressure(environmental_config))
        
        # Test bacteria
        bacteria = [
            Bacterium(id="sensitive", resistance_status=ResistanceStatus.SENSITIVE, fitness=1.0),
            Bacterium(id="resistant", resistance_status=ResistanceStatus.RESISTANT, fitness=0.9)
        ]
        
        context = {'total_population': 2000}  # Over carrying capacity
        
        results = env.apply_selection(bacteria, context, 0)
        
        # All bacteria should be affected by multiple pressures
        for result in results:
            assert result.modified_fitness < result.original_fitness
            assert len(result.pressure_effects) > 3  # Multiple pressure effects
        
        # Resistant bacterium should still have advantage in antimicrobial pressure
        # but may be disadvantaged by resource competition
        resistant_result = next(r for r in results if r.bacterium_id == "resistant")
        sensitive_result = next(r for r in results if r.bacterium_id == "sensitive")
        
        # The relative advantage depends on the specific combination of pressures
        # Both should have reduced fitness due to multiple stresses
        assert resistant_result.modified_fitness < 0.9
        assert sensitive_result.modified_fitness < 1.0 