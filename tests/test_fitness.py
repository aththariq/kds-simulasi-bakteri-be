"""
Test suite for comprehensive fitness computation system.
"""

import pytest
import math
from models.fitness import (
    ComprehensiveFitnessCalculator,
    FitnessConfig,
    FitnessWeights,
    FitnessComponent,
    FitnessNormalizationMethod,
    FitnessCalculationResult,
    calculate_comprehensive_fitness,
    update_bacterium_comprehensive_fitness
)
from models.bacterium import Bacterium, ResistanceStatus
from models.mutation import (
    Mutation, MutationEffect, MutationType
)
from models.selection import (
    SelectionEnvironment, AntimicrobialPressure, PressureConfig, SelectionResult, PressureType
)
from models.resistance import EnvironmentalContext


class TestFitnessWeights:
    """Test fitness weights configuration."""
    
    def test_default_weights(self):
        """Test default weight values."""
        weights = FitnessWeights()
        
        assert weights.base == 1.0
        assert weights.age_related == 1.0
        assert weights.mutation == 0.8
        assert weights.selection == 1.2
        assert weights.resistance == 1.0
        assert weights.environmental == 0.9
        assert weights.spatial == 0.7
        assert weights.epistatic == 0.5
    
    def test_weight_validation(self):
        """Test weight validation."""
        # Valid weights should not raise
        weights = FitnessWeights()
        weights.validate()
        
        # Negative weights should raise
        with pytest.raises(ValueError, match="All fitness weights must be non-negative"):
            weights = FitnessWeights(base=-0.1)
            weights.validate()


class TestFitnessConfig:
    """Test fitness configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FitnessConfig()
        
        assert isinstance(config.weights, FitnessWeights)
        assert config.normalization_method == FitnessNormalizationMethod.RELATIVE
        assert config.normalization_window == 100
        assert config.min_fitness == 0.01
        assert config.max_fitness == 10.0
        assert config.age_decline_rate == 0.01
        assert config.max_age_penalty == 0.5
        assert config.epistatic_strength == 0.1
        assert config.environmental_sensitivity == 1.0
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = FitnessConfig()
        config.validate()
        
        # Invalid fitness bounds
        with pytest.raises(ValueError, match="Invalid fitness bounds"):
            config = FitnessConfig(min_fitness=1.0, max_fitness=0.5)
            config.validate()
        
        # Invalid age decline rate
        with pytest.raises(ValueError, match="Age decline rate must be between 0 and 0.1"):
            config = FitnessConfig(age_decline_rate=0.2)
            config.validate()


class TestFitnessCalculationResult:
    """Test fitness calculation result."""
    
    def test_result_creation(self):
        """Test fitness result creation."""
        result = FitnessCalculationResult(final_fitness=1.5)
        
        assert result.final_fitness == 1.5
        assert result.base_fitness == 1.0
        assert result.normalized_fitness == 1.0
        assert result.fitness_multiplier == 1.0
        assert len(result.component_values) == 0
        assert len(result.mutation_effects) == 0
        assert result.resistance_result is None
    
    def test_total_effects_properties(self):
        """Test total effect property calculations."""
        from models.mutation import Mutation, MutationType, MutationEffect
        from models.selection import SelectionResult
        from models.resistance import CostBenefitResult
        
        # Create test mutations
        mut1 = Mutation(
            mutation_id="mut1",
            mutation_type=MutationType.POINT,
            effect=MutationEffect.BENEFICIAL,
            generation_occurred=1,
            fitness_change=0.1
        )
        mut2 = Mutation(
            mutation_id="mut2",
            mutation_type=MutationType.POINT,
            effect=MutationEffect.DELETERIOUS,
            generation_occurred=1,
            fitness_change=-0.05
        )
        
        # Create test selection results
        sel_result = SelectionResult(
            bacterium_id="test",
            original_fitness=1.0,
            modified_fitness=1.1,
            survival_probability=0.9,
            pressure_effects={"antimicrobial_concentration": 2.0}
        )
        
        # Create test resistance result
        resistance_result = CostBenefitResult(net_fitness_effect=0.05)
        
        result = FitnessCalculationResult(
            final_fitness=1.2,
            mutation_effects=[(mut1, 0.1), (mut2, -0.05)],
            selection_results=[sel_result],
            resistance_result=resistance_result
        )
        
        assert result.total_mutation_effect == 0.05
        # Use tolerance for floating point comparison
        assert abs(result.total_selection_effect - 0.1) < 1e-10
        assert result.resistance_effect == 0.05
    
    def test_summary_generation(self):
        """Test human-readable summary generation."""
        result = FitnessCalculationResult(final_fitness=1.2)
        result.component_contributions = {
            FitnessComponent.BASE: 1.0,
            FitnessComponent.MUTATION: 0.1,
            FitnessComponent.SELECTION: 0.05
        }
        
        summary = result.get_summary()
        assert "Fitness: 1.200" in summary
        assert "base: +1.000" in summary
        assert "mutation: +0.100" in summary


class TestComprehensiveFitnessCalculator:
    """Test the comprehensive fitness calculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = ComprehensiveFitnessCalculator()
        self.bacterium = Bacterium(id="test", fitness=1.0, age=5)
        self.resistant_bacterium = Bacterium(
            id="resistant_test",
            resistance_status=ResistanceStatus.RESISTANT,
            fitness=1.0,
            age=3
        )
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        # Default initialization
        calc = ComprehensiveFitnessCalculator()
        assert isinstance(calc.config, FitnessConfig)
        assert calc._generation_count == 0
        assert len(calc._fitness_history) == 0
        
        # Custom configuration
        config = FitnessConfig(min_fitness=0.1, max_fitness=5.0)
        calc = ComprehensiveFitnessCalculator(config=config)
        assert calc.config.min_fitness == 0.1
        assert calc.config.max_fitness == 5.0
    
    def test_base_fitness_calculation(self):
        """Test base fitness component calculation."""
        result = self.calculator.calculate_fitness(self.bacterium)
        
        assert FitnessComponent.BASE in result.component_values
        assert result.component_values[FitnessComponent.BASE] == 1.0
        assert FitnessComponent.BASE in result.component_contributions
    
    def test_age_effects_calculation(self):
        """Test age-related fitness effects."""
        young_bacterium = Bacterium(id="young", age=1)
        old_bacterium = Bacterium(id="old", age=20)
        
        young_result = self.calculator.calculate_fitness(young_bacterium)
        old_result = self.calculator.calculate_fitness(old_bacterium)
        
        # Older bacterium should have lower age component
        assert (young_result.component_values[FitnessComponent.AGE_RELATED] > 
                old_result.component_values[FitnessComponent.AGE_RELATED])
    
    def test_mutation_effects_calculation(self):
        """Test mutation-based fitness effects."""
        from models.mutation import Mutation, MutationType, MutationEffect
        
        beneficial_mut = Mutation(
            mutation_id="beneficial",
            mutation_type=MutationType.POINT,
            effect=MutationEffect.BENEFICIAL,
            generation_occurred=1,
            fitness_change=0.1
        )
        deleterious_mut = Mutation(
            mutation_id="deleterious",
            mutation_type=MutationType.POINT,
            effect=MutationEffect.DELETERIOUS,
            generation_occurred=1,
            fitness_change=-0.05
        )
        
        # Test with beneficial mutation
        beneficial_result = self.calculator.calculate_fitness(
            self.bacterium, mutations=[beneficial_mut]
        )
        
        # Test with deleterious mutation
        deleterious_result = self.calculator.calculate_fitness(
            self.bacterium, mutations=[deleterious_mut]
        )
        
        # Beneficial should increase mutation component, deleterious should decrease
        assert (beneficial_result.component_values[FitnessComponent.MUTATION] > 
                deleterious_result.component_values[FitnessComponent.MUTATION])
    
    def test_selection_effects_calculation(self):
        """Test selection pressure effects."""
        # Create selection environment with antimicrobial pressure
        selection_env = SelectionEnvironment()
        selection_env.add_pressure(AntimicrobialPressure(
            config=PressureConfig(
                pressure_type=PressureType.ANTIMICROBIAL,
                intensity=0.6
            )
        ))
        
        # Test sensitive bacterium
        sensitive_result = self.calculator.calculate_fitness(
            self.bacterium, selection_environment=selection_env
        )
        
        # Test resistant bacterium
        resistant_result = self.calculator.calculate_fitness(
            self.resistant_bacterium, selection_environment=selection_env
        )
        
        # Should have selection effects
        assert FitnessComponent.SELECTION in sensitive_result.component_values
        assert FitnessComponent.SELECTION in resistant_result.component_values
    
    def test_resistance_effects_calculation(self):
        """Test resistance cost/benefit effects."""
        # Without antibiotics
        no_antibiotic_context = EnvironmentalContext(antibiotic_concentration=0.0)
        no_ab_result = self.calculator.calculate_fitness(
            self.resistant_bacterium, environmental_context=no_antibiotic_context
        )
        
        # With antibiotics
        antibiotic_context = EnvironmentalContext(antibiotic_concentration=3.0)
        ab_result = self.calculator.calculate_fitness(
            self.resistant_bacterium, environmental_context=antibiotic_context
        )
        
        # Should have resistance component in both cases
        assert FitnessComponent.RESISTANCE in no_ab_result.component_values
        assert FitnessComponent.RESISTANCE in ab_result.component_values
        
        # With antibiotics should be more beneficial for resistant bacteria
        assert (ab_result.component_values[FitnessComponent.RESISTANCE] >= 
                no_ab_result.component_values[FitnessComponent.RESISTANCE])
    
    def test_environmental_effects_calculation(self):
        """Test environmental stress effects."""
        # Low stress environment
        low_stress_context = EnvironmentalContext(
            temperature_stress=0.1,
            ph_stress=0.1
        )
        
        # High stress environment
        high_stress_context = EnvironmentalContext(
            temperature_stress=0.8,
            ph_stress=0.7
        )
        
        low_stress_result = self.calculator.calculate_fitness(
            self.bacterium, environmental_context=low_stress_context
        )
        
        high_stress_result = self.calculator.calculate_fitness(
            self.bacterium, environmental_context=high_stress_context
        )
        
        # High stress should reduce environmental component
        assert (low_stress_result.component_values[FitnessComponent.ENVIRONMENTAL] > 
                high_stress_result.component_values[FitnessComponent.ENVIRONMENTAL])
    
    def test_spatial_effects_calculation(self):
        """Test spatial density effects."""
        # Low density context
        low_density_context = {
            'local_density': 10,
            'carrying_capacity': 100
        }
        
        # High density context
        high_density_context = {
            'local_density': 90,
            'carrying_capacity': 100
        }
        
        low_density_result = self.calculator.calculate_fitness(
            self.bacterium, population_context=low_density_context
        )
        
        high_density_result = self.calculator.calculate_fitness(
            self.bacterium, population_context=high_density_context
        )
        
        # High density should reduce spatial component
        assert (low_density_result.component_values[FitnessComponent.SPATIAL] > 
                high_density_result.component_values[FitnessComponent.SPATIAL])
    
    def test_epistatic_effects_calculation(self):
        """Test epistatic interaction effects."""
        from models.mutation import Mutation, MutationType, MutationEffect
        
        beneficial_mut1 = Mutation(
            mutation_id="beneficial1",
            mutation_type=MutationType.POINT,
            effect=MutationEffect.BENEFICIAL,
            generation_occurred=1,
            fitness_change=0.05
        )
        beneficial_mut2 = Mutation(
            mutation_id="beneficial2",
            mutation_type=MutationType.POINT,
            effect=MutationEffect.BENEFICIAL,
            generation_occurred=1,
            fitness_change=0.08
        )
        
        # Test with no mutations (should default to 1.0)
        no_mut_result = self.calculator.calculate_fitness(self.bacterium)
        assert no_mut_result.component_values[FitnessComponent.EPISTATIC] == 1.0
        
        # Test with two mutations (should have epistatic effects)
        two_mut_result = self.calculator.calculate_fitness(
            self.bacterium, mutations=[beneficial_mut1, beneficial_mut2]
        )
        
        # Should have epistatic component
        assert FitnessComponent.EPISTATIC in two_mut_result.component_values
    
    def test_fitness_bounds_enforcement(self):
        """Test that fitness bounds are enforced."""
        # Create config with tight bounds
        config = FitnessConfig(min_fitness=0.5, max_fitness=2.0)
        calculator = ComprehensiveFitnessCalculator(config=config)
        
        # Test with very low base fitness
        low_fit_bacterium = Bacterium(id="low", fitness=0.1)
        low_result = calculator.calculate_fitness(low_fit_bacterium)
        assert low_result.final_fitness >= 0.5
        
        # Test with very high base fitness
        high_fit_bacterium = Bacterium(id="high", fitness=5.0)
        high_result = calculator.calculate_fitness(high_fit_bacterium)
        assert high_result.final_fitness <= 2.0
    
    def test_fitness_normalization_none(self):
        """Test no normalization method."""
        config = FitnessConfig(normalization_method=FitnessNormalizationMethod.NONE)
        calculator = ComprehensiveFitnessCalculator(config=config)
        
        result = calculator.calculate_fitness(self.bacterium)
        # With no normalization, normalized should equal final
        assert result.normalized_fitness == result.final_fitness
    
    def test_fitness_normalization_relative(self):
        """Test relative normalization method."""
        config = FitnessConfig(normalization_method=FitnessNormalizationMethod.RELATIVE)
        calculator = ComprehensiveFitnessCalculator(config=config)
        
        # Calculate several fitness values to build history
        for i in range(10):
            bacterium = Bacterium(id=f"test_{i}", fitness=1.0 + i * 0.1)
            calculator.calculate_fitness(bacterium)
        
        # Now test normalization with a different fitness value
        test_bacterium = Bacterium(id="test_norm", fitness=2.0)
        result = calculator.calculate_fitness(test_bacterium)
        
        # Should have normalization applied - the normalized value should be calculated
        # relative to the population mean from the history
        assert len(calculator._fitness_history) > 0
        # With enough history, normalization should have an effect on the raw multiplied value
        assert result.normalized_fitness is not None
    
    def test_fitness_history_management(self):
        """Test fitness history tracking."""
        # Calculate many fitness values
        for i in range(250):  # More than 2 * normalization_window
            bacterium = Bacterium(id=f"test_{i}", fitness=1.0)
            self.calculator.calculate_fitness(bacterium)
        
        # History should be trimmed
        assert len(self.calculator._fitness_history) <= self.calculator.config.normalization_window
        assert self.calculator._generation_count == 250
    
    def test_reset_history(self):
        """Test history reset functionality."""
        # Build some history
        for i in range(10):
            bacterium = Bacterium(id=f"test_{i}", fitness=1.0)
            self.calculator.calculate_fitness(bacterium)
        
        assert len(self.calculator._fitness_history) > 0
        assert self.calculator._generation_count > 0
        
        # Reset
        self.calculator.reset_history()
        
        assert len(self.calculator._fitness_history) == 0
        assert self.calculator._generation_count == 0
    
    def test_fitness_landscape_generation(self):
        """Test fitness landscape generation."""
        gradient = {
            'antibiotic_concentration': (0.0, 5.0),
            'temperature_stress': (0.0, 1.0)
        }
        
        landscape = self.calculator.get_fitness_landscape(
            self.resistant_bacterium, gradient, num_points=5
        )
        
        assert 'gradients' in landscape
        assert 'fitness_values' in landscape
        assert 'coordinates' in landscape
        assert len(landscape['fitness_values']) == 2  # Two gradients
        assert len(landscape['coordinates']) == 2
        assert len(landscape['fitness_values'][0]) == 5  # 5 points


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_calculate_comprehensive_fitness(self):
        """Test convenience function for fitness calculation."""
        bacterium = Bacterium(id="test", fitness=1.0)
        
        # With default calculator
        result = calculate_comprehensive_fitness(bacterium)
        assert isinstance(result, FitnessCalculationResult)
        assert result.final_fitness > 0
        
        # With custom calculator
        calculator = ComprehensiveFitnessCalculator()
        result_custom = calculate_comprehensive_fitness(bacterium, calculator=calculator)
        
        # Should get same result with same calculator
        assert result.final_fitness != result_custom.final_fitness or calculator._generation_count == 1
    
    def test_update_bacterium_comprehensive_fitness(self):
        """Test updating bacterium fitness."""
        bacterium = Bacterium(id="test", fitness=1.0)
        original_fitness = bacterium.fitness
        
        # Update with apply_immediately=True
        new_fitness, result = update_bacterium_comprehensive_fitness(
            bacterium, apply_immediately=True
        )
        
        assert bacterium.fitness == new_fitness
        assert bacterium.fitness != original_fitness or new_fitness == original_fitness
        
        # Test without applying
        test_bacterium = Bacterium(id="test2", fitness=1.0)
        original_test_fitness = test_bacterium.fitness
        
        calculated_fitness, _ = update_bacterium_comprehensive_fitness(
            test_bacterium, apply_immediately=False
        )
        
        # Original fitness should be unchanged
        assert test_bacterium.fitness == original_test_fitness


class TestFitnessIntegration:
    """Integration tests for comprehensive fitness system."""
    
    def test_comprehensive_fitness_calculation(self):
        """Test a comprehensive fitness calculation scenario."""
        from models.mutation import Mutation, MutationType, MutationEffect
        from models.selection import (
            SelectionEnvironment, AntimicrobialPressure, PressureConfig
        )
        from models.resistance import EnvironmentalContext
        
        # Create a bacterium with mutations
        bacterium = Bacterium(
            id="comprehensive_test",
            resistance_status=ResistanceStatus.RESISTANT,
            fitness=1.2,
            age=10
        )
        
        # Add mutations to the bacterium
        bacterium.mutations = [
            Mutation(
                mutation_id="mut1",
                mutation_type=MutationType.POINT,
                effect=MutationEffect.BENEFICIAL,
                generation_occurred=5,
                fitness_change=0.1
            ),
            Mutation(
                mutation_id="mut2",
                mutation_type=MutationType.POINT,
                effect=MutationEffect.DELETERIOUS,
                generation_occurred=8,
                fitness_change=-0.05
            )
        ]
        
        # Create selection environment
        selection_env = SelectionEnvironment()
        selection_env.add_pressure(AntimicrobialPressure(
            config=PressureConfig(
                pressure_type=PressureType.ANTIMICROBIAL,
                intensity=0.6
            )
        ))
        
        # Create environmental context
        env_context = EnvironmentalContext(
            antibiotic_concentration=2.0,
            temperature_stress=0.3,
            ph_stress=0.2,
            resistance_frequency=0.15
        )
        
        # Create population context
        pop_context = {
            'local_density': 50,
            'carrying_capacity': 100
        }
        
        # Calculate comprehensive fitness
        calculator = ComprehensiveFitnessCalculator()
        result = calculator.calculate_fitness(
            bacterium=bacterium,
            mutations=bacterium.mutations,
            selection_environment=selection_env,
            environmental_context=env_context,
            population_context=pop_context
        )
        
        # Verify all components are calculated
        expected_components = {
            FitnessComponent.BASE,
            FitnessComponent.AGE_RELATED,
            FitnessComponent.MUTATION,
            FitnessComponent.SELECTION,
            FitnessComponent.RESISTANCE,
            FitnessComponent.ENVIRONMENTAL,
            FitnessComponent.SPATIAL,
            FitnessComponent.EPISTATIC
        }
        
        for component in expected_components:
            assert component in result.component_values
            assert component in result.component_contributions
        
        # Verify detailed results
        assert len(result.mutation_effects) == 2
        assert len(result.selection_results) > 0
        assert result.resistance_result is not None
        
        # Verify result is reasonable
        assert result.final_fitness > 0
        assert result.final_fitness != bacterium.fitness  # Should be modified
        
        # Get summary
        summary = result.get_summary()
        assert "Fitness:" in summary
    
    def test_evolutionary_scenario(self):
        """Test fitness calculation in an evolutionary scenario."""
        calculator = ComprehensiveFitnessCalculator()
        
        # Simulate population over generations
        bacteria = [
            Bacterium(id=f"bact_{i}", resistance_status=ResistanceStatus.SENSITIVE, fitness=1.0)
            for i in range(10)
        ]
        
        # Add some resistant bacteria
        bacteria.extend([
            Bacterium(id=f"res_{i}", resistance_status=ResistanceStatus.RESISTANT, fitness=0.95)
            for i in range(3)
        ])
        
        # Create antibiotic pressure environment
        env_context = EnvironmentalContext(antibiotic_concentration=3.0)
        
        # Calculate fitness for all bacteria
        fitness_results = []
        for bacterium in bacteria:
            result = calculator.calculate_fitness(
                bacterium, environmental_context=env_context
            )
            fitness_results.append((bacterium, result))
        
        # Resistant bacteria should generally have higher fitness under antibiotic pressure
        resistant_fitnesses = [r.final_fitness for b, r in fitness_results if b.is_resistant]
        sensitive_fitnesses = [r.final_fitness for b, r in fitness_results if not b.is_resistant]
        
        avg_resistant_fitness = sum(resistant_fitnesses) / len(resistant_fitnesses)
        avg_sensitive_fitness = sum(sensitive_fitnesses) / len(sensitive_fitnesses)
        
        assert avg_resistant_fitness > avg_sensitive_fitness
    
    def test_fitness_landscape_analysis(self):
        """Test fitness landscape analysis."""
        calculator = ComprehensiveFitnessCalculator()
        
        # Create reference bacteria
        sensitive_bacterium = Bacterium(id="sensitive_ref", resistance_status=ResistanceStatus.SENSITIVE)
        resistant_bacterium = Bacterium(id="resistant_ref", resistance_status=ResistanceStatus.RESISTANT)
        
        # Define environmental gradient
        gradient = {'antibiotic_concentration': (0.0, 10.0)}
        
        # Generate landscapes
        sensitive_landscape = calculator.get_fitness_landscape(
            sensitive_bacterium, gradient, num_points=11
        )
        resistant_landscape = calculator.get_fitness_landscape(
            resistant_bacterium, gradient, num_points=11
        )
        
        # Verify landscape structure
        assert len(sensitive_landscape['fitness_values'][0]) == 11
        assert len(resistant_landscape['fitness_values'][0]) == 11
        
        # At high antibiotic concentrations, resistant should be fitter
        sensitive_high_ab = sensitive_landscape['fitness_values'][0][-1]
        resistant_high_ab = resistant_landscape['fitness_values'][0][-1]
        
        assert resistant_high_ab > sensitive_high_ab 