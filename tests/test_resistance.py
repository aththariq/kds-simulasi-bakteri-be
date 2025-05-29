"""
Test suite for resistance cost/benefit calculation system.
"""

import pytest
import math
from models.resistance import (
    ResistanceCostBenefitCalculator,
    ResistanceCostConfig,
    ResistanceBenefitConfig,
    EnvironmentalContext,
    CostBenefitResult,
    CostType,
    BenefitType,
    calculate_resistance_fitness_effect,
    simple_resistance_cost_benefit
)
from models.bacterium import Bacterium, ResistanceStatus


class TestResistanceCostConfig:
    """Test resistance cost configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ResistanceCostConfig()
        
        assert config.base_metabolic_cost == 0.05
        assert config.base_fitness_cost == 0.03
        assert config.base_growth_cost == 0.02
        assert config.nutrient_stress_multiplier == 1.5
        assert config.temperature_stress_multiplier == 1.3
        assert config.ph_stress_multiplier == 1.2
        assert config.frequency_dependence_strength == 0.1
        assert config.max_cost_reduction == 0.3
        assert config.cost_reduction_rate == 0.01
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration should not raise
        config = ResistanceCostConfig()
        config.validate()
        
        # Invalid base cost
        with pytest.raises(ValueError, match="Base costs must be between 0.0 and 1.0"):
            config = ResistanceCostConfig(base_metabolic_cost=-0.1)
            config.validate()
        
        with pytest.raises(ValueError, match="Base costs must be between 0.0 and 1.0"):
            config = ResistanceCostConfig(base_fitness_cost=1.5)
            config.validate()
        
        # Invalid cost reduction
        with pytest.raises(ValueError, match="Max cost reduction must be between 0.0 and 1.0"):
            config = ResistanceCostConfig(max_cost_reduction=1.5)
            config.validate()


class TestResistanceBenefitConfig:
    """Test resistance benefit configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ResistanceBenefitConfig()
        
        assert config.base_survival_benefit == 0.8
        assert config.max_survival_benefit == 0.95
        assert config.mic_fold_change == 8.0
        assert config.hill_coefficient == 2.0
        assert config.resource_competition_benefit == 0.1
        assert config.spatial_advantage == 0.05
        assert config.minority_advantage == 0.05
        assert config.frequency_threshold == 0.1
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid configuration should not raise
        config = ResistanceBenefitConfig()
        config.validate()
        
        # Invalid survival benefit
        with pytest.raises(ValueError, match="Base survival benefit must be between 0.0 and 1.0"):
            config = ResistanceBenefitConfig(base_survival_benefit=-0.1)
            config.validate()
        
        with pytest.raises(ValueError, match="Max survival benefit must be between 0.0 and 1.0"):
            config = ResistanceBenefitConfig(max_survival_benefit=1.5)
            config.validate()


class TestEnvironmentalContext:
    """Test environmental context."""
    
    def test_default_context(self):
        """Test default environmental context."""
        context = EnvironmentalContext()
        
        assert context.antibiotic_concentration == 0.0
        assert context.antibiotic_type == "general"
        assert context.nutrient_availability == 1.0
        assert context.resource_competition == 0.0
        assert context.temperature_stress == 0.0
        assert context.ph_stress == 0.0
        assert context.osmotic_stress == 0.0
        assert context.resistance_frequency == 0.01
        assert context.population_density == 1.0
        assert context.generation == 0
    
    def test_context_validation(self):
        """Test environmental context validation."""
        # Valid context should not raise
        context = EnvironmentalContext()
        context.validate()
        
        # Invalid antibiotic concentration
        with pytest.raises(ValueError, match="Antibiotic concentration must be non-negative"):
            context = EnvironmentalContext(antibiotic_concentration=-1.0)
            context.validate()
        
        # Invalid bounded parameters
        with pytest.raises(ValueError, match="Environmental parameters must be between 0.0 and 1.0"):
            context = EnvironmentalContext(nutrient_availability=-0.1)
            context.validate()
        
        with pytest.raises(ValueError, match="Environmental parameters must be between 0.0 and 1.0"):
            context = EnvironmentalContext(resistance_frequency=1.5)
            context.validate()


class TestCostBenefitResult:
    """Test cost/benefit result calculations."""
    
    def test_result_creation(self):
        """Test cost/benefit result creation."""
        result = CostBenefitResult(
            total_cost=0.1,
            total_benefit=0.2,
            net_fitness_effect=0.1,
            cost_breakdown={CostType.METABOLIC: 0.05, CostType.FITNESS: 0.05},
            benefit_breakdown={BenefitType.SURVIVAL: 0.2}
        )
        
        assert result.total_cost == 0.1
        assert result.total_benefit == 0.2
        assert result.net_fitness_effect == 0.1
        assert result.fitness_multiplier == 1.1
        assert result.is_beneficial is True
    
    def test_fitness_multiplier_properties(self):
        """Test fitness multiplier calculations."""
        # Beneficial effect
        result = CostBenefitResult(net_fitness_effect=0.2)
        assert result.fitness_multiplier == 1.2
        assert result.is_beneficial is True
        
        # Costly effect
        result = CostBenefitResult(net_fitness_effect=-0.1)
        assert result.fitness_multiplier == 0.9
        assert result.is_beneficial is False
        
        # Very costly effect (should be capped at minimum)
        result = CostBenefitResult(net_fitness_effect=-2.0)
        assert result.fitness_multiplier == 0.01  # Minimum fitness
        assert result.is_beneficial is False
    
    def test_summary_properties(self):
        """Test human-readable summary properties."""
        result = CostBenefitResult(
            total_cost=0.15,
            total_benefit=0.25,
            cost_breakdown={
                CostType.METABOLIC: 0.05,
                CostType.FITNESS: 0.1
            },
            benefit_breakdown={
                BenefitType.SURVIVAL: 0.2,
                BenefitType.COMPETITIVE: 0.05
            }
        )
        
        cost_summary = result.cost_summary
        assert "Total: 15.0%" in cost_summary
        assert "metabolic: 5.0%" in cost_summary
        assert "fitness: 10.0%" in cost_summary
        
        benefit_summary = result.benefit_summary
        assert "Total: 25.0%" in benefit_summary
        assert "survival: 20.0%" in benefit_summary
        assert "competitive: 5.0%" in benefit_summary


class TestResistanceCostBenefitCalculator:
    """Test the main resistance cost/benefit calculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = ResistanceCostBenefitCalculator()
        self.resistant_bacterium = Bacterium(
            id="test_resistant",
            resistance_status=ResistanceStatus.RESISTANT
        )
        self.sensitive_bacterium = Bacterium(
            id="test_sensitive",
            resistance_status=ResistanceStatus.SENSITIVE
        )
        self.base_context = EnvironmentalContext()
    
    def test_calculator_initialization(self):
        """Test calculator initialization."""
        # Default initialization
        calc = ResistanceCostBenefitCalculator()
        assert isinstance(calc.cost_config, ResistanceCostConfig)
        assert isinstance(calc.benefit_config, ResistanceBenefitConfig)
        
        # Custom configuration
        cost_config = ResistanceCostConfig(base_metabolic_cost=0.1)
        benefit_config = ResistanceBenefitConfig(base_survival_benefit=0.9)
        calc = ResistanceCostBenefitCalculator(cost_config, benefit_config)
        assert calc.cost_config.base_metabolic_cost == 0.1
        assert calc.benefit_config.base_survival_benefit == 0.9
    
    def test_cost_calculation_sensitive_bacterium(self):
        """Test cost calculation for sensitive bacteria."""
        costs = self.calculator.calculate_costs(self.sensitive_bacterium, self.base_context)
        
        # Sensitive bacteria should have zero costs
        for cost_type in CostType:
            assert costs[cost_type] == 0.0
    
    def test_cost_calculation_resistant_bacterium_no_stress(self):
        """Test cost calculation for resistant bacteria without environmental stress."""
        costs = self.calculator.calculate_costs(self.resistant_bacterium, self.base_context)
        
        # Should have base costs
        assert costs[CostType.METABOLIC] == self.calculator.cost_config.base_metabolic_cost
        assert abs(costs[CostType.FITNESS] - self.calculator.cost_config.base_fitness_cost) < 0.001  # Use tolerance for floating point
        assert costs[CostType.GROWTH] == self.calculator.cost_config.base_growth_cost
        assert costs[CostType.MAINTENANCE] > 0
    
    def test_cost_calculation_with_environmental_stress(self):
        """Test cost calculation with environmental stress."""
        stressed_context = EnvironmentalContext(
            temperature_stress=0.8,
            ph_stress=0.6,
            nutrient_availability=0.3
        )
        
        costs = self.calculator.calculate_costs(self.resistant_bacterium, stressed_context)
        base_costs = self.calculator.calculate_costs(self.resistant_bacterium, self.base_context)
        
        # Costs should be higher under stress
        assert costs[CostType.METABOLIC] > base_costs[CostType.METABOLIC]
    
    def test_cost_calculation_with_resource_competition(self):
        """Test cost calculation with resource competition."""
        competitive_context = EnvironmentalContext(resource_competition=0.8)
        
        costs = self.calculator.calculate_costs(self.resistant_bacterium, competitive_context)
        base_costs = self.calculator.calculate_costs(self.resistant_bacterium, self.base_context)
        
        # Growth costs should be higher with competition
        assert costs[CostType.GROWTH] > base_costs[CostType.GROWTH]
    
    def test_cost_calculation_frequency_dependence(self):
        """Test frequency-dependent cost calculation."""
        rare_context = EnvironmentalContext(resistance_frequency=0.05)
        common_context = EnvironmentalContext(resistance_frequency=0.8)
        
        rare_costs = self.calculator.calculate_costs(self.resistant_bacterium, rare_context)
        common_costs = self.calculator.calculate_costs(self.resistant_bacterium, common_context)
        
        # Fitness costs should be lower when resistance is rare
        assert rare_costs[CostType.FITNESS] < common_costs[CostType.FITNESS]
    
    def test_cost_evolution_over_generations(self):
        """Test cost reduction over generations."""
        early_context = EnvironmentalContext(generation=1)
        late_context = EnvironmentalContext(generation=20)
        
        early_costs = self.calculator.calculate_costs(self.resistant_bacterium, early_context)
        late_costs = self.calculator.calculate_costs(self.resistant_bacterium, late_context)
        
        # Maintenance costs should decrease over generations
        assert late_costs[CostType.MAINTENANCE] < early_costs[CostType.MAINTENANCE]
    
    def test_benefit_calculation_sensitive_bacterium(self):
        """Test benefit calculation for sensitive bacteria."""
        benefits = self.calculator.calculate_benefits(self.sensitive_bacterium, self.base_context)
        
        # Sensitive bacteria should have zero benefits
        for benefit_type in BenefitType:
            assert benefits[benefit_type] == 0.0
    
    def test_benefit_calculation_no_antibiotics(self):
        """Test benefit calculation without antibiotics."""
        benefits = self.calculator.calculate_benefits(self.resistant_bacterium, self.base_context)
        
        # Should have no survival benefit without antibiotics
        assert benefits[BenefitType.SURVIVAL] == 0.0
    
    def test_benefit_calculation_with_antibiotics(self):
        """Test benefit calculation with antibiotics."""
        antibiotic_context = EnvironmentalContext(antibiotic_concentration=2.0)
        
        benefits = self.calculator.calculate_benefits(self.resistant_bacterium, antibiotic_context)
        
        # Should have survival benefit with antibiotics
        assert benefits[BenefitType.SURVIVAL] > 0
    
    def test_benefit_calculation_dose_response(self):
        """Test dose-response relationship in benefit calculation."""
        low_dose_context = EnvironmentalContext(antibiotic_concentration=0.5)  # Lower dose
        high_dose_context = EnvironmentalContext(antibiotic_concentration=8.0)  # Much higher dose
        
        low_benefits = self.calculator.calculate_benefits(self.resistant_bacterium, low_dose_context)
        high_benefits = self.calculator.calculate_benefits(self.resistant_bacterium, high_dose_context)
        
        # Higher antibiotic concentration should provide higher benefit
        assert high_benefits[BenefitType.SURVIVAL] > low_benefits[BenefitType.SURVIVAL]
    
    def test_benefit_calculation_resource_competition(self):
        """Test competitive benefit calculation."""
        competitive_context = EnvironmentalContext(resource_competition=0.8)
        
        benefits = self.calculator.calculate_benefits(self.resistant_bacterium, competitive_context)
        
        # Should have competitive benefit under resource competition
        assert benefits[BenefitType.COMPETITIVE] > 0
    
    def test_benefit_calculation_frequency_dependence(self):
        """Test frequency-dependent benefit calculation."""
        # Need antibiotics or resource competition for frequency-dependent benefits to apply
        rare_context = EnvironmentalContext(
            resistance_frequency=0.05,  # Below threshold
            antibiotic_concentration=1.0  # Add antibiotic pressure
        )
        common_context = EnvironmentalContext(
            resistance_frequency=0.5,   # Above threshold
            antibiotic_concentration=1.0  # Same antibiotic pressure
        )
        
        rare_benefits = self.calculator.calculate_benefits(self.resistant_bacterium, rare_context)
        common_benefits = self.calculator.calculate_benefits(self.resistant_bacterium, common_context)
        
        # Should have higher frequency-dependent benefit when rare (with antibiotic pressure)
        assert rare_benefits[BenefitType.FREQUENCY_DEPENDENT] > common_benefits[BenefitType.FREQUENCY_DEPENDENT]
    
    def test_net_effect_calculation(self):
        """Test net fitness effect calculation."""
        # No antibiotics - should be mostly costly but might have small frequency benefit
        no_antibiotic_result = self.calculator.calculate_net_effect(
            self.resistant_bacterium, self.base_context
        )
        # With very low resistance frequency (1%), there might be a small net benefit
        # but it should be small
        assert abs(no_antibiotic_result.net_fitness_effect) < 0.1  # Should be close to neutral
        
        # High antibiotics - should be beneficial
        high_antibiotic_context = EnvironmentalContext(antibiotic_concentration=5.0)
        high_antibiotic_result = self.calculator.calculate_net_effect(
            self.resistant_bacterium, high_antibiotic_context
        )
        assert high_antibiotic_result.net_fitness_effect > 0.3  # Should be clearly beneficial
        assert high_antibiotic_result.is_beneficial
    
    def test_update_bacterium_fitness(self):
        """Test updating bacterium fitness."""
        original_fitness = self.resistant_bacterium.fitness
        antibiotic_context = EnvironmentalContext(antibiotic_concentration=3.0)
        
        new_fitness, result = self.calculator.update_bacterium_fitness(
            self.resistant_bacterium, antibiotic_context, apply_immediately=True
        )
        
        # Fitness should be updated
        assert self.resistant_bacterium.fitness == new_fitness
        assert new_fitness != original_fitness
        
        # Test without applying
        test_bacterium = Bacterium(id="test", resistance_status=ResistanceStatus.RESISTANT)
        original_test_fitness = test_bacterium.fitness
        
        calculated_fitness, _ = self.calculator.update_bacterium_fitness(
            test_bacterium, antibiotic_context, apply_immediately=False
        )
        
        # Original fitness should be unchanged
        assert test_bacterium.fitness == original_test_fitness
        assert calculated_fitness != original_test_fitness
    
    def test_fitness_landscape_generation(self):
        """Test fitness landscape generation."""
        landscape = self.calculator.get_fitness_landscape(
            self.base_context,
            antibiotic_range=(0.0, 10.0),
            num_points=11
        )
        
        assert 'concentrations' in landscape
        assert 'resistant_fitness' in landscape
        assert 'sensitive_fitness' in landscape
        assert 'crossover_point' in landscape
        
        assert len(landscape['concentrations']) == 11
        assert len(landscape['resistant_fitness']) == 11
        assert len(landscape['sensitive_fitness']) == 11
        
        # At high antibiotic concentrations, resistant should be fitter
        assert landscape['resistant_fitness'][-1] > landscape['sensitive_fitness'][-1]


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_calculate_resistance_fitness_effect(self):
        """Test convenience function for fitness effect calculation."""
        resistant_bacterium = Bacterium(id="test", resistance_status=ResistanceStatus.RESISTANT)
        context = EnvironmentalContext(antibiotic_concentration=2.0)
        
        # With default calculator
        fitness_effect = calculate_resistance_fitness_effect(resistant_bacterium, context)
        assert isinstance(fitness_effect, float)
        assert fitness_effect > 0
        
        # With custom calculator
        calculator = ResistanceCostBenefitCalculator()
        fitness_effect_custom = calculate_resistance_fitness_effect(
            resistant_bacterium, context, calculator
        )
        assert fitness_effect == fitness_effect_custom  # Should be same with default
    
    def test_simple_resistance_cost_benefit(self):
        """Test simple cost/benefit function."""
        # Sensitive bacterium without antibiotics
        fitness = simple_resistance_cost_benefit(False, 0.0)
        assert fitness == 1.0
        
        # Sensitive bacterium with antibiotics
        fitness = simple_resistance_cost_benefit(False, 2.0)
        assert fitness < 1.0  # Should suffer
        
        # Resistant bacterium without antibiotics
        fitness = simple_resistance_cost_benefit(True, 0.0)
        assert fitness < 1.0  # Should pay cost
        
        # Resistant bacterium with antibiotics
        fitness = simple_resistance_cost_benefit(True, 2.0)
        assert fitness > 0.9  # Should benefit
        
        # Rare resistant bacterium (reduced cost)
        fitness_rare = simple_resistance_cost_benefit(True, 0.0, resistance_frequency=0.01)
        fitness_common = simple_resistance_cost_benefit(True, 0.0, resistance_frequency=0.5)
        assert fitness_rare > fitness_common  # Rare should have lower cost


class TestResistanceIntegration:
    """Integration tests for resistance cost/benefit system."""
    
    def test_realistic_antibiotic_treatment_scenario(self):
        """Test realistic antibiotic treatment scenario."""
        calculator = ResistanceCostBenefitCalculator()
        
        # Before treatment
        pre_treatment_context = EnvironmentalContext(
            antibiotic_concentration=0.0,
            resistance_frequency=0.2  # 20% resistance (higher baseline)
        )
        
        # During treatment
        treatment_context = EnvironmentalContext(
            antibiotic_concentration=4.0,  # High dose
            resistance_frequency=0.2
        )
        
        # After treatment (resistance increased)
        post_treatment_context = EnvironmentalContext(
            antibiotic_concentration=0.0,
            resistance_frequency=0.6  # 60% resistance after selection
        )
        
        resistant_bact = Bacterium(id="resistant", resistance_status=ResistanceStatus.RESISTANT)
        sensitive_bact = Bacterium(id="sensitive", resistance_status=ResistanceStatus.SENSITIVE)
        
        # Pre-treatment: resistance should be costly (no frequency advantage at 20%)
        pre_resistant_result = calculator.calculate_net_effect(resistant_bact, pre_treatment_context)
        pre_sensitive_result = calculator.calculate_net_effect(sensitive_bact, pre_treatment_context)
        
        assert pre_resistant_result.net_fitness_effect < 0  # Costly
        assert pre_sensitive_result.net_fitness_effect == 0  # No effect
        
        # During treatment: resistance should be beneficial
        treat_resistant_result = calculator.calculate_net_effect(resistant_bact, treatment_context)
        treat_sensitive_result = calculator.calculate_net_effect(sensitive_bact, treatment_context)
        
        assert treat_resistant_result.net_fitness_effect > 0  # Beneficial
        assert treat_sensitive_result.net_fitness_effect == 0  # Still no cost/benefit for sensitive
        
        # Post-treatment: resistance should be more costly (higher frequency)
        post_resistant_result = calculator.calculate_net_effect(resistant_bact, post_treatment_context)
        
        assert post_resistant_result.net_fitness_effect < pre_resistant_result.net_fitness_effect
        
    def test_environmental_stress_interaction(self):
        """Test interaction between resistance costs and environmental stress."""
        calculator = ResistanceCostBenefitCalculator()
        resistant_bact = Bacterium(id="resistant", resistance_status=ResistanceStatus.RESISTANT)
        
        # Optimal conditions
        optimal_context = EnvironmentalContext()
        
        # Stressed conditions
        stressed_context = EnvironmentalContext(
            temperature_stress=0.7,
            ph_stress=0.5,
            nutrient_availability=0.3
        )
        
        optimal_result = calculator.calculate_net_effect(resistant_bact, optimal_context)
        stressed_result = calculator.calculate_net_effect(resistant_bact, stressed_context)
        
        # Resistance should be more costly under stress
        assert stressed_result.total_cost > optimal_result.total_cost
        assert stressed_result.net_fitness_effect < optimal_result.net_fitness_effect
    
    def test_evolutionary_cost_reduction(self):
        """Test cost reduction over evolutionary time."""
        calculator = ResistanceCostBenefitCalculator()
        resistant_bact = Bacterium(id="resistant", resistance_status=ResistanceStatus.RESISTANT)
        
        # Early generation
        early_context = EnvironmentalContext(generation=1)
        
        # Late generation (resistance has evolved efficiency)
        late_context = EnvironmentalContext(generation=30)
        
        early_result = calculator.calculate_net_effect(resistant_bact, early_context)
        late_result = calculator.calculate_net_effect(resistant_bact, late_context)
        
        # Costs should decrease over time
        assert late_result.total_cost < early_result.total_cost
        assert late_result.net_fitness_effect > early_result.net_fitness_effect 