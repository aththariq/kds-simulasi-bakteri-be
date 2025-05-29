"""
Comprehensive Fitness Value Computation System.

This module provides a unified framework for calculating fitness values that integrates
mutation effects, selection pressures, and resistance costs/benefits. The system enables
dynamic fitness landscape changes and supports complex evolutionary scenarios.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np
from .bacterium import Bacterium
from .mutation import Mutation, MutationEffect
from .selection import SelectionEnvironment, SelectionResult
from .resistance import (
    ResistanceCostBenefitCalculator, 
    EnvironmentalContext, 
    CostBenefitResult
)


class FitnessComponent(Enum):
    """Types of fitness components."""
    BASE = "base"                    # Base genetic fitness
    AGE_RELATED = "age_related"      # Age-related fitness decline
    MUTATION = "mutation"            # Mutation effects
    SELECTION = "selection"          # Selection pressure effects
    RESISTANCE = "resistance"        # Resistance cost/benefit
    ENVIRONMENTAL = "environmental"  # Environmental stress factors
    SPATIAL = "spatial"             # Spatial/density effects
    EPISTATIC = "epistatic"         # Gene interaction effects


class FitnessNormalizationMethod(Enum):
    """Methods for fitness normalization."""
    NONE = "none"                   # No normalization
    RELATIVE = "relative"           # Relative to population mean
    LOGARITHMIC = "logarithmic"     # Logarithmic scaling
    SIGMOID = "sigmoid"             # Sigmoid function
    MIN_MAX = "min_max"            # Min-max scaling


@dataclass
class FitnessWeights:
    """Weights for different fitness components."""
    
    base: float = 1.0
    age_related: float = 1.0
    mutation: float = 0.8
    selection: float = 1.2
    resistance: float = 1.0
    environmental: float = 0.9
    spatial: float = 0.7
    epistatic: float = 0.5
    
    def validate(self) -> None:
        """Validate weight values."""
        weights = [self.base, self.age_related, self.mutation, self.selection,
                  self.resistance, self.environmental, self.spatial, self.epistatic]
        if any(w < 0 for w in weights):
            raise ValueError("All fitness weights must be non-negative")


@dataclass
class FitnessConfig:
    """Configuration for fitness calculations."""
    
    # Component weights
    weights: FitnessWeights = field(default_factory=FitnessWeights)
    
    # Normalization settings
    normalization_method: FitnessNormalizationMethod = FitnessNormalizationMethod.RELATIVE
    normalization_window: int = 100  # Generations for rolling normalization
    
    # Fitness bounds
    min_fitness: float = 0.01  # Minimum allowed fitness
    max_fitness: float = 10.0  # Maximum allowed fitness
    
    # Age-related decline
    age_decline_rate: float = 0.01  # Fitness loss per generation
    max_age_penalty: float = 0.5   # Maximum age penalty
    
    # Epistatic interaction strength
    epistatic_strength: float = 0.1  # Strength of gene interactions
    
    # Environmental integration
    environmental_sensitivity: float = 1.0  # How much environment affects fitness
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        self.weights.validate()
        
        if not 0 < self.min_fitness < self.max_fitness:
            raise ValueError("Invalid fitness bounds")
        
        if not 0 <= self.age_decline_rate <= 0.1:
            raise ValueError("Age decline rate must be between 0 and 0.1")


@dataclass
class FitnessCalculationResult:
    """Result of comprehensive fitness calculation."""
    
    # Final fitness value
    final_fitness: float
    
    # Component breakdown
    component_values: Dict[FitnessComponent, float] = field(default_factory=dict)
    component_contributions: Dict[FitnessComponent, float] = field(default_factory=dict)
    
    # Supporting calculations
    base_fitness: float = 1.0
    normalized_fitness: float = 1.0
    fitness_multiplier: float = 1.0
    
    # Detailed results from sub-systems
    mutation_effects: List[Tuple[Mutation, float]] = field(default_factory=list)
    selection_results: List[SelectionResult] = field(default_factory=list)
    resistance_result: Optional[CostBenefitResult] = None
    
    # Context information
    environmental_factors: Dict[str, float] = field(default_factory=dict)
    population_context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_mutation_effect(self) -> float:
        """Get total effect of all mutations."""
        return sum(effect for _, effect in self.mutation_effects)
    
    @property
    def total_selection_effect(self) -> float:
        """Get total effect of all selection pressures."""
        if not self.selection_results:
            return 0.0
        return sum(r.fitness_change for r in self.selection_results)
    
    @property
    def resistance_effect(self) -> float:
        """Get resistance cost/benefit effect."""
        return self.resistance_result.net_fitness_effect if self.resistance_result else 0.0
    
    def get_summary(self) -> str:
        """Get human-readable fitness summary."""
        components = []
        for comp, value in self.component_contributions.items():
            if abs(value) > 0.001:  # Only show significant contributions
                components.append(f"{comp.value}: {value:+.3f}")
        
        return f"Fitness: {self.final_fitness:.3f} ({', '.join(components)})"


class ComprehensiveFitnessCalculator:
    """
    Comprehensive fitness calculator that integrates all fitness-affecting factors.
    
    This calculator provides a unified framework for computing fitness values that
    incorporate mutation effects, selection pressures, resistance costs/benefits,
    environmental factors, and their complex interactions.
    """
    
    def __init__(
        self,
        config: Optional[FitnessConfig] = None,
        resistance_calculator: Optional[ResistanceCostBenefitCalculator] = None
    ):
        """
        Initialize the comprehensive fitness calculator.
        
        Args:
            config: Configuration for fitness calculations
            resistance_calculator: Calculator for resistance effects
        """
        self.config = config or FitnessConfig()
        self.config.validate()
        
        self.resistance_calculator = resistance_calculator or ResistanceCostBenefitCalculator()
        
        # Historical data for normalization
        self._fitness_history: List[float] = []
        self._generation_count = 0
    
    def calculate_fitness(
        self,
        bacterium: Bacterium,
        mutations: Optional[List[Mutation]] = None,
        selection_environment: Optional[SelectionEnvironment] = None,
        environmental_context: Optional[EnvironmentalContext] = None,
        population_context: Optional[Dict[str, Any]] = None
    ) -> FitnessCalculationResult:
        """
        Calculate comprehensive fitness for a bacterium.
        
        Args:
            bacterium: The bacterium to calculate fitness for
            mutations: List of mutations affecting the bacterium
            selection_environment: Current selection pressures
            environmental_context: Environmental conditions
            population_context: Population-level context information
            
        Returns:
            Comprehensive fitness calculation result
        """
        # Initialize result
        result = FitnessCalculationResult(
            final_fitness=bacterium.fitness,
            base_fitness=bacterium.fitness,
            environmental_factors=environmental_context.__dict__ if environmental_context else {},
            population_context=population_context or {}
        )
        
        # Calculate individual fitness components
        self._calculate_base_fitness(bacterium, result)
        self._calculate_age_effects(bacterium, result)
        self._calculate_mutation_effects(bacterium, mutations or [], result)
        self._calculate_selection_effects(bacterium, selection_environment, result)
        self._calculate_resistance_effects(bacterium, environmental_context, result)
        self._calculate_environmental_effects(bacterium, environmental_context, result)
        self._calculate_spatial_effects(bacterium, population_context, result)
        self._calculate_epistatic_effects(bacterium, mutations or [], result)
        
        # Combine all components
        self._combine_fitness_components(result)
        
        # Apply normalization
        self._normalize_fitness(result)
        
        # Update historical data
        self._update_fitness_history(result.final_fitness)
        
        return result
    
    def _calculate_base_fitness(
        self,
        bacterium: Bacterium,
        result: FitnessCalculationResult
    ) -> None:
        """Calculate base genetic fitness component."""
        base_value = bacterium.fitness
        contribution = base_value * self.config.weights.base
        
        result.component_values[FitnessComponent.BASE] = base_value
        result.component_contributions[FitnessComponent.BASE] = contribution
    
    def _calculate_age_effects(
        self,
        bacterium: Bacterium,
        result: FitnessCalculationResult
    ) -> None:
        """Calculate age-related fitness decline."""
        # Age penalty increases with age but caps at max penalty
        age_penalty = min(
            self.config.max_age_penalty,
            bacterium.age * self.config.age_decline_rate
        )
        
        age_value = 1.0 - age_penalty
        contribution = age_value * self.config.weights.age_related
        
        result.component_values[FitnessComponent.AGE_RELATED] = age_value
        result.component_contributions[FitnessComponent.AGE_RELATED] = contribution
    
    def _calculate_mutation_effects(
        self,
        bacterium: Bacterium,
        mutations: List[Mutation],
        result: FitnessCalculationResult
    ) -> None:
        """Calculate cumulative effects of mutations."""
        total_effect = 0.0
        mutation_effects = []
        
        for mutation in mutations:
            effect = self._get_mutation_fitness_effect(mutation)
            total_effect += effect
            mutation_effects.append((mutation, effect))
        
        # Apply diminishing returns for many mutations
        if len(mutations) > 5:
            diminishing_factor = 0.95 ** (len(mutations) - 5)
            total_effect *= diminishing_factor
        
        mutation_value = 1.0 + total_effect
        contribution = mutation_value * self.config.weights.mutation
        
        result.component_values[FitnessComponent.MUTATION] = mutation_value
        result.component_contributions[FitnessComponent.MUTATION] = contribution
        result.mutation_effects = mutation_effects
    
    def _calculate_selection_effects(
        self,
        bacterium: Bacterium,
        selection_environment: Optional[SelectionEnvironment],
        result: FitnessCalculationResult
    ) -> None:
        """Calculate effects of selection pressures."""
        if not selection_environment:
            result.component_values[FitnessComponent.SELECTION] = 1.0
            result.component_contributions[FitnessComponent.SELECTION] = self.config.weights.selection
            return
        
        # Get population context and generation from result if available
        population_context = result.population_context or {}
        generation = result.environmental_factors.get('generation', 0)
        
        # Apply all selection pressures
        selection_results = selection_environment.apply_selection([bacterium], population_context, generation)
        
        # Get the result for our bacterium
        bacterium_result = next((r for r in selection_results if r.bacterium_id == bacterium.id), None)
        
        if bacterium_result:
            # Calculate selection effect from the result
            selection_value = max(0.1, bacterium_result.modified_fitness / bacterium_result.original_fitness)
            contribution = selection_value * self.config.weights.selection
            
            result.component_values[FitnessComponent.SELECTION] = selection_value
            result.component_contributions[FitnessComponent.SELECTION] = contribution
            result.selection_results = [bacterium_result]
        else:
            # No selection result found, use defaults
            result.component_values[FitnessComponent.SELECTION] = 1.0
            result.component_contributions[FitnessComponent.SELECTION] = self.config.weights.selection
            result.selection_results = []
    
    def _calculate_resistance_effects(
        self,
        bacterium: Bacterium,
        environmental_context: Optional[EnvironmentalContext],
        result: FitnessCalculationResult
    ) -> None:
        """Calculate resistance cost/benefit effects."""
        if not environmental_context:
            result.component_values[FitnessComponent.RESISTANCE] = 1.0
            result.component_contributions[FitnessComponent.RESISTANCE] = self.config.weights.resistance
            return
        
        # Calculate resistance effects
        resistance_result = self.resistance_calculator.calculate_net_effect(
            bacterium, environmental_context
        )
        
        resistance_value = resistance_result.fitness_multiplier
        contribution = resistance_value * self.config.weights.resistance
        
        result.component_values[FitnessComponent.RESISTANCE] = resistance_value
        result.component_contributions[FitnessComponent.RESISTANCE] = contribution
        result.resistance_result = resistance_result
    
    def _calculate_environmental_effects(
        self,
        bacterium: Bacterium,
        environmental_context: Optional[EnvironmentalContext],
        result: FitnessCalculationResult
    ) -> None:
        """Calculate environmental stress effects."""
        if not environmental_context:
            result.component_values[FitnessComponent.ENVIRONMENTAL] = 1.0
            result.component_contributions[FitnessComponent.ENVIRONMENTAL] = self.config.weights.environmental
            return
        
        # Calculate environmental stress impact
        stress_factors = [
            environmental_context.temperature_stress,
            environmental_context.ph_stress,
            environmental_context.osmotic_stress
        ]
        
        # Combined stress with diminishing effects
        total_stress = 1.0 - np.prod([1.0 - s for s in stress_factors])
        
        # Apply environmental sensitivity
        stress_impact = total_stress * self.config.environmental_sensitivity
        environmental_value = max(0.1, 1.0 - stress_impact)
        contribution = environmental_value * self.config.weights.environmental
        
        result.component_values[FitnessComponent.ENVIRONMENTAL] = environmental_value
        result.component_contributions[FitnessComponent.ENVIRONMENTAL] = contribution
    
    def _calculate_spatial_effects(
        self,
        bacterium: Bacterium,
        population_context: Optional[Dict[str, Any]],
        result: FitnessCalculationResult
    ) -> None:
        """Calculate spatial density and competition effects."""
        if not population_context:
            result.component_values[FitnessComponent.SPATIAL] = 1.0
            result.component_contributions[FitnessComponent.SPATIAL] = self.config.weights.spatial
            return
        
        # Get local density from population context
        local_density = population_context.get('local_density', 1.0)
        carrying_capacity = population_context.get('carrying_capacity', 100)
        
        # Calculate crowding effect
        crowding_factor = min(1.0, local_density / carrying_capacity)
        crowding_penalty = crowding_factor * 0.3  # Max 30% penalty
        
        spatial_value = max(0.1, 1.0 - crowding_penalty)
        contribution = spatial_value * self.config.weights.spatial
        
        result.component_values[FitnessComponent.SPATIAL] = spatial_value
        result.component_contributions[FitnessComponent.SPATIAL] = contribution
    
    def _calculate_epistatic_effects(
        self,
        bacterium: Bacterium,
        mutations: List[Mutation],
        result: FitnessCalculationResult
    ) -> None:
        """Calculate gene interaction (epistatic) effects."""
        if len(mutations) < 2:
            result.component_values[FitnessComponent.EPISTATIC] = 1.0
            result.component_contributions[FitnessComponent.EPISTATIC] = self.config.weights.epistatic
            return
        
        # Calculate interactions between mutations
        epistatic_effect = 0.0
        num_interactions = 0
        
        for i, mut1 in enumerate(mutations):
            for mut2 in mutations[i+1:]:
                interaction = self._calculate_mutation_interaction(mut1, mut2)
                epistatic_effect += interaction
                num_interactions += 1
        
        # Normalize by number of interactions
        if num_interactions > 0:
            epistatic_effect /= num_interactions
        
        # Apply epistatic strength
        epistatic_effect *= self.config.epistatic_strength
        
        epistatic_value = 1.0 + epistatic_effect
        contribution = epistatic_value * self.config.weights.epistatic
        
        result.component_values[FitnessComponent.EPISTATIC] = epistatic_value
        result.component_contributions[FitnessComponent.EPISTATIC] = contribution
    
    def _combine_fitness_components(self, result: FitnessCalculationResult) -> None:
        """Combine all fitness components into final fitness value."""
        # Calculate weighted sum of contributions
        total_contribution = sum(result.component_contributions.values())
        
        # Normalize by total weight
        total_weight = sum([
            self.config.weights.base,
            self.config.weights.age_related,
            self.config.weights.mutation,
            self.config.weights.selection,
            self.config.weights.resistance,
            self.config.weights.environmental,
            self.config.weights.spatial,
            self.config.weights.epistatic
        ])
        
        # Calculate fitness multiplier
        result.fitness_multiplier = total_contribution / total_weight
        
        # Apply to base fitness
        result.final_fitness = result.base_fitness * result.fitness_multiplier
    
    def _normalize_fitness(self, result: FitnessCalculationResult) -> None:
        """Apply fitness normalization."""
        if self.config.normalization_method == FitnessNormalizationMethod.NONE:
            result.normalized_fitness = result.final_fitness
        elif self.config.normalization_method == FitnessNormalizationMethod.RELATIVE:
            # Normalize relative to recent population mean
            if len(self._fitness_history) > 0:
                recent_history = self._fitness_history[-self.config.normalization_window:]
                mean_fitness = np.mean(recent_history)
                result.normalized_fitness = result.final_fitness / max(0.1, mean_fitness)
            else:
                result.normalized_fitness = result.final_fitness
        elif self.config.normalization_method == FitnessNormalizationMethod.LOGARITHMIC:
            result.normalized_fitness = math.log1p(result.final_fitness)
        elif self.config.normalization_method == FitnessNormalizationMethod.SIGMOID:
            result.normalized_fitness = 1.0 / (1.0 + math.exp(-result.final_fitness + 1.0))
        elif self.config.normalization_method == FitnessNormalizationMethod.MIN_MAX:
            if len(self._fitness_history) > 0:
                recent_history = self._fitness_history[-self.config.normalization_window:]
                min_fitness = min(recent_history)
                max_fitness = max(recent_history)
                if max_fitness > min_fitness:
                    result.normalized_fitness = (result.final_fitness - min_fitness) / (max_fitness - min_fitness)
                else:
                    result.normalized_fitness = 1.0
            else:
                result.normalized_fitness = result.final_fitness

        # Update final fitness with normalized value
        result.final_fitness = result.normalized_fitness
        
        # Apply bounds after normalization
        result.final_fitness = max(
            self.config.min_fitness,
            min(self.config.max_fitness, result.final_fitness)
        )
    
    def _update_fitness_history(self, fitness: float) -> None:
        """Update fitness history for normalization."""
        self._fitness_history.append(fitness)
        
        # Keep only recent history within the normalization window
        if len(self._fitness_history) > self.config.normalization_window:
            self._fitness_history = self._fitness_history[-self.config.normalization_window:]
        
        self._generation_count += 1
    
    def _get_mutation_fitness_effect(self, mutation: Mutation) -> float:
        """Get fitness effect of a single mutation."""
        # Map mutation effects to fitness changes
        effect_map = {
            MutationEffect.BENEFICIAL: 0.1,
            MutationEffect.NEUTRAL: 0.0,
            MutationEffect.DELETERIOUS: -0.05,
            MutationEffect.LETHAL: -0.9
        }
        
        base_effect = effect_map.get(mutation.effect, 0.0)
        
        # Apply magnitude factor if available
        if hasattr(mutation, 'magnitude'):
            base_effect *= mutation.magnitude
        
        return base_effect
    
    def _calculate_mutation_interaction(self, mut1: Mutation, mut2: Mutation) -> float:
        """Calculate epistatic interaction between two mutations."""
        # Simple epistatic model - interactions depend on mutation types
        if mut1.effect == MutationEffect.BENEFICIAL and mut2.effect == MutationEffect.BENEFICIAL:
            # Positive synergy
            return 0.02
        elif mut1.effect == MutationEffect.DELETERIOUS and mut2.effect == MutationEffect.DELETERIOUS:
            # Negative synergy (mutations compound negatively)
            return -0.03
        elif (mut1.effect == MutationEffect.BENEFICIAL and mut2.effect == MutationEffect.DELETERIOUS) or \
             (mut1.effect == MutationEffect.DELETERIOUS and mut2.effect == MutationEffect.BENEFICIAL):
            # Compensation effect
            return 0.01
        else:
            # No significant interaction
            return 0.0
    
    def get_fitness_landscape(
        self,
        reference_bacterium: Bacterium,
        environmental_gradient: Dict[str, Tuple[float, float]],
        num_points: int = 20
    ) -> Dict[str, Any]:
        """
        Generate fitness landscape across environmental gradients.
        
        Args:
            reference_bacterium: Reference bacterium for calculations
            environmental_gradient: Dict mapping environmental factors to (min, max) ranges
            num_points: Number of points to sample along each gradient
            
        Returns:
            Fitness landscape data
        """
        # This is a simplified version - full implementation would be more complex
        landscape = {
            'gradients': environmental_gradient,
            'fitness_values': [],
            'coordinates': []
        }
        
        # For each environmental factor, sample along its gradient
        for factor, (min_val, max_val) in environmental_gradient.items():
            values = np.linspace(min_val, max_val, num_points)
            fitness_values = []
            
            for value in values:
                # Create environmental context with this factor value
                context = EnvironmentalContext(**{factor: value})
                
                # Calculate fitness
                result = self.calculate_fitness(
                    reference_bacterium,
                    environmental_context=context
                )
                fitness_values.append(result.final_fitness)
            
            landscape['fitness_values'].append(fitness_values)
            landscape['coordinates'].append(values.tolist())
        
        return landscape
    
    def reset_history(self) -> None:
        """Reset fitness history (useful for new simulations)."""
        self._fitness_history.clear()
        self._generation_count = 0


# Convenience functions
def calculate_comprehensive_fitness(
    bacterium: Bacterium,
    mutations: Optional[List[Mutation]] = None,
    selection_environment: Optional[SelectionEnvironment] = None,
    environmental_context: Optional[EnvironmentalContext] = None,
    population_context: Optional[Dict[str, Any]] = None,
    calculator: Optional[ComprehensiveFitnessCalculator] = None
) -> FitnessCalculationResult:
    """
    Convenience function for comprehensive fitness calculation.
    
    Args:
        bacterium: Bacterium to calculate fitness for
        mutations: List of mutations affecting the bacterium
        selection_environment: Current selection pressures
        environmental_context: Environmental conditions
        population_context: Population-level context
        calculator: Optional calculator instance
        
    Returns:
        Comprehensive fitness calculation result
    """
    if calculator is None:
        calculator = ComprehensiveFitnessCalculator()
    
    return calculator.calculate_fitness(
        bacterium=bacterium,
        mutations=mutations,
        selection_environment=selection_environment,
        environmental_context=environmental_context,
        population_context=population_context
    )


def update_bacterium_comprehensive_fitness(
    bacterium: Bacterium,
    mutations: Optional[List[Mutation]] = None,
    selection_environment: Optional[SelectionEnvironment] = None,
    environmental_context: Optional[EnvironmentalContext] = None,
    population_context: Optional[Dict[str, Any]] = None,
    calculator: Optional[ComprehensiveFitnessCalculator] = None,
    apply_immediately: bool = True
) -> Tuple[float, FitnessCalculationResult]:
    """
    Calculate and optionally apply comprehensive fitness to a bacterium.
    
    Args:
        bacterium: Bacterium to update
        mutations: List of mutations affecting the bacterium
        selection_environment: Current selection pressures
        environmental_context: Environmental conditions
        population_context: Population-level context
        calculator: Optional calculator instance
        apply_immediately: Whether to update bacterium.fitness immediately
        
    Returns:
        Tuple of (new_fitness, calculation_result)
    """
    result = calculate_comprehensive_fitness(
        bacterium=bacterium,
        mutations=mutations,
        selection_environment=selection_environment,
        environmental_context=environmental_context,
        population_context=population_context,
        calculator=calculator
    )
    
    if apply_immediately:
        bacterium.fitness = result.final_fitness
    
    return result.final_fitness, result 