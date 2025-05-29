"""
Resistance Cost/Benefit Calculation System.

This module provides comprehensive mathematical models for calculating the metabolic
costs and survival benefits of resistance mutations under different environmental
conditions.
"""

from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math
from .bacterium import Bacterium, ResistanceStatus


class CostType(Enum):
    """Types of resistance costs."""
    METABOLIC = "metabolic"  # Energy cost of resistance mechanisms
    FITNESS = "fitness"      # Direct reproductive fitness cost
    GROWTH = "growth"        # Growth rate penalty
    MAINTENANCE = "maintenance"  # Cost of maintaining resistance genes


class BenefitType(Enum):
    """Types of resistance benefits."""
    SURVIVAL = "survival"    # Direct survival advantage
    COMPETITIVE = "competitive"  # Advantage over sensitive bacteria
    FREQUENCY_DEPENDENT = "frequency_dependent"  # Benefits based on population composition


@dataclass
class ResistanceCostConfig:
    """Configuration for resistance cost calculations."""
    
    # Base cost parameters (0.0 to 1.0)
    base_metabolic_cost: float = 0.05  # 5% metabolic cost
    base_fitness_cost: float = 0.03    # 3% fitness reduction
    base_growth_cost: float = 0.02     # 2% growth rate reduction
    
    # Environmental modifiers
    nutrient_stress_multiplier: float = 1.5  # Increased cost under nutrient stress
    temperature_stress_multiplier: float = 1.3  # Increased cost under temperature stress
    ph_stress_multiplier: float = 1.2  # Increased cost under pH stress
    
    # Frequency-dependent factors
    frequency_dependence_strength: float = 0.1  # How much frequency affects costs
    
    # Cost reduction factors (evolved efficiency)
    max_cost_reduction: float = 0.3  # Maximum 30% cost reduction through evolution
    cost_reduction_rate: float = 0.01  # Rate of cost reduction per generation
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        costs = [self.base_metabolic_cost, self.base_fitness_cost, self.base_growth_cost]
        if any(cost < 0 or cost > 1 for cost in costs):
            raise ValueError("Base costs must be between 0.0 and 1.0")
        
        if self.max_cost_reduction < 0 or self.max_cost_reduction > 1:
            raise ValueError("Max cost reduction must be between 0.0 and 1.0")


@dataclass
class ResistanceBenefitConfig:
    """Configuration for resistance benefit calculations."""
    
    # Base benefit parameters
    base_survival_benefit: float = 0.8   # 80% survival advantage with antibiotics
    max_survival_benefit: float = 0.95   # Maximum 95% survival advantage
    
    # Antibiotic concentration response
    mic_fold_change: float = 8.0  # Fold increase in MIC for resistant bacteria
    hill_coefficient: float = 2.0  # Hill coefficient for dose-response
    
    # Competitive advantages
    resource_competition_benefit: float = 0.1  # 10% advantage in resource competition
    spatial_advantage: float = 0.05  # 5% advantage in spatial competition
    
    # Frequency-dependent benefits
    minority_advantage: float = 0.05  # 5% advantage when rare (reduced from 20%)
    frequency_threshold: float = 0.1  # Threshold below which minority advantage applies
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0 <= self.base_survival_benefit <= 1:
            raise ValueError("Base survival benefit must be between 0.0 and 1.0")
        if not 0 <= self.max_survival_benefit <= 1:
            raise ValueError("Max survival benefit must be between 0.0 and 1.0")


@dataclass
class EnvironmentalContext:
    """Environmental context for cost/benefit calculations."""
    
    # Antimicrobial pressure
    antibiotic_concentration: float = 0.0  # mg/L or MIC units
    antibiotic_type: str = "general"  # Type of antibiotic
    
    # Resource availability
    nutrient_availability: float = 1.0  # 0.0 (scarce) to 1.0 (abundant)
    resource_competition: float = 0.0   # 0.0 (no competition) to 1.0 (intense)
    
    # Environmental stressors
    temperature_stress: float = 0.0  # 0.0 (optimal) to 1.0 (extreme)
    ph_stress: float = 0.0          # 0.0 (optimal) to 1.0 (extreme)
    osmotic_stress: float = 0.0     # 0.0 (optimal) to 1.0 (extreme)
    
    # Population context
    resistance_frequency: float = 0.01  # Fraction of population that is resistant
    population_density: float = 1.0     # Relative population density
    
    # Temporal factors
    generation: int = 0  # Current generation (for evolved cost reduction)
    
    def validate(self) -> None:
        """Validate environmental context parameters."""
        if self.antibiotic_concentration < 0:
            raise ValueError("Antibiotic concentration must be non-negative")
        
        bounded_params = [
            self.nutrient_availability, self.resource_competition,
            self.temperature_stress, self.ph_stress, self.osmotic_stress,
            self.resistance_frequency, self.population_density
        ]
        
        if any(param < 0 or param > 1 for param in bounded_params):
            raise ValueError("Environmental parameters must be between 0.0 and 1.0")


@dataclass
class CostBenefitResult:
    """Result of cost/benefit calculation."""
    
    # Total effects
    total_cost: float = 0.0
    total_benefit: float = 0.0
    net_fitness_effect: float = 0.0  # benefit - cost
    
    # Detailed breakdown
    cost_breakdown: Dict[CostType, float] = field(default_factory=dict)
    benefit_breakdown: Dict[BenefitType, float] = field(default_factory=dict)
    
    # Environmental modifiers applied
    environmental_cost_modifier: float = 1.0
    frequency_modifier: float = 1.0
    
    @property
    def fitness_multiplier(self) -> float:
        """Get fitness multiplier from net effect."""
        return max(0.01, 1.0 + self.net_fitness_effect)  # Minimum 1% fitness
    
    @property
    def is_beneficial(self) -> bool:
        """Check if resistance is net beneficial."""
        return self.net_fitness_effect > 0
    
    @property
    def cost_summary(self) -> str:
        """Get human-readable cost summary."""
        costs = [f"{cost_type.value}: {value:.1%}" 
                for cost_type, value in self.cost_breakdown.items()]
        return f"Total: {self.total_cost:.1%} ({', '.join(costs)})"
    
    @property
    def benefit_summary(self) -> str:
        """Get human-readable benefit summary."""
        benefits = [f"{benefit_type.value}: {value:.1%}" 
                   for benefit_type, value in self.benefit_breakdown.items()]
        return f"Total: {self.total_benefit:.1%} ({', '.join(benefits)})"


class ResistanceCostBenefitCalculator:
    """
    Calculator for resistance costs and benefits under different conditions.
    
    This class implements mathematical models for:
    - Metabolic costs of resistance mechanisms
    - Fitness costs and trade-offs
    - Survival benefits under antimicrobial pressure
    - Frequency-dependent selection effects
    - Environmental context dependencies
    """
    
    def __init__(
        self,
        cost_config: Optional[ResistanceCostConfig] = None,
        benefit_config: Optional[ResistanceBenefitConfig] = None
    ):
        """
        Initialize calculator with configuration.
        
        Args:
            cost_config: Configuration for cost calculations
            benefit_config: Configuration for benefit calculations
        """
        self.cost_config = cost_config or ResistanceCostConfig()
        self.benefit_config = benefit_config or ResistanceBenefitConfig()
        
        # Validate configurations
        self.cost_config.validate()
        self.benefit_config.validate()
    
    def calculate_costs(
        self,
        bacterium: Bacterium,
        context: EnvironmentalContext
    ) -> Dict[CostType, float]:
        """
        Calculate resistance costs for a bacterium in given context.
        
        Args:
            bacterium: The bacterium to calculate costs for
            context: Environmental context
            
        Returns:
            Dictionary mapping cost types to values
        """
        if not bacterium.is_resistant:
            return {cost_type: 0.0 for cost_type in CostType}
        
        costs = {}
        
        # Base metabolic cost
        metabolic_cost = self.cost_config.base_metabolic_cost
        
        # Environmental stress multipliers
        stress_multiplier = 1.0
        stress_multiplier *= (1.0 + context.temperature_stress * 
                            (self.cost_config.temperature_stress_multiplier - 1.0))
        stress_multiplier *= (1.0 + context.ph_stress * 
                            (self.cost_config.ph_stress_multiplier - 1.0))
        
        # Nutrient stress increases metabolic costs
        if context.nutrient_availability < 0.5:
            nutrient_stress = 1.0 - context.nutrient_availability
            stress_multiplier *= (1.0 + nutrient_stress * 
                                (self.cost_config.nutrient_stress_multiplier - 1.0))
        
        costs[CostType.METABOLIC] = metabolic_cost * stress_multiplier
        
        # Fitness cost (direct reproductive cost)
        fitness_cost = self.cost_config.base_fitness_cost
        
        # Frequency-dependent cost modulation
        if context.resistance_frequency < 0.5:
            # Being rare reduces relative fitness cost (frequency-dependent selection)
            frequency_effect = context.resistance_frequency * self.cost_config.frequency_dependence_strength
            fitness_cost *= (1.0 - frequency_effect)
        
        # Round to avoid floating point precision issues
        costs[CostType.FITNESS] = round(fitness_cost, 6)
        
        # Growth cost (affects reproduction rate)
        growth_cost = self.cost_config.base_growth_cost
        
        # Resource competition increases growth costs
        if context.resource_competition > 0.3:
            competition_multiplier = 1.0 + context.resource_competition * 0.5
            growth_cost *= competition_multiplier
        
        costs[CostType.GROWTH] = growth_cost
        
        # Maintenance cost (cost of maintaining resistance genes)
        # Decreases over generations due to evolutionary optimization
        generation_reduction = min(
            self.cost_config.max_cost_reduction,
            context.generation * self.cost_config.cost_reduction_rate
        )
        
        base_maintenance = self.cost_config.base_metabolic_cost * 0.3  # 30% of metabolic cost
        maintenance_cost = base_maintenance * (1.0 - generation_reduction)
        
        costs[CostType.MAINTENANCE] = maintenance_cost
        
        return costs
    
    def calculate_benefits(
        self,
        bacterium: Bacterium,
        context: EnvironmentalContext
    ) -> Dict[BenefitType, float]:
        """
        Calculate resistance benefits for a bacterium in given context.
        
        Args:
            bacterium: The bacterium to calculate benefits for
            context: Environmental context
            
        Returns:
            Dictionary mapping benefit types to values
        """
        if not bacterium.is_resistant:
            return {benefit_type: 0.0 for benefit_type in BenefitType}
        
        benefits = {}
        
        # Survival benefit from antibiotic resistance
        if context.antibiotic_concentration > 0:
            # Hill equation for dose-response
            mic_ratio = context.antibiotic_concentration / self.benefit_config.mic_fold_change
            hill_numerator = mic_ratio ** self.benefit_config.hill_coefficient
            hill_denominator = 1.0 + hill_numerator
            survival_benefit = self.benefit_config.max_survival_benefit * (hill_numerator / hill_denominator)
            
            # Ensure minimum benefit for any antibiotic concentration
            min_benefit = self.benefit_config.base_survival_benefit * min(1.0, context.antibiotic_concentration / 2.0)
            survival_benefit = max(survival_benefit, min_benefit)
        else:
            survival_benefit = 0.0
        
        benefits[BenefitType.SURVIVAL] = survival_benefit
        
        # Competitive benefit in resource competition
        if context.resource_competition > 0.2:
            # Resistant bacteria may have competitive advantages in some contexts
            competitive_benefit = (context.resource_competition * 
                                 self.benefit_config.resource_competition_benefit)
        else:
            competitive_benefit = 0.0
        
        benefits[BenefitType.COMPETITIVE] = competitive_benefit
        
        # Frequency-dependent benefits (only applies under specific conditions)
        frequency_benefit = 0.0
        if (context.resistance_frequency < self.benefit_config.frequency_threshold and 
            (context.antibiotic_concentration > 0 or context.resource_competition > 0.3)):
            # Minority advantage only when there's actual selection pressure
            rarity_factor = (self.benefit_config.frequency_threshold - context.resistance_frequency) / \
                          self.benefit_config.frequency_threshold
            frequency_benefit = rarity_factor * self.benefit_config.minority_advantage * 0.5  # Reduced impact
        
        benefits[BenefitType.FREQUENCY_DEPENDENT] = frequency_benefit
        
        return benefits
    
    def calculate_net_effect(
        self,
        bacterium: Bacterium,
        context: EnvironmentalContext
    ) -> CostBenefitResult:
        """
        Calculate net fitness effect of resistance.
        
        Args:
            bacterium: The bacterium to analyze
            context: Environmental context
            
        Returns:
            Complete cost/benefit analysis result
        """
        # Calculate individual costs and benefits
        costs = self.calculate_costs(bacterium, context)
        benefits = self.calculate_benefits(bacterium, context)
        
        # Sum total costs and benefits
        total_cost = sum(costs.values())
        total_benefit = sum(benefits.values())
        
        # Calculate environmental and frequency modifiers
        env_modifier = 1.0
        if context.temperature_stress > 0.5 or context.ph_stress > 0.5:
            env_modifier += 0.2  # 20% cost increase under severe stress
        
        freq_modifier = 1.0
        if context.resistance_frequency < 0.1:
            freq_modifier *= 0.8  # 20% cost reduction when rare
        elif context.resistance_frequency > 0.9:
            freq_modifier *= 1.2  # 20% cost increase when dominant
        
        # Apply modifiers to costs
        adjusted_total_cost = total_cost * env_modifier * freq_modifier
        
        # Net effect
        net_effect = total_benefit - adjusted_total_cost
        
        return CostBenefitResult(
            total_cost=adjusted_total_cost,
            total_benefit=total_benefit,
            net_fitness_effect=net_effect,
            cost_breakdown=costs,
            benefit_breakdown=benefits,
            environmental_cost_modifier=env_modifier,
            frequency_modifier=freq_modifier
        )
    
    def update_bacterium_fitness(
        self,
        bacterium: Bacterium,
        context: EnvironmentalContext,
        apply_immediately: bool = True
    ) -> Tuple[float, CostBenefitResult]:
        """
        Update bacterium fitness based on resistance cost/benefit analysis.
        
        Args:
            bacterium: Bacterium to update
            context: Environmental context
            apply_immediately: Whether to apply fitness change to bacterium
            
        Returns:
            Tuple of (new_fitness, cost_benefit_result)
        """
        result = self.calculate_net_effect(bacterium, context)
        
        # Calculate new fitness
        new_fitness = bacterium.fitness * result.fitness_multiplier
        
        # Apply if requested
        if apply_immediately:
            bacterium.fitness = new_fitness
        
        return new_fitness, result
    
    def get_fitness_landscape(
        self,
        context: EnvironmentalContext,
        antibiotic_range: Tuple[float, float] = (0.0, 10.0),
        num_points: int = 50
    ) -> Dict[str, Any]:
        """
        Generate fitness landscape across antibiotic concentration range.
        
        Args:
            context: Base environmental context
            antibiotic_range: Range of antibiotic concentrations to test
            num_points: Number of points to sample
            
        Returns:
            Dictionary with fitness landscape data
        """
        import numpy as np
        
        concentrations = np.linspace(antibiotic_range[0], antibiotic_range[1], num_points)
        resistant_fitness = []
        sensitive_fitness = []
        
        # Create test bacteria
        resistant_bact = Bacterium(id="test_resistant", 
                                 resistance_status=ResistanceStatus.RESISTANT)
        sensitive_bact = Bacterium(id="test_sensitive", 
                                 resistance_status=ResistanceStatus.SENSITIVE)
        
        for conc in concentrations:
            # Update context
            test_context = EnvironmentalContext(
                antibiotic_concentration=float(conc),
                nutrient_availability=context.nutrient_availability,
                resource_competition=context.resource_competition,
                temperature_stress=context.temperature_stress,
                ph_stress=context.ph_stress,
                resistance_frequency=context.resistance_frequency,
                generation=context.generation
            )
            
            # Calculate fitness for both types
            resistant_result = self.calculate_net_effect(resistant_bact, test_context)
            sensitive_result = self.calculate_net_effect(sensitive_bact, test_context)
            
            resistant_fitness.append(resistant_result.fitness_multiplier)
            sensitive_fitness.append(sensitive_result.fitness_multiplier)
        
        return {
            'concentrations': concentrations.tolist(),
            'resistant_fitness': resistant_fitness,
            'sensitive_fitness': sensitive_fitness,
            'crossover_point': self._find_fitness_crossover(concentrations, 
                                                          resistant_fitness, 
                                                          sensitive_fitness)
        }
    
    def _find_fitness_crossover(
        self,
        concentrations: any,
        resistant_fitness: list,
        sensitive_fitness: list
    ) -> Optional[float]:
        """Find antibiotic concentration where resistant and sensitive fitness are equal."""
        for i in range(len(concentrations) - 1):
            if (resistant_fitness[i] <= sensitive_fitness[i] and 
                resistant_fitness[i + 1] > sensitive_fitness[i + 1]):
                # Linear interpolation to find exact crossover
                r1, r2 = resistant_fitness[i], resistant_fitness[i + 1]
                s1, s2 = sensitive_fitness[i], sensitive_fitness[i + 1]
                c1, c2 = concentrations[i], concentrations[i + 1]
                
                if (r2 - r1) != (s2 - s1):  # Avoid division by zero
                    crossover = c1 + (c2 - c1) * (s1 - r1) / ((r2 - r1) - (s2 - s1))
                    return float(crossover)
        
        return None


def calculate_resistance_fitness_effect(
    bacterium: Bacterium,
    context: EnvironmentalContext,
    calculator: Optional[ResistanceCostBenefitCalculator] = None
) -> float:
    """
    Convenience function to calculate resistance fitness effect.
    
    Args:
        bacterium: Bacterium to analyze
        context: Environmental context
        calculator: Optional calculator instance (creates default if None)
        
    Returns:
        Fitness multiplier (1.0 = no effect, >1.0 = beneficial, <1.0 = costly)
    """
    if calculator is None:
        calculator = ResistanceCostBenefitCalculator()
    
    result = calculator.calculate_net_effect(bacterium, context)
    return result.fitness_multiplier


# Convenience function for simple cost/benefit calculation
def simple_resistance_cost_benefit(
    is_resistant: bool,
    antibiotic_concentration: float,
    resistance_frequency: float = 0.01,
    base_cost: float = 0.05
) -> float:
    """
    Simple resistance cost/benefit calculation for basic use cases.
    
    Args:
        is_resistant: Whether bacterium is resistant
        antibiotic_concentration: Antibiotic concentration (0-10 scale)
        resistance_frequency: Population resistance frequency
        base_cost: Base fitness cost of resistance
        
    Returns:
        Fitness multiplier
    """
    if not is_resistant:
        if antibiotic_concentration > 0:
            # Sensitive bacteria suffer under antibiotics
            mortality = min(0.9, antibiotic_concentration * 0.3)
            return 1.0 - mortality
        else:
            return 1.0  # No effect without antibiotics
    
    # Resistant bacteria
    benefit = 0.0
    if antibiotic_concentration > 0:
        # Survival benefit under antibiotics
        benefit = min(0.8, antibiotic_concentration * 0.2)
    
    # Frequency-dependent cost reduction
    cost = base_cost
    if resistance_frequency < 0.1:
        cost *= (1.0 - 0.3 * (0.1 - resistance_frequency) / 0.1)  # Up to 30% cost reduction when rare
    
    return max(0.1, 1.0 + benefit - cost)  # Minimum 10% fitness 