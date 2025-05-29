"""
Selection pressure system for bacterial evolution simulation.

This module provides classes for modeling various types of selection pressures
that affect bacterial survival and reproduction, including antimicrobial
pressure, resource competition, environmental stress, and spatial effects.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import numpy as np
from .bacterium import Bacterium, ResistanceStatus


class PressureType(Enum):
    """Types of selection pressures."""
    ANTIMICROBIAL = "antimicrobial"
    RESOURCE = "resource"
    ENVIRONMENTAL = "environmental"
    SPATIAL = "spatial"
    COMPETITIVE = "competitive"
    CUSTOM = "custom"


@dataclass
class PressureConfig:
    """Configuration for selection pressure parameters."""
    pressure_type: PressureType
    intensity: float = 1.0  # Base pressure intensity (0.0 to 10.0)
    duration: Optional[int] = None  # Duration in generations (None = indefinite)
    time_profile: str = "constant"  # "constant", "linear", "exponential", "pulse"
    enabled: bool = True
    
    # Type-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.intensity < 0.0:
            raise ValueError("Pressure intensity must be non-negative")
        if self.duration is not None and self.duration < 0:
            raise ValueError("Duration must be non-negative")


@dataclass
class SelectionResult:
    """Result of applying selection pressure to a bacterium."""
    bacterium_id: str
    original_fitness: float
    modified_fitness: float
    survival_probability: float
    pressure_effects: Dict[str, float]
    selected_for_survival: bool = False
    
    @property
    def fitness_change(self) -> float:
        """Calculate the change in fitness due to selection pressure."""
        return self.modified_fitness - self.original_fitness


class SelectionPressure(ABC):
    """Abstract base class for selection pressures."""
    
    def __init__(self, config: PressureConfig):
        self.config = config
        self.generation_applied = 0
        self.history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def apply_to_bacterium(
        self, 
        bacterium: Bacterium, 
        population_context: Dict[str, Any],
        generation: int
    ) -> SelectionResult:
        """
        Apply selection pressure to a single bacterium.
        
        Args:
            bacterium: The bacterium to apply pressure to
            population_context: Context about the population state
            generation: Current generation number
        
        Returns:
            SelectionResult with modified fitness and survival probability
        """
        pass
    
    @abstractmethod
    def get_pressure_intensity(self, generation: int) -> float:
        """Get pressure intensity at a specific generation."""
        pass
    
    def is_active(self, generation: int) -> bool:
        """Check if pressure is active at given generation."""
        if not self.config.enabled:
            return False
        
        if self.config.duration is not None:
            return generation < self.generation_applied + self.config.duration
        
        return True
    
    def log_application(self, generation: int, results: List[SelectionResult]) -> None:
        """Log the application of this pressure for tracking purposes."""
        summary = {
            'generation': generation,
            'pressure_type': self.config.pressure_type.value,
            'intensity': self.get_pressure_intensity(generation),
            'bacteria_affected': len(results),
            'average_fitness_change': np.mean([r.fitness_change for r in results]) if results else 0.0,
            'survival_rate': np.mean([r.survival_probability for r in results]) if results else 0.0
        }
        self.history.append(summary)


class AntimicrobialPressure(SelectionPressure):
    """Selection pressure from antimicrobial agents."""
    
    def __init__(self, config: PressureConfig):
        super().__init__(config)
        
        # Default antimicrobial parameters
        self.mic_sensitive = config.parameters.get('mic_sensitive', 1.0)  # MIC for sensitive bacteria
        self.mic_resistant = config.parameters.get('mic_resistant', 8.0)  # MIC for resistant bacteria
        self.hill_coefficient = config.parameters.get('hill_coefficient', 2.0)  # Steepness of dose-response
        self.max_kill_rate = config.parameters.get('max_kill_rate', 0.95)  # Maximum kill rate
    
    def apply_to_bacterium(
        self, 
        bacterium: Bacterium, 
        population_context: Dict[str, Any],
        generation: int
    ) -> SelectionResult:
        """Apply antimicrobial pressure to bacterium."""
        concentration = self.get_pressure_intensity(generation)
        
        # Determine MIC based on resistance status
        if bacterium.is_resistant:
            mic = self.mic_resistant
            pressure_effect = f"resistant_mic_{mic}"
        else:
            mic = self.mic_sensitive
            pressure_effect = f"sensitive_mic_{mic}"
        
        # Calculate kill probability using Hill equation
        kill_probability = self._calculate_kill_probability(concentration, mic)
        
        # Survival probability is inverse of kill probability
        survival_prob = 1.0 - kill_probability
        
        # Modify fitness based on antimicrobial stress
        if bacterium.is_resistant:
            # Resistant bacteria survive better but may have fitness cost
            fitness_modifier = max(0.1, survival_prob * 0.9)  # Small fitness cost
        else:
            # Sensitive bacteria severely affected
            fitness_modifier = survival_prob
        
        modified_fitness = bacterium.effective_fitness * fitness_modifier
        
        return SelectionResult(
            bacterium_id=bacterium.id,
            original_fitness=bacterium.effective_fitness,
            modified_fitness=modified_fitness,
            survival_probability=survival_prob,
            pressure_effects={
                'antimicrobial_concentration': concentration,
                'mic_value': mic,
                'kill_probability': kill_probability,
                pressure_effect: fitness_modifier
            }
        )
    
    def _calculate_kill_probability(self, concentration: float, mic: float) -> float:
        """Calculate kill probability using Hill equation."""
        if concentration <= 0:
            return 0.0
        
        # Hill equation: E = Emax * (C^n) / (IC50^n + C^n)
        # Where C = concentration, IC50 = MIC, n = Hill coefficient
        numerator = self.max_kill_rate * (concentration ** self.hill_coefficient)
        denominator = (mic ** self.hill_coefficient) + (concentration ** self.hill_coefficient)
        
        return min(self.max_kill_rate, numerator / denominator)
    
    def get_pressure_intensity(self, generation: int) -> float:
        """Get antimicrobial concentration at specific generation."""
        base_intensity = self.config.intensity
        
        if self.config.time_profile == "constant":
            return base_intensity
        elif self.config.time_profile == "linear":
            # Linear increase over time
            slope = self.config.parameters.get('slope', 0.1)
            return base_intensity + (generation * slope)
        elif self.config.time_profile == "exponential":
            # Exponential decay (drug clearance)
            decay_rate = self.config.parameters.get('decay_rate', 0.1)
            return base_intensity * np.exp(-decay_rate * generation)
        elif self.config.time_profile == "pulse":
            # Periodic dosing
            pulse_interval = self.config.parameters.get('pulse_interval', 24)
            pulse_duration = self.config.parameters.get('pulse_duration', 6)
            
            cycle_position = generation % pulse_interval
            if cycle_position < pulse_duration:
                return base_intensity
            else:
                return 0.0
        
        return base_intensity


class ResourcePressure(SelectionPressure):
    """Selection pressure from resource competition."""
    
    def __init__(self, config: PressureConfig):
        super().__init__(config)
        
        self.carrying_capacity = config.parameters.get('carrying_capacity', 10000)
        self.competition_strength = config.parameters.get('competition_strength', 1.0)
        self.resource_efficiency_resistant = config.parameters.get('resource_efficiency_resistant', 0.9)
    
    def apply_to_bacterium(
        self, 
        bacterium: Bacterium, 
        population_context: Dict[str, Any],
        generation: int
    ) -> SelectionResult:
        """Apply resource competition pressure."""
        population_size = population_context.get('total_population', 1)
        local_density = population_context.get('local_density', 1.0)
        
        # Calculate competition intensity
        competition_factor = min(1.0, population_size / self.carrying_capacity)
        local_competition = min(1.0, local_density / 5.0)  # Assume max 5 per cell
        
        total_competition = (competition_factor + local_competition) / 2
        
        # Resource efficiency varies by resistance status
        if bacterium.is_resistant:
            resource_efficiency = self.resource_efficiency_resistant
        else:
            resource_efficiency = 1.0
        
        # Calculate fitness modification
        pressure_intensity = self.get_pressure_intensity(generation) * total_competition
        
        # Base fitness reduction due to competition
        base_fitness_reduction = pressure_intensity * self.competition_strength * 0.5  # Base 50% reduction at max competition
        
        # Additional reduction for resistant bacteria due to lower resource efficiency
        efficiency_penalty = (1.0 - resource_efficiency) * pressure_intensity * self.competition_strength * 0.3
        
        total_fitness_reduction = base_fitness_reduction + efficiency_penalty
        
        modified_fitness = bacterium.effective_fitness * (1.0 - total_fitness_reduction)
        
        # Survival probability based on competitive fitness
        survival_prob = max(0.1, modified_fitness / bacterium.effective_fitness)
        
        return SelectionResult(
            bacterium_id=bacterium.id,
            original_fitness=bacterium.effective_fitness,
            modified_fitness=modified_fitness,
            survival_probability=survival_prob,
            pressure_effects={
                'competition_factor': competition_factor,
                'local_competition': local_competition,
                'resource_efficiency': resource_efficiency,
                'fitness_reduction': total_fitness_reduction
            }
        )
    
    def get_pressure_intensity(self, generation: int) -> float:
        """Get resource pressure intensity."""
        return self.config.intensity


class EnvironmentalPressure(SelectionPressure):
    """Selection pressure from environmental stress factors."""
    
    def __init__(self, config: PressureConfig):
        super().__init__(config)
        
        # Environmental stress parameters
        self.stress_factors = config.parameters.get('stress_factors', ['temperature', 'ph', 'osmotic'])
        self.stress_tolerance_resistant = config.parameters.get('stress_tolerance_resistant', 1.1)
        self.baseline_stress = config.parameters.get('baseline_stress', 0.1)
    
    def apply_to_bacterium(
        self, 
        bacterium: Bacterium, 
        population_context: Dict[str, Any],
        generation: int
    ) -> SelectionResult:
        """Apply environmental stress pressure."""
        stress_level = self.get_pressure_intensity(generation)
        
        # Calculate stress tolerance
        if bacterium.is_resistant:
            # Resistant bacteria might have cross-resistance to some stresses
            stress_tolerance = self.stress_tolerance_resistant
        else:
            stress_tolerance = 1.0
        
        # Calculate effective stress
        effective_stress = max(0.0, stress_level - self.baseline_stress) / stress_tolerance
        
        # Fitness reduction due to stress
        stress_impact = min(0.8, effective_stress)  # Max 80% fitness reduction
        modified_fitness = bacterium.effective_fitness * (1.0 - stress_impact)
        
        # Survival probability
        survival_prob = max(0.05, 1.0 - effective_stress)
        
        return SelectionResult(
            bacterium_id=bacterium.id,
            original_fitness=bacterium.effective_fitness,
            modified_fitness=modified_fitness,
            survival_probability=survival_prob,
            pressure_effects={
                'stress_level': stress_level,
                'stress_tolerance': stress_tolerance,
                'effective_stress': effective_stress,
                'stress_impact': stress_impact
            }
        )
    
    def get_pressure_intensity(self, generation: int) -> float:
        """Get environmental stress intensity."""
        base_stress = self.config.intensity
        
        # Environmental stress can fluctuate
        if self.config.time_profile == "sine":
            # Sinusoidal variation (e.g., daily temperature cycles)
            period = self.config.parameters.get('period', 24)
            amplitude = self.config.parameters.get('amplitude', 0.3)
            return base_stress + amplitude * np.sin(2 * np.pi * generation / period)
        
        return base_stress


class SpatialPressure(SelectionPressure):
    """Selection pressure from spatial density effects."""
    
    def __init__(self, config: PressureConfig):
        super().__init__(config)
        
        self.crowding_threshold = config.parameters.get('crowding_threshold', 3)
        self.dispersal_advantage = config.parameters.get('dispersal_advantage', 0.1)
    
    def apply_to_bacterium(
        self, 
        bacterium: Bacterium, 
        population_context: Dict[str, Any],
        generation: int
    ) -> SelectionResult:
        """Apply spatial density pressure."""
        local_density = population_context.get('local_density', 1)
        neighbor_count = population_context.get('neighbor_count', 0)
        
        # Calculate crowding effect
        crowding_factor = max(0.0, local_density - self.crowding_threshold) / self.crowding_threshold
        
        # Fitness modification based on crowding
        crowding_penalty = crowding_factor * self.get_pressure_intensity(generation)
        modified_fitness = bacterium.effective_fitness * (1.0 - crowding_penalty)
        
        # Survival probability decreases with crowding
        survival_prob = max(0.1, 1.0 - (crowding_penalty * 0.5))
        
        return SelectionResult(
            bacterium_id=bacterium.id,
            original_fitness=bacterium.effective_fitness,
            modified_fitness=modified_fitness,
            survival_probability=survival_prob,
            pressure_effects={
                'local_density': local_density,
                'neighbor_count': neighbor_count,
                'crowding_factor': crowding_factor,
                'crowding_penalty': crowding_penalty
            }
        )
    
    def get_pressure_intensity(self, generation: int) -> float:
        """Get spatial pressure intensity."""
        return self.config.intensity


class CompetitivePressure(SelectionPressure):
    """Selection pressure from direct bacteria-to-bacteria competition."""
    
    def __init__(self, config: PressureConfig):
        super().__init__(config)
        
        self.competition_model = config.parameters.get('competition_model', 'fitness_based')
        self.interaction_radius = config.parameters.get('interaction_radius', 1.0)
        self.frequency_dependent = config.parameters.get('frequency_dependent', True)
        self.dominance_factor = config.parameters.get('dominance_factor', 1.2)
    
    def apply_to_bacterium(
        self, 
        bacterium: Bacterium, 
        population_context: Dict[str, Any],
        generation: int
    ) -> SelectionResult:
        """Apply competitive pressure to bacterium."""
        competitors = population_context.get('competitors', [])
        total_population = population_context.get('total_population', 1)
        
        if not competitors or total_population <= 1:
            # No competition
            return SelectionResult(
                bacterium_id=bacterium.id,
                original_fitness=bacterium.effective_fitness,
                modified_fitness=bacterium.effective_fitness,
                survival_probability=1.0,
                pressure_effects={'competition_strength': 0.0}
            )
        
        # Calculate competitive advantage/disadvantage
        competition_strength = self._calculate_competition_strength(
            bacterium, competitors, total_population
        )
        
        # Apply frequency-dependent selection if enabled
        if self.frequency_dependent:
            resistance_frequency = self._calculate_resistance_frequency(competitors)
            frequency_effect = self._apply_frequency_dependent_selection(
                bacterium, resistance_frequency
            )
            competition_strength *= frequency_effect
        
        # Modify fitness based on competitive interactions
        pressure_intensity = self.get_pressure_intensity(generation)
        fitness_modifier = 1.0 + (competition_strength * pressure_intensity * 0.3)  # Max 30% change
        modified_fitness = bacterium.effective_fitness * max(0.1, fitness_modifier)
        
        # Survival probability based on competitive fitness
        survival_prob = max(0.1, min(1.0, fitness_modifier))
        
        return SelectionResult(
            bacterium_id=bacterium.id,
            original_fitness=bacterium.effective_fitness,
            modified_fitness=modified_fitness,
            survival_probability=survival_prob,
            pressure_effects={
                'competition_strength': competition_strength,
                'competitor_count': len(competitors),
                'fitness_modifier': fitness_modifier,
                'frequency_effect': frequency_effect if self.frequency_dependent else 1.0
            }
        )
    
    def _calculate_competition_strength(
        self, 
        bacterium: Bacterium, 
        competitors: List[Bacterium],
        total_population: int
    ) -> float:
        """Calculate competitive strength relative to other bacteria."""
        if self.competition_model == 'fitness_based':
            # Compare fitness with competitors
            competitor_fitnesses = [c.effective_fitness for c in competitors]
            avg_competitor_fitness = np.mean(competitor_fitnesses) if competitor_fitnesses else 1.0
            
            relative_fitness = bacterium.effective_fitness / max(0.1, avg_competitor_fitness)
            return (relative_fitness - 1.0)  # Positive = advantage, negative = disadvantage
            
        elif self.competition_model == 'resistance_dominance':
            # Resistant bacteria have dominance advantage
            resistant_competitors = sum(1 for c in competitors if c.is_resistant)
            
            if bacterium.is_resistant:
                # Resistant bacteria benefit when rare, suffer when common
                resistance_frequency = resistant_competitors / len(competitors)
                return self.dominance_factor * (1.0 - resistance_frequency)
            else:
                # Sensitive bacteria compete better when resistant are common
                resistance_frequency = resistant_competitors / len(competitors)
                return -0.5 * resistance_frequency
        
        return 0.0
    
    def _calculate_resistance_frequency(self, competitors: List[Bacterium]) -> float:
        """Calculate frequency of resistant bacteria in population."""
        if not competitors:
            return 0.0
        
        resistant_count = sum(1 for c in competitors if c.is_resistant)
        return resistant_count / len(competitors)
    
    def _apply_frequency_dependent_selection(
        self, 
        bacterium: Bacterium, 
        resistance_frequency: float
    ) -> float:
        """Apply frequency-dependent selection effects."""
        if bacterium.is_resistant:
            # Resistant bacteria benefit from being rare (negative frequency dependence)
            return 1.0 + (0.3 * (1.0 - resistance_frequency))
        else:
            # Sensitive bacteria benefit when resistant are common
            return 1.0 + (0.2 * resistance_frequency)
    
    def get_pressure_intensity(self, generation: int) -> float:
        """Get competitive pressure intensity."""
        return self.config.intensity


class SelectionEnvironment:
    """Manages multiple selection pressures simultaneously."""
    
    def __init__(self):
        self.pressures: List[SelectionPressure] = []
        self.interaction_effects: Dict[str, Callable] = {}
    
    def add_pressure(self, pressure: SelectionPressure) -> None:
        """Add a selection pressure to the environment."""
        self.pressures.append(pressure)
    
    def remove_pressure(self, pressure_type: PressureType) -> bool:
        """Remove all pressures of given type."""
        initial_count = len(self.pressures)
        self.pressures = [p for p in self.pressures if p.config.pressure_type != pressure_type]
        return len(self.pressures) < initial_count
    
    def apply_selection(
        self, 
        bacteria: List[Bacterium], 
        population_context: Dict[str, Any],
        generation: int
    ) -> List[SelectionResult]:
        """
        Apply all active selection pressures to a list of bacteria.
        
        Args:
            bacteria: List of bacteria to apply selection to
            population_context: Context about population state
            generation: Current generation number
        
        Returns:
            List of SelectionResult objects with cumulative effects
        """
        results = []
        
        for bacterium in bacteria:
            # Apply each active pressure
            pressure_results = []
            for pressure in self.pressures:
                if pressure.is_active(generation):
                    result = pressure.apply_to_bacterium(bacterium, population_context, generation)
                    pressure_results.append(result)
            
            # Combine effects from multiple pressures
            if pressure_results:
                combined_result = self._combine_pressure_effects(bacterium, pressure_results)
                results.append(combined_result)
            else:
                # No active pressures
                results.append(SelectionResult(
                    bacterium_id=bacterium.id,
                    original_fitness=bacterium.effective_fitness,
                    modified_fitness=bacterium.effective_fitness,
                    survival_probability=1.0,
                    pressure_effects={}
                ))
        
        # Log applications
        for pressure in self.pressures:
            if pressure.is_active(generation):
                pressure_specific_results = [r for r in results if any(
                    pressure.config.pressure_type.value in effect 
                    for effect in r.pressure_effects.keys()
                )]
                pressure.log_application(generation, pressure_specific_results)
        
        return results
    
    def _combine_pressure_effects(
        self, 
        bacterium: Bacterium, 
        pressure_results: List[SelectionResult]
    ) -> SelectionResult:
        """Combine effects from multiple selection pressures."""
        if not pressure_results:
            return SelectionResult(
                bacterium_id=bacterium.id,
                original_fitness=bacterium.effective_fitness,
                modified_fitness=bacterium.effective_fitness,
                survival_probability=1.0,
                pressure_effects={}
            )
        
        # Combine fitness effects multiplicatively
        combined_fitness = bacterium.effective_fitness
        for result in pressure_results:
            # Safeguard against division by zero
            if result.original_fitness > 0.0:
                fitness_ratio = result.modified_fitness / result.original_fitness
            else:
                # If original fitness is 0, use the modified fitness directly as multiplier
                fitness_ratio = result.modified_fitness if result.modified_fitness > 0.0 else 1.0
            combined_fitness *= fitness_ratio
        
        # Combine survival probabilities multiplicatively (independent events)
        combined_survival = 1.0
        for result in pressure_results:
            combined_survival *= result.survival_probability
        
        # Merge pressure effects
        combined_effects = {}
        for result in pressure_results:
            combined_effects.update(result.pressure_effects)
        
        return SelectionResult(
            bacterium_id=bacterium.id,
            original_fitness=bacterium.effective_fitness,
            modified_fitness=combined_fitness,
            survival_probability=combined_survival,
            pressure_effects=combined_effects
        )
    
    def get_active_pressures(self, generation: int) -> List[SelectionPressure]:
        """Get list of pressures active at given generation."""
        return [p for p in self.pressures if p.is_active(generation)]
    
    def get_pressure_summary(self, generation: int) -> Dict[str, Any]:
        """Get summary of all pressure effects at given generation."""
        active_pressures = self.get_active_pressures(generation)
        
        return {
            'generation': generation,
            'active_pressure_count': len(active_pressures),
            'pressure_types': [p.config.pressure_type.value for p in active_pressures],
            'pressure_intensities': {
                p.config.pressure_type.value: p.get_pressure_intensity(generation) 
                for p in active_pressures
            }
        } 