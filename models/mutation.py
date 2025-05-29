"""
Mutation system for bacterial evolution simulation.

This module provides classes for modeling various types of genetic mutations
including point mutations, chromosomal mutations, and fitness-affecting mutations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import numpy as np
from .bacterium import ResistanceStatus


class MutationType(Enum):
    """Types of mutations that can occur."""
    POINT = "point"
    INSERTION = "insertion"
    DELETION = "deletion"
    INVERSION = "inversion"
    DUPLICATION = "duplication"
    FITNESS = "fitness"
    RESISTANCE = "resistance"


class MutationEffect(Enum):
    """Classification of mutation effects."""
    BENEFICIAL = "beneficial"
    NEUTRAL = "neutral"
    DELETERIOUS = "deleterious"
    LETHAL = "lethal"


@dataclass
class MutationConfig:
    """Configuration for mutation parameters."""
    base_mutation_rate: float = 1e-6  # Per base per generation
    point_mutation_rate: float = 1e-6
    indel_rate: float = 1e-7  # Insertion/deletion rate
    chromosomal_mutation_rate: float = 1e-8
    fitness_mutation_rate: float = 1e-5
    resistance_mutation_rate: float = 1e-7
    
    # Environmental modifiers
    stress_mutation_multiplier: float = 10.0
    antibiotic_mutation_multiplier: float = 5.0
    
    # Effect probabilities
    beneficial_probability: float = 0.1
    neutral_probability: float = 0.7
    deleterious_probability: float = 0.2
    
    # Fitness effect ranges
    beneficial_fitness_range: tuple = (0.01, 0.2)
    deleterious_fitness_range: tuple = (-0.2, -0.01)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        total_prob = (self.beneficial_probability + 
                     self.neutral_probability + 
                     self.deleterious_probability)
        if not np.isclose(total_prob, 1.0):
            raise ValueError("Effect probabilities must sum to 1.0")


@dataclass
class Mutation:
    """Represents a single mutation event."""
    mutation_id: str
    mutation_type: MutationType
    effect: MutationEffect
    fitness_change: float = 0.0
    resistance_change: Optional[ResistanceStatus] = None
    generation_occurred: int = 0
    parent_bacterium_id: str = ""
    description: str = ""
    
    # Additional mutation-specific data
    position: Optional[int] = None  # For point mutations
    sequence_change: str = ""  # Description of sequence change
    size: int = 0  # For indels and duplications
    
    def __post_init__(self):
        """Generate description if not provided."""
        if not self.description:
            self.description = self._generate_description()
    
    def _generate_description(self) -> str:
        """Generate a human-readable description of the mutation."""
        if self.mutation_type == MutationType.POINT:
            return f"Point mutation at position {self.position}"
        elif self.mutation_type == MutationType.INSERTION:
            return f"Insertion of {self.size} bases"
        elif self.mutation_type == MutationType.DELETION:
            return f"Deletion of {self.size} bases"
        elif self.mutation_type == MutationType.FITNESS:
            change_type = "beneficial" if self.fitness_change > 0 else "deleterious"
            return f"Fitness-affecting mutation ({change_type}, Δf={self.fitness_change:.3f})"
        elif self.mutation_type == MutationType.RESISTANCE:
            return f"Resistance mutation: {self.resistance_change.value if self.resistance_change else 'unknown'}"
        else:
            return f"{self.mutation_type.value} mutation"


class MutationEngine:
    """Engine for generating and applying mutations."""
    
    def __init__(self, config: MutationConfig):
        self.config = config
        self._mutation_counter = 0
    
    def generate_mutations(
        self,
        bacterium_id: str,
        generation: int,
        environmental_factors: Optional[Dict[str, float]] = None
    ) -> List[Mutation]:
        """
        Generate mutations for a bacterium based on mutation rates and environmental factors.
        
        Args:
            bacterium_id: ID of the bacterium undergoing mutation
            generation: Current generation number
            environmental_factors: Dict of environmental factors affecting mutation rates
        
        Returns:
            List of mutations to apply
        """
        mutations = []
        factors = environmental_factors or {}
        
        # Calculate effective mutation rates based on environment
        effective_rates = self._calculate_effective_rates(factors)
        
        # Generate different types of mutations
        for mutation_type, rate in effective_rates.items():
            if random.random() < rate:
                mutation = self._create_mutation(
                    mutation_type, bacterium_id, generation
                )
                if mutation:
                    mutations.append(mutation)
        
        return mutations
    
    def _calculate_effective_rates(self, factors: Dict[str, float]) -> Dict[MutationType, float]:
        """Calculate effective mutation rates based on environmental factors."""
        base_rates = {
            MutationType.POINT: self.config.point_mutation_rate,
            MutationType.INSERTION: self.config.indel_rate,
            MutationType.DELETION: self.config.indel_rate,
            MutationType.INVERSION: self.config.chromosomal_mutation_rate,
            MutationType.DUPLICATION: self.config.chromosomal_mutation_rate,
            MutationType.FITNESS: self.config.fitness_mutation_rate,
            MutationType.RESISTANCE: self.config.resistance_mutation_rate,
        }
        
        # Apply environmental modifiers
        stress_level = factors.get('stress', 0.0)
        antibiotic_level = factors.get('antibiotic_concentration', 0.0)
        
        stress_multiplier = 1.0 + (stress_level * (self.config.stress_mutation_multiplier - 1.0))
        antibiotic_multiplier = 1.0 + (antibiotic_level * (self.config.antibiotic_mutation_multiplier - 1.0))
        
        effective_rates = {}
        for mutation_type, base_rate in base_rates.items():
            effective_rate = base_rate * stress_multiplier
            
            # Antibiotics particularly affect resistance mutations
            if mutation_type == MutationType.RESISTANCE:
                effective_rate *= antibiotic_multiplier
            
            effective_rates[mutation_type] = min(effective_rate, 1.0)  # Cap at 100%
        
        return effective_rates
    
    def _create_mutation(
        self,
        mutation_type: MutationType,
        bacterium_id: str,
        generation: int
    ) -> Optional[Mutation]:
        """Create a specific type of mutation."""
        self._mutation_counter += 1
        mutation_id = f"mut_{self._mutation_counter}"
        
        # Determine mutation effect
        effect = self._determine_effect()
        
        # Calculate fitness change based on effect
        fitness_change = self._calculate_fitness_change(effect, mutation_type)
        
        # Handle resistance changes
        resistance_change = None
        if mutation_type == MutationType.RESISTANCE:
            resistance_change = self._determine_resistance_change()
        
        # Create mutation with type-specific parameters
        mutation = Mutation(
            mutation_id=mutation_id,
            mutation_type=mutation_type,
            effect=effect,
            fitness_change=fitness_change,
            resistance_change=resistance_change,
            generation_occurred=generation,
            parent_bacterium_id=bacterium_id
        )
        
        # Add type-specific details
        if mutation_type == MutationType.POINT:
            mutation.position = random.randint(1, 1000000)  # Arbitrary genome size
        elif mutation_type in [MutationType.INSERTION, MutationType.DELETION, MutationType.DUPLICATION]:
            mutation.size = self._generate_indel_size()
        
        return mutation
    
    def _determine_effect(self) -> MutationEffect:
        """Determine the effect classification of a mutation."""
        rand = random.random()
        
        if rand < self.config.beneficial_probability:
            return MutationEffect.BENEFICIAL
        elif rand < self.config.beneficial_probability + self.config.neutral_probability:
            return MutationEffect.NEUTRAL
        else:
            return MutationEffect.DELETERIOUS
    
    def _calculate_fitness_change(self, effect: MutationEffect, mutation_type: MutationType) -> float:
        """Calculate the fitness change associated with a mutation."""
        if effect == MutationEffect.NEUTRAL:
            return 0.0
        elif effect == MutationEffect.BENEFICIAL:
            min_change, max_change = self.config.beneficial_fitness_range
            return random.uniform(min_change, max_change)
        elif effect == MutationEffect.DELETERIOUS:
            min_change, max_change = self.config.deleterious_fitness_range
            return random.uniform(min_change, max_change)
        else:  # LETHAL
            return -1.0  # Complete fitness loss
    
    def _determine_resistance_change(self) -> Optional[ResistanceStatus]:
        """Determine resistance status change for resistance mutations."""
        # For simplicity, assume 50% chance of gaining resistance, 50% losing it
        return random.choice([ResistanceStatus.RESISTANT, ResistanceStatus.SENSITIVE])
    
    def _generate_indel_size(self) -> int:
        """Generate size for insertion/deletion mutations using geometric distribution."""
        # Most indels are small, following roughly geometric distribution
        return np.random.geometric(0.5)  # Mean size of ~2 bases
    
    def apply_mutations(self, bacterium, mutations: List[Mutation]) -> Dict[str, Any]:
        """
        Apply a list of mutations to a bacterium.
        
        Args:
            bacterium: Bacterium object to modify
            mutations: List of mutations to apply
        
        Returns:
            Dictionary summarizing the changes made
        """
        changes = {
            'fitness_change': 0.0,
            'resistance_changed': False,
            'mutations_applied': len(mutations),
            'mutation_details': []
        }
        
        for mutation in mutations:
            # Apply fitness changes
            if mutation.fitness_change != 0:
                bacterium.fitness = max(0.0, bacterium.fitness + mutation.fitness_change)
                changes['fitness_change'] += mutation.fitness_change
            
            # Apply resistance changes
            if mutation.resistance_change is not None:
                bacterium.resistance_status = mutation.resistance_change
                changes['resistance_changed'] = True
                # Update resistance-related properties
                if mutation.resistance_change == ResistanceStatus.RESISTANT:
                    bacterium._survival_bonus = 0.1
                else:
                    bacterium._survival_bonus = 0.0
            
            changes['mutation_details'].append({
                'id': mutation.mutation_id,
                'type': mutation.mutation_type.value,
                'effect': mutation.effect.value,
                'fitness_change': mutation.fitness_change,
                'description': mutation.description
            })
        
        return changes


class MutationTracker:
    """Tracks mutations across generations for phylogenetic analysis."""
    
    def __init__(self):
        self.mutation_history: Dict[str, List[Mutation]] = {}
        self.lineage_mutations: Dict[str, List[str]] = {}  # bacterium_id -> mutation_ids
    
    def record_mutations(self, bacterium_id: str, mutations: List[Mutation]) -> None:
        """Record mutations for a bacterium."""
        if bacterium_id not in self.mutation_history:
            self.mutation_history[bacterium_id] = []
        
        self.mutation_history[bacterium_id].extend(mutations)
        
        # Track lineage
        if bacterium_id not in self.lineage_mutations:
            self.lineage_mutations[bacterium_id] = []
        
        mutation_ids = [m.mutation_id for m in mutations]
        self.lineage_mutations[bacterium_id].extend(mutation_ids)
    
    def get_lineage_mutations(self, bacterium_id: str) -> List[Mutation]:
        """Get all mutations in a bacterium's lineage."""
        return self.mutation_history.get(bacterium_id, [])
    
    def get_mutation_statistics(self) -> Dict[str, Any]:
        """Get statistics about mutations across all bacteria."""
        all_mutations = [m for mutations in self.mutation_history.values() for m in mutations]
        
        if not all_mutations:
            return {'total_mutations': 0}
        
        type_counts = {}
        effect_counts = {}
        
        for mutation in all_mutations:
            mut_type = mutation.mutation_type.value
            mut_effect = mutation.effect.value
            
            type_counts[mut_type] = type_counts.get(mut_type, 0) + 1
            effect_counts[mut_effect] = effect_counts.get(mut_effect, 0) + 1
        
        return {
            'total_mutations': len(all_mutations),
            'mutation_types': type_counts,
            'mutation_effects': effect_counts,
            'average_fitness_change': np.mean([m.fitness_change for m in all_mutations]),
            'beneficial_mutations': sum(1 for m in all_mutations if m.effect == MutationEffect.BENEFICIAL),
            'deleterious_mutations': sum(1 for m in all_mutations if m.effect == MutationEffect.DELETERIOUS)
        } 