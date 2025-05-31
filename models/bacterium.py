"""
Bacterium class for individual bacterial cells in the simulation.
"""

from typing import Optional, Tuple, Dict, TYPE_CHECKING
import random
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from .mutation import MutationEngine


class ResistanceStatus(Enum):
    """Enum for bacterial resistance status."""
    SENSITIVE = "sensitive"
    RESISTANT = "resistant"


@dataclass
class Position:
    """2D position for spatial simulations."""
    x: int
    y: int
    
    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
    
    def is_adjacent(self, other: 'Position') -> bool:
        """Check if position is adjacent (8-neighborhood)."""
        return abs(self.x - other.x) <= 1 and abs(self.y - other.y) <= 1 and self != other


@dataclass
class Bacterium:
    """
    Individual bacterium with properties and behaviors for evolution simulation.
    
    Attributes:
        id: Unique identifier for the bacterium
        resistance_status: Whether the bacterium is resistant or sensitive
        age: Current age in generations
        fitness: Base reproductive fitness
        position: Spatial position (optional for non-spatial simulations)
        generation_born: Generation when bacterium was created
        parent_id: ID of parent bacterium (for tracking lineage)
    """
    
    id: str
    resistance_status: ResistanceStatus = ResistanceStatus.SENSITIVE
    age: int = 0
    fitness: float = 1.0
    position: Optional[Position] = None
    generation_born: int = 0
    parent_id: Optional[str] = None
    
    # Internal tracking
    _reproduction_attempts: int = field(default=0, init=False)
    _survival_bonus: float = field(default=0.0, init=False)
    
    def __post_init__(self):
        """Initialize derived properties after creation."""
        if self.resistance_status == ResistanceStatus.RESISTANT:
            # Apply resistance cost to fitness
            self._survival_bonus = 0.1  # Resistance provides survival advantage with antibiotics
    
    @property
    def is_resistant(self) -> bool:
        """Check if bacterium is antibiotic resistant."""
        return self.resistance_status == ResistanceStatus.RESISTANT
    
    def is_alive(self) -> bool:
        """
        Check if bacterium is alive.
        
        For this implementation, we consider bacteria alive if they have positive fitness
        and are within reasonable age limits. More complex mortality could be added later.
        
        Returns:
            True if bacterium is alive
        """
        # Basic viability checks
        if self.fitness <= 0:
            return False
        
        # Age limit (bacteria don't live forever)
        max_age = 50  # Arbitrary maximum age in generations
        if self.age > max_age:
            return False
        
        return True
    
    @property
    def effective_fitness(self) -> float:
        """
        Calculate effective fitness considering age and other factors.
        Fitness decreases slightly with age to model cellular aging.
        """
        age_factor = max(0.5, 1.0 - (self.age * 0.01))  # 1% fitness loss per generation
        return self.fitness * age_factor
    
    def age_one_generation(self) -> None:
        """Age the bacterium by one generation."""
        self.age += 1
    
    def calculate_survival_probability(
        self, 
        antibiotic_concentration: float = 0.0,
        resistance_cost: float = 0.1,
        environmental_stress: float = 0.0
    ) -> float:
        """
        Calculate probability of surviving to next generation.
        
        Args:
            antibiotic_concentration: Level of antibiotic pressure (0.0 to 10.0)
            resistance_cost: Fitness cost of resistance (0.0 to 1.0)
            environmental_stress: Additional environmental pressure (0.0 to 1.0)
        
        Returns:
            Probability of survival (0.0 to 1.0)
        """
        base_survival = self.effective_fitness
        
        # Apply antibiotic pressure
        if antibiotic_concentration > 0:
            if self.is_resistant:
                # Resistant bacteria survive better with antibiotics
                antibiotic_effect = 1.0 + (self._survival_bonus * antibiotic_concentration)
                # But pay fitness cost
                resistance_penalty = 1.0 - resistance_cost
                base_survival *= antibiotic_effect * resistance_penalty
            else:
                # Sensitive bacteria are severely affected by antibiotics
                antibiotic_mortality = 1.0 - min(0.95, antibiotic_concentration * 0.8)
                base_survival *= antibiotic_mortality
        else:
            # Without antibiotics, resistant bacteria pay cost without benefit
            if self.is_resistant:
                base_survival *= (1.0 - resistance_cost)
        
        # Apply environmental stress
        base_survival *= (1.0 - environmental_stress)
        
        # Age-related mortality increases survival difficulty
        age_mortality = 1.0 - (self.age * 0.005)  # 0.5% additional mortality per generation
        base_survival *= max(0.1, age_mortality)
        
        return max(0.0, min(1.0, base_survival))
    
    def can_reproduce(
        self, 
        min_age: int = 1,
        max_reproduction_attempts: int = 5,
        resource_availability: float = 1.0
    ) -> bool:
        """
        Determine if bacterium can reproduce.
        
        Args:
            min_age: Minimum age required for reproduction
            max_reproduction_attempts: Maximum number of reproduction attempts
            resource_availability: Available resources (0.0 to 1.0)
        
        Returns:
            True if bacterium can reproduce
        """
        if self.age < min_age:
            return False
        
        if self._reproduction_attempts >= max_reproduction_attempts:
            return False
        
        # Resource availability affects reproduction probability
        if resource_availability < 0.5:
            return random.random() < resource_availability
        
        return True
    
    def reproduce(
        self, 
        mutation_rate: float = 0.001,
        generation: int = 0,
        next_id_generator=None,
        mutation_engine: Optional['MutationEngine'] = None,
        environmental_factors: Optional[Dict[str, float]] = None
    ) -> Optional['Bacterium']:
        """
        Create offspring bacterium with potential mutations.
        
        Args:
            mutation_rate: Probability of mutation occurring (for legacy simple mutations)
            generation: Current generation number
            next_id_generator: Function to generate unique IDs
            mutation_engine: MutationEngine instance for advanced mutations (optional)
            environmental_factors: Dict of environmental factors affecting mutations
        
        Returns:
            New Bacterium instance or None if reproduction fails
        """
        if not self.can_reproduce():
            return None
        
        self._reproduction_attempts += 1
        
        # Generate new ID
        if next_id_generator:
            offspring_id = next_id_generator()
        else:
            offspring_id = f"{self.id}_offspring_{self._reproduction_attempts}"
        
        # Inherit parent traits
        offspring_resistance = self.resistance_status
        offspring_fitness = self.fitness
        
        # Create offspring near parent (for spatial simulations)
        offspring_position = None
        if self.position:
            # Place offspring in adjacent cell
            adjacent_positions = [
                Position(self.position.x + dx, self.position.y + dy)
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                if not (dx == 0 and dy == 0)
            ]
            offspring_position = random.choice(adjacent_positions)
        
        # Create offspring first
        offspring = Bacterium(
            id=offspring_id,
            resistance_status=offspring_resistance,
            age=0,
            fitness=offspring_fitness,
            position=offspring_position,
            generation_born=generation,
            parent_id=self.id
        )
        
        # Apply mutations using new system if available
        if mutation_engine is not None:
            # Use advanced mutation system - check if it has the required methods
            if hasattr(mutation_engine, 'generate_mutations') and hasattr(mutation_engine, 'apply_mutations'):
                mutations = mutation_engine.generate_mutations(
                    self.id, generation, environmental_factors
                )
                if mutations:
                    mutation_engine.apply_mutations(offspring, mutations)
        else:
            # Fall back to legacy simple mutation system
            if random.random() < mutation_rate:
                if self.resistance_status == ResistanceStatus.SENSITIVE:
                    # Mutation to resistance
                    offspring.resistance_status = ResistanceStatus.RESISTANT
                    # Apply resistance-related properties
                    offspring._survival_bonus = 0.1
                    # Slightly reduced fitness due to new resistance mechanism
                    offspring.fitness *= 0.95
                else:
                    # Resistant bacteria can mutate to become more or less fit
                    fitness_change = random.uniform(-0.1, 0.1)
                    offspring.fitness = max(0.1, offspring.fitness + fitness_change)
        
        return offspring
    
    def update_position(self, new_position: Position) -> None:
        """Update bacterium position for spatial simulations."""
        self.position = new_position
    
    def get_neighbors_in_radius(self, other_bacteria: list['Bacterium'], radius: float = 1.0) -> list['Bacterium']:
        """
        Get neighboring bacteria within specified radius.
        Used for horizontal gene transfer and local interactions.
        """
        if not self.position:
            return []
        
        neighbors = []
        for bacterium in other_bacteria:
            if bacterium != self and bacterium.position:
                if self.position.distance_to(bacterium.position) <= radius:
                    neighbors.append(bacterium)
        
        return neighbors
    
    def attempt_horizontal_gene_transfer(
        self, 
        neighbors: list['Bacterium'], 
        hgt_rate: float = 0.01
    ) -> bool:
        """
        Attempt horizontal gene transfer with neighboring bacteria.
        
        Args:
            neighbors: List of neighboring bacteria
            hgt_rate: Probability of HGT occurring
        
        Returns:
            True if HGT occurred and resistance status changed
        """
        if not neighbors or random.random() > hgt_rate:
            return False
        
        # Find resistant neighbors if this bacterium is sensitive
        if not self.is_resistant:
            resistant_neighbors = [b for b in neighbors if b.is_resistant]
            if resistant_neighbors:
                # Transfer resistance from a random resistant neighbor
                self.resistance_status = ResistanceStatus.RESISTANT
                # Apply slight fitness cost for acquiring resistance
                self.fitness *= 0.98
                self._survival_bonus = 0.1
                return True
        
        return False
    
    def __str__(self) -> str:
        """String representation of bacterium."""
        pos_str = f"({self.position.x},{self.position.y})" if self.position else "No position"
        return (f"Bacterium {self.id}: {self.resistance_status.value}, "
                f"age={self.age}, fitness={self.fitness:.3f}, pos={pos_str}")
    
    def __repr__(self) -> str:
        """Detailed representation of bacterium."""
        return (f"Bacterium(id='{self.id}', resistance_status={self.resistance_status}, "
                f"age={self.age}, fitness={self.fitness}, position={self.position})") 