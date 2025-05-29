"""
Population class for managing collections of bacteria.
"""

import random
import uuid
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import numpy as np

from .bacterium import Bacterium, ResistanceStatus, Position


@dataclass
class PopulationStats:
    """Statistics for tracking population metrics."""
    total_count: int = 0
    resistant_count: int = 0
    sensitive_count: int = 0
    average_age: float = 0.0
    average_fitness: float = 0.0
    resistance_frequency: float = 0.0
    generation: int = 0
    
    @property
    def resistance_percentage(self) -> float:
        """Get resistance frequency as percentage."""
        return self.resistance_frequency * 100


@dataclass  
class PopulationConfig:
    """Configuration for population initialization."""
    
    # Basic population parameters
    population_size: int = 10000
    initial_resistance_frequency: float = 0.01  # 1% resistant initially
    
    # Spatial parameters
    use_spatial: bool = True
    grid_width: int = 100
    grid_height: int = 100
    max_bacteria_per_cell: int = 5
    
    # Fitness parameters
    base_fitness_range: Tuple[float, float] = (0.8, 1.2)
    resistant_fitness_modifier: float = 0.9  # Resistance cost
    
    # Randomization
    random_seed: Optional[int] = None
    
    # Age distribution
    initial_age_range: Tuple[int, int] = (0, 3)
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.population_size < 0:
            raise ValueError("Population size must be non-negative")
        if not 0 <= self.initial_resistance_frequency <= 1:
            raise ValueError("Resistance frequency must be between 0 and 1")
        if self.use_spatial and (self.grid_width <= 0 or self.grid_height <= 0):
            raise ValueError("Grid dimensions must be positive")


class Population:
    """
    Manages a population of bacteria with spatial and non-spatial support.
    
    Provides methods for initialization, tracking, modification, and analysis
    of bacterial populations in evolution simulations.
    """
    
    def __init__(self, config: PopulationConfig):
        """
        Initialize population manager.
        
        Args:
            config: Population configuration parameters
        """
        self.config = config
        self.bacteria: List[Bacterium] = []
        self.generation = 0
        self._next_id = 0
        
        # Spatial tracking (if enabled)
        self.spatial_grid: Optional[Dict[Tuple[int, int], List[Bacterium]]] = None
        if config.use_spatial:
            self.spatial_grid = {}
        
        # Statistics tracking
        self.stats_history: List[PopulationStats] = []
        
        # Set random seed if provided
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
    
    def _generate_id(self) -> str:
        """Generate unique bacterium ID."""
        self._next_id += 1
        return f"bact_{self._next_id:06d}"
    
    def initialize_population(self) -> None:
        """
        Create initial population of bacteria according to configuration.
        """
        self.bacteria.clear()
        if self.spatial_grid:
            self.spatial_grid.clear()
        
        # Handle empty population case
        if self.config.population_size == 0:
            self._update_statistics()
            print("✅ Initialized empty population")
            return
        
        # Calculate how many resistant bacteria to create
        num_resistant = int(self.config.population_size * self.config.initial_resistance_frequency)
        num_sensitive = self.config.population_size - num_resistant
        
        # Create sensitive bacteria
        for _ in range(num_sensitive):
            bacterium = self._create_bacterium(ResistanceStatus.SENSITIVE)
            self.add_bacterium(bacterium)
        
        # Create resistant bacteria  
        for _ in range(num_resistant):
            bacterium = self._create_bacterium(ResistanceStatus.RESISTANT)
            self.add_bacterium(bacterium)
        
        # Shuffle to randomize positions
        random.shuffle(self.bacteria)
        
        # Update statistics
        self._update_statistics()
        
        print(f"✅ Initialized population: {self.config.population_size} bacteria "
              f"({num_resistant} resistant, {num_sensitive} sensitive)")
    
    def _create_bacterium(self, resistance_status: ResistanceStatus) -> Bacterium:
        """
        Create a single bacterium with randomized properties.
        
        Args:
            resistance_status: Whether bacterium should be resistant or sensitive
            
        Returns:
            New Bacterium instance
        """
        # Generate random fitness within specified range
        base_fitness = random.uniform(*self.config.base_fitness_range)
        
        # Apply resistance modifier
        if resistance_status == ResistanceStatus.RESISTANT:
            fitness = base_fitness * self.config.resistant_fitness_modifier
        else:
            fitness = base_fitness
        
        # Random initial age
        age = random.randint(*self.config.initial_age_range)
        
        # Random position (if spatial)
        position = None
        if self.config.use_spatial:
            position = self._find_available_position()
        
        bacterium = Bacterium(
            id=self._generate_id(),
            resistance_status=resistance_status,
            age=age,
            fitness=fitness,
            position=position,
            generation_born=self.generation
        )
        
        return bacterium
    
    def _find_available_position(self) -> Position:
        """
        Find an available position in the spatial grid.
        
        Returns:
            Available Position
        """
        max_attempts = 1000
        attempts = 0
        
        while attempts < max_attempts:
            x = random.randint(0, self.config.grid_width - 1)
            y = random.randint(0, self.config.grid_height - 1)
            position = Position(x, y)
            
            # Check if position is available
            if self._is_position_available(position):
                return position
            
            attempts += 1
        
        # If no position found after many attempts, allow overcrowding
        x = random.randint(0, self.config.grid_width - 1)
        y = random.randint(0, self.config.grid_height - 1)
        return Position(x, y)
    
    def _is_position_available(self, position: Position) -> bool:
        """
        Check if a position has space for another bacterium.
        
        Args:
            position: Position to check
            
        Returns:
            True if position is available
        """
        if not self.spatial_grid:
            return True
        
        grid_key = (position.x, position.y)
        bacteria_at_position = self.spatial_grid.get(grid_key, [])
        return len(bacteria_at_position) < self.config.max_bacteria_per_cell
    
    def add_bacterium(self, bacterium: Bacterium) -> bool:
        """
        Add a bacterium to the population.
        
        Args:
            bacterium: Bacterium to add
            
        Returns:
            True if successfully added
        """
        # Add to main list
        self.bacteria.append(bacterium)
        
        # Add to spatial grid if applicable
        if self.spatial_grid is not None and bacterium.position:
            grid_key = (bacterium.position.x, bacterium.position.y)
            if grid_key not in self.spatial_grid:
                self.spatial_grid[grid_key] = []
            self.spatial_grid[grid_key].append(bacterium)
        
        return True
    
    def remove_bacterium(self, bacterium: Bacterium) -> bool:
        """
        Remove a bacterium from the population.
        
        Args:
            bacterium: Bacterium to remove
            
        Returns:
            True if successfully removed
        """
        try:
            # Remove from main list
            self.bacteria.remove(bacterium)
            
            # Remove from spatial grid if applicable
            if self.spatial_grid is not None and bacterium.position:
                grid_key = (bacterium.position.x, bacterium.position.y)
                if grid_key in self.spatial_grid:
                    if bacterium in self.spatial_grid[grid_key]:
                        self.spatial_grid[grid_key].remove(bacterium)
                    if not self.spatial_grid[grid_key]:  # Remove empty lists
                        del self.spatial_grid[grid_key]
            
            return True
        except ValueError:
            return False
    
    def get_bacteria_at_position(self, position: Position) -> List[Bacterium]:
        """
        Get all bacteria at a specific position.
        
        Args:
            position: Position to check
            
        Returns:
            List of bacteria at the position
        """
        if not self.spatial_grid:
            return []
        
        grid_key = (position.x, position.y)
        return self.spatial_grid.get(grid_key, []).copy()
    
    def get_neighbors(self, bacterium: Bacterium, radius: float = 1.0) -> List[Bacterium]:
        """
        Get neighboring bacteria within specified radius.
        
        Args:
            bacterium: Central bacterium
            radius: Search radius
            
        Returns:
            List of neighboring bacteria
        """
        if not bacterium.position or not self.spatial_grid:
            return []
        
        neighbors = []
        search_range = int(radius) + 1
        
        for dx in range(-search_range, search_range + 1):
            for dy in range(-search_range, search_range + 1):
                if dx == 0 and dy == 0:
                    continue
                
                x = bacterium.position.x + dx
                y = bacterium.position.y + dy
                
                # Check bounds
                if 0 <= x < self.config.grid_width and 0 <= y < self.config.grid_height:
                    position = Position(x, y)
                    if bacterium.position.distance_to(position) <= radius:
                        neighbors.extend(self.get_bacteria_at_position(position))
        
        return neighbors
    
    def get_statistics(self) -> PopulationStats:
        """
        Calculate current population statistics.
        
        Returns:
            PopulationStats object with current metrics
        """
        if not self.bacteria:
            return PopulationStats()
        
        total_count = len(self.bacteria)
        resistant_count = sum(1 for b in self.bacteria if b.is_resistant)
        sensitive_count = total_count - resistant_count
        
        average_age = sum(b.age for b in self.bacteria) / total_count
        average_fitness = sum(b.effective_fitness for b in self.bacteria) / total_count
        resistance_frequency = resistant_count / total_count if total_count > 0 else 0.0
        
        return PopulationStats(
            total_count=total_count,
            resistant_count=resistant_count,
            sensitive_count=sensitive_count,
            average_age=average_age,
            average_fitness=average_fitness,
            resistance_frequency=resistance_frequency,
            generation=self.generation
        )
    
    def _update_statistics(self) -> None:
        """Update and store population statistics."""
        stats = self.get_statistics()
        self.stats_history.append(stats)
    
    def advance_generation(self) -> None:
        """
        Advance the simulation by one generation.

        This involves:
        1. Aging all bacteria.
        2. Determining survival based on individual probabilities.
        3. Updating population statistics.
        """
        self.generation += 1
        
        survivors: List[Bacterium] = []
        for bacterium in self.bacteria:
            # Placeholder for antibiotic concentration and environmental stress
            # These should eventually be configurable or dynamic
            antibiotic_concentration = 0.0 
            environmental_stress = 0.0
            resistance_cost = 0.1 # Example cost, should be configurable

            survival_prob = bacterium.calculate_survival_probability(
                antibiotic_concentration=antibiotic_concentration,
                resistance_cost=resistance_cost,
                environmental_stress=environmental_stress
            )
            
            if random.random() < survival_prob:
                bacterium.age_one_generation()
                survivors.append(bacterium)
        
        self.bacteria = survivors
        
        # Reproduction step
        offspring_list: List[Bacterium] = []
        for parent in self.bacteria: # Iterate over survivors
            # Placeholder for mutation rate, should be configurable
            mutation_rate = 0.001 
            # Placeholder for resource availability, should be dynamic or configurable
            resource_availability = 1.0 

            if parent.can_reproduce(resource_availability=resource_availability):
                offspring = parent.reproduce(
                    mutation_rate=mutation_rate,
                    generation=self.generation,
                    next_id_generator=self._generate_id # Pass the ID generator
                )
                if offspring:
                    offspring_list.append(offspring)

        # Add offspring to the population and spatial grid
        for child in offspring_list:
            self.add_bacterium(child) # This method handles spatial grid update

        # Update spatial grid if enabled - This part might be redundant if add_bacterium handles it
        # Re-check add_bacterium implementation. For now, let's assume add_bacterium handles it.
        # if self.config.use_spatial and self.spatial_grid is not None:
        #     self.spatial_grid.clear()
        #     for bacterium_item in self.bacteria: # self.bacteria now includes offspring
        #         if bacterium_item.position:
        #             pos_tuple = (bacterium_item.position.x, bacterium_item.position.y)
        #             if pos_tuple not in self.spatial_grid:
        #                 self.spatial_grid[pos_tuple] = []
        #             if len(self.spatial_grid[pos_tuple]) < self.config.max_bacteria_per_cell:
        #                 self.spatial_grid[pos_tuple].append(bacterium_item)

        self._update_statistics()
        print(f"Advanced to generation {self.generation}. Survivors: {len(survivors)}, Offspring: {len(offspring_list)}, Total: {len(self.bacteria)}")
    
    def get_random_sample(self, sample_size: int) -> List[Bacterium]:
        """
        Get a random sample of bacteria from the population.
        
        Args:
            sample_size: Number of bacteria to sample
            
        Returns:
            Random sample of bacteria
        """
        if sample_size >= len(self.bacteria):
            return self.bacteria.copy()
        
        return random.sample(self.bacteria, sample_size)
    
    def filter_bacteria(self, predicate: Callable[[Bacterium], bool]) -> List[Bacterium]:
        """
        Filter bacteria based on a predicate function.
        
        Args:
            predicate: Function that returns True for bacteria to include
            
        Returns:
            List of bacteria matching the predicate
        """
        return [b for b in self.bacteria if predicate(b)]
    
    def get_resistant_bacteria(self) -> List[Bacterium]:
        """Get all resistant bacteria."""
        return self.filter_bacteria(lambda b: b.is_resistant)
    
    def get_sensitive_bacteria(self) -> List[Bacterium]:
        """Get all sensitive bacteria."""
        return self.filter_bacteria(lambda b: not b.is_resistant)
    
    def get_reproducible_bacteria(self) -> List[Bacterium]:
        """Get bacteria that can reproduce."""
        return self.filter_bacteria(lambda b: b.can_reproduce())
    
    def clone_bacterium(self, bacterium: Bacterium) -> Bacterium:
        """
        Create a clone of a bacterium with a new ID.
        
        Args:
            bacterium: Bacterium to clone
            
        Returns:
            Cloned bacterium
        """
        position = None
        if bacterium.position and self.config.use_spatial:
            position = self._find_available_position()
        
        clone = Bacterium(
            id=self._generate_id(),
            resistance_status=bacterium.resistance_status,
            age=bacterium.age,
            fitness=bacterium.fitness,
            position=position,
            generation_born=self.generation,
            parent_id=bacterium.id
        )
        
        return clone
    
    def reset_population(self) -> None:
        """Reset population to empty state."""
        self.bacteria.clear()
        if self.spatial_grid:
            self.spatial_grid.clear()
        self.stats_history.clear()
        self.generation = 0
        self._next_id = 0
    
    def reset(self) -> None:
        """Alias for reset_population() for compatibility."""
        self.reset_population()
        # Re-initialize the population with original config
        self.initialize_population()
    
    def export_population_data(self) -> Dict[str, Any]:
        """
        Export population data for analysis or saving.
        
        Returns:
            Dictionary containing population data
        """
        return {
            'generation': self.generation,
            'current_population_size': len(self.bacteria),
            'config': {
                'population_size': self.config.population_size,
                'initial_resistance_frequency': self.config.initial_resistance_frequency,
                'use_spatial': self.config.use_spatial,
                'grid_dimensions': (self.config.grid_width, self.config.grid_height),
            },
            'stats_history': [
                {
                    'generation': stats.generation,
                    'total_count': stats.total_count,
                    'resistant_count': stats.resistant_count,
                    'resistance_frequency': stats.resistance_frequency,
                    'average_fitness': stats.average_fitness,
                    'average_age': stats.average_age,
                }
                for stats in self.stats_history
            ],
            'bacteria_data': [
                {
                    'id': b.id,
                    'is_resistant': b.is_resistant,
                    'age': b.age,
                    'fitness': b.fitness,
                    'position': (b.position.x, b.position.y) if b.position else None,
                }
                for b in self.bacteria
            ]
        }
    
    @property
    def size(self) -> int:
        """Get current population size."""
        return len(self.bacteria)
    
    @property
    def resistance_frequency(self) -> float:
        """Get current resistance frequency."""
        if not self.bacteria:
            return 0.0
        return sum(1 for b in self.bacteria if b.is_resistant) / len(self.bacteria)
    
    def __len__(self) -> int:
        """Support len() function."""
        return len(self.bacteria)
    
    def __iter__(self):
        """Support iteration over bacteria."""
        return iter(self.bacteria)
    
    def __str__(self) -> str:
        """String representation of population."""
        stats = self.get_statistics()
        return (f"Population(size={stats.total_count}, resistant={stats.resistant_count}, "
                f"sensitive={stats.sensitive_count}, gen={self.generation})")
    
    def __repr__(self) -> str:
        """Detailed representation of population."""
        return (f"Population(size={len(self.bacteria)}, generation={self.generation}, "
                f"spatial={'enabled' if self.config.use_spatial else 'disabled'})") 