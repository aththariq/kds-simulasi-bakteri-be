"""
Population class for managing collections of bacteria with performance optimizations.
"""

import random
import uuid
from typing import List, Dict, Optional, Tuple, Callable, Any, Set
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

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
    
    # Performance optimization flags
    enable_optimizations: bool = True
    batch_size: int = 1000  # Size for batch operations
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.population_size < 0:
            raise ValueError("Population size must be non-negative")
        if not 0 <= self.initial_resistance_frequency <= 1:
            raise ValueError("Resistance frequency must be between 0 and 1")
        if self.use_spatial and (self.grid_width <= 0 or self.grid_height <= 0):
            raise ValueError("Grid dimensions must be positive")


class OptimizedPopulation:
    """
    High-performance population manager with optimized data structures.
    
    Uses indexed collections for O(1) lookups and incremental statistics
    for better performance with large bacterial populations.
    """
    
    def __init__(self, config: PopulationConfig):
        """
        Initialize optimized population manager.
        
        Args:
            config: Population configuration parameters
        """
        self.config = config
        self.generation = 0
        self._next_id = 0
        
        # OPTIMIZATION: Use indexed collections instead of lists
        self.bacteria_by_id: Dict[str, Bacterium] = {}  # O(1) lookup by ID
        self.resistant_bacteria: Set[str] = set()  # O(1) resistance filtering
        self.sensitive_bacteria: Set[str] = set()  # O(1) sensitivity filtering
        
        # Spatial indexing (if enabled)
        self.spatial_grid: Optional[Dict[Tuple[int, int], Set[str]]] = None
        self.position_index: Dict[str, Position] = {}  # O(1) position lookup
        if config.use_spatial:
            self.spatial_grid = defaultdict(set)
        
        # OPTIMIZATION: Incremental statistics tracking
        self._cached_stats = PopulationStats()
        self._stats_dirty = True
        self.stats_history: List[PopulationStats] = []
        
        # Performance tracking
        self._operation_count = 0
        self._batch_updates_pending = []
        
        # Set random seed if provided
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
    
    @property
    def bacteria(self) -> List[Bacterium]:
        """Get all bacteria as list (for backward compatibility)."""
        return list(self.bacteria_by_id.values())
    
    def _generate_id(self) -> str:
        """Generate unique bacterium ID."""
        self._next_id += 1
        return f"bact_{self._next_id:06d}"
    
    def initialize_population(self) -> None:
        """
        Create initial population of bacteria according to configuration.
        """
        self.clear()
        
        # Handle empty population case
        if self.config.population_size == 0:
            self._update_statistics()
            print("✅ Initialized empty population")
            return
        
        # Calculate how many resistant bacteria to create
        num_resistant = int(self.config.population_size * self.config.initial_resistance_frequency)
        num_sensitive = self.config.population_size - num_resistant
        
        # OPTIMIZATION: Batch create bacteria to reduce overhead
        bacteria_to_add = []
        
        # Create sensitive bacteria
        for _ in range(num_sensitive):
            bacterium = self._create_bacterium(ResistanceStatus.SENSITIVE)
            bacteria_to_add.append(bacterium)
        
        # Create resistant bacteria  
        for _ in range(num_resistant):
            bacterium = self._create_bacterium(ResistanceStatus.RESISTANT)
            bacteria_to_add.append(bacterium)
        
        # Shuffle to randomize positions
        random.shuffle(bacteria_to_add)
        
        # Batch add all bacteria
        self._batch_add_bacteria(bacteria_to_add)
        
        # Update statistics
        self._update_statistics()
        
        print(f"✅ Initialized optimized population: {self.config.population_size} bacteria "
              f"({num_resistant} resistant, {num_sensitive} sensitive)")
    
    def _batch_add_bacteria(self, bacteria_list: List[Bacterium]) -> None:
        """
        Add multiple bacteria efficiently in batch operation.
        
        Args:
            bacteria_list: List of bacteria to add
        """
        for bacterium in bacteria_list:
            # Add to main index
            self.bacteria_by_id[bacterium.id] = bacterium
            
            # Add to resistance index
            if bacterium.is_resistant:
                self.resistant_bacteria.add(bacterium.id)
            else:
                self.sensitive_bacteria.add(bacterium.id)
            
            # Add to spatial index if applicable
            if self.config.use_spatial and bacterium.position:
                pos_key = (bacterium.position.x, bacterium.position.y)
                self.spatial_grid[pos_key].add(bacterium.id)
                self.position_index[bacterium.id] = bacterium.position
        
        self._stats_dirty = True
    
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
        Find an available position in the spatial grid using object pool.
        
        Returns:
            Available Position
        """
        max_attempts = 1000
        attempts = 0
        
        while attempts < max_attempts:
            x = random.randint(0, self.config.grid_width - 1)
            y = random.randint(0, self.config.grid_height - 1)
            position = Position.create(x, y)
            
            # Check if position is available
            if self._is_position_available(position):
                return position
            
            attempts += 1
        
        # If no position found after many attempts, allow overcrowding
        x = random.randint(0, self.config.grid_width - 1)
        y = random.randint(0, self.config.grid_height - 1)
        return Position.create(x, y)
    
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
        bacteria_at_position = self.spatial_grid.get(grid_key, set())
        return len(bacteria_at_position) < self.config.max_bacteria_per_cell
    
    def add_bacterium(self, bacterium: Bacterium) -> bool:
        """
        Add a bacterium to the population with optimized indexing.
        
        Args:
            bacterium: Bacterium to add
            
        Returns:
            True if successfully added
        """
        # Add to main index
        self.bacteria_by_id[bacterium.id] = bacterium
        
        # Update resistance indices
        if bacterium.is_resistant:
            self.resistant_bacteria.add(bacterium.id)
        else:
            self.sensitive_bacteria.add(bacterium.id)
        
        # Add to spatial grid if applicable
        if self.spatial_grid is not None and bacterium.position:
            grid_key = (bacterium.position.x, bacterium.position.y)
            self.spatial_grid[grid_key].add(bacterium.id)
            self.position_index[bacterium.id] = bacterium.position
        
        # Mark stats as dirty for incremental update
        self._stats_dirty = True
        
        return True
    
    def remove_bacterium(self, bacterium: Bacterium) -> bool:
        """
        Remove a bacterium from the population with optimized cleanup.
        
        Args:
            bacterium: Bacterium to remove
            
        Returns:
            True if successfully removed
        """
        try:
            # Remove from main index
            del self.bacteria_by_id[bacterium.id]
            
            # Remove from resistance indices
            self.resistant_bacteria.discard(bacterium.id)
            self.sensitive_bacteria.discard(bacterium.id)
            
            # Remove from spatial indices if applicable
            if self.spatial_grid is not None and bacterium.position:
                grid_key = (bacterium.position.x, bacterium.position.y)
                if grid_key in self.spatial_grid:
                    self.spatial_grid[grid_key].discard(bacterium.id)
                    if not self.spatial_grid[grid_key]:
                        del self.spatial_grid[grid_key]
                
                # Remove from position index
                self.position_index.pop(bacterium.id, None)
            
            # Mark stats as dirty for incremental update
            self._stats_dirty = True
            
            return True
        except KeyError:
            return False
    
    def clear(self) -> None:
        """Clear all data structures efficiently."""
        self.bacteria_by_id.clear()
        self.resistant_bacteria.clear()
        self.sensitive_bacteria.clear()
        if self.spatial_grid:
            self.spatial_grid.clear()
        self.position_index.clear()
        self._stats_dirty = True
    
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
        return [self.bacteria_by_id[id] for id in self.spatial_grid.get(grid_key, set())]
    
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
        Get current population statistics (cached for performance).
        
        Returns:
            PopulationStats object with current metrics
        """
        if self._stats_dirty:
            self._cached_stats = self._calculate_fresh_statistics()
            self._stats_dirty = False
        
        return self._cached_stats
    
    def _update_statistics(self) -> None:
        """Update and store population statistics using incremental approach."""
        if self._stats_dirty:
            self._cached_stats = self._calculate_fresh_statistics()
            self._stats_dirty = False
        
        self.stats_history.append(self._cached_stats)
    
    def _calculate_fresh_statistics(self) -> PopulationStats:
        """Calculate statistics from scratch when needed."""
        if not self.bacteria_by_id:
            return PopulationStats(generation=self.generation)
        
        total_count = len(self.bacteria_by_id)
        resistant_count = len(self.resistant_bacteria)
        sensitive_count = len(self.sensitive_bacteria)
        
        # Calculate averages efficiently
        total_age = sum(b.age for b in self.bacteria_by_id.values())
        total_fitness = sum(b.effective_fitness for b in self.bacteria_by_id.values())
        
        average_age = total_age / total_count
        average_fitness = total_fitness / total_count
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
    
    def advance_generation(self) -> None:
        """
        Advance the population by one generation with optimized operations.
        """
        self.generation += 1
        
        # OPTIMIZATION: Batch process survival and reproduction
        survivors = []
        new_offspring = []
        
        # Process in batches to reduce memory pressure
        bacteria_list = list(self.bacteria_by_id.values())
        batch_size = self.config.batch_size
        
        for i in range(0, len(bacteria_list), batch_size):
            batch = bacteria_list[i:i + batch_size]
            batch_survivors, batch_offspring = self._process_generation_batch(batch)
            survivors.extend(batch_survivors)
            new_offspring.extend(batch_offspring)
        
        # Clear current population
        self.clear()
        
        # Add survivors and offspring
        all_bacteria = survivors + new_offspring
        self._batch_add_bacteria(all_bacteria)
        
        # Update statistics
        self._update_statistics()
    
    def _process_generation_batch(self, bacteria_batch: List[Bacterium]) -> Tuple[List[Bacterium], List[Bacterium]]:
        """
        Process a batch of bacteria for survival and reproduction.
        
        Args:
            bacteria_batch: Batch of bacteria to process
            
        Returns:
            Tuple of (survivors, offspring)
        """
        survivors = []
        offspring = []
        
        for bacterium in bacteria_batch:
            # Age the bacterium
            bacterium.age_one_generation()
            
            # Check survival
            survival_prob = bacterium.calculate_survival_probability()
            if random.random() < survival_prob:
                survivors.append(bacterium)
                
                # Check reproduction
                if bacterium.can_reproduce():
                    child = bacterium.reproduce(generation=self.generation, next_id_generator=self._generate_id)
                    if child:
                        offspring.append(child)
        
        return survivors, offspring
    
    def get_random_sample(self, sample_size: int) -> List[Bacterium]:
        """
        Get a random sample of bacteria efficiently.
        
        Args:
            sample_size: Number of bacteria to sample
            
        Returns:
            Random sample of bacteria
        """
        if sample_size >= len(self.bacteria_by_id):
            return list(self.bacteria_by_id.values())
        
        # OPTIMIZATION: Sample IDs first, then lookup
        sampled_ids = random.sample(list(self.bacteria_by_id.keys()), sample_size)
        return [self.bacteria_by_id[id] for id in sampled_ids]
    
    def filter_bacteria(self, predicate: Callable[[Bacterium], bool]) -> List[Bacterium]:
        """
        Filter bacteria using a predicate function efficiently.
        
        Args:
            predicate: Function to test each bacterium
            
        Returns:
            List of bacteria matching the predicate
        """
        return [bacterium for bacterium in self.bacteria_by_id.values() if predicate(bacterium)]
    
    def get_resistant_bacteria(self) -> List[Bacterium]:
        """Get all resistant bacteria using optimized index."""
        return [self.bacteria_by_id[id] for id in self.resistant_bacteria]
    
    def get_sensitive_bacteria(self) -> List[Bacterium]:
        """Get all sensitive bacteria using optimized index."""
        return [self.bacteria_by_id[id] for id in self.sensitive_bacteria]
    
    def get_reproducible_bacteria(self) -> List[Bacterium]:
        """Get bacteria that can reproduce with optimized filtering."""
        return [bacterium for bacterium in self.bacteria_by_id.values() if bacterium.can_reproduce()]
    
    def clone_bacterium(self, bacterium: Bacterium) -> Bacterium:
        """
        Create an exact copy of a bacterium with new ID.
        
        Args:
            bacterium: Bacterium to clone
            
        Returns:
            Cloned bacterium with new ID
        """
        # Find position for clone
        clone_position = None
        if bacterium.position and self.config.use_spatial:
            clone_position = self._find_available_position()
        
        clone = Bacterium(
            id=self._generate_id(),
            resistance_status=bacterium.resistance_status,
            age=bacterium.age,
            fitness=bacterium.fitness,
            position=clone_position,
            generation_born=bacterium.generation_born,
            parent_id=bacterium.id
        )
        
        return clone
    
    def reset_population(self) -> None:
        """Reset population to empty state efficiently."""
        self.clear()
        self.generation = 0
        self._next_id = 0
        self.stats_history.clear()
    
    def reset(self) -> None:
        """Reset everything to initial state."""
        self.reset_population()
    
    def export_population_data(self) -> Dict[str, Any]:
        """
        Export population data for analysis with optimized serialization.
        
        Returns:
            Dictionary containing population data
        """
        stats = self.get_statistics()
        
        return {
            "metadata": {
                "generation": self.generation,
                "population_size": len(self.bacteria_by_id),
                "config": {
                    "population_size": self.config.population_size,
                    "grid_width": self.config.grid_width,
                    "grid_height": self.config.grid_height,
                    "use_spatial": self.config.use_spatial
                }
            },
            "statistics": {
                "total_count": stats.total_count,
                "resistant_count": stats.resistant_count,
                "sensitive_count": stats.sensitive_count,
                "average_age": stats.average_age,
                "average_fitness": stats.average_fitness,
                "resistance_frequency": stats.resistance_frequency
            },
            "bacteria": [
                {
                    "id": bacterium.id,
                    "resistance_status": bacterium.resistance_status.value,
                    "age": bacterium.age,
                    "fitness": bacterium.fitness,
                    "position": {
                        "x": bacterium.position.x,
                        "y": bacterium.position.y
                    } if bacterium.position else None,
                    "generation_born": bacterium.generation_born
                }
                for bacterium in self.bacteria_by_id.values()
            ]
        }
    
    @property
    def size(self) -> int:
        """Get population size efficiently."""
        return len(self.bacteria_by_id)
    
    @property
    def resistance_frequency(self) -> float:
        """Get resistance frequency efficiently using cached stats."""
        return self.get_statistics().resistance_frequency
    
    def __len__(self) -> int:
        """Get population size."""
        return len(self.bacteria_by_id)
    
    def __iter__(self):
        """Iterate over bacteria efficiently."""
        return iter(self.bacteria_by_id.values())
    
    def __str__(self) -> str:
        """String representation with current stats."""
        stats = self.get_statistics()
        return (f"OptimizedPopulation(size={stats.total_count}, "
                f"resistant={stats.resistant_count}, "
                f"generation={self.generation})")
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"OptimizedPopulation(config={self.config}, "
                f"size={len(self.bacteria_by_id)}, generation={self.generation})")

    def mutate(self, mutation_rate: float = 0.001) -> List[Dict[str, Any]]:
        """
        Apply mutations to the population with the given mutation rate.
        
        Args:
            mutation_rate: Probability of mutation per bacterium
            
        Returns:
            List of mutation events that occurred
        """
        mutation_events = []
        
        for bacterium in list(self.bacteria_by_id.values()):
            if random.random() < mutation_rate:
                # Track mutation before applying it
                old_resistance = bacterium.resistance_status
                
                # Apply mutation
                if bacterium.resistance_status == ResistanceStatus.SENSITIVE:
                    # Mutation to resistance
                    bacterium.resistance_status = ResistanceStatus.RESISTANT
                    bacterium._survival_bonus = 0.1
                    # Slightly reduced fitness due to resistance cost
                    bacterium.fitness *= 0.95
                    
                    # Update indices
                    self.sensitive_bacteria.discard(bacterium.id)
                    self.resistant_bacteria.add(bacterium.id)
                    
                else:
                    # Resistant bacteria can mutate to become more or less fit
                    fitness_change = random.uniform(-0.1, 0.1)
                    bacterium.fitness = max(0.1, bacterium.fitness + fitness_change)
                
                # Record mutation event
                mutation_events.append({
                    "bacterium_id": bacterium.id,
                    "generation": self.generation,
                    "old_resistance": old_resistance.value,
                    "new_resistance": bacterium.resistance_status.value,
                    "fitness_change": bacterium.fitness,
                    "mutation_type": "resistance_acquired" if old_resistance == ResistanceStatus.SENSITIVE else "fitness_modified"
                })
        
        # Mark stats as dirty since resistance composition changed
        self._stats_dirty = True
        
        return mutation_events
    
    def get_average_resistance(self) -> float:
        """
        Get the average resistance frequency in the population.
        
        Returns:
            Float between 0 and 1 representing the fraction of resistant bacteria
        """
        if len(self.bacteria_by_id) == 0:
            return 0.0
        
        return len(self.resistant_bacteria) / len(self.bacteria_by_id)
    
    def calculate_diversity_index(self) -> float:
        """
        Calculate the Shannon diversity index for the population.
        
        Returns:
            Shannon diversity index value
        """
        if len(self.bacteria_by_id) == 0:
            return 0.0
        
        total_count = len(self.bacteria_by_id)
        resistant_count = len(self.resistant_bacteria)
        sensitive_count = len(self.sensitive_bacteria)
        
        # Calculate Shannon diversity based on resistance status
        diversity = 0.0
        
        if resistant_count > 0:
            p_resistant = resistant_count / total_count
            diversity -= p_resistant * np.log2(p_resistant)
        
        if sensitive_count > 0:
            p_sensitive = sensitive_count / total_count
            diversity -= p_sensitive * np.log2(p_sensitive)
        
        return diversity


# Maintain backward compatibility
Population = OptimizedPopulation 