"""
Spatial Grid Model for Petri Dish Simulation

This module provides spatial representation for bacterial populations,
including 2D grid management, positioning, and proximity calculations
for horizontal gene transfer (HGT) mechanisms.
"""

import math
import logging
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import weakref
import numpy as np
import random
import heapq
from scipy.spatial import cKDTree  # Added for efficient spatial queries

# Import memory management components with try/catch for flexibility
try:
    from services.memory_manager import MemoryManager, Disposable, ObjectPool
except ImportError:
    try:
        from backend.services.memory_manager import MemoryManager, Disposable, ObjectPool
    except ImportError:
        # Fallback for testing - create minimal implementations
        class Disposable:
            def dispose(self): pass
            @property
            def is_disposed(self): return False
        
        class ObjectPool:
            def __init__(self, factory, *args, **kwargs): 
                self.factory = factory
            def acquire(self): 
                return self.factory()
            def release(self, obj): pass
            def clear(self): pass
            def get_stats(self): return type('Stats', (), {'hit_rate': 0, 'current_pool_size': 0})()
        
        class MemoryManager:
            def __init__(self): 
                self.resource_manager = type('ResourceManager', (), {
                    'register': lambda self, obj, group: None,
                    'unregister': lambda self, obj: None
                })()
            def create_pool(self, name, factory, *args, **kwargs): 
                return ObjectPool(factory, *args, **kwargs)
            def get_pool(self, name): return None
            def get_memory_metrics(self): 
                return type('Metrics', (), {
                    'rss_mb': 100, 'vms_mb': 200, 'heap_size_mb': 50, 
                    'object_count': 1000, 'timestamp': type('datetime', (), {'isoformat': lambda: '2023-01-01'})()
                })()
            def get_pool_stats(self): return {}
            def cleanup_all(self): return {}

logger = logging.getLogger(__name__)

# Global memory manager instance
_memory_manager = MemoryManager()

# Object pools for frequently created objects
_coordinate_pool: Optional[ObjectPool] = None
_grid_cell_pool: Optional[ObjectPool] = None


def _reset_coordinate(coord: 'Coordinate') -> None:
    """Reset function for coordinate pool."""
    coord.x = 0.0
    coord.y = 0.0


def _reset_grid_cell(cell: 'GridCell') -> None:
    """Reset function for grid cell pool."""
    cell.bacteria_ids.clear()
    cell.antibiotic_concentration = 0.0
    cell.nutrient_concentration = 1.0


def _initialize_pools():
    """Initialize object pools for spatial components."""
    global _coordinate_pool, _grid_cell_pool
    
    if _coordinate_pool is None:
        _coordinate_pool = _memory_manager.create_pool(
            name="coordinate_pool",
            factory=lambda: Coordinate(0.0, 0.0),
            reset_func=_reset_coordinate,
            max_size=5000
        )
    
    if _grid_cell_pool is None:
        _grid_cell_pool = _memory_manager.create_pool(
            name="grid_cell_pool", 
            factory=lambda: GridCell(Coordinate(0.0, 0.0)),
            reset_func=_reset_grid_cell,
            max_size=10000
        )


class BoundaryCondition(Enum):
    """Boundary conditions for the spatial grid."""
    CLOSED = "closed"  # Bacteria cannot move beyond boundaries
    PERIODIC = "periodic"  # Wrap-around boundaries (torus topology)
    REFLECTIVE = "reflective"  # Bacteria bounce off boundaries


@dataclass
class Coordinate:
    """Represents a 2D coordinate in the spatial grid."""
    x: float
    y: float
    
    @classmethod
    def create_pooled(cls, x: float, y: float) -> 'Coordinate':
        """Create a coordinate using object pool for efficiency."""
        _initialize_pools()
        coord = _coordinate_pool.acquire()
        coord.x = x
        coord.y = y
        return coord
    
    def release_to_pool(self) -> None:
        """Return this coordinate to the object pool."""
        if _coordinate_pool:
            _coordinate_pool.release(self)
    
    def distance_to(self, other: 'Coordinate') -> float:
        """Calculate Euclidean distance to another coordinate."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def manhattan_distance_to(self, other: 'Coordinate') -> float:
        """Calculate Manhattan distance to another coordinate."""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def is_within_distance(self, other: 'Coordinate', distance: float) -> bool:
        """Check if another coordinate is within specified distance."""
        return self.distance_to(other) <= distance
    
    def __hash__(self):
        return hash((round(self.x, 6), round(self.y, 6)))
    
    def __eq__(self, other):
        if not isinstance(other, Coordinate):
            return False
        return (abs(self.x - other.x) < 1e-3 and 
                abs(self.y - other.y) < 1e-3)


@dataclass
class GridCell:
    """Represents a cell in the spatial grid."""
    coordinate: Coordinate
    bacteria_ids: Set[str] = field(default_factory=set)
    antibiotic_concentration: float = 0.0
    nutrient_concentration: float = 1.0
    
    @classmethod
    def create_pooled(cls, coordinate: Coordinate) -> 'GridCell':
        """Create a grid cell using object pool for efficiency."""
        _initialize_pools()
        cell = _grid_cell_pool.acquire()
        cell.coordinate = coordinate
        cell.bacteria_ids.clear()
        cell.antibiotic_concentration = 0.0
        cell.nutrient_concentration = 1.0
        return cell
    
    def release_to_pool(self) -> None:
        """Return this grid cell to the object pool."""
        if _grid_cell_pool:
            _grid_cell_pool.release(self)
    
    def add_bacterium(self, bacterium_id: str):
        """Add a bacterium to this cell."""
        self.bacteria_ids.add(bacterium_id)
    
    def remove_bacterium(self, bacterium_id: str):
        """Remove a bacterium from this cell."""
        self.bacteria_ids.discard(bacterium_id)
    
    def get_bacteria_count(self) -> int:
        """Get the number of bacteria in this cell."""
        return len(self.bacteria_ids)
    
    def is_overcrowded(self, max_capacity: int = 10) -> bool:
        """Check if the cell is overcrowded."""
        return self.get_bacteria_count() > max_capacity


class SpatialGrid(Disposable):
    """
    2D spatial grid representing a Petri dish for bacterial simulation.
    
    Provides spatial positioning, neighborhood detection, and environmental
    factor management (antibiotics, nutrients) with lazy cell allocation
    and memory management integration.
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        cell_size: float = 1.0,
        boundary_condition: BoundaryCondition = BoundaryCondition.CLOSED
    ):
        """
        Initialize the spatial grid.
        
        Args:
            width: Width of the Petri dish in spatial units
            height: Height of the Petri dish in spatial units
            cell_size: Size of each grid cell
            boundary_condition: How to handle boundaries
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.boundary_condition = boundary_condition
        
        # Calculate grid dimensions
        self.grid_width = int(math.ceil(width / cell_size))
        self.grid_height = int(math.ceil(height / cell_size))
        
        # MEMORY OPTIMIZATION: Use lazy initialization for grid cells
        # Only create cells when they are actually accessed
        self.cells: Dict[Tuple[int, int], GridCell] = {}
        self._lazy_cell_creation_enabled = True
        
        # Track bacterial positions
        self.bacterial_positions: Dict[str, Coordinate] = {}
        
        # Environmental factors
        self.antibiotic_zones: List[Dict] = []
        self.nutrient_patches: List[Dict] = []
        
        # Memory management
        self._disposed = False
        _initialize_pools()
        _memory_manager.resource_manager.register(self, "spatial_grids")
        
        logger.info(f"Initialized spatial grid: {self.grid_width}x{self.grid_height} cells, "
                   f"cell size: {cell_size}, boundary: {boundary_condition.value}, "
                   f"lazy loading: {self._lazy_cell_creation_enabled}")
    
    def _get_or_create_cell(self, grid_index: Tuple[int, int]) -> GridCell:
        """
        Get existing cell or create new one lazily.
        
        This is the core of the lazy loading optimization - cells are only
        created when they are actually needed.
        """
        if grid_index in self.cells:
            return self.cells[grid_index]
        
        # Create cell lazily
        i, j = grid_index
        x = (i + 0.5) * self.cell_size
        y = (j + 0.5) * self.cell_size
        coord = Coordinate.create_pooled(x, y)
        cell = GridCell.create_pooled(coord)
        self.cells[grid_index] = cell
        
        logger.debug(f"Lazily created cell at grid index {grid_index}")
        return cell
    
    def _initialize_grid(self):
        """
        Initialize grid cells.
        
        MEMORY OPTIMIZATION: This method is now deprecated in favor of lazy loading.
        Kept for backward compatibility but does nothing by default.
        """
        if not self._lazy_cell_creation_enabled:
            # Fallback to eager initialization if lazy loading is disabled
            for i in range(self.grid_width):
                for j in range(self.grid_height):
                    x = (i + 0.5) * self.cell_size
                    y = (j + 0.5) * self.cell_size
                    coord = Coordinate.create_pooled(x, y)
                    self.cells[(i, j)] = GridCell.create_pooled(coord)
    
    def enable_lazy_loading(self, enabled: bool = True):
        """Enable or disable lazy cell creation."""
        self._lazy_cell_creation_enabled = enabled
        if not enabled:
            # Pre-create all cells if lazy loading is disabled
            self._initialize_grid()
    
    def _coordinate_to_grid_index(self, coord: Coordinate) -> Tuple[int, int]:
        """Convert coordinate to grid cell index."""
        i = int(coord.x / self.cell_size)
        j = int(coord.y / self.cell_size)
        
        # Handle boundary conditions
        if self.boundary_condition == BoundaryCondition.PERIODIC:
            i = i % self.grid_width
            j = j % self.grid_height
        elif self.boundary_condition == BoundaryCondition.CLOSED:
            i = max(0, min(i, self.grid_width - 1))
            j = max(0, min(j, self.grid_height - 1))
        
        return (i, j)
    
    def _grid_index_to_coordinate(self, i: int, j: int) -> Coordinate:
        """Convert grid cell index to coordinate."""
        x = (i + 0.5) * self.cell_size
        y = (j + 0.5) * self.cell_size
        return Coordinate.create_pooled(x, y)
    
    def is_valid_position(self, coord: Coordinate) -> bool:
        """Check if a coordinate is within the grid boundaries."""
        if self.boundary_condition == BoundaryCondition.PERIODIC:
            return True  # All positions are valid with periodic boundaries
        
        return (0 <= coord.x <= self.width and 
                0 <= coord.y <= self.height)
    
    def place_bacterium(self, bacterium_id: str, position: Coordinate) -> bool:
        """
        Place a bacterium at the specified position.
        
        Args:
            bacterium_id: Unique identifier for the bacterium
            position: Desired position
            
        Returns:
            True if placement was successful, False otherwise
        """
        if not self.is_valid_position(position):
            logger.warning(f"Invalid position for bacterium {bacterium_id}: {position}")
            return False
        
        # Remove from previous position if exists
        if bacterium_id in self.bacterial_positions:
            self.remove_bacterium(bacterium_id)
        
        # Add to new position using lazy cell creation
        grid_index = self._coordinate_to_grid_index(position)
        cell = self._get_or_create_cell(grid_index)
        cell.add_bacterium(bacterium_id)
        self.bacterial_positions[bacterium_id] = position
        
        logger.debug(f"Placed bacterium {bacterium_id} at {position}")
        return True
    
    def remove_bacterium(self, bacterium_id: str) -> bool:
        """
        Remove a bacterium from the grid.
        
        Args:
            bacterium_id: Unique identifier for the bacterium
            
        Returns:
            True if removal was successful, False if bacterium not found
        """
        if bacterium_id not in self.bacterial_positions:
            return False
        
        position = self.bacterial_positions[bacterium_id]
        grid_index = self._coordinate_to_grid_index(position)
        
        # Only access cell if it exists (lazy loading consideration)
        if grid_index in self.cells:
            cell = self.cells[grid_index]
            cell.remove_bacterium(bacterium_id)
        
        del self.bacterial_positions[bacterium_id]
        logger.debug(f"Removed bacterium {bacterium_id} from {position}")
        return True
    
    def move_bacterium(self, bacterium_id: str, new_position: Coordinate) -> bool:
        """
        Move a bacterium to a new position.
        
        Args:
            bacterium_id: Unique identifier for the bacterium
            new_position: New desired position
            
        Returns:
            True if movement was successful, False otherwise
        """
        if bacterium_id not in self.bacterial_positions:
            logger.warning(f"Cannot move bacterium {bacterium_id}: not found in grid")
            return False
        
        return self.place_bacterium(bacterium_id, new_position)
    
    def get_bacterium_position(self, bacterium_id: str) -> Optional[Coordinate]:
        """Get the position of a bacterium."""
        return self.bacterial_positions.get(bacterium_id)
    
    def get_neighbors(
        self, 
        bacterium_id: str, 
        radius: float,
        include_self: bool = False
    ) -> List[str]:
        """
        Find neighboring bacteria within specified radius.
        
        Args:
            bacterium_id: Target bacterium
            radius: Search radius
            include_self: Whether to include the target bacterium
            
        Returns:
            List of neighboring bacterium IDs
        """
        if bacterium_id not in self.bacterial_positions:
            return []
        
        center_pos = self.bacterial_positions[bacterium_id]
        neighbors = []
        
        # Get grid cells within radius
        cell_radius = int(math.ceil(radius / self.cell_size))
        center_grid = self._coordinate_to_grid_index(center_pos)
        
        for di in range(-cell_radius, cell_radius + 1):
            for dj in range(-cell_radius, cell_radius + 1):
                check_i = center_grid[0] + di
                check_j = center_grid[1] + dj
                
                # Handle boundary conditions
                if self.boundary_condition == BoundaryCondition.PERIODIC:
                    check_i = check_i % self.grid_width
                    check_j = check_j % self.grid_height
                elif (check_i < 0 or check_i >= self.grid_width or
                      check_j < 0 or check_j >= self.grid_height):
                    continue
                
                # LAZY LOADING: Only check cells that exist
                cell_index = (check_i, check_j)
                if cell_index in self.cells:
                    cell = self.cells[cell_index]
                    for other_id in cell.bacteria_ids:
                        if other_id == bacterium_id and not include_self:
                            continue
                        
                        other_pos = self.bacterial_positions.get(other_id)
                        if other_pos and center_pos.distance_to(other_pos) <= radius:
                            neighbors.append(other_id)
        
        return neighbors
    
    def get_local_density(self, position: Coordinate, radius: float) -> float:
        """
        Calculate local bacterial density around a position.
        
        Args:
            position: Center position
            radius: Search radius
            
        Returns:
            Number of bacteria per unit area
        """
        neighbors = self._get_bacteria_in_radius(position, radius)
        area = math.pi * radius**2
        return len(neighbors) / area
    
    def _get_bacteria_in_radius(
        self, 
        center: Coordinate, 
        radius: float
    ) -> List[str]:
        """Get all bacteria within radius of a position."""
        bacteria_in_radius = []
        
        cell_radius = int(math.ceil(radius / self.cell_size))
        center_grid = self._coordinate_to_grid_index(center)
        
        for di in range(-cell_radius, cell_radius + 1):
            for dj in range(-cell_radius, cell_radius + 1):
                check_i = center_grid[0] + di
                check_j = center_grid[1] + dj
                
                if self.boundary_condition == BoundaryCondition.PERIODIC:
                    check_i = check_i % self.grid_width
                    check_j = check_j % self.grid_height
                elif (check_i < 0 or check_i >= self.grid_width or
                      check_j < 0 or check_j >= self.grid_height):
                    continue
                
                cell = self.cells.get((check_i, check_j))
                if cell:
                    for bacterium_id in cell.bacteria_ids:
                        bacterium_pos = self.bacterial_positions[bacterium_id]
                        if center.is_within_distance(bacterium_pos, radius):
                            bacteria_in_radius.append(bacterium_id)
        
        return bacteria_in_radius
    
    def add_antibiotic_zone(
        self, 
        center: Coordinate, 
        radius: float, 
        concentration: float,
        zone_id: str = None
    ):
        """
        Add a circular antibiotic zone.
        
        Args:
            center: Center of the zone
            radius: Radius of the zone
            concentration: Antibiotic concentration
            zone_id: Unique identifier for the zone
        """
        zone = {
            'id': zone_id or f"antibiotic_{len(self.antibiotic_zones)}",
            'center': center,
            'radius': radius,
            'concentration': concentration,
            'type': 'circular'
        }
        self.antibiotic_zones.append(zone)
        
        # Update cell concentrations
        self._update_antibiotic_concentrations()
        logger.info(f"Added antibiotic zone at {center} with radius {radius}")
    
    def _update_antibiotic_concentrations(self):
        """Update antibiotic concentrations in all cells based on zones."""
        for (i, j), cell in self.cells.items():
            max_concentration = 0.0
            
            for zone in self.antibiotic_zones:
                if zone['type'] == 'circular':
                    distance = cell.coordinate.distance_to(zone['center'])
                    if distance <= zone['radius']:
                        if distance < 1e-6:  # Very close to center
                            concentration = zone['concentration']
                        else:
                            # Linear decay from center to edge
                            concentration = zone['concentration'] * (1 - distance / zone['radius'])
                        max_concentration = max(max_concentration, concentration)
            
            cell.antibiotic_concentration = max_concentration
    
    def get_antibiotic_concentration(self, position: Coordinate) -> float:
        """Get antibiotic concentration at a specific position."""
        max_concentration = 0.0
        
        # Calculate concentration directly from zones for exact position
        for zone in self.antibiotic_zones:
            if zone['type'] == 'circular':
                distance = position.distance_to(zone['center'])
                if distance <= zone['radius']:
                    if distance < 1e-6:  # Very close to center
                        concentration = zone['concentration']
                    else:
                        # Linear decay from center to edge
                        concentration = zone['concentration'] * (1 - distance / zone['radius'])
                    max_concentration = max(max_concentration, concentration)
        
        return max_concentration
    
    def get_grid_statistics(self) -> Dict:
        """Get comprehensive grid statistics."""
        total_bacteria = len(self.bacterial_positions)
        occupied_cells = sum(1 for cell in self.cells.values() 
                           if cell.get_bacteria_count() > 0)
        
        densities = [cell.get_bacteria_count() for cell in self.cells.values()]
        max_density = max(densities) if densities else 0
        avg_density = np.mean(densities) if densities else 0
        
        antibiotic_cells = sum(1 for cell in self.cells.values() 
                             if cell.antibiotic_concentration > 0)
        
        return {
            'total_bacteria': total_bacteria,
            'occupied_cells': occupied_cells,
            'total_cells': len(self.cells),
            'occupancy_rate': occupied_cells / len(self.cells),
            'max_cell_density': max_density,
            'average_cell_density': avg_density,
            'antibiotic_coverage': antibiotic_cells / len(self.cells),
            'grid_dimensions': (self.grid_width, self.grid_height),
            'physical_dimensions': (self.width, self.height)
        }
    
    def clear(self):
        """Clear all bacteria and environmental factors from the grid."""
        self.bacterial_positions.clear()
        self.antibiotic_zones.clear()
        self.nutrient_patches.clear()
        
        # MEMORY OPTIMIZATION: Return cells to object pool when clearing
        for cell in self.cells.values():
            cell.bacteria_ids.clear()
            cell.antibiotic_concentration = 0.0
            cell.nutrient_concentration = 1.0
            # Return coordinate and cell to their respective pools
            if hasattr(cell.coordinate, 'release_to_pool'):
                cell.coordinate.release_to_pool()
            if hasattr(cell, 'release_to_pool'):
                cell.release_to_pool()
        
        self.cells.clear()
        logger.info("Cleared spatial grid and returned objects to pools")
    
    def handle_boundary_condition(self, position: Coordinate) -> Coordinate:
        """
        Handle boundary conditions for a given position.
        
        Args:
            position: Position to check and potentially adjust
            
        Returns:
            Adjusted position based on boundary condition
        """
        x, y = position.x, position.y
        
        if self.boundary_condition == BoundaryCondition.PERIODIC:
            # Wrap around boundaries
            x = x % self.width
            y = y % self.height
        elif self.boundary_condition == BoundaryCondition.REFLECTIVE:
            # Reflect off boundaries
            if x < 0:
                x = -x
            elif x > self.width:
                x = 2 * self.width - x
            
            if y < 0:
                y = -y
            elif y > self.height:
                y = 2 * self.height - y
        else:  # CLOSED boundary condition
            # Clamp to boundaries
            x = max(0, min(x, self.width))
            y = max(0, min(y, self.height))
        
        return Coordinate(x, y)
    
    def get_cell_at_position(self, position: Coordinate) -> Optional[GridCell]:
        """
        Get the grid cell at a specific position.
        
        Args:
            position: Position to get cell for
            
        Returns:
            GridCell if position is valid, None otherwise
        """
        grid_index = self._coordinate_to_grid_index(position)
        return self.cells.get(grid_index)
    
    def dispose(self) -> None:
        """
        Dispose of the spatial grid and release all resources.
        
        This method implements the Disposable interface and ensures
        proper cleanup of all pooled objects and resources.
        """
        if self._disposed:
            return
        
        logger.info("Disposing spatial grid and releasing resources")
        
        # Clear all data and return objects to pools
        self.clear()
        
        # Mark as disposed
        self._disposed = True
        
        # Unregister from memory manager
        _memory_manager.resource_manager.unregister(self)
        
        logger.info("Spatial grid disposed successfully")
    
    @property
    def is_disposed(self) -> bool:
        """Check if the spatial grid has been disposed."""
        return self._disposed
    
    def get_memory_usage_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get memory usage statistics for this spatial grid.
        
        Returns:
            Dictionary with memory usage information
        """
        if self._disposed:
            return {"status": "disposed", "memory_usage_mb": 0}
        
        stats = {
            "total_cells_created": len(self.cells),
            "max_possible_cells": self.grid_width * self.grid_height,
            "memory_efficiency_percent": (len(self.cells) / (self.grid_width * self.grid_height)) * 100,
            "bacteria_count": len(self.bacterial_positions),
            "antibiotic_zones": len(self.antibiotic_zones),
            "lazy_loading_enabled": self._lazy_cell_creation_enabled
        }
        
        # Add pool statistics if available
        if _coordinate_pool:
            coord_stats = _coordinate_pool.get_stats()
            stats["coordinate_pool_hit_rate"] = coord_stats.hit_rate
            stats["coordinate_pool_size"] = coord_stats.current_pool_size
        
        if _grid_cell_pool:
            cell_stats = _grid_cell_pool.get_stats()
            stats["cell_pool_hit_rate"] = cell_stats.hit_rate
            stats["cell_pool_size"] = cell_stats.current_pool_size
        
        return stats
    
    def optimize_memory_usage(self) -> Dict[str, int]:
        """
        Perform memory optimization operations.
        
        Returns:
            Dictionary with optimization results
        """
        if self._disposed:
            return {"error": "Cannot optimize disposed grid"}
        
        results = {"cells_before": len(self.cells)}
        
        # Remove empty cells to save memory (only if lazy loading is enabled)
        if self._lazy_cell_creation_enabled:
            empty_cells = []
            for grid_index, cell in self.cells.items():
                if (cell.get_bacteria_count() == 0 and 
                    cell.antibiotic_concentration == 0.0 and 
                    cell.nutrient_concentration == 1.0):
                    empty_cells.append(grid_index)
            
            # Return empty cells to pools
            for grid_index in empty_cells:
                cell = self.cells.pop(grid_index)
                if hasattr(cell.coordinate, 'release_to_pool'):
                    cell.coordinate.release_to_pool()
                if hasattr(cell, 'release_to_pool'):
                    cell.release_to_pool()
            
            results["empty_cells_removed"] = len(empty_cells)
            results["cells_after"] = len(self.cells)
            results["memory_saved_percent"] = (len(empty_cells) / results["cells_before"]) * 100 if results["cells_before"] > 0 else 0
        else:
            results["empty_cells_removed"] = 0
            results["cells_after"] = results["cells_before"] 
            results["memory_saved_percent"] = 0
        
        # Trigger garbage collection for object pools
        if _coordinate_pool:
            _coordinate_pool.clear()
        if _grid_cell_pool:
            _grid_cell_pool.clear()
        
        logger.info(f"Memory optimization complete: removed {results['empty_cells_removed']} empty cells")
        return results


class SpatialManager:
    """
    High-level manager for spatial operations and grid management.
    Optimized for large bacterial populations with efficient data structures
    and memory management integration.
    """
    
    def __init__(self, grid: SpatialGrid):
        self.grid = grid
        self.bacterium_positions: Dict[str, Coordinate] = {}
        self.position_cache: Dict[str, Tuple[float, float]] = {}
        
        # Performance optimization data structures
        self._spatial_tree: Optional[cKDTree] = None
        self._tree_dirty = False
        self._batch_updates: List[Tuple[str, Coordinate]] = []
        self._update_threshold = 100  # Batch update threshold
        
        # Movement tracking for optimization
        self._movement_cache: Dict[str, float] = {}  # bacterium_id -> last movement distance
        self._static_bacteria: Set[str] = set()  # Bacteria that haven't moved recently
        
        # Memory management integration
        _initialize_pools()
        
    def initialize_random_population(
        self, 
        population_size: int, 
        bacterium_ids: List[str]
    ) -> Dict[str, Coordinate]:
        """
        Place bacteria randomly across the grid.
        
        Args:
            population_size: Number of bacteria to place
            bacterium_ids: List of bacterium identifiers
            
        Returns:
            Dictionary mapping bacterium_id to position
        """
        positions = {}
        
        for i, bacterium_id in enumerate(bacterium_ids[:population_size]):
            # Generate random position using pooled coordinates
            x = np.random.uniform(0, self.grid.width)
            y = np.random.uniform(0, self.grid.height)
            position = Coordinate.create_pooled(x, y)
            
            # Place bacterium in grid
            if self.grid.place_bacterium(bacterium_id, position):
                positions[bacterium_id] = position
                # Also update SpatialManager's tracking
                self.bacterium_positions[bacterium_id] = position
                self.position_cache[bacterium_id] = (position.x, position.y)
            else:
                logger.warning(f"Failed to place bacterium {bacterium_id}")
                # Return unused coordinate to pool
                if hasattr(position, 'release_to_pool'):
                    position.release_to_pool()
        
        # Mark spatial tree as dirty since we added new bacteria
        self._tree_dirty = True
        
        logger.info(f"Initialized {len(positions)} bacteria randomly on grid")
        return positions
    
    def calculate_hgt_candidates(
        self, 
        donor_id: str, 
        hgt_radius: float = 2.0,
        max_candidates: int = 10
    ) -> List[str]:
        """
        Calculate potential HGT candidates within radius (optimized for large populations).
        
        Args:
            donor_id: ID of the donor bacterium
            hgt_radius: Maximum distance for HGT
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of candidate bacterium IDs for HGT
        """
        if donor_id not in self.bacterium_positions:
            return []
        
        donor_pos = self.bacterium_positions[donor_id]
        
        # Use spatial tree for efficient neighbor search if available
        if self._spatial_tree is not None and not self._tree_dirty:
            donor_coords = np.array([[donor_pos.x, donor_pos.y]])
            indices = self._spatial_tree.query_ball_point(donor_coords[0], hgt_radius)
            
            # Convert indices back to bacterium IDs and limit results
            position_list = list(self.bacterium_positions.items())
            candidates = []
            for idx in indices:
                if idx < len(position_list):
                    bacterium_id, _ = position_list[idx]
                    if bacterium_id != donor_id:
                        candidates.append(bacterium_id)
                        if len(candidates) >= max_candidates:
                            break
            
            return candidates
        
        # Fallback to grid-based search for smaller populations
        candidates = []
        for bacterium_id, position in self.bacterium_positions.items():
            if bacterium_id != donor_id:
                distance = donor_pos.distance_to(position)
                if distance <= hgt_radius:
                    candidates.append(bacterium_id)
                    if len(candidates) >= max_candidates:
                        break
        
        return candidates
    
    def _rebuild_spatial_tree(self):
        """Rebuild the spatial tree for efficient queries."""
        if not self.bacterium_positions:
            self._spatial_tree = None
            return
        
        # Extract coordinates
        coordinates = np.array([
            [pos.x, pos.y] for pos in self.bacterium_positions.values()
        ])
        
        # Build KD-tree for efficient spatial queries
        self._spatial_tree = cKDTree(coordinates)
        self._tree_dirty = False
    
    def _should_rebuild_tree(self) -> bool:
        """Determine if spatial tree needs rebuilding."""
        return (
            self._tree_dirty or 
            self._spatial_tree is None or 
            len(self._batch_updates) > self._update_threshold
        )
    
    def update_bacterium_position_batch(self, updates: List[Tuple[str, Coordinate]]):
        """
        Update multiple bacterium positions in batch for better performance.
        
        Args:
            updates: List of (bacterium_id, new_position) tuples
        """
        for bacterium_id, new_position in updates:
            old_position = self.bacterium_positions.get(bacterium_id)
            
            # Update position
            self.bacterium_positions[bacterium_id] = new_position
            self.position_cache[bacterium_id] = (new_position.x, new_position.y)
            
            # Track movement for optimization
            if old_position:
                movement_distance = old_position.distance_to(new_position)
                self._movement_cache[bacterium_id] = movement_distance
                
                # Track static bacteria for optimization
                if movement_distance < 0.1:  # Consider static if moved less than 0.1 units
                    self._static_bacteria.add(bacterium_id)
                else:
                    self._static_bacteria.discard(bacterium_id)
            
            # Update grid occupancy
            if old_position:
                old_cell = self.grid.get_cell_at_position(old_position)
                if old_cell:
                    old_cell.bacteria_ids.discard(bacterium_id)
            
            new_cell = self.grid.get_cell_at_position(new_position)
            if new_cell:
                new_cell.bacteria_ids.add(bacterium_id)
        
        # Mark tree as dirty for rebuilding
        self._tree_dirty = True
        
        # Rebuild tree if threshold exceeded
        if len(updates) > self._update_threshold:
            self._rebuild_spatial_tree()
    
    def get_local_density(self, position: Coordinate, radius: float = 5.0) -> float:
        """
        Calculate local bacterial density around a position (optimized).
        
        Args:
            position: Center position
            radius: Radius to check
            
        Returns:
            Density (bacteria per unit area)
        """
        if self._should_rebuild_tree():
            self._rebuild_spatial_tree()
        
        if self._spatial_tree is not None:
            # Use spatial tree for efficient density calculation
            coords = np.array([[position.x, position.y]])
            indices = self._spatial_tree.query_ball_point(coords[0], radius)
            bacteria_count = len(indices)
        else:
            # Fallback method
            bacteria_count = sum(
                1 for pos in self.bacterium_positions.values()
                if position.distance_to(pos) <= radius
            )
        
        area = np.pi * radius ** 2
        return bacteria_count / area if area > 0 else 0.0
    
    def get_performance_metrics(self) -> Dict[str, Union[int, float]]:
        """
        Get performance metrics for monitoring and optimization.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "total_bacteria": len(self.bacterium_positions),
            "static_bacteria": len(self._static_bacteria),
            "movement_efficiency": len(self._static_bacteria) / max(len(self.bacterium_positions), 1),
            "spatial_tree_active": self._spatial_tree is not None,
            "tree_dirty": self._tree_dirty,
            "batch_updates_pending": len(self._batch_updates),
            "cache_hit_rate": len(self.position_cache) / max(len(self.bacterium_positions), 1),
            "occupied_cells": len([cell for cell in self.grid.cells.values() if cell.bacteria_ids]),
            "grid_utilization": len([cell for cell in self.grid.cells.values() if cell.bacteria_ids]) / len(self.grid.cells)
        }
    
    def optimize_for_large_population(self, enable_optimizations: bool = True):
        """
        Enable or disable optimizations for large populations.
        
        Args:
            enable_optimizations: Whether to enable performance optimizations
        """
        if enable_optimizations:
            # Rebuild spatial tree
            self._rebuild_spatial_tree()
            
            # Lower update threshold for more frequent optimizations
            self._update_threshold = max(50, len(self.bacterium_positions) // 20)
            
            logger.info(f"Enabled large population optimizations for {len(self.bacterium_positions)} bacteria")
        else:
            # Disable optimizations
            self._spatial_tree = None
            self._update_threshold = 100
            self._static_bacteria.clear()
            
            logger.info("Disabled large population optimizations")
    
    def cleanup_inactive_bacteria(self, active_bacteria_ids: Set[str]):
        """
        Remove data for bacteria that are no longer active in the simulation.
        
        Args:
            active_bacteria_ids: Set of currently active bacterium IDs
        """
        # Remove inactive bacteria from all data structures
        inactive_ids = set(self.bacterium_positions.keys()) - active_bacteria_ids
        
        for bacterium_id in inactive_ids:
            # Remove from position tracking and return coordinates to pool
            if bacterium_id in self.bacterium_positions:
                position = self.bacterium_positions[bacterium_id]
                cell = self.grid.get_cell_at_position(position)
                if cell:
                    cell.bacteria_ids.discard(bacterium_id)
                
                # MEMORY OPTIMIZATION: Return coordinate to pool
                if hasattr(position, 'release_to_pool'):
                    position.release_to_pool()
                
                del self.bacterium_positions[bacterium_id]
            
            # Remove from caches
            self.position_cache.pop(bacterium_id, None)
            self._movement_cache.pop(bacterium_id, None)
            self._static_bacteria.discard(bacterium_id)
        
        # Mark tree as dirty if bacteria were removed
        if inactive_ids:
            self._tree_dirty = True
            logger.info(f"Cleaned up {len(inactive_ids)} inactive bacteria")
    
    def cleanup_memory_resources(self) -> Dict[str, int]:
        """
        Perform comprehensive memory cleanup for the spatial manager.
        
        Returns:
            Dictionary with cleanup statistics
        """
        cleanup_stats = {"coordinates_released": 0, "caches_cleared": 0}
        
        # Clear position cache
        cache_size = len(self.position_cache)
        self.position_cache.clear()
        cleanup_stats["caches_cleared"] = cache_size
        
        # Clear movement tracking caches
        self._movement_cache.clear()
        self._static_bacteria.clear()
        
        # Clear spatial tree
        self._spatial_tree = None
        self._tree_dirty = True
        
        # Clear batch updates
        self._batch_updates.clear()
        
        # Trigger grid memory optimization
        if hasattr(self.grid, 'optimize_memory_usage'):
            grid_stats = self.grid.optimize_memory_usage()
            cleanup_stats.update(grid_stats)
        
        logger.info(f"SpatialManager memory cleanup complete: {cleanup_stats}")
        return cleanup_stats

    def simulate_bacterial_movement_batch(
        self, 
        bacterium_ids: List[str], 
        movement_radius: float = 0.5,
        movement_probability: float = 0.7
    ) -> List[Tuple[str, Optional[Coordinate]]]:
        """
        Simulate movement for multiple bacteria in batch for better performance.
        
        Args:
            bacterium_ids: List of bacterium IDs to move
            movement_radius: Maximum movement distance
            movement_probability: Probability of movement per bacterium
            
        Returns:
            List of (bacterium_id, new_position) tuples
        """
        movements = []
        
        for bacterium_id in bacterium_ids:
            if bacterium_id not in self.bacterium_positions:
                continue
            
            # Skip movement for bacteria that have been static
            if bacterium_id in self._static_bacteria and random.random() > 0.1:
                continue
            
            # Skip movement based on probability
            if random.random() > movement_probability:
                continue
            
            current_position = self.bacterium_positions[bacterium_id]
            new_position = self.simulate_bacterial_movement(bacterium_id, movement_radius)
            
            if new_position:
                movements.append((bacterium_id, new_position))
        
        # Apply movements in batch
        if movements:
            self.update_bacterium_position_batch(movements)
        
        return movements

    def simulate_bacterial_movement(
        self, 
        bacterium_id: str, 
        movement_radius: float = 0.5
    ) -> Optional[Coordinate]:
        """
        Simulate random bacterial movement for a single bacterium.
        
        Args:
            bacterium_id: ID of the bacterium to move
            movement_radius: Maximum movement distance
            
        Returns:
            New position if movement was successful, None otherwise
        """
        if bacterium_id not in self.bacterium_positions:
            return None
        
        current_pos = self.bacterium_positions[bacterium_id]
        
        # Generate random movement
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, movement_radius)
        
        new_x = current_pos.x + distance * np.cos(angle)
        new_y = current_pos.y + distance * np.sin(angle)
        
        # MEMORY OPTIMIZATION: Use pooled coordinate
        new_pos = Coordinate.create_pooled(new_x, new_y)
        
        # Check boundary conditions
        adjusted_pos = self.grid.handle_boundary_condition(new_pos)
        
        # If position was adjusted, release the original and create new one
        if adjusted_pos != new_pos:
            if hasattr(new_pos, 'release_to_pool'):
                new_pos.release_to_pool()
            new_pos = Coordinate.create_pooled(adjusted_pos.x, adjusted_pos.y)
        
        # Release old position to pool before updating
        if hasattr(current_pos, 'release_to_pool'):
            current_pos.release_to_pool()
        
        # Update position
        self.bacterium_positions[bacterium_id] = new_pos
        self.position_cache[bacterium_id] = (new_pos.x, new_pos.y)
        
        # Update grid occupancy
        old_cell = self.grid.get_cell_at_position(current_pos)
        if old_cell:
            old_cell.bacteria_ids.discard(bacterium_id)
        
        new_cell = self.grid.get_cell_at_position(new_pos)
        if new_cell:
            new_cell.bacteria_ids.add(bacterium_id)
        
        # Track movement for optimization
        movement_distance = current_pos.distance_to(new_pos)
        self._movement_cache[bacterium_id] = movement_distance
        
        if movement_distance < 0.1:
            self._static_bacteria.add(bacterium_id)
        else:
            self._static_bacteria.discard(bacterium_id)
        
        # Mark tree as dirty
        self._tree_dirty = True
        
        return new_pos
    
    def get_environmental_pressure(self, bacterium_id: str) -> Dict[str, float]:
        """
        Get environmental pressures affecting a bacterium.
        
        Args:
            bacterium_id: ID of the bacterium
            
        Returns:
            Dictionary of environmental factors
        """
        if bacterium_id not in self.bacterium_positions:
            return {}
        
        position = self.bacterium_positions[bacterium_id]
        antibiotic_conc = self.grid.get_antibiotic_concentration(position)
        local_density = self.get_local_density(position, radius=5.0)
        
        return {
            'antibiotic_concentration': antibiotic_conc,
            'crowding_pressure': local_density,
            'spatial_stress': min(antibiotic_conc + local_density * 0.1, 1.0)
        } 