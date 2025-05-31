"""
Memory-optimized spatial grid implementation for bacterial simulation.
"""

import math
import threading
from typing import Dict, Set, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import logging

from .spatial import BoundaryCondition, Coordinate, GridCell
from services.memory_manager import memory_manager, Disposable

logger = logging.getLogger(__name__)


@dataclass
class LazyGridCell:
    """Memory-optimized grid cell that creates data structures on demand."""
    coordinate: Coordinate
    _bacteria_ids: Optional[Set[str]] = field(default=None, init=False)
    _antibiotic_concentration: Optional[float] = field(default=None, init=False)
    _nutrient_concentration: Optional[float] = field(default=None, init=False)
    _is_active: bool = field(default=False, init=False)
    
    @property
    def bacteria_ids(self) -> Set[str]:
        """Get bacteria IDs, creating the set if needed."""
        if self._bacteria_ids is None:
            self._bacteria_ids = set()
            self._is_active = True
        return self._bacteria_ids
    
    @property
    def antibiotic_concentration(self) -> float:
        """Get antibiotic concentration."""
        return self._antibiotic_concentration if self._antibiotic_concentration is not None else 0.0
    
    @antibiotic_concentration.setter
    def antibiotic_concentration(self, value: float):
        """Set antibiotic concentration."""
        self._antibiotic_concentration = value
        self._is_active = True
    
    @property
    def nutrient_concentration(self) -> float:
        """Get nutrient concentration."""
        return self._nutrient_concentration if self._nutrient_concentration is not None else 1.0
    
    @nutrient_concentration.setter
    def nutrient_concentration(self, value: float):
        """Set nutrient concentration."""
        self._nutrient_concentration = value
        self._is_active = True
    
    def add_bacterium(self, bacterium_id: str):
        """Add a bacterium to this cell."""
        self.bacteria_ids.add(bacterium_id)
    
    def remove_bacterium(self, bacterium_id: str):
        """Remove a bacterium from this cell."""
        if self._bacteria_ids:
            self._bacteria_ids.discard(bacterium_id)
            # Clean up empty cell
            if not self._bacteria_ids and not self._has_environmental_factors():
                self._deactivate()
    
    def get_bacteria_count(self) -> int:
        """Get the number of bacteria in this cell."""
        return len(self._bacteria_ids) if self._bacteria_ids else 0
    
    def is_overcrowded(self, max_capacity: int = 10) -> bool:
        """Check if the cell is overcrowded."""
        return self.get_bacteria_count() > max_capacity
    
    def _has_environmental_factors(self) -> bool:
        """Check if cell has non-default environmental factors."""
        return (self._antibiotic_concentration is not None and self._antibiotic_concentration > 0) or \
               (self._nutrient_concentration is not None and self._nutrient_concentration != 1.0)
    
    def _deactivate(self):
        """Deactivate cell to save memory."""
        self._bacteria_ids = None
        self._antibiotic_concentration = None
        self._nutrient_concentration = None
        self._is_active = False
    
    @property
    def is_active(self) -> bool:
        """Check if cell is active (has data)."""
        return self._is_active
    
    def get_memory_usage(self) -> int:
        """Get estimated memory usage in bytes."""
        usage = 64  # Base object overhead
        if self._bacteria_ids:
            usage += len(self._bacteria_ids) * 24  # Set overhead + string refs
        if self._antibiotic_concentration is not None:
            usage += 8  # float
        if self._nutrient_concentration is not None:
            usage += 8  # float
        return usage


class OptimizedSpatialGrid(Disposable):
    """
    Memory-optimized spatial grid with lazy cell creation and automatic cleanup.
    """
    
    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        cell_size: float = 1.0,
        boundary_condition: BoundaryCondition = BoundaryCondition.CLOSED,
        lazy_loading: bool = True,
        cleanup_threshold: int = 1000  # Max inactive cells before cleanup
    ):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.boundary_condition = boundary_condition
        self.lazy_loading = lazy_loading
        self.cleanup_threshold = cleanup_threshold
        
        # Calculate grid dimensions
        self.grid_width = int(math.ceil(width / cell_size))
        self.grid_height = int(math.ceil(height / cell_size))
        
        # Lazy cell storage
        self.cells: Dict[Tuple[int, int], LazyGridCell] = {}
        self._cell_access_count: Dict[Tuple[int, int], int] = defaultdict(int)
        self._inactive_cell_count = 0
        
        # Track bacterial positions
        self.bacterial_positions: Dict[str, Coordinate] = {}
        
        # Environmental factors
        self.antibiotic_zones: List[Dict] = []
        self.nutrient_patches: List[Dict] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Disposal tracking
        self._disposed = False
        
        # Register with memory manager
        memory_manager.resource_manager.register(self, "spatial_grids")
        
        # Register cleanup strategy
        memory_manager.register_cleanup_strategy(self._cleanup_inactive_cells)
        
        logger.info(f"Initialized optimized spatial grid: {self.grid_width}x{self.grid_height} cells, "
                   f"lazy_loading={lazy_loading}, cleanup_threshold={cleanup_threshold}")
    
    def _get_or_create_cell(self, grid_index: Tuple[int, int]) -> LazyGridCell:
        """Get existing cell or create new one lazily."""
        with self._lock:
            if grid_index not in self.cells:
                i, j = grid_index
                x = (i + 0.5) * self.cell_size
                y = (j + 0.5) * self.cell_size
                coord = Coordinate(x, y)
                self.cells[grid_index] = LazyGridCell(coordinate=coord)
            
            self._cell_access_count[grid_index] += 1
            return self.cells[grid_index]
    
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
        return Coordinate(x, y)
    
    def is_valid_position(self, coord: Coordinate) -> bool:
        """Check if a coordinate is within the grid boundaries."""
        if self.boundary_condition == BoundaryCondition.PERIODIC:
            return True
        
        return (0 <= coord.x <= self.width and 
                0 <= coord.y <= self.height)
    
    def place_bacterium(self, bacterium_id: str, position: Coordinate) -> bool:
        """Place a bacterium at the specified position."""
        if self._disposed:
            return False
        
        if not self.is_valid_position(position):
            logger.warning(f"Invalid position for bacterium {bacterium_id}: {position}")
            return False
        
        with self._lock:
            # Remove from previous position if exists
            if bacterium_id in self.bacterial_positions:
                self.remove_bacterium(bacterium_id)
            
            # Add to new position
            grid_index = self._coordinate_to_grid_index(position)
            cell = self._get_or_create_cell(grid_index)
            cell.add_bacterium(bacterium_id)
            self.bacterial_positions[bacterium_id] = position
            
            logger.debug(f"Placed bacterium {bacterium_id} at {position}")
            return True
    
    def remove_bacterium(self, bacterium_id: str) -> bool:
        """Remove a bacterium from the grid."""
        if self._disposed:
            return False
        
        with self._lock:
            if bacterium_id not in self.bacterial_positions:
                return False
            
            position = self.bacterial_positions[bacterium_id]
            grid_index = self._coordinate_to_grid_index(position)
            
            if grid_index in self.cells:
                cell = self.cells[grid_index]
                cell.remove_bacterium(bacterium_id)
                
                # Track inactive cells for cleanup
                if not cell.is_active:
                    self._inactive_cell_count += 1
                    if self._inactive_cell_count >= self.cleanup_threshold:
                        self._cleanup_inactive_cells()
            
            del self.bacterial_positions[bacterium_id]
            logger.debug(f"Removed bacterium {bacterium_id} from {position}")
            return True
    
    def move_bacterium(self, bacterium_id: str, new_position: Coordinate) -> bool:
        """Move a bacterium to a new position."""
        if self._disposed:
            return False
        
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
        """Find neighboring bacteria within specified radius."""
        if self._disposed or bacterium_id not in self.bacterial_positions:
            return []
        
        center_pos = self.bacterial_positions[bacterium_id]
        neighbors = []
        
        # Get grid cells within radius
        cell_radius = int(math.ceil(radius / self.cell_size))
        center_grid = self._coordinate_to_grid_index(center_pos)
        
        with self._lock:
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
                    
                    grid_index = (check_i, check_j)
                    if grid_index in self.cells:
                        cell = self.cells[grid_index]
                        if cell._bacteria_ids:  # Only check active cells
                            for other_id in cell._bacteria_ids:
                                if other_id == bacterium_id and not include_self:
                                    continue
                                
                                other_pos = self.bacterial_positions.get(other_id)
                                if other_pos and center_pos.distance_to(other_pos) <= radius:
                                    neighbors.append(other_id)
        
        return neighbors
    
    def get_local_density(self, position: Coordinate, radius: float) -> float:
        """Calculate local bacterial density around a position."""
        if self._disposed:
            return 0.0
        
        bacteria_in_radius = self._get_bacteria_in_radius(position, radius)
        area = math.pi * radius * radius
        return len(bacteria_in_radius) / area if area > 0 else 0.0
    
    def _get_bacteria_in_radius(self, center: Coordinate, radius: float) -> List[str]:
        """Get all bacteria within radius of center position."""
        bacteria = []
        cell_radius = int(math.ceil(radius / self.cell_size))
        center_grid = self._coordinate_to_grid_index(center)
        
        with self._lock:
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
                    
                    grid_index = (check_i, check_j)
                    if grid_index in self.cells:
                        cell = self.cells[grid_index]
                        if cell._bacteria_ids:
                            for bacterium_id in cell._bacteria_ids:
                                bacterium_pos = self.bacterial_positions.get(bacterium_id)
                                if bacterium_pos and center.distance_to(bacterium_pos) <= radius:
                                    bacteria.append(bacterium_id)
        
        return bacteria
    
    def add_antibiotic_zone(
        self, 
        center: Coordinate, 
        radius: float, 
        concentration: float,
        zone_id: str = None
    ):
        """Add an antibiotic zone to the grid."""
        if self._disposed:
            return
        
        zone_info = {
            'center': center,
            'radius': radius,
            'concentration': concentration,
            'id': zone_id or f"zone_{len(self.antibiotic_zones)}"
        }
        
        self.antibiotic_zones.append(zone_info)
        self._update_antibiotic_concentrations()
        
        logger.info(f"Added antibiotic zone at {center} with radius {radius}")
    
    def _update_antibiotic_concentrations(self):
        """Update antibiotic concentrations based on zones."""
        if self._disposed:
            return
        
        # Update existing active cells
        with self._lock:
            for grid_index, cell in self.cells.items():
                if cell.is_active:
                    concentration = 0.0
                    for zone in self.antibiotic_zones:
                        distance = cell.coordinate.distance_to(zone['center'])
                        if distance <= zone['radius']:
                            # Linear decay from center
                            decay_factor = 1.0 - (distance / zone['radius'])
                            concentration += zone['concentration'] * decay_factor
                    
                    cell.antibiotic_concentration = concentration
    
    def get_antibiotic_concentration(self, position: Coordinate) -> float:
        """Get antibiotic concentration at a position."""
        if self._disposed:
            return 0.0
        
        grid_index = self._coordinate_to_grid_index(position)
        
        with self._lock:
            if grid_index in self.cells:
                return self.cells[grid_index].antibiotic_concentration
            else:
                # Calculate on demand for non-active cells
                concentration = 0.0
                for zone in self.antibiotic_zones:
                    distance = position.distance_to(zone['center'])
                    if distance <= zone['radius']:
                        decay_factor = 1.0 - (distance / zone['radius'])
                        concentration += zone['concentration'] * decay_factor
                return concentration
    
    def _cleanup_inactive_cells(self) -> int:
        """Clean up inactive cells to save memory."""
        if self._disposed:
            return 0
        
        cleaned_count = 0
        
        with self._lock:
            inactive_cells = [
                grid_index for grid_index, cell in self.cells.items()
                if not cell.is_active
            ]
            
            for grid_index in inactive_cells:
                del self.cells[grid_index]
                self._cell_access_count.pop(grid_index, None)
                cleaned_count += 1
            
            self._inactive_cell_count = 0
        
        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} inactive cells")
        
        return cleaned_count
    
    def get_grid_statistics(self) -> Dict[str, Any]:
        """Get comprehensive grid statistics."""
        if self._disposed:
            return {}
        
        with self._lock:
            active_cells = sum(1 for cell in self.cells.values() if cell.is_active)
            total_bacteria = len(self.bacterial_positions)
            occupied_cells = sum(1 for cell in self.cells.values() 
                               if cell._bacteria_ids and len(cell._bacteria_ids) > 0)
            
            memory_usage = sum(cell.get_memory_usage() for cell in self.cells.values())
            
            return {
                "total_bacteria": total_bacteria,
                "total_cells": len(self.cells),
                "active_cells": active_cells,
                "occupied_cells": occupied_cells,
                "occupancy_rate": occupied_cells / len(self.cells) if self.cells else 0,
                "antibiotic_zones": len(self.antibiotic_zones),
                "antibiotic_coverage": self._calculate_antibiotic_coverage(),
                "grid_dimensions": [self.grid_width, self.grid_height],
                "physical_dimensions": [self.width, self.height],
                "memory_usage_bytes": memory_usage,
                "memory_usage_mb": memory_usage / 1024 / 1024,
                "inactive_cell_count": self._inactive_cell_count
            }
    
    def _calculate_antibiotic_coverage(self) -> float:
        """Calculate percentage of grid covered by antibiotics."""
        if not self.antibiotic_zones:
            return 0.0
        
        # Simple approximation based on zone areas
        total_area = self.width * self.height
        covered_area = 0.0
        
        for zone in self.antibiotic_zones:
            zone_area = math.pi * zone['radius'] ** 2
            covered_area += min(zone_area, total_area)
        
        return min(100.0, (covered_area / total_area) * 100.0)
    
    def clear(self):
        """Clear all data from the grid."""
        if self._disposed:
            return
        
        with self._lock:
            self.cells.clear()
            self._cell_access_count.clear()
            self.bacterial_positions.clear()
            self.antibiotic_zones.clear()
            self.nutrient_patches.clear()
            self._inactive_cell_count = 0
        
        logger.info("Cleared spatial grid")
    
    def get_cell_at_position(self, position: Coordinate) -> Optional[LazyGridCell]:
        """Get the cell at a specific position."""
        if self._disposed:
            return None
        
        grid_index = self._coordinate_to_grid_index(position)
        with self._lock:
            return self.cells.get(grid_index)
    
    def dispose(self) -> None:
        """Dispose of the spatial grid and clean up resources."""
        if self._disposed:
            return
        
        logger.info("Disposing optimized spatial grid")
        
        with self._lock:
            self.clear()
            self._disposed = True
        
        # Unregister from memory manager
        memory_manager.resource_manager.unregister(self)
    
    @property
    def is_disposed(self) -> bool:
        """Check if the grid has been disposed."""
        return self._disposed
    
    def __del__(self):
        """Ensure disposal on deletion."""
        if not self._disposed:
            self.dispose() 