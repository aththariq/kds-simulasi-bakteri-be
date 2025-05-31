"""
Benchmark tests for spatial grid operations.

Tests the performance of spatial grid operations including memory management,
object pooling, and lazy loading that were optimized in performance optimization task 20.4.
"""

import pytest
import sys
import os
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from models.spatial import SpatialGrid, SpatialManager, Coordinate, BoundaryCondition
from tests.benchmarks.core import benchmark, performance_test


class SpatialBenchmarks:
    """Benchmark suite for spatial operations."""
    
    def setup_method(self):
        """Setup for each benchmark test."""
        self.grid_size = 100.0
        self.cell_size = 1.0
    
    @benchmark(
        name="spatial_grid_creation",
        iterations=10,
        categories=["spatial", "creation"],
        tags={"size": "100x100"}
    )
    def benchmark_spatial_grid_creation(self):
        """Benchmark spatial grid creation with lazy loading."""
        grid = SpatialGrid(
            width=self.grid_size,
            height=self.grid_size,
            cell_size=self.cell_size,
            boundary_condition=BoundaryCondition.PERIODIC
        )
        
        # Enable lazy loading to test memory optimization
        grid.enable_lazy_loading(True)
        
        return grid.cols * grid.rows
    
    @benchmark(
        name="spatial_coordinate_pooling",
        iterations=20,
        categories=["spatial", "memory"],
        tags={"optimization": "pooling"}
    )
    def benchmark_spatial_coordinate_pooling(self):
        """Benchmark coordinate object pooling."""
        operations_count = 0
        
        # Create many coordinates using pooling
        coordinates = []
        for i in range(1000):
            coord = Coordinate.create_pooled(
                x=float(i % 100),
                y=float(i // 100)
            )
            coordinates.append(coord)
            operations_count += 1
        
        # Return coordinates to pool
        for coord in coordinates:
            coord.release_to_pool()
            operations_count += 1
        
        return operations_count
    
    @benchmark(
        name="spatial_bacteria_placement",
        iterations=15,
        categories=["spatial", "placement"],
        tags={"operation": "place"}
    )
    def benchmark_spatial_bacteria_placement(self):
        """Benchmark bacteria placement on spatial grid."""
        grid = SpatialGrid(
            width=self.grid_size,
            height=self.grid_size,
            cell_size=self.cell_size
        )
        grid.enable_lazy_loading(True)
        
        # Place bacteria across the grid
        placement_count = 0
        for i in range(1000):
            x = (i % 100) * 1.0
            y = (i // 100) * 1.0
            position = Coordinate.create_pooled(x, y)
            
            if grid.place_bacterium(f"bacteria_{i}", position):
                placement_count += 1
            
            position.release_to_pool()
        
        return placement_count
    
    @benchmark(
        name="spatial_bacteria_movement",
        iterations=10,
        categories=["spatial", "movement"],
        tags={"operation": "move"}
    )
    def benchmark_spatial_bacteria_movement(self):
        """Benchmark bacteria movement operations."""
        grid = SpatialGrid(
            width=self.grid_size,
            height=self.grid_size,
            cell_size=self.cell_size
        )
        grid.enable_lazy_loading(True)
        
        # Place initial bacteria
        bacteria_ids = []
        for i in range(500):
            x = (i % 50) * 2.0
            y = (i // 50) * 2.0
            position = Coordinate.create_pooled(x, y)
            
            bacterium_id = f"bacteria_{i}"
            if grid.place_bacterium(bacterium_id, position):
                bacteria_ids.append(bacterium_id)
            
            position.release_to_pool()
        
        # Move bacteria to new positions
        movement_count = 0
        for i, bacterium_id in enumerate(bacteria_ids):
            new_x = ((i + 25) % 50) * 2.0
            new_y = ((i + 25) // 50) * 2.0
            new_position = Coordinate.create_pooled(new_x, new_y)
            
            if grid.move_bacterium(bacterium_id, new_position):
                movement_count += 1
            
            new_position.release_to_pool()
        
        return movement_count
    
    @benchmark(
        name="spatial_neighbor_search",
        iterations=20,
        categories=["spatial", "search"],
        tags={"operation": "neighbors"}
    )
    def benchmark_spatial_neighbor_search(self):
        """Benchmark neighbor finding operations."""
        grid = SpatialGrid(
            width=self.grid_size,
            height=self.grid_size,
            cell_size=self.cell_size
        )
        grid.enable_lazy_loading(True)
        
        # Place bacteria in a clustered pattern
        bacteria_ids = []
        for i in range(200):
            x = 50.0 + (i % 20) * 1.0  # Cluster around center
            y = 50.0 + (i // 20) * 1.0
            position = Coordinate.create_pooled(x, y)
            
            bacterium_id = f"bacteria_{i}"
            if grid.place_bacterium(bacterium_id, position):
                bacteria_ids.append(bacterium_id)
            
            position.release_to_pool()
        
        # Find neighbors for multiple bacteria
        neighbor_searches = 0
        for bacterium_id in bacteria_ids[::5]:  # Every 5th bacterium
            neighbors = grid.get_neighbors(bacterium_id, radius=5.0)
            neighbor_searches += len(neighbors)
        
        return neighbor_searches
    
    @benchmark(
        name="spatial_lazy_loading",
        iterations=5,
        categories=["spatial", "memory"],
        tags={"optimization": "lazy"}
    )
    def benchmark_spatial_lazy_loading(self):
        """Benchmark lazy loading cell creation."""
        # Large grid to test lazy loading benefits
        grid = SpatialGrid(
            width=1000.0,
            height=1000.0,
            cell_size=1.0
        )
        grid.enable_lazy_loading(True)
        
        # Only access a small portion of the grid
        operations_count = 0
        for i in range(100):
            x = float(i % 10) * 10.0  # Only use 10x10 area of 1000x1000 grid
            y = float(i // 10) * 10.0
            position = Coordinate.create_pooled(x, y)
            
            # This should only create cells as needed
            if grid.place_bacterium(f"bacteria_{i}", position):
                operations_count += 1
            
            position.release_to_pool()
        
        # Check that only a fraction of cells were created
        created_cells = len([cell for cell in grid.cells.values() if cell is not None])
        return created_cells  # Should be much less than 1,000,000
    
    @performance_test(
        max_execution_time=0.05,  # 50ms max
        max_memory_usage=30.0,    # 30MB max
        min_operations_per_second=1000.0,  # Min 1000 ops/sec
        name="spatial_high_performance",
        iterations=15,
        categories=["spatial", "performance"]
    )
    def benchmark_spatial_high_performance(self):
        """Performance test for spatial operations with strict constraints."""
        grid = SpatialGrid(
            width=50.0,
            height=50.0,
            cell_size=1.0
        )
        grid.enable_lazy_loading(True)
        
        operations_count = 0
        
        # Rapid placement and movement
        for i in range(100):
            # Place
            position = Coordinate.create_pooled(
                x=float(i % 50),
                y=float(i // 50)
            )
            if grid.place_bacterium(f"bacteria_{i}", position):
                operations_count += 1
            position.release_to_pool()
            
            # Immediate move
            new_position = Coordinate.create_pooled(
                x=float((i + 1) % 50),
                y=float((i + 1) // 50)
            )
            if grid.move_bacterium(f"bacteria_{i}", new_position):
                operations_count += 1
            new_position.release_to_pool()
        
        return operations_count
    
    @benchmark(
        name="spatial_manager_operations",
        iterations=10,
        categories=["spatial", "manager"],
        tags={"component": "manager"}
    )
    def benchmark_spatial_manager_operations(self):
        """Benchmark SpatialManager operations."""
        grid = SpatialGrid(
            width=self.grid_size,
            height=self.grid_size,
            cell_size=self.cell_size
        )
        grid.enable_lazy_loading(True)
        
        manager = SpatialManager(grid)
        
        # Initialize random population
        bacteria_ids = [f"bacteria_{i}" for i in range(500)]
        positions = manager.initialize_random_population(500, bacteria_ids)
        
        operations_count = len(positions)
        
        # Test batch position updates
        updates = []
        for i, (bacterium_id, position) in enumerate(positions.items()):
            if i % 2 == 0:  # Update every other bacterium
                new_position = Coordinate.create_pooled(
                    x=position.x + 1.0,
                    y=position.y + 1.0
                )
                updates.append((bacterium_id, new_position))
        
        manager.update_bacterium_position_batch(updates)
        operations_count += len(updates)
        
        # Clean up pooled coordinates
        for _, new_position in updates:
            new_position.release_to_pool()
        
        return operations_count
    
    @benchmark(
        name="spatial_memory_cleanup",
        iterations=8,
        categories=["spatial", "memory"],
        tags={"operation": "cleanup"}
    )
    def benchmark_spatial_memory_cleanup(self):
        """Benchmark memory cleanup operations."""
        grid = SpatialGrid(
            width=self.grid_size,
            height=self.grid_size,
            cell_size=self.cell_size
        )
        grid.enable_lazy_loading(True)
        
        manager = SpatialManager(grid)
        
        # Create population
        bacteria_ids = [f"bacteria_{i}" for i in range(1000)]
        positions = manager.initialize_random_population(1000, bacteria_ids)
        
        # Simulate some bacteria becoming inactive
        active_bacteria = set(bacteria_ids[::2])  # Keep every other bacterium
        
        # Test memory cleanup
        cleanup_stats = manager.cleanup_memory_resources()
        manager.cleanup_inactive_bacteria(active_bacteria)
        
        # Get memory optimization statistics
        memory_stats = grid.get_memory_usage_stats()
        optimization_stats = grid.optimize_memory_usage()
        
        return len(active_bacteria) + optimization_stats.get('cells_optimized', 0)


# Convenience functions for running benchmarks
def run_spatial_benchmarks():
    """Run all spatial benchmarks."""
    from tests.benchmarks.core import BenchmarkRunner
    
    runner = BenchmarkRunner()
    suite = runner.create_suite("spatial_operations", "Spatial grid performance benchmarks")
    
    benchmark_instance = SpatialBenchmarks()
    
    # Add all benchmark methods
    suite.add_benchmark(benchmark_instance.benchmark_spatial_grid_creation, "spatial_grid_creation")
    suite.add_benchmark(benchmark_instance.benchmark_spatial_coordinate_pooling, "spatial_coordinate_pooling")
    suite.add_benchmark(benchmark_instance.benchmark_spatial_bacteria_placement, "spatial_bacteria_placement")
    suite.add_benchmark(benchmark_instance.benchmark_spatial_bacteria_movement, "spatial_bacteria_movement")
    suite.add_benchmark(benchmark_instance.benchmark_spatial_neighbor_search, "spatial_neighbor_search")
    suite.add_benchmark(benchmark_instance.benchmark_spatial_lazy_loading, "spatial_lazy_loading")
    suite.add_benchmark(benchmark_instance.benchmark_spatial_high_performance, "spatial_high_performance")
    suite.add_benchmark(benchmark_instance.benchmark_spatial_manager_operations, "spatial_manager_operations")
    suite.add_benchmark(benchmark_instance.benchmark_spatial_memory_cleanup, "spatial_memory_cleanup")
    
    return runner.run_suite("spatial_operations")


if __name__ == "__main__":
    # Run benchmarks when script is executed directly
    report = run_spatial_benchmarks()
    print(f"Spatial benchmarks completed: {len(report.results)} tests run")
    if report.violations:
        print(f"Budget violations: {len(report.violations)}")
    print(f"Total execution time: {report.execution_time:.2f}s") 