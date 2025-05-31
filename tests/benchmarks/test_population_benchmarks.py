"""
Benchmark tests for population operations.

Tests the performance of population creation, indexing, filtering, and statistics
operations that were optimized in performance optimization task 20.1.
"""

import pytest
import sys
import os
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from models.population import Population, OptimizedPopulation, PopulationConfig
from models.bacterium import Bacterium
from tests.benchmarks.core import benchmark, performance_test


class PopulationBenchmarks:
    """Benchmark suite for population operations."""
    
    def setup_method(self):
        """Setup for each benchmark test."""
        self.config = PopulationConfig(
            initial_size=1000,
            max_size=10000,
            batch_size=100
        )
    
    @benchmark(
        name="population_creation_1k",
        iterations=10,
        categories=["population", "creation"],
        tags={"size": "1000"}
    )
    def benchmark_population_creation_1k(self):
        """Benchmark creating a population of 1000 bacteria."""
        population = OptimizedPopulation(config=self.config)
        
        # Generate bacteria
        bacteria = []
        for i in range(1000):
            bacterium = Bacterium(
                id=f"bacteria_{i}",
                fitness=0.5 + (i % 100) / 200,  # Vary fitness 0.5-1.0
                resistance_genes={"amp", "tet"} if i % 3 == 0 else {"amp"}
            )
            bacteria.append(bacterium)
        
        # Batch add bacteria (this is the optimized operation)
        population._batch_add_bacteria(bacteria)
        
        return len(population.bacteria_by_id)
    
    @benchmark(
        name="population_creation_5k",
        iterations=5,
        categories=["population", "creation"],
        tags={"size": "5000"}
    )
    def benchmark_population_creation_5k(self):
        """Benchmark creating a population of 5000 bacteria."""
        population = OptimizedPopulation(config=self.config)
        
        bacteria = []
        for i in range(5000):
            bacterium = Bacterium(
                id=f"bacteria_{i}",
                fitness=0.5 + (i % 100) / 200,
                resistance_genes={"amp", "tet", "chl"} if i % 4 == 0 else {"amp"}
            )
            bacteria.append(bacterium)
        
        population._batch_add_bacteria(bacteria)
        return len(population.bacteria_by_id)
    
    @benchmark(
        name="population_indexing_lookups",
        iterations=20,
        categories=["population", "indexing"],
        tags={"operation": "lookup"}
    )
    def benchmark_population_indexing_lookups(self):
        """Benchmark O(1) indexing operations."""
        population = OptimizedPopulation(config=self.config)
        
        # Setup population
        bacteria = []
        for i in range(1000):
            bacterium = Bacterium(
                id=f"bacteria_{i}",
                fitness=0.5 + (i % 100) / 200,
                resistance_genes={"amp", "tet"} if i % 2 == 0 else {"amp"}
            )
            bacteria.append(bacterium)
        
        population._batch_add_bacteria(bacteria)
        
        # Perform many lookups (this tests the indexed access)
        lookup_count = 0
        for i in range(0, 1000, 10):  # 100 lookups
            bacterium_id = f"bacteria_{i}"
            if bacterium_id in population.bacteria_by_id:
                lookup_count += 1
        
        return lookup_count
    
    @benchmark(
        name="population_resistance_filtering",
        iterations=15,
        categories=["population", "filtering"],
        tags={"operation": "filter"}
    )
    def benchmark_population_resistance_filtering(self):
        """Benchmark resistance-based filtering using optimized sets."""
        population = OptimizedPopulation(config=self.config)
        
        # Setup population with mixed resistance
        bacteria = []
        for i in range(2000):
            resistance_genes = set()
            if i % 3 == 0:
                resistance_genes.add("amp")
            if i % 5 == 0:
                resistance_genes.add("tet")
            if i % 7 == 0:
                resistance_genes.add("chl")
            
            bacterium = Bacterium(
                id=f"bacteria_{i}",
                fitness=0.5 + (i % 100) / 200,
                resistance_genes=resistance_genes
            )
            bacteria.append(bacterium)
        
        population._batch_add_bacteria(bacteria)
        
        # Test optimized filtering operations
        resistant_count = len(population.resistant_bacteria)
        sensitive_count = len(population.sensitive_bacteria)
        
        return resistant_count + sensitive_count
    
    @benchmark(
        name="population_statistics_cached",
        iterations=50,
        categories=["population", "statistics"],
        tags={"operation": "stats"}
    )
    def benchmark_population_statistics_cached(self):
        """Benchmark cached statistics calculation."""
        population = OptimizedPopulation(config=self.config)
        
        # Setup population
        bacteria = []
        for i in range(1000):
            bacterium = Bacterium(
                id=f"bacteria_{i}",
                fitness=0.3 + (i % 100) / 150,  # Vary fitness 0.3-1.0
                resistance_genes={"amp", "tet"} if i % 4 == 0 else {"amp"} if i % 2 == 0 else set()
            )
            bacteria.append(bacterium)
        
        population._batch_add_bacteria(bacteria)
        
        # Multiple statistics calls (should be cached after first)
        stats_calls = 0
        for _ in range(50):
            stats = population.get_statistics()
            if 'avg_fitness' in stats:
                stats_calls += 1
        
        return stats_calls
    
    @performance_test(
        max_execution_time=0.1,  # 100ms max
        max_memory_usage=50.0,   # 50MB max
        min_operations_per_second=100.0,  # Min 100 ops/sec
        name="population_batch_operations",
        iterations=10,
        categories=["population", "batch"]
    )
    def benchmark_population_batch_operations(self):
        """Performance test for batch operations with constraints."""
        population = OptimizedPopulation(config=self.config)
        
        # Test batch processing
        for batch_num in range(10):  # 10 batches
            bacteria = []
            for i in range(100):  # 100 bacteria per batch
                bacterium = Bacterium(
                    id=f"batch_{batch_num}_bacteria_{i}",
                    fitness=0.4 + (i % 60) / 100,
                    resistance_genes={"amp"} if i % 2 == 0 else set()
                )
                bacteria.append(bacterium)
            
            population._batch_add_bacteria(bacteria)
        
        return len(population.bacteria_by_id)
    
    @benchmark(
        name="population_memory_optimization",
        iterations=5,
        categories=["population", "memory"],
        tags={"optimization": "memory"}
    )
    def benchmark_population_memory_optimization(self):
        """Benchmark memory optimization features."""
        population = OptimizedPopulation(config=self.config)
        
        # Create large population to test memory efficiency
        bacteria = []
        for i in range(10000):
            bacterium = Bacterium(
                id=f"bacteria_{i}",
                fitness=0.1 + (i % 90) / 100,
                resistance_genes={"amp", "tet", "chl"} if i % 8 == 0 else {"amp"}
            )
            bacteria.append(bacterium)
        
        # Use optimized batch addition
        population._batch_add_bacteria(bacteria)
        
        # Test memory-efficient operations
        operations_count = 0
        
        # Statistics (cached)
        stats = population.get_statistics()
        operations_count += 1
        
        # Resistance filtering (indexed)
        resistant = len(population.resistant_bacteria)
        operations_count += 1
        
        # ID lookups (O(1))
        for i in range(0, 10000, 100):  # 100 lookups
            if f"bacteria_{i}" in population.bacteria_by_id:
                operations_count += 1
        
        return operations_count


# Convenience functions for running benchmarks
def run_population_benchmarks():
    """Run all population benchmarks."""
    from tests.benchmarks.core import BenchmarkRunner
    
    runner = BenchmarkRunner()
    suite = runner.create_suite("population_operations", "Population performance benchmarks")
    
    benchmark_instance = PopulationBenchmarks()
    
    # Add all benchmark methods
    suite.add_benchmark(benchmark_instance.benchmark_population_creation_1k, "population_creation_1k")
    suite.add_benchmark(benchmark_instance.benchmark_population_creation_5k, "population_creation_5k")
    suite.add_benchmark(benchmark_instance.benchmark_population_indexing_lookups, "population_indexing_lookups")
    suite.add_benchmark(benchmark_instance.benchmark_population_resistance_filtering, "population_resistance_filtering")
    suite.add_benchmark(benchmark_instance.benchmark_population_statistics_cached, "population_statistics_cached")
    suite.add_benchmark(benchmark_instance.benchmark_population_batch_operations, "population_batch_operations")
    suite.add_benchmark(benchmark_instance.benchmark_population_memory_optimization, "population_memory_optimization")
    
    return runner.run_suite("population_operations")


if __name__ == "__main__":
    # Run benchmarks when script is executed directly
    report = run_population_benchmarks()
    print(f"Population benchmarks completed: {len(report.results)} tests run")
    if report.violations:
        print(f"Budget violations: {len(report.violations)}")
    print(f"Total execution time: {report.execution_time:.2f}s") 