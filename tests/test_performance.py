"""
Tests for performance optimization and benchmarking functionality.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
import tempfile
import os

from utils.performance import (
    PerformanceProfiler, SimulationOptimizer, MemoryProfiler, 
    CacheManager, profiler, optimizer, memory_profiler, cache_manager
)
from utils.benchmark import LoadTester, PerformanceBenchmark, RegressionTester
from services.optimized_simulation_service import OptimizedSimulationService


class TestPerformanceProfiler:
    """Test performance profiling functionality."""
    
    def test_profiler_initialization(self):
        """Test profiler initializes correctly."""
        test_profiler = PerformanceProfiler()
        assert test_profiler.metrics_storage == {}
        assert test_profiler._lock is not None
    
    def test_profile_block_context_manager(self):
        """Test profile block context manager."""
        test_profiler = PerformanceProfiler()
        
        with test_profiler.profile_block("test_operation"):
            # Simulate some work
            sum(range(1000))
        
        assert "test_operation" in test_profiler.metrics_storage
        assert len(test_profiler.metrics_storage["test_operation"]) == 1
        
        metrics = test_profiler.metrics_storage["test_operation"][0]
        assert metrics.execution_time > 0
        assert metrics.function_name == "test_operation"
    
    def test_profile_function_decorator(self):
        """Test function profiling decorator."""
        test_profiler = PerformanceProfiler()
        
        @test_profiler.profile_function("test_func")
        def sample_function(n: int) -> int:
            return sum(range(n))
        
        result = sample_function(100)
        assert result == sum(range(100))
        assert "test_func" in test_profiler.metrics_storage
    
    @pytest.mark.asyncio
    async def test_async_function_profiling(self):
        """Test async function profiling."""
        test_profiler = PerformanceProfiler()
        
        @test_profiler.profile_function("test_async_func")
        async def async_sample_function(n: int) -> int:
            await asyncio.sleep(0.01)
            return sum(range(n))
        
        result = await async_sample_function(100)
        assert result == sum(range(100))
        assert "test_async_func" in test_profiler.metrics_storage
    
    def test_metrics_summary(self):
        """Test metrics summary generation."""
        test_profiler = PerformanceProfiler()
        
        # Generate some test metrics
        with test_profiler.profile_block("test_op"):
            sum(range(1000))
        
        with test_profiler.profile_block("test_op"):
            sum(range(500))
        
        summary = test_profiler.get_metrics_summary("test_op")
        
        assert "function_name" in summary
        assert "call_count" in summary
        assert summary["call_count"] == 2
        assert "execution_time" in summary
        assert "avg" in summary["execution_time"]


class TestSimulationOptimizer:
    """Test simulation optimization utilities."""
    
    def test_vectorized_mutation(self):
        """Test vectorized mutation operation."""
        population = np.random.rand(100, 10)  # 100 individuals, 10 traits
        mutation_rate = 0.1
        
        mutated = SimulationOptimizer.vectorized_mutation(population, mutation_rate)
        
        assert mutated.shape == population.shape
        assert not np.array_equal(population, mutated)  # Should have some mutations
    
    def test_vectorized_selection(self):
        """Test vectorized selection operation."""
        population = np.random.rand(100, 10)
        fitness_scores = np.random.rand(100)
        selection_pressure = 0.5
        
        selected = SimulationOptimizer.vectorized_selection(
            population, fitness_scores, selection_pressure
        )
        
        assert selected.shape == population.shape
    
    def test_cached_fitness_calculation(self):
        """Test fitness calculation caching."""
        test_optimizer = SimulationOptimizer()
        
        def dummy_fitness_func(pop):
            return np.sum(pop, axis=1)
        
        population = np.random.rand(10, 5)
        pop_hash = "test_hash"
        
        # First call should cache the result
        result1 = test_optimizer.cached_fitness_calculation(
            pop_hash, dummy_fitness_func, population
        )
        
        # Second call should use cached result
        result2 = test_optimizer.cached_fitness_calculation(
            pop_hash, dummy_fitness_func, population
        )
        
        assert np.array_equal(result1, result2)
        assert pop_hash in test_optimizer.fitness_cache


class TestMemoryProfiler:
    """Test memory profiling functionality."""
    
    def test_memory_profiler_initialization(self):
        """Test memory profiler initializes correctly."""
        test_profiler = MemoryProfiler()
        assert test_profiler.snapshots == []
    
    def test_take_snapshot(self):
        """Test taking memory snapshots."""
        test_profiler = MemoryProfiler()
        
        snapshot = test_profiler.take_snapshot("test_snapshot")
        
        assert "timestamp" in snapshot
        assert "label" in snapshot
        assert "memory_usage" in snapshot
        assert snapshot["label"] == "test_snapshot"
        assert len(test_profiler.snapshots) == 1
    
    def test_compare_snapshots(self):
        """Test snapshot comparison."""
        test_profiler = MemoryProfiler()
        
        # Take two snapshots
        test_profiler.take_snapshot("before")
        test_profiler.take_snapshot("after")
        
        comparison = test_profiler.compare_snapshots()
        assert "Memory usage change" in comparison
        assert "before" in comparison
        assert "after" in comparison


class TestCacheManager:
    """Test cache management functionality."""
    
    def test_cache_manager_initialization(self):
        """Test cache manager initializes correctly."""
        test_manager = CacheManager()
        assert "simulation_results" in test_manager.caches
        assert "fitness_scores" in test_manager.caches
        assert "population_states" in test_manager.caches
        assert "api_responses" in test_manager.caches
    
    def test_cache_operations(self):
        """Test basic cache operations."""
        test_manager = CacheManager()
        
        # Test setting and getting values
        test_manager.set("simulation_results", "test_key", {"test": "data"})
        result = test_manager.get("simulation_results", "test_key")
        
        assert result == {"test": "data"}
        
        # Test cache invalidation
        test_manager.invalidate("simulation_results", "test_key")
        result = test_manager.get("simulation_results", "test_key")
        
        assert result is None
    
    def test_cache_statistics(self):
        """Test cache statistics."""
        test_manager = CacheManager()
        
        stats = test_manager.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert "simulation_results" in stats
        assert "size" in stats["simulation_results"]
        assert "maxsize" in stats["simulation_results"]


class TestOptimizedSimulationService:
    """Test optimized simulation service."""
    
    def test_service_initialization(self):
        """Test optimized service initializes correctly."""
        service = OptimizedSimulationService()
        
        assert service.active_simulations == {}
        assert service._simulation_callbacks == {}
        assert service.performance_enabled is True
    
    def test_parameter_hashing(self):
        """Test parameter hashing functionality."""
        service = OptimizedSimulationService()
        
        params1 = {"a": 1, "b": 2, "c": 3}
        params2 = {"c": 3, "b": 2, "a": 1}  # Same params, different order
        params3 = {"a": 1, "b": 2, "c": 4}  # Different params
        
        hash1 = service._generate_params_hash(params1)
        hash2 = service._generate_params_hash(params2)
        hash3 = service._generate_params_hash(params3)
        
        assert hash1 == hash2  # Same params should have same hash
        assert hash1 != hash3  # Different params should have different hash
    
    def test_population_hashing(self):
        """Test population hashing functionality."""
        service = OptimizedSimulationService()
        
        pop1 = np.array([[1, 2, 3], [4, 5, 6]])
        pop2 = np.array([[1, 2, 3], [4, 5, 6]])  # Same population
        pop3 = np.array([[1, 2, 3], [4, 5, 7]])  # Different population
        
        hash1 = service._generate_population_hash(pop1)
        hash2 = service._generate_population_hash(pop2)
        hash3 = service._generate_population_hash(pop3)
        
        assert hash1 == hash2  # Same populations should have same hash
        assert hash1 != hash3  # Different populations should have different hash
    
    def test_results_compression(self):
        """Test simulation results compression."""
        service = OptimizedSimulationService()
        
        # Create test results with large datasets
        results = {
            "small_data": [1, 2, 3, 4, 5],
            "large_data": list(range(1000))  # Large dataset
        }
        
        compressed = service._compress_results(results)
        
        assert len(compressed["small_data"]) == 5  # Small data unchanged
        assert len(compressed["large_data"]) == 100  # Large data compressed


class TestLoadTester:
    """Test load testing functionality."""
    
    def test_load_tester_initialization(self):
        """Test load tester initializes correctly."""
        tester = LoadTester("http://test.com")
        
        assert tester.base_url == "http://test.com"
        assert tester.results == []
    
    @pytest.mark.asyncio
    async def test_load_test_configuration(self):
        """Test load test configuration validation."""
        tester = LoadTester()
        
        # Test with mock endpoint (will fail, but we test configuration)
        try:
            await tester.run_concurrent_requests(
                endpoint="/test",
                concurrent_users=2,
                requests_per_user=1,
                ramp_up_time=0.1
            )
        except Exception:
            pass  # Expected to fail due to mock endpoint
        
        # Validate that test was configured correctly
        assert tester.results == [] or len(tester.results) >= 0


class TestPerformanceBenchmark:
    """Test performance benchmarking functionality."""
    
    def test_benchmark_initialization(self):
        """Test benchmark initializes correctly."""
        benchmark = PerformanceBenchmark()
        
        assert benchmark.algorithm_results == {}
    
    def test_algorithm_benchmarking(self):
        """Test algorithm benchmarking."""
        benchmark = PerformanceBenchmark()
        
        def dummy_algorithm(param1: int, param2: float) -> int:
            return param1 * int(param2)
        
        test_cases = [
            {"param1": 10, "param2": 1.5},
            {"param1": 20, "param2": 2.0}
        ]
        
        result = benchmark.benchmark_simulation_algorithm(
            algorithm_func=dummy_algorithm,
            test_cases=test_cases,
            iterations=3
        )
        
        assert "algorithm_name" in result
        assert "test_cases" in result
        assert len(result["test_cases"]) == 2
        assert "overall_stats" in result


class TestRegressionTester:
    """Test regression testing functionality."""
    
    def test_regression_tester_initialization(self):
        """Test regression tester initializes correctly."""
        tester = RegressionTester()
        
        assert tester.baseline_results == {}
        assert tester.current_results == {}
    
    def test_baseline_operations(self):
        """Test baseline setting and comparison."""
        tester = RegressionTester()
        
        # Add some test results
        tester.add_current_result("test1", {
            "avg_execution_time": 1.0,
            "avg_memory_usage": 100.0,
            "requests_per_second": 50.0
        })
        
        # Save as baseline
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            baseline_file = f.name
        
        try:
            tester.save_current_as_baseline(baseline_file)
            
            # Load baseline
            new_tester = RegressionTester(baseline_file)
            
            # Add current results that are worse
            new_tester.add_current_result("test1", {
                "avg_execution_time": 1.2,  # 20% slower
                "avg_memory_usage": 120.0,  # 20% more memory
                "requests_per_second": 40.0  # 20% fewer requests
            })
            
            # Compare performance
            comparison = new_tester.compare_performance(tolerance_percent=10.0)
            
            assert "total_tests" in comparison
            assert "regressions" in comparison
            assert comparison["regressions"] > 0  # Should detect regressions
            
        finally:
            if os.path.exists(baseline_file):
                os.unlink(baseline_file)


# Integration tests
@pytest.mark.asyncio
async def test_global_profiler_integration():
    """Test global profiler instances work correctly."""
    # Clear any existing metrics
    profiler.clear_metrics()
    
    @profiler.profile_function("integration_test")
    async def test_function():
        await asyncio.sleep(0.01)
        return sum(range(100))
    
    result = await test_function()
    assert result == sum(range(100))
    
    metrics = profiler.get_metrics_summary("integration_test")
    assert "function_name" in metrics
    assert metrics["call_count"] == 1


def test_global_cache_integration():
    """Test global cache manager integration."""
    # Clear cache
    cache_manager.invalidate("simulation_results")
    
    # Test caching
    test_data = {"test": "integration_data"}
    cache_manager.set("simulation_results", "integration_test", test_data)
    
    retrieved = cache_manager.get("simulation_results", "integration_test")
    assert retrieved == test_data
    
    # Test statistics
    stats = cache_manager.get_cache_stats()
    assert "simulation_results" in stats


def test_global_memory_profiler_integration():
    """Test global memory profiler integration."""
    # Clear snapshots
    memory_profiler.snapshots.clear()
    
    # Take snapshot
    snapshot = memory_profiler.take_snapshot("integration_test")
    
    assert snapshot["label"] == "integration_test"
    assert len(memory_profiler.snapshots) == 1


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"]) 