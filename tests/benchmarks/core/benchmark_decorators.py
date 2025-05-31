"""
Decorators for creating benchmark tests with standardized execution and measurement.
"""

import asyncio
import functools
import time
import tracemalloc
from typing import Callable, Any, Dict, Optional
from .benchmark_utils import (
    BenchmarkConfig, BenchmarkResult, BenchmarkTimer, 
    ResourceMonitor, BenchmarkStorage
)


def benchmark(
    name: Optional[str] = None,
    iterations: int = 10,
    warmup_iterations: int = 3,
    timeout_seconds: Optional[float] = None,
    memory_tracking: bool = True,
    cpu_tracking: bool = True,
    categories: Optional[list] = None,
    tags: Optional[dict] = None,
    store_results: bool = True,
    compare_with_baseline: bool = False
):
    """
    Decorator to mark a function as a benchmark test.
    
    Args:
        name: Benchmark name (defaults to function name)
        iterations: Number of test iterations
        warmup_iterations: Number of warmup iterations before measurement
        timeout_seconds: Maximum time per iteration
        memory_tracking: Track memory usage
        cpu_tracking: Track CPU usage
        categories: Categories for organization
        tags: Additional metadata tags
        store_results: Save results to storage
        compare_with_baseline: Compare with previous results
    """
    def decorator(func: Callable) -> Callable:
        benchmark_name = name or func.__name__
        config = BenchmarkConfig(
            name=benchmark_name,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            timeout_seconds=timeout_seconds,
            memory_tracking=memory_tracking,
            cpu_tracking=cpu_tracking,
            categories=categories or [],
            tags=tags or {}
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> BenchmarkResult:
            return _run_sync_benchmark(func, config, store_results, compare_with_baseline, *args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> BenchmarkResult:
            return await _run_async_benchmark(func, config, store_results, compare_with_baseline, *args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def performance_test(
    max_execution_time: Optional[float] = None,
    max_memory_usage: Optional[float] = None,
    min_operations_per_second: Optional[float] = None,
    **benchmark_kwargs
):
    """
    Decorator that combines benchmark testing with performance assertions.
    
    Args:
        max_execution_time: Maximum allowed execution time in seconds
        max_memory_usage: Maximum allowed memory usage in MB
        min_operations_per_second: Minimum required operations per second
        **benchmark_kwargs: Arguments passed to @benchmark decorator
    """
    def decorator(func: Callable) -> Callable:
        # Apply benchmark decorator first
        benchmarked_func = benchmark(**benchmark_kwargs)(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> BenchmarkResult:
            result = benchmarked_func(*args, **kwargs)
            _validate_performance_constraints(
                result, max_execution_time, max_memory_usage, min_operations_per_second
            )
            return result
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> BenchmarkResult:
            result = await benchmarked_func(*args, **kwargs)
            _validate_performance_constraints(
                result, max_execution_time, max_memory_usage, min_operations_per_second
            )
            return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(benchmarked_func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def _run_sync_benchmark(
    func: Callable, 
    config: BenchmarkConfig, 
    store_results: bool,
    compare_with_baseline: bool,
    *args, 
    **kwargs
) -> BenchmarkResult:
    """Execute synchronous benchmark test."""
    result = BenchmarkResult(name=config.name, config=config)
    timer = BenchmarkTimer()
    monitor = ResourceMonitor() if (config.memory_tracking or config.cpu_tracking) else None
    
    try:
        # Warmup iterations
        for _ in range(config.warmup_iterations):
            try:
                if config.timeout_seconds:
                    _run_with_timeout(func, config.timeout_seconds, *args, **kwargs)
                else:
                    func(*args, **kwargs)
            except Exception:
                pass  # Ignore warmup errors
        
        # Start monitoring
        if monitor:
            monitor.start_monitoring()
        
        # Benchmark iterations
        successful_iterations = 0
        for iteration in range(config.iterations):
            try:
                if monitor:
                    monitor.sample_resources()
                
                with timer.time_it():
                    if config.timeout_seconds:
                        _run_with_timeout(func, config.timeout_seconds, *args, **kwargs)
                    else:
                        func(*args, **kwargs)
                
                result.execution_times.append(timer.duration)
                successful_iterations += 1
                
                if monitor:
                    monitor.sample_resources()
                    
            except Exception as e:
                result.errors.append(str(e))
        
        # Stop monitoring and collect resource data
        if monitor:
            resource_data = monitor.stop_monitoring()
            result.memory_usage = resource_data['memory']
            result.cpu_usage = resource_data['cpu']
        
        # Calculate success rate
        result.success_rate = successful_iterations / config.iterations if config.iterations > 0 else 0.0
        
        # Store and compare results if requested
        if store_results:
            storage = BenchmarkStorage()
            storage.save_result(result)
            
            if compare_with_baseline:
                baseline = storage.get_latest_result(config.name)
                if baseline and baseline.timestamp != result.timestamp:
                    comparison = storage.compare_results(result, baseline)
                    result.metadata['baseline_comparison'] = comparison
        
    except Exception as e:
        result.errors.append(f"Benchmark execution failed: {str(e)}")
        result.success_rate = 0.0
    
    return result


async def _run_async_benchmark(
    func: Callable, 
    config: BenchmarkConfig, 
    store_results: bool,
    compare_with_baseline: bool,
    *args, 
    **kwargs
) -> BenchmarkResult:
    """Execute asynchronous benchmark test."""
    result = BenchmarkResult(name=config.name, config=config)
    timer = BenchmarkTimer()
    monitor = ResourceMonitor() if (config.memory_tracking or config.cpu_tracking) else None
    
    try:
        # Warmup iterations
        for _ in range(config.warmup_iterations):
            try:
                if config.timeout_seconds:
                    await asyncio.wait_for(func(*args, **kwargs), timeout=config.timeout_seconds)
                else:
                    await func(*args, **kwargs)
            except Exception:
                pass  # Ignore warmup errors
        
        # Start monitoring
        if monitor:
            monitor.start_monitoring()
        
        # Benchmark iterations
        successful_iterations = 0
        for iteration in range(config.iterations):
            try:
                if monitor:
                    monitor.sample_resources()
                
                timer.start()
                if config.timeout_seconds:
                    await asyncio.wait_for(func(*args, **kwargs), timeout=config.timeout_seconds)
                else:
                    await func(*args, **kwargs)
                duration = timer.stop()
                
                result.execution_times.append(duration)
                successful_iterations += 1
                
                if monitor:
                    monitor.sample_resources()
                    
            except Exception as e:
                result.errors.append(str(e))
        
        # Stop monitoring and collect resource data
        if monitor:
            resource_data = monitor.stop_monitoring()
            result.memory_usage = resource_data['memory']
            result.cpu_usage = resource_data['cpu']
        
        # Calculate success rate
        result.success_rate = successful_iterations / config.iterations if config.iterations > 0 else 0.0
        
        # Store and compare results if requested
        if store_results:
            storage = BenchmarkStorage()
            storage.save_result(result)
            
            if compare_with_baseline:
                baseline = storage.get_latest_result(config.name)
                if baseline and baseline.timestamp != result.timestamp:
                    comparison = storage.compare_results(result, baseline)
                    result.metadata['baseline_comparison'] = comparison
        
    except Exception as e:
        result.errors.append(f"Async benchmark execution failed: {str(e)}")
        result.success_rate = 0.0
    
    return result


def _run_with_timeout(func: Callable, timeout: float, *args, **kwargs) -> Any:
    """Run a synchronous function with timeout using signal (Unix only) or threading."""
    import threading
    import time
    
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        raise TimeoutError(f"Function execution exceeded {timeout} seconds")
    
    if exception[0]:
        raise exception[0]
    
    return result[0]


def _validate_performance_constraints(
    result: BenchmarkResult,
    max_execution_time: Optional[float],
    max_memory_usage: Optional[float], 
    min_operations_per_second: Optional[float]
):
    """Validate that benchmark results meet performance constraints."""
    failures = []
    
    if max_execution_time and result.avg_execution_time > max_execution_time:
        failures.append(
            f"Execution time {result.avg_execution_time:.4f}s exceeds limit {max_execution_time}s"
        )
    
    if max_memory_usage and result.avg_memory_usage > max_memory_usage:
        failures.append(
            f"Memory usage {result.avg_memory_usage:.2f}MB exceeds limit {max_memory_usage}MB"
        )
    
    if min_operations_per_second and result.operations_per_second() < min_operations_per_second:
        failures.append(
            f"Operations per second {result.operations_per_second():.2f} below minimum {min_operations_per_second}"
        )
    
    if failures:
        raise AssertionError(f"Performance constraints failed: {'; '.join(failures)}")


# Convenience functions for common benchmark patterns
def quick_benchmark(func: Callable, iterations: int = 5) -> BenchmarkResult:
    """Quick benchmark with minimal configuration."""
    return benchmark(iterations=iterations, warmup_iterations=1)(func)()


def memory_benchmark(func: Callable, iterations: int = 10) -> BenchmarkResult:
    """Benchmark focused on memory usage."""
    return benchmark(
        iterations=iterations,
        memory_tracking=True,
        cpu_tracking=False
    )(func)()


def speed_benchmark(func: Callable, iterations: int = 20) -> BenchmarkResult:
    """Benchmark focused on execution speed."""
    return benchmark(
        iterations=iterations,
        memory_tracking=False,
        cpu_tracking=False
    )(func)() 