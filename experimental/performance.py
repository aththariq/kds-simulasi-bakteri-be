import time
import asyncio
import cProfile
import pstats
import io
import functools
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, Callable, Optional, List, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from memory_profiler import profile as memory_profile
from pympler import tracker, muppy, summary
from cachetools import TTLCache, LRUCache
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics."""
    function_name: str
    execution_time: float
    memory_usage_before: float
    memory_usage_after: float
    memory_peak: float
    cpu_usage: float
    timestamp: datetime
    call_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class PerformanceProfiler:
    """Comprehensive performance profiling utility."""
    
    def __init__(self):
        self.metrics_storage: Dict[str, List[PerformanceMetrics]] = {}
        self.memory_tracker = tracker.SummaryTracker()
        self._lock = threading.Lock()
    
    @contextmanager
    def profile_block(self, name: str):
        """Context manager for profiling code blocks."""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        start_cpu = psutil.cpu_percent()
        
        # Memory tracking
        memory_tracker = tracker.SummaryTracker()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            end_cpu = psutil.cpu_percent()
            
            # Calculate peak memory usage
            peak_memory = max(start_memory, end_memory)
            
            metrics = PerformanceMetrics(
                function_name=name,
                execution_time=end_time - start_time,
                memory_usage_before=start_memory,
                memory_usage_after=end_memory,
                memory_peak=peak_memory,
                cpu_usage=(start_cpu + end_cpu) / 2,
                timestamp=datetime.now()
            )
            
            self._store_metrics(name, metrics)
    
    def profile_function(self, func_name: Optional[str] = None):
        """Decorator for profiling function performance."""
        def decorator(func: Callable) -> Callable:
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    with self.profile_block(name):
                        return await func(*args, **kwargs)
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    with self.profile_block(name):
                        return func(*args, **kwargs)
                return sync_wrapper
        
        return decorator
    
    def _store_metrics(self, name: str, metrics: PerformanceMetrics):
        """Store performance metrics."""
        with self._lock:
            if name not in self.metrics_storage:
                self.metrics_storage[name] = []
            self.metrics_storage[name].append(metrics)
    
    def get_metrics_summary(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if function_name:
            if function_name not in self.metrics_storage:
                return {}
            
            metrics_list = self.metrics_storage[function_name]
            return self._calculate_summary(function_name, metrics_list)
        
        summary = {}
        for name, metrics_list in self.metrics_storage.items():
            summary[name] = self._calculate_summary(name, metrics_list)
        
        return summary
    
    def _calculate_summary(self, name: str, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate summary statistics for metrics."""
        if not metrics_list:
            return {}
        
        execution_times = [m.execution_time for m in metrics_list]
        memory_usage = [m.memory_usage_after - m.memory_usage_before for m in metrics_list]
        
        return {
            "function_name": name,
            "call_count": len(metrics_list),
            "execution_time": {
                "avg": np.mean(execution_times),
                "min": np.min(execution_times),
                "max": np.max(execution_times),
                "std": np.std(execution_times)
            },
            "memory_usage": {
                "avg": np.mean(memory_usage),
                "min": np.min(memory_usage),
                "max": np.max(memory_usage),
                "std": np.std(memory_usage)
            },
            "last_execution": metrics_list[-1].timestamp.isoformat()
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        summary = self.get_metrics_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def clear_metrics(self, function_name: Optional[str] = None):
        """Clear stored metrics."""
        with self._lock:
            if function_name:
                self.metrics_storage.pop(function_name, None)
            else:
                self.metrics_storage.clear()


class SimulationOptimizer:
    """Optimization utilities for simulation algorithms."""
    
    def __init__(self):
        self.cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour TTL
        self.fitness_cache = LRUCache(maxsize=5000)
    
    @staticmethod
    def vectorized_mutation(population: np.ndarray, mutation_rate: float) -> np.ndarray:
        """Optimized vectorized mutation operation."""
        # Generate random mutations for entire population at once
        mutation_mask = np.random.random(population.shape) < mutation_rate
        mutations = np.random.random(population.shape)
        
        # Apply mutations only where mask is True
        mutated_population = np.where(mutation_mask, mutations, population)
        return mutated_population
    
    @staticmethod
    def vectorized_selection(population: np.ndarray, fitness_scores: np.ndarray, 
                           selection_pressure: float) -> np.ndarray:
        """Optimized vectorized selection operation."""
        # Calculate selection probabilities
        fitness_normalized = fitness_scores / np.sum(fitness_scores)
        
        # Apply selection pressure
        selection_probs = np.power(fitness_normalized, selection_pressure)
        selection_probs = selection_probs / np.sum(selection_probs)
        
        # Select individuals based on probabilities
        selected_indices = np.random.choice(
            len(population), 
            size=len(population), 
            p=selection_probs, 
            replace=True
        )
        
        return population[selected_indices]
    
    def cached_fitness_calculation(self, population_hash: str, 
                                 fitness_func: Callable, 
                                 population: np.ndarray) -> np.ndarray:
        """Cache fitness calculations to avoid recomputation."""
        if population_hash in self.fitness_cache:
            return self.fitness_cache[population_hash]
        
        fitness_scores = fitness_func(population)
        self.fitness_cache[population_hash] = fitness_scores
        return fitness_scores
    
    @staticmethod
    def batch_process_populations(populations: List[np.ndarray], 
                                batch_size: int = 10) -> List[np.ndarray]:
        """Process multiple populations in optimized batches."""
        results = []
        for i in range(0, len(populations), batch_size):
            batch = populations[i:i + batch_size]
            # Process batch operations here
            results.extend(batch)
        return results


class MemoryProfiler:
    """Memory usage profiling and optimization."""
    
    def __init__(self):
        self.tracker = tracker.SummaryTracker()
        self.snapshots = []
    
    def take_snapshot(self, label: str = None):
        """Take a memory snapshot."""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'label': label or f"snapshot_{len(self.snapshots)}",
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'objects': muppy.get_objects()
        }
        self.snapshots.append(snapshot)
        return snapshot
    
    def compare_snapshots(self, snapshot1_idx: int = -2, snapshot2_idx: int = -1) -> str:
        """Compare two memory snapshots."""
        if len(self.snapshots) < 2:
            return "Need at least 2 snapshots to compare"
        
        snap1 = self.snapshots[snapshot1_idx]
        snap2 = self.snapshots[snapshot2_idx]
        
        diff = snap2['memory_usage'] - snap1['memory_usage']
        return f"Memory usage change: {diff:.2f} MB from {snap1['label']} to {snap2['label']}"
    
    def get_top_memory_objects(self, limit: int = 10) -> List[str]:
        """Get top memory consuming objects."""
        all_objects = muppy.get_objects()
        sum_obj = summary.summarize(all_objects)
        return summary.format_(sum_obj[:limit])


class CacheManager:
    """Centralized cache management for performance optimization."""
    
    def __init__(self):
        self.caches = {
            'simulation_results': TTLCache(maxsize=100, ttl=1800),  # 30 min
            'fitness_scores': LRUCache(maxsize=1000),
            'population_states': TTLCache(maxsize=50, ttl=3600),  # 1 hour
            'api_responses': TTLCache(maxsize=200, ttl=300)  # 5 min
        }
    
    def get(self, cache_name: str, key: str) -> Any:
        """Get value from specified cache."""
        cache = self.caches.get(cache_name)
        if cache:
            return cache.get(key)
        return None
    
    def set(self, cache_name: str, key: str, value: Any) -> bool:
        """Set value in specified cache."""
        cache = self.caches.get(cache_name)
        if cache:
            cache[key] = value
            return True
        return False
    
    def invalidate(self, cache_name: str, key: Optional[str] = None):
        """Invalidate cache entries."""
        cache = self.caches.get(cache_name)
        if cache:
            if key:
                cache.pop(key, None)
            else:
                cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get cache statistics."""
        stats = {}
        for name, cache in self.caches.items():
            stats[name] = {
                'size': len(cache),
                'maxsize': cache.maxsize,
                'hits': getattr(cache, 'hits', 0),
                'misses': getattr(cache, 'misses', 0)
            }
        return stats


# Global instances
profiler = PerformanceProfiler()
optimizer = SimulationOptimizer()
memory_profiler = MemoryProfiler()
cache_manager = CacheManager()


# Convenience decorators
def profile_performance(func_name: Optional[str] = None):
    """Decorator for profiling function performance."""
    return profiler.profile_function(func_name)


def cached_result(cache_name: str, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            result = cache_manager.get(cache_name, cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_name, cache_key, result)
            return result
        
        return wrapper
    return decorator 