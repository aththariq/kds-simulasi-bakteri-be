"""
Simplified performance utilities without heavy dependencies.
"""

import time
import asyncio
import functools
import psutil
from datetime import datetime
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


@dataclass
class SimplePerformanceMetrics:
    """Simple performance metrics without heavy dependencies."""
    function_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    timestamp: datetime
    call_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class SimpleProfiler:
    """Simple performance profiler."""
    
    def __init__(self):
        self.metrics_storage: Dict[str, List[SimplePerformanceMetrics]] = {}
    
    @contextmanager
    def profile_block(self, name: str):
        """Context manager for profiling code blocks."""
        start_time = time.perf_counter()
        
        try:
            # Get system metrics
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            cpu_percent = psutil.cpu_percent(interval=None)
        except Exception:
            start_memory = 0
            cpu_percent = 0
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            try:
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                final_cpu = psutil.cpu_percent(interval=None)
                avg_cpu = (cpu_percent + final_cpu) / 2
            except Exception:
                end_memory = start_memory
                avg_cpu = 0
            
            metrics = SimplePerformanceMetrics(
                function_name=name,
                execution_time=execution_time,
                memory_usage_mb=end_memory,
                cpu_percent=avg_cpu,
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
    
    def _store_metrics(self, name: str, metrics: SimplePerformanceMetrics):
        """Store performance metrics."""
        if name not in self.metrics_storage:
            self.metrics_storage[name] = []
        self.metrics_storage[name].append(metrics)
        
        # Keep only last 100 metrics per function to prevent memory growth
        if len(self.metrics_storage[name]) > 100:
            self.metrics_storage[name] = self.metrics_storage[name][-100:]
    
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
    
    def _calculate_summary(self, name: str, metrics_list: List[SimplePerformanceMetrics]) -> Dict[str, Any]:
        """Calculate summary statistics for metrics."""
        if not metrics_list:
            return {}
        
        execution_times = [m.execution_time for m in metrics_list]
        memory_usage = [m.memory_usage_mb for m in metrics_list]
        cpu_usage = [m.cpu_percent for m in metrics_list]
        
        return {
            "function_name": name,
            "call_count": len(metrics_list),
            "execution_time": {
                "avg": sum(execution_times) / len(execution_times),
                "min": min(execution_times),
                "max": max(execution_times),
            },
            "memory_usage_mb": {
                "avg": sum(memory_usage) / len(memory_usage),
                "min": min(memory_usage),
                "max": max(memory_usage),
            },
            "cpu_percent": {
                "avg": sum(cpu_usage) / len(cpu_usage),
                "min": min(cpu_usage),
                "max": max(cpu_usage),
            },
            "last_execution": metrics_list[-1].timestamp.isoformat()
        }
    
    def clear_metrics(self, function_name: Optional[str] = None):
        """Clear stored metrics."""
        if function_name:
            self.metrics_storage.pop(function_name, None)
        else:
            self.metrics_storage.clear()


class SimpleCache:
    """Simple in-memory cache without external dependencies."""
    
    def __init__(self, maxsize: int = 1000):
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
        self.maxsize = maxsize
    
    def get(self, category: str, key: str) -> Optional[Any]:
        """Get value from cache."""
        cache_key = f"{category}:{key}"
        if cache_key in self.cache:
            # Move to end (most recently used)
            if cache_key in self.access_order:
                self.access_order.remove(cache_key)
            self.access_order.append(cache_key)
            return self.cache[cache_key]
        return None
    
    def set(self, category: str, key: str, value: Any) -> None:
        """Set value in cache."""
        cache_key = f"{category}:{key}"
        
        # If cache is full, remove least recently used
        if len(self.cache) >= self.maxsize and cache_key not in self.cache:
            if self.access_order:
                lru_key = self.access_order.pop(0)
                self.cache.pop(lru_key, None)
        
        self.cache[cache_key] = value
        
        # Update access order
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)
    
    def clear(self, category: Optional[str] = None) -> None:
        """Clear cache."""
        if category:
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(f"{category}:")]
            for key in keys_to_remove:
                self.cache.pop(key, None)
                if key in self.access_order:
                    self.access_order.remove(key)
        else:
            self.cache.clear()
            self.access_order.clear()


# Global instances
simple_profiler = SimpleProfiler()
simple_cache = SimpleCache()

# Convenience decorators
def profile_performance(name: Optional[str] = None):
    """Convenience decorator for performance profiling."""
    return simple_profiler.profile_function(name)

def cached_result(category: str = "default", ttl: Optional[int] = None):
    """Simple result caching decorator."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Check cache
            cached = simple_cache.get(category, key)
            if cached is not None:
                return cached
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            simple_cache.set(category, key, result)
            return result
        
        return wrapper
    return decorator 