"""
Comprehensive Memory Management Service for Bacterial Simulation System
"""

import gc
import weakref
import threading
import asyncio
import tracemalloc
import psutil
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Set, List, Callable, TypeVar, Generic, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from contextlib import contextmanager, asynccontextmanager
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    heap_size_mb: float  # Python heap size in MB
    object_count: int  # Number of tracked objects
    pool_usage: Dict[str, int] = field(default_factory=dict)  # Pool usage statistics
    gc_stats: Dict[int, int] = field(default_factory=dict)  # Garbage collection stats
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PoolStats:
    """Object pool statistics."""
    total_created: int = 0
    total_reused: int = 0
    current_pool_size: int = 0
    max_pool_size: int = 0
    hits: int = 0
    misses: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class Disposable(ABC):
    """Interface for objects that need explicit disposal."""
    
    @abstractmethod
    def dispose(self) -> None:
        """Dispose of resources held by this object."""
        pass
    
    @property
    @abstractmethod
    def is_disposed(self) -> bool:
        """Check if the object has been disposed."""
        pass


class ObjectPool(Generic[T]):
    """Generic object pool for reducing allocation overhead."""
    
    def __init__(
        self, 
        factory: Callable[[], T], 
        reset_func: Optional[Callable[[T], None]] = None,
        max_size: int = 1000,
        name: str = "unnamed_pool"
    ):
        self.factory = factory
        self.reset_func = reset_func
        self.max_size = max_size
        self.name = name
        
        self._pool: deque[T] = deque()
        self._active_objects: Set[T] = set()
        self._lock = threading.Lock()
        self._stats = PoolStats(max_pool_size=max_size)
    
    def acquire(self) -> T:
        """Get an object from the pool or create a new one."""
        with self._lock:
            if self._pool:
                obj = self._pool.popleft()
                self._stats.hits += 1
                self._stats.total_reused += 1
            else:
                obj = self.factory()
                self._stats.misses += 1
                self._stats.total_created += 1
            
            self._active_objects.add(obj)
            self._stats.current_pool_size = len(self._pool)
            return obj
    
    def release(self, obj: T) -> None:
        """Return an object to the pool."""
        with self._lock:
            if obj in self._active_objects:
                self._active_objects.remove(obj)
                
                if len(self._pool) < self.max_size:
                    if self.reset_func:
                        self.reset_func(obj)
                    self._pool.append(obj)
                
                self._stats.current_pool_size = len(self._pool)
    
    def clear(self) -> None:
        """Clear all objects from the pool."""
        with self._lock:
            self._pool.clear()
            self._active_objects.clear()
            self._stats.current_pool_size = 0
    
    def get_stats(self) -> PoolStats:
        """Get pool statistics."""
        return self._stats


class WeakReferenceTracker:
    """Track objects using weak references to detect memory leaks."""
    
    def __init__(self):
        self._objects: Dict[str, Set[weakref.ref]] = defaultdict(set)
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def track(self, obj: Any, category: str = "default") -> None:
        """Track an object in the specified category."""
        def cleanup_callback(ref):
            with self._lock:
                self._objects[category].discard(ref)
        
        with self._lock:
            ref = weakref.ref(obj, cleanup_callback)
            self._objects[category].add(ref)
    
    def get_count(self, category: str = "default") -> int:
        """Get the count of live objects in a category."""
        with self._lock:
            # Clean up dead references
            dead_refs = {ref for ref in self._objects[category] if ref() is None}
            self._objects[category] -= dead_refs
            return len(self._objects[category])
    
    def get_all_counts(self) -> Dict[str, int]:
        """Get counts for all categories."""
        return {category: self.get_count(category) for category in self._objects}


class ResourceManager:
    """Manages lifecycle of disposable resources."""
    
    def __init__(self):
        self._resources: Set[Disposable] = set()
        self._resource_groups: Dict[str, Set[Disposable]] = defaultdict(set)
        self._lock = threading.Lock()
    
    def register(self, resource: Disposable, group: str = "default") -> None:
        """Register a disposable resource."""
        with self._lock:
            self._resources.add(resource)
            self._resource_groups[group].add(resource)
    
    def unregister(self, resource: Disposable) -> None:
        """Unregister a resource."""
        with self._lock:
            self._resources.discard(resource)
            for group in self._resource_groups.values():
                group.discard(resource)
    
    def dispose_group(self, group: str) -> int:
        """Dispose all resources in a group."""
        disposed_count = 0
        with self._lock:
            resources_to_dispose = list(self._resource_groups[group])
        
        for resource in resources_to_dispose:
            if not resource.is_disposed:
                try:
                    resource.dispose()
                    disposed_count += 1
                except Exception as e:
                    logger.error(f"Error disposing resource: {e}")
            
            self.unregister(resource)
        
        return disposed_count
    
    def dispose_all(self) -> int:
        """Dispose all registered resources."""
        disposed_count = 0
        with self._lock:
            resources_to_dispose = list(self._resources)
        
        for resource in resources_to_dispose:
            if not resource.is_disposed:
                try:
                    resource.dispose()
                    disposed_count += 1
                except Exception as e:
                    logger.error(f"Error disposing resource: {e}")
        
        self._resources.clear()
        self._resource_groups.clear()
        
        return disposed_count


class MemoryPressureMonitor:
    """Monitor memory pressure and trigger cleanup when needed."""
    
    def __init__(self, warning_threshold_mb: float = 1024, critical_threshold_mb: float = 2048):
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self._callbacks: Dict[str, List[Callable[[MemoryMetrics], None]]] = {
            'warning': [],
            'critical': [],
            'normal': []
        }
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_state = "normal"
    
    def register_callback(self, level: str, callback: Callable[[MemoryMetrics], None]) -> None:
        """Register a callback for memory pressure events."""
        if level in self._callbacks:
            self._callbacks[level].append(callback)
    
    def start_monitoring(self, interval: float = 5.0) -> None:
        """Start memory pressure monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
    
    def stop_monitoring(self) -> None:
        """Stop memory pressure monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
    
    async def _monitor_loop(self, interval: float) -> None:
        """Monitor memory usage in a loop."""
        while self._monitoring:
            try:
                metrics = self._get_current_metrics()
                current_state = self._assess_memory_state(metrics)
                
                if current_state != self._last_state:
                    logger.info(f"Memory state changed: {self._last_state} -> {current_state}")
                    await self._trigger_callbacks(current_state, metrics)
                    self._last_state = current_state
                
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(interval)
    
    def _get_current_metrics(self) -> MemoryMetrics:
        """Get current memory metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        heap_size = 0
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            heap_size = current / 1024 / 1024
        
        return MemoryMetrics(
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            heap_size_mb=heap_size,
            object_count=len(gc.get_objects()),
            gc_stats={i: gc.get_count()[i] for i in range(3)}
        )
    
    def _assess_memory_state(self, metrics: MemoryMetrics) -> str:
        """Assess current memory state."""
        if metrics.rss_mb >= self.critical_threshold_mb:
            return "critical"
        elif metrics.rss_mb >= self.warning_threshold_mb:
            return "warning"
        else:
            return "normal"
    
    async def _trigger_callbacks(self, state: str, metrics: MemoryMetrics) -> None:
        """Trigger callbacks for the current state."""
        for callback in self._callbacks[state]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metrics)
                else:
                    callback(metrics)
            except Exception as e:
                logger.error(f"Error in memory pressure callback: {e}")


class MemoryManager:
    """Central memory management service."""
    
    def __init__(self):
        self.pools: Dict[str, ObjectPool] = {}
        self.tracker = WeakReferenceTracker()
        self.resource_manager = ResourceManager()
        self.pressure_monitor = MemoryPressureMonitor()
        
        # Built-in cleanup strategies
        self._cleanup_strategies: List[Callable[[], int]] = []
        
        # Initialize memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        # Register built-in cleanup callbacks
        self.pressure_monitor.register_callback('warning', self._on_memory_warning)
        self.pressure_monitor.register_callback('critical', self._on_memory_critical)
    
    def create_pool(
        self, 
        name: str, 
        factory: Callable[[], T], 
        reset_func: Optional[Callable[[T], None]] = None,
        max_size: int = 1000
    ) -> ObjectPool[T]:
        """Create a new object pool."""
        pool = ObjectPool(factory, reset_func, max_size, name)
        self.pools[name] = pool
        logger.info(f"Created object pool '{name}' with max size {max_size}")
        return pool
    
    def get_pool(self, name: str) -> Optional[ObjectPool]:
        """Get an existing object pool."""
        return self.pools.get(name)
    
    def register_cleanup_strategy(self, strategy: Callable[[], int]) -> None:
        """Register a cleanup strategy that returns the number of objects cleaned."""
        self._cleanup_strategies.append(strategy)
    
    @contextmanager
    def managed_resource(self, resource: Disposable, group: str = "default"):
        """Context manager for automatic resource disposal."""
        self.resource_manager.register(resource, group)
        try:
            yield resource
        finally:
            if not resource.is_disposed:
                resource.dispose()
            self.resource_manager.unregister(resource)
    
    async def _on_memory_warning(self, metrics: MemoryMetrics) -> None:
        """Handle memory warning state."""
        logger.warning(f"Memory warning: {metrics.rss_mb:.1f} MB RSS")
        
        # Trigger gentle cleanup
        cleaned = 0
        for strategy in self._cleanup_strategies[:2]:  # Only run first 2 strategies
            try:
                cleaned += strategy()
            except Exception as e:
                logger.error(f"Error in cleanup strategy: {e}")
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} objects due to memory warning")
    
    async def _on_memory_critical(self, metrics: MemoryMetrics) -> None:
        """Handle critical memory state."""
        logger.error(f"Critical memory usage: {metrics.rss_mb:.1f} MB RSS")
        
        # Aggressive cleanup
        cleaned = 0
        
        # Run all cleanup strategies
        for strategy in self._cleanup_strategies:
            try:
                cleaned += strategy()
            except Exception as e:
                logger.error(f"Error in cleanup strategy: {e}")
        
        # Clear all object pools
        for pool in self.pools.values():
            pool.clear()
        
        # Force garbage collection
        collected = gc.collect()
        
        logger.warning(f"Emergency cleanup: {cleaned} objects cleaned, {collected} objects collected by GC")
    
    def get_memory_metrics(self) -> MemoryMetrics:
        """Get current memory metrics."""
        return self.pressure_monitor._get_current_metrics()
    
    def get_pool_stats(self) -> Dict[str, PoolStats]:
        """Get statistics for all pools."""
        return {name: pool.get_stats() for name, pool in self.pools.items()}
    
    def start_monitoring(self, interval: float = 5.0) -> None:
        """Start memory pressure monitoring."""
        self.pressure_monitor.start_monitoring(interval)
        logger.info("Memory pressure monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop memory pressure monitoring."""
        self.pressure_monitor.stop_monitoring()
        logger.info("Memory pressure monitoring stopped")
    
    def cleanup_all(self) -> Dict[str, int]:
        """Perform comprehensive cleanup."""
        results = {}
        
        # Run cleanup strategies
        strategy_cleaned = 0
        for i, strategy in enumerate(self._cleanup_strategies):
            try:
                cleaned = strategy()
                strategy_cleaned += cleaned
            except Exception as e:
                logger.error(f"Error in cleanup strategy {i}: {e}")
        
        results['strategy_cleaned'] = strategy_cleaned
        
        # Clear pools
        pool_cleared = 0
        for pool in self.pools.values():
            pool_cleared += pool._stats.current_pool_size
            pool.clear()
        
        results['pool_cleared'] = pool_cleared
        
        # Dispose resources
        results['resources_disposed'] = self.resource_manager.dispose_all()
        
        # Force garbage collection
        results['gc_collected'] = gc.collect()
        
        logger.info(f"Comprehensive cleanup completed: {results}")
        return results


# Global memory manager instance
memory_manager = MemoryManager()


# Convenience decorators and context managers
def pooled_object(pool_name: str):
    """Decorator to automatically use object pooling."""
    def decorator(cls):
        original_new = cls.__new__
        
        def pooled_new(cls, *args, **kwargs):
            pool = memory_manager.get_pool(pool_name)
            if pool:
                return pool.acquire()
            else:
                # Fallback to normal allocation
                return original_new(cls) if original_new is not object.__new__ else object.__new__(cls)
        
        cls.__new__ = pooled_new
        return cls
    
    return decorator


@contextmanager
def memory_tracked(category: str = "default"):
    """Context manager for tracking object creation."""
    created_objects = []
    
    class TrackedMeta(type):
        def __call__(cls, *args, **kwargs):
            obj = super().__call__(*args, **kwargs)
            memory_manager.tracker.track(obj, category)
            created_objects.append(obj)
            return obj
    
    try:
        yield created_objects
    finally:
        logger.debug(f"Created {len(created_objects)} objects in category '{category}'")


@asynccontextmanager
async def memory_pressure_context(warning_mb: float = 512, critical_mb: float = 1024):
    """Context manager for temporary memory pressure monitoring."""
    monitor = MemoryPressureMonitor(warning_mb, critical_mb)
    monitor.start_monitoring()
    try:
        yield monitor
    finally:
        monitor.stop_monitoring() 