"""
Memory Optimization Utilities for Bacterial Simulation System

This module provides system-wide memory optimization functions that can be
used across different components of the simulation to manage memory efficiently.
"""

import gc
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from contextlib import asynccontextmanager

# Import memory management components with try/catch for flexibility
try:
    from services.memory_manager import MemoryManager, MemoryMetrics
except ImportError:
    try:
        from backend.services.memory_manager import MemoryManager, MemoryMetrics
    except ImportError:
        # Fallback for testing - create minimal implementations
        class MemoryMetrics:
            def __init__(self):
                self.rss_mb = 100
                self.vms_mb = 200
                self.heap_size_mb = 50
                self.object_count = 1000
                self.timestamp = type('datetime', (), {'isoformat': lambda: '2023-01-01'})()
        
        class MemoryManager:
            def __init__(self): pass
            def get_pool(self, name): return None
            def get_memory_metrics(self): return MemoryMetrics()
            def get_pool_stats(self): return {}
            def cleanup_all(self): return {}

logger = logging.getLogger(__name__)

# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None


def get_global_memory_manager() -> MemoryManager:
    """Get or create the global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


class SimulationMemoryOptimizer:
    """
    Memory optimizer specifically designed for bacterial simulation workloads.
    
    Provides simulation-specific memory management patterns and optimization
    strategies for different phases of the simulation lifecycle.
    """
    
    def __init__(self):
        self.memory_manager = get_global_memory_manager()
        self._optimization_callbacks: List[Callable[[], Dict[str, Any]]] = []
        self._monitoring_active = False
    
    def register_optimization_callback(self, callback: Callable[[], Dict[str, Any]]) -> None:
        """
        Register a callback function that performs memory optimization.
        
        Args:
            callback: Function that returns optimization statistics
        """
        self._optimization_callbacks.append(callback)
    
    def optimize_for_simulation_phase(self, phase: str) -> Dict[str, Any]:
        """
        Perform memory optimization for specific simulation phases.
        
        Args:
            phase: Simulation phase ('initialization', 'evolution', 'analysis', 'cleanup')
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting memory optimization for phase: {phase}")
        
        optimization_results = {"phase": phase, "callbacks_executed": 0}
        
        # Execute registered optimization callbacks
        for i, callback in enumerate(self._optimization_callbacks):
            try:
                result = callback()
                optimization_results[f"callback_{i}"] = result
                optimization_results["callbacks_executed"] += 1
            except Exception as e:
                logger.error(f"Memory optimization callback {i} failed: {e}")
                optimization_results[f"callback_{i}_error"] = str(e)
        
        # Phase-specific optimizations
        if phase == "initialization":
            # Pre-allocate object pools for efficiency
            self._optimize_for_initialization()
        elif phase == "evolution":
            # Optimize for repeated object creation/destruction cycles
            self._optimize_for_evolution()
        elif phase == "analysis":
            # Free unused resources during analysis
            self._optimize_for_analysis()
        elif phase == "cleanup":
            # Comprehensive cleanup at simulation end
            self._optimize_for_cleanup()
        
        # Force garbage collection
        collected = gc.collect()
        optimization_results["gc_collected"] = collected
        
        # Get current memory metrics
        metrics = self.memory_manager.get_memory_metrics()
        optimization_results["memory_after_mb"] = metrics.rss_mb
        
        logger.info(f"Memory optimization complete for {phase}: {optimization_results}")
        return optimization_results
    
    def _optimize_for_initialization(self) -> None:
        """Optimize memory for simulation initialization phase."""
        # Pre-warm object pools
        logger.debug("Pre-warming object pools for initialization")
        
        # Get coordinate pool and pre-allocate some objects
        coord_pool = self.memory_manager.get_pool("coordinate_pool")
        if coord_pool:
            # Pre-allocate coordinates for initial population
            for _ in range(100):
                coord = coord_pool.acquire()
                coord_pool.release(coord)
        
        # Get grid cell pool and pre-allocate some objects
        cell_pool = self.memory_manager.get_pool("grid_cell_pool")
        if cell_pool:
            # Pre-allocate grid cells
            for _ in range(200):
                cell = cell_pool.acquire()
                cell_pool.release(cell)
    
    def _optimize_for_evolution(self) -> None:
        """Optimize memory during evolution simulation phase."""
        # Clear caches periodically during evolution
        logger.debug("Optimizing memory for evolution phase")
        
        # Trigger intermediate garbage collection
        gc.collect()
        
        # Get memory pressure state and respond accordingly
        metrics = self.memory_manager.get_memory_metrics()
        if metrics.rss_mb > 1024:  # If using more than 1GB
            logger.warning(f"High memory usage during evolution: {metrics.rss_mb:.1f}MB")
            self.memory_manager.cleanup_all()
    
    def _optimize_for_analysis(self) -> None:
        """Optimize memory during analysis phase."""
        logger.debug("Optimizing memory for analysis phase")
        
        # Analysis phase typically doesn't need large object pools
        # Clear unnecessary pools to free memory
        self.memory_manager.cleanup_all()
    
    def _optimize_for_cleanup(self) -> None:
        """Comprehensive cleanup at simulation end."""
        logger.debug("Performing comprehensive memory cleanup")
        
        # Cleanup all managed resources
        cleanup_results = self.memory_manager.cleanup_all()
        
        # Force multiple garbage collection passes
        for _ in range(3):
            collected = gc.collect()
            logger.debug(f"Garbage collection pass collected {collected} objects")
    
    def get_memory_usage_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive memory usage report.
        
        Returns:
            Dictionary with memory usage information
        """
        metrics = self.memory_manager.get_memory_metrics()
        pool_stats = self.memory_manager.get_pool_stats()
        
        report = {
            "timestamp": metrics.timestamp.isoformat(),
            "memory_usage": {
                "rss_mb": metrics.rss_mb,
                "vms_mb": metrics.vms_mb,
                "heap_size_mb": metrics.heap_size_mb,
                "object_count": metrics.object_count
            },
            "pools": {}
        }
        
        # Add pool statistics
        for pool_name, stats in pool_stats.items():
            report["pools"][pool_name] = {
                "hit_rate": stats.hit_rate,
                "current_size": stats.current_pool_size,
                "total_created": stats.total_created,
                "total_reused": stats.total_reused
            }
        
        return report
    
    @asynccontextmanager
    async def memory_monitored_operation(self, operation_name: str):
        """
        Context manager for monitoring memory usage during an operation.
        
        Args:
            operation_name: Name of the operation being monitored
        """
        logger.info(f"Starting memory-monitored operation: {operation_name}")
        
        # Get initial memory state
        initial_metrics = self.memory_manager.get_memory_metrics()
        
        try:
            yield
        finally:
            # Get final memory state
            final_metrics = self.memory_manager.get_memory_metrics()
            
            # Calculate memory delta
            memory_delta = final_metrics.rss_mb - initial_metrics.rss_mb
            
            logger.info(f"Memory-monitored operation '{operation_name}' complete. "
                       f"Memory delta: {memory_delta:+.1f}MB "
                       f"(from {initial_metrics.rss_mb:.1f}MB to {final_metrics.rss_mb:.1f}MB)")
            
            # Trigger optimization if memory increased significantly
            if memory_delta > 100:  # More than 100MB increase
                logger.warning(f"Large memory increase detected ({memory_delta:.1f}MB), "
                              "triggering optimization")
                self.optimize_for_simulation_phase("evolution")


# Global optimizer instance
_global_optimizer: Optional[SimulationMemoryOptimizer] = None


def get_simulation_memory_optimizer() -> SimulationMemoryOptimizer:
    """Get or create the global simulation memory optimizer."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = SimulationMemoryOptimizer()
    return _global_optimizer


def optimize_simulation_memory(phase: str = "evolution") -> Dict[str, Any]:
    """
    Convenience function for quick memory optimization.
    
    Args:
        phase: Simulation phase to optimize for
        
    Returns:
        Optimization results
    """
    optimizer = get_simulation_memory_optimizer()
    return optimizer.optimize_for_simulation_phase(phase)


def get_memory_report() -> Dict[str, Any]:
    """
    Convenience function to get current memory usage report.
    
    Returns:
        Memory usage report
    """
    optimizer = get_simulation_memory_optimizer()
    return optimizer.get_memory_usage_report()


async def memory_monitored_simulation_step(operation_name: str, operation_func: Callable):
    """
    Execute a simulation step with memory monitoring.
    
    Args:
        operation_name: Name of the operation
        operation_func: Function to execute
        
    Returns:
        Result of the operation function
    """
    optimizer = get_simulation_memory_optimizer()
    
    async with optimizer.memory_monitored_operation(operation_name):
        if asyncio.iscoroutinefunction(operation_func):
            return await operation_func()
        else:
            return operation_func() 