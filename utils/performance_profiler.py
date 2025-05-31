"""
Comprehensive Performance Profiling System
Extends the existing SimpleProfiler with advanced monitoring capabilities.
"""

import asyncio
import time
import json
import statistics
import tracemalloc
import gc
import functools
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set, AsyncGenerator
from dataclasses import dataclass, asdict, field
from contextlib import asynccontextmanager
import logging
import psutil
from pathlib import Path
import weakref
from collections import defaultdict, deque
import threading
import atexit

logger = logging.getLogger(__name__)


@dataclass
class AdvancedMetrics:
    """Extended performance metrics with memory and system details."""
    function_name: str
    execution_time: float
    memory_current: float
    memory_peak: float
    memory_allocated: float
    memory_freed: float
    cpu_percent: float
    gc_collections: Dict[int, int]
    thread_id: int
    call_stack_depth: int
    timestamp: datetime
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    function_name: str
    avg_execution_time: float
    p95_execution_time: float
    avg_memory_usage: float
    peak_memory_usage: float
    sample_count: int
    created_at: datetime
    confidence_level: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert baseline to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class PerformanceAlert:
    """Performance regression alert."""
    function_name: str
    metric_type: str
    baseline_value: float
    current_value: float
    deviation_percent: float
    threshold_percent: float
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class RealTimePerformanceCollector:
    """Collects performance metrics in real-time with minimal overhead."""
    
    def __init__(self, max_buffer_size: int = 1000):
        self.metrics_buffer: deque = deque(maxlen=max_buffer_size)
        self.subscribers: Set[Callable] = set()
        self._lock = threading.Lock()
        self._running = False
        self._collection_thread: Optional[threading.Thread] = None
        
    def start_collection(self, interval: float = 1.0):
        """Start real-time metrics collection."""
        if self._running:
            return
        
        self._running = True
        self._collection_thread = threading.Thread(
            target=self._collect_system_metrics,
            args=(interval,),
            daemon=True
        )
        self._collection_thread.start()
        logger.info("Real-time performance collection started")
    
    def stop_collection(self):
        """Stop real-time metrics collection."""
        self._running = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        logger.info("Real-time performance collection stopped")
    
    def _collect_system_metrics(self, interval: float):
        """Collect system metrics in background thread."""
        while self._running:
            try:
                # Collect system metrics
                process = psutil.Process()
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                
                system_metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'memory_rss': memory_info.rss / 1024 / 1024,  # MB
                    'memory_vms': memory_info.vms / 1024 / 1024,  # MB
                    'cpu_percent': cpu_percent,
                    'thread_count': process.num_threads(),
                    'fd_count': process.num_fds() if hasattr(process, 'num_fds') else 0,
                    'gc_stats': {str(i): gc.get_count()[i] for i in range(3)},
                }
                
                with self._lock:
                    self.metrics_buffer.append(system_metrics)
                
                # Notify subscribers
                for subscriber in self.subscribers.copy():
                    try:
                        subscriber(system_metrics)
                    except Exception as e:
                        logger.warning(f"Subscriber notification failed: {e}")
                        self.subscribers.discard(subscriber)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"System metrics collection error: {e}")
                time.sleep(interval)
    
    def subscribe(self, callback: Callable):
        """Subscribe to real-time metrics updates."""
        self.subscribers.add(callback)
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from real-time metrics updates."""
        self.subscribers.discard(callback)
    
    def get_recent_metrics(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent system metrics."""
        with self._lock:
            return list(self.metrics_buffer)[-count:]


class AdvancedProfiler:
    """Advanced performance profiler with regression detection and real-time monitoring."""
    
    def __init__(self, baseline_dir: str = "performance_baselines"):
        self.metrics_storage: Dict[str, List[AdvancedMetrics]] = defaultdict(list)
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.alerts: List[PerformanceAlert] = []
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(exist_ok=True)
        
        # Real-time collector
        self.realtime_collector = RealTimePerformanceCollector()
        
        # WebSocket subscribers for real-time updates
        self.websocket_subscribers: Set[Callable] = set()
        
        # Configuration
        self.regression_threshold = 20.0  # 20% degradation threshold
        self.auto_baseline_interval = timedelta(days=1)
        self.max_metrics_per_function = 1000
        
        # Start memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        # Load existing baselines
        self._load_baselines()
        
        # Register cleanup
        atexit.register(self._cleanup)
    
    @asynccontextmanager
    async def profile_async(self, name: str, correlation_id: Optional[str] = None):
        """Async context manager for profiling with advanced metrics."""
        # Start profiling
        start_time = time.perf_counter()
        gc_before = {i: gc.get_count()[i] for i in range(3)}
        
        try:
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            cpu_before = process.cpu_percent()
        except Exception:
            memory_before = 0
            cpu_before = 0
        
        # Memory tracing
        if tracemalloc.is_tracing():
            tracemalloc.clear_traces()
            snapshot_before = tracemalloc.take_snapshot()
        else:
            snapshot_before = None
        
        try:
            yield
        finally:
            # End profiling
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            gc_after = {i: gc.get_count()[i] for i in range(3)}
            gc_collections = {i: gc_after[i] - gc_before[i] for i in range(3)}
            
            try:
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                cpu_after = psutil.Process().cpu_percent()
                avg_cpu = (cpu_before + cpu_after) / 2
            except Exception:
                memory_after = memory_before
                avg_cpu = 0
            
            # Memory allocation tracking
            memory_allocated = memory_freed = 0
            memory_peak = memory_after
            
            if snapshot_before and tracemalloc.is_tracing():
                try:
                    snapshot_after = tracemalloc.take_snapshot()
                    
                    # Calculate memory differences
                    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
                    total_allocated = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
                    total_freed = sum(-stat.size_diff for stat in top_stats if stat.size_diff < 0)
                    
                    memory_allocated = total_allocated / 1024 / 1024  # MB
                    memory_freed = total_freed / 1024 / 1024  # MB
                    
                    # Peak memory during execution
                    peak_stats = snapshot_after.statistics('traceback')
                    if peak_stats:
                        memory_peak = max(memory_after, peak_stats[0].size / 1024 / 1024)
                
                except Exception as e:
                    logger.warning(f"Memory tracking error: {e}")
            
            # Create metrics
            metrics = AdvancedMetrics(
                function_name=name,
                execution_time=execution_time,
                memory_current=memory_after,
                memory_peak=memory_peak,
                memory_allocated=memory_allocated,
                memory_freed=memory_freed,
                cpu_percent=avg_cpu,
                gc_collections=gc_collections,
                thread_id=threading.get_ident(),
                call_stack_depth=len(traceback.extract_stack()),
                timestamp=datetime.now(),
                correlation_id=correlation_id
            )
            
            # Store metrics
            await self._store_metrics_async(name, metrics)
    
    def profile_sync(self, name: str, correlation_id: Optional[str] = None):
        """Synchronous context manager for profiling."""
        return self._SyncProfileContext(self, name, correlation_id)
    
    class _SyncProfileContext:
        """Synchronous profiling context."""
        
        def __init__(self, profiler: 'AdvancedProfiler', name: str, correlation_id: Optional[str]):
            self.profiler = profiler
            self.name = name
            self.correlation_id = correlation_id
        
        def __enter__(self):
            self.start_time = time.perf_counter()
            self.gc_before = {i: gc.get_count()[i] for i in range(3)}
            
            try:
                process = psutil.Process()
                self.memory_before = process.memory_info().rss / 1024 / 1024
                self.cpu_before = process.cpu_percent()
            except Exception:
                self.memory_before = 0
                self.cpu_before = 0
            
            if tracemalloc.is_tracing():
                tracemalloc.clear_traces()
                self.snapshot_before = tracemalloc.take_snapshot()
            else:
                self.snapshot_before = None
            
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            end_time = time.perf_counter()
            execution_time = end_time - self.start_time
            
            gc_after = {i: gc.get_count()[i] for i in range(3)}
            gc_collections = {i: gc_after[i] - self.gc_before[i] for i in range(3)}
            
            try:
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                cpu_after = psutil.Process().cpu_percent()
                avg_cpu = (self.cpu_before + cpu_after) / 2
            except Exception:
                memory_after = self.memory_before
                avg_cpu = 0
            
            # Memory tracking
            memory_allocated = memory_freed = 0
            memory_peak = memory_after
            
            if self.snapshot_before and tracemalloc.is_tracing():
                try:
                    snapshot_after = tracemalloc.take_snapshot()
                    top_stats = snapshot_after.compare_to(self.snapshot_before, 'lineno')
                    
                    total_allocated = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
                    total_freed = sum(-stat.size_diff for stat in top_stats if stat.size_diff < 0)
                    
                    memory_allocated = total_allocated / 1024 / 1024
                    memory_freed = total_freed / 1024 / 1024
                    
                    peak_stats = snapshot_after.statistics('traceback')
                    if peak_stats:
                        memory_peak = max(memory_after, peak_stats[0].size / 1024 / 1024)
                
                except Exception as e:
                    logger.warning(f"Memory tracking error: {e}")
            
            metrics = AdvancedMetrics(
                function_name=self.name,
                execution_time=execution_time,
                memory_current=memory_after,
                memory_peak=memory_peak,
                memory_allocated=memory_allocated,
                memory_freed=memory_freed,
                cpu_percent=avg_cpu,
                gc_collections=gc_collections,
                thread_id=threading.get_ident(),
                call_stack_depth=len(traceback.extract_stack()),
                timestamp=datetime.now(),
                correlation_id=self.correlation_id
            )
            
            # Store metrics synchronously
            asyncio.create_task(self.profiler._store_metrics_async(self.name, metrics))
    
    async def _store_metrics_async(self, name: str, metrics: AdvancedMetrics):
        """Store metrics and check for regressions asynchronously."""
        # Store metrics
        self.metrics_storage[name].append(metrics)
        
        # Limit storage size
        if len(self.metrics_storage[name]) > self.max_metrics_per_function:
            self.metrics_storage[name] = self.metrics_storage[name][-self.max_metrics_per_function:]
        
        # Check for regressions
        await self._check_regression(name, metrics)
        
        # Notify WebSocket subscribers
        await self._notify_websocket_subscribers({
            'type': 'performance_metrics',
            'data': metrics.to_dict()
        })
    
    async def _check_regression(self, name: str, metrics: AdvancedMetrics):
        """Check for performance regressions."""
        if name not in self.baselines:
            return
        
        baseline = self.baselines[name]
        alerts = []
        
        # Check execution time regression
        if metrics.execution_time > baseline.avg_execution_time * (1 + self.regression_threshold / 100):
            deviation = ((metrics.execution_time - baseline.avg_execution_time) / baseline.avg_execution_time) * 100
            severity = self._calculate_severity(deviation)
            
            alert = PerformanceAlert(
                function_name=name,
                metric_type='execution_time',
                baseline_value=baseline.avg_execution_time,
                current_value=metrics.execution_time,
                deviation_percent=deviation,
                threshold_percent=self.regression_threshold,
                timestamp=datetime.now(),
                severity=severity
            )
            alerts.append(alert)
        
        # Check memory regression
        if metrics.memory_peak > baseline.peak_memory_usage * (1 + self.regression_threshold / 100):
            deviation = ((metrics.memory_peak - baseline.peak_memory_usage) / baseline.peak_memory_usage) * 100
            severity = self._calculate_severity(deviation)
            
            alert = PerformanceAlert(
                function_name=name,
                metric_type='memory_usage',
                baseline_value=baseline.peak_memory_usage,
                current_value=metrics.memory_peak,
                deviation_percent=deviation,
                threshold_percent=self.regression_threshold,
                timestamp=datetime.now(),
                severity=severity
            )
            alerts.append(alert)
        
        # Store and notify alerts
        for alert in alerts:
            self.alerts.append(alert)
            logger.warning(f"Performance regression detected: {alert.function_name} - {alert.metric_type}")
            
            await self._notify_websocket_subscribers({
                'type': 'performance_alert',
                'data': alert.to_dict()
            })
    
    def _calculate_severity(self, deviation_percent: float) -> str:
        """Calculate alert severity based on deviation."""
        if deviation_percent >= 100:
            return 'critical'
        elif deviation_percent >= 50:
            return 'high'
        elif deviation_percent >= 25:
            return 'medium'
        else:
            return 'low'
    
    async def _notify_websocket_subscribers(self, message: Dict[str, Any]):
        """Notify WebSocket subscribers of performance updates."""
        for subscriber in self.websocket_subscribers.copy():
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(message)
                else:
                    subscriber(message)
            except Exception as e:
                logger.warning(f"WebSocket subscriber notification failed: {e}")
                self.websocket_subscribers.discard(subscriber)
    
    def subscribe_websocket(self, callback: Callable):
        """Subscribe to WebSocket performance updates."""
        self.websocket_subscribers.add(callback)
    
    def unsubscribe_websocket(self, callback: Callable):
        """Unsubscribe from WebSocket performance updates."""
        self.websocket_subscribers.discard(callback)
    
    def create_baseline(self, function_name: str, force: bool = False) -> Optional[PerformanceBaseline]:
        """Create performance baseline from recent metrics."""
        if function_name not in self.metrics_storage:
            logger.warning(f"No metrics found for function: {function_name}")
            return None
        
        metrics = self.metrics_storage[function_name]
        
        # Need at least 10 samples for reliable baseline
        if len(metrics) < 10:
            logger.warning(f"Insufficient metrics for baseline: {function_name} ({len(metrics)} samples)")
            return None
        
        # Use recent metrics (last 100 or all if fewer)
        recent_metrics = metrics[-100:]
        
        execution_times = [m.execution_time for m in recent_metrics]
        memory_usage = [m.memory_current for m in recent_metrics]
        peak_memory = [m.memory_peak for m in recent_metrics]
        
        baseline = PerformanceBaseline(
            function_name=function_name,
            avg_execution_time=statistics.mean(execution_times),
            p95_execution_time=statistics.quantiles(execution_times, n=20)[18],  # 95th percentile
            avg_memory_usage=statistics.mean(memory_usage),
            peak_memory_usage=max(peak_memory),
            sample_count=len(recent_metrics),
            created_at=datetime.now()
        )
        
        self.baselines[function_name] = baseline
        self._save_baseline(baseline)
        
        logger.info(f"Created baseline for {function_name}: "
                   f"avg_time={baseline.avg_execution_time:.4f}s, "
                   f"p95_time={baseline.p95_execution_time:.4f}s")
        
        return baseline
    
    def _save_baseline(self, baseline: PerformanceBaseline):
        """Save baseline to disk."""
        try:
            filename = f"{baseline.function_name.replace('/', '_').replace('.', '_')}.json"
            filepath = self.baseline_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(baseline.to_dict(), f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save baseline: {e}")
    
    def _load_baselines(self):
        """Load baselines from disk."""
        try:
            for filepath in self.baseline_dir.glob("*.json"):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                baseline = PerformanceBaseline(
                    function_name=data['function_name'],
                    avg_execution_time=data['avg_execution_time'],
                    p95_execution_time=data['p95_execution_time'],
                    avg_memory_usage=data['avg_memory_usage'],
                    peak_memory_usage=data['peak_memory_usage'],
                    sample_count=data['sample_count'],
                    created_at=datetime.fromisoformat(data['created_at']),
                    confidence_level=data.get('confidence_level', 0.95)
                )
                
                self.baselines[baseline.function_name] = baseline
            
            logger.info(f"Loaded {len(self.baselines)} performance baselines")
        
        except Exception as e:
            logger.error(f"Failed to load baselines: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'functions_monitored': len(self.metrics_storage),
            'total_metrics_collected': sum(len(metrics) for metrics in self.metrics_storage.values()),
            'baselines_established': len(self.baselines),
            'active_alerts': len([a for a in self.alerts if a.timestamp > datetime.now() - timedelta(hours=24)]),
            'system_metrics': self.realtime_collector.get_recent_metrics(1),
            'top_slowest_functions': self._get_slowest_functions(),
            'memory_intensive_functions': self._get_memory_intensive_functions(),
            'recent_alerts': [a.to_dict() for a in self.alerts[-10:]]
        }
        
        return summary
    
    def _get_slowest_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest functions by average execution time."""
        function_stats = []
        
        for name, metrics in self.metrics_storage.items():
            if len(metrics) >= 5:  # Need at least 5 samples
                recent_metrics = metrics[-50:]  # Last 50 calls
                avg_time = statistics.mean(m.execution_time for m in recent_metrics)
                function_stats.append({
                    'function_name': name,
                    'avg_execution_time': avg_time,
                    'call_count': len(recent_metrics),
                    'last_called': max(m.timestamp for m in recent_metrics).isoformat()
                })
        
        return sorted(function_stats, key=lambda x: x['avg_execution_time'], reverse=True)[:limit]
    
    def _get_memory_intensive_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most memory-intensive functions."""
        function_stats = []
        
        for name, metrics in self.metrics_storage.items():
            if len(metrics) >= 5:
                recent_metrics = metrics[-50:]
                avg_memory = statistics.mean(m.memory_peak for m in recent_metrics)
                function_stats.append({
                    'function_name': name,
                    'avg_memory_peak': avg_memory,
                    'call_count': len(recent_metrics),
                    'last_called': max(m.timestamp for m in recent_metrics).isoformat()
                })
        
        return sorted(function_stats, key=lambda x: x['avg_memory_peak'], reverse=True)[:limit]
    
    def start_realtime_monitoring(self):
        """Start real-time system monitoring."""
        self.realtime_collector.start_collection()
    
    def stop_realtime_monitoring(self):
        """Stop real-time system monitoring."""
        self.realtime_collector.stop_collection()
    
    def _cleanup(self):
        """Cleanup resources."""
        self.stop_realtime_monitoring()
        if tracemalloc.is_tracing():
            tracemalloc.stop()


# Global profiler instance
_global_profiler: Optional[AdvancedProfiler] = None


def get_profiler() -> AdvancedProfiler:
    """Get global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = AdvancedProfiler()
        _global_profiler.start_realtime_monitoring()
    return _global_profiler


def profile_async(name: str, correlation_id: Optional[str] = None):
    """Decorator for async function profiling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            profiler = get_profiler()
            async with profiler.profile_async(name, correlation_id):
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def profile_sync(name: str, correlation_id: Optional[str] = None):
    """Decorator for sync function profiling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            with profiler.profile_sync(name, correlation_id):
                return func(*args, **kwargs)
        return wrapper
    return decorator 