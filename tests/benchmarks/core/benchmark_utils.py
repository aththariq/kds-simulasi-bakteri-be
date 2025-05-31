"""
Core benchmark utilities for standardized performance testing.
"""

import time
import psutil
import tracemalloc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
import statistics
import json
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""
    name: str
    iterations: int = 10
    warmup_iterations: int = 3
    timeout_seconds: Optional[float] = None
    memory_tracking: bool = True
    cpu_tracking: bool = True
    categories: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'iterations': self.iterations,
            'warmup_iterations': self.warmup_iterations,
            'timeout_seconds': self.timeout_seconds,
            'memory_tracking': self.memory_tracking,
            'cpu_tracking': self.cpu_tracking,
            'categories': self.categories,
            'tags': self.tags
        }


@dataclass 
class BenchmarkResult:
    """Results from a benchmark test execution."""
    name: str
    config: BenchmarkConfig
    execution_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)
    success_rate: float = 1.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    @property
    def avg_execution_time(self) -> float:
        """Average execution time in seconds."""
        return statistics.mean(self.execution_times) if self.execution_times else 0.0
    
    @property
    def min_execution_time(self) -> float:
        """Minimum execution time in seconds."""
        return min(self.execution_times) if self.execution_times else 0.0
    
    @property
    def max_execution_time(self) -> float:
        """Maximum execution time in seconds."""
        return max(self.execution_times) if self.execution_times else 0.0
    
    @property
    def std_execution_time(self) -> float:
        """Standard deviation of execution times."""
        return statistics.stdev(self.execution_times) if len(self.execution_times) > 1 else 0.0
    
    @property
    def avg_memory_usage(self) -> float:
        """Average memory usage in MB."""
        return statistics.mean(self.memory_usage) if self.memory_usage else 0.0
    
    @property
    def peak_memory_usage(self) -> float:
        """Peak memory usage in MB."""
        return max(self.memory_usage) if self.memory_usage else 0.0
    
    @property
    def avg_cpu_usage(self) -> float:
        """Average CPU usage percentage."""
        return statistics.mean(self.cpu_usage) if self.cpu_usage else 0.0
    
    def operations_per_second(self, operations_count: int = 1) -> float:
        """Calculate operations per second based on average execution time."""
        if self.avg_execution_time > 0:
            return operations_count / self.avg_execution_time
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'name': self.name,
            'config': self.config.to_dict(),
            'execution_times': self.execution_times,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'success_rate': self.success_rate,
            'errors': self.errors,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'statistics': {
                'avg_execution_time': self.avg_execution_time,
                'min_execution_time': self.min_execution_time,
                'max_execution_time': self.max_execution_time,
                'std_execution_time': self.std_execution_time,
                'avg_memory_usage': self.avg_memory_usage,
                'peak_memory_usage': self.peak_memory_usage,
                'avg_cpu_usage': self.avg_cpu_usage
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResult':
        """Create result from dictionary."""
        config_data = data['config']
        config = BenchmarkConfig(**config_data)
        
        return cls(
            name=data['name'],
            config=config,
            execution_times=data.get('execution_times', []),
            memory_usage=data.get('memory_usage', []),
            cpu_usage=data.get('cpu_usage', []),
            success_rate=data.get('success_rate', 1.0),
            errors=data.get('errors', []),
            metadata=data.get('metadata', {}),
            timestamp=data.get('timestamp')
        )


class BenchmarkTimer:
    """High-precision timer for benchmarking."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start timing."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing and return duration in seconds."""
        if self.start_time is None:
            raise ValueError("Timer not started")
        self.end_time = time.perf_counter()
        return self.end_time - self.start_time
    
    @contextmanager
    def time_it(self):
        """Context manager for timing operations."""
        self.start()
        try:
            yield self
        finally:
            duration = self.stop()
            self.duration = duration


class ResourceMonitor:
    """Monitor system resources during benchmark execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.memory_samples = []
        self.cpu_samples = []
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.memory_samples.clear()
        self.cpu_samples.clear()
        
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start()
    
    def sample_resources(self):
        """Take a sample of current resource usage."""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        
        self.memory_samples.append(memory_mb)
        self.cpu_samples.append(cpu_percent)
    
    def stop_monitoring(self) -> Dict[str, List[float]]:
        """Stop monitoring and return collected samples."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        
        return {
            'memory': self.memory_samples.copy(),
            'cpu': self.cpu_samples.copy()
        }


class BenchmarkStorage:
    """Storage for benchmark results with comparison capabilities."""
    
    def __init__(self, storage_path: str = "benchmark_results"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
    
    def save_result(self, result: BenchmarkResult) -> Path:
        """Save benchmark result to file."""
        filename = f"{result.name}_{int(result.timestamp)}.json"
        filepath = self.storage_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        return filepath
    
    def load_result(self, filepath: Path) -> BenchmarkResult:
        """Load benchmark result from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return BenchmarkResult.from_dict(data)
    
    def get_latest_result(self, benchmark_name: str) -> Optional[BenchmarkResult]:
        """Get the most recent result for a benchmark."""
        pattern = f"{benchmark_name}_*.json"
        files = list(self.storage_path.glob(pattern))
        
        if not files:
            return None
        
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return self.load_result(latest_file)
    
    def get_results_history(self, benchmark_name: str, limit: int = 10) -> List[BenchmarkResult]:
        """Get recent results for a benchmark."""
        pattern = f"{benchmark_name}_*.json"
        files = list(self.storage_path.glob(pattern))
        
        if not files:
            return []
        
        # Sort by modification time, newest first
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        results = []
        for filepath in files[:limit]:
            try:
                result = self.load_result(filepath)
                results.append(result)
            except Exception as e:
                print(f"Warning: Could not load result from {filepath}: {e}")
        
        return results
    
    def compare_results(self, current: BenchmarkResult, baseline: BenchmarkResult) -> Dict[str, Any]:
        """Compare two benchmark results."""
        def percent_change(current_val: float, baseline_val: float) -> float:
            if baseline_val == 0:
                return 0.0
            return ((current_val - baseline_val) / baseline_val) * 100
        
        return {
            'execution_time': {
                'current': current.avg_execution_time,
                'baseline': baseline.avg_execution_time,
                'change_percent': percent_change(current.avg_execution_time, baseline.avg_execution_time),
                'improvement': current.avg_execution_time < baseline.avg_execution_time
            },
            'memory_usage': {
                'current': current.avg_memory_usage,
                'baseline': baseline.avg_memory_usage,
                'change_percent': percent_change(current.avg_memory_usage, baseline.avg_memory_usage),
                'improvement': current.avg_memory_usage < baseline.avg_memory_usage
            },
            'operations_per_second': {
                'current': current.operations_per_second(),
                'baseline': baseline.operations_per_second(),
                'change_percent': percent_change(current.operations_per_second(), baseline.operations_per_second()),
                'improvement': current.operations_per_second() > baseline.operations_per_second()
            }
        } 