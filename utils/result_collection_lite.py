"""
Lightweight Result Collection Framework for bacterial simulation engine.

This module provides basic tools for collecting and analyzing simulation results
without heavy dependencies like pandas/scipy for quick testing and validation.
"""

import asyncio
import json
import csv
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Generator, Union
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import logging

logger = logging.getLogger(__name__)


class ResultFormat(Enum):
    """Supported result export formats."""
    JSON = "json"
    CSV = "csv"
    PICKLE = "pickle"


class AggregationFunction(Enum):
    """Statistical aggregation functions."""
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    SUM = "sum"


@dataclass
class ResultMetrics:
    """Container for simulation result metrics."""
    simulation_id: str
    generation: int
    timestamp: datetime
    population_size: int
    resistant_count: int
    sensitive_count: int
    average_fitness: float
    fitness_std: float
    mutation_count: int
    extinction_occurred: bool
    diversity_index: float
    selection_pressure: float
    mutation_rate: float
    elapsed_time: float
    memory_usage: float
    cpu_usage: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResultMetrics':
        """Create metrics from dictionary."""
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class AggregatedResults:
    """Container for aggregated simulation results."""
    simulation_id: str
    start_time: datetime
    end_time: datetime
    total_generations: int
    final_population_size: int
    final_resistant_percentage: float
    average_fitness_trend: List[float]
    mutation_events: List[Dict[str, Any]]
    extinction_events: List[int]
    performance_stats: Dict[str, float]
    summary_statistics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat()
        return result


class ResultCollector:
    """Main class for collecting and managing simulation results."""
    
    def __init__(self, storage_path: str = "simulation_results"):
        """
        Initialize result collector.
        
        Args:
            storage_path: Directory path for storing results
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self._metrics_buffer: Dict[str, List[ResultMetrics]] = {}
        self._subscribers: List[Callable[[ResultMetrics], None]] = []
        self._streaming_active: Dict[str, bool] = {}
        
        logger.info(f"ResultCollector initialized with storage: {self.storage_path}")
    
    def add_subscriber(self, callback: Callable[[ResultMetrics], None]) -> None:
        """Add subscriber for real-time result streaming."""
        self._subscribers.append(callback)
        logger.debug(f"Added subscriber: {callback.__name__}")
    
    def remove_subscriber(self, callback: Callable[[ResultMetrics], None]) -> None:
        """Remove subscriber from streaming."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            logger.debug(f"Removed subscriber: {callback.__name__}")
    
    def collect_metrics(self, metrics: ResultMetrics) -> None:
        """
        Collect simulation metrics.
        
        Args:
            metrics: Result metrics to collect
        """
        simulation_id = metrics.simulation_id
        
        # Add to buffer
        if simulation_id not in self._metrics_buffer:
            self._metrics_buffer[simulation_id] = []
        self._metrics_buffer[simulation_id].append(metrics)
        
        # Notify subscribers for real-time streaming
        for subscriber in self._subscribers:
            try:
                subscriber(metrics)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
        
        logger.debug(f"Collected metrics for simulation {simulation_id}, generation {metrics.generation}")
    
    def get_metrics(self, simulation_id: str) -> List[ResultMetrics]:
        """Get all metrics for a simulation."""
        return self._metrics_buffer.get(simulation_id, [])
    
    def clear_metrics(self, simulation_id: str) -> None:
        """Clear metrics buffer for a simulation."""
        if simulation_id in self._metrics_buffer:
            del self._metrics_buffer[simulation_id]
            logger.debug(f"Cleared metrics for simulation {simulation_id}")
    
    def save_metrics(self, simulation_id: str, format_type: ResultFormat = ResultFormat.JSON) -> Path:
        """
        Save metrics to file.
        
        Args:
            simulation_id: Simulation identifier
            format_type: Export format
            
        Returns:
            Path to saved file
        """
        metrics = self.get_metrics(simulation_id)
        if not metrics:
            raise ValueError(f"No metrics found for simulation {simulation_id}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{simulation_id}_{timestamp}.{format_type.value}"
        filepath = self.storage_path / filename
        
        if format_type == ResultFormat.JSON:
            with open(filepath, 'w') as f:
                json.dump([m.to_dict() for m in metrics], f, indent=2)
        
        elif format_type == ResultFormat.CSV:
            # Use built-in CSV module instead of pandas
            with open(filepath, 'w', newline='') as f:
                if metrics:
                    writer = csv.DictWriter(f, fieldnames=metrics[0].to_dict().keys())
                    writer.writeheader()
                    for metric in metrics:
                        writer.writerow(metric.to_dict())
        
        elif format_type == ResultFormat.PICKLE:
            with open(filepath, 'wb') as f:
                pickle.dump(metrics, f)
        
        logger.info(f"Saved {len(metrics)} metrics to {filepath}")
        return filepath
    
    def load_metrics(self, filepath: Path) -> List[ResultMetrics]:
        """
        Load metrics from file.
        
        Args:
            filepath: Path to metrics file
            
        Returns:
            List of result metrics
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Metrics file not found: {filepath}")
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
                return [ResultMetrics.from_dict(item) for item in data]
        
        elif filepath.suffix == '.pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")


class LiteResultAnalyzer:
    """Lightweight analyzer for simulation results using built-in statistics."""
    
    def __init__(self, collector: ResultCollector):
        """
        Initialize analyzer.
        
        Args:
            collector: Result collector instance
        """
        self.collector = collector
    
    def aggregate_results(self, simulation_id: str) -> AggregatedResults:
        """
        Aggregate results for a simulation.
        
        Args:
            simulation_id: Simulation identifier
            
        Returns:
            Aggregated results
        """
        metrics = self.collector.get_metrics(simulation_id)
        if not metrics:
            raise ValueError(f"No metrics found for simulation {simulation_id}")
        
        # Calculate aggregations using built-in statistics
        fitness_values = [m.average_fitness for m in metrics]
        population_sizes = [m.population_size for m in metrics]
        resistant_counts = [m.resistant_count for m in metrics]
        
        # Basic statistics
        avg_fitness_trend = fitness_values
        final_population = population_sizes[-1] if population_sizes else 0
        final_resistant = resistant_counts[-1] if resistant_counts else 0
        final_resistant_percentage = (final_resistant / final_population * 100) if final_population > 0 else 0
        
        # Find mutation and extinction events
        mutation_events = []
        extinction_events = []
        
        for i, metric in enumerate(metrics):
            if metric.mutation_count > 0:
                mutation_events.append({
                    'generation': metric.generation,
                    'mutation_count': metric.mutation_count,
                    'timestamp': metric.timestamp.isoformat()
                })
            
            if metric.extinction_occurred:
                extinction_events.append(metric.generation)
        
        # Performance statistics
        performance_stats = {
            'avg_elapsed_time': statistics.mean([m.elapsed_time for m in metrics]),
            'avg_memory_usage': statistics.mean([m.memory_usage for m in metrics]),
            'avg_cpu_usage': statistics.mean([m.cpu_usage for m in metrics]),
            'total_generations': len(metrics)
        }
        
        # Summary statistics
        summary_statistics = {
            'avg_fitness': statistics.mean(fitness_values) if fitness_values else 0,
            'min_fitness': min(fitness_values) if fitness_values else 0,
            'max_fitness': max(fitness_values) if fitness_values else 0,
            'avg_population': statistics.mean(population_sizes) if population_sizes else 0,
            'total_mutations': sum(m.mutation_count for m in metrics),
            'extinction_count': len(extinction_events)
        }
        
        if len(fitness_values) > 1:
            summary_statistics['fitness_std'] = statistics.stdev(fitness_values)
        else:
            summary_statistics['fitness_std'] = 0
        
        return AggregatedResults(
            simulation_id=simulation_id,
            start_time=metrics[0].timestamp,
            end_time=metrics[-1].timestamp,
            total_generations=len(metrics),
            final_population_size=final_population,
            final_resistant_percentage=final_resistant_percentage,
            average_fitness_trend=avg_fitness_trend,
            mutation_events=mutation_events,
            extinction_events=extinction_events,
            performance_stats=performance_stats,
            summary_statistics=summary_statistics
        )
    
    def statistical_analysis(self, simulation_id: str) -> Dict[str, Any]:
        """
        Perform basic statistical analysis.
        
        Args:
            simulation_id: Simulation identifier
            
        Returns:
            Statistical analysis results
        """
        metrics = self.collector.get_metrics(simulation_id)
        if not metrics:
            raise ValueError(f"No metrics found for simulation {simulation_id}")
        
        # Extract numerical data
        fitness_values = [m.average_fitness for m in metrics]
        population_values = [m.population_size for m in metrics]
        resistance_ratios = [(m.resistant_count / m.population_size * 100) if m.population_size > 0 else 0 for m in metrics]
        
        analysis = {
            'simulation_id': simulation_id,
            'sample_size': len(metrics),
            'fitness_analysis': {
                'mean': statistics.mean(fitness_values),
                'median': statistics.median(fitness_values),
                'min': min(fitness_values),
                'max': max(fitness_values),
                'range': max(fitness_values) - min(fitness_values),
                'std': statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0
            },
            'population_analysis': {
                'mean': statistics.mean(population_values),
                'median': statistics.median(population_values),
                'min': min(population_values),
                'max': max(population_values),
                'std': statistics.stdev(population_values) if len(population_values) > 1 else 0
            },
            'resistance_analysis': {
                'mean_percentage': statistics.mean(resistance_ratios),
                'median_percentage': statistics.median(resistance_ratios),
                'min_percentage': min(resistance_ratios),
                'max_percentage': max(resistance_ratios),
                'std_percentage': statistics.stdev(resistance_ratios) if len(resistance_ratios) > 1 else 0
            },
            'trend_analysis': {
                'fitness_trend': 'increasing' if fitness_values[-1] > fitness_values[0] else 'decreasing',
                'population_trend': 'increasing' if population_values[-1] > population_values[0] else 'decreasing',
                'resistance_trend': 'increasing' if resistance_ratios[-1] > resistance_ratios[0] else 'decreasing'
            }
        }
        
        return analysis
    
    def generate_report(self, simulation_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive report.
        
        Args:
            simulation_id: Simulation identifier
            
        Returns:
            Complete simulation report
        """
        try:
            aggregated = self.aggregate_results(simulation_id)
            analysis = self.statistical_analysis(simulation_id)
            
            report = {
                'simulation_id': simulation_id,
                'generated_at': datetime.now().isoformat(),
                'aggregated_results': aggregated.to_dict(),
                'statistical_analysis': analysis,
                'recommendations': self._generate_recommendations(analysis)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report for {simulation_id}: {e}")
            raise
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        fitness_analysis = analysis.get('fitness_analysis', {})
        resistance_analysis = analysis.get('resistance_analysis', {})
        trend_analysis = analysis.get('trend_analysis', {})
        
        # Fitness recommendations
        if fitness_analysis.get('std', 0) > 0.5:
            recommendations.append("High fitness variability detected - consider adjusting selection pressure")
        
        if trend_analysis.get('fitness_trend') == 'decreasing':
            recommendations.append("Fitness is declining - mutation rate may be too high")
        
        # Resistance recommendations
        if resistance_analysis.get('mean_percentage', 0) > 80:
            recommendations.append("High resistance levels - consider increasing treatment intensity")
        
        if trend_analysis.get('resistance_trend') == 'increasing':
            recommendations.append("Resistance is spreading rapidly - early intervention recommended")
        
        # Population recommendations
        if trend_analysis.get('population_trend') == 'decreasing':
            recommendations.append("Population declining - monitor for extinction risk")
        
        return recommendations


class StreamingResultCollector(ResultCollector):
    """Lightweight streaming result collector with asyncio support."""
    
    def __init__(self, storage_path: str = "simulation_results", stream_interval: float = 1.0):
        """
        Initialize streaming collector.
        
        Args:
            storage_path: Directory for storing results
            stream_interval: Interval between stream updates (seconds)
        """
        super().__init__(storage_path)
        self.stream_interval = stream_interval
        self._streaming_tasks: Dict[str, asyncio.Task] = {}
        self._streaming_callbacks: Dict[str, Callable] = {}
    
    async def start_streaming(self, simulation_id: str, callback: Callable[[List[ResultMetrics]], None]) -> None:
        """
        Start streaming metrics for a simulation.
        
        Args:
            simulation_id: Simulation identifier
            callback: Function to call with metrics updates
        """
        if simulation_id in self._streaming_tasks:
            logger.warning(f"Streaming already active for simulation {simulation_id}")
            return
        
        self._streaming_callbacks[simulation_id] = callback
        
        async def stream_loop():
            while simulation_id in self._streaming_tasks:
                try:
                    metrics = self.get_metrics(simulation_id)
                    if metrics:
                        callback(metrics)
                    await asyncio.sleep(self.stream_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in streaming loop for {simulation_id}: {e}")
        
        task = asyncio.create_task(stream_loop())
        self._streaming_tasks[simulation_id] = task
        logger.info(f"Started streaming for simulation {simulation_id}")
    
    async def stop_streaming(self, simulation_id: str) -> None:
        """
        Stop streaming for a simulation.
        
        Args:
            simulation_id: Simulation identifier
        """
        if simulation_id in self._streaming_tasks:
            task = self._streaming_tasks[simulation_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            del self._streaming_tasks[simulation_id]
            if simulation_id in self._streaming_callbacks:
                del self._streaming_callbacks[simulation_id]
            
            logger.info(f"Stopped streaming for simulation {simulation_id}")
    
    async def cleanup(self) -> None:
        """Stop all streaming tasks."""
        for simulation_id in list(self._streaming_tasks.keys()):
            await self.stop_streaming(simulation_id)
        
        logger.info("All streaming tasks stopped") 