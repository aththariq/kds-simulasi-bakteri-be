"""
Result Collection Framework for bacterial simulation engine.

This module provides comprehensive tools for collecting, aggregating, and analyzing
simulation results with real-time streaming capabilities and export functionality.
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
import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ResultFormat(Enum):
    """Supported result export formats."""
    JSON = "json"
    CSV = "csv"
    PICKLE = "pickle"
    EXCEL = "excel"
    HDF5 = "hdf5"


class AggregationFunction(Enum):
    """Statistical aggregation functions."""
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    SUM = "sum"
    PERCENTILE_25 = "percentile_25"
    PERCENTILE_75 = "percentile_75"
    PERCENTILE_95 = "percentile_95"


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
            df = pd.DataFrame([m.to_dict() for m in metrics])
            df.to_csv(filepath, index=False)
        
        elif format_type == ResultFormat.PICKLE:
            with open(filepath, 'wb') as f:
                pickle.dump(metrics, f)
        
        elif format_type == ResultFormat.EXCEL:
            df = pd.DataFrame([m.to_dict() for m in metrics])
            df.to_excel(filepath, index=False)
        
        logger.info(f"Saved metrics for {simulation_id} to {filepath}")
        return filepath
    
    def load_metrics(self, filepath: Path) -> List[ResultMetrics]:
        """Load metrics from file."""
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return [ResultMetrics.from_dict(d) for d in data]
        
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
            return [ResultMetrics.from_dict(row.to_dict()) for _, row in df.iterrows()]
        
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")


class ResultAnalyzer:
    """Advanced analysis tools for simulation results."""
    
    def __init__(self, collector: ResultCollector):
        """Initialize analyzer with result collector."""
        self.collector = collector
        
    def aggregate_results(self, simulation_id: str) -> AggregatedResults:
        """
        Create aggregated results summary.
        
        Args:
            simulation_id: Simulation identifier
            
        Returns:
            Aggregated results summary
        """
        metrics = self.collector.get_metrics(simulation_id)
        if not metrics:
            raise ValueError(f"No metrics found for simulation {simulation_id}")
        
        # Sort by generation
        metrics.sort(key=lambda x: x.generation)
        
        # Basic statistics
        start_time = metrics[0].timestamp
        end_time = metrics[-1].timestamp
        total_generations = len(metrics)
        final_population = metrics[-1].population_size
        
        # Calculate trends
        fitness_trend = [m.average_fitness for m in metrics]
        resistance_percentages = [
            (m.resistant_count / m.population_size * 100) if m.population_size > 0 else 0
            for m in metrics
        ]
        
        # Find extinction events
        extinction_events = [
            m.generation for m in metrics 
            if m.extinction_occurred or m.population_size == 0
        ]
        
        # Performance statistics
        performance_stats = {
            'average_generation_time': np.mean([m.elapsed_time for m in metrics]),
            'total_runtime': (end_time - start_time).total_seconds(),
            'average_memory_usage': np.mean([m.memory_usage for m in metrics]),
            'peak_memory_usage': max([m.memory_usage for m in metrics]),
            'average_cpu_usage': np.mean([m.cpu_usage for m in metrics])
        }
        
        # Summary statistics
        summary_stats = {
            'final_resistance_percentage': resistance_percentages[-1] if resistance_percentages else 0,
            'peak_population': max([m.population_size for m in metrics]),
            'total_mutations': sum([m.mutation_count for m in metrics]),
            'average_diversity': np.mean([m.diversity_index for m in metrics]),
            'fitness_improvement': fitness_trend[-1] - fitness_trend[0] if len(fitness_trend) > 1 else 0,
            'extinction_rate': len(extinction_events) / total_generations if total_generations > 0 else 0
        }
        
        return AggregatedResults(
            simulation_id=simulation_id,
            start_time=start_time,
            end_time=end_time,
            total_generations=total_generations,
            final_population_size=final_population,
            final_resistant_percentage=resistance_percentages[-1] if resistance_percentages else 0,
            average_fitness_trend=fitness_trend,
            mutation_events=[],  # Could be enhanced with detailed mutation tracking
            extinction_events=extinction_events,
            performance_stats=performance_stats,
            summary_statistics=summary_stats
        )
    
    def statistical_analysis(self, simulation_id: str) -> Dict[str, Any]:
        """
        Perform statistical analysis on simulation results.
        
        Args:
            simulation_id: Simulation identifier
            
        Returns:
            Statistical analysis results
        """
        metrics = self.collector.get_metrics(simulation_id)
        if not metrics:
            raise ValueError(f"No metrics found for simulation {simulation_id}")
        
        # Extract time series data
        data = {
            'generation': [m.generation for m in metrics],
            'population_size': [m.population_size for m in metrics],
            'resistant_count': [m.resistant_count for m in metrics],
            'average_fitness': [m.average_fitness for m in metrics],
            'diversity_index': [m.diversity_index for m in metrics],
            'mutation_count': [m.mutation_count for m in metrics]
        }
        
        # Statistical analysis
        analysis = {}
        
        for key, values in data.items():
            if key == 'generation':
                continue
                
            analysis[key] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'percentile_25': np.percentile(values, 25),
                'percentile_75': np.percentile(values, 75),
                'percentile_95': np.percentile(values, 95),
                'skewness': stats.skew(values),
                'kurtosis': stats.kurtosis(values)
            }
            
            # Trend analysis
            if len(values) > 1:
                generations = data['generation']
                slope, intercept, r_value, p_value, std_err = stats.linregress(generations, values)
                analysis[key]['trend'] = {
                    'slope': slope,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                }
        
        return analysis
    
    def compare_simulations(self, simulation_ids: List[str]) -> Dict[str, Any]:
        """
        Compare multiple simulations.
        
        Args:
            simulation_ids: List of simulation identifiers to compare
            
        Returns:
            Comparison analysis
        """
        if len(simulation_ids) < 2:
            raise ValueError("At least 2 simulations required for comparison")
        
        aggregated_results = {}
        for sim_id in simulation_ids:
            try:
                aggregated_results[sim_id] = self.aggregate_results(sim_id)
            except ValueError:
                logger.warning(f"No data found for simulation {sim_id}")
                continue
        
        if len(aggregated_results) < 2:
            raise ValueError("Not enough valid simulations for comparison")
        
        # Comparison metrics
        comparison = {
            'simulation_count': len(aggregated_results),
            'final_population_comparison': {},
            'resistance_development': {},
            'performance_comparison': {},
            'statistical_tests': {}
        }
        
        # Extract comparison data
        final_populations = {sim_id: result.final_population_size 
                           for sim_id, result in aggregated_results.items()}
        resistance_percentages = {sim_id: result.final_resistant_percentage 
                                for sim_id, result in aggregated_results.items()}
        runtimes = {sim_id: result.performance_stats['total_runtime'] 
                   for sim_id, result in aggregated_results.items()}
        
        # Statistical comparisons
        pop_values = list(final_populations.values())
        resist_values = list(resistance_percentages.values())
        runtime_values = list(runtimes.values())
        
        if len(pop_values) > 1:
            comparison['final_population_comparison'] = {
                'mean': np.mean(pop_values),
                'std': np.std(pop_values),
                'coefficient_of_variation': np.std(pop_values) / np.mean(pop_values) if np.mean(pop_values) > 0 else 0,
                'values': final_populations
            }
            
            comparison['resistance_development'] = {
                'mean': np.mean(resist_values),
                'std': np.std(resist_values),
                'values': resistance_percentages
            }
            
            comparison['performance_comparison'] = {
                'mean_runtime': np.mean(runtime_values),
                'std_runtime': np.std(runtime_values),
                'values': runtimes
            }
        
        return comparison
    
    def generate_report(self, simulation_id: str, include_plots: bool = False) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.
        
        Args:
            simulation_id: Simulation identifier
            include_plots: Whether to include plot data
            
        Returns:
            Comprehensive report
        """
        try:
            aggregated = self.aggregate_results(simulation_id)
            statistical = self.statistical_analysis(simulation_id)
            
            report = {
                'simulation_id': simulation_id,
                'generated_at': datetime.now().isoformat(),
                'summary': aggregated.summary_statistics,
                'performance': aggregated.performance_stats,
                'statistical_analysis': statistical,
                'trends': {
                    'fitness_trend': aggregated.average_fitness_trend,
                    'extinction_events': aggregated.extinction_events
                }
            }
            
            if include_plots:
                # Add plot data for visualization
                metrics = self.collector.get_metrics(simulation_id)
                report['plot_data'] = {
                    'generations': [m.generation for m in metrics],
                    'population_sizes': [m.population_size for m in metrics],
                    'fitness_values': [m.average_fitness for m in metrics],
                    'resistance_counts': [m.resistant_count for m in metrics],
                    'diversity_indices': [m.diversity_index for m in metrics]
                }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report for {simulation_id}: {e}")
            raise


class StreamingResultCollector(ResultCollector):
    """Enhanced result collector with real-time streaming capabilities."""
    
    def __init__(self, storage_path: str = "simulation_results", stream_interval: float = 1.0):
        """
        Initialize streaming collector.
        
        Args:
            storage_path: Directory path for storing results
            stream_interval: Interval between stream updates in seconds
        """
        super().__init__(storage_path)
        self.stream_interval = stream_interval
        self._stream_tasks: Dict[str, asyncio.Task] = {}
    
    async def start_streaming(self, simulation_id: str, callback: Callable[[List[ResultMetrics]], None]) -> None:
        """
        Start streaming results for a simulation.
        
        Args:
            simulation_id: Simulation identifier
            callback: Function to call with streamed results
        """
        if simulation_id in self._stream_tasks:
            logger.warning(f"Streaming already active for {simulation_id}")
            return
        
        self._streaming_active[simulation_id] = True
        
        async def stream_loop():
            last_count = 0
            while self._streaming_active.get(simulation_id, False):
                try:
                    metrics = self.get_metrics(simulation_id)
                    if len(metrics) > last_count:
                        new_metrics = metrics[last_count:]
                        callback(new_metrics)
                        last_count = len(metrics)
                    
                    await asyncio.sleep(self.stream_interval)
                except Exception as e:
                    logger.error(f"Error in streaming loop for {simulation_id}: {e}")
                    break
        
        self._stream_tasks[simulation_id] = asyncio.create_task(stream_loop())
        logger.info(f"Started streaming for simulation {simulation_id}")
    
    async def stop_streaming(self, simulation_id: str) -> None:
        """Stop streaming for a simulation."""
        self._streaming_active[simulation_id] = False
        
        if simulation_id in self._stream_tasks:
            self._stream_tasks[simulation_id].cancel()
            try:
                await self._stream_tasks[simulation_id]
            except asyncio.CancelledError:
                pass
            del self._stream_tasks[simulation_id]
        
        logger.info(f"Stopped streaming for simulation {simulation_id}")
    
    async def cleanup(self) -> None:
        """Clean up all streaming tasks."""
        for simulation_id in list(self._stream_tasks.keys()):
            await self.stop_streaming(simulation_id) 