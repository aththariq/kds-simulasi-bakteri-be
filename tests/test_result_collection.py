"""
Comprehensive tests for the Result Collection Framework.

This test suite validates the accuracy, reliability, and efficiency of the 
result collection and analysis systems.
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import json
import csv
import pickle
import numpy as np
import pandas as pd

from utils.result_collection import (
    ResultMetrics, AggregatedResults, ResultCollector, ResultAnalyzer,
    StreamingResultCollector, ResultFormat, AggregationFunction
)


class TestResultMetrics:
    """Test ResultMetrics data class."""
    
    def test_metrics_creation(self):
        """Test basic metrics creation."""
        timestamp = datetime.now()
        metrics = ResultMetrics(
            simulation_id="test_sim",
            generation=10,
            timestamp=timestamp,
            population_size=1000,
            resistant_count=150,
            sensitive_count=850,
            average_fitness=1.2,
            fitness_std=0.3,
            mutation_count=25,
            extinction_occurred=False,
            diversity_index=0.8,
            selection_pressure=0.5,
            mutation_rate=1e-6,
            elapsed_time=2.5,
            memory_usage=128.5,
            cpu_usage=45.2
        )
        
        assert metrics.simulation_id == "test_sim"
        assert metrics.generation == 10
        assert metrics.timestamp == timestamp
        assert metrics.population_size == 1000
        assert metrics.resistant_count == 150
        assert metrics.sensitive_count == 850
    
    def test_metrics_to_dict(self):
        """Test conversion to dictionary."""
        timestamp = datetime.now()
        metrics = ResultMetrics(
            simulation_id="test_sim",
            generation=5,
            timestamp=timestamp,
            population_size=500,
            resistant_count=100,
            sensitive_count=400,
            average_fitness=1.0,
            fitness_std=0.2,
            mutation_count=10,
            extinction_occurred=False,
            diversity_index=0.7,
            selection_pressure=0.3,
            mutation_rate=1e-6,
            elapsed_time=1.0,
            memory_usage=64.0,
            cpu_usage=30.0
        )
        
        data = metrics.to_dict()
        assert data['simulation_id'] == "test_sim"
        assert data['generation'] == 5
        assert data['timestamp'] == timestamp.isoformat()
        assert data['population_size'] == 500
    
    def test_metrics_from_dict(self):
        """Test creation from dictionary."""
        timestamp_str = datetime.now().isoformat()
        data = {
            'simulation_id': "test_sim",
            'generation': 15,
            'timestamp': timestamp_str,
            'population_size': 750,
            'resistant_count': 200,
            'sensitive_count': 550,
            'average_fitness': 1.5,
            'fitness_std': 0.4,
            'mutation_count': 30,
            'extinction_occurred': True,
            'diversity_index': 0.6,
            'selection_pressure': 0.8,
            'mutation_rate': 2e-6,
            'elapsed_time': 3.2,
            'memory_usage': 256.0,
            'cpu_usage': 60.0
        }
        
        metrics = ResultMetrics.from_dict(data)
        assert metrics.simulation_id == "test_sim"
        assert metrics.generation == 15
        assert metrics.timestamp == datetime.fromisoformat(timestamp_str)
        assert metrics.extinction_occurred is True


class TestResultCollector:
    """Test ResultCollector functionality."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def collector(self, temp_storage):
        """Create result collector with temporary storage."""
        return ResultCollector(temp_storage)
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for testing."""
        return [
            ResultMetrics(
                simulation_id="test_sim",
                generation=i,
                timestamp=datetime.now() + timedelta(seconds=i),
                population_size=1000 - i * 10,
                resistant_count=i * 5,
                sensitive_count=1000 - i * 15,
                average_fitness=1.0 + i * 0.01,
                fitness_std=0.2,
                mutation_count=i * 2,
                extinction_occurred=False,
                diversity_index=0.8 - i * 0.01,
                selection_pressure=0.5,
                mutation_rate=1e-6,
                elapsed_time=1.0 + i * 0.1,
                memory_usage=100.0 + i,
                cpu_usage=50.0 + i
            )
            for i in range(10)
        ]
    
    def test_collector_initialization(self, temp_storage):
        """Test collector initialization."""
        collector = ResultCollector(temp_storage)
        assert collector.storage_path == Path(temp_storage)
        assert collector.storage_path.exists()
        assert len(collector._metrics_buffer) == 0
        assert len(collector._subscribers) == 0
    
    def test_collect_metrics(self, collector, sample_metrics):
        """Test metrics collection."""
        for metrics in sample_metrics:
            collector.collect_metrics(metrics)
        
        collected = collector.get_metrics("test_sim")
        assert len(collected) == 10
        assert collected[0].generation == 0
        assert collected[-1].generation == 9
    
    def test_subscriber_notification(self, collector):
        """Test subscriber notification system."""
        notifications = []
        
        def callback(metrics):
            notifications.append(metrics)
        
        collector.add_subscriber(callback)
        
        metrics = ResultMetrics(
            simulation_id="test_sim",
            generation=1,
            timestamp=datetime.now(),
            population_size=1000,
            resistant_count=100,
            sensitive_count=900,
            average_fitness=1.0,
            fitness_std=0.2,
            mutation_count=5,
            extinction_occurred=False,
            diversity_index=0.8,
            selection_pressure=0.5,
            mutation_rate=1e-6,
            elapsed_time=1.0,
            memory_usage=100.0,
            cpu_usage=50.0
        )
        
        collector.collect_metrics(metrics)
        
        assert len(notifications) == 1
        assert notifications[0].simulation_id == "test_sim"
        
        # Test subscriber removal
        collector.remove_subscriber(callback)
        collector.collect_metrics(metrics)
        assert len(notifications) == 1  # Should not increase
    
    def test_clear_metrics(self, collector, sample_metrics):
        """Test metrics clearing."""
        for metrics in sample_metrics:
            collector.collect_metrics(metrics)
        
        assert len(collector.get_metrics("test_sim")) == 10
        
        collector.clear_metrics("test_sim")
        assert len(collector.get_metrics("test_sim")) == 0
    
    def test_save_metrics_json(self, collector, sample_metrics):
        """Test saving metrics in JSON format."""
        for metrics in sample_metrics:
            collector.collect_metrics(metrics)
        
        filepath = collector.save_metrics("test_sim", ResultFormat.JSON)
        assert filepath.exists()
        assert filepath.suffix == '.json'
        
        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 10
        assert data[0]['simulation_id'] == "test_sim"
        assert data[0]['generation'] == 0
    
    def test_save_metrics_csv(self, collector, sample_metrics):
        """Test saving metrics in CSV format."""
        for metrics in sample_metrics:
            collector.collect_metrics(metrics)
        
        filepath = collector.save_metrics("test_sim", ResultFormat.CSV)
        assert filepath.exists()
        assert filepath.suffix == '.csv'
        
        # Verify content
        df = pd.read_csv(filepath)
        assert len(df) == 10
        assert df['simulation_id'].iloc[0] == "test_sim"
    
    def test_save_metrics_pickle(self, collector, sample_metrics):
        """Test saving metrics in pickle format."""
        for metrics in sample_metrics:
            collector.collect_metrics(metrics)
        
        filepath = collector.save_metrics("test_sim", ResultFormat.PICKLE)
        assert filepath.exists()
        assert filepath.suffix == '.pickle'
        
        # Verify content
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        assert len(data) == 10
        assert isinstance(data[0], ResultMetrics)
    
    def test_load_metrics(self, collector, sample_metrics):
        """Test loading metrics from files."""
        for metrics in sample_metrics:
            collector.collect_metrics(metrics)
        
        # Save and load JSON
        json_path = collector.save_metrics("test_sim", ResultFormat.JSON)
        loaded_json = collector.load_metrics(json_path)
        assert len(loaded_json) == 10
        assert isinstance(loaded_json[0], ResultMetrics)
        
        # Save and load pickle
        pickle_path = collector.save_metrics("test_sim", ResultFormat.PICKLE)
        loaded_pickle = collector.load_metrics(pickle_path)
        assert len(loaded_pickle) == 10
        assert isinstance(loaded_pickle[0], ResultMetrics)
    
    def test_save_empty_metrics_error(self, collector):
        """Test error when saving empty metrics."""
        with pytest.raises(ValueError, match="No metrics found"):
            collector.save_metrics("nonexistent_sim")
    
    def test_load_nonexistent_file_error(self, collector):
        """Test error when loading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            collector.load_metrics(Path("nonexistent.json"))


class TestResultAnalyzer:
    """Test ResultAnalyzer functionality."""
    
    @pytest.fixture
    def collector_with_data(self, temp_storage):
        """Create collector with sample data."""
        collector = ResultCollector(temp_storage)
        
        # Add sample data for multiple simulations
        for sim_num in range(3):
            sim_id = f"sim_{sim_num}"
            for gen in range(20):
                metrics = ResultMetrics(
                    simulation_id=sim_id,
                    generation=gen,
                    timestamp=datetime.now() + timedelta(seconds=gen),
                    population_size=1000 - gen * (sim_num + 1) * 5,
                    resistant_count=gen * (sim_num + 1) * 2,
                    sensitive_count=1000 - gen * (sim_num + 1) * 7,
                    average_fitness=1.0 + gen * 0.02 + sim_num * 0.1,
                    fitness_std=0.2 + gen * 0.01,
                    mutation_count=gen * (sim_num + 1),
                    extinction_occurred=gen > 15 and sim_num == 2,
                    diversity_index=0.8 - gen * 0.02,
                    selection_pressure=0.5 + gen * 0.01,
                    mutation_rate=1e-6 * (1 + sim_num),
                    elapsed_time=1.0 + gen * 0.05,
                    memory_usage=100.0 + gen * 2,
                    cpu_usage=50.0 + gen
                )
                collector.collect_metrics(metrics)
        
        return collector
    
    @pytest.fixture
    def analyzer(self, collector_with_data):
        """Create analyzer with sample data."""
        return ResultAnalyzer(collector_with_data)
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_aggregate_results(self, analyzer):
        """Test results aggregation."""
        aggregated = analyzer.aggregate_results("sim_0")
        
        assert aggregated.simulation_id == "sim_0"
        assert aggregated.total_generations == 20
        assert aggregated.final_population_size == 905  # 1000 - 19 * 1 * 5
        assert len(aggregated.average_fitness_trend) == 20
        assert isinstance(aggregated.performance_stats, dict)
        assert isinstance(aggregated.summary_statistics, dict)
    
    def test_statistical_analysis(self, analyzer):
        """Test statistical analysis."""
        analysis = analyzer.statistical_analysis("sim_0")
        
        assert 'population_size' in analysis
        assert 'average_fitness' in analysis
        assert 'resistant_count' in analysis
        
        # Check statistical measures
        pop_stats = analysis['population_size']
        assert 'mean' in pop_stats
        assert 'median' in pop_stats
        assert 'std' in pop_stats
        assert 'trend' in pop_stats
        
        # Check trend analysis
        assert 'slope' in pop_stats['trend']
        assert 'r_squared' in pop_stats['trend']
        assert 'direction' in pop_stats['trend']
    
    def test_compare_simulations(self, analyzer):
        """Test simulation comparison."""
        comparison = analyzer.compare_simulations(["sim_0", "sim_1", "sim_2"])
        
        assert comparison['simulation_count'] == 3
        assert 'final_population_comparison' in comparison
        assert 'resistance_development' in comparison
        assert 'performance_comparison' in comparison
        
        # Check comparison data
        pop_comp = comparison['final_population_comparison']
        assert 'mean' in pop_comp
        assert 'std' in pop_comp
        assert 'values' in pop_comp
        assert len(pop_comp['values']) == 3
    
    def test_compare_insufficient_simulations(self, analyzer):
        """Test error with insufficient simulations for comparison."""
        with pytest.raises(ValueError, match="At least 2 simulations required"):
            analyzer.compare_simulations(["sim_0"])
    
    def test_compare_nonexistent_simulations(self, analyzer):
        """Test comparison with nonexistent simulations."""
        with pytest.raises(ValueError, match="Not enough valid simulations"):
            analyzer.compare_simulations(["nonexistent_1", "nonexistent_2"])
    
    def test_generate_report(self, analyzer):
        """Test report generation."""
        report = analyzer.generate_report("sim_0", include_plots=True)
        
        assert report['simulation_id'] == "sim_0"
        assert 'generated_at' in report
        assert 'summary' in report
        assert 'performance' in report
        assert 'statistical_analysis' in report
        assert 'trends' in report
        assert 'plot_data' in report
        
        # Check plot data
        plot_data = report['plot_data']
        assert 'generations' in plot_data
        assert 'population_sizes' in plot_data
        assert 'fitness_values' in plot_data
        assert len(plot_data['generations']) == 20
    
    def test_generate_report_without_plots(self, analyzer):
        """Test report generation without plots."""
        report = analyzer.generate_report("sim_0", include_plots=False)
        
        assert 'plot_data' not in report
        assert 'summary' in report
        assert 'statistical_analysis' in report
    
    def test_analysis_nonexistent_simulation(self, analyzer):
        """Test analysis with nonexistent simulation."""
        with pytest.raises(ValueError, match="No metrics found"):
            analyzer.aggregate_results("nonexistent_sim")
        
        with pytest.raises(ValueError, match="No metrics found"):
            analyzer.statistical_analysis("nonexistent_sim")


class TestStreamingResultCollector:
    """Test streaming result collector."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def streaming_collector(self, temp_storage):
        """Create streaming collector."""
        return StreamingResultCollector(temp_storage, stream_interval=0.1)
    
    @pytest.mark.asyncio
    async def test_streaming_initialization(self, streaming_collector):
        """Test streaming collector initialization."""
        assert streaming_collector.stream_interval == 0.1
        assert len(streaming_collector._stream_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_start_stop_streaming(self, streaming_collector):
        """Test starting and stopping streaming."""
        streamed_data = []
        
        def callback(metrics_list):
            streamed_data.extend(metrics_list)
        
        # Start streaming
        await streaming_collector.start_streaming("test_sim", callback)
        assert "test_sim" in streaming_collector._stream_tasks
        
        # Add some metrics
        for i in range(3):
            metrics = ResultMetrics(
                simulation_id="test_sim",
                generation=i,
                timestamp=datetime.now(),
                population_size=1000,
                resistant_count=100,
                sensitive_count=900,
                average_fitness=1.0,
                fitness_std=0.2,
                mutation_count=5,
                extinction_occurred=False,
                diversity_index=0.8,
                selection_pressure=0.5,
                mutation_rate=1e-6,
                elapsed_time=1.0,
                memory_usage=100.0,
                cpu_usage=50.0
            )
            streaming_collector.collect_metrics(metrics)
            await asyncio.sleep(0.2)  # Wait for streaming to process
        
        # Stop streaming
        await streaming_collector.stop_streaming("test_sim")
        assert "test_sim" not in streaming_collector._stream_tasks
        
        # Should have received some data
        assert len(streamed_data) > 0
    
    @pytest.mark.asyncio
    async def test_streaming_duplicate_start(self, streaming_collector):
        """Test starting streaming when already active."""
        callback = Mock()
        
        await streaming_collector.start_streaming("test_sim", callback)
        
        # Try to start again - should not create new task
        initial_task = streaming_collector._stream_tasks["test_sim"]
        await streaming_collector.start_streaming("test_sim", callback)
        assert streaming_collector._stream_tasks["test_sim"] == initial_task
        
        await streaming_collector.stop_streaming("test_sim")
    
    @pytest.mark.asyncio
    async def test_cleanup_streaming(self, streaming_collector):
        """Test cleanup of all streaming tasks."""
        callback = Mock()
        
        # Start multiple streams
        for i in range(3):
            await streaming_collector.start_streaming(f"sim_{i}", callback)
        
        assert len(streaming_collector._stream_tasks) == 3
        
        # Cleanup all
        await streaming_collector.cleanup()
        assert len(streaming_collector._stream_tasks) == 0


class TestIntegrationScenarios:
    """Integration tests for complete result collection workflows."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_complete_simulation_workflow(self, temp_storage):
        """Test complete simulation result collection workflow."""
        # Initialize components
        collector = ResultCollector(temp_storage)
        analyzer = ResultAnalyzer(collector)
        
        # Simulate a complete simulation run
        simulation_id = "integration_test_sim"
        
        for generation in range(50):
            # Simulate changing population dynamics
            base_pop = 1000
            decay_rate = 0.98 if generation > 30 else 1.0
            
            population_size = int(base_pop * (decay_rate ** max(0, generation - 30)))
            resistant_count = min(population_size, generation * 3 + np.random.randint(0, 10))
            
            metrics = ResultMetrics(
                simulation_id=simulation_id,
                generation=generation,
                timestamp=datetime.now() + timedelta(seconds=generation * 10),
                population_size=population_size,
                resistant_count=resistant_count,
                sensitive_count=population_size - resistant_count,
                average_fitness=1.0 + generation * 0.01 + np.random.normal(0, 0.02),
                fitness_std=0.2 + np.random.normal(0, 0.05),
                mutation_count=np.random.poisson(5),
                extinction_occurred=population_size == 0,
                diversity_index=max(0.1, 0.8 - generation * 0.01 + np.random.normal(0, 0.05)),
                selection_pressure=0.5 + generation * 0.005,
                mutation_rate=1e-6 * (1 + generation * 0.01),
                elapsed_time=1.0 + np.random.exponential(0.5),
                memory_usage=100.0 + generation + np.random.normal(0, 5),
                cpu_usage=50.0 + generation * 0.5 + np.random.normal(0, 10)
            )
            
            collector.collect_metrics(metrics)
        
        # Test data collection
        collected_metrics = collector.get_metrics(simulation_id)
        assert len(collected_metrics) == 50
        
        # Test aggregation
        aggregated = analyzer.aggregate_results(simulation_id)
        assert aggregated.total_generations == 50
        assert aggregated.simulation_id == simulation_id
        
        # Test statistical analysis
        stats = analyzer.statistical_analysis(simulation_id)
        assert 'population_size' in stats
        assert 'average_fitness' in stats
        
        # Test file operations
        json_path = collector.save_metrics(simulation_id, ResultFormat.JSON)
        assert json_path.exists()
        
        csv_path = collector.save_metrics(simulation_id, ResultFormat.CSV)
        assert csv_path.exists()
        
        # Test loading
        loaded_metrics = collector.load_metrics(json_path)
        assert len(loaded_metrics) == 50
        assert isinstance(loaded_metrics[0], ResultMetrics)
        
        # Test report generation
        report = analyzer.generate_report(simulation_id, include_plots=True)
        assert 'plot_data' in report
        assert 'statistical_analysis' in report
    
    def test_multi_simulation_comparison_workflow(self, temp_storage):
        """Test workflow with multiple simulation comparison."""
        collector = ResultCollector(temp_storage)
        analyzer = ResultAnalyzer(collector)
        
        # Create three different simulation scenarios
        scenarios = {
            "low_pressure": {"mutation_multiplier": 1.0, "selection_strength": 0.3},
            "medium_pressure": {"mutation_multiplier": 2.0, "selection_strength": 0.6},
            "high_pressure": {"mutation_multiplier": 5.0, "selection_strength": 0.9}
        }
        
        simulation_ids = []
        
        for scenario_name, params in scenarios.items():
            sim_id = f"scenario_{scenario_name}"
            simulation_ids.append(sim_id)
            
            for generation in range(30):
                # Simulate different evolutionary pressures
                mut_mult = params["mutation_multiplier"]
                sel_strength = params["selection_strength"]
                
                population_size = max(10, 1000 - generation * sel_strength * 10)
                resistant_proportion = min(0.8, generation * 0.02 * mut_mult)
                resistant_count = int(population_size * resistant_proportion)
                
                metrics = ResultMetrics(
                    simulation_id=sim_id,
                    generation=generation,
                    timestamp=datetime.now() + timedelta(seconds=generation),
                    population_size=int(population_size),
                    resistant_count=resistant_count,
                    sensitive_count=int(population_size) - resistant_count,
                    average_fitness=1.0 + generation * 0.02 * (1 + sel_strength),
                    fitness_std=0.2,
                    mutation_count=int(5 * mut_mult),
                    extinction_occurred=population_size < 50,
                    diversity_index=max(0.2, 0.8 - generation * 0.01 * sel_strength),
                    selection_pressure=sel_strength,
                    mutation_rate=1e-6 * mut_mult,
                    elapsed_time=1.0,
                    memory_usage=100.0,
                    cpu_usage=50.0
                )
                
                collector.collect_metrics(metrics)
        
        # Test comparison analysis
        comparison = analyzer.compare_simulations(simulation_ids)
        assert comparison['simulation_count'] == 3
        assert len(comparison['final_population_comparison']['values']) == 3
        
        # Verify different scenarios produced different outcomes
        final_populations = comparison['final_population_comparison']['values']
        assert len(set(final_populations.values())) > 1  # Should have different results
        
        # Test individual analysis for each simulation
        for sim_id in simulation_ids:
            report = analyzer.generate_report(sim_id)
            assert report['simulation_id'] == sim_id
            assert 'summary' in report
    
    @pytest.mark.asyncio
    async def test_streaming_integration_workflow(self, temp_storage):
        """Test integration with streaming capabilities."""
        streaming_collector = StreamingResultCollector(temp_storage, stream_interval=0.05)
        analyzer = ResultAnalyzer(streaming_collector)
        
        # Track streamed data
        streamed_metrics = []
        
        def stream_callback(metrics_list):
            streamed_metrics.extend(metrics_list)
        
        simulation_id = "streaming_test"
        
        # Start streaming
        await streaming_collector.start_streaming(simulation_id, stream_callback)
        
        # Simulate real-time data collection
        for generation in range(20):
            metrics = ResultMetrics(
                simulation_id=simulation_id,
                generation=generation,
                timestamp=datetime.now(),
                population_size=1000 - generation * 10,
                resistant_count=generation * 5,
                sensitive_count=1000 - generation * 15,
                average_fitness=1.0 + generation * 0.01,
                fitness_std=0.2,
                mutation_count=generation,
                extinction_occurred=False,
                diversity_index=0.8 - generation * 0.01,
                selection_pressure=0.5,
                mutation_rate=1e-6,
                elapsed_time=1.0,
                memory_usage=100.0,
                cpu_usage=50.0
            )
            
            streaming_collector.collect_metrics(metrics)
            await asyncio.sleep(0.1)  # Allow streaming to process
        
        # Stop streaming
        await streaming_collector.stop_streaming(simulation_id)
        
        # Verify streaming worked
        assert len(streamed_metrics) > 0
        assert len(streamed_metrics) <= 20  # Should have received data
        
        # Test analysis on streamed data
        collected_metrics = streaming_collector.get_metrics(simulation_id)
        assert len(collected_metrics) == 20
        
        aggregated = analyzer.aggregate_results(simulation_id)
        assert aggregated.total_generations == 20


class TestPerformanceValidation:
    """Performance tests for result collection framework."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_large_dataset_collection_performance(self, temp_storage):
        """Test performance with large datasets."""
        collector = ResultCollector(temp_storage)
        
        # Generate large dataset
        start_time = datetime.now()
        
        for generation in range(1000):  # 1000 generations
            metrics = ResultMetrics(
                simulation_id="perf_test",
                generation=generation,
                timestamp=datetime.now(),
                population_size=1000,
                resistant_count=100,
                sensitive_count=900,
                average_fitness=1.0,
                fitness_std=0.2,
                mutation_count=5,
                extinction_occurred=False,
                diversity_index=0.8,
                selection_pressure=0.5,
                mutation_rate=1e-6,
                elapsed_time=1.0,
                memory_usage=100.0,
                cpu_usage=50.0
            )
            collector.collect_metrics(metrics)
        
        collection_time = (datetime.now() - start_time).total_seconds()
        
        # Should collect 1000 metrics efficiently (< 5 seconds)
        assert collection_time < 5.0
        assert len(collector.get_metrics("perf_test")) == 1000
    
    def test_analysis_performance_large_dataset(self, temp_storage):
        """Test analysis performance with large datasets."""
        collector = ResultCollector(temp_storage)
        analyzer = ResultAnalyzer(collector)
        
        # Create large dataset
        for generation in range(500):
            metrics = ResultMetrics(
                simulation_id="analysis_perf_test",
                generation=generation,
                timestamp=datetime.now() + timedelta(seconds=generation),
                population_size=1000 - generation,
                resistant_count=generation * 2,
                sensitive_count=1000 - generation * 3,
                average_fitness=1.0 + generation * 0.001,
                fitness_std=0.2,
                mutation_count=generation,
                extinction_occurred=False,
                diversity_index=0.8 - generation * 0.001,
                selection_pressure=0.5,
                mutation_rate=1e-6,
                elapsed_time=1.0,
                memory_usage=100.0 + generation,
                cpu_usage=50.0
            )
            collector.collect_metrics(metrics)
        
        # Test aggregation performance
        start_time = datetime.now()
        aggregated = analyzer.aggregate_results("analysis_perf_test")
        aggregation_time = (datetime.now() - start_time).total_seconds()
        
        assert aggregation_time < 2.0  # Should be fast
        assert aggregated.total_generations == 500
        
        # Test statistical analysis performance
        start_time = datetime.now()
        stats = analyzer.statistical_analysis("analysis_perf_test")
        stats_time = (datetime.now() - start_time).total_seconds()
        
        assert stats_time < 3.0  # Should be reasonably fast
        assert 'population_size' in stats
    
    def test_file_io_performance(self, temp_storage):
        """Test file I/O performance."""
        collector = ResultCollector(temp_storage)
        
        # Generate moderate dataset
        for generation in range(200):
            metrics = ResultMetrics(
                simulation_id="io_perf_test",
                generation=generation,
                timestamp=datetime.now(),
                population_size=1000,
                resistant_count=100,
                sensitive_count=900,
                average_fitness=1.0,
                fitness_std=0.2,
                mutation_count=5,
                extinction_occurred=False,
                diversity_index=0.8,
                selection_pressure=0.5,
                mutation_rate=1e-6,
                elapsed_time=1.0,
                memory_usage=100.0,
                cpu_usage=50.0
            )
            collector.collect_metrics(metrics)
        
        # Test save performance for different formats
        formats_to_test = [ResultFormat.JSON, ResultFormat.CSV, ResultFormat.PICKLE]
        
        for format_type in formats_to_test:
            start_time = datetime.now()
            filepath = collector.save_metrics("io_perf_test", format_type)
            save_time = (datetime.now() - start_time).total_seconds()
            
            assert save_time < 2.0  # Should save quickly
            assert filepath.exists()
            
            # Test load performance
            start_time = datetime.now()
            loaded_metrics = collector.load_metrics(filepath)
            load_time = (datetime.now() - start_time).total_seconds()
            
            assert load_time < 2.0  # Should load quickly
            assert len(loaded_metrics) == 200


if __name__ == "__main__":
    pytest.main([__file__]) 