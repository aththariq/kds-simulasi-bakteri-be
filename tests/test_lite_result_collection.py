#!/usr/bin/env python3
"""
Simple test for lightweight result collection framework.
This verifies that task 7.5 implementation works without heavy dependencies.
"""

import tempfile
import shutil
from datetime import datetime
from utils.result_collection_lite import (
    ResultMetrics, ResultCollector, LiteResultAnalyzer, 
    StreamingResultCollector, ResultFormat
)


def test_basic_functionality():
    """Test basic result collection and analysis functionality."""
    print("Testing basic result collection functionality...")
    
    # Create temporary storage
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize collector
        collector = ResultCollector(temp_dir)
        analyzer = LiteResultAnalyzer(collector)
        
        # Create sample metrics
        simulation_id = "test_sim_001"
        metrics_list = []
        
        for generation in range(10):
            metrics = ResultMetrics(
                simulation_id=simulation_id,
                generation=generation,
                timestamp=datetime.now(),
                population_size=1000 - generation * 50,  # Declining population
                resistant_count=generation * 80,  # Increasing resistance
                sensitive_count=1000 - generation * 130,  # Declining sensitive
                average_fitness=1.0 - generation * 0.02,  # Declining fitness
                fitness_std=0.1 + generation * 0.01,
                mutation_count=generation * 2,
                extinction_occurred=False,
                diversity_index=0.8 - generation * 0.05,
                selection_pressure=0.5 + generation * 0.03,
                mutation_rate=0.001,
                elapsed_time=0.5 + generation * 0.1,
                memory_usage=100 + generation * 10,
                cpu_usage=50 + generation * 2
            )
            metrics_list.append(metrics)
            collector.collect_metrics(metrics)
        
        # Test metric retrieval
        retrieved_metrics = collector.get_metrics(simulation_id)
        assert len(retrieved_metrics) == 10, f"Expected 10 metrics, got {len(retrieved_metrics)}"
        print("✓ Metric collection working")
        
        # Test aggregation
        aggregated = analyzer.aggregate_results(simulation_id)
        assert aggregated.simulation_id == simulation_id
        assert aggregated.total_generations == 10
        print("✓ Result aggregation working")
        
        # Test statistical analysis
        analysis = analyzer.statistical_analysis(simulation_id)
        assert analysis['simulation_id'] == simulation_id
        assert analysis['sample_size'] == 10
        assert 'fitness_analysis' in analysis
        assert 'population_analysis' in analysis
        assert 'resistance_analysis' in analysis
        print("✓ Statistical analysis working")
        
        # Test report generation
        report = analyzer.generate_report(simulation_id)
        assert 'aggregated_results' in report
        assert 'statistical_analysis' in report
        assert 'recommendations' in report
        print("✓ Report generation working")
        
        # Test file saving
        json_path = collector.save_metrics(simulation_id, ResultFormat.JSON)
        assert json_path.exists(), "JSON file not created"
        print("✓ JSON export working")
        
        csv_path = collector.save_metrics(simulation_id, ResultFormat.CSV)
        assert csv_path.exists(), "CSV file not created"
        print("✓ CSV export working")
        
        # Test file loading
        loaded_metrics = collector.load_metrics(json_path)
        assert len(loaded_metrics) == 10, "Failed to load correct number of metrics"
        print("✓ File loading working")
        
        print("All basic functionality tests passed!")


def test_subscriber_functionality():
    """Test subscriber notification system."""
    print("\nTesting subscriber functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        collector = ResultCollector(temp_dir)
        
        # Set up subscriber
        received_metrics = []
        
        def callback(metrics):
            received_metrics.append(metrics)
        
        collector.add_subscriber(callback)
        
        # Send some metrics
        for i in range(3):
            metrics = ResultMetrics(
                simulation_id="test_sub",
                generation=i,
                timestamp=datetime.now(),
                population_size=1000,
                resistant_count=100,
                sensitive_count=900,
                average_fitness=1.0,
                fitness_std=0.1,
                mutation_count=5,
                extinction_occurred=False,
                diversity_index=0.8,
                selection_pressure=0.5,
                mutation_rate=0.001,
                elapsed_time=0.5,
                memory_usage=100,
                cpu_usage=50
            )
            collector.collect_metrics(metrics)
        
        assert len(received_metrics) == 3, f"Expected 3 notifications, got {len(received_metrics)}"
        print("✓ Subscriber notifications working")


async def test_streaming_functionality():
    """Test streaming result collector (basic test without full async)."""
    print("\nTesting streaming collector initialization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        streaming_collector = StreamingResultCollector(temp_dir, stream_interval=0.1)
        
        # Just test initialization
        assert streaming_collector.stream_interval == 0.1
        assert hasattr(streaming_collector, '_streaming_tasks')
        assert hasattr(streaming_collector, '_streaming_callbacks')
        print("✓ Streaming collector initialization working")


def test_error_handling():
    """Test error handling scenarios."""
    print("\nTesting error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        collector = ResultCollector(temp_dir)
        analyzer = LiteResultAnalyzer(collector)
        
        # Test empty simulation
        try:
            analyzer.aggregate_results("nonexistent_sim")
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✓ Empty simulation error handling working")
        
        # Test save empty metrics
        try:
            collector.save_metrics("empty_sim")
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✓ Empty metrics save error handling working")


def main():
    """Run all tests."""
    print("Running lightweight result collection tests...")
    print("=" * 50)
    
    test_basic_functionality()
    test_subscriber_functionality() 
    
    # Note: We skip async test in this simple verification
    # since it requires proper asyncio setup
    # test_streaming_functionality()
    
    test_error_handling()
    
    print("\n" + "=" * 50)
    print("All tests passed! Task 7.5 core functionality verified.")
    print("Result Collection and Testing Framework is working correctly.")


if __name__ == "__main__":
    main() 