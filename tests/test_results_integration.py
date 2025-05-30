"""
Integration tests for the Result Collection Framework with the simulation system.

This test suite validates the end-to-end functionality of result collection,
including integration with the simulation engine and API endpoints.
"""

import pytest
import asyncio
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

from main import app
from utils.result_collection import (
    ResultCollector, ResultAnalyzer, StreamingResultCollector,
    ResultMetrics, ResultFormat
)
from services.simulation_service import SimulationService
from models.population import Population
from models.bacterium import Bacterium, ResistanceStatus


class TestResultCollectionIntegration:
    """Integration tests for result collection with simulation system."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_api_key(self):
        """Mock API key for testing."""
        return "test-api-key-12345"
    
    @pytest.fixture
    def sample_result_metrics(self):
        """Sample result metrics for testing."""
        return {
            "simulation_id": "integration_test_sim",
            "generation": 10,
            "timestamp": datetime.now().isoformat(),
            "population_size": 1000,
            "resistant_count": 150,
            "sensitive_count": 850,
            "average_fitness": 1.2,
            "fitness_std": 0.3,
            "mutation_count": 25,
            "extinction_occurred": False,
            "diversity_index": 0.8,
            "selection_pressure": 0.5,
            "mutation_rate": 1e-6,
            "elapsed_time": 2.5,
            "memory_usage": 128.5,
            "cpu_usage": 45.2
        }
    
    def test_api_health_check(self, client):
        """Test result collection API health check."""
        response = client.get("/api/results/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "storage_path" in data
        assert "active_simulations" in data
    
    @patch('utils.auth.verify_api_key')
    def test_collect_metrics_endpoint(self, mock_auth, client, sample_result_metrics):
        """Test metrics collection endpoint."""
        mock_auth.return_value = "test-api-key"
        
        response = client.post(
            "/api/results/collect",
            json=sample_result_metrics,
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "integration_test_sim" in data["message"]
        assert "generation 10" in data["message"]
    
    @patch('utils.auth.verify_api_key')
    def test_list_simulations_endpoint(self, mock_auth, client, sample_result_metrics):
        """Test simulations listing endpoint."""
        mock_auth.return_value = "test-api-key"
        
        # First collect some metrics
        client.post(
            "/api/results/collect",
            json=sample_result_metrics,
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        # Then list simulations
        response = client.get(
            "/api/results/simulations",
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "simulations" in data
        assert data["total_count"] >= 1
        
        # Check simulation details
        simulations = data["simulations"]
        sim = next((s for s in simulations if s["simulation_id"] == "integration_test_sim"), None)
        assert sim is not None
        assert sim["total_generations"] == 1
        assert sim["latest_generation"] == 10
        assert sim["status"] == "active"
    
    @patch('utils.auth.verify_api_key')
    def test_get_simulation_metrics_endpoint(self, mock_auth, client, sample_result_metrics):
        """Test getting simulation metrics endpoint."""
        mock_auth.return_value = "test-api-key"
        
        # Collect metrics first
        client.post(
            "/api/results/collect",
            json=sample_result_metrics,
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        # Get metrics
        response = client.get(
            "/api/results/simulations/integration_test_sim/metrics",
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["simulation_id"] == "integration_test_sim"
        assert data["count"] == 1
        assert len(data["metrics"]) == 1
        
        metrics = data["metrics"][0]
        assert metrics["generation"] == 10
        assert metrics["population_size"] == 1000
        assert metrics["resistant_count"] == 150
    
    @patch('utils.auth.verify_api_key')
    def test_aggregated_results_endpoint(self, mock_auth, client):
        """Test aggregated results endpoint."""
        mock_auth.return_value = "test-api-key"
        
        # Collect multiple generations of metrics
        for generation in range(5):
            metrics = {
                "simulation_id": "aggregation_test",
                "generation": generation,
                "timestamp": (datetime.now() + timedelta(seconds=generation)).isoformat(),
                "population_size": 1000 - generation * 10,
                "resistant_count": generation * 20,
                "sensitive_count": 1000 - generation * 30,
                "average_fitness": 1.0 + generation * 0.1,
                "fitness_std": 0.2,
                "mutation_count": generation * 5,
                "extinction_occurred": False,
                "diversity_index": 0.8 - generation * 0.05,
                "selection_pressure": 0.5,
                "mutation_rate": 1e-6,
                "elapsed_time": 1.0 + generation * 0.2,
                "memory_usage": 100.0 + generation * 5,
                "cpu_usage": 50.0 + generation * 2
            }
            
            client.post(
                "/api/results/collect",
                json=metrics,
                headers={"Authorization": "Bearer test-api-key"}
            )
        
        # Get aggregated results
        response = client.get(
            "/api/results/simulations/aggregation_test/aggregate",
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["simulation_id"] == "aggregation_test"
        
        aggregated = data["aggregated_results"]
        assert aggregated["total_generations"] == 5
        assert aggregated["final_population_size"] == 960  # 1000 - 4 * 10
        assert len(aggregated["average_fitness_trend"]) == 5
        assert "performance_stats" in aggregated
        assert "summary_statistics" in aggregated
    
    @patch('utils.auth.verify_api_key')
    def test_statistical_analysis_endpoint(self, mock_auth, client):
        """Test statistical analysis endpoint."""
        mock_auth.return_value = "test-api-key"
        
        # Collect metrics with varying data
        for generation in range(10):
            metrics = {
                "simulation_id": "stats_test",
                "generation": generation,
                "timestamp": (datetime.now() + timedelta(seconds=generation)).isoformat(),
                "population_size": 1000 + generation * 5,  # Increasing trend
                "resistant_count": generation * 15,  # Linear growth
                "sensitive_count": 1000 - generation * 10,
                "average_fitness": 1.0 + generation * 0.05,  # Fitness improvement
                "fitness_std": 0.2 + generation * 0.01,
                "mutation_count": generation * 3,
                "extinction_occurred": False,
                "diversity_index": 0.8 - generation * 0.02,  # Decreasing diversity
                "selection_pressure": 0.5 + generation * 0.02,
                "mutation_rate": 1e-6,
                "elapsed_time": 1.0,
                "memory_usage": 100.0,
                "cpu_usage": 50.0
            }
            
            client.post(
                "/api/results/collect",
                json=metrics,
                headers={"Authorization": "Bearer test-api-key"}
            )
        
        # Get statistical analysis
        response = client.get(
            "/api/results/simulations/stats_test/analysis",
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["simulation_id"] == "stats_test"
        
        analysis = data["statistical_analysis"]
        assert "population_size" in analysis
        assert "average_fitness" in analysis
        assert "resistant_count" in analysis
        
        # Check trend analysis for population size (should be increasing)
        pop_trend = analysis["population_size"]["trend"]
        assert pop_trend["direction"] == "increasing"
        assert pop_trend["slope"] > 0
        
        # Check fitness trend (should be increasing)
        fitness_trend = analysis["average_fitness"]["trend"]
        assert fitness_trend["direction"] == "increasing"
    
    @patch('utils.auth.verify_api_key')
    def test_report_generation_endpoint(self, mock_auth, client):
        """Test report generation endpoint."""
        mock_auth.return_value = "test-api-key"
        
        # Collect comprehensive metrics
        for generation in range(15):
            metrics = {
                "simulation_id": "report_test",
                "generation": generation,
                "timestamp": (datetime.now() + timedelta(seconds=generation * 10)).isoformat(),
                "population_size": max(10, 1000 - generation * 30),
                "resistant_count": min(1000, generation * 25),
                "sensitive_count": max(0, 1000 - generation * 55),
                "average_fitness": 1.0 + generation * 0.03,
                "fitness_std": 0.2 + generation * 0.005,
                "mutation_count": generation * 4,
                "extinction_occurred": generation > 12,
                "diversity_index": max(0.1, 0.8 - generation * 0.04),
                "selection_pressure": 0.5 + generation * 0.03,
                "mutation_rate": 1e-6 * (1 + generation * 0.1),
                "elapsed_time": 1.0 + generation * 0.1,
                "memory_usage": 100.0 + generation * 3,
                "cpu_usage": 50.0 + generation * 1.5
            }
            
            client.post(
                "/api/results/collect",
                json=metrics,
                headers={"Authorization": "Bearer test-api-key"}
            )
        
        # Generate report with plots
        response = client.get(
            "/api/results/simulations/report_test/report?include_plots=true",
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["simulation_id"] == "report_test"
        
        report = data["report"]
        assert "generated_at" in report
        assert "summary" in report
        assert "performance" in report
        assert "statistical_analysis" in report
        assert "trends" in report
        assert "plot_data" in report
        
        # Check plot data
        plot_data = report["plot_data"]
        assert "generations" in plot_data
        assert "population_sizes" in plot_data
        assert "fitness_values" in plot_data
        assert len(plot_data["generations"]) == 15
    
    @patch('utils.auth.verify_api_key')
    def test_simulation_comparison_endpoint(self, mock_auth, client):
        """Test simulation comparison endpoint."""
        mock_auth.return_value = "test-api-key"
        
        # Create two different simulation scenarios
        scenarios = [
            {"id": "compare_sim_1", "resistance_rate": 0.1, "fitness_boost": 0.02},
            {"id": "compare_sim_2", "resistance_rate": 0.3, "fitness_boost": 0.05}
        ]
        
        for scenario in scenarios:
            for generation in range(8):
                metrics = {
                    "simulation_id": scenario["id"],
                    "generation": generation,
                    "timestamp": (datetime.now() + timedelta(seconds=generation)).isoformat(),
                    "population_size": 1000 - generation * 20,
                    "resistant_count": int(generation * 100 * scenario["resistance_rate"]),
                    "sensitive_count": 1000 - int(generation * 120 * scenario["resistance_rate"]),
                    "average_fitness": 1.0 + generation * scenario["fitness_boost"],
                    "fitness_std": 0.2,
                    "mutation_count": generation * 6,
                    "extinction_occurred": False,
                    "diversity_index": 0.8 - generation * 0.03,
                    "selection_pressure": 0.5,
                    "mutation_rate": 1e-6,
                    "elapsed_time": 1.0,
                    "memory_usage": 100.0,
                    "cpu_usage": 50.0
                }
                
                client.post(
                    "/api/results/collect",
                    json=metrics,
                    headers={"Authorization": "Bearer test-api-key"}
                )
        
        # Compare simulations
        response = client.post(
            "/api/results/compare",
            json=["compare_sim_1", "compare_sim_2"],
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["simulation_ids"]) == 2
        
        comparison = data["comparison"]
        assert comparison["simulation_count"] == 2
        assert "final_population_comparison" in comparison
        assert "resistance_development" in comparison
        assert "performance_comparison" in comparison
        
        # Verify different outcomes
        final_pops = comparison["final_population_comparison"]["values"]
        assert len(final_pops) == 2
        assert "compare_sim_1" in final_pops
        assert "compare_sim_2" in final_pops
    
    @patch('utils.auth.verify_api_key')
    def test_export_functionality(self, mock_auth, client):
        """Test metrics export functionality."""
        mock_auth.return_value = "test-api-key"
        
        # Collect metrics for export
        for generation in range(5):
            metrics = {
                "simulation_id": "export_test",
                "generation": generation,
                "timestamp": (datetime.now() + timedelta(seconds=generation)).isoformat(),
                "population_size": 1000,
                "resistant_count": 100,
                "sensitive_count": 900,
                "average_fitness": 1.0,
                "fitness_std": 0.2,
                "mutation_count": 5,
                "extinction_occurred": False,
                "diversity_index": 0.8,
                "selection_pressure": 0.5,
                "mutation_rate": 1e-6,
                "elapsed_time": 1.0,
                "memory_usage": 100.0,
                "cpu_usage": 50.0
            }
            
            client.post(
                "/api/results/collect",
                json=metrics,
                headers={"Authorization": "Bearer test-api-key"}
            )
        
        # Test JSON export
        response = client.post(
            "/api/results/simulations/export_test/export?format_type=json",
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Test CSV export
        response = client.post(
            "/api/results/simulations/export_test/export?format_type=csv",
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
    
    @patch('utils.auth.verify_api_key')
    def test_batch_collection(self, mock_auth, client):
        """Test batch metrics collection."""
        mock_auth.return_value = "test-api-key"
        
        # Prepare batch metrics
        batch_metrics = []
        for generation in range(3):
            metrics = {
                "simulation_id": "batch_test",
                "generation": generation,
                "timestamp": (datetime.now() + timedelta(seconds=generation)).isoformat(),
                "population_size": 1000,
                "resistant_count": 100,
                "sensitive_count": 900,
                "average_fitness": 1.0,
                "fitness_std": 0.2,
                "mutation_count": 5,
                "extinction_occurred": False,
                "diversity_index": 0.8,
                "selection_pressure": 0.5,
                "mutation_rate": 1e-6,
                "elapsed_time": 1.0,
                "memory_usage": 100.0,
                "cpu_usage": 50.0
            }
            batch_metrics.append(metrics)
        
        # Submit batch
        response = client.post(
            "/api/results/batch/collect",
            json=batch_metrics,
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["collected_count"] == 3
        assert data["total_count"] == 3
        assert data["error_count"] == 0
    
    @patch('utils.auth.verify_api_key')
    def test_storage_info_endpoint(self, mock_auth, client):
        """Test storage information endpoint."""
        mock_auth.return_value = "test-api-key"
        
        response = client.get(
            "/api/results/storage/info",
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "storage_path" in data
        assert "file_count" in data
        assert "total_size_bytes" in data
        assert "active_simulations" in data
    
    @patch('utils.auth.verify_api_key')
    def test_metrics_validation(self, mock_auth, client):
        """Test metrics format validation."""
        mock_auth.return_value = "test-api-key"
        
        # Valid metrics
        valid_metrics = {
            "simulation_id": "validation_test",
            "generation": 5,
            "timestamp": datetime.now().isoformat(),
            "population_size": 1000,
            "resistant_count": 100,
            "sensitive_count": 900,
            "average_fitness": 1.0,
            "fitness_std": 0.2,
            "mutation_count": 5,
            "extinction_occurred": False,
            "diversity_index": 0.8,
            "selection_pressure": 0.5,
            "mutation_rate": 1e-6,
            "elapsed_time": 1.0,
            "memory_usage": 100.0,
            "cpu_usage": 50.0
        }
        
        response = client.post(
            "/api/results/utilities/validate-metrics",
            json=valid_metrics,
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "parsed_metrics" in data
        
        # Invalid metrics (missing field)
        invalid_metrics = valid_metrics.copy()
        del invalid_metrics["population_size"]
        
        response = client.post(
            "/api/results/utilities/validate-metrics",
            json=invalid_metrics,
            headers={"Authorization": "Bearer test-api-key"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert "Missing required field" in data["error"]
    
    def test_error_handling(self, client):
        """Test error handling in result collection endpoints."""
        # Test without authentication
        response = client.get("/api/results/simulations")
        assert response.status_code == 403  # Unauthorized
        
        # Test with non-existent simulation
        with patch('utils.auth.verify_api_key', return_value="test-api-key"):
            response = client.get(
                "/api/results/simulations/nonexistent_sim/metrics",
                headers={"Authorization": "Bearer test-api-key"}
            )
            assert response.status_code == 404


class TestSimulationEngineIntegration:
    """Integration tests for result collection with the simulation engine."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_simulation_with_result_collection(self, temp_storage):
        """Test simulation engine integration with result collection."""
        # Initialize result collection components
        collector = ResultCollector(temp_storage)
        analyzer = ResultAnalyzer(collector)
        
        # Track collected metrics
        collected_metrics = []
        
        def metrics_callback(metrics):
            collected_metrics.append(metrics)
        
        collector.add_subscriber(metrics_callback)
        
        # Simulate a basic simulation run with result collection
        simulation_id = "engine_integration_test"
        
        # Simulate 5 generations
        for generation in range(5):
            # Simulate typical simulation metrics
            metrics = ResultMetrics(
                simulation_id=simulation_id,
                generation=generation,
                timestamp=datetime.now(),
                population_size=1000 - generation * 50,
                resistant_count=generation * 30,
                sensitive_count=1000 - generation * 80,
                average_fitness=1.0 + generation * 0.05,
                fitness_std=0.2 + generation * 0.01,
                mutation_count=generation * 8,
                extinction_occurred=False,
                diversity_index=0.8 - generation * 0.08,
                selection_pressure=0.5 + generation * 0.05,
                mutation_rate=1e-6 * (1 + generation * 0.2),
                elapsed_time=1.0 + generation * 0.3,
                memory_usage=100.0 + generation * 10,
                cpu_usage=50.0 + generation * 5
            )
            
            collector.collect_metrics(metrics)
        
        # Verify collection worked
        assert len(collected_metrics) == 5
        assert all(m.simulation_id == simulation_id for m in collected_metrics)
        
        # Test analysis
        aggregated = analyzer.aggregate_results(simulation_id)
        assert aggregated.total_generations == 5
        assert aggregated.simulation_id == simulation_id
        
        # Test statistical analysis
        stats = analyzer.statistical_analysis(simulation_id)
        assert "population_size" in stats
        assert "average_fitness" in stats
        
        # Test file export
        json_path = collector.save_metrics(simulation_id, ResultFormat.JSON)
        assert json_path.exists()
        
        # Verify file content
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        assert len(data) == 5
        assert data[0]["simulation_id"] == simulation_id
    
    @pytest.mark.asyncio
    async def test_streaming_integration(self, temp_storage):
        """Test streaming result collection integration."""
        streaming_collector = StreamingResultCollector(temp_storage, stream_interval=0.1)
        
        # Track streamed data
        streamed_batches = []
        
        def stream_callback(metrics_list):
            streamed_batches.append(metrics_list)
        
        simulation_id = "streaming_integration_test"
        
        # Start streaming
        await streaming_collector.start_streaming(simulation_id, stream_callback)
        
        # Simulate metrics collection during streaming
        for generation in range(10):
            metrics = ResultMetrics(
                simulation_id=simulation_id,
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
            
            streaming_collector.collect_metrics(metrics)
            await asyncio.sleep(0.05)  # Small delay to allow streaming
        
        # Wait for streaming to process
        await asyncio.sleep(0.5)
        
        # Stop streaming
        await streaming_collector.stop_streaming(simulation_id)
        
        # Verify streaming worked
        assert len(streamed_batches) > 0
        total_streamed = sum(len(batch) for batch in streamed_batches)
        assert total_streamed > 0  # Should have streamed some data
        
        # Verify collected data
        collected = streaming_collector.get_metrics(simulation_id)
        assert len(collected) == 10


if __name__ == "__main__":
    pytest.main([__file__]) 