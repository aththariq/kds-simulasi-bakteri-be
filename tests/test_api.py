"""
Tests for API endpoints.
"""

import pytest
import uuid
from fastapi.testclient import TestClient
from main import app
from services.simulation_service import SimulationService


class TestRootEndpoints:
    """Test root and health endpoints."""
    
    def test_root_endpoint(self, client):
        """Test the root endpoint returns correct information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Bacterial Resistance Simulation API"
        assert "version" in data
        assert data["status"] == "running"
        assert data["docs_url"] == "/docs"
        assert data["redoc_url"] == "/redoc"
    
    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestSimulationEndpoints:
    """Test simulation API endpoints."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Clear any existing simulations
        from routes.simulation import simulation_service
        simulation_service.active_simulations.clear()
    
    def test_create_simulation_success(self, client):
        """Test successful simulation creation."""
        payload = {
            "initial_population_size": 1000,
            "mutation_rate": 0.001,
            "selection_pressure": 0.1,
            "antibiotic_concentration": 1.0,
            "simulation_time": 100
        }
        
        response = client.post("/api/simulations/", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "simulation_id" in data
        assert data["status"] == "initialized"
        assert data["parameters"]["initial_population_size"] == 1000
        assert data["parameters"]["mutation_rate"] == 0.001
    
    def test_create_simulation_with_defaults(self, client):
        """Test simulation creation with default parameters."""
        payload = {}
        
        response = client.post("/api/simulations/", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "initialized"
        assert data["parameters"]["initial_population_size"] == 1000  # default
        assert data["parameters"]["mutation_rate"] == 0.001  # default
    
    def test_create_simulation_validation_error(self, client):
        """Test simulation creation with invalid parameters."""
        payload = {
            "initial_population_size": 5,  # Too small
            "mutation_rate": -0.1,  # Negative
            "selection_pressure": 1.5,  # Too high
            "simulation_time": 0  # Too small
        }
        
        response = client.post("/api/simulations/", json=payload)
        
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
    
    def test_list_simulations_empty(self, client):
        """Test listing simulations when none exist."""
        response = client.get("/api/simulations/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["active_simulations"] == 0
        assert data["simulations"] == []
    
    def test_list_simulations_with_data(self, client):
        """Test listing simulations with existing data."""
        # Create a simulation first
        payload = {"initial_population_size": 500}
        create_response = client.post("/api/simulations/", json=payload)
        assert create_response.status_code == 201
        
        # List simulations
        response = client.get("/api/simulations/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["active_simulations"] == 1
        assert len(data["simulations"]) == 1
        assert data["simulations"][0]["status"] == "initialized"
    
    def test_get_simulation_status_success(self, client):
        """Test getting simulation status for existing simulation."""
        # Create a simulation first
        payload = {"initial_population_size": 500}
        create_response = client.post("/api/simulations/", json=payload)
        simulation_id = create_response.json()["simulation_id"]
        
        # Get status
        response = client.get(f"/api/simulations/{simulation_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["simulation_id"] == simulation_id
        assert data["status"] == "initialized"
        assert data["current_generation"] == 0
    
    def test_get_simulation_status_not_found(self, client):
        """Test getting simulation status for non-existent simulation."""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/simulations/{fake_id}")
        
        assert response.status_code == 404
        error_data = response.json()
        assert "detail" in error_data
    
    def test_run_simulation_success(self, client):
        """Test running a simulation successfully."""
        # Create a simulation first with small parameters for quick testing
        payload = {
            "initial_population_size": 100,
            "simulation_time": 5  # Very short simulation
        }
        create_response = client.post("/api/simulations/", json=payload)
        simulation_id = create_response.json()["simulation_id"]
        
        # Run the simulation
        response = client.post(f"/api/simulations/{simulation_id}/run")
        
        assert response.status_code == 200
        data = response.json()
        assert data["simulation_id"] == simulation_id
        assert data["status"] == "completed"
        assert data["generations_completed"] == 5
        assert "final_population_size" in data
        assert "results" in data
    
    def test_run_simulation_not_found(self, client):
        """Test running a non-existent simulation."""
        fake_id = str(uuid.uuid4())
        response = client.post(f"/api/simulations/{fake_id}/run")
        
        assert response.status_code == 404
        error_data = response.json()
        assert "detail" in error_data
    
    def test_delete_simulation_success(self, client):
        """Test deleting a simulation successfully."""
        # Create a simulation first
        payload = {"initial_population_size": 500}
        create_response = client.post("/api/simulations/", json=payload)
        simulation_id = create_response.json()["simulation_id"]
        
        # Delete the simulation
        response = client.delete(f"/api/simulations/{simulation_id}")
        
        assert response.status_code == 204
        
        # Verify it's gone
        get_response = client.get(f"/api/simulations/{simulation_id}")
        assert get_response.status_code == 404
    
    def test_delete_simulation_not_found(self, client):
        """Test deleting a non-existent simulation."""
        fake_id = str(uuid.uuid4())
        response = client.delete(f"/api/simulations/{fake_id}")
        
        assert response.status_code == 404
        error_data = response.json()
        assert "detail" in error_data


class TestSimulationWorkflow:
    """Test complete simulation workflow."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Clear any existing simulations
        from routes.simulation import simulation_service
        simulation_service.active_simulations.clear()
    
    def test_complete_simulation_workflow(self):
        """Test the complete simulation workflow from creation to deletion."""
        # 1. Create simulation
        payload = {
            "initial_population_size": 100,
            "mutation_rate": 0.01,
            "selection_pressure": 0.2,
            "simulation_time": 10
        }
        
        create_response = client.post("/api/simulations/", json=payload)
        assert create_response.status_code == 201
        simulation_id = create_response.json()["simulation_id"]
        
        # 2. Check initial status
        status_response = client.get(f"/api/simulations/{simulation_id}")
        assert status_response.status_code == 200
        assert status_response.json()["status"] == "initialized"
        
        # 3. Run simulation
        run_response = client.post(f"/api/simulations/{simulation_id}/run")
        assert run_response.status_code == 200
        run_data = run_response.json()
        assert run_data["status"] == "completed"
        assert len(run_data["results"]["population_history"]) == 10
        
        # 4. Check final status
        final_status_response = client.get(f"/api/simulations/{simulation_id}")
        assert final_status_response.status_code == 200
        final_data = final_status_response.json()
        assert final_data["status"] == "completed"
        
        # 5. Verify in list
        list_response = client.get("/api/simulations/")
        assert list_response.status_code == 200
        assert list_response.json()["active_simulations"] == 1
        
        # 6. Delete simulation
        delete_response = client.delete(f"/api/simulations/{simulation_id}")
        assert delete_response.status_code == 204
        
        # 7. Verify empty list
        final_list_response = client.get("/api/simulations/")
        assert final_list_response.status_code == 200
        assert final_list_response.json()["active_simulations"] == 0 