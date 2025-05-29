"""
Pytest fixtures for API and service testing.
"""

import pytest
from fastapi.testclient import TestClient
from main import app
from services.simulation_service import SimulationService


@pytest.fixture
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.fixture
def simulation_service():
    """Clean simulation service fixture."""
    service = SimulationService()
    service.active_simulations.clear()
    return service


@pytest.fixture
def sample_simulation_params():
    """Sample simulation parameters for testing."""
    return {
        "initial_population_size": 1000,
        "mutation_rate": 0.001,
        "selection_pressure": 0.1,
        "antibiotic_concentration": 1.0,
        "simulation_time": 100
    }


@pytest.fixture
def quick_simulation_params():
    """Quick simulation parameters for fast testing."""
    return {
        "initial_population_size": 100,
        "mutation_rate": 0.01,
        "selection_pressure": 0.2,
        "antibiotic_concentration": 1.0,
        "simulation_time": 5
    }


@pytest.fixture
def invalid_simulation_params():
    """Invalid simulation parameters for error testing."""
    return {
        "initial_population_size": 5,  # Too small
        "mutation_rate": -0.1,  # Negative
        "selection_pressure": 1.5,  # Too high
        "antibiotic_concentration": -1.0,  # Negative
        "simulation_time": 0  # Too small
    }


@pytest.fixture
def created_simulation(client, quick_simulation_params):
    """Fixture that creates a simulation and returns its ID."""
    response = client.post("/api/simulations/", json=quick_simulation_params)
    assert response.status_code == 201
    return response.json()["simulation_id"]


@pytest.fixture
def completed_simulation(client, created_simulation):
    """Fixture that creates and runs a simulation to completion."""
    # Run the simulation
    response = client.post(f"/api/simulations/{created_simulation}/run")
    assert response.status_code == 200
    return created_simulation


@pytest.fixture(autouse=True)
def clean_simulations():
    """Automatically clean up simulations before each test."""
    from routes.simulation import simulation_service
    simulation_service.active_simulations.clear()
    yield
    simulation_service.active_simulations.clear() 