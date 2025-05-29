"""
Tests for validation utilities and Pydantic schemas.
"""

import pytest
from pydantic import ValidationError
from utils.validation import (
    validate_population_size,
    validate_mutation_rate,
    validate_selection_pressure,
    validate_simulation_time,
    validate_antibiotic_concentration,
    validate_simulation_parameters
)
from schemas.simulation import SimulationCreateRequest, SimulationResults


class TestValidationUtilities:
    """Test custom validation utility functions."""
    
    def test_validate_population_size_valid(self):
        """Test population size validation with valid values."""
        assert validate_population_size(100) == 100
        assert validate_population_size(1000) == 1000
        assert validate_population_size(50000) == 50000
    
    def test_validate_population_size_too_small(self):
        """Test population size validation with too small values."""
        with pytest.raises(ValueError, match="Population size must be at least 10"):
            validate_population_size(5)
        
        with pytest.raises(ValueError, match="Population size must be at least 10"):
            validate_population_size(0)
    
    def test_validate_population_size_too_large(self):
        """Test population size validation with too large values."""
        with pytest.raises(ValueError, match="Population size cannot exceed"):
            validate_population_size(200000)  # Exceeds max_population_size of 100000
    
    def test_validate_mutation_rate_valid(self):
        """Test mutation rate validation with valid values."""
        assert validate_mutation_rate(0.0) == 0.0
        assert validate_mutation_rate(0.001) == 0.001
        assert validate_mutation_rate(0.01) == 0.01
        assert validate_mutation_rate(1.0) == 1.0
    
    def test_validate_mutation_rate_negative(self):
        """Test mutation rate validation with negative values."""
        with pytest.raises(ValueError, match="Mutation rate cannot be negative"):
            validate_mutation_rate(-0.1)
    
    def test_validate_mutation_rate_too_high(self):
        """Test mutation rate validation with values exceeding 1.0."""
        with pytest.raises(ValueError, match="Mutation rate cannot exceed 1.0"):
            validate_mutation_rate(1.5)
    
    def test_validate_mutation_rate_warning(self):
        """Test mutation rate validation with high values that trigger warnings."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_mutation_rate(0.15)
            assert result == 0.15
            assert len(w) == 1
            assert "very high" in str(w[0].message)
    
    def test_validate_selection_pressure_valid(self):
        """Test selection pressure validation with valid values."""
        assert validate_selection_pressure(0.0) == 0.0
        assert validate_selection_pressure(0.1) == 0.1
        assert validate_selection_pressure(0.5) == 0.5
        assert validate_selection_pressure(1.0) == 1.0
    
    def test_validate_selection_pressure_invalid(self):
        """Test selection pressure validation with invalid values."""
        with pytest.raises(ValueError, match="Selection pressure cannot be negative"):
            validate_selection_pressure(-0.1)
        
        with pytest.raises(ValueError, match="Selection pressure cannot exceed 1.0"):
            validate_selection_pressure(1.5)
    
    def test_validate_simulation_time_valid(self):
        """Test simulation time validation with valid values."""
        assert validate_simulation_time(1) == 1
        assert validate_simulation_time(100) == 100
        assert validate_simulation_time(1000) == 1000
    
    def test_validate_simulation_time_invalid(self):
        """Test simulation time validation with invalid values."""
        with pytest.raises(ValueError, match="Simulation time must be at least 1"):
            validate_simulation_time(0)
        
        with pytest.raises(ValueError, match="Simulation time cannot exceed"):
            validate_simulation_time(2000)  # Exceeds max_simulation_time of 1000
    
    def test_validate_antibiotic_concentration_valid(self):
        """Test antibiotic concentration validation with valid values."""
        assert validate_antibiotic_concentration(0.0) == 0.0
        assert validate_antibiotic_concentration(1.0) == 1.0
        assert validate_antibiotic_concentration(10.0) == 10.0
    
    def test_validate_antibiotic_concentration_negative(self):
        """Test antibiotic concentration validation with negative values."""
        with pytest.raises(ValueError, match="Antibiotic concentration cannot be negative"):
            validate_antibiotic_concentration(-1.0)
    
    def test_validate_simulation_parameters_complete(self):
        """Test complete parameter validation with valid parameters."""
        params = {
            "initial_population_size": 1000,
            "mutation_rate": 0.001,
            "selection_pressure": 0.1,
            "simulation_time": 100,
            "antibiotic_concentration": 1.0
        }
        
        validated = validate_simulation_parameters(params)
        
        assert validated == params
    
    def test_validate_simulation_parameters_partial(self):
        """Test parameter validation with partial parameters."""
        params = {
            "initial_population_size": 500,
            "mutation_rate": 0.01
        }
        
        validated = validate_simulation_parameters(params)
        
        assert len(validated) == 2
        assert validated["initial_population_size"] == 500
        assert validated["mutation_rate"] == 0.01
    
    def test_validate_simulation_parameters_invalid(self):
        """Test parameter validation with invalid parameters."""
        params = {
            "initial_population_size": 5,  # Too small
            "mutation_rate": -0.1  # Negative
        }
        
        with pytest.raises(ValueError):
            validate_simulation_parameters(params)


class TestPydanticSchemas:
    """Test Pydantic schema validation."""
    
    def test_simulation_create_request_valid(self):
        """Test SimulationCreateRequest with valid data."""
        data = {
            "initial_population_size": 1000,
            "mutation_rate": 0.001,
            "selection_pressure": 0.1,
            "antibiotic_concentration": 1.0,
            "simulation_time": 100
        }
        
        request = SimulationCreateRequest(**data)
        
        assert request.initial_population_size == 1000
        assert request.mutation_rate == 0.001
        assert request.selection_pressure == 0.1
        assert request.antibiotic_concentration == 1.0
        assert request.simulation_time == 100
    
    def test_simulation_create_request_defaults(self):
        """Test SimulationCreateRequest with default values."""
        request = SimulationCreateRequest()
        
        assert request.initial_population_size == 1000
        assert request.mutation_rate == 0.001
        assert request.selection_pressure == 0.1
        assert request.antibiotic_concentration == 1.0
        assert request.simulation_time == 100
    
    def test_simulation_create_request_invalid_population(self):
        """Test SimulationCreateRequest with invalid population size."""
        with pytest.raises(ValidationError) as exc_info:
            SimulationCreateRequest(initial_population_size=5)
        
        errors = exc_info.value.errors()
        assert any("Population size must be at least 10" in str(error) for error in errors)
    
    def test_simulation_create_request_invalid_mutation_rate(self):
        """Test SimulationCreateRequest with invalid mutation rate."""
        with pytest.raises(ValidationError) as exc_info:
            SimulationCreateRequest(mutation_rate=-0.1)
        
        errors = exc_info.value.errors()
        assert any("Mutation rate cannot be negative" in str(error) for error in errors)
    
    def test_simulation_create_request_high_mutation_selection_combo(self):
        """Test SimulationCreateRequest with unrealistic mutation/selection combination."""
        with pytest.raises(ValidationError) as exc_info:
            SimulationCreateRequest(mutation_rate=0.02, selection_pressure=0.9)
        
        errors = exc_info.value.errors()
        assert any("Very high mutation rate with very high selection pressure" in str(error) for error in errors)
    
    def test_simulation_results_valid(self):
        """Test SimulationResults with valid data."""
        data = {
            "population_history": [1000, 950, 900],
            "resistance_history": [0.1, 0.15, 0.2],
            "fitness_history": [0.8, 0.75, 0.7]
        }
        
        results = SimulationResults(**data)
        
        assert results.population_history == [1000, 950, 900]
        assert results.resistance_history == [0.1, 0.15, 0.2]
        assert results.fitness_history == [0.8, 0.75, 0.7]
    
    def test_simulation_results_empty(self):
        """Test SimulationResults with empty data."""
        data = {
            "population_history": [],
            "resistance_history": [],
            "fitness_history": []
        }
        
        results = SimulationResults(**data)
        
        assert results.population_history == []
        assert results.resistance_history == []
        assert results.fitness_history == []
    
    def test_simulation_results_invalid_population(self):
        """Test SimulationResults with invalid population data."""
        with pytest.raises(ValidationError) as exc_info:
            SimulationResults(
                population_history=[1000, -50, 900],  # Negative population
                resistance_history=[0.1, 0.15, 0.2],
                fitness_history=[0.8, 0.75, 0.7]
            )
        
        errors = exc_info.value.errors()
        assert any("Population history cannot contain negative values" in str(error) for error in errors)
    
    def test_simulation_results_invalid_resistance(self):
        """Test SimulationResults with invalid resistance data."""
        with pytest.raises(ValidationError) as exc_info:
            SimulationResults(
                population_history=[1000, 950, 900],
                resistance_history=[0.1, 1.5, 0.2],  # Resistance > 1.0
                fitness_history=[0.8, 0.75, 0.7]
            )
        
        errors = exc_info.value.errors()
        assert any("Resistance values must be between 0 and 1" in str(error) for error in errors)
    
    def test_simulation_results_invalid_fitness(self):
        """Test SimulationResults with invalid fitness data."""
        with pytest.raises(ValidationError) as exc_info:
            SimulationResults(
                population_history=[1000, 950, 900],
                resistance_history=[0.1, 0.15, 0.2],
                fitness_history=[0.8, -0.1, 0.7]  # Negative fitness
            )
        
        errors = exc_info.value.errors()
        assert any("Fitness values cannot be negative" in str(error) for error in errors)


class TestValidationIntegration:
    """Test integration between validation utilities and schemas."""
    
    def test_schema_uses_custom_validators(self):
        """Test that Pydantic schemas use custom validation functions."""
        # This should trigger the custom population size validator
        with pytest.raises(ValidationError) as exc_info:
            SimulationCreateRequest(initial_population_size=5)
        
        # Check that the error message comes from our custom validator
        errors = exc_info.value.errors()
        assert any("Population size must be at least 10" in str(error) for error in errors)
    
    def test_field_constraints_and_custom_validators(self):
        """Test that both Pydantic field constraints and custom validators work."""
        # Field constraint should catch this (ge=10)
        with pytest.raises(ValidationError):
            SimulationCreateRequest(initial_population_size=5)
        
        # Custom validator should catch this
        with pytest.raises(ValidationError):
            SimulationCreateRequest(initial_population_size=200000)
    
    def test_cross_field_validation(self):
        """Test cross-field validation between mutation rate and selection pressure."""
        # This should pass
        request1 = SimulationCreateRequest(mutation_rate=0.001, selection_pressure=0.9)
        assert request1.mutation_rate == 0.001
        
        # This should fail due to cross-field validation
        with pytest.raises(ValidationError) as exc_info:
            SimulationCreateRequest(mutation_rate=0.05, selection_pressure=0.95)
        
        errors = exc_info.value.errors()
        assert any("Very high mutation rate with very high selection pressure" in str(error) for error in errors) 