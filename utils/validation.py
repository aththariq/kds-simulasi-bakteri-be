"""
Custom validation utilities for simulation parameters.
"""

from typing import Any, Dict
from pydantic import validator
from config import settings


def validate_population_size(value: int) -> int:
    """
    Validate population size is within acceptable limits.
    
    Args:
        value: Population size to validate
        
    Returns:
        Validated population size
        
    Raises:
        ValueError: If population size is invalid
    """
    if value < 10:
        raise ValueError("Population size must be at least 10")
    if value > settings.max_population_size:
        raise ValueError(f"Population size cannot exceed {settings.max_population_size}")
    return value


def validate_mutation_rate(value: float) -> float:
    """
    Validate mutation rate is within biological realistic ranges.
    
    Args:
        value: Mutation rate to validate
        
    Returns:
        Validated mutation rate
        
    Raises:
        ValueError: If mutation rate is invalid
    """
    if value < 0.0:
        raise ValueError("Mutation rate cannot be negative")
    if value > 1.0:
        raise ValueError("Mutation rate cannot exceed 1.0 (100%)")
    if value > 0.1:
        # Warning for unrealistically high mutation rates
        import warnings
        warnings.warn(
            f"Mutation rate {value} is very high. "
            "Typical bacterial mutation rates are 10^-6 to 10^-3",
            UserWarning
        )
    return value


def validate_selection_pressure(value: float) -> float:
    """
    Validate selection pressure is within reasonable ranges.
    
    Args:
        value: Selection pressure to validate
        
    Returns:
        Validated selection pressure
        
    Raises:
        ValueError: If selection pressure is invalid
    """
    if value < 0.0:
        raise ValueError("Selection pressure cannot be negative")
    if value > 1.0:
        raise ValueError("Selection pressure cannot exceed 1.0")
    return value


def validate_simulation_time(value: int) -> int:
    """
    Validate simulation time is within acceptable limits.
    
    Args:
        value: Simulation time to validate
        
    Returns:
        Validated simulation time
        
    Raises:
        ValueError: If simulation time is invalid
    """
    if value < 1:
        raise ValueError("Simulation time must be at least 1 generation")
    if value > settings.max_simulation_time:
        raise ValueError(f"Simulation time cannot exceed {settings.max_simulation_time} generations")
    return value


def validate_antibiotic_concentration(value: float) -> float:
    """
    Validate antibiotic concentration is positive.
    
    Args:
        value: Antibiotic concentration to validate
        
    Returns:
        Validated antibiotic concentration
        
    Raises:
        ValueError: If concentration is invalid
    """
    if value < 0.0:
        raise ValueError("Antibiotic concentration cannot be negative")
    return value


def validate_simulation_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a complete set of simulation parameters.
    
    Args:
        parameters: Dictionary of simulation parameters
        
    Returns:
        Validated parameters dictionary
        
    Raises:
        ValueError: If any parameter is invalid
    """
    validated = {}
    
    if "initial_population_size" in parameters:
        validated["initial_population_size"] = validate_population_size(
            parameters["initial_population_size"]
        )
    
    if "mutation_rate" in parameters:
        validated["mutation_rate"] = validate_mutation_rate(
            parameters["mutation_rate"]
        )
    
    if "selection_pressure" in parameters:
        validated["selection_pressure"] = validate_selection_pressure(
            parameters["selection_pressure"]
        )
    
    if "simulation_time" in parameters:
        validated["simulation_time"] = validate_simulation_time(
            parameters["simulation_time"]
        )
    
    if "antibiotic_concentration" in parameters:
        validated["antibiotic_concentration"] = validate_antibiotic_concentration(
            parameters["antibiotic_concentration"]
        )
    
    return validated 