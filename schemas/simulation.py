"""
Pydantic schemas for simulation API requests and responses.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from utils.validation import (
    validate_population_size,
    validate_mutation_rate,
    validate_selection_pressure,
    validate_simulation_time,
    validate_antibiotic_concentration
)


class SimulationCreateRequest(BaseModel):
    """Request model for creating a new simulation."""
    
    initial_population_size: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="Starting population size (10-100,000)"
    )
    mutation_rate: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Mutation rate per generation (0.0-1.0)"
    )
    selection_pressure: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Selection pressure strength (0.0-1.0)"
    )
    antibiotic_concentration: float = Field(
        default=1.0,
        ge=0.0,
        description="Antibiotic concentration"
    )
    simulation_time: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of generations to simulate (1-1,000)"
    )
    
    # Custom validators
    @validator('initial_population_size')
    def validate_pop_size(cls, v):
        return validate_population_size(v)
    
    @validator('mutation_rate')
    def validate_mut_rate(cls, v):
        return validate_mutation_rate(v)
    
    @validator('selection_pressure')
    def validate_sel_pressure(cls, v):
        return validate_selection_pressure(v)
    
    @validator('simulation_time')
    def validate_sim_time(cls, v):
        return validate_simulation_time(v)
    
    @validator('antibiotic_concentration')
    def validate_antibiotic_conc(cls, v):
        return validate_antibiotic_concentration(v)
    
    @validator('mutation_rate', 'selection_pressure')
    def validate_rate_combination(cls, v, values):
        """Validate that mutation rate and selection pressure combination is reasonable."""
        if 'mutation_rate' in values and 'selection_pressure' in values:
            mut_rate = values.get('mutation_rate', 0)
            sel_pressure = v if hasattr(v, '__float__') else values.get('selection_pressure', 0)
            
            # Check for unrealistic combinations
            if mut_rate > 0.01 and sel_pressure > 0.8:
                raise ValueError(
                    "Very high mutation rate with very high selection pressure "
                    "may lead to unrealistic simulation results"
                )
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "initial_population_size": 1000,
                "mutation_rate": 0.001,
                "selection_pressure": 0.1,
                "antibiotic_concentration": 1.0,
                "simulation_time": 100
            }
        }


class SimulationParameters(BaseModel):
    """Simulation parameters model."""
    
    initial_population_size: int
    mutation_rate: float
    selection_pressure: float
    antibiotic_concentration: float
    simulation_time: int


class SimulationResults(BaseModel):
    """Simulation results model."""
    
    population_history: List[int] = Field(description="Population size over time")
    resistance_history: List[float] = Field(description="Average resistance over time")
    fitness_history: List[float] = Field(description="Average fitness over time")
    
    @validator('population_history')
    def validate_population_history(cls, v):
        if not v:
            return v
        if any(pop < 0 for pop in v):
            raise ValueError("Population history cannot contain negative values")
        return v
    
    @validator('resistance_history')
    def validate_resistance_history(cls, v):
        if not v:
            return v
        if any(res < 0 or res > 1 for res in v):
            raise ValueError("Resistance values must be between 0 and 1")
        return v
    
    @validator('fitness_history')
    def validate_fitness_history(cls, v):
        if not v:
            return v
        if any(fit < 0 for fit in v):
            raise ValueError("Fitness values cannot be negative")
        return v


class SimulationResponse(BaseModel):
    """Response model for simulation creation."""
    
    simulation_id: str = Field(description="Unique simulation identifier")
    status: str = Field(description="Current simulation status")
    parameters: SimulationParameters = Field(description="Simulation parameters")
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['initialized', 'running', 'completed', 'failed', 'cancelled']
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return v


class SimulationStatusResponse(BaseModel):
    """Response model for simulation status and results."""
    
    simulation_id: str = Field(description="Unique simulation identifier")
    status: str = Field(description="Current simulation status")
    current_generation: Optional[int] = Field(description="Current generation (if running)")
    generations_completed: Optional[int] = Field(description="Total generations completed")
    final_population_size: Optional[int] = Field(description="Final population size")
    final_resistance: Optional[float] = Field(description="Final average resistance")
    parameters: SimulationParameters = Field(description="Simulation parameters")
    results: SimulationResults = Field(description="Simulation results data")
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['initialized', 'running', 'completed', 'failed', 'cancelled']
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return v
    
    @validator('current_generation')
    def validate_current_generation(cls, v):
        if v is not None and v < 0:
            raise ValueError("Current generation cannot be negative")
        return v
    
    @validator('final_resistance')
    def validate_final_resistance(cls, v):
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Final resistance must be between 0 and 1")
        return v


class SimulationSummary(BaseModel):
    """Summary model for simulation listing."""
    
    simulation_id: str = Field(description="Unique simulation identifier")
    status: str = Field(description="Current simulation status")
    current_generation: int = Field(description="Current generation")
    parameters: SimulationParameters = Field(description="Simulation parameters")
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['initialized', 'running', 'completed', 'failed', 'cancelled']
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return v


class SimulationListResponse(BaseModel):
    """Response model for listing simulations."""
    
    active_simulations: int = Field(description="Number of active simulations")
    simulations: List[SimulationSummary] = Field(description="List of simulations")
    
    @validator('active_simulations')
    def validate_active_simulations(cls, v):
        if v < 0:
            raise ValueError("Active simulations count cannot be negative")
        return v 