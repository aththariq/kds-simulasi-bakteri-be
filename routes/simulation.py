"""
Simulation API routes for bacterial resistance simulation.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
import uuid
from services.simulation_service import SimulationService
from schemas.simulation import (
    SimulationCreateRequest,
    SimulationResponse,
    SimulationStatusResponse,
    SimulationListResponse
)

router = APIRouter(prefix="/api/simulations", tags=["Simulations"])

# Global simulation service instance
simulation_service = SimulationService()


@router.post("/", response_model=SimulationResponse, status_code=status.HTTP_201_CREATED)
async def create_simulation(request: SimulationCreateRequest) -> SimulationResponse:
    """
    Create a new bacterial resistance simulation.
    
    Args:
        request: Simulation parameters including population size, mutation rate, etc.
        
    Returns:
        Simulation metadata with unique ID and parameters
    """
    try:
        # Generate unique simulation ID
        simulation_id = str(uuid.uuid4())
        
        # Create simulation using service
        result = simulation_service.create_simulation(
            simulation_id=simulation_id,
            initial_population_size=request.initial_population_size,
            mutation_rate=request.mutation_rate,
            selection_pressure=request.selection_pressure,
            antibiotic_concentration=request.antibiotic_concentration,
            simulation_time=request.simulation_time
        )
        
        return SimulationResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create simulation: {str(e)}"
        )


@router.post("/{simulation_id}/run", response_model=SimulationStatusResponse)
async def run_simulation(simulation_id: str) -> SimulationStatusResponse:
    """
    Run a simulation by its ID.
    
    Args:
        simulation_id: Unique identifier of the simulation to run
        
    Returns:
        Complete simulation results including population history and resistance data
    """
    try:
        result = simulation_service.run_simulation(simulation_id)
        return SimulationStatusResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run simulation: {str(e)}"
        )


@router.get("/{simulation_id}", response_model=SimulationStatusResponse)
async def get_simulation_status(simulation_id: str) -> SimulationStatusResponse:
    """
    Get the current status and results of a simulation.
    
    Args:
        simulation_id: Unique identifier of the simulation
        
    Returns:
        Current simulation status, progress, and available results
    """
    try:
        result = simulation_service.get_simulation_status(simulation_id)
        return SimulationStatusResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get simulation status: {str(e)}"
        )


@router.delete("/{simulation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_simulation(simulation_id: str):
    """
    Delete a simulation from memory.
    
    Args:
        simulation_id: Unique identifier of the simulation to delete
    """
    try:
        success = simulation_service.delete_simulation(simulation_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation {simulation_id} not found"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete simulation: {str(e)}"
        )


@router.get("/", response_model=SimulationListResponse)
async def list_simulations() -> SimulationListResponse:
    """
    List all active simulations.
    
    Returns:
        List of all active simulations with their current status
    """
    try:
        result = simulation_service.list_simulations()
        return SimulationListResponse(**result)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list simulations: {str(e)}"
        ) 