"""
Simulation API routes for bacterial resistance simulation.
"""

from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends, Query, Request
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional
import json
import uuid
from datetime import datetime
from services.simulation_service import SimulationService
from schemas.simulation import (
    SimulationCreateRequest,
    SimulationResponse,
    SimulationStatusResponse,
    SimulationListResponse
)
from utils.data_transform import DataTransformer, ResponseBuilder
from utils.auth import verify_api_key, optional_verify_api_key, SecurityValidator, RequestValidator

router = APIRouter(prefix="/api/simulations", tags=["Simulations"])

# Global simulation service instance
simulation_service = SimulationService()


@router.post("/", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_simulation(
    request: SimulationCreateRequest,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Create a new bacterial resistance simulation.
    
    Args:
        request: Simulation parameters including population size, mutation rate, etc.
        api_key: Validated API key
        
    Returns:
        Simulation metadata with unique ID and parameters
    """
    try:
        # Validate simulation parameters
        validated_params = DataTransformer.validate_simulation_parameters(request.dict())
        
        # Generate unique simulation ID
        simulation_id = str(uuid.uuid4())
        
        # Create simulation using service
        result = simulation_service.create_simulation(
            simulation_id=simulation_id,
            **validated_params
        )
        
        # Format response data
        formatted_result = DataTransformer.format_simulation_data(result)
        
        return ResponseBuilder.success(
            data=formatted_result,
            message="Simulation created successfully"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create simulation: {str(e)}"
        )


@router.post("/{simulation_id}/run", response_model=Dict[str, Any])
async def run_simulation(
    simulation_id: str, 
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Run a simulation synchronously (legacy method).
    
    Args:
        simulation_id: ID of the simulation to run
        background_tasks: FastAPI background tasks for async execution
        api_key: Validated API key
        
    Returns:
        Complete simulation results
    """
    try:
        # Sanitize simulation ID
        clean_simulation_id = SecurityValidator.sanitize_simulation_id(simulation_id)
        
        # Run simulation synchronously
        result = simulation_service.run_simulation(clean_simulation_id)
        
        # Format response data
        formatted_result = DataTransformer.format_simulation_data(result)
        
        return ResponseBuilder.success(
            data=formatted_result,
            message="Simulation completed successfully"
        )
        
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


@router.post("/{simulation_id}/run-stream")
async def run_simulation_stream(
    simulation_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Run a simulation with real-time streaming updates.
    
    Args:
        simulation_id: ID of the simulation to run
        api_key: Validated API key
        
    Returns:
        Server-sent events stream with real-time progress updates
    """
    try:
        # Sanitize simulation ID
        clean_simulation_id = SecurityValidator.sanitize_simulation_id(simulation_id)
        
        async def generate_progress():
            """Generate server-sent events for simulation progress."""
            try:
                async for progress_data in simulation_service.run_simulation_async(clean_simulation_id):
                    # Format progress data
                    formatted_data = DataTransformer.format_simulation_data(progress_data)
                    
                    # Create standardized message
                    message = DataTransformer.format_websocket_message(
                        message_type="simulation_progress",
                        data=formatted_data,
                        simulation_id=clean_simulation_id
                    )
                    
                    yield f"data: {message}\n\n"
                    
            except ValueError as e:
                error_response = ResponseBuilder.error(str(e), "SIMULATION_NOT_FOUND")
                yield f"data: {json.dumps(error_response)}\n\n"
            except Exception as e:
                error_response = ResponseBuilder.error(str(e), "SIMULATION_ERROR")
                yield f"data: {json.dumps(error_response)}\n\n"
        
        return StreamingResponse(
            generate_progress(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start simulation stream: {str(e)}"
        )


@router.get("/{simulation_id}/status", response_model=Dict[str, Any])
async def get_simulation_status(
    simulation_id: str,
    api_key: str = Depends(optional_verify_api_key)
) -> Dict[str, Any]:
    """
    Get the current status and progress of a simulation.
    
    Args:
        simulation_id: ID of the simulation
        api_key: Optional API key for authentication
        
    Returns:
        Current simulation status and progress data
    """
    try:
        # Sanitize simulation ID
        clean_simulation_id = SecurityValidator.sanitize_simulation_id(simulation_id)
        
        result = simulation_service.get_simulation_status(clean_simulation_id)
        
        # Format response data
        formatted_result = DataTransformer.format_simulation_data(result)
        
        return ResponseBuilder.success(
            data=formatted_result,
            message="Simulation status retrieved successfully"
        )
        
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


@router.get("/", response_model=Dict[str, Any])
async def list_simulations(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    created_after: Optional[datetime] = Query(None, description="Filter by creation date"),
    created_before: Optional[datetime] = Query(None, description="Filter by creation date"),
    api_key: str = Depends(optional_verify_api_key)
) -> Dict[str, Any]:
    """
    List all simulations with filtering and pagination.
    
    Args:
        page: Page number for pagination
        page_size: Number of items per page
        status_filter: Filter simulations by status
        created_after: Filter simulations created after this date
        created_before: Filter simulations created before this date
        api_key: Optional API key for authentication
        
    Returns:
        Paginated list of simulations with metadata
    """
    try:
        # Validate pagination parameters
        validated_page, validated_page_size = RequestValidator.validate_pagination(page, page_size)
        
        # Validate filters
        filters = RequestValidator.validate_simulation_filters(
            status=status_filter,
            created_after=created_after,
            created_before=created_before
        )
        
        # Get simulations from service
        result = simulation_service.list_simulations()
        simulations = result.get("simulations", [])
        
        # Apply filters
        if filters.get("status"):
            simulations = [s for s in simulations if s.get("status") == filters["status"]]
        
        if filters.get("created_after"):
            simulations = [s for s in simulations 
                         if s.get("created_at") and s["created_at"] >= filters["created_after"]]
        
        if filters.get("created_before"):
            simulations = [s for s in simulations 
                         if s.get("created_at") and s["created_at"] <= filters["created_before"]]
        
        # Format simulation data
        formatted_simulations = [
            DataTransformer.format_simulation_data(sim) for sim in simulations
        ]
        
        # Apply pagination
        paginated_result = DataTransformer.paginate_results(
            formatted_simulations, 
            validated_page, 
            validated_page_size
        )
        
        return ResponseBuilder.success(
            data=paginated_result,
            message="Simulations retrieved successfully"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list simulations: {str(e)}"
        )


@router.delete("/{simulation_id}", response_model=Dict[str, Any])
async def delete_simulation(
    simulation_id: str,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Delete a simulation from memory.
    
    Args:
        simulation_id: ID of the simulation to delete
        api_key: Validated API key
        
    Returns:
        Deletion confirmation
    """
    try:
        # Sanitize simulation ID
        clean_simulation_id = SecurityValidator.sanitize_simulation_id(simulation_id)
        
        success = simulation_service.delete_simulation(clean_simulation_id)
        if success:
            return ResponseBuilder.success(
                data={"simulation_id": clean_simulation_id},
                message=f"Simulation {clean_simulation_id} deleted successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation {clean_simulation_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete simulation: {str(e)}"
        )


@router.post("/{simulation_id}/pause", response_model=Dict[str, Any])
async def pause_simulation(
    simulation_id: str,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Pause a running simulation.
    
    Args:
        simulation_id: ID of the simulation to pause
        api_key: Validated API key
        
    Returns:
        Updated simulation status
    """
    try:
        # Sanitize simulation ID
        clean_simulation_id = SecurityValidator.sanitize_simulation_id(simulation_id)
        
        result = simulation_service.pause_simulation(clean_simulation_id)
        
        # Format response data
        formatted_result = DataTransformer.format_simulation_data(result)
        
        return ResponseBuilder.success(
            data=formatted_result,
            message="Simulation paused successfully"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to pause simulation: {str(e)}"
        )


@router.post("/{simulation_id}/resume", response_model=Dict[str, Any])
async def resume_simulation(
    simulation_id: str,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Resume a paused simulation.
    
    Args:
        simulation_id: ID of the simulation to resume
        api_key: Validated API key
        
    Returns:
        Updated simulation status
    """
    try:
        # Sanitize simulation ID
        clean_simulation_id = SecurityValidator.sanitize_simulation_id(simulation_id)
        
        result = simulation_service.resume_simulation(clean_simulation_id)
        
        # Format response data
        formatted_result = DataTransformer.format_simulation_data(result)
        
        return ResponseBuilder.success(
            data=formatted_result,
            message="Simulation resumed successfully"
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume simulation: {str(e)}"
        ) 