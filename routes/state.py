"""
State management API routes for monitoring and control.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import Dict, Any, List, Optional
from utils.state_manager import state_manager
from utils.auth import optional_verify_api_key
from utils.data_transform import ResponseBuilder

router = APIRouter(prefix="/api/state", tags=["State Management"])


@router.get("/health", response_model=Dict[str, Any])
async def get_state_health(
    api_key: str = Depends(optional_verify_api_key)
) -> Dict[str, Any]:
    """
    Get health status of the state management system.
    
    Returns:
        Health check results and system metrics
    """
    try:
        health_data = state_manager.health_check()
        return ResponseBuilder.success(
            data=health_data,
            message="State management health check completed"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get state health: {str(e)}"
        )


@router.get("/memory", response_model=Dict[str, Any])
async def get_memory_usage(
    api_key: str = Depends(optional_verify_api_key)
) -> Dict[str, Any]:
    """
    Get current memory usage statistics.
    
    Returns:
        Memory usage information
    """
    try:
        memory_data = state_manager.get_memory_usage()
        return ResponseBuilder.success(
            data=memory_data,
            message="Memory usage retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory usage: {str(e)}"
        )


@router.get("/simulations", response_model=Dict[str, Any])
async def list_active_states(
    api_key: str = Depends(optional_verify_api_key)
) -> Dict[str, Any]:
    """
    List all active simulation states.
    
    Returns:
        List of active simulation states with basic info
    """
    try:
        active_states = {}
        for sim_id, state_data in state_manager.active_states.items():
            active_states[sim_id] = {
                "simulation_id": sim_id,
                "state": state_data.get("state"),
                "created_at": state_data.get("created_at"),
                "updated_at": state_data.get("updated_at"),
                "current_generation": state_data.get("current_generation", 0),
                "progress_percentage": state_data.get("progress_percentage", 0.0),
                "population_size": state_data.get("population_size", 0)
            }
        
        return ResponseBuilder.success(
            data={
                "active_simulations": active_states,
                "count": len(active_states)
            },
            message="Active simulation states retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list active states: {str(e)}"
        )


@router.get("/simulations/{simulation_id}", response_model=Dict[str, Any])
async def get_simulation_state(
    simulation_id: str,
    api_key: str = Depends(optional_verify_api_key)
) -> Dict[str, Any]:
    """
    Get detailed state information for a specific simulation.
    
    Args:
        simulation_id: ID of the simulation
        
    Returns:
        Detailed simulation state data
    """
    try:
        state_data = state_manager.get_simulation_state(simulation_id)
        if not state_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation {simulation_id} not found"
            )
        
        return ResponseBuilder.success(
            data=state_data,
            message="Simulation state retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get simulation state: {str(e)}"
        )


@router.get("/simulations/{simulation_id}/snapshots", response_model=Dict[str, Any])
async def get_simulation_snapshots(
    simulation_id: str,
    limit: Optional[int] = 20,
    api_key: str = Depends(optional_verify_api_key)
) -> Dict[str, Any]:
    """
    Get snapshots for a specific simulation.
    
    Args:
        simulation_id: ID of the simulation
        limit: Maximum number of snapshots to return
        
    Returns:
        List of simulation snapshots
    """
    try:
        snapshots = state_manager.state_snapshots.get(simulation_id, [])
        
        # Limit results
        if limit and len(snapshots) > limit:
            snapshots = snapshots[-limit:]
        
        # Convert snapshots to dict format
        snapshot_data = [snapshot.to_dict() for snapshot in snapshots]
        
        return ResponseBuilder.success(
            data={
                "snapshots": snapshot_data,
                "count": len(snapshot_data),
                "simulation_id": simulation_id
            },
            message="Simulation snapshots retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get simulation snapshots: {str(e)}"
        )


@router.get("/simulations/{simulation_id}/checkpoints", response_model=Dict[str, Any])
async def get_simulation_checkpoints(
    simulation_id: str,
    api_key: str = Depends(optional_verify_api_key)
) -> Dict[str, Any]:
    """
    Get available checkpoints for a specific simulation.
    
    Args:
        simulation_id: ID of the simulation
        
    Returns:
        List of available checkpoints
    """
    try:
        state_data = state_manager.get_simulation_state(simulation_id)
        if not state_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Simulation {simulation_id} not found"
            )
        
        checkpoints = state_data.get("checkpoints", [])
        
        return ResponseBuilder.success(
            data={
                "checkpoints": checkpoints,
                "count": len(checkpoints),
                "simulation_id": simulation_id
            },
            message="Simulation checkpoints retrieved successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get simulation checkpoints: {str(e)}"
        )


@router.post("/simulations/{simulation_id}/cleanup", response_model=Dict[str, Any])
async def cleanup_simulation_state(
    simulation_id: str,
    remove_files: bool = False,
    api_key: str = Depends(optional_verify_api_key)
) -> Dict[str, Any]:
    """
    Clean up simulation state and optionally remove files.
    
    Args:
        simulation_id: ID of the simulation
        remove_files: Whether to remove saved files
        
    Returns:
        Cleanup confirmation
    """
    try:
        state_manager.cleanup_simulation(simulation_id, remove_files)
        
        return ResponseBuilder.success(
            data={"simulation_id": simulation_id, "files_removed": remove_files},
            message="Simulation state cleaned up successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup simulation state: {str(e)}"
        )


@router.get("/statistics", response_model=Dict[str, Any])
async def get_state_statistics(
    api_key: str = Depends(optional_verify_api_key)
) -> Dict[str, Any]:
    """
    Get overall state management statistics.
    
    Returns:
        Comprehensive statistics about the state management system
    """
    try:
        stats = {
            "active_simulations": len(state_manager.active_states),
            "total_snapshots": sum(len(snapshots) for snapshots in state_manager.state_snapshots.values()),
            "auto_save_tasks": len(state_manager._auto_save_tasks),
            "memory_usage": state_manager.get_memory_usage(),
            "health_status": state_manager.health_check(),
            "simulation_states": {}
        }
        
        # Count simulations by state
        for sim_id, state_data in state_manager.active_states.items():
            sim_state = str(state_data.get("state", "unknown"))
            if sim_state not in stats["simulation_states"]:
                stats["simulation_states"][sim_state] = 0
            stats["simulation_states"][sim_state] += 1
        
        return ResponseBuilder.success(
            data=stats,
            message="State management statistics retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get state statistics: {str(e)}"
        ) 