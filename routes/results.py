"""
API routes for result collection and analysis.

This module provides RESTful endpoints for accessing the result collection
framework functionality including metrics collection, analysis, and reporting.
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from typing import Dict, List, Optional, Any
import asyncio
import json
import io
from datetime import datetime
from pathlib import Path

from utils.result_collection import (
    ResultCollector, ResultAnalyzer, StreamingResultCollector,
    ResultMetrics, ResultFormat
)
from utils.auth import verify_api_key
from schemas.simulation import SimulationMetrics, SimulationProgressUpdate


router = APIRouter(prefix="/api/results", tags=["Results"])

# Global instances - in production, these should be dependency injected
result_collector = ResultCollector("simulation_results")
result_analyzer = ResultAnalyzer(result_collector)
streaming_collector = StreamingResultCollector("simulation_results", stream_interval=1.0)


@router.get("/health")
async def health_check():
    """Health check for results service."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "storage_path": str(result_collector.storage_path),
        "active_simulations": len(result_collector._metrics_buffer)
    }


@router.post("/collect")
async def collect_metrics(
    metrics: Dict[str, Any],
    api_key: str = Depends(verify_api_key)
):
    """
    Collect simulation metrics.
    
    Args:
        metrics: Result metrics dictionary
        api_key: API authentication key
        
    Returns:
        Collection confirmation
    """
    try:
        # Convert dict to ResultMetrics
        result_metrics = ResultMetrics(
            simulation_id=metrics["simulation_id"],
            generation=metrics["generation"],
            timestamp=datetime.fromisoformat(metrics["timestamp"]) if isinstance(metrics["timestamp"], str) else metrics["timestamp"],
            population_size=metrics["population_size"],
            resistant_count=metrics["resistant_count"],
            sensitive_count=metrics["sensitive_count"],
            average_fitness=metrics["average_fitness"],
            fitness_std=metrics["fitness_std"],
            mutation_count=metrics["mutation_count"],
            extinction_occurred=metrics["extinction_occurred"],
            diversity_index=metrics["diversity_index"],
            selection_pressure=metrics["selection_pressure"],
            mutation_rate=metrics["mutation_rate"],
            elapsed_time=metrics["elapsed_time"],
            memory_usage=metrics["memory_usage"],
            cpu_usage=metrics["cpu_usage"]
        )
        
        result_collector.collect_metrics(result_metrics)
        
        return {
            "status": "success",
            "message": f"Metrics collected for simulation {metrics['simulation_id']}, generation {metrics['generation']}",
            "timestamp": datetime.now().isoformat()
        }
        
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error collecting metrics: {str(e)}")


@router.get("/simulations")
async def list_simulations(api_key: str = Depends(verify_api_key)):
    """
    List all tracked simulations.
    
    Args:
        api_key: API authentication key
        
    Returns:
        List of simulation IDs and basic info
    """
    try:
        simulations = []
        
        for sim_id, metrics_list in result_collector._metrics_buffer.items():
            if metrics_list:
                latest_metrics = max(metrics_list, key=lambda m: m.generation)
                simulations.append({
                    "simulation_id": sim_id,
                    "total_generations": len(metrics_list),
                    "latest_generation": latest_metrics.generation,
                    "latest_timestamp": latest_metrics.timestamp.isoformat(),
                    "final_population": latest_metrics.population_size,
                    "status": "extinct" if latest_metrics.extinction_occurred else "active"
                })
        
        return {
            "simulations": simulations,
            "total_count": len(simulations),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing simulations: {str(e)}")


@router.get("/simulations/{simulation_id}/metrics")
async def get_simulation_metrics(
    simulation_id: str,
    format_type: Optional[str] = Query("json", regex="^(json|dict)$"),
    api_key: str = Depends(verify_api_key)
):
    """
    Get all metrics for a specific simulation.
    
    Args:
        simulation_id: Simulation identifier
        format_type: Response format (json or dict)
        api_key: API authentication key
        
    Returns:
        Simulation metrics
    """
    try:
        metrics = result_collector.get_metrics(simulation_id)
        
        if not metrics:
            raise HTTPException(status_code=404, detail=f"No metrics found for simulation {simulation_id}")
        
        if format_type == "dict":
            return {
                "simulation_id": simulation_id,
                "metrics": [m.to_dict() for m in metrics],
                "count": len(metrics),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "simulation_id": simulation_id,
                "metrics": [m.to_dict() for m in metrics],
                "count": len(metrics),
                "timestamp": datetime.now().isoformat()
            }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")


@router.get("/simulations/{simulation_id}/aggregate")
async def get_aggregated_results(
    simulation_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get aggregated results for a simulation.
    
    Args:
        simulation_id: Simulation identifier
        api_key: API authentication key
        
    Returns:
        Aggregated simulation results
    """
    try:
        aggregated = result_analyzer.aggregate_results(simulation_id)
        
        return {
            "simulation_id": simulation_id,
            "aggregated_results": aggregated.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error aggregating results: {str(e)}")


@router.get("/simulations/{simulation_id}/analysis")
async def get_statistical_analysis(
    simulation_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Get statistical analysis for a simulation.
    
    Args:
        simulation_id: Simulation identifier
        api_key: API authentication key
        
    Returns:
        Statistical analysis results
    """
    try:
        analysis = result_analyzer.statistical_analysis(simulation_id)
        
        return {
            "simulation_id": simulation_id,
            "statistical_analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing analysis: {str(e)}")


@router.get("/simulations/{simulation_id}/report")
async def generate_report(
    simulation_id: str,
    include_plots: bool = Query(False, description="Include plot data in report"),
    api_key: str = Depends(verify_api_key)
):
    """
    Generate comprehensive analysis report for a simulation.
    
    Args:
        simulation_id: Simulation identifier
        include_plots: Whether to include plot data
        api_key: API authentication key
        
    Returns:
        Comprehensive analysis report
    """
    try:
        report = result_analyzer.generate_report(simulation_id, include_plots=include_plots)
        
        return {
            "simulation_id": simulation_id,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


@router.post("/compare")
async def compare_simulations(
    simulation_ids: List[str],
    api_key: str = Depends(verify_api_key)
):
    """
    Compare multiple simulations.
    
    Args:
        simulation_ids: List of simulation identifiers to compare
        api_key: API authentication key
        
    Returns:
        Comparison analysis
    """
    try:
        if len(simulation_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 simulations required for comparison")
        
        comparison = result_analyzer.compare_simulations(simulation_ids)
        
        return {
            "simulation_ids": simulation_ids,
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing simulations: {str(e)}")


@router.post("/simulations/{simulation_id}/export")
async def export_metrics(
    simulation_id: str,
    format_type: ResultFormat,
    api_key: str = Depends(verify_api_key)
):
    """
    Export simulation metrics to file.
    
    Args:
        simulation_id: Simulation identifier
        format_type: Export format
        api_key: API authentication key
        
    Returns:
        File download response
    """
    try:
        filepath = result_collector.save_metrics(simulation_id, format_type)
        
        media_type_map = {
            ResultFormat.JSON: "application/json",
            ResultFormat.CSV: "text/csv",
            ResultFormat.PICKLE: "application/octet-stream",
            ResultFormat.EXCEL: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
        
        return FileResponse(
            path=filepath,
            media_type=media_type_map.get(format_type, "application/octet-stream"),
            filename=filepath.name
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting metrics: {str(e)}")


@router.delete("/simulations/{simulation_id}")
async def clear_simulation_metrics(
    simulation_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Clear metrics for a specific simulation.
    
    Args:
        simulation_id: Simulation identifier
        api_key: API authentication key
        
    Returns:
        Deletion confirmation
    """
    try:
        metrics_count = len(result_collector.get_metrics(simulation_id))
        result_collector.clear_metrics(simulation_id)
        
        return {
            "status": "success",
            "message": f"Cleared {metrics_count} metrics for simulation {simulation_id}",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing metrics: {str(e)}")


# Streaming endpoints
@router.post("/streaming/{simulation_id}/start")
async def start_streaming(
    simulation_id: str,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Start streaming results for a simulation.
    
    Args:
        simulation_id: Simulation identifier
        background_tasks: FastAPI background tasks
        api_key: API authentication key
        
    Returns:
        Streaming start confirmation
    """
    try:
        streaming_buffer = []
        
        def stream_callback(metrics_list):
            streaming_buffer.extend([m.to_dict() for m in metrics_list])
        
        await streaming_collector.start_streaming(simulation_id, stream_callback)
        
        return {
            "status": "success",
            "message": f"Started streaming for simulation {simulation_id}",
            "simulation_id": simulation_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting streaming: {str(e)}")


@router.post("/streaming/{simulation_id}/stop")
async def stop_streaming(
    simulation_id: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Stop streaming for a simulation.
    
    Args:
        simulation_id: Simulation identifier
        api_key: API authentication key
        
    Returns:
        Streaming stop confirmation
    """
    try:
        await streaming_collector.stop_streaming(simulation_id)
        
        return {
            "status": "success",
            "message": f"Stopped streaming for simulation {simulation_id}",
            "simulation_id": simulation_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error stopping streaming: {str(e)}")


@router.get("/streaming/status")
async def get_streaming_status(api_key: str = Depends(verify_api_key)):
    """
    Get status of all active streaming sessions.
    
    Args:
        api_key: API authentication key
        
    Returns:
        Streaming status information
    """
    try:
        active_streams = list(streaming_collector._stream_tasks.keys())
        
        return {
            "active_streams": active_streams,
            "stream_count": len(active_streams),
            "stream_interval": streaming_collector.stream_interval,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting streaming status: {str(e)}")


# Batch operations
@router.post("/batch/collect")
async def batch_collect_metrics(
    metrics_list: List[Dict[str, Any]],
    api_key: str = Depends(verify_api_key)
):
    """
    Collect multiple metrics in batch.
    
    Args:
        metrics_list: List of metrics dictionaries
        api_key: API authentication key
        
    Returns:
        Batch collection results
    """
    try:
        collected_count = 0
        errors = []
        
        for i, metrics_data in enumerate(metrics_list):
            try:
                result_metrics = ResultMetrics(
                    simulation_id=metrics_data["simulation_id"],
                    generation=metrics_data["generation"],
                    timestamp=datetime.fromisoformat(metrics_data["timestamp"]) if isinstance(metrics_data["timestamp"], str) else metrics_data["timestamp"],
                    population_size=metrics_data["population_size"],
                    resistant_count=metrics_data["resistant_count"],
                    sensitive_count=metrics_data["sensitive_count"],
                    average_fitness=metrics_data["average_fitness"],
                    fitness_std=metrics_data["fitness_std"],
                    mutation_count=metrics_data["mutation_count"],
                    extinction_occurred=metrics_data["extinction_occurred"],
                    diversity_index=metrics_data["diversity_index"],
                    selection_pressure=metrics_data["selection_pressure"],
                    mutation_rate=metrics_data["mutation_rate"],
                    elapsed_time=metrics_data["elapsed_time"],
                    memory_usage=metrics_data["memory_usage"],
                    cpu_usage=metrics_data["cpu_usage"]
                )
                
                result_collector.collect_metrics(result_metrics)
                collected_count += 1
                
            except Exception as e:
                errors.append({
                    "index": i,
                    "error": str(e),
                    "simulation_id": metrics_data.get("simulation_id", "unknown")
                })
        
        return {
            "status": "completed",
            "collected_count": collected_count,
            "total_count": len(metrics_list),
            "error_count": len(errors),
            "errors": errors,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in batch collection: {str(e)}")


@router.get("/storage/info")
async def get_storage_info(api_key: str = Depends(verify_api_key)):
    """
    Get information about result storage.
    
    Args:
        api_key: API authentication key
        
    Returns:
        Storage information
    """
    try:
        storage_path = result_collector.storage_path
        
        # Count files in storage
        file_count = len([f for f in storage_path.iterdir() if f.is_file()])
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in storage_path.rglob('*') if f.is_file())
        
        return {
            "storage_path": str(storage_path),
            "file_count": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "active_simulations": len(result_collector._metrics_buffer),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting storage info: {str(e)}")


# Utility endpoints
@router.post("/utilities/validate-metrics")
async def validate_metrics_format(
    metrics: Dict[str, Any],
    api_key: str = Depends(verify_api_key)
):
    """
    Validate metrics format without collecting.
    
    Args:
        metrics: Metrics dictionary to validate
        api_key: API authentication key
        
    Returns:
        Validation results
    """
    try:
        # Try to create ResultMetrics object
        result_metrics = ResultMetrics(
            simulation_id=metrics["simulation_id"],
            generation=metrics["generation"],
            timestamp=datetime.fromisoformat(metrics["timestamp"]) if isinstance(metrics["timestamp"], str) else metrics["timestamp"],
            population_size=metrics["population_size"],
            resistant_count=metrics["resistant_count"],
            sensitive_count=metrics["sensitive_count"],
            average_fitness=metrics["average_fitness"],
            fitness_std=metrics["fitness_std"],
            mutation_count=metrics["mutation_count"],
            extinction_occurred=metrics["extinction_occurred"],
            diversity_index=metrics["diversity_index"],
            selection_pressure=metrics["selection_pressure"],
            mutation_rate=metrics["mutation_rate"],
            elapsed_time=metrics["elapsed_time"],
            memory_usage=metrics["memory_usage"],
            cpu_usage=metrics["cpu_usage"]
        )
        
        return {
            "valid": True,
            "message": "Metrics format is valid",
            "parsed_metrics": result_metrics.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
    except KeyError as e:
        return {
            "valid": False,
            "error": f"Missing required field: {e}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Validation error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        } 