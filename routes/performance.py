"""
Performance monitoring API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from utils.performance_profiler import get_profiler, AdvancedProfiler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/performance", tags=["performance"])


@router.get("/summary")
async def get_performance_summary() -> Dict[str, Any]:
    """Get comprehensive performance summary."""
    try:
        profiler = get_profiler()
        summary = profiler.get_performance_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance summary")


@router.get("/metrics/{function_name}")
async def get_function_metrics(
    function_name: str,
    limit: int = Query(default=100, ge=1, le=1000, description="Number of recent metrics to return")
) -> Dict[str, Any]:
    """Get metrics for a specific function."""
    try:
        profiler = get_profiler()
        
        if function_name not in profiler.metrics_storage:
            raise HTTPException(status_code=404, detail=f"No metrics found for function: {function_name}")
        
        metrics = profiler.metrics_storage[function_name][-limit:]
        
        return {
            "function_name": function_name,
            "metric_count": len(metrics),
            "metrics": [metric.to_dict() for metric in metrics]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get function metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve function metrics")


@router.get("/realtime")
async def get_realtime_metrics(
    count: int = Query(default=50, ge=1, le=500, description="Number of recent system metrics to return")
) -> List[Dict[str, Any]]:
    """Get recent real-time system metrics."""
    try:
        profiler = get_profiler()
        metrics = profiler.realtime_collector.get_recent_metrics(count)
        return metrics
    except Exception as e:
        logger.error(f"Failed to get realtime metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve realtime metrics")


@router.get("/baselines")
async def get_baselines() -> Dict[str, Any]:
    """Get all performance baselines."""
    try:
        profiler = get_profiler()
        baselines = {
            name: baseline.to_dict() 
            for name, baseline in profiler.baselines.items()
        }
        return {
            "baseline_count": len(baselines),
            "baselines": baselines
        }
    except Exception as e:
        logger.error(f"Failed to get baselines: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve baselines")


@router.post("/baselines/{function_name}")
async def create_baseline(function_name: str, force: bool = Query(default=False)) -> Dict[str, Any]:
    """Create or update a performance baseline for a function."""
    try:
        profiler = get_profiler()
        baseline = profiler.create_baseline(function_name, force=force)
        
        if baseline is None:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot create baseline for {function_name}. Insufficient metrics or function not found."
            )
        
        return {
            "message": f"Baseline created for {function_name}",
            "baseline": baseline.to_dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create baseline: {e}")
        raise HTTPException(status_code=500, detail="Failed to create baseline")


@router.get("/alerts")
async def get_alerts(
    hours: int = Query(default=24, ge=1, le=168, description="Hours to look back for alerts"),
    severity: Optional[str] = Query(default=None, description="Filter by severity: low, medium, high, critical")
) -> Dict[str, Any]:
    """Get performance alerts."""
    try:
        profiler = get_profiler()
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter alerts by time and optionally by severity
        filtered_alerts = [
            alert for alert in profiler.alerts
            if alert.timestamp > cutoff_time and (severity is None or alert.severity == severity)
        ]
        
        # Sort by timestamp (most recent first)
        filtered_alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return {
            "alert_count": len(filtered_alerts),
            "alerts": [alert.to_dict() for alert in filtered_alerts],
            "summary": {
                "critical": len([a for a in filtered_alerts if a.severity == "critical"]),
                "high": len([a for a in filtered_alerts if a.severity == "high"]),
                "medium": len([a for a in filtered_alerts if a.severity == "medium"]),
                "low": len([a for a in filtered_alerts if a.severity == "low"])
            }
        }
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


@router.get("/functions")
async def get_monitored_functions() -> Dict[str, Any]:
    """Get list of functions being monitored."""
    try:
        profiler = get_profiler()
        
        functions = []
        for name, metrics in profiler.metrics_storage.items():
            if metrics:
                latest_metric = max(metrics, key=lambda m: m.timestamp)
                functions.append({
                    "function_name": name,
                    "metric_count": len(metrics),
                    "last_called": latest_metric.timestamp.isoformat(),
                    "has_baseline": name in profiler.baselines,
                    "avg_execution_time": sum(m.execution_time for m in metrics[-50:]) / min(len(metrics), 50),
                    "avg_memory_usage": sum(m.memory_current for m in metrics[-50:]) / min(len(metrics), 50)
                })
        
        # Sort by last called (most recent first)
        functions.sort(key=lambda x: x["last_called"], reverse=True)
        
        return {
            "function_count": len(functions),
            "functions": functions
        }
    except Exception as e:
        logger.error(f"Failed to get monitored functions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve monitored functions")


@router.delete("/metrics/{function_name}")
async def clear_function_metrics(function_name: str) -> Dict[str, str]:
    """Clear metrics for a specific function."""
    try:
        profiler = get_profiler()
        
        if function_name not in profiler.metrics_storage:
            raise HTTPException(status_code=404, detail=f"No metrics found for function: {function_name}")
        
        # Clear metrics
        del profiler.metrics_storage[function_name]
        
        return {"message": f"Metrics cleared for {function_name}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear metrics")


@router.delete("/metrics")
async def clear_all_metrics() -> Dict[str, str]:
    """Clear all performance metrics."""
    try:
        profiler = get_profiler()
        profiler.metrics_storage.clear()
        return {"message": "All metrics cleared"}
    except Exception as e:
        logger.error(f"Failed to clear all metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear all metrics")


@router.get("/health")
async def performance_health_check() -> Dict[str, Any]:
    """Health check for performance monitoring system."""
    try:
        profiler = get_profiler()
        
        return {
            "status": "healthy",
            "realtime_collector_running": profiler.realtime_collector._running,
            "memory_tracking_enabled": tracemalloc.is_tracing(),
            "functions_monitored": len(profiler.metrics_storage),
            "baselines_loaded": len(profiler.baselines),
            "recent_alerts": len([a for a in profiler.alerts if a.timestamp > datetime.now() - timedelta(hours=1)])
        }
    except Exception as e:
        logger.error(f"Performance health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.post("/config")
async def update_config(
    regression_threshold: Optional[float] = Query(default=None, ge=0, le=100),
    max_metrics_per_function: Optional[int] = Query(default=None, ge=100, le=10000)
) -> Dict[str, Any]:
    """Update performance monitoring configuration."""
    try:
        profiler = get_profiler()
        config_updates = {}
        
        if regression_threshold is not None:
            profiler.regression_threshold = regression_threshold
            config_updates["regression_threshold"] = regression_threshold
        
        if max_metrics_per_function is not None:
            profiler.max_metrics_per_function = max_metrics_per_function
            config_updates["max_metrics_per_function"] = max_metrics_per_function
        
        return {
            "message": "Configuration updated successfully",
            "updates": config_updates,
            "current_config": {
                "regression_threshold": profiler.regression_threshold,
                "max_metrics_per_function": profiler.max_metrics_per_function
            }
        }
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update configuration")


# Import tracemalloc for health check
import tracemalloc 