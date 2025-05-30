"""
Performance monitoring and benchmarking API routes.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import Dict, Any, Optional, List
import json
import asyncio
from datetime import datetime

from utils.auth import verify_api_key, optional_verify_api_key
from utils.performance_simple import simple_profiler, simple_cache
from utils.data_transform import ResponseBuilder

router = APIRouter(prefix="/api/performance", tags=["Performance"])


@router.get("/metrics", response_model=Dict[str, Any])
async def get_performance_metrics(
    function_name: Optional[str] = Query(None, description="Specific function to get metrics for"),
    api_key: str = Depends(optional_verify_api_key)
) -> Dict[str, Any]:
    """Get performance metrics summary."""
    try:
        metrics = simple_profiler.get_metrics_summary(function_name)
        
        return ResponseBuilder.success(
            data={
                "metrics": metrics,
                "timestamp": datetime.now().isoformat(),
                "function_filter": function_name
            },
            message="Performance metrics retrieved successfully"
        )
    
    except Exception as e:
        return ResponseBuilder.error(f"Failed to get metrics: {str(e)}")


@router.get("/memory", response_model=Dict[str, Any])
async def get_memory_usage(
    api_key: str = Depends(optional_verify_api_key)
) -> Dict[str, Any]:
    """Get simplified memory usage information."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return ResponseBuilder.success(
            data={
                "current_memory_mb": memory_info.rss / 1024 / 1024,
                "timestamp": datetime.now().isoformat(),
                "memory_percent": process.memory_percent()
            },
            message="Memory usage retrieved successfully"
        )
    
    except Exception as e:
        return ResponseBuilder.error(f"Failed to get memory usage: {str(e)}")


@router.get("/cache-stats", response_model=Dict[str, Any])
async def get_cache_statistics(
    api_key: str = Depends(optional_verify_api_key)
) -> Dict[str, Any]:
    """Get simple cache statistics."""
    try:
        return ResponseBuilder.success(
            data={
                "cache_size": len(simple_cache.cache),
                "max_size": simple_cache.maxsize,
                "access_order_length": len(simple_cache.access_order),
                "timestamp": datetime.now().isoformat()
            },
            message="Cache statistics retrieved successfully"
        )
    
    except Exception as e:
        return ResponseBuilder.error(f"Failed to get cache stats: {str(e)}")


@router.post("/cache/clear", response_model=Dict[str, Any])
async def clear_cache(
    cache_category: Optional[str] = Query(None, description="Specific cache category to clear"),
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """Clear cache entries."""
    try:
        if cache_category:
            simple_cache.clear(cache_category)
            message = f"Cache category '{cache_category}' cleared successfully"
        else:
            simple_cache.clear()
            message = "All caches cleared successfully"
        
        return ResponseBuilder.success(
            data={"cleared_cache": cache_category or "all"},
            message=message
        )
    
    except Exception as e:
        return ResponseBuilder.error(f"Failed to clear cache: {str(e)}")


@router.get("/status", response_model=Dict[str, Any])
async def get_performance_status(
    api_key: str = Depends(optional_verify_api_key)
) -> Dict[str, Any]:
    """Get overall performance status."""
    try:
        import psutil
        
        return ResponseBuilder.success(
            data={
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "active_functions": len(simple_profiler.metrics_storage),
                "cache_utilization": len(simple_cache.cache) / simple_cache.maxsize * 100,
                "timestamp": datetime.now().isoformat()
            },
            message="Performance status retrieved successfully"
        )
    
    except Exception as e:
        return ResponseBuilder.error(f"Failed to get performance status: {str(e)}") 