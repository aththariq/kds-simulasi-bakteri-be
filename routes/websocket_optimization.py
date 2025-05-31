"""
WebSocket Optimization API Endpoints
Task 20.3: WebSocket Communication Efficiency Enhancement

This module provides API endpoints for monitoring and configuring
WebSocket communication optimizations including compression, batching,
and delta updates.
"""

from fastapi import APIRouter, Query, HTTPException, BackgroundTasks
from typing import Dict, List, Optional, Any
import logging

from services.websocket_optimizations import (
    get_optimized_websocket_service,
    MessageOptimizationLevel,
    MessageEncoding,
    CompressionType
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/websocket-optimization", tags=["WebSocket Optimization"])


@router.get("/stats")
async def get_optimization_stats() -> Dict[str, Any]:
    """Get comprehensive WebSocket optimization statistics."""
    try:
        service = get_optimized_websocket_service()
        stats = service.get_performance_summary()
        
        return {
            "status": "success",
            "data": stats,
            "timestamp": "2025-05-31T04:42:00Z"
        }
    
    except Exception as e:
        logger.error(f"Failed to get optimization stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/encoding-stats")
async def get_encoding_stats() -> Dict[str, Any]:
    """Get detailed encoding and compression statistics."""
    try:
        service = get_optimized_websocket_service()
        encoder_stats = service.encoder.get_optimization_stats()
        
        return {
            "status": "success",
            "data": {
                "encoder_stats": encoder_stats,
                "supported_encodings": [encoding.value for encoding in MessageEncoding],
                "supported_compressions": [comp.value for comp in CompressionType],
                "optimization_levels": [level.value for level in MessageOptimizationLevel]
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get encoding stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch-stats")
async def get_batch_stats() -> Dict[str, Any]:
    """Get message batching statistics."""
    try:
        service = get_optimized_websocket_service()
        batch_stats = service.message_batcher.get_batch_stats()
        
        return {
            "status": "success",
            "data": batch_stats
        }
    
    except Exception as e:
        logger.error(f"Failed to get batch stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/client/{client_id}/optimization-level")
async def set_client_optimization_level(
    client_id: str,
    level: str = Query(..., description="Optimization level: none, basic, compressed, delta, batched")
) -> Dict[str, Any]:
    """Set optimization level for a specific client."""
    try:
        # Validate optimization level
        try:
            optimization_level = MessageOptimizationLevel(level)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid optimization level. Must be one of: {[l.value for l in MessageOptimizationLevel]}"
            )
        
        service = get_optimized_websocket_service()
        service.set_client_optimization_level(client_id, optimization_level)
        
        return {
            "status": "success",
            "message": f"Set optimization level for client {client_id} to {level}",
            "client_id": client_id,
            "optimization_level": level
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set client optimization level: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/client/{client_id}/optimization-level")
async def get_client_optimization_level(client_id: str) -> Dict[str, Any]:
    """Get optimization level for a specific client."""
    try:
        service = get_optimized_websocket_service()
        level = service.get_client_optimization_level(client_id)
        
        return {
            "status": "success",
            "client_id": client_id,
            "optimization_level": level.value
        }
    
    except Exception as e:
        logger.error(f"Failed to get client optimization level: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enable")
async def enable_optimization() -> Dict[str, str]:
    """Enable WebSocket optimizations globally."""
    try:
        service = get_optimized_websocket_service()
        service.optimization_enabled = True
        
        return {
            "status": "success",
            "message": "WebSocket optimizations enabled"
        }
    
    except Exception as e:
        logger.error(f"Failed to enable optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disable")
async def disable_optimization() -> Dict[str, str]:
    """Disable WebSocket optimizations globally (fallback to JSON)."""
    try:
        service = get_optimized_websocket_service()
        service.optimization_enabled = False
        
        return {
            "status": "success",
            "message": "WebSocket optimizations disabled (using JSON fallback)"
        }
    
    except Exception as e:
        logger.error(f"Failed to disable optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def optimization_health_check() -> Dict[str, Any]:
    """Health check for WebSocket optimization services."""
    try:
        service = get_optimized_websocket_service()
        
        # Check if MessagePack is available
        try:
            import msgpack
            msgpack_available = True
        except ImportError:
            msgpack_available = False
        
        health_status = {
            "optimization_enabled": service.optimization_enabled,
            "msgpack_available": msgpack_available,
            "active_clients": len(service.client_preferences),
            "total_messages_sent": service.total_messages_sent,
            "total_bandwidth_saved": service.total_bandwidth_saved,
            "delta_states_active": len(service.delta_compressor.client_states),
            "encoder_stats_available": len(service.encoder.stats) > 0
        }
        
        # Determine overall health
        is_healthy = (
            msgpack_available and
            service.optimization_enabled
        )
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "data": health_status,
            "recommendations": [] if is_healthy else [
                "Install msgpack package for binary encoding support" if not msgpack_available else None,
                "Enable optimizations for better performance" if not service.optimization_enabled else None
            ]
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/client/{client_id}")
async def cleanup_client_optimization_data(client_id: str) -> Dict[str, str]:
    """Clean up optimization data for a specific client."""
    try:
        service = get_optimized_websocket_service()
        service.cleanup_client(client_id)
        
        return {
            "status": "success",
            "message": f"Cleaned up optimization data for client {client_id}"
        }
    
    except Exception as e:
        logger.error(f"Failed to cleanup client data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/benchmark")
async def run_optimization_benchmark(
    background_tasks: BackgroundTasks,
    message_count: int = Query(default=1000, ge=100, le=10000, description="Number of test messages"),
    message_size: int = Query(default=1024, ge=100, le=10240, description="Size of test messages in bytes")
) -> Dict[str, Any]:
    """Run a benchmark to measure optimization performance."""
    try:
        def run_benchmark():
            """Background task to run the benchmark."""
            import time
            import json
            import random
            import string
            
            service = get_optimized_websocket_service()
            
            # Generate test data
            def generate_test_message(size: int) -> Dict[str, Any]:
                # Create a message with specified size
                content = ''.join(random.choices(string.ascii_letters + string.digits, k=size//4))
                return {
                    'id': f"test_{random.randint(1000, 9999)}",
                    'type': 'simulation_update',
                    'simulation_id': 'benchmark_simulation',
                    'timestamp': time.time(),
                    'data': {
                        'generation': random.randint(1, 1000),
                        'population': random.randint(100, 10000),
                        'content': content
                    }
                }
            
            results = {
                'message_count': message_count,
                'message_size': message_size,
                'optimization_levels': {}
            }
            
            # Test each optimization level
            for level in MessageOptimizationLevel:
                start_time = time.time()
                total_original_size = 0
                total_optimized_size = 0
                
                for i in range(message_count):
                    test_message = generate_test_message(message_size)
                    
                    # Encode with current optimization level
                    encoded_data, metrics = service.encoder.encode_message(test_message, level)
                    
                    total_original_size += metrics.original_json_size
                    total_optimized_size += metrics.optimized_size
                
                end_time = time.time()
                
                results['optimization_levels'][level.value] = {
                    'total_original_size': total_original_size,
                    'total_optimized_size': total_optimized_size,
                    'compression_ratio': total_optimized_size / total_original_size if total_original_size > 0 else 1.0,
                    'bandwidth_saved': total_original_size - total_optimized_size,
                    'encoding_time': end_time - start_time,
                    'messages_per_second': message_count / (end_time - start_time)
                }
            
            # Store results for later retrieval
            service._benchmark_results = results
            logger.info(f"Benchmark completed: {results}")
        
        # Start benchmark in background
        background_tasks.add_task(run_benchmark)
        
        return {
            "status": "success",
            "message": "Benchmark started in background",
            "parameters": {
                "message_count": message_count,
                "message_size": message_size
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to start benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark/results")
async def get_benchmark_results() -> Dict[str, Any]:
    """Get the latest benchmark results."""
    try:
        service = get_optimized_websocket_service()
        
        if not hasattr(service, '_benchmark_results'):
            return {
                "status": "no_results",
                "message": "No benchmark results available. Run /benchmark first."
            }
        
        return {
            "status": "success",
            "data": service._benchmark_results
        }
    
    except Exception as e:
        logger.error(f"Failed to get benchmark results: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 