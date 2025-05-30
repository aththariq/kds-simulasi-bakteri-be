"""
Main entry point for the bacterial simulation backend.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import time
import logging
from config import settings
from routes.simulation import router as simulation_router
from routes.websocket import router as websocket_router
from routes.state import router as state_router
from routes.performance import router as performance_router
from routes.results import router as results_router
from utils.data_transform import DataTransformer, ResponseBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    debug=settings.debug
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=settings.allow_credentials,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Enhanced error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with formatted error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ResponseBuilder.error(
            message=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        )
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    return JSONResponse(
        status_code=422,
        content=ResponseBuilder.error(
            message="Validation error",
            error_code="VALIDATION_ERROR",
            details=exc.errors()
        )
    )

@app.exception_handler(StarletteHTTPException)
async def starlette_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle Starlette HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ResponseBuilder.error(
            message=exc.detail,
            error_code=f"STARLETTE_{exc.status_code}"
        )
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ResponseBuilder.error(
            message="Internal server error",
            error_code="INTERNAL_ERROR"
        )
    )

# Include routers
app.include_router(simulation_router)
app.include_router(websocket_router)
app.include_router(state_router)
app.include_router(performance_router)
app.include_router(results_router)

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint returning API information."""
    return ResponseBuilder.success(
        data={
            "message": "Bacterial Resistance Simulation API",
            "version": settings.api_version,
            "status": "running",
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "websocket_endpoints": {
                "simulation": "/ws/simulation",
                "global": "/ws/global"
            },
            "performance_endpoints": {
                "metrics": "/api/performance/metrics",
                "memory": "/api/performance/memory",
                "benchmark": "/api/performance/benchmark/load-test",
                "health": "/api/performance/health"
            },
            "results_endpoints": {
                "health": "/api/results/health",
                "collect": "/api/results/collect",
                "simulations": "/api/results/simulations",
                "analysis": "/api/results/simulations/{id}/analysis",
                "reports": "/api/results/simulations/{id}/report",
                "export": "/api/results/simulations/{id}/export"
            }
        },
        message="API is running successfully"
    )

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return ResponseBuilder.success(
        data={
            "status": "healthy",
            "version": settings.api_version,
            "timestamp": time.time(),
            "performance_optimization": "enabled"
        },
        message="Service is healthy"
    )

@app.get("/info", tags=["Info"])
async def api_info():
    """Detailed API information endpoint."""
    return ResponseBuilder.success(
        data={
            "title": settings.api_title,
            "description": settings.api_description,
            "version": settings.api_version,
            "debug_mode": settings.debug,
            "endpoints": {
                "simulation": "/api/simulations",
                "websocket": "/ws",
                "state": "/api/state",
                "performance": "/api/performance",
                "health": "/health",
                "docs": "/docs"
            },
            "features": [
                "Real-time simulation streaming",
                "WebSocket support",
                "Rate limiting",
                "Input validation",
                "Error handling",
                "Performance optimization",
                "Memory profiling",
                "Load testing",
                "Benchmarking",
                "Regression testing"
            ]
        },
        message="API information retrieved successfully"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    ) 