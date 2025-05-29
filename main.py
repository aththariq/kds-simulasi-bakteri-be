"""
Main entry point for the bacterial simulation backend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
from routes.simulation import router as simulation_router

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

# Include routers
app.include_router(simulation_router)

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Bacterial Resistance Simulation API",
        "version": settings.api_version,
        "status": "running",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.api_version}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload
    ) 