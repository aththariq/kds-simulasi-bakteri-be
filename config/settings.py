"""
Application settings and configuration.
"""

import os
from typing import List

class Settings:
    """Application settings"""
    
    # API Configuration
    api_title: str = "Bacterial Resistance Simulation API"
    api_description: str = "API for simulating bacterial resistance evolution"
    api_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Server Configuration
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    reload: bool = os.getenv("RELOAD", "true").lower() == "true"
    
    # CORS Configuration
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]
    allow_credentials: bool = True
    allowed_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allowed_headers: List[str] = ["*"]
    
    # Database Configuration
    mongodb_url: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    database_name: str = os.getenv("DATABASE_NAME", "bacterial_simulation")
    
    # Redis Configuration (if needed)
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Performance Configuration
    max_concurrent_simulations: int = int(os.getenv("MAX_CONCURRENT_SIMULATIONS", "5"))
    simulation_timeout: int = int(os.getenv("SIMULATION_TIMEOUT", "3600"))  # 1 hour
    
    # Logging Configuration
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

# Create global settings instance
settings = Settings() 