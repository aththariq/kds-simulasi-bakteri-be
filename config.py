"""
Configuration settings for the bacterial simulation API.
"""

import os
from typing import List


class Settings:
    """Application settings with default values."""
    
    # API Settings
    api_title: str = "Bacterial Resistance Simulation API"
    api_description: str = "API for simulating bacterial antibiotic resistance evolution"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # CORS Settings
    allowed_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    allowed_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allowed_headers: List[str] = ["*"]
    allow_credentials: bool = True
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # Simulation Settings
    max_population_size: int = 100000
    max_simulation_time: int = 1000
    default_mutation_rate: float = 0.001
    default_selection_pressure: float = 0.1


# Create a global settings instance
settings = Settings() 