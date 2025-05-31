from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from bson import ObjectId
from enum import Enum

class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class SimulationStatus(str, Enum):
    """Simulation status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BacteriumData(BaseModel):
    """Individual bacterium data model"""
    id: str = Field(..., description="Unique bacterium identifier")
    fitness: float = Field(..., ge=0.0, description="Fitness value")
    resistance_genes: List[str] = Field(default_factory=list, description="List of resistance genes")
    position_x: Optional[float] = Field(None, description="X coordinate in spatial grid")
    position_y: Optional[float] = Field(None, description="Y coordinate in spatial grid")
    generation: int = Field(..., ge=0, description="Generation number")
    parent_id: Optional[str] = Field(None, description="Parent bacterium ID")
    mutation_count: int = Field(default=0, ge=0, description="Number of mutations")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "bact_001",
                "fitness": 0.85,
                "resistance_genes": ["ampR", "tetR"],
                "position_x": 10.5,
                "position_y": 15.2,
                "generation": 5,
                "parent_id": "bact_parent_001",
                "mutation_count": 2
            }
        }

class PopulationSnapshot(BaseModel):
    """Population state at a specific generation"""
    generation: int = Field(..., ge=0, description="Generation number")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Snapshot timestamp")
    total_population: int = Field(..., ge=0, description="Total population count")
    resistant_population: int = Field(..., ge=0, description="Resistant bacteria count")
    susceptible_population: int = Field(..., ge=0, description="Susceptible bacteria count")
    average_fitness: float = Field(..., ge=0.0, description="Average population fitness")
    resistance_frequency: float = Field(..., ge=0.0, le=1.0, description="Resistance frequency")
    bacteria: List[BacteriumData] = Field(default_factory=list, description="Individual bacteria data")
    
    @validator('resistant_population', 'susceptible_population')
    def validate_population_counts(cls, v, values):
        if 'total_population' in values:
            total = values['total_population']
            if v > total:
                raise ValueError("Population subset cannot exceed total population")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "generation": 10,
                "total_population": 1000,
                "resistant_population": 350,
                "susceptible_population": 650,
                "average_fitness": 0.75,
                "resistance_frequency": 0.35,
                "bacteria": []
            }
        }

class SimulationParameters(BaseModel):
    """Simulation input parameters"""
    initial_population: int = Field(..., gt=0, description="Initial population size")
    generations: int = Field(..., gt=0, description="Number of generations to simulate")
    mutation_rate: float = Field(..., ge=0.0, le=1.0, description="Mutation rate per generation")
    antibiotic_concentration: float = Field(..., ge=0.0, description="Antibiotic concentration")
    selection_pressure: float = Field(default=1.0, ge=0.0, description="Selection pressure strength")
    spatial_enabled: bool = Field(default=False, description="Enable spatial simulation")
    grid_size: Optional[int] = Field(None, gt=0, description="Spatial grid size")
    hgt_enabled: bool = Field(default=False, description="Enable horizontal gene transfer")
    hgt_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="HGT rate")
    
    @validator('grid_size')
    def validate_grid_size(cls, v, values):
        if values.get('spatial_enabled') and v is None:
            raise ValueError("Grid size required when spatial simulation is enabled")
        return v
    
    @validator('hgt_rate')
    def validate_hgt_rate(cls, v, values):
        if values.get('hgt_enabled') and v is None:
            raise ValueError("HGT rate required when HGT is enabled")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "initial_population": 1000,
                "generations": 50,
                "mutation_rate": 0.01,
                "antibiotic_concentration": 0.5,
                "selection_pressure": 1.0,
                "spatial_enabled": True,
                "grid_size": 100,
                "hgt_enabled": True,
                "hgt_rate": 0.001
            }
        }

class SimulationMetadata(BaseModel):
    """Simulation metadata and configuration"""
    simulation_id: str = Field(..., description="Unique simulation identifier")
    name: Optional[str] = Field(None, description="User-defined simulation name")
    description: Optional[str] = Field(None, description="Simulation description")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    status: SimulationStatus = Field(default=SimulationStatus.PENDING, description="Simulation status")
    parameters: SimulationParameters = Field(..., description="Simulation parameters")
    total_runtime: Optional[float] = Field(None, ge=0.0, description="Total runtime in seconds")
    error_message: Optional[str] = Field(None, description="Error message if simulation failed")
    tags: List[str] = Field(default_factory=list, description="User-defined tags")
    
    class Config:
        schema_extra = {
            "example": {
                "simulation_id": "sim_20231201_001",
                "name": "High Mutation Rate Test",
                "description": "Testing bacterial evolution with increased mutation rate",
                "status": "completed",
                "parameters": {
                    "initial_population": 1000,
                    "generations": 50,
                    "mutation_rate": 0.02,
                    "antibiotic_concentration": 0.5
                },
                "total_runtime": 45.6,
                "tags": ["high_mutation", "test"]
            }
        }

class SimulationResults(BaseModel):
    """Complete simulation results"""
    simulation_id: str = Field(..., description="Simulation identifier")
    metadata: SimulationMetadata = Field(..., description="Simulation metadata")
    population_history: List[PopulationSnapshot] = Field(default_factory=list, description="Population evolution history")
    final_statistics: Dict[str, Any] = Field(default_factory=dict, description="Final simulation statistics")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "simulation_id": "sim_20231201_001",
                "metadata": {},
                "population_history": [],
                "final_statistics": {
                    "final_population": 1200,
                    "final_resistance_frequency": 0.45,
                    "generations_to_resistance": 15,
                    "extinction_events": 0
                },
                "performance_metrics": {
                    "avg_generation_time": 0.85,
                    "memory_usage_mb": 125.5,
                    "cpu_time_seconds": 42.3
                }
            }
        }

# MongoDB Collection Models
class SimulationDocument(BaseModel):
    """MongoDB document model for simulations"""
    _id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    simulation_id: str = Field(..., description="Unique simulation identifier")
    metadata: SimulationMetadata = Field(..., description="Simulation metadata")
    results: Optional[SimulationResults] = Field(None, description="Simulation results")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class PopulationDocument(BaseModel):
    """MongoDB document model for population snapshots"""
    _id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    simulation_id: str = Field(..., description="Associated simulation ID")
    snapshot: PopulationSnapshot = Field(..., description="Population snapshot data")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class BacteriumDocument(BaseModel):
    """MongoDB document model for individual bacteria"""
    _id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    simulation_id: str = Field(..., description="Associated simulation ID")
    generation: int = Field(..., ge=0, description="Generation number")
    bacterium: BacteriumData = Field(..., description="Bacterium data")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# Index definitions for MongoDB collections
SIMULATION_INDEXES = [
    [("simulation_id", 1)],  # Unique index on simulation_id
    [("metadata.status", 1)],  # Index on status for filtering
    [("metadata.created_at", -1)],  # Index on creation date for sorting
    [("metadata.tags", 1)],  # Index on tags for filtering
]

POPULATION_INDEXES = [
    [("simulation_id", 1), ("snapshot.generation", 1)],  # Compound index for queries
    [("simulation_id", 1)],  # Index on simulation_id
    [("snapshot.generation", 1)],  # Index on generation
    [("created_at", -1)],  # Index on creation date
]

BACTERIUM_INDEXES = [
    [("simulation_id", 1), ("generation", 1)],  # Compound index
    [("simulation_id", 1)],  # Index on simulation_id
    [("generation", 1)],  # Index on generation
    [("bacterium.id", 1)],  # Index on bacterium ID
    [("bacterium.fitness", -1)],  # Index on fitness for sorting
] 