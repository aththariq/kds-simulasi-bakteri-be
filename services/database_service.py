from datetime import datetime
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError, BulkWriteError
from bson import ObjectId
import logging

from models.database_models import (
    SimulationDocument, PopulationDocument, BacteriumDocument,
    SimulationMetadata, PopulationSnapshot, BacteriumData,
    SimulationResults, SimulationStatus
)
from utils.db_connection import DatabaseManager, DatabaseOperationError
from config.database import DatabaseConfig

logger = logging.getLogger(__name__)

class DatabaseService:
    """Service for MongoDB create operations"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.simulations: Optional[AsyncIOMotorCollection] = None
        self.populations: Optional[AsyncIOMotorCollection] = None
        self.bacteria: Optional[AsyncIOMotorCollection] = None
    
    async def initialize(self):
        """Initialize database connection and collections"""
        try:
            self.db = await self.db_manager.get_database()
            self.simulations = self.db.simulations
            self.populations = self.db.populations
            self.bacteria = self.db.bacteria
            
            # Create indexes
            await self._create_indexes()
            logger.info("Database service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database service: {e}")
            raise DatabaseOperationError(f"Database initialization failed: {e}")
    
    async def _create_indexes(self):
        """Create database indexes for optimal performance"""
        try:
            # Simulation collection indexes
            await self.simulations.create_index("simulation_id", unique=True)
            await self.simulations.create_index("metadata.status")
            await self.simulations.create_index([("metadata.created_at", -1)])
            await self.simulations.create_index("metadata.tags")
            
            # Population collection indexes
            await self.populations.create_index([("simulation_id", 1), ("snapshot.generation", 1)])
            await self.populations.create_index("simulation_id")
            await self.populations.create_index("snapshot.generation")
            await self.populations.create_index([("created_at", -1)])
            
            # Bacterium collection indexes
            await self.bacteria.create_index([("simulation_id", 1), ("generation", 1)])
            await self.bacteria.create_index("simulation_id")
            await self.bacteria.create_index("generation")
            await self.bacteria.create_index("bacterium.id")
            await self.bacteria.create_index([("bacterium.fitness", -1)])
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    async def create_simulation(self, metadata: SimulationMetadata) -> str:
        """
        Create a new simulation entry
        
        Args:
            metadata: Simulation metadata
            
        Returns:
            str: MongoDB document ID
            
        Raises:
            DatabaseOperationError: If creation fails
        """
        try:
            # Validate simulation_id uniqueness
            existing = await self.simulations.find_one({"simulation_id": metadata.simulation_id})
            if existing:
                raise DatabaseOperationError(f"Simulation with ID {metadata.simulation_id} already exists")
            
            # Create simulation document
            simulation_doc = SimulationDocument(
                simulation_id=metadata.simulation_id,
                metadata=metadata,
                results=None
            )
            
            # Insert document
            result = await self.simulations.insert_one(simulation_doc.dict(by_alias=True))
            
            logger.info(f"Created simulation: {metadata.simulation_id}")
            return str(result.inserted_id)
            
        except DuplicateKeyError:
            raise DatabaseOperationError(f"Simulation {metadata.simulation_id} already exists")
        except Exception as e:
            logger.error(f"Failed to create simulation {metadata.simulation_id}: {e}")
            raise DatabaseOperationError(f"Simulation creation failed: {e}")
    
    async def save_population_snapshot(self, simulation_id: str, snapshot: PopulationSnapshot) -> str:
        """
        Save a population snapshot
        
        Args:
            simulation_id: Associated simulation ID
            snapshot: Population snapshot data
            
        Returns:
            str: MongoDB document ID
        """
        try:
            # Validate simulation exists
            await self._validate_simulation_exists(simulation_id)
            
            # Create population document
            population_doc = PopulationDocument(
                simulation_id=simulation_id,
                snapshot=snapshot
            )
            
            # Insert document
            result = await self.populations.insert_one(population_doc.dict(by_alias=True))
            
            logger.debug(f"Saved population snapshot for simulation {simulation_id}, generation {snapshot.generation}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to save population snapshot: {e}")
            raise DatabaseOperationError(f"Population snapshot save failed: {e}")
    
    async def save_bacteria_batch(self, simulation_id: str, generation: int, bacteria: List[BacteriumData]) -> List[str]:
        """
        Save multiple bacteria in a batch operation
        
        Args:
            simulation_id: Associated simulation ID
            generation: Generation number
            bacteria: List of bacteria data
            
        Returns:
            List[str]: List of MongoDB document IDs
        """
        try:
            if not bacteria:
                return []
            
            # Validate simulation exists
            await self._validate_simulation_exists(simulation_id)
            
            # Create bacterium documents
            bacterium_docs = [
                BacteriumDocument(
                    simulation_id=simulation_id,
                    generation=generation,
                    bacterium=bacterium
                ).dict(by_alias=True)
                for bacterium in bacteria
            ]
            
            # Batch insert
            result = await self.bacteria.insert_many(bacterium_docs, ordered=False)
            
            logger.debug(f"Saved {len(bacteria)} bacteria for simulation {simulation_id}, generation {generation}")
            return [str(oid) for oid in result.inserted_ids]
            
        except BulkWriteError as e:
            logger.error(f"Bulk write error saving bacteria: {e.details}")
            raise DatabaseOperationError(f"Bacteria batch save failed: {e}")
        except Exception as e:
            logger.error(f"Failed to save bacteria batch: {e}")
            raise DatabaseOperationError(f"Bacteria batch save failed: {e}")
    
    async def save_individual_bacterium(self, simulation_id: str, generation: int, bacterium: BacteriumData) -> str:
        """
        Save a single bacterium
        
        Args:
            simulation_id: Associated simulation ID
            generation: Generation number
            bacterium: Bacterium data
            
        Returns:
            str: MongoDB document ID
        """
        try:
            # Validate simulation exists
            await self._validate_simulation_exists(simulation_id)
            
            # Create bacterium document
            bacterium_doc = BacteriumDocument(
                simulation_id=simulation_id,
                generation=generation,
                bacterium=bacterium
            )
            
            # Insert document
            result = await self.bacteria.insert_one(bacterium_doc.dict(by_alias=True))
            
            logger.debug(f"Saved bacterium {bacterium.id} for simulation {simulation_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to save bacterium: {e}")
            raise DatabaseOperationError(f"Bacterium save failed: {e}")
    
    async def update_simulation_status(self, simulation_id: str, status: SimulationStatus, 
                                     error_message: Optional[str] = None) -> bool:
        """
        Update simulation status
        
        Args:
            simulation_id: Simulation ID
            status: New status
            error_message: Optional error message
            
        Returns:
            bool: True if updated successfully
        """
        try:
            update_data = {
                "metadata.status": status.value,
                "updated_at": datetime.utcnow()
            }
            
            if error_message:
                update_data["metadata.error_message"] = error_message
            
            result = await self.simulations.update_one(
                {"simulation_id": simulation_id},
                {"$set": update_data}
            )
            
            if result.matched_count == 0:
                raise DatabaseOperationError(f"Simulation {simulation_id} not found")
            
            logger.info(f"Updated simulation {simulation_id} status to {status.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update simulation status: {e}")
            raise DatabaseOperationError(f"Status update failed: {e}")
    
    async def save_simulation_results(self, simulation_id: str, results: SimulationResults) -> bool:
        """
        Save complete simulation results
        
        Args:
            simulation_id: Simulation ID
            results: Complete simulation results
            
        Returns:
            bool: True if saved successfully
        """
        try:
            # Update simulation document with results
            result = await self.simulations.update_one(
                {"simulation_id": simulation_id},
                {
                    "$set": {
                        "results": results.dict(),
                        "metadata.status": SimulationStatus.COMPLETED.value,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            if result.matched_count == 0:
                raise DatabaseOperationError(f"Simulation {simulation_id} not found")
            
            logger.info(f"Saved results for simulation {simulation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save simulation results: {e}")
            raise DatabaseOperationError(f"Results save failed: {e}")
    
    async def create_simulation_with_initial_data(self, metadata: SimulationMetadata, 
                                                initial_population: PopulationSnapshot) -> Dict[str, str]:
        """
        Create simulation with initial population data in a transaction-like operation
        
        Args:
            metadata: Simulation metadata
            initial_population: Initial population snapshot
            
        Returns:
            Dict[str, str]: Dictionary with simulation_id and population_id
        """
        try:
            # Create simulation
            simulation_doc_id = await self.create_simulation(metadata)
            
            # Save initial population
            population_doc_id = await self.save_population_snapshot(
                metadata.simulation_id, 
                initial_population
            )
            
            # Save individual bacteria if present
            bacteria_ids = []
            if initial_population.bacteria:
                bacteria_ids = await self.save_bacteria_batch(
                    metadata.simulation_id,
                    initial_population.generation,
                    initial_population.bacteria
                )
            
            logger.info(f"Created simulation {metadata.simulation_id} with initial data")
            
            return {
                "simulation_id": simulation_doc_id,
                "population_id": population_doc_id,
                "bacteria_count": len(bacteria_ids)
            }
            
        except Exception as e:
            # Attempt cleanup on failure
            try:
                await self._cleanup_failed_simulation(metadata.simulation_id)
            except:
                pass  # Cleanup failure is not critical
            
            logger.error(f"Failed to create simulation with initial data: {e}")
            raise DatabaseOperationError(f"Simulation creation with data failed: {e}")
    
    async def _validate_simulation_exists(self, simulation_id: str) -> bool:
        """Validate that a simulation exists"""
        exists = await self.simulations.find_one({"simulation_id": simulation_id})
        if not exists:
            raise DatabaseOperationError(f"Simulation {simulation_id} does not exist")
        return True
    
    async def _cleanup_failed_simulation(self, simulation_id: str):
        """Clean up data for a failed simulation creation"""
        try:
            # Remove simulation document
            await self.simulations.delete_one({"simulation_id": simulation_id})
            
            # Remove associated population snapshots
            await self.populations.delete_many({"simulation_id": simulation_id})
            
            # Remove associated bacteria
            await self.bacteria.delete_many({"simulation_id": simulation_id})
            
            logger.info(f"Cleaned up failed simulation {simulation_id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup simulation {simulation_id}: {e}")
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get database collection statistics"""
        try:
            stats = {}
            
            # Simulation collection stats
            sim_count = await self.simulations.count_documents({})
            stats["simulations"] = {
                "total_count": sim_count,
                "status_breakdown": {}
            }
            
            # Get status breakdown
            pipeline = [
                {"$group": {"_id": "$metadata.status", "count": {"$sum": 1}}}
            ]
            async for doc in self.simulations.aggregate(pipeline):
                stats["simulations"]["status_breakdown"][doc["_id"]] = doc["count"]
            
            # Population collection stats
            pop_count = await self.populations.count_documents({})
            stats["populations"] = {"total_count": pop_count}
            
            # Bacteria collection stats
            bact_count = await self.bacteria.count_documents({})
            stats["bacteria"] = {"total_count": bact_count}
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)} 