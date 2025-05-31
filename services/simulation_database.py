import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import DuplicateKeyError
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
from pydantic import ValidationError

from models.database_models import (
    SimulationDocument, 
    PopulationDocument, 
    BacteriumDocument,
    SimulationMetadata,
    SimulationResults,
    PopulationSnapshot,
    BacteriumData,
    SimulationStatus,
    PyObjectId
)
from utils.db_connection import DatabaseSession, ensure_database_connection, retry_on_connection_failure

logger = logging.getLogger(__name__)

def _convert_mongo_document(doc: dict) -> dict:
    """
    Convert MongoDB document field names to Pydantic model compatible format.
    
    Args:
        doc: Raw MongoDB document with _id field
        
    Returns:
        dict: Document with _id converted to id for Pydantic compatibility
    """
    if doc is None:
        return doc
    
    # Create a copy to avoid modifying the original
    converted_doc = doc.copy()
    
    # Convert _id to id for Pydantic models
    if '_id' in converted_doc:
        # Convert ObjectId to string for Pydantic compatibility
        object_id = converted_doc.pop('_id')
        logger.debug(f"Converting ObjectId: {object_id} (type: {type(object_id)})")
        
        # Force string conversion for ObjectId
        if isinstance(object_id, ObjectId):
            converted_doc['id'] = str(object_id)
            logger.debug(f"Converted ObjectId to string: {converted_doc['id']}")
        elif hasattr(object_id, '__str__'):
            converted_doc['id'] = str(object_id)
            logger.debug(f"Converted to string using __str__: {converted_doc['id']}")
        else:
            converted_doc['id'] = object_id
            logger.debug(f"Used as-is: {converted_doc['id']} (type: {type(converted_doc['id'])})")
    
    return converted_doc

class SimulationDatabaseError(Exception):
    """Custom exception for simulation database operations"""
    pass

class SimulationCreateOperations:
    """Database create operations for simulation data"""
    
    def __init__(self):
        self.collections = {
            'simulations': 'simulations',
            'populations': 'population_snapshots', 
            'bacteria': 'bacteria'
        }
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def create_simulation(
        self, 
        simulation_metadata: SimulationMetadata
    ) -> str:
        """
        Create a new simulation record in MongoDB
        
        Args:
            simulation_metadata: Complete simulation metadata including parameters
            
        Returns:
            str: The MongoDB ObjectId of the created simulation document
            
        Raises:
            SimulationDatabaseError: If simulation creation fails
            ValidationError: If data validation fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Create simulation document
                simulation_doc = SimulationDocument(
                    simulation_id=simulation_metadata.simulation_id,
                    metadata=simulation_metadata,
                    results=None  # Results will be added later
                )
                
                # Validate document before insertion
                doc_dict = simulation_doc.dict(by_alias=True, exclude={'id'})
                
                # Insert document
                result = await collection.insert_one(doc_dict)
                
                if not result.inserted_id:
                    raise SimulationDatabaseError("Failed to insert simulation document")
                
                logger.info(f"Created simulation {simulation_metadata.simulation_id} with ObjectId {result.inserted_id}")
                return str(result.inserted_id)
                
        except DuplicateKeyError as e:
            error_msg = f"Simulation with ID {simulation_metadata.simulation_id} already exists"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
            
        except ValidationError as e:
            error_msg = f"Simulation metadata validation failed: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Failed to create simulation {simulation_metadata.simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def save_simulation_results(
        self,
        simulation_id: str,
        results: SimulationResults
    ) -> bool:
        """
        Save or update simulation results
        
        Args:
            simulation_id: Unique simulation identifier
            results: Complete simulation results data
            
        Returns:
            bool: True if update was successful
            
        Raises:
            SimulationDatabaseError: If save operation fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Validate results data
                results_dict = results.dict()
                
                # Update simulation document with results
                update_result = await collection.update_one(
                    {"simulation_id": simulation_id},
                    {
                        "$set": {
                            "results": results_dict,
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                
                if update_result.matched_count == 0:
                    raise SimulationDatabaseError(f"Simulation {simulation_id} not found")
                
                if update_result.modified_count == 0:
                    logger.warning(f"No changes made to simulation {simulation_id} results")
                
                logger.info(f"Saved results for simulation {simulation_id}")
                return True
                
        except ValidationError as e:
            error_msg = f"Results validation failed for simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Failed to save results for simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def batch_insert_population_snapshots(
        self,
        simulation_id: str,
        snapshots: List[PopulationSnapshot]
    ) -> List[str]:
        """
        Batch insert population snapshots for a simulation
        
        Args:
            simulation_id: Unique simulation identifier
            snapshots: List of population snapshots to insert
            
        Returns:
            List[str]: List of inserted document ObjectIds
            
        Raises:
            SimulationDatabaseError: If batch insertion fails
        """
        try:
            if not snapshots:
                logger.warning(f"No snapshots to insert for simulation {simulation_id}")
                return []
            
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['populations']]
                
                # Prepare documents for batch insertion
                documents = []
                for snapshot in snapshots:
                    # Validate snapshot data
                    pop_doc = PopulationDocument(
                        simulation_id=simulation_id,
                        snapshot=snapshot
                    )
                    doc_dict = pop_doc.dict(by_alias=True, exclude={'id'})
                    documents.append(doc_dict)
                
                # Batch insert documents
                result = await collection.insert_many(documents, ordered=False)
                
                if not result.inserted_ids:
                    raise SimulationDatabaseError("Failed to insert population snapshots")
                
                inserted_count = len(result.inserted_ids)
                logger.info(f"Inserted {inserted_count} population snapshots for simulation {simulation_id}")
                
                return [str(obj_id) for obj_id in result.inserted_ids]
                
        except ValidationError as e:
            error_msg = f"Population snapshot validation failed for simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Failed to batch insert population snapshots for simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def batch_insert_bacteria(
        self,
        simulation_id: str,
        generation: int,
        bacteria: List[BacteriumData]
    ) -> List[str]:
        """
        Batch insert bacteria data for a specific generation
        
        Args:
            simulation_id: Unique simulation identifier
            generation: Generation number
            bacteria: List of bacteria to insert
            
        Returns:
            List[str]: List of inserted document ObjectIds
            
        Raises:
            SimulationDatabaseError: If batch insertion fails
        """
        try:
            if not bacteria:
                logger.warning(f"No bacteria to insert for simulation {simulation_id} generation {generation}")
                return []
            
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['bacteria']]
                
                # Prepare documents for batch insertion
                documents = []
                for bacterium in bacteria:
                    # Validate bacterium data
                    bact_doc = BacteriumDocument(
                        simulation_id=simulation_id,
                        generation=generation,
                        bacterium=bacterium
                    )
                    doc_dict = bact_doc.dict(by_alias=True, exclude={'id'})
                    documents.append(doc_dict)
                
                # Batch insert documents
                result = await collection.insert_many(documents, ordered=False)
                
                if not result.inserted_ids:
                    raise SimulationDatabaseError("Failed to insert bacteria data")
                
                inserted_count = len(result.inserted_ids)
                logger.info(f"Inserted {inserted_count} bacteria for simulation {simulation_id} generation {generation}")
                
                return [str(obj_id) for obj_id in result.inserted_ids]
                
        except ValidationError as e:
            error_msg = f"Bacteria validation failed for simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Failed to batch insert bacteria for simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def update_simulation_status(
        self,
        simulation_id: str,
        status: SimulationStatus,
        error_message: Optional[str] = None,
        runtime: Optional[float] = None
    ) -> bool:
        """
        Update simulation status and metadata
        
        Args:
            simulation_id: Unique simulation identifier
            status: New simulation status
            error_message: Error message if status is FAILED
            runtime: Total runtime if status is COMPLETED
            
        Returns:
            bool: True if update was successful
            
        Raises:
            SimulationDatabaseError: If update fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Prepare update document
                update_doc = {
                    "metadata.status": status.value,
                    "metadata.updated_at": datetime.utcnow()
                }
                
                if error_message is not None:
                    update_doc["metadata.error_message"] = error_message
                
                if runtime is not None:
                    update_doc["metadata.total_runtime"] = runtime
                
                # Update simulation document
                result = await collection.update_one(
                    {"simulation_id": simulation_id},
                    {"$set": update_doc}
                )
                
                if result.matched_count == 0:
                    raise SimulationDatabaseError(f"Simulation {simulation_id} not found")
                
                logger.info(f"Updated simulation {simulation_id} status to {status.value}")
                return True
                
        except Exception as e:
            error_msg = f"Failed to update simulation {simulation_id} status: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def create_simulation_with_initial_data(
        self,
        simulation_metadata: SimulationMetadata,
        initial_population: Optional[PopulationSnapshot] = None
    ) -> str:
        """
        Create a simulation with optional initial population data in a transaction
        
        Args:
            simulation_metadata: Complete simulation metadata
            initial_population: Optional initial population snapshot
            
        Returns:
            str: The MongoDB ObjectId of the created simulation
            
        Raises:
            SimulationDatabaseError: If creation fails
        """
        try:
            # Create the simulation first
            simulation_object_id = await self.create_simulation(simulation_metadata)
            
            # If initial population data is provided, insert it
            if initial_population:
                await self.batch_insert_population_snapshots(
                    simulation_metadata.simulation_id,
                    [initial_population]
                )
                
                # If the snapshot contains bacteria data, insert that too
                if initial_population.bacteria:
                    await self.batch_insert_bacteria(
                        simulation_metadata.simulation_id,
                        initial_population.generation,
                        initial_population.bacteria
                    )
            
            logger.info(f"Successfully created simulation {simulation_metadata.simulation_id} with initial data")
            return simulation_object_id
            
        except Exception as e:
            # If something fails, we should clean up any partial data
            # This is a simplified approach - in production, you might want proper transaction support
            error_msg = f"Failed to create simulation with initial data: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e

class SimulationReadOperations:
    """Database read operations for simulation data"""
    
    def __init__(self):
        self.collections = {
            'simulations': 'simulations',
            'populations': 'population_snapshots',
            'bacteria': 'bacteria'
        }
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def get_simulation_by_id(self, simulation_id: str) -> Optional[SimulationDocument]:
        """
        Retrieve a single simulation by its ID
        
        Args:
            simulation_id: Unique simulation identifier
            
        Returns:
            SimulationDocument: The simulation document or None if not found
            
        Raises:
            SimulationDatabaseError: If retrieval fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Find simulation by ID
                doc = await collection.find_one({"simulation_id": simulation_id})
                
                if not doc:
                    logger.info(f"Simulation {simulation_id} not found")
                    return None
                
                # Convert MongoDB document format to Pydantic-compatible format
                doc = _convert_mongo_document(doc)
                
                # Convert to Pydantic model
                simulation_doc = SimulationDocument(**doc)
                logger.debug(f"Retrieved simulation {simulation_id}")
                return simulation_doc
                
        except ValidationError as e:
            error_msg = f"Failed to validate simulation document {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Failed to retrieve simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def get_simulation_by_object_id(self, object_id: str) -> Optional[SimulationDocument]:
        """
        Retrieve a single simulation by its MongoDB ObjectId
        
        Args:
            object_id: MongoDB ObjectId as string
            
        Returns:
            SimulationDocument: The simulation document or None if not found
            
        Raises:
            SimulationDatabaseError: If retrieval fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Convert string to ObjectId
                if not ObjectId.is_valid(object_id):
                    raise SimulationDatabaseError(f"Invalid ObjectId format: {object_id}")
                
                # Find simulation by ObjectId
                doc = await collection.find_one({"_id": ObjectId(object_id)})
                
                if not doc:
                    logger.info(f"Simulation with ObjectId {object_id} not found")
                    return None
                
                # Convert MongoDB document format to Pydantic-compatible format
                doc = _convert_mongo_document(doc)
                
                # Convert to Pydantic model
                simulation_doc = SimulationDocument(**doc)
                logger.debug(f"Retrieved simulation by ObjectId {object_id}")
                return simulation_doc
                
        except ValidationError as e:
            error_msg = f"Failed to validate simulation document {object_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Failed to retrieve simulation by ObjectId {object_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def list_simulations(
        self,
        status_filter: Optional[SimulationStatus] = None,
        tags_filter: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> Tuple[List[SimulationDocument], int]:
        """
        List simulations with pagination and filtering
        
        Args:
            status_filter: Filter by simulation status
            tags_filter: Filter by tags (any of the provided tags)
            limit: Maximum number of results to return
            offset: Number of results to skip (for pagination)
            sort_by: Field to sort by (created_at, updated_at, simulation_id, name)
            sort_order: Sort order (asc or desc)
            
        Returns:
            Tuple[List[SimulationDocument], int]: List of simulations and total count
            
        Raises:
            SimulationDatabaseError: If retrieval fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Build query filter
                query = {}
                if status_filter:
                    query["metadata.status"] = status_filter.value
                
                if tags_filter:
                    query["metadata.tags"] = {"$in": tags_filter}
                
                # Validate sort parameters
                valid_sort_fields = ["created_at", "updated_at", "simulation_id", "metadata.name"]
                if sort_by not in valid_sort_fields:
                    sort_by = "created_at"
                
                sort_direction = DESCENDING if sort_order.lower() == "desc" else ASCENDING
                
                # Get total count for pagination
                total_count = await collection.count_documents(query)
                
                # Execute query with pagination and sorting
                cursor = collection.find(query).sort(sort_by, sort_direction).skip(offset).limit(limit)
                docs = await cursor.to_list(length=limit)
                
                # Convert to Pydantic models
                simulations = []
                for doc in docs:
                    try:
                        # Convert MongoDB document format to Pydantic-compatible format
                        doc = _convert_mongo_document(doc)
                        
                        # Convert to Pydantic model
                        simulation_doc = SimulationDocument(**doc)
                        simulations.append(simulation_doc)
                    except ValidationError as e:
                        logger.warning(f"Skipping invalid simulation document: {e}")
                        continue
                
                logger.info(f"Retrieved {len(simulations)} simulations (total: {total_count})")
                return simulations, total_count
                
        except Exception as e:
            error_msg = f"Failed to list simulations: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def get_population_history(
        self,
        simulation_id: str,
        generation_start: Optional[int] = None,
        generation_end: Optional[int] = None,
        include_bacteria: bool = False
    ) -> List[PopulationSnapshot]:
        """
        Retrieve population history for a simulation
        
        Args:
            simulation_id: Unique simulation identifier
            generation_start: Starting generation (inclusive)
            generation_end: Ending generation (inclusive)
            include_bacteria: Whether to include individual bacteria data
            
        Returns:
            List[PopulationSnapshot]: Population snapshots in chronological order
            
        Raises:
            SimulationDatabaseError: If retrieval fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['populations']]
                
                # Build query filter
                query = {"simulation_id": simulation_id}
                
                if generation_start is not None or generation_end is not None:
                    generation_filter = {}
                    if generation_start is not None:
                        generation_filter["$gte"] = generation_start
                    if generation_end is not None:
                        generation_filter["$lte"] = generation_end
                    query["snapshot.generation"] = generation_filter
                
                # Execute query sorted by generation
                cursor = collection.find(query).sort("snapshot.generation", ASCENDING)
                docs = await cursor.to_list(length=None)
                
                # Convert to Pydantic models
                snapshots = []
                for doc in docs:
                    try:
                        # Convert MongoDB document format to Pydantic-compatible format
                        doc = _convert_mongo_document(doc)
                        pop_doc = PopulationDocument(**doc)
                        snapshot = pop_doc.snapshot
                        
                        # Optionally load bacteria data
                        if include_bacteria and not snapshot.bacteria:
                            bacteria = await self.get_bacteria_by_generation(
                                simulation_id, 
                                snapshot.generation
                            )
                            snapshot.bacteria = bacteria
                        
                        snapshots.append(snapshot)
                    except ValidationError as e:
                        logger.warning(f"Skipping invalid population document: {e}")
                        continue
                
                logger.debug(f"Retrieved {len(snapshots)} population snapshots for {simulation_id}")
                return snapshots
                
        except Exception as e:
            error_msg = f"Failed to retrieve population history for {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def get_bacteria_by_generation(
        self,
        simulation_id: str,
        generation: int,
        limit: Optional[int] = None,
        fitness_threshold: Optional[float] = None
    ) -> List[BacteriumData]:
        """
        Retrieve bacteria for a specific generation
        
        Args:
            simulation_id: Unique simulation identifier
            generation: Generation number
            limit: Maximum number of bacteria to return
            fitness_threshold: Minimum fitness threshold
            
        Returns:
            List[BacteriumData]: List of bacteria for the generation
            
        Raises:
            SimulationDatabaseError: If retrieval fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['bacteria']]
                
                # Build query filter
                query = {
                    "simulation_id": simulation_id,
                    "generation": generation
                }
                
                if fitness_threshold is not None:
                    query["bacterium.fitness"] = {"$gte": fitness_threshold}
                
                # Execute query with optional limit
                cursor = collection.find(query).sort("bacterium.fitness", DESCENDING)
                
                if limit:
                    cursor = cursor.limit(limit)
                
                docs = await cursor.to_list(length=limit)
                
                # Convert to Pydantic models
                bacteria = []
                for doc in docs:
                    try:
                        # Convert MongoDB document format to Pydantic-compatible format
                        doc = _convert_mongo_document(doc)
                        bact_doc = BacteriumDocument(**doc)
                        bacteria.append(bact_doc.bacterium)
                    except ValidationError as e:
                        logger.warning(f"Skipping invalid bacterium document: {e}")
                        continue
                
                logger.debug(f"Retrieved {len(bacteria)} bacteria for {simulation_id} generation {generation}")
                return bacteria
                
        except Exception as e:
            error_msg = f"Failed to retrieve bacteria for {simulation_id} generation {generation}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def get_simulation_statistics(self, simulation_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a simulation
        
        Args:
            simulation_id: Unique simulation identifier
            
        Returns:
            Dict[str, Any]: Statistics including generation count, population metrics, etc.
            
        Raises:
            SimulationDatabaseError: If retrieval fails
        """
        try:
            async with DatabaseSession() as db:
                simulations_collection = db[self.collections['simulations']]
                populations_collection = db[self.collections['populations']]
                bacteria_collection = db[self.collections['bacteria']]
                
                # Get simulation metadata
                simulation_doc = await simulations_collection.find_one({"simulation_id": simulation_id})
                if not simulation_doc:
                    raise SimulationDatabaseError(f"Simulation {simulation_id} not found")
                
                # Population statistics
                population_stats = await populations_collection.aggregate([
                    {"$match": {"simulation_id": simulation_id}},
                    {"$group": {
                        "_id": None,
                        "total_generations": {"$sum": 1},
                        "min_population": {"$min": "$snapshot.total_population"},
                        "max_population": {"$max": "$snapshot.total_population"},
                        "avg_population": {"$avg": "$snapshot.total_population"},
                        "final_resistance_frequency": {"$last": "$snapshot.resistance_frequency"},
                        "max_resistance_frequency": {"$max": "$snapshot.resistance_frequency"}
                    }}
                ]).to_list(length=1)
                
                # Bacteria statistics
                bacteria_stats = await bacteria_collection.aggregate([
                    {"$match": {"simulation_id": simulation_id}},
                    {"$group": {
                        "_id": None,
                        "total_bacteria_tracked": {"$sum": 1},
                        "avg_fitness": {"$avg": "$bacterium.fitness"},
                        "max_fitness": {"$max": "$bacterium.fitness"},
                        "min_fitness": {"$min": "$bacterium.fitness"},
                        "avg_mutations": {"$avg": "$bacterium.mutation_count"}
                    }}
                ]).to_list(length=1)
                
                # Compile statistics
                stats = {
                    "simulation_id": simulation_id,
                    "simulation_name": simulation_doc.get("metadata", {}).get("name"),
                    "status": simulation_doc.get("metadata", {}).get("status"),
                    "created_at": simulation_doc.get("created_at"),
                    "updated_at": simulation_doc.get("updated_at"),
                    "population_statistics": population_stats[0] if population_stats else {},
                    "bacteria_statistics": bacteria_stats[0] if bacteria_stats else {},
                    "has_results": simulation_doc.get("results") is not None
                }
                
                # Remove MongoDB-specific fields
                for key in ["_id"]:
                    stats["population_statistics"].pop(key, None)
                    stats["bacteria_statistics"].pop(key, None)
                
                logger.debug(f"Compiled statistics for simulation {simulation_id}")
                return stats
                
        except Exception as e:
            error_msg = f"Failed to get statistics for simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def search_simulations(
        self,
        search_query: str,
        search_fields: List[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> Tuple[List[SimulationDocument], int]:
        """
        Search simulations by text query
        
        Args:
            search_query: Text to search for
            search_fields: Fields to search in (name, description, tags, simulation_id)
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            Tuple[List[SimulationDocument], int]: Search results and total count
            
        Raises:
            SimulationDatabaseError: If search fails
        """
        try:
            if not search_fields:
                search_fields = ["metadata.name", "metadata.description", "metadata.tags", "simulation_id"]
            
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Build search query using regex for flexible matching
                search_conditions = []
                for field in search_fields:
                    if field == "metadata.tags":
                        # For tags, use exact match
                        search_conditions.append({field: {"$in": [search_query]}})
                    else:
                        # For text fields, use case-insensitive regex
                        search_conditions.append({field: {"$regex": search_query, "$options": "i"}})
                
                query = {"$or": search_conditions}
                
                # Get total count
                total_count = await collection.count_documents(query)
                
                # Execute search query
                cursor = collection.find(query).sort("metadata.created_at", DESCENDING).skip(offset).limit(limit)
                docs = await cursor.to_list(length=limit)
                
                # Convert to Pydantic models
                simulations = []
                for doc in docs:
                    try:
                        # Convert MongoDB document format to Pydantic-compatible format
                        doc = _convert_mongo_document(doc)
                        
                        # Convert to Pydantic model
                        simulation_doc = SimulationDocument(**doc)
                        simulations.append(simulation_doc)
                    except ValidationError as e:
                        logger.warning(f"Skipping invalid simulation document in search: {e}")
                        continue
                
                logger.info(f"Search for '{search_query}' found {len(simulations)} results (total: {total_count})")
                return simulations, total_count
                
        except Exception as e:
            error_msg = f"Failed to search simulations with query '{search_query}': {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def get_recent_simulations(self, limit: int = 10) -> List[SimulationDocument]:
        """
        Get the most recently created simulations
        
        Args:
            limit: Maximum number of recent simulations to return
            
        Returns:
            List[SimulationDocument]: Recent simulations ordered by creation date
            
        Raises:
            SimulationDatabaseError: If retrieval fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Get recent simulations
                cursor = collection.find().sort("metadata.created_at", DESCENDING).limit(limit)
                docs = await cursor.to_list(length=limit)
                
                # Convert to Pydantic models
                simulations = []
                for doc in docs:
                    try:
                        # Convert MongoDB document format to Pydantic-compatible format
                        doc = _convert_mongo_document(doc)
                        
                        # Convert to Pydantic model
                        simulation_doc = SimulationDocument(**doc)
                        simulations.append(simulation_doc)
                    except ValidationError as e:
                        logger.warning(f"Skipping invalid simulation document: {e}")
                        continue
                
                logger.debug(f"Retrieved {len(simulations)} recent simulations")
                return simulations
                
        except Exception as e:
            error_msg = f"Failed to retrieve recent simulations: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def simulation_exists(self, simulation_id: str) -> bool:
        """
        Check if a simulation exists
        
        Args:
            simulation_id: Unique simulation identifier
            
        Returns:
            bool: True if simulation exists, False otherwise
            
        Raises:
            SimulationDatabaseError: If check fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Check if simulation exists
                exists = await collection.find_one({"simulation_id": simulation_id}, {"_id": 1})
                result = exists is not None
                
                logger.debug(f"Simulation {simulation_id} exists: {result}")
                return result
                
        except Exception as e:
            error_msg = f"Failed to check if simulation {simulation_id} exists: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e

class SimulationUpdateOperations:
    """Database update operations for simulation data"""
    
    def __init__(self):
        self.collections = {
            'simulations': 'simulations',
            'populations': 'population_snapshots', 
            'bacteria': 'bacteria'
        }
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def update_simulation_metadata(
        self,
        simulation_id: str,
        metadata_updates: Dict[str, Any]
    ) -> bool:
        """
        Update simulation metadata fields
        
        Args:
            simulation_id: Unique simulation identifier
            metadata_updates: Dictionary of metadata fields to update
            
        Returns:
            bool: True if update was successful
            
        Raises:
            SimulationDatabaseError: If update operation fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Prepare update document with metadata fields
                update_doc = {}
                for key, value in metadata_updates.items():
                    update_doc[f"metadata.{key}"] = value
                
                # Always update the updated_at timestamp
                update_doc["metadata.updated_at"] = datetime.utcnow()
                update_doc["updated_at"] = datetime.utcnow()
                
                # Perform update
                result = await collection.update_one(
                    {"simulation_id": simulation_id},
                    {"$set": update_doc}
                )
                
                if result.matched_count == 0:
                    raise SimulationDatabaseError(f"Simulation {simulation_id} not found")
                
                logger.info(f"Updated metadata for simulation {simulation_id}")
                return result.modified_count > 0
                
        except Exception as e:
            error_msg = f"Failed to update metadata for simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def update_simulation_parameters(
        self,
        simulation_id: str,
        parameter_updates: Dict[str, Any]
    ) -> bool:
        """
        Update simulation parameters
        
        Args:
            simulation_id: Unique simulation identifier
            parameter_updates: Dictionary of parameter fields to update
            
        Returns:
            bool: True if update was successful
            
        Raises:
            SimulationDatabaseError: If update operation fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Prepare update document with parameter fields
                update_doc = {}
                for key, value in parameter_updates.items():
                    update_doc[f"metadata.parameters.{key}"] = value
                
                # Update timestamps
                update_doc["metadata.updated_at"] = datetime.utcnow()
                update_doc["updated_at"] = datetime.utcnow()
                
                # Perform update
                result = await collection.update_one(
                    {"simulation_id": simulation_id},
                    {"$set": update_doc}
                )
                
                if result.matched_count == 0:
                    raise SimulationDatabaseError(f"Simulation {simulation_id} not found")
                
                logger.info(f"Updated parameters for simulation {simulation_id}")
                return result.modified_count > 0
                
        except Exception as e:
            error_msg = f"Failed to update parameters for simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def append_population_snapshot(
        self,
        simulation_id: str,
        snapshot: PopulationSnapshot
    ) -> str:
        """
        Append a new population snapshot to existing simulation results
        
        Args:
            simulation_id: Unique simulation identifier
            snapshot: Population snapshot to append
            
        Returns:
            str: ObjectId of the created population document
            
        Raises:
            SimulationDatabaseError: If append operation fails
        """
        try:
            async with DatabaseSession() as db:
                # Insert population snapshot document
                pop_collection: AsyncIOMotorCollection = db[self.collections['populations']]
                
                pop_doc = PopulationDocument(
                    simulation_id=simulation_id,
                    snapshot=snapshot
                )
                
                pop_result = await pop_collection.insert_one(
                    pop_doc.dict(by_alias=True, exclude={'id'})
                )
                
                # Update simulation with new snapshot in results
                sim_collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Append to population_history array in results
                sim_result = await sim_collection.update_one(
                    {"simulation_id": simulation_id},
                    {
                        "$push": {"results.population_history": snapshot.dict()},
                        "$set": {
                            "metadata.updated_at": datetime.utcnow(),
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                
                if sim_result.matched_count == 0:
                    # If simulation doesn't exist or doesn't have results, create results structure
                    await sim_collection.update_one(
                        {"simulation_id": simulation_id},
                        {
                            "$set": {
                                "results.population_history": [snapshot.dict()],
                                "metadata.updated_at": datetime.utcnow(),
                                "updated_at": datetime.utcnow()
                            }
                        }
                    )
                
                logger.info(f"Appended population snapshot for simulation {simulation_id}, generation {snapshot.generation}")
                return str(pop_result.inserted_id)
                
        except Exception as e:
            error_msg = f"Failed to append population snapshot for simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def update_simulation_results(
        self,
        simulation_id: str,
        results_updates: Dict[str, Any]
    ) -> bool:
        """
        Update specific fields in simulation results
        
        Args:
            simulation_id: Unique simulation identifier
            results_updates: Dictionary of result fields to update
            
        Returns:
            bool: True if update was successful
            
        Raises:
            SimulationDatabaseError: If update operation fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Prepare update document with results fields
                update_doc = {}
                for key, value in results_updates.items():
                    update_doc[f"results.{key}"] = value
                
                # Update timestamps
                update_doc["metadata.updated_at"] = datetime.utcnow()
                update_doc["updated_at"] = datetime.utcnow()
                
                # Perform update
                result = await collection.update_one(
                    {"simulation_id": simulation_id},
                    {"$set": update_doc}
                )
                
                if result.matched_count == 0:
                    raise SimulationDatabaseError(f"Simulation {simulation_id} not found")
                
                logger.info(f"Updated results for simulation {simulation_id}")
                return result.modified_count > 0
                
        except Exception as e:
            error_msg = f"Failed to update results for simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def add_simulation_tags(
        self,
        simulation_id: str,
        tags: List[str]
    ) -> bool:
        """
        Add tags to a simulation (avoiding duplicates)
        
        Args:
            simulation_id: Unique simulation identifier
            tags: List of tags to add
            
        Returns:
            bool: True if update was successful
            
        Raises:
            SimulationDatabaseError: If update operation fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Use $addToSet to avoid duplicate tags
                result = await collection.update_one(
                    {"simulation_id": simulation_id},
                    {
                        "$addToSet": {"metadata.tags": {"$each": tags}},
                        "$set": {
                            "metadata.updated_at": datetime.utcnow(),
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                
                if result.matched_count == 0:
                    raise SimulationDatabaseError(f"Simulation {simulation_id} not found")
                
                logger.info(f"Added tags {tags} to simulation {simulation_id}")
                return result.modified_count > 0
                
        except Exception as e:
            error_msg = f"Failed to add tags to simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def remove_simulation_tags(
        self,
        simulation_id: str,
        tags: List[str]
    ) -> bool:
        """
        Remove tags from a simulation
        
        Args:
            simulation_id: Unique simulation identifier
            tags: List of tags to remove
            
        Returns:
            bool: True if update was successful
            
        Raises:
            SimulationDatabaseError: If update operation fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Use $pullAll to remove specified tags
                result = await collection.update_one(
                    {"simulation_id": simulation_id},
                    {
                        "$pullAll": {"metadata.tags": tags},
                        "$set": {
                            "metadata.updated_at": datetime.utcnow(),
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                
                if result.matched_count == 0:
                    raise SimulationDatabaseError(f"Simulation {simulation_id} not found")
                
                logger.info(f"Removed tags {tags} from simulation {simulation_id}")
                return result.modified_count > 0
                
        except Exception as e:
            error_msg = f"Failed to remove tags from simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e

class SimulationDeleteOperations:
    """Database delete operations for simulation data"""
    
    def __init__(self):
        self.collections = {
            'simulations': 'simulations',
            'populations': 'population_snapshots', 
            'bacteria': 'bacteria'
        }
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def soft_delete_simulation(
        self,
        simulation_id: str,
        deletion_reason: Optional[str] = None
    ) -> bool:
        """
        Soft delete a simulation by marking it as deleted
        
        Args:
            simulation_id: Unique simulation identifier
            deletion_reason: Optional reason for deletion
            
        Returns:
            bool: True if soft delete was successful
            
        Raises:
            SimulationDatabaseError: If soft delete operation fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Mark simulation as deleted
                update_doc = {
                    "metadata.status": SimulationStatus.CANCELLED,
                    "metadata.is_deleted": True,
                    "metadata.deleted_at": datetime.utcnow(),
                    "metadata.updated_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                if deletion_reason:
                    update_doc["metadata.deletion_reason"] = deletion_reason
                
                result = await collection.update_one(
                    {"simulation_id": simulation_id},
                    {"$set": update_doc}
                )
                
                if result.matched_count == 0:
                    raise SimulationDatabaseError(f"Simulation {simulation_id} not found")
                
                logger.info(f"Soft deleted simulation {simulation_id}")
                return result.modified_count > 0
                
        except Exception as e:
            error_msg = f"Failed to soft delete simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def hard_delete_simulation(
        self,
        simulation_id: str,
        delete_related_data: bool = True
    ) -> Dict[str, int]:
        """
        Permanently delete a simulation and optionally its related data
        
        Args:
            simulation_id: Unique simulation identifier
            delete_related_data: Whether to delete population and bacteria data
            
        Returns:
            Dict[str, int]: Count of deleted documents by collection
            
        Raises:
            SimulationDatabaseError: If hard delete operation fails
        """
        try:
            async with DatabaseSession() as db:
                deleted_counts = {}
                
                # Delete main simulation document
                sim_collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                sim_result = await sim_collection.delete_one({"simulation_id": simulation_id})
                deleted_counts['simulations'] = sim_result.deleted_count
                
                if delete_related_data:
                    # Delete population snapshots
                    pop_collection: AsyncIOMotorCollection = db[self.collections['populations']]
                    pop_result = await pop_collection.delete_many({"simulation_id": simulation_id})
                    deleted_counts['populations'] = pop_result.deleted_count
                    
                    # Delete bacteria documents
                    bact_collection: AsyncIOMotorCollection = db[self.collections['bacteria']]
                    bact_result = await bact_collection.delete_many({"simulation_id": simulation_id})
                    deleted_counts['bacteria'] = bact_result.deleted_count
                
                if deleted_counts.get('simulations', 0) == 0:
                    raise SimulationDatabaseError(f"Simulation {simulation_id} not found")
                
                logger.info(f"Hard deleted simulation {simulation_id} and related data: {deleted_counts}")
                return deleted_counts
                
        except Exception as e:
            error_msg = f"Failed to hard delete simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def delete_population_snapshots(
        self,
        simulation_id: str,
        generation_start: Optional[int] = None,
        generation_end: Optional[int] = None
    ) -> int:
        """
        Delete population snapshots for a simulation within a generation range
        
        Args:
            simulation_id: Unique simulation identifier
            generation_start: Start generation (inclusive), None for no lower bound
            generation_end: End generation (inclusive), None for no upper bound
            
        Returns:
            int: Number of deleted population documents
            
        Raises:
            SimulationDatabaseError: If delete operation fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['populations']]
                
                # Build query with generation range
                query = {"simulation_id": simulation_id}
                
                if generation_start is not None or generation_end is not None:
                    generation_query = {}
                    if generation_start is not None:
                        generation_query["$gte"] = generation_start
                    if generation_end is not None:
                        generation_query["$lte"] = generation_end
                    query["snapshot.generation"] = generation_query
                
                # Delete matching documents
                result = await collection.delete_many(query)
                
                logger.info(f"Deleted {result.deleted_count} population snapshots for simulation {simulation_id}")
                return result.deleted_count
                
        except Exception as e:
            error_msg = f"Failed to delete population snapshots for simulation {simulation_id}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def delete_bacteria_by_generation(
        self,
        simulation_id: str,
        generation: int
    ) -> int:
        """
        Delete all bacteria documents for a specific generation
        
        Args:
            simulation_id: Unique simulation identifier
            generation: Generation number to delete
            
        Returns:
            int: Number of deleted bacteria documents
            
        Raises:
            SimulationDatabaseError: If delete operation fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['bacteria']]
                
                # Delete bacteria for specific generation
                result = await collection.delete_many({
                    "simulation_id": simulation_id,
                    "generation": generation
                })
                
                logger.info(f"Deleted {result.deleted_count} bacteria for simulation {simulation_id}, generation {generation}")
                return result.deleted_count
                
        except Exception as e:
            error_msg = f"Failed to delete bacteria for simulation {simulation_id}, generation {generation}: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def cleanup_old_simulations(
        self,
        days_old: int = 30,
        status_filter: Optional[SimulationStatus] = None,
        dry_run: bool = True
    ) -> Dict[str, int]:
        """
        Clean up old simulations based on age and status
        
        Args:
            days_old: Delete simulations older than this many days
            status_filter: Only delete simulations with this status (None for any)
            dry_run: If True, only count what would be deleted without deleting
            
        Returns:
            Dict[str, int]: Count of simulations that would be/were deleted
            
        Raises:
            SimulationDatabaseError: If cleanup operation fails
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections['simulations']]
                
                # Calculate cutoff date
                cutoff_date = datetime.utcnow() - timedelta(days=days_old)
                
                # Build query
                query = {"created_at": {"$lt": cutoff_date}}
                if status_filter:
                    query["metadata.status"] = status_filter.value
                
                if dry_run:
                    # Count documents that would be deleted
                    count = await collection.count_documents(query)
                    logger.info(f"Would delete {count} simulations older than {days_old} days")
                    return {"would_delete": count}
                else:
                    # Get simulation IDs before deletion for cleanup of related data
                    cursor = collection.find(query, {"simulation_id": 1})
                    simulation_ids = [doc["simulation_id"] async for doc in cursor]
                    
                    # Delete simulations
                    result = await collection.delete_many(query)
                    deleted_simulations = result.deleted_count
                    
                    # Clean up related data
                    deleted_populations = 0
                    deleted_bacteria = 0
                    
                    if simulation_ids:
                        pop_collection: AsyncIOMotorCollection = db[self.collections['populations']]
                        pop_result = await pop_collection.delete_many({
                            "simulation_id": {"$in": simulation_ids}
                        })
                        deleted_populations = pop_result.deleted_count
                        
                        bact_collection: AsyncIOMotorCollection = db[self.collections['bacteria']]
                        bact_result = await bact_collection.delete_many({
                            "simulation_id": {"$in": simulation_ids}
                        })
                        deleted_bacteria = bact_result.deleted_count
                    
                    cleanup_counts = {
                        "simulations": deleted_simulations,
                        "populations": deleted_populations,
                        "bacteria": deleted_bacteria
                    }
                    
                    logger.info(f"Cleanup completed: {cleanup_counts}")
                    return cleanup_counts
                
        except Exception as e:
            error_msg = f"Failed to cleanup old simulations: {e}"
            logger.error(error_msg)
            raise SimulationDatabaseError(error_msg) from e

# Factory functions and global instances
async def get_simulation_create_operations() -> SimulationCreateOperations:
    """Get simulation create operations instance"""
    await ensure_database_connection()
    return SimulationCreateOperations()

async def get_simulation_read_operations() -> SimulationReadOperations:
    """Get simulation read operations instance"""
    await ensure_database_connection()
    return SimulationReadOperations()

async def get_simulation_update_operations() -> SimulationUpdateOperations:
    """Get simulation update operations instance"""
    await ensure_database_connection()
    return SimulationUpdateOperations()

async def get_simulation_delete_operations() -> SimulationDeleteOperations:
    """Get simulation delete operations instance"""
    await ensure_database_connection()
    return SimulationDeleteOperations()

# Global instances
create_ops = SimulationCreateOperations()
read_ops = SimulationReadOperations()
update_ops = SimulationUpdateOperations()
delete_ops = SimulationDeleteOperations()

# Legacy global instance (deprecated, use specific operations)
simulation_db = create_ops 