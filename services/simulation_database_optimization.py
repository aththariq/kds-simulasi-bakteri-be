import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import ASCENDING, DESCENDING, TEXT
from pymongo.errors import OperationFailure
from functools import wraps
import time

from utils.db_connection import DatabaseSession, ensure_database_connection, retry_on_connection_failure
from models.database_models import SimulationStatus

logger = logging.getLogger(__name__)

class DatabaseOptimizationError(Exception):
    """Custom exception for database optimization operations"""
    pass

def performance_monitor(func):
    """Decorator to monitor query performance"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f}s: {e}")
            raise
    return wrapper

class SimulationDatabaseOptimizer:
    """Database optimization and index management for simulation data"""
    
    def __init__(self):
        self.collections = {
            'simulations': 'simulations',
            'populations': 'population_snapshots', 
            'bacteria': 'bacteria'
        }
        
        # Index definitions for optimal query performance
        self.index_definitions = {
            'simulations': [
                # Unique index on simulation_id for fast lookups
                [("simulation_id", ASCENDING), {"unique": True, "background": True}],
                
                # Compound index for status and creation date filtering
                [("metadata.status", ASCENDING), ("created_at", DESCENDING), {"background": True}],
                
                # Index on creation date for time-based queries
                [("created_at", DESCENDING), {"background": True}],
                
                # Index on updated date for recently modified simulations
                [("updated_at", DESCENDING), {"background": True}],
                
                # Index on tags for tag-based filtering
                [("metadata.tags", ASCENDING), {"background": True}],
                
                # Text search index for name and description
                [("metadata.name", TEXT), ("metadata.description", TEXT), {"background": True}],
                
                # Compound index for status and tags filtering
                [("metadata.status", ASCENDING), ("metadata.tags", ASCENDING), {"background": True}],
                
                # Index for simulation parameters queries
                [("metadata.parameters.initial_population", ASCENDING), {"background": True}],
                [("metadata.parameters.mutation_rate", ASCENDING), {"background": True}],
                [("metadata.parameters.antibiotic_concentration", ASCENDING), {"background": True}],
            ],
            
            'populations': [
                # Compound index for simulation and generation lookups
                [("simulation_id", ASCENDING), ("snapshot.generation", ASCENDING), {"background": True}],
                
                # Index on simulation_id for simulation-specific queries
                [("simulation_id", ASCENDING), {"background": True}],
                
                # Index on generation for cross-simulation generation analysis
                [("snapshot.generation", ASCENDING), {"background": True}],
                
                # Index on creation date for time-based queries
                [("created_at", DESCENDING), {"background": True}],
                
                # Index for population metrics
                [("snapshot.total_population", ASCENDING), {"background": True}],
                [("snapshot.resistance_frequency", ASCENDING), {"background": True}],
                [("snapshot.average_fitness", ASCENDING), {"background": True}],
                
                # Compound index for simulation and population size queries
                [("simulation_id", ASCENDING), ("snapshot.total_population", DESCENDING), {"background": True}],
            ],
            
            'bacteria': [
                # Compound index for simulation and generation lookups
                [("simulation_id", ASCENDING), ("generation", ASCENDING), {"background": True}],
                
                # Index on simulation_id for simulation-specific queries
                [("simulation_id", ASCENDING), {"background": True}],
                
                # Index on generation for cross-simulation analysis
                [("generation", ASCENDING), {"background": True}],
                
                # Index on bacterium ID for individual bacteria lookups
                [("bacterium.id", ASCENDING), {"background": True}],
                
                # Index on fitness for fitness-based queries
                [("bacterium.fitness", DESCENDING), {"background": True}],
                
                # Index on resistance genes for resistance analysis
                [("bacterium.resistance_genes", ASCENDING), {"background": True}],
                
                # Spatial indexes for spatial queries
                [("bacterium.position_x", ASCENDING), ("bacterium.position_y", ASCENDING), {"background": True}],
                
                # Index on creation date for time-based queries
                [("created_at", DESCENDING), {"background": True}],
            ]
        }
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def create_indexes(self) -> Dict[str, List[str]]:
        """
        Create all necessary indexes for optimal query performance
        
        Returns:
            Dict[str, List[str]]: Created indexes by collection
            
        Raises:
            DatabaseOptimizationError: If index creation fails
        """
        try:
            async with DatabaseSession() as db:
                created_indexes = {}
                
                for collection_name, index_list in self.index_definitions.items():
                    collection: AsyncIOMotorCollection = db[self.collections[collection_name]]
                    collection_indexes = []
                    
                    logger.info(f"Creating indexes for collection: {collection_name}")
                    
                    for index_spec, index_options in index_list:
                        try:
                            # Create index
                            index_name = await collection.create_index(index_spec, **index_options)
                            collection_indexes.append(index_name)
                            logger.info(f"Created index: {index_name} on {collection_name}")
                            
                        except OperationFailure as e:
                            if "already exists" in str(e):
                                logger.info(f"Index already exists on {collection_name}: {index_spec}")
                            else:
                                logger.error(f"Failed to create index on {collection_name}: {e}")
                                raise
                    
                    created_indexes[collection_name] = collection_indexes
                
                logger.info("All indexes created successfully")
                return created_indexes
                
        except Exception as e:
            error_msg = f"Failed to create database indexes: {e}"
            logger.error(error_msg)
            raise DatabaseOptimizationError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def list_indexes(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all existing indexes in the database
        
        Returns:
            Dict[str, List[Dict[str, Any]]]: Existing indexes by collection
        """
        try:
            async with DatabaseSession() as db:
                all_indexes = {}
                
                for collection_name in self.collections:
                    collection: AsyncIOMotorCollection = db[self.collections[collection_name]]
                    indexes = await collection.list_indexes().to_list(None)
                    all_indexes[collection_name] = indexes
                    
                    logger.info(f"Collection {collection_name} has {len(indexes)} indexes")
                
                return all_indexes
                
        except Exception as e:
            error_msg = f"Failed to list database indexes: {e}"
            logger.error(error_msg)
            raise DatabaseOptimizationError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def analyze_query_performance(
        self, 
        collection_name: str, 
        query: Dict[str, Any],
        explain: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze query performance and execution plan
        
        Args:
            collection_name: Name of the collection to query
            query: MongoDB query to analyze
            explain: Whether to return execution plan
            
        Returns:
            Dict[str, Any]: Query performance analysis
        """
        try:
            async with DatabaseSession() as db:
                collection: AsyncIOMotorCollection = db[self.collections[collection_name]]
                
                if explain:
                    # Get query execution plan
                    cursor = collection.find(query)
                    explain_result = await cursor.explain()
                    
                    # Extract performance metrics
                    execution_stats = explain_result.get('executionStats', {})
                    
                    analysis = {
                        'query': query,
                        'collection': collection_name,
                        'execution_time_ms': execution_stats.get('executionTimeMillis', 0),
                        'documents_examined': execution_stats.get('totalDocsExamined', 0),
                        'documents_returned': execution_stats.get('totalDocsReturned', 0),
                        'index_used': execution_stats.get('indexUsed', False),
                        'winning_plan': explain_result.get('queryPlanner', {}).get('winningPlan', {}),
                        'full_explain': explain_result
                    }
                    
                    logger.info(f"Query analysis for {collection_name}: {analysis['execution_time_ms']}ms, "
                              f"examined: {analysis['documents_examined']}, "
                              f"returned: {analysis['documents_returned']}")
                    
                    return analysis
                else:
                    # Simple performance timing
                    start_time = time.time()
                    count = await collection.count_documents(query)
                    execution_time = (time.time() - start_time) * 1000
                    
                    return {
                        'query': query,
                        'collection': collection_name,
                        'execution_time_ms': execution_time,
                        'document_count': count
                    }
                
        except Exception as e:
            error_msg = f"Failed to analyze query performance: {e}"
            logger.error(error_msg)
            raise DatabaseOptimizationError(error_msg) from e
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def get_collection_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed statistics for all collections
        
        Returns:
            Dict[str, Dict[str, Any]]: Collection statistics
        """
        try:
            async with DatabaseSession() as db:
                stats = {}
                
                for collection_name in self.collections:
                    collection: AsyncIOMotorCollection = db[self.collections[collection_name]]
                    
                    # Get collection stats
                    collection_stats = await db.command("collStats", self.collections[collection_name])
                    
                    # Get document count
                    doc_count = await collection.count_documents({})
                    
                    # Get index stats
                    indexes = await collection.list_indexes().to_list(None)
                    
                    stats[collection_name] = {
                        'document_count': doc_count,
                        'storage_size_bytes': collection_stats.get('storageSize', 0),
                        'index_count': len(indexes),
                        'index_sizes': collection_stats.get('indexSizes', {}),
                        'avg_obj_size': collection_stats.get('avgObjSize', 0),
                        'total_size_bytes': collection_stats.get('size', 0)
                    }
                    
                    logger.info(f"Collection {collection_name}: {doc_count} documents, "
                              f"{len(indexes)} indexes, "
                              f"{collection_stats.get('storageSize', 0)} bytes")
                
                return stats
                
        except Exception as e:
            error_msg = f"Failed to get collection statistics: {e}"
            logger.error(error_msg)
            raise DatabaseOptimizationError(error_msg) from e
    
    @performance_monitor
    async def benchmark_common_queries(self) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark common query patterns for performance analysis
        
        Returns:
            Dict[str, Dict[str, Any]]: Benchmark results
        """
        benchmark_queries = {
            'simulations': [
                # Query by status
                {"metadata.status": SimulationStatus.COMPLETED},
                
                # Query by creation date range
                {"created_at": {"$gte": datetime.now().replace(day=1)}},
                
                # Query by tags
                {"metadata.tags": {"$in": ["test", "production"]}},
                
                # Complex compound query
                {
                    "metadata.status": {"$in": [SimulationStatus.COMPLETED, SimulationStatus.RUNNING]},
                    "metadata.parameters.mutation_rate": {"$gte": 0.01}
                }
            ],
            
            'populations': [
                # Query by simulation
                {"simulation_id": "sim_example_001"},
                
                # Query by generation range
                {"snapshot.generation": {"$gte": 10, "$lte": 50}},
                
                # Query by population metrics
                {"snapshot.total_population": {"$gte": 1000}},
                
                # Complex compound query
                {
                    "simulation_id": "sim_example_001",
                    "snapshot.generation": {"$gte": 20},
                    "snapshot.resistance_frequency": {"$gte": 0.5}
                }
            ],
            
            'bacteria': [
                # Query by simulation and generation
                {"simulation_id": "sim_example_001", "generation": 25},
                
                # Query by fitness threshold
                {"bacterium.fitness": {"$gte": 0.8}},
                
                # Query by resistance genes
                {"bacterium.resistance_genes": {"$in": ["ampR", "tetR"]}},
                
                # Spatial query
                {
                    "bacterium.position_x": {"$gte": 10, "$lte": 90},
                    "bacterium.position_y": {"$gte": 10, "$lte": 90}
                }
            ]
        }
        
        results = {}
        
        for collection_name, queries in benchmark_queries.items():
            collection_results = []
            
            for i, query in enumerate(queries):
                try:
                    analysis = await self.analyze_query_performance(collection_name, query)
                    collection_results.append({
                        'query_index': i,
                        'query': query,
                        'performance': analysis
                    })
                except Exception as e:
                    logger.warning(f"Benchmark query {i} failed for {collection_name}: {e}")
                    collection_results.append({
                        'query_index': i,
                        'query': query,
                        'error': str(e)
                    })
            
            results[collection_name] = collection_results
        
        return results
    
    async def optimize_database(self) -> Dict[str, Any]:
        """
        Perform complete database optimization
        
        Returns:
            Dict[str, Any]: Optimization results
        """
        logger.info("Starting database optimization process")
        
        optimization_results = {
            'started_at': datetime.utcnow(),
            'steps_completed': [],
            'errors': []
        }
        
        try:
            # Step 1: Create indexes
            logger.info("Step 1: Creating database indexes")
            created_indexes = await self.create_indexes()
            optimization_results['created_indexes'] = created_indexes
            optimization_results['steps_completed'].append('create_indexes')
            
            # Step 2: Analyze performance
            logger.info("Step 2: Analyzing query performance")
            benchmark_results = await self.benchmark_common_queries()
            optimization_results['benchmark_results'] = benchmark_results
            optimization_results['steps_completed'].append('benchmark_queries')
            
            # Step 3: Get statistics
            logger.info("Step 3: Collecting collection statistics")
            collection_stats = await self.get_collection_statistics()
            optimization_results['collection_statistics'] = collection_stats
            optimization_results['steps_completed'].append('collect_statistics')
            
            optimization_results['completed_at'] = datetime.utcnow()
            optimization_results['success'] = True
            
            logger.info("Database optimization completed successfully")
            
        except Exception as e:
            optimization_results['errors'].append(str(e))
            optimization_results['completed_at'] = datetime.utcnow()
            optimization_results['success'] = False
            logger.error(f"Database optimization failed: {e}")
            raise DatabaseOptimizationError(f"Database optimization failed: {e}") from e
        
        return optimization_results

# Factory function
async def get_database_optimizer() -> SimulationDatabaseOptimizer:
    """Get database optimizer instance"""
    await ensure_database_connection()
    return SimulationDatabaseOptimizer()

# Global instance
db_optimizer = SimulationDatabaseOptimizer() 