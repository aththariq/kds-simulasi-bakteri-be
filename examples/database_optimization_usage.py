"""
Database Optimization Usage Examples

This module demonstrates how to use the database optimization functionality
for the bacterial simulation database system.

Examples include:
- Index creation and management
- Query performance analysis
- Database optimization workflow
- Load testing and benchmarking
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from services.simulation_database_optimization import (
    get_database_optimizer,
    SimulationDatabaseOptimizer,
    DatabaseOptimizationError
)
from models.database_models import SimulationStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def example_create_indexes():
    """Example: Create database indexes for optimal performance"""
    logger.info("=== Database Index Creation Example ===")
    
    try:
        # Get database optimizer instance
        optimizer = await get_database_optimizer()
        
        # Create all necessary indexes
        logger.info("Creating database indexes...")
        created_indexes = await optimizer.create_indexes()
        
        # Display results
        for collection_name, indexes in created_indexes.items():
            logger.info(f"Collection '{collection_name}': Created {len(indexes)} indexes")
            for index_name in indexes:
                logger.info(f"  - {index_name}")
        
        logger.info("Index creation completed successfully!")
        
    except DatabaseOptimizationError as e:
        logger.error(f"Index creation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

async def example_list_existing_indexes():
    """Example: List all existing database indexes"""
    logger.info("=== List Database Indexes Example ===")
    
    try:
        optimizer = await get_database_optimizer()
        
        # List all existing indexes
        all_indexes = await optimizer.list_indexes()
        
        for collection_name, indexes in all_indexes.items():
            logger.info(f"Collection '{collection_name}' has {len(indexes)} indexes:")
            for index in indexes:
                index_name = index.get('name', 'unknown')
                index_keys = index.get('key', {})
                logger.info(f"  - {index_name}: {index_keys}")
        
    except Exception as e:
        logger.error(f"Failed to list indexes: {e}")

async def example_analyze_query_performance():
    """Example: Analyze the performance of common queries"""
    logger.info("=== Query Performance Analysis Example ===")
    
    try:
        optimizer = await get_database_optimizer()
        
        # Define test queries for different collections
        test_queries = {
            'simulations': [
                # Query by status
                {"metadata.status": SimulationStatus.COMPLETED},
                
                # Query by creation date
                {"created_at": {"$gte": datetime.now() - timedelta(days=7)}},
                
                # Complex compound query
                {
                    "metadata.status": {"$in": [SimulationStatus.COMPLETED, SimulationStatus.RUNNING]},
                    "metadata.parameters.mutation_rate": {"$gte": 0.01}
                }
            ],
            
            'populations': [
                # Query by simulation ID
                {"simulation_id": "sim_example_001"},
                
                # Query by generation range
                {"snapshot.generation": {"$gte": 10, "$lte": 50}},
                
                # Query by population metrics
                {"snapshot.resistance_frequency": {"$gte": 0.5}}
            ],
            
            'bacteria': [
                # Query by simulation and generation
                {"simulation_id": "sim_example_001", "generation": 25},
                
                # Query by fitness threshold
                {"bacterium.fitness": {"$gte": 0.8}},
                
                # Spatial query
                {
                    "bacterium.position_x": {"$gte": 10, "$lte": 90},
                    "bacterium.position_y": {"$gte": 10, "$lte": 90}
                }
            ]
        }
        
        # Analyze each query
        for collection_name, queries in test_queries.items():
            logger.info(f"Analyzing queries for collection: {collection_name}")
            
            for i, query in enumerate(queries):
                try:
                    analysis = await optimizer.analyze_query_performance(
                        collection_name, query, explain=True
                    )
                    
                    logger.info(f"  Query {i+1}: {query}")
                    logger.info(f"    Execution time: {analysis['execution_time_ms']}ms")
                    logger.info(f"    Documents examined: {analysis['documents_examined']}")
                    logger.info(f"    Documents returned: {analysis['documents_returned']}")
                    logger.info(f"    Index used: {analysis['index_used']}")
                    
                    # Check if query is slow
                    if analysis['execution_time_ms'] > 100:
                        logger.warning(f"    ⚠️  Slow query detected!")
                    else:
                        logger.info(f"    ✅ Query performance is good")
                
                except Exception as e:
                    logger.error(f"    ❌ Query analysis failed: {e}")
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")

async def example_get_collection_statistics():
    """Example: Get detailed statistics for all collections"""
    logger.info("=== Collection Statistics Example ===")
    
    try:
        optimizer = await get_database_optimizer()
        
        # Get collection statistics
        stats = await optimizer.get_collection_statistics()
        
        logger.info("Database collection statistics:")
        
        total_documents = 0
        total_storage = 0
        
        for collection_name, collection_stats in stats.items():
            doc_count = collection_stats['document_count']
            storage_bytes = collection_stats['storage_size_bytes']
            index_count = collection_stats['index_count']
            avg_size = collection_stats['avg_obj_size']
            
            total_documents += doc_count
            total_storage += storage_bytes
            
            logger.info(f"Collection '{collection_name}':")
            logger.info(f"  Documents: {doc_count:,}")
            logger.info(f"  Storage size: {storage_bytes / 1024 / 1024:.2f} MB")
            logger.info(f"  Indexes: {index_count}")
            logger.info(f"  Average document size: {avg_size} bytes")
            
            # Index breakdown
            index_sizes = collection_stats.get('index_sizes', {})
            if index_sizes:
                logger.info(f"  Index sizes:")
                for index_name, size in index_sizes.items():
                    logger.info(f"    {index_name}: {size / 1024:.2f} KB")
        
        # Overall statistics
        logger.info("Overall database statistics:")
        logger.info(f"  Total documents: {total_documents:,}")
        logger.info(f"  Total storage: {total_storage / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        logger.error(f"Failed to get collection statistics: {e}")

async def example_benchmark_common_queries():
    """Example: Benchmark common query patterns"""
    logger.info("=== Query Benchmarking Example ===")
    
    try:
        optimizer = await get_database_optimizer()
        
        # Run benchmark tests
        benchmark_results = await optimizer.benchmark_common_queries()
        
        logger.info("Benchmark results:")
        
        for collection_name, collection_results in benchmark_results.items():
            logger.info(f"Collection '{collection_name}':")
            
            for result in collection_results:
                query_index = result['query_index']
                query = result['query']
                
                if 'error' in result:
                    logger.error(f"  Query {query_index + 1}: ❌ {result['error']}")
                else:
                    performance = result['performance']
                    execution_time = performance['execution_time_ms']
                    
                    # Performance classification
                    if execution_time < 10:
                        status = "🚀 Excellent"
                    elif execution_time < 50:
                        status = "✅ Good"
                    elif execution_time < 200:
                        status = "⚠️  Acceptable"
                    else:
                        status = "🐌 Slow"
                    
                    logger.info(f"  Query {query_index + 1}: {status} ({execution_time}ms)")
                    logger.info(f"    Query: {query}")
                    
                    if 'documents_examined' in performance:
                        examined = performance['documents_examined']
                        returned = performance['documents_returned']
                        efficiency = (returned / examined * 100) if examined > 0 else 0
                        logger.info(f"    Efficiency: {efficiency:.1f}% ({returned}/{examined})")
        
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")

async def example_complete_optimization_workflow():
    """Example: Complete database optimization workflow"""
    logger.info("=== Complete Database Optimization Workflow ===")
    
    try:
        optimizer = await get_database_optimizer()
        
        # Run complete optimization
        optimization_results = await optimizer.optimize_database()
        
        if optimization_results['success']:
            logger.info("✅ Database optimization completed successfully!")
            
            # Display optimization results
            steps_completed = optimization_results['steps_completed']
            logger.info(f"Steps completed: {', '.join(steps_completed)}")
            
            # Show index creation results
            if 'created_indexes' in optimization_results:
                created_indexes = optimization_results['created_indexes']
                total_indexes = sum(len(indexes) for indexes in created_indexes.values())
                logger.info(f"Total indexes created: {total_indexes}")
            
            # Show benchmark summary
            if 'benchmark_results' in optimization_results:
                benchmark_results = optimization_results['benchmark_results']
                total_queries = sum(len(results) for results in benchmark_results.values())
                logger.info(f"Total queries benchmarked: {total_queries}")
            
            # Show collection statistics summary
            if 'collection_statistics' in optimization_results:
                stats = optimization_results['collection_statistics']
                total_docs = sum(s['document_count'] for s in stats.values())
                total_storage = sum(s['storage_size_bytes'] for s in stats.values())
                logger.info(f"Total documents: {total_docs:,}")
                logger.info(f"Total storage: {total_storage / 1024 / 1024:.2f} MB")
            
            # Execution time
            start_time = optimization_results['started_at']
            end_time = optimization_results['completed_at']
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Optimization completed in {duration:.2f} seconds")
            
        else:
            logger.error("❌ Database optimization failed!")
            errors = optimization_results.get('errors', [])
            for error in errors:
                logger.error(f"  Error: {error}")
                
    except DatabaseOptimizationError as e:
        logger.error(f"Optimization workflow failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in optimization: {e}")

async def example_monitor_optimization_impact():
    """Example: Monitor the impact of optimization on query performance"""
    logger.info("=== Optimization Impact Monitoring Example ===")
    
    try:
        optimizer = await get_database_optimizer()
        
        # Define a test query
        test_query = {"metadata.status": SimulationStatus.COMPLETED}
        collection_name = 'simulations'
        
        # Analyze performance before optimization
        logger.info("Analyzing performance before optimization...")
        performance_before = await optimizer.analyze_query_performance(
            collection_name, test_query, explain=True
        )
        
        logger.info("Performance before optimization:")
        logger.info(f"  Execution time: {performance_before['execution_time_ms']}ms")
        logger.info(f"  Documents examined: {performance_before['documents_examined']}")
        logger.info(f"  Index used: {performance_before['index_used']}")
        
        # Create indexes
        logger.info("Creating indexes...")
        await optimizer.create_indexes()
        
        # Analyze performance after optimization
        logger.info("Analyzing performance after optimization...")
        performance_after = await optimizer.analyze_query_performance(
            collection_name, test_query, explain=True
        )
        
        logger.info("Performance after optimization:")
        logger.info(f"  Execution time: {performance_after['execution_time_ms']}ms")
        logger.info(f"  Documents examined: {performance_after['documents_examined']}")
        logger.info(f"  Index used: {performance_after['index_used']}")
        
        # Calculate improvement
        time_before = performance_before['execution_time_ms']
        time_after = performance_after['execution_time_ms']
        
        if time_before > 0 and time_after > 0:
            improvement_ratio = time_before / time_after
            improvement_percent = ((time_before - time_after) / time_before) * 100
            
            logger.info("Performance improvement:")
            logger.info(f"  Speed improvement: {improvement_ratio:.2f}x faster")
            logger.info(f"  Time reduction: {improvement_percent:.1f}%")
            
            if improvement_ratio > 2:
                logger.info("  🎉 Significant performance improvement achieved!")
            elif improvement_ratio > 1.5:
                logger.info("  ✅ Good performance improvement")
            else:
                logger.info("  📊 Modest performance improvement")
        
    except Exception as e:
        logger.error(f"Impact monitoring failed: {e}")

async def example_database_health_check():
    """Example: Comprehensive database health check"""
    logger.info("=== Database Health Check Example ===")
    
    try:
        optimizer = await get_database_optimizer()
        
        logger.info("Performing database health check...")
        
        # Check 1: Collection statistics
        logger.info("1. Checking collection statistics...")
        stats = await optimizer.get_collection_statistics()
        
        for collection_name, collection_stats in stats.items():
            doc_count = collection_stats['document_count']
            storage_mb = collection_stats['storage_size_bytes'] / 1024 / 1024
            index_count = collection_stats['index_count']
            
            logger.info(f"  {collection_name}: {doc_count:,} docs, {storage_mb:.1f}MB, {index_count} indexes")
            
            # Health warnings
            if doc_count > 1000000:
                logger.warning(f"    ⚠️  Large collection ({doc_count:,} documents)")
            
            if index_count < 2:
                logger.warning(f"    ⚠️  Few indexes ({index_count})")
            
            if storage_mb > 1000:
                logger.warning(f"    ⚠️  Large storage usage ({storage_mb:.1f}MB)")
        
        # Check 2: Index coverage
        logger.info("2. Checking index coverage...")
        existing_indexes = await optimizer.list_indexes()
        
        for collection_name, indexes in existing_indexes.items():
            index_names = [idx.get('name', 'unknown') for idx in indexes]
            logger.info(f"  {collection_name}: {len(index_names)} indexes")
            
            # Check for essential indexes
            has_compound_index = any('_1_' in name for name in index_names)
            has_text_index = any('text' in str(idx.get('key', {})) for idx in indexes)
            
            if not has_compound_index:
                logger.warning(f"    ⚠️  No compound indexes found")
            
            if collection_name == 'simulations' and not has_text_index:
                logger.info(f"    💡 Consider adding text search index")
        
        # Check 3: Query performance sampling
        logger.info("3. Sampling query performance...")
        
        sample_queries = {
            'simulations': {"metadata.status": SimulationStatus.COMPLETED},
            'populations': {"simulation_id": "test_sim"},
            'bacteria': {"generation": 1}
        }
        
        for collection_name, query in sample_queries.items():
            try:
                analysis = await optimizer.analyze_query_performance(
                    collection_name, query, explain=False
                )
                
                execution_time = analysis['execution_time_ms']
                
                if execution_time < 50:
                    status = "✅ Good"
                elif execution_time < 200:
                    status = "⚠️  Acceptable"
                else:
                    status = "🚫 Slow"
                
                logger.info(f"  {collection_name}: {status} ({execution_time:.1f}ms)")
                
            except Exception as e:
                logger.warning(f"  {collection_name}: ❌ Query failed ({e})")
        
        logger.info("Database health check completed!")
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")

async def main():
    """Run all optimization examples"""
    logger.info("Starting Database Optimization Examples")
    
    examples = [
        ("Create Indexes", example_create_indexes),
        ("List Existing Indexes", example_list_existing_indexes),
        ("Analyze Query Performance", example_analyze_query_performance),
        ("Get Collection Statistics", example_get_collection_statistics),
        ("Benchmark Common Queries", example_benchmark_common_queries),
        ("Complete Optimization Workflow", example_complete_optimization_workflow),
        ("Monitor Optimization Impact", example_monitor_optimization_impact),
        ("Database Health Check", example_database_health_check)
    ]
    
    for example_name, example_func in examples:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running example: {example_name}")
            logger.info(f"{'='*60}")
            
            await example_func()
            
            logger.info(f"✅ {example_name} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ {example_name} failed: {e}")
        
        # Brief pause between examples
        await asyncio.sleep(1)
    
    logger.info("\n🎉 All database optimization examples completed!")

if __name__ == "__main__":
    asyncio.run(main()) 