"""
Example usage of the MongoDB Database Service for bacterial simulation data

This example demonstrates how to:
1. Initialize the database service
2. Create simulations with metadata
3. Save population snapshots and bacteria data
4. Update simulation status and save results
5. Handle errors and cleanup
"""

import asyncio
import logging
from datetime import datetime
from typing import List

from services.database_factory import database_service_context, get_singleton_database_service
from models.database_models import (
    SimulationMetadata, SimulationParameters, SimulationStatus,
    PopulationSnapshot, BacteriumData, SimulationResults
)
from config.database import DatabaseConfig
from utils.db_connection import DatabaseOperationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def create_sample_simulation_data():
    """Create sample simulation data for demonstration"""
    
    # Create simulation parameters
    parameters = SimulationParameters(
        initial_population=1000,
        generations=50,
        mutation_rate=0.01,
        antibiotic_concentration=0.5,
        selection_pressure=1.0,
        spatial_enabled=True,
        grid_size=100,
        hgt_enabled=True,
        hgt_rate=0.001
    )
    
    # Create simulation metadata
    metadata = SimulationMetadata(
        simulation_id=f"example_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        name="Example Bacterial Evolution Simulation",
        description="Demonstration of bacterial resistance evolution under antibiotic pressure",
        status=SimulationStatus.PENDING,
        parameters=parameters,
        tags=["example", "resistance", "evolution"]
    )
    
    return metadata

async def create_sample_population_data(generation: int = 0) -> PopulationSnapshot:
    """Create sample population snapshot data"""
    
    # Create sample bacteria
    bacteria = []
    for i in range(5):  # Small sample for demonstration
        bacterium = BacteriumData(
            id=f"bact_{generation}_{i:03d}",
            fitness=0.7 + (i * 0.05),  # Varying fitness
            resistance_genes=["ampR"] if i % 2 == 0 else ["tetR", "ampR"],
            position_x=float(i * 10),
            position_y=float(i * 15),
            generation=generation,
            parent_id=f"bact_{generation-1}_{i:03d}" if generation > 0 else None,
            mutation_count=generation
        )
        bacteria.append(bacterium)
    
    # Create population snapshot
    snapshot = PopulationSnapshot(
        generation=generation,
        total_population=1000,
        resistant_population=350 + (generation * 10),  # Increasing resistance
        susceptible_population=650 - (generation * 10),
        average_fitness=0.75 + (generation * 0.01),
        resistance_frequency=(350 + generation * 10) / 1000,
        bacteria=bacteria
    )
    
    return snapshot

async def example_basic_operations():
    """Demonstrate basic database operations"""
    logger.info("=== Basic Database Operations Example ===")
    
    try:
        # Use context manager for automatic cleanup
        async with database_service_context() as db_service:
            
            # 1. Create simulation
            logger.info("1. Creating simulation...")
            metadata = await create_sample_simulation_data()
            simulation_doc_id = await db_service.create_simulation(metadata)
            logger.info(f"Created simulation: {metadata.simulation_id} (Doc ID: {simulation_doc_id})")
            
            # 2. Update status to running
            logger.info("2. Updating simulation status...")
            await db_service.update_simulation_status(
                metadata.simulation_id, 
                SimulationStatus.RUNNING
            )
            logger.info("Status updated to RUNNING")
            
            # 3. Save initial population snapshot
            logger.info("3. Saving initial population snapshot...")
            initial_population = await create_sample_population_data(generation=0)
            pop_doc_id = await db_service.save_population_snapshot(
                metadata.simulation_id, 
                initial_population
            )
            logger.info(f"Saved population snapshot (Doc ID: {pop_doc_id})")
            
            # 4. Save bacteria batch
            logger.info("4. Saving bacteria batch...")
            bacteria_ids = await db_service.save_bacteria_batch(
                metadata.simulation_id,
                0,
                initial_population.bacteria
            )
            logger.info(f"Saved {len(bacteria_ids)} bacteria")
            
            # 5. Save individual bacterium
            logger.info("5. Saving individual bacterium...")
            new_bacterium = BacteriumData(
                id="special_bact_001",
                fitness=0.95,
                resistance_genes=["ampR", "tetR", "strR"],
                position_x=50.0,
                position_y=50.0,
                generation=0,
                parent_id=None,
                mutation_count=3
            )
            bact_doc_id = await db_service.save_individual_bacterium(
                metadata.simulation_id,
                0,
                new_bacterium
            )
            logger.info(f"Saved individual bacterium (Doc ID: {bact_doc_id})")
            
            # 6. Complete simulation with results
            logger.info("6. Completing simulation with results...")
            results = SimulationResults(
                simulation_id=metadata.simulation_id,
                metadata=metadata,
                population_history=[initial_population],
                final_statistics={
                    "final_population": 1200,
                    "final_resistance_frequency": 0.45,
                    "generations_to_resistance": 15,
                    "extinction_events": 0
                },
                performance_metrics={
                    "avg_generation_time": 0.85,
                    "memory_usage_mb": 125.5,
                    "cpu_time_seconds": 42.3
                }
            )
            
            await db_service.save_simulation_results(metadata.simulation_id, results)
            logger.info("Simulation completed and results saved")
            
            # 7. Get collection statistics
            logger.info("7. Getting collection statistics...")
            stats = await db_service.get_collection_stats()
            logger.info(f"Database stats: {stats}")
            
    except DatabaseOperationError as e:
        logger.error(f"Database operation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

async def example_simulation_with_initial_data():
    """Demonstrate creating simulation with initial data in one operation"""
    logger.info("=== Simulation with Initial Data Example ===")
    
    try:
        async with database_service_context() as db_service:
            
            # Create simulation metadata and initial population
            metadata = await create_sample_simulation_data()
            metadata.simulation_id = f"batch_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            initial_population = await create_sample_population_data(generation=0)
            
            # Create simulation with initial data in one operation
            logger.info("Creating simulation with initial data...")
            result = await db_service.create_simulation_with_initial_data(
                metadata, 
                initial_population
            )
            
            logger.info(f"Created simulation with initial data:")
            logger.info(f"  Simulation Doc ID: {result['simulation_id']}")
            logger.info(f"  Population Doc ID: {result['population_id']}")
            logger.info(f"  Bacteria Count: {result['bacteria_count']}")
            
    except DatabaseOperationError as e:
        logger.error(f"Database operation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

async def example_multi_generation_simulation():
    """Demonstrate saving data for multiple generations"""
    logger.info("=== Multi-Generation Simulation Example ===")
    
    try:
        async with database_service_context() as db_service:
            
            # Create simulation
            metadata = await create_sample_simulation_data()
            metadata.simulation_id = f"multi_gen_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            await db_service.create_simulation(metadata)
            await db_service.update_simulation_status(
                metadata.simulation_id, 
                SimulationStatus.RUNNING
            )
            
            # Simulate multiple generations
            population_history = []
            for generation in range(5):  # 5 generations for demo
                logger.info(f"Processing generation {generation}...")
                
                # Create population snapshot for this generation
                snapshot = await create_sample_population_data(generation)
                population_history.append(snapshot)
                
                # Save population snapshot
                await db_service.save_population_snapshot(
                    metadata.simulation_id, 
                    snapshot
                )
                
                # Save bacteria for this generation
                await db_service.save_bacteria_batch(
                    metadata.simulation_id,
                    generation,
                    snapshot.bacteria
                )
                
                logger.info(f"Saved generation {generation} data")
            
            # Complete simulation
            results = SimulationResults(
                simulation_id=metadata.simulation_id,
                metadata=metadata,
                population_history=population_history,
                final_statistics={
                    "final_population": 1200,
                    "final_resistance_frequency": 0.55,
                    "generations_to_resistance": 3,
                    "extinction_events": 0
                },
                performance_metrics={
                    "avg_generation_time": 1.2,
                    "memory_usage_mb": 200.0,
                    "cpu_time_seconds": 60.0
                }
            )
            
            await db_service.save_simulation_results(metadata.simulation_id, results)
            logger.info("Multi-generation simulation completed")
            
    except DatabaseOperationError as e:
        logger.error(f"Database operation failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

async def example_error_handling():
    """Demonstrate error handling scenarios"""
    logger.info("=== Error Handling Example ===")
    
    try:
        async with database_service_context() as db_service:
            
            # 1. Try to create duplicate simulation
            logger.info("1. Testing duplicate simulation creation...")
            metadata = await create_sample_simulation_data()
            metadata.simulation_id = "duplicate_test_sim"
            
            # Create first time (should succeed)
            await db_service.create_simulation(metadata)
            logger.info("First creation succeeded")
            
            # Try to create again (should fail)
            try:
                await db_service.create_simulation(metadata)
                logger.error("Duplicate creation should have failed!")
            except DatabaseOperationError as e:
                logger.info(f"Expected error caught: {e}")
            
            # 2. Try to save data for non-existent simulation
            logger.info("2. Testing operations on non-existent simulation...")
            try:
                snapshot = await create_sample_population_data()
                await db_service.save_population_snapshot("non_existent_sim", snapshot)
                logger.error("Should have failed for non-existent simulation!")
            except DatabaseOperationError as e:
                logger.info(f"Expected error caught: {e}")
            
            # 3. Try to update non-existent simulation status
            logger.info("3. Testing status update on non-existent simulation...")
            try:
                await db_service.update_simulation_status(
                    "non_existent_sim", 
                    SimulationStatus.COMPLETED
                )
                logger.error("Should have failed for non-existent simulation!")
            except DatabaseOperationError as e:
                logger.info(f"Expected error caught: {e}")
            
    except Exception as e:
        logger.error(f"Unexpected error in error handling example: {e}")

async def example_singleton_service():
    """Demonstrate using singleton service pattern"""
    logger.info("=== Singleton Service Example ===")
    
    try:
        # Get singleton service (creates if doesn't exist)
        db_service = await get_singleton_database_service()
        
        # Create a simple simulation
        metadata = await create_sample_simulation_data()
        metadata.simulation_id = f"singleton_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        await db_service.create_simulation(metadata)
        logger.info(f"Created simulation using singleton service: {metadata.simulation_id}")
        
        # Get the same singleton instance
        db_service2 = await get_singleton_database_service()
        assert db_service is db_service2, "Should be the same instance"
        logger.info("Confirmed singleton pattern working")
        
        # Get stats using the singleton
        stats = await db_service.get_collection_stats()
        logger.info(f"Stats from singleton: {stats}")
        
    except Exception as e:
        logger.error(f"Singleton service example error: {e}")

async def main():
    """Run all examples"""
    logger.info("Starting MongoDB Database Service Examples")
    
    # Run examples
    await example_basic_operations()
    await example_simulation_with_initial_data()
    await example_multi_generation_simulation()
    await example_error_handling()
    await example_singleton_service()
    
    logger.info("All examples completed")

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main()) 