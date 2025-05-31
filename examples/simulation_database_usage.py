"""
Example usage of MongoDB create operations for bacterial simulation data

This script demonstrates how to use the SimulationCreateOperations class
to save simulation data to MongoDB during a simulation run.
"""

import asyncio
import logging
from datetime import datetime
from typing import List

from services.simulation_database import get_simulation_create_operations, SimulationDatabaseError
from models.database_models import (
    SimulationMetadata,
    SimulationParameters,
    SimulationResults,
    PopulationSnapshot,
    BacteriumData,
    SimulationStatus
)
from utils.db_connection import ensure_database_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_create_simulation():
    """Example: Create a new simulation record"""
    try:
        # Get create operations instance
        create_ops = await get_simulation_create_operations()
        
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
        simulation_id = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        metadata = SimulationMetadata(
            simulation_id=simulation_id,
            name="High Mutation Rate Experiment",
            description="Testing bacterial evolution with increased mutation rate",
            status=SimulationStatus.PENDING,
            parameters=parameters,
            tags=["experiment", "high_mutation"]
        )
        
        # Create simulation in database
        object_id = await create_ops.create_simulation(metadata)
        logger.info(f"Created simulation {simulation_id} with ObjectId: {object_id}")
        
        return simulation_id, object_id
        
    except SimulationDatabaseError as e:
        logger.error(f"Failed to create simulation: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

async def example_save_population_data(simulation_id: str):
    """Example: Save population snapshots during simulation"""
    try:
        create_ops = await get_simulation_create_operations()
        
        # Simulate population data for multiple generations
        population_snapshots = []
        
        for generation in range(5):  # Save first 5 generations
            # Create sample bacteria data
            bacteria_data = []
            for i in range(10):  # Sample 10 bacteria per generation
                bacterium = BacteriumData(
                    id=f"bact_{generation}_{i:03d}",
                    fitness=0.5 + (i * 0.05),  # Variable fitness
                    resistance_genes=["ampR"] if i % 2 == 0 else ["ampR", "tetR"],
                    position_x=float(i * 10),
                    position_y=float(generation * 5),
                    generation=generation,
                    parent_id=f"bact_{generation-1}_{i//2:03d}" if generation > 0 else None,
                    mutation_count=generation
                )
                bacteria_data.append(bacterium)
            
            # Create population snapshot
            resistant_count = len([b for b in bacteria_data if "ampR" in b.resistance_genes])
            snapshot = PopulationSnapshot(
                generation=generation,
                total_population=1000 + (generation * 50),  # Growing population
                resistant_population=resistant_count * 100,  # Scale up from sample
                susceptible_population=(1000 + generation * 50) - (resistant_count * 100),
                average_fitness=sum(b.fitness for b in bacteria_data) / len(bacteria_data),
                resistance_frequency=resistant_count / len(bacteria_data),
                bacteria=bacteria_data
            )
            population_snapshots.append(snapshot)
        
        # Batch insert population snapshots
        pop_ids = await create_ops.batch_insert_population_snapshots(
            simulation_id, 
            population_snapshots
        )
        logger.info(f"Inserted {len(pop_ids)} population snapshots")
        
        # Insert bacteria data separately (for detailed analysis)
        for generation, snapshot in enumerate(population_snapshots):
            bact_ids = await create_ops.batch_insert_bacteria(
                simulation_id,
                generation,
                snapshot.bacteria
            )
            logger.info(f"Inserted {len(bact_ids)} bacteria for generation {generation}")
        
    except SimulationDatabaseError as e:
        logger.error(f"Failed to save population data: {e}")
        raise

async def example_complete_simulation_workflow():
    """Example: Complete simulation workflow from creation to completion"""
    simulation_id = None
    
    try:
        # Ensure database connection
        await ensure_database_connection()
        logger.info("Database connection established")
        
        # Step 1: Create simulation
        simulation_id, object_id = await example_create_simulation()
        
        # Step 2: Update status to running
        create_ops = await get_simulation_create_operations()
        await create_ops.update_simulation_status(
            simulation_id,
            SimulationStatus.RUNNING
        )
        logger.info(f"Started simulation {simulation_id}")
        
        # Step 3: Run simulation and save data
        start_time = asyncio.get_event_loop().time()
        
        # Save population data during simulation
        await example_save_population_data(simulation_id)
        
        # Step 4: Create final results
        end_time = asyncio.get_event_loop().time()
        runtime = end_time - start_time
        
        results = SimulationResults(
            simulation_id=simulation_id,
            metadata=SimulationMetadata(
                simulation_id=simulation_id,
                name="High Mutation Rate Experiment",
                description="Testing bacterial evolution with increased mutation rate",
                status=SimulationStatus.COMPLETED,
                parameters=SimulationParameters(
                    initial_population=1000,
                    generations=50,
                    mutation_rate=0.01,
                    antibiotic_concentration=0.5
                ),
                total_runtime=runtime
            ),
            population_history=[],  # Already saved separately
            final_statistics={
                "final_population": 1250,
                "final_resistance_frequency": 0.65,
                "generations_to_resistance": 8,
                "extinction_events": 0
            },
            performance_metrics={
                "avg_generation_time": runtime / 5,
                "memory_usage_mb": 85.2,
                "cpu_time_seconds": runtime * 0.8
            }
        )
        
        # Step 5: Save final results
        await create_ops.save_simulation_results(simulation_id, results)
        
        # Step 6: Update status to completed
        await create_ops.update_simulation_status(
            simulation_id,
            SimulationStatus.COMPLETED,
            runtime=runtime
        )
        
        logger.info(f"Successfully completed simulation {simulation_id}")
        
    except Exception as e:
        logger.error(f"Simulation workflow failed: {e}")
        
        # Update simulation status to failed if we have a simulation_id
        if simulation_id:
            try:
                create_ops = await get_simulation_create_operations()
                await create_ops.update_simulation_status(
                    simulation_id,
                    SimulationStatus.FAILED,
                    error_message=str(e)
                )
            except Exception as cleanup_error:
                logger.error(f"Failed to update error status: {cleanup_error}")
        
        raise

async def example_create_with_initial_data():
    """Example: Create simulation with initial population data"""
    try:
        create_ops = await get_simulation_create_operations()
        
        # Create initial bacteria
        initial_bacteria = [
            BacteriumData(
                id=f"initial_{i:03d}",
                fitness=0.7 + (i * 0.01),
                resistance_genes=["ampR"] if i < 5 else [],
                position_x=float(i * 2),
                position_y=0.0,
                generation=0,
                parent_id=None,
                mutation_count=0
            )
            for i in range(10)
        ]
        
        # Create initial population snapshot
        initial_population = PopulationSnapshot(
            generation=0,
            total_population=1000,
            resistant_population=500,
            susceptible_population=500,
            average_fitness=0.75,
            resistance_frequency=0.5,
            bacteria=initial_bacteria
        )
        
        # Create simulation parameters
        parameters = SimulationParameters(
            initial_population=1000,
            generations=25,
            mutation_rate=0.005,
            antibiotic_concentration=0.3
        )
        
        # Create simulation metadata
        simulation_id = f"initial_data_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        metadata = SimulationMetadata(
            simulation_id=simulation_id,
            name="Simulation with Initial Data",
            description="Example of creating simulation with predefined initial population",
            status=SimulationStatus.PENDING,
            parameters=parameters
        )
        
        # Create simulation with initial data
        object_id = await create_ops.create_simulation_with_initial_data(
            metadata,
            initial_population
        )
        
        logger.info(f"Created simulation {simulation_id} with initial data. ObjectId: {object_id}")
        return simulation_id
        
    except SimulationDatabaseError as e:
        logger.error(f"Failed to create simulation with initial data: {e}")
        raise

async def main():
    """Main function to run all examples"""
    logger.info("Starting MongoDB create operations examples...")
    
    try:
        # Example 1: Basic simulation creation
        logger.info("\n=== Example 1: Basic Simulation Creation ===")
        await example_create_simulation()
        
        # Example 2: Complete workflow
        logger.info("\n=== Example 2: Complete Simulation Workflow ===")
        await example_complete_simulation_workflow()
        
        # Example 3: Creation with initial data
        logger.info("\n=== Example 3: Simulation with Initial Data ===")
        await example_create_with_initial_data()
        
        logger.info("\nAll examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Examples failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 