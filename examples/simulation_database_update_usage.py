"""
Example usage of MongoDB Update Operations for Bacterial Simulation Database

This module demonstrates how to use the SimulationUpdateOperations class
to update simulation data in MongoDB.
"""

import asyncio
import logging
from datetime import datetime
from models.database_models import (
    SimulationMetadata,
    SimulationParameters,
    PopulationSnapshot,
    BacteriumData,
    SimulationStatus
)
from services.simulation_database import (
    get_simulation_update_operations,
    get_simulation_create_operations,
    get_simulation_read_operations
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_update_simulation_metadata():
    """Example: Update simulation metadata"""
    logger.info("=== Update Simulation Metadata Example ===")
    
    update_ops = await get_simulation_update_operations()
    simulation_id = "example_sim_001"
    
    # Update metadata fields
    metadata_updates = {
        "name": "Updated Simulation Name",
        "description": "Modified simulation description",
        "status": SimulationStatus.COMPLETED,
        "total_runtime": 125.5,
        "error_message": None  # Clear any error message
    }
    
    try:
        success = await update_ops.update_simulation_metadata(
            simulation_id, metadata_updates
        )
        
        if success:
            logger.info(f"Successfully updated metadata for simulation {simulation_id}")
        else:
            logger.warning(f"No changes made to simulation {simulation_id}")
            
    except Exception as e:
        logger.error(f"Failed to update metadata: {e}")

async def example_update_simulation_parameters():
    """Example: Update simulation parameters"""
    logger.info("=== Update Simulation Parameters Example ===")
    
    update_ops = await get_simulation_update_operations()
    simulation_id = "example_sim_001"
    
    # Update specific parameters
    parameter_updates = {
        "mutation_rate": 0.015,  # Increase mutation rate
        "antibiotic_concentration": 0.75,  # Increase antibiotic concentration
        "selection_pressure": 1.2,  # Adjust selection pressure
        "spatial_enabled": True,  # Enable spatial simulation
        "grid_size": 150  # Set grid size
    }
    
    try:
        success = await update_ops.update_simulation_parameters(
            simulation_id, parameter_updates
        )
        
        if success:
            logger.info(f"Successfully updated parameters for simulation {simulation_id}")
        else:
            logger.warning(f"No changes made to simulation {simulation_id}")
            
    except Exception as e:
        logger.error(f"Failed to update parameters: {e}")

async def example_append_population_snapshot():
    """Example: Append new population snapshot to existing simulation"""
    logger.info("=== Append Population Snapshot Example ===")
    
    update_ops = await get_simulation_update_operations()
    simulation_id = "example_sim_001"
    
    # Create sample bacteria data
    bacteria = [
        BacteriumData(
            id="new_bact_001",
            fitness=0.92,
            resistance_genes=["ampR", "tetR"],
            generation=25,
            position_x=50.5,
            position_y=75.2,
            mutation_count=3
        ),
        BacteriumData(
            id="new_bact_002",
            fitness=0.88,
            resistance_genes=["ampR"],
            generation=25,
            position_x=60.1,
            position_y=80.4,
            mutation_count=1
        )
    ]
    
    # Create population snapshot
    snapshot = PopulationSnapshot(
        generation=25,
        total_population=1250,
        resistant_population=800,
        susceptible_population=450,
        average_fitness=0.85,
        resistance_frequency=0.64,
        bacteria=bacteria
    )
    
    try:
        document_id = await update_ops.append_population_snapshot(
            simulation_id, snapshot
        )
        
        logger.info(f"Successfully appended population snapshot with ID: {document_id}")
        
    except Exception as e:
        logger.error(f"Failed to append population snapshot: {e}")

async def example_update_simulation_results():
    """Example: Update simulation results"""
    logger.info("=== Update Simulation Results Example ===")
    
    update_ops = await get_simulation_update_operations()
    simulation_id = "example_sim_001"
    
    # Update final statistics and performance metrics
    results_updates = {
        "final_statistics": {
            "final_population": 1500,
            "final_resistance_frequency": 0.72,
            "generations_to_resistance": 18,
            "extinction_events": 0,
            "max_fitness_achieved": 0.95,
            "genetic_diversity_index": 0.68
        },
        "performance_metrics": {
            "avg_generation_time": 1.2,
            "memory_usage_mb": 185.3,
            "cpu_time_seconds": 67.8,
            "peak_memory_mb": 225.1,
            "total_mutations": 1250
        }
    }
    
    try:
        success = await update_ops.update_simulation_results(
            simulation_id, results_updates
        )
        
        if success:
            logger.info(f"Successfully updated results for simulation {simulation_id}")
        else:
            logger.warning(f"No changes made to simulation {simulation_id}")
            
    except Exception as e:
        logger.error(f"Failed to update results: {e}")

async def example_manage_simulation_tags():
    """Example: Add and remove simulation tags"""
    logger.info("=== Manage Simulation Tags Example ===")
    
    update_ops = await get_simulation_update_operations()
    simulation_id = "example_sim_001"
    
    # Add tags
    new_tags = ["high_resistance", "spatial_analysis", "production_run"]
    
    try:
        success = await update_ops.add_simulation_tags(simulation_id, new_tags)
        
        if success:
            logger.info(f"Successfully added tags {new_tags} to simulation {simulation_id}")
        
        # Remove old tags
        old_tags = ["test", "experimental"]
        success = await update_ops.remove_simulation_tags(simulation_id, old_tags)
        
        if success:
            logger.info(f"Successfully removed tags {old_tags} from simulation {simulation_id}")
            
    except Exception as e:
        logger.error(f"Failed to manage tags: {e}")

async def example_complete_simulation_update_workflow():
    """Example: Complete workflow for updating a simulation"""
    logger.info("=== Complete Simulation Update Workflow ===")
    
    # Get operations instances
    create_ops = await get_simulation_create_operations()
    read_ops = await get_simulation_read_operations()
    update_ops = await get_simulation_update_operations()
    
    simulation_id = "workflow_sim_001"
    
    try:
        # 1. Create initial simulation
        parameters = SimulationParameters(
            initial_population=1000,
            generations=50,
            mutation_rate=0.01,
            antibiotic_concentration=0.5,
            spatial_enabled=True,
            grid_size=100
        )
        
        metadata = SimulationMetadata(
            simulation_id=simulation_id,
            name="Workflow Example Simulation",
            description="Example simulation for update workflow",
            parameters=parameters,
            status=SimulationStatus.PENDING
        )
        
        object_id = await create_ops.create_simulation(metadata)
        logger.info(f"Created simulation with ObjectId: {object_id}")
        
        # 2. Update status to running
        await update_ops.update_simulation_metadata(
            simulation_id, {"status": SimulationStatus.RUNNING}
        )
        logger.info("Updated status to RUNNING")
        
        # 3. Add some tags
        await update_ops.add_simulation_tags(
            simulation_id, ["workflow", "example", "batch_1"]
        )
        logger.info("Added initial tags")
        
        # 4. Simulate adding population data over time
        for generation in range(0, 10, 5):
            snapshot = PopulationSnapshot(
                generation=generation,
                total_population=1000 + generation * 20,
                resistant_population=int((1000 + generation * 20) * 0.3),
                susceptible_population=int((1000 + generation * 20) * 0.7),
                average_fitness=0.7 + generation * 0.01,
                resistance_frequency=0.3,
                bacteria=[]
            )
            
            await update_ops.append_population_snapshot(simulation_id, snapshot)
            logger.info(f"Added population snapshot for generation {generation}")
        
        # 5. Complete the simulation
        await update_ops.update_simulation_metadata(
            simulation_id, {
                "status": SimulationStatus.COMPLETED,
                "total_runtime": 45.2
            }
        )
        
        # 6. Add final results
        await update_ops.update_simulation_results(
            simulation_id, {
                "final_statistics": {
                    "final_population": 1200,
                    "final_resistance_frequency": 0.35,
                    "generations_to_resistance": 8
                },
                "performance_metrics": {
                    "avg_generation_time": 0.9,
                    "memory_usage_mb": 120.5
                }
            }
        )
        
        # 7. Add completion tags
        await update_ops.add_simulation_tags(
            simulation_id, ["completed", "successful"]
        )
        
        # 8. Verify final state
        final_simulation = await read_ops.get_simulation_by_id(simulation_id)
        if final_simulation:
            logger.info("Final simulation state:")
            logger.info(f"  Status: {final_simulation.metadata.status}")
            logger.info(f"  Runtime: {final_simulation.metadata.total_runtime}s")
            logger.info(f"  Tags: {final_simulation.metadata.tags}")
            
            if final_simulation.results:
                pop_count = len(final_simulation.results.population_history)
                logger.info(f"  Population snapshots: {pop_count}")
        
        logger.info("Workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        
        # Update simulation with error status
        try:
            await update_ops.update_simulation_metadata(
                simulation_id, {
                    "status": SimulationStatus.FAILED,
                    "error_message": str(e)
                }
            )
        except:
            pass  # Best effort error logging

async def main():
    """Run all update operation examples"""
    logger.info("Starting MongoDB Update Operations Examples")
    
    try:
        await example_update_simulation_metadata()
        await example_update_simulation_parameters()
        await example_append_population_snapshot()
        await example_update_simulation_results()
        await example_manage_simulation_tags()
        await example_complete_simulation_update_workflow()
        
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Examples failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 