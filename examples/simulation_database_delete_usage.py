"""
Example usage of MongoDB Delete Operations for Bacterial Simulation Database

This module demonstrates how to use the SimulationDeleteOperations class
to delete simulation data from MongoDB.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from models.database_models import (
    SimulationMetadata,
    SimulationParameters,
    PopulationSnapshot,
    BacteriumData,
    SimulationStatus
)
from services.simulation_database import (
    get_simulation_delete_operations,
    get_simulation_create_operations,
    get_simulation_update_operations,
    get_simulation_read_operations
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_soft_delete_simulation():
    """Example: Soft delete a simulation"""
    logger.info("=== Soft Delete Simulation Example ===")
    
    delete_ops = await get_simulation_delete_operations()
    simulation_id = "delete_example_001"
    deletion_reason = "Test simulation no longer needed"
    
    try:
        success = await delete_ops.soft_delete_simulation(
            simulation_id, deletion_reason
        )
        
        if success:
            logger.info(f"Successfully soft deleted simulation {simulation_id}")
            logger.info(f"Reason: {deletion_reason}")
        else:
            logger.warning(f"No changes made to simulation {simulation_id}")
            
    except Exception as e:
        logger.error(f"Failed to soft delete simulation: {e}")

async def example_hard_delete_simulation():
    """Example: Hard delete a simulation with related data"""
    logger.info("=== Hard Delete Simulation Example ===")
    
    delete_ops = await get_simulation_delete_operations()
    simulation_id = "delete_example_002"
    
    try:
        # Delete simulation and all related data
        deleted_counts = await delete_ops.hard_delete_simulation(
            simulation_id, delete_related_data=True
        )
        
        logger.info(f"Successfully hard deleted simulation {simulation_id}")
        logger.info(f"Deleted counts: {deleted_counts}")
        
    except Exception as e:
        logger.error(f"Failed to hard delete simulation: {e}")

async def example_hard_delete_simulation_only():
    """Example: Hard delete simulation without related data"""
    logger.info("=== Hard Delete Simulation Only Example ===")
    
    delete_ops = await get_simulation_delete_operations()
    simulation_id = "delete_example_003"
    
    try:
        # Delete only the main simulation document
        deleted_counts = await delete_ops.hard_delete_simulation(
            simulation_id, delete_related_data=False
        )
        
        logger.info(f"Successfully deleted simulation document {simulation_id}")
        logger.info(f"Deleted counts: {deleted_counts}")
        
    except Exception as e:
        logger.error(f"Failed to delete simulation: {e}")

async def example_delete_population_snapshots():
    """Example: Delete population snapshots by generation range"""
    logger.info("=== Delete Population Snapshots Example ===")
    
    delete_ops = await get_simulation_delete_operations()
    simulation_id = "delete_example_004"
    
    try:
        # Delete snapshots for generations 10-20
        deleted_count = await delete_ops.delete_population_snapshots(
            simulation_id, 
            generation_start=10,
            generation_end=20
        )
        
        logger.info(f"Deleted {deleted_count} population snapshots for simulation {simulation_id}")
        logger.info(f"Generation range: 10-20")
        
        # Delete all remaining snapshots
        remaining_deleted = await delete_ops.delete_population_snapshots(simulation_id)
        logger.info(f"Deleted {remaining_deleted} remaining population snapshots")
        
    except Exception as e:
        logger.error(f"Failed to delete population snapshots: {e}")

async def example_delete_bacteria_by_generation():
    """Example: Delete bacteria by specific generation"""
    logger.info("=== Delete Bacteria by Generation Example ===")
    
    delete_ops = await get_simulation_delete_operations()
    simulation_id = "delete_example_005"
    generation = 15
    
    try:
        deleted_count = await delete_ops.delete_bacteria_by_generation(
            simulation_id, generation
        )
        
        logger.info(f"Deleted {deleted_count} bacteria for simulation {simulation_id}")
        logger.info(f"Generation: {generation}")
        
    except Exception as e:
        logger.error(f"Failed to delete bacteria: {e}")

async def example_cleanup_old_simulations_dry_run():
    """Example: Cleanup old simulations in dry run mode"""
    logger.info("=== Cleanup Old Simulations (Dry Run) Example ===")
    
    delete_ops = await get_simulation_delete_operations()
    days_old = 30
    status_filter = SimulationStatus.COMPLETED
    
    try:
        result = await delete_ops.cleanup_old_simulations(
            days_old=days_old,
            status_filter=status_filter,
            dry_run=True
        )
        
        logger.info(f"Dry run results: {result}")
        logger.info(f"Would delete simulations older than {days_old} days with status {status_filter}")
        
    except Exception as e:
        logger.error(f"Failed to run cleanup dry run: {e}")

async def example_cleanup_old_simulations_actual():
    """Example: Actual cleanup of old simulations"""
    logger.info("=== Cleanup Old Simulations (Actual) Example ===")
    
    delete_ops = await get_simulation_delete_operations()
    days_old = 60  # More conservative for actual deletion
    status_filter = SimulationStatus.FAILED
    
    try:
        # First run dry run to see what would be deleted
        dry_run_result = await delete_ops.cleanup_old_simulations(
            days_old=days_old,
            status_filter=status_filter,
            dry_run=True
        )
        logger.info(f"Dry run: {dry_run_result}")
        
        # Ask for confirmation (in real usage, you might want user input)
        proceed = dry_run_result.get("would_delete", 0) > 0
        
        if proceed:
            # Perform actual cleanup
            actual_result = await delete_ops.cleanup_old_simulations(
                days_old=days_old,
                status_filter=status_filter,
                dry_run=False
            )
            logger.info(f"Cleanup completed: {actual_result}")
        else:
            logger.info("No simulations to clean up")
        
    except Exception as e:
        logger.error(f"Failed to cleanup old simulations: {e}")

async def example_complete_deletion_workflow():
    """Example: Complete workflow for managing simulation lifecycle"""
    logger.info("=== Complete Deletion Workflow Example ===")
    
    # Get all operations instances
    create_ops = await get_simulation_create_operations()
    update_ops = await get_simulation_update_operations()
    read_ops = await get_simulation_read_operations()
    delete_ops = await get_simulation_delete_operations()
    
    simulation_id = "lifecycle_sim_001"
    
    try:
        # 1. Create a test simulation
        parameters = SimulationParameters(
            initial_population=500,
            generations=20,
            mutation_rate=0.01,
            antibiotic_concentration=0.3
        )
        
        metadata = SimulationMetadata(
            simulation_id=simulation_id,
            name="Lifecycle Test Simulation",
            description="Simulation for testing deletion lifecycle",
            parameters=parameters,
            status=SimulationStatus.PENDING
        )
        
        object_id = await create_ops.create_simulation(metadata)
        logger.info(f"1. Created test simulation with ObjectId: {object_id}")
        
        # 2. Add some population data
        for generation in range(0, 6, 2):
            snapshot = PopulationSnapshot(
                generation=generation,
                total_population=500 + generation * 10,
                resistant_population=int((500 + generation * 10) * 0.2),
                susceptible_population=int((500 + generation * 10) * 0.8),
                average_fitness=0.75,
                resistance_frequency=0.2,
                bacteria=[]
            )
            
            await update_ops.append_population_snapshot(simulation_id, snapshot)
        
        logger.info("2. Added population snapshots for generations 0, 2, 4")
        
        # 3. Add some bacteria data
        bacteria_data = [
            BacteriumData(
                id=f"test_bact_{i}",
                fitness=0.8,
                resistance_genes=["ampR"] if i % 2 == 0 else [],
                generation=4,
                position_x=float(i * 10),
                position_y=float(i * 5)
            ) for i in range(10)
        ]
        
        await create_ops.batch_insert_bacteria(simulation_id, 4, bacteria_data)
        logger.info("3. Added bacteria data for generation 4")
        
        # 4. Verify initial state
        initial_simulation = await read_ops.get_simulation_by_id(simulation_id)
        if initial_simulation and initial_simulation.results:
            logger.info(f"4. Initial state: {len(initial_simulation.results.population_history)} snapshots")
        
        # 5. Delete specific generation data
        deleted_bacteria = await delete_ops.delete_bacteria_by_generation(simulation_id, 4)
        logger.info(f"5. Deleted {deleted_bacteria} bacteria from generation 4")
        
        # 6. Delete some population snapshots
        deleted_snapshots = await delete_ops.delete_population_snapshots(
            simulation_id, generation_start=2, generation_end=4
        )
        logger.info(f"6. Deleted {deleted_snapshots} population snapshots (generations 2-4)")
        
        # 7. Update simulation status to mark for deletion
        await update_ops.update_simulation_metadata(
            simulation_id, {
                "status": SimulationStatus.CANCELLED,
                "error_message": "Marked for deletion in test workflow"
            }
        )
        logger.info("7. Marked simulation for deletion")
        
        # 8. Soft delete the simulation
        soft_deleted = await delete_ops.soft_delete_simulation(
            simulation_id, "Completed lifecycle test"
        )
        if soft_deleted:
            logger.info("8. Soft deleted the simulation")
        
        # 9. Verify soft delete state
        soft_deleted_simulation = await read_ops.get_simulation_by_id(simulation_id)
        if soft_deleted_simulation:
            status = soft_deleted_simulation.metadata.status
            logger.info(f"9. Simulation status after soft delete: {status}")
        
        # 10. Finally, hard delete everything
        final_deleted = await delete_ops.hard_delete_simulation(
            simulation_id, delete_related_data=True
        )
        logger.info(f"10. Hard deleted simulation and related data: {final_deleted}")
        
        # 11. Verify complete deletion
        final_check = await read_ops.get_simulation_by_id(simulation_id)
        if final_check is None:
            logger.info("11. Confirmed: Simulation completely removed")
        else:
            logger.warning("11. Warning: Simulation still exists after hard delete")
        
        logger.info("Lifecycle workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Lifecycle workflow failed: {e}")
        
        # Cleanup on failure
        try:
            await delete_ops.hard_delete_simulation(simulation_id, delete_related_data=True)
            logger.info("Cleaned up test simulation after failure")
        except:
            pass  # Best effort cleanup

async def example_selective_data_management():
    """Example: Selective deletion for data management"""
    logger.info("=== Selective Data Management Example ===")
    
    delete_ops = await get_simulation_delete_operations()
    read_ops = await get_simulation_read_operations()
    
    simulation_id = "data_mgmt_sim_001"
    
    try:
        # Get current statistics
        stats = await read_ops.get_simulation_statistics(simulation_id)
        logger.info(f"Initial statistics: {stats}")
        
        # Strategy 1: Keep only recent generations (last 10)
        recent_generations_to_keep = 10
        
        # Get population history to determine what to delete
        pop_history = await read_ops.get_population_history(simulation_id)
        
        if pop_history:
            max_generation = max(snapshot.generation for snapshot in pop_history)
            cutoff_generation = max_generation - recent_generations_to_keep
            
            if cutoff_generation > 0:
                # Delete old population snapshots
                deleted_old_pops = await delete_ops.delete_population_snapshots(
                    simulation_id,
                    generation_start=0,
                    generation_end=cutoff_generation
                )
                logger.info(f"Deleted {deleted_old_pops} old population snapshots (gen 0-{cutoff_generation})")
                
                # Delete bacteria data for old generations
                total_deleted_bacteria = 0
                for gen in range(0, cutoff_generation + 1):
                    deleted_bacteria = await delete_ops.delete_bacteria_by_generation(
                        simulation_id, gen
                    )
                    total_deleted_bacteria += deleted_bacteria
                
                logger.info(f"Deleted {total_deleted_bacteria} bacteria from old generations")
        
        # Strategy 2: Archive simulation by soft delete but keep data
        await delete_ops.soft_delete_simulation(
            simulation_id, "Archived for data management"
        )
        logger.info(f"Archived simulation {simulation_id}")
        
        # Get final statistics
        final_stats = await read_ops.get_simulation_statistics(simulation_id)
        logger.info(f"Final statistics: {final_stats}")
        
    except Exception as e:
        logger.error(f"Selective data management failed: {e}")

async def main():
    """Run all delete operation examples"""
    logger.info("Starting MongoDB Delete Operations Examples")
    
    try:
        await example_soft_delete_simulation()
        await example_hard_delete_simulation()
        await example_hard_delete_simulation_only()
        await example_delete_population_snapshots()
        await example_delete_bacteria_by_generation()
        await example_cleanup_old_simulations_dry_run()
        await example_cleanup_old_simulations_actual()
        await example_complete_deletion_workflow()
        await example_selective_data_management()
        
        logger.info("All delete operation examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Delete operation examples failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 