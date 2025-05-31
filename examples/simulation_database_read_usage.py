"""
Example usage of MongoDB read operations for bacterial simulation data

This script demonstrates how to:
1. Retrieve individual simulations by ID or ObjectId
2. List simulations with pagination and filtering
3. Get population history with generation filtering
4. Retrieve bacteria data for specific generations
5. Get comprehensive simulation statistics
6. Search simulations by text queries
7. Get recent simulations
8. Check simulation existence

This example assumes that simulation data has already been created
using the create operations (see simulation_database_usage.py).
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional

from services.simulation_database import get_simulation_read_operations, SimulationDatabaseError
from models.database_models import SimulationStatus
from utils.db_connection import ensure_database_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def example_get_simulation_by_id():
    """Example: Retrieve a simulation by its unique ID"""
    logger.info("=== Get Simulation By ID Example ===")
    
    try:
        # Get read operations instance
        read_ops = await get_simulation_read_operations()
        
        # Try to get a specific simulation
        simulation_id = "example_sim_20231201_001"
        simulation = await read_ops.get_simulation_by_id(simulation_id)
        
        if simulation:
            logger.info(f"Found simulation: {simulation.simulation_id}")
            logger.info(f"  Name: {simulation.metadata.name}")
            logger.info(f"  Status: {simulation.metadata.status}")
            logger.info(f"  Created: {simulation.metadata.created_at}")
            logger.info(f"  Parameters: {simulation.metadata.parameters.dict()}")
        else:
            logger.info(f"Simulation {simulation_id} not found")
            
    except SimulationDatabaseError as e:
        logger.error(f"Failed to retrieve simulation: {e}")

async def example_list_simulations_with_pagination():
    """Example: List simulations with pagination and filtering"""
    logger.info("=== List Simulations with Pagination Example ===")
    
    try:
        read_ops = await get_simulation_read_operations()
        
        # List all completed simulations
        simulations, total_count = await read_ops.list_simulations(
            status_filter=SimulationStatus.COMPLETED,
            limit=5,
            offset=0,
            sort_by="created_at",
            sort_order="desc"
        )
        
        logger.info(f"Found {len(simulations)} completed simulations (total: {total_count})")
        
        for sim in simulations:
            logger.info(f"  - {sim.simulation_id}: {sim.metadata.name}")
            logger.info(f"    Status: {sim.metadata.status}, Created: {sim.metadata.created_at}")
        
        # List simulations with specific tags
        logger.info("\n--- Filtering by tags ---")
        tagged_sims, tag_count = await read_ops.list_simulations(
            tags_filter=["experiment", "high_mutation"],
            limit=10,
            offset=0
        )
        
        logger.info(f"Found {len(tagged_sims)} simulations with tags (total: {tag_count})")
        for sim in tagged_sims:
            logger.info(f"  - {sim.simulation_id}: Tags = {sim.metadata.tags}")
            
    except SimulationDatabaseError as e:
        logger.error(f"Failed to list simulations: {e}")

async def example_get_population_history():
    """Example: Retrieve population history for a simulation"""
    logger.info("=== Get Population History Example ===")
    
    try:
        read_ops = await get_simulation_read_operations()
        
        simulation_id = "example_sim_20231201_001"
        
        # Get complete population history
        logger.info("Getting complete population history...")
        all_snapshots = await read_ops.get_population_history(simulation_id)
        
        if all_snapshots:
            logger.info(f"Retrieved {len(all_snapshots)} population snapshots")
            logger.info(f"  Generation range: {all_snapshots[0].generation} - {all_snapshots[-1].generation}")
            logger.info(f"  Initial population: {all_snapshots[0].total_population}")
            logger.info(f"  Final population: {all_snapshots[-1].total_population}")
            logger.info(f"  Final resistance frequency: {all_snapshots[-1].resistance_frequency:.3f}")
        
        # Get population history for specific generation range
        logger.info("\n--- Getting generation range 10-20 ---")
        range_snapshots = await read_ops.get_population_history(
            simulation_id,
            generation_start=10,
            generation_end=20,
            include_bacteria=False  # Don't load individual bacteria for performance
        )
        
        logger.info(f"Retrieved {len(range_snapshots)} snapshots for generations 10-20")
        for snapshot in range_snapshots:
            logger.info(f"  Gen {snapshot.generation}: {snapshot.total_population} bacteria, "
                      f"{snapshot.resistance_frequency:.3f} resistance frequency")
            
    except SimulationDatabaseError as e:
        logger.error(f"Failed to retrieve population history: {e}")

async def example_get_bacteria_by_generation():
    """Example: Retrieve bacteria for a specific generation"""
    logger.info("=== Get Bacteria by Generation Example ===")
    
    try:
        read_ops = await get_simulation_read_operations()
        
        simulation_id = "example_sim_20231201_001"
        generation = 25
        
        # Get top 10 fittest bacteria for generation 25
        logger.info(f"Getting top 10 fittest bacteria for generation {generation}...")
        top_bacteria = await read_ops.get_bacteria_by_generation(
            simulation_id,
            generation=generation,
            limit=10,
            fitness_threshold=None
        )
        
        if top_bacteria:
            logger.info(f"Retrieved {len(top_bacteria)} bacteria")
            for i, bacterium in enumerate(top_bacteria, 1):
                logger.info(f"  {i}. {bacterium.id}: fitness={bacterium.fitness:.3f}, "
                          f"mutations={bacterium.mutation_count}, "
                          f"resistance_genes={bacterium.resistance_genes}")
        
        # Get highly fit bacteria (fitness > 0.9)
        logger.info(f"\n--- Getting highly fit bacteria (fitness > 0.9) ---")
        fit_bacteria = await read_ops.get_bacteria_by_generation(
            simulation_id,
            generation=generation,
            limit=None,
            fitness_threshold=0.9
        )
        
        logger.info(f"Found {len(fit_bacteria)} highly fit bacteria")
        
    except SimulationDatabaseError as e:
        logger.error(f"Failed to retrieve bacteria: {e}")

async def example_get_simulation_statistics():
    """Example: Get comprehensive simulation statistics"""
    logger.info("=== Get Simulation Statistics Example ===")
    
    try:
        read_ops = await get_simulation_read_operations()
        
        simulation_id = "example_sim_20231201_001"
        
        # Get comprehensive statistics
        stats = await read_ops.get_simulation_statistics(simulation_id)
        
        logger.info(f"Statistics for simulation: {stats['simulation_id']}")
        logger.info(f"  Name: {stats['simulation_name']}")
        logger.info(f"  Status: {stats['status']}")
        logger.info(f"  Created: {stats['created_at']}")
        logger.info(f"  Has Results: {stats['has_results']}")
        
        # Population statistics
        pop_stats = stats.get('population_statistics', {})
        if pop_stats:
            logger.info(f"\nPopulation Statistics:")
            logger.info(f"  Total Generations: {pop_stats.get('total_generations', 0)}")
            logger.info(f"  Population Range: {pop_stats.get('min_population', 0)} - {pop_stats.get('max_population', 0)}")
            logger.info(f"  Average Population: {pop_stats.get('avg_population', 0):.1f}")
            logger.info(f"  Final Resistance Frequency: {pop_stats.get('final_resistance_frequency', 0):.3f}")
            logger.info(f"  Max Resistance Frequency: {pop_stats.get('max_resistance_frequency', 0):.3f}")
        
        # Bacteria statistics
        bact_stats = stats.get('bacteria_statistics', {})
        if bact_stats:
            logger.info(f"\nBacteria Statistics:")
            logger.info(f"  Total Bacteria Tracked: {bact_stats.get('total_bacteria_tracked', 0)}")
            logger.info(f"  Average Fitness: {bact_stats.get('avg_fitness', 0):.3f}")
            logger.info(f"  Fitness Range: {bact_stats.get('min_fitness', 0):.3f} - {bact_stats.get('max_fitness', 0):.3f}")
            logger.info(f"  Average Mutations: {bact_stats.get('avg_mutations', 0):.2f}")
            
    except SimulationDatabaseError as e:
        logger.error(f"Failed to get simulation statistics: {e}")

async def example_search_simulations():
    """Example: Search simulations by text query"""
    logger.info("=== Search Simulations Example ===")
    
    try:
        read_ops = await get_simulation_read_operations()
        
        # Search for simulations containing "mutation" in name or description
        search_query = "mutation"
        simulations, total_count = await read_ops.search_simulations(
            search_query=search_query,
            search_fields=["metadata.name", "metadata.description"],
            limit=10,
            offset=0
        )
        
        logger.info(f"Search for '{search_query}' found {len(simulations)} results (total: {total_count})")
        
        for sim in simulations:
            logger.info(f"  - {sim.simulation_id}: {sim.metadata.name}")
            logger.info(f"    Description: {sim.metadata.description}")
        
        # Search by simulation ID pattern
        logger.info(f"\n--- Search by ID pattern ---")
        id_results, id_count = await read_ops.search_simulations(
            search_query="example_sim",
            search_fields=["simulation_id"],
            limit=5
        )
        
        logger.info(f"Found {len(id_results)} simulations with ID pattern (total: {id_count})")
        for sim in id_results:
            logger.info(f"  - {sim.simulation_id}")
            
    except SimulationDatabaseError as e:
        logger.error(f"Failed to search simulations: {e}")

async def example_get_recent_simulations():
    """Example: Get recently created simulations"""
    logger.info("=== Get Recent Simulations Example ===")
    
    try:
        read_ops = await get_simulation_read_operations()
        
        # Get 5 most recent simulations
        recent_sims = await read_ops.get_recent_simulations(limit=5)
        
        logger.info(f"Retrieved {len(recent_sims)} recent simulations:")
        
        for sim in recent_sims:
            logger.info(f"  - {sim.simulation_id}: {sim.metadata.name}")
            logger.info(f"    Created: {sim.metadata.created_at}, Status: {sim.metadata.status}")
            
    except SimulationDatabaseError as e:
        logger.error(f"Failed to get recent simulations: {e}")

async def example_check_simulation_existence():
    """Example: Check if simulations exist"""
    logger.info("=== Check Simulation Existence Example ===")
    
    try:
        read_ops = await get_simulation_read_operations()
        
        # Check if various simulations exist
        simulation_ids = [
            "example_sim_20231201_001",
            "nonexistent_simulation",
            "another_test_sim"
        ]
        
        for sim_id in simulation_ids:
            exists = await read_ops.simulation_exists(sim_id)
            status = "EXISTS" if exists else "NOT FOUND"
            logger.info(f"  {sim_id}: {status}")
            
    except SimulationDatabaseError as e:
        logger.error(f"Failed to check simulation existence: {e}")

async def example_complex_query_workflow():
    """Example: Complex workflow combining multiple read operations"""
    logger.info("=== Complex Query Workflow Example ===")
    
    try:
        read_ops = await get_simulation_read_operations()
        
        # 1. Find all completed simulations
        completed_sims, _ = await read_ops.list_simulations(
            status_filter=SimulationStatus.COMPLETED,
            limit=50
        )
        
        logger.info(f"Found {len(completed_sims)} completed simulations")
        
        # 2. For each simulation, get basic statistics
        for sim in completed_sims[:3]:  # Limit to first 3 for demo
            logger.info(f"\nAnalyzing simulation: {sim.simulation_id}")
            
            # Get population history summary
            snapshots = await read_ops.get_population_history(
                sim.simulation_id,
                include_bacteria=False
            )
            
            if snapshots:
                initial_pop = snapshots[0].total_population
                final_pop = snapshots[-1].total_population
                final_resistance = snapshots[-1].resistance_frequency
                
                logger.info(f"  Generations: {len(snapshots)}")
                logger.info(f"  Population: {initial_pop} → {final_pop}")
                logger.info(f"  Final resistance frequency: {final_resistance:.3f}")
                
                # Get top bacteria from final generation
                final_gen = snapshots[-1].generation
                top_bacteria = await read_ops.get_bacteria_by_generation(
                    sim.simulation_id,
                    generation=final_gen,
                    limit=3
                )
                
                if top_bacteria:
                    logger.info(f"  Top bacteria fitness: {[b.fitness for b in top_bacteria]}")
            
    except SimulationDatabaseError as e:
        logger.error(f"Complex workflow failed: {e}")

async def main():
    """Main function to run all read operation examples"""
    logger.info("Starting MongoDB read operations examples...")
    
    try:
        # Ensure database connection
        await ensure_database_connection()
        
        # Run all examples
        examples = [
            example_get_simulation_by_id,
            example_list_simulations_with_pagination,
            example_get_population_history,
            example_get_bacteria_by_generation,
            example_get_simulation_statistics,
            example_search_simulations,
            example_get_recent_simulations,
            example_check_simulation_existence,
            example_complex_query_workflow
        ]
        
        for example_func in examples:
            try:
                await example_func()
                logger.info("")  # Add spacing between examples
            except Exception as e:
                logger.error(f"Example {example_func.__name__} failed: {e}")
                logger.info("")
        
        logger.info("All read operation examples completed!")
        
    except Exception as e:
        logger.error(f"Examples failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 