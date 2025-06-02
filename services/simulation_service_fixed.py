"""
Simulation service for handling bacterial resistance evolution simulations.
"""

from typing import Dict, Any, Optional, Callable, AsyncGenerator
import asyncio
import numpy as np
import random
from datetime import datetime
import uuid
from models.population import Population, PopulationConfig
from models.selection import SelectionPressure
from models.fitness import ComprehensiveFitnessCalculator
from models.spatial import SpatialGrid, SpatialManager, BoundaryCondition
from models.resistance import EnvironmentalContext
from utils.state_manager import state_manager, SimulationState
import logging

logger = logging.getLogger(__name__)

class SimulationService:
    """Service class for managing bacterial resistance simulations."""
    
    def __init__(self):
        self.active_simulations: Dict[str, Dict[str, Any]] = {}
        self._simulation_callbacks: Dict[str, list] = {}
    
    def create_simulation(
        self,
        simulation_id: str,
        initial_population_size: int = 1000,
        mutation_rate: float = 0.001,
        selection_pressure: float = 0.1,
        antibiotic_concentration: float = 1.0,
        simulation_time: int = 100
    ) -> Dict[str, Any]:
        """
        Create a new simulation with the given parameters.
        
        Args:
            simulation_id: Unique identifier for the simulation
            initial_population_size: Starting population size
            mutation_rate: Rate of mutation per generation
            selection_pressure: Strength of selection pressure
            antibiotic_concentration: Concentration of antibiotic
            simulation_time: Number of generations to simulate
            
        Returns:
            Dictionary containing simulation metadata
        """
        # Prepare simulation parameters
        simulation_params = {
            "initial_population_size": initial_population_size,
            "mutation_rate": mutation_rate,
            "selection_pressure": selection_pressure,
            "antibiotic_concentration": antibiotic_concentration,
            "simulation_time": simulation_time
        }
        
        # Create state in state manager
        state_data = state_manager.create_simulation_state(
            simulation_id=simulation_id,
            initial_params=simulation_params,
            metadata={
                "created_by": "simulation_service",
                "version": "1.0"
            }
        )
        
        try:
            # Initialize population
            population_config = PopulationConfig(population_size=initial_population_size)
            population = Population(config=population_config)
            population.initialize_population()
            
            # Initialize selection pressure
            selection = SelectionPressure(
                pressure=selection_pressure,
                antibiotic_concentration=antibiotic_concentration
            )
            
            # Initialize fitness calculator
            fitness_calc = ComprehensiveFitnessCalculator()
            
            # Initialize spatial grid system for spatial representation
            spatial_grid = SpatialGrid(
                width=100.0,
                height=100.0,
                cell_size=1.0,
                boundary_condition=BoundaryCondition.CLOSED
            )
            spatial_manager = SpatialManager(spatial_grid)
            
            # Initialize bacterial positions randomly
            bacterium_ids = [f"bacterium_{i}" for i in range(initial_population_size)]
            bacterium_positions = spatial_manager.initialize_random_population(
                population_size=initial_population_size,
                bacterium_ids=bacterium_ids
            )
            
            # Enable performance optimizations for large populations
            if initial_population_size > 1000:
                spatial_manager.optimize_for_large_population(True)
                logger.info(f"Enabled performance optimizations for {initial_population_size} bacteria")
            
            # Store simulation data with enhanced metadata
            simulation_data = {
                "id": simulation_id,
                "status": "initialized",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "parameters": simulation_params,
                "population": population,
                "selection": selection,
                "fitness_calculator": fitness_calc,
                "spatial_grid": spatial_grid,
                "spatial_manager": spatial_manager,
                "bacterium_positions": bacterium_positions,
                "current_generation": 0,
                "progress_percentage": 0.0,
                "estimated_completion": None,
                "results": {
                    "population_history": [],
                    "resistance_history": [],
                    "fitness_history": [],
                    "generation_times": [],
                    "mutation_events": []
                },
                "metrics": {
                    "total_mutations": 0,
                    "extinction_events": 0,
                    "resistance_peaks": [],
                    "diversity_index": []
                }
            }
            
            self.active_simulations[simulation_id] = simulation_data
            self._simulation_callbacks[simulation_id] = []
            
            # Update state manager with simulation objects
            state_manager.update_simulation_state(
                simulation_id,
                {
                    "population_size": initial_population_size,
                    "status": SimulationState.INITIALIZED
                }
            )
            
            return {
                "simulation_id": simulation_id,
                "status": "initialized",
                "created_at": simulation_data["created_at"],
                "parameters": simulation_data["parameters"]
            }
            
        except Exception as e:
            # Update state with error
            state_manager.update_simulation_state(
                simulation_id,
                {
                    "status": SimulationState.ERROR,
                    "error_message": str(e)
                }
            )
            raise
    
    def register_progress_callback(self, simulation_id: str, callback: Callable):
        """Register a callback function for simulation progress updates."""
        if simulation_id in self._simulation_callbacks:
            self._simulation_callbacks[simulation_id].append(callback)
    
    async def _notify_progress(self, simulation_id: str, data: Dict[str, Any]):
        """Notify all registered callbacks about simulation progress."""
        if simulation_id in self._simulation_callbacks:
            for callback in self._simulation_callbacks[simulation_id]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    print(f"Error in progress callback: {e}")
    
    async def run_simulation_async(self, simulation_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run a simulation asynchronously with real-time progress updates.
        
        Args:
            simulation_id: ID of the simulation to run
            
        Yields:
            Progress updates throughout the simulation
        """
        if simulation_id not in self.active_simulations:
            raise ValueError(f"Simulation {simulation_id} not found")
        
        sim_data = self.active_simulations[simulation_id]
        sim_data["status"] = "running"
        sim_data["updated_at"] = datetime.utcnow().isoformat()
        
        # Update state manager
        state_manager.update_simulation_state(
            simulation_id,
            {
                "state": SimulationState.RUNNING,
                "start_time": datetime.utcnow()
            }        )
        
        population = sim_data["population"]
        selection = sim_data["selection"]
        fitness_calc = sim_data["fitness_calculator"]
        max_generations = sim_data["parameters"]["simulation_time"]
        mutation_rate = sim_data["parameters"]["mutation_rate"]
        
        start_time = datetime.utcnow()
        
        try:
            # Run simulation for specified number of generations
            for generation in range(max_generations):
                generation_start = datetime.utcnow()
                
                # Create environmental context with antibiotic concentration
                environmental_context = EnvironmentalContext(
                    antibiotic_concentration=sim_data["parameters"]["antibiotic_concentration"],
                    generation=generation
                )
                
                # Calculate fitness for current population with environmental context
                fitness_scores = fitness_calc.calculate_fitness(population, environmental_context)
                
                # Apply selection pressure
                population = selection.apply_selection(population, fitness_scores)
                
                # Track mutations before applying them
                pre_mutation_resistance = population.get_average_resistance()
                
                # CRITICAL FIX: Apply reproduction and aging FIRST before mutations
                # This ensures population growth happens
                population.advance_generation()
                
                # Apply mutations after reproduction
                mutation_events = population.mutate(mutation_rate)
                sim_data["metrics"]["total_mutations"] += len(mutation_events)
                
                # Calculate post-mutation metrics
                post_mutation_resistance = population.get_average_resistance()
                generation_time = (datetime.utcnow() - generation_start).total_seconds()
                
                # Record data for this generation
                sim_data["results"]["population_history"].append(population.size)
                sim_data["results"]["resistance_history"].append(post_mutation_resistance)
                sim_data["results"]["fitness_history"].append(float(np.mean(fitness_scores)))
                sim_data["results"]["generation_times"].append(generation_time)
                sim_data["results"]["mutation_events"].extend(mutation_events)
                
                # Update progress tracking
                sim_data["current_generation"] = generation + 1
                sim_data["progress_percentage"] = ((generation + 1) / max_generations) * 100
                
                # Calculate diversity index
                diversity = population.calculate_diversity_index()
                sim_data["metrics"]["diversity_index"].append(diversity)
                
                # Update spatial positions and extract spatial data
                spatial_grid = sim_data["spatial_grid"]
                spatial_manager = sim_data["spatial_manager"]
                bacterium_positions = sim_data["bacterium_positions"]
                
                # Simulate bacterial movement for this generation
                bacterium_ids_to_move = list(bacterium_positions.keys())[:population.size]
                
                # Use batch movement for better performance with large populations
                if len(bacterium_ids_to_move) > 100:
                    movements = spatial_manager.simulate_bacterial_movement_batch(
                        bacterium_ids=bacterium_ids_to_move,
                        movement_radius=0.3,
                        movement_probability=0.8
                    )
                    # Update positions from batch movement
                    for bacterium_id, new_position in movements:
                        if new_position:
                            bacterium_positions[bacterium_id] = new_position
                else:
                    # Use individual movement for smaller populations
                    for bacterium_id in bacterium_ids_to_move:
                        new_position = spatial_manager.simulate_bacterial_movement(
                            bacterium_id, 
                            movement_radius=0.3
                        )
                        if new_position:
                            bacterium_positions[bacterium_id] = new_position
                
                # Get spatial statistics
                grid_stats = spatial_grid.get_grid_statistics()
                  # Prepare spatial data for client updates with proper bacteria data
                actual_bacteria = list(population.bacteria_by_id.values())[:min(population.size, 100)]  # Limit for performance
                spatial_data = {
                    "bacteria": [
                        {
                            "id": bacterium.id,
                            "position": {"x": bacterium.position.x if bacterium.position else random.uniform(0, 100), 
                                       "y": bacterium.position.y if bacterium.position else random.uniform(0, 100)},
                            "resistance_status": bacterium.resistance_status.value if hasattr(bacterium.resistance_status, 'value') else bacterium.resistance_status,
                            "isResistant": bacterium.is_resistant,  # Add this for frontend compatibility
                            "fitness": float(bacterium.fitness),
                            "age": bacterium.age,
                            "generation": sim_data["current_generation"]
                        }
                        for bacterium in actual_bacteria
                    ],
                    "antibiotic_zones": [
                        {
                            "id": f"zone_{i}",
                            "center": {"x": zone[0], "y": zone[1]},
                            "radius": zone[2],
                            "concentration": zone[3]
                        }
                        for i, zone in enumerate(getattr(spatial_grid, 'antibiotic_zones', []))
                    ],
                    "grid_statistics": grid_stats,
                    "timestamp": datetime.utcnow().timestamp()
                }
                
                # Update state manager with progress
                state_manager.update_simulation_state(
                    simulation_id,
                    {
                        "current_generation": sim_data["current_generation"],
                        "progress_percentage": sim_data["progress_percentage"],
                        "population_size": population.size,
                        "resistance_frequency": post_mutation_resistance
                    },
                    create_snapshot=(generation % 10 == 0)  # Create snapshot every 10 generations
                )
                
                # Create checkpoint every 50 generations
                if generation % 50 == 0:
                    state_manager.create_checkpoint(simulation_id, {
                        "population": population,
                        "generation": generation,
                        "simulation_data": sim_data
                    })
                
                # Check for extinction
                if population.size == 0:
                    sim_data["metrics"]["extinction_events"] += 1
                    break
                
                # Prepare progress update
                progress_data = {
                    "simulation_id": simulation_id,
                    "status": "running",
                    "current_generation": sim_data["current_generation"],
                    "progress_percentage": sim_data["progress_percentage"],
                    "population_size": population.size,
                    "average_resistance": post_mutation_resistance,
                    "average_fitness": float(np.mean(fitness_scores)),
                    "diversity_index": diversity,
                    "generation_time": generation_time,
                    "mutations_this_generation": len(mutation_events),
                    "spatial_data": spatial_data
                }
                
                # Notify callbacks and yield progress
                await self._notify_progress(simulation_id, progress_data)
                yield progress_data
                
                # Small delay to allow other operations
                await asyncio.sleep(0.01)
            
            # Mark simulation as completed
            final_status = "completed" if population.size > 0 else "extinct"
            sim_data["status"] = final_status
            sim_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Update state manager with completion
            state_manager.update_simulation_state(
                simulation_id,
                {
                    "state": SimulationState.COMPLETED,
                    "end_time": datetime.utcnow(),
                    "final_population_size": population.size,
                    "final_resistance": population.get_average_resistance() if population.size > 0 else 0
                }
            )
            
            # Create final checkpoint
            state_manager.create_checkpoint(simulation_id, {
                "population": population,
                "generation": sim_data["current_generation"],
                "simulation_data": sim_data,
                "final_results": sim_data["results"]
            })
            
            final_results = {
                "simulation_id": simulation_id,
                "status": sim_data["status"],
                "generations_completed": sim_data["current_generation"],
                "final_population_size": population.size,
                "final_resistance": population.get_average_resistance() if population.size > 0 else 0,
                "total_time": (datetime.utcnow() - start_time).total_seconds(),
                "results": sim_data["results"],
                "metrics": sim_data["metrics"]
            }
            
            yield final_results
            
        except Exception as e:
            sim_data["status"] = "error"
            sim_data["error"] = str(e)
            sim_data["updated_at"] = datetime.utcnow().isoformat()
            
            # Update state manager with error
            state_manager.update_simulation_state(
                simulation_id,
                {
                    "state": SimulationState.ERROR,
                    "error_message": str(e),
                    "error_time": datetime.utcnow()
                }
            )
            
            error_data = {
                "simulation_id": simulation_id,
                "status": "error",
                "error": str(e),
                "current_generation": sim_data["current_generation"]
            }
            
            await self._notify_progress(simulation_id, error_data)
            yield error_data
    
    def run_simulation(self, simulation_id: str) -> Dict[str, Any]:
        """
        Run a complete simulation (legacy synchronous method).
        
        Args:
            simulation_id: ID of the simulation to run
            
        Returns:
            Dictionary containing simulation results
        """
        if simulation_id not in self.active_simulations:
            raise ValueError(f"Simulation {simulation_id} not found")

        sim_data = self.active_simulations[simulation_id]
        sim_data["status"] = "running"
        sim_data["updated_at"] = datetime.utcnow().isoformat()

        population = sim_data["population"]
        selection = sim_data["selection"]
        fitness_calc = sim_data["fitness_calculator"]
        max_generations = sim_data["parameters"]["simulation_time"]
        mutation_rate = sim_data["parameters"]["mutation_rate"]

        # Run simulation for specified number of generations
        for generation in range(max_generations):
            # Calculate fitness for current population
            fitness_scores = fitness_calc.calculate_fitness(population)

            # Apply selection pressure
            population = selection.apply_selection(population, fitness_scores)

            # Apply mutations
            population.mutate(mutation_rate)

            # Record data for this generation
            sim_data["results"]["population_history"].append(population.size)
            sim_data["results"]["resistance_history"].append(
                population.get_average_resistance()
            )
            sim_data["results"]["fitness_history"].append(
                float(np.mean(fitness_scores))
            )

            sim_data["current_generation"] = generation + 1
            sim_data["progress_percentage"] = ((generation + 1) / max_generations) * 100

        sim_data["status"] = "completed"
        sim_data["updated_at"] = datetime.utcnow().isoformat()

        return {
            "simulation_id": simulation_id,
            "status": "completed",
            "generations_completed": sim_data["current_generation"],
            "final_population_size": population.size,
            "final_resistance": population.get_average_resistance(),
            "results": sim_data["results"]
        }

    def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
        """
        Get the current status of a simulation.
        
        Args:
            simulation_id: ID of the simulation
            
        Returns:
            Dictionary containing simulation status and current data
        """
        if simulation_id not in self.active_simulations:
            raise ValueError(f"Simulation {simulation_id} not found")

        sim_data = self.active_simulations[simulation_id]

        return {
            "simulation_id": simulation_id,
            "status": sim_data["status"],
            "created_at": sim_data.get("created_at"),
            "updated_at": sim_data.get("updated_at"),
            "current_generation": sim_data["current_generation"],
            "progress_percentage": sim_data.get("progress_percentage", 0.0),
            "parameters": sim_data["parameters"],
            "results": sim_data["results"],
            "metrics": sim_data.get("metrics", {})
        }

    def delete_simulation(self, simulation_id: str) -> bool:
        """
        Delete a simulation from memory.
        
        Args:
            simulation_id: ID of the simulation to delete
            
        Returns:
            True if deleted successfully, False if not found
        """
        if simulation_id in self.active_simulations:
            del self.active_simulations[simulation_id]
            if simulation_id in self._simulation_callbacks:
                del self._simulation_callbacks[simulation_id]
            return True
        return False

    def list_simulations(self) -> Dict[str, Any]:
        """
        List all active simulations.
        
        Returns:
            Dictionary containing list of simulation summaries
        """
        simulations = []
        for sim_id, sim_data in self.active_simulations.items():
            simulations.append({
                "simulation_id": sim_id,
                "status": sim_data["status"],
                "created_at": sim_data.get("created_at"),
                "updated_at": sim_data.get("updated_at"),
                "current_generation": sim_data["current_generation"],
                "progress_percentage": sim_data.get("progress_percentage", 0.0),
                "parameters": sim_data["parameters"]
            })

        return {
            "active_simulations": len(simulations),
            "simulations": simulations
        }

    def pause_simulation(self, simulation_id: str) -> Dict[str, Any]:
        """Pause a running simulation."""
        if simulation_id not in self.active_simulations:
            raise ValueError(f"Simulation {simulation_id} not found")
        
        sim_data = self.active_simulations[simulation_id]
        if sim_data["status"] == "running":
            sim_data["status"] = "paused"
            sim_data["updated_at"] = datetime.utcnow().isoformat()
        
        return {"simulation_id": simulation_id, "status": sim_data["status"]}
    
    def resume_simulation(self, simulation_id: str) -> Dict[str, Any]:
        """Resume a paused simulation."""
        if simulation_id not in self.active_simulations:
            raise ValueError(f"Simulation {simulation_id} not found")
        
        sim_data = self.active_simulations[simulation_id]
        if sim_data["status"] == "paused":
            sim_data["status"] = "running"
            sim_data["updated_at"] = datetime.utcnow().isoformat()
        
        return {"simulation_id": simulation_id, "status": sim_data["status"]}
