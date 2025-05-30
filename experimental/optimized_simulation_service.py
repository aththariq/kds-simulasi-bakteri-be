"""
Optimized simulation service with performance enhancements.
"""

from typing import Dict, Any, Optional, Callable, AsyncGenerator
import asyncio
import numpy as np
from datetime import datetime
import uuid
import hashlib
import logging

from models.population import Population, PopulationConfig
from models.selection import SelectionPressure
from models.fitness import ComprehensiveFitnessCalculator
from utils.state_manager import state_manager, SimulationState
from utils.performance import (
    profiler, optimizer, cache_manager, memory_profiler,
    profile_performance, cached_result
)

logger = logging.getLogger(__name__)


class OptimizedSimulationService:
    """Optimized simulation service with performance enhancements."""
    
    def __init__(self):
        self.active_simulations: Dict[str, Dict[str, Any]] = {}
        self._simulation_callbacks: Dict[str, list] = {}
        
        # Performance tracking
        self.performance_enabled = True
        memory_profiler.take_snapshot("service_initialization")
    
    @profile_performance("create_simulation")
    def create_simulation(
        self,
        simulation_id: str,
        initial_population_size: int = 1000,
        mutation_rate: float = 0.001,
        selection_pressure: float = 0.1,
        antibiotic_concentration: float = 1.0,
        simulation_time: int = 100
    ) -> Dict[str, Any]:
        """Create a new simulation with optimized initialization."""
        
        # Check cache for similar simulations
        params_hash = self._generate_params_hash({
            "initial_population_size": initial_population_size,
            "mutation_rate": mutation_rate,
            "selection_pressure": selection_pressure,
            "antibiotic_concentration": antibiotic_concentration
        })
        
        cached_init = cache_manager.get('simulation_results', f"init_{params_hash}")
        if cached_init:
            logger.info(f"Using cached initialization for similar parameters")
        
        simulation_params = {
            "initial_population_size": initial_population_size,
            "mutation_rate": mutation_rate,
            "selection_pressure": selection_pressure,
            "antibiotic_concentration": antibiotic_concentration,
            "simulation_time": simulation_time,
            "params_hash": params_hash
        }
        
        # Create state in state manager
        state_data = state_manager.create_simulation_state(
            simulation_id=simulation_id,
            initial_params=simulation_params,
            metadata={
                "created_by": "optimized_simulation_service",
                "version": "2.0",
                "performance_enabled": self.performance_enabled
            }
        )
        
        try:
            # Initialize population with optimized configuration
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
                "current_generation": 0,
                "progress_percentage": 0.0,
                "estimated_completion": None,
                "performance_metrics": {
                    "generation_times": [],
                    "memory_usage": [],
                    "cache_hits": 0,
                    "cache_misses": 0
                },
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
            
            # Cache initialization data if successful
            if not cached_init:
                cache_manager.set('simulation_results', f"init_{params_hash}", {
                    "population_config": population_config,
                    "initialization_successful": True
                })
            
            # Update state manager
            state_manager.update_simulation_state(
                simulation_id,
                {
                    "population_size": initial_population_size,
                    "status": SimulationState.INITIALIZED
                }
            )
            
            memory_profiler.take_snapshot(f"simulation_created_{simulation_id}")
            
            return {
                "simulation_id": simulation_id,
                "status": "initialized",
                "created_at": simulation_data["created_at"],
                "parameters": simulation_data["parameters"],
                "optimization_enabled": True
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
    
    @profile_performance("run_simulation_async")
    async def run_simulation_async(self, simulation_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Run simulation with vectorized operations and optimization."""
        
        if simulation_id not in self.active_simulations:
            raise ValueError(f"Simulation {simulation_id} not found")
        
        sim_data = self.active_simulations[simulation_id]
        sim_data["status"] = "running"
        sim_data["updated_at"] = datetime.utcnow().isoformat()
        
        # Performance monitoring
        memory_profiler.take_snapshot(f"simulation_start_{simulation_id}")
        
        # Update state manager
        state_manager.update_simulation_state(
            simulation_id,
            {
                "state": SimulationState.RUNNING,
                "start_time": datetime.utcnow()
            }
        )
        
        population = sim_data["population"]
        selection = sim_data["selection"]
        fitness_calc = sim_data["fitness_calculator"]
        max_generations = sim_data["parameters"]["simulation_time"]
        mutation_rate = sim_data["parameters"]["mutation_rate"]
        
        start_time = datetime.utcnow()
        batch_size = min(10, max_generations // 10)  # Process in batches
        
        try:
            # Run simulation for specified number of generations
            for generation in range(max_generations):
                generation_start = datetime.utcnow()
                
                # Get population as numpy array for vectorized operations
                pop_array = population.get_population_array()
                
                # Check cache for fitness calculations
                pop_hash = self._generate_population_hash(pop_array)
                cached_fitness = cache_manager.get('fitness_scores', pop_hash)
                
                if cached_fitness is not None:
                    fitness_scores = cached_fitness
                    sim_data["performance_metrics"]["cache_hits"] += 1
                else:
                    # Use optimized fitness calculation
                    fitness_scores = self._optimized_fitness_calculation(
                        population, fitness_calc, pop_hash
                    )
                    sim_data["performance_metrics"]["cache_misses"] += 1
                
                # Apply vectorized selection
                if hasattr(selection, 'apply_vectorized_selection'):
                    selected_pop = selection.apply_vectorized_selection(
                        pop_array, fitness_scores
                    )
                    population.update_from_array(selected_pop)
                else:
                    # Fallback to regular selection
                    population = selection.apply_selection(population, fitness_scores)
                
                # Apply vectorized mutation
                if hasattr(population, 'mutate_vectorized'):
                    mutation_events = population.mutate_vectorized(mutation_rate)
                else:
                    # Use optimizer for vectorized mutation
                    mutated_pop = optimizer.vectorized_mutation(
                        population.get_population_array(), 
                        mutation_rate
                    )
                    population.update_from_array(mutated_pop)
                    mutation_events = []
                
                sim_data["metrics"]["total_mutations"] += len(mutation_events)
                
                # Calculate metrics
                post_mutation_resistance = population.get_average_resistance()
                generation_time = (datetime.utcnow() - generation_start).total_seconds()
                
                # Record compressed data for large simulations
                if generation % 10 == 0 or generation < 100:
                    sim_data["results"]["population_history"].append(population.size)
                    sim_data["results"]["resistance_history"].append(post_mutation_resistance)
                    sim_data["results"]["fitness_history"].append(float(np.mean(fitness_scores)))
                
                sim_data["results"]["generation_times"].append(generation_time)
                sim_data["performance_metrics"]["generation_times"].append(generation_time)
                
                # Memory usage tracking
                current_memory = memory_profiler.take_snapshot(
                    f"generation_{generation}"
                )['memory_usage']
                sim_data["performance_metrics"]["memory_usage"].append(current_memory)
                
                # Update progress tracking
                sim_data["current_generation"] = generation + 1
                sim_data["progress_percentage"] = ((generation + 1) / max_generations) * 100
                
                # Calculate diversity index (cached)
                diversity_key = f"diversity_{pop_hash}"
                diversity = cache_manager.get('api_responses', diversity_key)
                if diversity is None:
                    diversity = population.calculate_diversity_index()
                    cache_manager.set('api_responses', diversity_key, diversity)
                
                sim_data["metrics"]["diversity_index"].append(diversity)
                
                # Optimized state updates (batch every 5 generations)
                if generation % 5 == 0:
                    state_manager.update_simulation_state(
                        simulation_id,
                        {
                            "current_generation": sim_data["current_generation"],
                            "progress_percentage": sim_data["progress_percentage"],
                            "population_size": population.size,
                            "resistance_frequency": post_mutation_resistance,
                            "avg_generation_time": np.mean(sim_data["performance_metrics"]["generation_times"][-10:])
                        },
                        create_snapshot=(generation % 10 == 0)
                    )
                
                # Create checkpoint every 50 generations
                if generation % 50 == 0:
                    checkpoint_data = {
                        "population_array": population.get_population_array(),
                        "generation": generation,
                        "performance_metrics": sim_data["performance_metrics"]
                    }
                    state_manager.create_checkpoint(simulation_id, checkpoint_data)
                
                # Check for extinction
                if population.size == 0:
                    sim_data["metrics"]["extinction_events"] += 1
                    break
                
                # Prepare optimized progress update
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
                    "performance": {
                        "cache_hit_ratio": sim_data["performance_metrics"]["cache_hits"] / 
                                         max(1, sim_data["performance_metrics"]["cache_hits"] + 
                                             sim_data["performance_metrics"]["cache_misses"]),
                        "avg_generation_time": np.mean(sim_data["performance_metrics"]["generation_times"][-10:]) if sim_data["performance_metrics"]["generation_times"] else 0,
                        "memory_usage_mb": current_memory
                    }
                }
                
                # Notify callbacks and yield progress
                await self._notify_progress(simulation_id, progress_data)
                yield progress_data
                
                # Adaptive delay based on performance
                delay = 0.001 if generation_time < 0.1 else 0.01
                await asyncio.sleep(delay)
            
            # Simulation completion
            final_status = "completed" if population.size > 0 else "extinct"
            sim_data["status"] = final_status
            sim_data["updated_at"] = datetime.utcnow().isoformat()
            
            total_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Performance summary
            performance_summary = {
                "total_execution_time": total_time,
                "avg_generation_time": np.mean(sim_data["performance_metrics"]["generation_times"]),
                "cache_hit_ratio": sim_data["performance_metrics"]["cache_hits"] / 
                                 max(1, sim_data["performance_metrics"]["cache_hits"] + 
                                     sim_data["performance_metrics"]["cache_misses"]),
                "peak_memory_usage": max(sim_data["performance_metrics"]["memory_usage"]) if sim_data["performance_metrics"]["memory_usage"] else 0,
                "generations_per_second": sim_data["current_generation"] / total_time if total_time > 0 else 0
            }
            
            # Update state manager with completion
            state_manager.update_simulation_state(
                simulation_id,
                {
                    "state": SimulationState.COMPLETED,
                    "end_time": datetime.utcnow(),
                    "final_population_size": population.size,
                    "final_resistance": population.get_average_resistance() if population.size > 0 else 0,
                    "performance_summary": performance_summary
                }
            )
            
            # Create final optimized checkpoint
            final_checkpoint = {
                "population_array": population.get_population_array(),
                "generation": sim_data["current_generation"],
                "compressed_results": self._compress_results(sim_data["results"]),
                "performance_summary": performance_summary
            }
            state_manager.create_checkpoint(simulation_id, final_checkpoint)
            
            memory_profiler.take_snapshot(f"simulation_complete_{simulation_id}")
            
            final_results = {
                "simulation_id": simulation_id,
                "status": sim_data["status"],
                "generations_completed": sim_data["current_generation"],
                "final_population_size": population.size,
                "final_resistance": population.get_average_resistance() if population.size > 0 else 0,
                "total_time": total_time,
                "results": sim_data["results"],
                "metrics": sim_data["metrics"],
                "performance": performance_summary
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
    
    def _generate_params_hash(self, params: Dict[str, Any]) -> str:
        """Generate hash for simulation parameters."""
        params_str = str(sorted(params.items()))
        return hashlib.md5(params_str.encode()).hexdigest()[:12]
    
    def _generate_population_hash(self, pop_array: np.ndarray) -> str:
        """Generate hash for population state."""
        return hashlib.md5(pop_array.tobytes()).hexdigest()[:12]
    
    @cached_result('fitness_scores', lambda self, pop, calc, hash_key: hash_key)
    def _optimized_fitness_calculation(self, population, fitness_calc, pop_hash: str):
        """Optimized fitness calculation with caching."""
        return fitness_calc.calculate_fitness(population)
    
    def _compress_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compress simulation results for storage."""
        compressed = {}
        for key, data in results.items():
            if isinstance(data, list) and len(data) > 100:
                # Keep every 10th element for large datasets
                compressed[key] = data[::10]
            else:
                compressed[key] = data
        return compressed
    
    @profile_performance("register_progress_callback")
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
                    logger.error(f"Error in progress callback: {e}")
    
    def get_performance_metrics(self, simulation_id: str) -> Dict[str, Any]:
        """Get performance metrics for a simulation."""
        if simulation_id not in self.active_simulations:
            return {}
        
        sim_data = self.active_simulations[simulation_id]
        return sim_data.get("performance_metrics", {})
    
    def get_simulation_status(self, simulation_id: str) -> Dict[str, Any]:
        """Get the current status of a simulation with performance data."""
        if simulation_id not in self.active_simulations:
            raise ValueError(f"Simulation {simulation_id} not found")
        
        sim_data = self.active_simulations[simulation_id]
        
        # Get state manager data
        state_data = state_manager.get_simulation_state(simulation_id)
        
        return {
            "id": simulation_id,
            "status": sim_data["status"],
            "created_at": sim_data["created_at"],
            "updated_at": sim_data["updated_at"],
            "current_generation": sim_data["current_generation"],
            "progress_percentage": sim_data["progress_percentage"],
            "parameters": sim_data["parameters"],
            "performance_metrics": sim_data.get("performance_metrics", {}),
            "state_manager_data": state_data
        }
    
    def clear_simulation_cache(self, simulation_id: str):
        """Clear cached data for a specific simulation."""
        if simulation_id in self.active_simulations:
            params_hash = self.active_simulations[simulation_id]["parameters"]["params_hash"]
            cache_manager.invalidate('simulation_results', f"init_{params_hash}")
            cache_manager.invalidate('fitness_scores')
            cache_manager.invalidate('api_responses')
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return cache_manager.get_cache_stats()


# Global optimized instance
optimized_simulation_service = OptimizedSimulationService() 