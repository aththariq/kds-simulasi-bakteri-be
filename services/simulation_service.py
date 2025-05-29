"""
Simulation service for handling bacterial resistance evolution simulations.
"""

from typing import Dict, Any, Optional
import numpy as np
from models.population import Population
from models.selection import SelectionPressure
from models.fitness import FitnessCalculator
from models.resistance import ResistanceProfile


class SimulationService:
    """Service class for managing bacterial resistance simulations."""
    
    def __init__(self):
        self.active_simulations: Dict[str, Dict[str, Any]] = {}
    
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
        # Initialize population
        population = Population(size=initial_population_size)
        
        # Initialize selection pressure
        selection = SelectionPressure(
            pressure=selection_pressure,
            antibiotic_concentration=antibiotic_concentration
        )
        
        # Initialize fitness calculator
        fitness_calc = FitnessCalculator()
        
        # Store simulation data
        simulation_data = {
            "id": simulation_id,
            "status": "initialized",
            "parameters": {
                "initial_population_size": initial_population_size,
                "mutation_rate": mutation_rate,
                "selection_pressure": selection_pressure,
                "antibiotic_concentration": antibiotic_concentration,
                "simulation_time": simulation_time
            },
            "population": population,
            "selection": selection,
            "fitness_calculator": fitness_calc,
            "current_generation": 0,
            "results": {
                "population_history": [],
                "resistance_history": [],
                "fitness_history": []
            }
        }
        
        self.active_simulations[simulation_id] = simulation_data
        
        return {
            "simulation_id": simulation_id,
            "status": "initialized",
            "parameters": simulation_data["parameters"]
        }
    
    def run_simulation(self, simulation_id: str) -> Dict[str, Any]:
        """
        Run a complete simulation.
        
        Args:
            simulation_id: ID of the simulation to run
            
        Returns:
            Dictionary containing simulation results
        """
        if simulation_id not in self.active_simulations:
            raise ValueError(f"Simulation {simulation_id} not found")
        
        sim_data = self.active_simulations[simulation_id]
        sim_data["status"] = "running"
        
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
        
        sim_data["status"] = "completed"
        
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
            "current_generation": sim_data["current_generation"],
            "parameters": sim_data["parameters"],
            "results": sim_data["results"]
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
                "current_generation": sim_data["current_generation"],
                "parameters": sim_data["parameters"]
            })
        
        return {
            "active_simulations": len(simulations),
            "simulations": simulations
        } 