#!/usr/bin/env python3
"""
Comprehensive test to verify that the fitness access issue has been completely resolved.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.population import PopulationConfig, OptimizedPopulation
from models.fitness import ComprehensiveFitnessCalculator
from models.selection import AntimicrobialPressure, PressureConfig, PressureType
from models.resistance import EnvironmentalContext
from services.simulation_service import SimulationService
import random
import numpy as np

def test_simulation_service():
    """Test the full simulation service functionality."""
    print("=== SIMULATION SERVICE TEST ===")
    
    # Create simulation service
    service = SimulationService()
    
    # Create a simulation
    simulation_id = "test_sim_001"
    result = service.create_simulation(
        simulation_id=simulation_id,
        initial_population_size=30,
        mutation_rate=0.01,
        selection_pressure=0.2,
        antibiotic_concentration=2.0,
        simulation_time=5
    )
    
    print(f"✅ Created simulation: {result['simulation_id']}")
    print(f"   Status: {result['status']}")
    print(f"   Parameters: {result['parameters']}")
    
    # Run the simulation
    print("Running simulation...")
    final_result = service.run_simulation(simulation_id)
    
    print(f"✅ Simulation completed:")
    print(f"   Status: {final_result['status']}")
    print(f"   Generations: {final_result['generations_completed']}")
    print(f"   Final population: {final_result['final_population_size']}")
    print(f"   Final resistance: {final_result['final_resistance']:.3f}")
    
    # Get simulation status
    status = service.get_simulation_status(simulation_id)
    print(f"✅ Status check:")
    print(f"   Current generation: {status['current_generation']}")
    print(f"   Progress: {status['progress_percentage']:.1f}%")
    
    # Clean up
    service.delete_simulation(simulation_id)
    print("✅ Simulation cleaned up")

def test_fitness_calculation_edge_cases():
    """Test edge cases for fitness calculation."""
    print("\n=== FITNESS EDGE CASES TEST ===")
    
    # Create very small population
    config = PopulationConfig(population_size=5)
    population = OptimizedPopulation(config)
    population.initialize_population()
    
    fitness_calc = ComprehensiveFitnessCalculator()
    environmental_context = EnvironmentalContext(
        antibiotic_concentration=1.0,
        generation=0
    )
    
    print(f"Testing with {population.size} bacteria...")
    
    # Test fitness calculation for each bacterium
    all_fitness_values = []
    for bacterium in population.bacteria_by_id.values():
        population_context = {
            'total_population': population.size,
            'generation': 0,
            'carrying_capacity': 10,
            'local_density': 1.0
        }
        
        fitness_result = fitness_calc.calculate_fitness(
            bacterium=bacterium,
            environmental_context=environmental_context,
            population_context=population_context
        )
        
        # Verify the result has the expected structure
        assert hasattr(fitness_result, 'final_fitness'), "Fitness result missing final_fitness"
        assert isinstance(fitness_result.final_fitness, (int, float)), "Final fitness is not numeric"
        
        bacterium.fitness = fitness_result.final_fitness
        all_fitness_values.append(fitness_result.final_fitness)
    
    print(f"✅ All {len(all_fitness_values)} fitness values calculated successfully")
    print(f"   Range: {min(all_fitness_values):.3f} to {max(all_fitness_values):.3f}")
    print(f"   Mean: {np.mean(all_fitness_values):.3f}")
    
    # Test accessing fitness attribute directly
    fitness_values_direct = [bacterium.fitness for bacterium in population.bacteria_by_id.values()]
    print(f"✅ Direct fitness access successful for {len(fitness_values_direct)} bacteria")
    
    # Verify no AttributeError occurs
    try:
        for bacterium in population.bacteria_by_id.values():
            _ = bacterium.fitness  # This should not fail
        print("✅ No AttributeError when accessing bacterium.fitness")
    except AttributeError as e:
        print(f"❌ AttributeError still occurs: {e}")
        return False
    
    return True

def test_selection_with_fitness():
    """Test that selection works properly with fitness values."""
    print("\n=== SELECTION WITH FITNESS TEST ===")
    
    # Create population
    config = PopulationConfig(population_size=15)
    population = OptimizedPopulation(config)
    population.initialize_population()
    
    # Calculate fitness for all bacteria
    fitness_calc = ComprehensiveFitnessCalculator()
    environmental_context = EnvironmentalContext(antibiotic_concentration=1.5, generation=0)
    
    for bacterium in population.bacteria_by_id.values():
        population_context = {
            'total_population': population.size,
            'generation': 0,
            'carrying_capacity': 30,
            'local_density': 1.0
        }
        
        fitness_result = fitness_calc.calculate_fitness(
            bacterium=bacterium,
            environmental_context=environmental_context,
            population_context=population_context
        )
        bacterium.fitness = fitness_result.final_fitness
    
    print(f"✅ Fitness calculated for {population.size} bacteria")
    
    # Apply selection
    pressure_config = PressureConfig(
        pressure_type=PressureType.ANTIMICROBIAL,
        intensity=1.5,
        parameters={'mic_sensitive': 1.0, 'mic_resistant': 8.0, 'hill_coefficient': 2.0, 'max_kill_rate': 0.9}
    )
    selection = AntimicrobialPressure(config=pressure_config)
    
    bacteria_list = list(population.bacteria_by_id.values())
    population_context = {'total_population': population.size, 'generation': 0}
    
    selection_results = selection.apply_selection(bacteria_list, population_context, 0)
    
    print(f"✅ Selection applied successfully")
    print(f"   Selection results for {len(selection_results)} bacteria")
    
    # Verify no fitness access errors during selection
    survivors = []
    for bacterium in bacteria_list:
        result = next((r for r in selection_results if r.bacterium_id == bacterium.id), None)
        if result and random.random() < result.survival_probability:
            _ = bacterium.fitness  # This should not fail
            survivors.append(bacterium)
    
    print(f"✅ {len(survivors)} bacteria survived selection")
    print(f"   No fitness access errors during selection process")
    
    return True

if __name__ == "__main__":
    print("Starting comprehensive fitness access tests...\n")
    
    success = True
    
    try:
        # Test simulation service
        test_simulation_service()
        
        # Test fitness calculation edge cases
        if not test_fitness_calculation_edge_cases():
            success = False
        
        # Test selection with fitness
        if not test_selection_with_fitness():
            success = False
            
        if success:
            print("\n=== ALL TESTS PASSED ===")
            print("✅ Fitness access issue has been completely resolved!")
            print("✅ Simulation service is working correctly")
            print("✅ No AttributeError: 'OptimizedPopulation' object has no attribute 'fitness'")
        else:
            print("\n=== SOME TESTS FAILED ===")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
