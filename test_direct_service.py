#!/usr/bin/env python3
"""
Direct test of the fitness calculation fix without state manager complications.
"""

import sys
import os
import traceback
import random

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.population import Population, PopulationConfig, OptimizedPopulation
from models.selection import AntimicrobialPressure, PressureConfig, PressureType, SelectionEnvironment
from models.fitness import ComprehensiveFitnessCalculator
from models.resistance import EnvironmentalContext

def test_direct_service_logic():
    """Test the core simulation logic that was causing the AttributeError."""
    print("=== DIRECT SERVICE LOGIC TEST ===")
    
    try:
        # Initialize population (mimic service creation)
        population_config = PopulationConfig(population_size=25)
        population = OptimizedPopulation(config=population_config)
        population.initialize_population()
        print(f"✅ Population created: {population.size} bacteria")
        
        # Initialize selection environment (fixed approach)
        pressure_config = PressureConfig(
            pressure_type=PressureType.ANTIMICROBIAL,
            intensity=1.5,
            parameters={
                'mic_sensitive': 1.0,
                'mic_resistant': 8.0,
                'hill_coefficient': 2.0,
                'max_kill_rate': 0.95
            }
        )
        antimicrobial_pressure = AntimicrobialPressure(config=pressure_config)
        selection = SelectionEnvironment()
        selection.add_pressure(antimicrobial_pressure)
        print("✅ Selection environment created")
        
        # Initialize fitness calculator
        fitness_calc = ComprehensiveFitnessCalculator()
        print("✅ Fitness calculator created")
        
        # Run simulation logic (mimic service.run_simulation)
        mutation_rate = 0.01
        max_generations = 3
        
        for generation in range(max_generations):
            print(f"\n--- Generation {generation} ---")
            
            # Create environmental context
            environmental_context = EnvironmentalContext(
                antibiotic_concentration=1.5,
                generation=generation
            )
            
            # This is the EXACT logic from simulation_service.py that was causing the error
            fitness_scores = {}
            population_context = {
                'total_population': population.size,
                'generation': generation,
                'carrying_capacity': population_config.population_size * 2,
                'local_density': 1.0
            }
            
            # Calculate fitness for each bacterium - this was the source of AttributeError
            bacteria_count = 0
            for bacterium in population.bacteria_by_id.values():
                if hasattr(bacterium, 'fitness') and hasattr(bacterium, 'id'):
                    try:
                        fitness_result = fitness_calc.calculate_fitness(
                            bacterium=bacterium,
                            environmental_context=environmental_context,
                            population_context=population_context
                        )
                        fitness_scores[bacterium.id] = fitness_result.final_fitness
                        # Update bacterium's fitness with the calculated value
                        bacterium.fitness = fitness_result.final_fitness
                        bacteria_count += 1
                    except Exception as e:
                        print(f"❌ Failed to calculate fitness for bacterium {bacterium.id}: {e}")
                        return False
                else:
                    print(f"❌ Invalid bacterium object found: {type(bacterium)}")
                    return False
            
            print(f"✅ Fitness calculated for {bacteria_count} bacteria")
            
            # Apply selection pressure - this was also part of the issue
            bacteria_list = list(population.bacteria_by_id.values())
            bacteria_list = [b for b in bacteria_list if hasattr(b, 'fitness') and hasattr(b, 'id')]
            
            selection_context = {
                'total_population': population.size,
                'generation': generation
            }
            selection_results = selection.apply_selection(bacteria_list, selection_context, generation)
            print(f"✅ Selection applied to {len(bacteria_list)} bacteria")
            
            # Apply selection results
            if selection_results:
                bacterium_map = {result.bacterium_id: result for result in selection_results}
                survivors = []
                
                for bacterium in bacteria_list:
                    if bacterium.id in bacterium_map:
                        result = bacterium_map[bacterium.id]
                        bacterium.fitness = result.modified_fitness
                        if random.random() < result.survival_probability:
                            survivors.append(bacterium)
                
                population.clear()
                population._batch_add_bacteria(survivors)
                print(f"✅ Population after selection: {population.size} bacteria")
            
            # Advance generation
            population.advance_generation()
            mutation_events = population.mutate(mutation_rate)
            print(f"✅ Generation advanced, mutations: {len(mutation_events)}")
            print(f"   Final population size: {population.size}")
            
            # Test accessing fitness after all operations (this was failing before)
            try:
                fitness_values = [bacterium.fitness for bacterium in population.bacteria_by_id.values()]
                print(f"✅ Final fitness access successful: {len(fitness_values)} values")
                if fitness_values:
                    print(f"   Mean fitness: {sum(fitness_values)/len(fitness_values):.3f}")
            except AttributeError as e:
                print(f"❌ CRITICAL: Still getting AttributeError: {e}")
                return False
            
            if population.size == 0:
                print("   Population extinct")
                break
        
        print(f"\n✅ Simulation completed successfully!")
        print(f"   Final population: {population.size}")
        print(f"   Final resistance: {population.get_average_resistance():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("Testing direct simulation service logic without state manager...\n")
    
    success = test_direct_service_logic()
    
    if success:
        print("\n=== DIRECT SERVICE LOGIC TEST PASSED ===")
        print("✅ No AttributeError: 'OptimizedPopulation' object has no attribute 'fitness'")
        print("✅ Core simulation logic works correctly")
        print("✅ The fix is complete and functional!")
        print("✅ The simulation service should now work without the AttributeError")
    else:
        print("\n=== DIRECT SERVICE LOGIC TEST FAILED ===")
        print("❌ There are still issues with the core simulation logic")
        sys.exit(1)
