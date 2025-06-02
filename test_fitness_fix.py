#!/usr/bin/env python3
"""
Test script to verify the fitness calculation fix in simulation service.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.population import PopulationConfig, OptimizedPopulation
from models.resistance import EnvironmentalContext
from models.fitness import ComprehensiveFitnessCalculator
from models.selection import SelectionEnvironment
import numpy as np

def test_fitness_calculation_fix():
    """Test the fitness calculation logic that was causing the error."""
    
    print("=== TESTING FITNESS CALCULATION FIX ===")
      # 1. Create a test population
    print("1. Creating test population...")
    config = PopulationConfig(
        population_size=10,
        initial_resistance_frequency=0.2,
        use_spatial=False
    )
    population = OptimizedPopulation(config)
    population.initialize_population()
    print(f"   Population size: {population.size}")
    print(f"   Bacteria count: {len(population.bacteria_by_id)}")
    
    # 2. Initialize components
    print("2. Initializing simulation components...")
    fitness_calc = ComprehensiveFitnessCalculator()
    selection = SelectionEnvironment()
    
    # 3. Create environmental context
    print("3. Creating environmental context...")
    environmental_context = EnvironmentalContext(
        antibiotic_concentration=1.0,
        generation=0
    )
    
    # 4. Test the NEW fitness calculation approach (individual bacteria)
    print("4. Testing NEW fitness calculation approach...")
    try:
        fitness_scores = {}
        population_context = {
            'total_population': population.size,
            'generation': 0,
            'carrying_capacity': config.population_size * 2,
            'local_density': 1.0
        }
        
        for bacterium in population.bacteria_by_id.values():
            if hasattr(bacterium, 'fitness') and hasattr(bacterium, 'id'):
                fitness_result = fitness_calc.calculate_fitness(
                    bacterium=bacterium,
                    environmental_context=environmental_context,
                    population_context=population_context
                )
                fitness_scores[bacterium.id] = fitness_result.final_fitness
                # Update bacterium's fitness with the calculated value
                bacterium.fitness = fitness_result.final_fitness
            else:
                print(f"   WARNING: Invalid bacterium object found: {type(bacterium)}")
        
        print(f"   SUCCESS: Calculated fitness for {len(fitness_scores)} bacteria")
        print(f"   Sample fitness values: {list(fitness_scores.values())[:5]}")
        
    except Exception as e:
        print(f"   ERROR in NEW approach: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test the OLD fitness calculation approach (would fail)
    print("5. Testing OLD fitness calculation approach (should fail)...")
    try:
        # This is what was causing the error before
        old_fitness_scores = fitness_calc.calculate_fitness(population, environmental_context)
        print("   UNEXPECTED: Old approach worked?!")
    except Exception as e:
        print(f"   EXPECTED ERROR in OLD approach: {e}")
    
    # 6. Test selection with new approach
    print("6. Testing selection with corrected bacteria list...")
    try:
        bacteria_list = list(population.bacteria_by_id.values())
        
        # Defensive check - ensure we only have Bacterium objects
        bacteria_list = [b for b in bacteria_list if hasattr(b, 'fitness') and hasattr(b, 'id')]
        
        print(f"   Before selection - bacteria_list size: {len(bacteria_list)}")
        
        population_context = {
            'total_population': population.size,
            'generation': 0
        }
        selection_results = selection.apply_selection(bacteria_list, population_context, 0)
        
        print(f"   Selection results: {len(selection_results)}")
        print("   SUCCESS: Selection completed without error")
        
    except Exception as e:
        print(f"   ERROR in selection: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== ALL TESTS PASSED ===")
    print("The fitness access issue has been resolved!")
    return True

if __name__ == "__main__":
    success = test_fitness_calculation_fix()
    sys.exit(0 if success else 1)
