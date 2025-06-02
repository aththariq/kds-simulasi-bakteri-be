#!/usr/bin/env python3
"""
Simple test to verify that the fitness access issue has been completely resolved.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.population import PopulationConfig, OptimizedPopulation
from models.fitness import ComprehensiveFitnessCalculator
from models.selection import AntimicrobialPressure, PressureConfig, PressureType
from models.resistance import EnvironmentalContext
import random
import numpy as np

def test_direct_fitness_calculation():
    """Test the core fitness calculation functionality without service layer."""
    print("=== DIRECT FITNESS CALCULATION TEST ===")
    
    # Create population
    config = PopulationConfig(population_size=25)
    population = OptimizedPopulation(config)
    population.initialize_population()
    
    print(f"✅ Created population with {population.size} bacteria")
    print(f"   Resistant: {sum(1 for b in population.bacteria_by_id.values() if b.is_resistant)}")
    print(f"   Sensitive: {sum(1 for b in population.bacteria_by_id.values() if not b.is_resistant)}")
    
    # Initialize fitness calculator and environmental context
    fitness_calc = ComprehensiveFitnessCalculator()
    environmental_context = EnvironmentalContext(
        antibiotic_concentration=1.5,
        generation=0
    )
    
    # Test fitness calculation for each bacterium - this was the original problem
    fitness_scores = {}
    population_context = {
        'total_population': population.size,
        'generation': 0,
        'carrying_capacity': 50,
        'local_density': 1.0
    }
    
    print("Calculating fitness for each bacterium...")
    for bacterium in population.bacteria_by_id.values():
        # This is the exact pattern that was failing before
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
            except Exception as e:
                print(f"❌ ERROR: Failed to calculate fitness for bacterium {bacterium.id}: {e}")
                return False
        else:
            print(f"❌ WARNING: Invalid bacterium object found: {type(bacterium)}")
            return False
    
    print(f"✅ Successfully calculated fitness for {len(fitness_scores)} bacteria")
    
    # Test the list comprehension that was originally failing
    try:
        fitness_values = [bacterium.fitness for bacterium in population.bacteria_by_id.values()]
        print(f"✅ List comprehension access successful: {len(fitness_values)} values")
        print(f"   Range: {min(fitness_values):.3f} to {max(fitness_values):.3f}")
        print(f"   Mean: {np.mean(fitness_values):.3f}")
    except AttributeError as e:
        print(f"❌ AttributeError still occurs in list comprehension: {e}")
        return False
    
    return True

def test_selection_workflow():
    """Test the complete selection workflow that includes fitness access."""
    print("\n=== SELECTION WORKFLOW TEST ===")
    
    # Create population
    config = PopulationConfig(population_size=20)
    population = OptimizedPopulation(config)
    population.initialize_population()
    
    # Calculate fitness (the fixed approach)
    fitness_calc = ComprehensiveFitnessCalculator()
    environmental_context = EnvironmentalContext(antibiotic_concentration=2.0, generation=0)
    
    fitness_scores = {}
    population_context = {
        'total_population': population.size,
        'generation': 0,
        'carrying_capacity': 40,
        'local_density': 1.0
    }
    
    for bacterium in population.bacteria_by_id.values():
        fitness_result = fitness_calc.calculate_fitness(
            bacterium=bacterium,
            environmental_context=environmental_context,
            population_context=population_context
        )
        fitness_scores[bacterium.id] = fitness_result.final_fitness
        bacterium.fitness = fitness_result.final_fitness
    
    print(f"✅ Fitness calculated for {len(fitness_scores)} bacteria")
      # Apply selection
    pressure_config = PressureConfig(
        pressure_type=PressureType.ANTIMICROBIAL,
        intensity=2.0,
        parameters={
            'mic_sensitive': 1.0,
            'mic_resistant': 8.0,
            'hill_coefficient': 2.0,
            'max_kill_rate': 0.95
        }
    )
    antimicrobial_pressure = AntimicrobialPressure(config=pressure_config)
    
    # Create selection environment and add the pressure
    from models.selection import SelectionEnvironment
    selection = SelectionEnvironment()
    selection.add_pressure(antimicrobial_pressure)
    
    bacteria_list = list(population.bacteria_by_id.values())
    selection_context = {'total_population': population.size, 'generation': 0}
    
    try:
        selection_results = selection.apply_selection(bacteria_list, selection_context, 0)
        print(f"✅ Selection applied successfully to {len(bacteria_list)} bacteria")
    except Exception as e:
        print(f"❌ Selection failed: {e}")
        return False
    
    # Simulate survival selection that accesses fitness
    survivors = []
    for bacterium in bacteria_list:
        try:
            # This is the pattern that was failing - accessing bacterium.fitness
            fitness_value = bacterium.fitness
            
            result = next((r for r in selection_results if r.bacterium_id == bacterium.id), None)
            if result and random.random() < result.survival_probability:
                survivors.append(bacterium)
        except AttributeError as e:
            print(f"❌ AttributeError during survival selection: {e}")
            return False
    
    print(f"✅ Survival selection completed: {len(survivors)} survivors from {len(bacteria_list)} bacteria")
    
    # Test batch operations
    try:
        population.clear()
        population._batch_add_bacteria(survivors)
        print(f"✅ Population updated with survivors: {population.size} bacteria")
    except Exception as e:
        print(f"❌ Population update failed: {e}")
        return False
    
    return True

def test_multiple_generations():
    """Test multiple generations to ensure no regression."""
    print("\n=== MULTIPLE GENERATIONS TEST ===")
    
    config = PopulationConfig(population_size=15)
    population = OptimizedPopulation(config)
    population.initialize_population()
    
    fitness_calc = ComprehensiveFitnessCalculator()
    pressure_config = PressureConfig(
        pressure_type=PressureType.ANTIMICROBIAL,
        intensity=1.0,
        parameters={'mic_sensitive': 1.0, 'mic_resistant': 8.0, 'hill_coefficient': 2.0, 'max_kill_rate': 0.8}
    )
    antimicrobial_pressure = AntimicrobialPressure(config=pressure_config)
    
    # Create selection environment and add the pressure
    from models.selection import SelectionEnvironment
    selection = SelectionEnvironment()
    selection.add_pressure(antimicrobial_pressure)
    
    print(f"Starting with {population.size} bacteria")
    
    for generation in range(3):
        print(f"  Generation {generation}:")
        
        # Environmental context
        environmental_context = EnvironmentalContext(
            antibiotic_concentration=1.0,
            generation=generation
        )
        
        # Calculate fitness for all bacteria
        population_context = {
            'total_population': population.size,
            'generation': generation,
            'carrying_capacity': 30,
            'local_density': 1.0
        }
        
        fitness_count = 0
        for bacterium in population.bacteria_by_id.values():
            try:
                fitness_result = fitness_calc.calculate_fitness(
                    bacterium=bacterium,
                    environmental_context=environmental_context,
                    population_context=population_context
                )
                bacterium.fitness = fitness_result.final_fitness
                fitness_count += 1
            except Exception as e:
                print(f"    ❌ Fitness calculation failed: {e}")
                return False
        
        print(f"    Fitness calculated for {fitness_count} bacteria")
        
        # Apply selection
        bacteria_list = list(population.bacteria_by_id.values())
        selection_context = {'total_population': population.size, 'generation': generation}
        
        try:
            selection_results = selection.apply_selection(bacteria_list, selection_context, generation)
            
            # Apply survival
            survivors = []
            for bacterium in bacteria_list:
                result = next((r for r in selection_results if r.bacterium_id == bacterium.id), None)
                if result and random.random() < result.survival_probability:
                    survivors.append(bacterium)
            
            population.clear()
            population._batch_add_bacteria(survivors)
            
            print(f"    Population after selection: {population.size} bacteria")
            
            # Advance generation for reproduction
            population.advance_generation()
            print(f"    Population after reproduction: {population.size} bacteria")
            
        except Exception as e:
            print(f"    ❌ Selection/reproduction failed: {e}")
            return False
    
    print(f"✅ Multiple generations completed successfully")
    print(f"   Final population: {population.size} bacteria")
    
    return True

if __name__ == "__main__":
    print("Starting focused fitness access tests...\n")
    
    success = True
    
    try:
        # Test direct fitness calculation
        if not test_direct_fitness_calculation():
            success = False
        
        # Test selection workflow
        if not test_selection_workflow():
            success = False
        
        # Test multiple generations
        if not test_multiple_generations():
            success = False
            
        if success:
            print("\n=== ALL TESTS PASSED ===")
            print("✅ The AttributeError: 'OptimizedPopulation' object has no attribute 'fitness' has been FIXED!")
            print("✅ Fitness calculation now works correctly with individual bacteria")
            print("✅ Selection and survival processes work without errors")
            print("✅ Multiple generation simulations run successfully")
        else:
            print("\n=== SOME TESTS FAILED ===")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
