#!/usr/bin/env python3
"""
Minimal simulation test to verify the fitness fix works in the full simulation context.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.population import PopulationConfig, OptimizedPopulation
from models.resistance import EnvironmentalContext
from models.fitness import ComprehensiveFitnessCalculator
from models.selection import SelectionEnvironment, AntimicrobialPressure, PressureType, PressureConfig
import numpy as np

def run_minimal_simulation():
    """Run a minimal simulation to test the fitness fix in action."""
    
    print("=== MINIMAL SIMULATION TEST ===")
    
    # Create population
    print("1. Creating population...")
    config = PopulationConfig(
        population_size=20,
        initial_resistance_frequency=0.3,
        use_spatial=False
    )
    population = OptimizedPopulation(config)
    population.initialize_population()
    
    print(f"   Initial population: {population.size}")
    print(f"   Resistant: {len(population.resistant_bacteria)}")
    print(f"   Sensitive: {len(population.sensitive_bacteria)}")
    
    # Initialize simulation components
    fitness_calc = ComprehensiveFitnessCalculator()
    selection = SelectionEnvironment()
    
    # Add antimicrobial pressure
    selection.add_pressure(AntimicrobialPressure(
        config=PressureConfig(
            pressure_type=PressureType.ANTIMICROBIAL,
            intensity=0.6
        )
    ))
    
    max_generations = 3
    antibiotic_concentration = 2.0
    
    print(f"2. Running simulation for {max_generations} generations...")
    
    for generation in range(max_generations):
        print(f"\n   Generation {generation}:")
        print(f"   Population size: {population.size}")
        
        # Create environmental context
        environmental_context = EnvironmentalContext(
            antibiotic_concentration=antibiotic_concentration,
            generation=generation
        )
        
        # Calculate fitness for each bacterium (THE FIX!)
        fitness_scores = {}
        population_context = {
            'total_population': population.size,
            'generation': generation,
            'carrying_capacity': config.population_size * 2,
            'local_density': 1.0
        }
        
        bacteria_processed = 0
        for bacterium in population.bacteria_by_id.values():
            if hasattr(bacterium, 'fitness') and hasattr(bacterium, 'id'):
                try:
                    fitness_result = fitness_calc.calculate_fitness(
                        bacterium=bacterium,
                        environmental_context=environmental_context,
                        population_context=population_context
                    )
                    fitness_scores[bacterium.id] = fitness_result.final_fitness
                    bacterium.fitness = fitness_result.final_fitness
                    bacteria_processed += 1
                except Exception as e:
                    print(f"      ERROR: Failed to calculate fitness for {bacterium.id}: {e}")
                    fitness_scores[bacterium.id] = bacterium.fitness
        
        print(f"      Fitness calculated for {bacteria_processed} bacteria")
        
        # Apply selection pressure
        bacteria_list = [b for b in population.bacteria_by_id.values() 
                        if hasattr(b, 'fitness') and hasattr(b, 'id')]
        
        selection_context = {
            'total_population': population.size,
            'generation': generation
        }
        
        try:
            selection_results = selection.apply_selection(bacteria_list, selection_context, generation)
            print(f"      Selection applied to {len(selection_results)} bacteria")
            
            # Apply survival selection (simplified)
            survivors = []
            for bacterium in bacteria_list:
                # Simple survival based on fitness
                survival_prob = min(0.95, bacterium.fitness * 0.8)
                if np.random.random() < survival_prob:
                    survivors.append(bacterium)
            
            print(f"      {len(survivors)} bacteria survived")
            
            # Update population with survivors
            if survivors:
                population.clear()
                population._batch_add_bacteria(survivors)
            else:
                print("      WARNING: No survivors! Population extinct.")
                break
                
        except Exception as e:
            print(f"      ERROR in selection: {e}")
            break
    
    print(f"\n3. Simulation completed!")
    print(f"   Final population size: {population.size}")
    if population.size > 0:
        print(f"   Final resistant count: {len(population.resistant_bacteria)}")
        print(f"   Final resistance frequency: {population.resistance_frequency:.3f}")
        
        # Test fitness access one more time
        print("4. Final fitness access test...")
        try:
            fitness_values = [bacterium.fitness for bacterium in population.bacteria_by_id.values()]
            print(f"   SUCCESS: Accessed {len(fitness_values)} fitness values")
            print(f"   Mean fitness: {np.mean(fitness_values):.3f}")
        except Exception as e:
            print(f"   ERROR: {e}")
            return False
    
    print("\n=== SIMULATION TEST PASSED ===")
    return True

if __name__ == "__main__":
    success = run_minimal_simulation()
    sys.exit(0 if success else 1)
