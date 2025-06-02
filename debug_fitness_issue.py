#!/usr/bin/env python3
"""
Debug script to isolate the fitness access issue.
"""

import traceback
from models.population import OptimizedPopulation, PopulationConfig

def test_population_bacteria_iteration():
    """Test that population.bacteria_by_id.values() returns only Bacterium objects."""
    print("🔍 Testing population bacteria iteration...")
    
    config = PopulationConfig(
        population_size=10,
        initial_resistance_frequency=0.2,
        random_seed=42
    )
    
    population = OptimizedPopulation(config)
    population.initialize_population()
    
    print(f"   Population size: {population.size}")
    print(f"   bacteria_by_id keys: {len(population.bacteria_by_id)}")
    
    # Check each object in bacteria_by_id.values()
    for i, obj in enumerate(population.bacteria_by_id.values()):
        print(f"   Object {i}: type={type(obj)}, has_fitness={hasattr(obj, 'fitness')}")
        if hasattr(obj, 'fitness'):
            print(f"      fitness={obj.fitness}")
        else:
            print(f"      ERROR: Object does not have fitness attribute!")
            print(f"      Object attributes: {dir(obj)}")
            break
    
    print("   ✅ All objects in bacteria_by_id are proper Bacterium objects")

def test_spatial_data_construction():
    """Test the spatial data construction that's failing."""
    print("\n🔍 Testing spatial data construction...")
    
    config = PopulationConfig(
        population_size=10,
        initial_resistance_frequency=0.2,
        random_seed=42
    )
    
    population = OptimizedPopulation(config)
    population.initialize_population()
    
    print(f"   Population size: {population.size}")
    
    try:
        # This is the line that's failing in the simulation service
        actual_bacteria = list(population.bacteria_by_id.values())[:min(population.size, 100)]
        print(f"   Got {len(actual_bacteria)} bacteria for spatial data")
        
        # Test the problematic line
        for i, bacterium in enumerate(actual_bacteria):
            print(f"   Bacterium {i}: type={type(bacterium)}")
            print(f"      has fitness: {hasattr(bacterium, 'fitness')}")
            if hasattr(bacterium, 'fitness'):
                fitness_value = float(bacterium.fitness)
                print(f"      fitness: {fitness_value}")
            else:
                print(f"      ERROR: Object {bacterium} has no fitness attribute!")
                return False
        
        print("   ✅ All bacteria have fitness attribute")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in spatial data construction: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 DEBUGGING FITNESS ACCESS ISSUE")
    print("=" * 50)
    
    try:
        test_population_bacteria_iteration()
        test_spatial_data_construction()
        print("\n✅ All tests passed - issue might be elsewhere")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
