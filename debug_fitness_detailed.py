import traceback
import sys
import os

# Add the backend directory to sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from models.population import OptimizedPopulation
from models.bacterium import Bacterium
from models.resistance import EnvironmentalContext
from services.simulation_service import BacterialSimulationService

def debug_fitness_access():
    """Debug exactly where the fitness access error occurs"""
    print("=== DETAILED FITNESS ACCESS DEBUG ===")
    
    try:
        # Create a minimal simulation
        print("1. Creating simulation service...")
        sim_service = BacterialSimulationService()
        
        print("2. Creating environment context...")
        env_context = EnvironmentalContext(
            antibiotic_concentration=1.0,
            temperature=37.0,
            ph=7.0,
            oxygen_level=0.8,
            nutrient_concentration=1.0,
            generation=1
        )
        
        print("3. Creating simulation...")
        simulation = sim_service.create_simulation(
            initial_population_size=100,
            antibiotic_concentration=1.0,
            environmental_context=env_context
        )
        
        print(f"4. Simulation created: {simulation.id} (status: {simulation.status})")
        
        # Access the population directly
        population = simulation.population
        print(f"5. Population type: {type(population)}")
        print(f"   Population size: {population.size}")
        print(f"   Bacteria by ID size: {len(population.bacteria_by_id)}")
        
        # Check what's in bacteria_by_id
        print("6. Examining bacteria_by_id contents...")
        for i, (key, obj) in enumerate(list(population.bacteria_by_id.items())[:5]):
            print(f"   [{i}] Key: {key}, Type: {type(obj)}")
            if hasattr(obj, 'fitness'):
                print(f"       fitness: {obj.fitness}")
            else:
                print(f"       ERROR: No fitness attribute!")
                print(f"       Object attributes: {dir(obj)}")
        
        # Try list comprehension that causes the error
        print("7. Testing list comprehension...")
        try:
            fitness_values = [bacterium.fitness for bacterium in population.bacteria_by_id.values()]
            print(f"   SUCCESS: Got {len(fitness_values)} fitness values")
        except AttributeError as e:
            print(f"   ERROR in list comprehension: {e}")
            print("   Examining objects that don't have fitness...")
            for obj in population.bacteria_by_id.values():
                if not hasattr(obj, 'fitness'):
                    print(f"   BAD OBJECT: {type(obj)} - {obj}")
                    print(f"   Attributes: {[attr for attr in dir(obj) if not attr.startswith('_')]}")
                    break
        
        # Now try to run one step
        print("8. Attempting one simulation step...")
        sim_service.run_simulation(simulation.id, max_generations=1)
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("TRACEBACK:")
        traceback.print_exc()

if __name__ == "__main__":
    debug_fitness_access()
