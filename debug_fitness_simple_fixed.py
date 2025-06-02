import traceback
import sys
import os

# Add the backend directory to sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

def trace_fitness_error():
    """Simple debug to trace the fitness error"""
    print("=== SIMPLIFIED FITNESS ERROR TRACE ===")
    
    try:
        from models.population import OptimizedPopulation, PopulationConfig
        from models.bacterium import Bacterium
        from models.resistance import EnvironmentalContext
        
        print("1. Creating minimal population...")
        config = PopulationConfig(population_size=10, use_spatial=False)
        population = OptimizedPopulation(config)
        
        print("2. Creating test bacteria...")
        for i in range(5):
            bacterium = Bacterium(id=f"test_{i}", fitness=1.0)
            population.add_bacterium(bacterium)
        
        print(f"3. Population size: {population.size}")
        print(f"   bacteria_by_id keys: {list(population.bacteria_by_id.keys())}")
        
        print("4. Testing direct fitness access...")
        for key, obj in population.bacteria_by_id.items():
            print(f"   Key: {key}, Type: {type(obj)}, fitness: {getattr(obj, 'fitness', 'NO FITNESS ATTR')}")
        
        print("5. Testing list comprehension (THIS IS WHERE THE ERROR OCCURS)...")
        try:
            fitness_values = [bacterium.fitness for bacterium in population.bacteria_by_id.values()]
            print(f"   SUCCESS: Got {len(fitness_values)} fitness values")
        except AttributeError as e:
            print(f"   ERROR: {e}")
            
            print("6. Finding the problematic object...")
            for i, obj in enumerate(population.bacteria_by_id.values()):
                print(f"   Object {i}: type={type(obj)}, has_fitness={hasattr(obj, 'fitness')}")
                if not hasattr(obj, 'fitness'):
                    print(f"   BAD OBJECT: {obj}")
                    print(f"   Attributes: {[attr for attr in dir(obj) if not attr.startswith('_')]}")
                    break
        
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    trace_fitness_error()
