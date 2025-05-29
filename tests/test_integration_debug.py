"""
Debug version of integration test to find performance bottlenecks.
"""

import time
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.bacterium import Bacterium, ResistanceStatus
from models.mutation import MutationEngine, MutationConfig
from models.selection import SelectionEnvironment, AntimicrobialPressure, PressureConfig, PressureType
from models.fitness import ComprehensiveFitnessCalculator, FitnessConfig
from models.resistance import ResistanceCostBenefitCalculator, EnvironmentalContext
from models.population import Population, PopulationConfig


def profile_integration_components():
    """Profile each component of the integration test."""
    
    print("=== Profiling Integration Components ===\n")
    
    # Test 1: Population initialization
    print("1. Testing Population Initialization...")
    start = time.time()
    population = Population(config=PopulationConfig(population_size=100, use_spatial=False))
    population.initialize_population()
    print(f"   - Time: {time.time() - start:.3f}s")
    print(f"   - Population size: {len(population.bacteria)}\n")
    
    # Test 2: Mutation engine
    print("2. Testing Mutation Engine...")
    mutation_engine = MutationEngine(config=MutationConfig())
    start = time.time()
    total_mutations = 0
    for bacterium in population.bacteria[:10]:  # Test on 10 bacteria
        mutations = mutation_engine.generate_mutations(bacterium.id, 0)
        total_mutations += len(mutations)
    elapsed = time.time() - start
    print(f"   - Time for 10 bacteria: {elapsed:.3f}s")
    print(f"   - Mutations generated: {total_mutations}")
    print(f"   - Time per bacterium: {elapsed/10*1000:.1f}ms\n")
    
    # Test 3: Fitness calculation
    print("3. Testing Fitness Calculation...")
    fitness_calculator = ComprehensiveFitnessCalculator()
    selection_env = SelectionEnvironment()
    env_context = EnvironmentalContext()
    
    start = time.time()
    for bacterium in population.bacteria[:10]:  # Test on 10 bacteria
        fitness_result = fitness_calculator.calculate_fitness(
            bacterium=bacterium,
            mutations=[],
            selection_environment=selection_env,
            environmental_context=env_context,
            population_context={'generation': 0}
        )
    elapsed = time.time() - start
    print(f"   - Time for 10 bacteria: {elapsed:.3f}s")
    print(f"   - Time per bacterium: {elapsed/10*1000:.1f}ms\n")
    
    # Test 4: Full generation with timing breakdown
    print("4. Testing Full Generation Processing...")
    print("   Processing 1 generation with 50 bacteria:")
    
    small_pop = Population(config=PopulationConfig(population_size=50, use_spatial=False))
    small_pop.initialize_population()
    
    # Time mutations
    start = time.time()
    for bacterium in small_pop.bacteria:
        mutations = mutation_engine.generate_mutations(bacterium.id, 0)
        if mutations:
            mutation_engine.apply_mutations(bacterium, mutations)
    mutation_time = time.time() - start
    print(f"   - Mutation time: {mutation_time:.3f}s")
    
    # Time fitness calculations
    start = time.time()
    for bacterium in small_pop.bacteria:
        fitness_result = fitness_calculator.calculate_fitness(
            bacterium=bacterium,
            mutations=[],
            selection_environment=selection_env,
            environmental_context=env_context,
            population_context={'generation': 0}
        )
        bacterium.fitness = fitness_result.final_fitness
    fitness_time = time.time() - start
    print(f"   - Fitness calculation time: {fitness_time:.3f}s")
    
    # Time generation advance
    start = time.time()
    small_pop.advance_generation()
    advance_time = time.time() - start
    print(f"   - Generation advance time: {advance_time:.3f}s")
    
    total_time = mutation_time + fitness_time + advance_time
    print(f"   - Total time: {total_time:.3f}s\n")
    
    # Estimate for full test
    print("5. Estimated Time for Full Integration Test:")
    generations = 50
    population_size = 100
    time_per_generation = (total_time / 50) * population_size  # Scale up
    total_estimated = time_per_generation * generations
    print(f"   - Per generation (100 bacteria): {time_per_generation:.1f}s")
    print(f"   - Total for 50 generations: {total_estimated:.1f}s ({total_estimated/60:.1f} minutes)")
    
    if total_estimated > 60:
        print("\n⚠️  WARNING: Integration test will likely take too long!")
        print("   Consider reducing population size or generations for tests.")


if __name__ == "__main__":
    profile_integration_components() 