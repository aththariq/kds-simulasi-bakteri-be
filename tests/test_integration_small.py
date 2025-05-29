"""
Small integration test to diagnose performance issues.
"""

import pytest
import time
from models.bacterium import Bacterium, ResistanceStatus
from models.mutation import MutationEngine, MutationConfig
from models.population import Population, PopulationConfig


def test_small_simulation():
    """Test a very small simulation to check performance."""
    print("Starting small simulation test...")
    
    # Create tiny population
    config = PopulationConfig(
        population_size=10,  # Very small
        use_spatial=False
    )
    population = Population(config=config)
    
    print("Initializing population...")
    start = time.time()
    population.initialize_population()
    print(f"Population initialized in {time.time() - start:.2f}s")
    
    # Run just one generation
    print("Running one generation...")
    start = time.time()
    population.advance_generation()
    print(f"Generation completed in {time.time() - start:.2f}s")
    
    # Check results
    stats = population.get_statistics()
    assert len(population.bacteria) > 0
    print(f"Final population size: {len(population.bacteria)}")
    print("Test completed successfully!")


def test_mutation_engine_performance():
    """Test mutation engine performance."""
    print("Testing mutation engine performance...")
    
    engine = MutationEngine(config=MutationConfig())
    bacterium = Bacterium(id="test_1")
    
    # Generate mutations 100 times
    start = time.time()
    total_mutations = 0
    for i in range(100):
        mutations = engine.generate_mutations(bacterium.id, i)
        total_mutations += len(mutations)
    
    elapsed = time.time() - start
    print(f"Generated {total_mutations} mutations in {elapsed:.2f}s")
    print(f"Average time per generation: {elapsed/100*1000:.2f}ms")
    
    assert elapsed < 1.0  # Should be very fast


if __name__ == "__main__":
    test_small_simulation()
    test_mutation_engine_performance() 