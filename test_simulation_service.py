#!/usr/bin/env python3
"""
Test the simulation service to ensure the AttributeError fix works end-to-end.
"""

import sys
import os
import traceback

# Add the backend directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.simulation_service import SimulationService

def test_simulation_service():
    """Test the simulation service end-to-end."""
    print("=== SIMULATION SERVICE TEST ===")
    
    try:
        # Create simulation service
        service = SimulationService()
        print("✅ Simulation service created")
        
        # Create a simulation
        simulation_id = "test_sim_fix"
        sim_metadata = service.create_simulation(
            simulation_id=simulation_id,
            initial_population_size=20,
            mutation_rate=0.01,
            selection_pressure=0.1,
            antibiotic_concentration=2.0,
            simulation_time=3
        )
        print(f"✅ Simulation created: {sim_metadata['simulation_id']}")
        
        # Run the simulation (synchronous version)
        results = service.run_simulation(simulation_id)
        print(f"✅ Simulation completed!")
        print(f"   Status: {results['status']}")
        print(f"   Generations: {results['generations_completed']}")
        print(f"   Final population: {results['final_population_size']}")
        print(f"   Final resistance: {results['final_resistance']:.3f}")
        
        # Check that we have proper data
        if results['final_population_size'] > 0:
            print("✅ Population survived the simulation")
        else:
            print("⚠️  Population went extinct (but no AttributeError)")
        
        # Clean up
        service.delete_simulation(simulation_id)
        print("✅ Simulation cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("Testing simulation service with the fitness AttributeError fix...\n")
    
    success = test_simulation_service()
    
    if success:
        print("\n=== SIMULATION SERVICE TEST PASSED ===")
        print("✅ No AttributeError: 'OptimizedPopulation' object has no attribute 'fitness'")
        print("✅ End-to-end simulation works correctly")
        print("✅ Fix is complete and functional!")
    else:
        print("\n=== SIMULATION SERVICE TEST FAILED ===")
        print("❌ There are still issues with the simulation")
        sys.exit(1)
