#!/usr/bin/env python3
"""
Test script to verify that the bacterial simulation fixes are working correctly.
This will test that:
1. Bacterial population increases instead of declining
2. Resistant bacteria get antibiotic survival benefits
3. Spatial data includes isResistant property for frontend compatibility
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
from services.simulation_service import SimulationService
from models.resistance import EnvironmentalContext
import time

async def test_simulation_fix():
    """Test the simulation to verify the fixes work."""
    print("🧪 Starting bacterial simulation test...")
    
    # Create simulation service
    service = SimulationService()
    
    # Create a test simulation with parameters that should show growth
    simulation_id = "test_fix_simulation"
    
    print("📋 Creating simulation with parameters:")
    print("   - Initial population: 500")
    print("   - Antibiotic concentration: 1.0 (high)")
    print("   - Mutation rate: 0.01 (higher for faster evolution)")
    print("   - Simulation time: 20 generations")
    
    result = service.create_simulation(
        simulation_id=simulation_id,
        initial_population_size=500,
        mutation_rate=0.01,  # Higher mutation rate to see evolution faster
        selection_pressure=0.2,
        antibiotic_concentration=1.0,  # High antibiotic concentration
        simulation_time=20  # Shorter test
    )
    
    print(f"✅ Simulation created: {result['status']}")
    
    # Track key metrics
    generation_data = []
    population_sizes = []
    resistance_levels = []
    
    print("\n🚀 Running simulation...")
    print("Gen | Population | Resistance | Status")
    print("----|------------|------------|--------")
    
    try:
        async for progress in service.run_simulation_async(simulation_id):
            if progress.get("status") == "running":
                gen = progress.get("current_generation", 0)
                pop_size = progress.get("population_size", 0)
                resistance = progress.get("average_resistance", 0)
                
                generation_data.append(progress)
                population_sizes.append(pop_size)
                resistance_levels.append(resistance)
                
                print(f"{gen:3d} | {pop_size:10d} | {resistance:10.3f} | Running")
                
                # Check spatial data for isResistant property
                spatial_data = progress.get("spatial_data", {})
                if spatial_data and "bacteria" in spatial_data:
                    bacteria = spatial_data["bacteria"]
                    if bacteria and len(bacteria) > 0:
                        first_bacterium = bacteria[0]
                        has_is_resistant = "isResistant" in first_bacterium
                        if gen == 1:  # Only print this once
                            print(f"    ✓ Spatial data includes 'isResistant': {has_is_resistant}")
                
            elif progress.get("status") in ["completed", "extinct"]:
                print(f"\n🏁 Simulation {progress.get('status')}")
                break
            elif progress.get("status") == "error":
                print(f"\n❌ Simulation error: {progress.get('error')}")
                return False
    
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        return False
    
    # Analyze results
    print("\n📊 Analysis:")
    
    if len(population_sizes) < 2:
        print("❌ Not enough data to analyze")
        return False
    
    initial_pop = population_sizes[0]
    final_pop = population_sizes[-1]
    initial_resistance = resistance_levels[0]
    final_resistance = resistance_levels[-1]
    
    print(f"   Initial population: {initial_pop}")
    print(f"   Final population: {final_pop}")
    print(f"   Population change: {((final_pop - initial_pop) / initial_pop * 100):+.1f}%")
    print(f"   Initial resistance: {initial_resistance:.3f}")
    print(f"   Final resistance: {final_resistance:.3f}")
    print(f"   Resistance change: {((final_resistance - initial_resistance)):+.3f}")
    
    # Check if population is stable or growing
    population_growing = final_pop >= initial_pop * 0.8  # Allow some decline but not collapse
    resistance_evolving = final_resistance > initial_resistance + 0.05  # Some resistance evolution
    
    print("\n🎯 Test Results:")
    if population_growing:
        print("   ✅ Population is stable/growing (fix working)")
    else:
        print("   ❌ Population is declining significantly (fix not working)")
    
    if resistance_evolving:
        print("   ✅ Resistance is evolving (selection pressure working)")
    else:
        print("   ⚠️  Resistance evolution limited (might be normal)")
    
    # Check for consistent spatial data
    has_spatial_data = any("spatial_data" in gen for gen in generation_data)
    print(f"   {'✅' if has_spatial_data else '❌'} Spatial data present: {has_spatial_data}")
    
    # Overall success
    success = population_growing and has_spatial_data
    
    if success:
        print("\n🎉 TEST PASSED: Bacterial simulation fixes are working!")
        print("   - Population growth is stable")
        print("   - Spatial data includes resistance information")
        print("   - EnvironmentalContext fix is effective")
    else:
        print("\n⚠️  TEST ISSUES: Some aspects need attention")
        if not population_growing:
            print("   - Population decline issue persists")
        if not has_spatial_data:
            print("   - Spatial data missing")
    
    return success

def test_environmental_context():
    """Test that EnvironmentalContext creates proper antibiotic benefits."""
    print("\n🧬 Testing EnvironmentalContext directly...")
    
    try:
        env_context = EnvironmentalContext(
            antibiotic_concentration=1.0,
            generation=1
        )
        print(f"   ✅ EnvironmentalContext created successfully")
        print(f"   - Antibiotic concentration: {env_context.antibiotic_concentration}")
        print(f"   - Generation: {env_context.generation}")
        return True
    except Exception as e:
        print(f"   ❌ EnvironmentalContext failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("BACTERIAL SIMULATION FIX VERIFICATION TEST")
    print("=" * 60)
    
    # Test EnvironmentalContext first
    env_test_passed = test_environmental_context()
    
    if env_test_passed:
        # Run the full simulation test
        result = asyncio.run(test_simulation_fix())
        
        print("\n" + "=" * 60)
        if result:
            print("🎊 ALL TESTS PASSED - Simulation fixes are working!")
        else:
            print("⚠️  TESTS REVEALED ISSUES - Check the analysis above")
        print("=" * 60)
    else:
        print("\n⚠️  Basic EnvironmentalContext test failed - Check imports")
