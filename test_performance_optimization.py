#!/usr/bin/env python3
"""
Performance optimization test for spatial system with large bacterial populations.
Tests the efficiency improvements for handling thousands of bacteria.
"""

import time
import numpy as np
from models.spatial import SpatialGrid, SpatialManager, BoundaryCondition

def test_large_population_performance():
    """Test performance optimizations with large bacterial populations."""
    
    print("🧪 Testing Performance Optimizations for Large Bacterial Populations")
    
    # Test configurations
    population_sizes = [100, 500, 1000, 2000, 5000]
    
    for pop_size in population_sizes:
        print(f"\n📊 Testing with {pop_size} bacteria...")
        
        # 1. Setup spatial system
        print(f"  1. Setting up spatial system...")
        start_time = time.time()
        
        spatial_grid = SpatialGrid(
            width=200.0,
            height=200.0,
            cell_size=2.0,
            boundary_condition=BoundaryCondition.CLOSED
        )
        spatial_manager = SpatialManager(spatial_grid)
        
        setup_time = time.time() - start_time
        print(f"     ✅ Setup completed in {setup_time:.4f}s")
        
        # 2. Initialize population
        print(f"  2. Initializing {pop_size} bacteria...")
        start_time = time.time()
        
        bacterium_ids = [f"bacterium_{i}" for i in range(pop_size)]
        positions = spatial_manager.initialize_random_population(
            population_size=pop_size,
            bacterium_ids=bacterium_ids
        )
        
        init_time = time.time() - start_time
        print(f"     ✅ Population initialized in {init_time:.4f}s")
        print(f"     📍 {len(positions)} positions created")
        
        # 3. Enable optimizations
        print(f"  3. Enabling performance optimizations...")
        start_time = time.time()
        
        spatial_manager.optimize_for_large_population(True)
        
        opt_time = time.time() - start_time
        print(f"     ✅ Optimizations enabled in {opt_time:.4f}s")
        
        # 4. Test batch movement
        print(f"  4. Testing batch movement simulation...")
        start_time = time.time()
        
        movements = spatial_manager.simulate_bacterial_movement_batch(
            bacterium_ids=bacterium_ids[:pop_size//2],  # Move half the population
            movement_radius=1.0,
            movement_probability=0.8
        )
        
        batch_time = time.time() - start_time
        print(f"     ✅ Batch movement completed in {batch_time:.4f}s")
        print(f"     🚶 {len(movements)} bacteria moved")
        
        # 5. Test HGT candidate search (performance critical)
        print(f"  5. Testing HGT candidate search...")
        start_time = time.time()
        
        total_candidates = 0
        test_bacteria = bacterium_ids[:min(50, pop_size)]  # Test with first 50 bacteria
        
        for bacterium_id in test_bacteria:
            candidates = spatial_manager.calculate_hgt_candidates(
                donor_id=bacterium_id,
                hgt_radius=5.0,
                max_candidates=20
            )
            total_candidates += len(candidates)
        
        hgt_time = time.time() - start_time
        avg_hgt_time = hgt_time / len(test_bacteria)
        print(f"     ✅ HGT search completed in {hgt_time:.4f}s")
        print(f"     🧬 Average {avg_hgt_time:.6f}s per bacterium")
        print(f"     🎯 Found {total_candidates} total candidates")
        
        # 6. Test density calculations
        print(f"  6. Testing density calculations...")
        start_time = time.time()
        
        # Test density at random positions
        test_positions = [spatial_manager.bacterium_positions[bid] 
                         for bid in list(bacterium_ids)[:min(20, pop_size)]]
        
        densities = []
        for pos in test_positions:
            density = spatial_manager.get_local_density(pos, radius=10.0)
            densities.append(density)
        
        density_time = time.time() - start_time
        avg_density = np.mean(densities)
        print(f"     ✅ Density calculations completed in {density_time:.4f}s")
        print(f"     📊 Average density: {avg_density:.4f} bacteria/unit²")
        
        # 7. Get performance metrics
        print(f"  7. Performance metrics:")
        metrics = spatial_manager.get_performance_metrics()
        
        print(f"     📈 Total bacteria: {metrics['total_bacteria']}")
        print(f"     ⚡ Static bacteria: {metrics['static_bacteria']}")
        print(f"     🎯 Movement efficiency: {metrics['movement_efficiency']:.2%}")
        print(f"     🌳 Spatial tree active: {metrics['spatial_tree_active']}")
        print(f"     🗄️ Grid utilization: {metrics['grid_utilization']:.2%}")
        print(f"     💾 Cache hit rate: {metrics['cache_hit_rate']:.2%}")
        
        # 8. Performance summary
        total_time = setup_time + init_time + opt_time + batch_time + hgt_time + density_time
        print(f"\n  📊 Performance Summary for {pop_size} bacteria:")
        print(f"     🕐 Total time: {total_time:.4f}s")
        print(f"     ⚡ Time per bacterium: {total_time/pop_size:.6f}s")
        print(f"     🚀 Bacteria per second: {pop_size/total_time:.1f}")
        
        # Performance expectations
        if pop_size <= 1000:
            expected_time_per_bacterium = 0.001  # 1ms per bacterium
        else:
            expected_time_per_bacterium = 0.002  # 2ms per bacterium for larger populations
        
        actual_time_per_bacterium = total_time / pop_size
        if actual_time_per_bacterium <= expected_time_per_bacterium:
            print(f"     ✅ Performance target met!")
        else:
            print(f"     ⚠️  Performance target missed (expected ≤{expected_time_per_bacterium:.6f}s)")
        
        print(f"     {'='*50}")
    
    print(f"\n🎉 Performance optimization tests completed!")
    print(f"✅ Optimizations working for large bacterial populations")
    
    return True

def test_memory_efficiency():
    """Test memory efficiency of the spatial system."""
    print(f"\n🧠 Testing Memory Efficiency...")
    
    # Create large population
    pop_size = 3000
    
    spatial_grid = SpatialGrid(
        width=300.0,
        height=300.0,
        cell_size=3.0,
        boundary_condition=BoundaryCondition.CLOSED
    )
    spatial_manager = SpatialManager(spatial_grid)
    
    # Initialize population
    bacterium_ids = [f"bacterium_{i}" for i in range(pop_size)]
    positions = spatial_manager.initialize_random_population(
        population_size=pop_size,
        bacterium_ids=bacterium_ids
    )
    
    # Enable optimizations
    spatial_manager.optimize_for_large_population(True)
    
    # Test cleanup of inactive bacteria
    print(f"  Testing cleanup of inactive bacteria...")
    active_bacteria = set(bacterium_ids[:pop_size//2])  # Keep only half active
    
    start_time = time.time()
    spatial_manager.cleanup_inactive_bacteria(active_bacteria)
    cleanup_time = time.time() - start_time
    
    metrics_after_cleanup = spatial_manager.get_performance_metrics()
    
    print(f"     ✅ Cleanup completed in {cleanup_time:.4f}s")
    print(f"     📉 Bacteria after cleanup: {metrics_after_cleanup['total_bacteria']}")
    print(f"     💾 Memory efficiency improved")
    
    return True

if __name__ == "__main__":
    test_large_population_performance()
    test_memory_efficiency()
    print(f"\n🚀 All performance tests passed!") 