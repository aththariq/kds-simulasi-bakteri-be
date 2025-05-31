"""
Test script for WebSocket communication optimizations.
Task 20.3: WebSocket Communication Efficiency Enhancement

This script tests the various optimization features including:
- Binary encoding with MessagePack
- Compression with GZIP/ZLIB
- Delta compression for state updates
- Message batching
- Performance measurement and comparison
"""

import asyncio
import json
import time
import random
import string
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_optimizations():
    """Main test function for WebSocket optimizations."""
    
    # Import services
    from services.websocket_optimizations import (
        get_optimized_websocket_service,
        MessageOptimizationLevel,
        BinaryMessageEncoder,
        DeltaCompressor,
        MessageBatcher
    )
    
    print("=" * 60)
    print("WebSocket Communication Optimization Tests")
    print("=" * 60)
    
    # Test 1: Binary Message Encoding
    print("\n1. Testing Binary Message Encoding...")
    encoder = BinaryMessageEncoder()
    
    # Generate test message
    test_message = {
        'id': 'test_001',
        'type': 'simulation_update',
        'timestamp': time.time(),
        'simulation_id': 'test_simulation',
        'data': {
            'generation': 42,
            'population': 1000,
            'resistant_count': 150,
            'fitness_scores': [random.random() for _ in range(100)],
            'metadata': {
                'description': 'Test simulation data with various data types',
                'parameters': {'mutation_rate': 0.01, 'selection_pressure': 0.8}
            }
        }
    }
    
    # Test different optimization levels
    results = {}
    for level in MessageOptimizationLevel:
        encoded_data, metrics = encoder.encode_message(test_message, level)
        compression_ratio = metrics.optimized_size / metrics.original_json_size if metrics.original_json_size > 0 else 1.0
        results[level.value] = {
            'original_size': metrics.original_json_size,
            'optimized_size': metrics.optimized_size,
            'compression_ratio': compression_ratio,
            'bandwidth_saved': metrics.bandwidth_saved,
            'encoding_time': metrics.encoding_time,
            'compression_time': metrics.compression_time
        }
        
        print(f"  {level.value:12}: {metrics.original_json_size:6} -> {metrics.optimized_size:6} bytes "
              f"({metrics.bandwidth_saved:4} saved, {(1-compression_ratio)*100:5.1f}% reduction)")
    
    # Find best optimization
    best_level = min(results.keys(), key=lambda x: results[x]['optimized_size'])
    print(f"  Best optimization: {best_level} "
          f"({results[best_level]['bandwidth_saved']} bytes saved)")
    
    # Test 2: Delta Compression
    print("\n2. Testing Delta Compression...")
    delta_compressor = DeltaCompressor()
    
    # Simulate state updates
    client_id = "test_client_001"
    simulation_id = "test_simulation"
    
    # Initial state
    initial_state = {
        'generation': 1,
        'population': 1000,
        'resistant_bacteria': 100,
        'sensitive_bacteria': 900,
        'mutation_events': 5,
        'environment': {
            'antibiotic_concentration': 0.5,
            'temperature': 37.0,
            'ph': 7.2
        }
    }
    
    # Create first update (should be full state)
    delta_data, is_full = delta_compressor.create_delta_update(
        client_id, simulation_id, initial_state
    )
    
    print(f"  First update: {'Full state' if is_full else 'Delta'} "
          f"({len(json.dumps(delta_data))} bytes)")
    
    # Create incremental updates
    delta_sizes = []
    full_sizes = []
    
    for generation in range(2, 12):
        # Simulate small changes
        updated_state = initial_state.copy()
        updated_state['generation'] = generation
        updated_state['population'] += random.randint(-10, 10)
        updated_state['resistant_bacteria'] += random.randint(-5, 5)
        updated_state['sensitive_bacteria'] = updated_state['population'] - updated_state['resistant_bacteria']
        updated_state['mutation_events'] += random.randint(0, 3)
        
        # Small environment changes
        updated_state['environment']['antibiotic_concentration'] += random.uniform(-0.1, 0.1)
        
        delta_data, is_full = delta_compressor.create_delta_update(
            client_id, simulation_id, updated_state
        )
        
        delta_size = len(json.dumps(delta_data))
        full_size = len(json.dumps(updated_state))
        
        delta_sizes.append(delta_size)
        full_sizes.append(full_size)
        
        print(f"  Generation {generation:2}: {'Full' if is_full else 'Delta'} "
              f"({delta_size:3} vs {full_size:3} bytes, "
              f"{(1 - delta_size/full_size)*100:5.1f}% reduction)")
    
    avg_delta_savings = (1 - sum(delta_sizes) / sum(full_sizes)) * 100
    print(f"  Average delta compression: {avg_delta_savings:.1f}% bandwidth reduction")
    
    # Test 3: Message Batching
    print("\n3. Testing Message Batching...")
    batcher = MessageBatcher(max_batch_size=5, max_batch_age=0.2)
    
    # Simulate rapid message sending
    client_id = "test_client_002"
    batch_count = 0
    total_messages = 15
    total_individual_size = 0
    total_batch_size = 0
    
    for i in range(total_messages):
        message = {
            'id': f'msg_{i:03d}',
            'type': 'simulation_update',
            'timestamp': time.time(),
            'data': {
                'generation': i,
                'update_type': 'incremental',
                'changes': random.randint(1, 10)
            }
        }
        
        individual_size = len(json.dumps(message))
        total_individual_size += individual_size
        
        # Add to batch
        batch_data = batcher.add_message(client_id, message)
        
        if batch_data:
            batch_count += 1
            batch_size = len(batch_data)
            total_batch_size += batch_size
            print(f"  Batch {batch_count}: {batch_size} bytes (compressed batch)")
        
        # Small delay to test time-based batching
        if i % 3 == 0:
            await asyncio.sleep(0.05)
    
    # Flush remaining messages
    final_batch = batcher.flush_client_batches(client_id)
    if final_batch:
        batch_count += 1
        total_batch_size += len(final_batch)
        print(f"  Final batch: {len(final_batch)} bytes")
    
    batch_efficiency = (1 - total_batch_size / total_individual_size) * 100 if total_individual_size > 0 else 0
    print(f"  Batching efficiency: {batch_efficiency:.1f}% size reduction "
          f"({total_messages} messages -> {batch_count} batches)")
    
    # Test 4: Performance Benchmark
    print("\n4. Performance Benchmark...")
    service = get_optimized_websocket_service()
    
    # Generate test dataset
    test_messages = []
    for i in range(100):
        message = {
            'id': f'perf_test_{i:03d}',
            'type': 'simulation_update',
            'simulation_id': 'performance_test',
            'timestamp': time.time(),
            'data': {
                'generation': i,
                'population_data': {
                    'total': random.randint(800, 1200),
                    'resistant': random.randint(100, 300),
                    'fitness_distribution': [random.random() for _ in range(50)]
                },
                'environmental_data': {
                    'antibiotic_zones': [
                        {
                            'id': f'zone_{j}',
                            'concentration': random.uniform(0.1, 1.0),
                            'radius': random.uniform(5.0, 15.0)
                        } for j in range(random.randint(1, 5))
                    ]
                },
                'metadata': {
                    'processing_time': random.uniform(0.001, 0.1),
                    'memory_usage': random.randint(1000000, 10000000)
                }
            }
        }
        test_messages.append(message)
    
    # Benchmark different optimization levels
    benchmark_results = {}
    
    for level in [MessageOptimizationLevel.NONE, MessageOptimizationLevel.COMPRESSED, MessageOptimizationLevel.DELTA]:
        start_time = time.time()
        total_original = 0
        total_optimized = 0
        
        for message in test_messages:
            encoded_data, metrics = service.encoder.encode_message(message, level)
            total_original += metrics.original_json_size
            total_optimized += metrics.optimized_size
        
        end_time = time.time()
        
        benchmark_results[level.value] = {
            'total_time': end_time - start_time,
            'messages_per_second': len(test_messages) / (end_time - start_time),
            'total_original_size': total_original,
            'total_optimized_size': total_optimized,
            'compression_ratio': total_optimized / total_original if total_original > 0 else 1.0,
            'bandwidth_savings': total_original - total_optimized
        }
    
    print("  Optimization Level    | Time (s) | Msgs/sec | Original | Optimized | Savings")
    print("  " + "-" * 75)
    
    for level, stats in benchmark_results.items():
        print(f"  {level:18} | {stats['total_time']:8.3f} | {stats['messages_per_second']:8.1f} | "
              f"{stats['total_original_size']:8} | {stats['total_optimized_size']:9} | "
              f"{stats['bandwidth_savings']:7}")
    
    # Test 5: Integration Test
    print("\n5. Integration Test...")
    
    # Test the optimization service end-to-end
    optimization_service = get_optimized_websocket_service()
    
    # Enable optimizations
    optimization_service.optimization_enabled = True
    
    # Set different optimization levels for different clients
    clients = ['client_001', 'client_002', 'client_003']
    levels = [MessageOptimizationLevel.BASIC, MessageOptimizationLevel.COMPRESSED, MessageOptimizationLevel.DELTA]
    
    for client, level in zip(clients, levels):
        optimization_service.set_client_optimization_level(client, level)
        client_level = optimization_service.get_client_optimization_level(client)
        print(f"  {client}: {client_level.value}")
    
    # Simulate message transmission
    total_savings = 0
    for client in clients:
        message = {
            'id': f'integration_test_{client}',
            'type': 'simulation_update',
            'client_id': client,
            'data': {'test': 'integration', 'value': random.randint(1, 100)}
        }
        
        level = optimization_service.get_client_optimization_level(client)
        encoded_data, metrics = optimization_service.encoder.encode_message(message, level)
        total_savings += metrics.bandwidth_saved
        
        print(f"    {client} ({level.value}): {metrics.bandwidth_saved} bytes saved")
    
    print(f"  Total bandwidth saved: {total_savings} bytes")
    
    # Get performance summary
    summary = optimization_service.get_performance_summary()
    print(f"  Total messages processed: {summary['total_messages_sent']}")
    print(f"  Total bandwidth saved: {summary['total_bandwidth_saved']} bytes")
    
    print("\n" + "=" * 60)
    print("WebSocket Optimization Tests Completed Successfully!")
    print("=" * 60)
    
    return {
        'encoding_results': results,
        'delta_compression_savings': avg_delta_savings,
        'batching_efficiency': batch_efficiency,
        'benchmark_results': benchmark_results,
        'integration_savings': total_savings
    }


if __name__ == "__main__":
    # Run the test
    try:
        results = asyncio.run(test_websocket_optimizations())
        print(f"\nTest Summary:")
        print(f"- Best encoding savings: {max(r['bandwidth_saved'] for r in results['encoding_results'].values())} bytes")
        print(f"- Delta compression: {results['delta_compression_savings']:.1f}% average savings")
        print(f"- Batching efficiency: {results['batching_efficiency']:.1f}% size reduction")
        print(f"- Integration test savings: {results['integration_savings']} bytes")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise 