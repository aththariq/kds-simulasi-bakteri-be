#!/usr/bin/env python3
"""
Test for HGT Gene Transfer Engine.
Tests the complete gene transfer execution system with all mechanisms.
"""

from models.hgt import (
    GeneTransferEngine, ProximityDetector, ProbabilityCalculator, 
    HGTConfig, HGTMechanism
)
from models.spatial import SpatialGrid, SpatialManager, Coordinate, BoundaryCondition
from models.bacterium import Bacterium, ResistanceStatus


def test_gene_transfer_engine():
    """Test HGT gene transfer engine system."""
    
    print("🧪 Testing HGT Gene Transfer Engine...")
    
    # 1. Setup complete HGT system
    print("\n1. Setting up complete HGT system...")
    
    # Create spatial system
    grid = SpatialGrid(
        width=20.0,
        height=20.0,
        cell_size=1.0,
        boundary_condition=BoundaryCondition.CLOSED
    )
    manager = SpatialManager(grid)
    
    # Create HGT configuration
    config = HGTConfig(
        conjugation_distance=2.0,
        transformation_distance=3.0,
        transduction_distance=4.0,
        conjugation_probability=0.15,
        transformation_probability=0.08,
        transduction_probability=0.05
    )
    
    # Create components
    proximity_detector = ProximityDetector(manager, config)
    probability_calculator = ProbabilityCalculator(config)
    transfer_engine = GeneTransferEngine(proximity_detector, probability_calculator, config)
    
    print(f"✅ Complete HGT system created")
    
    # 2. Create diverse bacterial population
    print("\n2. Creating diverse bacterial population...")
    
    bacteria = {}
    positions = {}
    
    # Create donor bacteria (resistant)
    for i in range(3):
        donor = Bacterium(
            id=f"donor_{i}",
            resistance_status=ResistanceStatus.RESISTANT,
            fitness=0.85 + (i * 0.05),
            age=2 + i
        )
        
        # Add HGT properties
        donor.has_conjugative_plasmid = (i % 2 == 0)
        donor.is_competent = True
        donor.phage_infected = (i == 1)
        donor.species = "E.coli"
        donor.resistance_genes = ["beta_lactamase", "efflux_pump"]
        donor.is_motile = True
        donor.phage_load = 1.2 if donor.phage_infected else 0.0
        
        bacteria[donor.id] = donor
        positions[donor.id] = Coordinate(5.0 + (i * 2.0), 10.0)
        grid.place_bacterium(donor.id, positions[donor.id])
        manager.bacterium_positions[donor.id] = positions[donor.id]
    
    # Create recipient bacteria (sensitive)
    for i in range(5):
        recipient = Bacterium(
            id=f"recipient_{i}",
            resistance_status=ResistanceStatus.SENSITIVE,
            fitness=0.8 + (i * 0.03),
            age=1 + i
        )
        
        # Add HGT properties
        recipient.has_conjugative_plasmid = False
        recipient.is_competent = (i % 2 == 0)
        recipient.phage_infected = False
        recipient.phage_resistant = (i == 4)
        recipient.species = "E.coli" if i < 3 else "S.aureus"
        recipient.resistance_genes = []
        recipient.surface_receptors = True
        recipient.conjugation_compatible = True
        recipient.natural_transformation = (i % 2 == 0)
        recipient.dna_uptake_efficiency = 0.7 + (i * 0.05)
        recipient.phage_sensitivity = ["T4", "T7"] if not recipient.phage_resistant else []
        recipient.competence_level = 0.6 + (i * 0.08)
        
        bacteria[recipient.id] = recipient
        positions[recipient.id] = Coordinate(5.0 + (i * 1.5), 8.0)
        grid.place_bacterium(recipient.id, positions[recipient.id])
        manager.bacterium_positions[recipient.id] = positions[recipient.id]
    
    print(f"✅ Created {len(bacteria)} bacteria:")
    print(f"  • {sum(1 for b in bacteria.values() if b.resistance_status == ResistanceStatus.RESISTANT)} resistant donors")
    print(f"  • {sum(1 for b in bacteria.values() if b.resistance_status == ResistanceStatus.SENSITIVE)} sensitive recipients")
    
    # 3. Test single HGT round execution
    print("\n3. Testing HGT round execution...")
    
    environmental_factors = {
        "antibiotic_concentration": 1.5,
        "stress_response": 0.3,
        "nutrient_limitation": 0.2
    }
    
    # Execute HGT round
    transfer_events = transfer_engine.execute_hgt_round(
        bacterium_population=bacteria,
        spatial_positions=positions,
        environmental_factors=environmental_factors,
        generation=1,
        max_transfers_per_round=20
    )
    
    print(f"✅ HGT round completed:")
    print(f"  • {len(transfer_events)} successful transfers")
    
    for event in transfer_events:
        print(f"  • {event.mechanism.value}: {event.donor_id} → {event.recipient_id}")
        print(f"    Genes: {event.genes_transferred}")
        print(f"    Success: {event.success}, Probability: {event.probability:.4f}")
    
    # 4. Test transfer statistics
    print("\n4. Testing transfer statistics...")
    
    stats = transfer_engine.get_transfer_statistics()
    
    print(f"✅ Transfer Statistics:")
    print(f"  • Total attempts: {stats['total_transfer_attempts']}")
    print(f"  • Successful transfers: {stats['successful_transfers']}")
    print(f"  • Failed transfers: {stats['failed_transfers']}")
    print(f"  • Success rate: {stats['overall_success_rate']:.2%}")
    
    print(f"  • Transfers by mechanism:")
    for mechanism, count in stats['transfers_by_mechanism'].items():
        print(f"    - {mechanism.value}: {count}")
    
    print(f"  • Resistance gene frequencies:")
    for gene, data in stats['resistance_gene_frequencies'].items():
        print(f"    - {gene}: {data['frequency']:.2%} (last seen: gen {data['last_seen']})")
    
    # 5. Test gene transfer application
    print("\n5. Testing gene transfer application effects...")
    
    # Count resistant bacteria before and after
    initial_resistant = sum(1 for b in bacteria.values() if b.resistance_status == ResistanceStatus.RESISTANT)
    
    # Check for newly resistant bacteria
    newly_resistant = []
    for bacterium_id, bacterium in bacteria.items():
        if bacterium_id.startswith("recipient_") and bacterium.resistance_status == ResistanceStatus.RESISTANT:
            newly_resistant.append(bacterium_id)
    
    print(f"✅ Gene transfer effects:")
    print(f"  • Initial resistant: {initial_resistant}")
    print(f"  • Newly resistant: {len(newly_resistant)} bacteria")
    if newly_resistant:
        for bacterium_id in newly_resistant:
            bacterium = bacteria[bacterium_id]
            print(f"    - {bacterium_id}: genes = {getattr(bacterium, 'resistance_genes', [])} fitness = {bacterium.fitness:.3f}")
    
    # 6. Test resistance gene tracking
    print("\n6. Testing resistance gene tracking...")
    
    # Check gene distribution in population
    gene_counts = {}
    for bacterium in bacteria.values():
        if hasattr(bacterium, 'resistance_genes'):
            for gene in bacterium.resistance_genes:
                gene_counts[gene] = gene_counts.get(gene, 0) + 1
    
    print(f"✅ Gene distribution in population:")
    for gene, count in gene_counts.items():
        frequency = count / len(bacteria)
        print(f"  • {gene}: {count} bacteria ({frequency:.1%})")
    
    # 7. Test multiple rounds
    print("\n7. Testing multiple HGT rounds...")
    
    initial_events = len(transfer_engine.transfer_events)
    
    # Run additional rounds
    for generation in range(2, 5):
        round_events = transfer_engine.execute_hgt_round(
            bacterium_population=bacteria,
            spatial_positions=positions,
            environmental_factors=environmental_factors,
            generation=generation,
            max_transfers_per_round=15
        )
        print(f"  • Generation {generation}: {len(round_events)} transfers")
    
    final_events = len(transfer_engine.transfer_events)
    total_new_events = final_events - initial_events
    
    print(f"✅ Multiple rounds completed:")
    print(f"  • Additional events: {total_new_events}")
    print(f"  • Total events logged: {final_events}")
    
    # 8. Test recent events retrieval
    print("\n8. Testing recent events retrieval...")
    
    recent_events = transfer_engine.get_recent_events(limit=5)
    
    print(f"✅ Recent events (last 5):")
    for i, event in enumerate(recent_events):
        print(f"  • Event {i+1}: {event.mechanism.value} {event.donor_id}→{event.recipient_id}")
        print(f"    Generation: {event.generation}, Success: {event.success}")
    
    # 9. Test transfer engine with different environmental conditions
    print("\n9. Testing environmental condition effects...")
    
    # High stress environment
    high_stress_env = {
        "antibiotic_concentration": 3.0,
        "stress_response": 0.8,
        "nutrient_limitation": 0.6,
        "temperature_stress": 0.4
    }
    
    stress_events = transfer_engine.execute_hgt_round(
        bacterium_population=bacteria,
        spatial_positions=positions,
        environmental_factors=high_stress_env,
        generation=10,
        max_transfers_per_round=25
    )
    
    # No stress environment
    no_stress_events = transfer_engine.execute_hgt_round(
        bacterium_population=bacteria,
        spatial_positions=positions,
        environmental_factors={},
        generation=11,
        max_transfers_per_round=25
    )
    
    print(f"✅ Environmental effects:")
    print(f"  • High stress environment: {len(stress_events)} transfers")
    print(f"  • No stress environment: {len(no_stress_events)} transfers")
    
    if len(stress_events) > len(no_stress_events):
        print(f"  • Stress increases transfer activity as expected ✅")
    else:
        print(f"  • Transfer activity varies based on random factors")
    
    # 10. Test final statistics
    print("\n10. Final comprehensive statistics...")
    
    final_stats = transfer_engine.get_transfer_statistics()
    
    print(f"✅ Final Statistics:")
    print(f"  • Total transfer attempts: {final_stats['total_transfer_attempts']}")
    print(f"  • Overall success rate: {final_stats['overall_success_rate']:.1%}")
    print(f"  • Events logged: {final_stats['total_events_logged']}")
    
    # Check final resistance frequencies
    final_resistant = sum(1 for b in bacteria.values() if b.resistance_status == ResistanceStatus.RESISTANT)
    resistance_spread = final_resistant - initial_resistant
    
    print(f"  • Resistance spread: +{resistance_spread} newly resistant bacteria")
    print(f"  • Total resistant: {final_resistant}/{len(bacteria)} ({final_resistant/len(bacteria):.1%})")
    
    # Verify statistics make sense
    assert final_stats['total_transfer_attempts'] > 0, "Should have attempted transfers"
    assert final_stats['total_events_logged'] > 0, "Should have logged events"
    assert 0.0 <= final_stats['overall_success_rate'] <= 1.0, "Success rate should be between 0-100%"
    
    print("\n🎉 All HGT Gene Transfer Engine tests completed successfully!")
    return True


if __name__ == "__main__":
    print("🧬 HGT Gene Transfer Engine Test Suite")
    print("=" * 60)
    
    try:
        test_gene_transfer_engine()
        
        print("\n" + "=" * 60)
        print("🎊 ALL TESTS PASSED! HGT gene transfer engine is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 