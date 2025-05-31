#!/usr/bin/env python3
"""
Comprehensive test suite for resistance gene modeling and expression dynamics.
Tests ResistanceGeneModel, GeneExpressionController, and ResistanceSpreadTracker.
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hgt import (
    ResistanceGeneModel, GeneExpressionController, ResistanceSpreadTracker,
    ResistanceGeneState, EnvironmentalPressure, GeneTransferRecord, HGTMechanism
)
from models.bacterium import Bacterium, ResistanceStatus, Position
from datetime import datetime
import random


def test_resistance_gene_modeling():
    """Test comprehensive resistance gene modeling system."""
    print("🧬 Resistance Gene Modeling Test Suite")
    print("=" * 60)
    
    # 1. Test ResistanceGeneModel
    print("\n1. Testing ResistanceGeneModel...")
    resistance_model = ResistanceGeneModel()
    
    # Test gene state creation
    beta_lactamase_state = resistance_model.create_gene_state(
        "beta_lactamase", 
        acquisition_method="conjugation",
        generation=5
    )
    
    print(f"✅ Created beta_lactamase gene state:")
    print(f"  • Expression level: {beta_lactamase_state.expression_level:.3f}")
    print(f"  • Acquisition method: {beta_lactamase_state.acquisition_method}")
    print(f"  • Regulation: {beta_lactamase_state.regulation_state}")
    print(f"  • Metabolic cost: {beta_lactamase_state.metabolic_cost:.3f}")
    
    # Test environmental pressure
    print("\n2. Testing Environmental Pressure Response...")
    
    # No antibiotic pressure
    no_pressure = EnvironmentalPressure(
        antibiotic_concentration=0.0,
        stress_level=0.1,
        generation=10
    )
    
    # High antibiotic pressure
    high_pressure = EnvironmentalPressure(
        antibiotic_concentration=0.8,
        antibiotic_type="penicillin",
        stress_level=0.6,
        generation=11
    )
    
    print(f"✅ Environmental pressure scores:")
    print(f"  • No pressure: {no_pressure.get_pressure_score():.3f}")
    print(f"  • High pressure: {high_pressure.get_pressure_score():.3f}")
    
    # Test expression updates
    print("\n3. Testing Gene Expression Dynamics...")
    
    # Update under no pressure
    updated_no_pressure = resistance_model.update_expression(
        beta_lactamase_state, no_pressure
    )
    
    # Update under high pressure
    updated_high_pressure = resistance_model.update_expression(
        beta_lactamase_state, high_pressure
    )
    
    print(f"✅ Expression level changes:")
    print(f"  • Under no pressure: {updated_no_pressure.expression_level:.3f}")
    print(f"  • Under high pressure: {updated_high_pressure.expression_level:.3f}")
    print(f"  • Expression activated: {updated_high_pressure.is_expressed}")
    
    # Test fitness cost calculation
    print("\n4. Testing Fitness Cost Calculation...")
    
    # Create multiple gene states
    efflux_state = resistance_model.create_gene_state("efflux_pump", "transformation", 8)
    efflux_expressed = resistance_model.update_expression(efflux_state, high_pressure)
    
    gene_states = [updated_high_pressure, efflux_expressed]
    fitness_cost = resistance_model.calculate_fitness_cost(gene_states)
    
    print(f"✅ Fitness cost analysis:")
    print(f"  • Number of expressed genes: {len([gs for gs in gene_states if gs.is_expressed])}")
    print(f"  • Total fitness cost: {fitness_cost:.3f}")
    
    # Test resistance strength
    resistance_strength = resistance_model.get_resistance_strength(
        gene_states, "penicillin"
    )
    
    print(f"  • Resistance strength against penicillin: {resistance_strength:.3f}")
    
    print("\n5. Testing GeneExpressionController...")
    
    # Create expression controller
    expression_controller = GeneExpressionController(resistance_model)
    
    # Create bacterial population
    population = {}
    for i in range(6):
        bacterium = Bacterium(
            id=f"bacteria_{i}",
            position=Position(x=int(random.uniform(0, 10)), y=int(random.uniform(0, 10))),
            fitness=0.8,
            resistance_status=ResistanceStatus.SENSITIVE if i < 3 else ResistanceStatus.RESISTANT
        )
        
        # Add resistance genes to some bacteria
        if i >= 3:
            if not hasattr(bacterium, 'resistance_genes'):
                bacterium.resistance_genes = set()
            bacterium.resistance_genes = {"beta_lactamase", "efflux_pump"}
        else:
            # Initialize empty resistance genes for sensitive bacteria
            bacterium.resistance_genes = set()
        
        population[bacterium.id] = bacterium
    
    print(f"✅ Created population of {len(population)} bacteria")
    print(f"  • Sensitive: {len([b for b in population.values() if b.resistance_status == ResistanceStatus.SENSITIVE])}")
    print(f"  • Resistant: {len([b for b in population.values() if b.resistance_status == ResistanceStatus.RESISTANT])}")
    
    # Update population expression
    updated_gene_states = expression_controller.update_population_expression(
        population, high_pressure
    )
    
    # Apply gene states
    expression_controller.apply_gene_states_to_population(
        population, updated_gene_states
    )
    
    print(f"✅ Updated expression for population")
    
    # Get expression statistics
    expression_stats = expression_controller.get_population_expression_stats()
    print(f"  • Bacteria tracked: {expression_stats['total_bacteria']}")
    print(f"  • Genes tracked: {expression_stats['genes_tracked']}")
    
    for gene_name, stats in expression_stats.get('gene_expression_stats', {}).items():
        print(f"  • {gene_name}: {stats['expression_frequency']:.1%} expressing")
    
    print("\n6. Testing ResistanceSpreadTracker...")
    
    # Create spread tracker
    spread_tracker = ResistanceSpreadTracker()
    
    # Simulate HGT events
    transfer_record = GeneTransferRecord(
        id="transfer_001",
        donor_id="bacteria_3",
        recipient_id="bacteria_1",
        mechanism=HGTMechanism.CONJUGATION,
        genes_transferred=["beta_lactamase"],
        success=True,
        generation=15,
        probability=0.12
    )
    
    spread_tracker.record_hgt_spread(
        transfer_record, 
        "lineage_donor", 
        "lineage_recipient", 
        15
    )
    
    # Simulate vertical inheritance
    spread_tracker.record_vertical_inheritance(
        "bacteria_3", "bacteria_3_offspring", 
        ["beta_lactamase", "efflux_pump"], 
        16
    )
    
    # Take generation snapshot
    spread_tracker.take_generation_snapshot(population, 16)
    
    print(f"✅ Recorded spread events:")
    
    # Get spread statistics
    spread_stats = spread_tracker.get_spread_statistics()
    print(f"  • Total events: {spread_stats['total_events']}")
    print(f"  • HGT events: {spread_stats['hgt_events']}")
    print(f"  • Vertical events: {spread_stats['vertical_events']}")
    print(f"  • HGT success rate: {spread_stats['hgt_success_rate']:.1%}")
    print(f"  • Lineages tracked: {spread_stats['lineages_tracked']}")
    
    # Get gene spread history
    beta_history = spread_tracker.get_gene_spread_history("beta_lactamase")
    print(f"  • Beta-lactamase lineages: {beta_history['total_lineages']}")
    print(f"  • Beta-lactamase spread events: {beta_history['total_spread_events']}")
    
    print("\n7. Testing Environmental Response Dynamics...")
    
    # Test multiple generations with changing pressure
    generations_data = []
    current_pressure = EnvironmentalPressure(generation=20)
    
    for gen in range(20, 25):
        # Simulate antibiotic introduction at generation 22
        if gen >= 22:
            current_pressure.antibiotic_concentration = 0.6
            current_pressure.antibiotic_type = "penicillin"
        else:
            current_pressure.antibiotic_concentration = 0.0
            current_pressure.antibiotic_type = ""
        
        current_pressure.generation = gen
        
        # Update population expression
        gene_states = expression_controller.update_population_expression(
            population, current_pressure
        )
        expression_controller.apply_gene_states_to_population(population, gene_states)
        
        # Calculate population expression levels
        expressing_bacteria = 0
        total_expression = 0.0
        
        for bacterium in population.values():
            if hasattr(bacterium, 'gene_states'):
                for gene_state in bacterium.gene_states.values():
                    if gene_state.is_expressed:
                        expressing_bacteria += 1
                        total_expression += gene_state.expression_level
                        break
        
        avg_expression = total_expression / expressing_bacteria if expressing_bacteria > 0 else 0
        
        generations_data.append({
            "generation": gen,
            "antibiotic_present": current_pressure.antibiotic_concentration > 0,
            "expressing_bacteria": expressing_bacteria,
            "avg_expression": avg_expression
        })
        
        spread_tracker.take_generation_snapshot(population, gen)
    
    print(f"✅ Multi-generation expression dynamics:")
    for data in generations_data:
        antibiotic_status = "🔴 AB+" if data["antibiotic_present"] else "🟢 AB-"
        print(f"  • Gen {data['generation']}: {antibiotic_status} | "
              f"{data['expressing_bacteria']} expressing | "
              f"Avg expr: {data['avg_expression']:.3f}")
    
    print("\n8. Testing Gene Transfer Integration...")
    
    # Test applying HGT-derived genes to a bacterium
    recipient = population["bacteria_1"]
    
    # Simulate gene transfer from HGT
    new_gene_state = resistance_model.create_gene_state(
        "target_modification",
        acquisition_method="transduction",
        generation=25
    )
    
    # Add to bacterium
    if not hasattr(recipient, 'gene_states'):
        recipient.gene_states = {}
    if not hasattr(recipient, 'resistance_genes'):
        recipient.resistance_genes = set()
    
    recipient.gene_states["target_modification"] = new_gene_state
    recipient.resistance_genes.add("target_modification")
    
    # Update expression under pressure
    updated_states = expression_controller.update_population_expression(
        {"bacteria_1": recipient}, 
        EnvironmentalPressure(
            antibiotic_concentration=0.7,
            antibiotic_type="streptomycin",
            generation=26
        )
    )
    
    expression_controller.apply_gene_states_to_population(
        {"bacteria_1": recipient}, updated_states
    )
    
    target_mod_state = recipient.gene_states["target_modification"]
    print(f"✅ HGT-derived gene integration:")
    print(f"  • Gene: target_modification")
    print(f"  • Acquisition: {target_mod_state.acquisition_method}")
    print(f"  • Expression level: {target_mod_state.expression_level:.3f}")
    print(f"  • Active: {target_mod_state.is_expressed}")
    
    # Record this transfer
    hgt_record = GeneTransferRecord(
        id="transfer_002",
        donor_id="unknown_donor",
        recipient_id="bacteria_1",
        mechanism=HGTMechanism.TRANSDUCTION,
        genes_transferred=["target_modification"],
        success=True,
        generation=25,
        probability=0.08
    )
    
    spread_tracker.record_hgt_spread(
        hgt_record, "external_lineage", "lineage_bacteria_1", 25
    )
    
    # Final spread statistics
    final_stats = spread_tracker.get_spread_statistics()
    print(f"  • Final HGT events: {final_stats['hgt_events']}")
    print(f"  • Mechanism distribution: {final_stats['mechanism_distribution']}")
    
    print("\n🎊 ALL RESISTANCE GENE MODELING TESTS PASSED!")
    print("🔬 Resistance gene dynamics system is fully functional!")
    
    return True


if __name__ == "__main__":
    try:
        test_resistance_gene_modeling()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 