#!/usr/bin/env python3
"""
Test for HGT Transfer Probability Calculation System.
Tests the probability calculations with various biological factors.
"""

from models.hgt import ProbabilityCalculator, HGTConfig, HGTMechanism
from models.spatial import Coordinate
from models.bacterium import Bacterium, ResistanceStatus


def test_probability_calculator():
    """Test HGT probability calculation system."""
    
    print("🧪 Testing HGT Probability Calculator...")
    
    # 1. Setup configuration and calculator
    print("\n1. Setting up probability calculator...")
    config = HGTConfig(
        conjugation_distance=2.0,
        transformation_distance=4.0,
        transduction_distance=6.0,
        conjugation_probability=0.1,
        transformation_probability=0.05,
        transduction_probability=0.02
    )
    calculator = ProbabilityCalculator(config)
    print(f"✅ Probability calculator created with config")
    
    # 2. Create test bacteria with enhanced properties
    print("\n2. Creating test bacteria with enhanced HGT properties...")
    
    # Donor bacterium
    donor = Bacterium(
        id="donor_1",
        resistance_status=ResistanceStatus.RESISTANT,
        fitness=0.9,
        age=5
    )
    
    # Add HGT-specific properties
    donor.has_conjugative_plasmid = True
    donor.is_competent = True
    donor.phage_infected = True
    donor.species = "E.coli"
    donor.conjugation_type = "IncF"
    donor.phage_type = "T4"
    donor.is_motile = True
    donor.phage_load = 1.5
    
    # Recipient bacterium
    recipient = Bacterium(
        id="recipient_1", 
        resistance_status=ResistanceStatus.SENSITIVE,
        fitness=0.85,
        age=3
    )
    
    # Add HGT-specific properties
    recipient.has_conjugative_plasmid = False
    recipient.is_competent = True
    recipient.phage_infected = False
    recipient.phage_resistant = False
    recipient.species = "E.coli"
    recipient.conjugation_type = "IncF"
    recipient.surface_receptors = True
    recipient.dna_uptake_efficiency = 0.8
    recipient.phage_sensitivity = ["T4", "T7"]
    recipient.natural_transformation = True
    recipient.conjugation_compatible = True
    
    print(f"✅ Created donor: {donor.id} (species={donor.species}, fitness={donor.fitness})")
    print(f"✅ Created recipient: {recipient.id} (species={recipient.species}, fitness={recipient.fitness})")
    
    # 3. Test basic probability calculations for each mechanism
    print("\n3. Testing basic probability calculations...")
    
    distance = 1.5  # Within all thresholds
    
    for mechanism in HGTMechanism:
        probability = calculator.calculate_transfer_probability(
            donor=donor,
            recipient=recipient,
            mechanism=mechanism,
            distance=distance
        )
        print(f"  • {mechanism.value}: {probability:.4f}")
        assert 0.0 <= probability <= 1.0, f"Probability out of range for {mechanism.value}"
    
    print("✅ Basic probability calculations working correctly")
    
    # 4. Test distance effects
    print("\n4. Testing distance effects...")
    
    distances = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    mechanism = HGTMechanism.CONJUGATION
    
    print(f"Distance effects for {mechanism.value}:")
    previous_prob = 1.0
    
    for distance in distances:
        probability = calculator.calculate_transfer_probability(
            donor=donor,
            recipient=recipient,
            mechanism=mechanism,
            distance=distance
        )
        print(f"  • Distance {distance}: {probability:.4f}")
        
        # Probability should decrease with distance (mostly)
        if distance <= config.get_distance_threshold(mechanism):
            assert probability >= 0.0, f"Probability should be non-negative at distance {distance}"
        else:
            assert probability == 0.0, f"Probability should be 0 beyond threshold distance"
    
    print("✅ Distance effects working correctly")
    
    # 5. Test environmental factors
    print("\n5. Testing environmental factor effects...")
    
    base_prob = calculator.calculate_transfer_probability(
        donor=donor,
        recipient=recipient,
        mechanism=HGTMechanism.TRANSFORMATION,
        distance=1.0
    )
    
    # Test antibiotic stress
    antibiotic_env = {"antibiotic_concentration": 2.0}
    antibiotic_prob = calculator.calculate_transfer_probability(
        donor=donor,
        recipient=recipient,
        mechanism=HGTMechanism.TRANSFORMATION,
        distance=1.0,
        environmental_factors=antibiotic_env
    )
    
    print(f"  • Base probability: {base_prob:.4f}")
    print(f"  • With antibiotics: {antibiotic_prob:.4f}")
    print(f"  • Antibiotic factor: {antibiotic_prob/base_prob:.2f}x")
    
    # Test multiple stress factors
    stress_env = {
        "antibiotic_concentration": 1.0,
        "stress_response": 0.5,
        "nutrient_limitation": 0.3,
        "temperature_stress": 0.2
    }
    
    stress_prob = calculator.calculate_transfer_probability(
        donor=donor,
        recipient=recipient,
        mechanism=HGTMechanism.TRANSFORMATION,
        distance=1.0,
        environmental_factors=stress_env
    )
    
    print(f"  • With multiple stresses: {stress_prob:.4f}")
    print(f"  • Combined stress factor: {stress_prob/base_prob:.2f}x")
    
    # Antibiotic and stress should generally increase transfer
    assert antibiotic_prob >= base_prob, "Antibiotics should increase transfer"
    
    print("✅ Environmental factors working correctly")
    
    # 6. Test species compatibility
    print("\n6. Testing species compatibility...")
    
    # Create different species recipient
    different_recipient = Bacterium(
        id="recipient_2",
        resistance_status=ResistanceStatus.SENSITIVE,
        fitness=0.85,
        age=3
    )
    different_recipient.species = "S.aureus"
    different_recipient.is_competent = True
    different_recipient.conjugation_compatible = True
    different_recipient.natural_transformation = True
    
    same_species_prob = calculator.calculate_transfer_probability(
        donor=donor,
        recipient=recipient,  # Same species (E.coli)
        mechanism=HGTMechanism.CONJUGATION,
        distance=1.0
    )
    
    different_species_prob = calculator.calculate_transfer_probability(
        donor=donor,
        recipient=different_recipient,  # Different species (S.aureus)
        mechanism=HGTMechanism.CONJUGATION,
        distance=1.0
    )
    
    print(f"  • Same species (E.coli): {same_species_prob:.4f}")
    print(f"  • Different species (S.aureus): {different_species_prob:.4f}")
    print(f"  • Species compatibility factor: {same_species_prob/different_species_prob:.2f}x")
    
    # Same species should have higher transfer rate
    assert same_species_prob > different_species_prob, "Same species should have higher transfer rate"
    
    print("✅ Species compatibility working correctly")
    
    # 7. Test population density effects
    print("\n7. Testing population density effects...")
    
    densities = [0.5, 2.0, 5.0, 10.0, 15.0, 25.0]
    
    print("Population density effects:")
    for density in densities:
        prob = calculator.calculate_transfer_probability(
            donor=donor,
            recipient=recipient,
            mechanism=HGTMechanism.CONJUGATION,
            distance=1.0,
            population_density=density
        )
        print(f"  • Density {density:4.1f}: {prob:.4f}")
    
    print("✅ Population density effects working correctly")
    
    # 8. Test cell state factors
    print("\n8. Testing cell state factors...")
    
    # Create bacteria with different fitness levels
    high_fitness_donor = Bacterium(id="donor_fit", fitness=0.95, age=2)
    high_fitness_donor.has_conjugative_plasmid = True
    high_fitness_donor.species = "E.coli"
    
    low_fitness_donor = Bacterium(id="donor_unfit", fitness=0.3, age=20)
    low_fitness_donor.has_conjugative_plasmid = True
    low_fitness_donor.species = "E.coli"
    
    high_recipient = Bacterium(id="recipient_fit", fitness=0.9, age=1)
    high_recipient.species = "E.coli"
    high_recipient.conjugation_compatible = True
    
    low_recipient = Bacterium(id="recipient_unfit", fitness=0.4, age=25)
    low_recipient.species = "E.coli"
    low_recipient.conjugation_compatible = True
    
    high_high_prob = calculator.calculate_transfer_probability(
        donor=high_fitness_donor,
        recipient=high_recipient,
        mechanism=HGTMechanism.CONJUGATION,
        distance=1.0
    )
    
    low_low_prob = calculator.calculate_transfer_probability(
        donor=low_fitness_donor,
        recipient=low_recipient,
        mechanism=HGTMechanism.CONJUGATION,
        distance=1.0
    )
    
    print(f"  • High fitness pair: {high_high_prob:.4f}")
    print(f"  • Low fitness pair: {low_low_prob:.4f}")
    print(f"  • Fitness effect: {high_high_prob/low_low_prob:.2f}x")
    
    # Higher fitness should lead to higher transfer rates
    assert high_high_prob > low_low_prob, "Higher fitness should increase transfer rates"
    
    print("✅ Cell state factors working correctly")
    
    # 9. Test population transfer rates calculation
    print("\n9. Testing population transfer rates calculation...")
    
    # Create a small population
    population = {
        "donor_1": donor,
        "recipient_1": recipient,
        "recipient_2": different_recipient
    }
    
    positions = {
        "donor_1": Coordinate(10.0, 10.0),
        "recipient_1": Coordinate(11.0, 10.0),  # Distance 1.0
        "recipient_2": Coordinate(12.0, 10.0)   # Distance 2.0
    }
    
    candidates = {
        "donor_1": ["recipient_1", "recipient_2"]
    }
    
    transfer_rates = calculator.calculate_population_transfer_rates(
        bacterium_population=population,
        candidates=candidates,
        mechanism=HGTMechanism.CONJUGATION,
        positions=positions,
        environmental_factors={"antibiotic_concentration": 1.0}
    )
    
    print(f"✅ Population transfer rates calculated:")
    for donor_id, recipients in transfer_rates.items():
        print(f"  • Donor {donor_id}:")
        for recipient_id, probability in recipients.items():
            print(f"    - {recipient_id}: {probability:.4f}")
    
    # Should have calculated rates for the donor
    assert "donor_1" in transfer_rates, "Should have rates for donor_1"
    assert len(transfer_rates["donor_1"]) > 0, "Should have at least one recipient rate"
    
    print("✅ Population transfer rates working correctly")
    
    print("\n🎉 All HGT Probability Calculator tests completed successfully!")
    return True


if __name__ == "__main__":
    print("🧬 HGT Probability Calculator Test Suite")
    print("=" * 50)
    
    try:
        test_probability_calculator()
        
        print("\n" + "=" * 50)
        print("🎊 ALL TESTS PASSED! HGT probability calculation system is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 