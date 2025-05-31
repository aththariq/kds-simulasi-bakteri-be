"""
Horizontal Gene Transfer (HGT) implementation for bacterial resistance simulation.
This module provides comprehensive HGT modeling including proximity detection,
probability calculation, and gene transfer mechanics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Set, Any
from datetime import datetime
import random
import uuid
import math
import asyncio
from collections import defaultdict, deque

# Import bacterium components
from models.bacterium import Bacterium, ResistanceStatus

# Import spatial components
from models.spatial import Coordinate, SpatialGrid, SpatialManager


class HGTMechanism(Enum):
    """Types of horizontal gene transfer mechanisms."""
    CONJUGATION = "conjugation"  # Direct cell contact via pili
    TRANSFORMATION = "transformation"  # Uptake of free DNA from environment
    TRANSDUCTION = "transduction"  # Transfer via bacteriophages


class GeneTransferEvent(Enum):
    """Types of gene transfer events."""
    RESISTANCE_GENE = "resistance_gene"
    PLASMID = "plasmid" 
    CHROMOSOMAL_DNA = "chromosomal_dna"
    VIRULENCE_FACTOR = "virulence_factor"


@dataclass
class HGTConfig:
    """Configuration for horizontal gene transfer simulation."""
    
    # Distance thresholds for different mechanisms (in spatial units)
    conjugation_distance: float = 1.0
    transformation_distance: float = 3.0
    transduction_distance: float = 5.0
    
    # Base transfer probabilities per mechanism
    conjugation_probability: float = 0.1
    transformation_probability: float = 0.05
    transduction_probability: float = 0.02
    
    # Environmental factors
    stress_factor_multiplier: float = 2.0  # Increase HGT under stress
    antibiotic_presence_multiplier: float = 1.5  # More transfer when antibiotics present
    
    # Population density effects
    high_density_threshold: float = 10.0  # bacteria per unit area
    high_density_multiplier: float = 1.3
    
    # Genetic compatibility factors
    same_species_multiplier: float = 2.0
    different_species_multiplier: float = 0.5
    
    # Transfer limitations
    max_transfers_per_generation: int = 100
    max_transfers_per_bacterium: int = 3
    cooldown_generations: int = 5  # Generations before bacterium can transfer again
    
    # Gene-specific transfer rates
    resistance_gene_transfer_rate: float = 0.8
    plasmid_transfer_rate: float = 0.6
    chromosomal_transfer_rate: float = 0.3
    
    def get_distance_threshold(self, mechanism: HGTMechanism) -> float:
        """Get distance threshold for a specific HGT mechanism."""
        return {
            HGTMechanism.CONJUGATION: self.conjugation_distance,
            HGTMechanism.TRANSFORMATION: self.transformation_distance,
            HGTMechanism.TRANSDUCTION: self.transduction_distance
        }[mechanism]
    
    def get_base_probability(self, mechanism: HGTMechanism) -> float:
        """Get base transfer probability for a specific mechanism."""
        return {
            HGTMechanism.CONJUGATION: self.conjugation_probability,
            HGTMechanism.TRANSFORMATION: self.transformation_probability,
            HGTMechanism.TRANSDUCTION: self.transduction_probability
        }[mechanism]


@dataclass
class HGTEvent:
    """Record of a horizontal gene transfer event."""
    
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    generation: int = 0
    
    # Transfer participants
    donor_id: str = ""
    recipient_id: str = ""
    
    # Transfer details
    mechanism: HGTMechanism = HGTMechanism.CONJUGATION
    transfer_type: GeneTransferEvent = GeneTransferEvent.RESISTANCE_GENE
    genes_transferred: List[str] = field(default_factory=list)
    
    # Spatial information
    distance: float = 0.0
    donor_position: Optional[Coordinate] = None
    recipient_position: Optional[Coordinate] = None
    
    # Environmental conditions
    antibiotic_concentration: float = 0.0
    local_density: float = 0.0
    stress_level: float = 0.0
    
    # Success metrics
    transfer_probability: float = 0.0
    successful: bool = False
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "generation": self.generation,
            "donor_id": self.donor_id,
            "recipient_id": self.recipient_id,
            "mechanism": self.mechanism.value,
            "transfer_type": self.transfer_type.value,
            "genes_transferred": self.genes_transferred,
            "distance": self.distance,
            "donor_position": {"x": self.donor_position.x, "y": self.donor_position.y} if self.donor_position else None,
            "recipient_position": {"x": self.recipient_position.x, "y": self.recipient_position.y} if self.recipient_position else None,
            "antibiotic_concentration": self.antibiotic_concentration,
            "local_density": self.local_density,
            "stress_level": self.stress_level,
            "transfer_probability": self.transfer_probability,
            "successful": self.successful
        }


@dataclass
class GeneTransferRecord:
    """
    Record of a completed gene transfer event with detailed information.
    """
    id: str
    donor_id: str
    recipient_id: str
    mechanism: HGTMechanism
    genes_transferred: List[str]
    success: bool
    generation: int
    probability: float
    failure_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        """Convert transfer record to dictionary for serialization."""
        return {
            "id": self.id,
            "donor_id": self.donor_id,
            "recipient_id": self.recipient_id,
            "mechanism": self.mechanism.value,
            "genes_transferred": self.genes_transferred,
            "success": self.success,
            "generation": self.generation,
            "probability": self.probability,
            "failure_reason": self.failure_reason,
            "timestamp": self.timestamp.isoformat()
        }


class ProximityDetector:
    """
    Advanced proximity detection system for horizontal gene transfer.
    
    Leverages the spatial grid system to efficiently identify candidate
    donor-recipient pairs for gene transfer based on multiple biological
    and environmental factors.
    """
    
    def __init__(self, spatial_manager: SpatialManager, config: HGTConfig):
        self.spatial_manager = spatial_manager
        self.config = config
        
        # Performance tracking
        self._detection_cache: Dict[str, Dict[str, List[str]]] = {}
        self._cache_generation = -1
        self._detection_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_detections": 0,
            "avg_candidates_per_detection": 0.0
        }
        
    def detect_hgt_candidates(
        self,
        bacterium_population: Dict[str, Bacterium],
        mechanism: HGTMechanism,
        current_generation: int = 0,
        use_cache: bool = True
    ) -> Dict[str, List[str]]:
        """
        Detect all potential HGT candidates for a given mechanism.
        
        Args:
            bacterium_population: Dictionary of bacterium_id -> Bacterium
            mechanism: HGT mechanism to detect candidates for
            current_generation: Current simulation generation
            use_cache: Whether to use detection caching
            
        Returns:
            Dictionary mapping donor_id -> list of recipient_ids
        """
        # Update cache if generation changed
        if self._cache_generation != current_generation:
            self._detection_cache.clear()
            self._cache_generation = current_generation
        
        cache_key = f"{mechanism.value}_{len(bacterium_population)}"
        
        # Check cache first
        if use_cache and cache_key in self._detection_cache:
            self._detection_stats["cache_hits"] += 1
            return self._detection_cache[cache_key]
        
        self._detection_stats["cache_misses"] += 1
        self._detection_stats["total_detections"] += 1
        
        # Get distance threshold for this mechanism
        distance_threshold = self.config.get_distance_threshold(mechanism)
        
        # Find candidates for each potential donor
        candidates = {}
        total_candidates = 0
        
        for donor_id, donor in bacterium_population.items():
            if not self._is_viable_donor(donor, mechanism):
                continue
            
            # Use spatial system to find nearby bacteria
            nearby_ids = self.spatial_manager.calculate_hgt_candidates(
                donor_id=donor_id,
                hgt_radius=distance_threshold,
                max_candidates=50  # Limit for performance
            )
            
            # Filter to viable recipients
            viable_recipients = []
            for recipient_id in nearby_ids:
                if recipient_id in bacterium_population:
                    recipient = bacterium_population[recipient_id]
                    if self._is_viable_recipient(recipient, donor, mechanism):
                        viable_recipients.append(recipient_id)
            
            if viable_recipients:
                candidates[donor_id] = viable_recipients
                total_candidates += len(viable_recipients)
        
        # Update statistics
        if candidates:
            self._detection_stats["avg_candidates_per_detection"] = total_candidates / len(candidates)
        
        # Cache results
        if use_cache:
            self._detection_cache[cache_key] = candidates
        
        logger.debug(f"Detected {len(candidates)} donors with {total_candidates} total candidates for {mechanism.value}")
        return candidates
    
    def _is_viable_donor(self, bacterium: Bacterium, mechanism: HGTMechanism) -> bool:
        """
        Check if a bacterium can act as a gene donor.
        
        Args:
            bacterium: Bacterium to check
            mechanism: HGT mechanism
            
        Returns:
            True if bacterium can be a donor
        """
        # Basic viability checks
        if not bacterium.is_alive():
            return False
        
        # Check if bacterium has genetic material to transfer
        if mechanism == HGTMechanism.CONJUGATION:
            # Need conjugative plasmids or mobile elements
            return hasattr(bacterium, 'has_conjugative_plasmid') and bacterium.has_conjugative_plasmid
        
        elif mechanism == HGTMechanism.TRANSFORMATION:
            # Dead bacteria can still contribute DNA for transformation
            return True
        
        elif mechanism == HGTMechanism.TRANSDUCTION:
            # Need phage infection or lysogenic state
            return hasattr(bacterium, 'phage_infected') and bacterium.phage_infected
        
        return True
    
    def _is_viable_recipient(
        self, 
        recipient: Bacterium, 
        donor: Bacterium, 
        mechanism: HGTMechanism
    ) -> bool:
        """
        Check if a bacterium can act as a gene recipient.
        
        Args:
            recipient: Potential recipient bacterium
            donor: Donor bacterium
            mechanism: HGT mechanism
            
        Returns:
            True if bacterium can be a recipient
        """
        # Basic viability checks
        if not recipient.is_alive():
            return False
        
        # Can't transfer to self
        if recipient.id == donor.id:
            return False
        
        # Mechanism-specific checks
        if mechanism == HGTMechanism.CONJUGATION:
            # Check for compatible surface receptors
            return self._check_conjugation_compatibility(recipient, donor)
        
        elif mechanism == HGTMechanism.TRANSFORMATION:
            # Check for competence state
            return hasattr(recipient, 'is_competent') and recipient.is_competent
        
        elif mechanism == HGTMechanism.TRANSDUCTION:
            # Check for phage susceptibility
            return not (hasattr(recipient, 'phage_resistant') and recipient.phage_resistant)
        
        return True
    
    def _check_conjugation_compatibility(self, recipient: Bacterium, donor: Bacterium) -> bool:
        """
        Check if two bacteria are compatible for conjugation.
        
        Args:
            recipient: Recipient bacterium
            donor: Donor bacterium
            
        Returns:
            True if compatible for conjugation
        """
        # Check species compatibility
        if hasattr(recipient, 'species') and hasattr(donor, 'species'):
            # Same species are more compatible
            return recipient.species == donor.species or random.random() < 0.3
        
        # Default compatibility
        return True
    
    def get_proximity_metrics(self, bacterium_population: Dict[str, Bacterium]) -> Dict[str, Union[int, float]]:
        """
        Get proximity detection performance metrics.
        
        Args:
            bacterium_population: Current bacterial population
            
        Returns:
            Dictionary with performance metrics
        """
        total_bacteria = len(bacterium_population)
        
        # Calculate potential interactions for each mechanism
        potential_interactions = {}
        for mechanism in HGTMechanism:
            candidates = self.detect_hgt_candidates(bacterium_population, mechanism, use_cache=False)
            potential_interactions[mechanism.value] = sum(len(recipients) for recipients in candidates.values())
        
        return {
            "total_bacteria": total_bacteria,
            "detection_cache_size": len(self._detection_cache),
            "cache_hit_rate": (
                self._detection_stats["cache_hits"] / 
                max(self._detection_stats["cache_hits"] + self._detection_stats["cache_misses"], 1)
            ),
            "avg_candidates_per_detection": self._detection_stats["avg_candidates_per_detection"],
            "potential_conjugation_pairs": potential_interactions.get(HGTMechanism.CONJUGATION.value, 0),
            "potential_transformation_pairs": potential_interactions.get(HGTMechanism.TRANSFORMATION.value, 0),
            "potential_transduction_pairs": potential_interactions.get(HGTMechanism.TRANSDUCTION.value, 0),
            "total_potential_transfers": sum(potential_interactions.values())
        }
    
    def clear_cache(self):
        """Clear the detection cache."""
        self._detection_cache.clear()
        self._cache_generation = -1
        logger.debug("Proximity detection cache cleared")


class ProbabilityCalculator:
    """
    Advanced probability calculation system for horizontal gene transfer.
    
    Calculates transfer probabilities based on multiple biological factors
    including environmental conditions, genetic compatibility, and cell states.
    """
    
    def __init__(self, config: HGTConfig):
        self.config = config
        
        # Environmental factor weights
        self._environmental_weights = {
            "stress_response": 1.5,      # Higher stress increases HGT
            "nutrient_limitation": 1.3,   # Resource scarcity promotes transfer
            "temperature_stress": 1.2,    # Temperature changes affect transfer
            "pH_stress": 1.1,            # pH variations influence rates
            "osmotic_stress": 1.2        # Osmotic pressure effects
        }
        
        # Population density curve parameters
        self._density_curve = {
            "optimal_density": 5.0,      # Optimal bacteria per unit area
            "overcrowding_penalty": 0.7, # Penalty for high density
            "isolation_penalty": 0.8     # Penalty for low density
        }
    
    def calculate_transfer_probability(
        self,
        donor: Bacterium,
        recipient: Bacterium,
        mechanism: HGTMechanism,
        distance: float,
        environmental_factors: Optional[Dict[str, float]] = None,
        population_density: float = 1.0,
        generation: int = 0
    ) -> float:
        """
        Calculate comprehensive transfer probability between two bacteria.
        
        Args:
            donor: Donor bacterium
            recipient: Recipient bacterium
            mechanism: HGT mechanism
            distance: Distance between bacteria
            environmental_factors: Dict of environmental stress factors
            population_density: Local population density
            generation: Current generation
            
        Returns:
            Transfer probability (0.0 to 1.0)
        """
        # Start with base probability from configuration
        base_prob = self.config.get_base_probability(mechanism)
        
        # Apply distance factor
        distance_factor = self._calculate_distance_factor(distance, mechanism)
        
        # Apply environmental factors
        env_factor = self._calculate_environmental_factor(
            environmental_factors or {}
        )
        
        # Apply genetic compatibility
        compatibility_factor = self._calculate_compatibility_factor(
            donor, recipient, mechanism
        )
        
        # Apply population density effects
        density_factor = self._calculate_density_factor(population_density)
        
        # Apply cell state factors
        cell_state_factor = self._calculate_cell_state_factor(
            donor, recipient, mechanism
        )
        
        # Apply mechanism-specific adjustments
        mechanism_factor = self._calculate_mechanism_factor(
            donor, recipient, mechanism
        )
        
        # Combine all factors
        probability = (
            base_prob *
            distance_factor *
            env_factor *
            compatibility_factor *
            density_factor *
            cell_state_factor *
            mechanism_factor
        )
        
        # Apply upper and lower bounds
        probability = max(0.0, min(1.0, probability))
        
        logger.debug(
            f"Transfer probability {donor.id}->{recipient.id} ({mechanism.value}): "
            f"base={base_prob:.3f}, dist={distance_factor:.3f}, env={env_factor:.3f}, "
            f"compat={compatibility_factor:.3f}, density={density_factor:.3f}, "
            f"state={cell_state_factor:.3f}, mech={mechanism_factor:.3f} = {probability:.3f}"
        )
        
        return probability
    
    def _calculate_distance_factor(self, distance: float, mechanism: HGTMechanism) -> float:
        """Calculate distance-based probability reduction factor."""
        max_distance = self.config.get_distance_threshold(mechanism)
        
        if distance > max_distance:
            return 0.0
        
        # Use exponential decay for distance effect
        if mechanism == HGTMechanism.CONJUGATION:
            # Conjugation requires very close contact
            return np.exp(-distance * 2.0)
        elif mechanism == HGTMechanism.TRANSFORMATION:
            # Transformation less sensitive to distance
            return np.exp(-distance * 0.5)
        else:  # TRANSDUCTION
            # Transduction can work at longer distances
            return np.exp(-distance * 0.3)
    
    def _calculate_environmental_factor(self, env_factors: Dict[str, float]) -> float:
        """Calculate environmental stress effects on transfer probability."""
        if not env_factors:
            return 1.0
        
        total_factor = 1.0
        
        # Antibiotic pressure increases HGT
        if "antibiotic_concentration" in env_factors:
            antibiotic_level = env_factors["antibiotic_concentration"]
            antibiotic_factor = 1.0 + (antibiotic_level * self.config.antibiotic_presence_multiplier * 0.1)
            total_factor *= antibiotic_factor
        
        # General stress factors
        for factor_name, stress_level in env_factors.items():
            if factor_name in self._environmental_weights:
                weight = self._environmental_weights[factor_name]
                stress_factor = 1.0 + (stress_level * (weight - 1.0) * 0.1)
                total_factor *= stress_factor
        
        return min(self.config.stress_factor_multiplier, total_factor)
    
    def _calculate_compatibility_factor(
        self, 
        donor: Bacterium, 
        recipient: Bacterium, 
        mechanism: HGTMechanism
    ) -> float:
        """Calculate genetic compatibility factor."""
        compatibility = 1.0
        
        # Species compatibility
        if hasattr(donor, 'species') and hasattr(recipient, 'species'):
            if donor.species == recipient.species:
                compatibility *= self.config.same_species_multiplier
            else:
                compatibility *= self.config.different_species_multiplier
        
        # Mechanism-specific compatibility
        if mechanism == HGTMechanism.CONJUGATION:
            # Check for conjugation compatibility
            if hasattr(donor, 'conjugation_type') and hasattr(recipient, 'conjugation_type'):
                if donor.conjugation_type == recipient.conjugation_type:
                    compatibility *= 1.5  # Same conjugation system
                else:
                    compatibility *= 0.7  # Different systems
        
        elif mechanism == HGTMechanism.TRANSFORMATION:
            # DNA uptake compatibility
            if hasattr(recipient, 'dna_uptake_efficiency'):
                compatibility *= recipient.dna_uptake_efficiency
        
        elif mechanism == HGTMechanism.TRANSDUCTION:
            # Phage host range compatibility
            if hasattr(donor, 'phage_type') and hasattr(recipient, 'phage_sensitivity'):
                if donor.phage_type in recipient.phage_sensitivity:
                    compatibility *= 2.0  # Compatible phage
                else:
                    compatibility *= 0.1  # Incompatible phage
        
        return compatibility
    
    def _calculate_density_factor(self, population_density: float) -> float:
        """Calculate population density effects."""
        optimal = self._density_curve["optimal_density"]
        
        if population_density < optimal:
            # Low density - isolation penalty
            factor = self._density_curve["isolation_penalty"]
            return factor + (1.0 - factor) * (population_density / optimal)
        
        elif population_density > self.config.high_density_threshold:
            # High density - both crowding penalty and enhanced contact
            crowding_penalty = self._density_curve["overcrowding_penalty"]
            contact_bonus = self.config.high_density_multiplier
            
            # Balance between penalty and bonus
            penalty_weight = min(1.0, population_density / (self.config.high_density_threshold * 2))
            return (crowding_penalty * penalty_weight) + (contact_bonus * (1 - penalty_weight))
        
        else:
            # Optimal to high density - gradual increase
            excess = population_density - optimal
            max_excess = self.config.high_density_threshold - optimal
            factor = 1.0 + (excess / max_excess) * (self.config.high_density_multiplier - 1.0)
            return factor
    
    def _calculate_cell_state_factor(
        self, 
        donor: Bacterium, 
        recipient: Bacterium, 
        mechanism: HGTMechanism
    ) -> float:
        """Calculate cell state and health effects."""
        factor = 1.0
        
        # Donor fitness affects transfer capability
        donor_fitness = getattr(donor, 'effective_fitness', donor.fitness)
        factor *= (0.5 + 0.5 * donor_fitness)  # Range: 0.5 to 1.0
        
        # Recipient fitness affects receptivity
        recipient_fitness = getattr(recipient, 'effective_fitness', recipient.fitness)
        factor *= (0.7 + 0.3 * recipient_fitness)  # Range: 0.7 to 1.0
        
        # Age effects
        donor_age_factor = max(0.5, 1.0 - (donor.age * 0.01))
        recipient_age_factor = max(0.5, 1.0 - (recipient.age * 0.01))
        factor *= (donor_age_factor * recipient_age_factor)
        
        # Mechanism-specific cell state requirements
        if mechanism == HGTMechanism.CONJUGATION:
            # Both cells need to be active for physical contact
            if hasattr(donor, 'is_motile') and not donor.is_motile:
                factor *= 0.8
            if hasattr(recipient, 'surface_receptors') and not recipient.surface_receptors:
                factor *= 0.7
        
        elif mechanism == HGTMechanism.TRANSFORMATION:
            # Recipient needs to be competent
            if hasattr(recipient, 'competence_level'):
                factor *= recipient.competence_level
            elif hasattr(recipient, 'is_competent') and not recipient.is_competent:
                factor *= 0.3
        
        elif mechanism == HGTMechanism.TRANSDUCTION:
            # Donor needs phage infection
            if hasattr(donor, 'phage_load'):
                factor *= min(2.0, donor.phage_load)
            elif hasattr(donor, 'phage_infected') and not donor.phage_infected:
                factor *= 0.1
        
        return factor
    
    def _calculate_mechanism_factor(
        self, 
        donor: Bacterium, 
        recipient: Bacterium, 
        mechanism: HGTMechanism
    ) -> float:
        """Calculate mechanism-specific adjustment factors."""
        factor = 1.0
        
        if mechanism == HGTMechanism.CONJUGATION:
            # Requires donor to have conjugative elements
            if hasattr(donor, 'has_conjugative_plasmid'):
                if donor.has_conjugative_plasmid:
                    factor *= 1.5
                else:
                    factor *= 0.1
            
            # Recipient compatibility
            if hasattr(recipient, 'conjugation_compatible'):
                if not recipient.conjugation_compatible:
                    factor *= 0.3
        
        elif mechanism == HGTMechanism.TRANSFORMATION:
            # Natural transformation capability
            if hasattr(recipient, 'natural_transformation'):
                if recipient.natural_transformation:
                    factor *= 2.0
                else:
                    factor *= 0.5
        
        elif mechanism == HGTMechanism.TRANSDUCTION:
            # Phage-mediated transfer
            if hasattr(recipient, 'phage_resistant'):
                if recipient.phage_resistant:
                    factor *= 0.2
                else:
                    factor *= 1.8
        
        return factor
    
    def calculate_population_transfer_rates(
        self,
        bacterium_population: Dict[str, Bacterium],
        candidates: Dict[str, List[str]],
        mechanism: HGTMechanism,
        positions: Dict[str, Coordinate],
        environmental_factors: Optional[Dict[str, float]] = None,
        generation: int = 0
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate transfer probabilities for all candidate pairs.
        
        Args:
            bacterium_population: Population of bacteria
            candidates: Donor -> recipient pairs from proximity detection
            mechanism: HGT mechanism
            positions: Bacterial positions for distance calculation
            environmental_factors: Environmental conditions
            generation: Current generation
            
        Returns:
            Dict of donor_id -> {recipient_id: probability}
        """
        transfer_rates = {}
        
        # Calculate local population density
        total_area = 100.0 * 100.0  # Assuming 100x100 grid
        population_density = len(bacterium_population) / total_area
        
        for donor_id, recipient_ids in candidates.items():
            if donor_id not in bacterium_population:
                continue
            
            donor = bacterium_population[donor_id]
            donor_rates = {}
            
            for recipient_id in recipient_ids:
                if recipient_id not in bacterium_population:
                    continue
                
                recipient = bacterium_population[recipient_id]
                
                # Calculate distance
                distance = 0.0
                if donor_id in positions and recipient_id in positions:
                    distance = positions[donor_id].distance_to(positions[recipient_id])
                
                # Calculate transfer probability
                probability = self.calculate_transfer_probability(
                    donor=donor,
                    recipient=recipient,
                    mechanism=mechanism,
                    distance=distance,
                    environmental_factors=environmental_factors,
                    population_density=population_density,
                    generation=generation
                )
                
                if probability > 0.0:
                    donor_rates[recipient_id] = probability
            
            if donor_rates:
                transfer_rates[donor_id] = donor_rates
        
        return transfer_rates


class GeneTransferEngine:
    """
    Core engine for executing horizontal gene transfer events.
    
    Combines proximity detection and probability calculation to perform
    actual gene transfers between bacterial populations.
    """
    
    def __init__(
        self, 
        proximity_detector: ProximityDetector,
        probability_calculator: ProbabilityCalculator,
        config: HGTConfig
    ):
        self.proximity_detector = proximity_detector
        self.probability_calculator = probability_calculator
        self.config = config
        
        # Transfer tracking
        self.transfer_events: List[GeneTransferRecord] = []
        self.transfer_stats = {
            "total_attempts": 0,
            "successful_transfers": 0,
            "failed_transfers": 0,
            "by_mechanism": {mechanism: 0 for mechanism in HGTMechanism}
        }
        
        # Resistance gene tracking
        self.resistance_genes = {
            "beta_lactamase": {"frequency": 0.0, "last_seen": 0},
            "efflux_pump": {"frequency": 0.0, "last_seen": 0},
            "target_modification": {"frequency": 0.0, "last_seen": 0},
            "drug_inactivation": {"frequency": 0.0, "last_seen": 0},
            "permeability_reduction": {"frequency": 0.0, "last_seen": 0}
        }
        
    def execute_hgt_round(
        self,
        bacterium_population: Dict[str, Bacterium],
        spatial_positions: Dict[str, Coordinate],
        environmental_factors: Optional[Dict[str, float]] = None,
        generation: int = 0,
        max_transfers_per_round: int = 1000
    ) -> List[GeneTransferRecord]:
        """
        Execute a complete round of horizontal gene transfer.
        
        Args:
            bacterium_population: Current bacterial population
            spatial_positions: Spatial positions of bacteria
            environmental_factors: Environmental conditions
            generation: Current generation number
            max_transfers_per_round: Limit on transfers per round
            
        Returns:
            List of successful transfer events
        """
        round_events = []
        transfer_count = 0
        
        logger.info(f"Starting HGT round for generation {generation} with {len(bacterium_population)} bacteria")
        
        # Process each HGT mechanism
        for mechanism in HGTMechanism:
            if transfer_count >= max_transfers_per_round:
                break
                
            mechanism_events = self._execute_mechanism_transfers(
                mechanism=mechanism,
                bacterium_population=bacterium_population,
                spatial_positions=spatial_positions,
                environmental_factors=environmental_factors,
                generation=generation,
                remaining_transfers=max_transfers_per_round - transfer_count
            )
            
            round_events.extend(mechanism_events)
            transfer_count += len(mechanism_events)
        
        # Update resistance gene frequencies
        self._update_resistance_frequencies(bacterium_population, generation)
        
        # Store events
        self.transfer_events.extend(round_events)
        
        logger.info(
            f"Completed HGT round: {len(round_events)} successful transfers, "
            f"{transfer_count} total transfers"
        )
        
        return round_events
    
    def _execute_mechanism_transfers(
        self,
        mechanism: HGTMechanism,
        bacterium_population: Dict[str, Bacterium],
        spatial_positions: Dict[str, Coordinate],
        environmental_factors: Optional[Dict[str, float]] = None,
        generation: int = 0,
        remaining_transfers: int = 1000
    ) -> List[GeneTransferRecord]:
        """Execute transfers for a specific mechanism."""
        
        # 1. Detect proximity candidates
        candidates = self.proximity_detector.detect_hgt_candidates(
            bacterium_population, mechanism, generation
        )
        
        if not candidates:
            return []
        
        # 2. Calculate transfer probabilities
        transfer_rates = self.probability_calculator.calculate_population_transfer_rates(
            bacterium_population=bacterium_population,
            candidates=candidates,
            mechanism=mechanism,
            positions=spatial_positions,
            environmental_factors=environmental_factors,
            generation=generation
        )
        
        # 3. Execute transfers based on probabilities
        successful_events = []
        attempts = 0
        
        for donor_id, recipient_rates in transfer_rates.items():
            if attempts >= remaining_transfers:
                break
            
            donor = bacterium_population.get(donor_id)
            if not donor:
                continue
            
            for recipient_id, probability in recipient_rates.items():
                if attempts >= remaining_transfers:
                    break
                
                recipient = bacterium_population.get(recipient_id)
                if not recipient:
                    continue
                
                # Roll for transfer success
                if random.random() < probability:
                    # Attempt transfer
                    event = self._attempt_gene_transfer(
                        donor=donor,
                        recipient=recipient,
                        mechanism=mechanism,
                        probability=probability,
                        generation=generation,
                        environmental_factors=environmental_factors
                    )
                    
                    if event and event.success:
                        successful_events.append(event)
                        self.transfer_stats["successful_transfers"] += 1
                    else:
                        self.transfer_stats["failed_transfers"] += 1
                    
                    self.transfer_stats["total_attempts"] += 1
                    self.transfer_stats["by_mechanism"][mechanism] += 1
                    attempts += 1
        
        return successful_events
    
    def _attempt_gene_transfer(
        self,
        donor: Bacterium,
        recipient: Bacterium,
        mechanism: HGTMechanism,
        probability: float,
        generation: int,
        environmental_factors: Optional[Dict[str, float]] = None
    ) -> Optional[GeneTransferRecord]:
        """
        Attempt a single gene transfer between two bacteria.
        
        Returns:
            GeneTransferRecord if transfer attempted, None if invalid
        """
        # Determine what genes can be transferred
        transferable_genes = self._get_transferable_genes(donor, mechanism)
        
        if not transferable_genes:
            return GeneTransferRecord(
                id=f"transfer_{uuid.uuid4().hex[:8]}",
                donor_id=donor.id,
                recipient_id=recipient.id,
                mechanism=mechanism,
                genes_transferred=[],
                success=False,
                generation=generation,
                probability=probability,
                failure_reason="No transferable genes available"
            )
        
        # Select genes to transfer (may be subset)
        genes_to_transfer = self._select_genes_for_transfer(
            transferable_genes, mechanism, environmental_factors
        )
        
        # Check recipient barriers
        transfer_success, failure_reason = self._check_recipient_barriers(
            recipient, genes_to_transfer, mechanism
        )
        
        # Create transfer event
        event = GeneTransferRecord(
            id=f"transfer_{uuid.uuid4().hex[:8]}",
            donor_id=donor.id,
            recipient_id=recipient.id,
            mechanism=mechanism,
            genes_transferred=genes_to_transfer if transfer_success else [],
            success=transfer_success,
            generation=generation,
            probability=probability,
            failure_reason=failure_reason if not transfer_success else None
        )
        
        # If successful, apply gene transfer to recipient
        if transfer_success:
            self._apply_gene_transfer(recipient, genes_to_transfer, mechanism)
        
        return event
    
    def _get_transferable_genes(self, donor: Bacterium, mechanism: HGTMechanism) -> List[str]:
        """Determine which genes can be transferred from donor."""
        transferable = []
        
        # Base resistance genes based on donor status
        if donor.resistance_status == ResistanceStatus.RESISTANT:
            # Resistant bacteria have resistance genes to transfer
            transferable.extend(["beta_lactamase", "efflux_pump"])
        
        # Add mechanism-specific genes
        if mechanism == HGTMechanism.CONJUGATION:
            # Plasmid-mediated genes
            if hasattr(donor, 'has_conjugative_plasmid') and donor.has_conjugative_plasmid:
                transferable.extend(["drug_inactivation", "target_modification"])
        
        elif mechanism == HGTMechanism.TRANSFORMATION:
            # Environmental DNA uptake
            transferable.extend(["permeability_reduction"])
            
        elif mechanism == HGTMechanism.TRANSDUCTION:
            # Phage-mediated transfer
            if hasattr(donor, 'phage_infected') and donor.phage_infected:
                transferable.extend(["efflux_pump", "drug_inactivation"])
        
        # Add donor-specific genes if available
        if hasattr(donor, 'resistance_genes'):
            transferable.extend(donor.resistance_genes)
        
        return list(set(transferable))  # Remove duplicates
    
    def _select_genes_for_transfer(
        self, 
        available_genes: List[str], 
        mechanism: HGTMechanism,
        environmental_factors: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """Select subset of genes for transfer based on mechanism constraints."""
        
        if not available_genes:
            return []
        
        # Mechanism-specific transfer limitations
        if mechanism == HGTMechanism.CONJUGATION:
            # Conjugation can transfer multiple genes efficiently
            max_genes = min(len(available_genes), 3)
            transfer_efficiency = 0.8
            
        elif mechanism == HGTMechanism.TRANSFORMATION:
            # Transformation typically transfers fewer genes
            max_genes = min(len(available_genes), 2)
            transfer_efficiency = 0.6
            
        elif mechanism == HGTMechanism.TRANSDUCTION:
            # Transduction limited by phage capacity
            max_genes = min(len(available_genes), 2)
            transfer_efficiency = 0.5
        
        # Environmental stress can affect transfer efficiency
        if environmental_factors:
            stress_level = sum(environmental_factors.values()) / len(environmental_factors)
            transfer_efficiency *= (1.0 + stress_level * 0.2)  # Stress increases transfer
        
        # Randomly select genes based on efficiency
        selected = []
        for gene in available_genes[:max_genes]:
            if random.random() < transfer_efficiency:
                selected.append(gene)
        
        return selected
    
    def _check_recipient_barriers(
        self, 
        recipient: Bacterium, 
        genes: List[str], 
        mechanism: HGTMechanism
    ) -> Tuple[bool, Optional[str]]:
        """Check if recipient can accept the gene transfer."""
        
        if not genes:
            return False, "No genes to transfer"
        
        # Basic viability check
        if not recipient.is_alive():
            return False, "Recipient not viable"
        
        # Mechanism-specific barriers
        if mechanism == HGTMechanism.CONJUGATION:
            # Check surface compatibility
            if hasattr(recipient, 'surface_receptors') and not recipient.surface_receptors:
                return False, "Recipient lacks surface receptors"
        
        elif mechanism == HGTMechanism.TRANSFORMATION:
            # Check competence
            if hasattr(recipient, 'is_competent') and not recipient.is_competent:
                if random.random() > 0.1:  # 10% chance even if not naturally competent
                    return False, "Recipient not competent for transformation"
        
        elif mechanism == HGTMechanism.TRANSDUCTION:
            # Check phage sensitivity
            if hasattr(recipient, 'phage_resistant') and recipient.phage_resistant:
                return False, "Recipient resistant to phage"
        
        # Fitness cost consideration
        if recipient.fitness < 0.3:
            return False, "Recipient too weak to accept new genes"
        
        # Genetic compatibility
        if hasattr(recipient, 'species'):
            # Same species transfers are more likely to succeed
            compatibility_threshold = 0.9  # 90% success for same species
            if random.random() > compatibility_threshold:
                return False, "Genetic incompatibility"
        
        return True, None
    
    def _apply_gene_transfer(
        self, 
        recipient: Bacterium, 
        genes: List[str], 
        mechanism: HGTMechanism
    ) -> None:
        """Apply successful gene transfer to recipient bacterium."""
        
        # Add resistance genes to recipient
        if not hasattr(recipient, 'resistance_genes'):
            recipient.resistance_genes = []
        
        for gene in genes:
            if gene not in recipient.resistance_genes:
                recipient.resistance_genes.append(gene)
        
        # Update resistance status if resistance genes were transferred
        if any(gene in genes for gene in ["beta_lactamase", "efflux_pump", "drug_inactivation"]):
            recipient.resistance_status = ResistanceStatus.RESISTANT
        
        # Apply fitness cost for new resistance genes
        fitness_cost = len(genes) * 0.05  # 5% cost per gene
        recipient.fitness = max(0.1, recipient.fitness - fitness_cost)
        
        # Add mechanism-specific properties
        if mechanism == HGTMechanism.CONJUGATION:
            if "conjugative_plasmid" in genes:
                recipient.has_conjugative_plasmid = True
        
        elif mechanism == HGTMechanism.TRANSFORMATION:
            # Transformation may increase competence
            if hasattr(recipient, 'competence_level'):
                recipient.competence_level = min(1.0, recipient.competence_level + 0.1)
        
        elif mechanism == HGTMechanism.TRANSDUCTION:
            # May acquire phage immunity
            if "phage_immunity" in genes:
                recipient.phage_resistant = True
    
    def _update_resistance_frequencies(
        self, 
        population: Dict[str, Bacterium], 
        generation: int
    ) -> None:
        """Update population-level resistance gene frequencies."""
        
        total_bacteria = len(population)
        if total_bacteria == 0:
            return
        
        for gene_name in self.resistance_genes:
            count = 0
            for bacterium in population.values():
                if hasattr(bacterium, 'resistance_genes'):
                    if gene_name in bacterium.resistance_genes:
                        count += 1
                elif gene_name in ["beta_lactamase", "efflux_pump"] and bacterium.resistance_status == ResistanceStatus.RESISTANT:
                    count += 1
            
            frequency = count / total_bacteria
            self.resistance_genes[gene_name]["frequency"] = frequency
            if frequency > 0:
                self.resistance_genes[gene_name]["last_seen"] = generation
    
    def get_transfer_statistics(self) -> Dict[str, Union[int, float, Dict]]:
        """Get comprehensive transfer statistics."""
        
        total_attempts = self.transfer_stats["total_attempts"]
        if total_attempts == 0:
            success_rate = 0.0
        else:
            success_rate = self.transfer_stats["successful_transfers"] / total_attempts
        
        return {
            "total_transfer_attempts": total_attempts,
            "successful_transfers": self.transfer_stats["successful_transfers"],
            "failed_transfers": self.transfer_stats["failed_transfers"],
            "overall_success_rate": success_rate,
            "transfers_by_mechanism": dict(self.transfer_stats["by_mechanism"]),
            "resistance_gene_frequencies": dict(self.resistance_genes),
            "total_events_logged": len(self.transfer_events)
        }
    
    def get_recent_events(self, limit: int = 10) -> List[GeneTransferRecord]:
        """Get most recent transfer events."""
        return self.transfer_events[-limit:] if self.transfer_events else []
    
    def clear_statistics(self) -> None:
        """Clear all statistics and event logs."""
        self.transfer_events.clear()
        self.transfer_stats = {key: 0 if isinstance(value, int) else {} for key, value in self.transfer_stats.items()}
        self.resistance_genes.clear()


# ============================
# RESISTANCE GENE MODELING
# ============================

@dataclass
class ResistanceGeneState:
    """
    State information for a resistance gene in a bacterium.
    """
    gene_name: str
    is_expressed: bool = False
    expression_level: float = 0.0  # 0.0 to 1.0
    acquisition_generation: int = 0
    acquisition_method: str = "vertical"  # vertical, conjugation, transformation, transduction
    activation_threshold: float = 0.1  # antibiotic concentration threshold
    metabolic_cost: float = 0.05  # fitness cost when expressed
    regulation_state: str = "constitutive"  # constitutive, inducible, repressible
    last_activation_gen: int = 0
    
    def copy(self) -> 'ResistanceGeneState':
        """Create a copy of the gene state."""
        return ResistanceGeneState(
            gene_name=self.gene_name,
            is_expressed=self.is_expressed,
            expression_level=self.expression_level,
            acquisition_generation=self.acquisition_generation,
            acquisition_method=self.acquisition_method,
            activation_threshold=self.activation_threshold,
            metabolic_cost=self.metabolic_cost,
            regulation_state=self.regulation_state,
            last_activation_gen=self.last_activation_gen
        )


@dataclass
class EnvironmentalPressure:
    """
    Environmental pressure information affecting gene expression.
    """
    antibiotic_concentration: float = 0.0
    antibiotic_type: str = ""
    stress_level: float = 0.0
    nutrient_availability: float = 1.0
    temperature: float = 37.0  # Celsius
    ph_level: float = 7.0
    generation: int = 0
    
    def get_pressure_score(self) -> float:
        """Calculate overall environmental pressure score."""
        pressure = 0.0
        
        # Antibiotic pressure (strongest factor)
        if self.antibiotic_concentration > 0:
            pressure += min(self.antibiotic_concentration * 2.0, 1.0)
        
        # General stress
        pressure += self.stress_level * 0.5
        
        # Nutrient limitation stress
        if self.nutrient_availability < 0.5:
            pressure += (0.5 - self.nutrient_availability) * 0.3
        
        # Temperature stress
        temp_stress = abs(self.temperature - 37.0) / 10.0
        pressure += min(temp_stress * 0.2, 0.3)
        
        # pH stress
        ph_stress = abs(self.ph_level - 7.0) / 2.0
        pressure += min(ph_stress * 0.1, 0.2)
        
        return min(pressure, 1.0)


class ResistanceGeneModel:
    """
    Model for individual resistance gene behavior and expression dynamics.
    """
    
    def __init__(self):
        # Gene-specific properties
        self.gene_properties = {
            "beta_lactamase": {
                "type": "enzyme_production",
                "induction_time": 2,  # generations to fully activate
                "metabolic_cost": 0.08,
                "target_antibiotics": ["penicillin", "ampicillin", "amoxicillin"],
                "regulation": "inducible",
                "baseline_expression": 0.1,
                "max_expression": 1.0
            },
            "efflux_pump": {
                "type": "membrane_transport",
                "induction_time": 1,
                "metabolic_cost": 0.12,
                "target_antibiotics": ["tetracycline", "chloramphenicol", "fluoroquinolones"],
                "regulation": "constitutive",
                "baseline_expression": 0.3,
                "max_expression": 0.8
            },
            "target_modification": {
                "type": "protein_modification",
                "induction_time": 3,
                "metabolic_cost": 0.06,
                "target_antibiotics": ["streptomycin", "gentamicin"],
                "regulation": "inducible",
                "baseline_expression": 0.05,
                "max_expression": 0.9
            },
            "drug_inactivation": {
                "type": "enzyme_production",
                "induction_time": 2,
                "metabolic_cost": 0.10,
                "target_antibiotics": ["chloramphenicol", "aminoglycosides"],
                "regulation": "inducible",
                "baseline_expression": 0.1,
                "max_expression": 1.0
            },
            "permeability_reduction": {
                "type": "membrane_modification",
                "induction_time": 4,
                "metabolic_cost": 0.05,
                "target_antibiotics": ["beta_lactams", "quinolones"],
                "regulation": "constitutive",
                "baseline_expression": 0.2,
                "max_expression": 0.7
            }
        }
    
    def create_gene_state(
        self, 
        gene_name: str, 
        acquisition_method: str = "vertical",
        generation: int = 0
    ) -> ResistanceGeneState:
        """Create a new resistance gene state."""
        if gene_name not in self.gene_properties:
            raise ValueError(f"Unknown resistance gene: {gene_name}")
        
        props = self.gene_properties[gene_name]
        initial_expression = props["baseline_expression"] if props["regulation"] == "constitutive" else 0.0
        
        return ResistanceGeneState(
            gene_name=gene_name,
            is_expressed=initial_expression > 0,
            expression_level=initial_expression,
            acquisition_generation=generation,
            acquisition_method=acquisition_method,
            metabolic_cost=props["metabolic_cost"],
            regulation_state=props["regulation"]
        )
    
    def update_expression(
        self, 
        gene_state: ResistanceGeneState, 
        environmental_pressure: EnvironmentalPressure,
        previous_expression: float = None
    ) -> ResistanceGeneState:
        """Update gene expression based on environmental conditions."""
        props = self.gene_properties[gene_state.gene_name]
        updated_state = gene_state.copy()
        
        # Determine target expression level
        target_expression = self._calculate_target_expression(
            gene_state, environmental_pressure, props
        )
        
        # Apply temporal dynamics (genes don't instantly change expression)
        updated_state.expression_level = self._apply_expression_dynamics(
            current_level=gene_state.expression_level,
            target_level=target_expression,
            induction_time=props["induction_time"],
            previous_level=previous_expression
        )
        
        # Update expression state
        updated_state.is_expressed = updated_state.expression_level > 0.05
        
        # Track activation
        if updated_state.is_expressed and not gene_state.is_expressed:
            updated_state.last_activation_gen = environmental_pressure.generation
        
        return updated_state
    
    def _calculate_target_expression(
        self, 
        gene_state: ResistanceGeneState, 
        env_pressure: EnvironmentalPressure,
        props: Dict
    ) -> float:
        """Calculate target expression level based on conditions."""
        base_level = props["baseline_expression"]
        max_level = props["max_expression"]
        
        if props["regulation"] == "constitutive":
            # Constitutive genes maintain baseline but can increase under pressure
            pressure_score = env_pressure.get_pressure_score()
            return min(base_level + (pressure_score * 0.3), max_level)
        
        elif props["regulation"] == "inducible":
            # Inducible genes respond strongly to specific antibiotics
            if self._is_target_antibiotic_present(gene_state.gene_name, env_pressure):
                # Strong induction by target antibiotic
                induction_strength = min(env_pressure.antibiotic_concentration * 2.0, 1.0)
                return min(base_level + (induction_strength * max_level), max_level)
            else:
                # General stress can cause low-level expression
                general_pressure = env_pressure.get_pressure_score()
                if general_pressure > 0.5:
                    return min(base_level + (general_pressure * 0.2), max_level * 0.4)
                else:
                    return base_level
        
        return base_level
    
    def _is_target_antibiotic_present(self, gene_name: str, env_pressure: EnvironmentalPressure) -> bool:
        """Check if target antibiotic for this gene is present."""
        if env_pressure.antibiotic_concentration <= 0 or not env_pressure.antibiotic_type:
            return False
        
        target_antibiotics = self.gene_properties[gene_name]["target_antibiotics"]
        return any(antibiotic in env_pressure.antibiotic_type.lower() for antibiotic in target_antibiotics)
    
    def _apply_expression_dynamics(
        self, 
        current_level: float, 
        target_level: float, 
        induction_time: int,
        previous_level: float = None
    ) -> float:
        """Apply temporal dynamics to expression changes."""
        # Expression changes gradually over generations
        change_rate = 1.0 / max(induction_time, 1)
        
        if target_level > current_level:
            # Induction (activation)
            delta = (target_level - current_level) * change_rate
            new_level = current_level + delta
        else:
            # Repression (deactivation) - generally faster than induction
            fast_change_rate = change_rate * 1.5
            delta = (current_level - target_level) * fast_change_rate
            new_level = current_level - delta
        
        return max(0.0, min(1.0, new_level))
    
    def calculate_fitness_cost(self, gene_states: List[ResistanceGeneState]) -> float:
        """Calculate total fitness cost from expressed resistance genes."""
        total_cost = 0.0
        
        for gene_state in gene_states:
            if gene_state.is_expressed:
                # Cost is proportional to expression level
                gene_cost = gene_state.metabolic_cost * gene_state.expression_level
                total_cost += gene_cost
        
        # Multiple resistance genes have synergistic costs
        if len([gs for gs in gene_states if gs.is_expressed]) > 1:
            total_cost *= 1.2  # 20% penalty for multiple resistance
        
        return min(total_cost, 0.5)  # Cap total cost at 50% fitness reduction
    
    def get_resistance_strength(
        self, 
        gene_states: List[ResistanceGeneState], 
        antibiotic_type: str
    ) -> float:
        """Calculate resistance strength against specific antibiotic."""
        total_resistance = 0.0
        
        for gene_state in gene_states:
            if not gene_state.is_expressed:
                continue
            
            props = self.gene_properties[gene_state.gene_name]
            if self._is_gene_effective_against(gene_state.gene_name, antibiotic_type):
                # Resistance strength based on expression level
                gene_resistance = gene_state.expression_level * 0.8
                total_resistance += gene_resistance
        
        return min(total_resistance, 1.0)
    
    def _is_gene_effective_against(self, gene_name: str, antibiotic_type: str) -> bool:
        """Check if gene provides resistance against specific antibiotic."""
        target_antibiotics = self.gene_properties[gene_name]["target_antibiotics"]
        return any(antibiotic in antibiotic_type.lower() for antibiotic in target_antibiotics)


class GeneExpressionController:
    """
    Controller for managing gene expression dynamics across populations.
    """
    
    def __init__(self, resistance_model: ResistanceGeneModel):
        self.resistance_model = resistance_model
        self.expression_history: Dict[str, List[float]] = {}  # bacterium_id -> expression_levels
        self.environmental_history: List[EnvironmentalPressure] = []
    
    def update_population_expression(
        self,
        bacterium_population: Dict[str, Bacterium],
        environmental_pressure: EnvironmentalPressure
    ) -> Dict[str, Dict[str, ResistanceGeneState]]:
        """Update resistance gene expression for entire population."""
        updated_gene_states = {}
        
        for bacterium_id, bacterium in bacterium_population.items():
            if not hasattr(bacterium, 'resistance_genes'):
                continue
            
            bacterium_genes = {}
            previous_expression = self._get_previous_expression(bacterium_id)
            
            for gene_name in bacterium.resistance_genes:
                # Get or create gene state
                current_state = getattr(bacterium, 'gene_states', {}).get(
                    gene_name, 
                    self.resistance_model.create_gene_state(
                        gene_name, 
                        acquisition_method="vertical",
                        generation=environmental_pressure.generation
                    )
                )
                
                # Update expression
                updated_state = self.resistance_model.update_expression(
                    current_state, 
                    environmental_pressure,
                    previous_expression.get(gene_name) if previous_expression else None
                )
                
                bacterium_genes[gene_name] = updated_state
            
            updated_gene_states[bacterium_id] = bacterium_genes
            
            # Store expression history
            self._record_expression_history(bacterium_id, bacterium_genes)
        
        # Store environmental history
        self.environmental_history.append(environmental_pressure)
        if len(self.environmental_history) > 100:  # Keep last 100 generations
            self.environmental_history.pop(0)
        
        return updated_gene_states
    
    def apply_gene_states_to_population(
        self,
        bacterium_population: Dict[str, Bacterium],
        gene_states: Dict[str, Dict[str, ResistanceGeneState]]
    ) -> None:
        """Apply updated gene states back to bacterial population."""
        for bacterium_id, bacterium in bacterium_population.items():
            if bacterium_id in gene_states:
                bacterium.gene_states = gene_states[bacterium_id]
                
                # Update resistance status based on expression
                bacterium.resistance_genes = set(
                    gene_name for gene_name, state in gene_states[bacterium_id].items()
                    if state.is_expressed
                )
                
                # Calculate and apply fitness cost
                fitness_cost = self.resistance_model.calculate_fitness_cost(
                    list(gene_states[bacterium_id].values())
                )
                
                # Apply fitness penalty (store original fitness if not already stored)
                if not hasattr(bacterium, 'base_fitness'):
                    bacterium.base_fitness = bacterium.fitness
                
                bacterium.fitness = bacterium.base_fitness * (1.0 - fitness_cost)
    
    def _get_previous_expression(self, bacterium_id: str) -> Optional[Dict[str, float]]:
        """Get previous generation expression levels for bacterium."""
        if bacterium_id not in self.expression_history:
            return None
        
        history = self.expression_history[bacterium_id]
        if not history:
            return None
        
        return history[-1] if history else None
    
    def _record_expression_history(
        self, 
        bacterium_id: str, 
        gene_states: Dict[str, ResistanceGeneState]
    ) -> None:
        """Record expression levels for historical tracking."""
        if bacterium_id not in self.expression_history:
            self.expression_history[bacterium_id] = []
        
        expression_levels = {
            gene_name: state.expression_level 
            for gene_name, state in gene_states.items()
        }
        
        self.expression_history[bacterium_id].append(expression_levels)
        
        # Keep only last 50 generations per bacterium
        if len(self.expression_history[bacterium_id]) > 50:
            self.expression_history[bacterium_id].pop(0)
    
    def get_population_expression_stats(self) -> Dict[str, Union[float, Dict]]:
        """Get population-wide expression statistics."""
        if not self.expression_history:
            return {"total_bacteria": 0, "genes_tracked": 0}
        
        # Collect current expression data
        current_expression = {}
        active_bacteria = 0
        
        for bacterium_id, history in self.expression_history.items():
            if history:
                active_bacteria += 1
                latest = history[-1]
                for gene_name, expression_level in latest.items():
                    if gene_name not in current_expression:
                        current_expression[gene_name] = []
                    current_expression[gene_name].append(expression_level)
        
        # Calculate statistics
        stats = {
            "total_bacteria": active_bacteria,
            "genes_tracked": len(current_expression),
            "gene_expression_stats": {}
        }
        
        for gene_name, expression_levels in current_expression.items():
            if expression_levels:
                stats["gene_expression_stats"][gene_name] = {
                    "mean_expression": sum(expression_levels) / len(expression_levels),
                    "max_expression": max(expression_levels),
                    "expressing_count": len([e for e in expression_levels if e > 0.05]),
                    "expression_frequency": len([e for e in expression_levels if e > 0.05]) / len(expression_levels)
                }
        
        return stats


class ResistanceSpreadTracker:
    """
    Tracker for monitoring the spread of resistance genes through HGT.
    """
    
    def __init__(self):
        self.spread_events: List[Dict] = []
        self.lineage_tracking: Dict[str, Dict] = {}  # gene_id -> lineage_info
        self.generation_snapshots: List[Dict] = []
    
    def record_hgt_spread(
        self, 
        transfer_record: GeneTransferRecord,
        donor_lineage: str,
        recipient_lineage: str,
        generation: int
    ) -> None:
        """Record a resistance gene spread event via HGT."""
        spread_event = {
            "event_id": transfer_record.id,
            "generation": generation,
            "donor_id": transfer_record.donor_id,
            "recipient_id": transfer_record.recipient_id,
            "donor_lineage": donor_lineage,
            "recipient_lineage": recipient_lineage,
            "mechanism": transfer_record.mechanism.value,
            "genes_transferred": transfer_record.genes_transferred.copy(),
            "success": transfer_record.success,
            "timestamp": transfer_record.timestamp,
            "spread_type": "horizontal"
        }
        
        self.spread_events.append(spread_event)
        
        # Update lineage tracking for transferred genes
        for gene_name in transfer_record.genes_transferred:
            if transfer_record.success:
                self._update_gene_lineage(
                    gene_name, 
                    recipient_lineage, 
                    generation,
                    acquisition_method=transfer_record.mechanism.value,
                    source_lineage=donor_lineage
                )
    
    def record_vertical_inheritance(
        self,
        parent_id: str,
        offspring_id: str,
        inherited_genes: List[str],
        generation: int
    ) -> None:
        """Record vertical inheritance of resistance genes."""
        spread_event = {
            "generation": generation,
            "parent_id": parent_id,
            "offspring_id": offspring_id,
            "genes_inherited": inherited_genes,
            "spread_type": "vertical",
            "timestamp": datetime.utcnow()
        }
        
        self.spread_events.append(spread_event)
        
        # Update lineage tracking
        for gene_name in inherited_genes:
            offspring_lineage = f"lineage_{offspring_id}"
            self._update_gene_lineage(
                gene_name,
                offspring_lineage,
                generation,
                acquisition_method="vertical",
                source_lineage=f"lineage_{parent_id}"
            )
    
    def _update_gene_lineage(
        self,
        gene_name: str,
        lineage_id: str,
        generation: int,
        acquisition_method: str,
        source_lineage: str = None
    ) -> None:
        """Update lineage tracking for a gene."""
        gene_id = f"{gene_name}_{lineage_id}"
        
        if gene_id not in self.lineage_tracking:
            self.lineage_tracking[gene_id] = {
                "gene_name": gene_name,
                "lineage_id": lineage_id,
                "first_appearance": generation,
                "acquisition_method": acquisition_method,
                "source_lineage": source_lineage,
                "spread_events": [],
                "current_generation": generation
            }
        
        # Record spread event
        self.lineage_tracking[gene_id]["spread_events"].append({
            "generation": generation,
            "acquisition_method": acquisition_method,
            "source_lineage": source_lineage
        })
        
        self.lineage_tracking[gene_id]["current_generation"] = generation
    
    def take_generation_snapshot(
        self,
        bacterium_population: Dict[str, Bacterium],
        generation: int
    ) -> None:
        """Take a snapshot of resistance gene distribution."""
        snapshot = {
            "generation": generation,
            "timestamp": datetime.utcnow(),
            "total_bacteria": len(bacterium_population),
            "gene_frequencies": {},
            "resistance_patterns": {},
            "hgt_derived_resistance": {}
        }
        
        # Calculate gene frequencies
        gene_counts = {}
        total_bacteria = len(bacterium_population)
        hgt_derived_counts = {}
        
        for bacterium_id, bacterium in bacterium_population.items():
            bacterium_genes = getattr(bacterium, 'resistance_genes', set())
            gene_states = getattr(bacterium, 'gene_states', {})
            
            for gene_name in bacterium_genes:
                # Count total occurrences
                if gene_name not in gene_counts:
                    gene_counts[gene_name] = 0
                gene_counts[gene_name] += 1
                
                # Count HGT-derived occurrences
                gene_state = gene_states.get(gene_name)
                if gene_state and gene_state.acquisition_method != "vertical":
                    if gene_name not in hgt_derived_counts:
                        hgt_derived_counts[gene_name] = 0
                    hgt_derived_counts[gene_name] += 1
        
        # Calculate frequencies
        for gene_name, count in gene_counts.items():
            snapshot["gene_frequencies"][gene_name] = {
                "count": count,
                "frequency": count / total_bacteria if total_bacteria > 0 else 0,
                "hgt_derived": hgt_derived_counts.get(gene_name, 0),
                "hgt_frequency": hgt_derived_counts.get(gene_name, 0) / total_bacteria if total_bacteria > 0 else 0
            }
        
        # Analyze resistance patterns
        pattern_counts = {}
        for bacterium in bacterium_population.values():
            genes = getattr(bacterium, 'resistance_genes', set())
            pattern = tuple(sorted(genes))
            if pattern not in pattern_counts:
                pattern_counts[pattern] = 0
            pattern_counts[pattern] += 1
        
        snapshot["resistance_patterns"] = {
            str(pattern): count for pattern, count in pattern_counts.items()
        }
        
        self.generation_snapshots.append(snapshot)
        
        # Keep only last 200 snapshots
        if len(self.generation_snapshots) > 200:
            self.generation_snapshots.pop(0)
    
    def get_spread_statistics(self, recent_generations: int = 10) -> Dict:
        """Get resistance spread statistics."""
        if not self.spread_events:
            return {"total_events": 0, "hgt_events": 0, "vertical_events": 0}
        
        # Filter recent events
        if self.generation_snapshots:
            current_gen = self.generation_snapshots[-1]["generation"]
            recent_events = [
                event for event in self.spread_events
                if event.get("generation", 0) >= (current_gen - recent_generations)
            ]
        else:
            recent_events = self.spread_events[-50:]  # Last 50 events as fallback
        
        # Count by type
        hgt_events = [e for e in recent_events if e.get("spread_type") == "horizontal"]
        vertical_events = [e for e in recent_events if e.get("spread_type") == "vertical"]
        
        # Count by mechanism
        mechanism_counts = {}
        for event in hgt_events:
            mechanism = event.get("mechanism", "unknown")
            if mechanism not in mechanism_counts:
                mechanism_counts[mechanism] = 0
            mechanism_counts[mechanism] += 1
        
        # Calculate HGT effectiveness
        successful_hgt = len([e for e in hgt_events if e.get("success", False)])
        hgt_success_rate = successful_hgt / len(hgt_events) if hgt_events else 0
        
        return {
            "total_events": len(recent_events),
            "hgt_events": len(hgt_events),
            "vertical_events": len(vertical_events),
            "hgt_success_rate": hgt_success_rate,
            "mechanism_distribution": mechanism_counts,
            "recent_generations_analyzed": recent_generations,
            "lineages_tracked": len(self.lineage_tracking)
        }
    
    def get_gene_spread_history(self, gene_name: str) -> Dict:
        """Get spread history for a specific gene."""
        gene_lineages = {
            lid: linfo for lid, linfo in self.lineage_tracking.items()
            if linfo["gene_name"] == gene_name
        }
        
        if not gene_lineages:
            return {"gene_name": gene_name, "lineages": 0, "spread_events": []}
        
        # Collect all spread events for this gene
        all_events = []
        for lineage_info in gene_lineages.values():
            all_events.extend(lineage_info["spread_events"])
        
        # Sort by generation
        all_events.sort(key=lambda x: x["generation"])
        
        # Calculate spread statistics
        hgt_events = [e for e in all_events if e["acquisition_method"] != "vertical"]
        vertical_events = [e for e in all_events if e["acquisition_method"] == "vertical"]
        
        return {
            "gene_name": gene_name,
            "total_lineages": len(gene_lineages),
            "total_spread_events": len(all_events),
            "hgt_spread_events": len(hgt_events),
            "vertical_inheritance_events": len(vertical_events),
            "first_appearance": min(linfo["first_appearance"] for linfo in gene_lineages.values()),
            "spread_timeline": all_events
        }


# ============================
# POPULATION IMPACT TRACKING
# ============================

import math
from collections import defaultdict, deque
from typing import Tuple, Set
import numpy as np


@dataclass
class PopulationMetrics:
    """
    Snapshot of population metrics at a specific generation.
    """
    generation: int
    total_population: int
    resistant_count: int
    sensitive_count: int
    average_fitness: float
    fitness_variance: float
    gene_diversity_shannon: float
    gene_diversity_simpson: float
    hgt_events_this_gen: int
    new_resistance_acquisitions: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def resistance_frequency(self) -> float:
        """Calculate resistance frequency in population."""
        return self.resistant_count / self.total_population if self.total_population > 0 else 0.0
    
    @property
    def fitness_coefficient_of_variation(self) -> float:
        """Calculate coefficient of variation for fitness."""
        return (self.fitness_variance ** 0.5) / self.average_fitness if self.average_fitness > 0 else 0.0


@dataclass
class TransferNetworkNode:
    """
    Node in the HGT transfer network representing a bacterium.
    """
    bacterium_id: str
    position: Tuple[float, float]
    resistance_genes: Set[str]
    fitness: float
    generation_born: int
    transfers_sent: int = 0
    transfers_received: int = 0
    
    @property
    def transfer_activity(self) -> int:
        """Total transfer activity (sent + received)."""
        return self.transfers_sent + self.transfers_received


@dataclass
class TransferNetworkEdge:
    """
    Edge in the HGT transfer network representing a transfer event.
    """
    donor_id: str
    recipient_id: str
    mechanism: HGTMechanism
    genes_transferred: List[str]
    generation: int
    success: bool
    distance: float
    probability: float


class PopulationImpactTracker:
    """
    Comprehensive tracker for monitoring HGT impact on bacterial populations.
    """
    
    def __init__(self, history_limit: int = 500):
        self.history_limit = history_limit
        
        # Population snapshots over time
        self.population_history: deque = deque(maxlen=history_limit)
        
        # Transfer network tracking
        self.transfer_network: Dict[str, TransferNetworkNode] = {}
        self.transfer_edges: List[TransferNetworkEdge] = []
        
        # Gene spread tracking
        self.gene_spread_velocities: Dict[str, List[float]] = defaultdict(list)
        self.gene_doubling_times: Dict[str, List[float]] = defaultdict(list)
        
        # Fitness impact tracking
        self.fitness_before_hgt: Dict[str, float] = {}  # bacterium_id -> fitness
        self.fitness_after_hgt: Dict[str, float] = {}
        
        # Hotspot tracking
        self.spatial_hotspots: List[Dict] = []
        self.temporal_hotspots: List[Dict] = []
        
        # Environmental correlation data
        self.environmental_correlations: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    
    def record_population_snapshot(
        self,
        bacterium_population: Dict[str, Bacterium],
        generation: int,
        hgt_events_this_gen: int = 0
    ) -> PopulationMetrics:
        """Record a snapshot of the population state."""
        
        # Basic population counts
        total_pop = len(bacterium_population)
        resistant_count = 0
        fitness_values = []
        all_genes = []
        new_acquisitions = 0
        
        for bacterium in bacterium_population.values():
            # Count resistance status
            if getattr(bacterium, 'resistance_genes', set()):
                resistant_count += 1
            
            # Collect fitness values
            fitness_values.append(bacterium.fitness)
            
            # Collect genes for diversity calculation
            genes = getattr(bacterium, 'resistance_genes', set())
            all_genes.extend(list(genes))
            
            # Count new acquisitions this generation
            gene_states = getattr(bacterium, 'gene_states', {})
            for gene_state in gene_states.values():
                if hasattr(gene_state, 'acquisition_generation') and gene_state.acquisition_generation == generation:
                    new_acquisitions += 1
        
        # Calculate fitness statistics
        avg_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else 0.0
        fitness_variance = np.var(fitness_values) if len(fitness_values) > 1 else 0.0
        
        # Calculate gene diversity
        shannon_diversity = self._calculate_shannon_diversity(all_genes)
        simpson_diversity = self._calculate_simpson_diversity(all_genes)
        
        # Create metrics snapshot
        metrics = PopulationMetrics(
            generation=generation,
            total_population=total_pop,
            resistant_count=resistant_count,
            sensitive_count=total_pop - resistant_count,
            average_fitness=avg_fitness,
            fitness_variance=fitness_variance,
            gene_diversity_shannon=shannon_diversity,
            gene_diversity_simpson=simpson_diversity,
            hgt_events_this_gen=hgt_events_this_gen,
            new_resistance_acquisitions=new_acquisitions
        )
        
        self.population_history.append(metrics)
        
        # Update network nodes
        self._update_network_nodes(bacterium_population, generation)
        
        return metrics
    
    def record_transfer_event(
        self,
        transfer_record: GeneTransferRecord,
        donor_position: Tuple[float, float],
        recipient_position: Tuple[float, float],
        distance: float
    ) -> None:
        """Record an HGT transfer event for network analysis."""
        
        # Create network edge
        edge = TransferNetworkEdge(
            donor_id=transfer_record.donor_id,
            recipient_id=transfer_record.recipient_id,
            mechanism=transfer_record.mechanism,
            genes_transferred=transfer_record.genes_transferred.copy(),
            generation=transfer_record.generation,
            success=transfer_record.success,
            distance=distance,
            probability=transfer_record.probability
        )
        
        self.transfer_edges.append(edge)
        
        # Update node transfer counts
        if transfer_record.donor_id in self.transfer_network:
            self.transfer_network[transfer_record.donor_id].transfers_sent += 1
        
        if transfer_record.recipient_id in self.transfer_network:
            self.transfer_network[transfer_record.recipient_id].transfers_received += 1
        
        # Keep only recent edges (last 1000)
        if len(self.transfer_edges) > 1000:
            self.transfer_edges = self.transfer_edges[-1000:]
    
    def calculate_gene_spread_velocity(
        self, 
        gene_name: str, 
        time_window: int = 10
    ) -> float:
        """Calculate the velocity of gene spread through population."""
        if len(self.population_history) < 2:
            return 0.0
        
        # Get recent population snapshots
        recent_snapshots = list(self.population_history)[-time_window:]
        
        if len(recent_snapshots) < 2:
            return 0.0
        
        # Count gene occurrences over time
        frequencies = []
        generations = []
        
        for snapshot in recent_snapshots:
            # This is simplified - in real implementation would track gene frequencies
            frequency = snapshot.resistance_frequency  # Approximation
            frequencies.append(frequency)
            generations.append(snapshot.generation)
        
        # Calculate velocity as change in frequency per generation
        if len(frequencies) >= 2:
            delta_frequency = frequencies[-1] - frequencies[0]
            delta_generations = generations[-1] - generations[0]
            velocity = delta_frequency / delta_generations if delta_generations > 0 else 0.0
            
            self.gene_spread_velocities[gene_name].append(velocity)
            return velocity
        
        return 0.0
    
    def calculate_gene_doubling_time(self, gene_name: str) -> Optional[float]:
        """Calculate doubling time for gene frequency."""
        velocities = self.gene_spread_velocities.get(gene_name, [])
        
        if not velocities or velocities[-1] <= 0:
            return None
        
        # Doubling time = ln(2) / growth_rate
        growth_rate = velocities[-1]
        doubling_time = math.log(2) / growth_rate if growth_rate > 0 else float('inf')
        
        self.gene_doubling_times[gene_name].append(doubling_time)
        return doubling_time
    
    def identify_transfer_hotspots(
        self,
        spatial_radius: float = 2.0,
        temporal_window: int = 5,
        min_transfers: int = 3
    ) -> Tuple[List[Dict], List[Dict]]:
        """Identify spatial and temporal hotspots of HGT activity."""
        
        spatial_hotspots = []
        temporal_hotspots = []
        
        if not self.transfer_edges:
            return spatial_hotspots, temporal_hotspots
        
        # Spatial hotspot detection
        recent_edges = [e for e in self.transfer_edges if e.success]
        
        # Group transfers by spatial proximity
        spatial_clusters = self._cluster_transfers_spatially(recent_edges, spatial_radius)
        
        for cluster in spatial_clusters:
            if len(cluster) >= min_transfers:
                center_x = sum(e.distance for e in cluster) / len(cluster)  # Simplified
                center_y = 0  # Simplified - would calculate actual center
                
                hotspot = {
                    "type": "spatial",
                    "center": (center_x, center_y),
                    "radius": spatial_radius,
                    "transfer_count": len(cluster),
                    "mechanisms": list(set(e.mechanism.value for e in cluster)),
                    "genes_involved": list(set(gene for e in cluster for gene in e.genes_transferred))
                }
                spatial_hotspots.append(hotspot)
        
        # Temporal hotspot detection
        if len(self.population_history) >= temporal_window:
            recent_generations = list(range(
                self.population_history[-1].generation - temporal_window,
                self.population_history[-1].generation + 1
            ))
            
            for gen in recent_generations:
                gen_transfers = [e for e in recent_edges if e.generation == gen]
                
                if len(gen_transfers) >= min_transfers:
                    hotspot = {
                        "type": "temporal",
                        "generation": gen,
                        "transfer_count": len(gen_transfers),
                        "mechanisms": list(set(e.mechanism.value for e in gen_transfers)),
                        "success_rate": sum(1 for e in gen_transfers if e.success) / len(gen_transfers)
                    }
                    temporal_hotspots.append(hotspot)
        
        self.spatial_hotspots.extend(spatial_hotspots)
        self.temporal_hotspots.extend(temporal_hotspots)
        
        return spatial_hotspots, temporal_hotspots
    
    def analyze_fitness_impact(self) -> Dict[str, float]:
        """Analyze fitness impact of HGT events."""
        if not self.fitness_before_hgt or not self.fitness_after_hgt:
            return {"average_fitness_change": 0.0, "fitness_improvement_rate": 0.0}
        
        # Calculate fitness changes
        fitness_changes = []
        improvements = 0
        
        for bacterium_id in self.fitness_before_hgt:
            if bacterium_id in self.fitness_after_hgt:
                before = self.fitness_before_hgt[bacterium_id]
                after = self.fitness_after_hgt[bacterium_id]
                change = after - before
                fitness_changes.append(change)
                
                if change > 0:
                    improvements += 1
        
        avg_change = sum(fitness_changes) / len(fitness_changes) if fitness_changes else 0.0
        improvement_rate = improvements / len(fitness_changes) if fitness_changes else 0.0
        
        return {
            "average_fitness_change": avg_change,
            "fitness_improvement_rate": improvement_rate,
            "total_assessments": len(fitness_changes)
        }
    
    def get_population_diversity_trends(self, window: int = 20) -> Dict[str, List[float]]:
        """Get trends in population diversity over time."""
        if len(self.population_history) < 2:
            return {"shannon": [], "simpson": [], "generations": []}
        
        recent_history = list(self.population_history)[-window:]
        
        return {
            "shannon": [h.gene_diversity_shannon for h in recent_history],
            "simpson": [h.gene_diversity_simpson for h in recent_history],
            "generations": [h.generation for h in recent_history],
            "resistance_frequencies": [h.resistance_frequency for h in recent_history]
        }
    
    def correlate_with_environment(
        self,
        environmental_factor: str,
        factor_value: float,
        hgt_rate: float
    ) -> None:
        """Record correlation between environmental factors and HGT rates."""
        self.environmental_correlations[environmental_factor].append((factor_value, hgt_rate))
    
    def get_transfer_network_metrics(self) -> Dict[str, Union[int, float]]:
        """Calculate network-level metrics for HGT transfers."""
        if not self.transfer_network or not self.transfer_edges:
            return {"node_count": 0, "edge_count": 0}
        
        successful_edges = [e for e in self.transfer_edges if e.success]
        
        # Calculate network metrics
        node_count = len(self.transfer_network)
        edge_count = len(successful_edges)
        
        # Calculate degree distribution
        degrees = [node.transfer_activity for node in self.transfer_network.values()]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0.0
        max_degree = max(degrees) if degrees else 0
        
        # Calculate clustering coefficient (simplified)
        clustering_coefficient = self._calculate_clustering_coefficient()
        
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "average_degree": avg_degree,
            "max_degree": max_degree,
            "clustering_coefficient": clustering_coefficient,
            "network_density": edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0.0
        }
    
    def get_comprehensive_impact_report(self) -> Dict:
        """Generate a comprehensive impact analysis report."""
        if not self.population_history:
            return {"status": "insufficient_data"}
        
        # Get latest metrics
        latest = self.population_history[-1]
        
        # Calculate trends
        diversity_trends = self.get_population_diversity_trends()
        fitness_impact = self.analyze_fitness_impact()
        network_metrics = self.get_transfer_network_metrics()
        spatial_hotspots, temporal_hotspots = self.identify_transfer_hotspots()
        
        # Calculate gene spread statistics
        gene_velocities = {
            gene: velocities[-1] if velocities else 0.0
            for gene, velocities in self.gene_spread_velocities.items()
        }
        
        report = {
            "generation": latest.generation,
            "population_status": {
                "total_population": latest.total_population,
                "resistance_frequency": latest.resistance_frequency,
                "average_fitness": latest.average_fitness,
                "diversity_shannon": latest.gene_diversity_shannon,
                "diversity_simpson": latest.gene_diversity_simpson
            },
            "hgt_impact": {
                "events_this_generation": latest.hgt_events_this_gen,
                "new_acquisitions": latest.new_resistance_acquisitions,
                "fitness_impact": fitness_impact,
                "gene_spread_velocities": gene_velocities
            },
            "network_analysis": network_metrics,
            "hotspots": {
                "spatial_count": len(spatial_hotspots),
                "temporal_count": len(temporal_hotspots),
                "spatial_hotspots": spatial_hotspots[:5],  # Top 5
                "temporal_hotspots": temporal_hotspots[:5]
            },
            "trends": diversity_trends,
            "environmental_correlations": {
                factor: len(correlations) for factor, correlations in self.environmental_correlations.items()
            }
        }
        
        return report
    
    def _calculate_shannon_diversity(self, gene_list: List[str]) -> float:
        """Calculate Shannon diversity index for genes."""
        if not gene_list:
            return 0.0
        
        # Count gene frequencies
        gene_counts = defaultdict(int)
        for gene in gene_list:
            gene_counts[gene] += 1
        
        total = len(gene_list)
        shannon = 0.0
        
        for count in gene_counts.values():
            if count > 0:
                proportion = count / total
                shannon -= proportion * math.log(proportion)
        
        return shannon
    
    def _calculate_simpson_diversity(self, gene_list: List[str]) -> float:
        """Calculate Simpson diversity index for genes."""
        if not gene_list:
            return 0.0
        
        # Count gene frequencies
        gene_counts = defaultdict(int)
        for gene in gene_list:
            gene_counts[gene] += 1
        
        total = len(gene_list)
        simpson = 0.0
        
        for count in gene_counts.values():
            if count > 0:
                proportion = count / total
                simpson += proportion ** 2
        
        return 1.0 - simpson
    
    def _update_network_nodes(
        self,
        bacterium_population: Dict[str, Bacterium],
        generation: int
    ) -> None:
        """Update network nodes with current population state."""
        for bacterium_id, bacterium in bacterium_population.items():
            if bacterium_id not in self.transfer_network:
                position = (0.0, 0.0)  # Simplified - would use actual position
                if hasattr(bacterium, 'position') and bacterium.position:
                    position = (float(bacterium.position.x), float(bacterium.position.y))
                
                self.transfer_network[bacterium_id] = TransferNetworkNode(
                    bacterium_id=bacterium_id,
                    position=position,
                    resistance_genes=getattr(bacterium, 'resistance_genes', set()).copy(),
                    fitness=bacterium.fitness,
                    generation_born=getattr(bacterium, 'generation_born', generation)
                )
            else:
                # Update existing node
                node = self.transfer_network[bacterium_id]
                node.resistance_genes = getattr(bacterium, 'resistance_genes', set()).copy()
                node.fitness = bacterium.fitness
    
    def _cluster_transfers_spatially(
        self,
        transfer_edges: List[TransferNetworkEdge],
        radius: float
    ) -> List[List[TransferNetworkEdge]]:
        """Cluster transfer events by spatial proximity."""
        clusters = []
        processed = set()
        
        for edge in transfer_edges:
            if id(edge) in processed:
                continue
            
            cluster = [edge]
            processed.add(id(edge))
            
            # Find nearby transfers (simplified clustering)
            for other_edge in transfer_edges:
                if id(other_edge) in processed:
                    continue
                
                # Simplified distance check
                if abs(edge.distance - other_edge.distance) <= radius:
                    cluster.append(other_edge)
                    processed.add(id(other_edge))
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_clustering_coefficient(self) -> float:
        """Calculate network clustering coefficient."""
        if len(self.transfer_network) < 3:
            return 0.0
        
        # Simplified clustering calculation
        total_possible_triangles = 0
        actual_triangles = 0
        
        # This is a simplified version - full implementation would be more complex
        nodes_with_connections = [
            node for node in self.transfer_network.values()
            if node.transfer_activity > 0
        ]
        
        if len(nodes_with_connections) < 3:
            return 0.0
        
        return 0.1  # Placeholder - real clustering coefficient calculation needed


class HGTVisualizationEngine:
    """
    Engine for generating visualizations of HGT data and population impact.
    """
    
    def __init__(self, impact_tracker: PopulationImpactTracker):
        self.impact_tracker = impact_tracker
    
    def generate_transfer_network_data(self) -> Dict:
        """Generate data structure for transfer network visualization."""
        nodes = []
        edges = []
        
        # Convert network nodes to visualization format
        for node in self.impact_tracker.transfer_network.values():
            nodes.append({
                "id": node.bacterium_id,
                "x": node.position[0],
                "y": node.position[1],
                "resistance_genes": list(node.resistance_genes),
                "fitness": node.fitness,
                "transfer_activity": node.transfer_activity,
                "transfers_sent": node.transfers_sent,
                "transfers_received": node.transfers_received
            })
        
        # Convert edges to visualization format
        for edge in self.impact_tracker.transfer_edges:
            if edge.success:
                edges.append({
                    "source": edge.donor_id,
                    "target": edge.recipient_id,
                    "mechanism": edge.mechanism.value,
                    "genes": edge.genes_transferred,
                    "generation": edge.generation,
                    "distance": edge.distance,
                    "probability": edge.probability
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "mechanisms": list(set(e["mechanism"] for e in edges))
            }
        }
    
    def generate_population_trend_data(self, window: int = 50) -> Dict:
        """Generate data for population trend visualization."""
        if not self.impact_tracker.population_history:
            return {"generations": [], "metrics": {}}
        
        recent_history = list(self.impact_tracker.population_history)[-window:]
        
        return {
            "generations": [h.generation for h in recent_history],
            "metrics": {
                "population_size": [h.total_population for h in recent_history],
                "resistance_frequency": [h.resistance_frequency for h in recent_history],
                "average_fitness": [h.average_fitness for h in recent_history],
                "shannon_diversity": [h.gene_diversity_shannon for h in recent_history],
                "simpson_diversity": [h.gene_diversity_simpson for h in recent_history],
                "hgt_events": [h.hgt_events_this_gen for h in recent_history]
            }
        }
    
    def generate_hotspot_visualization_data(self) -> Dict:
        """Generate data for hotspot visualization."""
        spatial_hotspots, temporal_hotspots = self.impact_tracker.identify_transfer_hotspots()
        
        return {
            "spatial_hotspots": spatial_hotspots,
            "temporal_hotspots": temporal_hotspots,
            "hotspot_summary": {
                "total_spatial": len(spatial_hotspots),
                "total_temporal": len(temporal_hotspots),
                "most_active_mechanisms": self._get_most_active_mechanisms()
            }
        }
    
    def _get_most_active_mechanisms(self) -> List[str]:
        """Get the most active HGT mechanisms."""
        mechanism_counts = defaultdict(int)
        
        for edge in self.impact_tracker.transfer_edges:
            if edge.success:
                mechanism_counts[edge.mechanism.value] += 1
        
        # Sort by frequency
        sorted_mechanisms = sorted(
            mechanism_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [mechanism for mechanism, count in sorted_mechanisms]


class PopulationAnalytics:
    """
    Advanced statistical analysis tools for population impact assessment.
    """
    
    def __init__(self, impact_tracker: PopulationImpactTracker):
        self.impact_tracker = impact_tracker
    
    def calculate_hgt_efficiency_by_mechanism(self) -> Dict[str, Dict[str, float]]:
        """Calculate HGT efficiency metrics by mechanism type."""
        mechanism_stats = defaultdict(lambda: {"attempts": 0, "successes": 0, "total_probability": 0.0})
        
        for edge in self.impact_tracker.transfer_edges:
            mechanism = edge.mechanism.value
            mechanism_stats[mechanism]["attempts"] += 1
            mechanism_stats[mechanism]["total_probability"] += edge.probability
            
            if edge.success:
                mechanism_stats[mechanism]["successes"] += 1
        
        # Calculate efficiency metrics
        efficiency_results = {}
        for mechanism, stats in mechanism_stats.items():
            if stats["attempts"] > 0:
                success_rate = stats["successes"] / stats["attempts"]
                avg_probability = stats["total_probability"] / stats["attempts"]
                
                efficiency_results[mechanism] = {
                    "success_rate": success_rate,
                    "average_probability": avg_probability,
                    "total_attempts": stats["attempts"],
                    "total_successes": stats["successes"],
                    "efficiency_score": success_rate / avg_probability if avg_probability > 0 else 0.0
                }
        
        return efficiency_results
    
    def analyze_population_fitness_distribution(self) -> Dict:
        """Analyze the distribution of fitness in the population over time."""
        if not self.impact_tracker.population_history:
            return {"status": "no_data"}
        
        recent_history = list(self.impact_tracker.population_history)[-20:]
        
        fitness_data = {
            "generations": [h.generation for h in recent_history],
            "mean_fitness": [h.average_fitness for h in recent_history],
            "fitness_variance": [h.fitness_variance for h in recent_history],
            "coefficient_of_variation": [h.fitness_coefficient_of_variation for h in recent_history]
        }
        
        # Calculate fitness trends
        if len(fitness_data["mean_fitness"]) >= 2:
            fitness_trend = (fitness_data["mean_fitness"][-1] - fitness_data["mean_fitness"][0]) / len(fitness_data["mean_fitness"])
        else:
            fitness_trend = 0.0
        
        return {
            "fitness_data": fitness_data,
            "fitness_trend": fitness_trend,
            "current_mean": fitness_data["mean_fitness"][-1] if fitness_data["mean_fitness"] else 0.0,
            "current_variance": fitness_data["fitness_variance"][-1] if fitness_data["fitness_variance"] else 0.0
        }
    
    def calculate_gene_flow_efficiency(self) -> Dict[str, float]:
        """Calculate how efficiently genes flow through the population."""
        if not self.impact_tracker.transfer_edges:
            return {"overall_efficiency": 0.0}
        
        successful_transfers = [e for e in self.impact_tracker.transfer_edges if e.success]
        
        if not successful_transfers:
            return {"overall_efficiency": 0.0}
        
        # Calculate average transfer distance
        avg_distance = sum(e.distance for e in successful_transfers) / len(successful_transfers)
        
        # Calculate transfer rate per generation
        if self.impact_tracker.population_history:
            generations_span = max(1, self.impact_tracker.population_history[-1].generation - self.impact_tracker.population_history[0].generation)
            transfer_rate = len(successful_transfers) / generations_span
        else:
            transfer_rate = 0.0
        
        # Calculate gene diversity effect
        genes_transferred = set()
        for edge in successful_transfers:
            genes_transferred.update(edge.genes_transferred)
        
        gene_diversity_effect = len(genes_transferred)
        
        return {
            "overall_efficiency": transfer_rate * gene_diversity_effect / (avg_distance + 1),
            "average_transfer_distance": avg_distance,
            "transfer_rate_per_generation": transfer_rate,
            "unique_genes_transferred": gene_diversity_effect,
            "total_successful_transfers": len(successful_transfers)
        } 


# ============================
# SPATIAL GRID INTEGRATION
# ============================

from typing import Protocol, runtime_checkable
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid


@runtime_checkable
class SpatialGridInterface(Protocol):
    """Protocol for spatial grid systems."""
    
    def get_neighbors(self, position: Coordinate, radius: float) -> List[str]:
        """Get neighbors within radius of position."""
        ...
    
    def get_density(self, position: Coordinate, radius: float) -> float:
        """Get population density at position."""
        ...
    
    def update_position(self, bacterium_id: str, new_position: Coordinate) -> None:
        """Update bacterium position in grid."""
        ...
    
    def get_position(self, bacterium_id: str) -> Optional[Coordinate]:
        """Get bacterium position from grid."""
        ...


@dataclass
class SpatialHGTConfig:
    """Configuration for spatial HGT integration."""
    
    # Grid optimization settings
    enable_spatial_indexing: bool = True
    use_quadtree_optimization: bool = True
    cache_neighbor_queries: bool = True
    
    # Performance tuning
    max_concurrent_transfers: int = 50
    batch_size: int = 100
    enable_async_processing: bool = True
    
    # Spatial constraints
    enforce_physical_barriers: bool = False
    enable_density_effects: bool = True
    dynamic_proximity_scaling: bool = True
    
    # Memory management
    max_cached_queries: int = 1000
    query_cache_ttl: int = 10  # generations
    
    def get_effective_radius(self, mechanism: HGTMechanism, base_radius: float) -> float:
        """Calculate effective transfer radius based on spatial constraints."""
        if self.dynamic_proximity_scaling:
            # Scale radius based on mechanism type
            scaling_factors = {
                HGTMechanism.CONJUGATION: 0.8,    # Direct contact
                HGTMechanism.TRANSFORMATION: 1.2, # Environmental DNA
                HGTMechanism.TRANSDUCTION: 1.5    # Virus-mediated
            }
            return base_radius * scaling_factors.get(mechanism, 1.0)
        return base_radius


class SpatialHGTCache:
    """
    Cache system for spatial HGT queries to improve performance.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 10):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[Any, int]] = {}  # key -> (data, generation)
        self.access_times: Dict[str, int] = {}
    
    def get(self, key: str, current_generation: int) -> Optional[Any]:
        """Get cached data if still valid."""
        if key in self.cache:
            data, cached_generation = self.cache[key]
            if current_generation - cached_generation <= self.ttl:
                self.access_times[key] = current_generation
                return data
            else:
                # Remove expired entry
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        return None
    
    def put(self, key: str, data: Any, current_generation: int) -> None:
        """Cache data with generation timestamp."""
        # Evict old entries if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = (data, current_generation)
        self.access_times[key] = current_generation
    
    def _evict_oldest(self) -> None:
        """Evict least recently used cache entry."""
        if self.access_times:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        self.access_times.clear()


class HGTSpatialIntegration:
    """
    Integration layer between HGT system and spatial grid.
    Provides optimized spatial queries and transfer processing.
    """
    
    def __init__(
        self,
        spatial_grid: SpatialGridInterface,
        hgt_config: HGTConfig,
        spatial_config: SpatialHGTConfig,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        self.spatial_grid = spatial_grid
        self.hgt_config = hgt_config
        self.spatial_config = spatial_config
        self.executor = executor or ThreadPoolExecutor(max_workers=4)
        
        # Performance caching
        self.neighbor_cache = SpatialHGTCache(
            max_size=spatial_config.max_cached_queries,
            ttl=spatial_config.query_cache_ttl
        ) if spatial_config.cache_neighbor_queries else None
        
        # Statistics tracking
        self.query_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "spatial_queries": 0,
            "concurrent_transfers": 0
        }
    
    async def find_transfer_candidates_optimized(
        self,
        bacterium_population: Dict[str, Bacterium],
        mechanism: HGTMechanism,
        current_generation: int = 0,
        batch_size: Optional[int] = None
    ) -> Dict[str, List[str]]:
        """
        Find HGT transfer candidates using optimized spatial queries.
        """
        batch_size = batch_size or self.spatial_config.batch_size
        candidates = {}
        
        # Get all viable donors first
        viable_donors = [
            bacterium for bacterium in bacterium_population.values()
            if self._is_viable_donor(bacterium, mechanism)
        ]
        
        if not viable_donors:
            return candidates
        
        # Process in batches for memory efficiency
        if self.spatial_config.enable_async_processing and len(viable_donors) > batch_size:
            candidates = await self._process_donors_async(
                viable_donors, bacterium_population, mechanism, current_generation, batch_size
            )
        else:
            candidates = await self._process_donors_sync(
                viable_donors, bacterium_population, mechanism, current_generation
            )
        
        return candidates
    
    async def _process_donors_async(
        self,
        donors: List[Bacterium],
        population: Dict[str, Bacterium],
        mechanism: HGTMechanism,
        generation: int,
        batch_size: int
    ) -> Dict[str, List[str]]:
        """Process donors in parallel batches."""
        candidates = {}
        
        # Split into batches
        batches = [donors[i:i + batch_size] for i in range(0, len(donors), batch_size)]
        
        # Process batches concurrently
        tasks = [
            self._process_donor_batch(batch, population, mechanism, generation)
            for batch in batches
        ]
        
        batch_results = await asyncio.gather(*tasks)
        
        # Merge results
        for batch_candidates in batch_results:
            candidates.update(batch_candidates)
        
        return candidates
    
    async def _process_donor_batch(
        self,
        donors: List[Bacterium],
        population: Dict[str, Bacterium],
        mechanism: HGTMechanism,
        generation: int
    ) -> Dict[str, List[str]]:
        """Process a batch of donors."""
        batch_candidates = {}
        
        for donor in donors:
            recipients = await self._find_recipients_for_donor(
                donor, population, mechanism, generation
            )
            if recipients:
                batch_candidates[donor.id] = recipients
        
        return batch_candidates
    
    async def _process_donors_sync(
        self,
        donors: List[Bacterium],
        population: Dict[str, Bacterium],
        mechanism: HGTMechanism,
        generation: int
    ) -> Dict[str, List[str]]:
        """Process donors synchronously."""
        candidates = {}
        
        for donor in donors:
            recipients = await self._find_recipients_for_donor(
                donor, population, mechanism, generation
            )
            if recipients:
                candidates[donor.id] = recipients
        
        return candidates
    
    async def _find_recipients_for_donor(
        self,
        donor: Bacterium,
        population: Dict[str, Bacterium],
        mechanism: HGTMechanism,
        generation: int
    ) -> List[str]:
        """Find viable recipients for a specific donor."""
        if not hasattr(donor, 'position') or not donor.position:
            return []
        
        # Get effective transfer radius
        base_radius = self.hgt_config.get_distance_threshold(mechanism)
        effective_radius = self.spatial_config.get_effective_radius(mechanism, base_radius)
        
        # Check cache first
        cache_key = f"{donor.id}_{mechanism.value}_{effective_radius}_{generation}"
        
        if self.neighbor_cache:
            cached_recipients = self.neighbor_cache.get(cache_key, generation)
            if cached_recipients is not None:
                self.query_stats["cache_hits"] += 1
                return cached_recipients
            self.query_stats["cache_misses"] += 1
        
        # Perform spatial query
        self.query_stats["spatial_queries"] += 1
        
        neighbor_ids = self.spatial_grid.get_neighbors(Coordinate(x=donor.position.x, y=donor.position.y), effective_radius)
        
        # Filter for viable recipients
        recipients = []
        for neighbor_id in neighbor_ids:
            if neighbor_id in population and neighbor_id != donor.id:
                recipient = population[neighbor_id]
                if self._is_viable_recipient(recipient, donor, mechanism):
                    recipients.append(neighbor_id)
        
        # Apply density effects if enabled
        if self.spatial_config.enable_density_effects:
            recipients = self._apply_density_filtering(
                recipients, Coordinate(x=donor.position.x, y=donor.position.y), effective_radius
            )
        
        # Cache the result
        if self.neighbor_cache:
            self.neighbor_cache.put(cache_key, recipients, generation)
        
        return recipients
    
    def _apply_density_filtering(
        self,
        recipient_ids: List[str],
        donor_position: Coordinate,
        radius: float
    ) -> List[str]:
        """Apply density-based filtering to recipient list."""
        if not recipient_ids:
            return recipient_ids
        
        # Get local density
        local_density = self.spatial_grid.get_density(donor_position, radius)
        
        # Apply density-based probability filter
        if local_density > self.hgt_config.high_density_threshold:
            # Higher density increases transfer opportunities
            density_factor = min(local_density / self.hgt_config.high_density_threshold, 2.0)
            # Keep all recipients but note the density effect
            return recipient_ids
        else:
            # Lower density might reduce transfer opportunities
            density_factor = max(local_density / self.hgt_config.high_density_threshold, 0.5)
            # Randomly filter some recipients based on density
            filtered_count = max(1, int(len(recipient_ids) * density_factor))
            return random.sample(recipient_ids, min(filtered_count, len(recipient_ids)))
    
    def calculate_spatial_transfer_probability(
        self,
        donor: Bacterium,
        recipient: Bacterium,
        mechanism: HGTMechanism,
        environmental_factors: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate transfer probability with spatial considerations."""
        if not hasattr(donor, 'position') or not hasattr(recipient, 'position'):
            return 0.0
        
        if not donor.position or not recipient.position:
            return 0.0
        
        # Calculate distance
        distance = self._calculate_distance(Coordinate(x=donor.position.x, y=donor.position.y), 
                                           Coordinate(x=recipient.position.x, y=recipient.position.y))
        
        # Get base probability
        base_prob = self.hgt_config.get_base_probability(mechanism)
        
        # Apply distance decay
        max_distance = self.hgt_config.get_distance_threshold(mechanism)
        if distance > max_distance:
            return 0.0
        
        distance_factor = 1.0 - (distance / max_distance)
        
        # Apply density effects
        local_density = self.spatial_grid.get_density(Coordinate(x=donor.position.x, y=donor.position.y), max_distance)
        density_factor = self._calculate_density_factor(local_density)
        
        # Apply environmental factors
        env_factor = 1.0
        if environmental_factors:
            if 'antibiotic_concentration' in environmental_factors:
                env_factor *= (1.0 + environmental_factors['antibiotic_concentration'] * 
                              self.hgt_config.antibiotic_presence_multiplier)
        
        # Calculate final probability
        final_probability = base_prob * distance_factor * density_factor * env_factor
        
        return min(final_probability, 1.0)
    
    def get_spatial_statistics(self) -> Dict[str, Union[int, float]]:
        """Get spatial integration performance statistics."""
        total_queries = self.query_stats["cache_hits"] + self.query_stats["cache_misses"]
        cache_hit_rate = (self.query_stats["cache_hits"] / total_queries 
                         if total_queries > 0 else 0.0)
        
        return {
            **self.query_stats,
            "cache_hit_rate": cache_hit_rate,
            "total_queries": total_queries,
            "cached_entries": len(self.neighbor_cache.cache) if self.neighbor_cache else 0
        }
    
    def optimize_for_generation(self, generation: int) -> None:
        """Optimize system for a new generation."""
        # Clear expired cache entries
        if self.neighbor_cache and generation % 10 == 0:  # Every 10 generations
            # Keep only recent entries
            expired_keys = []
            for key, (_, cached_gen) in self.neighbor_cache.cache.items():
                if generation - cached_gen > self.spatial_config.query_cache_ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.neighbor_cache.cache[key]
                if key in self.neighbor_cache.access_times:
                    del self.neighbor_cache.access_times[key]
    
    def _is_viable_donor(self, bacterium: Bacterium, mechanism: HGTMechanism) -> bool:
        """Check if bacterium can be an HGT donor."""
        # Must have transferable genetic material
        if not hasattr(bacterium, 'resistance_genes') or not bacterium.resistance_genes:
            return False
        
        # Must be healthy enough to transfer
        if bacterium.fitness < 0.3:
            return False
        
        # Mechanism-specific checks
        if mechanism == HGTMechanism.CONJUGATION:
            # Requires direct contact capability
            return bacterium.fitness > 0.5
        elif mechanism == HGTMechanism.TRANSFORMATION:
            # Can release DNA when dying or stressed
            return True
        elif mechanism == HGTMechanism.TRANSDUCTION:
            # Requires virus compatibility
            return bacterium.fitness > 0.4
        
        return True
    
    def _is_viable_recipient(
        self, 
        recipient: Bacterium, 
        donor: Bacterium, 
        mechanism: HGTMechanism
    ) -> bool:
        """Check if bacterium can receive HGT."""
        # Must be alive and capable of receiving
        if recipient.fitness < 0.1:
            return False
        
        # Avoid self-transfer
        if recipient.id == donor.id:
            return False
        
        # Mechanism-specific compatibility
        if mechanism == HGTMechanism.CONJUGATION:
            # Direct contact capability
            return True
        elif mechanism == HGTMechanism.TRANSFORMATION:
            # Must be competent for DNA uptake
            return getattr(recipient, 'is_competent', True)
        elif mechanism == HGTMechanism.TRANSDUCTION:
            # Must be susceptible to phage
            return not getattr(recipient, 'phage_resistant', False)
        
        return True
    
    def _calculate_distance(self, pos1: Coordinate, pos2: Coordinate) -> float:
        """Calculate distance between two coordinates."""
        return math.sqrt((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2)
    
    def _calculate_density_factor(self, density: float) -> float:
        """Calculate density effect on transfer probability."""
        if density <= 0:
            return 0.5  # Low density reduces transfer
        elif density > self.hgt_config.high_density_threshold:
            return min(1.5, 1.0 + density / self.hgt_config.high_density_threshold * 0.3)
        else:
            return 1.0


class HGTSimulationOrchestrator:
    """
    High-level orchestrator for HGT simulation rounds.
    
    Coordinates spatial integration with gene transfer mechanics to
    run complete HGT simulation rounds with performance optimization.
    """
    
    def __init__(
        self, 
        spatial_grid: SpatialGridInterface, 
        hgt_config: HGTConfig, 
        spatial_config: SpatialHGTConfig
    ):
        self.spatial_integration = HGTSpatialIntegration(spatial_grid, hgt_config, spatial_config)
        self.gene_transfer_engine = GeneTransferEngine(hgt_config)
        self.hgt_config = hgt_config
        self.spatial_config = spatial_config
        
        # Performance tracking
        self.round_stats = {
            "total_rounds": 0,
            "total_transfers": 0,
            "average_transfers_per_round": 0.0,
            "performance_metrics": []
        }
    
    async def run_hgt_round(
        self, 
        population: Dict[str, Bacterium], 
        generation: int,
        environmental_factors: Optional[Dict[str, float]] = None
    ) -> Dict[str, Union[int, float, Dict]]:
        """
        Run a complete HGT round for a population.
        
        Args:
            population: Current bacterial population
            generation: Current generation number
            environmental_factors: Environmental conditions
        
        Returns:
            Dictionary with round statistics and results
        """
        round_start_time = datetime.utcnow()
        
        # Track round statistics
        total_transfers = 0
        successful_transfers = 0
        failed_transfers = 0
        transfers_by_mechanism = {mechanism: 0 for mechanism in HGTMechanism}
        
        # Run HGT for each mechanism
        for mechanism in HGTMechanism:
            # Find transfer candidates
            candidates = await self.spatial_integration.find_transfer_candidates(
                population, mechanism, generation
            )
            
            # Execute transfers
            for donor_id, recipient_ids in candidates.items():
                if donor_id not in population:
                    continue
                
                donor = population[donor_id]
                
                for recipient_id in recipient_ids:
                    if recipient_id not in population:
                        continue
                    
                    recipient = population[recipient_id]
                    
                    # Calculate transfer probability
                    transfer_prob = self.spatial_integration.calculate_spatial_transfer_probability(
                        donor, recipient, mechanism, environmental_factors
                    )
                    
                    # Attempt transfer
                    transfer_record = self.gene_transfer_engine.execute_gene_transfer(
                        donor, recipient, mechanism, generation, transfer_prob
                    )
                    
                    total_transfers += 1
                    transfers_by_mechanism[mechanism] += 1
                    
                    if transfer_record.success:
                        successful_transfers += 1
                    else:
                        failed_transfers += 1
        
        # Update performance statistics
        round_duration = (datetime.utcnow() - round_start_time).total_seconds()
        self.round_stats["total_rounds"] += 1
        self.round_stats["total_transfers"] += total_transfers
        self.round_stats["average_transfers_per_round"] = (
            self.round_stats["total_transfers"] / self.round_stats["total_rounds"]
        )
        
        # Get spatial statistics
        spatial_stats = self.spatial_integration.get_spatial_statistics()
        
        # Optimize for next generation
        self.optimize_for_generation(generation)
        
        return {
            "total_transfers": total_transfers,
            "successful_transfers": successful_transfers,
            "failed_transfers": failed_transfers,
            "success_rate": successful_transfers / total_transfers if total_transfers > 0 else 0.0,
            "transfers_by_mechanism": {m.value: count for m, count in transfers_by_mechanism.items()},
            "spatial_statistics": spatial_stats,
            "round_duration": round_duration,
            "generation": generation
        }
    
    def optimize_for_generation(self, generation: int) -> None:
        """Optimize performance for a generation."""
        self.spatial_integration.optimize_for_generation(generation)
    
    def get_performance_statistics(self) -> Dict[str, Union[int, float]]:
        """Get comprehensive performance statistics."""
        spatial_stats = self.spatial_integration.get_spatial_statistics()
        transfer_stats = self.gene_transfer_engine.get_transfer_statistics()
        
        return {
            **self.round_stats,
            **spatial_stats,
            "transfer_statistics": transfer_stats
        }