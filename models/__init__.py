"""
Models package for bacterial resistance evolution simulation.

This package contains all data models and business logic for:
- Bacterial populations and individual bacterium modeling
- Population dynamics and fitness calculations
- Mutation mechanisms and genetic variations
- Selection pressures and environmental factors
- Spatial grid system for Petri dish modeling
- Horizontal gene transfer mechanisms
"""

from .bacterium import Bacterium
from .population import Population
from .spatial import (
    SpatialGrid, SpatialManager, Coordinate, GridCell, 
    BoundaryCondition
)
from .hgt import (
    ProximityDetector, ProbabilityCalculator, GeneTransferEngine, HGTConfig, HGTEvent, HGTMechanism,
    GeneTransferEvent, GeneTransferRecord, ResistanceGeneState, EnvironmentalPressure, 
    ResistanceGeneModel, GeneExpressionController, ResistanceSpreadTracker,
    PopulationMetrics, TransferNetworkNode, TransferNetworkEdge, PopulationImpactTracker,
    HGTVisualizationEngine, PopulationAnalytics, SpatialGridInterface, SpatialHGTConfig,
    SpatialHGTCache, HGTSpatialIntegration, HGTSimulationOrchestrator
)

__all__ = [
    # Core bacterium and population models
    "Bacterium",
    "Population",
    
    # Spatial system
    "SpatialGrid", "SpatialManager", "Coordinate", "GridCell", 
    "BoundaryCondition",
    
    # Horizontal gene transfer
    "ProximityDetector", "ProbabilityCalculator", "GeneTransferEngine", "HGTConfig", "HGTEvent", "HGTMechanism",
    "GeneTransferEvent", "GeneTransferRecord", "ResistanceGeneState", "EnvironmentalPressure", 
    "ResistanceGeneModel", "GeneExpressionController", "ResistanceSpreadTracker",
    "PopulationMetrics", "TransferNetworkNode", "TransferNetworkEdge", "PopulationImpactTracker",
    "HGTVisualizationEngine", "PopulationAnalytics", "SpatialGridInterface", "SpatialHGTConfig",
    "SpatialHGTCache", "HGTSpatialIntegration", "HGTSimulationOrchestrator"
] 