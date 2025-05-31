import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from models.database_models import (
    SimulationDocument, 
    PopulationDocument, 
    BacteriumDocument,
    SimulationMetadata,
    SimulationParameters,
    PopulationSnapshot,
    BacteriumData,
    SimulationStatus
)
from services.simulation_database import (
    SimulationUpdateOperations,
    SimulationDatabaseError,
    get_simulation_update_operations
)

@pytest.fixture
def sample_metadata():
    """Sample simulation metadata for testing"""
    parameters = SimulationParameters(
        initial_population=1000,
        generations=50,
        mutation_rate=0.01,
        antibiotic_concentration=0.5
    )
    
    return SimulationMetadata(
        simulation_id="test_sim_001",
        name="Test Simulation",
        description="Test simulation for update operations",
        parameters=parameters,
        status=SimulationStatus.RUNNING
    )

@pytest.fixture
def sample_population_snapshot():
    """Sample population snapshot for testing"""
    bacteria = [
        BacteriumData(
            id="bact_001",
            fitness=0.8,
            resistance_genes=["ampR"],
            generation=5,
            position_x=10.0,
            position_y=15.0
        ),
        BacteriumData(
            id="bact_002", 
            fitness=0.9,
            resistance_genes=["tetR"],
            generation=5,
            position_x=20.0,
            position_y=25.0
        )
    ]
    
    return PopulationSnapshot(
        generation=5,
        total_population=1000,
        resistant_population=400,
        susceptible_population=600,
        average_fitness=0.75,
        resistance_frequency=0.4,
        bacteria=bacteria
    )

@pytest.fixture
def update_operations():
    """Create update operations instance for testing"""
    return SimulationUpdateOperations()

class TestSimulationUpdateOperations:
    """Test suite for simulation database update operations"""

    @pytest.mark.asyncio
    async def test_update_simulation_metadata_success(self, update_operations, sample_metadata):
        """Test successful metadata update"""
        simulation_id = "test_sim_001"
        metadata_updates = {
            "name": "Updated Test Simulation",
            "description": "Updated description",
            "status": SimulationStatus.COMPLETED
        }
        
        # Mock database session and collection
        mock_db = MagicMock()
        mock_collection = AsyncMock()
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock successful update result
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.modified_count = 1
        mock_collection.update_one.return_value = mock_result
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            result = await update_operations.update_simulation_metadata(
                simulation_id, metadata_updates
            )
            
            assert result is True
            mock_collection.update_one.assert_called_once()
            call_args = mock_collection.update_one.call_args
            assert call_args[0][0] == {"simulation_id": simulation_id}
            
            # Check that update document includes the metadata updates
            update_doc = call_args[0][1]["$set"]
            assert "metadata.name" in update_doc
            assert "metadata.description" in update_doc
            assert "metadata.status" in update_doc
            assert "metadata.updated_at" in update_doc

    @pytest.mark.asyncio
    async def test_update_simulation_metadata_not_found(self, update_operations):
        """Test metadata update when simulation not found"""
        simulation_id = "nonexistent_sim"
        metadata_updates = {"name": "Updated Name"}
        
        # Mock database session and collection
        mock_db = MagicMock()
        mock_collection = AsyncMock()
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock update result with no matches
        mock_result = MagicMock()
        mock_result.matched_count = 0
        mock_collection.update_one.return_value = mock_result
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            with pytest.raises(SimulationDatabaseError, match="not found"):
                await update_operations.update_simulation_metadata(
                    simulation_id, metadata_updates
                )

    @pytest.mark.asyncio
    async def test_update_simulation_parameters_success(self, update_operations):
        """Test successful parameter update"""
        simulation_id = "test_sim_001"
        parameter_updates = {
            "mutation_rate": 0.02,
            "antibiotic_concentration": 0.7,
            "selection_pressure": 1.5
        }
        
        # Mock database session and collection
        mock_db = MagicMock()
        mock_collection = AsyncMock()
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock successful update result
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.modified_count = 1
        mock_collection.update_one.return_value = mock_result
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            result = await update_operations.update_simulation_parameters(
                simulation_id, parameter_updates
            )
            
            assert result is True
            mock_collection.update_one.assert_called_once()
            call_args = mock_collection.update_one.call_args
            
            # Check that update document includes the parameter updates
            update_doc = call_args[0][1]["$set"]
            assert "metadata.parameters.mutation_rate" in update_doc
            assert "metadata.parameters.antibiotic_concentration" in update_doc
            assert "metadata.parameters.selection_pressure" in update_doc

    @pytest.mark.asyncio
    async def test_append_population_snapshot_success(self, update_operations, sample_population_snapshot):
        """Test successful population snapshot append"""
        simulation_id = "test_sim_001"
        
        # Mock database session and collections
        mock_db = MagicMock()
        mock_pop_collection = AsyncMock()
        mock_sim_collection = AsyncMock()
        mock_db.__getitem__.side_effect = lambda key: {
            'population_snapshots': mock_pop_collection,
            'simulations': mock_sim_collection
        }[key]
        
        # Mock successful insert and update results
        mock_insert_result = MagicMock()
        mock_insert_result.inserted_id = "pop_doc_id_123"
        mock_pop_collection.insert_one.return_value = mock_insert_result
        
        mock_update_result = MagicMock()
        mock_update_result.matched_count = 1
        mock_sim_collection.update_one.return_value = mock_update_result
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            result = await update_operations.append_population_snapshot(
                simulation_id, sample_population_snapshot
            )
            
            assert result == "pop_doc_id_123"
            mock_pop_collection.insert_one.assert_called_once()
            mock_sim_collection.update_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_simulation_results_success(self, update_operations):
        """Test successful simulation results update"""
        simulation_id = "test_sim_001"
        results_updates = {
            "final_statistics": {"final_population": 1200, "extinction_events": 0},
            "performance_metrics": {"avg_generation_time": 0.95}
        }
        
        # Mock database session and collection
        mock_db = MagicMock()
        mock_collection = AsyncMock()
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock successful update result
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.modified_count = 1
        mock_collection.update_one.return_value = mock_result
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            result = await update_operations.update_simulation_results(
                simulation_id, results_updates
            )
            
            assert result is True
            mock_collection.update_one.assert_called_once()
            call_args = mock_collection.update_one.call_args
            
            # Check that update document includes the results updates
            update_doc = call_args[0][1]["$set"]
            assert "results.final_statistics" in update_doc
            assert "results.performance_metrics" in update_doc

    @pytest.mark.asyncio
    async def test_add_simulation_tags_success(self, update_operations):
        """Test successful tag addition"""
        simulation_id = "test_sim_001"
        tags = ["experiment", "high_mutation", "test_run"]
        
        # Mock database session and collection
        mock_db = MagicMock()
        mock_collection = AsyncMock()
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock successful update result
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.modified_count = 1
        mock_collection.update_one.return_value = mock_result
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            result = await update_operations.add_simulation_tags(simulation_id, tags)
            
            assert result is True
            mock_collection.update_one.assert_called_once()
            call_args = mock_collection.update_one.call_args
            
            # Check that $addToSet is used to avoid duplicates
            update_doc = call_args[0][1]
            assert "$addToSet" in update_doc
            assert "metadata.tags" in update_doc["$addToSet"]

    @pytest.mark.asyncio
    async def test_remove_simulation_tags_success(self, update_operations):
        """Test successful tag removal"""
        simulation_id = "test_sim_001"
        tags = ["old_tag", "deprecated"]
        
        # Mock database session and collection
        mock_db = MagicMock()
        mock_collection = AsyncMock()
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock successful update result
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_result.modified_count = 1
        mock_collection.update_one.return_value = mock_result
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            result = await update_operations.remove_simulation_tags(simulation_id, tags)
            
            assert result is True
            mock_collection.update_one.assert_called_once()
            call_args = mock_collection.update_one.call_args
            
            # Check that $pullAll is used to remove specified tags
            update_doc = call_args[0][1]
            assert "$pullAll" in update_doc
            assert "metadata.tags" in update_doc["$pullAll"]

    @pytest.mark.asyncio 
    async def test_get_simulation_update_operations(self):
        """Test factory function for update operations"""
        with patch('services.simulation_database.ensure_database_connection') as mock_ensure:
            mock_ensure.return_value = None
            
            operations = await get_simulation_update_operations()
            
            assert isinstance(operations, SimulationUpdateOperations)
            mock_ensure.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__]) 