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
    SimulationDeleteOperations,
    SimulationDatabaseError,
    get_simulation_delete_operations
)

@pytest.fixture
def delete_operations():
    """Create delete operations instance for testing"""
    return SimulationDeleteOperations()

class TestSimulationDeleteOperations:
    """Test suite for simulation database delete operations"""

    @pytest.mark.asyncio
    async def test_soft_delete_simulation_success(self, delete_operations):
        """Test successful soft delete of simulation"""
        simulation_id = "test_sim_001"
        deletion_reason = "Test cleanup"
        
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
            
            result = await delete_operations.soft_delete_simulation(
                simulation_id, deletion_reason
            )
            
            assert result is True
            mock_collection.update_one.assert_called_once()
            call_args = mock_collection.update_one.call_args
            assert call_args[0][0] == {"simulation_id": simulation_id}
            
            # Check that soft delete fields are set
            update_doc = call_args[0][1]["$set"]
            assert update_doc["metadata.status"] == SimulationStatus.CANCELLED
            assert update_doc["metadata.is_deleted"] is True
            assert "metadata.deleted_at" in update_doc
            assert update_doc["metadata.deletion_reason"] == deletion_reason

    @pytest.mark.asyncio
    async def test_soft_delete_simulation_not_found(self, delete_operations):
        """Test soft delete when simulation not found"""
        simulation_id = "nonexistent_sim"
        
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
                await delete_operations.soft_delete_simulation(simulation_id)

    @pytest.mark.asyncio
    async def test_hard_delete_simulation_success(self, delete_operations):
        """Test successful hard delete with related data"""
        simulation_id = "test_sim_001"
        
        # Mock database session and collections
        mock_db = MagicMock()
        mock_sim_collection = AsyncMock()
        mock_pop_collection = AsyncMock()
        mock_bact_collection = AsyncMock()
        
        mock_db.__getitem__.side_effect = lambda key: {
            'simulations': mock_sim_collection,
            'population_snapshots': mock_pop_collection,
            'bacteria': mock_bact_collection
        }[key]
        
        # Mock successful delete results
        mock_sim_result = MagicMock()
        mock_sim_result.deleted_count = 1
        mock_sim_collection.delete_one.return_value = mock_sim_result
        
        mock_pop_result = MagicMock()
        mock_pop_result.deleted_count = 10
        mock_pop_collection.delete_many.return_value = mock_pop_result
        
        mock_bact_result = MagicMock()
        mock_bact_result.deleted_count = 500
        mock_bact_collection.delete_many.return_value = mock_bact_result
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            result = await delete_operations.hard_delete_simulation(
                simulation_id, delete_related_data=True
            )
            
            expected_counts = {
                'simulations': 1,
                'populations': 10,
                'bacteria': 500
            }
            assert result == expected_counts
            
            # Verify all delete operations were called
            mock_sim_collection.delete_one.assert_called_once_with({"simulation_id": simulation_id})
            mock_pop_collection.delete_many.assert_called_once_with({"simulation_id": simulation_id})
            mock_bact_collection.delete_many.assert_called_once_with({"simulation_id": simulation_id})

    @pytest.mark.asyncio
    async def test_hard_delete_simulation_without_related_data(self, delete_operations):
        """Test hard delete without deleting related data"""
        simulation_id = "test_sim_001"
        
        # Mock database session and collection
        mock_db = MagicMock()
        mock_sim_collection = AsyncMock()
        mock_db.__getitem__.return_value = mock_sim_collection
        
        # Mock successful delete result
        mock_result = MagicMock()
        mock_result.deleted_count = 1
        mock_sim_collection.delete_one.return_value = mock_result
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            result = await delete_operations.hard_delete_simulation(
                simulation_id, delete_related_data=False
            )
            
            expected_counts = {'simulations': 1}
            assert result == expected_counts
            
            # Verify only simulation delete was called
            mock_sim_collection.delete_one.assert_called_once_with({"simulation_id": simulation_id})

    @pytest.mark.asyncio
    async def test_hard_delete_simulation_not_found(self, delete_operations):
        """Test hard delete when simulation not found"""
        simulation_id = "nonexistent_sim"
        
        # Mock database session and collection
        mock_db = MagicMock()
        mock_collection = AsyncMock()
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock delete result with no deletions
        mock_result = MagicMock()
        mock_result.deleted_count = 0
        mock_collection.delete_one.return_value = mock_result
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            with pytest.raises(SimulationDatabaseError, match="not found"):
                await delete_operations.hard_delete_simulation(simulation_id)

    @pytest.mark.asyncio
    async def test_delete_population_snapshots_with_generation_range(self, delete_operations):
        """Test deleting population snapshots within generation range"""
        simulation_id = "test_sim_001"
        generation_start = 5
        generation_end = 10
        
        # Mock database session and collection
        mock_db = MagicMock()
        mock_collection = AsyncMock()
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock successful delete result
        mock_result = MagicMock()
        mock_result.deleted_count = 6
        mock_collection.delete_many.return_value = mock_result
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            result = await delete_operations.delete_population_snapshots(
                simulation_id, generation_start, generation_end
            )
            
            assert result == 6
            mock_collection.delete_many.assert_called_once()
            call_args = mock_collection.delete_many.call_args
            
            # Check that generation range query is constructed correctly
            query = call_args[0][0]
            assert query["simulation_id"] == simulation_id
            assert "$gte" in query["snapshot.generation"]
            assert "$lte" in query["snapshot.generation"]
            assert query["snapshot.generation"]["$gte"] == generation_start
            assert query["snapshot.generation"]["$lte"] == generation_end

    @pytest.mark.asyncio
    async def test_delete_population_snapshots_all(self, delete_operations):
        """Test deleting all population snapshots for a simulation"""
        simulation_id = "test_sim_001"
        
        # Mock database session and collection
        mock_db = MagicMock()
        mock_collection = AsyncMock()
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock successful delete result
        mock_result = MagicMock()
        mock_result.deleted_count = 50
        mock_collection.delete_many.return_value = mock_result
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            result = await delete_operations.delete_population_snapshots(simulation_id)
            
            assert result == 50
            mock_collection.delete_many.assert_called_once()
            call_args = mock_collection.delete_many.call_args
            
            # Check that only simulation_id is in query (no generation filter)
            query = call_args[0][0]
            assert query == {"simulation_id": simulation_id}

    @pytest.mark.asyncio
    async def test_delete_bacteria_by_generation_success(self, delete_operations):
        """Test successful deletion of bacteria by generation"""
        simulation_id = "test_sim_001"
        generation = 15
        
        # Mock database session and collection
        mock_db = MagicMock()
        mock_collection = AsyncMock()
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock successful delete result
        mock_result = MagicMock()
        mock_result.deleted_count = 1000
        mock_collection.delete_many.return_value = mock_result
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            result = await delete_operations.delete_bacteria_by_generation(
                simulation_id, generation
            )
            
            assert result == 1000
            mock_collection.delete_many.assert_called_once_with({
                "simulation_id": simulation_id,
                "generation": generation
            })

    @pytest.mark.asyncio
    async def test_cleanup_old_simulations_dry_run(self, delete_operations):
        """Test cleanup old simulations in dry run mode"""
        days_old = 30
        status_filter = SimulationStatus.COMPLETED
        
        # Mock database session and collection
        mock_db = MagicMock()
        mock_collection = AsyncMock()
        mock_db.__getitem__.return_value = mock_collection
        
        # Mock count result
        mock_collection.count_documents.return_value = 5
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            result = await delete_operations.cleanup_old_simulations(
                days_old=days_old,
                status_filter=status_filter,
                dry_run=True
            )
            
            assert result == {"would_delete": 5}
            mock_collection.count_documents.assert_called_once()
            
            # Check query includes cutoff date and status filter
            call_args = mock_collection.count_documents.call_args
            query = call_args[0][0]
            assert "created_at" in query
            assert "$lt" in query["created_at"]
            assert query["metadata.status"] == status_filter.value

    @pytest.mark.asyncio
    async def test_cleanup_old_simulations_actual_deletion(self, delete_operations):
        """Test cleanup old simulations with actual deletion"""
        days_old = 30
        
        # Mock database session and collections
        mock_db = MagicMock()
        mock_sim_collection = AsyncMock()
        mock_pop_collection = AsyncMock()
        mock_bact_collection = AsyncMock()
        
        mock_db.__getitem__.side_effect = lambda key: {
            'simulations': mock_sim_collection,
            'population_snapshots': mock_pop_collection,
            'bacteria': mock_bact_collection
        }[key]
        
        # Mock find cursor for simulation IDs
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__.return_value = [
            {"simulation_id": "old_sim_1"},
            {"simulation_id": "old_sim_2"}
        ]
        mock_sim_collection.find.return_value = mock_cursor
        
        # Mock delete results
        mock_sim_delete_result = MagicMock()
        mock_sim_delete_result.deleted_count = 2
        mock_sim_collection.delete_many.return_value = mock_sim_delete_result
        
        mock_pop_delete_result = MagicMock()
        mock_pop_delete_result.deleted_count = 20
        mock_pop_collection.delete_many.return_value = mock_pop_delete_result
        
        mock_bact_delete_result = MagicMock()
        mock_bact_delete_result.deleted_count = 2000
        mock_bact_collection.delete_many.return_value = mock_bact_delete_result
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_db
            
            result = await delete_operations.cleanup_old_simulations(
                days_old=days_old,
                dry_run=False
            )
            
            expected_counts = {
                "simulations": 2,
                "populations": 20,
                "bacteria": 2000
            }
            assert result == expected_counts
            
            # Verify all cleanup operations were called
            mock_sim_collection.delete_many.assert_called_once()
            mock_pop_collection.delete_many.assert_called_once()
            mock_bact_collection.delete_many.assert_called_once()

    @pytest.mark.asyncio 
    async def test_get_simulation_delete_operations(self):
        """Test factory function for delete operations"""
        with patch('services.simulation_database.ensure_database_connection') as mock_ensure:
            mock_ensure.return_value = None
            
            operations = await get_simulation_delete_operations()
            
            assert isinstance(operations, SimulationDeleteOperations)
            mock_ensure.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__]) 