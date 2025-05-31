import pytest
import asyncio
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, patch

from services.simulation_database import (
    SimulationCreateOperations,
    SimulationDatabaseError,
    get_simulation_create_operations
)
from models.database_models import (
    SimulationMetadata,
    SimulationParameters,
    SimulationResults,
    PopulationSnapshot,
    BacteriumData,
    SimulationStatus
)
from utils.db_connection import DatabaseSession

class TestSimulationCreateOperations:
    """Test suite for simulation database create operations"""
    
    @pytest.fixture
    def sample_simulation_metadata(self):
        """Create sample simulation metadata for testing"""
        parameters = SimulationParameters(
            initial_population=1000,
            generations=50,
            mutation_rate=0.01,
            antibiotic_concentration=0.5,
            selection_pressure=1.0,
            spatial_enabled=True,
            grid_size=100,
            hgt_enabled=True,
            hgt_rate=0.001
        )
        
        return SimulationMetadata(
            simulation_id="test_sim_001",
            name="Test Simulation",
            description="A test simulation for unit testing",
            status=SimulationStatus.PENDING,
            parameters=parameters,
            tags=["test", "unit_test"]
        )
    
    @pytest.fixture
    def sample_bacteria_data(self):
        """Create sample bacteria data for testing"""
        return [
            BacteriumData(
                id="bact_001",
                fitness=0.85,
                resistance_genes=["ampR", "tetR"],
                position_x=10.5,
                position_y=15.2,
                generation=0,
                parent_id=None,
                mutation_count=0
            ),
            BacteriumData(
                id="bact_002",
                fitness=0.92,
                resistance_genes=["ampR"],
                position_x=20.1,
                position_y=25.8,
                generation=0,
                parent_id=None,
                mutation_count=0
            )
        ]
    
    @pytest.fixture
    def sample_population_snapshot(self, sample_bacteria_data):
        """Create sample population snapshot for testing"""
        return PopulationSnapshot(
            generation=0,
            total_population=1000,
            resistant_population=350,
            susceptible_population=650,
            average_fitness=0.75,
            resistance_frequency=0.35,
            bacteria=sample_bacteria_data
        )
    
    @pytest.fixture
    def sample_simulation_results(self, sample_population_snapshot):
        """Create sample simulation results for testing"""
        return SimulationResults(
            simulation_id="test_sim_001",
            metadata=self.sample_simulation_metadata(),
            population_history=[sample_population_snapshot],
            final_statistics={
                "final_population": 1200,
                "final_resistance_frequency": 0.45,
                "generations_to_resistance": 15,
                "extinction_events": 0
            },
            performance_metrics={
                "avg_generation_time": 0.85,
                "memory_usage_mb": 125.5,
                "cpu_time_seconds": 42.3
            }
        )
    
    @pytest.fixture
    def create_operations(self):
        """Create SimulationCreateOperations instance for testing"""
        return SimulationCreateOperations()
    
    @pytest.mark.asyncio
    async def test_create_simulation_success(self, create_operations, sample_simulation_metadata):
        """Test successful simulation creation"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            # Mock database collection and operations
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock successful insertion
            mock_result = AsyncMock()
            mock_result.inserted_id = "507f1f77bcf86cd799439011"
            mock_collection.insert_one.return_value = mock_result
            
            # Test creation
            result = await create_operations.create_simulation(sample_simulation_metadata)
            
            assert result == "507f1f77bcf86cd799439011"
            mock_collection.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_simulation_duplicate_error(self, create_operations, sample_simulation_metadata):
        """Test simulation creation with duplicate ID"""
        from pymongo.errors import DuplicateKeyError
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock duplicate key error
            mock_collection.insert_one.side_effect = DuplicateKeyError("Duplicate key error")
            
            # Test duplicate error
            with pytest.raises(SimulationDatabaseError) as exc_info:
                await create_operations.create_simulation(sample_simulation_metadata)
            
            assert "already exists" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_save_simulation_results_success(self, create_operations, sample_simulation_results):
        """Test successful simulation results saving"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock successful update
            mock_result = AsyncMock()
            mock_result.matched_count = 1
            mock_result.modified_count = 1
            mock_collection.update_one.return_value = mock_result
            
            # Test saving results
            result = await create_operations.save_simulation_results(
                "test_sim_001", 
                sample_simulation_results
            )
            
            assert result is True
            mock_collection.update_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_simulation_results_not_found(self, create_operations, sample_simulation_results):
        """Test simulation results saving when simulation not found"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock no match found
            mock_result = AsyncMock()
            mock_result.matched_count = 0
            mock_collection.update_one.return_value = mock_result
            
            # Test not found error
            with pytest.raises(SimulationDatabaseError) as exc_info:
                await create_operations.save_simulation_results(
                    "nonexistent_sim", 
                    sample_simulation_results
                )
            
            assert "not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_batch_insert_population_snapshots_success(
        self, 
        create_operations, 
        sample_population_snapshot
    ):
        """Test successful batch insertion of population snapshots"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock successful batch insertion
            mock_result = AsyncMock()
            mock_result.inserted_ids = ["id1", "id2"]
            mock_collection.insert_many.return_value = mock_result
            
            # Test batch insertion
            result = await create_operations.batch_insert_population_snapshots(
                "test_sim_001",
                [sample_population_snapshot, sample_population_snapshot]
            )
            
            assert len(result) == 2
            assert result == ["id1", "id2"]
            mock_collection.insert_many.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_insert_population_snapshots_empty(self, create_operations):
        """Test batch insertion with empty snapshots list"""
        result = await create_operations.batch_insert_population_snapshots(
            "test_sim_001",
            []
        )
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_batch_insert_bacteria_success(self, create_operations, sample_bacteria_data):
        """Test successful batch insertion of bacteria data"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock successful batch insertion
            mock_result = AsyncMock()
            mock_result.inserted_ids = ["bact_id1", "bact_id2"]
            mock_collection.insert_many.return_value = mock_result
            
            # Test batch insertion
            result = await create_operations.batch_insert_bacteria(
                "test_sim_001",
                0,
                sample_bacteria_data
            )
            
            assert len(result) == 2
            assert result == ["bact_id1", "bact_id2"]
            mock_collection.insert_many.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_insert_bacteria_empty(self, create_operations):
        """Test batch insertion with empty bacteria list"""
        result = await create_operations.batch_insert_bacteria(
            "test_sim_001",
            0,
            []
        )
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_update_simulation_status_success(self, create_operations):
        """Test successful simulation status update"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock successful update
            mock_result = AsyncMock()
            mock_result.matched_count = 1
            mock_collection.update_one.return_value = mock_result
            
            # Test status update
            result = await create_operations.update_simulation_status(
                "test_sim_001",
                SimulationStatus.COMPLETED,
                runtime=45.6
            )
            
            assert result is True
            mock_collection.update_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_simulation_status_with_error(self, create_operations):
        """Test simulation status update with error message"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock successful update
            mock_result = AsyncMock()
            mock_result.matched_count = 1
            mock_collection.update_one.return_value = mock_result
            
            # Test status update with error
            result = await create_operations.update_simulation_status(
                "test_sim_001",
                SimulationStatus.FAILED,
                error_message="Test error message"
            )
            
            assert result is True
            
            # Verify update document includes error message
            call_args = mock_collection.update_one.call_args
            update_doc = call_args[0][1]['$set']
            assert 'metadata.error_message' in update_doc
            assert update_doc['metadata.error_message'] == "Test error message"
    
    @pytest.mark.asyncio
    async def test_create_simulation_with_initial_data_success(
        self, 
        create_operations, 
        sample_simulation_metadata,
        sample_population_snapshot
    ):
        """Test successful creation of simulation with initial data"""
        with patch.object(create_operations, 'create_simulation') as mock_create, \
             patch.object(create_operations, 'batch_insert_population_snapshots') as mock_insert_pop, \
             patch.object(create_operations, 'batch_insert_bacteria') as mock_insert_bact:
            
            # Mock successful operations
            mock_create.return_value = "sim_object_id"
            mock_insert_pop.return_value = ["pop_id"]
            mock_insert_bact.return_value = ["bact_id1", "bact_id2"]
            
            # Test creation with initial data
            result = await create_operations.create_simulation_with_initial_data(
                sample_simulation_metadata,
                sample_population_snapshot
            )
            
            assert result == "sim_object_id"
            mock_create.assert_called_once_with(sample_simulation_metadata)
            mock_insert_pop.assert_called_once()
            mock_insert_bact.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_simulation_with_initial_data_no_bacteria(
        self, 
        create_operations, 
        sample_simulation_metadata,
        sample_population_snapshot
    ):
        """Test creation with initial data but no bacteria"""
        # Remove bacteria from snapshot
        sample_population_snapshot.bacteria = []
        
        with patch.object(create_operations, 'create_simulation') as mock_create, \
             patch.object(create_operations, 'batch_insert_population_snapshots') as mock_insert_pop, \
             patch.object(create_operations, 'batch_insert_bacteria') as mock_insert_bact:
            
            # Mock successful operations
            mock_create.return_value = "sim_object_id"
            mock_insert_pop.return_value = ["pop_id"]
            
            # Test creation with initial data but no bacteria
            result = await create_operations.create_simulation_with_initial_data(
                sample_simulation_metadata,
                sample_population_snapshot
            )
            
            assert result == "sim_object_id"
            mock_create.assert_called_once_with(sample_simulation_metadata)
            mock_insert_pop.assert_called_once()
            # Bacteria insertion should not be called when no bacteria data
            mock_insert_bact.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_simulation_create_operations(self):
        """Test getting global create operations instance"""
        ops = await get_simulation_create_operations()
        assert isinstance(ops, SimulationCreateOperations)
        
        # Should return the same instance
        ops2 = await get_simulation_create_operations()
        assert ops is ops2

if __name__ == "__main__":
    pytest.main([__file__]) 