import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId

from services.database_service import DatabaseService
from services.database_factory import DatabaseServiceFactory, database_service_context
from models.database_models import (
    SimulationMetadata, PopulationSnapshot, BacteriumData,
    SimulationParameters, SimulationStatus, SimulationResults
)
from utils.db_connection import DatabaseManager, DatabaseOperationError
from config.database import DatabaseConfig

class TestDatabaseService:
    """Test suite for DatabaseService create operations"""
    
    @pytest.fixture
    async def mock_db_manager(self):
        """Create a mock database manager"""
        manager = AsyncMock(spec=DatabaseManager)
        
        # Mock database and collections
        mock_db = AsyncMock()
        mock_simulations = AsyncMock()
        mock_populations = AsyncMock()
        mock_bacteria = AsyncMock()
        
        mock_db.simulations = mock_simulations
        mock_db.populations = mock_populations
        mock_db.bacteria = mock_bacteria
        
        manager.get_database.return_value = mock_db
        
        return manager, mock_db, mock_simulations, mock_populations, mock_bacteria
    
    @pytest.fixture
    async def database_service(self, mock_db_manager):
        """Create a database service with mocked dependencies"""
        manager, mock_db, mock_simulations, mock_populations, mock_bacteria = mock_db_manager
        
        service = DatabaseService(manager)
        
        # Mock the initialize method to avoid actual database calls
        with patch.object(service, '_create_indexes', new_callable=AsyncMock):
            await service.initialize()
        
        # Set up the mocked collections
        service.db = mock_db
        service.simulations = mock_simulations
        service.populations = mock_populations
        service.bacteria = mock_bacteria
        
        return service, mock_simulations, mock_populations, mock_bacteria
    
    @pytest.fixture
    def sample_simulation_metadata(self):
        """Create sample simulation metadata"""
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
            description="Test simulation for unit tests",
            status=SimulationStatus.PENDING,
            parameters=parameters,
            tags=["test", "unit_test"]
        )
    
    @pytest.fixture
    def sample_population_snapshot(self):
        """Create sample population snapshot"""
        bacteria = [
            BacteriumData(
                id="bact_001",
                fitness=0.85,
                resistance_genes=["ampR"],
                position_x=10.5,
                position_y=15.2,
                generation=0,
                parent_id=None,
                mutation_count=0
            ),
            BacteriumData(
                id="bact_002",
                fitness=0.92,
                resistance_genes=["tetR", "ampR"],
                position_x=20.1,
                position_y=25.8,
                generation=0,
                parent_id=None,
                mutation_count=1
            )
        ]
        
        return PopulationSnapshot(
            generation=0,
            total_population=1000,
            resistant_population=350,
            susceptible_population=650,
            average_fitness=0.75,
            resistance_frequency=0.35,
            bacteria=bacteria
        )
    
    @pytest.mark.asyncio
    async def test_create_simulation_success(self, database_service, sample_simulation_metadata):
        """Test successful simulation creation"""
        service, mock_simulations, _, _ = database_service
        
        # Mock successful creation
        mock_simulations.find_one.return_value = None  # No existing simulation
        mock_simulations.insert_one.return_value = MagicMock(inserted_id=ObjectId())
        
        # Execute
        result = await service.create_simulation(sample_simulation_metadata)
        
        # Verify
        assert isinstance(result, str)
        mock_simulations.find_one.assert_called_once()
        mock_simulations.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_simulation_duplicate_id(self, database_service, sample_simulation_metadata):
        """Test simulation creation with duplicate ID"""
        service, mock_simulations, _, _ = database_service
        
        # Mock existing simulation
        mock_simulations.find_one.return_value = {"simulation_id": "test_sim_001"}
        
        # Execute and verify exception
        with pytest.raises(DatabaseOperationError, match="already exists"):
            await service.create_simulation(sample_simulation_metadata)
    
    @pytest.mark.asyncio
    async def test_save_population_snapshot_success(self, database_service, sample_population_snapshot):
        """Test successful population snapshot save"""
        service, mock_simulations, mock_populations, _ = database_service
        
        # Mock simulation exists
        mock_simulations.find_one.return_value = {"simulation_id": "test_sim_001"}
        mock_populations.insert_one.return_value = MagicMock(inserted_id=ObjectId())
        
        # Execute
        result = await service.save_population_snapshot("test_sim_001", sample_population_snapshot)
        
        # Verify
        assert isinstance(result, str)
        mock_populations.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_population_snapshot_invalid_simulation(self, database_service, sample_population_snapshot):
        """Test population snapshot save with invalid simulation ID"""
        service, mock_simulations, _, _ = database_service
        
        # Mock simulation doesn't exist
        mock_simulations.find_one.return_value = None
        
        # Execute and verify exception
        with pytest.raises(DatabaseOperationError, match="does not exist"):
            await service.save_population_snapshot("invalid_sim", sample_population_snapshot)
    
    @pytest.mark.asyncio
    async def test_save_bacteria_batch_success(self, database_service, sample_population_snapshot):
        """Test successful bacteria batch save"""
        service, mock_simulations, _, mock_bacteria = database_service
        
        # Mock simulation exists
        mock_simulations.find_one.return_value = {"simulation_id": "test_sim_001"}
        mock_bacteria.insert_many.return_value = MagicMock(
            inserted_ids=[ObjectId(), ObjectId()]
        )
        
        # Execute
        result = await service.save_bacteria_batch(
            "test_sim_001", 
            0, 
            sample_population_snapshot.bacteria
        )
        
        # Verify
        assert len(result) == 2
        assert all(isinstance(id_str, str) for id_str in result)
        mock_bacteria.insert_many.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_save_bacteria_batch_empty_list(self, database_service):
        """Test bacteria batch save with empty list"""
        service, _, _, _ = database_service
        
        # Execute
        result = await service.save_bacteria_batch("test_sim_001", 0, [])
        
        # Verify
        assert result == []
    
    @pytest.mark.asyncio
    async def test_save_individual_bacterium_success(self, database_service, sample_population_snapshot):
        """Test successful individual bacterium save"""
        service, mock_simulations, _, mock_bacteria = database_service
        
        # Mock simulation exists
        mock_simulations.find_one.return_value = {"simulation_id": "test_sim_001"}
        mock_bacteria.insert_one.return_value = MagicMock(inserted_id=ObjectId())
        
        # Execute
        result = await service.save_individual_bacterium(
            "test_sim_001", 
            0, 
            sample_population_snapshot.bacteria[0]
        )
        
        # Verify
        assert isinstance(result, str)
        mock_bacteria.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_simulation_status_success(self, database_service):
        """Test successful simulation status update"""
        service, mock_simulations, _, _ = database_service
        
        # Mock successful update
        mock_simulations.update_one.return_value = MagicMock(matched_count=1)
        
        # Execute
        result = await service.update_simulation_status(
            "test_sim_001", 
            SimulationStatus.RUNNING
        )
        
        # Verify
        assert result is True
        mock_simulations.update_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_simulation_status_not_found(self, database_service):
        """Test simulation status update with non-existent simulation"""
        service, mock_simulations, _, _ = database_service
        
        # Mock no match
        mock_simulations.update_one.return_value = MagicMock(matched_count=0)
        
        # Execute and verify exception
        with pytest.raises(DatabaseOperationError, match="not found"):
            await service.update_simulation_status("invalid_sim", SimulationStatus.RUNNING)
    
    @pytest.mark.asyncio
    async def test_save_simulation_results_success(self, database_service, sample_simulation_metadata):
        """Test successful simulation results save"""
        service, mock_simulations, _, _ = database_service
        
        # Mock successful update
        mock_simulations.update_one.return_value = MagicMock(matched_count=1)
        
        # Create sample results
        results = SimulationResults(
            simulation_id="test_sim_001",
            metadata=sample_simulation_metadata,
            population_history=[],
            final_statistics={"final_population": 1200},
            performance_metrics={"runtime": 45.6}
        )
        
        # Execute
        result = await service.save_simulation_results("test_sim_001", results)
        
        # Verify
        assert result is True
        mock_simulations.update_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_simulation_with_initial_data_success(
        self, database_service, sample_simulation_metadata, sample_population_snapshot
    ):
        """Test successful simulation creation with initial data"""
        service, mock_simulations, mock_populations, mock_bacteria = database_service
        
        # Mock successful operations
        mock_simulations.find_one.return_value = None  # No existing simulation
        mock_simulations.insert_one.return_value = MagicMock(inserted_id=ObjectId())
        mock_populations.insert_one.return_value = MagicMock(inserted_id=ObjectId())
        mock_bacteria.insert_many.return_value = MagicMock(
            inserted_ids=[ObjectId(), ObjectId()]
        )
        
        # Execute
        result = await service.create_simulation_with_initial_data(
            sample_simulation_metadata, 
            sample_population_snapshot
        )
        
        # Verify
        assert "simulation_id" in result
        assert "population_id" in result
        assert "bacteria_count" in result
        assert result["bacteria_count"] == 2
    
    @pytest.mark.asyncio
    async def test_get_collection_stats_success(self, database_service):
        """Test successful collection statistics retrieval"""
        service, mock_simulations, mock_populations, mock_bacteria = database_service
        
        # Mock collection counts
        mock_simulations.count_documents.return_value = 10
        mock_populations.count_documents.return_value = 500
        mock_bacteria.count_documents.return_value = 50000
        
        # Mock aggregation for status breakdown
        mock_simulations.aggregate.return_value = [
            {"_id": "completed", "count": 7},
            {"_id": "running", "count": 2},
            {"_id": "failed", "count": 1}
        ]
        
        # Execute
        stats = await service.get_collection_stats()
        
        # Verify
        assert stats["simulations"]["total_count"] == 10
        assert stats["populations"]["total_count"] == 500
        assert stats["bacteria"]["total_count"] == 50000
        assert "status_breakdown" in stats["simulations"]

class TestDatabaseServiceFactory:
    """Test suite for DatabaseServiceFactory"""
    
    @pytest.mark.asyncio
    async def test_create_service_success(self):
        """Test successful service creation"""
        with patch('services.database_factory.DatabaseManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            
            with patch('services.database_factory.DatabaseService') as mock_service_class:
                mock_service = AsyncMock()
                mock_service_class.return_value = mock_service
                
                # Execute
                result = await DatabaseServiceFactory.create_service()
                
                # Verify
                assert result == mock_service
                mock_service.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_database_service_context(self):
        """Test database service context manager"""
        with patch('services.database_factory.DatabaseManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            
            with patch('services.database_factory.DatabaseService') as mock_service_class:
                mock_service = AsyncMock()
                mock_service_class.return_value = mock_service
                
                # Execute
                async with database_service_context() as service:
                    assert service == mock_service
                
                # Verify cleanup
                mock_manager.close.assert_called_once()

class TestDatabaseIntegration:
    """Integration tests for database operations"""
    
    @pytest.mark.asyncio
    async def test_full_simulation_workflow(self):
        """Test complete simulation workflow with mocked database"""
        # This would be an integration test with a real test database
        # For now, we'll use mocks to simulate the workflow
        
        with patch('services.database_factory.DatabaseManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            
            # Mock database operations
            mock_db = AsyncMock()
            mock_simulations = AsyncMock()
            mock_populations = AsyncMock()
            mock_bacteria = AsyncMock()
            
            mock_db.simulations = mock_simulations
            mock_db.populations = mock_populations
            mock_db.bacteria = mock_bacteria
            mock_manager.get_database.return_value = mock_db
            
            # Mock successful operations
            mock_simulations.find_one.return_value = None
            mock_simulations.insert_one.return_value = MagicMock(inserted_id=ObjectId())
            mock_simulations.update_one.return_value = MagicMock(matched_count=1)
            mock_populations.insert_one.return_value = MagicMock(inserted_id=ObjectId())
            mock_bacteria.insert_many.return_value = MagicMock(inserted_ids=[ObjectId()])
            
            async with database_service_context() as service:
                # Create simulation
                metadata = SimulationMetadata(
                    simulation_id="integration_test_001",
                    name="Integration Test",
                    description="Full workflow test",
                    status=SimulationStatus.PENDING,
                    parameters=SimulationParameters(
                        initial_population=100,
                        generations=10,
                        mutation_rate=0.01,
                        antibiotic_concentration=0.5
                    )
                )
                
                sim_id = await service.create_simulation(metadata)
                assert isinstance(sim_id, str)
                
                # Update status to running
                await service.update_simulation_status(
                    "integration_test_001", 
                    SimulationStatus.RUNNING
                )
                
                # Save population snapshot
                snapshot = PopulationSnapshot(
                    generation=0,
                    total_population=100,
                    resistant_population=10,
                    susceptible_population=90,
                    average_fitness=0.8,
                    resistance_frequency=0.1,
                    bacteria=[
                        BacteriumData(
                            id="test_bact_001",
                            fitness=0.8,
                            resistance_genes=[],
                            generation=0
                        )
                    ]
                )
                
                pop_id = await service.save_population_snapshot(
                    "integration_test_001", 
                    snapshot
                )
                assert isinstance(pop_id, str)
                
                # Save bacteria batch
                bacteria_ids = await service.save_bacteria_batch(
                    "integration_test_001",
                    0,
                    snapshot.bacteria
                )
                assert len(bacteria_ids) == 1
                
                # Complete simulation
                results = SimulationResults(
                    simulation_id="integration_test_001",
                    metadata=metadata,
                    population_history=[snapshot],
                    final_statistics={"final_population": 120},
                    performance_metrics={"runtime": 5.2}
                )
                
                success = await service.save_simulation_results(
                    "integration_test_001", 
                    results
                )
                assert success is True

if __name__ == "__main__":
    pytest.main([__file__]) 