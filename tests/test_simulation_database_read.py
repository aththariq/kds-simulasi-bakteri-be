import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from bson import ObjectId

from services.simulation_database import SimulationReadOperations, SimulationDatabaseError
from models.database_models import (
    SimulationDocument, PopulationDocument, BacteriumDocument,
    SimulationMetadata, PopulationSnapshot, BacteriumData,
    SimulationParameters, SimulationStatus, SimulationResults
)

class TestSimulationReadOperations:
    """Test suite for simulation database read operations"""
    
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
            description="A test simulation for validation",
            status=SimulationStatus.COMPLETED,
            parameters=parameters,
            tags=["test", "validation"]
        )
    
    @pytest.fixture
    def sample_bacterium_data(self):
        """Create sample bacterium data for testing"""
        return BacteriumData(
            id="bact_001",
            fitness=0.85,
            resistance_genes=["ampR", "tetR"],
            position_x=10.5,
            position_y=15.2,
            generation=5,
            parent_id="bact_parent_001",
            mutation_count=2
        )
    
    @pytest.fixture
    def sample_population_snapshot(self, sample_bacterium_data):
        """Create sample population snapshot for testing"""
        return PopulationSnapshot(
            generation=10,
            total_population=1000,
            resistant_population=350,
            susceptible_population=650,
            average_fitness=0.75,
            resistance_frequency=0.35,
            bacteria=[sample_bacterium_data]
        )
    
    @pytest.fixture
    def sample_simulation_document(self, sample_simulation_metadata):
        """Create sample simulation document for testing"""
        return SimulationDocument(
            simulation_id=sample_simulation_metadata.simulation_id,
            metadata=sample_simulation_metadata,
            results=None
        )
    
    @pytest.fixture
    def read_operations(self):
        """Create SimulationReadOperations instance for testing"""
        return SimulationReadOperations()
    
    @pytest.mark.asyncio
    async def test_get_simulation_by_id_success(self, read_operations, sample_simulation_document):
        """Test successful simulation retrieval by ID"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            # Mock database collection and operations
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock successful find - simulate what MongoDB actually returns
            # MongoDB returns documents with _id field containing ObjectId
            mock_doc = sample_simulation_document.model_dump(by_alias=True)
            # Simulate MongoDB's _id field structure
            if 'id' in mock_doc:
                mock_doc['_id'] = ObjectId(mock_doc.pop('id'))
            
            mock_collection.find_one.return_value = mock_doc
            
            # Test retrieval
            result = await read_operations.get_simulation_by_id("test_sim_001")
            
            # Verify results
            assert result is not None
            assert result.simulation_id == "test_sim_001"
            assert isinstance(result, SimulationDocument)
            mock_collection.find_one.assert_called_once_with({"simulation_id": "test_sim_001"})
    
    @pytest.mark.asyncio
    async def test_get_simulation_by_id_not_found(self, read_operations):
        """Test simulation retrieval when ID doesn't exist"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock no result found
            mock_collection.find_one.return_value = None
            
            # Test retrieval
            result = await read_operations.get_simulation_by_id("nonexistent_sim")
            
            assert result is None
            mock_collection.find_one.assert_called_once_with({"simulation_id": "nonexistent_sim"})
    
    @pytest.mark.asyncio
    async def test_get_simulation_by_object_id_success(self, read_operations, sample_simulation_document):
        """Test successful simulation retrieval by ObjectId"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock successful find with ObjectId conversion
            mock_doc = sample_simulation_document.model_dump(by_alias=True)
            if 'id' in mock_doc:
                mock_doc['_id'] = ObjectId(mock_doc.pop('id'))
            
            mock_collection.find_one.return_value = mock_doc
            
            # Test retrieval by ObjectId
            test_object_id = "507f1f77bcf86cd799439011"
            result = await read_operations.get_simulation_by_object_id(test_object_id)
            
            # Verify results
            assert result is not None
            assert result.simulation_id == "test_sim_001"
            assert isinstance(result, SimulationDocument)
    
    @pytest.mark.asyncio
    async def test_get_simulation_by_object_id_invalid_format(self, read_operations):
        """Test simulation retrieval with invalid ObjectId format"""
        with pytest.raises(SimulationDatabaseError) as exc_info:
            await read_operations.get_simulation_by_object_id("invalid_object_id")
        
        assert "Invalid ObjectId format" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_list_simulations_success(self, read_operations, sample_simulation_document):
        """Test successful simulation listing"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = MagicMock()  # Use MagicMock for collection to avoid coroutine issues
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock cursor operations properly for async
            mock_doc = sample_simulation_document.model_dump(by_alias=True)
            if 'id' in mock_doc:
                mock_doc['_id'] = ObjectId(mock_doc.pop('id'))
            
            # Create proper sync mock cursor chain
            mock_cursor = MagicMock()
            mock_collection.find.return_value = mock_cursor
            mock_cursor.sort.return_value = mock_cursor
            mock_cursor.skip.return_value = mock_cursor
            mock_cursor.limit.return_value = mock_cursor
            mock_cursor.to_list = AsyncMock(return_value=[mock_doc])
            
            # Mock count_documents as async
            mock_collection.count_documents = AsyncMock(return_value=1)
            
            # Test listing
            simulations, total_count = await read_operations.list_simulations(limit=10, offset=0)
            
            # Verify results
            assert len(simulations) == 1
            assert total_count == 1
            assert simulations[0].simulation_id == "test_sim_001"
            assert isinstance(simulations[0], SimulationDocument)
    
    @pytest.mark.asyncio
    async def test_list_simulations_with_filters(self, read_operations):
        """Test simulation listing with status and tags filters"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock cursor operations
            mock_cursor = AsyncMock()
            mock_cursor.sort.return_value = mock_cursor
            mock_cursor.skip.return_value = mock_cursor
            mock_cursor.limit.return_value = mock_cursor
            mock_cursor.to_list.return_value = []
            
            mock_collection.find.return_value = mock_cursor
            mock_collection.count_documents.return_value = 0
            
            # Test listing with filters
            simulations, total_count = await read_operations.list_simulations(
                status_filter=SimulationStatus.RUNNING,
                tags_filter=["test", "experiment"],
                limit=5,
                offset=10
            )
            
            assert len(simulations) == 0
            assert total_count == 0
            
            # Verify filter was applied correctly
            expected_query = {
                "metadata.status": "running",
                "metadata.tags": {"$in": ["test", "experiment"]}
            }
            mock_collection.count_documents.assert_called_once_with(expected_query)
            mock_collection.find.assert_called_once_with(expected_query)
    
    @pytest.mark.asyncio
    async def test_get_population_history_success(self, read_operations, sample_population_snapshot):
        """Test successful population history retrieval"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock cursor operations
            mock_cursor = AsyncMock()
            mock_cursor.sort.return_value = mock_cursor
            mock_cursor.to_list.return_value = [
                PopulationDocument(
                    simulation_id="test_sim_001",
                    snapshot=sample_population_snapshot
                ).dict(by_alias=True)
            ]
            
            mock_collection.find.return_value = mock_cursor
            
            # Test population history retrieval
            snapshots = await read_operations.get_population_history(
                "test_sim_001",
                generation_start=5,
                generation_end=15
            )
            
            assert len(snapshots) == 1
            assert snapshots[0].generation == 10
            assert snapshots[0].total_population == 1000
            
            # Verify query was built correctly
            expected_query = {
                "simulation_id": "test_sim_001",
                "snapshot.generation": {"$gte": 5, "$lte": 15}
            }
            mock_collection.find.assert_called_once_with(expected_query)
    
    @pytest.mark.asyncio
    async def test_get_bacteria_by_generation_success(self, read_operations, sample_bacterium_data):
        """Test successful bacteria retrieval by generation"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock cursor operations
            mock_cursor = AsyncMock()
            mock_cursor.sort.return_value = mock_cursor
            mock_cursor.limit.return_value = mock_cursor
            mock_cursor.to_list.return_value = [
                BacteriumDocument(
                    simulation_id="test_sim_001",
                    generation=5,
                    bacterium=sample_bacterium_data
                ).dict(by_alias=True)
            ]
            
            mock_collection.find.return_value = mock_cursor
            
            # Test bacteria retrieval
            bacteria = await read_operations.get_bacteria_by_generation(
                "test_sim_001",
                generation=5,
                limit=100,
                fitness_threshold=0.8
            )
            
            assert len(bacteria) == 1
            assert bacteria[0].id == "bact_001"
            assert bacteria[0].fitness == 0.85
            
            # Verify query was built correctly
            expected_query = {
                "simulation_id": "test_sim_001",
                "generation": 5,
                "bacterium.fitness": {"$gte": 0.8}
            }
            mock_collection.find.assert_called_once_with(expected_query)
    
    @pytest.mark.asyncio
    async def test_get_simulation_statistics_success(self, read_operations, sample_simulation_document):
        """Test successful simulation statistics retrieval"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_simulations = AsyncMock()
            mock_populations = AsyncMock()
            mock_bacteria = AsyncMock()
            
            mock_db.__getitem__.side_effect = lambda key: {
                'simulations': mock_simulations,
                'population_snapshots': mock_populations,
                'bacteria': mock_bacteria
            }[key]
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock aggregation results
            mock_simulations.find_one.return_value = sample_simulation_document.dict(by_alias=True)
            
            mock_pop_cursor = AsyncMock()
            mock_pop_cursor.to_list.return_value = [{
                "_id": None,
                "total_generations": 50,
                "min_population": 800,
                "max_population": 1200,
                "avg_population": 1000,
                "final_resistance_frequency": 0.35,
                "max_resistance_frequency": 0.45
            }]
            mock_populations.aggregate.return_value = mock_pop_cursor
            
            mock_bact_cursor = AsyncMock()
            mock_bact_cursor.to_list.return_value = [{
                "_id": None,
                "total_bacteria_tracked": 50000,
                "avg_fitness": 0.75,
                "max_fitness": 0.95,
                "min_fitness": 0.45,
                "avg_mutations": 2.5
            }]
            mock_bacteria.aggregate.return_value = mock_bact_cursor
            
            # Test statistics retrieval
            stats = await read_operations.get_simulation_statistics("test_sim_001")
            
            assert stats["simulation_id"] == "test_sim_001"
            assert stats["simulation_name"] == "Test Simulation"
            assert stats["population_statistics"]["total_generations"] == 50
            assert stats["bacteria_statistics"]["total_bacteria_tracked"] == 50000
            assert stats["has_results"] is False
    
    @pytest.mark.asyncio
    async def test_search_simulations_success(self, read_operations, sample_simulation_document):
        """Test successful simulation search"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = MagicMock()  # Use MagicMock for collection to avoid coroutine issues
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock cursor operations properly for async
            mock_doc = sample_simulation_document.model_dump(by_alias=True)
            if 'id' in mock_doc:
                mock_doc['_id'] = ObjectId(mock_doc.pop('id'))
            
            # Create proper sync mock cursor chain
            mock_cursor = MagicMock()
            mock_collection.find.return_value = mock_cursor
            mock_cursor.sort.return_value = mock_cursor
            mock_cursor.skip.return_value = mock_cursor
            mock_cursor.limit.return_value = mock_cursor
            mock_cursor.to_list = AsyncMock(return_value=[mock_doc])
            
            mock_collection.count_documents = AsyncMock(return_value=1)
            
            # Test search
            simulations, total_count = await read_operations.search_simulations(
                search_query="test",
                search_fields=["metadata.name", "metadata.description"],
                limit=10,
                offset=0
            )
            
            # Verify results
            assert len(simulations) == 1
            assert total_count == 1
            assert simulations[0].simulation_id == "test_sim_001"
    
    @pytest.mark.asyncio
    async def test_get_recent_simulations_success(self, read_operations, sample_simulation_document):
        """Test successful recent simulations retrieval"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = MagicMock()  # Use MagicMock for collection to avoid coroutine issues
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock cursor operations properly for async
            mock_doc = sample_simulation_document.model_dump(by_alias=True)
            if 'id' in mock_doc:
                mock_doc['_id'] = ObjectId(mock_doc.pop('id'))
            
            # Create proper sync mock cursor chain
            mock_cursor = MagicMock()
            mock_collection.find.return_value = mock_cursor
            mock_cursor.sort.return_value = mock_cursor
            mock_cursor.limit.return_value = mock_cursor
            mock_cursor.to_list = AsyncMock(return_value=[mock_doc])
            
            # Test recent simulations retrieval
            simulations = await read_operations.get_recent_simulations(limit=5)
            
            # Verify results
            assert len(simulations) == 1
            assert simulations[0].simulation_id == "test_sim_001"
    
    @pytest.mark.asyncio
    async def test_simulation_exists_true(self, read_operations):
        """Test simulation existence check - exists"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock simulation exists
            mock_collection.find_one.return_value = {"_id": ObjectId()}
            
            # Test existence check
            exists = await read_operations.simulation_exists("test_sim_001")
            
            assert exists is True
            mock_collection.find_one.assert_called_once_with(
                {"simulation_id": "test_sim_001"}, 
                {"_id": 1}
            )
    
    @pytest.mark.asyncio
    async def test_simulation_exists_false(self, read_operations):
        """Test simulation existence check - doesn't exist"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock simulation doesn't exist
            mock_collection.find_one.return_value = None
            
            # Test existence check
            exists = await read_operations.simulation_exists("nonexistent_sim")
            
            assert exists is False
            mock_collection.find_one.assert_called_once_with(
                {"simulation_id": "nonexistent_sim"}, 
                {"_id": 1}
            )
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, read_operations):
        """Test database error handling in read operations"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock database error
            mock_collection.find_one.side_effect = Exception("Database connection failed")
            
            # Test error handling
            with pytest.raises(SimulationDatabaseError) as exc_info:
                await read_operations.get_simulation_by_id("test_sim_001")
            
            assert "Failed to retrieve simulation" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, read_operations):
        """Test validation error handling in read operations"""
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = AsyncMock()
            mock_collection = AsyncMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_session.return_value.__aenter__.return_value = mock_db
            
            # Mock invalid document data
            invalid_doc = {"simulation_id": "test", "invalid_field": "invalid_value"}
            mock_collection.find_one.return_value = invalid_doc
            
            # Test validation error handling
            with pytest.raises(SimulationDatabaseError) as exc_info:
                await read_operations.get_simulation_by_id("test_sim_001")
            
            assert "Failed to validate simulation document" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main([__file__]) 