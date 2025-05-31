import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from models.database_models import SimulationStatus
from services.simulation_database_optimization import (
    SimulationDatabaseOptimizer,
    DatabaseOptimizationError,
    get_database_optimizer,
    performance_monitor
)

@pytest.fixture
def optimizer():
    """Create database optimizer instance for testing"""
    return SimulationDatabaseOptimizer()

@pytest.fixture
def mock_collection():
    """Mock MongoDB collection"""
    collection = AsyncMock()
    collection.create_index = AsyncMock()
    collection.list_indexes = AsyncMock()
    collection.find = AsyncMock()
    collection.count_documents = AsyncMock()
    return collection

@pytest.fixture
def mock_database():
    """Mock MongoDB database"""
    db = MagicMock()
    db.command = AsyncMock()
    return db

class TestSimulationDatabaseOptimizer:
    """Test suite for database optimization functionality"""

    @pytest.mark.asyncio
    async def test_create_indexes_success(self, optimizer, mock_database):
        """Test successful index creation"""
        # Mock collections
        mock_sim_collection = AsyncMock()
        mock_pop_collection = AsyncMock()
        mock_bact_collection = AsyncMock()
        
        mock_database.__getitem__.side_effect = lambda key: {
            'simulations': mock_sim_collection,
            'population_snapshots': mock_pop_collection,
            'bacteria': mock_bact_collection
        }[key]
        
        # Mock successful index creation
        mock_sim_collection.create_index.return_value = "simulation_id_1"
        mock_pop_collection.create_index.return_value = "simulation_id_1_generation_1"
        mock_bact_collection.create_index.return_value = "simulation_id_1_generation_1"
        
        with patch('services.simulation_database_optimization.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_database
            
            result = await optimizer.create_indexes()
            
            assert isinstance(result, dict)
            assert 'simulations' in result
            assert 'populations' in result
            assert 'bacteria' in result
            
            # Verify index creation was called
            assert mock_sim_collection.create_index.called
            assert mock_pop_collection.create_index.called
            assert mock_bact_collection.create_index.called

    @pytest.mark.asyncio
    async def test_create_indexes_existing_index(self, optimizer, mock_database):
        """Test index creation when index already exists"""
        from pymongo.errors import OperationFailure
        
        mock_collection = AsyncMock()
        mock_database.__getitem__.return_value = mock_collection
        
        # Mock index already exists error
        mock_collection.create_index.side_effect = OperationFailure("already exists")
        
        with patch('services.simulation_database_optimization.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_database
            
            # Should not raise exception for existing indexes
            result = await optimizer.create_indexes()
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_list_indexes_success(self, optimizer, mock_database):
        """Test successful index listing"""
        mock_collections = {
            'simulations': AsyncMock(),
            'population_snapshots': AsyncMock(),
            'bacteria': AsyncMock()
        }
        
        # Mock index listings
        for collection_name, collection in mock_collections.items():
            collection.list_indexes.return_value.to_list.return_value = [
                {'name': f'{collection_name}_index_1'},
                {'name': f'{collection_name}_index_2'}
            ]
        
        mock_database.__getitem__.side_effect = lambda key: mock_collections[key]
        
        with patch('services.simulation_database_optimization.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_database
            
            result = await optimizer.list_indexes()
            
            assert isinstance(result, dict)
            assert len(result) == 3
            for collection_name in ['simulations', 'populations', 'bacteria']:
                assert collection_name in result
                assert len(result[collection_name]) == 2

    @pytest.mark.asyncio
    async def test_analyze_query_performance_with_explain(self, optimizer, mock_database):
        """Test query performance analysis with explain plan"""
        collection_name = 'simulations'
        query = {"metadata.status": SimulationStatus.COMPLETED}
        
        mock_collection = AsyncMock()
        mock_database.__getitem__.return_value = mock_collection
        
        # Mock explain result
        mock_cursor = AsyncMock()
        mock_collection.find.return_value = mock_cursor
        
        explain_result = {
            'executionStats': {
                'executionTimeMillis': 5,
                'totalDocsExamined': 100,
                'totalDocsReturned': 10,
                'indexUsed': True
            },
            'queryPlanner': {
                'winningPlan': {'stage': 'IXSCAN', 'indexName': 'status_1'}
            }
        }
        mock_cursor.explain.return_value = explain_result
        
        with patch('services.simulation_database_optimization.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_database
            
            result = await optimizer.analyze_query_performance(collection_name, query, explain=True)
            
            assert result['collection'] == collection_name
            assert result['query'] == query
            assert result['execution_time_ms'] == 5
            assert result['documents_examined'] == 100
            assert result['documents_returned'] == 10
            assert result['index_used'] is True

    @pytest.mark.asyncio
    async def test_analyze_query_performance_without_explain(self, optimizer, mock_database):
        """Test query performance analysis without explain plan"""
        collection_name = 'simulations'
        query = {"metadata.status": SimulationStatus.COMPLETED}
        
        mock_collection = AsyncMock()
        mock_database.__getitem__.return_value = mock_collection
        
        # Mock count result
        mock_collection.count_documents.return_value = 25
        
        with patch('services.simulation_database_optimization.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_database
            
            with patch('time.time', side_effect=[0.0, 0.005]):  # 5ms execution time
                result = await optimizer.analyze_query_performance(collection_name, query, explain=False)
            
            assert result['collection'] == collection_name
            assert result['query'] == query
            assert result['execution_time_ms'] == 5.0
            assert result['document_count'] == 25

    @pytest.mark.asyncio
    async def test_get_collection_statistics_success(self, optimizer, mock_database):
        """Test successful collection statistics retrieval"""
        mock_collections = {
            'simulations': AsyncMock(),
            'population_snapshots': AsyncMock(),
            'bacteria': AsyncMock()
        }
        
        # Mock collection stats
        collection_stats = {
            'storageSize': 1024,
            'indexSizes': {'_id_': 512, 'simulation_id_1': 256},
            'avgObjSize': 128,
            'size': 2048
        }
        
        mock_database.command.return_value = collection_stats
        
        # Mock document counts and indexes
        for collection in mock_collections.values():
            collection.count_documents.return_value = 100
            collection.list_indexes.return_value.to_list.return_value = [
                {'name': '_id_'},
                {'name': 'simulation_id_1'}
            ]
        
        mock_database.__getitem__.side_effect = lambda key: mock_collections[key]
        
        with patch('services.simulation_database_optimization.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.return_value = mock_database
            
            result = await optimizer.get_collection_statistics()
            
            assert isinstance(result, dict)
            assert len(result) == 3
            
            for collection_name in ['simulations', 'populations', 'bacteria']:
                stats = result[collection_name]
                assert stats['document_count'] == 100
                assert stats['storage_size_bytes'] == 1024
                assert stats['index_count'] == 2
                assert stats['avg_obj_size'] == 128

    @pytest.mark.asyncio
    async def test_benchmark_common_queries_success(self, optimizer):
        """Test successful benchmarking of common queries"""
        
        # Mock the analyze_query_performance method
        async def mock_analyze_performance(collection_name, query):
            return {
                'query': query,
                'collection': collection_name,
                'execution_time_ms': 10,
                'documents_examined': 50,
                'documents_returned': 5
            }
        
        optimizer.analyze_query_performance = mock_analyze_performance
        
        result = await optimizer.benchmark_common_queries()
        
        assert isinstance(result, dict)
        assert 'simulations' in result
        assert 'populations' in result
        assert 'bacteria' in result
        
        # Check that each collection has benchmark results
        for collection_name, collection_results in result.items():
            assert isinstance(collection_results, list)
            assert len(collection_results) > 0
            
            for query_result in collection_results:
                assert 'query_index' in query_result
                assert 'query' in query_result
                assert 'performance' in query_result

    @pytest.mark.asyncio
    async def test_benchmark_common_queries_with_errors(self, optimizer):
        """Test benchmarking with some queries failing"""
        
        # Mock the analyze_query_performance method with errors
        async def mock_analyze_performance(collection_name, query):
            if 'error_trigger' in str(query):
                raise Exception("Test query error")
            return {
                'query': query,
                'collection': collection_name,
                'execution_time_ms': 10
            }
        
        optimizer.analyze_query_performance = mock_analyze_performance
        
        # Inject error-triggering query
        original_queries = optimizer.benchmark_common_queries.__wrapped__.__defaults__
        
        result = await optimizer.benchmark_common_queries()
        
        # Should still return results structure even with errors
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_optimize_database_success(self, optimizer):
        """Test complete database optimization success"""
        
        # Mock all optimization methods
        optimizer.create_indexes = AsyncMock(return_value={'simulations': ['index1', 'index2']})
        optimizer.benchmark_common_queries = AsyncMock(return_value={'results': 'benchmark_data'})
        optimizer.get_collection_statistics = AsyncMock(return_value={'stats': 'collection_data'})
        
        result = await optimizer.optimize_database()
        
        assert result['success'] is True
        assert 'created_indexes' in result
        assert 'benchmark_results' in result
        assert 'collection_statistics' in result
        assert len(result['steps_completed']) == 3
        assert 'create_indexes' in result['steps_completed']
        assert 'benchmark_queries' in result['steps_completed']
        assert 'collect_statistics' in result['steps_completed']

    @pytest.mark.asyncio
    async def test_optimize_database_failure(self, optimizer):
        """Test database optimization failure handling"""
        
        # Mock create_indexes to fail
        optimizer.create_indexes = AsyncMock(side_effect=Exception("Index creation failed"))
        
        with pytest.raises(DatabaseOptimizationError):
            await optimizer.optimize_database()

    def test_performance_monitor_decorator_success(self):
        """Test performance monitor decorator with successful function"""
        
        @performance_monitor
        async def test_function():
            await asyncio.sleep(0.001)  # Simulate work
            return "success"
        
        # Test that decorator works
        result = asyncio.run(test_function())
        assert result == "success"

    def test_performance_monitor_decorator_failure(self):
        """Test performance monitor decorator with failing function"""
        
        @performance_monitor
        async def test_function():
            raise ValueError("Test error")
        
        # Test that decorator properly handles exceptions
        with pytest.raises(ValueError, match="Test error"):
            asyncio.run(test_function())

    @pytest.mark.asyncio
    async def test_get_database_optimizer(self):
        """Test factory function for database optimizer"""
        
        with patch('services.simulation_database_optimization.ensure_database_connection') as mock_ensure:
            mock_ensure.return_value = None
            
            optimizer = await get_database_optimizer()
            
            assert isinstance(optimizer, SimulationDatabaseOptimizer)
            mock_ensure.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_indexes_database_error(self, optimizer, mock_database):
        """Test index creation with database connection error"""
        
        with patch('services.simulation_database_optimization.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.side_effect = Exception("Database connection failed")
            
            with pytest.raises(DatabaseOptimizationError, match="Failed to create database indexes"):
                await optimizer.create_indexes()

    @pytest.mark.asyncio
    async def test_list_indexes_database_error(self, optimizer, mock_database):
        """Test index listing with database connection error"""
        
        with patch('services.simulation_database_optimization.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.side_effect = Exception("Database connection failed")
            
            with pytest.raises(DatabaseOptimizationError, match="Failed to list database indexes"):
                await optimizer.list_indexes()

    @pytest.mark.asyncio
    async def test_query_analysis_database_error(self, optimizer):
        """Test query analysis with database connection error"""
        
        with patch('services.simulation_database_optimization.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.side_effect = Exception("Database connection failed")
            
            with pytest.raises(DatabaseOptimizationError, match="Failed to analyze query performance"):
                await optimizer.analyze_query_performance('simulations', {})

    @pytest.mark.asyncio
    async def test_collection_statistics_database_error(self, optimizer):
        """Test collection statistics with database connection error"""
        
        with patch('services.simulation_database_optimization.DatabaseSession') as mock_session:
            mock_session.return_value.__aenter__.side_effect = Exception("Database connection failed")
            
            with pytest.raises(DatabaseOptimizationError, match="Failed to get collection statistics"):
                await optimizer.get_collection_statistics() 