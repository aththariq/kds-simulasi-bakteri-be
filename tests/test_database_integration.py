import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List
import random
import string
from unittest.mock import patch

from models.database_models import (
    SimulationMetadata, SimulationParameters, SimulationResults,
    PopulationSnapshot, BacteriumData, SimulationStatus
)
from services.simulation_database import (
    get_simulation_create_operations,
    get_simulation_read_operations,
    get_simulation_update_operations,
    get_simulation_delete_operations
)
from services.simulation_database_optimization import get_database_optimizer

class TestDatabaseIntegration:
    """Integration tests for complete database operations workflow"""
    
    @pytest.fixture
    def sample_simulation_parameters(self):
        """Generate realistic simulation parameters"""
        return SimulationParameters(
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
    
    @pytest.fixture
    def sample_simulation_metadata(self, sample_simulation_parameters):
        """Generate realistic simulation metadata"""
        return SimulationMetadata(
            simulation_id=f"sim_test_{random.randint(1000, 9999)}",
            name="Integration Test Simulation",
            description="Testing database integration with realistic data",
            status=SimulationStatus.PENDING,
            parameters=sample_simulation_parameters,
            tags=["test", "integration", "benchmark"]
        )
    
    def generate_population_snapshot(self, generation: int, total_pop: int = 1000) -> PopulationSnapshot:
        """Generate realistic population snapshot data"""
        resistant_pop = random.randint(0, total_pop)
        susceptible_pop = total_pop - resistant_pop
        
        bacteria = []
        for i in range(min(100, total_pop)):  # Sample subset for testing
            bacterium = BacteriumData(
                id=f"bact_{generation}_{i}",
                fitness=random.uniform(0.1, 1.0),
                resistance_genes=random.choices(
                    ["ampR", "tetR", "strR", "chlR"], 
                    k=random.randint(0, 3)
                ),
                position_x=random.uniform(0, 100),
                position_y=random.uniform(0, 100),
                generation=generation,
                parent_id=f"bact_{generation-1}_{random.randint(0, 99)}" if generation > 0 else None,
                mutation_count=random.randint(0, 5)
            )
            bacteria.append(bacterium)
        
        return PopulationSnapshot(
            generation=generation,
            total_population=total_pop,
            resistant_population=resistant_pop,
            susceptible_population=susceptible_pop,
            average_fitness=random.uniform(0.4, 0.9),
            resistance_frequency=resistant_pop / total_pop if total_pop > 0 else 0.0,
            bacteria=bacteria
        )
    
    @pytest.mark.asyncio
    async def test_complete_simulation_lifecycle(self, sample_simulation_metadata):
        """Test complete simulation lifecycle: create, read, update, delete"""
        
        # Mock database operations for integration testing
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = mock_session.return_value.__aenter__.return_value
            mock_collection = mock_db.__getitem__.return_value
            
            # Mock successful operations
            mock_collection.insert_one.return_value.inserted_id = "mock_object_id"
            mock_collection.find_one.return_value = {
                "_id": "mock_object_id",
                "simulation_id": sample_simulation_metadata.simulation_id,
                "metadata": sample_simulation_metadata.model_dump(),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            mock_collection.update_one.return_value.modified_count = 1
            mock_collection.delete_one.return_value.deleted_count = 1
            
            # Step 1: Create simulation
            create_ops = await get_simulation_create_operations()
            simulation_id = await create_ops.create_simulation(sample_simulation_metadata)
            assert simulation_id == sample_simulation_metadata.simulation_id
            
            # Step 2: Add population snapshots
            for generation in range(5):  # Add 5 generations
                snapshot = self.generate_population_snapshot(generation)
                snapshot_id = await create_ops.save_population_snapshot(simulation_id, snapshot)
                assert snapshot_id is not None
            
            # Step 3: Read operations
            read_ops = await get_simulation_read_operations()
            retrieved_sim = await read_ops.get_simulation_by_id(simulation_id)
            assert retrieved_sim is not None
            
            # Mock population history response
            mock_collection.find.return_value.to_list.return_value = [
                {
                    "_id": f"mock_pop_{i}",
                    "simulation_id": simulation_id,
                    "snapshot": self.generate_population_snapshot(i).model_dump(),
                    "created_at": datetime.utcnow()
                }
                for i in range(5)
            ]
            
            population_history = await read_ops.get_population_history(simulation_id)
            assert len(population_history) == 5
            
            # Step 4: Update operations
            update_ops = await get_simulation_update_operations()
            
            # Update metadata
            metadata_updates = {
                "name": "Updated Integration Test",
                "description": "Updated description for testing"
            }
            update_success = await update_ops.update_simulation_metadata(
                simulation_id, metadata_updates
            )
            assert update_success is True
            
            # Add tags
            tag_success = await update_ops.add_simulation_tags(
                simulation_id, ["updated", "modified"]
            )
            assert tag_success is True
            
            # Step 5: Delete operations
            delete_ops = await get_simulation_delete_operations()
            
            # Soft delete first
            soft_delete_success = await delete_ops.soft_delete_simulation(
                simulation_id, "Integration test cleanup"
            )
            assert soft_delete_success is True
            
            # Hard delete
            deletion_results = await delete_ops.hard_delete_simulation(
                simulation_id, delete_related_data=True
            )
            assert deletion_results["simulations_deleted"] == 1
    
    @pytest.mark.asyncio
    async def test_load_testing_simulation_data(self):
        """Test database performance with large datasets"""
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = mock_session.return_value.__aenter__.return_value
            mock_collection = mock_db.__getitem__.return_value
            
            # Mock bulk operations
            mock_collection.insert_many.return_value.inserted_ids = [
                f"mock_id_{i}" for i in range(1000)
            ]
            mock_collection.find.return_value.to_list.return_value = []
            
            create_ops = await get_simulation_create_operations()
            
            # Test batch insert of large population snapshots
            large_snapshot = self.generate_population_snapshot(0, total_pop=10000)
            bacteria_batch = large_snapshot.bacteria * 100  # Create large dataset
            
            simulation_id = f"load_test_{random.randint(1000, 9999)}"
            
            # Test batch bacteria insertion
            batch_ids = await create_ops.batch_insert_bacteria(
                simulation_id, 0, bacteria_batch
            )
            assert len(batch_ids) == len(bacteria_batch)
            
            # Test multiple population snapshots
            snapshots = [
                self.generate_population_snapshot(i, total_pop=5000)
                for i in range(100)
            ]
            
            snapshot_ids = await create_ops.batch_insert_population_snapshots(
                simulation_id, snapshots
            )
            assert len(snapshot_ids) == len(snapshots)
    
    @pytest.mark.asyncio
    async def test_database_optimization_workflow(self):
        """Test complete database optimization workflow"""
        
        with patch('services.simulation_database_optimization.DatabaseSession') as mock_session:
            mock_db = mock_session.return_value.__aenter__.return_value
            mock_collection = mock_db.__getitem__.return_value
            
            # Mock index operations
            mock_collection.create_index.return_value = "test_index_1"
            mock_collection.list_indexes.return_value.to_list.return_value = [
                {"name": "_id_"},
                {"name": "test_index_1"}
            ]
            
            # Mock collection stats
            mock_db.command.return_value = {
                'storageSize': 1024000,
                'indexSizes': {'_id_': 512000, 'test_index_1': 256000},
                'avgObjSize': 1024,
                'size': 2048000
            }
            
            mock_collection.count_documents.return_value = 1000
            
            # Mock query explanation
            mock_cursor = mock_collection.find.return_value
            mock_cursor.explain.return_value = {
                'executionStats': {
                    'executionTimeMillis': 15,
                    'totalDocsExamined': 1000,
                    'totalDocsReturned': 100,
                    'indexUsed': True
                },
                'queryPlanner': {
                    'winningPlan': {'stage': 'IXSCAN', 'indexName': 'test_index_1'}
                }
            }
            
            optimizer = await get_database_optimizer()
            
            # Step 1: Create indexes
            created_indexes = await optimizer.create_indexes()
            assert isinstance(created_indexes, dict)
            assert len(created_indexes) > 0
            
            # Step 2: Analyze performance
            test_query = {"metadata.status": SimulationStatus.COMPLETED}
            performance_analysis = await optimizer.analyze_query_performance(
                'simulations', test_query
            )
            assert performance_analysis['execution_time_ms'] == 15
            assert performance_analysis['index_used'] is True
            
            # Step 3: Get statistics
            collection_stats = await optimizer.get_collection_statistics()
            assert isinstance(collection_stats, dict)
            assert 'simulations' in collection_stats
            
            # Step 4: Run benchmarks
            benchmark_results = await optimizer.benchmark_common_queries()
            assert isinstance(benchmark_results, dict)
            
            # Step 5: Complete optimization
            optimization_results = await optimizer.optimize_database()
            assert optimization_results['success'] is True
            assert len(optimization_results['steps_completed']) == 3
    
    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self, sample_simulation_metadata):
        """Test concurrent database operations for thread safety"""
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = mock_session.return_value.__aenter__.return_value
            mock_collection = mock_db.__getitem__.return_value
            
            # Mock concurrent operations
            mock_collection.insert_one.return_value.inserted_id = "mock_object_id"
            mock_collection.find_one.return_value = {
                "_id": "mock_object_id",
                "simulation_id": sample_simulation_metadata.simulation_id,
                "metadata": sample_simulation_metadata.model_dump(),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            mock_collection.update_one.return_value.modified_count = 1
            
            create_ops = await get_simulation_create_operations()
            read_ops = await get_simulation_read_operations()
            update_ops = await get_simulation_update_operations()
            
            # Create multiple simulations concurrently
            concurrent_tasks = []
            for i in range(10):
                metadata = sample_simulation_metadata.model_copy()
                metadata.simulation_id = f"concurrent_test_{i}"
                task = create_ops.create_simulation(metadata)
                concurrent_tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            
            # Verify all succeeded
            for result in results:
                assert not isinstance(result, Exception)
                assert isinstance(result, str)
            
            # Test concurrent reads
            read_tasks = [
                read_ops.get_simulation_by_id(f"concurrent_test_{i}")
                for i in range(10)
            ]
            
            read_results = await asyncio.gather(*read_tasks, return_exceptions=True)
            
            # Verify reads succeeded
            for result in read_results:
                assert not isinstance(result, Exception)
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self, sample_simulation_metadata):
        """Test error recovery and system resilience"""
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            # Test connection failure recovery
            mock_session.side_effect = [
                Exception("Connection failed"),  # First attempt fails
                Exception("Connection failed"),  # Second attempt fails
                mock_session.return_value  # Third attempt succeeds
            ]
            
            mock_db = mock_session.return_value.__aenter__.return_value
            mock_collection = mock_db.__getitem__.return_value
            mock_collection.insert_one.return_value.inserted_id = "mock_object_id"
            
            create_ops = await get_simulation_create_operations()
            
            # This should succeed after retries
            simulation_id = await create_ops.create_simulation(sample_simulation_metadata)
            assert simulation_id == sample_simulation_metadata.simulation_id
    
    @pytest.mark.asyncio
    async def test_data_validation_and_integrity(self):
        """Test data validation and integrity constraints"""
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = mock_session.return_value.__aenter__.return_value
            mock_collection = mock_db.__getitem__.return_value
            mock_collection.insert_one.return_value.inserted_id = "mock_object_id"
            
            create_ops = await get_simulation_create_operations()
            
            # Test invalid simulation parameters
            invalid_params = SimulationParameters(
                initial_population=-100,  # Invalid negative population
                generations=0,  # Invalid zero generations
                mutation_rate=1.5,  # Invalid rate > 1.0
                antibiotic_concentration=-0.5,  # Invalid negative concentration
                spatial_enabled=True,
                grid_size=None,  # Missing required grid_size
                hgt_enabled=True,
                hgt_rate=None  # Missing required hgt_rate
            )
            
            # This should raise validation errors
            with pytest.raises(Exception):  # Pydantic validation error
                invalid_metadata = SimulationMetadata(
                    simulation_id="test_invalid",
                    parameters=invalid_params
                )
    
    @pytest.mark.asyncio
    async def test_query_optimization_impact(self):
        """Test the impact of query optimization on performance"""
        
        with patch('services.simulation_database_optimization.DatabaseSession') as mock_session:
            mock_db = mock_session.return_value.__aenter__.return_value
            mock_collection = mock_db.__getitem__.return_value
            
            # Mock performance before optimization (slow)
            mock_cursor_slow = mock_collection.find.return_value
            mock_cursor_slow.explain.return_value = {
                'executionStats': {
                    'executionTimeMillis': 500,  # Slow query
                    'totalDocsExamined': 10000,
                    'totalDocsReturned': 100,
                    'indexUsed': False
                }
            }
            
            optimizer = await get_database_optimizer()
            
            # Analyze performance before optimization
            slow_query = {"metadata.status": SimulationStatus.COMPLETED}
            performance_before = await optimizer.analyze_query_performance(
                'simulations', slow_query
            )
            
            assert performance_before['execution_time_ms'] == 500
            assert performance_before['index_used'] is False
            
            # Mock index creation
            mock_collection.create_index.return_value = "status_1"
            await optimizer.create_indexes()
            
            # Mock performance after optimization (fast)
            mock_cursor_fast = mock_collection.find.return_value
            mock_cursor_fast.explain.return_value = {
                'executionStats': {
                    'executionTimeMillis': 5,  # Fast query with index
                    'totalDocsExamined': 100,
                    'totalDocsReturned': 100,
                    'indexUsed': True
                }
            }
            
            # Analyze performance after optimization
            performance_after = await optimizer.analyze_query_performance(
                'simulations', slow_query
            )
            
            assert performance_after['execution_time_ms'] == 5
            assert performance_after['index_used'] is True
            
            # Verify performance improvement
            improvement_ratio = performance_before['execution_time_ms'] / performance_after['execution_time_ms']
            assert improvement_ratio == 100  # 100x improvement
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_large_datasets(self):
        """Test memory efficiency with large simulation datasets"""
        
        with patch('services.simulation_database.DatabaseSession') as mock_session:
            mock_db = mock_session.return_value.__aenter__.return_value
            mock_collection = mock_db.__getitem__.return_value
            
            # Mock large dataset responses
            mock_collection.find.return_value.to_list.return_value = [
                {
                    "_id": f"mock_bact_{i}",
                    "simulation_id": "memory_test",
                    "generation": i // 1000,
                    "bacterium": self.generate_population_snapshot(i // 1000).bacteria[0].model_dump(),
                    "created_at": datetime.utcnow()
                }
                for i in range(50000)  # Large dataset
            ]
            
            read_ops = await get_simulation_read_operations()
            
            # Test streaming/pagination approach for large datasets
            bacteria_data = await read_ops.get_bacteria_by_generation(
                "memory_test", 
                generation=1,
                limit=1000  # Limit to control memory usage
            )
            
            # Verify we got limited results to control memory
            assert len(bacteria_data) <= 1000
    
    def test_data_model_compatibility(self):
        """Test backward compatibility of data models"""
        
        # Test that old data structure can be converted to new models
        old_simulation_data = {
            "simulation_id": "legacy_sim_001",
            "name": "Legacy Simulation",
            "parameters": {
                "initial_population": 1000,
                "generations": 50,
                "mutation_rate": 0.01,
                "antibiotic_concentration": 0.5
                # Missing new fields like spatial_enabled, hgt_enabled
            },
            "status": "completed",
            "created_at": "2023-01-01T00:00:00Z"
        }
        
        # Should be able to create modern model from legacy data with defaults
        try:
            # Fill in missing required fields with defaults
            old_simulation_data["parameters"]["spatial_enabled"] = False
            old_simulation_data["parameters"]["hgt_enabled"] = False
            
            params = SimulationParameters(**old_simulation_data["parameters"])
            metadata = SimulationMetadata(
                simulation_id=old_simulation_data["simulation_id"],
                name=old_simulation_data["name"],
                status=SimulationStatus.COMPLETED,
                parameters=params
            )
            
            assert metadata.simulation_id == "legacy_sim_001"
            assert metadata.parameters.spatial_enabled is False
            assert metadata.parameters.hgt_enabled is False
            
        except Exception as e:
            pytest.fail(f"Failed to convert legacy data: {e}") 