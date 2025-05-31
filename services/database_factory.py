from typing import Optional
import logging
from contextlib import asynccontextmanager

from services.database_service import DatabaseService
from utils.db_connection import DatabaseManager
from config.database import DatabaseConfig

logger = logging.getLogger(__name__)

class DatabaseServiceFactory:
    """Factory for creating and managing database service instances"""
    
    _instance: Optional[DatabaseService] = None
    _db_manager: Optional[DatabaseManager] = None
    
    @classmethod
    async def create_service(cls, config: Optional[DatabaseConfig] = None) -> DatabaseService:
        """
        Create a database service instance
        
        Args:
            config: Optional database configuration
            
        Returns:
            DatabaseService: Initialized database service
        """
        if config is None:
            config = DatabaseConfig()
        
        # Create database manager
        db_manager = DatabaseManager(config)
        
        # Create and initialize service
        service = DatabaseService(db_manager)
        await service.initialize()
        
        return service
    
    @classmethod
    async def get_singleton_service(cls, config: Optional[DatabaseConfig] = None) -> DatabaseService:
        """
        Get or create a singleton database service instance
        
        Args:
            config: Optional database configuration
            
        Returns:
            DatabaseService: Singleton database service
        """
        if cls._instance is None:
            cls._instance = await cls.create_service(config)
            logger.info("Created singleton database service")
        
        return cls._instance
    
    @classmethod
    async def close_singleton(cls):
        """Close the singleton database service"""
        if cls._instance is not None:
            if cls._db_manager is not None:
                await cls._db_manager.close()
            cls._instance = None
            cls._db_manager = None
            logger.info("Closed singleton database service")
    
    @classmethod
    @asynccontextmanager
    async def get_service_context(cls, config: Optional[DatabaseConfig] = None):
        """
        Context manager for database service with automatic cleanup
        
        Args:
            config: Optional database configuration
            
        Yields:
            DatabaseService: Database service instance
        """
        service = None
        db_manager = None
        
        try:
            if config is None:
                config = DatabaseConfig()
            
            db_manager = DatabaseManager(config)
            service = DatabaseService(db_manager)
            await service.initialize()
            
            yield service
            
        except Exception as e:
            logger.error(f"Database service context error: {e}")
            raise
        finally:
            if db_manager is not None:
                await db_manager.close()
                logger.debug("Closed database service context")

# Convenience functions for common usage patterns
async def get_database_service(config: Optional[DatabaseConfig] = None) -> DatabaseService:
    """Get a database service instance"""
    return await DatabaseServiceFactory.create_service(config)

async def get_singleton_database_service(config: Optional[DatabaseConfig] = None) -> DatabaseService:
    """Get the singleton database service instance"""
    return await DatabaseServiceFactory.get_singleton_service(config)

@asynccontextmanager
async def database_service_context(config: Optional[DatabaseConfig] = None):
    """Context manager for database service"""
    async with DatabaseServiceFactory.get_service_context(config) as service:
        yield service 