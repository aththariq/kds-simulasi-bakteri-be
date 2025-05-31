import asyncio
import logging
from typing import Optional, Callable, Any
from functools import wraps
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo.errors import (
    ConnectionFailure, 
    ServerSelectionTimeoutError, 
    NetworkTimeout,
    OperationFailure
)
from config.database import get_database, init_database, close_database

logger = logging.getLogger(__name__)

class DatabaseConnectionError(Exception):
    """Custom exception for database connection errors"""
    pass

class DatabaseOperationError(Exception):
    """Custom exception for database operation errors"""
    pass

def retry_on_connection_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry database operations on connection failure"""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except (ConnectionFailure, ServerSelectionTimeoutError, NetworkTimeout) as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Database connection failed (attempt {attempt + 1}/{max_retries + 1}). "
                            f"Retrying in {wait_time} seconds. Error: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Database connection failed after {max_retries + 1} attempts")
                        
                except Exception as e:
                    # Don't retry on non-connection errors
                    logger.error(f"Database operation failed: {e}")
                    raise DatabaseOperationError(f"Database operation failed: {e}") from e
            
            raise DatabaseConnectionError(f"Failed to connect after {max_retries + 1} attempts") from last_exception
        
        return wrapper
    return decorator

class DatabaseManager:
    """Database connection and operation manager"""
    
    def __init__(self):
        self._database: Optional[AsyncIOMotorDatabase] = None
        self._connected = False
    
    async def connect(self) -> None:
        """Initialize database connection with retry logic"""
        try:
            await init_database()
            self._database = await get_database()
            self._connected = True
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to establish database connection: {e}")
            self._connected = False
            raise DatabaseConnectionError(f"Database connection failed: {e}") from e
    
    async def disconnect(self) -> None:
        """Close database connection"""
        try:
            await close_database()
            self._connected = False
            logger.info("Database connection closed")
            
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self._connected
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def ping(self) -> bool:
        """Test database connection"""
        try:
            if not self._database:
                raise DatabaseConnectionError("Database not initialized")
                
            await self._database.command('ping')
            return True
            
        except Exception as e:
            logger.error(f"Database ping failed: {e}")
            raise
    
    async def get_database(self) -> AsyncIOMotorDatabase:
        """Get database instance with connection check"""
        if not self._connected or not self._database:
            await self.connect()
        
        # Test connection
        await self.ping()
        return self._database
    
    @retry_on_connection_failure(max_retries=3, delay=1.0)
    async def execute_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute database operation with error handling"""
        try:
            database = await self.get_database()
            return await operation(database, *args, **kwargs)
            
        except OperationFailure as e:
            logger.error(f"Database operation failed: {e}")
            raise DatabaseOperationError(f"Operation failed: {e}") from e
            
        except Exception as e:
            logger.error(f"Unexpected error during database operation: {e}")
            raise

# Global database manager instance
db_manager = DatabaseManager()

async def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    return db_manager

async def ensure_database_connection() -> AsyncIOMotorDatabase:
    """Ensure database connection and return database instance"""
    return await db_manager.get_database()

# Context manager for database operations
class DatabaseSession:
    """Context manager for database operations"""
    
    def __init__(self):
        self.database: Optional[AsyncIOMotorDatabase] = None
    
    async def __aenter__(self) -> AsyncIOMotorDatabase:
        """Enter context and get database"""
        self.database = await ensure_database_connection()
        return self.database
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context"""
        # Connection pooling handles cleanup automatically
        pass

# Health check function
async def check_database_health() -> dict:
    """Check database health and return status"""
    try:
        manager = await get_db_manager()
        
        if not manager.is_connected:
            return {
                "status": "disconnected",
                "message": "Database not connected"
            }
        
        # Test connection
        await manager.ping()
        
        # Get database stats
        database = await manager.get_database()
        stats = await database.command("dbStats")
        
        return {
            "status": "healthy",
            "message": "Database connection is healthy",
            "stats": {
                "collections": stats.get("collections", 0),
                "dataSize": stats.get("dataSize", 0),
                "storageSize": stats.get("storageSize", 0),
                "indexes": stats.get("indexes", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Database health check failed: {e}"
        } 