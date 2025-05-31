import os
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import MongoClient
from pymongo.database import Database
import logging

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration settings"""
    
    def __init__(self):
        self.mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
        self.database_name = os.getenv("DATABASE_NAME", "bacterial_simulation")
        self.max_pool_size = int(os.getenv("MONGODB_MAX_POOL_SIZE", "10"))
        self.min_pool_size = int(os.getenv("MONGODB_MIN_POOL_SIZE", "1"))
        self.max_idle_time_ms = int(os.getenv("MONGODB_MAX_IDLE_TIME_MS", "30000"))
        self.server_selection_timeout_ms = int(os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "5000"))
        self.connect_timeout_ms = int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "10000"))
        self.socket_timeout_ms = int(os.getenv("MONGODB_SOCKET_TIMEOUT_MS", "20000"))

class AsyncMongoDBConnection:
    """Async MongoDB connection manager"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._client: Optional[AsyncIOMotorClient] = None
        self._database: Optional[AsyncIOMotorDatabase] = None
    
    async def connect(self) -> None:
        """Establish connection to MongoDB"""
        try:
            self._client = AsyncIOMotorClient(
                self.config.mongodb_url,
                maxPoolSize=self.config.max_pool_size,
                minPoolSize=self.config.min_pool_size,
                maxIdleTimeMS=self.config.max_idle_time_ms,
                serverSelectionTimeoutMS=self.config.server_selection_timeout_ms,
                connectTimeoutMS=self.config.connect_timeout_ms,
                socketTimeoutMS=self.config.socket_timeout_ms,
            )
            
            # Test the connection
            await self._client.admin.command('ping')
            
            self._database = self._client[self.config.database_name]
            logger.info(f"Successfully connected to MongoDB database: {self.config.database_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")
    
    @property
    def database(self) -> AsyncIOMotorDatabase:
        """Get the database instance"""
        if not self._database:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._database
    
    @property
    def client(self) -> AsyncIOMotorClient:
        """Get the client instance"""
        if not self._client:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._client

class SyncMongoDBConnection:
    """Sync MongoDB connection manager for non-async operations"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._client: Optional[MongoClient] = None
        self._database: Optional[Database] = None
    
    def connect(self) -> None:
        """Establish connection to MongoDB"""
        try:
            self._client = MongoClient(
                self.config.mongodb_url,
                maxPoolSize=self.config.max_pool_size,
                minPoolSize=self.config.min_pool_size,
                maxIdleTimeMS=self.config.max_idle_time_ms,
                serverSelectionTimeoutMS=self.config.server_selection_timeout_ms,
                connectTimeoutMS=self.config.connect_timeout_ms,
                socketTimeoutMS=self.config.socket_timeout_ms,
            )
            
            # Test the connection
            self._client.admin.command('ping')
            
            self._database = self._client[self.config.database_name]
            logger.info(f"Successfully connected to MongoDB database: {self.config.database_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def disconnect(self) -> None:
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")
    
    @property
    def database(self) -> Database:
        """Get the database instance"""
        if not self._database:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._database
    
    @property
    def client(self) -> MongoClient:
        """Get the client instance"""
        if not self._client:
            raise RuntimeError("Client not connected. Call connect() first.")
        return self._client

# Global instances
db_config = DatabaseConfig()
async_db = AsyncMongoDBConnection(db_config)
sync_db = SyncMongoDBConnection(db_config)

async def get_database() -> AsyncIOMotorDatabase:
    """Get async database instance"""
    return async_db.database

def get_sync_database() -> Database:
    """Get sync database instance"""
    return sync_db.database

async def init_database() -> None:
    """Initialize database connection"""
    await async_db.connect()

async def close_database() -> None:
    """Close database connection"""
    await async_db.disconnect()

def init_sync_database() -> None:
    """Initialize sync database connection"""
    sync_db.connect()

def close_sync_database() -> None:
    """Close sync database connection"""
    sync_db.disconnect() 