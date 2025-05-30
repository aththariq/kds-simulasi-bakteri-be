"""
State management system for bacterial simulation tracking and persistence.
"""

import json
import pickle
import gzip
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
import logging
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class SimulationState(Enum):
    """Enumeration of possible simulation states."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    RECOVERING = "recovering"


@dataclass
class StateSnapshot:
    """Represents a snapshot of simulation state at a specific point."""
    simulation_id: str
    generation: int
    timestamp: datetime
    state: SimulationState
    progress_percentage: float
    population_size: int
    resistance_frequency: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary format."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['state'] = self.state.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateSnapshot':
        """Create snapshot from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['state'] = SimulationState(data['state'])
        return cls(**data)


@dataclass
class StateConfig:
    """Configuration for state management system."""
    storage_dir: str = "./simulation_states"
    backup_dir: str = "./simulation_backups"
    auto_save_interval: int = 30  # seconds
    max_snapshots_per_simulation: int = 100
    compression_enabled: bool = True
    backup_retention_days: int = 7
    checkpoint_interval: int = 50  # generations
    max_memory_usage_mb: int = 1024


class StateManager:
    """
    Comprehensive state management system for bacterial simulations.
    
    Provides state tracking, persistence, recovery, and memory management
    capabilities for simulation lifecycle management.
    """
    
    def __init__(self, config: StateConfig = None):
        """
        Initialize state manager.
        
        Args:
            config: State management configuration
        """
        self.config = config or StateConfig()
        self.active_states: Dict[str, Dict[str, Any]] = {}
        self.state_snapshots: Dict[str, List[StateSnapshot]] = {}
        self.state_observers: Dict[str, List[Callable]] = {}
        self._auto_save_tasks: Dict[str, asyncio.Task] = {}
        
        # Ensure directories exist
        self._setup_directories()
        
        # Load existing states on startup
        self._load_existing_states()
    
    def _setup_directories(self):
        """Create necessary directories for state management."""
        Path(self.config.storage_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        Path(self.config.storage_dir, "checkpoints").mkdir(exist_ok=True)
        Path(self.config.storage_dir, "snapshots").mkdir(exist_ok=True)
        Path(self.config.storage_dir, "metadata").mkdir(exist_ok=True)
    
    def _load_existing_states(self):
        """Load existing simulation states from disk."""
        try:
            metadata_dir = Path(self.config.storage_dir, "metadata")
            for metadata_file in metadata_dir.glob("*.json"):
                simulation_id = metadata_file.stem
                try:
                    with open(metadata_file, 'r') as f:
                        state_data = json.load(f)
                    self.active_states[simulation_id] = state_data
                    logger.info(f"Loaded state for simulation {simulation_id}")
                except Exception as e:
                    logger.error(f"Failed to load state for {simulation_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to load existing states: {e}")
    
    def create_simulation_state(
        self,
        simulation_id: str,
        initial_params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new simulation state.
        
        Args:
            simulation_id: Unique simulation identifier
            initial_params: Initial simulation parameters
            metadata: Additional metadata
            
        Returns:
            Created state data
        """
        state_data = {
            "simulation_id": simulation_id,
            "state": SimulationState.INITIALIZED,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "parameters": initial_params,
            "current_generation": 0,
            "progress_percentage": 0.0,
            "metadata": metadata or {},
            "performance_metrics": {
                "start_time": None,
                "end_time": None,
                "generation_times": [],
                "memory_usage": [],
                "state_transitions": []
            },
            "checkpoints": [],
            "last_snapshot": None,
            "error_history": [],
            "recovery_points": []
        }
        
        self.active_states[simulation_id] = state_data
        self.state_snapshots[simulation_id] = []
        self.state_observers[simulation_id] = []
        
        # Save initial state
        self._save_state_metadata(simulation_id)
        
        # Start auto-save task
        self._start_auto_save(simulation_id)
        
        return state_data
    
    def update_simulation_state(
        self,
        simulation_id: str,
        updates: Dict[str, Any],
        create_snapshot: bool = True
    ) -> bool:
        """
        Update simulation state with new data.
        
        Args:
            simulation_id: Simulation identifier
            updates: Updates to apply
            create_snapshot: Whether to create a snapshot
            
        Returns:
            True if successful, False otherwise
        """
        if simulation_id not in self.active_states:
            logger.error(f"Simulation {simulation_id} not found")
            return False
        
        try:
            state_data = self.active_states[simulation_id]
            old_state = state_data.get("state")
            
            # Apply updates
            for key, value in updates.items():
                if key == "state" and isinstance(value, str):
                    value = SimulationState(value)
                state_data[key] = value
            
            # Update timestamp
            state_data["updated_at"] = datetime.utcnow()
            
            # Track state transitions
            new_state = state_data.get("state")
            if old_state != new_state:
                transition = {
                    "from_state": old_state.value if isinstance(old_state, SimulationState) else old_state,
                    "to_state": new_state.value if isinstance(new_state, SimulationState) else new_state,
                    "timestamp": datetime.utcnow().isoformat(),
                    "generation": state_data.get("current_generation", 0)
                }
                state_data["performance_metrics"]["state_transitions"].append(transition)
            
            # Create snapshot if requested
            if create_snapshot:
                self._create_snapshot(simulation_id)
            
            # Notify observers
            self._notify_observers(simulation_id, updates)
            
            # Save state
            self._save_state_metadata(simulation_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update state for {simulation_id}: {e}")
            return False
    
    def get_simulation_state(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current simulation state.
        
        Args:
            simulation_id: Simulation identifier
            
        Returns:
            State data or None if not found
        """
        return self.active_states.get(simulation_id)
    
    def _create_snapshot(self, simulation_id: str) -> Optional[StateSnapshot]:
        """
        Create a state snapshot.
        
        Args:
            simulation_id: Simulation identifier
            
        Returns:
            Created snapshot or None if failed
        """
        try:
            state_data = self.active_states.get(simulation_id)
            if not state_data:
                return None
            
            snapshot = StateSnapshot(
                simulation_id=simulation_id,
                generation=state_data.get("current_generation", 0),
                timestamp=datetime.utcnow(),
                state=state_data.get("state", SimulationState.INITIALIZED),
                progress_percentage=state_data.get("progress_percentage", 0.0),
                population_size=state_data.get("population_size", 0),
                resistance_frequency=state_data.get("resistance_frequency", 0.0),
                metadata=state_data.get("metadata", {})
            )
            
            # Add to snapshots list
            if simulation_id not in self.state_snapshots:
                self.state_snapshots[simulation_id] = []
            
            self.state_snapshots[simulation_id].append(snapshot)
            
            # Maintain max snapshots limit
            max_snapshots = self.config.max_snapshots_per_simulation
            if len(self.state_snapshots[simulation_id]) > max_snapshots:
                self.state_snapshots[simulation_id] = self.state_snapshots[simulation_id][-max_snapshots:]
            
            # Update last snapshot reference
            state_data["last_snapshot"] = snapshot.to_dict()
            
            # Save snapshot to disk
            self._save_snapshot(snapshot)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to create snapshot for {simulation_id}: {e}")
            return None
    
    def create_checkpoint(self, simulation_id: str, simulation_data: Any) -> bool:
        """
        Create a checkpoint with full simulation data.
        
        Args:
            simulation_id: Simulation identifier
            simulation_data: Complete simulation data to checkpoint
            
        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint_path = Path(
                self.config.storage_dir,
                "checkpoints",
                f"{simulation_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.checkpoint"
            )
            
            # Serialize simulation data
            if self.config.compression_enabled:
                with gzip.open(checkpoint_path, 'wb') as f:
                    pickle.dump(simulation_data, f)
            else:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(simulation_data, f)
            
            # Update state with checkpoint info
            state_data = self.active_states.get(simulation_id, {})
            checkpoint_info = {
                "path": str(checkpoint_path),
                "timestamp": datetime.utcnow().isoformat(),
                "generation": state_data.get("current_generation", 0),
                "size_bytes": checkpoint_path.stat().st_size
            }
            
            if "checkpoints" not in state_data:
                state_data["checkpoints"] = []
            state_data["checkpoints"].append(checkpoint_info)
            
            # Maintain checkpoint limit (keep last 10)
            if len(state_data["checkpoints"]) > 10:
                old_checkpoint = state_data["checkpoints"].pop(0)
                # Remove old checkpoint file
                try:
                    Path(old_checkpoint["path"]).unlink()
                except Exception:
                    pass
            
            self._save_state_metadata(simulation_id)
            
            logger.info(f"Created checkpoint for simulation {simulation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint for {simulation_id}: {e}")
            return False
    
    def load_checkpoint(self, simulation_id: str, checkpoint_index: int = -1) -> Optional[Any]:
        """
        Load simulation data from checkpoint.
        
        Args:
            simulation_id: Simulation identifier
            checkpoint_index: Index of checkpoint to load (-1 for latest)
            
        Returns:
            Loaded simulation data or None if failed
        """
        try:
            state_data = self.active_states.get(simulation_id)
            if not state_data or not state_data.get("checkpoints"):
                return None
            
            checkpoint_info = state_data["checkpoints"][checkpoint_index]
            checkpoint_path = Path(checkpoint_info["path"])
            
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return None
            
            # Load simulation data
            if self.config.compression_enabled and checkpoint_path.suffix == '.checkpoint':
                with gzip.open(checkpoint_path, 'rb') as f:
                    simulation_data = pickle.load(f)
            else:
                with open(checkpoint_path, 'rb') as f:
                    simulation_data = pickle.load(f)
            
            logger.info(f"Loaded checkpoint for simulation {simulation_id}")
            return simulation_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint for {simulation_id}: {e}")
            return None
    
    def add_state_observer(self, simulation_id: str, observer: Callable):
        """
        Add an observer for state changes.
        
        Args:
            simulation_id: Simulation identifier
            observer: Callback function to notify on state changes
        """
        if simulation_id not in self.state_observers:
            self.state_observers[simulation_id] = []
        self.state_observers[simulation_id].append(observer)
    
    def remove_state_observer(self, simulation_id: str, observer: Callable):
        """
        Remove a state observer.
        
        Args:
            simulation_id: Simulation identifier
            observer: Observer to remove
        """
        if simulation_id in self.state_observers:
            try:
                self.state_observers[simulation_id].remove(observer)
            except ValueError:
                pass
    
    def _notify_observers(self, simulation_id: str, updates: Dict[str, Any]):
        """Notify all observers of state changes."""
        observers = self.state_observers.get(simulation_id, [])
        for observer in observers:
            try:
                observer(simulation_id, updates)
            except Exception as e:
                logger.error(f"Error in state observer: {e}")
    
    def _save_state_metadata(self, simulation_id: str):
        """Save state metadata to disk."""
        try:
            state_data = self.active_states.get(simulation_id)
            if not state_data:
                return
            
            # Convert datetime objects to ISO format for JSON serialization
            serializable_data = self._make_json_serializable(state_data.copy())
            
            metadata_path = Path(self.config.storage_dir, "metadata", f"{simulation_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save state metadata for {simulation_id}: {e}")
    
    def _save_snapshot(self, snapshot: StateSnapshot):
        """Save snapshot to disk."""
        try:
            snapshot_path = Path(
                self.config.storage_dir,
                "snapshots",
                f"{snapshot.simulation_id}_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
    
    def _make_json_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format."""
        if isinstance(data, datetime):
            return data.isoformat()
        elif isinstance(data, SimulationState):
            return data.value
        elif isinstance(data, dict):
            return {key: self._make_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        else:
            return data
    
    def _start_auto_save(self, simulation_id: str):
        """Start auto-save task for simulation."""
        async def auto_save_task():
            while simulation_id in self.active_states:
                try:
                    await asyncio.sleep(self.config.auto_save_interval)
                    if simulation_id in self.active_states:
                        self._save_state_metadata(simulation_id)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in auto-save for {simulation_id}: {e}")
        
        # Cancel existing task if any
        if simulation_id in self._auto_save_tasks:
            self._auto_save_tasks[simulation_id].cancel()
        
        # Start new task
        self._auto_save_tasks[simulation_id] = asyncio.create_task(auto_save_task())
    
    def cleanup_simulation(self, simulation_id: str, remove_files: bool = False):
        """
        Clean up simulation state and files.
        
        Args:
            simulation_id: Simulation identifier
            remove_files: Whether to remove saved files
        """
        try:
            # Cancel auto-save task
            if simulation_id in self._auto_save_tasks:
                self._auto_save_tasks[simulation_id].cancel()
                del self._auto_save_tasks[simulation_id]
            
            # Remove from memory
            self.active_states.pop(simulation_id, None)
            self.state_snapshots.pop(simulation_id, None)
            self.state_observers.pop(simulation_id, None)
            
            if remove_files:
                # Remove metadata file
                metadata_path = Path(self.config.storage_dir, "metadata", f"{simulation_id}.json")
                if metadata_path.exists():
                    metadata_path.unlink()
                
                # Remove snapshots
                snapshot_dir = Path(self.config.storage_dir, "snapshots")
                for snapshot_file in snapshot_dir.glob(f"{simulation_id}_*.json"):
                    snapshot_file.unlink()
                
                # Remove checkpoints
                checkpoint_dir = Path(self.config.storage_dir, "checkpoints")
                for checkpoint_file in checkpoint_dir.glob(f"{simulation_id}_*.checkpoint"):
                    checkpoint_file.unlink()
            
            logger.info(f"Cleaned up simulation {simulation_id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup simulation {simulation_id}: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        import psutil
        import sys
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "process_memory_mb": memory_info.rss / 1024 / 1024,
            "active_simulations": len(self.active_states),
            "total_snapshots": sum(len(snapshots) for snapshots in self.state_snapshots.values()),
            "python_objects_size_mb": sys.getsizeof(self.active_states) / 1024 / 1024
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on state management system."""
        try:
            storage_dir = Path(self.config.storage_dir)
            backup_dir = Path(self.config.backup_dir)
            
            health_status = {
                "healthy": True,
                "storage_accessible": storage_dir.exists() and os.access(storage_dir, os.W_OK),
                "backup_accessible": backup_dir.exists() and os.access(backup_dir, os.W_OK),
                "active_simulations": len(self.active_states),
                "auto_save_tasks": len(self._auto_save_tasks),
                "memory_usage": self.get_memory_usage(),
                "errors": []
            }
            
            # Check for any issues
            if not health_status["storage_accessible"]:
                health_status["healthy"] = False
                health_status["errors"].append("Storage directory not accessible")
            
            if not health_status["backup_accessible"]:
                health_status["healthy"] = False
                health_status["errors"].append("Backup directory not accessible")
            
            return health_status
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Global state manager instance
state_manager = StateManager() 