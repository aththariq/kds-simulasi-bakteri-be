"""
WebSocket Communication Optimization Service
Task 20.3: WebSocket Communication Efficiency Enhancement

This module implements advanced WebSocket communication optimizations including:
- Binary message encoding using MessagePack
- Payload compression with automatic threshold detection
- Delta compression for simulation state updates
- Message batching for high-frequency data streams
- Lightweight message variants for frequent updates
"""

import asyncio
import gzip
import json
import time
import zlib
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import uuid
import logging
import struct
from collections import defaultdict, deque
from schemas.websocket_protocol import MessageType

try:
    import msgpack
except ImportError:
    msgpack = None

logger = logging.getLogger(__name__)


class MessageEncoding(Enum):
    """Message encoding formats."""
    JSON = "json"
    MSGPACK = "msgpack"
    BINARY = "binary"


class CompressionType(Enum):
    """Compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    DEFLATE = "deflate"


class MessageOptimizationLevel(Enum):
    """Optimization levels for different message types."""
    NONE = "none"              # No optimization (JSON only)
    BASIC = "basic"            # Binary encoding only
    COMPRESSED = "compressed"  # Binary + compression
    DELTA = "delta"           # Binary + compression + delta updates
    BATCHED = "batched"       # All optimizations + batching


@dataclass
class CompressionStats:
    """Statistics for compression performance."""
    original_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    compression_time: float = 0.0
    decompression_time: float = 0.0
    algorithm: CompressionType = CompressionType.NONE
    
    def calculate_ratio(self):
        """Calculate compression ratio."""
        if self.original_size > 0:
            self.compression_ratio = self.compressed_size / self.original_size
        else:
            self.compression_ratio = 1.0


@dataclass
class MessageMetrics:
    """Metrics for message transmission optimization."""
    message_id: str
    encoding: MessageEncoding
    compression: CompressionType
    optimization_level: MessageOptimizationLevel
    
    # Size metrics
    original_json_size: int = 0
    optimized_size: int = 0
    size_reduction: float = 0.0
    
    # Timing metrics
    encoding_time: float = 0.0
    compression_time: float = 0.0
    total_optimization_time: float = 0.0
    
    # Performance metrics
    bandwidth_saved: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def calculate_savings(self):
        """Calculate bandwidth savings."""
        self.bandwidth_saved = max(0, self.original_json_size - self.optimized_size)
        if self.original_json_size > 0:
            self.size_reduction = self.bandwidth_saved / self.original_json_size


@dataclass
class DeltaState:
    """State tracking for delta compression."""
    client_id: str
    simulation_id: str
    last_full_state: Dict[str, Any] = field(default_factory=dict)
    sequence_number: int = 0
    last_update_time: datetime = field(default_factory=datetime.now)
    delta_threshold: float = 0.3  # Send full state if delta is >30% of original


@dataclass
class BatchMessage:
    """Container for batched messages."""
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Dict[str, Any]] = field(default_factory=list)
    batch_size: int = 0
    creation_time: datetime = field(default_factory=datetime.now)
    compression_stats: Optional[CompressionStats] = None
    
    def add_message(self, message: Dict[str, Any]):
        """Add a message to the batch."""
        self.messages.append(message)
        self.batch_size = len(self.messages)
    
    def is_ready(self, max_size: int = 10, max_age_seconds: float = 0.1) -> bool:
        """Check if batch is ready for transmission."""
        if self.batch_size >= max_size:
            return True
        
        age = (datetime.now() - self.creation_time).total_seconds()
        return age >= max_age_seconds


class BinaryMessageEncoder:
    """Binary message encoder using MessagePack."""
    
    def __init__(self):
        self.compression_threshold = 1024  # Compress messages > 1KB
        self.stats = defaultdict(list)
    
    def encode_message(
        self, 
        message: Dict[str, Any], 
        optimization_level: MessageOptimizationLevel = MessageOptimizationLevel.COMPRESSED
    ) -> Tuple[bytes, MessageMetrics]:
        """Encode a message with specified optimization level."""
        start_time = time.time()
        message_id = message.get('id', str(uuid.uuid4()))
        
        # Create metrics object
        metrics = MessageMetrics(
            message_id=message_id,
            encoding=MessageEncoding.JSON,
            compression=CompressionType.NONE,
            optimization_level=optimization_level
        )
        
        # Helper for JSON serialization
        def serializer_helper(obj):
            if isinstance(obj, Enum): # Catches MessageType and other enums
                return obj.value
            # For objects not handled by this, let json.dumps raise the default TypeError
            raise TypeError(f"Object of type {obj.__class__.__name__} is not handled by serializer_helper for JSON encoding")

        # Calculate original JSON size
        try:
            json_data = json.dumps(message, separators=(',', ':'), default=serializer_helper).encode('utf-8')
            metrics.original_json_size = len(json_data)
        except TypeError as e:
            logger.error(f"JSON serialization error in encode_message: {e}")
            # Consider logging parts of the message for debugging, carefully if it contains sensitive data
            logger.debug(f"Message structure that caused serialization error (first 500 chars): {str(message)[:500]}")
            raise
        
        if optimization_level == MessageOptimizationLevel.NONE:
            # No optimization - return JSON
            metrics.optimized_size = metrics.original_json_size
            metrics.total_optimization_time = time.time() - start_time
            return json_data, metrics
        
        # Binary encoding
        if msgpack is None:
            logger.warning("MessagePack not available, falling back to JSON")
            metrics.optimized_size = metrics.original_json_size
            metrics.total_optimization_time = time.time() - start_time
            return json_data, metrics
        
        # Encode with MessagePack
        encoding_start = time.time()
        try:
            binary_data = msgpack.packb(message, use_bin_type=True, default=serializer_helper)
            metrics.encoding_time = time.time() - encoding_start
            metrics.encoding = MessageEncoding.MSGPACK
        except TypeError as e:
            logger.error(f"MessagePack serialization error in encode_message: {e}")
            logger.debug(f"Message structure that caused msgpack serialization error (first 500 chars): {str(message)[:500]}")
            # Fallback to JSON if msgpack fails fundamentally with this structure even with helper
            logger.warning("Falling back to JSON due to MessagePack serialization error.")
            json_data, metrics = self.encode_message(message, MessageOptimizationLevel.NONE) # Recursive call with NO optimization
            return json_data, metrics 
        
        result_data = binary_data
        metrics.optimized_size = len(binary_data)
        
        # Apply compression if enabled and beneficial
        if optimization_level in [MessageOptimizationLevel.COMPRESSED, MessageOptimizationLevel.DELTA, MessageOptimizationLevel.BATCHED]:
            if len(binary_data) >= self.compression_threshold:
                compressed_data, compression_stats = self._compress_data(binary_data)
                if compression_stats.compression_ratio < 0.9:  # Only use if >10% reduction
                    result_data = compressed_data
                    metrics.compression = compression_stats.algorithm
                    metrics.compression_time = compression_stats.compression_time
                    metrics.optimized_size = len(compressed_data)
        
        metrics.total_optimization_time = time.time() - start_time
        metrics.calculate_savings()
        
        # Store metrics for analysis
        self.stats[optimization_level.value].append(metrics)
        
        return result_data, metrics
    
    def _compress_data(self, data: bytes) -> Tuple[bytes, CompressionStats]:
        """Compress data using the best available algorithm."""
        stats = CompressionStats(original_size=len(data))
        
        start_time = time.time()
        
        # Try different compression algorithms
        compression_results = []
        
        # GZIP compression
        try:
            gzip_start = time.time()
            gzip_data = gzip.compress(data, compresslevel=6)
            gzip_time = time.time() - gzip_start
            compression_results.append((
                gzip_data, 
                CompressionType.GZIP, 
                gzip_time
            ))
        except Exception as e:
            logger.warning(f"GZIP compression failed: {e}")
        
        # ZLIB compression
        try:
            zlib_start = time.time()
            zlib_data = zlib.compress(data, level=6)
            zlib_time = time.time() - zlib_start
            compression_results.append((
                zlib_data, 
                CompressionType.ZLIB, 
                zlib_time
            ))
        except Exception as e:
            logger.warning(f"ZLIB compression failed: {e}")
        
        # Choose best compression (smallest size)
        if compression_results:
            best_data, best_algorithm, best_time = min(
                compression_results, 
                key=lambda x: len(x[0])
            )
            
            stats.compressed_size = len(best_data)
            stats.compression_time = best_time
            stats.algorithm = best_algorithm
            stats.calculate_ratio()
            
            return best_data, stats
        
        # No compression worked
        stats.compressed_size = stats.original_size
        stats.compression_time = time.time() - start_time
        stats.calculate_ratio()
        return data, stats
    
    def decode_message(self, data: bytes, encoding: MessageEncoding, compression: CompressionType) -> Dict[str, Any]:
        """Decode a message from binary format."""
        try:
            # Decompress if needed
            if compression != CompressionType.NONE:
                data = self._decompress_data(data, compression)
            
            # Decode based on encoding
            if encoding == MessageEncoding.MSGPACK:
                if msgpack is None:
                    raise ValueError("MessagePack not available for decoding")
                return msgpack.unpackb(data, raw=False)
            else:
                # Assume JSON
                return json.loads(data.decode('utf-8'))
        
        except Exception as e:
            logger.error(f"Failed to decode message: {e}")
            raise
    
    def _decompress_data(self, data: bytes, compression: CompressionType) -> bytes:
        """Decompress data using specified algorithm."""
        if compression == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression == CompressionType.ZLIB:
            return zlib.decompress(data)
        else:
            return data
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        stats = {}
        
        for level, metrics_list in self.stats.items():
            if not metrics_list:
                continue
            
            total_original = sum(m.original_json_size for m in metrics_list)
            total_optimized = sum(m.optimized_size for m in metrics_list)
            total_saved = sum(m.bandwidth_saved for m in metrics_list)
            avg_encoding_time = sum(m.encoding_time for m in metrics_list) / len(metrics_list)
            avg_compression_time = sum(m.compression_time for m in metrics_list) / len(metrics_list)
            
            stats[level] = {
                'message_count': len(metrics_list),
                'total_original_size': total_original,
                'total_optimized_size': total_optimized,
                'total_bandwidth_saved': total_saved,
                'compression_ratio': total_optimized / total_original if total_original > 0 else 1.0,
                'avg_encoding_time': avg_encoding_time,
                'avg_compression_time': avg_compression_time,
                'bandwidth_savings_percent': (total_saved / total_original * 100) if total_original > 0 else 0
            }
        
        return stats


class DeltaCompressor:
    """Delta compression for simulation state updates."""
    
    def __init__(self):
        self.client_states: Dict[str, DeltaState] = {}
        self.delta_threshold = 0.3  # Send full state if delta is >30% of original
    
    def create_delta_update(
        self, 
        client_id: str, 
        simulation_id: str, 
        new_state: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool]:
        """Create a delta update for a client, returns (delta_data, is_full_state)."""
        state_key = f"{client_id}:{simulation_id}"
        
        # Initialize client state if not exists
        if state_key not in self.client_states:
            self.client_states[state_key] = DeltaState(
                client_id=client_id,
                simulation_id=simulation_id
            )
            delta_state = self.client_states[state_key]
            delta_state.last_full_state = new_state.copy()
            delta_state.sequence_number = 1
            return new_state, True  # First update is always full state
        
        delta_state = self.client_states[state_key]
        
        # Calculate delta
        delta = self._calculate_delta(delta_state.last_full_state, new_state)
        
        # Estimate delta size vs full state size
        delta_size = len(json.dumps(delta, separators=(',', ':')))
        full_size = len(json.dumps(new_state, separators=(',', ':')))
        
        # Use full state if delta is too large or too old
        time_since_full = (datetime.now() - delta_state.last_update_time).total_seconds()
        should_send_full = (
            delta_size > full_size * delta_state.delta_threshold or
            time_since_full > 30 or  # Force full state every 30 seconds
            delta_state.sequence_number % 10 == 0  # Full state every 10 updates
        )
        
        if should_send_full:
            delta_state.last_full_state = new_state.copy()
            delta_state.sequence_number += 1
            delta_state.last_update_time = datetime.now()
            
            result = {
                'type': 'full_state',
                'sequence': delta_state.sequence_number,
                'data': new_state
            }
            return result, True
        else:
            delta_state.sequence_number += 1
            delta_state.last_update_time = datetime.now()
            
            result = {
                'type': 'delta',
                'sequence': delta_state.sequence_number,
                'delta': delta,
                'base_sequence': delta_state.sequence_number - 1
            }
            return result, False
    
    def _calculate_delta(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate delta between two states."""
        delta = {
            'added': {},
            'modified': {},
            'removed': []
        }
        
        # Find added and modified items
        for key, new_value in new_state.items():
            if key not in old_state:
                delta['added'][key] = new_value
            elif old_state[key] != new_value:
                delta['modified'][key] = new_value
        
        # Find removed items
        for key in old_state:
            if key not in new_state:
                delta['removed'].append(key)
        
        return delta
    
    def apply_delta(self, base_state: Dict[str, Any], delta: Dict[str, Any]) -> Dict[str, Any]:
        """Apply delta to base state to reconstruct full state."""
        result = base_state.copy()
        
        # Apply additions
        for key, value in delta.get('added', {}).items():
            result[key] = value
        
        # Apply modifications
        for key, value in delta.get('modified', {}).items():
            result[key] = value
        
        # Apply removals
        for key in delta.get('removed', []):
            result.pop(key, None)
        
        return result
    
    def cleanup_client_state(self, client_id: str, simulation_id: str = None):
        """Clean up client state tracking."""
        if simulation_id:
            state_key = f"{client_id}:{simulation_id}"
            self.client_states.pop(state_key, None)
        else:
            # Remove all states for client
            keys_to_remove = [
                key for key in self.client_states.keys() 
                if key.startswith(f"{client_id}:")
            ]
            for key in keys_to_remove:
                self.client_states.pop(key, None)


class MessageBatcher:
    """Message batching for high-frequency updates."""
    
    def __init__(self, max_batch_size: int = 10, max_batch_age: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_batch_age = max_batch_age  # seconds
        self.batches: Dict[str, BatchMessage] = {}
        self.encoder = BinaryMessageEncoder()
    
    def add_message(self, client_id: str, message: Dict[str, Any]) -> Optional[bytes]:
        """Add message to batch, returns encoded batch if ready."""
        if client_id not in self.batches:
            self.batches[client_id] = BatchMessage()
        
        batch = self.batches[client_id]
        batch.add_message(message)
        
        if batch.is_ready(self.max_batch_size, self.max_batch_age):
            # Batch is ready, encode and return
            return self._encode_batch(client_id)
        
        return None
    
    def flush_client_batches(self, client_id: str) -> Optional[bytes]:
        """Force flush batches for a client."""
        if client_id in self.batches and self.batches[client_id].messages:
            return self._encode_batch(client_id)
        return None
    
    def _encode_batch(self, client_id: str) -> bytes:
        """Encode and remove batch for client."""
        batch = self.batches.pop(client_id)
        
        batch_data = {
            'type': 'batch',
            'batch_id': batch.batch_id,
            'message_count': batch.batch_size,
            'messages': batch.messages,
            'timestamp': datetime.now().isoformat()
        }
        
        # Encode with high optimization
        encoded_data, metrics = self.encoder.encode_message(
            batch_data, 
            MessageOptimizationLevel.BATCHED
        )
        
        batch.compression_stats = CompressionStats(
            original_size=metrics.original_json_size,
            compressed_size=metrics.optimized_size,
            compression_ratio=metrics.size_reduction,
            compression_time=metrics.compression_time
        )
        
        return encoded_data
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batching performance statistics."""
        active_batches = len(self.batches)
        total_pending_messages = sum(
            len(batch.messages) for batch in self.batches.values()
        )
        
        return {
            'active_batches': active_batches,
            'total_pending_messages': total_pending_messages,
            'encoder_stats': self.encoder.get_optimization_stats()
        }


class OptimizedWebSocketService:
    """Optimized WebSocket service with binary protocols, compression, and batching."""
    
    def __init__(self):
        self.encoder = BinaryMessageEncoder()
        self.delta_compressor = DeltaCompressor()
        self.message_batcher = MessageBatcher()
        
        # Performance tracking
        self.total_messages_sent = 0
        self.total_bandwidth_saved = 0
        self.optimization_enabled = True
        
        # Client optimization preferences
        self.client_preferences: Dict[str, MessageOptimizationLevel] = {}
    
    def set_client_optimization_level(self, client_id: str, level: MessageOptimizationLevel):
        """Set optimization level for a specific client."""
        self.client_preferences[client_id] = level
    
    def get_client_optimization_level(self, client_id: str) -> MessageOptimizationLevel:
        """Get optimization level for a client."""
        return self.client_preferences.get(client_id, MessageOptimizationLevel.COMPRESSED)
    
    async def send_optimized_message(
        self, 
        websocket, 
        client_id: str, 
        message: Dict[str, Any], 
        use_delta: bool = False,
        use_batching: bool = False
    ) -> MessageMetrics:
        """Send an optimized message to a WebSocket client."""
        start_time = time.time()
        
        if not self.optimization_enabled:
            # Fallback to JSON
            json_data = json.dumps(message)
            await websocket.send_text(json_data)
            
            metrics = MessageMetrics(
                message_id=message.get('id', 'unknown'),
                encoding=MessageEncoding.JSON,
                compression=CompressionType.NONE,
                optimization_level=MessageOptimizationLevel.NONE,
                original_json_size=len(json_data.encode('utf-8')),
                optimized_size=len(json_data.encode('utf-8'))
            )
            return metrics
        
        optimization_level = self.get_client_optimization_level(client_id)
        
        # Apply delta compression if requested and appropriate
        processed_message = message
        is_full_state = True
        
        if use_delta and 'simulation_id' in message:
            simulation_id = message['simulation_id']
            if 'data' in message:
                delta_data, is_full_state = self.delta_compressor.create_delta_update(
                    client_id, simulation_id, message['data']
                )
                processed_message = message.copy()
                processed_message['data'] = delta_data
                processed_message['is_delta'] = not is_full_state
        
        # Handle batching if requested
        if use_batching and optimization_level == MessageOptimizationLevel.BATCHED:
            batch_data = self.message_batcher.add_message(client_id, processed_message)
            if batch_data is not None:
                # Batch is ready, send it
                await websocket.send_bytes(batch_data)
                
                # Create synthetic metrics for batch
                json_size = len(json.dumps(processed_message).encode('utf-8'))
                metrics = MessageMetrics(
                    message_id=processed_message.get('id', 'batch'),
                    encoding=MessageEncoding.MSGPACK,
                    compression=CompressionType.GZIP,
                    optimization_level=optimization_level,
                    original_json_size=json_size,
                    optimized_size=len(batch_data)
                )
                metrics.calculate_savings()
                self._update_performance_stats(metrics)
                return metrics
            else:
                # Message added to batch, not sent yet
                metrics = MessageMetrics(
                    message_id=processed_message.get('id', 'batched'),
                    encoding=MessageEncoding.JSON,
                    compression=CompressionType.NONE,
                    optimization_level=MessageOptimizationLevel.NONE,
                    original_json_size=0,
                    optimized_size=0
                )
                return metrics
        
        # Encode message
        encoded_data, metrics = self.encoder.encode_message(processed_message, optimization_level)
        
        # Send based on encoding
        if metrics.encoding == MessageEncoding.JSON:
            await websocket.send_text(encoded_data.decode('utf-8'))
        else:
            # Create a protocol header for binary messages
            header = self._create_binary_header(
                metrics.encoding, 
                metrics.compression, 
                len(encoded_data)
            )
            await websocket.send_bytes(header + encoded_data)
        
        self._update_performance_stats(metrics)
        return metrics
    
    def _create_binary_header(
        self, 
        encoding: MessageEncoding, 
        compression: CompressionType, 
        data_length: int
    ) -> bytes:
        """Create binary header for protocol identification."""
        # Simple binary header format:
        # [4 bytes: magic number] [1 byte: encoding] [1 byte: compression] [4 bytes: length]
        magic = b'WSOP'  # WebSocket Optimized Protocol
        encoding_byte = {
            MessageEncoding.JSON: 0,
            MessageEncoding.MSGPACK: 1,
            MessageEncoding.BINARY: 2
        }.get(encoding, 0)
        
        compression_byte = {
            CompressionType.NONE: 0,
            CompressionType.GZIP: 1,
            CompressionType.ZLIB: 2,
            CompressionType.DEFLATE: 3
        }.get(compression, 0)
        
        return magic + struct.pack('<BBI', encoding_byte, compression_byte, data_length)
    
    def decode_binary_message(self, data: bytes) -> Dict[str, Any]:
        """Decode a binary message with header."""
        if len(data) < 10:  # Minimum header size
            raise ValueError("Invalid binary message: too short")
        
        # Parse header
        magic = data[:4]
        if magic != b'WSOP':
            raise ValueError("Invalid binary message: bad magic number")
        
        encoding_byte, compression_byte, data_length = struct.unpack('<BBI', data[4:10])
        
        encoding = {
            0: MessageEncoding.JSON,
            1: MessageEncoding.MSGPACK,
            2: MessageEncoding.BINARY
        }.get(encoding_byte, MessageEncoding.JSON)
        
        compression = {
            0: CompressionType.NONE,
            1: CompressionType.GZIP,
            2: CompressionType.ZLIB,
            3: CompressionType.DEFLATE
        }.get(compression_byte, CompressionType.NONE)
        
        # Extract and decode payload
        payload = data[10:10+data_length]
        return self.encoder.decode_message(payload, encoding, compression)
    
    def _update_performance_stats(self, metrics: MessageMetrics):
        """Update performance statistics."""
        self.total_messages_sent += 1
        self.total_bandwidth_saved += metrics.bandwidth_saved
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        encoder_stats = self.encoder.get_optimization_stats()
        batch_stats = self.message_batcher.get_batch_stats()
        
        return {
            'total_messages_sent': self.total_messages_sent,
            'total_bandwidth_saved': self.total_bandwidth_saved,
            'optimization_enabled': self.optimization_enabled,
            'encoder_stats': encoder_stats,
            'batch_stats': batch_stats,
            'active_clients': len(self.client_preferences),
            'delta_compressor_stats': {
                'active_client_states': len(self.delta_compressor.client_states)
            }
        }
    
    def cleanup_client(self, client_id: str):
        """Clean up client-specific optimization state."""
        self.client_preferences.pop(client_id, None)
        self.delta_compressor.cleanup_client_state(client_id)
        self.message_batcher.flush_client_batches(client_id)


# Global optimized service instance
optimized_websocket_service = OptimizedWebSocketService()


def get_optimized_websocket_service() -> OptimizedWebSocketService:
    """Get the global optimized WebSocket service instance."""
    return optimized_websocket_service 