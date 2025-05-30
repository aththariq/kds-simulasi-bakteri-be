"""
Data transformation utilities for API request/response handling.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import numpy as np
from pydantic import BaseModel


class DataTransformer:
    """Utility class for transforming data between different formats."""
    
    @staticmethod
    def serialize_numpy(obj: Any) -> Any:
        """
        Convert numpy types to Python native types for JSON serialization.
        
        Args:
            obj: Object that may contain numpy types
            
        Returns:
            Object with numpy types converted to Python types
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: DataTransformer.serialize_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [DataTransformer.serialize_numpy(item) for item in obj]
        else:
            return obj
    
    @staticmethod
    def format_simulation_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format raw simulation data for API response.
        
        Args:
            raw_data: Raw simulation data from the engine
            
        Returns:
            Formatted data suitable for API response
        """
        formatted = {}
        
        # Handle timestamps
        if 'created_at' in raw_data and isinstance(raw_data['created_at'], datetime):
            formatted['created_at'] = raw_data['created_at'].isoformat()
        
        if 'updated_at' in raw_data and isinstance(raw_data['updated_at'], datetime):
            formatted['updated_at'] = raw_data['updated_at'].isoformat()
        
        # Handle numerical data
        for key, value in raw_data.items():
            if key not in ['created_at', 'updated_at']:
                formatted[key] = DataTransformer.serialize_numpy(value)
        
        return formatted
    
    @staticmethod
    def validate_simulation_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize simulation parameters.
        
        Args:
            params: Raw simulation parameters
            
        Returns:
            Validated and normalized parameters
            
        Raises:
            ValueError: If parameters are invalid
        """
        validated = {}
        
        # Required parameters with validation (matching Pydantic schema)
        required_params = {
            'initial_population_size': (int, 10, 100000),
            'mutation_rate': (float, 0.0, 1.0),
            'selection_pressure': (float, 0.0, 1.0),
            'antibiotic_concentration': (float, 0.0, 1000.0),
            'simulation_time': (int, 1, 1000)
        }
        
        for param_name, (param_type, min_val, max_val) in required_params.items():
            if param_name not in params:
                raise ValueError(f"Missing required parameter: {param_name}")
            
            try:
                value = param_type(params[param_name])
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"Parameter {param_name} must be between {min_val} and {max_val}, got {value}"
                    )
                validated[param_name] = value
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for {param_name}: {e}")
        
        return validated
    
    @staticmethod
    def format_error_response(error: Exception, status_code: int = 500) -> Dict[str, Any]:
        """
        Format error information for API response.
        
        Args:
            error: Exception object
            status_code: HTTP status code
            
        Returns:
            Formatted error response
        """
        return {
            "error": {
                "message": str(error),
                "type": type(error).__name__,
                "status_code": status_code,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    @staticmethod
    def paginate_results(
        data: List[Any], 
        page: int = 1, 
        page_size: int = 10
    ) -> Dict[str, Any]:
        """
        Paginate a list of results.
        
        Args:
            data: List of items to paginate
            page: Page number (1-based)
            page_size: Number of items per page
            
        Returns:
            Paginated results with metadata
        """
        total_items = len(data)
        total_pages = (total_items + page_size - 1) // page_size
        
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_items)
        
        paginated_data = data[start_idx:end_idx]
        
        return {
            "data": paginated_data,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": total_items,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }
    
    @staticmethod
    def compress_simulation_history(history: List[Dict[str, Any]], max_points: int = 100) -> List[Dict[str, Any]]:
        """
        Compress simulation history by sampling points to reduce data size.
        
        Args:
            history: Full simulation history
            max_points: Maximum number of points to keep
            
        Returns:
            Compressed history
        """
        if len(history) <= max_points:
            return history
        
        # Use uniform sampling to reduce points
        indices = np.linspace(0, len(history) - 1, max_points, dtype=int)
        compressed = [history[i] for i in indices]
        
        return compressed
    
    @staticmethod
    def format_websocket_message(
        message_type: str, 
        data: Any, 
        simulation_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Format a message for WebSocket transmission.
        
        Args:
            message_type: Type of the message
            data: Message data
            simulation_id: Optional simulation ID
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            JSON-formatted string ready for WebSocket transmission
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        message = {
            "type": message_type,
            "data": DataTransformer.serialize_numpy(data),
            "timestamp": timestamp.isoformat()
        }
        
        if simulation_id:
            message["simulation_id"] = simulation_id
        
        return json.dumps(message)


class ResponseBuilder:
    """Helper class for building consistent API responses."""
    
    @staticmethod
    def success(data: Any, message: str = "Success") -> Dict[str, Any]:
        """Build a success response."""
        return {
            "success": True,
            "message": message,
            "data": DataTransformer.serialize_numpy(data),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def error(message: str, error_code: str = "UNKNOWN_ERROR", details: Any = None) -> Dict[str, Any]:
        """Build an error response."""
        response = {
            "success": False,
            "error": {
                "message": message,
                "code": error_code,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        if details:
            response["error"]["details"] = DataTransformer.serialize_numpy(details)
        
        return response
    
    @staticmethod
    def simulation_status(
        simulation_id: str,
        status: str,
        progress: float = 0.0,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build a simulation status response."""
        response = {
            "simulation_id": simulation_id,
            "status": status,
            "progress": progress,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if metrics:
            response["metrics"] = DataTransformer.serialize_numpy(metrics)
        
        return response 