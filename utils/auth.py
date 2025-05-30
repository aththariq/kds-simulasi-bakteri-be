"""
Authentication and validation utilities for API security.
"""

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import hashlib
import secrets
import time
from datetime import datetime, timedelta


security = HTTPBearer(auto_error=False)


class AuthenticationManager:
    """Manages API authentication and authorization."""
    
    def __init__(self):
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
    def generate_api_key(self, client_name: str, rate_limit: int = 100) -> str:
        """
        Generate a new API key for a client.
        
        Args:
            client_name: Name of the client
            rate_limit: Requests per hour limit
            
        Returns:
            Generated API key
        """
        api_key = secrets.token_urlsafe(32)
        
        self.api_keys[api_key] = {
            "client_name": client_name,
            "created_at": datetime.utcnow(),
            "rate_limit": rate_limit,
            "active": True
        }
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if valid, False otherwise
        """
        return api_key in self.api_keys and self.api_keys[api_key]["active"]
    
    def check_rate_limit(self, api_key: str) -> bool:
        """
        Check if API key has exceeded rate limit.
        
        Args:
            api_key: API key to check
            
        Returns:
            True if within limit, False if exceeded
        """
        if api_key not in self.api_keys:
            return False
        
        current_time = time.time()
        hour_ago = current_time - 3600  # 1 hour
        
        if api_key not in self.rate_limits:
            self.rate_limits[api_key] = {"requests": [], "blocked_until": None}
        
        # Clean old requests
        self.rate_limits[api_key]["requests"] = [
            req_time for req_time in self.rate_limits[api_key]["requests"] 
            if req_time > hour_ago
        ]
        
        # Check if still blocked
        if (self.rate_limits[api_key]["blocked_until"] and 
            current_time < self.rate_limits[api_key]["blocked_until"]):
            return False
        
        # Check rate limit
        request_count = len(self.rate_limits[api_key]["requests"])
        rate_limit = self.api_keys[api_key]["rate_limit"]
        
        if request_count >= rate_limit:
            # Block for 1 hour
            self.rate_limits[api_key]["blocked_until"] = current_time + 3600
            return False
        
        # Record this request
        self.rate_limits[api_key]["requests"].append(current_time)
        return True
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            api_key: API key to revoke
            
        Returns:
            True if revoked, False if not found
        """
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            return True
        return False


# Global authentication manager
auth_manager = AuthenticationManager()

# For development, generate a default API key
DEFAULT_API_KEY = auth_manager.generate_api_key("development", rate_limit=1000)


class SecurityValidator:
    """Validates request security and input sanitization."""
    
    @staticmethod
    def sanitize_simulation_id(simulation_id: str) -> str:
        """
        Sanitize and validate simulation ID.
        
        Args:
            simulation_id: Raw simulation ID
            
        Returns:
            Sanitized simulation ID
            
        Raises:
            HTTPException: If ID is invalid
        """
        if not simulation_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Simulation ID cannot be empty"
            )
        
        # Remove any potentially dangerous characters
        sanitized = ''.join(c for c in simulation_id if c.isalnum() or c in '-_')
        
        if len(sanitized) < 10 or len(sanitized) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Simulation ID must be between 10 and 100 characters"
            )
        
        return sanitized
    
    @staticmethod
    def validate_request_size(content_length: Optional[str], max_size: int = 1048576) -> None:
        """
        Validate request content size.
        
        Args:
            content_length: Content length header value
            max_size: Maximum allowed size in bytes
            
        Raises:
            HTTPException: If request is too large
        """
        if content_length:
            try:
                size = int(content_length)
                if size > max_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Request too large. Maximum size: {max_size} bytes"
                    )
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid content length"
                )
    
    @staticmethod
    def generate_request_hash(request_data: Dict[str, Any]) -> str:
        """
        Generate a hash for request deduplication.
        
        Args:
            request_data: Request data to hash
            
        Returns:
            SHA-256 hash of the request
        """
        import json
        
        # Sort keys for consistent hashing
        sorted_data = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify API key from request headers.
    
    Args:
        credentials: Authorization credentials
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        # For development, allow requests without API key
        return DEFAULT_API_KEY
    
    api_key = credentials.credentials
    
    if not auth_manager.validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not auth_manager.check_rate_limit(api_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": "3600"},
        )
    
    return api_key


async def optional_verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """
    Optional API key verification (for public endpoints).
    
    Args:
        credentials: Authorization credentials
        
    Returns:
        API key if provided and valid, None otherwise
    """
    if not credentials:
        return None
    
    try:
        return await verify_api_key(credentials)
    except HTTPException:
        return None


class RequestValidator:
    """Validates common request patterns."""
    
    @staticmethod
    def validate_pagination(page: int = 1, page_size: int = 10) -> tuple[int, int]:
        """
        Validate pagination parameters.
        
        Args:
            page: Page number
            page_size: Items per page
            
        Returns:
            Validated page and page_size
            
        Raises:
            HTTPException: If parameters are invalid
        """
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page number must be >= 1"
            )
        
        if page_size < 1 or page_size > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page size must be between 1 and 100"
            )
        
        return page, page_size
    
    @staticmethod
    def validate_simulation_filters(
        status: Optional[str] = None,
        created_after: Optional[datetime] = None,
        created_before: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Validate simulation filtering parameters.
        
        Args:
            status: Status filter
            created_after: Created after timestamp
            created_before: Created before timestamp
            
        Returns:
            Validated filters
            
        Raises:
            HTTPException: If filters are invalid
        """
        filters = {}
        
        if status:
            valid_statuses = ["pending", "running", "completed", "paused", "error", "cancelled"]
            if status not in valid_statuses:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
                )
            filters["status"] = status
        
        if created_after and created_before:
            if created_after >= created_before:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="created_after must be before created_before"
                )
        
        if created_after:
            filters["created_after"] = created_after
        if created_before:
            filters["created_before"] = created_before
        
        return filters 