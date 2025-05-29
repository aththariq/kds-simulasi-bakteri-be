"""
Error response schemas for API error handling.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ErrorDetail(BaseModel):
    """Individual error detail model."""
    
    type: str = Field(description="Error type")
    message: str = Field(description="Error message")
    field: Optional[str] = Field(description="Field that caused the error", default=None)
    value: Optional[Any] = Field(description="Invalid value that caused the error", default=None)


class ValidationErrorResponse(BaseModel):
    """Response model for validation errors."""
    
    error: str = Field(default="Validation Error", description="Error type")
    message: str = Field(description="General error message")
    details: List[ErrorDetail] = Field(description="List of validation errors")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Validation Error",
                "message": "Request validation failed",
                "details": [
                    {
                        "type": "value_error",
                        "message": "Population size must be at least 10",
                        "field": "initial_population_size",
                        "value": 5
                    }
                ]
            }
        }


class NotFoundErrorResponse(BaseModel):
    """Response model for not found errors."""
    
    error: str = Field(default="Not Found", description="Error type")
    message: str = Field(description="Error message")
    resource: Optional[str] = Field(description="Resource that was not found", default=None)
    id: Optional[str] = Field(description="ID of the resource that was not found", default=None)
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Not Found",
                "message": "Simulation not found",
                "resource": "simulation",
                "id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }


class InternalServerErrorResponse(BaseModel):
    """Response model for internal server errors."""
    
    error: str = Field(default="Internal Server Error", description="Error type")
    message: str = Field(description="Error message")
    timestamp: Optional[str] = Field(description="Error timestamp", default=None)
    request_id: Optional[str] = Field(description="Request ID for tracking", default=None)
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Internal Server Error",
                "message": "An unexpected error occurred while processing the simulation",
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_123456789"
            }
        }


class BadRequestErrorResponse(BaseModel):
    """Response model for bad request errors."""
    
    error: str = Field(default="Bad Request", description="Error type")
    message: str = Field(description="Error message")
    suggestion: Optional[str] = Field(description="Suggestion for fixing the request", default=None)
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Bad Request",
                "message": "Invalid simulation parameters provided",
                "suggestion": "Check the API documentation for valid parameter ranges"
            }
        } 