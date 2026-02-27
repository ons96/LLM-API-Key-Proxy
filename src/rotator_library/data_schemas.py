"""Pydantic schemas for data validation across the rotator library."""

from pydantic import BaseModel, Field, HttpUrl, validator
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from enum import Enum


class ModelCreator(BaseModel):
    """Schema for model creator information."""
    name: Optional[str] = None
    slug: Optional[str] = None


class Pricing(BaseModel):
    """Schema for model pricing information."""
    input_price_per_million_tokens: Optional[float] = None
    output_price_per_million_tokens: Optional[float] = None
    
    class Config:
        extra = 'allow'


class Evaluations(BaseModel):
    """Schema for model evaluations - flexible dict structure."""
    artificial_analysis_intelligence_index: Optional[float] = None
    context_window: Optional[int] = None
    
    class Config:
        extra = 'allow'


class ArtificialAnalysisModel(BaseModel):
    """Schema for Artificial Analysis API model data."""
    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable model name")
    slug: str = Field(..., description="URL-friendly model identifier")
    model_creator: Optional[ModelCreator] = None
    evaluations: Optional[Dict[str, Any]] = None
    pricing: Optional[Pricing] = None
    median_output_tokens_per_second: Optional[float] = None
    
    @validator('median_output_tokens_per_second')
    def validate_positive_throughput(cls, v):
        if v is not None and v < 0:
            raise ValueError('Throughput must be positive')
        return v
    
    class Config:
        extra = 'allow'


class ProviderCapability(str, Enum):
    """Enumeration of provider capabilities."""
    CHAT = "chat"
    CODE = "code"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
    IMAGE = "image"


class FreeModel(BaseModel):
    """Schema for free tier model configuration."""
    id: str
    name: Optional[str] = None
    context: int = Field(..., gt=0, description="Context window size")
    rpm: Optional[int] = Field(None, ge=0, description="Requests per minute limit")
    daily_limit: Optional[int] = Field(None, ge=0, description="Daily usage limit")
    note: Optional[str] = None


class Provider(BaseModel):
    """Schema for provider configuration."""
    id: str = Field(..., description="Unique provider identifier")
    name: str = Field(..., description="Display name")
    signup_url: Optional[HttpUrl] = None
    api_base: Optional[str] = None
    free_tier: bool = False
    free_models: List[FreeModel] = []
    capabilities: List[ProviderCapability] = []
    last_verified: Optional[str] = None
    note: Optional[str] = None
    
    @validator('last_verified')
    def validate_date_format(cls, v):
        if v is not None:
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError(f"Invalid ISO date format: {v}")
        return v


class ProvidersDatabase(BaseModel):
    """Schema for the providers database root."""
    providers: List[Provider]
    
    @validator('providers')
    def validate_unique_provider_ids(cls, v):
        ids = [p.id for p in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate provider IDs found")
        return v
    
    @validator('providers')
    def validate_unique_model_ids_per_provider(cls, v):
        for provider in v:
            model_ids = [m.id for m in provider.free_models]
            if len(model_ids) != len(set(model_ids)):
                raise ValueError(f"Duplicate model IDs found in provider {provider.id}")
        return v


class ValidationErrorDetail(BaseModel):
    """Schema for validation error details."""
    loc: List[str]
    msg: str
    type: str


class DataValidationReport(BaseModel):
    """Schema for validation report output."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    record_count: int
    valid_count: int
    error_count: int
    warning_count: int
    errors: List[str] = []
    warnings: List[str] = []
    is_valid: bool = False
