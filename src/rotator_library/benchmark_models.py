"""
Phase 2.1 Benchmark Data Schema Models

Pydantic models for type-safe benchmark data handling across the LLM proxy system.
Defines the canonical schema for benchmark records fetched from various sources
(Artificial Analysis, manual entry, etc.) used for routing decisions.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import json


class BenchmarkSource(str, Enum):
    """Enumeration of supported benchmark data sources."""
    ARTIFICIAL_ANALYSIS = "artificial_analysis"
    CUSTOM = "custom"
    MANUAL = "manual"
    OPENLLM = "openllm_leaderboard"
    LMSYS = "lmsys_chatbot_arena"


class ModelCreator(BaseModel):
    """Information about the organization that created the model."""
    name: Optional[str] = Field(None, description="Human-readable organization name")
    slug: Optional[str] = Field(None, description="URL-friendly identifier")
    
    class Config:
        extra = "allow"


class PricingInfo(BaseModel):
    """Pricing information for the model."""
    input_price_per_1k: Optional[float] = Field(
        None, 
        description="Price per 1000 input tokens in USD",
        ge=0
    )
    output_price_per_1k: Optional[float] = Field(
        None, 
        description="Price per 1000 output tokens in USD",
        ge=0
    )
    currency: str = Field("USD", description="ISO 4217 currency code")
    
    # Raw pricing data for provider-specific variations
    raw_pricing: Dict[str, Any] = Field(
        default_factory=dict,
        description="Unprocessed pricing data from source"
    )

    @validator('input_price_per_1k', 'output_price_per_1k', pre=True)
    def normalize_pricing(cls, v):
        """Convert string prices to float, handle None."""
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None
    
    class Config:
        extra = "allow"


class PerformanceMetrics(BaseModel):
    """Throughput and latency metrics."""
    tokens_per_second: Optional[float] = Field(
        None, 
        alias="median_output_tokens_per_second",
        description="Median output tokens per second",
        ge=0
    )
    latency_first_token_ms: Optional[float] = Field(
        None, 
        description="Time to first token in milliseconds",
        ge=0
    )
    latency_first_token_seconds: Optional[float] = Field(
        None,
        description="Time to first token in seconds (original source field)",
        ge=0
    )
    context_window: Optional[int] = Field(
        None, 
        description="Maximum context window size in tokens",
        gt=0
    )
    
    class Config:
        extra = "allow"
        allow_population_by_field_name = True


class EvaluationMetrics(BaseModel):
    """
    Standardized benchmark scores.
    
    Scores are typically 0-100 or 0.0-1.0 depending on the benchmark.
    """
    # Artificial Analysis specific
    artificial_analysis_intelligence_index: Optional[float] = Field(
        None, alias="eval_artificial_analysis_intelligence_index", ge=0, le=100
    )
    
    # Standard academic benchmarks
    mmlu: Optional[float] = Field(None, alias="eval_mmlu", ge=0, le=100)
    hellaswag: Optional[float] = Field(None, alias="eval_hellaswag", ge=0, le=100)
    truthfulqa: Optional[float] = Field(None, alias="eval_truthfulqa", ge=0, le=100)
    winogrande: Optional[float] = Field(None, alias="eval_winogrande", ge=0, le=100)
    gsm8k: Optional[float] = Field(None, alias="eval_gsm8k", ge=0, le=100)
    humaneval: Optional[float] = Field(None, alias="eval_humaneval", ge=0, le=100)
    mbpp: Optional[float] = Field(None, alias="eval_mbpp", ge=0, le=100)
    arc: Optional[float] = Field(None, alias="eval_arc", ge=0, le=100)
    
    # LMSYS Arena specific
    arena_elo: Optional[float] = Field(None, ge=0)
    
    # Storage for any additional evaluations not explicitly modeled
    raw_evaluations: Dict[str, Any] = Field(
        default_factory=dict,
        description="All evaluation data including unmapped fields"
    )
    
    @root_validator(pre=True)
    def extract_raw_evaluations(cls, values):
        """Capture all eval_ prefixed fields into raw_evaluations."""
        raw = {}
        eval_fields = {}
        
        for key, value in list(values.items()):
            if key.startswith('eval_'):
                eval_fields[key] = value
                if key not in raw:
                    raw[key] = value
            elif isinstance(value, (int, float)):
                raw[key] = value
                
        values['raw_evaluations'] = raw
        return values
    
    class Config:
        extra = "allow"
        allow_population_by_field_name = True


class BenchmarkRecord(BaseModel):
    """
    Single model benchmark record conforming to Phase 2.1 schema.
    
    This is the canonical representation of a model's capabilities,
    performance, and pricing used by the router for decision-making.
    """
    
    # === Identification ===
    id: str = Field(..., description="Unique model identifier from source")
    name: str = Field(..., description="Human-readable model name")
    slug: str = Field(..., description="URL-friendly model identifier")
    
    # === Source Metadata ===
    source: BenchmarkSource = Field(
        ..., 
        description="Origin of this benchmark data"
    )
    fetched_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when data was retrieved"
    )
    source_version: Optional[str] = Field(
        None, 
        description="API version or data revision from source"
    )
    source_url: Optional[str] = Field(
        None,
        description="Direct URL to source data if available"
    )
    
    # === Model Metadata ===
    model_creator: Optional[ModelCreator] = None
    release_date: Optional[str] = None
    context_window: Optional[int] = Field(None, gt=0)
    description: Optional[str] = None
    
    # === Benchmark Data ===
    evaluations: EvaluationMetrics = Field(default_factory=EvaluationMetrics)
    pricing: PricingInfo = Field(default_factory=PricingInfo)
    performance: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    
    # === Provider Associations ===
    # Populated by pipeline based on providers_database.yaml matching
    provider_ids: List[str] = Field(
        default_factory=list,
        description="Provider IDs where this model is available"
    )
    provider_model_ids: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of provider_id -> provider-specific model ID"
    )
    
    # === Schema Metadata ===
    schema_version: str = Field("2.1.0", description="Schema version of this record")
    
    # === Raw Data Preservation ===
    _raw_api_response: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            BenchmarkSource: lambda v: v.value
        }
    
    @property
    def is_fresh(self, max_age_hours: int = 168) -> bool:
        """Check if record is younger than max_age_hours (default 1 week)."""
        age = datetime.utcnow() - self.fetched_at
        return age.total_seconds() < (max_age_hours * 3600)
    
    @property
    def intelligence_score(self) -> Optional[float]:
        """Convenience accessor for primary intelligence ranking."""
        return self.evaluations.artificial_analysis_intelligence_index
    
    def to_flattened_dict(self) -> Dict[str, Any]:
        """
        Convert to flat dictionary suitable for CSV export.
        Flattens nested objects with underscore notation.
        """
        result = {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "source": self.source.value,
            "fetched_at": self.fetched_at.isoformat(),
            "source_version": self.source_version,
            "schema_version": self.schema_version,
            "model_creator_name": self.model_creator.name if self.model_creator else None,
            "model_creator_slug": self.model_creator.slug if self.model_creator else None,
            "release_date": self.release_date,
            "context_window": self.context_window,
            "intelligence_index": self.intelligence_score,
            "provider_ids": ",".join(self.provider_ids),
        }
        
        # Flatten evaluations (eval_ prefix)
        if self.evaluations:
            for key, value in self.evaluations.raw_evaluations.items():
                if value is not None:
                    result[key] = value
        
        # Flatten pricing (price_ prefix)
        if self.pricing:
            if self.pricing.input_price_per_1k is not None:
                result["price_input_per_1k"] = self.pricing.input_price_per_1k
            if self.pricing.output_price_per_1k is not None:
                result["price_output_per_1k"] = self.pricing.output_price_per_1k
            for key, value in self.pricing.raw_pricing.items():
                if key not in result:
                    result[f"price_{key}"] = value
        
        # Flatten performance
        if self.performance:
            if self.performance.tokens_per_second:
                result["median_output_tokens_per_second"] = self.performance.tokens_per_second
            if self.performance.latency_first_token_ms:
                result["latency_first_token_ms"] = self.performance.latency_first_token_ms
        
        return result


class BenchmarkDataset(BaseModel):
    """
    Collection of benchmark records with metadata.
    
    Represents a complete dataset fetched from a source,
    suitable for serialization to JSON or database storage.
    """
    schema_version: str = Field("2.1.0", description="Benchmark schema version")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    source: BenchmarkSource
    source_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the fetch operation"
    )
    records: List[BenchmarkRecord] = Field(default_factory=list)
    
    def get_by_id(self, model_id: str) -> Optional[BenchmarkRecord]:
        """Retrieve record by model ID."""
        for record in self.records:
            if record.id == model_id:
                return record
        return None
    
    def get_by_slug(self, slug: str) -> Optional[BenchmarkRecord]:
        """Retrieve record by slug."""
        for record in self.records:
            if record.slug == slug:
                return record
        return None
    
    def filter_by_provider(self, provider_id: str) -> List[BenchmarkRecord]:
        """Get all records available on a specific provider."""
        return [r for r in self.records if provider_id in r.provider_ids]
    
    def filter_fresh(self, max_age_hours: int = 168) -> List[BenchmarkRecord]:
        """Return only records newer than max_age_hours."""
        cutoff = datetime.utcnow() - __import__('datetime').timedelta(hours=max_age_hours)
        return [r for r in self.records if r.fetched_at > cutoff]
    
    def to_dataframe(self):
        """Convert to pandas DataFrame (requires pandas)."""
        import pandas as pd
        data = [r.to_flattened_dict() for r in self.records]
        return pd.DataFrame(data)
    
    def save_json(self, filepath: str):
        """Save dataset to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.dict(), f, indent=2, default=str)
    
    @classmethod
    def load_json(cls, filepath: str) -> "BenchmarkDataset":
        """Load dataset from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)


class BenchmarkSchemaConfig(BaseModel):
    """
    Configuration for benchmark data schema validation and mapping.
    Loaded from config/benchmark_schema.yaml.
    """
    schema_version: str
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    field_mappings: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    normalization_rules: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"
