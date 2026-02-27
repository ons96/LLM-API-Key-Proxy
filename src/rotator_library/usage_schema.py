"""
Usage Tracking Schema for Phase 3.2 Analytics

Defines Pydantic models for type-safe, comprehensive usage tracking
including request metrics, token counts, costs, and error classification.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class RequestStatus(str, Enum):
    """Classification of request completion status."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    CANCELLED = "cancelled"


class ErrorCategory(str, Enum):
    """High-level error classification for analytics."""
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    NETWORK = "network"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


class UsageEvent(BaseModel):
    """
    Individual usage event representing a single API request.
    Stored in rolling window for recent activity analysis.
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_id: str = Field(default_factory=lambda: f"evt_{datetime.utcnow().timestamp()}")
    provider: str
    credential_id: str
    model: Optional[str] = None
    virtual_model: Optional[str] = None  # Router virtual model name if applicable
    
    # Request metadata
    status: RequestStatus
    error_category: Optional[ErrorCategory] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    # Token metrics (if available)
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    
    # Performance metrics
    latency_ms: float = 0.0
    time_to_first_token_ms: Optional[float] = None  # For streaming
    
    # Cost tracking
    cost_usd: float = 0.0
    
    # Routing metadata
    endpoint: Optional[str] = None
    region: Optional[str] = None
    priority: int = 5  # 1 = highest
    rotation_mode: str = "balanced"
    
    # Request context
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    client_ip_hash: Optional[str] = None  # Hashed for privacy
    
    @validator('tokens_total', always=True)
    def calculate_total_tokens(cls, v, values):
        """Ensure tokens_total is calculated if not provided."""
        if v == 0 and values.get('tokens_prompt') and values.get('tokens_completion'):
            return values['tokens_prompt'] + values['tokens_completion']
        return v


class ModelStats(BaseModel):
    """Aggregated statistics per model."""
    model_id: str
    first_used: Optional[datetime] = None
    last_used: Optional[datetime] = None
    
    # Request counts
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    # Token aggregates
    total_tokens_prompt: int = 0
    total_tokens_completion: int = 0
    total_tokens: int = 0
    
    # Cost aggregates
    total_cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0  # For providers without precise billing
    
    # Performance metrics
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    p95_latency_ms: Optional[float] = None  # Calculated periodically
    
    # Error tracking
    error_counts: Dict[str, int] = Field(default_factory=dict)
    
    def update_latency_stats(self, latency_ms: float):
        """Update latency statistics with new measurement."""
        self.total_latency_ms += latency_ms
        self.request_count += 1
        self.avg_latency_ms = self.total_latency_ms / self.request_count
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)


class TimeWindowStats(BaseModel):
    """Statistics for a specific time window (hourly/daily)."""
    window_start: datetime
    window_end: datetime
    request_count: int = 0
    token_count: int = 0
    cost_usd: float = 0.0
    error_count: int = 0
    unique_models: set = Field(default_factory=set)
    unique_credentials: set = Field(default_factory=set)
    
    class Config:
        arbitrary_types_allowed = True


class CredentialStats(BaseModel):
    """
    Comprehensive statistics per credential.
    Replaces and extends the simple usage count in legacy format.
    """
    credential_id: str
    provider: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Current operational state (legacy compatibility)
    current_daily_count: int = 0
    last_reset_date: str = ""  # YYYY-MM-DD format
    cooldown_until: Optional[datetime] = None
    is_active: bool = True
    
    # Aggregate metrics
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    cooldown_count: int = 0
    
    # Token and cost tracking
    total_tokens: int = 0
    total_tokens_prompt: int = 0
    total_tokens_completion: int = 0
    total_cost_usd: float = 0.0
    
    # Temporal tracking
    first_used: Optional[datetime] = None
    last_used: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    
    # Model breakdown
    model_breakdown: Dict[str, ModelStats] = Field(default_factory=dict)
    
    # Error classification breakdown
    error_breakdown: Dict[str, int] = Field(default_factory=dict)
    error_category_breakdown: Dict[str, int] = Field(default_factory=dict)
    
    # Hourly windows for last 24h (rolling)
    hourly_stats: Dict[str, TimeWindowStats] = Field(default_factory=dict)
    
    # Quality metrics
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    success_rate_24h: float = 1.0
    
    def record_event(self, event: UsageEvent):
        """Update stats with a new usage event."""
        self.updated_at = datetime.utcnow()
        self.last_used = event.timestamp
        
        if event.status == RequestStatus.SUCCESS:
            self.success_count += 1
            self.last_success = event.timestamp
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.failure_count += 1
            self.last_failure = event.timestamp
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            
            # Update error breakdowns
            error_key = event.error_code or "unknown"
            self.error_breakdown[error_key] = self.error_breakdown.get(error_key, 0) + 1
            
            if event.error_category:
                cat_key = event.error_category.value
                self.error_category_breakdown[cat_key] = self.error_category_breakdown.get(cat_key, 0) + 1
        
        # Update token counts
        self.total_tokens += event.tokens_total
        self.total_tokens_prompt += event.tokens_prompt
        self.total_tokens_completion += event.tokens_completion
        
        # Update cost
        self.total_cost_usd += event.cost_usd
        
        # Update model breakdown
        if event.model:
            if event.model not in self.model_breakdown:
                self.model_breakdown[event.model] = ModelStats(model_id=event.model)
            
            model_stats = self.model_breakdown[event.model]
            model_stats.last_used = event.timestamp
            if not model_stats.first_used:
                model_stats.first_used = event.timestamp
            
            if event.status == RequestStatus.SUCCESS:
                model_stats.success_count += 1
            else:
                model_stats.failure_count += 1
            
            model_stats.total_tokens_prompt += event.tokens_prompt
            model_stats.total_tokens_completion += event.tokens_completion
            model_stats.total_tokens += event.tokens_total
            model_stats.total_cost_usd += event.cost_usd
            model_stats.update_latency_stats(event.latency_ms)
            
            if event.error_code:
                model_stats.error_counts[event.error_code] = model_stats.error_counts.get(event.error_code, 0) + 1


class DailySummary(BaseModel):
    """Aggregated daily statistics."""
    date: str  # YYYY-MM-DD
    provider_breakdown: Dict[str, int] = Field(default_factory=dict)
    model_breakdown: Dict[str, int] = Field(default_factory=dict)
    status_breakdown: Dict[str, int] = Field(default_factory=dict)
    
    total_requests: int = 0
    total_success: int = 0
    total_failures: int = 0
    
    total_tokens: int = 0
    total_tokens_prompt: int = 0
    total_tokens_completion: int = 0
    
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    
    unique_credentials: int = 0
    unique_models: int = 0


class ProviderAnalytics(BaseModel):
    """Analytics aggregated at the provider level."""
    provider: str
    total_requests: int = 0
    active_credentials: int = 0
    total_credentials: int = 0
    avg_success_rate: float = 0.0
    total_cost_24h: float = 0.0
    models_available: List[str] = Field(default_factory=list)


class UsageSchema(BaseModel):
    """
    Root schema container for all usage analytics data.
    Version 3.2.0 introduces comprehensive event tracking and aggregation.
    """
    schema_version: str = "3.2.0"
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    data_retention_days: int = 30
    
    # Primary data stores
    credentials: Dict[str, CredentialStats] = Field(default_factory=dict)
    daily_summaries: Dict[str, DailySummary] = Field(default_factory=dict)  # Key: YYYY-MM-DD
    provider_analytics: Dict[str, ProviderAnalytics] = Field(default_factory=dict)
    
    # Rolling window of recent events (configurable size, default 1000)
    recent_events: List[UsageEvent] = Field(default_factory=list)
    max_recent_events: int = 1000
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    migration_history: List[Dict] = Field(default_factory=list)
    
    def add_event(self, event: UsageEvent) -> None:
        """Add event to recent events with FIFO overflow."""
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events = self.recent_events[-self.max_recent_events:]
        self.last_updated = datetime.utcnow()
    
    def get_or_create_credential_stats(self, credential_id: str, provider: str) -> CredentialStats:
        """Get existing or create new credential stats."""
        if credential_id not in self.credentials:
            self.credentials[credential_id] = CredentialStats(
                credential_id=credential_id,
                provider=provider
            )
        return self.credentials[credential_id]
    
    def get_daily_summary(self, date_str: str) -> DailySummary:
        """Get or create daily summary for date."""
        if date_str not in self.daily_summaries:
            self.daily_summaries[date_str] = DailySummary(date=date_str)
        return self.daily_summaries[date_str]
    
    def update_provider_analytics(self, provider: str) -> None:
        """Recalculate provider-level analytics from credential data."""
        creds = [c for c in self.credentials.values() if c.provider == provider]
        
        total_reqs = sum(c.request_count for c in creds)
        success_reqs = sum(c.success_count for c in creds)
        
        self.provider_analytics[provider] = ProviderAnalytics(
            provider=provider,
            total_requests=total_reqs,
            active_credentials=sum(1 for c in creds if c.is_active),
            total_credentials=len(creds),
            avg_success_rate=success_reqs / total_reqs if total_reqs > 0 else 0.0
        )


# Legacy compatibility aliases for migration
KeyUsageData = UsageSchema
CredentialUsageData = CredentialStats
