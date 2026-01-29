"""
Core Router Module for Multi-Provider LLM Routing

Implements $0-only routing with virtual models, MoE support, and web-search augmentation.
Maintains OpenAI API compatibility throughout.
"""

import asyncio
import time
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, AsyncGenerator, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yaml
from pathlib import Path
import os

import litellm
from fastapi import HTTPException
import httpx


logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider health status."""

    HEALTHY = "healthy"
    COOLDOWN = "cooldown"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ErrorCategory(Enum):
    """Error classification for routing decisions."""

    TRANSIENT = "transient"  # Retry same provider
    RATE_LIMIT = "rate_limit"  # Cooldown provider, try next
    AUTH_ERROR = "auth_error"  # Disable provider
    INVALID_REQUEST = "invalid_request"  # Fail immediately
    PROVIDER_ERROR = "provider_error"  # Try next provider


@dataclass
class CapabilityRequirements:
    """Requirements extracted from incoming request."""

    needs_tools: bool = False
    needs_vision: bool = False
    needs_structured_output: bool = False
    needs_long_context: bool = False
    min_context_tokens: int = 0
    streaming: bool = False
    search_requested: bool = False
    moe_mode: bool = False


@dataclass
class ProviderMetrics:
    """Tracked metrics per provider/model combination."""

    cooldown_until: float = 0.0
    consecutive_failures: int = 0
    last_success_ts: float = 0.0
    last_error_ts: float = 0.0
    ewma_latency_ms: float = 0.0
    total_requests: int = 0
    total_errors: int = 0
    success_rate: float = 1.0

    def update_latency(self, latency_ms: float, alpha: float = 0.3):
        """Update EWMA latency."""
        if self.ewma_latency_ms == 0:
            self.ewma_latency_ms = latency_ms
        else:
            self.ewma_latency_ms = (alpha * latency_ms) + (
                (1 - alpha) * self.ewma_latency_ms
            )

    def record_success(self):
        """Record successful request."""
        self.last_success_ts = time.time()
        self.total_requests += 1
        self.consecutive_failures = 0
        self.update_success_rate()

    def record_error(self):
        """Record failed request."""
        self.last_error_ts = time.time()
        self.total_requests += 1
        self.total_errors += 1
        self.consecutive_failures += 1
        self.update_success_rate()

    def update_success_rate(self):
        """Update success rate."""
        if self.total_requests > 0:
            self.success_rate = (
                self.total_requests - self.total_errors
            ) / self.total_requests

    def is_healthy(self, current_time: Optional[float] = None) -> bool:
        """Check if provider is healthy (not in cooldown)."""
        if current_time is None:
            current_time = time.time()
        return current_time >= self.cooldown_until

    def set_cooldown(self, duration_seconds: int):
        """Set provider in cooldown."""
        self.cooldown_until = time.time() + duration_seconds
        logger.info(f"Provider set to cooldown for {duration_seconds}s")


@dataclass
class ProviderCandidate:
    """A provider candidate for routing."""

    provider: str
    model: str
    priority: int = 5
    capabilities: set = field(default_factory=set)
    fallback_only: bool = False
    role: Optional[str] = None  # For MoE mode
    search_enabled: bool = False
    free_tier_only: bool = True

    def matches_requirements(self, requirements: CapabilityRequirements) -> bool:
        """Check if candidate matches capability requirements."""
        if (
            requirements.needs_tools
            and "tools" not in self.capabilities
            and "function_calling" not in self.capabilities
        ):
            return False
        if requirements.needs_vision and "vision" not in self.capabilities:
            return False
        if (
            requirements.needs_structured_output
            and "structured_output" not in self.capabilities
        ):
            return False
        if requirements.needs_long_context and "long_context" not in self.capabilities:
            return False
        return True


@dataclass
class SearchProviderConfig:
    """Configuration for search providers."""

    name: str
    enabled: bool
    priority: int
    free_tier_only: bool
    paid_available: bool = False
    client: Optional[Any] = None


class SearchProvider:
    """Base class for search providers."""

    def __init__(self, config: SearchProviderConfig, api_key: Optional[str] = None):
        self.config = config
        self.api_key = api_key
        self.metrics = ProviderMetrics()

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Execute search query."""
        raise NotImplementedError

    def is_available(self, free_only_mode: bool) -> bool:
        """Check if provider is available given FREE_ONLY_MODE."""
        if not self.config.enabled:
            return False
        if free_only_mode and not self.config.free_tier_only:
            return False
        return self.api_key is not None


class BraveSearchProvider(SearchProvider):
    """Brave Search API provider."""

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Execute Brave search."""
        if not self.api_key:
            raise ValueError("Brave API key not configured")

        start_time = time.time()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": max_results},
                    headers={"X-Subscription-Token": self.api_key},
                )
                response.raise_for_status()
                data = response.json()

                latency_ms = (time.time() - start_time) * 1000
                self.metrics.update_latency(latency_ms)
                self.metrics.record_success()

                # Extract results
                results = []
                for result in data.get("web", {}).get("results", [])[:max_results]:
                    results.append(
                        {
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "description": result.get("description"),
                            "source": "brave",
                        }
                    )

                return results

        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Brave search failed: {e}")
            raise


class TavilySearchProvider(SearchProvider):
    """Tavily Search API provider."""

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Execute Tavily search."""
        if not self.api_key:
            raise ValueError("Tavily API key not configured")

        start_time = time.time()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max_results,
                        "include_answer": True,
                    },
                )
                response.raise_for_status()
                data = response.json()

                latency_ms = (time.time() - start_time) * 1000
                self.metrics.update_latency(latency_ms)
                self.metrics.record_success()

                # Extract results
                results = []
                for result in data.get("results", [])[:max_results]:
                    results.append(
                        {
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "content": result.get("content"),
                            "source": "tavily",
                        }
                    )

                return results

        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Tavily search failed: {e}")
            raise


class DuckDuckGoSearchProvider(SearchProvider):
    """DuckDuckGo Instant Answer API provider (free, no API key required)."""

    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        start_time = time.time()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_html": 1},
                )
                response.raise_for_status()
                data = response.json()

                latency_ms = (time.time() - start_time) * 1000
                self.metrics.update_latency(latency_ms)
                self.metrics.record_success()

                results = []

                abstract = data.get("Abstract")
                if abstract:
                    results.append(
                        {
                            "title": data.get("Heading") or "Instant Answer",
                            "url": data.get("AbstractURL")
                            or data.get("AbstractSource"),
                            "content": abstract,
                            "description": data.get("AbstractText"),
                            "source": "duckduckgo",
                        }
                    )

                related_topics = data.get("RelatedTopics", [])
                for topic in related_topics[: max_results - len(results)]:
                    if isinstance(topic, dict):
                        text = topic.get("Text")
                        if text:
                            url = topic.get("FirstURL") if "FirstURL" in topic else None
                            results.append(
                                {
                                    "title": topic.get("FirstURL", text).split("/")[-1],
                                    "url": url,
                                    "content": text,
                                    "description": text,
                                    "source": "duckduckgo",
                                }
                            )

                return results[:max_results]

        except Exception as e:
            self.metrics.record_error()
            logger.error(f"DuckDuckGo search failed: {e}")
            raise


from .rate_limiter import RateLimitTracker
from .model_ranker import ModelRanker
from .provider_adapter import ProviderAdapterFactory


class RouterCore:
    def __init__(self, config_path: str = "config/router_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(self.config_path)

        self.free_only_mode = self.config.get("free_only_mode", True)

        self.provider_metrics: Dict[Tuple[str, str], ProviderMetrics] = {}
        self.metrics = (
            self.provider_metrics
        )  # Alias for backward compatibility if needed

        self.aliases: Dict[str, Any] = {}
        self.virtual_models: Dict[str, Dict[str, Any]] = {}
        self.search_providers: Dict[str, Any] = {}

        self.rate_limiter = RateLimitTracker()
        self.model_ranker = ModelRanker()

        self._initialize_components()

    # ... existing code ...

    def _update_metrics(
        self,
        provider: str,
        model: str,
        latency_ms: float,
        success: bool,
        error_type: Optional[ErrorCategory] = None,
    ):
        """Update metrics for a provider."""
        metrics = self._get_metrics(provider, model)
        if success:
            metrics.record_success()
            metrics.update_latency(latency_ms)
        else:
            metrics.record_error()
            if error_type == ErrorCategory.RATE_LIMIT:
                metrics.set_cooldown(
                    self.config.get("routing", {}).get(
                        "rate_limit_cooldown_seconds", 300
                    )
                )

    async def _check_conditions(
        self, candidate_cfg: Dict[str, Any], provider: str, model: str
    ) -> bool:
        """Check if candidate conditions are met using RateLimitTracker."""
        conditions = candidate_cfg.get("conditions", {})
        if not conditions:
            # Check configured provider-level limits if any
            # (Logic to look up provider-specific defaults could go here)
            return True

        # Map config keys to RateLimitTracker keys
        limits = {}
        if "max_rpm" in conditions:
            limits["rpm"] = conditions["max_rpm"]
        if "max_daily_requests" in conditions:
            limits["daily"] = conditions["max_daily_requests"]

        can_use = await self.rate_limiter.can_use_provider(provider, model, limits)
        return can_use

    async def _execute_single_candidate(
        self, candidate: ProviderCandidate, request: Dict[str, Any], request_id: str
    ) -> Dict[str, Any]:
        """Execute request against a single candidate."""
        # Record request attempt
        await self.rate_limiter.record_request(candidate.provider, candidate.model)

        start_time = time.time()
        try:
            # Handle search if needed and enabled
            if candidate.search_enabled and request.get("_search_query"):
                # ... (search logic would go here)
                pass

            # Prepare request
            # Remove router-specific fields
            request_clean = {k: v for k, v in request.items() if not k.startswith("_")}
            request_clean["model"] = f"{candidate.provider}/{candidate.model}"

            logger.info(
                f"[{request_id}] Executing via {candidate.provider}/{candidate.model}"
            )

            # Check if we have a custom adapter for this provider
            supported_providers = ProviderAdapterFactory.list_supported_providers()
            if candidate.provider in supported_providers:
                # Get API key from environment
                api_key = None
                if candidate.provider == "groq":
                    api_key = os.getenv("GROQ_API_KEY")
                elif candidate.provider == "gemini":
                    api_key = os.getenv("GEMINI_API_KEY")

                # Create adapter
                adapter = ProviderAdapterFactory.create_adapter(
                    candidate.provider, api_key
                )

                # Execute via adapter
                response = await adapter.chat_completions(request_clean)
            else:
                # Fallback to LiteLLM direct usage
                # Special handling for G4F if not in adapter factory (but it is now)
                if candidate.provider == "g4f":
                    # This path shouldn't be reached if G4F is in factory
                    pass

                # Execute
                response = await litellm.acompletion(**request_clean, stream=False)

            # Record success
            self._update_metrics(
                candidate.provider, candidate.model, time.time() - start_time, True
            )
            return response

        except Exception as e:
            # Record failure
            error_type = (await self._classify_error(e))[0]

            if error_type == ErrorCategory.RATE_LIMIT:
                await self.rate_limiter.record_rate_limit_hit(
                    candidate.provider, candidate.model
                )

            self._update_metrics(
                candidate.provider,
                candidate.model,
                time.time() - start_time,
                False,
                error_type,
            )
            logger.warning(
                f"[{request_id}] Candidate {candidate.provider}/{candidate.model} failed: {e} ({error_type.value})"
            )
            raise e

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            # Return default config
            return {
                "free_only_mode": True,
                "routing": {
                    "default_cooldown_seconds": 60,
                    "rate_limit_cooldown_seconds": 300,
                },
            }

    def _initialize_components(self):
        """Initialize all router components."""
        self._initialize_search_providers()
        self._initialize_virtual_models()
        self._load_new_virtual_models()  # NEW: Load from config/virtual_models.yaml
        self._load_aliases()  # NEW: Load aliases from config/aliases.yaml
        logger.info(f"Router initialized with FREE_ONLY_MODE={self.free_only_mode}")

    def _load_new_virtual_models(self):
        """Load virtual models from config/virtual_models.yaml."""
        root_dir = Path(__file__).resolve().parent.parent.parent
        vm_path = root_dir / "config" / "virtual_models.yaml"

        if vm_path.exists():
            try:
                with open(vm_path, "r") as f:
                    data = yaml.safe_load(f)
                    new_models = data.get("virtual_models", {})
                    # Convert new format to internal format if needed, or just merge
                    # The internal format expects a dictionary of model_id -> config
                    # The new format is model_id -> {description, fallback_chain, settings}
                    # The old format used 'candidates' instead of 'fallback_chain'.
                    # We can normalize this.
                    for model_id, model_cfg in new_models.items():
                        if "fallback_chain" in model_cfg:
                            # Map fallback_chain to candidates for compatibility
                            model_cfg["candidates"] = model_cfg["fallback_chain"]
                        self.virtual_models[model_id] = model_cfg

                logger.info(f"Loaded {len(new_models)} virtual models from {vm_path}")
            except Exception as e:
                logger.error(f"Failed to load virtual models: {e}")
        else:
            logger.warning(f"Virtual models config not found at {vm_path}")

    def _load_aliases(self):
        """Load virtual model aliases from config/aliases.yaml."""
        # Use path relative to this file to be CWD-independent
        root_dir = Path(__file__).resolve().parent.parent.parent
        alias_path = root_dir / "config" / "aliases.yaml"

        self.aliases = {}
        if alias_path.exists():
            try:
                with open(alias_path, "r") as f:
                    data = yaml.safe_load(f)
                    self.aliases = data.get("aliases", {})
                logger.info(f"Loaded {len(self.aliases)} aliases from {alias_path}")
            except Exception as e:
                logger.error(f"Failed to load aliases: {e}")
        else:
            logger.warning(f"Alias config not found at {alias_path}")

    def _resolve_alias(self, model_id: str) -> List[Dict[str, str]]:
        """Resolve a model alias to a list of candidate dicts."""
        if model_id in self.aliases:
            return self.aliases[model_id].get("candidates", [])
        return []

    def _initialize_search_providers(self):
        """Initialize search providers."""
        search_config = self.config.get("search", {})
        provider_configs = search_config.get("providers", [])

        for provider_cfg in provider_configs:
            name = provider_cfg.get("name")
            enabled = provider_cfg.get("enabled", False)

            if not enabled:
                continue

            config = SearchProviderConfig(
                name=name,
                enabled=enabled,
                priority=provider_cfg.get("priority", 1),
                free_tier_only=provider_cfg.get("free_only", True),
                paid_available=provider_cfg.get("paid_available", False),
            )

            # Get API key from environment
            env_var_map = {
                "brave": "BRAVE_API_KEY",
                "tavily": "TAVILY_API_KEY",
                "exa": "EXA_API_KEY",
            }

            api_key = os.getenv(env_var_map.get(name, ""))

            if name == "brave":
                self.search_providers[name] = BraveSearchProvider(config, api_key)
            elif name == "tavily":
                self.search_providers[name] = TavilySearchProvider(config, api_key)
            else:
                logger.warning(f"Unknown search provider: {name}")

        logger.info(f"Initialized {len(self.search_providers)} search providers")

    def _initialize_virtual_models(self):
        """Initialize virtual router models."""
        self.virtual_models = self.config.get("router_models", {})
        logger.info(f"Initialized {len(self.virtual_models)} virtual models")

    def get_virtual_model_names(self) -> List[str]:
        """Get list of virtual model names."""
        return list(self.virtual_models.keys())

    def _get_metrics(self, provider: str, model: str) -> ProviderMetrics:
        """Get or create metrics for provider/model pair."""
        key = (provider, model)
        if key not in self.provider_metrics:
            self.provider_metrics[key] = ProviderMetrics()
        return self.provider_metrics[key]

    async def _classify_error(
        self, error: Exception
    ) -> Tuple[ErrorCategory, Optional[int]]:
        """Classify error and determine retry behavior."""
        error_str = str(error).lower()

        # Rate limit errors
        if any(
            keyword in error_str
            for keyword in ["rate_limit", "too many requests", "429"]
        ):
            retry_after = None
            response = getattr(error, "response", None)
            if response:
                retry_after = response.headers.get("retry-after")
                if retry_after:
                    try:
                        retry_after = int(retry_after)
                    except ValueError:
                        retry_after = 60  # Default 1 minute
            return ErrorCategory.RATE_LIMIT, retry_after or 60

        # Authentication errors
        if any(
            keyword in error_str
            for keyword in ["unauthorized", "invalid_api_key", "401", "403"]
        ):
            return ErrorCategory.AUTH_ERROR, None

        # Transient errors
        if any(
            keyword in error_str
            for keyword in ["timeout", "connection", "503", "502", "504"]
        ):
            return ErrorCategory.TRANSIENT, None

        # Invalid request
        if any(keyword in error_str for keyword in ["bad request", "invalid", "400"]):
            return ErrorCategory.INVALID_REQUEST, None

        # Default to provider error
        return ErrorCategory.PROVIDER_ERROR, None

    def _extract_requirements(self, request: Dict[str, Any]) -> CapabilityRequirements:
        """Extract capability requirements from request."""
        req = CapabilityRequirements()

        # Check for tools/functions
        if "tools" in request or "functions" in request:
            req.needs_tools = True

        # Check for vision (image content)
        messages = request.get("messages", [])
        for message in messages:
            if isinstance(message.get("content"), list):
                for content_item in message["content"]:
                    if (
                        isinstance(content_item, dict)
                        and content_item.get("type") == "image_url"
                    ):
                        req.needs_vision = True
                        break

        # Check for structured output
        if "response_format" in request and request["response_format"] != {
            "type": "text"
        }:
            req.needs_structured_output = True

        # Check streaming
        req.streaming = request.get("stream", False)

        # Check MoE mode
        model = request.get("model", "")
        if model in self.virtual_models and self.virtual_models[model].get("moe_mode"):
            req.moe_mode = True

        # Check for search indicators in prompt
        last_message = messages[-1] if messages else {}
        if isinstance(last_message.get("content"), str):
            content = last_message["content"].lower()
            search_indicators = [
                "latest",
                "recent",
                "current",
                "news",
                "shopping",
                "best",
                "top",
                "compare",
                "vs",
                "sources",
                "citations",
            ]
            req.search_requested = any(
                indicator in content for indicator in search_indicators
            )

        return req

    def _get_candidates(
        self, model_id: str, requirements: CapabilityRequirements
    ) -> List[ProviderCandidate]:
        """Get candidate providers for the requested model."""
        candidates = []

        if model_id in self.virtual_models:
            # Virtual model - get configured candidates
            virtual_cfg = self.virtual_models[model_id]
            # Support both 'candidates' (old) and 'fallback_chain' (new) keys
            chain = virtual_cfg.get("fallback_chain", virtual_cfg.get("candidates", []))

            # [NEW] Apply Intelligent Ranking if enabled
            # We can toggle this via config or assume it's always on for virtual models
            # Let's check a flag 'auto_order' in virtual model config
            if virtual_cfg.get("auto_order", False):
                chain = self.model_ranker.rank_candidates(model_id, chain)

            for candidate_cfg in chain:
                # Check FREE_ONLY_MODE restrictions
                if self.free_only_mode and not candidate_cfg.get(
                    "free_tier_only", True
                ):
                    continue

                # Check forbidden providers
                forbidden = self.config.get("safety", {}).get(
                    "forbidden_providers_under_free_mode", []
                )
                if self.free_only_mode and candidate_cfg["provider"] in forbidden:
                    continue

                # Check conditions (NEW)
                if not self._check_conditions(
                    candidate_cfg, candidate_cfg["provider"], candidate_cfg["model"]
                ):
                    continue

                candidate = ProviderCandidate(
                    provider=candidate_cfg["provider"],
                    model=candidate_cfg["model"],
                    priority=candidate_cfg.get("priority", 5),
                    capabilities=set(candidate_cfg.get("capabilities", [])),
                    fallback_only=candidate_cfg.get("fallback_only", False),
                    role=candidate_cfg.get("role"),
                    search_enabled=candidate_cfg.get("search_enabled", False),
                )

                if candidate.matches_requirements(requirements):
                    candidates.append(candidate)
        else:
            # Direct model reference - parse provider/model format
            if "/" in model_id:
                provider, model = model_id.split("/", 1)
                candidates.append(
                    ProviderCandidate(provider=provider, model=model, priority=5)
                )
            else:
                # Assume provider-agnostic model, check all providers
                for provider_name in ["groq", "gemini", "g4f"]:
                    candidates.append(
                        ProviderCandidate(
                            provider=provider_name, model=model_id, priority=5
                        )
                    )

        # Sort by priority
        candidates.sort(key=lambda c: c.priority)
        return candidates

    def _should_perform_search(self, requirements: CapabilityRequirements) -> bool:
        """Determine if search should be performed."""
        if not self.search_providers:
            return False

        search_config = self.config.get("search", {})
        if (
            not search_config.get("default_enabled", False)
            and not requirements.search_requested
        ):
            return False

        return True

    async def _perform_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform search using available providers."""
        if not self._should_perform_search(CapabilityRequirements()):
            return []

        # Sort providers by priority
        available_providers = [
            p
            for p in self.search_providers.values()
            if p.is_available(self.free_only_mode) and p.metrics.is_healthy()
        ]
        available_providers.sort(key=lambda p: p.config.priority)

        for provider in available_providers:
            try:
                results = await provider.search(query, max_results=5)
                logger.info(f"Search successful via {provider.config.name}")
                return results
            except Exception as e:
                logger.warning(f"Search via {provider.config.name} failed: {e}")
                provider.metrics.record_error()
                provider.metrics.set_cooldown(60)  # 1 minute cooldown

        return []

    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for prompt injection."""
        if not results:
            return ""

        formatted = "\n\n<search_results>\n"
        for i, result in enumerate(results, 1):
            formatted += f"[{i}] {result.get('title', 'No title')}\n"
            formatted += f"URL: {result.get('url', 'No URL')}\n"
            desc = result.get("description") or result.get("content", "")
            if desc:
                formatted += f"Description: {desc[:200]}...\n"
            formatted += "\n"
        formatted += "</search_results>\n"

        return formatted

    async def execute_moe(
        self, router_cfg: Dict[str, Any], request: Dict[str, Any], request_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute MoE (Mixture of Experts) mode."""
        max_experts = router_cfg.get("max_experts", 3)
        aggregator_model = router_cfg.get(
            "aggregator_model", "groq/llama-3.3-70b-versatile"
        )

        # Filter expert candidates
        expert_candidates = [
            c
            for c in self._get_candidates(
                request.get("model", ""), self._extract_requirements(request)
            )
            if c.role and "expert" in c.role
        ]

        # Limit to max_experts
        expert_candidates = expert_candidates[:max_experts]

        if not expert_candidates:
            raise ValueError("No expert candidates available for MoE mode")

        logger.info(f"MoE mode: Running {len(expert_candidates)} experts")

        # Execute experts in parallel
        expert_tasks = []
        for i, candidate in enumerate(expert_candidates):
            expert_request = request.copy()
            expert_request["model"] = f"{candidate.provider}/{candidate.model}"
            expert_request["stream"] = False  # Non-streaming for experts

            task = asyncio.create_task(
                self._execute_single_candidate(
                    candidate, expert_request, f"{request_id}-expert-{i}"
                ),
                name=f"expert-{i}-{candidate.provider}",
            )
            expert_tasks.append((candidate, task))

        expert_results = []
        for candidate, task in expert_tasks:
            try:
                result = await task
                expert_results.append(
                    {
                        "provider": candidate.provider,
                        "model": candidate.model,
                        "role": candidate.role,
                        "output": result["choices"][0]["message"]["content"]
                        if result.get("choices")
                        else "",
                    }
                )
            except Exception as e:
                logger.error(f"Expert {candidate.provider} failed: {e}")
                expert_results.append(
                    {
                        "provider": candidate.provider,
                        "model": candidate.model,
                        "role": candidate.role,
                        "error": str(e),
                    }
                )

        # Prepare aggregator prompt
        aggregator_messages = [
            {
                "role": "system",
                "content": "You are an expert aggregator synthesizing responses from multiple AI experts. "
                "Analyze the different expert opinions below and provide a comprehensive, "
                "well-reasoned synthesis. Note any disagreements and explain the consensus or "
                "majority view. If some experts failed, acknowledge this and work with "
                "the available responses.",
            },
            {
                "role": "user",
                "content": f"Original question: {request.get('messages', [{}])[-1].get('content', '')}\n\n"
                f"<expert_responses>\n"
                f"{json.dumps(expert_results, indent=2)}\n"
                f"</expert_responses>",
            },
        ]

        aggregator_request = {
            "model": aggregator_model,
            "messages": aggregator_messages,
            "stream": request.get("stream", False),
            "temperature": 0.3,  # Lower temperature for synthesis
        }

        # Execute aggregator
        provider_name, model_name = aggregator_model.split("/", 1)
        aggregator_candidate = next(
            (
                c
                for c in self._get_candidates(
                    aggregator_model, self._extract_requirements(aggregator_request)
                )
                if c.provider == provider_name and c.model == model_name
            ),
            None,
        )

        if not aggregator_candidate:
            raise ValueError(f"Aggregator model {aggregator_model} not available")

        # Stream aggregator response
        if aggregator_request["stream"]:
            async for chunk in self._execute_single_candidate_stream(
                aggregator_candidate, aggregator_request, f"{request_id}-aggregator"
            ):
                yield chunk
        else:
            result = await self._execute_single_candidate(
                aggregator_candidate, aggregator_request, f"{request_id}-aggregator"
            )
            yield result

    async def _execute_single_candidate_stream(
        self, candidate: ProviderCandidate, request: Dict[str, Any], request_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a single candidate (streaming)."""
        start_time = time.time()
        metrics = self._get_metrics(candidate.provider, candidate.model)

        # Record attempt
        await self.rate_limiter.record_request(candidate.provider, candidate.model)

        try:
            # Update model reference
            request = request.copy()
            request["model"] = f"{candidate.provider}/{candidate.model}"
            request_clean = {k: v for k, v in request.items() if not k.startswith("_")}

            logger.info(
                f"[{request_id}] Streaming {candidate.provider}/{candidate.model}"
            )

            # Execute via LiteLLM
            response = await litellm.acompletion(**request_clean, stream=True)

            chunk_count = 0
            async for chunk in response:
                chunk_count += 1
                # Convert chunk to dict if necessary
                if hasattr(chunk, "dict"):
                    chunk_dict = chunk.dict()
                elif hasattr(chunk, "__dict__"):
                    chunk_dict = chunk.__dict__
                else:
                    chunk_dict = chunk

                if chunk_dict.get("choices"):
                    pass  # We could track final response here if needed

                # Check for error in chunk
                if "error" in chunk_dict:
                    raise Exception(
                        f"Provider returned error in stream: {chunk_dict['error']}"
                    )

                yield chunk_dict

            if chunk_count == 0:
                raise Exception("Stream ended with no chunks")

            latency_ms = (time.time() - start_time) * 1000
            self._update_metrics(candidate.provider, candidate.model, latency_ms, True)

            logger.info(
                f"[{request_id}] Stream success: {candidate.provider}/{candidate.model} ({latency_ms:.1f}ms)"
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            error_classification = await self._classify_error(e)
            error_category = error_classification[0]
            retry_after = error_classification[1]

            if retry_after:
                metrics.set_cooldown(retry_after)
            elif error_category == ErrorCategory.RATE_LIMIT:
                metrics.set_cooldown(
                    self.config.get("routing", {}).get(
                        "rate_limit_cooldown_seconds", 300
                    )
                )

            self._update_metrics(
                candidate.provider,
                candidate.model,
                latency_ms,
                False,
                error_category,
            )

            logger.error(
                f"[{request_id}] Stream error {candidate.provider}/{candidate.model}: {e} ({error_category.value})"
            )
            raise e

    async def _stream_with_fallback(
        self,
        candidates: List[ProviderCandidate],
        request: Dict[str, Any],
        request_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute streaming with fallback logic."""
        last_error = None

        for candidate in candidates:
            try:
                # Search injection logic
                request_to_use = request
                if candidate.search_enabled:
                    search_query = request.get("messages", [{}])[-1].get("content", "")
                    if isinstance(search_query, str) and len(search_query) > 0:
                        search_results = await self._perform_search(search_query)
                        if search_results:
                            modified_request = request.copy()
                            modified_messages = modified_request.get(
                                "messages", []
                            ).copy()
                            last_message = modified_messages[-1].copy()
                            search_content = self._format_search_results(search_results)
                            if isinstance(last_message.get("content"), str):
                                last_message["content"] += search_content
                            elif isinstance(last_message.get("content"), list):
                                last_message["content"].append(
                                    {"type": "text", "text": search_content}
                                )
                            modified_messages[-1] = last_message
                            modified_request["messages"] = modified_messages
                            request_to_use = modified_request

                async for chunk in self._execute_single_candidate_stream(
                    candidate, request_to_use, request_id
                ):
                    yield chunk

                return

            except Exception as e:
                last_error = e
                error_classification = await self._classify_error(e)
                error_category = error_classification[0]

                if error_category in [
                    ErrorCategory.INVALID_REQUEST,
                    ErrorCategory.AUTH_ERROR,
                ]:
                    raise e

                logger.warning(
                    f"[{request_id}] Stream candidate {candidate.provider} failed, trying next..."
                )
                continue

        if last_error:
            raise last_error
        raise HTTPException(status_code=503, detail="All stream providers failed")

    async def route_request(
        self, request: Dict[str, Any], request_id: str
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Main routing entry point."""
        model_id = request.get("model", "")

        # --- NEW: Alias Resolution & Chain Fallback Logic ---
        alias_candidates = self._resolve_alias(model_id)
        if alias_candidates:
            logger.info(
                f"[{request_id}] Resolving alias '{model_id}' to {len(alias_candidates)} candidates"
            )
            last_error = None

            for candidate_cfg in alias_candidates:
                # Construct temporary request for this candidate
                candidate_request = request.copy()
                candidate_provider = candidate_cfg["provider"]
                candidate_model = candidate_cfg["model"]

                # Override model field to be explicit "provider/model" for internal routing
                candidate_request["model"] = f"{candidate_provider}/{candidate_model}"

                try:
                    logger.info(
                        f"[{request_id}] Alias '{model_id}' -> Trying {candidate_provider}/{candidate_model}"
                    )

                    # Recursively call route_request (but now with a resolved explicit model)
                    # We avoid infinite recursion because the new model id (e.g. "openai/gpt-4")
                    # won't match an alias key in the next pass.
                    return await self.route_request(candidate_request, request_id)

                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"[{request_id}] Alias chain: {candidate_provider}/{candidate_model} failed: {e}"
                    )

                    # Check if we should abort chain (e.g. Auth Error on a paid provider might mean stop,
                    # but for now we assume failover is desired)
                    continue

            # If alias chain exhausted
            logger.error(
                f"[{request_id}] All candidates for alias '{model_id}' failed."
            )
            if last_error:
                raise last_error
            raise HTTPException(
                status_code=503, detail=f"All providers for alias '{model_id}' failed"
            )

        # --- End Alias Logic ---

        # Extract requirements
        requirements = self._extract_requirements(request)

        # Check for MoE mode
        if model_id in self.virtual_models and self.virtual_models[model_id].get(
            "moe_mode"
        ):
            logger.info(f"[{request_id}] MoE mode requested for {model_id}")
            return self.execute_moe(self.virtual_models[model_id], request, request_id)

        # Get candidates
        candidates = self._get_candidates(model_id, requirements)

        if not candidates:
            raise HTTPException(
                status_code=400,
                detail=f"No suitable providers found for model: {model_id}",
            )

        # Filter candidates by health and FREE_ONLY_MODE
        available_candidates = []
        for candidate in candidates:
            metrics = self._get_metrics(candidate.provider, candidate.model)

            # Check cooldown
            if not metrics.is_healthy():
                logger.debug(
                    f"[{request_id}] Candidate {candidate.provider}/{candidate.model} in cooldown"
                )
                continue

            # Check FREE_ONLY_MODE
            if self.free_only_mode:
                provider_config = self.config.get("providers", {}).get(
                    candidate.provider, {}
                )
                if not provider_config.get("enabled", False):
                    continue

                # Check if provider has any free tier models
                free_models = provider_config.get("free_tier_models", [])
                if free_models and candidate.model not in free_models:
                    # Allow if no free model filter specified or model is in free list
                    logger.debug(
                        f"[{request_id}] Candidate {candidate.provider}/{candidate.model} not in free tier list"
                    )
                    continue

            available_candidates.append(candidate)

        if not available_candidates:
            raise HTTPException(
                status_code=503, detail="No healthy providers available for the request"
            )

        # Sort by success rate and latency
        available_candidates.sort(
            key=lambda c: (
                -self._get_metrics(c.provider, c.model).success_rate,
                self._get_metrics(c.provider, c.model).ewma_latency_ms
                if self._get_metrics(c.provider, c.model).ewma_latency_ms > 0
                else float("inf"),
            )
        )

        if requirements.streaming:
            return self._stream_with_fallback(available_candidates, request, request_id)

        # Try candidates in order
        last_error = None
        for i, candidate in enumerate(available_candidates):
            try:
                # Check if search augmentation needed
                if requirements.search_requested or candidate.search_enabled:
                    search_query = request.get("messages", [{}])[-1].get("content", "")
                    if isinstance(search_query, str) and len(search_query) > 0:
                        search_results = await self._perform_search(search_query)
                        if search_results:
                            # Inject search results into last message
                            modified_request = request.copy()
                            modified_messages = modified_request.get(
                                "messages", []
                            ).copy()
                            last_message = modified_messages[-1].copy()

                            search_content = self._format_search_results(search_results)
                            if isinstance(last_message.get("content"), str):
                                last_message["content"] += search_content
                            elif isinstance(last_message.get("content"), list):
                                last_message["content"].append(
                                    {"type": "text", "text": search_content}
                                )

                            modified_messages[-1] = last_message
                            modified_request["messages"] = modified_messages
                            request = modified_request

                # Execute request
                return await self._execute_single_candidate(
                    candidate, request, request_id
                )

            except Exception as e:
                last_error = e
                error_category, _ = await self._classify_error(e)

                if error_category in [
                    ErrorCategory.INVALID_REQUEST,
                    ErrorCategory.AUTH_ERROR,
                ]:
                    # Don't retry on invalid requests or auth errors
                    break

                logger.warning(
                    f"[{request_id}] Candidate {candidate.provider}/{candidate.model} failed, trying next"
                )
                continue

        # All candidates failed
        if last_error:
            raise last_error
        else:
            raise HTTPException(
                status_code=503, detail="All provider candidates failed"
            )

    def get_model_list(self) -> List[Dict[str, Any]]:
        """Get list of available models for /v1/models endpoint."""
        models = []
        current_time = int(time.time())

        # Add virtual models
        for model_id, config in self.virtual_models.items():
            models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": current_time,
                    "owned_by": "router",
                    "description": config.get("description", ""),
                    "routing_type": "virtual",
                }
            )

        # Add real models from providers (simplified - in real implementation would query providers)
        providers = self.config.get("providers", {})
        for provider_name, provider_config in providers.items():
            if not provider_config.get("enabled", False):
                continue

            free_models = provider_config.get("free_tier_models", [])
            for model in free_models:
                if self.free_only_mode and model not in free_models:
                    continue

                models.append(
                    {
                        "id": f"{provider_name}/{model}",
                        "object": "model",
                        "created": current_time,
                        "owned_by": provider_name,
                        "capabilities": self._infer_capabilities(provider_name, model),
                    }
                )

        return models

    def _infer_capabilities(self, provider: str, model: str) -> List[str]:
        """Infer model capabilities from provider and model name."""
        capabilities = ["text"]

        model_lower = model.lower()

        if "vision" in model_lower or any(
            x in model_lower for x in ["claude-3", "gemini-pro", "gpt-4"]
        ):
            capabilities.append("vision")

        if "function" in model_lower or any(
            x in model_lower for x in ["claude", "gpt", "gemini", "llama"]
        ):
            capabilities.extend(["tools", "function_calling"])

        if "pro" in model_lower or "large" in model_lower:
            capabilities.append("long_context")

        return capabilities

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all providers."""
        status = {
            "free_only_mode": self.free_only_mode,
            "providers": {},
            "search_providers": {},
            "timestamp": time.time(),
        }

        # Provider metrics
        for (provider, model), metrics in self.provider_metrics.items():
            if provider not in status["providers"]:
                status["providers"][provider] = {}

            status["providers"][provider][model] = {
                "status": ProviderStatus.HEALTHY.value
                if metrics.is_healthy()
                else ProviderStatus.COOLDOWN.value,
                "success_rate": metrics.success_rate,
                "ewma_latency_ms": metrics.ewma_latency_ms,
                "consecutive_failures": metrics.consecutive_failures,
                "total_requests": metrics.total_requests,
                "cooldown_until": metrics.cooldown_until,
            }

        # Search provider metrics
        for name, provider in self.search_providers.items():
            status["search_providers"][name] = {
                "status": ProviderStatus.HEALTHY.value
                if provider.metrics.is_healthy()
                else ProviderStatus.COOLDOWN.value,
                "success_rate": provider.metrics.success_rate,
                "enabled": provider.config.enabled,
                "available": provider.is_available(self.free_only_mode),
            }

        return status
