"""
Router Integration Module

Integrates the new router system with the existing proxy infrastructure.
Maintains backward compatibility while adding new features.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator as AsyncGeneratorABC
from typing import Dict, Any, AsyncGenerator, Union, List

from fastapi import HTTPException, Request

from .router_core import RouterCore, CapabilityRequirements
from .provider_adapter import ProviderAdapterFactory

logger = logging.getLogger(__name__)


class RouterIntegration:
    """Integration layer between the new router and existing proxy."""

    def __init__(
        self,
        rotating_client: Any = None,
        config_path: str = "config/router_config.yaml",
    ):
        self.router = RouterCore(config_path)
        self.rotating_client = rotating_client
        self.adapter_factory = ProviderAdapterFactory()

        # Initialize provider adapters from environment
        self.adapters: Dict[str, Any] = {}
        self._initialize_adapters()

        logger.info(
            f"Router integration initialized with {len(self.adapters)} providers"
        )

    def _initialize_adapters(self):
        """Initialize provider adapters dynamically from router_config.yaml providers section."""
        # Build provider_configs from router_config: {name: env_var or None}
        # Skip non-LLM providers (search tools etc.) and virtual router models
        _SEARCH_ONLY = {"brave_search", "tavily", "duckduckgo", "exa", "jina"}
        _SKIP_PREFIXES = ("router/",)
        raw_providers = self.router.config.get("providers", {})
        provider_configs = {}
        for pname, pcfg in raw_providers.items():
            if pname in _SEARCH_ONLY:
                continue
            if any(pname.startswith(p) for p in _SKIP_PREFIXES):
                continue
            if not isinstance(pcfg, dict):
                continue
            # Skip virtual model groups (coding-smart etc.) that snuck into providers
            if "candidates" in pcfg or "description" in pcfg:
                continue
            env_var = pcfg.get("env_var") or None
            no_key = pcfg.get("no_api_key_required", False)
            if no_key:
                env_var = None  # signal that no key needed
                provider_configs[pname] = None
            else:
                provider_configs[pname] = env_var
        # Also ensure legacy providers that may not be in config are still initialized
        _LEGACY_NO_KEY = {"g4f", "g4f_ollama", "g4f_pollinations", "g4f_nvidia", "g4f_gemini", "g4f_groq"}
        for p in _LEGACY_NO_KEY:
            if p not in provider_configs:
                provider_configs[p] = None

        # Load active-day window configs from providers_database.yaml so the
        # rate limiter can enforce sliding-window caps (e.g. g4f: 3 active days
        # per 12-day window).
        active_days_windows: Dict[str, Dict[str, Any]] = {}
        try:
            import yaml as _yaml
            from pathlib import Path as _Path
            # Resolve repo root from this file's location
            # (src/proxy_app/router_integration.py -> parents[2] = repo root)
            _repo_root = _Path(__file__).resolve().parent.parent.parent
            _db_file = _repo_root / "config" / "providers_database.yaml"
            if _db_file.exists():
                with open(_db_file, "r") as _f:
                    _db = _yaml.safe_load(_f) or {}
                for _p in _db.get("providers", []) or []:
                    if not isinstance(_p, dict):
                        continue
                    _pid = _p.get("id")
                    _adw = _p.get("active_days_window")
                    if _pid and isinstance(_adw, dict):
                        active_days_windows[_pid] = _adw
        except Exception as _e:
            logger.debug(
                f"Could not load active_days_window configs from database: {_e}"
            )
        if active_days_windows:
            try:
                self.router.rate_limiter.configure_active_days_windows(
                    active_days_windows
                )
            except Exception as _e:  # pragma: no cover - defensive
                logger.warning(
                    f"Failed to configure active-days windows: {_e}"
                )

        for provider_name, env_var in provider_configs.items():
            if env_var is None:  # No API key needed
                try:
                    adapter = self.adapter_factory.create_adapter(provider_name, None)
                    self.adapters[provider_name] = adapter
                    logger.info(
                        f"Initialized {provider_name} adapter (no API key required)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider_name} adapter: {e}")
                continue

            # Check if API key is available
            import os

            api_key = os.getenv(env_var)

            if api_key:
                try:
                    provider_cfg = self.router.config.get("providers", {}).get(
                        provider_name, {}
                    )
                    api_base = provider_cfg.get("base_url")
                    model_list = provider_cfg.get("free_tier_models", [])
                    
                    # Also load models from providers_database.yaml for providers
                    # that only have models listed there (e.g. xinjianya)
                    if not model_list and api_base:
                        try:
                            import yaml as _yaml
                            from pathlib import Path
                            _db_file = Path("/home/ubuntu/LLM-API-Key-Proxy/config/providers_database.yaml")
                            if _db_file.exists():
                                with open(_db_file, "r") as _f:
                                    _db = _yaml.safe_load(_f) or {}
                                for _p in _db.get("providers", []):
                                    if _p.get("id") == provider_name:
                                        _models = _p.get("free_models", [])
                                        model_list = [m["id"] for m in _models if isinstance(m, dict) and "id" in m]
                                        logger.info(f"Loaded {len(model_list)} models from providers_database.yaml for {provider_name}")
                                        break
                        except Exception as _e:
                            logger.debug(f"Could not load models from database for {provider_name}: {_e}")

                    adapter = self.adapter_factory.create_adapter(
                        provider_name,
                        api_key,
                        api_base,
                        model_list,
                    )
                    self.adapters[provider_name] = adapter
                    logger.info(f"Initialized {provider_name} adapter")
                except Exception as e:
                    logger.warning(f"Failed to initialize {provider_name} adapter: {e}")
            else:
                logger.debug(f"No API key found for {provider_name}, skipping")

    async def chat_completions(
        self,
        request_data: Dict[str, Any],
        raw_request: Request,
        enable_logging: bool = True,
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Main chat completions endpoint integrated with router."""

        # Generate request ID
        request_id = f"req_{uuid.uuid4().hex[:16]}"

        # Log request if enabled
        if enable_logging:
            logger.info(
                f"[{request_id}] Incoming request for model: {request_data.get('model', 'unknown')}"
            )
            logger.debug(f"[{request_id}] Request data: {request_data}")

        # Use new router for all requests (legacy rotating client removed)
        try:
            return await self._handle_with_router(request_data, request_id)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Router failed: {e}")
            raise HTTPException(status_code=500, detail=f"Router error: {str(e)}")

    async def _handle_with_router(
        self, request_data: Dict[str, Any], request_id: str
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """Handle request with new router system."""

        # Check if streaming is requested
        streaming = request_data.get("stream", False)

        # Route the request
        start_time = time.time()

        try:
            response = await self.router.route_request(request_data, request_id)

            # Handle streaming response
            if streaming:
                if not isinstance(response, AsyncGeneratorABC):
                    raise HTTPException(
                        status_code=500,
                        detail="Streaming requested but non-stream response returned",
                    )
                return self._wrap_streaming_response(response, request_id, start_time)
            else:
                # Log completion
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"[{request_id}] Request completed in {duration_ms:.1f}ms")
                return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[{request_id}] Router request failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _wrap_streaming_response(
        self,
        response: AsyncGenerator[Dict[str, Any], None],
        request_id: str,
        start_time: float,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Wrap streaming response with logging and final stats."""
        try:
            chunk_count = 0
            async for chunk in response:
                chunk_count += 1
                # Pass through the chunk
                yield chunk

            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"[{request_id}] Stream completed in {duration_ms:.1f}ms ({chunk_count} chunks)"
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"[{request_id}] Stream failed after {duration_ms:.1f}ms: {e}")
            raise

    def get_models(self) -> List[Dict[str, Any]]:
        """Get available models (combines router and legacy models)."""

        # Get models from router
        router_models = self.router.get_model_list()

        # Get models from adapters
        adapter_models = []
        for provider_name, adapter in self.adapters.items():
            models = asyncio.run(adapter.list_models())
            for model in models:
                caps = adapter.get_model_capabilities(model)
                model_entry = {
                    "id": f"{provider_name}/{model}",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": provider_name,
                }

                if caps:
                    model_entry["capabilities"] = caps.tags
                    model_entry["max_context_tokens"] = caps.max_context_tokens
                    model_entry["free_tier"] = caps.free_tier_available

                adapter_models.append(model_entry)

        # Combine and deduplicate
        all_models = router_models + adapter_models

        # Remove duplicates (prefer router models)
        model_lookup = {model["id"]: model for model in all_models}
        return list(model_lookup.values())

    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        health_status = self.router.get_health_status()

        # Add adapter-specific health info
        health_status["adapters"] = {}
        for provider_name, adapter in self.adapters.items():
            health_status["adapters"][provider_name] = {
                "available": True,
                "models": len(adapter.models),
            }

        return health_status

    def refresh_configuration(self):
        """Refresh router configuration."""
        # Reinitialize router with potential new config
        self.router = RouterCore()
        self._initialize_adapters()
        logger.info("Router configuration refreshed")

    @property
    def free_only_mode(self) -> bool:
        """Get FREE_ONLY_MODE status."""
        return self.router.free_only_mode


def extract_search_requirements(request_data: Dict[str, Any]) -> tuple[bool, str]:
    """Extract whether search is needed and the search query."""

    # Check for explicit search indicators in the last message
    messages = request_data.get("messages", [])
    if not messages:
        return False, ""

    last_message = messages[-1]
    content = ""

    if isinstance(last_message.get("content"), str):
        content = last_message["content"]
    elif isinstance(last_message.get("content"), list):
        # Extract text content
        for item in last_message["content"]:
            if isinstance(item, dict) and item.get("type") == "text":
                content = item.get("text", "")
                break

    # Check for search indicators
    search_indicators = [
        "latest",
        "recent",
        "current",
        "news",
        "shopping",
        "best",
        "top",
        "compare",
        "sources",
        "citations",
        "what's new",
        "up to date",
        "2024",
        "2025",
    ]

    content_lower = content.lower()
    needs_search = any(indicator in content_lower for indicator in search_indicators)

    # Only search if content is substantial enough
    if needs_search and len(content.split()) > 3:
        # Extract a search query (simplified - could be improved)
        search_query = content[:200]  # First 200 chars as search query
        return True, search_query

    return False, ""
