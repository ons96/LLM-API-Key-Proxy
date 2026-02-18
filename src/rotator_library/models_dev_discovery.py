"""
models.dev Auto-Discovery Module

Fetches free LLM providers/models from https://models.dev/api.json,
compares against existing router_config.yaml, and reports new discoveries.
Can optionally update the config with newly found free providers.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
import yaml

logger = logging.getLogger(__name__)

MODELS_DEV_API_URL = "https://models.dev/api.json"

SKIP_PROVIDERS = frozenset(
    {
        "lmstudio",
        "ollama-cloud",
        "github-copilot",
        "gitlab",
        "opencode",
        "privatemode-ai",
    }
)

CONFIG_ROOT = Path(__file__).resolve().parent.parent.parent / "config"
ROUTER_CONFIG_PATH = CONFIG_ROOT / "router_config.yaml"
DISCOVERY_CACHE_PATH = CONFIG_ROOT / "discovered_models.json"


def _is_free_model(model: Dict[str, Any]) -> bool:
    cost = model.get("cost", {})
    return cost.get("input", 1) == 0 and cost.get("output", 1) == 0


def _is_text_model(model: Dict[str, Any]) -> bool:
    modalities = model.get("modalities", {})
    return "text" in modalities.get("input", []) and "text" in modalities.get(
        "output", []
    )


def _is_usable_model(model: Dict[str, Any]) -> bool:
    if model.get("status") == "deprecated":
        return False
    if not _is_free_model(model):
        return False
    if not _is_text_model(model):
        return False
    return True


def _has_remote_api(provider: Dict[str, Any]) -> bool:
    api = provider.get("api", "")
    if not api:
        return False
    if "127.0.0.1" in api or "localhost" in api:
        return False
    return True


def _load_router_config() -> Dict[str, Any]:
    if not ROUTER_CONFIG_PATH.exists():
        return {}
    with open(ROUTER_CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def _get_configured_providers(config: Dict[str, Any]) -> Set[str]:
    providers = config.get("providers", {})
    return set(providers.keys())


def _get_configured_models(config: Dict[str, Any]) -> Set[str]:
    models: Set[str] = set()
    providers = config.get("providers", {})
    for provider_cfg in providers.values():
        if isinstance(provider_cfg, dict):
            for m in provider_cfg.get("free_tier_models", []):
                models.add(str(m).lower())
    return models


async def fetch_models_dev_data(timeout: float = 60.0) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(MODELS_DEV_API_URL)
        response.raise_for_status()
        return response.json()


def discover_free_providers(
    api_data: Dict[str, Any],
    existing_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if existing_config is None:
        existing_config = _load_router_config()

    configured_providers = _get_configured_providers(existing_config)
    configured_models = _get_configured_models(existing_config)

    discoveries: List[Dict[str, Any]] = []

    for provider_id, provider_data in api_data.items():
        if provider_id in SKIP_PROVIDERS:
            continue

        if not _has_remote_api(provider_data):
            continue

        free_text_models = {}
        for model_id, model_data in provider_data.get("models", {}).items():
            if _is_usable_model(model_data):
                free_text_models[model_id] = model_data

        if not free_text_models:
            continue

        is_new_provider = provider_id not in configured_providers

        new_models = {}
        for model_id, model_data in free_text_models.items():
            model_lower = model_id.lower()
            if model_lower not in configured_models:
                new_models[model_id] = model_data

        if not new_models and not is_new_provider:
            continue

        discovery = {
            "provider_id": provider_id,
            "provider_name": provider_data.get("name", provider_id),
            "api_url": provider_data.get("api", ""),
            "env_vars": provider_data.get("env", []),
            "doc_url": provider_data.get("doc", ""),
            "is_new_provider": is_new_provider,
            "new_models": [],
            "all_free_models": [],
        }

        for model_id, model_data in new_models.items():
            discovery["new_models"].append(
                {
                    "id": model_id,
                    "name": model_data.get("name", model_id),
                    "context_window": model_data.get("limit", {}).get("context", 0),
                    "max_output": model_data.get("limit", {}).get("output", 0),
                    "tool_call": model_data.get("tool_call", False),
                    "reasoning": model_data.get("reasoning", False),
                    "open_weights": model_data.get("open_weights", False),
                    "family": model_data.get("family", ""),
                }
            )

        for model_id, model_data in free_text_models.items():
            discovery["all_free_models"].append(
                {
                    "id": model_id,
                    "name": model_data.get("name", model_id),
                    "context_window": model_data.get("limit", {}).get("context", 0),
                    "tool_call": model_data.get("tool_call", False),
                    "reasoning": model_data.get("reasoning", False),
                }
            )

        discoveries.append(discovery)

    discoveries.sort(key=lambda d: (-len(d["new_models"]), d["provider_id"]))
    return discoveries


def generate_config_entries(
    discoveries: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}

    for discovery in discoveries:
        if not discovery["new_models"]:
            continue

        pid = discovery["provider_id"]
        env_vars = discovery["env_vars"]

        entry: Dict[str, Any] = {
            "enabled": False,
            "free_tier_models": [m["id"] for m in discovery["new_models"]],
        }

        if env_vars:
            entry["env_var"] = env_vars[0]

        if discovery["api_url"]:
            entry["api_base"] = discovery["api_url"]

        entries[pid] = entry

    return entries


def save_discovery_cache(
    discoveries: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    path = output_path or DISCOVERY_CACHE_PATH
    cache = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_discoveries": len(discoveries),
        "new_providers": sum(1 for d in discoveries if d["is_new_provider"]),
        "total_new_models": sum(len(d["new_models"]) for d in discoveries),
        "discoveries": discoveries,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)

    return path


def update_router_config(
    discoveries: List[Dict[str, Any]],
    config_path: Optional[Path] = None,
    auto_enable: bool = False,
) -> List[str]:
    path = config_path or ROUTER_CONFIG_PATH
    if not path.exists():
        logger.error(f"Config not found: {path}")
        return []

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    providers = config.setdefault("providers", {})
    changes: List[str] = []

    for discovery in discoveries:
        pid = discovery["provider_id"]
        new_model_ids = [m["id"] for m in discovery["new_models"]]

        if not new_model_ids:
            continue

        if pid in providers and isinstance(providers[pid], dict):
            existing_models = providers[pid].get("free_tier_models", [])
            existing_lower = {str(m).lower() for m in existing_models}

            added = []
            for mid in new_model_ids:
                if mid.lower() not in existing_lower:
                    existing_models.append(mid)
                    added.append(mid)

            if added:
                providers[pid]["free_tier_models"] = existing_models
                changes.append(
                    f"{pid}: added {len(added)} models ({', '.join(added[:3])})"
                )
        else:
            env_vars = discovery["env_vars"]
            entry: Dict[str, Any] = {
                "enabled": auto_enable,
                "free_tier_models": new_model_ids,
            }
            if env_vars:
                entry["env_var"] = env_vars[0]

            providers[pid] = entry
            changes.append(f"{pid}: NEW provider with {len(new_model_ids)} free models")

    if changes:
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Updated {path} with {len(changes)} changes")

    return changes


async def run_discovery(
    update_config: bool = False,
    auto_enable: bool = False,
    save_cache: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    logger.info("Fetching models.dev API data...")
    api_data = await fetch_models_dev_data()
    logger.info(f"Loaded {len(api_data)} providers from models.dev")

    config = _load_router_config()
    discoveries = discover_free_providers(api_data, config)

    total_new_models = sum(len(d["new_models"]) for d in discoveries)
    new_providers = sum(1 for d in discoveries if d["is_new_provider"])
    logger.info(
        f"Found {total_new_models} new free models across "
        f"{len(discoveries)} providers ({new_providers} entirely new)"
    )

    if save_cache:
        cache_path = save_discovery_cache(discoveries)
        logger.info(f"Discovery cache saved to {cache_path}")

    changes: List[str] = []
    if update_config and discoveries:
        changes = update_router_config(discoveries, auto_enable=auto_enable)
        for change in changes:
            logger.info(f"  Config change: {change}")

    return discoveries, changes
