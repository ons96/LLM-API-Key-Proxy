"""Read credential-free model-group chains from llm-provider-manager SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from router_core import Capability, Deployment


def _tokens(value: object) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return {str(item).lower() for item in parsed}
            if isinstance(parsed, dict):
                return {str(key).lower() for key, enabled in parsed.items() if enabled}
        except json.JSONDecodeError:
            pass
        return {part.strip().lower() for part in value.replace(",", " ").split() if part.strip()}
    return {str(value).lower()}


def _capability(group: str, capabilities: object) -> Capability:
    name = group.lower()
    tags = _tokens(capabilities)
    if name.startswith("coding") or "code" in tags or "coding" in tags:
        return Capability.CODE_IMPLEMENT
    if name == "glm5-elite" or "reasoning" in tags:
        return Capability.DEEP_REASONING
    if "tools" in tags or "tool" in tags:
        return Capability.FAST_TOOL
    return Capability.FAST_GENERAL


def _env_prefix(env_var: object, provider: str) -> str:
    value = str(env_var or provider).strip()
    if value.endswith("_API_KEY"):
        value = value[:-8]
    return value


def load_provider_group_deployments(
    path: str | Path,
    groups: list[str] | None = None,
    *,
    free_only: bool = True,
) -> list[Deployment]:
    """Build ordered deployments from provider metadata without reading secrets.

    The database contributes model/provider identity, endpoint metadata,
    capability hints, context limits, and chain priority. API keys remain in
    the process environment and are never queried from the database.
    """
    requested = {group for group in (groups or []) if group}
    database = Path(path).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(database)
    uri = f"file:{database}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            """SELECT vm.name, fc.priority, fc.provider_key, fc.model_id,
                      fc.capabilities, p.base_url, p.env_var, p.enabled,
                      p.free_tier, p.no_api_key_required, p.free_unlimited,
                      p.free_daily, m.context_window, m.tps, m.free_tier,
                      m.capabilities
                 FROM fallback_chains AS fc
                 JOIN virtual_models AS vm ON vm.id = fc.virtual_model_id
                 LEFT JOIN providers AS p ON p.key_name = fc.provider_key
                 LEFT JOIN models AS m ON m.provider_id = p.id AND m.model_id = fc.model_id
                ORDER BY vm.name, fc.priority, fc.id"""
        ).fetchall()
    finally:
        connection.close()

    deployments: list[Deployment] = []
    seen: set[str] = set()
    for (
        group,
        priority,
        provider,
        model,
        chain_capabilities,
        base_url,
        env_var,
        enabled,
        provider_free,
        no_api_key,
        free_unlimited,
        free_daily,
        context_window,
        tps,
        model_free,
        model_capabilities,
    ) in rows:
        group = str(group)
        provider = str(provider or "")
        model = str(model or "")
        if requested and group not in requested:
            continue
        if not provider or not model or enabled == 0:
            continue
        if free_only and not any((provider_free, no_api_key, free_unlimited, free_daily, model_free)):
            continue
        deployment_id = f"{group}:{priority}:{provider}:{model}"
        if deployment_id in seen:
            continue
        seen.add(deployment_id)
        capabilities = _tokens(chain_capabilities) | _tokens(model_capabilities)
        capability = _capability(group, capabilities)
        tools = capability == Capability.CODE_IMPLEMENT or "tools" in capabilities or "tool" in capabilities
        structured = bool({"json", "structured_output", "structured"} & capabilities)
        context_limit = max(1024, int(context_window or 16_384))
        tokens_per_second = max(1, int(tps or 1))
        latency_ms = max(50.0, 1000.0 / tokens_per_second)
        quota_class = "free" if free_only else "metadata"
        deployments.append(
            Deployment(
                deployment_id=deployment_id,
                capability=capability,
                context_limit=context_limit,
                tools=tools,
                structured_output=structured,
                latency_ms=latency_ms,
                success_rate=0.5,
                provider=provider,
                base_url=str(base_url or ""),
                config_fingerprint="provider-db",
                quota_class=quota_class,
                model=model,
                group=group,
                chain_priority=max(1, int(priority)),
                env_prefix=_env_prefix(env_var, provider),
            )
        )
    if not deployments:
        raise ValueError("provider metadata produced no eligible deployments")
    return deployments
