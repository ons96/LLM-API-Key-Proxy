#!/usr/bin/env python3
"""
Unified Discovery Pipeline

Orchestrates:
1. Fetch models.dev free models
2. Query provider /v1/models endpoints
3. Merge with VPS telemetry
4. Compute composite scores
5. Generate agent-specific virtual models
6. Update all configs
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotator_library.models_dev_discovery import (
    fetch_models_dev_data,
    discover_free_providers,
)
from optimization.telemetry_analyzer import TelemetryAnalyzer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(path: Path) -> Dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_yaml(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )


def load_json(path: Path) -> Dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


AGENT_PROFILES = {
    "oracle": {
        "needs": ["reasoning", "tools", "long_context"],
        "prefers": ["accuracy", "depth"],
        "weights": {"benchmark": 0.75, "success_rate": 0.20, "latency": 0.05},
        "base_model": "coding-elite",
    },
    "explore": {
        "needs": ["speed", "tools"],
        "prefers": ["latency", "throughput"],
        "weights": {"benchmark": 0.15, "success_rate": 0.40, "latency": 0.45},
        "base_model": "coding-fast",
    },
    "librarian": {
        "needs": ["tools", "search"],
        "prefers": ["availability", "reliability"],
        "weights": {"benchmark": 0.25, "success_rate": 0.55, "latency": 0.20},
        "base_model": "chat-smart",
    },
    "build": {
        "needs": ["coding", "tools"],
        "prefers": ["balanced"],
        "weights": {"benchmark": 0.55, "success_rate": 0.35, "latency": 0.10},
        "base_model": "coding-smart",
    },
    "metis": {
        "needs": ["reasoning", "analysis"],
        "prefers": ["accuracy"],
        "weights": {"benchmark": 0.70, "success_rate": 0.25, "latency": 0.05},
        "base_model": "coding-elite",
    },
    "momus": {
        "needs": ["reasoning", "analysis"],
        "prefers": ["quality"],
        "weights": {"benchmark": 0.65, "success_rate": 0.30, "latency": 0.05},
        "base_model": "coding-elite",
    },
}


async def discover_models():
    logger.info("Fetching models.dev data...")
    try:
        api_data = await fetch_models_dev_data()
        discoveries = discover_free_providers(
            api_data, load_yaml(CONFIG_DIR / "router_config.yaml")
        )
        logger.info(f"Found {len(discoveries)} providers with free models")
        return discoveries
    except Exception as e:
        logger.error(f"Discovery failed: {e}")
        return []


def analyze_telemetry(telemetry_path: Path) -> Dict:
    logger.info(f"Analyzing telemetry from {telemetry_path}")
    analyzer = TelemetryAnalyzer(telemetry_path)

    try:
        return analyzer.get_provider_scores()
    except Exception as e:
        logger.warning(f"Telemetry analysis failed, using defaults: {e}")
        return {}


def merge_scores(discoveries: List[Dict], telemetry_scores: Dict) -> List[Dict]:
    rankings = load_yaml(CONFIG_DIR / "model_rankings.yaml")
    benchmark_scores = {}
    for m in rankings.get("models", []):
        model_id = m.get("id", "")
        scores = m.get("scores", {})
        benchmark_scores[model_id] = {
            "swe_bench": scores.get("swe_bench", 0),
            "intelligence": scores.get("intelligence", 0),
            "coding": scores.get("coding", 0),
            "tps": scores.get("speed_tps", 50),
        }

    merged = []
    for discovery in discoveries:
        for model in discovery.get("all_free_models", []):
            model_id = f"{discovery['provider_id']}/{model['id']}"

            bench = benchmark_scores.get(
                model_id, {"swe_bench": 0, "intelligence": 50, "coding": 0, "tps": 100}
            )
            tel = telemetry_scores.get(
                model_id, {"success_rate": 0.8, "avg_latency_ms": 2000}
            )

            bench_score = (
                bench["swe_bench"] * 0.5
                + bench["coding"] * 0.3
                + bench["intelligence"] * 0.2
            ) / 100
            success_score = tel.get("success_rate", 0.5)
            latency_score = max(0, 1 - tel.get("avg_latency_ms", 5000) / 30000)

            composite = bench_score * 0.4 + success_score * 0.4 + latency_score * 0.2

            merged.append(
                {
                    "provider": discovery["provider_id"],
                    "model": model["id"],
                    "model_id": model_id,
                    "composite_score": round(composite, 3),
                    "capabilities": model.get("capabilities", []),
                    "context_window": model.get("context_window", 0),
                    "benchmark_score": round(bench_score, 3),
                    "success_rate": round(success_score, 3),
                    "latency_score": round(latency_score, 3),
                }
            )

    merged.sort(key=lambda x: x["composite_score"], reverse=True)
    return merged


def generate_agent_virtual_models(scored_models: List[Dict]) -> Dict:
    virtual_models = load_yaml(CONFIG_DIR / "virtual_models.yaml")
    agent_models = {}

    for agent_name, profile in AGENT_PROFILES.items():
        weights = profile["weights"]
        candidates = []

        for m in scored_models[:50]:
            score = (
                m["benchmark_score"] * weights["benchmark"]
                + m["success_rate"] * weights["success_rate"]
                + m["latency_score"] * weights["latency"]
            )
            candidates.append(
                {
                    "provider": m["provider"],
                    "model": m["model"],
                    "score": round(score, 3),
                    "capabilities": m.get("capabilities", []),
                }
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)

        agent_model_name = f"agent-{agent_name}"
        agent_models[agent_model_name] = {
            "description": f"Optimized for {agent_name} agent - {profile['prefers']}",
            "fallback_chain": [
                {"provider": c["provider"], "model": c["model"], "priority": i + 1}
                for i, c in enumerate(candidates[:20])
            ],
            "settings": {
                "timeout_ms": 180000 if "reasoning" in profile["needs"] else 120000,
                "retry_on_rate_limit": True,
            },
        }

    all_virtual = {**virtual_models.get("virtual_models", {}), **agent_models}

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sources": ["models.dev", "telemetry", "leaderboards"],
            "agent_profiles": list(AGENT_PROFILES.keys()),
        },
        "virtual_models": all_virtual,
        "agent_profiles": AGENT_PROFILES,
    }


def update_providers_database(discoveries: List[Dict]) -> bool:
    db_path = CONFIG_DIR / "providers_database.yaml"
    db = load_yaml(db_path)
    providers = db.get("providers", [])

    existing_ids = {p["id"] for p in providers}
    changed = False

    for discovery in discoveries:
        pid = discovery["provider_id"]
        if pid not in existing_ids and discovery.get("new_models"):
            providers.append(
                {
                    "id": pid,
                    "name": discovery.get("provider_name", pid),
                    "enabled": False,
                    "free_tier": True,
                    "no_api_key_required": False,
                    "free_models": [
                        {"id": m["id"], "context": m.get("context_window", 0)}
                        for m in discovery.get("new_models", [])[:10]
                    ],
                    "notes": f"Discovered {datetime.now().strftime('%Y-%m-%d')}",
                }
            )
            existing_ids.add(pid)
            changed = True
            logger.info(f"Added new provider: {pid}")

    if changed:
        db["providers"] = providers
        save_yaml(db_path, db)

    return changed


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", required=True, help="Path to telemetry JSON")
    parser.add_argument("--output-dir", default="config/", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force update")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discoveries = await discover_models()

    telemetry_scores = analyze_telemetry(Path(args.telemetry))

    scored_models = merge_scores(discoveries, telemetry_scores)

    updated_virtual = generate_agent_virtual_models(scored_models)
    save_yaml(output_dir / "virtual_models.yaml", updated_virtual)

    update_providers_database(discoveries)

    discovery_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "new_providers": sum(1 for d in discoveries if d.get("is_new_provider")),
        "updated_providers": sum(
            1
            for d in discoveries
            if not d.get("is_new_provider") and d.get("new_models")
        ),
        "total_free_models": sum(
            len(d.get("all_free_models", [])) for d in discoveries
        ),
        "top_models": scored_models[:10],
    }
    save_json(output_dir / "discovery_report.json", discovery_report)

    optimization_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "virtual_models_generated": len(updated_virtual["virtual_models"]),
        "agent_profiles": len(AGENT_PROFILES),
        "top_scored_models": scored_models[:5],
    }
    save_json(output_dir / "optimization_report.json", optimization_report)

    logger.info(
        f"Pipeline complete. Generated {len(updated_virtual['virtual_models'])} virtual models"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
