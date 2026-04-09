#!/usr/bin/env python3
"""Update OpenCode config with agent-specific virtual models."""

import json
import sys
from pathlib import Path
from typing import Dict, Any

OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "opencode.json"
OH_MY_OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "oh-my-opencode.json"

AGENT_MODEL_MAPPING = {
    "oracle": "agent-oracle",
    "explore": "agent-explore",
    "librarian": "agent-librarian",
    "build": "agent-build",
    "metis": "agent-metis",
    "momus": "agent-momus",
}

DEFAULT_MODEL = "coding-elite"


def load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def update_opencode_main_config() -> bool:
    config = load_json(OPENCODE_CONFIG)
    changed = False

    if "provider" not in config:
        config["provider"] = {}

    if "vps-gateway" not in config["provider"]:
        config["provider"]["vps-gateway"] = {
            "name": "VPS LLM Gateway",
            "options": {
                "baseURL": "http://40.233.101.233:8000/v1",
                "apiKey": "CHANGE_ME_TO_A_STRONG_SECRET_KEY",
            },
        }
        changed = True

    if config.get("model") != f"vps-gateway/{DEFAULT_MODEL}":
        config["model"] = f"vps-gateway/{DEFAULT_MODEL}"
        changed = True

    if changed:
        save_json(OPENCODE_CONFIG, config)
        print(f"Updated {OPENCODE_CONFIG}")

    return changed


def update_agent_models() -> bool:
    config = load_json(OH_MY_OPENCODE_CONFIG)
    changed = False

    if "agents" not in config:
        config["agents"] = {}

    for agent_name, virtual_model in AGENT_MODEL_MAPPING.items():
        if (
            config["agents"].get(agent_name, {}).get("model")
            != f"vps-gateway/{virtual_model}"
        ):
            config["agents"][agent_name] = {
                "model": f"vps-gateway/{virtual_model}",
                "provider": "vps-gateway",
            }
            changed = True
            print(f"Set {agent_name} → {virtual_model}")

    if changed:
        save_json(OH_MY_OPENCODE_CONFIG, config)
        print(f"Updated {OH_MY_OPENCODE_CONFIG}")

    return changed


def print_current_config():
    print("\n" + "=" * 60)
    print("CURRENT OPENCODE CONFIGURATION")
    print("=" * 60)

    main_config = load_json(OPENCODE_CONFIG)
    print(f"\nDefault model: {main_config.get('model', 'not set')}")

    agent_config = load_json(OH_MY_OPENCODE_CONFIG)
    agents = agent_config.get("agents", {})

    print("\nAgent models:")
    for agent_name in sorted(AGENT_MODEL_MAPPING.keys()):
        current = agents.get(agent_name, {}).get("model", "not configured")
        target = f"vps-gateway/{AGENT_MODEL_MAPPING[agent_name]}"
        status = "✓" if current == target else "✗"
        print(f"  {status} {agent_name:12s}: {current}")


def main():
    print("=" * 60)
    print("OPENCODE CONFIG UPDATER")
    print("=" * 60)

    main_changed = update_opencode_main_config()
    agent_changed = update_agent_models()

    print_current_config()

    if main_changed or agent_changed:
        print("\n✓ Configuration updated successfully")
        print("  Restart OpenCode to apply changes")
    else:
        print("\n✓ Configuration already up to date")

    return 0


if __name__ == "__main__":
    sys.exit(main())
