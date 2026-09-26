#!/usr/bin/env python3
"""
Chain-regeneration runner (task-board #928).

One cycle:
  1. Build a candidate pool from providers_database.yaml (every free_model of
     enabled + free_tier providers).
  2. For each live virtual model in virtual_models.yaml, add candidates whose
     model capabilities match the chain family. Candidates found in the
     benchmark rankings are scored with generate_virtual_models.WEIGHTS +
     calculate_score and must clear the family min_score floor; unranked
     candidates (i.e. brand-new providers) bootstrap onto the tail, where
     reorder_chains keeps them after telemetry-proven entries.
  3. Merged chains go through chain_policy.sanitize_chain (blocked filter,
     dedup, direct-free-fallback guarantee, renumber).
  4. The merged config feeds reorder_chains.reorder_config for live-telemetry
     ordering, then (only with --apply) replaces config/virtual_models.yaml
     atomically, with a timestamped backup.

Dry-run is the DEFAULT: nothing outside a temp dir is written unless --apply
is passed (#928 AC). Telemetry DB default comes from reorder_chains — the
durable repo data/ path, never /dev/shm (reboot-wiped tmpfs made reorder a
no-op; gateway-writer alignment is operator follow-up #758).

Usage:
    python scripts/regenerate_chains.py            # dry-run, print plan
    python scripts/regenerate_chains.py --apply    # write config (backed up)

Exit codes:
    0 = cycle ran and changed something (candidates added and/or reordered)
    1 = input/config error
    2 = cycle ran, nothing to change
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import chain_policy  # noqa: E402
import generate_virtual_models as gvm  # noqa: E402
import reorder_chains  # noqa: E402

logger = logging.getLogger("regenerate_chains")

DEFAULT_CONFIG_PATH = reorder_chains.DEFAULT_CONFIG_PATH
DEFAULT_PROVIDERS_DB = _REPO_ROOT / "config" / "providers_database.yaml"
DEFAULT_POLICY_PATH = chain_policy.DEFAULT_POLICY_PATH

# Per-VM score floors, mirroring generate_virtual_models.generate_virtual_models_yaml
# thresholds. Unmapped VM names default to 0.30 (ponytail: single default floor;
# per-VM overrides only if a chain demonstratively drifts).
MIN_SCORES = {
    "coding-elite": 0.5,
    "coding-smart": 0.35,
    "coding-fast": 0.3,
    "chat-elite": 0.5,
    "chat-smart": 0.4,
    "chat-fast": 0.3,
    "chat-rp": 0.15,
}
DEFAULT_MIN_SCORE = 0.30


def family_for(vm_name: str) -> Tuple[str, str, float, int]:
    """Resolve (weights key, required capability, min_score, chain cap).

    ponytail: prefix mapping keeps the 17 live chain names (agent-*, auto,
    glm5-elite, title-fast, ...) working without a config table; exact names
    reuse generate_virtual_models.WEIGHTS directly.
    """
    if vm_name in gvm.WEIGHTS:
        weights_key = vm_name
    elif vm_name.startswith("coding") or vm_name.startswith("agent"):
        weights_key = "coding-smart"
    elif vm_name.endswith("-rp"):
        weights_key = "chat-rp"
    else:
        weights_key = "chat-smart"
    if vm_name.startswith("coding"):
        capability = "code"
    elif vm_name.startswith("agent"):
        capability = "tools"
    else:
        capability = "chat"
    cap = 20 if weights_key == "chat-rp" else 30
    return weights_key, capability, MIN_SCORES.get(vm_name, DEFAULT_MIN_SCORE), cap


def load_db_candidates(db_path: Path) -> List[Dict[str, Any]]:
    """providers_database.yaml -> flat candidate list.

    Guards against explicitly-null YAML lists (capabilities: null was a live
    gateway crasher — see repo gotcha); treats null as empty.
    """
    doc = yaml.safe_load(Path(db_path).read_text()) or {}
    candidates: List[Dict[str, Any]] = []
    for p in doc.get("providers") or []:
        if not isinstance(p, dict):
            continue
        if not p.get("enabled", False) or not p.get("free_tier", False):
            continue
        provider_id = str(p.get("id", "")).strip()
        if not provider_id:
            continue
        provider_caps = [str(c).lower() for c in (p.get("capabilities") or [])]
        for m in p.get("free_models") or []:
            if not isinstance(m, dict) or not m.get("id"):
                continue
            model_caps = [str(c).lower() for c in (m.get("capabilities") or [])]
            candidates.append(
                {
                    "provider": provider_id,
                    "model": str(m["id"]).strip(),
                    "capabilities": model_caps or provider_caps,
                    "context": m.get("context", 0),
                    "tps": m.get("tps", 0),
                }
            )
    logger.info("loaded %d DB candidates from %s", len(candidates), db_path)
    return candidates


def _ranked_metrics(models: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """Rankings rows -> (by "provider/model" exact key, by bare model key)."""
    by_exact: Dict[str, Dict] = {}
    by_model: Dict[str, Dict] = {}
    for m in models:
        mid = m.get("id", "")
        if not mid:
            continue
        by_exact[mid] = m
        by_model.setdefault(mid.rsplit("/", 1)[-1].lower(), m)
    return by_exact, by_model


def merge_candidates_for_vm(
    vm_name: str,
    model_cfg: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    coding_maps: Tuple[Dict[str, Dict], Dict[str, Dict]],
    chat_maps: Tuple[Dict[str, Dict], Dict[str, Dict]],
    policy: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extend one virtual model's chain with policy-safe DB candidates.

    Returns (new_chain, log_lines). Existing entries always keep their
    relative head positions; new candidates append after them.
    """
    weights_key, required_cap, min_score, cap = family_for(vm_name)
    chain = model_cfg.get("fallback_chain") or []
    existing_keys = {
        chain_policy.candidate_key(c.get("provider"), c.get("model")) for c in chain
    }

    is_coding = required_cap in ("code", "tools")
    by_exact, by_model = coding_maps if is_coding else chat_maps

    scored: List[Tuple[float, Dict[str, Any]]] = []
    bootstrap: List[Dict[str, Any]] = []
    seen = set()
    log_lines: List[str] = []
    for cand in candidates:
        if required_cap not in cand["capabilities"]:
            continue
        key = chain_policy.candidate_key(cand["provider"], cand["model"])
        if key in existing_keys or key in seen:
            continue
        if chain_policy.blocked_reason(cand["provider"], cand["model"], policy):
            continue
        seen.add(key)
        exact = by_exact.get(f"{cand['provider']}/{cand['model']}")
        metrics = exact or by_model.get(cand["model"].lower())
        if metrics:
            score = gvm.calculate_score(metrics, gvm.WEIGHTS[weights_key])
            if score >= min_score:
                scored.append((score, cand))
            else:
                log_lines.append(
                    f"[{vm_name}] skip {cand['provider']}/{cand['model']}: "
                    f"score {score:.2f} < min {min_score}"
                )
        else:
            bootstrap.append(cand)

    merged = [dict(c) for c in chain]
    scored.sort(key=lambda t: (-t[0], t[1]["provider"], t[1]["model"]))
    for _, cand in scored:
        entry = {
            "provider": cand["provider"],
            "model": cand["model"],
            "reasoning_effort": "None",
        }
        merged.append(entry)
        log_lines.append(f"[{vm_name}] +candidate {cand['provider']}/{cand['model']} (scored)")
    for cand in bootstrap:
        entry = {
            "provider": cand["provider"],
            "model": cand["model"],
            "reasoning_effort": "None",
        }
        merged.append(entry)
        log_lines.append(
            f"[{vm_name}] +candidate {cand['provider']}/{cand['model']} (bootstrap)"
        )

    max_entries = max(len(chain), cap)
    new_chain = chain_policy.sanitize_chain(merged, policy, max_entries=max_entries)
    return new_chain, log_lines


def build_merged_doc(
    live_doc: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    coding_rankings_path: Optional[Path],
    chat_rankings_path: Optional[Path],
    policy: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, List[str]], List[str]]:
    """Merge DB candidates into every eligible live virtual model.

    Skips policy-fixed static virtual models (e.g. safe-coding, #405) so the
    regeneration pipeline never mutates operator-pinned chains.
    Returns (merged_doc, added_per_vm, log_lines).
    """
    coding_maps = _ranked_metrics(gvm.load_coding_models(coding_rankings_path))
    chat_maps = _ranked_metrics(gvm.load_chat_models(chat_rankings_path))
    static_vms = reorder_chains.load_static_virtual_models(
        reorder_chains.STATIC_VIRTUAL_MODELS_PATH
    )

    merged_doc = dict(live_doc)
    virtual_models = dict(live_doc.get("virtual_models") or {})
    added_per_vm: Dict[str, List[str]] = {}
    log_lines: List[str] = []

    for vm_name, model_cfg in virtual_models.items():
        if vm_name in static_vms:
            log_lines.append(f"[{vm_name}] skipped (static/policy-fixed)")
            continue
        if not isinstance(model_cfg, dict):
            continue
        new_chain, vm_log = merge_candidates_for_vm(
            vm_name, model_cfg, candidates, coding_maps, chat_maps, policy
        )
        log_lines.extend(vm_log)
        before = {
            chain_policy.candidate_key(c.get("provider"), c.get("model"))
            for c in (model_cfg.get("fallback_chain") or [])
        }
        added = [
            f"{c['provider']}/{c['model']}"
            for c in new_chain
            if chain_policy.candidate_key(c.get("provider"), c.get("model")) not in before
        ]
        if added:
            added_per_vm[vm_name] = added
        virtual_models[vm_name] = dict(model_cfg, fallback_chain=new_chain)

    merged_doc["virtual_models"] = virtual_models
    return merged_doc, added_per_vm, log_lines


def regenerate(
    config_path: Path,
    providers_db_path: Path,
    policy_path: Optional[Path] = None,
    coding_rankings_path: Optional[Path] = None,
    chat_rankings_path: Optional[Path] = None,
    telemetry_db: Optional[str] = None,
    window_h: int = reorder_chains.DEFAULT_WINDOW_H,
    min_samples: int = reorder_chains.DEFAULT_MIN_SAMPLES,
    max_tps: float = reorder_chains.DEFAULT_MAX_TPS,
    max_ttft_ms: float = reorder_chains.DEFAULT_MAX_TTFT_MS,
    apply_changes: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    """Run one regeneration cycle. Returns (exit_code, report)."""
    config_path = Path(config_path)
    providers_db_path = Path(providers_db_path)
    if not config_path.exists():
        logger.error("config not found: %s", config_path)
        return 1, {"error": f"config not found: {config_path}"}
    if not providers_db_path.exists():
        logger.error("providers_database not found: %s", providers_db_path)
        return 1, {"error": f"providers_database not found: {providers_db_path}"}

    policy = chain_policy.load_policy(policy_path)
    candidates = load_db_candidates(providers_db_path)
    live_doc = yaml.safe_load(config_path.read_text()) or {}

    merged_doc, added_per_vm, log_lines = build_merged_doc(
        live_doc,
        candidates,
        coding_rankings_path,
        chat_rankings_path,
        policy,
    )
    added_total = sum(len(v) for v in added_per_vm.values())

    metadata = merged_doc.setdefault("metadata", {})
    metadata["last_regenerate_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["last_regenerate_stats"] = {
        "db_candidates": len(candidates),
        "candidates_added": added_total,
        "vms_updated": len(added_per_vm),
    }

    # Stage the merged config in a scratch dir, then let reorder_config do the
    # telemetry pass (it re-reads from disk). Scratch writes never touch the
    # live config; only --apply copies the final result over.
    report: Dict[str, Any] = {
        "added": added_per_vm,
        "candidates_total": len(candidates),
        "log": log_lines,
        "wrote": False,
    }
    with tempfile.TemporaryDirectory(prefix="regen-chains-") as tmp:
        staging = Path(tmp) / "virtual_models.staging.yaml"
        chain_policy.write_yaml_atomic(staging, merged_doc, allow_unicode=True)
        total, reordered, reorder_log = reorder_chains.reorder_config(
            config_path=staging,
            telemetry_db=telemetry_db or reorder_chains.DEFAULT_TELEMETRY_DB,
            window_h=window_h,
            min_samples=min_samples,
            max_tps=max_tps,
            max_ttft_ms=max_ttft_ms,
            dry_run=False,  # staging only; the live write is gated below
        )
        report["total_models"] = total
        report["reordered"] = reordered
        report["log"] = log_lines + reorder_log

        if apply_changes:
            ts = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            backup_path = config_path.with_name(f"{config_path.name}.bak-pre-regen-{ts}")
            shutil.copy2(config_path, backup_path)
            shutil.copy2(staging, config_path)
            report["wrote"] = True
            report["log"].append(f"WROTE {config_path} (backup: {backup_path.name})")
        else:
            report["log"].append(f"DRY RUN — would write {config_path}")

    if added_total == 0 and reordered == 0:
        return 2, report
    return 0, report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate + reorder virtual model chains from providers_database.yaml"
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help=f"Path to virtual_models.yaml (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--providers-db", type=Path, default=DEFAULT_PROVIDERS_DB,
        help=f"Path to providers_database.yaml (default: {DEFAULT_PROVIDERS_DB})",
    )
    parser.add_argument(
        "--policy", type=Path, default=DEFAULT_POLICY_PATH,
        help=f"Path to fallback-chain exclusion policy (default: {DEFAULT_POLICY_PATH})",
    )
    parser.add_argument(
        "--coding-rankings", type=Path, default=None,
        help="Path to model_rankings.yaml (default: config/model_rankings.yaml)",
    )
    parser.add_argument(
        "--chat-rankings", type=Path, default=None,
        help="Path to chat_model_rankings.yaml (default: config/chat_model_rankings.yaml)",
    )
    parser.add_argument(
        "--telemetry-db", default=None,
        help=(
            "Path to telemetry SQLite "
            f"(default: TELEMETRY_DB_PATH env or {reorder_chains.DEFAULT_TELEMETRY_DB})"
        ),
    )
    parser.add_argument("--window-h", type=int, default=reorder_chains.DEFAULT_WINDOW_H)
    parser.add_argument("--min-samples", type=int, default=reorder_chains.DEFAULT_MIN_SAMPLES)
    parser.add_argument("--max-tps", type=float, default=reorder_chains.DEFAULT_MAX_TPS)
    parser.add_argument("--max-ttft-ms", type=float, default=reorder_chains.DEFAULT_MAX_TTFT_MS)
    parser.add_argument(
        "--apply", action="store_true",
        help="Write the regenerated config (default: dry-run, no writes)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        code, report = regenerate(
            config_path=args.config,
            providers_db_path=args.providers_db,
            policy_path=args.policy,
            coding_rankings_path=args.coding_rankings,
            chat_rankings_path=args.chat_rankings,
            telemetry_db=args.telemetry_db,
            window_h=args.window_h,
            min_samples=args.min_samples,
            max_tps=args.max_tps,
            max_ttft_ms=args.max_ttft_ms,
            apply_changes=args.apply,
        )
    except Exception as exc:  # surface as exit code, not traceback
        logger.error("regeneration failed: %r", exc)
        return 1

    for line in report.get("log", []):
        print(line)
    added = sum(len(v) for v in report.get("added", {}).values())
    logger.info(
        "done: %d candidates added across %d VMs, %d/%d models reordered (wrote=%s)",
        added,
        len(report.get("added", {})),
        report.get("reordered", 0),
        report.get("total_models", 0),
        report.get("wrote", False),
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
