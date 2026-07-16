"""Hermetic coverage for non-destructive fallback-chain exclusions (#401)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from stat import S_IMODE

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import chain_policy
import generate_virtual_models
import rebuild_chains


class TestChainPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = chain_policy.load_policy()

    def test_required_blocks_and_allowed_neighbors(self) -> None:
        cases = (
            ("kilo", "model", True),
            ("kilocode", "model", False),
            ("antigravity", "model", True),
            ("g4f_nvidia", "model", True),
            ("g4f", "model", False),
            ("cliproxyapi", "gemini-3-pro", True),
            ("cliproxyapi", "gemini-3-flash", False),
            ("cliproxyapi", "gemini-3-flash-preview", True),
            ("cliproxyapi", "gemini-2.5-flash", False),
            ("cerebras", "qwen-3-235b-a22b-instruct-2507", True),
            ("supacoder", "gpt-5.4", True),
            ("gemini", "gemini-3-pro", True),
            ("gemini", "gemini-3.1-pro", True),
        )
        for provider, model, blocked in cases:
            with self.subTest(provider=provider, model=model):
                self.assertEqual(
                    chain_policy.blocked_reason(provider, model, self.policy) is not None,
                    blocked,
                )

    def test_matching_is_case_insensitive_and_uses_model_component(self) -> None:
        self.assertIsNotNone(
            chain_policy.blocked_reason(
                "CEREBRAS", "vendor/QWEN-3-235B-A22B-INSTRUCT-2507", self.policy
            )
        )
        self.assertIsNone(
            chain_policy.blocked_reason("KILOCODE", "model", self.policy)
        )

    def test_sanitize_filters_dedupes_and_preserves_metadata(self) -> None:
        chain = chain_policy.sanitize_chain(
            [
                {"provider": "nvidia", "model": "model-a", "notes": "keep"},
                {"provider": "kilo", "model": "dead"},
                {"provider": "NVIDIA", "model": "MODEL-A"},
                {"provider": "g4f_nvidia", "model": "dead"},
            ],
            self.policy,
            max_entries=2,
        )
        self.assertEqual(len(chain), 2)
        self.assertEqual(chain[0]["notes"], "keep")
        self.assertEqual(
            [(entry["provider"], entry["priority"]) for entry in chain],
            [("nvidia", 1), ("groq", 2)],
        )
        self.assertEqual(chain[1]["model"], "llama-3.3-70b-versatile")

    def test_sanitize_empty_chain_injects_direct_fallback(self) -> None:
        chain = chain_policy.sanitize_chain([], self.policy)
        self.assertEqual(
            chain,
            [
                {
                    "provider": "groq",
                    "model": "llama-3.3-70b-versatile",
                    "priority": 1,
                }
            ],
        )

    def test_direct_fallback_requires_full_model_identity(self) -> None:
        chain = chain_policy.sanitize_chain(
            [{"provider": "groq", "model": "vendor-a/llama-3.3-70b-versatile"}],
            self.policy,
        )
        self.assertEqual(
            [(entry["provider"], entry["model"]) for entry in chain],
            [
                ("groq", "vendor-a/llama-3.3-70b-versatile"),
                ("groq", "llama-3.3-70b-versatile"),
            ],
        )

    def test_generator_filters_candidates_before_cap(self) -> None:
        models = [
            {"id": "kilo/model", "swe_bench": 100},
            {"id": "cerebras/qwen-3-235b-a22b-instruct-2507", "swe_bench": 99},
            {"id": "groq/llama-3.3-70b-versatile", "swe_bench": 98},
        ]
        chain = generate_virtual_models.generate_fallback_chain(
            models,
            {"swe_bench": 1.0},
            "coding",
            min_score=0,
            max_models=2,
            policy=self.policy,
        )
        self.assertEqual(
            [(entry["provider"], entry["model"]) for entry in chain],
            [("groq", "llama-3.3-70b-versatile")],
        )

    def test_generated_merge_preserves_live_only_configuration(self) -> None:
        live = {
            "agent_profiles": {"agent": {"model": "keep"}},
            "virtual_models": {
                "agent-oracle": {"fallback_chain": [{"provider": "kilo", "model": "kept"}]},
                "coding-fast": {
                    "description": "old",
                    "auto_continue_on_truncate": True,
                    "default_max_tokens": 2048,
                    "auto_route": {"keep": True},
                    "settings": {"timeout_ms": 1000},
                    "fallback_chain": [{"provider": "groq", "model": "old"}],
                },
            },
        }
        generated = {
            "virtual_models": {
                "coding-fast": {
                    "description": "new",
                    "settings": {"timeout_ms": 2},
                    "fallback_chain": [{"provider": "groq", "model": "new"}],
                }
            }
        }
        merged = generate_virtual_models.merge_generated_virtual_models(live, generated)
        self.assertEqual(merged["agent_profiles"], live["agent_profiles"])
        self.assertEqual(merged["virtual_models"]["agent-oracle"], live["virtual_models"]["agent-oracle"])
        coding_fast = merged["virtual_models"]["coding-fast"]
        self.assertEqual(coding_fast["description"], "old")
        self.assertTrue(coding_fast["auto_continue_on_truncate"])
        self.assertEqual(coding_fast["default_max_tokens"], 2048)
        self.assertEqual(coding_fast["auto_route"], {"keep": True})
        self.assertEqual(coding_fast["settings"], {"timeout_ms": 1000})
        self.assertEqual(
            coding_fast["fallback_chain"], [{"provider": "groq", "model": "new"}]
        )

    def test_generated_merge_rejects_malformed_live_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "live coding-fast virtual model"):
            generate_virtual_models.merge_generated_virtual_models(
                {"virtual_models": {"coding-fast": "invalid"}},
                {
                    "virtual_models": {
                        "coding-fast": {
                            "fallback_chain": [{"provider": "groq", "model": "new"}]
                        }
                    }
                },
            )

    def test_atomic_yaml_write_preserves_existing_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("old: value\n")
            path.chmod(0o640)
            chain_policy.write_yaml_atomic(path, {"new": "value"})
            self.assertEqual(yaml.safe_load(path.read_text()), {"new": "value"})
            self.assertEqual(S_IMODE(path.stat().st_mode), 0o640)
            self.assertFalse(list(Path(directory).glob(".config.yaml.*.tmp")))


class TestRebuildChains(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = chain_policy.load_policy()

    def test_rebuild_filters_target_only_and_preserves_other_data(self) -> None:
        document = {
            "metadata": {"owner": "keep"},
            "providers": {"catalog": {"endpoint": "keep"}},
            "agent_profiles": {"agent": {"model": "keep"}},
            "virtual_models": {
                "coding-fast": {
                    "description": "target",
                    "fallback_chain": [
                        {"provider": "kilo", "model": "dead"},
                        {"provider": "nvidia", "model": "kept", "notes": "metadata"},
                    ],
                },
                "agent-oracle": {
                    "fallback_chain": [{"provider": "kilo", "model": "catalog-only"}]
                },
            },
        }
        rebuilt, changed = rebuild_chains.rebuild_document(
            document,
            {
                "coding-fast": [
                    ("supacoder", "gpt-5.4"),
                    ("kilocode", "allowed"),
                    ("groq", "llama-3.3-70b-versatile"),
                ]
            },
            self.policy,
        )
        self.assertEqual(changed, ["coding-fast"])
        chain = rebuilt["virtual_models"]["coding-fast"]["fallback_chain"]
        self.assertEqual(
            [(entry["provider"], entry["model"]) for entry in chain],
            [
                ("kilocode", "allowed"),
                ("groq", "llama-3.3-70b-versatile"),
                ("nvidia", "kept"),
            ],
        )
        self.assertEqual(chain[-1]["notes"], "metadata")
        self.assertEqual(rebuilt["metadata"], document["metadata"])
        self.assertEqual(rebuilt["providers"], document["providers"])
        self.assertEqual(rebuilt["agent_profiles"], document["agent_profiles"])
        self.assertEqual(
            rebuilt["virtual_models"]["agent-oracle"],
            document["virtual_models"]["agent-oracle"],
        )

    def test_static_candidates_require_current_telemetry(self) -> None:
        candidates = {
            "coding-fast": [("groq", "llama-3.3-70b-versatile"), ("mistral", "x")],
            "chat-fast": [("gemini", "gemini-2.5-flash")],
        }
        filtered = rebuild_chains.filter_new_tops_by_observed_models(
            candidates,
            {"groq": [{"model": "llama-3.3-70b-versatile"}, "malformed"]},
        )
        self.assertEqual(filtered, {"coding-fast": [("groq", "llama-3.3-70b-versatile")]})
        self.assertNotIn("chat-fast", filtered)
        self.assertEqual(rebuild_chains.filter_new_tops_by_observed_models(candidates, {}), {})

    def test_rebuild_dry_run_creates_no_backup_or_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "virtual_models.yaml"
            original = {
                "virtual_models": {
                    "coding-fast": {
                        "fallback_chain": [{"provider": "kilo", "model": "dead"}]
                    }
                }
            }
            config.write_text(yaml.safe_dump(original, sort_keys=False))
            before = config.read_bytes()
            result = rebuild_chains.main(
                [
                    str(config),
                    "--working-models",
                    str(Path(directory) / "missing.json"),
                    "--dry-run",
                ]
            )
            self.assertEqual(result, 0)
            self.assertEqual(config.read_bytes(), before)
            self.assertFalse(list(Path(directory).glob("*.bak-pre-rebuild")))


if __name__ == "__main__":
    unittest.main()
