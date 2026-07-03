"""Tests for #343 free-model baseline enforcement in reorder_chains.py.

Verifies:
- is_free_provider classifies the 4 free kinds + paid/missing correctly
- apply_free_baseline demotes non-free entries below the free floor to slot 99
- free entries are never demoted regardless of score
- entries with invalid scores (< 0, insufficient samples) are not demoted
- when no free entries have valid scores, chain is unchanged (fallback path)
- log line per demotion matches the #343-required format
- demotion preserves order of kept entries + extra dict keys

Run: python -m pytest tests/test_free_baseline.py -v
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "reorder_chains",
    _REPO_ROOT / "scripts" / "reorder_chains.py",
)
assert _spec is not None and _spec.loader is not None
reorder_chains = importlib.util.module_from_spec(_spec)
sys.modules["reorder_chains"] = reorder_chains
_spec.loader.exec_module(reorder_chains)

from reorder_chains import (  # type: ignore
    apply_free_baseline,
    is_free_provider,
)


@dataclass
class _Cat:
    """Minimal stand-in for rank_models.ProviderCategory."""

    name: str
    kind: str = "paid"


class TestIsFreeProvider(unittest.TestCase):
    def test_free_kinds(self):
        for kind in ("free_unlimited", "free_daily", "free_one_time", "no_key"):
            cats = {"p": _Cat("p", kind)}
            self.assertTrue(is_free_provider("p", cats), f"{kind} should be free")

    def test_paid_is_not_free(self):
        cats = {"p": _Cat("p", "paid")}
        self.assertFalse(is_free_provider("p", cats))

    def test_missing_provider_not_free(self):
        self.assertFalse(is_free_provider("ghost", {}))

    def test_unknown_kind_not_free(self):
        cats = {"p": _Cat("p", "unknown_kind")}
        self.assertFalse(is_free_provider("p", cats))


class TestApplyFreeBaseline(unittest.TestCase):
    def _chain(self, entries: List[Tuple[str, str, int]]) -> List[Dict]:
        return [{"provider": p, "model": m, "priority": pr} for p, m, pr in entries]

    def test_demotes_non_free_below_floor(self):
        # free floor = min(0.80, 0.50) = 0.50. paid at 0.30 < floor -> demote.
        chain = self._chain(
            [("groq", "good", 1), ("nvidia", "ok", 2), ("openai", "bad", 3)]
        )
        cats = {
            "groq": _Cat("groq", "free_unlimited"),
            "nvidia": _Cat("nvidia", "free_unlimited"),
            "openai": _Cat("openai", "paid"),
        }
        scores = {
            ("groq", "good"): 0.80,
            ("nvidia", "ok"): 0.50,
            ("openai", "bad"): 0.30,
        }
        new_chain, logs = apply_free_baseline(chain, scores, cats)
        # openai demoted to slot 99; groq + nvidia keep order at 1, 2
        self.assertEqual(new_chain[0]["provider"], "groq")
        self.assertEqual(new_chain[0]["priority"], 1)
        self.assertEqual(new_chain[1]["provider"], "nvidia")
        self.assertEqual(new_chain[1]["priority"], 2)
        self.assertEqual(new_chain[2]["provider"], "openai")
        self.assertEqual(new_chain[2]["priority"], 99)
        self.assertEqual(len(logs), 1)
        self.assertIn("[BASELINE]", logs[0])
        self.assertIn("openai/bad", logs[0])
        self.assertIn("demoted below free baseline", logs[0])

    def test_free_entry_never_demoted_even_if_lowest(self):
        # free entry has the lowest score but must not be demoted.
        chain = self._chain([("groq", "low", 1), ("openai", "high", 2)])
        cats = {
            "groq": _Cat("groq", "free_unlimited"),
            "openai": _Cat("openai", "paid"),
        }
        scores = {("groq", "low"): 0.20, ("openai", "high"): 0.90}
        new_chain, logs = apply_free_baseline(chain, scores, cats)
        # floor = 0.20; openai 0.90 >= floor -> not demoted; groq kept.
        # No demotions -> chain returned unchanged.
        self.assertEqual(len(logs), 0)
        self.assertEqual([c["provider"] for c in new_chain], ["groq", "openai"])

    def test_invalid_scores_not_demoted(self):
        # score = -1 (insufficient samples) must NOT be demoted even if
        # non-free and below floor.
        chain = self._chain([("groq", "free1", 1), ("openai", "unknown", 2)])
        cats = {
            "groq": _Cat("groq", "free_unlimited"),
            "openai": _Cat("openai", "paid"),
        }
        scores = {("groq", "free1"): 0.60, ("openai", "unknown"): -1.0}
        new_chain, logs = apply_free_baseline(chain, scores, cats)
        self.assertEqual(len(logs), 0)
        # openai has invalid score -> not demoted, stays in ranked order
        self.assertEqual(new_chain[-1]["provider"], "openai")
        self.assertNotEqual(new_chain[-1]["priority"], 99)

    def test_no_free_data_returns_unchanged(self):
        # No free providers -> fallback to current behaviour (unchanged).
        chain = self._chain([("openai", "a", 1), ("openai", "b", 2)])
        cats = {"openai": _Cat("openai", "paid")}
        scores = {("openai", "a"): 0.90, ("openai", "b"): 0.10}
        new_chain, logs = apply_free_baseline(chain, scores, cats)
        self.assertEqual(len(logs), 0)
        self.assertEqual([c["priority"] for c in new_chain], [1, 2])

    def test_no_free_valid_scores_returns_unchanged(self):
        # Free providers exist but all have invalid scores -> unchanged.
        chain = self._chain([("groq", "free1", 1), ("openai", "paid1", 2)])
        cats = {
            "groq": _Cat("groq", "free_unlimited"),
            "openai": _Cat("openai", "paid"),
        }
        scores = {("groq", "free1"): -1.0, ("openai", "paid1"): 0.90}
        new_chain, logs = apply_free_baseline(chain, scores, cats)
        self.assertEqual(len(logs), 0)
        self.assertEqual([c["priority"] for c in new_chain], [1, 2])

    def test_multiple_demotions_get_99_100_101(self):
        chain = self._chain(
            [
                ("groq", "free1", 1),
                ("openai", "bad1", 2),
                ("openai", "bad2", 3),
                ("openai", "bad3", 4),
            ]
        )
        cats = {
            "groq": _Cat("groq", "free_unlimited"),
            "openai": _Cat("openai", "paid"),
        }
        scores = {
            ("groq", "free1"): 0.70,
            ("openai", "bad1"): 0.10,
            ("openai", "bad2"): 0.20,
            ("openai", "bad3"): 0.30,
        }
        new_chain, logs = apply_free_baseline(chain, scores, cats)
        # floor = 0.70; all 3 openai entries below -> demoted to 99, 100, 101
        demoted = [c for c in new_chain if c["provider"] == "openai"]
        self.assertEqual([c["priority"] for c in demoted], [99, 100, 101])
        self.assertEqual(len(logs), 3)
        kept = [c for c in new_chain if c["provider"] == "groq"]
        self.assertEqual(kept[0]["priority"], 1)

    def test_preserves_extra_keys(self):
        chain = [
            {"provider": "groq", "model": "f", "priority": 1, "notes": "keep"},
            {
                "provider": "openai",
                "model": "p",
                "priority": 2,
                "capabilities": {"tools": True},
            },
        ]
        cats = {
            "groq": _Cat("groq", "free_unlimited"),
            "openai": _Cat("openai", "paid"),
        }
        scores = {("groq", "f"): 0.80, ("openai", "p"): 0.10}
        new_chain, _ = apply_free_baseline(chain, scores, cats)
        free_entry = next(c for c in new_chain if c["provider"] == "groq")
        paid_entry = next(c for c in new_chain if c["provider"] == "openai")
        self.assertEqual(free_entry["notes"], "keep")
        self.assertEqual(paid_entry["capabilities"], {"tools": True})

    def test_baseline_floor_min_argument(self):
        # floor = max(baseline, baseline_floor_min). baseline=0.50 but
        # floor_min=0.60 -> paid at 0.55 (< 0.60) demoted despite > baseline.
        chain = self._chain([("groq", "free1", 1), ("openai", "mid", 2)])
        cats = {
            "groq": _Cat("groq", "free_unlimited"),
            "openai": _Cat("openai", "paid"),
        }
        scores = {("groq", "free1"): 0.50, ("openai", "mid"): 0.55}
        new_chain, logs = apply_free_baseline(
            chain, scores, cats, baseline_floor_min=0.60
        )
        self.assertEqual(len(logs), 1)
        self.assertEqual(new_chain[-1]["provider"], "openai")
        self.assertEqual(new_chain[-1]["priority"], 99)


if __name__ == "__main__":
    unittest.main()
