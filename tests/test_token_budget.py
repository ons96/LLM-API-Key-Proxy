"""Tests for src/proxy_app/routing/token_budget.py — task-board #479.

Verifies:
- Per-class max_tokens defaults (greeting small, code-gen larger)
- Context-limit clamping edge cases (huge input, tiny context)
- Tier multiplier raises cap within the clamp
- Explicit client max_tokens is NEVER overridden
- Dry-run does not mutate the request; enabled mode applies it
- TOKEN_BUDGET_OVERRIDES env override

Run: python -m pytest tests/test_token_budget.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from proxy_app.routing.request_features import (  # noqa: E402
    extract_request_features,
)
from proxy_app.routing.token_budget import (  # noqa: E402
    FLOOR_TOKEN_BUDGET,
    apply_token_budget,
    compute_max_tokens,
    env_overrides,
    load_budget_config,
)

_REAL_YAML = _REPO_ROOT / "config" / "token_budget.yaml"


class TestComputeMaxTokens(unittest.TestCase):
    def test_per_class_defaults(self):
        # A greeting must not burn 4096 tokens.
        self.assertEqual(compute_max_tokens("greeting", 20, 131072), 128)
        self.assertEqual(compute_max_tokens("short-qa", 30, 131072), 512)
        self.assertEqual(compute_max_tokens("code-gen", 50, 131072), 2048)
        self.assertEqual(compute_max_tokens("reasoning", 50, 131072), 4096)
        self.assertEqual(compute_max_tokens("agentic-multi-step", 50, 131072), 8192)

    def test_unknown_class_falls_back(self):
        self.assertEqual(compute_max_tokens("mystery-class", 10, 131072), 512)

    def test_context_clamp_small_context(self):
        # Tiny context forces a hard clamp well below the class default.
        cap = compute_max_tokens("code-edit", 200, 1024)
        self.assertLess(cap, 1024)  # clamped by 1024 - 200 - 256 = 568

    def test_context_clamp_huge_input(self):
        # Input already exceeds context -> floor (never negative/zero).
        cap = compute_max_tokens("reasoning", 100_000, 8192)
        self.assertGreaterEqual(cap, FLOOR_TOKEN_BUDGET)

    def test_tier_multiplier_from_config(self):
        # Through the real config, T4 raises code-edit 1024 -> 2048.
        cfg = load_budget_config(str(_REAL_YAML))
        base = compute_max_tokens("code-edit", 200, 131072,
                                  tier=4,
                                  tier_multiplier=cfg["tier_multiplier"])
        self.assertEqual(base, 2048)
        tier0 = compute_max_tokens("code-edit", 200, 131072,
                                   tier=0,
                                   tier_multiplier=cfg["tier_multiplier"])
        self.assertEqual(tier0, 1024)

    def test_env_override(self):
        os.environ["TOKEN_BUDGET_OVERRIDES"] = '{"reasoning": 1234}'
        try:
            self.assertEqual(env_overrides().get("reasoning"), 1234)
            self.assertEqual(compute_max_tokens("reasoning", 20, 131072), 1234)
        finally:
            del os.environ["TOKEN_BUDGET_OVERRIDES"]


class TestApplyTokenBudgetRequest(unittest.TestCase):
    def _features(self, body):
        return extract_request_features(body)

    def test_client_max_tokens_never_overridden(self):
        # Both OpenAI max_tokens and responses max_completion_tokens respect intent.
        for key in ("max_tokens", "max_completion_tokens"):
            body = {"model": "x", "messages": [{"role": "user", "content": "hi"}], key: 1}
            feats = self._features(body)
            hdr = apply_token_budget(body, feats, 131072, enabled=True)
            self.assertIsNone(hdr, f"{key} should return None but got {hdr}")
            self.assertEqual(body[key], 1, f"{key} must not be overridden")

    def test_dry_run_does_not_mutate(self):
        body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
        feats = self._features(body)
        hdr = apply_token_budget(body, feats, 131072, enabled=False)
        self.assertIn("dry-run", hdr)
        self.assertNotIn("max_tokens", body)  # unchanged in dry-run
        self.assertNotIn("max_completion_tokens", body)

    def test_enabled_applies_task_class_cap(self):
        body = {"model": "x", "messages": [{"role": "user", "content": "hello"}]}
        feats = self._features(body)
        hdr = apply_token_budget(body, feats, 131072, enabled=True)
        self.assertIn("applied", hdr)
        self.assertEqual(body["max_tokens"], 128)  # greeting class

    def test_backs_off_when_absent(self):
        # No budget changes if features indicate has_max_tokens even pre-parse.
        body = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 2048}
        feats = self._features(body)
        self.assertIsNone(apply_token_budget(body, feats, 131072, enabled=True))


if __name__ == "__main__":
    unittest.main()
