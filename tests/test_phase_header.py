"""
Tests for X-Phase-Router-Phase header routing (task-board #252 / #347).

When an opencode client (the opencode-phase-router plugin) sends the
`X-Phase-Router-Phase` header, the gateway's `auto` virtual model should
trust the client's classification and route directly to the phase-mapped
chain, bypassing semantic/keyword routing. The tool-capability guard
still applies.

These tests exercise resolve_auto directly with the phase_header kwarg.
They do not require pytest fixtures for the FastAPI Request — the header
extraction in router_wrapper.py is a one-liner covered by integration
tests. pytest is used to match the existing test_semantic_router.py
style; they also run under unittest via pytest-free shims where needed.
"""

import unittest

try:
    import pytest  # noqa: F401
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False

# Base class: use pytest.TestCase if available, else unittest.TestCase.
_BaseTestCase = unittest.TestCase
from proxy_app.semantic_router import (
    PHASE_CHAIN_MAP,
    TOOL_CAPABLE_CHAINS,
    resolve_auto,
)
from proxy_app.intent_detector import MessageIntent


class TestPhaseHeaderRouting(_BaseTestCase):
    """resolve_auto(phase_header=...) routes to the phase chain."""

    def test_plan_routes_to_agent_oracle(self):
        req = {"messages": [{"role": "user", "content": "design the auth module"}]}
        result = resolve_auto(req, phase_header="plan")
        assert result.chain == "agent-oracle"
        assert result.source == "phase-header"
        assert result.confidence == 1.0
        assert result.reasoning == "client phase=plan"

    def test_build_exploration_routes_to_coding_elite(self):
        req = {"messages": [{"role": "user", "content": "implement feature X"}]}
        result = resolve_auto(req, phase_header="build_exploration")
        assert result.chain == "coding-elite"
        assert result.source == "phase-header"

    def test_build_iteration_routes_to_coding_fast(self):
        req = {"messages": [{"role": "user", "content": "fix the typo"}]}
        result = resolve_auto(req, phase_header="build_iteration")
        assert result.chain == "coding-fast"
        assert result.source == "phase-header"

    def test_build_verification_routes_to_coding_elite(self):
        req = {"messages": [{"role": "user", "content": "run the tests"}]}
        result = resolve_auto(req, phase_header="build_verification")
        assert result.chain == "coding-elite"
        assert result.source == "phase-header"

    def test_unknown_phase_falls_through_to_keyword(self):
        """An unrecognized phase header must NOT break routing — it falls
        through to the normal semantic/keyword path."""
        req = {"messages": [{"role": "user", "content": "hi"}]}
        result = resolve_auto(req, phase_header="bogus_phase")
        # Should NOT be source=phase-header; should be keyword/default.
        assert result.source != "phase-header"
        assert result.chain  # any valid chain

    def test_no_phase_header_normal_flow(self):
        """Omitting phase_header (the default for non-opencode clients)
        must produce the normal routing result."""
        req = {"messages": [{"role": "user", "content": "hi"}]}
        result_no_header = resolve_auto(req)
        result_none_header = resolve_auto(req, phase_header=None)
        assert result_no_header.chain == result_none_header.chain
        assert result_no_header.source == result_none_header.source

    def test_empty_phase_header_treated_as_absent(self):
        """An empty-string header (e.g. plugin sent the header but had no
        classification yet) must not crash and must fall through."""
        req = {"messages": [{"role": "user", "content": "hi"}]}
        result = resolve_auto(req, phase_header="")
        # Empty string is falsy → falls through to keyword/default.
        assert result.source != "phase-header"
        assert result.chain

    def test_intent_is_unknown_when_phase_header_used(self):
        """When the client classifies the phase, the gateway doesn't run
        intent detection — intent reflects the CLIENT's view (UNKNOWN),
        not a re-derived one."""
        req = {"messages": [{"role": "user", "content": "anything"}]}
        result = resolve_auto(req, phase_header="plan")
        assert result.intent == MessageIntent.UNKNOWN


class TestPhaseHeaderToolGuard(_BaseTestCase):
    """The tool-capability guard applies even when a phase header is
    present. A phase chain that isn't tool-capable gets rerouted when
    the request carries tools."""

    def test_plan_with_tools_keeps_agent_oracle(self):
        """agent-oracle is tool-capable, so tools don't reroute it."""
        req = {
            "messages": [{"role": "user", "content": "plan the work"}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
        }
        result = resolve_auto(req, phase_header="plan")
        assert result.chain == "agent-oracle"
        assert result.chain in TOOL_CAPABLE_CHAINS

    def test_build_iteration_with_tools_keeps_coding_fast(self):
        req = {
            "messages": [{"role": "user", "content": "fix it"}],
            "tools": [{"type": "function", "function": {"name": "edit"}}],
        }
        result = resolve_auto(req, phase_header="build_iteration")
        assert result.chain == "coding-fast"

    def test_tool_choice_alone_respected_with_phase_header(self):
        req = {
            "messages": [{"role": "user", "content": "plan"}],
            "tool_choice": "auto",
        }
        result = resolve_auto(req, phase_header="plan")
        # plan -> agent-oracle is tool-capable, so unchanged.
        assert result.chain == "agent-oracle"


class TestPhaseChainMapConsistency(_BaseTestCase):
    """PHASE_CHAIN_MAP must cover all four phases and map to known chains."""

    def test_covers_four_phases(self):
        assert set(PHASE_CHAIN_MAP.keys()) == {
            "plan",
            "build_exploration",
            "build_iteration",
            "build_verification",
        }

    def test_all_phase_chains_are_tool_capable(self):
        """Every phase chain must be tool-capable — agentic coding work
        always carries tools, and a phase chain that can't handle tools
        would break the opencode agent loop."""
        for phase, chain in PHASE_CHAIN_MAP.items():
            assert chain in TOOL_CAPABLE_CHAINS, (
                f"phase {phase!r} maps to non-tool chain {chain!r}"
            )
