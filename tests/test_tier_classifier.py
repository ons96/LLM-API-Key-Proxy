"""Tier classifier tests (#476).

Covers the #465 fixture suite (trivial / edit / refactor / architecture),
X-Route-Tier override handling, escalation rules, and tier-floor checks.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proxy_app.routing.request_features import (
    Capabilities,
    PromptBucket,
    RequestFeatures,
    TaskClass,
    extract_request_features,
)
from proxy_app.routing.tier_classifier import (
    TIER_FLOORS,
    Tier,
    classify_request,
    load_model_scores,
    model_meets_floor,
    parse_tier_header,
)


def _feats(
    task_class: TaskClass = TaskClass.SHORT_QA,
    chars: int = 50,
    tools: int = 0,
    images: int = 0,
    files: int = 0,
    bucket: PromptBucket = PromptBucket.SHORT,
) -> RequestFeatures:
    return RequestFeatures(
        task_class=task_class,
        text_char_count=chars,
        tool_count=tools,
        image_count=images,
        file_count=files,
        prompt_length_bucket=bucket,
    )


# ---------------------------------------------------------------------------
# #465 fixture suite: task class -> base tier
# ---------------------------------------------------------------------------


class TestBaseTiers:
    def test_trivial_greeting(self):
        assert classify_request(_feats(TaskClass.GREETING)) == Tier.T0

    def test_short_qa(self):
        assert classify_request(_feats(TaskClass.SHORT_QA)) == Tier.T1

    @pytest.mark.parametrize(
        "tc,want",
        [
            (TaskClass.CODE_EDIT, Tier.T2),
            (TaskClass.SUMMARIZATION, Tier.T2),
            (TaskClass.VISION_CAPTION, Tier.T2),
            (TaskClass.CODE_GEN, Tier.T3),
            (TaskClass.REASONING, Tier.T3),
            (TaskClass.FILE_ANALYSIS, Tier.T3),
            (TaskClass.AGENTIC, Tier.T4),
        ],
    )
    def test_task_tier_map(self, tc, want):
        assert classify_request(_feats(tc)) == want

    def test_unknown_task_defaults_to_t1(self):
        assert classify_request(_feats(None)) == Tier.T1  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Escalations
# ---------------------------------------------------------------------------


class TestEscalations:
    def test_very_long_escalates_to_t3(self):
        feats = _feats(TaskClass.SHORT_QA, chars=9000, bucket=PromptBucket.VERY_LONG)
        assert classify_request(feats) == Tier.T3

    def test_long_escalates_t1_to_t2(self):
        feats = _feats(TaskClass.SHORT_QA, chars=3000, bucket=PromptBucket.LONG)
        assert classify_request(feats) == Tier.T2

    def test_long_does_not_escalate_t2(self):
        feats = _feats(TaskClass.CODE_EDIT, chars=3000, bucket=PromptBucket.LONG)
        assert classify_request(feats) == Tier.T2

    def test_many_tools_escalate_to_t3(self):
        feats = _feats(TaskClass.CODE_GEN, tools=3)
        assert classify_request(feats) == Tier.T3

    def test_single_tool_no_escalation(self):
        feats = _feats(TaskClass.SHORT_QA, tools=1)
        assert classify_request(feats) == Tier.T1

    def test_many_attachments_escalate_to_t3(self):
        feats = _feats(TaskClass.CODE_EDIT, images=2)
        assert classify_request(feats) == Tier.T3

    def test_single_attachment_escalates_to_t2(self):
        feats = _feats(TaskClass.SHORT_QA, images=1)
        assert classify_request(feats) == Tier.T2
        feats2 = _feats(TaskClass.SHORT_QA, files=1)
        assert classify_request(feats2) == Tier.T2

    def test_escalations_cap_at_t4(self):
        feats = _feats(TaskClass.AGENTIC, chars=9000, tools=5, images=3)
        assert classify_request(feats) == Tier.T4


# ---------------------------------------------------------------------------
# X-Route-Tier header override
# ---------------------------------------------------------------------------


class TestHeaderOverride:
    def test_header_wins_over_heuristics(self):
        assert classify_request(_feats(TaskClass.GREETING), "T4") == Tier.T4
        assert classify_request(_feats(TaskClass.AGENTIC), "T0") == Tier.T0

    def test_case_insensitive_and_int_form(self):
        assert classify_request(_feats(), "t2") == Tier.T2
        assert classify_request(_feats(), "3") == Tier.T3

    def test_invalid_header_ignored(self):
        assert classify_request(_feats(TaskClass.GREETING), "T9") == Tier.T0
        assert classify_request(_feats(TaskClass.GREETING), "banana") == Tier.T0
        assert classify_request(_feats(TaskClass.GREETING), "") == Tier.T0
        assert classify_request(_feats(TaskClass.GREETING), None) == Tier.T0

    def test_parse_tier_header(self):
        assert parse_tier_header("T0") == Tier.T0
        assert parse_tier_header("4") == Tier.T4
        assert parse_tier_header("T42") is None
        assert parse_tier_header("-1") is None


# ---------------------------------------------------------------------------
# End-to-end through extract_request_features (fixture bodies)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_greeting_body(self):
        feats = extract_request_features(
            {"messages": [{"role": "user", "content": "hello"}]}
        )
        assert classify_request(feats) == Tier.T0

    def test_edit_request_body(self):
        feats = extract_request_features(
            {"messages": [{"role": "user", "content": "fix the bug in parse() and update the tests"}]}
        )
        assert classify_request(feats) == Tier.T2

    def test_architecture_request_body(self):
        # short design+refactor ask lands CODE_EDIT -> T2; heavy design work
        # (long prompt / tools) escalates separately
        feats = extract_request_features(
            {"messages": [{"role": "user", "content": "design the routing architecture, explain the tradeoffs, then refactor the chain selector"}]}
        )
        assert classify_request(feats) == Tier.T2

    def test_architecture_body_with_tools_escalates(self):
        # 3 tool declarations -> extractor marks AGENTIC -> T4
        body = {
            "messages": [
                {"role": "user", "content": "design the routing architecture, explain the tradeoffs, then refactor the chain selector"},
            ],
            "tools": [{"type": "function", "function": {"name": "run", "parameters": {}}}] * 3,
        }
        feats = extract_request_features(body)
        assert classify_request(feats) == Tier.T4

    def test_agentic_body_with_tools(self):
        body = {
            "messages": [
                {"role": "user", "content": "first gather the logs then cross-check the failures then execute the fix"},
            ],
            "tools": [{"type": "function", "function": {"name": "run", "parameters": {}}}] * 3,
        }
        feats = extract_request_features(body)
        assert feats.requires(Capabilities.TOOL_CALLING)
        assert classify_request(feats) == Tier.T4


# ---------------------------------------------------------------------------
# Tier floors
# ---------------------------------------------------------------------------


class TestTierFloors:
    def test_t0_always_meets_floor(self):
        assert model_meets_floor(0.0, Tier.T0)
        assert model_meets_floor(-1.0, Tier.T0)

    def test_floor_bounds(self):
        assert model_meets_floor(0.8, Tier.T4)
        assert not model_meets_floor(0.5, Tier.T4)
        assert not model_meets_floor(0.0, Tier.T1)
        assert TIER_FLOORS[Tier.T0] <= TIER_FLOORS[Tier.T1] <= \
            TIER_FLOORS[Tier.T2] <= TIER_FLOORS[Tier.T3] <= TIER_FLOORS[Tier.T4]

    def test_load_model_scores_from_real_config(self):
        rankings = Path(__file__).resolve().parents[1] / "config" / "model_rankings.yaml"
        if not rankings.exists():
            pytest.skip("model_rankings.yaml not present")
        scores = load_model_scores(str(rankings))
        assert len(scores) >= 200
        # composite 0.0 = unranked model; everything else on (0, 1]
        assert all(0.0 <= v <= 1.0 for v in scores.values())
        nonzero = [v for v in scores.values() if v > 0.0]
        assert nonzero
        # top-tier model should clear T4 floor
        assert model_meets_floor(max(nonzero), Tier.T4)

    def test_missing_rankings_returns_empty(self, tmp_path):
        assert load_model_scores(str(tmp_path / "nope.yaml")) == {}
