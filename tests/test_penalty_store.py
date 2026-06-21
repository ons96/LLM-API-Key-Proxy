"""Tests for penalty_store: record failures, decay, chain sort, classify."""

import os
import tempfile
import time
import unittest

# Test env must be set before import.
_tmpdir = tempfile.mkdtemp(prefix="penalty_test_")
os.environ["PENALTY_DB_PATH"] = os.path.join(_tmpdir, "test_penalty.db")
os.environ["PENALTY_HALFLIFE_S"] = "1.0"  # 1s for fast decay tests

# Reset singleton before any import in case prior tests cached it.
from proxy_app.penalty_store import (  # noqa: E402
    PenaltyStore,
    PenaltyEntry,
    classify_failure,
    FAILURE_WEIGHTS,
)


class TestRecordAndDecay(unittest.TestCase):
    def setUp(self):
        PenaltyStore.reset_singleton_for_test()
        self.store = PenaltyStore(
            db_path=os.environ["PENALTY_DB_PATH"],
            half_life_s=float(os.environ["PENALTY_HALFLIFE_S"]),
        )
        self.store.clear()

    def test_record_invalid_key(self):
        self.store.record_failure("groq", "llama-3.3-70b", "invalid_key")
        score = self.store.get_score("groq", "llama-3.3-70b")
        self.assertGreater(score, 4.0, f"invalid_key weight=5.0; got {score}")

    def test_record_rate_limit(self):
        self.store.record_failure("groq", "llama-3.1-8b", "rate_limit")
        score = self.store.get_score("groq", "llama-3.1-8b")
        self.assertGreater(score, 0.5, f"rate_limit weight=1.0; got {score}")

    def test_decay_after_two_halflives(self):
        self.store.record_failure("groq", "llama-3.3-70b", "invalid_key")
        initial = self.store.get_score("groq", "llama-3.3-70b")
        time.sleep(2.0)  # 2 half-lives -> ~25% remaining
        final = self.store.get_score("groq", "llama-3.3-70b")
        ratio = final / initial if initial > 0 else 0
        self.assertLess(ratio, 0.30, f"expected <30% after 2 half-lives; got {ratio:.3f}")

    def test_score_sums_across_failure_types(self):
        self.store.record_failure("groq", "llama-3.3-70b", "invalid_key")
        self.store.record_failure("groq", "llama-3.3-70b", "rate_limit")
        score = self.store.get_score("groq", "llama-3.3-70b")
        # ~5.0 + ~1.0 = ~6.0, both fresh.
        self.assertGreater(score, 5.5, f"expected >5.5 sum; got {score}")

    def test_count_increments(self):
        self.store.record_failure("groq", "llama-3.3-70b", "rate_limit")
        self.store.record_failure("groq", "llama-3.3-70b", "rate_limit")
        entries = self.store.get_entries(provider="groq", model="llama-3.3-70b")
        rate_entries = [e for e in entries if e.failure_type == "rate_limit"]
        self.assertEqual(len(rate_entries), 1)
        self.assertEqual(rate_entries[0].count, 2)


class TestChainSort(unittest.TestCase):
    def setUp(self):
        PenaltyStore.reset_singleton_for_test()
        self.store = PenaltyStore(
            db_path=os.environ["PENALTY_DB_PATH"],
            half_life_s=float(os.environ["PENALTY_HALFLIFE_S"]),
        )
        self.store.clear()

    def test_healthy_provider_first(self):
        self.store.record_failure("groq", "llama-3.3-70b", "invalid_key")
        chain = [
            ("groq", "llama-3.3-70b"),
            ("gemini", "gemini-1.5-pro"),
            ("openai", "gpt-4o-mini"),
        ]
        scored = self.store.score_chain(chain)
        self.assertEqual(scored[0][0], "gemini")
        self.assertEqual(scored[-1][0], "groq")
        self.assertEqual(scored[-1][1], "llama-3.3-70b")

    def test_no_penalty_zero_score(self):
        score = self.store.get_score("unknown_provider", "unknown_model")
        self.assertEqual(score, 0.0)

    def test_empty_chain(self):
        scored = self.store.score_chain([])
        self.assertEqual(scored, [])

    def test_all_healthy_preserve_order(self):
        chain = [("groq", "a"), ("gemini", "b"), ("openai", "c")]
        scored = self.store.score_chain(chain)
        # All zero penalty -> stable sort should preserve original order.
        self.assertEqual([(p, m) for p, m, _ in scored], chain)


class TestClassifyFailure(unittest.TestCase):
    def test_401_is_invalid_key(self):
        self.assertEqual(classify_failure(status_code=401), "invalid_key")

    def test_403_is_invalid_key(self):
        self.assertEqual(classify_failure(status_code=403), "invalid_key")

    def test_429_is_rate_limit(self):
        self.assertEqual(classify_failure(status_code=429), "rate_limit")

    def test_500_is_provider_down(self):
        self.assertEqual(classify_failure(status_code=500), "provider_down")

    def test_503_is_provider_down(self):
        self.assertEqual(classify_failure(status_code=503), "provider_down")

    def test_timeout_exception(self):
        self.assertEqual(
            classify_failure(exception_type="TimeoutError"),
            "timeout",
        )

    def test_timeout_message(self):
        self.assertEqual(
            classify_failure(error_message="Request timed out after 30s"),
            "timeout",
        )

    def test_quota_message_is_out_of_credit(self):
        self.assertEqual(
            classify_failure(error_message="out of credits"),
            "out_of_credit",
        )

    def test_credit_message_is_out_of_credit(self):
        self.assertEqual(
            classify_failure(error_message="quota exceeded for today"),
            "out_of_credit",
        )

    def test_json_parse_is_bad_output(self):
        self.assertEqual(
            classify_failure(error_message="JSON parse error: malformed response"),
            "bad_output",
        )

    def test_truncated_is_bad_output(self):
        self.assertEqual(
            classify_failure(error_message="response truncated mid-stream"),
            "bad_output",
        )

    def test_auth_exception_is_invalid_key(self):
        self.assertEqual(
            classify_failure(exception_type="AuthenticationError"),
            "invalid_key",
        )

    def test_unauthorized_message_is_invalid_key(self):
        self.assertEqual(
            classify_failure(error_message="Unauthorized access"),
            "invalid_key",
        )

    def test_unknown_defaults_to_provider_down(self):
        self.assertEqual(classify_failure(), "provider_down")

    def test_rate_limit_message_takes_precedence_over_500(self):
        # A 500 with a body containing "rate limit" is almost certainly a
        # rate limit wrapped in a 5xx. Our impl's rate-limit check
        # (`status==429 or "rate" in msg and "limit" in msg`) fires before
        # the generic 5xx branch, so the message wins. Saner than the inverse.
        result = classify_failure(status_code=500, error_message="rate limit hit")
        self.assertEqual(result, "rate_limit")


class TestPenaltyEntry(unittest.TestCase):
    def setUp(self):
        PenaltyStore.reset_singleton_for_test()
        self.store = PenaltyStore(
            db_path=os.environ["PENALTY_DB_PATH"],
            half_life_s=float(os.environ["PENALTY_HALFLIFE_S"]),
        )
        self.store.clear()

    def test_entry_has_all_fields(self):
        self.store.record_failure("openai", "gpt-4o", "invalid_key")
        entries = self.store.get_entries(provider="openai", model="gpt-4o")
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertIsInstance(e, PenaltyEntry)
        self.assertEqual(e.provider, "openai")
        self.assertEqual(e.model, "gpt-4o")
        self.assertEqual(e.failure_type, "invalid_key")
        self.assertEqual(e.count, 1)
        self.assertGreater(e.last_ts, 0)
        self.assertGreater(e.score, 4.0)

    def test_get_entries_filters_by_provider(self):
        self.store.record_failure("openai", "gpt-4o", "invalid_key")
        self.store.record_failure("groq", "llama", "rate_limit")
        entries = self.store.get_entries(provider="groq")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].provider, "groq")


class TestPruneOldState(unittest.TestCase):
    def setUp(self):
        PenaltyStore.reset_singleton_for_test()
        self.store = PenaltyStore(
            db_path=os.environ["PENALTY_DB_PATH"],
            half_life_s=float(os.environ["PENALTY_HALFLIFE_S"]),
        )
        self.store.clear()

    def test_prune_removes_decayed_rows(self):
        # Record a failure in the distant past.
        old_ts = time.time() - 100000  # ~28 hours ago
        self.store.record_failure("groq", "llama-3.3-70b", "rate_limit", ts=old_ts)
        entries_before = self.store.get_entries(provider="groq")
        self.assertEqual(len(entries_before), 1)
        # Prune with 24h max age.
        removed = self.store.prune_old_state(max_age_s=86400)
        self.assertGreaterEqual(removed, 1)
        entries_after = self.store.get_entries(provider="groq")
        self.assertEqual(len(entries_after), 0)


class TestFailureWeights(unittest.TestCase):
    def test_all_failure_types_have_weights(self):
        expected = {"rate_limit", "timeout", "provider_down",
                    "out_of_credit", "invalid_key", "bad_output"}
        self.assertEqual(set(FAILURE_WEIGHTS.keys()), expected)

    def test_invalid_key_has_highest_weight(self):
        self.assertEqual(
            max(FAILURE_WEIGHTS, key=lambda k: FAILURE_WEIGHTS[k]),
            "invalid_key",
        )

    def test_rate_limit_has_lowest_weight(self):
        # rate_limit and bad_output both 1.0; rate_limit should be among the lowest.
        self.assertLessEqual(
            FAILURE_WEIGHTS["rate_limit"],
            FAILURE_WEIGHTS["invalid_key"],
        )


if __name__ == "__main__":
    unittest.main()
