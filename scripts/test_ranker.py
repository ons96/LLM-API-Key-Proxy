import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from proxy_app.model_ranker import ModelRanker


def test_ranker():
    print("Testing Model Ranker...")
    ranker = ModelRanker()

    # Mock candidates
    candidates = [
        {"provider": "groq", "model": "llama-3.1-8b-instant", "priority": 1},
        {"provider": "google", "model": "gemini-1.5-pro", "priority": 2},  # High score
        {"provider": "g4f", "model": "gpt-4", "priority": 3},
    ]

    print("Original order:", [c["model"] for c in candidates])

    # Rank for 'coding-smart' (should prioritize Gemini 1.5 Pro due to high humaneval)
    ranked = ranker.rank_candidates("coding-smart", candidates)
    print("Ranked order (coding-smart):", [c["model"] for c in ranked])

    if ranked[0]["model"] == "gemini-1.5-pro":
        print("✅ Gemini 1.5 Pro promoted to top (correct).")
        return True
    else:
        print(f"❌ Unexpected top model: {ranked[0]['model']}")
        return False


if __name__ == "__main__":
    if test_ranker():
        print("PASS")
    else:
        print("FAIL")
