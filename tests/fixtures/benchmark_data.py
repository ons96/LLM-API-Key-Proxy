"""Sample benchmark data for testing."""

# Sample model rankings data (mimicking model_rankings.yaml structure)
SAMPLE_MODEL_RANKINGS = {
    "models": [
        {
            "id": "gpt-4o",
            "name": "GPT-4o",
            "provider": "openai",
            "scores": {
                "humaneval": 90.2,
                "swe_bench": 38.1,
                "livebench": 48.3,
                "speed_tps": 85.0,
                "ttft_ms": 250.0
            },
            "best_for": ["coding-smart", "chat-smart"]
        },
        {
            "id": "claude-3.5-sonnet",
            "name": "Claude 3.5 Sonnet",
            "provider": "anthropic",
            "scores": {
                "humaneval": 92.0,
                "swe_bench": 49.0,
                "livebench": 55.1,
                "speed_tps": 75.0,
                "ttft_ms": 300.0
            },
            "best_for": ["coding-smart"]
        },
        {
            "id": "o1-mini",
            "name": "O1 Mini",
            "provider": "openai",
            "scores": {
                "humaneval": 87.5,
                "swe_bench": 34.7,
                "livebench": 42.5,
                "speed_tps": 65.0,
                "ttft_ms": 400.0
            },
            "best_for": ["coding-smart"]
        },
        {
            "id": "groq/llama-3.3-70b-versatile",
            "name": "Llama 3.3 70B (Groq)",
            "provider": "groq",
            "scores": {
                "humaneval": 72.3,
                "swe_bench": 24.1,
                "livebench": 38.2,
                "speed_tps": 1000.0,
                "ttft_ms": 50.0
            },
            "best_for": ["coding-fast"]
        },
        {
            "id": "groq/llama-3.1-8b-instant",
            "name": "Llama 3.1 8B Instant (Groq)",
            "provider": "groq",
            "scores": {
                "humaneval": 62.8,
                "swe_bench": 15.3,
                "livebench": 28.7,
                "speed_tps": 1200.0,
                "ttft_ms": 30.0
            },
            "best_for": ["coding-fast", "chat-fast"]
        },
        {
            "id": "cerebras/llama-3.1-70b",
            "name": "Llama 3.1 70B (Cerebras)",
            "provider": "cerebras",
            "scores": {
                "humaneval": 73.1,
                "swe_bench": 25.2,
                "livebench": 39.5,
                "speed_tps": 3000.0,
                "ttft_ms": 20.0
            },
            "best_for": ["coding-fast"]
        },
        {
            "id": "gemini/gemini-1.5-flash",
            "name": "Gemini 1.5 Flash",
            "provider": "google",
            "scores": {
                "humaneval": 68.9,
                "swe_bench": 18.2,
                "livebench": 32.4,
                "speed_tps": 250.0,
                "ttft_ms": 150.0
            },
            "best_for": ["chat-fast"]
        }
    ]
}

# Performance metrics by provider (mimicking provider_speeds.json)
SAMPLE_PROVIDER_PERFORMANCE = {
    "groq": {
        "llama-3.3-70b-versatile": {
            "avg_tps": 1000.0,
            "avg_ttft_ms": 50.0,
            "success_rate": 0.99,
            "avg_latency_ms": 150.0
        },
        "llama-3.1-8b-instant": {
            "avg_tps": 1200.0,
            "avg_ttft_ms": 30.0,
            "success_rate": 0.995,
            "avg_latency_ms": 80.0
        }
    },
    "cerebras": {
        "llama-3.1-70b": {
            "avg_tps": 3000.0,
            "avg_ttft_ms": 20.0,
            "success_rate": 0.98,
            "avg_latency_ms": 50.0
        },
        "llama-3.1-8b": {
            "avg_tps": 3500.0,
            "avg_ttft_ms": 15.0,
            "success_rate": 0.98,
            "avg_latency_ms": 40.0
        }
    },
    "google": {
        "gemini-1.5-flash": {
            "avg_tps": 250.0,
            "avg_ttft_ms": 150.0,
            "success_rate": 0.97,
            "avg_latency_ms": 300.0
        }
    },
    "g4f": {
        "gpt-4o": {
            "avg_tps": 85.0,
            "avg_ttft_ms": 250.0,
            "success_rate": 0.85,
            "avg_latency_ms": 500.0
        }
    }
}

# Virtual model configurations for testing
SAMPLE_VIRTUAL_MODELS = {
    "coding-smart": {
        "description": "Best coding models",
        "fallback_chain": [
            {"provider": "g4f", "model": "gpt-4o", "priority": 1, "free_tier_only": True},
            {"provider": "g4f", "model": "claude-3.5-sonnet", "priority": 2, "free_tier_only": True},
            {"provider": "g4f", "model": "o1-mini", "priority": 3, "free_tier_only": True}
        ],
        "auto_order": True
    },
    "coding-fast": {
        "description": "Fast coding models",
        "fallback_chain": [
            {"provider": "cerebras", "model": "llama-3.1-70b", "priority": 1, "free_tier_only": True},
            {"provider": "groq", "model": "llama-3.3-70b-versatile", "priority": 2, "free_tier_only": True},
            {"provider": "groq", "model": "llama-3.1-8b-instant", "priority": 3, "free_tier_only": True}
        ],
        "auto_order": True
    },
    "chat-smart": {
        "description": "Best chat models",
        "fallback_chain": [
            {"provider": "g4f", "model": "gpt-4o", "priority": 1, "free_tier_only": True},
            {"provider": "g4f", "model": "claude-3.5-sonnet", "priority": 2, "free_tier_only": True}
        ],
        "auto_order": True
    },
    "chat-fast": {
        "description": "Fast chat models",
        "fallback_chain": [
            {"provider": "cerebras", "model": "llama-3.1-8b", "priority": 1, "free_tier_only": True},
            {"provider": "groq", "model": "llama-3.1-8b-instant", "priority": 2, "free_tier_only": True}
        ],
        "auto_order": True
    }
}
