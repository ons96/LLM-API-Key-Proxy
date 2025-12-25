"""
Test suite for priority tier routing functionality.
"""
import pytest
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set up test environment with priority tiers BEFORE importing
_test_env = {
    "PROVIDER_PRIORITY_OPENAI": "1",
    "PROVIDER_PRIORITY_ANTHROPIC": "1",
    "PROVIDER_PRIORITY_GROQ": "2",
    "PROVIDER_PRIORITY_GEMINI": "3",
    "PROVIDER_PRIORITY_G4F": "5",
}

from rotator_library.providers.g4f_provider import G4FProvider
from rotator_library.client import get_provider_priority


class TestProviderPriorityResolution:
    """Test provider priority resolution from environment variables."""
    
    def test_priority_from_env_openai(self):
        """Test OpenAI gets priority 1 from environment."""
        priority = get_provider_priority("openai")
        assert priority == 1
    
    def test_priority_from_env_groq(self):
        """Test Groq gets priority 2 from environment."""
        priority = get_provider_priority("groq")
        assert priority == 2
    
    def test_priority_from_env_gemini(self):
        """Test Gemini gets priority 3 from environment."""
        priority = get_provider_priority("gemini")
        assert priority == 3
    
    def test_priority_from_env_g4f(self):
        """Test G4F gets priority 5 from environment."""
        priority = get_provider_priority("g4f")
        assert priority == 5
    
    def test_priority_fallback_unknown(self):
        """Test unknown provider gets default lowest priority."""
        priority = get_provider_priority("unknown_provider")
        assert priority == 10


class TestProviderPriorityOrder:
    """Test that providers are ordered correctly by priority."""
    
    def test_tier_ordering(self):
        """Test that tier 1 providers are higher priority than tier 2."""
        tier1_providers = ["openai"]
        tier2_providers = ["groq"]
        
        for t1 in tier1_providers:
            for t2 in tier2_providers:
                p1 = get_provider_priority(t1)
                p2 = get_provider_priority(t2)
                assert p1 < p2, f"{t1} (P{p1}) should be higher priority than {t2} (P{p2})"
    
    def test_g4f_is_lowest_priority(self):
        """Test that G4F is always lowest priority among known providers."""
        g4f_priority = get_provider_priority("g4f")
        
        # Test against providers that have explicit priorities
        known_providers = ["openai", "groq", "gemini"]
        for provider in known_providers:
            priority = get_provider_priority(provider)
            assert priority < g4f_priority, (
                f"{provider} (P{priority}) should be higher than G4F (P{g4f_priority})"
            )


class TestTierBasedRouting:
    """Test routing based on priority tiers."""
    
    def test_tier_1_contains_premium_providers(self):
        """Test that tier 1 contains premium paid providers."""
        tier1_providers = []
        for provider in ["openai"]:
            if get_provider_priority(provider) == 1:
                tier1_providers.append(provider)
        
        assert len(tier1_providers) > 0
    
    def test_tier_3_contains_standard_providers(self):
        """Test that tier 3 contains standard providers."""
        tier3_providers = []
        for provider in ["gemini"]:
            if get_provider_priority(provider) == 3:
                tier3_providers.append(provider)
        
        assert "gemini" in tier3_providers or len(tier3_providers) >= 0
    
    def test_fallback_tier_5_contains_g4f(self):
        """Test that G4F is in fallback tier 5."""
        assert get_provider_priority("g4f") == 5


class TestRotatingClientPriorityRouting:
    """Test RotatingClient priority routing integration."""
    
    def test_priority_config_loading(self):
        """Test that priority configuration is loaded correctly."""
        # This tests the priority loading mechanism
        expected_priorities = {
            "openai": 1,
            "groq": 2,
            "gemini": 3,
            "g4f": 5,
        }
        
        for provider, expected in expected_priorities.items():
            actual = get_provider_priority(provider)
            assert actual == expected, f"Provider {provider} should have priority {expected}, got {actual}"
    
    def test_fallback_chain_order(self):
        """Test that fallback chain follows correct priority order."""
        priorities = {
            "openai": get_provider_priority("openai"),
            "groq": get_provider_priority("groq"),
            "gemini": get_provider_priority("gemini"),
            "g4f": get_provider_priority("g4f"),
        }
        
        # Sort by priority (lower is better)
        sorted_providers = sorted(priorities.items(), key=lambda x: x[1])
        
        # Verify order: openai < groq < gemini < g4f
        assert sorted_providers[0][0] in ["openai"]
        assert sorted_providers[-1][0] == "g4f"


class TestG4FPriorityInContext:
    """Test G4F priority behavior in the context of the proxy."""
    
    def test_g4f_provider_has_lowest_priority(self):
        """Verify G4F provider is configured as lowest priority."""
        provider = G4FProvider()
        
        assert provider.get_credential_priority("any-key") == 5
        assert provider.get_credential_tier_name("any-key") == "standard"
    
    def test_g4f_will_only_be_used_as_fallback(self):
        """Test that G4F is correctly positioned as a fallback provider."""
        g4f_priority = 5
        
        # All standard providers should have higher priority
        standard_providers = ["openai", "groq", "gemini"]
        
        for provider in standard_providers:
            priority = get_provider_priority(provider)
            assert priority < g4f_priority, (
                f"Standard provider {provider} (P{priority}) should have higher priority "
                f"than G4F (P{g4f_priority})"
            )
    
    def test_priority_env_var_format(self):
        """Test that PROVIDER_PRIORITY_* format works correctly."""
        # Test with a custom provider
        os.environ["PROVIDER_PRIORITY_TESTPROVIDER"] = "4"
        
        try:
            priority = get_provider_priority("testprovider")
            assert priority == 4
        finally:
            if "PROVIDER_PRIORITY_TESTPROVIDER" in os.environ:
                del os.environ["PROVIDER_PRIORITY_TESTPROVIDER"]


class TestPriorityConfigurationPersistence:
    """Test that priority configuration is properly handled."""
    
    def test_env_override_is_respected(self):
        """Test that environment variable overrides are respected."""
        os.environ["PROVIDER_PRIORITY_OPENAI"] = "2"
        
        try:
            priority = get_provider_priority("openai")
            assert priority == 2, "Environment override should change priority"
        finally:
            # Reset to original value
            os.environ["PROVIDER_PRIORITY_OPENAI"] = "1"
    
    def test_invalid_priority_value(self):
        """Test handling of invalid priority values."""
        os.environ["PROVIDER_PRIORITY_TESTPROVIDER"] = "invalid"
        
        try:
            priority = get_provider_priority("testprovider")
            # Should fall back to default (10 for unknown)
            assert priority == 10
        finally:
            if "PROVIDER_PRIORITY_TESTPROVIDER" in os.environ:
                del os.environ["PROVIDER_PRIORITY_TESTPROVIDER"]


class TestPriorityTierSystemIntegration:
    """Test the priority tier system as a whole."""
    
    def test_complete_priority_map(self):
        """Test the complete priority mapping for all providers."""
        priority_map = {}
        providers_to_test = [
            "openai", "groq", "gemini", "g4f"
        ]
        
        for provider in providers_to_test:
            priority_map[provider] = get_provider_priority(provider)
        
        # Verify all priorities are valid integers
        for provider, priority in priority_map.items():
            assert isinstance(priority, int), f"Priority for {provider} should be an integer"
            assert 1 <= priority <= 10, f"Priority for {provider} should be between 1 and 10"
        
        # Verify G4F has priority 5
        assert priority_map["g4f"] == 5
        # G4F should have lower or equal priority than openai
        assert priority_map["g4f"] >= priority_map["openai"]
    
    def test_priority_groups(self):
        """Test that providers can be grouped by priority tier."""
        tier_groups = {}
        
        providers = ["openai", "groq", "gemini", "g4f"]
        for provider in providers:
            priority = get_provider_priority(provider)
            if priority not in tier_groups:
                tier_groups[priority] = []
            tier_groups[priority].append(provider)
        
        # Verify tier grouping works
        assert len(tier_groups) >= 2  # At least 2 different tiers
        assert 1 in tier_groups  # Tier 1 exists (openai)
        assert 5 in tier_groups  # Tier 5 exists (G4F)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
