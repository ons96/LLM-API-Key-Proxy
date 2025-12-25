#!/usr/bin/env python3
"""
G4F Fallback Demonstration Script

This script demonstrates how G4F fallback works when primary providers fail.
It shows the priority tier system and failover behavior.
"""
import os
import asyncio
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rotator_library.providers.g4f_provider import G4FProvider


async def demonstrate_g4f_priority_tier():
    """Demonstrate G4F priority tier positioning."""
    print("=" * 70)
    print("G4F Priority Tier System Demonstration")
    print("=" * 70)
    
    # Show default priorities
    print("\n1. Default Provider Priorities:")
    print("-" * 40)
    
    providers = [
        ("openai", "Premium (Paid)"),
        ("anthropic", "Premium (Paid)"),
        ("groq", "Fast/_affordable"),
        ("openrouter", "Multi-provider"),
        ("gemini", "Standard"),
        ("mistral", "Standard"),
        ("g4f", "Fallback (Free)"),
    ]
    
    for provider, tier_desc in providers:
        priority = G4FProvider.get_provider_priority(provider)
        print(f"   {provider:15} -> Tier {priority:2} ({tier_desc})")
    
    # Show G4F provider properties
    print("\n2. G4F Provider Properties:")
    print("-" * 40)
    
    provider = G4FProvider()
    print(f"   Provider Name:        {provider.provider_name}")
    print(f"   Tier Priority:        {provider.get_credential_priority('any-key')}")
    print(f"   Tier Name:            {provider.get_credential_tier_name('any-key')}")
    print(f"   Has Custom Logic:     {provider.has_custom_logic()}")
    print(f"   Rotation Mode:        {provider.get_rotation_mode('g4f')}")
    print(f"   Supports Embeddings:  False (Not Implemented)")
    
    # Show endpoint configuration
    print("\n3. Configured G4F Endpoints:")
    print("-" * 40)
    
    if provider._endpoints:
        for endpoint_name, endpoint_url in provider._endpoints.items():
            print(f"   {endpoint_name:10} -> {endpoint_url}")
    else:
        print("   No endpoints configured")
        print("   Set G4F_MAIN_API_BASE or other G4F_*_API_BASE variables")
    
    # Demonstrate endpoint routing
    print("\n4. Endpoint Routing Based on Model Name:")
    print("-" * 40)
    
    test_models = [
        ("groq/llama-3.1-70b", "Routes to Groq endpoint"),
        ("grok-2", "Routes to Grok endpoint"),
        ("gemini-1.5-pro", "Routes to Gemini endpoint"),
        ("nemotron-70b", "Routes to NVIDIA endpoint"),
        ("gpt-4o", "Routes to Main endpoint"),
        ("unknown-model", "Routes to Main endpoint (fallback)"),
    ]
    
    for model, expected_behavior in test_models:
        endpoint = provider._get_endpoint_for_model(model)
        if endpoint:
            endpoint_name = endpoint.split("/")[-1] if "/" in endpoint else endpoint
            print(f"   {model:25} -> {endpoint_name[:30]}")
        else:
            print(f"   {model:25} -> No endpoint configured")
        print(f"                              ({expected_behavior})")
    
    # Show fallback order
    print("\n5. Failover Order (Priority-Based):")
    print("-" * 40)
    print("   Tier 1: OpenAI, Anthropic (Highest Priority)")
    print("   Tier 2: Groq, OpenRouter")
    print("   Tier 3: Gemini, Mistral")
    print("   Tier 5: G4F (Lowest Priority - Last Resort)")
    
    print("\n6. When to Use G4F:")
    print("-" * 40)
    print("   - Primary providers exhausted or rate-limited")
    print("   - Cost savings needed (free tier)")
    print("   - Backup for development/testing")
    print("   - Provider doesn't support the required model")
    
    print("\n" + "=" * 70)
    print("Configuration Required in .env:")
    print("=" * 70)
    print("""
# G4F Fallback Configuration
G4F_API_KEY=""                              # Optional API key
G4F_MAIN_API_BASE="https://your-g4f-proxy"  # Main endpoint
G4F_GROQ_API_BASE=""                        # Groq-compatible (optional)
G4F_GROK_API_BASE=""                        # Grok-compatible (optional)
G4F_GEMINI_API_BASE=""                      # Gemini-compatible (optional)
G4F_NVIDIA_API_BASE=""                      # NVIDIA-compatible (optional)

# Provider Priority Tiers (optional - defaults shown)
PROVIDER_PRIORITY_OPENAI=1
PROVIDER_PRIORITY_GROQ=2
PROVIDER_PRIORITY_GEMINI=3
PROVIDER_PRIORITY_G4F=5
""")


async def demonstrate_error_scenarios():
    """Demonstrate error handling scenarios."""
    print("\n" + "=" * 70)
    print("G4F Error Handling Scenarios")
    print("=" * 70)
    
    provider = G4FProvider()
    
    print("\n1. No Endpoints Configured:")
    print("-" * 40)
    print("   Error: No G4F endpoint configured")
    print("   Action: Set G4F_MAIN_API_BASE or other G4F_*_API_BASE variables")
    print("   Fallback: Request fails immediately")
    
    print("\n2. Embedding Request:")
    print("-" * 40)
    print("   Error: NotImplementedError")
    print("   Message: G4F providers do not support embeddings")
    print("   Action: Use a different provider for embeddings")
    
    print("\n3. Rate Limit Error:")
    print("-" * 40)
    print("   Error: httpx.HTTPStatusError (429)")
    print("   Action: Rotate to next G4F endpoint or fail to next tier")
    print("   Fallback: Try different G4F endpoint, then other providers")
    
    print("\n4. Provider Tiers Priority:")
    print("-" * 40)
    print("   1. Try Tier 1 providers (OpenAI, Anthropic)")
    print("   2. Try Tier 2 providers (Groq, OpenRouter)")
    print("   3. Try Tier 3 providers (Gemini, Mistral)")
    print("   4. Try Tier 5 providers (G4F - Last Resort)")


async def main():
    """Run the demonstration."""
    print("\n" + "=" * 70)
    print(" G4F Fallback Provider - Feature Demonstration")
    print("=" * 70)
    
    await demonstrate_g4f_priority_tier()
    await demonstrate_error_scenarios()
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
