import sys
import os
import pytest
import httpx
import respx
import litellm

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from rotator_library.providers.g4f_provider import G4FProvider

@pytest.mark.asyncio
async def test_g4f_waf_detection_non_streaming():
    """Verify G4F provider raises APIConnectionError on HTML response (WAF)."""
    provider = G4FProvider()
    
    async with httpx.AsyncClient() as client:
        with respx.mock(base_url="https://g4f.dev/v1") as mock:
            # Mock a 200 OK response but with HTML body (Cloudflare style)
            html_body = "<html><head><title>Just a moment...</title></head><body>Blocked</body></html>"
            mock.post("/chat/completions").return_value = httpx.Response(200, text=html_body)
            
            with pytest.raises(litellm.APIConnectionError) as excinfo:
                await provider.acompletion(
                    client=client,
                    model="g4f/gpt-4o",
                    messages=[{"role": "user", "content": "Hello"}],
                )
            
            assert "WAF/Cloudflare" in str(excinfo.value)

@pytest.mark.asyncio
async def test_g4f_waf_detection_streaming():
    """Verify G4F provider raises APIConnectionError on HTML streaming response."""
    provider = G4FProvider()
    
    async with httpx.AsyncClient() as client:
        with respx.mock(base_url="https://g4f.dev/v1") as mock:
            # Mock a 200 OK response that is actually HTML
            html_body = "<html><body>Cloudflare security check</body></html>"
            mock.post("/chat/completions").return_value = httpx.Response(200, text=html_body)
            
            # Start the stream
            gen = await provider.acompletion(
                client=client,
                model="g4f/gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True
            )
            
            # Iterating should trigger the error
            with pytest.raises(litellm.APIConnectionError) as excinfo:
                async for _ in gen:
                    pass
            
            assert "WAF/Cloudflare" in str(excinfo.value)

@pytest.mark.asyncio
async def test_g4f_model_mapping_clean():
    """Verify G4F provider strips 'g4f/' prefix correctly."""
    provider = G4FProvider()
    
    # Test _strip_provider_prefix directly
    assert provider._strip_provider_prefix("g4f/gpt-4o") == "gpt-4o"
    assert provider._strip_provider_prefix("g4f/claude-3-opus") == "claude-3-opus"
    # Ensure it doesn't break other strings
    assert provider._strip_provider_prefix("gpt-4o") == "gpt-4o"
