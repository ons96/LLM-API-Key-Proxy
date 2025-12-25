# G4F Provider Integration Documentation

## Phase 2.4 Complete: G4F Provider Factory Integration

### Overview
Phase 2.4 has been successfully implemented, integrating the G4F provider into the provider factory system. This enables the G4F provider to be used as a fallback provider through the proxy routing system.

### What Was Implemented

#### 1. Provider Factory Updates (`src/rotator_library/provider_factory.py`)

**Key Changes:**
- Added G4F provider to the provider mapping system
- Created separate maps for OAuth providers vs direct providers
- Added G4F-specific configuration loading from environment variables
- Implemented provider type detection (OAuth vs Direct)
- Added comprehensive provider validation

**New Functions Added:**
```python
# Provider type detection
is_oauth_provider(provider_name: str) -> bool
is_direct_provider(provider_name: str) -> bool

# G4F configuration management
get_provider_config(provider_name: str) -> Dict[str, Any]
validate_provider_config(provider_name: str) -> bool

# Enhanced provider retrieval
get_provider_class(provider_name: str) -> Type
get_oauth_providers() -> List[str]
get_direct_providers() -> List[str]
```

**Environment Variables Supported:**
- `G4F_API_KEY` - Optional API key for G4F endpoints
- `G4F_MAIN_API_BASE` - Main G4F API base URL
- `G4F_GROQ_API_BASE` - G4F Groq-compatible endpoint
- `G4F_GROK_API_BASE` - G4F Grok-compatible endpoint
- `G4F_GEMINI_API_BASE` - G4F Gemini-compatible endpoint
- `G4F_NVIDIA_API_BASE` - G4F Nvidia-compatible endpoint

#### 2. Automatic Provider Registration

The G4F provider is automatically registered through the existing plugin system in `src/rotator_library/providers/__init__.py`:
- ✅ Already included in the list of known providers to skip dynamic registration
- ✅ Automatically discovered and loaded when the providers module is imported
- ✅ Available in `PROVIDER_PLUGINS` dictionary

#### 3. Provider Classification

**OAuth Providers** (require credential management):
- gemini_cli, qwen_code, iflow, antigravity

**Direct Providers** (use API keys/env vars):
- g4f (newly added)

### Integration Points

#### 1. Client Integration
The G4F provider integrates seamlessly with the existing `RotatingClient`:
- Automatically discovered via `PROVIDER_PLUGINS`
- Uses environment-based configuration
- Supports the provider tier/priority system (default tier 5 - fallback)
- Handles both streaming and non-streaming requests

#### 2. Credential Management
- G4F uses API keys from environment variables (no OAuth credentials needed)
- Can be configured with multiple API base URLs for different endpoints
- Follows the same error handling and retry patterns as other providers

#### 3. Model Support
The G4F provider includes:
- Dynamic model discovery from API endpoints
- Static fallback model list with common G4F models
- Support for multiple model formats (OpenAI-compatible, G4F-specific)

### Usage Examples

#### 1. Basic Configuration
```bash
# Add to your .env file
G4F_API_KEY=your_g4f_api_key
G4F_MAIN_API_BASE=https://your-g4f-endpoint.com
```

#### 2. Programmatic Usage
```python
from rotator_library.provider_factory import get_provider_class, get_provider_config

# Get G4F provider class
g4f_class = get_provider_class("g4f")

# Load G4F configuration
config = get_provider_config("g4f")
print(f"G4F configured with: {list(config.keys())}")

# Check if provider is valid
is_valid = validate_provider_config("g4f")
```

#### 3. Client Usage
```python
from rotator_library.client import RotatingClient

# G4F will be automatically available if configured
client = RotatingClient(api_keys={"g4f": ["your-api-key"]})

# Use G4F models
response = await client.acompletion(
    model="g4f/gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Testing Instructions

#### Manual Test 1: Provider Factory Integration
```python
# Run this in a Python environment with the project installed
from src.rotator_library.provider_factory import (
    get_available_providers, 
    is_direct_provider, 
    get_provider_class
)

# Check if G4F is available
providers = get_available_providers()
assert "g4f" in providers

# Verify G4F is a direct provider
assert is_direct_provider("g4f")

# Get G4F provider class
g4f_class = get_provider_class("g4f")
print(f"G4F Provider Class: {g4f_class}")
```

#### Manual Test 2: Provider Instantiation
```python
import os
from src.rotator_library.providers.g4f_provider import G4FProvider

# Set up environment
os.environ["G4F_API_KEY"] = "test-key"
os.environ["G4F_MAIN_API_BASE"] = "https://test.example.com"

# Create provider instance
provider = G4FProvider()
print(f"Provider: {provider.provider_name}")
print(f"Has Custom Logic: {provider.has_custom_logic()}")
print(f"Default Tier: {provider.default_tier_priority}")
```

#### Manual Test 3: Client Integration
```python
from src.rotator_library.client import RotatingClient

# Test with configured G4F credentials
client = RotatingClient(api_keys={"g4f": ["test-key"]})

# Verify G4F is in the client's provider list
assert "g4f" in client.all_credentials

# Test model listing (if G4F endpoint is available)
try:
    models = await client.get_available_models("g4f")
    print(f"G4F Models: {len(models)} found")
except Exception as e:
    print(f"G4F model listing failed (expected if no endpoint): {e}")
```

#### Manual Test 4: Provider Plugin System
```python
from src.rotator_library.providers import PROVIDER_PLUGINS

# Verify G4F is registered
assert "g4f" in PROVIDER_PLUGINS

# Get the G4F plugin
g4f_plugin = PROVIDER_PLUGINS["g4f"]
print(f"G4F Plugin Class: {g4f_plugin}")

# Verify it's the correct class
from src.rotator_library.providers.g4f_provider import G4FProvider
assert g4f_plugin is G4FProvider
```

### Expected Behavior

#### 1. Provider Discovery
- G4F should appear in `get_available_providers()` list
- G4F should be classified as a direct provider (not OAuth)
- G4F should be in the `PROVIDER_PLUGINS` dictionary

#### 2. Configuration Loading
- Environment variables should be properly loaded
- Configuration should include `default_tier_priority: 5`
- Provider should validate successfully

#### 3. Integration with Client
- G4F credentials should be accepted by the RotatingClient
- G4F should be available for model requests
- Error handling should work correctly

### Troubleshooting

#### Issue: G4F not appearing in provider list
**Solution:** Check that `src/rotator_library/providers/g4f_provider.py` exists and contains a class inheriting from `ProviderInterface`

#### Issue: G4F configuration not loading
**Solution:** Verify environment variables are set and `get_provider_config("g4f")` is called after environment setup

#### Issue: G4F models not available
**Solution:** Check G4F endpoint configuration and network connectivity. The provider has fallback static models.

### Files Modified/Created

1. **Modified:** `src/rotator_library/provider_factory.py`
   - Added G4F provider mapping
   - Added configuration management functions
   - Added provider type detection

2. **Created:** `src/rotator_library/test_g4f_integration.py`
   - Comprehensive test suite for G4F integration
   - Tests all factory integration points
   - Validates provider instantiation

### Next Steps

The G4F provider integration is now complete and ready for use. The provider can be used as:
1. A fallback provider when other providers are rate-limited
2. A cost-effective option for non-critical requests
3. A backup when primary providers fail

To use G4F in production:
1. Configure G4F environment variables
2. Add G4F API keys to your credential configuration
3. Use G4F models in your requests (e.g., "g4f/gpt-3.5-turbo")
4. Monitor G4F usage through the existing usage tracking system

### Verification Complete ✅

Phase 2.4 (G4F Provider Factory Integration) has been successfully implemented with:
- ✅ Provider factory updated to handle G4F
- ✅ Environment-based configuration support
- ✅ Provider cache integration
- ✅ Known providers list updated
- ✅ Proper integration with existing provider management system
- ✅ Support for provider tier/priority system
- ✅ Comprehensive testing framework provided

The G4F provider is now fully integrated and accessible through the proxy routing system.