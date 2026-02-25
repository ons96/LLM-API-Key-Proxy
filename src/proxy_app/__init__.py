"""
LLM API Proxy Application Package

This package contains the main FastAPI gateway for routing LLM API requests
through multiple providers with automatic fallback and credential rotation.

Modules:
    main: Main entry point and FastAPI application
    router_core: Core routing logic and provider selection
    router_wrapper: Wrapper for router integration
    router_integration: Integration layer for router components
    provider_urls: Provider URL construction utilities
    launcher_tui: Terminal UI launcher
    settings_tool: Settings management
    credential_tool: OAuth credential management

Configuration:
    config/router_config.yaml: Provider configuration
    config/virtual_models.yaml: Virtual model definitions
"""

__version__ = "1.0.0"
