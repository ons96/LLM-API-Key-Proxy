"""Smoke tests to verify basic functionality and imports."""

import pytest
import sys
from pathlib import Path

# Ensure src is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_proxy_app_imports():
    """Test that proxy_app modules can be imported."""
    from proxy_app.main import app, EmbeddingRequest, ModelCard
    from proxy_app.router_wrapper import initialize_router, get_router
    from proxy_app.health_checker import HealthChecker
    from proxy_app.batch_manager import EmbeddingBatcher
    assert app is not None


def test_rotator_library_imports():
    """Test that rotator_library modules can be imported."""
    from rotator_library import RotatingClient
    from rotator_library.credential_manager import CredentialManager
    from rotator_library.model_info_service import init_model_info_service
    from rotator_library import PROVIDER_PLUGINS
    assert isinstance(PROVIDER_PLUGINS, dict)


def test_fastapi_app_creation():
    """Test that FastAPI app can be instantiated."""
    from fastapi.testclient import TestClient
    from proxy_app.main import app
    
    client = TestClient(app)
    # Test root endpoint or docs
    response = client.get("/docs")
    assert response.status_code == 200


def test_pydantic_models():
    """Test that Pydantic models validate correctly."""
    from proxy_app.main import EmbeddingRequest, ModelCard, EnrichedModelCard
    
    # Test embedding request
    req = EmbeddingRequest(model="text-embedding-3-small", input="test text")
    assert req.model == "text-embedding-3-small"
    assert req.input == "test text"
    
    # Test model card
    card = ModelCard(id="gpt-4")
    assert card.id == "gpt-4"
    assert card.object == "model"


def test_environment_loading():
    """Test that environment configuration can be loaded."""
    import os
    from dotenv import load_dotenv
    
    # Ensure no error is raised during load
    from proxy_app.main import _root_dir, _env_files_found
    assert _root_dir is not None
    assert isinstance(_env_files_found, list)
