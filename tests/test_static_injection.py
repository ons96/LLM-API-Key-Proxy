"""#405: static virtual-model injection (policy-fixed chains survive rebuild)."""
from pathlib import Path
import os
import sys
import tempfile

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from reorder_chains import load_static_virtual_models


def test_load_static_virtual_models():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(
            "virtual_models:\n"
            "  safe-coding:\n"
            "    description: x\n"
            "    fallback_chain: []\n"
        )
        path = f.name
    try:
        vms = load_static_virtual_models(Path(path))
        assert "safe-coding" in vms
    finally:
        os.unlink(path)


def test_missing_file_returns_empty():
    assert load_static_virtual_models(Path("/no/such/file.yaml")) == {}
