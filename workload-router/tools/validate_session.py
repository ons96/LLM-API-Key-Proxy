"""Validate the portable evidence for the deployed router review."""

from pathlib import Path


REQUIRED_TERMS = (
    "mapping-based dead-provider policy",
    "deployment-aware fallback ranking",
    "unused stream-timeout request field",
    "systemd service and socket",
)


def validate_project(root: Path) -> None:
    """Raise if the project record is incomplete."""
    readme = (root / "README.md").read_text(encoding="utf-8")
    missing = [term for term in REQUIRED_TERMS if term not in readme]
    if missing:
        raise ValueError(f"missing review terms: {', '.join(missing)}")
    if "authenticated model-list probe" not in readme:
        raise ValueError("missing authenticated probe status")


if __name__ == "__main__":
    validate_project(Path(__file__).resolve().parents[1])
    print("router session validation: OK")
