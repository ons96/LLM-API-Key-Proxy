import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import litellm_parity as parity  # noqa: E402


def write(root: Path, name: str, body: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_scan_records_file_and_line(tmp_path):
    write(tmp_path, "app/main.py", "import litellm\nawait litellm.acompletion(x)\n")
    found = parity.scan(tmp_path)
    assert found == {"acompletion": [("app/main.py", 2)]}


def test_scan_ignores_comments_and_unrelated_attributes(tmp_path):
    write(tmp_path, "a.py", "# litellm.acompletion in a comment\nnot_litellm.foo = 1\n")
    assert parity.scan(tmp_path) == {}


def test_counts_sites_across_files(tmp_path):
    write(tmp_path, "a.py", "litellm.acompletion(1)\n")
    write(tmp_path, "b.py", "litellm.acompletion(2)\nlitellm.RateLimitError\n")
    coverage = parity.Coverage(parity.scan(tmp_path))
    assert coverage.sites == 3
    assert len(coverage.occurrences) == 2


def test_declared_replaced_symbols_do_not_block(tmp_path):
    write(tmp_path, "a.py", "litellm.acompletion(1)\nlitellm.RateLimitError\n")
    coverage = parity.Coverage(parity.scan(tmp_path))
    assert coverage.blocking() == []
    assert coverage.undeclared == []


def test_undeclared_symbol_blocks_and_fails(tmp_path):
    write(tmp_path, "a.py", "litellm.brand_new_thing(1)\n")
    assert parity.main(["--source", str(tmp_path)]) == 1
    coverage = parity.Coverage(parity.scan(tmp_path))
    assert coverage.undeclared == ["brand_new_thing"]


def test_gap_symbol_blocks_without_being_undeclared(tmp_path):
    write(tmp_path, "a.py", "litellm.drop_params(model, kwargs)\n")
    coverage = parity.Coverage(parity.scan(tmp_path))
    assert coverage.undeclared == []
    assert coverage.blocking() == ["drop_params"]


def test_clean_tree_exits_zero(tmp_path):
    write(tmp_path, "a.py", "litellm.acompletion(1)\n")
    assert parity.main(["--source", str(tmp_path)]) == 0


def test_render_lists_status_sections_and_blocking(tmp_path, capsys):
    write(tmp_path, "a.py", "litellm.acompletion(1)\nlitellm.drop_params(2)\nlitellm.unknown(3)\n")
    parity.main(["--source", str(tmp_path)])
    output = capsys.readouterr().out
    assert "GAP" in output and "REPLACED" in output
    assert "UNDECLARED" in output and "unknown" in output
    assert "blocking: drop_params, unknown" in output


def test_json_report_is_machine_readable(tmp_path, capsys):
    write(tmp_path, "a.py", "litellm.acompletion(1)\n")
    assert parity.main(["--source", str(tmp_path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["sites"] == 1
    assert report["by_status"]["replaced"] == ["acompletion"]
    assert report["blocking"] == []


def test_missing_source_directory_fails_closed(tmp_path):
    missing = tmp_path / "nope"
    try:
        parity.scan(missing)
    except SystemExit as exc:
        assert "source directory not found" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_repository_gateway_source_is_declared():
    """The real gateway tree must stay fully declared, or this test names the gap."""
    root = Path(__file__).resolve().parents[2] / "src"
    if not root.is_dir():
        return
    coverage = parity.Coverage(parity.scan(root))
    assert coverage.undeclared == []
