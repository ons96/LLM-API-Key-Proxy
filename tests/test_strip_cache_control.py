"""Tests for _strip_cache_control (#346).

Extracts the ACTUAL function source from router_core.py via AST and tests it
in isolation, avoiding the module's heavy import chain (litellm/fastapi/etc
not installed locally — they live on the VPS).
"""

import ast
import pathlib
import textwrap


def _extract_function():
    """Pull _strip_cache_control source from router_core.py, exec in clean ns."""
    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "proxy_app"
        / "router_core.py"
    ).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_strip_cache_control":
            fn_src = ast.get_source_segment(src, node)
            ns = {"Any": object}
            exec(textwrap.dedent(fn_src), ns)
            return ns["_strip_cache_control"]
    raise RuntimeError("_strip_cache_control not found in router_core.py")


strip = _extract_function()


class TestStripCacheControl:
    def test_strips_from_message_content_blocks(self):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": "there"},
                    ],
                }
            ]
        }
        out = strip(payload)
        assert "cache_control" not in out["messages"][0]["content"][0]
        assert out["messages"][0]["content"][0]["text"] == "hi"

    def test_strips_from_tool_calls(self):
        payload = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "foo", "arguments": "{}"},
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ]
        }
        out = strip(payload)
        assert "cache_control" not in out["messages"][0]["tool_calls"][0]
        assert out["messages"][0]["tool_calls"][0]["function"]["name"] == "foo"

    def test_preserves_other_fields(self):
        payload = {"model": "x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
        out = strip(payload)
        assert out == payload

    def test_empty_and_scalars_passthrough(self):
        assert strip({}) == {}
        assert strip([]) == []
        assert strip("str") == "str"
        assert strip(42) == 42

    def test_nested_lists_and_dicts(self):
        payload = {
            "messages": [
                {
                    "content": [
                        {"cache_control": {"type": "ephemeral"}, "nested": [{"cache_control": "x", "v": 1}]}
                    ],
                }
            ]
        }
        out = strip(payload)
        block = out["messages"][0]["content"][0]
        assert "cache_control" not in block
        assert "cache_control" not in block["nested"][0]
        assert block["nested"][0]["v"] == 1

    def test_in_place_mutation_documented(self):
        """Function mutates in place (perf) AND returns the obj — callers can use either."""
        payload = {"messages": [{"content": [{"cache_control": {"type": "ephemeral"}, "text": "hi"}]}]}
        ret = strip(payload)
        assert ret is payload
        assert "cache_control" not in payload["messages"][0]["content"][0]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-q"])
