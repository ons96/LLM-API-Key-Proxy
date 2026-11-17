"""LiteLLM removal checklist.

The gateway cannot drop the LiteLLM library until every symbol it still calls
has a sidecar equivalent, or a recorded reason to disappear along with the
library. This tool scans the gateway source and reports that coverage.

It deliberately fails when a symbol appears that has no declared disposition, so
the checklist cannot silently rot as new LiteLLM calls are added.

Read-only: the scanned code is never imported, no network is used, and no
provider is contacted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

SYMBOL_RE = re.compile(r"\blitellm\.([A-Za-z_][A-Za-z0-9_]*)")

# status -> what the sidecar does about it.
#   replaced: the sidecar performs this function without LiteLLM
#   partial:  the sidecar covers the common subset only
#   gap:      nothing in the sidecar yet; LiteLLM cannot go until this is closed
#   retired:  internal LiteLLM plumbing that simply disappears with the library
DISPOSITIONS: dict[str, tuple[str, str]] = {
    "acompletion": ("replaced", "newapi_runtime dispatches through New-API over urllib"),
    "aembedding": ("partial", "only operator-allowlisted SIDECAR_EMBEDDING_ALIASES"),
    "ModelResponse": ("replaced", "sidecar builds OpenAI-shaped dict responses"),
    "EmbeddingResponse": ("replaced", "sidecar builds OpenAI-shaped dict responses"),
    "Choices": ("replaced", "sidecar builds OpenAI-shaped dict responses"),
    "Message": ("replaced", "sidecar builds OpenAI-shaped dict responses"),
    "Usage": ("replaced", "sidecar normalises usage from the upstream body"),
    "RateLimitError": ("replaced", "route_policy failure class rate_limit"),
    "ServiceUnavailableError": ("replaced", "route_policy failure class upstream_error"),
    "InternalServerError": ("replaced", "route_policy failure class upstream_error"),
    "APIConnectionError": ("replaced", "route_policy failure class transport"),
    "AuthenticationError": ("replaced", "route_policy failure class auth_error"),
    "Timeout": ("replaced", "route_policy failure class transport"),
    "OpenAIError": ("replaced", "sidecar raises its own bounded failure classes"),
    "InvalidRequestError": ("replaced", "sidecar validates the request before dispatch"),
    "ContextWindowExceededError": ("partial", "byte upper bound only, no exact tokenizer"),
    "completion_cost": ("partial", "ledger route prices; no upstream pricing registry"),
    "get_model_info": ("partial", "ledger route metadata; no upstream model-info registry"),
    "drop_params": ("gap", "no sidecar request-parameter filtering equivalent"),
    "callbacks": ("partial", "quota_ledger and provider_status instead of callbacks"),
    "set_verbose": ("retired", "library-level logging switch"),
    "exceptions": ("retired", "library namespace"),
    "types": ("retired", "library namespace"),
    "integrations": ("retired", "library namespace"),
    "litellm_core_utils": ("retired", "library namespace"),
}

STATUS_ORDER = ("gap", "partial", "replaced", "retired")


def scan(source: Path) -> dict[str, list[tuple[str, int]]]:
    """Map every ``litellm.<symbol>`` to the file and line that uses it."""
    if not source.is_dir():
        raise SystemExit("source directory not found: " + str(source))
    found: dict[str, list[tuple[str, int]]] = {}
    for path in sorted(source.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for symbol in SYMBOL_RE.findall(line):
                found.setdefault(symbol, []).append((str(path.relative_to(source)), lineno))
    return found


@dataclass(frozen=True)
class Coverage:
    occurrences: dict[str, list[tuple[str, int]]]

    @property
    def sites(self) -> int:
        return sum(len(rows) for rows in self.occurrences.values())

    @property
    def undeclared(self) -> list[str]:
        return sorted(set(self.occurrences) - set(DISPOSITIONS))

    def by_status(self) -> dict[str, list[str]]:
        grouped = {status: [] for status in STATUS_ORDER}
        for symbol in sorted(self.occurrences):
            status = DISPOSITIONS.get(symbol, ("gap", ""))[0]
            grouped[status].append(symbol)
        return grouped

    def blocking(self) -> list[str]:
        """Symbols that must be closed before LiteLLM can be removed."""
        return sorted(set(self.undeclared) | set(self.by_status()["gap"]))


def render(coverage: Coverage) -> str:
    lines = [
        "LiteLLM removal checklist",
        "sites: %d  symbols: %d" % (coverage.sites, len(coverage.occurrences)),
        "",
    ]
    for status in STATUS_ORDER:
        symbols = coverage.by_status()[status]
        if not symbols:
            continue
        lines.append(status.upper())
        for symbol in symbols:
            note = DISPOSITIONS.get(symbol, ("gap", "not declared"))[1]
            count = len(coverage.occurrences[symbol])
            lines.append("  %-28s %2d  %s" % (symbol, count, note))
        lines.append("")
    if coverage.undeclared:
        lines.append("UNDECLARED (declare a disposition in DISPOSITIONS)")
        lines.extend("  " + symbol for symbol in coverage.undeclared)
        lines.append("")
    blocking = coverage.blocking()
    lines.append("blocking: " + (", ".join(blocking) if blocking else "none"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="", help="gateway source root (default: ../src)")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args(argv)

    source = Path(args.source) if args.source else Path(__file__).resolve().parents[2] / "src"
    coverage = Coverage(scan(source))
    if args.json:
        print(json.dumps({
            "sites": coverage.sites,
            "symbols": len(coverage.occurrences),
            "by_status": coverage.by_status(),
            "undeclared": coverage.undeclared,
            "blocking": coverage.blocking(),
        }, indent=2, sort_keys=True))
    else:
        print(render(coverage))
    return 1 if coverage.blocking() else 0


if __name__ == "__main__":
    sys.exit(main())
