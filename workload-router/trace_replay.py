"""Offline routing replay for shadow evaluation against a fixed baseline."""

import json
from dataclasses import dataclass
from pathlib import Path

from router_core import Deployment, RequestFeatures, select


@dataclass(frozen=True)
class Trace:
    prompt: str
    context_tokens: int = 0
    tools: bool = False
    phase: str = ""


@dataclass(frozen=True)
class ReplayResult:
    total: int
    routed: int
    baseline_deployment: str
    selected: tuple[str, ...]
    baseline_matches: int
    selected_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "routed": self.routed,
            "baseline_deployment": self.baseline_deployment,
            "baseline_matches": self.baseline_matches,
            "selected": list(self.selected),
            "selected_counts": self.selected_counts,
            "baseline_match_rate": (
                self.baseline_matches / self.total if self.total else 0.0
            ),
        }


def load_traces(path: str | Path) -> list[Trace]:
    """Load newline-delimited trace metadata; reject malformed records."""
    traces = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise ValueError("each trace requires a string prompt")
        traces.append(Trace(
            prompt=item["prompt"],
            context_tokens=int(item.get("context_tokens", 0)),
            tools=bool(item.get("tools", False)),
            phase=str(item.get("phase", "")),
        ))
    return traces


def replay(traces: list[Trace], deployments: list[Deployment], baseline_deployment: str) -> ReplayResult:
    """Replay traces locally and report route coverage, without model calls."""
    selected = []
    for trace in traces:
        selected.append(select(RequestFeatures(
            prompt=trace.prompt,
            context_tokens=trace.context_tokens,
            tools=trace.tools,
            phase=trace.phase,
        ), deployments).deployment_id)
    return ReplayResult(
        len(traces),
        len(selected),
        baseline_deployment,
        tuple(selected),
        sum(item == baseline_deployment for item in selected),
        {deployment_id: selected.count(deployment_id) for deployment_id in sorted(set(selected))},
    )


def write_report(result: ReplayResult, path: str | Path) -> None:
    """Write aggregate replay metrics without request contents."""
    Path(path).write_text(json.dumps(result.as_dict(), indent=2) + "\n", encoding="utf-8")
