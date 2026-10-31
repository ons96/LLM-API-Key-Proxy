#!/usr/bin/env python3
"""Replayable evaluation for the prompt classifier (#949).

Usage:
    PROMPT_CLASSIFIER_ENABLED=1 python3 scripts/eval_prompt_classifier.py \
        [--corpus tests/fixtures/prompt_classifier_eval.jsonl]

Reports accuracy, per-category precision/recall, latency, fallback rate.
Stdlib only.
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from proxy_app import prompt_classifier as pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="tests/fixtures/prompt_classifier_eval.jsonl")
    args = ap.parse_args()
    corpus_path = (ROOT / args.corpus) if not os.path.isabs(args.corpus) else Path(args.corpus)

    rows = []
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    pc.reset_metrics()
    correct = 0
    total = len(rows)
    per_cat_tp = Counter()
    per_cat_fn = Counter()
    per_cat_fp = Counter()

    for row in rows:
        text = row["text"]
        label = row["label"]
        req = {"messages": [{"role": "user", "content": text}]}
        r = pc.classify(req)
        if r.category.value == label:
            correct += 1
            per_cat_tp[label] += 1
        else:
            per_cat_fn[label] += 1
            per_cat_fp[r.category.value] += 1

    m = pc.get_metrics()
    print(f"Corpus: {corpus_path}  ({total} cases)")
    print(f"Accuracy: {correct}/{total} = {correct/total:.3f}")
    print(f"Fallback rate: {m['fallback_rate']:.3f} ({m['fallback']}/{m['total']})")
    print(f"Latency avg: {m['latency_ms_avg']:.3f} ms (n={m['latency_ms_count']})")
    print()
    print("Per-category:")
    cats = sorted(set(r["label"] for r in rows))
    for c in cats:
        tp = per_cat_tp[c]
        fn = per_cat_fn[c]
        fp = per_cat_fp[c]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  {c:10s} tp={tp} fn={fn} fp={fp} precision={prec:.2f} recall={rec:.2f}")
    print()
    print("Metrics:", json.dumps(m, indent=2))
    return 0 if correct == total else 1


if __name__ == "__main__":
    sys.exit(main())
PYEOF
filePath="/home/osees/CodingProjects/LLM-API-Key-Proxy/scripts/eval_prompt_classifier.py"