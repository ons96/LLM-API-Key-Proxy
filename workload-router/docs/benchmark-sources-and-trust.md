# Benchmark sources, trust, and the ingest path

Status: source decision only. No ingest code is written yet, and nothing here
calls a provider or an LLM.

Purpose: give S8 (cold-start priors) measured quality, latency and per-task cost
for each model without inventing numbers, without a paid API, and without a
scheduled job on VPS 40.

## Decision

The router does not scrape and does not call the Artificial Analysis API. It
reads a **pinned git artifact** that an existing GitHub Action already produces
once a day. That is the whole ingest path:

```
coding-agents-leaderboard (GitHub Action, daily 06:00 UTC, free runner)
    check_aa_change.py --mode check    # hashes only the embedded dataset
    main.py + intel_index_full_extract.py
    commit data/*.csv + rebuild GitHub Pages
        |
        v
raw.githubusercontent.com/ons96/coding-agents-leaderboard/main/data/*.csv
        |
        v
router: one HTTP GET per day, checksum-pinned, cached, last-good retained
```

Consequences that matter:

- No scraping code in this repository, so no new dependency and no parser to
  maintain here.
- No network access in tests: the consumer takes bytes from a fixture path.
- No scheduled job or model call on the 1 GB VPS.
- The data is versioned in git, so an upstream bad run cannot silently corrupt
  the ledger; a pin can be rolled back by commit.
- Fully automatic. There is no manual JSON import step anywhere.

The Artificial Analysis web API (`api/v2/data/llms/models`) is **not** used. Its
free tier exposes fewer fields than the public payload, it needs a key that has
to be stored and rotated, and it adds a rate limit for no extra signal. The
public scrape already carries everything needed, including per-task cost and
time. If a key exists it stays unused as a fallback.

`llm-leaderboard-aggregate` is **not** the ingest path. Its benchmark builder
installs Playwright and Chromium and is designed to run on a GitHub runner, not
on a 1 GB box; its value is the normalisation layer (`canonical_names.yaml`,
`model_alias_mapping.json`, `source_confidence.yaml`, `reasoning_defaults.yaml`),
which can be copied as data without taking the dependency.

## What the router consumes

From `data/artificial_analysis_intelligence_index_full.csv`:

| Column | Used for |
| --- | --- |
| `intelligenceIndex` (+ `intelligenceIndexIsEstimated`) | cold-start `a(m)` prior |
| `est_time_per_task_min` | `E[T]` prior, and therefore `v` |
| `est_output_tokens_per_task` | expected output tokens per task |
| `est_reasoning_tokens_per_task` | separates thinking cost from answer cost |
| `est_cost_per_task_usd` | sanity check against the route's own credit tariff |
| `price1mInputTokens`, `price1mOutputTokens` | cross-check provider prices |
| `outputSpeed` | `T` prior for generated tokens |
| `isReasoning`, `suffix`, effort-tagged rows | see "Reasoning effort" below |

`intelligenceIndexIsEstimated` must be honoured: an estimated row is a prior of
lower confidence, not a measurement.

## Reasoning effort is a route dimension

`effort_variants.csv` (`model_id, effort, metric, value, source`) and
`config/reasoning_defaults.yaml` already model this in the aggregate repo. The
router must treat effort as a separate route property:

- The same model at `low` and `high` effort is two products with different `T`
  and different quality. Collapsing them into one number is what makes ladders
  fail.
- Escalating effort on one model is usually cheaper than escalating the model,
  so the ladder should try effort first. That requires effort to be visible in
  the route record, not folded into a single score.
- `reasoning_defaults.yaml` decides which effort row stands for "the base
  model" when a caller does not specify one.

## Trust: which benchmarks still mean anything

Rule: prefer held-out or rotating evaluations (fresh issues, time-gated
problems, or an evaluator that runs the models itself) over static public sets.
Any static public benchmark is saturated within about two model releases.

| Benchmark | Verdict | Why | Use |
| --- | --- | --- | --- |
| Artificial Analysis Intelligence Index | **use, primary** | AA runs the models itself on a fixed harness, so the usual "lab reports its own number" vector is absent. It aggregates third-party suites, so it inherits their contamination. | Main cold-start prior, with `isEstimated` respected. |
| AA coding-agents / agentic tabs | **use, primary** | Harness is held constant across rows, so model comparisons are meaningful; carries per-task time and cost. | Agentic priors and `E[T]`. |
| SWE-rebench | **use** | Continuous, evaluated on issues published after the model's cutoff, which is the strongest contamination defence available. | Main agentic coding signal. |
| BenchLM agentic | **use with caution** | Only wired source with agentic scores for current 2026 releases, but methodology and weights are not published. | Signal, never truth. Cap its weight. |
| SWE-bench Verified | **do not rank on** | `swebench.com` has accepted no new run since 2026-02-26 (recorded in `llm-leaderboard-aggregate/docs/benchmark-sources.md`); the set is also the classic contamination target. | Excluded from priors; keep for historical context only. |
| Aider polyglot | **do not rank on** | Leaderboard stale since 2025; missing every 2026 frontier model. Their own registry already downweighted it 0.20 -> 0.10. | Excluded. |
| LiveCodeBench | **floor only** | Time-gated problems resist contamination, but LeetCode-style tasks are narrow and frontier scores are saturating. | Sanity floor ("can it code at all"), not a ranking. |
| MMLU / MMLU-Pro | **excluded** | Saturated for frontier models and present in training corpora for years. | No signal. |
| GPQA (Diamond) | **low weight** | Useful for small models, saturating at the top; questions have circulated publicly long enough to be trained on. | Weak tiebreak only. |
| HLE | **low weight, prefer AA's run** | Designed to resist saturation, but the question set leaked and is treated as training data by some labs. | Only as measured by AA, never as a self-report. |
| LMArena | **chat preference only** | Pairwise human votes measure style and helpfulness, not coding ability; longer, heavily formatted answers win, and the arena has a history of vote-integrity problems. | Chat-style tiebreak for `chat` profiles only. Never for `code`. |
| Terminal-Bench | **context only** | Results depend on the scaffold as much as the model, so cross-row comparison is not model comparison. | Read AA's harness-controlled row instead. |
| Vals AI | **excluded** | Their own analysis excludes it: enterprise voice/workflow focus, not agentic coding. | None. |
| llm-stats | **speed only** | Provider-side self-reported TPS/TTFT. | Provider speed fallback. |
| AwesomeAgents | **speed fallback** | Third-party aggregation of AA speed figures. | Only when AA speed is missing. |

Two more rules that follow from the table:

1. **Never compare raw percentages across suites.** Percentile-normalise within
   each benchmark first, as `benchmarks/scoring.py` already does. A 64% on one
   suite and a 64% on another are not the same ability.
2. **A missing score is never a zero.** AA writes `0.0` for "not run" on some
   columns; a literal zero must be read as missing, or models get penalised for
   evaluations that never happened.

## Model identity

Join on the existing alias data rather than inventing a matcher:

- `canonical_names.yaml` and `model_alias_mapping.json` for id normalisation.
- Exact normalised slug first; when an id is missing a size or effort token,
  fall back to a looser match and **mark it** (the `~` convention that
  `free_frontier.py` already uses).
- An unmatched model keeps its learned value and gets no prior. It must not
  inherit a similarly named model's score.

## Prior policy

Owner decision (2026-11-17): the **intelligence index is the primary prior**,
not the coding index. Artificial Analysis now weights agentic work inside the
intelligence index, and the coding index covers far fewer models, so using the
coding index alone would leave most of a deliberately wide catalog without any
prior at all.

Precedence when several numbers exist for one model:

1. `intelligenceIndex` - broad coverage, agentic-weighted.
2. Coding / agentic index columns where the model has them - corroboration and
   a tiebreak, never a replacement.
3. Per-benchmark rows from the trusted list above, percentile-normalised.
4. No prior at all - the model keeps its learned value and inherits nothing.

`intelligenceIndexIsEstimated` marks a prior as low-confidence. An estimated row
must never outrank a measured one at equal capability.

## Catalog breadth

The catalog is deliberately wide: the goal is never to run out of usable
capacity, accepting a slow or weak model as the fallback. Breadth is therefore
not something to prune, and the earlier "add a provider only if it fixes
scarcity" advice is withdrawn.

Breadth still needs a gate, or a wide catalog becomes a slow one:

- `unverified` - newly added, has never answered a probe. Dispatchable only
  when nothing verified is available.
- `verified` - answered one cheap probe successfully. Normal dispatch.
- `dead` - failed its probe or returned a hard provider error. Excluded from
  dispatch entirely, so no request is spent on a route already known to be
  down.
- Re-probe `dead` and `unverified` rows weekly. Providers recover, and an API
  key can be replaced; a route that is dead today can be capacity tomorrow.
- Never delete a row. Archiving is a dispatch decision, not a data decision.

This is what keeps a wide catalog from turning into retry fan-out: the existing
`supacoder`, `jasper` and `crax` failures are routes being attempted while
known-dead, which costs attempts and pollutes latency telemetry.

## Open items

- Which artefact to pin: the 3.4 MB intelligence-index CSV (rich, slower to
  fetch) or the 11 KB coding-agents CSV (small, fewer columns). Recommendation:
  pin the full index CSV for priors, and fetch it at most daily.
- Copy `source_confidence.yaml`, `reasoning_defaults.yaml` and
  `canonical_names.yaml` into this repo as data so the router does not depend on
  a second checkout at runtime.
- No ingest code exists yet; this document is the source decision it will
  follow.

## Starting balances are not queryable — learn them from refusals

Probed 24 enabled providers with their real keys on 2026-11-17 against
`/api/user/self`, `/api/v1/key` and `/user/balance`: 3 returned HTTP 200 and
none returned a numeric balance field. The relays authenticate their balance
dashboards by session cookie, not by API key, so balances cannot be harvested
with the keys we hold.

Decision: do not track starting balances.

- Usage comes from `telemetry.db` (`llm_events`, per-provider tokens and cost).
- Exhaustion is learned from the provider's own refusal: 402/403
  "insufficient balance" marks that route exhausted until its reset window.
- The moment of refusal is more accurate than any cached number, and costs no
  extra requests.

## What may be marked dead

Only a transport-level failure is evidence that a provider's API is gone:

- DNS failure, connection refused, TLS failure, or the gateway edge reporting
  the host unreachable.

These are **not** dead and must never be archived out of dispatch:

- 401/403 — our key is bad or revoked
- 402/429 — quota exhausted or rate limited
- 404 model-not-found — our `free_tier_models` mapping is wrong
- 5xx from the provider's application layer

Those mark the route `unverified` or raise a config task; the provider stays in
the catalog and keeps its models. "Dead" is a dispatch filter, not a deletion:
dead routes are excluded from dispatch and re-probed weekly, and the provider
entry is never removed.
