# Revision 3 review notes

Design under review: `DESIGN-v2-verified-routing.md` revision 3 (PR #367, branch
`design-v3-money-objective`). Section 0 says disagreements belong here rather
than in a silent rewrite. This file is written on `feat/newapi-sidecar`, where
revision 3 is not merged yet, so read it beside PR #367.

What I built from revision 3 is on branch `s0-cost-vector`: S0 only (cost vector,
lambda_T plumbing, free-only audit). Nothing after S0.

## 1. Wrong or oversold

**The free-only guardrail contradicts itself and would break a feature.**
§6.5 wants a priced route in a free-only process to be a startup error; §14 Q9
asks whether paid routes should stay visible for "what would it have cost".
Both cannot hold, and the code already chose. `/v1/cost-estimate` prices paid
routes from the ledger (`QuotaLedger.estimate_cost`), and
`test_newapi_server.py::test_cost_estimate_is_ledger_backed_and_read_only`
configures a `quota_type="paid"` route into a runtime with `free_only=True`.
Refusing to boot would delete that capability and turn one stale row from a
6,827-model catalogue into a full sidecar outage - for no safety gain, because
`score_route` already returns `paid_excluded` / `free_price_not_verified` and
`embed()` refuses paid aliases. So: audit, not refusal.

**"lambda_T is the only free parameter" is not true of this code.** `Weights`
also carries `quota_lambda=0.01` and `escalation_cost=0.03`, and `Route` carries
`shadow_price`, `prefill_ms_per_token`, `handoff_ms`. `escalation_cost` is a
second money<->probability exchange rate and it contributes to every score. S0
does not leave one knob; it leaves one *new* knob.

**"Free-only behaviour is identical to revision 2" holds for coding only.**
Revision 2 scored 0.003/s for `auto-coding` and 0.001/s for `auto-chat`. One
lambda_T of 0.003 triples the weight of time on chat traffic. I kept both
defaults and made the uniform value an override. If you want one number, chat
rankings move the day it is set - deliberately, not accidentally.

**S0 renames a known-wrong quantity without changing it.** §1 "Lie 2" says
`quota_lambda * pressure` is utilisation, not opportunity, and §2.1 says `O` is
displacement. S0 stores the utilisation term in a column called
`cost_opportunity`. That is the only split that leaves ranking untouched, but the
name now claims more than the number measures. Reports therefore carry
`pricing: "utilisation"`, and S4 must redefine the column - every weekly dollar
and opportunity figure before S4 should be read as an estimate of pressure, not
of displacement.

**S8's pseudo-counts quietly change how fast evidence wins.** The doc keeps
`1 + k*q` and `1 + k*(1-q)` with `k=4`, so the prior *total* goes from 4 to 6 for
a perfect model, and the total is what sets how much evidence it takes to move.
A `q=0.9` model starts at `Beta(4.6, 1.4)`, p=0.77: above neutral, so a
leaderboard can outrank an unseeded route for roughly the same ~8 attempts the
doc worries about. Deferred rather than implemented, because the honest `k` needs
your graded traces, not arithmetic.

## 2. Gaps the design does not cover

1. **Credit routes drop their customer price from the score.** For
   `quota_type="credit_based"`, `score_route` computes `money` and discards it
   (`cost_money = 0`), while `_amounts()` still charges the `money_day` axis. A
   credit route with a real price is ranked as free but budget-capped as paid -
   rank and feasibility disagree. I preserved that to keep S0 behaviour-neutral,
   but §2.2 ("a multiplier is just a price") makes such routes reachable.
2. **`pressure` collapses every axis, so S3's ladder cannot reserve per axis.**
   §6.4 requires per-axis lambda (rpm/tpm/rpd/tpd) and per-axis reservation;
   `Capacity` exposes one `max()` float. S3 needs `pressure_by_axis` from the
   ledger, and no stage schedules adding it. Cheap now, awkward later.
3. **`evidence()` cannot tell a timeout from wrong-but-confident.** Every failure
   adds one to beta. §4.3 wants retry-on-operational and escalate-on-semantic,
   but the bandit needs them as separate counts, or a flaky provider and a wrong
   model stay indistinguishable. `error_class` and `verdict` are already stored;
   nothing reads them.
4. **`money_day` is daily, but §6.5 promises a monthly cap.** `_start()` floors
   to 86400, so `daily_budget=5` is $5/day, not $5/month. A monthly policy
   cannot be expressed today, and the prose says it can.
5. **Verification time is not in the ledger.** `latency_ms` is provider time; the
   protocol's `duration_ms` on the outcome POST is not stored. `E[T]` in the
   ladder includes verification, so S3's gate is unfalsifiable until that number
   is captured.
6. **`free_only` means two different things.** The sidecar (ledger) excludes
   priced routes at score time; `provider_metadata.load_provider_group_deployments`
   drops them at load time with a `continue`. Same flag, different layer, and the
   `ROUTER_*` path loses them entirely instead of reporting them.

## 3. The real bottleneck, and whether S3/S4/S5 are in the right order

Once the plugin posts verdicts, the bottleneck is **oracle latency and the
verdict round-trip**, not routing intelligence.

- Every attempt pays V0+V4+V2 before the next tier can start, so a two-tier
  cascade costs about two oracle runs. A 30 s verifier dominates a 4 s model
  difference, and the ladder's most important input becomes `T_verify`, which
  (gap 5) is not measured.
- Learning is gated on a POST that arrives after the task finishes, and the
  client has no queue or retry. A dropped POST is a silently lost sample.

So the ordering is wrong in one place: **S4's pricing should precede S3's
ladder**, or S3 must be explicitly mechanical. S3's DP consumes `E[O]`, and
today's `O` is the utilisation term §1 calls a lie; building the DP on it means
S4 rewrites every constant under a shipped feature. The ladder is 20 lines of
enumeration; the pricing is the substance. S5 can stay after S3 as long as the
ladder refuses to escalate into a credit route until `a(m, class)` exists.

Also: S3's gate ("live p90 T2G improves vs S1") is blocked on S2's plugin, which
the doc itself says is missing. That is the critical path, not S3 code.

## 4. Questions that change what I build next

1. **lambda_T**: what is an hour worth, and is one number right for both
   `auto-coding` and `auto-chat`? If chat should stay cheaper per second, §2.1's
   single-rate model is wrong and `Weights` keeps two values.
2. **Paid routes under free_only**: (a) refuse to boot, (b) keep them in the
   ledger, excluded from scoring, visible to cost-estimate and reports (what I
   built), or (c) score them but never dispatch? §6.5 and Q9 disagree.
3. **Is the dollar cap monthly or daily?** If monthly I need a real window axis;
   if daily, §6.5's prose needs fixing before someone implements the wrong one.
4. **May the router grade on a number it does not measure?** If yes, S3 ships on
   provider-only `T` and `T_verify` lands later. If no, the plugin must post
   `duration_ms` before the ladder means anything.
5. **Credit routes with a non-zero price**: forbid the combination, or score the
   money (gap 1)?
