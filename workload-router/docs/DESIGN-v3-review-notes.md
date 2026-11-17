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

1. **Nothing enforces the credit-price convention.** For
   `quota_type="credit_based"`, `score_route` sets `cost_money = 0` and prices
   the credit tariff as `O`, while `_amounts()` still charges the `money_day`
   axis. That is correct when the convention holds (customer price 0, provider
   tariff in `credit_*_price`) and contradictory when it does not (ranked free,
   budgeted as paid). The fix is a configure-time warning naming the route, not
   an exclusion: a priced credit route is usually a metadata mistake, and the
   owner's own providers are credit-funded at a real tariff.
2. **`pressure` collapses every axis, so S3's ladder cannot reserve per axis.**
   §6.4 requires per-axis lambda (rpm/tpm/rpd/tpd) and per-axis reservation;
   `Capacity` exposes one `max()` float. S3 needs `pressure_by_axis` from the
   ledger, and no stage schedules adding it. Cheap now, awkward later.
3. **`evidence()` cannot tell a timeout from wrong-but-confident.** Every failure
   adds one to beta. §4.3 wants retry-on-operational and escalate-on-semantic,
   but the bandit needs them as separate counts, or a flaky provider and a wrong
   model stay indistinguishable. `error_class` and `verdict` are already stored;
   nothing reads them.
4. ~~**`money_day` is daily, but §6.5 promises a monthly cap.**~~ Fixed in the
   S0 follow-up: `monthly_budget` is a separate UTC calendar-month axis, and
   both caps are enforced independently, so the stricter one binds. What is
   still missing is *pacing*: nothing re-prices a dollar when the month is
   nearly spent. The formula is `monthly_remaining / days_remaining`, and it
   belongs in S4's shadow price rather than in a rewritten limit - rewriting
   would refuse a legal spend on day 3 and erase the difference between "out
   today" and "out this month".
5. ~~**Verification time is not in the ledger.**~~ Fixed: the outcome POST's
   `duration_ms` is stored as `oracle_ms`, reported beside provider time as
   `oracle_s`, and never folded into `elapsed_s`. What is still missing is S3
   actually using it in `E[T]`.
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

## 5. Answers received (2026-11-17), and what they change

This supersedes questions 1, 2, 3 and 5 in section 4; question 4 is settled in
the same list.

Owner's goal, restated: **maximise completed tasks per unit of time, at the
lowest real cost.** Never pay out of pocket; renewable free tiers first;
one-time-credit providers as overflow, never as the foundation.

Settled:

- **lambda_T stays at 0.003, and the question behind it was the wrong one.** The
  owner's answer to "wait 30 s or burn a whole allowance" depends on how big the
  allowance is and how many other providers could take the work. That is a
  scarcity question, not a wage. So the exchange rate is now stated once, in
  seconds: `allowance_seconds` is how much waiting one whole allowance is worth,
  and the quota price is derived from it (`lambda_T x allowance_seconds`)
  instead of being a second constant that drifts. Defaults reproduce the old
  0.01 exactly (3.3 s coding, 10 s chat). Config: `SIDECAR_LAMBDA_T` and
  `SIDECAR_ALLOWANCE_SECONDS`.
- **Paid routes**: ledger-visible, excluded from scoring (unchanged). `$` stays
  pinned at zero - the owner will not spend from their own wallet - while
  provider credits are fine to spend and are priced as `O`. No route needs
  excluding for being "credit-based"; only a route that would bill the owner
  does.
- **One-time credits**: a preference, not a ban. Renewable free tiers first; a
  one-time grant is worth using when the model or the grant is genuinely better
  (§6.3's `a > 0`), never as the only foundation.
- **Dollar caps**: daily and monthly, blank = no limit, both checked so the
  stricter one binds; monthly resets on the calendar month. Implemented here as
  the `money_month` axis (a UTC calendar month, ignoring `day_offset`). Pacing -
  "if the month only has $10 left, today's real budget is $10/days_left" - is
  deliberately not folded into the limit: it is the price of a dollar in S4, so
  a derived average never refuses a spend the real cap allows.
- **Verification time** is captured before S3, not after.
- **Credit-funded routes are not paid routes** (revised after the owner's
  provider example: a $4/1M model funded by a large balance plus daily check-ins
  costs no cash). Convention: customer `input_price`/`output_price` stay 0, the
  provider's tariff goes in `credit_*_price`, and `O` prices depletion.
  `credit_renewal` is now a field: `daily` meters the balance on a day window so
  a check-in grant refills instead of draining forever, `none` keeps a one-time
  grant as a lifetime total. Excluding by price would have locked out exactly
  the providers the owner wants.

Still not modelled, and why:

1. **How big the allowance is.** `pressure` is used/ceiling, so a small allowance
   saturates quickly and is priced as dear - which matches "a small allowance is
   easy to burn". What it cannot see is *time to reset*: an allowance resetting
   in ten minutes should be worth nothing. That is the S4 forecast.
2. **How many alternative providers exist.** Nothing in the score knows that
   five other routes could absorb this task, which is exactly when burning one
   provider's quota is cheap. The principled fix is the shadow auction (§6.2):
   price falls when substitutable supply is large. It needs the demand
   forecast, so it is S4, not a per-request divisor - a divisor would also
   ignore per-axis admission, which does not exist yet.
3. **Throughput.** "Completed tasks per hour" needs the ladder to know how many
   attempts a task really costs; dividing by `s` is S0's approximation of
   expected effort per completed task.

One consequence worth knowing: quota and time now move together, so raising
`SIDECAR_LAMBDA_T` makes the router both faster-preferring and more willing to
burn quota, in the same proportion. Tuning one no longer silently unbalances
the other.

## 6. Which way the ladder runs (answer 4)

Cheap-first is not a law; it is what the arithmetic says, and the arithmetic was
missing its largest term.

The scorer never ranks by price per token. It divides by `s`, the probability
that one attempt is correct, so it ranks by expected cost per *completed* task.
A model that is expensive per token but reliable already wins outright on that
basis - that half of the owner's intuition is satisfied by the formula as it
stands.

The open question is the *order* of a ladder. For a cheap tier A then a strong
tier B, cascading beats one-shot B only when

    T_A + v  <  s_A * T_B

where `T` is provider time and `v` is verification time, paid by every attempt.
With the appendix's own numbers (`T_A = 4 s`, `T_B = 40 s`, `s_A = 0.55`):

| `v` per attempt | cascade A->B | one-shot B | winner |
|---|---|---|---|
| 0 s | 4 + 0.45x40 = 22 s | 40 s | cascade |
| 30 s | 34 + 0.45x70 = 65.5 s | 70 s | cascade, barely |
| 60 s | 64 + 0.45x100 = 109 s | 100 s | one-shot |

So the direction depends mostly on how long the oracle takes - which is why
`oracle_ms` is stored now, before the ladder exists, rather than after.

Recommendation for S3: do not hardcode either direction. Enumerate the feasible
tiers (<= 4, so <= 24 orderings), score every prefix of every ordering on
`lambda_T*E[T] + E[O] + E[M]`, and take the winner. "Start strong and stop" is
then a prefix of length one, and "cheap first" is an output rather than an
assumption. Two caveats: attempts are not independent (a compiler error makes
the next attempt more likely to succeed than `s` suggests), and that is why the
choice has to be replayed against stored traces rather than settled on paper.

On benchmark seeding (the owner's other suggestion): agreed in principle as the
cold-start prior, and S8 already specifies the shape (bounded pseudo-counts,
`prior_source`, expiry). It is blocked on data, not on code: the source list,
and whether the import is a manual JSON file or a scheduled fetch, are still
open - and inventing scores is worse than starting from `2, 2`.
