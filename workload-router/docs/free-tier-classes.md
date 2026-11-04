# Free-tier classes and ordering

Status: implemented locally, not deployed. Every provider currently configured on
the gateway is treated as free; the sidecar must never present its list price as
money the owner spends.

## The four route classes

`Route.quota_type` remains the capacity mechanism. `Route.free_class` is the
operator-facing funding label derived from it.

| Funding class | `quota_type` | Meaning | Required route data | Score effect |
| --- | --- | --- | --- | --- |
| `free_unlimited` | `unlimited` | No finite credit balance. Rate limits may still apply. | `input_price=0`, `output_price=0`; optional `rpm`/`rpd`/`tpm`/`tpd` limits. | Zero money; finite limits still add quota pressure. |
| `free_limited` | `rate_limited` | Free but capped by a renewable quota window. | Zero actual prices plus at least one configured or learned `rpm`/`rpd`/`tpm`/`tpd` limit. | Zero money plus quota pressure, then latency/success. |
| `free_credits` | `credit_based` | One-time or finite provider credits. | Zero actual prices, `credit_balance`, and `credit_input_price` / `credit_output_price`. | Notional credit consumption plus quota pressure. |
| `paid` | `paid` | Real money would be charged. | Normal prices and limits. | Excluded when free-only is on. |
| `unknown` | `unknown` | Not enough metadata to route safely. | None. | Hard-infeasible, not silently treated as free. |

`free_limited` covers daily-renewable free tiers and other finite windows.
`free_credits` covers one-time credit grants. A trial grant is a finite credit
unless it renews; if it renews, use `rate_limited` with a daily window instead.

## Existing provider metadata is the source

The sidecar should not make operators re-tag providers that the gateway already
classifies. The read-only source of truth is
`/home/ubuntu/llm-provider-manager/llm_providers.db`; its `providers` table has
`free_one_time`, `free_daily`, `free_unlimited`, `checkin_required`,
`checkin_unlimited`, `provider_type`, `classification_source`, and
`human_confirmed`. `providers_database.yaml` is the human-maintained fallback.

Use the operator preview:

```sh
python3 newapi_operator.py classify \
    --provider-db /home/ubuntu/llm-provider-manager/llm_providers.db
```

The adapter preserves explicit route metadata and reports mismatches for review.
It does not copy credentials or write the provider database. `quota_amount`,
`quota_unit`, and `quota_reset` are frequently `NULL`; a credit balance or
tariff is not inferred from a missing value.

## Zero customer price, non-zero credit cost

Free credit routes need two different prices:

- `input_price` and `output_price` are the actual customer price. They stay
  zero for every free route.
- `credit_input_price` and `credit_output_price` are a notional provider-credit
  tariff, in the same currency units per million tokens. They are used only to
  deplete `credit_balance` and rank the route.

This separation is deliberate. If a one-time credit route used its list price
in `input_price`, `/v1/cost-estimate` would claim the owner spent money. If it
used zero for both, the credit balance would never deplete.

A `credit_based` route with a balance but no credit tariff and zero actual
prices is rejected by capacity checking as `unknown_credit_price`. Existing
paid-style credit routes that set a positive actual price keep the compatibility
fallback of using that price as the credit tariff.

## Gateway metadata mapping

The gateway and provider database use several related vocabularies. Map them
when seeding a sidecar route:

| Existing signal | Sidecar `quota_type` | Sidecar `free_class` |
| --- | --- | --- |
| `free_unlimited`, `no_api_key_required`, `fully_free`, `unlimited_rate_limited` | `unlimited` | `free_unlimited` |
| `free_daily`, `daily_renewable`, `free_tier_limited` | `rate_limited` | `free_limited` |
| `free_one_time`, `one_time_credit` | `credit_based` | `free_credits` |
| `freemium` with a free allowance | `rate_limited` or `credit_based`, according to whether the allowance renews | `free_limited` or `free_credits` |
| paid/unknown | `paid` or `unknown` | `paid` or `unknown` |

`unlimited_rate_limited` and `fully_free` mean no finite credit balance, even
when the provider has RPM or daily request limits. Keep those routes as
`unlimited`; the limits still feed quota pressure in scoring. Use
`rate_limited` for a genuinely capped free allowance such as a daily renewable
tier.

## Examples

No credit or usage allowance:

```json
{
  "route_id": "provider-model",
  "quota_type": "unlimited",
  "input_price": 0,
  "output_price": 0
}
```

Daily renewable free tier:

```json
{
  "route_id": "provider-model",
  "quota_type": "rate_limited",
  "input_price": 0,
  "output_price": 0,
  "limits": {"rpd": 1000, "tpd": 5000000},
  "day_offset": 0
}
```

One-time provider credits:

```json
{
  "route_id": "provider-model",
  "quota_type": "credit_based",
  "input_price": 0,
  "output_price": 0,
  "credit_input_price": 1.0,
  "credit_output_price": 2.0,
  "credit_balance": 5.0
}
```

The credit prices above are notional: 1 currency unit per million input tokens
and 2 per million output tokens. They only decide how fast the 5-unit grant is
consumed.

## Ranking and visibility

- `free_unlimited` has zero effective money; if finite rate limits are present,
  it still adds `quota_lambda * pressure`.
- `free_limited` adds `quota_lambda * pressure`; it moves down as its window
  fills.
- `free_credits` adds `credit_cost * shadow_price` plus quota pressure; it moves
  down before the grant is exhausted.
- `paid` is excluded under `free_only`.
- `unknown` and unknown pricing are hard-infeasible.

`GET /v1/router/diagnostics` includes a `free_classes` count. The operator
`routes` preview shows `free_class` and `quota_type` for every proposed route.
`POST /v1/cost-estimate` returns the selected route's `free_class`, a `free`
boolean, cost zero, and `source: "free_tier"` when the actual price is zero.

## Limits

- Route seeding is still explicit JSON; provider-database import does not
  automatically choose a class. This avoids guessing whether a provider's
  allowance renews or is a one-time grant.
- Class/price contradictions are visible in scoring and cost output; operators
  should set actual prices to zero for all free routes.
- No live provider quotas were probed for this change. Tests are provider-free.
