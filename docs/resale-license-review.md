# Resale review: Free LLM Gateway Starter Kit

Scope: can `gumroad-bundle/` be sold as a paid product, given the repository
licenses and the provider terms it touches. Reviewed 2026-11-21.

## Verdict

Yes, as maintained. The bundle sells setup materials. It does not redistribute
gateway source, credentials, or provider access, so neither the LGPL-3.0
obligations nor the provider terms attach to the sale.

## License facts

| Component | License | File |
|---|---|---|
| `src/rotator_library/` (core library) | LGPL-3.0 | `src/rotator_library/COPYING`, `COPYING.LESSER` |
| `src/proxy_app/` (proxy application) | MIT | `src/proxy_app/LICENSE` |

- MIT permits commercial sale. The obligation is to keep the copyright and
  permission notice with any copy of the software. The bundle ships no copy of
  `src/proxy_app/`, so there is nothing to notice.
- LGPL-3.0 is a free-software license, not a non-commercial one: selling is
  permitted. Its obligations (source availability, relinking, no extra
  restrictions) trigger on **distribution of the library**. The bundle ships no
  library code; `quickstart.sh` requires the buyer to run it inside their own
  checkout of the public repository.
- Because the buyer obtains the source themselves from
  `https://github.com/ons96/LLM-API-Key-Proxy`, the upstream project is the
  distributor of the LGPL component, not this product.

## Provider terms

`README.md` and `.env.starter` already require the buyer to use accounts they
control, and the bundle carries no keys, no tokens, and no shared access. That
keeps the sale outside every provider's resale and key-sharing clauses. The
one rule that must stay true: a sold config may only reference the buyer's own
vendor-approved keys. Free-tier and keyless routing in a config sold to a
stranger gets that stranger's keys and IPs banned.

## Guards in place

- `package-release.sh` packs an explicit allowlist of seven files; it cannot
  walk the checkout.
- `package-release.sh` scans every packaged file for credential shapes
  (`sk-`, `ghp_`, `xai-`, `AKIA`, private-key blocks, key/secret/password/token
  assignments of 24+ characters) and aborts the build on a hit.
  Self-check: `bash package-release.sh --self-test`.

## Residual risks

1. A future edit that vendors `src/rotator_library/` into the bundle would
   create LGPL distribution obligations. The allowlist prevents this today.
2. Marketplace terms (Lemon Squeezy, Gumroad) add their own refund, tax, and
   prohibited-product rules on top of this review.
3. The listing copy must keep saying the source is public and that the product
   is setup materials, not exclusive access. `GUMROAD_LISTING.md` does.
