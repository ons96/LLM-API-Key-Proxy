# QUESTIONS.md

Open questions with the default this run took. None of these blocked progress.

1. **Is the vps-gateway-tunnel failure permanent?**
   `vps-gateway-tunnel.service` has restarted 2965 times;
   `ssh 100.71.95.75:22` times out while `40.233.101.233:22` works, so the
   Tailscale path is down, not the host.
   Default taken: bypass the tunnel and measure over direct ssh. Not repaired
   because it needs network access outside this repo.

2. **What are the undocumented loopback daemons on 8081/8090/8095/8100/8101/8123?**
   Four `python`/`uvicorn` processes listen on loopback ports and are not systemd
   units known to this repo. They are outside the brief's four-service list.
   Default taken: not touched, not counted in the 250 MB budget, reported here.

3. **Is `g4f*` free or paid?**
   The DB says `fully_free`; all six variants returned HTTP 402 (payment
   required) to a 1-token probe today.
   Default taken: treat the live 402 as authoritative and demote them in M1;
   leave the DB row unchanged until the catalog pipeline writes it, so the change
   is auditable.

4. **Where do per-provider quota numbers come from?**
   `quota_amount` is NULL for all 203 providers. The only concrete number found
   in the wild is septorlabs' 50 requests/day, learned from a 429 body.
   Default taken: header-first parsing with body/`observed_429` fallback, and an
   explicit `unknown` state rather than an invented number.

5. **Does the host's `cooldown_policy.yaml` matter?**
   The live process logs `CooldownPolicy loaded ... 10 providers configured,
   default wait_on_429=False`, and the file is on this branch but not at the
   host's HEAD.
   Default taken: treat the file as live behaviour, include it in the M6
   reconcile.

6. **Should the pre-existing order-dependent MoE test be fixed?**
   `TestMoEMode::test_expert_candidate_selection` fails when
   `tests/test_router_core.py` runs alone and passes in the full suite; reproduced
   on the pristine tree.
   Default taken: out of scope, documented, not "fixed" by weakening it.
