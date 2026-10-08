# AI Workload Router

## Session startup

Read `STATE.md`, `HANDOFF.md`, `README.md`, and the P0 specification at
`docs/specs/p0-router.md` before changing files. The full architecture plan is
at `/home/osees/CodingProjects/temp-omnirush/docs/ai-workload-router-architecture.md`.

Preserve existing work. Inspect `git status` before edits. Work in small,
verified milestones and update `STATE.md` after each milestone. Do not print,
copy, rotate, or commit credentials. Keep implementation free-only and suitable
for a 1 GB VPS. Run the narrowest relevant validation first, then broader checks
when justified by the change.

The current repository contains portable artifacts for an implementation that
also exists in `/home/ubuntu/LLM-API-Key-Proxy` on the VPS. Do not modify the VPS
repository from this local session unless explicitly requested.

Deployment target: Oracle VPS 40 at `40.233.101.233`, with 1 GB RAM, zram,
weak CPU, and no GPU. Keep code and dependencies suitable for that host.
Local tests must remain provider-free; deployment requires a separate explicit
step.
