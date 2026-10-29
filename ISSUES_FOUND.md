# Issues Found

*This file documents all bugs, issues, and incomplete items discovered during the overnight review.*

---

## Critical Blockers

| Project | Issue | Details | Status |
|---------|--------|---------|--------|
| llm-leaderboard | Type errors in llm_aggregated_leaderboard.py | Lines 115, 132 had None.find_all() errors. Line 305 has type error. | 🔴 Fixed |

---

## Core Functionality Broken

| Project | Issue | Details | Status |
|---------|--------|---------|--------|
| TBD | TBD | TBD | TBD |

---

## Compilation/Runtime Errors

| Project | Issue | Details | Status |
|---------|--------|---------|--------|
| llm-leaderboard | Potential pandas type error | LSP reports DataFrame columns parameter type mismatch (may be false positive) | ⚠️ Needs Testing |

---

## Missing Dependencies

| Project | Issue | Details | Status |
|---------|--------|---------|--------|
| TBD | TBD | TBD | TBD |

---

## Incomplete Features

| Project | Issue | Details | Status |
|---------|--------|---------|--------|
| free_research_agent | Multiple chat versions exist | flask_app, free_research_agent, puter-free-chatbot have overlapping functionality - needs consolidation | 🟡 Pending |
| agentic_gateway | Provider discovery not fully automated | Scanning implemented but could be more automatic | 🟡 Pending |
| agentic_gateway | Rate/Usage limit skip logic untested | Tracking exists but skip mechanism needs testing | 🟡 Pending |
| agentic_gateway | Real-time dynamic ordering untested | Speed tests exist but real-time tracking needs verification | 🟡 Pending |

---

## Minor Bugs

| Project | Issue | Details | Status |
|---------|--------|---------|--------|
| TBD | TBD | TBD | TBD |

---

## Documentation Missing

| Project | Issue | Details | Status |
|---------|--------|---------|--------|
| Multiple | README files missing | Many utility projects lack README.md | 🟡 Pending |
| Phase 5-7 | No dedicated projects | AI Research Assistant, Writing Assistant, Code Assistant, etc. have no dedicated implementations | 🟡 Pending |
| Tasks 1.6, 1.7 | No implementation found | YouTube Video Analysis and Android Agentic Tools are research-only, no code | 🟡 Pending |
