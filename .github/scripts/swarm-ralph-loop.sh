#!/usr/bin/env bash
set -euo pipefail

max_attempts="${RALPH_MAX_ATTEMPTS:-3}"
attempt=1
ISSUE_PROMPT_FILE="${ISSUE_PROMPT_FILE:?ISSUE_PROMPT_FILE is required}"
ARCHITECT_OUTPUT_FILE="${ARCHITECT_OUTPUT_FILE:?ARCHITECT_OUTPUT_FILE is required}"
CODER_OUTPUT_FILE="${CODER_OUTPUT_FILE:?CODER_OUTPUT_FILE is required}"
SWARM_WORKDIR="${SWARM_WORKDIR:?SWARM_WORKDIR is required}"

while true; do
  echo "Starting Ralph Loop attempt $attempt of $max_attempts"
  ISSUE_PROMPT=$(cat "$ISSUE_PROMPT_FILE")
  AGENT_OUTPUT=$(cat "$ARCHITECT_OUTPUT_FILE")
  CODER_OUTPUT=$(cat "$CODER_OUTPUT_FILE")
  TEST_COMMANDS=$(awk '
    /^\[TEST_COMMANDS\]$/ {flag=1; next}
    /^\[[A-Z_]+\]$/ {if (flag) exit}
    flag {print}
  ' "$CODER_OUTPUT_FILE")
  QA_COMMANDS=$(awk '
    /^\[QA_COMMANDS\]$/ {flag=1; next}
    /^\[[A-Z_]+\]$/ {if (flag) exit}
    flag {print}
  ' "$ARCHITECT_OUTPUT_FILE")
  ISSUE_COMMAND=$(sed -n 's/.*`\(.*\)`.*/\1/p' "$ISSUE_PROMPT_FILE" | tail -n 1)
  TEST_COMMANDS="${TEST_COMMANDS:-$QA_COMMANDS}"
  TEST_COMMANDS="${TEST_COMMANDS:-$ISSUE_COMMAND}"

  if [ -z "$TEST_COMMANDS" ]; then
    echo "FAILURE_DIAGNOSTICS_REPORT"
    echo "[FAILED_TESTS]: QA phase could not find TEST_COMMANDS in coder output."
    echo "[DIAGNOSIS]: The coder report omitted the verification commands required for deterministic QA."
    echo "[SUGGESTED_NEXT_STEPS]: Tighten the coder output format so it always includes a concrete shell command block under [TEST_COMMANDS]."
    exit 1
  fi

  set +e
  TEST_OUTPUT=$(cd "$SWARM_WORKDIR" && bash -lc "$TEST_COMMANDS" 2>&1)
  TEST_EXIT_CODE=$?
  set -e

  if [ "$TEST_EXIT_CODE" -eq 0 ]; then
    echo "QA_SUCCESS_REPORT"
    echo "[VERIFIED_TEST_COMMANDS]"
    printf '%s\n' "$TEST_COMMANDS"
    echo "[TEST_OUTPUT]"
    printf '%s\n' "$TEST_OUTPUT"
    exit 0
  fi

  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "FAILURE_DIAGNOSTICS_REPORT"
    echo "[FAILED_TESTS]"
    printf '%s\n' "$TEST_COMMANDS"
    echo "[DIAGNOSIS]: The deterministic Ralph Loop test command exited with status $TEST_EXIT_CODE."
    echo "[TEST_OUTPUT]"
    printf '%s\n' "$TEST_OUTPUT"
    echo "[SUGGESTED_NEXT_STEPS]: Update the coder patch or test command so the repository reaches a passing terminal exit code."
    exit 1
  fi

  attempt=$((attempt + 1))
done
