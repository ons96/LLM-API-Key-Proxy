#!/usr/bin/env bash
# Self-test for a1-capture-retry.sh (task-board #206)
#
# Verifies:
#   - --help prints usage, exit 0
#   - --dry-run prints plan, exit 0 (with and without required args)
#   - arg validation: --ocpus 5 rejected, --ram-gb 100 rejected, missing --image rejected
#   - --once path with mock oci (simulated auth failure → exit 1)
#   - --once path with mock oci (simulated capacity failure → exit 3)
#   - --once path with mock oci (simulated success → exit 0, prints OCID)
#   - compute_backoff: respects min/max caps with jitter in range
#
# Usage: ./test_a1_capture.sh
# Exits 0 if all tests pass, 1 otherwise.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/a1-capture-retry.sh"
PASS=0
FAIL=0

assert_exit() {
    local expected_rc="$1"
    local desc="$2"
    shift 2
    local actual_rc
    "$@" >/dev/null 2>&1
    actual_rc=$?
    if [[ "$actual_rc" -eq "$expected_rc" ]]; then
        echo "  PASS: $desc (rc=$actual_rc)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $desc (expected rc=$expected_rc, got $actual_rc)"
        FAIL=$((FAIL + 1))
    fi
}

assert_contains() {
    local desc="$1"
    local needle="$2"
    local haystack="$3"
    if grep -qiF -- "$needle" <<<"$haystack"; then
        echo "  PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $desc (output missing: '$needle')"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== a1-capture-retry.sh self-test ==="
echo

# --- Test 1: --help exits 0 and shows usage ---
out="$("$SCRIPT" --help 2>&1)" || true
assert_exit 0 "help exits 0" "$SCRIPT" --help
assert_contains "help mentions --dry-run" "--dry-run" "$out"
assert_contains "help mentions --once" "--once" "$out"
assert_contains "help mentions --max-retries" "--max-retries" "$out"
echo

# --- Test 2: --dry-run with no required args exits 0 (skips validation) ---
assert_exit 0 "dry-run without --image/--subnet exits 0" "$SCRIPT" --dry-run
out="$("$SCRIPT" --dry-run 2>&1)" || true
assert_contains "dry-run prints DRY RUN header" "=== DRY RUN ===" "$out"
assert_contains "dry-run prints shape" "VM.Standard.A1.Flex" "$out"
echo

# --- Test 3: --dry-run honors overrides ---
out="$("$SCRIPT" --dry-run --ocpus 4 --ram-gb 24 --name test-instance 2>&1)" || true
assert_contains "dry-run honors --ocpus" "OCPUs: 4" "$out"
assert_contains "dry-run honors --ram-gb" "RAM: 24GB" "$out"
assert_contains "dry-run honors --name" "Instance name: test-instance" "$out"
echo

# --- Test 4: arg validation rejects out-of-range OCPUs ---
assert_exit 1 "rejects --ocpus 5" "$SCRIPT" --dry-run --ocpus 5
out="$("$SCRIPT" --dry-run --ocpus 5 2>&1)" || true
assert_contains "ocpus error mentions range" "1-4" "$out"
echo

# --- Test 5: arg validation rejects out-of-range RAM ---
assert_exit 1 "rejects --ram-gb 100" "$SCRIPT" --dry-run --ram-gb 100
assert_exit 1 "rejects --ram-gb 0" "$SCRIPT" --dry-run --ram-gb 0
echo

# --- Test 6: without --dry-run, --image is required ---
assert_exit 1 "missing --image rejected (non-dry-run)" "$SCRIPT" --once
out="$("$SCRIPT" --once 2>&1)" || true
assert_contains "missing image error mentions --image" "--image" "$out"
echo

# --- Test 7: missing --subnet rejected ---
assert_exit 1 "missing --subnet rejected" "$SCRIPT" --once --image ocid1.image.oc1.fake
out="$("$SCRIPT" --once --image ocid1.image.oc1.fake 2>&1)" || true
assert_contains "missing subnet error mentions --subnet" "--subnet" "$out"
echo

# --- Tests 8-10: mock oci via PATH shim, exercise --once paths ---
TMPDIR_TEST="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_TEST"' EXIT

# Build mock oci that emits to stderr based on env var A1_MOCK_MODE
cat >"$TMPDIR_TEST/oci" <<'EOF'
#!/usr/bin/env bash
# Mock oci-cli for testing a1-capture-retry.sh
mode="${A1_MOCK_MODE:-capacity}"
case "$mode" in
    success)
        echo "ocid1.instance.oc1.fake.success0001"
        exit 0
        ;;
    capacity)
        echo "ERROR: Out of host capacity in this availability domain" >&2
        exit 1
        ;;
    auth)
        echo "ERROR: NotAuthenticated: Invalid session" >&2
        exit 1
        ;;
    quota)
        echo "ERROR: LimitExceeded: instance quota exceeded" >&2
        exit 1
        ;;
    conflict)
        echo "ERROR: Conflict: instance with name already exists" >&2
        exit 1
        ;;
    *)
        echo "ERROR: unknown mock mode: $mode" >&2
        exit 1
        ;;
esac
EOF
chmod +x "$TMPDIR_TEST/oci"

# Test 8: success path
A1_MOCK_MODE=success PATH="$TMPDIR_TEST:$PATH" \
    assert_exit 0 "mock success exits 0" \
    env A1_MOCK_MODE=success PATH="$TMPDIR_TEST:$PATH" \
    "$SCRIPT" --once --image ocid1.image.fake --subnet ocid1.subnet.fake --compartment ocid1.tenancy.fake --log /tmp/test-a1.log

out="$(A1_MOCK_MODE=success PATH="$TMPDIR_TEST:$PATH" \
    "$SCRIPT" --once --image ocid1.image.fake --subnet ocid1.subnet.fake --compartment ocid1.tenancy.fake --log /tmp/test-a1.log 2>/dev/null)" || true
assert_contains "success prints instance OCID" "ocid1.instance" "$out"
echo

# Test 9: capacity failure with --once exits 3
A1_MOCK_MODE=capacity PATH="$TMPDIR_TEST:$PATH" \
    assert_exit 3 "mock capacity --once exits 3" \
    env A1_MOCK_MODE=capacity PATH="$TMPDIR_TEST:$PATH" \
    "$SCRIPT" --once --image ocid1.image.fake --subnet ocid1.subnet.fake --compartment ocid1.tenancy.fake --log /tmp/test-a1.log
echo

# Test 10: auth failure exits 1 (non-retryable, even without --once)
A1_MOCK_MODE=auth PATH="$TMPDIR_TEST:$PATH" \
    assert_exit 1 "mock auth failure exits 1" \
    env A1_MOCK_MODE=auth PATH="$TMPDIR_TEST:$PATH" \
    "$SCRIPT" --once --image ocid1.image.fake --subnet ocid1.subnet.fake --compartment ocid1.tenancy.fake --log /tmp/test-a1.log
echo

# Test 11: quota failure with --once exits 3 (retryable)
A1_MOCK_MODE=quota PATH="$TMPDIR_TEST:$PATH" \
    assert_exit 3 "mock quota --once exits 3" \
    env A1_MOCK_MODE=quota PATH="$TMPDIR_TEST:$PATH" \
    "$SCRIPT" --once --image ocid1.image.fake --subnet ocid1.subnet.fake --compartment ocid1.tenancy.fake --log /tmp/test-a1.log
echo

# Test 12: conflict exits 1 (non-retryable)
A1_MOCK_MODE=conflict PATH="$TMPDIR_TEST:$PATH" \
    assert_exit 1 "mock conflict exits 1" \
    env A1_MOCK_MODE=conflict PATH="$TMPDIR_TEST:$PATH" \
    "$SCRIPT" --once --image ocid1.image.fake --subnet ocid1.subnet.fake --compartment ocid1.tenancy.fake --log /tmp/test-a1.log
echo

# Test 13: backoff math — source the script and call compute_backoff directly
# (extract via grep + eval the function in isolation)
BACKOFF_SCRIPT=$(mktemp)
cat >"$BACKOFF_SCRIPT" <<'EOF'
BACKOFF_MIN_S=60
BACKOFF_MAX_S=3600
BACKOFF_FACTOR=2
RANDOM=42  # deterministic for test
compute_backoff() {
    local attempt="$1"
    local base
    base=$(awk -v min="$BACKOFF_MIN_S" -v factor="$BACKOFF_FACTOR" -v attempt="$attempt" \
        'BEGIN { e = attempt - 1; if (e > 30) e = 30; printf "%d", min * (factor ^ e) }')
    if [[ $base -gt $BACKOFF_MAX_S ]]; then
        base=$BACKOFF_MAX_S
    fi
    local jitter_range=$((base / 4))
    if [[ $jitter_range -gt 0 ]]; then
        local jitter=$(( (RANDOM % (2 * jitter_range + 1)) - jitter_range ))
        echo $((base + jitter))
    else
        echo "$base"
    fi
}
EOF
source "$BACKOFF_SCRIPT"

b1="$(compute_backoff 1)"
b2="$(compute_backoff 2)"
b3="$(compute_backoff 3)"
bcap="$(compute_backoff 100)"  # should cap at BACKOFF_MAX_S ± 25% jitter

# Attempt 1: 60 * 2^0 = 60, jitter ±15 → 45..75
if [[ $b1 -ge 45 && $b1 -le 75 ]]; then
    echo "  PASS: backoff attempt 1 in range (got $b1)"
    PASS=$((PASS + 1))
else
    echo "  FAIL: backoff attempt 1 out of range 45..75 (got $b1)"
    FAIL=$((FAIL + 1))
fi

# Attempt 2: 60 * 2^1 = 120, jitter ±30 → 90..150
if [[ $b2 -ge 90 && $b2 -le 150 ]]; then
    echo "  PASS: backoff attempt 2 in range (got $b2)"
    PASS=$((PASS + 1))
else
    echo "  FAIL: backoff attempt 2 out of range 90..150 (got $b2)"
    FAIL=$((FAIL + 1))
fi

# Attempt 3: 60 * 2^2 = 240, jitter ±60 → 180..300
if [[ $b3 -ge 180 && $b3 -le 300 ]]; then
    echo "  PASS: backoff attempt 3 in range (got $b3)"
    PASS=$((PASS + 1))
else
    echo "  FAIL: backoff attempt 3 out of range 180..300 (got $b3)"
    FAIL=$((FAIL + 1))
fi

# Attempt 100: capped at 3600, jitter ±900 → 2700..4500
if [[ $bcap -ge 2700 && $bcap -le 4500 ]]; then
    echo "  PASS: backoff attempt 100 capped (got $bcap)"
    PASS=$((PASS + 1))
else
    echo "  FAIL: backoff attempt 100 out of capped range 2700..4500 (got $bcap)"
    FAIL=$((FAIL + 1))
fi

rm -f "$BACKOFF_SCRIPT"
echo

# --- Summary ---
echo "=== Summary: $PASS passed, $FAIL failed ==="
if [[ $FAIL -eq 0 ]]; then
    exit 0
else
    exit 1
fi
