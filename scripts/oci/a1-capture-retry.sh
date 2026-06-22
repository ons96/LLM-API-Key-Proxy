#!/usr/bin/env bash
# OCI A1.Flex capture/retry script (task-board #206)
#
# Oracle Cloud Always-Free A1.Flex instances fail ~90% of launch attempts with
# "Out of host capacity in this availability domain". The fix is a retry loop
# that spans hours/days until Oracle releases capacity.
#
# Uses oci-cli (free, Python). Reads config from ~/.oci/config or env.
# Idempotent: safe to re-run. Backs off exponentially with jitter.
#
# Usage:
#   ./a1-capture-retry.sh --dry-run                  # print plan, exit 0
#   ./a1-capture-retry.sh --max-retries 100          # loop up to 100 times
#   ./a1-capture-retry.sh --once                     # single attempt, exit on failure
#   ./a1-capture-retry.sh --instance-name gateway-a1 --ocpus 2 --ram-gb 12
#
# Env vars (all optional, flags override):
#   OCI_CLI_USER, OCI_CLI_FINGERPRINT, OCI_CLI_TENANCY, OCI_CLI_REGION,
#   OCI_CLI_KEY_FILE, OCI_CLI_CONFIG_FILE
#   A1_REGION, A1_AD, A1_SHAPE, A1_OCPUS, A1_RAM_GB, A1_IMAGE_OCID,
#   A1_SUBNET_OCID, A1_SSH_KEY_OCID, A1_COMPARTMENT_OCID,
#   A1_INSTANCE_NAME, A1_MAX_RETRIES, A1_BACKOFF_MIN_S, A1_BACKOFF_MAX_S
#
# Exit codes:
#   0  success (instance created or --dry-run)
#   1  bad args / missing deps / launch error after retries
#   3  --once and attempt failed (no retry)
#
# Refs: task-board #206 (this), #205 (A1 migration), #194 (epic).
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REGION="${A1_REGION:-us-sanjose-1}"
AD="${A1_AD:-1}"
SHAPE="${A1_SHAPE:-VM.Standard.A1.Flex}"
OCPUS="${A1_OCPUS:-2}"
RAM_GB="${A1_RAM_GB:-12}"
IMAGE_OCID="${A1_IMAGE_OCID:-}"
SUBNET_OCID="${A1_SUBNET_OCID:-}"
SSH_KEY_OCID="${A1_SSH_KEY_OCID:-}"
COMPARTMENT_OCID="${A1_COMPARTMENT_OCID:-}"
INSTANCE_NAME="${A1_INSTANCE_NAME:-gateway-a1}"
MAX_RETRIES="${A1_MAX_RETRIES:-200}"
BACKOFF_MIN_S="${A1_BACKOFF_MIN_S:-60}"
BACKOFF_MAX_S="${A1_BACKOFF_MAX_S:-3600}"
BACKOFF_FACTOR="${A1_BACKOFF_FACTOR:-1.5}"
DRY_RUN=0
ONCE=0
LOG_FILE="${A1_LOG_FILE:-/tmp/oci-a1-capture.log}"

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Capture an OCI Always-Free A1.Flex instance via retry loop.

Options:
  --region R            OCI region (default: $REGION)
  --ad N                Availability domain number (default: $AD)
  --shape S             Shape (default: $SHAPE)
  --ocpus N             OCPUs 1-4 (default: $OCPUS)
  --ram-gb N            RAM in GB, 1-24 (default: $RAM_GB)
  --image OCID          Image OCID (required unless --dry-run)
  --subnet OCID         Subnet OCID (required unless --dry-run)
  --ssh-key OCID        SSH public key OCID (optional)
  --compartment OCID    Compartment OCID (optional, defaults to tenancy root)
  --name NAME           Instance display name (default: $INSTANCE_NAME)
  --max-retries N       Max retry attempts (default: $MAX_RETRIES)
  --backoff-min S       Min backoff seconds (default: $BACKOFF_MIN_S)
  --backoff-max S       Max backoff seconds (default: $BACKOFF_MAX_S)
  --once                Single attempt, exit on failure (exit 3)
  --dry-run             Print planned command, exit 0
  --log FILE            Log file path (default: $LOG_FILE)
  -h, --help            Show this help

Env: A1_REGION, A1_AD, A1_SHAPE, A1_OCPUS, A1_RAM_GB, A1_IMAGE_OCID,
     A1_SUBNET_OCID, A1_SSH_KEY_OCID, A1_COMPARTMENT_OCID, A1_INSTANCE_NAME,
     A1_MAX_RETRIES, A1_BACKOFF_MIN_S, A1_BACKOFF_MAX_S
     OCI_CLI_* vars (oci-cli native)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --region) REGION="$2"; shift 2;;
        --ad) AD="$2"; shift 2;;
        --shape) SHAPE="$2"; shift 2;;
        --ocpus) OCPUS="$2"; shift 2;;
        --ram-gb) RAM_GB="$2"; shift 2;;
        --image) IMAGE_OCID="$2"; shift 2;;
        --subnet) SUBNET_OCID="$2"; shift 2;;
        --ssh-key) SSH_KEY_OCID="$2"; shift 2;;
        --compartment) COMPARTMENT_OCID="$2"; shift 2;;
        --name) INSTANCE_NAME="$2"; shift 2;;
        --max-retries) MAX_RETRIES="$2"; shift 2;;
        --backoff-min) BACKOFF_MIN_S="$2"; shift 2;;
        --backoff-max) BACKOFF_MAX_S="$2"; shift 2;;
        --once) ONCE=1; shift;;
        --dry-run) DRY_RUN=1; shift;;
        --log) LOG_FILE="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 1;;
    esac
done

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log() {
    local ts
    ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo "[$ts] $*" | tee -a "$LOG_FILE" >&2
}

# ---------------------------------------------------------------------------
# Preflight: numeric validation (always runs, even for --dry-run)
# ---------------------------------------------------------------------------
if ! [[ "$OCPUS" =~ ^[1-4]$ ]]; then
    echo "ERROR: --ocpus must be 1-4 (free tier max 4)" >&2
    exit 1
fi
if ! [[ "$RAM_GB" =~ ^[0-9]+$ ]] || [[ "$RAM_GB" -lt 1 ]] || [[ "$RAM_GB" -gt 24 ]]; then
    echo "ERROR: --ram-gb must be 1-24 (free tier max 24)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Backoff math (defined early so --dry-run can show it)
# ---------------------------------------------------------------------------
compute_backoff() {
    local attempt="$1"
    # Exponential: BASE * FACTOR^(attempt-1), capped at MAX, with up to 25% jitter.
    # ponytail: use awk to avoid bash integer overflow on large exponents.
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

# ---------------------------------------------------------------------------
# Dry-run: print plan and exit 0 (no deps required)
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "=== DRY RUN ==="
    echo "Would run up to $MAX_RETRIES attempts with backoff ${BACKOFF_MIN_S}s→${BACKOFF_MAX_S}s (factor ${BACKOFF_FACTOR})"
    echo "Region: $REGION"
    echo "AD: $AD"
    echo "Shape: $SHAPE"
    echo "OCPUs: $OCPUS"
    echo "RAM: ${RAM_GB}GB"
    echo "Image: ${IMAGE_OCID:-<not set — would fail>}"
    echo "Subnet: ${SUBNET_OCID:-<not set — would fail>}"
    echo "Compartment: ${COMPARTMENT_OCID:-<auto-resolve at runtime>}"
    echo "Instance name: $INSTANCE_NAME"
    echo "SSH key OCID: ${SSH_KEY_OCID:-<none>}"
    echo "Log file: $LOG_FILE"
    echo
    echo "Launch command would be:"
    echo "  oci compute instance launch --region $REGION --availability-domain ${AD} \\"
    echo "    --shape $SHAPE --shape-config '{\"ocpus\":${OCPUS},\"memoryInGBs\":${RAM_GB}}' \\"
    echo "    --image-id <IMAGE> --subnet-id <SUBNET> --display-name $INSTANCE_NAME \\"
    echo "    --compartment-id <COMPARTMENT> --assign-public-ip false"
    exit 0
fi

# ---------------------------------------------------------------------------
# Non-dry-run preflight: required args + deps
# ---------------------------------------------------------------------------
if [[ -z "$IMAGE_OCID" ]]; then
    echo "ERROR: --image OCID required (or set A1_IMAGE_OCID env)" >&2
    exit 1
fi
if [[ -z "$SUBNET_OCID" ]]; then
    echo "ERROR: --subnet OCID required (or set A1_SUBNET_OCID env)" >&2
    exit 1
fi
if ! command -v oci >/dev/null 2>&1; then
    echo "ERROR: oci-cli not found. Install: pip install oci-cli" >&2
    exit 1
fi

# Resolve compartment if not set: default to tenancy root.
# ponytail: Always Free tenancy root == tenancy OCID; oci-cli can resolve it
# from config without an extra API call in most cases.
if [[ -z "$COMPARTMENT_OCID" ]]; then
    log "Resolving tenancy root compartment..."
    if ! COMPARTMENT_OCID="$(oci iam compartment list --query 'data[0]."compartment-id"' --raw-output 2>/dev/null)"; then
        if ! COMPARTMENT_OCID="$(oci iam tenancy get --query 'data.id' --raw-output 2>/dev/null)"; then
            log "ERROR: cannot resolve compartment OCID; set --compartment explicitly"
            exit 1
        fi
    fi
    log "Using compartment: $COMPARTMENT_OCID"
fi

# ---------------------------------------------------------------------------
# Build launch args
# ---------------------------------------------------------------------------
LAUNCH_ARGS=(
    compute instance launch
    --region "$REGION"
    --availability-domain "${AD}"
    --shape "$SHAPE"
    --shape-config "{\"ocpus\":${OCPUS},\"memoryInGBs\":${RAM_GB}}"
    --image-id "$IMAGE_OCID"
    --subnet-id "$SUBNET_OCID"
    --display-name "$INSTANCE_NAME"
    --compartment-id "$COMPARTMENT_OCID"
    --assign-public-ip false
    --query 'data.id' --raw-output
)

if [[ -n "$SSH_KEY_OCID" ]]; then
    LAUNCH_ARGS+=(--metadata "{\"ssh_authorized_keys\":\"${SSH_KEY_OCID}\"}")
fi

# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------
attempt() {
    local attempt_num="$1"
    log "Attempt $attempt_num/$MAX_RETRIES: launching $INSTANCE_NAME ($SHAPE ${OCPUS}OCPU ${RAM_GB}GB)"

    local instance_ocid
    local rc
    local stderr_out

    # ponytail: capture stderr to file (avoid subshell + pipefail interaction
    # with grep classification). Note: `local rc=$?` captures `local`'s own
    # exit, not oci's — assign in separate statement.
    local err_file="/tmp/oci-a1-capture.err.$$"
    if oci "${LAUNCH_ARGS[@]}" >/tmp/oci-a1-capture.out 2>"$err_file"; then
        instance_ocid="$(cat /tmp/oci-a1-capture.out)"
        if [[ -n "$instance_ocid" && "$instance_ocid" == ocid1.instance.* ]]; then
            log "SUCCESS: instance $instance_ocid created"
            echo "$instance_ocid"
            rm -f "$err_file"
            return 0
        fi
        log "WARNING: launch returned 0 but output not a valid instance OCID: $instance_ocid"
        rm -f "$err_file"
        return 1
    fi
    rc=$?
    stderr_out="$(cat "$err_file" 2>/dev/null)"
    rm -f "$err_file"

    # Classify failure (grep on variable, no pipe — safe under set -e)
    if grep -qi "out of host capacity" <<<"$stderr_out"; then
        log "  capacity unavailable (expected); will retry"
    elif grep -qi "limitexceeded\|quota" <<<"$stderr_out"; then
        log "  quota exceeded (tenancy cap hit); will retry"
    elif grep -qi "notauthorized\|authenticat" <<<"$stderr_out"; then
        log "  ERROR: auth failure — not retrying"
        log "  $stderr_out"
        return 2
    elif grep -qi "conflict\|already exists" <<<"$stderr_out"; then
        log "  instance name conflict — check tenancy for orphan instances"
        log "  $stderr_out"
        return 2
    else
        log "  unexpected failure (rc=$rc): $stderr_out"
    fi
    return 1
}

log "=== A1.Flex capture retry starting ==="
log "Target: $INSTANCE_NAME ($SHAPE ${OCPUS}OCPU ${RAM_GB}GB in $REGION AD$AD)"
log "Max retries: $MAX_RETRIES, backoff ${BACKOFF_MIN_S}s→${BACKOFF_MAX_S}s (factor ${BACKOFF_FACTOR})"

attempt_num=0
while [[ $attempt_num -lt $MAX_RETRIES ]]; do
    attempt_num=$((attempt_num + 1))

    rc=0
    attempt "$attempt_num" || rc=$?
    if [[ $rc -eq 0 ]]; then
        log "=== Capture complete after $attempt_num attempts ==="
        exit 0
    fi

    # rc=2 means non-retryable (auth/conflict); rc=1 means retryable
    if [[ $rc -eq 2 ]]; then
        log "=== Aborting: non-retryable error ==="
        exit 1
    fi

    if [[ $ONCE -eq 1 ]]; then
        log "=== --once set, not retrying ==="
        exit 3
    fi

    if [[ $attempt_num -ge $MAX_RETRIES ]]; then
        log "=== Max retries ($MAX_RETRIES) exhausted ==="
        exit 1
    fi

    wait_s="$(compute_backoff "$attempt_num")"
    log "  backing off ${wait_s}s before attempt $((attempt_num + 1))"
    sleep "$wait_s"
done

log "=== Max retries ($MAX_RETRIES) exhausted ==="
exit 1
