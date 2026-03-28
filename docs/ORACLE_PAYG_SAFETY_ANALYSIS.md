# Oracle Cloud PAYG Safety Analysis
Generated: 2026-03-28

## Decision Summary

**Recommendation: Switch to PAYG with strict controls is feasible but requires 3 prerequisites first.**

Do NOT switch until all three are in place:
1. Budget alert at $1 configured (Oracle Budget)
2. IAM policy denying paid compute shapes (blocks accidental paid instances)
3. You have verified Always Free resource limits apply post-upgrade

---

## Why Consider PAYG?

Oracle Always Free (current) gives:
- 2x VM.Standard.E2.1.Micro (AMD, 1 OCPU, 1GB RAM each)
- OR up to 4 OCPU + 24GB RAM total on VM.Standard.A1.Flex (ARM)

**The A1.Flex limit is the main reason to upgrade.** Always Free accounts are sometimes
capped at only the AMD micro instances in practice, and the A1.Flex availability is
unreliable (waitlisted in many regions). PAYG accounts get consistent access to A1.Flex
(4 OCPU, 24GB RAM, ARM) which remains permanently free.

---

## What Stays Free After Upgrading to PAYG

Oracle's Always Free tier resources remain free **forever** even on a PAYG account:

| Resource | Always Free Limit | Notes |
|---|---|---|
| VM.Standard.E2.1.Micro | 2 instances | AMD, 1 OCPU, 1GB RAM each |
| VM.Standard.A1.Flex | 4 OCPU + 24GB RAM total | ARM — can be 1x4OCPU or 4x1OCPU |
| Block Volume | 200GB total | Boot + data volumes |
| Object Storage | 20GB | Standard tier |
| Outbound data transfer | 10TB/month | **This is the key risk — see below** |
| Autonomous Database | 2 instances, 20GB each | |
| Load Balancer | 1x 10Mbps | |

**Source**: https://www.oracle.com/cloud/free/

---

## Billing Risk Analysis

### Risk 1: Data Egress — MEDIUM RISK

**Free limit**: 10TB/month outbound to internet.
**Your usage**: LLM API proxy forwards requests/responses. A typical LLM response is ~2-5KB.
At 14,400 requests/day (Groq daily limit as proxy): ~70MB/day → ~2.1GB/month. Well within 10TB.

However: If you accidentally expose a public endpoint with no auth and get scraped/abused,
you could spike. The proxy already requires `PROXY_API_KEY` auth — this is your protection.

**Mitigation**: Auth is already required. Monitor with Oracle Metrics (Networking → VCN Metrics → Bytes Egress).

### Risk 2: Paid Compute Shapes — LOW RISK (if IAM policy is set)

The only way to incur compute charges is to launch a non-Always-Free instance shape.
An IAM deny policy blocks this at the account level.

**Mitigation**: See IAM Policy section below.

### Risk 3: Block Volume Over 200GB — LOW RISK

Your current usage is likely <100GB. Oracle charges ~$0.0255/GB-month over 200GB.
Easy to avoid if you don't intentionally add storage.

### Risk 4: Object Storage Over 20GB — LOW RISK

Not currently using Object Storage for the gateway. Stay under 20GB.

### Risk 5: Paid Database or Load Balancer features — VERY LOW

Not using these beyond free tier.

---

## Hard Safety Controls

### Step 1: Set a Budget with Hard Alert

In Oracle Cloud Console → Billing → Budgets:

```
Budget Name: safety-limit
Budget Amount: $5 USD/month
Alert Threshold: 80% ($4)
Alert Type: ACTUAL
Email: your-email@example.com
```

This sends email when you hit $4 of actual charges. Does NOT automatically stop resources
(Oracle doesn't support hard spending stops), but gives early warning.

**Also set a second alert at 100% ($5).**

### Step 2: IAM Policy — Block Paid Compute Shapes

In Oracle Cloud Console → Identity → Policies → Create Policy:

```
Name: block-paid-compute
Description: Prevent accidental launch of paid instance shapes
Statement:
  deny group <your-group-or-user-OCID> to manage instance-family in tenancy
    where request.instance.shape not in ('VM.Standard.E2.1.Micro', 'VM.Standard.A1.Flex')
```

OR more permissive (allow compute management but block specific paid families):

```
deny any-user to launch instance in tenancy
  where request.instance.shape = 'VM.Standard3.Flex'
deny any-user to launch instance in tenancy
  where request.instance.shape = 'VM.Standard.E3.Flex'
deny any-user to launch instance in tenancy
  where request.instance.shape = 'VM.Standard.E4.Flex'
```

**Important**: Test these policies before upgrading. IAM policies in Oracle use
`deny` statements that override `allow`. Verify you can still launch A1.Flex after
applying the policy.

### Step 3: Set Spending Limit Notification in Account Settings

Billing → Payment Method → Spending Limit:
- Set credit card spending limit to $10/month if Oracle allows it
- Note: Oracle may not enforce hard credit limits on PAYG — the budget alert is your real safeguard

### Step 4: Disable Data Egress Auto-scaling (if applicable)

For the LLM gateway: ensure the proxy has rate limiting enabled so it can't be abused.
The `proxy_api_keys` field in `router_config.yaml` already enforces auth.

---

## Upgrade Procedure

1. Log into Oracle Cloud Console
2. Go to: Billing → Upgrade to Paid Account
3. Enter credit card (required for PAYG)
4. **Before upgrading**: Screenshot current Always Free resource inventory
5. After upgrading: Verify existing VMs are still marked Always Free in their details
6. Immediately set Budget alert ($5 limit)
7. Apply IAM deny policy for paid shapes
8. Test: Try to launch a VM.Standard3.Flex — it should be denied by IAM

**Rollback**: Oracle does not allow downgrading from PAYG back to Always Free.
This is a one-way operation. Ensure controls are in place FIRST.

---

## A1.Flex Benefits (What You Get)

After PAYG upgrade, your free A1.Flex allocation (4 OCPU, 24GB RAM) enables:

| Option | Config | Use |
|---|---|---|
| Option A | 1x A1.Flex: 4 OCPU, 24GB RAM | Consolidate VPS1+VPS2 into one powerful instance |
| Option B | 2x A1.Flex: 2 OCPU, 12GB RAM each | Keep VPS1 (gateway) and VPS2 (ZeroClaw) separate but more powerful |
| Option C | 4x A1.Flex: 1 OCPU, 6GB RAM each | Maximum instance count, still free |

**Recommendation: Option B** — 2x A1.Flex. VPS1 gets 2 OCPU + 12GB RAM (handles LiteLLM + gateway much better),
VPS2 gets 2 OCPU + 12GB RAM (handles opencode agent loops with room to spare).

The AMD E2.1.Micro instances (current) can be deleted or repurposed.

---

## Decision Matrix

| Factor | Always Free | PAYG + controls |
|---|---|---|
| Billing risk | Zero | Low (with IAM + budget) |
| A1.Flex access | Unreliable/waitlisted | Consistent |
| Current VPS RAM | 1GB each | 12GB each (Option B) |
| Gateway memory pressure | High | Eliminated |
| ZeroClaw stability | OOM-prone | Stable |
| Action required | None | IAM policy + budget first |

**Verdict**: Switch when you have 30 minutes to set up the safety controls properly.
Do not switch on a whim. The IAM deny policy is the critical safeguard.

---

## Monitoring After Switch

```bash
# Check current month charges
# Oracle Console → Billing → Cost and Usage

# OR via CLI (if oci-cli installed)
oci usage-api usage-summary request-summarized-usages \
  --tenant-id <your-tenancy-ocid> \
  --granularity DAILY \
  --query-type COST \
  --date-range-name LAST_30_DAYS
```

Check weekly for the first month after switching.
