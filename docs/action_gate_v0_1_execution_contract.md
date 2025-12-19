# Action Gate v0.1 — Execution Contract

**Version:** 0.1
**Date:** 2025-12-19
**Scope:** PC-side execution layer

---

## 1. Observed vs Actionable Data

### What Replay v1.9 CSV **is**

The replay CSV (`replay_core0_events_v1_9.py`) is designed for **offline analysis and observability**:

- Reconstructs the pipeline from stored JSONL event records
- Outputs tile-level metrics: `cycles_physical`, `compass_global_score`, `rotor_state`
- Provides post-hoc analysis of direction lock transitions
- Enables debugging and verification of pipeline behavior

**Key characteristic:** All timestamps are **static** (from recorded events). There is no real-time delta or data staleness.

### What Replay v1.9 CSV **is NOT**

The replay CSV does **not** provide execution-worthy inputs because:

| Missing Input | Why It's Missing |
|---------------|------------------|
| `data_age_ms` | No wall-clock delta; time is embedded in events |
| Real-time coherence | Confidence computed post-hoc, not live |
| Stale data detection | Cannot detect "no new events" condition |
| Force-fallback triggers | No external safety signals in replay |

**Consequence:** Feeding replay CSV data to Action Gate will produce `FALLBACK` or `OBSERVE` states, never `ACTIVE`.

---

## 2. Why Replay CSV Does Not Provide LOCKED/Coherence

### Design Choice: Offline vs Live

The `RealtimePipeline` in `realtime_states_v1_9_canonical.py` contains a **simplified** `MovementBody` class that differs from the full `MovementBodyV3`:

| Feature | `MovementBody` (v1.9 replay) | `MovementBodyV3` (live) |
|---------|------------------------------|-------------------------|
| Lock mechanism | Simple tile counter | Threshold-based with confidence |
| Coherence score | Computed from cycles | Live compass + direction fusion |
| Data staleness | Not tracked | Real-time age computation |
| Idle timeout | Not implemented | Active decay and reset |

### Implications

1. **Replay lock_state** is based on consecutive tile counts, not execution-ready confidence thresholds.
2. **Replay coherence** reflects cycle consistency, not real-time pipeline health.
3. The replay pipeline **intentionally omits** execution-layer signals to maintain its role as an observability tool.

---

## 3. Action Gate Execution Contract

### 3.1 Required Inputs

The Action Gate expects a `GateInput` dataclass with:

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `now_ms` | `int` | `time.time() * 1000` | Current wall-clock time (external, deterministic) |
| `coherence_score` | `float` | `compass_snapshot.global_score` | Pipeline coherence (0.0–1.0) |
| `lock_state` | `str` | `direction_lock_state` | "LOCKED", "SOFT_LOCK", or "UNLOCKED" |
| `data_age_ms` | `int` | Computed from `ageE_s` | Time since last event |
| `rotor_active` | `bool` | `rotor_state == "MOVEMENT"` | Whether motor is active |
| `force_fallback` | `bool` | External signal | Manual safety trigger |
| `arm_signal` | `bool` | Derived from lock | Signal to arm the gate |
| `activate_signal` | `bool` | Derived from lock + coherence | Signal to activate |

### 3.2 Output: GateOutput

| Field | Type | Description |
|-------|------|-------------|
| `state` | `GateState` | Current gate state (IDLE/OBSERVE/ARMED/ACTIVE/FALLBACK) |
| `decision` | `GateDecision` | ALLOW_ACTIVE, HOLD_OBSERVE, or FORCE_FALLBACK |
| `reason` | `str` | Execution-neutral reason token |
| `timestamp_ms` | `int` | Timestamp of decision |
| `allowed` | `bool` | Whether action is permitted |
| `log_entries` | `List[str]` | Gate log entries for this evaluation |

### 3.3 Gate States

| State | Meaning | Entry Conditions |
|-------|---------|------------------|
| IDLE | Waiting for input | Initial state, or after reset from FALLBACK |
| OBSERVE | Monitoring | First input received, or conditions lost |
| ARMED | Ready to activate | Coherence + lock conditions met |
| ACTIVE | Executing allowed | Activation triggered with sufficient coherence |
| FALLBACK | Safe mode | Data stale, coherence drop, or manual trigger |

### 3.4 Reason Tokens (Execution-Neutral)

```
data_stale           — Data age exceeded threshold
coherence_drop       — Coherence score dropped below threshold
lock_lost            — Direction lock state became UNLOCKED
insufficient_context — Not enough data to proceed
safety_reset         — Recovery from fallback
armed_condition_met  — Conditions met to arm gate
input_received       — First input processed
activation_triggered — Gate activated
manual_fallback      — External fallback request
```

---

## 4. Integration Point

### 4.1 Location

**Script:** `scripts/live_symphonia_v2_0.py`
**Hook location:** After `l1.update()` call (line ~453)

### 4.2 Integration Moment

The gate evaluates **after** the L1/L2 pipeline snapshot is computed:

```
Event received → L2.feed_event() → L1.update() → Gate.evaluate()
```

This ensures the gate has access to:
- Fresh `direction_lock_state` from L2
- Current `compass_snapshot.global_score`
- Computed `ageE_s` from L1

### 4.3 Why This Location

1. **After L2:** Lock state and coherence are updated
2. **After L1:** Age metrics are computed
3. **Per-tick:** Gate evaluates on each pipeline tick (50ms or event-driven)
4. **Non-blocking:** Gate evaluation adds minimal latency

### 4.4 What the Hook Does

```python
# Build gate input from live pipeline snapshot fields
gate_input = GateInput(
    now_ms=int(now * 1000),
    coherence_score=dc,              # compass confidence
    lock_state=lk,                   # direction_lock_state
    data_age_ms=int(snap.ageE_s * 1000),
    rotor_active=(rotor_state == "MOVEMENT"),
    arm_signal=(lk in ("SOFT_LOCK", "LOCKED")),
    activate_signal=(lk == "LOCKED" and dc >= 0.5),
)
gate_output = action_gate.evaluate(gate_input)
```

### 4.5 What the Hook Does NOT Do

- No motor actions triggered
- No pipeline state modified
- No CSV export changes
- No replay logic changes

---

## 5. Test Strategy

### 5.1 How to Verify the Gate Works

**Live pipeline test:**
```bash
python3 scripts/live_symphonia_v2_0.py --log --scoreboard
```

Check the JSONL log for `_gate` entries:
```json
{
  "_gate": {
    "state": "OBSERVE",
    "decision": "HOLD_OBSERVE",
    "reason": "insufficient_context",
    "allowed": false
  }
}
```

### 5.2 Relevant Log Lines

In gate-internal logs (written to logger):
```
GATE_ENTER state=OBSERVE reason=input_received from_state=IDLE t_ms=...
GATE_BASIS fields={'coherence': '0.00', 'lock': 'UNLOCKED', ...}
GATE_DECISION state=OBSERVE output=HOLD_OBSERVE
```

### 5.3 Summary Section

At session end, the summary includes:
```
----------------------------------------------------------------------
ACTION GATE v0.1 (execution layer observability):
  Final state:        OBSERVE
  Total transitions:  3
  Last decision:      HOLD_OBSERVE
  Last reason:        insufficient_context
  Action allowed:     False
```

### 5.4 Why CSV-Based Tests Always Produce FALLBACK

When feeding replay CSV data through the gate:

1. **No real-time aging:** `data_age_ms` cannot be computed from static CSV timestamps
2. **Stale threshold exceeded:** Default `stale_data_threshold_ms` is 5000ms
3. **No live coherence:** Confidence values in CSV are post-hoc, not live-computed
4. **Result:** Gate defaults to `FALLBACK` or `OBSERVE`, never `ACTIVE`

This is **by design**: the gate requires live pipeline data to make execution decisions.

---

## 6. Module Location

| File | Purpose |
|------|---------|
| `sym_cycles/action_gate_v0_1.py` | Gate state machine implementation |
| `scripts/live_symphonia_v2_0.py` | Live pipeline with gate hook |
| `scripts/smoke_action_gate_v0_1.py` | Standalone smoke test |
| `docs/action_gate_v0_1_execution_contract.md` | This document |

---

## 7. Contract Summary

**The Action Gate:**
- Evaluates live pipeline snapshots only
- Uses execution-neutral terminology
- Defaults to safe posture (FALLBACK always reachable)
- Logs decisions for observability
- Does NOT execute motor actions (v0.1)

**Replay CSV:**
- Provides observability data only
- Is NOT suitable for execution-layer decisions
- Will always result in non-ACTIVE gate states

---

*Contract version: 1.0 — SORA CodeX Contract v1.0*
