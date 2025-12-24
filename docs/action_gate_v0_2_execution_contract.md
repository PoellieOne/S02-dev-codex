# Action Gate v0.2 — Execution Contract

**Version:** 0.2
**Date:** 2025-12-24
**Scope:** PC-side execution layer
**Predecessor:** Action Gate v0.1

---

## 1. Action Intent

### 1.1 Definition

Action Intent is an input to the Action Gate with the following properties:

| Property | Value |
|----------|-------|
| Delivery | Externally provided |
| Frequency | Per tick |
| Persistence | None (must be re-sent each tick) |
| Default | `INTENT_NONE` |
| Revocable | Yes (by sending different value next tick) |

### 1.2 Explicit Constraints

Action Intent is:
- **Not a command** — The gate may reject it
- **Not truth** — It expresses external input, not system state
- **Not semantic** — No interpretation layer exists

Action Intent does **not**:
- Override gate states
- Bypass safety rules
- Introduce semantic meaning

### 1.3 Permitted Values (Exact)

| Value | Description |
|-------|-------------|
| `INTENT_NONE` | No intent expressed (default) |
| `INTENT_ACTIVATE` | Request to transition to ACTIVE |
| `INTENT_HOLD` | Request to maintain current allowance |
| `INTENT_RELEASE` | Request to disengage |

No other values are permitted.

---

## 2. Gate States

Unchanged from v0.1:

| State | Description |
|-------|-------------|
| `IDLE` | No active processing, waiting for input |
| `OBSERVE` | Monitoring inputs, not yet armed |
| `ARMED` | Conditions approaching threshold, ready to activate |
| `ACTIVE` | Actively allowing actions |
| `FALLBACK` | Safe mode, blocking actions, always reachable |

---

## 3. Execution Context Inputs

### 3.1 Existing Inputs (v0.1)

| Field | Type | Description |
|-------|------|-------------|
| `now_ms` | `int` | Current wall-clock time (external) |
| `coherence_score` | `float` | Pipeline coherence (0.0–1.0) |
| `lock_state` | `str` | "LOCKED", "SOFT_LOCK", or "UNLOCKED" |
| `data_age_ms` | `int` | Time since last event |
| `rotor_active` | `bool` | Whether motor is active |
| `force_fallback` | `bool` | External safety trigger |
| `arm_signal` | `bool` | Signal to arm the gate |

### 3.2 New Inputs (v0.2)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `action_intent` | `ActionIntent` | `INTENT_NONE` | Externally provided intent |
| `intent_source` | `str` | `"unknown"` | Source identifier for logging |

---

## 4. Binding Behavioral Rules

### 4.1 INTENT_ACTIVATE

| Condition | Result |
|-----------|--------|
| Gate state = `ARMED` | Allowed (if activation conditions met) |
| Gate state != `ARMED` | Rejected |
| Activation conditions not met | Rejected |

### 4.2 INTENT_HOLD

| Condition | Result |
|-----------|--------|
| Gate state = `ARMED` | Allowed |
| Gate state = `ACTIVE` | Allowed (maintains ACTIVE) |
| Gate state = other | Rejected |

### 4.3 INTENT_RELEASE

| Condition | Result |
|-----------|--------|
| Any state | `FORCE_FALLBACK` |

`INTENT_RELEASE` always forces transition to `FALLBACK`.

### 4.4 INTENT_NONE

| Condition | Result |
|-----------|--------|
| Gate state = `ARMED` | Cannot transition to `ACTIVE` |
| Gate state = `ACTIVE` | Transitions to `OBSERVE` |

Without Action Intent, no `ACTIVE` state is reachable.

---

## 5. State Transitions

### 5.1 Transition Table

| From | To | Conditions |
|------|----|------------|
| `IDLE` | `OBSERVE` | Any input received |
| `OBSERVE` | `ARMED` | Arm conditions met |
| `OBSERVE` | `FALLBACK` | Safety conditions OR `INTENT_RELEASE` |
| `ARMED` | `ACTIVE` | Activation conditions AND (`INTENT_ACTIVATE` OR `INTENT_HOLD`) |
| `ARMED` | `OBSERVE` | Arm conditions lost |
| `ARMED` | `FALLBACK` | Safety conditions OR `INTENT_RELEASE` |
| `ACTIVE` | `OBSERVE` | Coherence drop OR lock lost OR `INTENT_NONE` |
| `ACTIVE` | `FALLBACK` | Safety conditions OR `INTENT_RELEASE` |
| `FALLBACK` | `IDLE` | No force_fallback AND coherence >= threshold AND NOT `INTENT_RELEASE` |
| `*` | `FALLBACK` | Always allowed (dominant) |

### 5.2 Arm Conditions

| Condition | Threshold |
|-----------|-----------|
| `coherence_score` >= `arm_coherence_min` | 0.4 |
| `lock_state` != "UNLOCKED" | — |
| `arm_signal` = true OR (`lock_state` = "LOCKED" AND `coherence_score` >= 0.5) | — |

### 5.3 Activation Conditions

| Condition | Threshold |
|-----------|-----------|
| `coherence_score` >= `activation_coherence_min` | 0.7 |
| `lock_state` = "LOCKED" | — |

### 5.4 Fallback Conditions

| Condition | Reason Token |
|-----------|--------------|
| `action_intent` = `INTENT_RELEASE` | `intent_release` |
| `force_fallback` = true | `manual_fallback` |
| `data_age_ms` > `stale_data_threshold_ms` | `data_stale` |
| `coherence_score` < 0.1 | `coherence_drop` |

---

## 6. Gate Output

| Field | Type | Description |
|-------|------|-------------|
| `state` | `GateState` | Current gate state |
| `decision` | `GateDecision` | `ALLOW_ACTIVE`, `HOLD_OBSERVE`, or `FORCE_FALLBACK` |
| `reason` | `str` | Reason token |
| `timestamp_ms` | `int` | Timestamp of decision |
| `allowed` | `bool` | Whether action is permitted |
| `intent_received` | `ActionIntent` | Intent that was evaluated |
| `intent_accepted` | `bool` | Whether intent was accepted |
| `log_entries` | `List[str]` | Gate log entries |

---

## 7. Logging Contract

### 7.1 Log Events

| Event | Fields | Description |
|-------|--------|-------------|
| `ACTION_INTENT` | `value`, `source` | Logs received intent per tick |
| `GATE_BASIS` | `fields` | Logs execution context |
| `GATE_ENTER` | `state`, `reason`, `from_state`, `t_ms` | Logs state transition |
| `GATE_DECISION` | `state`, `output`, `intent`, `basis` | Logs decision with intent |
| `GATE_FALLBACK` | `reason`, `t_ms` | Logs fallback trigger |

### 7.2 Example Log Sequence

```
ACTION_INTENT value=INTENT_ACTIVATE source=external
GATE_BASIS fields={'coherence': '0.80', 'lock': 'LOCKED', 'data_age_ms': 100, 'rotor': False}
GATE_ENTER state=ACTIVE reason=intent_activate_accepted from_state=ARMED t_ms=12345
GATE_DECISION state=ACTIVE output=ALLOW_ACTIVE intent=INTENT_ACTIVATE basis={...}
```

### 7.3 Reason Tokens (v0.2 additions)

| Token | Trigger |
|-------|---------|
| `intent_activate_accepted` | `INTENT_ACTIVATE` accepted from `ARMED` |
| `intent_activate_rejected` | `INTENT_ACTIVATE` rejected (not in `ARMED` or conditions not met) |
| `intent_hold_accepted` | `INTENT_HOLD` accepted from `ARMED` or `ACTIVE` |
| `intent_hold_rejected` | `INTENT_HOLD` rejected (not in `ARMED` or `ACTIVE`) |
| `intent_release` | `INTENT_RELEASE` received |
| `no_intent` | No intent provided, `ACTIVE` not reachable |

---

## 8. Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coherence_threshold` | `float` | 0.6 | Min coherence to stay active |
| `stale_data_threshold_ms` | `int` | 5000 | Max data age before stale |
| `arm_coherence_min` | `float` | 0.4 | Min coherence to arm |
| `activation_coherence_min` | `float` | 0.7 | Min coherence to activate |
| `fallback_always_allowed` | `bool` | True | Fallback is always reachable |
| `require_intent_for_active` | `bool` | True | Without intent, no ACTIVE |

---

## 9. Module Location

| File | Purpose |
|------|---------|
| `sym_cycles/action_gate_v0_2.py` | Gate state machine v0.2 |
| `scripts/smoke_action_gate_v0_2.py` | Smoke test suite |
| `docs/action_gate_v0_2_execution_contract.md` | This document |

---

## 10. Contract Summary

**Action Gate v0.2:**
- Evaluates live pipeline snapshots
- Requires explicit Action Intent for ACTIVE state
- Uses execution-neutral terminology
- Defaults to safe posture (FALLBACK always reachable)
- Logs decisions with intent tracking

**Action Intent:**
- Externally provided, per tick, revocable
- Not a command, not truth, not semantic
- Does not override states or safety rules

---

*Contract version: 0.2 — SORA CodeX Contract v1.0 / D-C Contract v0.2*
