# Action Gate v0.2 — Integration Note

**Version:** 0.2
**Date:** 2025-12-24
**Scope:** PC-side integration

---

## 1. Action Intent Delivery Point

### 1.1 Location

| Component | File | Entry Point |
|-----------|------|-------------|
| `GateInput` | `sym_cycles/action_gate_v0_2.py` | `action_intent` parameter |
| `GateInput` | `sym_cycles/action_gate_v0_2.py` | `intent_source` parameter |

### 1.2 Input Construction

```python
from sym_cycles.action_gate_v0_2 import GateInput, ActionIntent

gate_input = GateInput(
    now_ms=<timestamp>,
    coherence_score=<float>,
    lock_state=<str>,
    data_age_ms=<int>,
    rotor_active=<bool>,
    action_intent=<ActionIntent>,    # <-- External delivery
    intent_source=<str>,             # <-- Source identifier
)
```

---

## 2. Default Behavior

| Condition | Default Value |
|-----------|---------------|
| `action_intent` not provided | `ActionIntent.INTENT_NONE` |
| `intent_source` not provided | `"unknown"` |

---

## 3. External Requirement

Action Intent must be externally provided:

| Requirement | Description |
|-------------|-------------|
| Source | External to pipeline (keyboard, API, other) |
| Derivation | Not derived from pipeline state |
| Persistence | Must be re-sent each tick |

---

## 4. Live Pipeline Integration

### 4.1 Current State

| Component | Status |
|-----------|--------|
| `action_gate_v0_2.py` | Implemented |
| `live_symphonia_v2_0.py` | v0.1 integrated, v0.2 not yet |

### 4.2 Integration Point

| Location | File | Line Reference |
|----------|------|----------------|
| Gate evaluation hook | `scripts/live_symphonia_v2_0.py` | After `l1.update()` |

### 4.3 Required Changes for v0.2

Replace v0.1 import and instantiation with v0.2:

| Current (v0.1) | Required (v0.2) |
|----------------|-----------------|
| `from sym_cycles.action_gate_v0_1 import ...` | `from sym_cycles.action_gate_v0_2 import ...` |
| `ActionGateV0_1()` | `ActionGateV0_2()` |

Add external intent delivery mechanism.

---

## 5. Module Dependencies

| Module | Dependency |
|--------|------------|
| `action_gate_v0_2.py` | None (standalone) |
| `live_symphonia_v2_0.py` | `action_gate_v0_2.py` (optional) |

---

*Integration Note v0.2 — SORA CodeX Contract v1.0*
