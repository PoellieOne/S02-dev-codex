# Action Gate — Changelog

---

## v0.2 (2025-12-24)

### Added

| Addition | Location |
|----------|----------|
| `ActionIntent` enum | `action_gate_v0_2.py:44-55` |
| `GateInput.action_intent` field | `action_gate_v0_2.py:98` |
| `GateInput.intent_source` field | `action_gate_v0_2.py:99` |
| `GateOutput.intent_received` field | `action_gate_v0_2.py:112` |
| `GateOutput.intent_accepted` field | `action_gate_v0_2.py:113` |
| `GateConfig.require_intent_for_active` parameter | `action_gate_v0_2.py:126` |
| `_check_intent_for_activation()` method | `action_gate_v0_2.py:225-252` |

### New Log Events

| Event | Fields |
|-------|--------|
| `ACTION_INTENT` | `value`, `source` |

### Modified Log Events

| Event | Change |
|-------|--------|
| `GATE_DECISION` | Added `intent` and `basis` fields |

### New Reason Tokens

| Token | Description |
|-------|-------------|
| `intent_activate_accepted` | INTENT_ACTIVATE accepted |
| `intent_activate_rejected` | INTENT_ACTIVATE rejected |
| `intent_hold_accepted` | INTENT_HOLD accepted |
| `intent_hold_rejected` | INTENT_HOLD rejected |
| `intent_release` | INTENT_RELEASE received |
| `no_intent` | No intent provided |

### Behavioral Changes

| Change | Description |
|--------|-------------|
| ACTIVE requires intent | Without Action Intent, ACTIVE state not reachable |
| INTENT_RELEASE | Always forces FALLBACK |

### No Change

| Component | Status |
|-----------|--------|
| Gate states | Unchanged |
| Existing execution context inputs | Unchanged |
| Safety rules | Unchanged |
| Fallback dominance | Unchanged |

### Semantic Impact

None. No semantic layers modified.

---

## v0.1 (2025-12-19)

### Initial Release

| Component | Description |
|-----------|-------------|
| `ActionGateV0_1` | Deterministic execution gate state machine |
| `GateState` | IDLE, OBSERVE, ARMED, ACTIVE, FALLBACK |
| `GateInput` | Execution context inputs |
| `GateOutput` | Decision outputs |
| `GateConfig` | Configuration parameters |

### Log Events

| Event | Description |
|-------|-------------|
| `GATE_ENTER` | State transition |
| `GATE_BASIS` | Execution context |
| `GATE_DECISION` | Decision output |
| `GATE_FALLBACK` | Fallback trigger |

---

*Changelog — SORA CodeX Contract v1.0*
