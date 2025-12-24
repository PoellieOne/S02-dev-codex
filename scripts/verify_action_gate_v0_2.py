#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_action_gate_v0_2.py — Verification & Regression Tests

Task Brief 04: Verify Action Gate v0.2 against v0.1 behavior.

Tests:
    A. Regression: v0.2 with INTENT_NONE == v0.1 behavior
    B. Intent stability: repeated scenarios, revocability
    C. Logging consistency: complete, no semantic terms

Contract: SORA CodeX Contract v1.0 / D-C Contract v0.2
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sym_cycles.action_gate_v0_1 import (
    ActionGateV0_1,
    GateInput as GateInputV1,
    GateState,
    GateDecision,
)
from sym_cycles.action_gate_v0_2 import (
    ActionGateV0_2,
    GateInput as GateInputV2,
    ActionIntent,
)


def print_separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_result(name: str, passed: bool, details: str = ""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if details:
        print(f"         {details}")


# =============================================================================
# A. REGRESSION TESTS
# =============================================================================

def test_regression_idle_to_observe():
    """v0.2 INTENT_NONE: IDLE -> OBSERVE identical to v0.1"""
    gate_v1 = ActionGateV0_1()
    gate_v2 = ActionGateV0_2()

    inp_v1 = GateInputV1(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0)
    inp_v2 = GateInputV2(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0,
                         action_intent=ActionIntent.INTENT_NONE)

    out_v1 = gate_v1.evaluate(inp_v1)
    out_v2 = gate_v2.evaluate(inp_v2)

    # Compare by .value to avoid enum instance mismatch
    match = (
        gate_v1.state.value == gate_v2.state.value and
        out_v1.decision.value == out_v2.decision.value and
        out_v1.allowed == out_v2.allowed
    )
    return match, f"v1={gate_v1.state.value}/{out_v1.decision.value} v2={gate_v2.state.value}/{out_v2.decision.value}"


def test_regression_observe_to_armed():
    """v0.2 INTENT_NONE: OBSERVE -> ARMED identical to v0.1"""
    gate_v1 = ActionGateV0_1()
    gate_v2 = ActionGateV0_2()

    # Move to OBSERVE
    gate_v1.evaluate(GateInputV1(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0))
    gate_v2.evaluate(GateInputV2(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0,
                                 action_intent=ActionIntent.INTENT_NONE))

    # Arm conditions
    inp_v1 = GateInputV1(now_ms=200, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True)
    inp_v2 = GateInputV2(now_ms=200, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True,
                         action_intent=ActionIntent.INTENT_NONE)

    out_v1 = gate_v1.evaluate(inp_v1)
    out_v2 = gate_v2.evaluate(inp_v2)

    # Compare by .value to avoid enum instance mismatch
    match = (
        gate_v1.state.value == gate_v2.state.value and
        out_v1.decision.value == out_v2.decision.value
    )
    return match, f"v1={gate_v1.state.value}/{out_v1.decision.value} v2={gate_v2.state.value}/{out_v2.decision.value}"


def test_regression_armed_no_active_without_intent():
    """v0.2 INTENT_NONE: ARMED does NOT go to ACTIVE (v0.1 with activate_signal would)"""
    # This is expected DIFFERENCE: v0.2 requires intent
    gate_v2 = ActionGateV0_2()

    # Move to ARMED
    gate_v2.evaluate(GateInputV2(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0))
    gate_v2.evaluate(GateInputV2(now_ms=200, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True))

    # Perfect conditions but INTENT_NONE
    inp = GateInputV2(now_ms=300, coherence_score=0.9, lock_state="LOCKED", data_age_ms=100,
                      action_intent=ActionIntent.INTENT_NONE)
    out = gate_v2.evaluate(inp)

    # v0.2 should stay ARMED (not go to ACTIVE without intent)
    correct = gate_v2.state.value == "ARMED" and out.decision.value == "HOLD_OBSERVE"
    return correct, f"state={gate_v2.state.value} decision={out.decision.value}"


def test_regression_fallback_patterns():
    """v0.2 INTENT_NONE: Fallback patterns identical to v0.1"""
    gate_v1 = ActionGateV0_1()
    gate_v2 = ActionGateV0_2()

    # Test force_fallback
    inp_v1 = GateInputV1(now_ms=100, coherence_score=0.8, lock_state="LOCKED", data_age_ms=0, force_fallback=True)
    inp_v2 = GateInputV2(now_ms=100, coherence_score=0.8, lock_state="LOCKED", data_age_ms=0, force_fallback=True,
                         action_intent=ActionIntent.INTENT_NONE)

    out_v1 = gate_v1.evaluate(inp_v1)
    out_v2 = gate_v2.evaluate(inp_v2)

    match_force = (
        gate_v1.state.value == gate_v2.state.value == "FALLBACK" and
        out_v1.decision.value == out_v2.decision.value == "FORCE_FALLBACK"
    )

    # Test stale data
    gate_v1 = ActionGateV0_1()
    gate_v2 = ActionGateV0_2()

    inp_v1 = GateInputV1(now_ms=100, coherence_score=0.8, lock_state="LOCKED", data_age_ms=6000)
    inp_v2 = GateInputV2(now_ms=100, coherence_score=0.8, lock_state="LOCKED", data_age_ms=6000,
                         action_intent=ActionIntent.INTENT_NONE)

    out_v1 = gate_v1.evaluate(inp_v1)
    out_v2 = gate_v2.evaluate(inp_v2)

    match_stale = (
        gate_v1.state.value == gate_v2.state.value == "FALLBACK" and
        out_v1.decision.value == out_v2.decision.value == "FORCE_FALLBACK"
    )

    return match_force and match_stale, f"force_fb={match_force} stale={match_stale}"


def test_regression_no_extra_active_transitions():
    """v0.2 INTENT_NONE: No extra ACTIVE transitions compared to v0.1 baseline"""
    # Run identical sequence through both
    gate_v1 = ActionGateV0_1()
    gate_v2 = ActionGateV0_2()

    sequence = [
        {"coherence": 0.3, "lock": "UNLOCKED", "data_age": 0},
        {"coherence": 0.5, "lock": "SOFT_LOCK", "data_age": 50},
        {"coherence": 0.6, "lock": "LOCKED", "data_age": 100},
        {"coherence": 0.8, "lock": "LOCKED", "data_age": 150},
        {"coherence": 0.4, "lock": "UNLOCKED", "data_age": 200},
        {"coherence": 0.7, "lock": "LOCKED", "data_age": 250},
    ]

    v1_active_count = 0
    v2_active_count = 0

    for i, s in enumerate(sequence):
        t = (i + 1) * 100
        inp_v1 = GateInputV1(now_ms=t, coherence_score=s["coherence"], lock_state=s["lock"],
                             data_age_ms=s["data_age"], arm_signal=(s["lock"] != "UNLOCKED"))
        inp_v2 = GateInputV2(now_ms=t, coherence_score=s["coherence"], lock_state=s["lock"],
                             data_age_ms=s["data_age"], arm_signal=(s["lock"] != "UNLOCKED"),
                             action_intent=ActionIntent.INTENT_NONE)

        out_v1 = gate_v1.evaluate(inp_v1)
        out_v2 = gate_v2.evaluate(inp_v2)

        if out_v1.decision.value == "ALLOW_ACTIVE":
            v1_active_count += 1
        if out_v2.decision.value == "ALLOW_ACTIVE":
            v2_active_count += 1

    # v0.2 should have NO active transitions without intent
    # v0.1 may have some if activate_signal was implicit
    no_extra = v2_active_count == 0
    return no_extra, f"v1_active={v1_active_count} v2_active={v2_active_count}"


# =============================================================================
# B. INTENT STABILITY TESTS
# =============================================================================

def test_intent_repeated_activate():
    """INTENT_ACTIVATE from ARMED transitions to ACTIVE"""
    gate = ActionGateV0_2()

    # Setup to ARMED
    gate.evaluate(GateInputV2(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0))
    gate.evaluate(GateInputV2(now_ms=200, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True))
    assert gate.state.value == "ARMED"

    # First ACTIVATE from ARMED -> should go to ACTIVE
    out1 = gate.evaluate(GateInputV2(now_ms=300, coherence_score=0.8, lock_state="LOCKED", data_age_ms=100,
                                     action_intent=ActionIntent.INTENT_ACTIVATE))
    first_active = out1.decision.value == "ALLOW_ACTIVE" and gate.state.value == "ACTIVE"

    # Per contract: INTENT_ACTIVATE only allowed from ARMED
    # From ACTIVE, must use INTENT_HOLD to maintain
    # Test that ACTIVATE from ACTIVE causes transition to OBSERVE (correct behavior)
    out2 = gate.evaluate(GateInputV2(now_ms=400, coherence_score=0.8, lock_state="LOCKED", data_age_ms=150,
                                     action_intent=ActionIntent.INTENT_ACTIVATE))
    second_rejected = out2.decision.value == "HOLD_OBSERVE"

    return first_active and second_rejected, f"first_active={first_active} second_rejected={second_rejected}"


def test_intent_repeated_hold():
    """Repeated INTENT_HOLD maintains ACTIVE"""
    gate = ActionGateV0_2()

    # Setup to ACTIVE
    gate.evaluate(GateInputV2(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0))
    gate.evaluate(GateInputV2(now_ms=200, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True))
    gate.evaluate(GateInputV2(now_ms=300, coherence_score=0.8, lock_state="LOCKED", data_age_ms=100,
                              action_intent=ActionIntent.INTENT_ACTIVATE))

    # Repeated HOLD
    hold_active_count = 0
    for i in range(5):
        t = 400 + i * 100
        inp = GateInputV2(now_ms=t, coherence_score=0.75, lock_state="LOCKED", data_age_ms=150 + i*50,
                          action_intent=ActionIntent.INTENT_HOLD)
        out = gate.evaluate(inp)
        if out.decision.value == "ALLOW_ACTIVE":
            hold_active_count += 1

    return hold_active_count == 5, f"hold_active_count={hold_active_count}/5"


def test_intent_revocability():
    """ACTIVATE -> NONE revokes active state"""
    gate = ActionGateV0_2()

    # Setup to ACTIVE
    gate.evaluate(GateInputV2(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0))
    gate.evaluate(GateInputV2(now_ms=200, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True))
    out1 = gate.evaluate(GateInputV2(now_ms=300, coherence_score=0.8, lock_state="LOCKED", data_age_ms=100,
                                     action_intent=ActionIntent.INTENT_ACTIVATE))

    was_active = out1.decision.value == "ALLOW_ACTIVE"

    # Revoke with NONE
    out2 = gate.evaluate(GateInputV2(now_ms=400, coherence_score=0.8, lock_state="LOCKED", data_age_ms=150,
                                     action_intent=ActionIntent.INTENT_NONE))

    revoked = out2.decision.value != "ALLOW_ACTIVE" and gate.state.value == "OBSERVE"

    return was_active and revoked, f"was_active={was_active} revoked={revoked} state={gate.state.value}"


def test_intent_release_dominant():
    """RELEASE is always dominant"""
    results = []

    # Test from OBSERVE
    gate = ActionGateV0_2()
    gate.evaluate(GateInputV2(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0))
    out = gate.evaluate(GateInputV2(now_ms=200, coherence_score=0.8, lock_state="LOCKED", data_age_ms=50,
                                    action_intent=ActionIntent.INTENT_RELEASE))
    results.append(("OBSERVE", gate.state.value == "FALLBACK"))

    # Test from ARMED
    gate = ActionGateV0_2()
    gate.evaluate(GateInputV2(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0))
    gate.evaluate(GateInputV2(now_ms=200, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True))
    out = gate.evaluate(GateInputV2(now_ms=300, coherence_score=0.8, lock_state="LOCKED", data_age_ms=100,
                                    action_intent=ActionIntent.INTENT_RELEASE))
    results.append(("ARMED", gate.state.value == "FALLBACK"))

    # Test from ACTIVE
    gate = ActionGateV0_2()
    gate.evaluate(GateInputV2(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0))
    gate.evaluate(GateInputV2(now_ms=200, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True))
    gate.evaluate(GateInputV2(now_ms=300, coherence_score=0.8, lock_state="LOCKED", data_age_ms=100,
                              action_intent=ActionIntent.INTENT_ACTIVATE))
    out = gate.evaluate(GateInputV2(now_ms=400, coherence_score=0.9, lock_state="LOCKED", data_age_ms=150,
                                    action_intent=ActionIntent.INTENT_RELEASE))
    results.append(("ACTIVE", gate.state.value == "FALLBACK"))

    all_pass = all(r[1] for r in results)
    details = " ".join([f"{r[0]}={'OK' if r[1] else 'FAIL'}" for r in results])
    return all_pass, details


def test_intent_alternating():
    """Alternating ACTIVATE/HOLD/NONE"""
    gate = ActionGateV0_2()

    # Setup to ARMED
    gate.evaluate(GateInputV2(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0))
    gate.evaluate(GateInputV2(now_ms=200, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True))

    sequence = [
        (ActionIntent.INTENT_ACTIVATE, "ALLOW_ACTIVE"),  # ARMED -> ACTIVE
        (ActionIntent.INTENT_HOLD, "ALLOW_ACTIVE"),      # Stay ACTIVE
        (ActionIntent.INTENT_NONE, "HOLD_OBSERVE"),      # ACTIVE -> OBSERVE
    ]

    # After OBSERVE, need to re-arm
    results = []
    t = 300
    for intent, expected in sequence:
        inp = GateInputV2(now_ms=t, coherence_score=0.8, lock_state="LOCKED", data_age_ms=100,
                          action_intent=intent, arm_signal=True)
        out = gate.evaluate(inp)
        results.append(out.decision.value == expected)
        t += 100

    all_pass = all(results)
    return all_pass, f"sequence_results={results}"


# =============================================================================
# C. LOGGING CONSISTENCY TESTS
# =============================================================================

def test_logging_action_intent_present():
    """ACTION_INTENT log event present"""
    gate = ActionGateV0_2()
    inp = GateInputV2(now_ms=100, coherence_score=0.5, lock_state="LOCKED", data_age_ms=0,
                      action_intent=ActionIntent.INTENT_ACTIVATE, intent_source="test_source")
    out = gate.evaluate(inp)

    has_action_intent = any("ACTION_INTENT" in entry for entry in out.log_entries)
    has_value = any("value=INTENT_ACTIVATE" in entry for entry in out.log_entries)
    has_source = any("source=test_source" in entry for entry in out.log_entries)

    return has_action_intent and has_value and has_source, f"entries={len(out.log_entries)}"


def test_logging_gate_decision_has_intent():
    """GATE_DECISION includes intent field"""
    gate = ActionGateV0_2()
    gate.evaluate(GateInputV2(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0))
    gate.evaluate(GateInputV2(now_ms=200, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True))
    out = gate.evaluate(GateInputV2(now_ms=300, coherence_score=0.8, lock_state="LOCKED", data_age_ms=100,
                                    action_intent=ActionIntent.INTENT_ACTIVATE))

    decision_entries = [e for e in out.log_entries if "GATE_DECISION" in e]
    has_intent = any("intent=" in e for e in decision_entries)
    has_basis = any("basis=" in e for e in decision_entries)

    return has_intent and has_basis, f"decision_entries={len(decision_entries)}"


def test_logging_no_semantic_terms():
    """Logs contain no semantic terms"""
    gate = ActionGateV0_2()

    # Run full sequence
    gate.evaluate(GateInputV2(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0))
    gate.evaluate(GateInputV2(now_ms=200, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True))
    out = gate.evaluate(GateInputV2(now_ms=300, coherence_score=0.8, lock_state="LOCKED", data_age_ms=100,
                                    action_intent=ActionIntent.INTENT_ACTIVATE))

    # Forbidden terms
    forbidden = ["truth", "belief", "desire", "want", "feel", "think", "meaning", "semantic"]

    all_entries = " ".join(out.log_entries).lower()
    found_forbidden = [term for term in forbidden if term in all_entries]

    return len(found_forbidden) == 0, f"forbidden_found={found_forbidden}"


def test_logging_complete_fields():
    """All required log fields present"""
    gate = ActionGateV0_2()

    gate.evaluate(GateInputV2(now_ms=100, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0))
    gate.evaluate(GateInputV2(now_ms=200, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True))
    out = gate.evaluate(GateInputV2(now_ms=300, coherence_score=0.8, lock_state="LOCKED", data_age_ms=100,
                                    action_intent=ActionIntent.INTENT_ACTIVATE))

    # Check required events
    all_text = " ".join(out.log_entries)
    has_action_intent = "ACTION_INTENT" in all_text
    has_gate_basis = "GATE_BASIS" in all_text
    has_gate_enter = "GATE_ENTER" in all_text
    has_gate_decision = "GATE_DECISION" in all_text

    all_present = has_action_intent and has_gate_basis and has_gate_enter and has_gate_decision
    return all_present, f"AI={has_action_intent} GB={has_gate_basis} GE={has_gate_enter} GD={has_gate_decision}"


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_separator("Action Gate v0.2 — Verification & Regression Tests")
    print(f"\nTask Brief 04: Verification / Regression")
    print(f"Date: 2025-12-24")

    results = []

    # A. REGRESSION TESTS
    print_separator("A. REGRESSION TESTS")
    print("\nComparing v0.2 (INTENT_NONE) against v0.1 behavior:\n")

    tests_a = [
        ("IDLE -> OBSERVE identical", test_regression_idle_to_observe),
        ("OBSERVE -> ARMED identical", test_regression_observe_to_armed),
        ("ARMED stays ARMED without intent", test_regression_armed_no_active_without_intent),
        ("Fallback patterns identical", test_regression_fallback_patterns),
        ("No extra ACTIVE transitions", test_regression_no_extra_active_transitions),
    ]

    for name, test_fn in tests_a:
        passed, details = test_fn()
        print_result(name, passed, details)
        results.append((f"A: {name}", passed))

    # B. INTENT STABILITY TESTS
    print_separator("B. INTENT STABILITY TESTS")
    print("\nTesting repeated scenarios and revocability:\n")

    tests_b = [
        ("Repeated ACTIVATE over ticks", test_intent_repeated_activate),
        ("Repeated HOLD maintains ACTIVE", test_intent_repeated_hold),
        ("ACTIVATE -> NONE revokes", test_intent_revocability),
        ("RELEASE dominant from all states", test_intent_release_dominant),
        ("Alternating ACTIVATE/HOLD/NONE", test_intent_alternating),
    ]

    for name, test_fn in tests_b:
        passed, details = test_fn()
        print_result(name, passed, details)
        results.append((f"B: {name}", passed))

    # C. LOGGING CONSISTENCY TESTS
    print_separator("C. LOGGING CONSISTENCY TESTS")
    print("\nVerifying log completeness and correctness:\n")

    tests_c = [
        ("ACTION_INTENT event present", test_logging_action_intent_present),
        ("GATE_DECISION has intent field", test_logging_gate_decision_has_intent),
        ("No semantic terms in logs", test_logging_no_semantic_terms),
        ("All required log fields present", test_logging_complete_fields),
    ]

    for name, test_fn in tests_c:
        passed, details = test_fn()
        print_result(name, passed, details)
        results.append((f"C: {name}", passed))

    # SUMMARY
    print_separator("VERIFICATION SUMMARY")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    failed = total - passed

    print(f"\n  Total tests: {total}")
    print(f"  Passed:      {passed}")
    print(f"  Failed:      {failed}")

    if failed > 0:
        print("\n  Failed tests:")
        for name, p in results:
            if not p:
                print(f"    - {name}")

    print()
    if failed == 0:
        print("="*70)
        print("  GEEN REGRESSIES GEVONDEN")
        print("  All verification scenarios passed.")
        print("="*70)
        return 0
    else:
        print("="*70)
        print("  WEL REGRESSIES GEVONDEN")
        print(f"  {failed} test(s) failed.")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
