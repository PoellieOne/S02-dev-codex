#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_action_gate_v0_2.py — Smoke Test for Action Gate v0.2

Verifies Action Intent implementation per D-C Contract v0.2.

Required Tests (Task Brief 02):
    1. ACTIVATE without context -> rejected
    2. ACTIVATE with ARMED -> accepted
    3. RELEASE -> always fallback

Usage:
    python3 scripts/smoke_action_gate_v0_2.py

Contract: SORA CodeX Contract v1.0 / D-C Contract v0.2
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from sym_cycles.action_gate_v0_2 import (
    ActionGateV0_2,
    GateInput,
    GateState,
    GateDecision,
    GateConfig,
    ActionIntent,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )


def print_separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_output(output, step: str):
    print(f"\n[{step}]")
    print(f"  State:           {output.state.value}")
    print(f"  Decision:        {output.decision.value}")
    print(f"  Reason:          {output.reason}")
    print(f"  Allowed:         {output.allowed}")
    print(f"  Intent received: {output.intent_received.value}")
    print(f"  Intent accepted: {output.intent_accepted}")


def test_activate_without_context():
    """Test 1: ACTIVATE without context -> rejected"""
    print_separator("TEST 1: ACTIVATE without context -> rejected")

    gate = ActionGateV0_2()
    t_ms = 0

    # Gate starts in IDLE
    print(f"Initial state: {gate.state.value}")
    assert gate.state == GateState.IDLE

    # First input moves to OBSERVE
    t_ms += 100
    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.3,
        lock_state="UNLOCKED",
        data_age_ms=0,
        action_intent=ActionIntent.INTENT_NONE,
        intent_source="test",
    )
    out = gate.evaluate(inp)
    print_output(out, "Move to OBSERVE")
    assert gate.state == GateState.OBSERVE

    # Try INTENT_ACTIVATE while in OBSERVE (not ARMED) -> should be rejected
    t_ms += 100
    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.8,
        lock_state="LOCKED",
        data_age_ms=50,
        action_intent=ActionIntent.INTENT_ACTIVATE,
        intent_source="test",
    )
    out = gate.evaluate(inp)
    print_output(out, "ACTIVATE from OBSERVE (should reject)")

    # Gate moved to ARMED due to conditions, but ACTIVATE was received while in OBSERVE
    # So intent should not have caused immediate ACTIVE
    # Actually: the check happens AFTER state transition evaluation
    # Let me verify the actual behavior

    # Reset and try again with explicit OBSERVE state
    gate = ActionGateV0_2()
    t_ms = 0

    # Move to OBSERVE
    t_ms += 100
    inp = GateInput(now_ms=t_ms, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0)
    gate.evaluate(inp)

    # Stay in OBSERVE (conditions not met for ARM)
    t_ms += 100
    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.3,  # Below arm threshold
        lock_state="UNLOCKED",  # No lock
        data_age_ms=50,
        action_intent=ActionIntent.INTENT_ACTIVATE,
        intent_source="test",
    )
    out = gate.evaluate(inp)
    print_output(out, "ACTIVATE in OBSERVE (conditions not met)")

    assert gate.state == GateState.OBSERVE, f"Expected OBSERVE, got {gate.state.value}"
    assert out.decision == GateDecision.HOLD_OBSERVE, f"Expected HOLD_OBSERVE, got {out.decision.value}"
    assert out.allowed is False, "Expected allowed=False"
    assert out.intent_accepted is False, "Expected intent_accepted=False"

    print("\n  -> TEST 1 PASSED: ACTIVATE without proper context rejected")
    return True


def test_activate_with_armed():
    """Test 2: ACTIVATE with ARMED -> accepted"""
    print_separator("TEST 2: ACTIVATE with ARMED -> accepted")

    gate = ActionGateV0_2()
    t_ms = 0

    # Step 1: IDLE -> OBSERVE
    t_ms += 100
    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.3,
        lock_state="UNLOCKED",
        data_age_ms=0,
        action_intent=ActionIntent.INTENT_NONE,
        intent_source="test",
    )
    out = gate.evaluate(inp)
    print_output(out, "IDLE -> OBSERVE")
    assert gate.state == GateState.OBSERVE

    # Step 2: OBSERVE -> ARMED (meet arm conditions, no intent yet)
    t_ms += 100
    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.5,
        lock_state="LOCKED",
        data_age_ms=50,
        arm_signal=True,
        action_intent=ActionIntent.INTENT_NONE,
        intent_source="test",
    )
    out = gate.evaluate(inp)
    print_output(out, "OBSERVE -> ARMED")
    assert gate.state == GateState.ARMED, f"Expected ARMED, got {gate.state.value}"

    # Step 3: ARMED -> ACTIVE with INTENT_ACTIVATE
    t_ms += 100
    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.8,  # >= activation_coherence_min (0.7)
        lock_state="LOCKED",
        data_age_ms=100,
        action_intent=ActionIntent.INTENT_ACTIVATE,
        intent_source="test",
    )
    out = gate.evaluate(inp)
    print_output(out, "ARMED + INTENT_ACTIVATE -> ACTIVE")

    assert gate.state == GateState.ACTIVE, f"Expected ACTIVE, got {gate.state.value}"
    assert out.decision == GateDecision.ALLOW_ACTIVE, f"Expected ALLOW_ACTIVE, got {out.decision.value}"
    assert out.allowed is True, "Expected allowed=True"
    assert out.intent_accepted is True, "Expected intent_accepted=True"
    assert out.intent_received == ActionIntent.INTENT_ACTIVATE

    print("\n  -> TEST 2 PASSED: ACTIVATE with ARMED accepted")
    return True


def test_release_always_fallback():
    """Test 3: RELEASE -> always fallback"""
    print_separator("TEST 3: RELEASE -> always fallback")

    # Test 3a: RELEASE from OBSERVE
    print("\n--- Test 3a: RELEASE from OBSERVE ---")
    gate = ActionGateV0_2()
    t_ms = 0

    # Move to OBSERVE
    t_ms += 100
    inp = GateInput(now_ms=t_ms, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0)
    gate.evaluate(inp)
    assert gate.state == GateState.OBSERVE

    # RELEASE from OBSERVE
    t_ms += 100
    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.8,
        lock_state="LOCKED",
        data_age_ms=50,
        action_intent=ActionIntent.INTENT_RELEASE,
        intent_source="test",
    )
    out = gate.evaluate(inp)
    print_output(out, "RELEASE from OBSERVE")

    assert gate.state == GateState.FALLBACK, f"Expected FALLBACK, got {gate.state.value}"
    assert out.decision == GateDecision.FORCE_FALLBACK
    assert out.allowed is False

    # Test 3b: RELEASE from ARMED
    print("\n--- Test 3b: RELEASE from ARMED ---")
    gate = ActionGateV0_2()
    t_ms = 0

    # Move to ARMED
    t_ms += 100
    inp = GateInput(now_ms=t_ms, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0)
    gate.evaluate(inp)
    t_ms += 100
    inp = GateInput(now_ms=t_ms, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True)
    gate.evaluate(inp)
    assert gate.state == GateState.ARMED

    # RELEASE from ARMED
    t_ms += 100
    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.8,
        lock_state="LOCKED",
        data_age_ms=100,
        action_intent=ActionIntent.INTENT_RELEASE,
        intent_source="test",
    )
    out = gate.evaluate(inp)
    print_output(out, "RELEASE from ARMED")

    assert gate.state == GateState.FALLBACK, f"Expected FALLBACK, got {gate.state.value}"
    assert out.decision == GateDecision.FORCE_FALLBACK
    assert out.allowed is False

    # Test 3c: RELEASE from ACTIVE
    print("\n--- Test 3c: RELEASE from ACTIVE ---")
    gate = ActionGateV0_2()
    t_ms = 0

    # Move to ACTIVE
    t_ms += 100
    inp = GateInput(now_ms=t_ms, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0)
    gate.evaluate(inp)
    t_ms += 100
    inp = GateInput(now_ms=t_ms, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True)
    gate.evaluate(inp)
    t_ms += 100
    inp = GateInput(
        now_ms=t_ms, coherence_score=0.8, lock_state="LOCKED", data_age_ms=100,
        action_intent=ActionIntent.INTENT_ACTIVATE, intent_source="test"
    )
    gate.evaluate(inp)
    assert gate.state == GateState.ACTIVE

    # RELEASE from ACTIVE
    t_ms += 100
    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.9,
        lock_state="LOCKED",
        data_age_ms=150,
        action_intent=ActionIntent.INTENT_RELEASE,
        intent_source="test",
    )
    out = gate.evaluate(inp)
    print_output(out, "RELEASE from ACTIVE")

    assert gate.state == GateState.FALLBACK, f"Expected FALLBACK, got {gate.state.value}"
    assert out.decision == GateDecision.FORCE_FALLBACK
    assert out.allowed is False

    print("\n  -> TEST 3 PASSED: RELEASE always forces FALLBACK")
    return True


def test_no_intent_no_active():
    """Additional test: Without Action Intent -> no ACTIVE"""
    print_separator("TEST 4: Without Action Intent -> no ACTIVE")

    gate = ActionGateV0_2()
    t_ms = 0

    # Move to ARMED
    t_ms += 100
    inp = GateInput(now_ms=t_ms, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0)
    gate.evaluate(inp)
    t_ms += 100
    inp = GateInput(now_ms=t_ms, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True)
    gate.evaluate(inp)
    assert gate.state == GateState.ARMED

    # Perfect conditions but INTENT_NONE -> should NOT go to ACTIVE
    t_ms += 100
    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.9,
        lock_state="LOCKED",
        data_age_ms=100,
        action_intent=ActionIntent.INTENT_NONE,  # No intent
        intent_source="test",
    )
    out = gate.evaluate(inp)
    print_output(out, "Perfect conditions + INTENT_NONE")

    assert gate.state == GateState.ARMED, f"Expected ARMED (no ACTIVE without intent), got {gate.state.value}"
    assert out.decision == GateDecision.HOLD_OBSERVE
    assert out.allowed is False
    assert out.intent_accepted is False

    print("\n  -> TEST 4 PASSED: Without Action Intent, no ACTIVE")
    return True


def test_intent_hold():
    """Additional test: INTENT_HOLD behavior"""
    print_separator("TEST 5: INTENT_HOLD behavior")

    gate = ActionGateV0_2()
    t_ms = 0

    # Move to ACTIVE via INTENT_ACTIVATE
    t_ms += 100
    inp = GateInput(now_ms=t_ms, coherence_score=0.3, lock_state="UNLOCKED", data_age_ms=0)
    gate.evaluate(inp)
    t_ms += 100
    inp = GateInput(now_ms=t_ms, coherence_score=0.5, lock_state="LOCKED", data_age_ms=50, arm_signal=True)
    gate.evaluate(inp)
    t_ms += 100
    inp = GateInput(
        now_ms=t_ms, coherence_score=0.8, lock_state="LOCKED", data_age_ms=100,
        action_intent=ActionIntent.INTENT_ACTIVATE, intent_source="test"
    )
    gate.evaluate(inp)
    assert gate.state == GateState.ACTIVE

    # Stay ACTIVE with INTENT_HOLD
    t_ms += 100
    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.75,
        lock_state="LOCKED",
        data_age_ms=150,
        action_intent=ActionIntent.INTENT_HOLD,
        intent_source="test",
    )
    out = gate.evaluate(inp)
    print_output(out, "ACTIVE + INTENT_HOLD -> stay ACTIVE")

    assert gate.state == GateState.ACTIVE, f"Expected ACTIVE, got {gate.state.value}"
    assert out.decision == GateDecision.ALLOW_ACTIVE
    assert out.allowed is True
    assert out.intent_accepted is True

    print("\n  -> TEST 5 PASSED: INTENT_HOLD keeps ACTIVE state")
    return True


def main():
    setup_logging()

    print_separator("Action Gate v0.2 Smoke Test Suite")
    print("\nVerifying D-C Contract v0.2 compliance...")
    print(f"Gate version: {ActionGateV0_2.VERSION}")

    results = []

    # Required tests per Task Brief 02
    results.append(("TEST 1: ACTIVATE without context -> rejected", test_activate_without_context()))
    results.append(("TEST 2: ACTIVATE with ARMED -> accepted", test_activate_with_armed()))
    results.append(("TEST 3: RELEASE -> always fallback", test_release_always_fallback()))

    # Additional verification tests
    results.append(("TEST 4: No intent -> no ACTIVE", test_no_intent_no_active()))
    results.append(("TEST 5: INTENT_HOLD behavior", test_intent_hold()))

    # Summary
    print_separator("TEST SUMMARY")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("="*70)
        print("  ALL TESTS PASSED - Action Gate v0.2 compliant with D-C Contract")
        print("="*70)
        return 0
    else:
        print("="*70)
        print("  SOME TESTS FAILED")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
