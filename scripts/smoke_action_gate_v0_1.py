#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_action_gate_v0_1.py — Smoke Test for Action Gate v0.1

Standalone test script to verify gate transitions without pipeline integration.
Demonstrates:
  - IDLE → OBSERVE → ARMED → ACTIVE transitions
  - ACTIVE → FALLBACK transition

Usage:
    python3 scripts/smoke_action_gate_v0_1.py

Expected output:
    GATE_ENTER, GATE_DECISION, GATE_BASIS, GATE_FALLBACK log entries
    with execution-neutral reason tokens.

Contract: SORA CodeX Contract v1.0
"""

import sys
import os

# Add parent directory to path for import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from sym_cycles.action_gate_v0_1 import (
    ActionGateV0_1,
    GateInput,
    GateState,
    GateDecision,
    GateConfig,
)


def setup_logging():
    """Configure logging for visibility."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )


def print_separator(title: str):
    """Print a section separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_output(output, step: str):
    """Print gate output summary."""
    print(f"\n[{step}]")
    print(f"  State:    {output.state.value}")
    print(f"  Decision: {output.decision.value}")
    print(f"  Reason:   {output.reason}")
    print(f"  Allowed:  {output.allowed}")


def main():
    """Run smoke test demonstrating gate transitions."""
    setup_logging()

    print_separator("Action Gate v0.1 Smoke Test")
    print("\nInitializing gate...")

    # Create gate with default config
    gate = ActionGateV0_1()
    print(f"Initial state: {gate.state.value}")

    # Simulated time (deterministic, external)
    t_ms = 0

    # === Step 1: IDLE → OBSERVE ===
    print_separator("Step 1: IDLE -> OBSERVE (first input)")
    t_ms += 100

    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.3,
        lock_state="UNLOCKED",
        data_age_ms=0,
    )
    out = gate.evaluate(inp)
    print_output(out, "First input received")
    assert gate.state == GateState.OBSERVE, f"Expected OBSERVE, got {gate.state}"

    # === Step 2: OBSERVE → ARMED ===
    print_separator("Step 2: OBSERVE -> ARMED (coherence + lock)")
    t_ms += 100

    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.5,
        lock_state="LOCKED",
        data_age_ms=50,
        arm_signal=True,
    )
    out = gate.evaluate(inp)
    print_output(out, "Arm conditions met")
    assert gate.state == GateState.ARMED, f"Expected ARMED, got {gate.state}"

    # === Step 3: ARMED → ACTIVE ===
    print_separator("Step 3: ARMED -> ACTIVE (activation trigger)")
    t_ms += 100

    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.8,
        lock_state="LOCKED",
        data_age_ms=100,
        activate_signal=True,
    )
    out = gate.evaluate(inp)
    print_output(out, "Activation triggered")
    assert gate.state == GateState.ACTIVE, f"Expected ACTIVE, got {gate.state}"
    assert out.decision == GateDecision.ALLOW_ACTIVE, "Expected ALLOW_ACTIVE"
    assert out.allowed is True, "Expected allowed=True"

    # === Step 4: ACTIVE (sustained) ===
    print_separator("Step 4: ACTIVE sustained")
    t_ms += 100

    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.75,
        lock_state="LOCKED",
        data_age_ms=150,
    )
    out = gate.evaluate(inp)
    print_output(out, "Sustained active")
    assert gate.state == GateState.ACTIVE, f"Expected ACTIVE, got {gate.state}"

    # === Step 5: ACTIVE → FALLBACK ===
    print_separator("Step 5: ACTIVE -> FALLBACK (forced fallback)")
    t_ms += 100

    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.75,
        lock_state="LOCKED",
        data_age_ms=200,
        force_fallback=True,
    )
    out = gate.evaluate(inp)
    print_output(out, "Fallback forced")
    assert gate.state == GateState.FALLBACK, f"Expected FALLBACK, got {gate.state}"
    assert out.decision == GateDecision.FORCE_FALLBACK, "Expected FORCE_FALLBACK"
    assert out.allowed is False, "Expected allowed=False"

    # === Step 6: Verify FALLBACK → IDLE recovery ===
    print_separator("Step 6: FALLBACK -> IDLE (recovery)")
    t_ms += 100

    inp = GateInput(
        now_ms=t_ms,
        coherence_score=0.8,
        lock_state="LOCKED",
        data_age_ms=50,
        force_fallback=False,
    )
    out = gate.evaluate(inp)
    print_output(out, "Recovery from fallback")
    assert gate.state == GateState.IDLE, f"Expected IDLE, got {gate.state}"

    # === Summary ===
    print_separator("Summary")
    debug = gate.get_debug_state()
    print(f"\nTotal transitions: {debug['transition_count']}")
    print(f"Final state: {debug['state']}")

    print("\n" + "="*60)
    print("  SMOKE TEST PASSED - All transitions verified")
    print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
