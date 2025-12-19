#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
action_gate_v0_1.py — Execution Gate State Machine v0.1

PC-side execution-layer module: deterministic gate between pipeline outputs
and executing actions.

Execution States (exact):
    IDLE       - No active processing, waiting for input
    OBSERVE    - Monitoring inputs, not yet armed
    ARMED      - Conditions approaching threshold, ready to activate
    ACTIVE     - Actively executing/allowing actions
    FALLBACK   - Safe mode, blocking actions, always reachable

Contract: SORA CodeX Contract v1.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
import logging


# === Execution States ===

class GateState(Enum):
    """Exact execution states for the action gate."""
    IDLE = "IDLE"
    OBSERVE = "OBSERVE"
    ARMED = "ARMED"
    ACTIVE = "ACTIVE"
    FALLBACK = "FALLBACK"


# === Reason Tokens (execution-neutral) ===

class ReasonToken:
    """Execution-neutral reason tokens for gate transitions."""
    DATA_STALE = "data_stale"
    COHERENCE_DROP = "coherence_drop"
    LOCK_LOST = "lock_lost"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    SAFETY_RESET = "safety_reset"
    ARMED_CONDITION_MET = "armed_condition_met"
    INPUT_RECEIVED = "input_received"
    THRESHOLD_REACHED = "threshold_reached"
    ACTIVATION_TRIGGERED = "activation_triggered"
    MANUAL_FALLBACK = "manual_fallback"
    TIMEOUT_EXPIRED = "timeout_expired"
    INIT_COMPLETE = "init_complete"
    OBSERVE_STARTED = "observe_started"


# === Decision Outputs ===

class GateDecision(Enum):
    """Gate decision outputs."""
    ALLOW_ACTIVE = "ALLOW_ACTIVE"
    HOLD_OBSERVE = "HOLD_OBSERVE"
    FORCE_FALLBACK = "FORCE_FALLBACK"


# === Input/Output Dataclasses ===

@dataclass
class GateInput:
    """Input data for gate evaluation."""
    now_ms: int                           # Current timestamp (external, deterministic)
    coherence_score: float = 1.0          # 0.0 - 1.0, pipeline coherence measure
    lock_state: str = "UNLOCKED"          # LOCKED / SOFT_LOCK / UNLOCKED
    data_age_ms: int = 0                  # Age of input data in ms
    rotor_active: bool = False            # Whether rotor is active
    force_fallback: bool = False          # External fallback trigger
    arm_signal: bool = False              # Signal to arm the gate
    activate_signal: bool = False         # Signal to activate
    fields: Dict[str, Any] = field(default_factory=dict)  # Additional basis fields


@dataclass
class GateOutput:
    """Output from gate evaluation."""
    state: GateState
    decision: GateDecision
    reason: str
    timestamp_ms: int
    allowed: bool
    log_entries: List[str] = field(default_factory=list)


# === Gate Configuration ===

@dataclass
class GateConfig:
    """Configuration for the action gate."""
    coherence_threshold: float = 0.6       # Min coherence to stay active
    stale_data_threshold_ms: int = 5000    # Max data age before stale
    arm_coherence_min: float = 0.4         # Min coherence to arm
    activation_coherence_min: float = 0.7  # Min coherence to activate
    fallback_always_allowed: bool = True   # Fallback is always reachable


# === Logging ===

def _format_log(event_type: str, **kwargs) -> str:
    """Format a gate log entry."""
    parts = [event_type]
    for k, v in kwargs.items():
        parts.append(f"{k}={v}")
    return " ".join(parts)


# === State Machine ===

class ActionGateV0_1:
    """
    Deterministic execution gate state machine.

    Transitions:
        IDLE     → OBSERVE    (on input received)
        OBSERVE  → ARMED      (on coherence/lock conditions met)
        OBSERVE  → FALLBACK   (on safety conditions)
        ARMED    → ACTIVE     (on activation trigger)
        ARMED    → OBSERVE    (on conditions lost)
        ARMED    → FALLBACK   (on safety conditions)
        ACTIVE   → OBSERVE    (on coherence drop)
        ACTIVE   → FALLBACK   (on safety conditions)
        FALLBACK → IDLE       (on reset)
        *        → FALLBACK   (always allowed, dominant)

    All transitions are deterministic based on GateInput.
    No randomness. Time only via now_ms input.
    """

    def __init__(self, config: GateConfig = None, logger: logging.Logger = None):
        self._config = config or GateConfig()
        self._logger = logger or logging.getLogger(__name__)

        self._state = GateState.IDLE
        self._last_transition_ms: Optional[int] = None
        self._transition_count = 0
        self._log_buffer: List[str] = []

    @property
    def state(self) -> GateState:
        """Current gate state."""
        return self._state

    @property
    def transition_count(self) -> int:
        """Total number of state transitions."""
        return self._transition_count

    def _log(self, entry: str) -> None:
        """Add log entry to buffer and emit via logger."""
        self._log_buffer.append(entry)
        self._logger.info(entry)

    def _enter_state(self, new_state: GateState, reason: str, now_ms: int) -> None:
        """Transition to a new state with logging."""
        old_state = self._state
        self._state = new_state
        self._last_transition_ms = now_ms
        self._transition_count += 1

        self._log(_format_log(
            "GATE_ENTER",
            state=new_state.value,
            reason=reason,
            from_state=old_state.value,
            t_ms=now_ms
        ))

    def _check_fallback_conditions(self, inp: GateInput) -> Optional[str]:
        """
        Check if fallback conditions are met.
        Returns reason token if fallback should be forced, None otherwise.

        Fallback is always possible and dominant.
        """
        # Explicit fallback request
        if inp.force_fallback:
            return ReasonToken.MANUAL_FALLBACK

        # Data too stale
        if inp.data_age_ms > self._config.stale_data_threshold_ms:
            return ReasonToken.DATA_STALE

        # Coherence dropped critically
        if inp.coherence_score < 0.1:
            return ReasonToken.COHERENCE_DROP

        return None

    def _check_arm_conditions(self, inp: GateInput) -> bool:
        """Check if conditions are met to arm the gate."""
        # Need sufficient coherence
        if inp.coherence_score < self._config.arm_coherence_min:
            return False

        # Need some form of lock
        if inp.lock_state == "UNLOCKED":
            return False

        # Need explicit arm signal or sufficient conditions
        return inp.arm_signal or (inp.lock_state == "LOCKED" and inp.coherence_score >= 0.5)

    def _check_activation_conditions(self, inp: GateInput) -> bool:
        """Check if conditions are met to activate."""
        # Need strong coherence
        if inp.coherence_score < self._config.activation_coherence_min:
            return False

        # Need locked state
        if inp.lock_state != "LOCKED":
            return False

        # Need activation trigger
        return inp.activate_signal

    def evaluate(self, inp: GateInput) -> GateOutput:
        """
        Evaluate gate state based on input. Deterministic.

        Returns GateOutput with new state, decision, and logs.
        """
        self._log_buffer = []  # Reset per evaluation

        # Log basis fields
        basis_fields = {
            "coherence": f"{inp.coherence_score:.2f}",
            "lock": inp.lock_state,
            "data_age_ms": inp.data_age_ms,
            "rotor": inp.rotor_active,
        }
        if inp.fields:
            basis_fields.update({k: str(v) for k, v in inp.fields.items()})

        self._log(_format_log("GATE_BASIS", fields=basis_fields))

        # === Fallback check (always first, always dominant) ===
        fallback_reason = self._check_fallback_conditions(inp)
        if fallback_reason and self._config.fallback_always_allowed:
            if self._state != GateState.FALLBACK:
                self._enter_state(GateState.FALLBACK, fallback_reason, inp.now_ms)

            self._log(_format_log(
                "GATE_FALLBACK",
                reason=fallback_reason,
                t_ms=inp.now_ms
            ))
            self._log(_format_log(
                "GATE_DECISION",
                state=self._state.value,
                output=GateDecision.FORCE_FALLBACK.value
            ))

            return GateOutput(
                state=self._state,
                decision=GateDecision.FORCE_FALLBACK,
                reason=fallback_reason,
                timestamp_ms=inp.now_ms,
                allowed=False,
                log_entries=list(self._log_buffer)
            )

        # === State-specific transitions ===
        decision = GateDecision.HOLD_OBSERVE
        reason = ReasonToken.INSUFFICIENT_CONTEXT

        if self._state == GateState.IDLE:
            # IDLE → OBSERVE on any input
            self._enter_state(GateState.OBSERVE, ReasonToken.INPUT_RECEIVED, inp.now_ms)
            reason = ReasonToken.OBSERVE_STARTED
            decision = GateDecision.HOLD_OBSERVE

        elif self._state == GateState.OBSERVE:
            # OBSERVE → ARMED if conditions met
            if self._check_arm_conditions(inp):
                self._enter_state(GateState.ARMED, ReasonToken.ARMED_CONDITION_MET, inp.now_ms)
                reason = ReasonToken.ARMED_CONDITION_MET
                decision = GateDecision.HOLD_OBSERVE
            else:
                reason = ReasonToken.INSUFFICIENT_CONTEXT
                decision = GateDecision.HOLD_OBSERVE

        elif self._state == GateState.ARMED:
            # ARMED → ACTIVE if activation conditions met
            if self._check_activation_conditions(inp):
                self._enter_state(GateState.ACTIVE, ReasonToken.ACTIVATION_TRIGGERED, inp.now_ms)
                reason = ReasonToken.ACTIVATION_TRIGGERED
                decision = GateDecision.ALLOW_ACTIVE
            # ARMED → OBSERVE if conditions lost
            elif not self._check_arm_conditions(inp):
                self._enter_state(GateState.OBSERVE, ReasonToken.LOCK_LOST, inp.now_ms)
                reason = ReasonToken.LOCK_LOST
                decision = GateDecision.HOLD_OBSERVE
            else:
                reason = ReasonToken.ARMED_CONDITION_MET
                decision = GateDecision.HOLD_OBSERVE

        elif self._state == GateState.ACTIVE:
            # ACTIVE → OBSERVE if coherence drops
            if inp.coherence_score < self._config.coherence_threshold:
                self._enter_state(GateState.OBSERVE, ReasonToken.COHERENCE_DROP, inp.now_ms)
                reason = ReasonToken.COHERENCE_DROP
                decision = GateDecision.HOLD_OBSERVE
            # ACTIVE → OBSERVE if lock lost
            elif inp.lock_state == "UNLOCKED":
                self._enter_state(GateState.OBSERVE, ReasonToken.LOCK_LOST, inp.now_ms)
                reason = ReasonToken.LOCK_LOST
                decision = GateDecision.HOLD_OBSERVE
            else:
                reason = ReasonToken.ACTIVATION_TRIGGERED
                decision = GateDecision.ALLOW_ACTIVE

        elif self._state == GateState.FALLBACK:
            # FALLBACK → IDLE on reset (no force_fallback, good coherence)
            if not inp.force_fallback and inp.coherence_score >= self._config.coherence_threshold:
                self._enter_state(GateState.IDLE, ReasonToken.SAFETY_RESET, inp.now_ms)
                reason = ReasonToken.SAFETY_RESET
                decision = GateDecision.HOLD_OBSERVE
            else:
                reason = ReasonToken.SAFETY_RESET
                decision = GateDecision.FORCE_FALLBACK

        # Log decision
        self._log(_format_log(
            "GATE_DECISION",
            state=self._state.value,
            output=decision.value
        ))

        return GateOutput(
            state=self._state,
            decision=decision,
            reason=reason,
            timestamp_ms=inp.now_ms,
            allowed=(decision == GateDecision.ALLOW_ACTIVE),
            log_entries=list(self._log_buffer)
        )

    def force_fallback(self, now_ms: int, reason: str = None) -> GateOutput:
        """
        Force immediate transition to FALLBACK state.
        Always allowed, always succeeds.
        """
        reason = reason or ReasonToken.MANUAL_FALLBACK

        if self._state != GateState.FALLBACK:
            self._enter_state(GateState.FALLBACK, reason, now_ms)

        self._log(_format_log(
            "GATE_FALLBACK",
            reason=reason,
            t_ms=now_ms
        ))
        self._log(_format_log(
            "GATE_DECISION",
            state=self._state.value,
            output=GateDecision.FORCE_FALLBACK.value
        ))

        return GateOutput(
            state=self._state,
            decision=GateDecision.FORCE_FALLBACK,
            reason=reason,
            timestamp_ms=now_ms,
            allowed=False,
            log_entries=list(self._log_buffer)
        )

    def reset(self, now_ms: int) -> None:
        """Reset gate to IDLE state."""
        self._log_buffer = []
        self._enter_state(GateState.IDLE, ReasonToken.INIT_COMPLETE, now_ms)

    def get_debug_state(self) -> Dict[str, Any]:
        """Get debug state snapshot."""
        return {
            "state": self._state.value,
            "transition_count": self._transition_count,
            "last_transition_ms": self._last_transition_ms,
        }


# === Convenience factory ===

def create_gate(config: GateConfig = None) -> ActionGateV0_1:
    """Create a new ActionGate instance."""
    return ActionGateV0_1(config=config)
