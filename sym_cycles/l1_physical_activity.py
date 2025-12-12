#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
l1_physical_activity.py — L1 PhysicalActivity Layer v0.3 (Encoder-Aware + Decay)

S02.HandEncoder-Awareness v0.1 — Patch A/B/C implementatie.

Changelog v0.3:
- PATCH A: θ̂ nu ALTIJD uit cycles_physical_total / cycles_per_rot (niet rotations!)
- PATCH B1: encoder_conf time-decay: conf *= exp(-dt_s / tau_s)
- PATCH B2: Hard gap reset: bij dt > hard_reset_s → conf=0, state=STILL
- FIX: activity_score decay bij geen events
- FIX: disp_score is altijd unsigned |Δθ̂|

Kernprincipes:
- θ̂ = unsigned fysieke progress (cycles_total / cycles_per_rot)
- θ̂ verandert VÓÓR lock (zodra cycles toenemen)
- encoder_conf vervalt exponentieel zonder nieuwe input
- State valt terug naar STILL bij stilte

States:
- STILL:        Geen activiteit, geen displacement
- FEELING:      Lichte activiteit (events), geen displacement
- SCRAPE:       Hoge activiteit, geen displacement (edge oscillatie)
- DISPLACEMENT: Netto displacement, direction nog onzeker
- MOVING:       Displacement + direction stabiel
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from collections import deque
import math


class L1State(Enum):
    """L1 PhysicalActivity states (v0.3 encoder-aware)."""
    STILL = "STILL"
    FEELING = "FEELING"
    SCRAPE = "SCRAPE"
    DISPLACEMENT = "DISPLACEMENT"
    MOVING = "MOVING"


@dataclass
class L1Config:
    """
    Configuratie voor L1 PhysicalActivity (v0.3 met decay).
    
    Patch A: θ̂ bron
    - cycles_per_rot: cycles per volledige rotatie (default 12)
    
    Patch B: Time decay
    - encoder_tau_s: exponentiële decay tijdconstante
    - hard_reset_s: force reset na deze stilte
    - activity_decay_rate: activity score decay per seconde
    """
    # Gap threshold
    gap_ms: float = 500.0
    
    # Activity thresholds (events per seconde)
    activity_threshold_low: float = 1.0    # A0: STILL → FEELING
    activity_threshold_high: float = 5.0   # A1: FEELING → SCRAPE
    
    # Displacement threshold (rotations per update)
    displacement_threshold: float = 0.005  # D0: min |Δθ̂| voor DISPLACEMENT
    
    # Direction thresholds
    direction_conf_threshold: float = 0.5
    lock_states_for_moving: tuple = ("SOFT_LOCK", "LOCKED")
    
    # Cycles per rotation (Patch A)
    cycles_per_rot: float = 12.0
    
    # Time decay parameters (Patch B)
    encoder_tau_s: float = 0.6       # Decay tijdconstante
    hard_reset_s: float = 1.5        # Hard reset na stilte
    activity_decay_rate: float = 5.0  # Activity decay per seconde


@dataclass
class L1Snapshot:
    """Snapshot van L1 state (v0.3 met decay info)."""
    state: L1State
    
    # Virtuele hoek θ̂ (encoder-achtig, uit cycles!)
    theta_hat_rot: float = 0.0
    delta_theta_rot: float = 0.0
    
    # Activity metrics
    activity_score: float = 0.0
    disp_score: float = 0.0
    
    # Doorgeluste L2 info
    direction_effective: str = "UNDECIDED"
    direction_conf: float = 0.0
    lock_state: str = "UNLOCKED"
    
    # Encoder confidence (met decay!)
    encoder_conf: float = 0.0
    
    # Timing
    dt_s: float = 0.0                       # Tijd sinds vorige update
    t_last_cycle: Optional[float] = None
    t_last_event: Optional[float] = None
    gap_since_cycle_ms: float = float('inf')
    gap_since_event_ms: float = float('inf')
    
    # Counters
    total_cycles: float = 0.0
    total_events: int = 0
    delta_cycles: float = 0.0
    delta_events: int = 0
    events_without_cycles: int = 0


class L1PhysicalActivity:
    """
    L1 PhysicalActivity Layer v0.3 (Encoder-Aware + Time Decay).
    
    Patch A: θ̂ = cycles_physical_total / cycles_per_rot (unsigned progress)
    Patch B: encoder_conf decays exponentially, hard reset na stilte
    """
    
    def __init__(self, config: L1Config = None):
        self.config = config or L1Config()
        
        # State
        self._state: L1State = L1State.STILL
        
        # θ̂ integrator (Patch A: uit cycles, niet rotations!)
        self._theta_hat_rot: float = 0.0
        self._prev_theta_hat_rot: float = 0.0
        
        # Timing
        self._t_last_update: Optional[float] = None
        self._t_last_cycle: Optional[float] = None
        self._t_last_event: Optional[float] = None
        
        # Counters
        self._prev_cycles_total: float = 0.0
        self._total_events: int = 0
        self._events_without_cycles: int = 0
        
        # Activity tracking (met decay)
        self._activity_score: float = 0.0
        
        # Encoder confidence (met decay)
        self._encoder_conf: float = 0.0
        
        # L2 doorlus
        self._direction_effective: str = "UNDECIDED"
        self._direction_conf: float = 0.0
        self._lock_state: str = "UNLOCKED"
    
    @property
    def state(self) -> L1State:
        return self._state
    
    @property
    def theta_hat_rot(self) -> float:
        return self._theta_hat_rot
    
    def update(
        self,
        wall_time: float,
        cycles_physical_total: float,
        events_this_batch: int = 0,
        # Optionele L2 doorlus (voor MOVING state)
        direction_conf: float = None,
        lock_state: str = None,
        direction_effective: str = None,
        # NIET MEER GEBRUIKEN voor θ̂:
        rotations: float = None,  # Ignored! θ̂ komt uit cycles
    ) -> L1Snapshot:
        """
        Update L1 state.
        
        Patch A: θ̂ = cycles_physical_total / cycles_per_rot
        Patch B: Time decay op activity en encoder_conf
        Patch C: Wordt aangeroepen met events_this_batch=0 bij idle tick
        """
        cfg = self.config
        
        # === Timing (Patch B) ===
        dt_s = 0.0
        if self._t_last_update is not None:
            dt_s = wall_time - self._t_last_update
        self._t_last_update = wall_time
        
        # === Hard reset check (Patch B2) ===
        if dt_s > cfg.hard_reset_s:
            self._hard_reset()
            dt_s = 0.0  # Na reset, geen decay toepassen
        
        # === Delta berekeningen ===
        delta_cycles = cycles_physical_total - self._prev_cycles_total
        self._prev_cycles_total = cycles_physical_total
        
        delta_events = events_this_batch
        self._total_events += events_this_batch
        
        # === θ̂ uit cycles (Patch A - NIET uit rotations!) ===
        self._prev_theta_hat_rot = self._theta_hat_rot
        self._theta_hat_rot = cycles_physical_total / cfg.cycles_per_rot
        delta_theta_rot = self._theta_hat_rot - self._prev_theta_hat_rot
        
        # === Timing updates ===
        if delta_cycles > 0:
            self._t_last_cycle = wall_time
            self._events_without_cycles = 0
        
        if delta_events > 0:
            self._t_last_event = wall_time
            if delta_cycles == 0:
                self._events_without_cycles += delta_events
        
        # === L2 doorlus ===
        if direction_conf is not None:
            self._direction_conf = direction_conf
        if lock_state is not None:
            self._lock_state = lock_state
        if direction_effective is not None:
            self._direction_effective = direction_effective
        
        # === Gap berekening ===
        gap_since_cycle_ms = float('inf')
        if self._t_last_cycle is not None:
            gap_since_cycle_ms = (wall_time - self._t_last_cycle) * 1000.0
        
        gap_since_event_ms = float('inf')
        if self._t_last_event is not None:
            gap_since_event_ms = (wall_time - self._t_last_event) * 1000.0
        
        # === Activity score met decay (Patch B) ===
        # Decay bestaande score
        if dt_s > 0:
            decay_factor = math.exp(-dt_s * cfg.activity_decay_rate)
            self._activity_score *= decay_factor
        
        # Voeg nieuwe events toe
        self._activity_score += delta_events
        
        # === Displacement score ===
        disp_score = abs(delta_theta_rot)
        
        # === Encoder confidence met decay (Patch B1) ===
        if dt_s > 0:
            decay_factor = math.exp(-dt_s / cfg.encoder_tau_s)
            self._encoder_conf *= decay_factor
        
        # Boost encoder_conf bij activiteit/displacement
        if delta_cycles > 0:
            self._encoder_conf = min(1.0, self._encoder_conf + 0.15)
        elif delta_events > 0:
            self._encoder_conf = min(1.0, self._encoder_conf + 0.05)
        
        # Lock/direction boost
        if self._lock_state == "LOCKED":
            self._encoder_conf = min(1.0, self._encoder_conf + 0.1 * dt_s)
        
        # Clamp
        self._encoder_conf = max(0.0, min(1.0, self._encoder_conf))
        
        # === State machine ===
        self._state = self._compute_state(
            activity_score=self._activity_score,
            disp_score=disp_score,
            gap_since_cycle_ms=gap_since_cycle_ms,
            gap_since_event_ms=gap_since_event_ms,
        )
        
        return L1Snapshot(
            state=self._state,
            theta_hat_rot=self._theta_hat_rot,
            delta_theta_rot=delta_theta_rot,
            activity_score=self._activity_score,
            disp_score=disp_score,
            direction_effective=self._direction_effective,
            direction_conf=self._direction_conf,
            lock_state=self._lock_state,
            encoder_conf=self._encoder_conf,
            dt_s=dt_s,
            t_last_cycle=self._t_last_cycle,
            t_last_event=self._t_last_event,
            gap_since_cycle_ms=gap_since_cycle_ms,
            gap_since_event_ms=gap_since_event_ms,
            total_cycles=cycles_physical_total,
            total_events=self._total_events,
            delta_cycles=delta_cycles,
            delta_events=delta_events,
            events_without_cycles=self._events_without_cycles,
        )
    
    def _compute_state(
        self,
        activity_score: float,
        disp_score: float,
        gap_since_cycle_ms: float,
        gap_since_event_ms: float,
    ) -> L1State:
        """
        State machine:
        - Gap timeout → STILL
        - Displacement → DISPLACEMENT/MOVING
        - Activity only → FEELING/SCRAPE
        """
        cfg = self.config
        
        # Gap timeout → STILL
        if gap_since_cycle_ms >= cfg.gap_ms and gap_since_event_ms >= cfg.gap_ms:
            return L1State.STILL
        
        # Low activity → STILL
        if activity_score < cfg.activity_threshold_low and disp_score < cfg.displacement_threshold:
            return L1State.STILL
        
        # Displacement gate
        has_displacement = disp_score >= cfg.displacement_threshold
        
        if has_displacement:
            # Check direction stability
            direction_stable = (
                self._direction_conf >= cfg.direction_conf_threshold or
                self._lock_state in cfg.lock_states_for_moving
            )
            return L1State.MOVING if direction_stable else L1State.DISPLACEMENT
        
        # Alleen activiteit, geen displacement
        if activity_score >= cfg.activity_threshold_high:
            return L1State.SCRAPE
        elif activity_score >= cfg.activity_threshold_low:
            return L1State.FEELING
        
        return L1State.STILL
    
    def _hard_reset(self):
        """Hard reset na lange stilte (Patch B2)."""
        self._state = L1State.STILL
        self._encoder_conf = 0.0
        self._activity_score = 0.0
        self._events_without_cycles = 0
        # θ̂ behouden (fysieke positie blijft)
    
    def reset(self):
        """Volledige reset."""
        self._state = L1State.STILL
        self._theta_hat_rot = 0.0
        self._prev_theta_hat_rot = 0.0
        self._t_last_update = None
        self._t_last_cycle = None
        self._t_last_event = None
        self._prev_cycles_total = 0.0
        self._total_events = 0
        self._events_without_cycles = 0
        self._activity_score = 0.0
        self._encoder_conf = 0.0
        self._direction_effective = "UNDECIDED"
        self._direction_conf = 0.0
        self._lock_state = "UNLOCKED"


# === Preset Configs ===

L1_CONFIG_HUMAN = L1Config(
    gap_ms=500.0,
    activity_threshold_low=1.0,
    activity_threshold_high=5.0,
    displacement_threshold=0.005,
    direction_conf_threshold=0.5,
    cycles_per_rot=12.0,
    encoder_tau_s=0.6,
    hard_reset_s=1.5,
    activity_decay_rate=5.0,
)

L1_CONFIG_BENCH = L1Config(
    gap_ms=800.0,
    activity_threshold_low=2.0,
    activity_threshold_high=8.0,
    displacement_threshold=0.01,
    direction_conf_threshold=0.6,
    cycles_per_rot=12.0,
    encoder_tau_s=0.8,
    hard_reset_s=2.0,
    activity_decay_rate=3.0,
)
