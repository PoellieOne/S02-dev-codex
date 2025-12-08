#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
movement_body_v3

Awareness v3 â€” MovementBody bovenop sym_cycles realtime kompas.

Integratie met sym_cycles.realtime_compass:

- "Snel" vensterkompas = CompassRealtimeState / window:
    window_direction, window_confidence
- "Traag" inertiaal kompas = InertialCompassState:
    global_direction, global_score (signed)

MovementBody gebruikt:
- global_direction + |global_score| als richtingsanker (lock),
- window_direction + window_confidence voor flow/tegenstroom.

Zie ook:
  sym_cycles.realtime_compass.InertialCompassState
  sym_cycles.realtime_compass.InertialCompassSnapshot
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import statistics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def _get_time_us(node: Dict[str, Any]) -> Optional[int]:
    """
    Haal een betekenisvol tijdstempel uit een backbone-node of step.
    Voorkeur: t_center_us, anders t_us, anders t.
    """
    for key in ("t_center_us", "t_us", "t"):
        if key in node:
            try:
                return int(node[key])
            except (TypeError, ValueError):
                continue
    return None


def _get_projection_score(step_or_node: Dict[str, Any]) -> Optional[float]:
    """
    Extractie van een 'projection score' (0..1) uit ProjectedStep/backbone-node.
    Verwacht bij ProjectedStep: obj["projection"]["total_score"].
    Bij backbone-node kun je evt. zelf een veld 'projection_score' toevoegen.
    """
    if "projection" in step_or_node and isinstance(step_or_node["projection"], dict):
        if "total_score" in step_or_node["projection"]:
            try:
                return float(step_or_node["projection"]["total_score"])
            except (TypeError, ValueError):
                return None

    if "projection_score" in step_or_node:
        try:
            return float(step_or_node["projection_score"])
        except (TypeError, ValueError):
            return None

    return None


def _opposite_dir(d: str) -> str:
    if d == "CW":
        return "CCW"
    if d == "CCW":
        return "CW"
    return "UNDECIDED"


# ---------------------------------------------------------------------------
# State-dataclass
# ---------------------------------------------------------------------------

@dataclass
class MovementBodyStateV3:
    # Tijd
    t_us: Optional[int] = None           # laatst verwerkte tijdstempel

    # CompassRealtimeState / InertialCompassSnapshot (laatste snapshot)
    compass_global_direction: str = "UNDECIDED"
    compass_global_score_signed: float = 0.0   # ruwe signed score uit InertialCompassState
    compass_global_score: float = 0.0          # |score| â†’ 0..1 confidence
    compass_window_direction: str = "UNDECIDED"
    compass_window_score: float = 0.0          # window_confidence
    compass_trend: str = "UNKNOWN"
    compass_stable: bool = False

    # Direction lock
    direction_lock_state: str = "UNLOCKED"   # "UNLOCKED" | "SOFT_LOCK" | "LOCKED"
    direction_locked_dir: str = "UNDECIDED"  # richting waar het lichaam op vast zit
    direction_locked_conf: float = 0.0       # lock-confidence 0..1

    # Effectieve richting voor sign
    direction_global_effective: str = "UNDECIDED"
    direction_global_conf: float = 0.0

    # Cycli & hoek
    cycles_per_rot: float = 12.0        # C_active
    cycle_index: int = 0                # signed: CW = +1/cycle, CCW = -1/cycle
    rotations: float = 0.0              # signed = cycle_index / C
    theta_deg: float = 0.0              # 0..360 mechanische hoek
    theta_wrap_count: int = 0           # integer wraps voor cumulatieve hoek

    # Snelheid
    rpm_inst: float = 0.0               # instant rpm uit laatste cycle
    rpm_est: float = 0.0                # gesmoothed rpm (EMA)
    rpm_jitter: float = 0.0             # sigma/mean over recent venster

    # Flow / weerstand (window vs locked)
    flow_state: str = "NEUTRAL"         # "FLOW" | "RESIST" | "CHAOTIC" | "NEUTRAL"
    flow_score: float = 0.0             # 0..1 â€“ hoe lekker het loopt
    resist_score: float = 0.0           # 0..1 â€“ hoeveel tegenstroom

    # Bewegingstoestand
    motion_state: str = "STATIC"        # "STATIC" | "EVALUATING" | "MOVING"
    motion_conf: float = 0.0            # 0..1

    # Backbone-koppeling / kwaliteit
    last_backbone_node_id: Optional[int] = None
    last_cycle_type: Optional[str] = None
    last_projection_score: float = 0.0
    cadence_ok: bool = False            # op basis van rpm_jitter e.d.

    # Awareness
    awareness_conf: float = 0.0         # globale state-confidence 0..1


# ---------------------------------------------------------------------------
# MovementBody v3 â€“ kernklasse
# ---------------------------------------------------------------------------

class MovementBodyV3:
    """
    Kernklasse voor Awareness v3 MovementBody met CompassRealtimeState /
    InertialCompassSnapshot als richting-input.

    Typische integratie met sym_cycles.realtime_compass:

        from sym_cycles.realtime_compass import InertialCompassState

        ics = InertialCompassState(window_tiles=20, phase_class="any")
        mb  = MovementBodyV3(cycles_per_rot=12.0)

        for tile in tiles:
            snap = ics.feed_tile(tile, tile_duration_us=tile_duration_us)
            mb.set_compass_realtime(snap)  # object of snap.__dict__
            mb.feed_cycle_node(backbone_node_for_same_time)

        state = mb.snapshot()

    Thresholds tunen
        inertial global_score typisch â‰ˆ 0.4â€“0.6,
        window_confidence â‰ˆ 0.6â€“1.0 in stabiele fases.

    Die twee getallen zijn feitelijk je natuurlijke schaal. Dus:
        lock_global_hi ergens ~0.4â€“0.5,
        compass_min_conf â‰ˆ 0.25â€“0.3 is logisch,
        lock_window_min iets onder typische window_conf (bijv. 0.6).
    """

    def __init__(self,
                cycles_per_rot: float = 12.0,
                auto_learn_cycles_per_rot: bool = False,
                rpm_alpha: float = 0.3,
                compass_min_conf: float = 0.25,
                min_proj_score: float = 0.0,
                rpm_slow_thresh: float = 20.0,
                rpm_move_thresh: float = 60.0,
                idle_timeout_us: int = 800_000,
                jitter_window_size: int = 10,
                jitter_max_rel: float = 0.4,
                awareness_rpm_norm: float = 100.0,
                # Lock thresholds
                lock_global_hi: float = 0.4,
                lock_window_min: float = 0.4,
                lock_cycles_min: int = 3,
                lock_promote_cycles: int = 4,
                unlock_global_lo: float = 0.25,
                unlock_window_hi: float = 0.8,
                unlock_window_conflict_cycles: int = 3,
                hard_flip_cycles: int = 6,
                # Flow thresholds
                flow_hi: float = 0.6,
                resist_hi: float = 0.6,
                # Uitadem logica
                idle_lock_decay: float = 0.9,
                idle_unlock_time_us: int = 2_000_000,
                idle_awareness_floor: float = 0.05):
        # Config â€“ algemene
        self.cycles_per_rot_nominal = float(cycles_per_rot)
        self.auto_learn_cycles_per_rot = bool(auto_learn_cycles_per_rot)
        self.rpm_alpha = float(rpm_alpha)
        self.compass_min_conf = float(compass_min_conf)
        self.min_proj_score = float(min_proj_score)
        self.rpm_slow_thresh = float(rpm_slow_thresh)
        self.rpm_move_thresh = float(rpm_move_thresh)
        self.idle_timeout_us = int(idle_timeout_us)
        self.jitter_window_size = int(jitter_window_size)
        self.jitter_max_rel = float(jitter_max_rel)
        self.awareness_rpm_norm = float(awareness_rpm_norm)

        # Config â€“ lock
        self.lock_global_hi = float(lock_global_hi)
        self.lock_window_min = float(lock_window_min)
        self.lock_cycles_min = int(lock_cycles_min)
        self.lock_promote_cycles = int(lock_promote_cycles)
        self.unlock_global_lo = float(unlock_global_lo)
        self.unlock_window_hi = float(unlock_window_hi)
        self.unlock_window_conflict_cycles = int(unlock_window_conflict_cycles)
        self.hard_flip_cycles = int(hard_flip_cycles)

        # Config â€“ flow
        self.flow_hi = float(flow_hi)
        self.resist_hi = float(resist_hi)

        # Uitadem-config
        self.idle_lock_decay = float(idle_lock_decay)
        self.idle_unlock_time_us = int(idle_unlock_time_us)
        self.idle_awareness_floor = float(idle_awareness_floor)

        # Interne state
        self._state = MovementBodyStateV3(
            cycles_per_rot=self.cycles_per_rot_nominal
        )

        self._last_cycle_t_us: Optional[int] = None
        self._rpm_window: List[float] = []

        # Lock/flow interne counters
        self._lock_candidate_dir: str = "UNDECIDED"
        self._lock_candidate_count: int = 0
        self._conflict_count: int = 0
        self._hard_flip_conflict_count: int = 0

        # Idle-cumulatief (voor totale idle-tijd)
        self._idle_start_us: Optional[int] = None

    # ------------------------------------------------------------------
    # CompassRealtimeState / InertialCompassSnapshot ingest
    # ------------------------------------------------------------------

    def set_compass_realtime(self, compass: Any) -> None:
        """
        Update compass-gerelateerde state vanuit een realtime kompas-snapshot.

        Ondersteunde bronnen:
        - InertialCompassSnapshot object:
            .window_direction, .window_confidence,
            .global_direction, .global_score
        - dict met dezelfde velden (zoals snapshot.__dict__)
        - eventueel aanvullende keys:
            trend, stability_flags
        """
        # Object of dict?
        if hasattr(compass, "global_direction"):
            gd = getattr(compass, "global_direction", "UNDECIDED")
            gs_signed = getattr(compass, "global_score", 0.0)
            wd = getattr(compass, "window_direction", "UNDECIDED")
            ws = getattr(compass, "window_confidence", None)
            if ws is None:
                ws = getattr(compass, "window_score", 0.0)
            tr = getattr(compass, "trend", None)
            stf = getattr(compass, "stability_flags", None)
        elif isinstance(compass, dict):
            gd = compass.get("global_direction", "UNDECIDED")
            gs_signed = compass.get("global_score", 0.0)
            wd = compass.get("window_direction", "UNDECIDED")
            ws = compass.get("window_confidence", compass.get("window_score", 0.0))
            tr = compass.get("trend", None)
            stf = compass.get("stability_flags", None)
        else:
            return  # onbekend type

        gd = gd if gd in ("CW", "CCW", "UNDECIDED") else "UNDECIDED"
        wd = wd if wd in ("CW", "CCW", "UNDECIDED") else "UNDECIDED"

        # global_score is signed (CW=+, CCW=-) â†’ magnitude is confidence
        try:
            gs_signed = float(gs_signed)
        except (TypeError, ValueError):
            gs_signed = 0.0
        gs = abs(gs_signed)

        try:
            ws = float(ws)
        except (TypeError, ValueError):
            ws = 0.0

        st = self._state
        st.compass_global_direction = gd
        st.compass_global_score_signed = gs_signed
        st.compass_global_score = _clamp(gs, 0.0, 1.0)
        st.compass_window_direction = wd
        st.compass_window_score = _clamp(ws, 0.0, 1.0)
        st.compass_trend = str(tr) if tr is not None else "UNKNOWN"

        stable_flag = False
        if isinstance(stf, dict):
            stable_flag = bool(stf.get("jitter_low", False)) or bool(
                stf.get("stable", False)
            )
        st.compass_stable = stable_flag

        # Lock-update + afgeleide richting voor sign
        self._update_direction_lock()
        self._update_flow_state()
        self._update_awareness_conf()

    # Backwards-compat wrapper voor losse CompassSignResult
    def set_compass_result(self, compass: Any) -> None:
        """
        Wrapper voor oude CompassSignResult:
        treated as inertial snapshot met alleen global_* gevuld.
        """
        if hasattr(compass, "global_direction"):
            gd = getattr(compass, "global_direction", "UNDECIDED")
            conf = getattr(compass, "confidence", 0.0)
        elif isinstance(compass, dict):
            gd = compass.get("global_direction", "UNDECIDED")
            conf = compass.get("confidence", 0.0)
        else:
            return

        self.set_compass_realtime({
            "global_direction": gd,
            "global_score": float(conf),  # positief, dus magnitude = conf
            "window_direction": "UNDECIDED",
            "window_confidence": 0.0,
        })

    def merge_core1_direction(self, dir_label: str, dir_conf: float) -> None:
        """
        Optionele fusie met Core1 direction-engine (SROT/RCC).
        We verhogen direction_locked_conf als Core1 sterker is.
        """
        if dir_label not in ("CW", "CCW", "UNDECIDED"):
            return
        dir_conf = _clamp(float(dir_conf), 0.0, 1.0)

        if dir_label == "UNDECIDED":
            return

        st = self._state

        # Als we al locked zijn op dezelfde richting â†’ conf liften
        if st.direction_locked_dir == dir_label:
            st.direction_locked_conf = max(st.direction_locked_conf, dir_conf)
        # Als we nog niets locked hebben, kunnen we dit als start gebruiken
        elif st.direction_lock_state == "UNLOCKED":
            st.direction_locked_dir = dir_label
            st.direction_locked_conf = dir_conf
            st.direction_lock_state = "SOFT_LOCK"

        self._update_effective_direction()
        self._update_awareness_conf()

    # ------------------------------------------------------------------
    # Direction sign en lock mechaniek
    # ------------------------------------------------------------------

    def _update_effective_direction(self) -> None:
        """
        Stel direction_global_effective + direction_global_conf afgeleid in:
        - bij LOCKED/SOFT_LOCK: locked_dir + max(locked_conf, global_score)
        - bij UNLOCKED: global_direction + global_score
        """
        st = self._state
        if st.direction_lock_state in ("LOCKED", "SOFT_LOCK") and \
           st.direction_locked_dir in ("CW", "CCW"):
            st.direction_global_effective = st.direction_locked_dir
            st.direction_global_conf = max(
                _clamp(st.direction_locked_conf, 0.0, 1.0),
                st.compass_global_score,
            )
        else:
            st.direction_global_effective = st.compass_global_direction
            st.direction_global_conf = st.compass_global_score

    def _update_direction_lock(self) -> None:
        """
        Lock/unlock/flip logica op basis van inertiaal/global + window kompas.
        """
        st = self._state
        gd = st.compass_global_direction
        gs = st.compass_global_score     # magnitude 0..1
        wd = st.compass_window_direction
        ws = st.compass_window_score

        # alignment / conflict (alleen nog conceptueel nodig)
        same_dir = gd in ("CW", "CCW") and wd == gd
        opp_dir = gd in ("CW", "CCW") and wd == _opposite_dir(gd)

        # UNLOCKED â†’ SOFT_LOCK
        if st.direction_lock_state == "UNLOCKED":
            if gd in ("CW", "CCW") and gs >= self.lock_global_hi:
                window_ok = (
                    wd == "UNDECIDED"
                    or (wd == gd and ws >= self.lock_window_min)
                )
                if window_ok:
                    if self._lock_candidate_dir == gd:
                        self._lock_candidate_count += 1
                    else:
                        self._lock_candidate_dir = gd
                        self._lock_candidate_count = 1

                    if self._lock_candidate_count >= self.lock_cycles_min:
                        st.direction_lock_state = "SOFT_LOCK"
                        st.direction_locked_dir = gd
                        st.direction_locked_conf = gs
                        self._lock_candidate_count = 0
                        self._conflict_count = 0
                        self._hard_flip_conflict_count = 0
                else:
                    self._lock_candidate_dir = "UNDECIDED"
                    self._lock_candidate_count = 0
            else:
                self._lock_candidate_dir = "UNDECIDED"
                self._lock_candidate_count = 0

        # SOFT_LOCK â†’ LOCKED of terug
        elif st.direction_lock_state == "SOFT_LOCK":
            locked_dir = st.direction_locked_dir
            if locked_dir in ("CW", "CCW"):
                if gd == locked_dir and gs >= self.lock_global_hi:
                    window_ok = (
                        wd == "UNDECIDED"
                        or (wd == locked_dir and ws >= self.lock_window_min)
                    )
                    if window_ok:
                        self._lock_candidate_count += 1
                        if self._lock_candidate_count >= self.lock_promote_cycles:
                            st.direction_lock_state = "LOCKED"
                            st.direction_locked_conf = max(st.direction_locked_conf, gs)
                            self._lock_candidate_count = 0
                            self._conflict_count = 0
                            self._hard_flip_conflict_count = 0
                    else:
                        self._lock_candidate_count = 0
                if gs < self.unlock_global_lo:
                    st.direction_lock_state = "UNLOCKED"
                    st.direction_locked_dir = "UNDECIDED"
                    st.direction_locked_conf = 0.0
                    self._lock_candidate_count = 0
                    self._conflict_count = 0
                    self._hard_flip_conflict_count = 0

        # LOCKED â€“ monitor tegenstroom/stabiliteit
        if st.direction_lock_state == "LOCKED":
            locked_dir = st.direction_locked_dir

            # Stabiliteit valt weg â†’ degrade naar SOFT_LOCK
            if gs < self.unlock_global_lo:
                st.direction_lock_state = "SOFT_LOCK"

            # Tegenstroom via window:
            if locked_dir in ("CW", "CCW") and wd == _opposite_dir(locked_dir) and ws >= self.unlock_window_hi:
                self._conflict_count += 1
                self._hard_flip_conflict_count += 1
            else:
                self._conflict_count = 0

            # Bij voldoende tegenstroom â†’ terug naar SOFT_LOCK
            if self._conflict_count >= self.unlock_window_conflict_cycles:
                st.direction_lock_state = "SOFT_LOCK"

            # Eventuele hard flip bij langdurige sterke tegenstroom
            if self._hard_flip_conflict_count >= self.hard_flip_cycles:
                new_dir = _opposite_dir(locked_dir)
                if new_dir in ("CW", "CCW"):
                    st.direction_lock_state = "SOFT_LOCK"
                    st.direction_locked_dir = new_dir
                    st.direction_locked_conf = gs
                self._hard_flip_conflict_count = 0

        self._update_effective_direction()

    def _direction_sign(self) -> int:
        """
        +1 bij CW, -1 bij CCW, 0 bij UNDECIDED of te lage confidence.

        N.B. Dit is de sign voor cycle_index/rotations/Î¸.
        RPM zelf blijft fysisch positief.
        """
        st = self._state
        d = st.direction_global_effective
        conf = st.direction_global_conf

        if d not in ("CW", "CCW"):
            return 0
        if conf < self.compass_min_conf:
            return 0

        return +1 if d == "CW" else -1

    # ------------------------------------------------------------------
    # Cycle-feeds (backbone)
    # ------------------------------------------------------------------

    def feed_cycle_node(self,
                        node: Dict[str, Any],
                        proj_score: Optional[float] = None) -> None:
        """
        Verwerk Ã©Ã©n backbone-cycle.

        Vereist:
        - tijdstempel (t_center_us of t_us)
        Optioneel:
        - "id" en "cycle_type"
        - "projection_score" of projection.total_score (als proj_score None is)
        """
        t_us = _get_time_us(node)
        if t_us is None:
            return

        st = self._state
        st.t_us = t_us

        if "id" in node:
            try:
                st.last_backbone_node_id = int(node["id"])
            except (TypeError, ValueError):
                st.last_backbone_node_id = None

        if "cycle_type" in node:
            st.last_cycle_type = str(node["cycle_type"])

        if proj_score is None:
            proj_score = _get_projection_score(node)
        if proj_score is None:
            proj_score = 0.0

        st.last_projection_score = float(proj_score)

        if proj_score < self.min_proj_score:
            self._update_motion_state_idle_like()
            self._update_awareness_conf()
            return

        s = self._direction_sign()

        if s != 0:
            self._update_cycles_and_angle(t_us, s)
            self._update_rpm(t_us)
        else:
            self._update_motion_state_idle_like()

        self._update_awareness_conf()

    def _update_cycles_and_angle(self, t_us: int, sign: int) -> None:
        st = self._state
        C = st.cycles_per_rot if st.cycles_per_rot > 0 else self.cycles_per_rot_nominal

        st.cycle_index += int(sign)
        st.rotations = st.cycle_index / C

        theta_prev = st.theta_deg
        theta_new = (st.rotations * 360.0) % 360.0

        delta = theta_new - theta_prev
        if delta > 180.0:
            st.theta_wrap_count -= 1
        elif delta < -180.0:
            st.theta_wrap_count += 1

        st.theta_deg = theta_new
        st.t_us = t_us

    def _update_rpm(self, t_us: int) -> None:
        st = self._state

        if self._last_cycle_t_us is None:
            self._last_cycle_t_us = t_us
            return

        dt_us = t_us - self._last_cycle_t_us
        self._last_cycle_t_us = t_us

        if dt_us <= 0:
            return

        dt_s = dt_us * 1e-6
        C = st.cycles_per_rot if st.cycles_per_rot > 0 else self.cycles_per_rot_nominal

        rpm_inst = 60.0 / (dt_s * C)

        if st.rpm_est <= 0:
            rpm_est = rpm_inst
        else:
            alpha = self.rpm_alpha
            rpm_est = (1.0 - alpha) * st.rpm_est + alpha * rpm_inst

        st.rpm_inst = rpm_inst
        st.rpm_est = rpm_est

        self._rpm_window.append(rpm_inst)
        if len(self._rpm_window) > self.jitter_window_size:
            self._rpm_window.pop(0)

        if len(self._rpm_window) >= 2:
            mean_rpm = statistics.mean(self._rpm_window)
            if mean_rpm > 0:
                sigma_rpm = statistics.pstdev(self._rpm_window)
                st.rpm_jitter = _clamp(sigma_rpm / mean_rpm, 0.0, 1.0)
            else:
                st.rpm_jitter = 0.0
        else:
            st.rpm_jitter = 0.0

        st.cadence_ok = st.rpm_jitter <= self.jitter_max_rel
        self._update_motion_state_from_rpm()

    # ------------------------------------------------------------------
    # Flow / tegenstroom uit window vs locked
    # ------------------------------------------------------------------

    def _update_flow_state(self) -> None:
        st = self._state

        if st.direction_lock_state == "UNLOCKED" or \
           st.direction_locked_dir not in ("CW", "CCW"):
            st.flow_state = "NEUTRAL"
            st.flow_score = 0.0
            st.resist_score = 0.0
            return

        locked_dir = st.direction_locked_dir
        wd = st.compass_window_direction
        ws = st.compass_window_score

        alignment = 0.0
        if wd == locked_dir:
            alignment = ws
        elif wd == _opposite_dir(locked_dir):
            alignment = -ws

        st.flow_score = max(0.0, alignment)
        st.resist_score = max(0.0, -alignment)

        if st.flow_score > self.flow_hi and st.resist_score < 0.3:
            st.flow_state = "FLOW"
        elif st.resist_score > self.resist_hi and st.flow_score < 0.3:
            st.flow_state = "RESIST"
        elif st.flow_score == 0.0 and st.resist_score == 0.0:
            st.flow_state = "NEUTRAL"
        else:
            st.flow_state = "CHAOTIC"

    # ------------------------------------------------------------------
    # Motion-state & idle
    # ------------------------------------------------------------------

    def _update_motion_state_from_rpm(self) -> None:
        st = self._state
        rpm = st.rpm_est
        dir_conf = st.direction_locked_conf

        if rpm < 1.0 and dir_conf < 0.3:
            st.motion_state = "STATIC"
        elif rpm < self.rpm_slow_thresh or not st.cadence_ok:
            st.motion_state = "EVALUATING"
        elif rpm < self.rpm_move_thresh:
            st.motion_state = "EVALUATING"
        else:
            st.motion_state = "MOVING"

        base_conf = _clamp(rpm / self.rpm_move_thresh, 0.0, 1.0)
        if not st.cadence_ok:
            base_conf *= 0.5

        if st.motion_state == "STATIC":
            st.motion_conf = 0.0
        else:
            st.motion_conf = base_conf

    def _update_motion_state_idle_like(self) -> None:
        st = self._state
        rpm = st.rpm_est
        dir_conf = st.direction_locked_conf

        if rpm < 1.0 and dir_conf < 0.3:
            st.motion_state = "STATIC"
            st.motion_conf = 0.0
        elif rpm < self.rpm_slow_thresh:
            st.motion_state = "EVALUATING"
            st.motion_conf = _clamp(rpm / self.rpm_slow_thresh, 0.0, 1.0)

    def update_idle(self, t_us: int) -> None:
        """
        Bijwerken van idle-detectie / timeouts als er geen nieuwe cycles zijn.

        - Als dt_idle > idle_timeout_us â†’ rpm langzaam laten loslaten, motion -> richting STATIC.
        - Na lange idle (idle_unlock_time_us) â†’ lock + rpm + motion volledig naar STILL.
        """
        st = self._state

        if st.t_us is None:
            st.t_us = int(t_us)
            self._idle_start_us = int(t_us)
            return

        t_us = int(t_us)
        dt_us = t_us - st.t_us
        if dt_us <= 0:
            return

        # cumulatieve idle-tijd sinds laatste non-idle update
        if self._idle_start_us is None:
            self._idle_start_us = st.t_us
        total_idle_us = t_us - self._idle_start_us

        # 1) "Korte" idle: rpm & motion laten afbouwen
        if dt_us >= self.idle_timeout_us:
            # rpm langzaam kleiner maken
            decay = 0.8
            st.rpm_est *= decay
            st.rpm_inst *= decay

            # jitter/cadence resetten â€“ we kunnen niet vertrouwen op oude cadence
            self._rpm_window.clear()
            st.rpm_jitter = 0.0
            st.cadence_ok = False

            # Motion-state updaten op basis van lager rpm
            # -> geen nieuwe cycles, dus geen echte beweging meer
            if st.rpm_est < 1.0:
                st.rpm_est = 0.0
                st.rpm_inst = 0.0
                st.motion_state = "STATIC"
                st.motion_conf = 0.0
            else:
                # Tussenfase: niet meer "volle MOVING", maar EVALUATING
                st.motion_state = "EVALUATING"
                st.motion_conf = 0.5 * _clamp(st.rpm_est / self.rpm_move_thresh, 0.0, 1.0)

            # Flow neutraliseren â€“ geen actuele tegentrend meer zonder nieuwe tiles
            st.flow_state = "NEUTRAL"
            st.flow_score = 0.0
            st.resist_score = 0.0

            # Lock-confidence langzaam afbouwen
            st.direction_locked_conf *= self.idle_lock_decay
            if st.direction_locked_conf < 0.1:
                st.direction_lock_state = "UNLOCKED"

        # 2) "Lange" idle: volledige STILL-reset
        if total_idle_us >= self.idle_unlock_time_us:
            # alles wat met richting en beweging te maken heeft terug naar nul
            st.direction_lock_state = "UNLOCKED"
            st.direction_locked_dir = "UNDECIDED"
            st.direction_locked_conf = 0.0

            st.rpm_est = 0.0
            st.rpm_inst = 0.0
            st.motion_state = "STATIC"
            st.motion_conf = 0.0

            st.flow_state = "NEUTRAL"
            st.flow_score = 0.0
            st.resist_score = 0.0

        st.t_us = t_us

        # Awareness opnieuw berekenen
        self._update_awareness_conf()
        # Optioneel: awareness in diepe STILL niet boven een zachte vloer laten
        if st.motion_state == "STATIC" and st.direction_lock_state == "UNLOCKED":
            if st.awareness_conf < self.idle_awareness_floor:
                st.awareness_conf = self.idle_awareness_floor

    # ------------------------------------------------------------------
    # PureStep-feed (optioneel, placeholder)
    # ------------------------------------------------------------------

    def feed_pure_step(self, step: Dict[str, Any]) -> None:
        t_us = _get_time_us(step)
        if t_us is None:
            return
        self._state.t_us = t_us

    # ------------------------------------------------------------------
    # Awareness-confidence
    # ------------------------------------------------------------------

    def _update_awareness_conf(self) -> None:
        st = self._state

        dir_term = _clamp(st.direction_locked_conf, 0.0, 1.0)
        mot_term = _clamp(st.motion_conf, 0.0, 1.0)
        rpm_term = _clamp(st.rpm_est / self.awareness_rpm_norm, 0.0, 1.0)
        flow_term = _clamp(st.flow_score, 0.0, 1.0)
        resist_term = 1.0 - _clamp(st.resist_score, 0.0, 1.0)

        score = (
            0.30 * dir_term +
            0.25 * mot_term +
            0.20 * rpm_term +
            0.15 * flow_term +
            0.10 * resist_term
        )

        st.awareness_conf = _clamp(score, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def snapshot(self) -> MovementBodyStateV3:
        return MovementBodyStateV3(**self._state.__dict__.copy())


if __name__ == "__main__":
    # Mini rooktest â€“ integreert met een synthetische "inertial" snapshot
    mb = MovementBodyV3()

    # Simuleer een inertiaal snapshot zoals uit InertialCompassState
    snap_like = {
        "global_direction": "CW",
        "global_score": 0.5,         # signed score, hier positief
        "window_direction": "CW",
        "window_confidence": 0.9,
    }
    mb.set_compass_realtime(snap_like)

    # Een paar "cycles" met 5 ms tussenruimte (~1000 rpm bij C=12)
    t = 0
    for i in range(12):
        t += 5000  # Âµs
        mb.feed_cycle_node({"t_center_us": t, "id": i, "cycle_type": "cycle"})

    state = mb.snapshot()
    print(
        "lock_state:", state.direction_lock_state,
        "locked_dir:", state.direction_locked_dir,
        "rpm_est:", round(state.rpm_est, 1),
        "rotations:", round(state.rotations, 3),
        "theta_deg:", round(state.theta_deg, 1),
        "flow_state:", state.flow_state,
        "awareness_conf:", round(state.awareness_conf, 3),
    )
