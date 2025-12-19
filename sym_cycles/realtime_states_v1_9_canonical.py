#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_states_v1_9_canonical.py — Realtime Pipeline with CycleBuilder TruthProbe v0.4.1+v0.4.2

S02.CycleBuilder-TruthProbe — Reject Telemetry + Normalization + Trace.

Changelog v0.4.1:
- Reject reasons telemetry (cb_reject counter)
- Last reason tracking
- Pools tail buffer per sensor

Changelog v0.4.2:
- canon_event24() normalization function
- Key/type guards (string to_pool -> int, etc.)
- Trace buffer for debug

Chain: EVENT24 → CyclesState → TilesState → InertialCompass → MovementBodyV3
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, Counter
import statistics
import math

# === Constants ===
POOL_NEU = 0
POOL_N = 1
POOL_S = 2

# === Compatibility constants for tooling (replay/scripts) ===
ROTOR_STATE_STILL = "STILL"
ROTOR_STATE_MOVEMENT = "MOVEMENT"

LOCK_STATE_UNLOCKED = "UNLOCKED"
LOCK_STATE_SOFT_LOCK = "SOFT_LOCK"
LOCK_STATE_LOCKED = "LOCKED"

# === Canonicalization v0.4.2 ===

def canon_event24(ev: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    """
    Canonicalize event to standard keys/types.
    
    Returns: (ok, canonical_event, reject_reason)
    
    Canon keys:
    - kind: "event24" 
    - sensor: int 0 or 1
    - to_pool: int 0, 1, or 2
    - t_abs_us: int/float
    - dt_us: int/float
    """
    canon = {}
    
    # Kind
    kind = ev.get("kind")
    if kind is None:
        # Infer kind if has required fields
        if "sensor" in ev or "to_pool" in ev or "toPool" in ev:
            kind = "event24"
        else:
            return False, {}, "NO_EVENT_KIND"
    canon["kind"] = str(kind)
    
    # Sensor - normalize "A"/"B" to 0/1
    sensor = ev.get("sensor")
    if sensor is None:
        sensor = ev.get("sensorId") or ev.get("sensor_id")
    
    if sensor is None:
        return False, {}, "NO_SENSOR"
    
    if sensor == "A" or sensor == "a":
        sensor = 0
    elif sensor == "B" or sensor == "b":
        sensor = 1
    elif isinstance(sensor, str):
        try:
            sensor = int(sensor)
        except ValueError:
            return False, {}, "SENSOR_INVALID"
    elif isinstance(sensor, float):
        sensor = int(sensor)
    
    if sensor not in (0, 1):
        return False, {}, "SENSOR_INVALID"
    canon["sensor"] = sensor
    
    # to_pool - try multiple key names and normalize type
    to_pool = ev.get("to_pool")
    if to_pool is None:
        to_pool = ev.get("toPool") or ev.get("pool_to") or ev.get("pool")
    
    if to_pool is None:
        return False, {}, "NO_TO_POOL"
    
    # Type normalization
    if isinstance(to_pool, str):
        try:
            to_pool = int(to_pool)
        except ValueError:
            return False, {}, "TO_POOL_INVALID_TYPE"
    elif isinstance(to_pool, float):
        to_pool = int(to_pool)
    elif not isinstance(to_pool, int):
        return False, {}, "TO_POOL_INVALID_TYPE"
    
    if to_pool not in (0, 1, 2):
        return False, {}, "TO_POOL_OUT_OF_RANGE"
    canon["to_pool"] = to_pool
    
    # from_pool (optional)
    from_pool = ev.get("from_pool")
    if from_pool is None:
        from_pool = ev.get("fromPool") or ev.get("pool_from")
    if from_pool is not None:
        if isinstance(from_pool, str):
            try:
                from_pool = int(from_pool)
            except:
                from_pool = None
        elif isinstance(from_pool, float):
            from_pool = int(from_pool)
    canon["from_pool"] = from_pool
    
    # t_abs_us
    t_abs = ev.get("t_abs_us")
    if t_abs is None:
        t_abs = ev.get("tAbsUs") or ev.get("t_us") or ev.get("t")
    if t_abs is not None:
        try:
            t_abs = int(t_abs)
        except:
            t_abs = 0
    canon["t_abs_us"] = t_abs or 0
    
    # dt_us
    dt_us = ev.get("dt_us")
    if dt_us is None:
        dt_us = ev.get("dtUs") or ev.get("dt")
    if dt_us is not None:
        try:
            dt_us = int(dt_us)
        except:
            dt_us = 0
    canon["dt_us"] = dt_us or 0
    
    # Pass through other fields
    for k in ["flags0", "flags1", "polarity", "qlevel", "pair", "dir_hint", "edge_kind",
              "dvdt_q15", "mono_q8", "snr_q8", "fit_err_q8", "rpm_hint_q", "score_q8", "seq"]:
        if k in ev:
            canon[k] = ev[k]
    
    return True, canon, "CANON_OK"


# === Cycle Detection with TruthProbe ===

class CyclesState:
    """
    Incremental 3-point cycle detection per sensor.
    With v0.4.1/v0.4.2 TruthProbe: reject telemetry + trace.
    """
    
    def __init__(self, dt_min_us: int = 100, dt_max_us: int = 5_000_000):
        self._windows = {"A": deque(maxlen=3), "B": deque(maxlen=3)}
        self._cycle_counts = {"A": 0, "B": 0}
        self._dt_samples = {"A": [], "B": []}
        
        self._dt_min_us = dt_min_us
        self._dt_max_us = dt_max_us
        
        # === TruthProbe v0.4.1 ===
        self.cb_events_seen_total = 0
        self.cb_cycles_emitted_total = 0
        self.cb_reject = Counter()
        self.cb_last_reason = None
        self.cb_last_event = None
        self.cb_last_pools_tail_A = deque(maxlen=6)
        self.cb_last_pools_tail_B = deque(maxlen=6)
        
        # === v0.4.2 Canonicalization counters ===
        self.cb_canon_ok_total = 0
        self.cb_canon_fail_total = 0
        self.cb_canon_fail_reasons = Counter()
        
        # === v0.4.2 Trace buffer ===
        self._trace_armed = True
        self._trace_buffer = deque(maxlen=30)
        self._trace_arm_events = 40
    
    @staticmethod
    def _sensor_label(sensor_idx: int) -> Optional[str]:
        if sensor_idx == 0: return "A"
        if sensor_idx == 1: return "B"
        return None
    
    def _record_reject(self, reason: str, ev: Dict = None):
        """Record a reject reason."""
        self.cb_reject[reason] += 1
        self.cb_last_reason = reason
        if ev:
            self.cb_last_event = {k: ev.get(k) for k in ["sensor", "to_pool", "t_abs_us", "dt_us"]}
        
        # Trace
        if self._trace_armed:
            t = ev.get("t_abs_us", 0) if ev else 0
            s = ev.get("sensor", -1) if ev else -1
            p = ev.get("to_pool", -1) if ev else -1
            dt = ev.get("dt_us", 0) if ev else 0
            self._trace_buffer.append((t, s, p, dt, reason))
    
    def feed_event(self, ev: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process one event with canonicalization and reject tracking."""
        self.cb_events_seen_total += 1
        
        # === Canonicalize v0.4.2 ===
        ok, canon, reason = canon_event24(ev)
        
        if not ok:
            self.cb_canon_fail_total += 1
            self.cb_canon_fail_reasons[reason] += 1
            self._record_reject(reason, ev)
            return []
        
        self.cb_canon_ok_total += 1
        
        # Use canonical event
        ev = canon
        
        # Kind check
        if ev.get("kind") != "event24":
            self._record_reject("NO_EVENT_KIND", ev)
            return []
        
        # Sensor
        sensor_idx = ev.get("sensor")
        s_label = self._sensor_label(sensor_idx)
        if s_label is None:
            self._record_reject("SENSOR_INVALID", ev)
            return []
        
        # to_pool
        to_pool = ev.get("to_pool")
        if to_pool is None:
            self._record_reject("NO_TO_POOL", ev)
            return []
        
        # Record pool in tail buffer
        if s_label == "A":
            self.cb_last_pools_tail_A.append(to_pool)
        else:
            self.cb_last_pools_tail_B.append(to_pool)
        
        # t_abs_us
        t_us = ev.get("t_abs_us")
        if t_us is None:
            self._record_reject("NO_T_ABS_US", ev)
            return []
        
        # Window
        win = self._windows[s_label]
        
        # Same pool repeat check
        if len(win) > 0 and win[-1]["to_pool"] == to_pool:
            self._record_reject("SAME_POOL_REPEAT", ev)
            return []
        
        win.append({"t_us": int(t_us), "to_pool": int(to_pool)})
        
        cycles = []
        
        # 3-point cycle detection
        if len(win) == 3:
            p0, p1, p2 = (w["to_pool"] for w in win)
            t0, t1, t2 = (w["t_us"] for w in win)
            unique = {p0, p1, p2}
            
            if unique == {POOL_NEU, POOL_N, POOL_S}:
                dt = t2 - t0
                
                # DT bounds check
                if dt < self._dt_min_us:
                    self._record_reject("DT_TOO_SMALL", ev)
                    return []
                if dt > self._dt_max_us:
                    self._record_reject("DT_TOO_LARGE", ev)
                    return []
                
                # Determine cycle type
                if [p0, p1, p2] == [POOL_N, POOL_NEU, POOL_S]:
                    ctype = "cycle_up"
                elif [p0, p1, p2] == [POOL_S, POOL_NEU, POOL_N]:
                    ctype = "cycle_down"
                else:
                    ctype = "cycle_mixed"
                
                cycles.append({
                    "sensor": s_label,
                    "cycle_type": ctype,
                    "t_start_us": t0,
                    "t_end_us": t2,
                    "t_center_us": 0.5 * (t0 + t2),
                    "dt_us": dt,
                })
                
                self._cycle_counts[s_label] += 1
                self._dt_samples[s_label].append(float(dt))
                self.cb_cycles_emitted_total += 1
                self.cb_last_reason = "CYCLE_EMITTED"
                
                # Trace: stop arming after first cycle
                if self._trace_armed and self.cb_cycles_emitted_total > 0:
                    self._trace_armed = False
                
                # Record trace
                if self._trace_armed:
                    self._trace_buffer.append((t_us, sensor_idx, to_pool, dt, "CYCLE_EMITTED"))
            else:
                # Not a valid 3-pool pattern
                self._record_reject("SEQ_NOT_MATCH", ev)
        elif len(win) < 3:
            self._record_reject("WINDOW_NOT_READY", ev)
        
        # Arm trace after N events if still 0 cycles
        if (self.cb_events_seen_total >= self._trace_arm_events and 
            self.cb_cycles_emitted_total == 0 and not self._trace_armed):
            self._trace_armed = True
        
        return cycles
    
    def get_cb_debug(self) -> Dict[str, Any]:
        """Get CycleBuilder debug state for export."""
        # Top 3 reject reasons
        top3 = self.cb_reject.most_common(3)
        
        return {
            "events_total": self.cb_events_seen_total,
            "cycles_total": self.cb_cycles_emitted_total,
            "canon_ok": self.cb_canon_ok_total,
            "canon_fail": self.cb_canon_fail_total,
            "canon_fail_reasons": dict(self.cb_canon_fail_reasons),
            "last_reason": self.cb_last_reason,
            "last_event": self.cb_last_event,
            "reject_top3": {r: c for r, c in top3},
            "reject_all": dict(self.cb_reject),
            "pools_tail_A": list(self.cb_last_pools_tail_A),
            "pools_tail_B": list(self.cb_last_pools_tail_B),
            "trace": list(self._trace_buffer) if self._trace_buffer else [],
        }
    
    def debug_summary(self) -> Dict[str, Any]:
        out = {"per_sensor": {}}
        for s in ("A", "B"):
            dts = sorted(self._dt_samples[s])
            n = len(dts)
            if n:
                med = statistics.median(dts)
                mn, mx = dts[0], dts[-1]
            else:
                med = mn = mx = None
            out["per_sensor"][s] = {
                "cycles": self._cycle_counts[s],
                "dt_us_n": n,
                "dt_us_min": mn,
                "dt_us_median": med,
                "dt_us_max": mx,
            }
        out["cb_debug"] = self.get_cb_debug()
        return out


# === Tiles State ===

class TilesState:
    """Incremental tile building."""
    
    def __init__(self, tile_span_cycles: float = 1.0, boot_cycles_for_median: int = 24):
        self.tile_span_cycles = float(tile_span_cycles)
        self.boot_cycles_for_median = int(boot_cycles_for_median)
        
        self._boot_dt_samples = []
        self._tile_duration_us = None
        self._t0_us = None
        self._current_tile_index = None
        self._current_tile_data = {"A": [], "B": []}
        self._tiles_emitted = 0
    
    @property
    def tile_duration_us(self):
        return self._tile_duration_us
    
    def _observe_dt(self, cycle):
        dt = cycle.get("dt_us")
        if not isinstance(dt, (int, float)) or dt <= 0:
            return
        self._boot_dt_samples.append(float(dt))
        
        if (self._tile_duration_us is None and 
            len(self._boot_dt_samples) >= self.boot_cycles_for_median):
            median_dt = statistics.median(self._boot_dt_samples)
            if median_dt > 0:
                self._tile_duration_us = self.tile_span_cycles * median_dt
    
    def _tile_index_for_time(self, t_us):
        if self._t0_us is None:
            self._t0_us = t_us
        if not self._tile_duration_us or self._tile_duration_us <= 0:
            return 0
        rel = (t_us - self._t0_us) / self._tile_duration_us
        return 0 if rel < 0 else int(rel)
    
    def _flush_current_tile(self):
        if self._current_tile_index is None:
            return None
        
        idx = self._current_tile_index
        t_start = (self._t0_us + idx * self._tile_duration_us) if self._tile_duration_us else self._t0_us or 0
        t_end = t_start + (self._tile_duration_us or 0)
        
        ts_A = self._current_tile_data["A"]
        ts_B = self._current_tile_data["B"]
        
        nA, nB = len(ts_A), len(ts_B)
        cycles_phys = 0.5 * (nA + nB)
        
        tile = {
            "tile_index": idx,
            "t_start_us": t_start,
            "t_end_us": t_end,
            "t_center_us": 0.5 * (t_start + t_end),
            "nA": nA, "nB": nB,
            "cycles_physical": cycles_phys,
            "samples_A": ts_A,
            "samples_B": ts_B,
        }
        
        self._current_tile_data = {"A": [], "B": []}
        self._tiles_emitted += 1
        return tile
    
    def feed_cycles(self, cycles):
        tiles = []
        
        for cyc in cycles:
            self._observe_dt(cyc)
            
            if self._tile_duration_us is None:
                continue
            
            t_c = cyc.get("t_center_us", 0)
            new_idx = self._tile_index_for_time(t_c)
            
            if self._current_tile_index is None:
                self._current_tile_index = new_idx
            
            if new_idx != self._current_tile_index:
                flushed = self._flush_current_tile()
                if flushed:
                    tiles.append(flushed)
                self._current_tile_index = new_idx
            
            s = cyc.get("sensor", "A")
            if s in self._current_tile_data:
                self._current_tile_data[s].append({
                    "t_us": cyc.get("t_center_us"),
                    "cycle_type": cyc.get("cycle_type"),
                    "dt_us": cyc.get("dt_us"),
                })
        
        return tiles


# === Compass ===

@dataclass
class CompassSnapshot:
    global_score: float = 0.0
    conf: float = 0.0
    direction: str = "UNDECIDED"
    
    def to_dict(self):
        return {"global_score": self.global_score, "conf": self.conf, "direction": self.direction}


class InertialCompass:
    """Direction compass with EMA scoring."""
    
    def __init__(self, alpha: float = 0.15, threshold_high: float = 0.6, threshold_low: float = 0.3):
        self._alpha = alpha
        self._th_high = threshold_high
        self._th_low = threshold_low
        self._score = 0.0
        self._direction = "UNDECIDED"
    
    def feed_tile(self, tile):
        nA, nB = tile.get("nA", 0), tile.get("nB", 0)
        
        # Count up/down cycles
        ups = downs = 0
        for s in ["samples_A", "samples_B"]:
            for samp in tile.get(s, []):
                ct = samp.get("cycle_type", "")
                if ct == "cycle_up": ups += 1
                elif ct == "cycle_down": downs += 1
        
        total = ups + downs
        if total > 0:
            delta = (ups - downs) / total
            self._score = (1 - self._alpha) * self._score + self._alpha * delta
        
        # Direction decision
        if self._score >= self._th_high:
            self._direction = "CW"
        elif self._score <= -self._th_high:
            self._direction = "CCW"
        elif abs(self._score) < self._th_low:
            self._direction = "UNDECIDED"
    
    def snapshot(self):
        return CompassSnapshot(
            global_score=self._score,
            conf=abs(self._score),
            direction=self._direction,
        )


# === Movement Body ===

class MovementBody:
    """Simple movement state machine."""
    
    def __init__(self, rpm_move_thresh: float = 1.0, cycles_per_rot: float = 12.0):
        self._rpm_thresh = rpm_move_thresh
        self._cycles_per_rot = cycles_per_rot
        
        self._rotor_state = "STILL"
        self._lock_state = "UNLOCKED"
        self._total_cycles = 0.0
        self._rotations = 0.0
        self._theta_deg = 0.0
        self._rpm = 0.0
        self._last_t_us = None
        self._direction = "UNDECIDED"
        
        # Lock tracking
        self._consistent_dir_tiles = 0
    
    def feed_tile(self, tile, compass: CompassSnapshot):
        cycles_phys = tile.get("cycles_physical", 0)
        self._total_cycles += cycles_phys
        self._rotations = self._total_cycles / self._cycles_per_rot
        self._theta_deg = (self._rotations * 360.0) % 360.0
        
        # RPM estimate
        t_us = tile.get("t_center_us", 0)
        if self._last_t_us and t_us > self._last_t_us:
            dt_s = (t_us - self._last_t_us) / 1e6
            if dt_s > 0:
                cycles_per_s = cycles_phys / dt_s
                rpm = (cycles_per_s / self._cycles_per_rot) * 60
                self._rpm = 0.8 * self._rpm + 0.2 * rpm
        self._last_t_us = t_us
        
        # Rotor state
        if self._rpm >= self._rpm_thresh:
            self._rotor_state = "MOVEMENT"
        else:
            self._rotor_state = "STILL"
        
        # Direction tracking
        self._direction = compass.direction
        
        # Lock state
        if compass.direction in ("CW", "CCW"):
            self._consistent_dir_tiles += 1
            if self._consistent_dir_tiles >= 8:
                self._lock_state = "LOCKED"
            elif self._consistent_dir_tiles >= 4:
                self._lock_state = "SOFT_LOCK"
        else:
            self._consistent_dir_tiles = 0
            self._lock_state = "UNLOCKED"
    
    def snapshot(self):
        return {
            "rotor_state": self._rotor_state,
            "direction_lock_state": self._lock_state,
            "direction_global_effective": self._direction,
            "total_cycles_physical": self._total_cycles,
            "rotations": self._rotations,
            "theta_deg": self._theta_deg,
            "rpm_est": self._rpm,
            "cycle_index": int(self._total_cycles),
        }


# === Pipeline Profile ===

@dataclass
class PipelineProfile:
    name: str = "default"
    compass_alpha: float = 0.15
    compass_threshold_high: float = 0.6
    compass_threshold_low: float = 0.3
    compass_window_tiles: int = 8
    compass_deadzone_us: int = 1000
    compass_min_tiles: int = 2
    compass_max_abs_dt_us: int = 500000
    lock_confidence_threshold: float = 0.7
    lock_soft_threshold: float = 0.5
    unlock_tiles_threshold: int = 3
    rpm_alpha: float = 0.2
    jitter_max_rel: float = 0.5
    jitter_window_size: int = 5
    rpm_move_thresh: float = 1.0
    rpm_slow_thresh: float = 5.0
    rpm_still_thresh: float = 0.5
    tile_span_cycles: float = 1.0
    min_normal_tile: int = 2
    stereo_fusion: bool = True
    cycles_per_rot: float = 12.0
    dt_min_us: int = 100
    dt_max_us: int = 5_000_000


PROFILE_PRODUCTION = PipelineProfile(name="production")
PROFILE_BENCH = PipelineProfile(name="bench", compass_alpha=0.2, rpm_move_thresh=0.5)
PROFILE_BENCH_TOLERANT = PipelineProfile(
    name="bench_tolerant",
    compass_alpha=0.25,
    compass_threshold_high=0.4,
    compass_threshold_low=0.15,
    rpm_move_thresh=0.3,
    dt_min_us=50,
    dt_max_us=10_000_000,
)


def load_profile_from_xram(path: str, profile_name: str = "production") -> PipelineProfile:
    """Load profile from xram JSON."""
    import json
    with open(path) as f:
        xram = json.load(f)
    
    # Basic extraction - can be expanded
    return PROFILE_BENCH_TOLERANT


# === Pipeline Result ===

@dataclass
class PipelineResult:
    tiles_emitted: List[Dict[str, Any]]
    compass_snapshot: CompassSnapshot
    movement_state: Dict[str, Any]


# === Main Pipeline ===

class RealtimePipeline:
    """
    Complete realtime pipeline with CycleBuilder TruthProbe.
    
    EVENT24 → CyclesState → TilesState → InertialCompass → MovementBody
    """
    
    def __init__(self, profile: PipelineProfile = None):
        self._profile = profile or PROFILE_PRODUCTION
        
        self._cycles = CyclesState(
            dt_min_us=self._profile.dt_min_us,
            dt_max_us=self._profile.dt_max_us,
        )
        self._tiles = TilesState(
            tile_span_cycles=self._profile.tile_span_cycles,
        )
        self._compass = InertialCompass(
            alpha=self._profile.compass_alpha,
            threshold_high=self._profile.compass_threshold_high,
            threshold_low=self._profile.compass_threshold_low,
        )
        self._movement = MovementBody(
            rpm_move_thresh=self._profile.rpm_move_thresh,
            cycles_per_rot=self._profile.cycles_per_rot,
        )
        
        self._tiles_emitted_total = 0
    
    def feed_event(self, ev: Dict[str, Any]) -> PipelineResult:
        """Process one event through full pipeline."""
        # Cycles
        cycles = self._cycles.feed_event(ev)
        
        # Tiles
        tiles = self._tiles.feed_cycles(cycles)
        
        # Compass + Movement
        for tile in tiles:
            self._compass.feed_tile(tile)
            self._movement.feed_tile(tile, self._compass.snapshot())
            self._tiles_emitted_total += 1
        
        # Build result
        compass_snap = self._compass.snapshot()
        mv_state = self._movement.snapshot()
        
        # Add CycleBuilder debug (_cb) v0.4.1/v0.4.2
        mv_state["_cb"] = self._cycles.get_cb_debug()
        mv_state["_debug"] = {
            "pools_recent_A": list(self._cycles.cb_last_pools_tail_A),
            "pools_recent_B": list(self._cycles.cb_last_pools_tail_B),
            "cycles_emitted_n": len(cycles),
        }
        
        return PipelineResult(
            tiles_emitted=tiles,
            compass_snapshot=compass_snap,
            movement_state=mv_state,
        )
    
    def snapshot(self) -> PipelineResult:
        """Get current state without feeding event."""
        return PipelineResult(
            tiles_emitted=[],
            compass_snapshot=self._compass.snapshot(),
            movement_state=self._movement.snapshot(),
        )
    
    def get_cb_debug(self) -> Dict[str, Any]:
        """Get CycleBuilder debug state."""
        return self._cycles.get_cb_debug()
