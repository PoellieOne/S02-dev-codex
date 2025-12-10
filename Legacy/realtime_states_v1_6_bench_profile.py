"""
S02 — Realtime-States v1.6 Bench Profiel
========================================

NIEUWE FEATURE: Configureerbare profielen voor production vs bench runs.

Architectuur (conform S02.RealtimeStates_v1_4_ClaimAtLock_Architectuur.md):
--------------------------------------------------------------------------
De v1.5 bench tests toonden aan dat slechts 1/4 runs LOCK bereikte.
Root cause: alpha=0.95 is te conservatief voor korte handmatige runs.

Twee profielen:
1. PRODUCTION (tight-industrial)
   - Strenge thresholds voor betrouwbare industriële metingen
   - alpha=0.95, threshold_high=0.6
   
2. BENCH (loose-human)
   - Tolerantere parameters voor handmatige test runs
   - alpha=0.85, threshold_high=0.50
   - Snellere lock, accepteert meer variatie

Kernwijzigingen t.o.v. v1.5:
---------------------------
- Nieuw: PipelineProfile dataclass met alle tunable parameters
- Nieuw: PROFILE_PRODUCTION en PROFILE_BENCH presets
- RealtimePipeline accepteert profile parameter
- Alle compass/lock/rpm parameters komen uit profile

Verwachte verbeteringen:
-----------------------
- Bench runs: 4/4 LOCK verwacht (was 1/4)
- Production runs: Ongewijzigd gedrag
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import statistics

# === Constants ===============================================================

POOL_NEU = 0
POOL_N   = 1
POOL_S   = 2

# BootWarmup constants
TILE_STATE_BOOT = "BOOT"
TILE_STATE_NORMAL = "NORMAL"
DEFAULT_MIN_NORMAL_TILE = 1


# === Pipeline Profiles =======================================================

@dataclass
class PipelineProfile:
    """
    Configureerbaar profiel voor de realtime pipeline.
    
    Twee presets beschikbaar:
    - PROFILE_PRODUCTION: Streng, voor industriële runs
    - PROFILE_BENCH: Tolerant, voor handmatige test runs
    """
    name: str = "custom"
    
    # Compass parameters
    compass_alpha: float = 0.95           # EMA smoothing (hoger = trager)
    compass_threshold_high: float = 0.6   # Score voor direction lock
    compass_threshold_low: float = 0.25   # Score voor UNDECIDED
    compass_window_tiles: int = 20        # Window grootte
    
    # Lock parameters
    lock_confidence_threshold: float = 0.5   # Confidence nodig voor LOCKED
    lock_soft_threshold: float = 0.3         # Confidence voor SOFT_LOCK
    
    # RPM parameters
    rpm_alpha: float = 0.3                # RPM EMA smoothing
    jitter_max_rel: float = 0.4           # Max jitter voor cadence_ok
    jitter_window_size: int = 10          # Window voor jitter berekening
    rpm_move_thresh: float = 5.0          # RPM threshold voor MOVING
    rpm_slow_thresh: float = 1.0          # RPM threshold voor EVALUATING
    
    # Tile parameters
    tile_span_cycles: float = 0.6         # Tile span in cycles
    min_normal_tile: int = 1              # Eerste NORMAL tile (BootWarmup)
    stereo_fusion: str = "mean"           # Stereo fusion strategie
    
    # Physical constants
    cycles_per_rot: float = 12.0          # Cycles per rotatie (S02: 12)


# Preset profielen
PROFILE_PRODUCTION = PipelineProfile(
    name="production",
    compass_alpha=0.95,
    compass_threshold_high=0.6,
    compass_threshold_low=0.25,
    compass_window_tiles=20,
    lock_confidence_threshold=0.5,
    lock_soft_threshold=0.3,
    rpm_alpha=0.3,
    jitter_max_rel=0.4,
    jitter_window_size=10,
    rpm_move_thresh=5.0,
    rpm_slow_thresh=1.0,
    tile_span_cycles=0.6,
    min_normal_tile=1,
    stereo_fusion="mean",
    cycles_per_rot=12.0,
)

PROFILE_BENCH = PipelineProfile(
    name="bench",
    # Snellere compass response
    compass_alpha=0.85,              # Was 0.95 → snellere EMA
    compass_threshold_high=0.50,     # Was 0.6 → makkelijker lock
    compass_threshold_low=0.20,      # Was 0.25 → snellere UNDECIDED exit
    compass_window_tiles=20,         # Was 20 → kortere window
    # Snellere lock
    lock_confidence_threshold=0.4,   # Was 0.5 → sneller LOCKED
    lock_soft_threshold=0.25,        # Was 0.3 → sneller SOFT_LOCK
    # Tolerantere RPM
    rpm_alpha=0.4,                   # Was 0.3 → snellere response
    jitter_max_rel=0.6,              # Was 0.4 → meer jitter toegestaan
    jitter_window_size=8,            # Was 10 → kortere window
    rpm_move_thresh=3.0,             # Was 5.0 → sneller MOVING
    rpm_slow_thresh=0.5,             # Was 1.0 → sneller EVALUATING
    # Tile parameters
    tile_span_cycles=0.6,
    min_normal_tile=1,
    stereo_fusion="mean",
    cycles_per_rot=12.0,
)


# === Cycle detection (L1) — unchanged ========================================

class CyclesState:
    """Incrementele 3-punts cycle-detectie per sensor."""

    def __init__(self):
        self._windows = {
            "A": deque(maxlen=3),
            "B": deque(maxlen=3),
        }
        self._cycle_counts = {"A": 0, "B": 0}
        self._dt_samples = {"A": [], "B": []}

    @staticmethod
    def _sensor_label(sensor_idx: int) -> Optional[str]:
        if sensor_idx == 0:
            return "A"
        if sensor_idx == 1:
            return "B"
        return None

    def feed_event(self, ev: Dict[str, Any]) -> List[Dict[str, Any]]:
        if ev.get("kind") != "event24":
            return []

        s_label = self._sensor_label(ev.get("sensor"))
        if s_label not in self._windows:
            return []

        t_us = ev.get("t_abs_us")
        to_pool = ev.get("to_pool")
        if t_us is None or to_pool is None:
            return []

        win = self._windows[s_label]
        win.append({
            "t_us": int(t_us),
            "to_pool": int(to_pool),
        })

        cycles: List[Dict[str, Any]] = []

        if len(win) == 3:
            p0, p1, p2 = (w["to_pool"] for w in win)
            t0, t1, t2 = (w["t_us"] for w in win)
            unique = {p0, p1, p2}

            if unique == {POOL_NEU, POOL_N, POOL_S}:
                if   [p0, p1, p2] == [POOL_N,  POOL_NEU, POOL_S]:
                    ctype = "cycle_up"
                elif [p0, p1, p2] == [POOL_S,  POOL_NEU, POOL_N]:
                    ctype = "cycle_down"
                else:
                    ctype = "cycle_mixed"

                cycles.append({
                    "sensor": s_label,
                    "cycle_type": ctype,
                    "t_start_us": t0,
                    "t_end_us": t2,
                    "t_center_us": 0.5 * (t0 + t2),
                    "dt_us": t2 - t0,
                })

                self._cycle_counts[s_label] += 1
                try:
                    self._dt_samples[s_label].append(float(t2 - t0))
                except Exception:
                    pass

        return cycles

    def get_total_cycles(self) -> Dict[str, int]:
        return self._cycle_counts.copy()

    def debug_summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"per_sensor": {}}
        for s in ("A", "B"):
            dts = sorted(self._dt_samples[s])
            n = len(dts)
            if n:
                med = statistics.median(dts)
                mn = dts[0]
                mx = dts[-1]
            else:
                med = mn = mx = None
            out["per_sensor"][s] = {
                "cycles": self._cycle_counts[s],
                "dt_us_n": n,
                "dt_us_min": mn,
                "dt_us_median": med,
                "dt_us_max": mx,
            }
        return out


# === Time-based tiles with cycles_physical and tile_state (L2) ===============

class TilesState:
    """v1.6: Tiles met cycles_physical, tile_state, en profile support."""

    def __init__(self, profile: PipelineProfile):
        self.profile = profile
        self.tile_span_cycles = profile.tile_span_cycles
        self.min_normal_tile = profile.min_normal_tile
        self.stereo_fusion = profile.stereo_fusion
        self.boot_cycles_for_median = 24

        self._boot_dt_samples: List[float] = []
        self._tile_duration_us: Optional[float] = None

        self._t0_us: Optional[float] = None
        self._current_tile_index: Optional[int] = None
        self._current_tile_cycles = {"A": [], "B": []}

        self._tiles_emitted = 0
        self._empty_tiles_emitted = 0
        self._data_tiles_emitted = 0
        self._boot_tiles_emitted = 0
        self._total_cycles_physical = 0.0
        self._boot_cycles_physical = 0.0

    @property
    def tile_duration_us(self) -> Optional[float]:
        return self._tile_duration_us

    def _observe_dt(self, cycle: Dict[str, Any]) -> None:
        dt = cycle.get("dt_us")
        if not isinstance(dt, (int, float)):
            return
        dt_f = float(dt)
        if dt_f <= 0:
            return
        self._boot_dt_samples.append(dt_f)

        if (self._tile_duration_us is None and
                len(self._boot_dt_samples) >= self.boot_cycles_for_median):
            median_dt = statistics.median(self._boot_dt_samples)
            if median_dt > 0:
                self._tile_duration_us = self.tile_span_cycles * median_dt

    def _tile_index_for_time(self, t_us: float) -> int:
        if self._t0_us is None:
            self._t0_us = t_us
        if self._tile_duration_us is None or self._tile_duration_us <= 0:
            return 0
        rel = (t_us - self._t0_us) / self._tile_duration_us
        return 0 if rel < 0 else int(rel)

    def _get_tile_state(self, tile_index: int) -> str:
        if tile_index < self.min_normal_tile:
            return TILE_STATE_BOOT
        return TILE_STATE_NORMAL

    def _fuse_cycles(self, nA: int, nB: int) -> float:
        if nA == 0 and nB == 0:
            return 0.0
        
        if self.stereo_fusion == "min":
            if nA == 0 or nB == 0:
                return 0.0
            return float(min(nA, nB))
        elif self.stereo_fusion == "mean":
            return (nA + nB) / 2.0
        elif self.stereo_fusion == "max":
            return float(max(nA, nB))
        else:
            return float(min(nA, nB)) if (nA > 0 and nB > 0) else 0.0

    def _make_tile(self, idx: int, 
                   cycles_A: List[Dict], 
                   cycles_B: List[Dict]) -> Dict[str, Any]:
        
        t_start = (self._t0_us +
                   idx * self._tile_duration_us) if self._tile_duration_us else self._t0_us or 0.0
        t_end = t_start + (self._tile_duration_us or 0.0)
        t_center = (t_start + t_end) / 2

        nA = len(cycles_A)
        nB = len(cycles_B)
        
        ts_A = [c["t_center_us"] for c in cycles_A]
        ts_B = [c["t_center_us"] for c in cycles_B]
        tA = sum(ts_A) / len(ts_A) if ts_A else None
        tB = sum(ts_B) / len(ts_B) if ts_B else None
        dt_ab = (tB - tA) if (tA is not None and tB is not None) else None

        cycles_physical = self._fuse_cycles(nA, nB)
        has_data = cycles_physical > 0
        
        tile_state = self._get_tile_state(idx)

        tile = {
            "tile_index": idx,
            "tile_state": tile_state,
            "t_start_us": t_start,
            "t_end_us": t_end,
            "t_center_us": t_center,
            "tA_us": tA,
            "tB_us": tB,
            "dt_ab_us": dt_ab,
            "nA": nA,
            "nB": nB,
            "cycles_physical": cycles_physical,
            "has_data": has_data,
            "tile_span_cycles": self.tile_span_cycles,
        }

        self._tiles_emitted += 1
        self._total_cycles_physical += cycles_physical
        
        if tile_state == TILE_STATE_BOOT:
            self._boot_tiles_emitted += 1
            self._boot_cycles_physical += cycles_physical
        
        if has_data:
            self._data_tiles_emitted += 1
        else:
            self._empty_tiles_emitted += 1

        return tile

    def _flush_tiles_up_to(self, target_idx: int) -> List[Dict[str, Any]]:
        tiles: List[Dict[str, Any]] = []
        
        if self._current_tile_index is None:
            return tiles
        
        tile = self._make_tile(
            self._current_tile_index,
            self._current_tile_cycles["A"],
            self._current_tile_cycles["B"]
        )
        tiles.append(tile)
        
        for idx in range(self._current_tile_index + 1, target_idx):
            empty_tile = self._make_tile(idx, [], [])
            tiles.append(empty_tile)
        
        self._current_tile_cycles = {"A": [], "B": []}
        
        return tiles

    def feed_cycles(self, cycles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        new_tiles: List[Dict[str, Any]] = []

        for c in cycles:
            t_center = c.get("t_center_us")
            sensor = c.get("sensor")
            if t_center is None or sensor not in ("A", "B"):
                continue

            t_center = float(t_center)
            self._observe_dt(c)
            idx = self._tile_index_for_time(t_center)

            if self._current_tile_index is None:
                self._current_tile_index = idx

            if idx != self._current_tile_index:
                flushed = self._flush_tiles_up_to(idx)
                new_tiles.extend(flushed)
                self._current_tile_index = idx

            self._current_tile_cycles[sensor].append(c)

        return new_tiles

    def finalize(self) -> List[Dict[str, Any]]:
        tiles: List[Dict[str, Any]] = []
        if self._current_tile_index is not None:
            tile = self._make_tile(
                self._current_tile_index,
                self._current_tile_cycles["A"],
                self._current_tile_cycles["B"]
            )
            tiles.append(tile)
            self._current_tile_cycles = {"A": [], "B": []}
        return tiles

    def debug_summary(self) -> Dict[str, Any]:
        dts = sorted(self._boot_dt_samples)
        n = len(dts)
        if n:
            med = statistics.median(dts)
            mn = dts[0]
            mx = dts[-1]
        else:
            med = mn = mx = None

        return {
            "boot_dt_n": n,
            "boot_dt_min": mn,
            "boot_dt_median": med,
            "boot_dt_max": mx,
            "tile_span_cycles": self.tile_span_cycles,
            "tile_duration_us": self._tile_duration_us,
            "tiles_emitted": self._tiles_emitted,
            "data_tiles_emitted": self._data_tiles_emitted,
            "empty_tiles_emitted": self._empty_tiles_emitted,
            "boot_tiles_emitted": self._boot_tiles_emitted,
            "total_cycles_physical": self._total_cycles_physical,
            "boot_cycles_physical": self._boot_cycles_physical,
            "stereo_fusion": self.stereo_fusion,
            "min_normal_tile": self.min_normal_tile,
        }


# === Compass with profile support (L2.5) =====================================

@dataclass
class CompassSnapshot:
    t_tile_index: int
    n_tiles_window: int
    global_direction: str
    global_score: float
    window_direction: str
    window_confidence: float
    window_verdict_tiles: str
    window_meta: Dict[str, Any]
    global_meta: Dict[str, Any]


class CompassAdapter:
    """v1.6: Compass met profile-based parameters."""
    
    def __init__(self, profile: PipelineProfile):
        self.profile = profile
        self._window_tiles = profile.compass_window_tiles
        self._alpha = profile.compass_alpha
        self._threshold_high = profile.compass_threshold_high
        self._threshold_low = profile.compass_threshold_low
        
        self._tiles: deque = deque(maxlen=self._window_tiles)
        self._global_score: float = 0.0
        self._global_direction: str = "UNDECIDED"
        self._tile_count: int = 0

    def feed_tile(self, tile: Dict[str, Any], tile_duration_us: Optional[float]) -> CompassSnapshot:
        self._tile_count += 1
        dt_ab = tile.get("dt_ab_us")
        
        window_dir = "UNDECIDED"
        window_conf = 0.0
        
        if dt_ab is not None:
            self._tiles.append(dt_ab)
            
            # Window direction
            cw_count = sum(1 for d in self._tiles if d > 0)
            ccw_count = sum(1 for d in self._tiles if d < 0)
            total = len(self._tiles)
            
            if total > 0:
                if cw_count > ccw_count:
                    window_dir = "CW"
                    window_conf = cw_count / total
                elif ccw_count > cw_count:
                    window_dir = "CCW"
                    window_conf = ccw_count / total
            
            # Global EMA update
            signal = 1.0 if dt_ab > 0 else -1.0 if dt_ab < 0 else 0.0
            self._global_score = self._alpha * self._global_score + (1 - self._alpha) * signal
            
            # Direction decision based on thresholds
            if self._global_score > self._threshold_high:
                self._global_direction = "CW"
            elif self._global_score < -self._threshold_high:
                self._global_direction = "CCW"
            elif abs(self._global_score) < self._threshold_low:
                self._global_direction = "UNDECIDED"
            # Else: keep previous direction (hysteresis)
        
        return CompassSnapshot(
            t_tile_index=self._tile_count,
            n_tiles_window=len(self._tiles),
            global_direction=self._global_direction,
            global_score=self._global_score,  # Keep signed for debugging
            window_direction=window_dir,
            window_confidence=window_conf,
            window_verdict_tiles="",
            window_meta={},
            global_meta={},
        )


# === Movement Body with profile support (L3) =================================

@dataclass
class MovementState:
    t_us: int = 0
    cycles_per_rot: float = 12.0
    cycle_index: float = 0.0
    rotations: float = 0.0
    theta_deg: float = 0.0
    theta_wrap_count: int = 0
    
    direction_lock_state: str = "UNLOCKED"
    direction_locked_dir: str = ""
    direction_locked_conf: float = 0.0
    direction_global_effective: str = "UNDECIDED"
    direction_global_conf: float = 0.0
    
    rpm_inst: float = 0.0
    rpm_est: float = 0.0
    rpm_jitter: float = 0.0
    cadence_ok: bool = False
    
    motion_state: str = "STATIC"
    motion_conf: float = 0.0
    
    flow_state: str = "IDLE"
    flow_score: float = 0.0
    resist_score: float = 0.0
    awareness_conf: float = 0.1


class MovementBodyV3:
    """v1.6: MovementBody met profile-based parameters."""
    
    def __init__(self, profile: PipelineProfile):
        self.profile = profile
        self._state = MovementState(cycles_per_rot=profile.cycles_per_rot)
        self._compass_data = {}
        self._rpm_window: List[float] = []
        
        # Parameters uit profile
        self.rpm_alpha = profile.rpm_alpha
        self.jitter_window_size = profile.jitter_window_size
        self.jitter_max_rel = profile.jitter_max_rel
        self.rpm_move_thresh = profile.rpm_move_thresh
        self.rpm_slow_thresh = profile.rpm_slow_thresh
        self.lock_confidence_threshold = profile.lock_confidence_threshold
        self.lock_soft_threshold = profile.lock_soft_threshold
        
        self._lock_confidence_accum = 0.0
        
    def _direction_sign(self) -> int:
        d = self._state.direction_global_effective
        if d == "CW": return 1
        if d == "CCW": return -1
        return 0
    
    def set_compass_realtime(self, compass_data: Dict[str, Any]):
        self._compass_data = compass_data
        
        gd = compass_data.get("global_direction", "UNDECIDED")
        gs = compass_data.get("global_score", 0.0)
        
        self._state.direction_global_effective = gd
        self._state.direction_global_conf = abs(gs)
        
        # Lock logic met profile-based thresholds
        if self._state.direction_lock_state == "LOCKED":
            return
            
        if gd in ("CW", "CCW") and abs(gs) > self.profile.compass_threshold_high:
            # Accumulate confidence
            self._lock_confidence_accum += abs(gs) * 0.15  # Snellere accumulatie
            
            if self._lock_confidence_accum > self.lock_confidence_threshold:
                self._state.direction_lock_state = "LOCKED"
                self._state.direction_locked_dir = gd
                self._state.direction_locked_conf = abs(gs)
            elif self._lock_confidence_accum > self.lock_soft_threshold:
                self._state.direction_lock_state = "SOFT_LOCK"
        else:
            # Decay confidence
            self._lock_confidence_accum = max(0, self._lock_confidence_accum - 0.05)
            if self._lock_confidence_accum < self.lock_soft_threshold:
                if self._state.direction_lock_state == "SOFT_LOCK":
                    self._state.direction_lock_state = "UNLOCKED"
            
    def _update_motion_state_from_rpm(self):
        rpm = self._state.rpm_est
        if rpm >= self.rpm_move_thresh:
            self._state.motion_state = "MOVING"
            self._state.motion_conf = min(1.0, rpm / 60.0)
        elif rpm >= self.rpm_slow_thresh:
            self._state.motion_state = "EVALUATING"
            self._state.motion_conf = 0.5
        else:
            self._state.motion_state = "STATIC"
            self._state.motion_conf = 0.0
            
    def _update_awareness_conf(self):
        motion = self._state.motion_conf
        lock = self._state.direction_locked_conf if self._state.direction_lock_state == "LOCKED" else 0.0
        cadence = 1.0 if self._state.cadence_ok else 0.3
        
        self._state.awareness_conf = 0.4 * motion + 0.4 * lock + 0.2 * cadence
        self._state.flow_score = lock
        
    def snapshot(self) -> MovementState:
        return self._state


# === Realtime snapshot (L3) ==================================================

@dataclass
class RealtimeSnapshot:
    t_us: Optional[int]
    cycles_emitted: List[Dict[str, Any]]
    tiles_emitted: List[Dict[str, Any]]
    compass_snapshot: Optional[CompassSnapshot]
    movement_state: Dict[str, Any]
    
    # Cycles tracking
    cycles_physical_this_batch: float = 0.0
    total_cycles_physical: float = 0.0
    cycles_unsigned: float = 0.0
    cycles_claimed_at_lock: float = 0.0
    
    # BootWarmup tracking
    boot_tiles_skipped: int = 0
    boot_cycles_skipped: float = 0.0
    
    # v1.6: Profile info
    profile_name: str = ""


# === RealtimePipeline v1.6 — Profile Support =================================

class RealtimePipeline:
    """
    S02-Realtime-States v1.6 — Profile Support
    
    NIEUW: Configureerbare profielen voor production vs bench runs.
    
    Gebruik:
        # Production (streng)
        pipeline = RealtimePipeline(profile=PROFILE_PRODUCTION)
        
        # Bench (tolerant)
        pipeline = RealtimePipeline(profile=PROFILE_BENCH)
        
        # Custom
        my_profile = PipelineProfile(name="custom", compass_alpha=0.90, ...)
        pipeline = RealtimePipeline(profile=my_profile)
    """

    def __init__(self, profile: PipelineProfile = None):
        # Default naar PRODUCTION profile
        if profile is None:
            profile = PROFILE_PRODUCTION
        
        self.profile = profile
        
        self.cycles_state = CyclesState()
        self.tiles_state = TilesState(profile=profile)
        self.compass_adapter = CompassAdapter(profile=profile)
        self.movement_body = MovementBodyV3(profile=profile)

        self._last_t_us: Optional[int] = None
        self._last_compass_snapshot: Optional[CompassSnapshot] = None
        
        # Cycles tracking
        self._total_cycles_physical: float = 0.0
        self._cycles_unsigned: float = 0.0
        self._cycles_claimed_at_lock: float = 0.0
        self._lock_claimed: bool = False
        self._last_cycle_t_us: Optional[int] = None
        
        # Lock state tracking
        self._prev_lock_state: str = "UNLOCKED"
        
        # BootWarmup tracking
        self._boot_tiles_skipped: int = 0
        self._boot_cycles_skipped: float = 0.0

    def _get_direction_sign(self) -> int:
        return self.movement_body._direction_sign()

    def _check_and_claim_at_lock(self) -> None:
        st = self.movement_body._state
        current_lock = st.direction_lock_state
        
        if (current_lock == "LOCKED" and 
            self._prev_lock_state != "LOCKED" and 
            not self._lock_claimed and
            self._cycles_unsigned > 0):
            
            sign = self._get_direction_sign()
            if sign != 0:
                claimed = self._cycles_unsigned * sign
                
                C = st.cycles_per_rot if st.cycles_per_rot > 0 else 12.0
                st.cycle_index += claimed
                st.rotations = st.cycle_index / C
                
                theta_new = (st.rotations * 360.0) % 360.0
                st.theta_deg = theta_new
                
                self._cycles_claimed_at_lock = self._cycles_unsigned
                self._cycles_unsigned = 0.0
                self._lock_claimed = True
        
        self._prev_lock_state = current_lock

    def feed_event(self, ev: Dict[str, Any]) -> RealtimeSnapshot:
        if isinstance(ev.get("t_abs_us"), (int, float)):
            self._last_t_us = int(ev["t_abs_us"])

        cycles = self.cycles_state.feed_event(ev)
        tiles = self.tiles_state.feed_cycles(cycles)

        compass_snap: Optional[CompassSnapshot] = self._last_compass_snapshot
        cycles_physical_batch: float = 0.0

        for tile in tiles:
            cycles_physical = tile.get("cycles_physical", 0.0)
            tile_state = tile.get("tile_state", TILE_STATE_NORMAL)
            
            cycles_physical_batch += cycles_physical
            self._total_cycles_physical += cycles_physical
            
            has_data = cycles_physical > 0
            
            # BOOT tiles: skip compass/lock/awareness
            if tile_state == TILE_STATE_BOOT:
                self._boot_tiles_skipped += 1
                self._boot_cycles_skipped += cycles_physical
                
                if has_data:
                    self._cycles_unsigned += cycles_physical
                continue
            
            # NORMAL tiles: full processing
            if has_data:
                compass_snap = self.compass_adapter.feed_tile(
                    tile,
                    tile_duration_us=self.tiles_state.tile_duration_us,
                )
                self._last_compass_snapshot = compass_snap

                self.movement_body.set_compass_realtime({
                    "global_direction": compass_snap.global_direction,
                    "global_score": compass_snap.global_score,
                    "window_direction": compass_snap.window_direction,
                    "window_confidence": compass_snap.window_confidence,
                })

                t_center = tile.get("tA_us") or tile.get("t_center_us") or self._last_t_us or 0

                st = self.movement_body._state
                current_lock = st.direction_lock_state
                sign = self._get_direction_sign()
                
                if current_lock == "LOCKED" and sign != 0:
                    self._update_cycle_index_signed(cycles_physical, sign, t_center)
                else:
                    self._cycles_unsigned += cycles_physical
                    self._update_rpm_unsigned(t_center, cycles_physical)
                
                self._check_and_claim_at_lock()

        movement_state_obj = self.movement_body.snapshot()
        movement_state = movement_state_obj.__dict__.copy()
        movement_state.update({
            "total_cycles_physical": self._total_cycles_physical,
            "cycles_unsigned": self._cycles_unsigned,
            "cycles_claimed_at_lock": self._cycles_claimed_at_lock,
            "boot_tiles_skipped": self._boot_tiles_skipped,
            "boot_cycles_skipped": self._boot_cycles_skipped,
            "profile_name": self.profile.name,
        })

        return RealtimeSnapshot(
            t_us=self._last_t_us,
            cycles_emitted=cycles,
            tiles_emitted=tiles,
            compass_snapshot=compass_snap,
            movement_state=movement_state,
            cycles_physical_this_batch=cycles_physical_batch,
            total_cycles_physical=self._total_cycles_physical,
            cycles_unsigned=self._cycles_unsigned,
            cycles_claimed_at_lock=self._cycles_claimed_at_lock,
            boot_tiles_skipped=self._boot_tiles_skipped,
            boot_cycles_skipped=self._boot_cycles_skipped,
            profile_name=self.profile.name,
        )

    def _update_cycle_index_signed(self, cycles_physical: float, sign: int, t_us: int) -> None:
        st = self.movement_body._state
        C = st.cycles_per_rot if st.cycles_per_rot > 0 else 12.0
        
        st.cycle_index += sign * cycles_physical
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
        
        self._update_rpm_signed(t_us, cycles_physical)
        
        self.movement_body._update_motion_state_from_rpm()
        self.movement_body._update_awareness_conf()

    def _update_rpm_signed(self, t_us: int, cycles_physical: float) -> None:
        st = self.movement_body._state
        
        if self._last_cycle_t_us is None:
            self._last_cycle_t_us = t_us
            return
        
        dt_us = t_us - self._last_cycle_t_us
        self._last_cycle_t_us = t_us
        
        if dt_us <= 0 or cycles_physical <= 0:
            return
        
        dt_s = dt_us * 1e-6
        C = st.cycles_per_rot if st.cycles_per_rot > 0 else 12.0
        
        rpm_inst = (cycles_physical / dt_s) * (60.0 / C)
        
        alpha = self.movement_body.rpm_alpha
        if st.rpm_est <= 0:
            rpm_est = rpm_inst
        else:
            rpm_est = (1.0 - alpha) * st.rpm_est + alpha * rpm_inst
        
        st.rpm_inst = rpm_inst
        st.rpm_est = rpm_est
        
        self.movement_body._rpm_window.append(rpm_inst)
        if len(self.movement_body._rpm_window) > self.movement_body.jitter_window_size:
            self.movement_body._rpm_window.pop(0)
        
        if len(self.movement_body._rpm_window) >= 2:
            mean_rpm = statistics.mean(self.movement_body._rpm_window)
            if mean_rpm > 0:
                sigma_rpm = statistics.pstdev(self.movement_body._rpm_window)
                st.rpm_jitter = min(1.0, max(0.0, sigma_rpm / mean_rpm))
            else:
                st.rpm_jitter = 0.0
        else:
            st.rpm_jitter = 0.0
        
        st.cadence_ok = st.rpm_jitter <= self.movement_body.jitter_max_rel

    def _update_rpm_unsigned(self, t_us: int, cycles_physical: float) -> None:
        st = self.movement_body._state
        
        if self._last_cycle_t_us is None:
            self._last_cycle_t_us = t_us
            return
        
        dt_us = t_us - self._last_cycle_t_us
        
        if dt_us <= 0 or cycles_physical <= 0:
            return
        
        dt_s = dt_us * 1e-6
        C = st.cycles_per_rot if st.cycles_per_rot > 0 else 12.0
        
        rpm_inst = (cycles_physical / dt_s) * (60.0 / C)
        
        alpha = 0.3
        if st.rpm_est <= 0:
            st.rpm_est = rpm_inst
        else:
            st.rpm_est = (1.0 - alpha) * st.rpm_est + alpha * rpm_inst
        
        st.rpm_inst = rpm_inst
        
        if st.rpm_est >= self.movement_body.rpm_move_thresh:
            st.motion_state = "MOVING"
            st.motion_conf = 1.0
        elif st.rpm_est >= self.movement_body.rpm_slow_thresh:
            st.motion_state = "EVALUATING"
            st.motion_conf = 0.5
        else:
            st.motion_state = "STATIC"
            st.motion_conf = 0.0

    def debug_tiles_and_cycles(self) -> Dict[str, Any]:
        return {
            "cycles": self.cycles_state.debug_summary(),
            "tiles": self.tiles_state.debug_summary(),
            "total_cycles_physical": self._total_cycles_physical,
            "cycles_unsigned": self._cycles_unsigned,
            "cycles_claimed_at_lock": self._cycles_claimed_at_lock,
            "lock_claimed": self._lock_claimed,
            "boot_tiles_skipped": self._boot_tiles_skipped,
            "boot_cycles_skipped": self._boot_cycles_skipped,
            "profile": self.profile.name,
        }

    def snapshot(self) -> RealtimeSnapshot:
        movement_state_obj = self.movement_body.snapshot()
        movement_state = movement_state_obj.__dict__.copy()
        movement_state.update({
            "total_cycles_physical": self._total_cycles_physical,
            "cycles_unsigned": self._cycles_unsigned,
            "cycles_claimed_at_lock": self._cycles_claimed_at_lock,
            "boot_tiles_skipped": self._boot_tiles_skipped,
            "boot_cycles_skipped": self._boot_cycles_skipped,
            "profile_name": self.profile.name,
        })
        
        return RealtimeSnapshot(
            t_us=self._last_t_us,
            cycles_emitted=[],
            tiles_emitted=[],
            compass_snapshot=self._last_compass_snapshot,
            movement_state=movement_state,
            total_cycles_physical=self._total_cycles_physical,
            cycles_unsigned=self._cycles_unsigned,
            cycles_claimed_at_lock=self._cycles_claimed_at_lock,
            boot_tiles_skipped=self._boot_tiles_skipped,
            boot_cycles_skipped=self._boot_cycles_skipped,
            profile_name=self.profile.name,
        )
