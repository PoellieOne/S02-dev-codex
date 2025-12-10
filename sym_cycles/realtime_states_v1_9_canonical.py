"""
S02 — Realtime-States v1.9 Canonical
====================================

CANONICAL ARCHITECTUUR volgens SoRa afstemming met Sophia (OpenAI).

Kernprincipes:
-------------
1. CompassSign v3 blijft canonical voor confidence berekening
2. Hiërarchische RotorState: STILL → MOVEMENT (met lock substates)
3. Unlock mechanisme: LOCKED kan terugvallen naar SOFT_LOCK → UNLOCKED
4. STILL als default state bij rpm → 0

State Machine:
-------------
```
RotorState (top-level)
├── STILL (default)
│   └── rpm < rpm_still_thresh
│   └── direction = UNDECIDED
│   └── geen cycles accumulation
└── MOVEMENT
    ├── UNLOCKED
    │   └── accumulate unsigned cycles
    │   └── compass_score < threshold_high
    ├── SOFT_LOCK  
    │   └── tentative direction
    │   └── threshold_low < compass_score < threshold_high
    └── LOCKED
        └── committed direction
        └── compass_score >= threshold_high (stabiel)
        └── Claim-at-Lock bij entry
        └── unlock bij score daling
```

Unlock Regels:
-------------
- LOCKED → SOFT_LOCK: als compass_score < threshold_low
- SOFT_LOCK → UNLOCKED: als score laag blijft OF cadence instort
- MOVEMENT → STILL: als rpm_est < rpm_still_thresh

Compatibiliteit:
---------------
- CompassSign v3 (canonical)
- PhaseTiles v3 (dt_ab_us)
- MovementBody v3
- Claim-at-Lock
- BootWarmup
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

# Tile states
TILE_STATE_BOOT = "BOOT"
TILE_STATE_NORMAL = "NORMAL"

# Rotor states (top-level)
ROTOR_STATE_STILL = "STILL"
ROTOR_STATE_MOVEMENT = "MOVEMENT"

# Direction lock states (within MOVEMENT)
LOCK_STATE_UNLOCKED = "UNLOCKED"
LOCK_STATE_SOFT_LOCK = "SOFT_LOCK"
LOCK_STATE_LOCKED = "LOCKED"


# === Pipeline Profiles =======================================================

@dataclass
class PipelineProfile:
    """Configureerbaar profiel voor de realtime pipeline."""
    name: str = "custom"
    
    # Compass parameters (voor InertialCompass EMA)
    compass_alpha: float = 0.95
    compass_threshold_high: float = 0.6
    compass_threshold_low: float = 0.25
    compass_window_tiles: int = 20
    
    # CompassSign v3 parameters
    compass_deadzone_us: float = 100.0
    compass_min_tiles: int = 3
    compass_max_abs_dt_us: Optional[float] = None
    
    # Lock parameters
    lock_confidence_threshold: float = 0.5
    lock_soft_threshold: float = 0.3
    unlock_tiles_threshold: int = 3  # Aantal tiles onder threshold voor unlock
    
    # RPM parameters
    rpm_alpha: float = 0.3
    jitter_max_rel: float = 0.4
    jitter_window_size: int = 10
    rpm_move_thresh: float = 5.0
    rpm_slow_thresh: float = 1.0
    rpm_still_thresh: float = 1.0  # Onder deze waarde → STILL
    
    # Tile parameters
    tile_span_cycles: float = 0.6
    min_normal_tile: int = 1
    stereo_fusion: str = "mean"
    
    # Physical constants
    cycles_per_rot: float = 12.0


# Preset profielen
PROFILE_PRODUCTION = PipelineProfile(
    name="production",
    compass_alpha=0.95,
    compass_threshold_high=0.6,
    compass_threshold_low=0.25,
    compass_window_tiles=20,
    compass_deadzone_us=100.0,
    compass_min_tiles=3,
    compass_max_abs_dt_us=None,
    lock_confidence_threshold=0.5,
    lock_soft_threshold=0.3,
    unlock_tiles_threshold=3,
    rpm_alpha=0.3,
    jitter_max_rel=0.4,
    jitter_window_size=10,
    rpm_move_thresh=5.0,
    rpm_slow_thresh=1.0,
    rpm_still_thresh=1.0,
    tile_span_cycles=0.6,
    min_normal_tile=1,
    stereo_fusion="mean",
    cycles_per_rot=12.0,
)

PROFILE_BENCH = PipelineProfile(
    name="bench",
    compass_alpha=0.90,
    compass_threshold_high=0.55,
    compass_threshold_low=0.20,
    compass_window_tiles=20,
    compass_deadzone_us=100.0,
    compass_min_tiles=3,
    compass_max_abs_dt_us=None,
    lock_confidence_threshold=0.45,
    lock_soft_threshold=0.25,
    unlock_tiles_threshold=3,
    rpm_alpha=0.35,
    jitter_max_rel=0.5,
    jitter_window_size=10,
    rpm_move_thresh=3.0,
    rpm_slow_thresh=0.5,
    rpm_still_thresh=0.5,
    tile_span_cycles=0.6,
    min_normal_tile=1,
    stereo_fusion="mean",
    cycles_per_rot=12.0,
)

PROFILE_BENCH_TOLERANT = PipelineProfile(
    name="bench_tolerant",
    compass_alpha=0.85,
    compass_threshold_high=0.45,
    compass_threshold_low=0.15,
    compass_window_tiles=15,
    compass_deadzone_us=50.0,
    compass_min_tiles=2,
    compass_max_abs_dt_us=None,
    lock_confidence_threshold=0.35,
    lock_soft_threshold=0.20,
    unlock_tiles_threshold=4,
    rpm_alpha=0.4,
    jitter_max_rel=0.6,
    jitter_window_size=8,
    rpm_move_thresh=2.0,
    rpm_slow_thresh=0.3,
    rpm_still_thresh=0.3,
    tile_span_cycles=0.6,
    min_normal_tile=1,
    stereo_fusion="mean",
    cycles_per_rot=12.0,
)


def load_profile_from_xram(xram_path: str, profile_name: str) -> PipelineProfile:
    """Load a PipelineProfile from xram JSON configuration."""
    import json
    
    with open(xram_path, 'r') as f:
        xram = json.load(f)
    
    profiles = xram.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"Profile '{profile_name}' not found in xram. Available: {list(profiles.keys())}")
    
    p = profiles[profile_name]["params"]
    
    return PipelineProfile(
        name=profile_name,
        compass_alpha=p["compass"]["alpha"],
        compass_threshold_high=p["compass"]["threshold_high"],
        compass_threshold_low=p["compass"]["threshold_low"],
        compass_window_tiles=p["compass"]["window_tiles"],
        compass_deadzone_us=p["compass"]["deadzone_us"],
        compass_min_tiles=p["compass"]["min_tiles"],
        compass_max_abs_dt_us=p["compass"].get("max_abs_dt_us"),
        lock_confidence_threshold=p["lock"]["confidence_threshold"],
        lock_soft_threshold=p["lock"]["soft_threshold"],
        unlock_tiles_threshold=p["lock"]["unlock_tiles_threshold"],
        rpm_alpha=p["rpm"]["alpha"],
        jitter_max_rel=p["rpm"]["jitter_max_rel"],
        jitter_window_size=p["rpm"]["jitter_window_size"],
        rpm_move_thresh=p["rpm"]["move_thresh"],
        rpm_slow_thresh=p["rpm"]["slow_thresh"],
        rpm_still_thresh=p["rpm"]["still_thresh"],
        tile_span_cycles=p["tiles"]["span_cycles"],
        min_normal_tile=p["tiles"]["min_normal_tile"],
        stereo_fusion=p["tiles"]["stereo_fusion"],
        cycles_per_rot=p["physical"]["cycles_per_rot"],
    )


# === Cycle detection (L1) ====================================================

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
                    "dt_us": t2 - t0,
                })

                self._cycle_counts[s_label] += 1
                self._dt_samples[s_label].append(float(t2 - t0))

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


# === TilesState ==============================================================

class TilesState:
    """Tile aggregatie met pure stereo marking."""

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
        self._pure_stereo_tiles = 0
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
        
        # Pure stereo: alleen betrouwbare dt_ab bij nA=1, nB=1
        is_pure_stereo = (nA == 1 and nB == 1)
        
        if is_pure_stereo:
            dt_ab = (tB - tA) if (tA is not None and tB is not None) else None
            self._pure_stereo_tiles += 1
        else:
            dt_ab = None

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
            "is_pure_stereo": is_pure_stereo,
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
            "pure_stereo_tiles": self._pure_stereo_tiles,
            "total_cycles_physical": self._total_cycles_physical,
            "boot_cycles_physical": self._boot_cycles_physical,
            "stereo_fusion": self.stereo_fusion,
            "min_normal_tile": self.min_normal_tile,
        }


# === CompassSign v3 Inline Implementation ====================================
# (Canonical implementation, compatible with compass_sign_v3.py)

@dataclass
class CompassSignConfig:
    """CompassSign v3 configuratie."""
    deadzone_us: float = 100.0
    min_tiles: int = 3
    max_abs_dt_us: Optional[float] = None
    phase_class: str = "any"
    mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "A_LEADS_B": "CW",
            "B_LEADS_A": "CCW",
            "ONBEPAALD": "UNDECIDED",
        }
    )


@dataclass
class CompassSignResult:
    """CompassSign v3 resultaat."""
    global_direction: str
    confidence: float
    verdict_tiles: str
    meta: Dict[str, Any]


def _summarize_sorted(values: List[float]) -> Dict[str, Optional[float]]:
    """Statistieken uit gesorteerde lijst."""
    n = len(values)
    if n == 0:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None}
    
    def pick(p: float) -> float:
        idx = int(round((n - 1) * p))
        return values[max(0, min(n - 1, idx))]
    
    return {
        "n": n,
        "min": values[0],
        "p25": pick(0.25),
        "median": pick(0.50),
        "p75": pick(0.75),
        "max": values[-1],
    }


def _compute_compass_sign_from_tiles(
    tiles: List[Dict[str, Any]],
    config: CompassSignConfig,
    tile_duration_us: Optional[float] = None,
) -> CompassSignResult:
    """CompassSign v3 canonical implementatie."""
    
    # Extract dt_ab values
    dt_values: List[float] = []
    for t in tiles:
        dt = t.get("dt_ab_us")
        if not isinstance(dt, (int, float)):
            continue
        dt_f = float(dt)
        if config.max_abs_dt_us is not None and abs(dt_f) > config.max_abs_dt_us:
            continue
        dt_values.append(dt_f)
    
    dt_values.sort()
    dt_stats = _summarize_sorted(dt_values)
    
    # Tiles verdict
    n = dt_stats.get("n") or 0
    median = dt_stats.get("median")
    
    verdict = "ONBEPAALD"
    reason = ""
    
    if n < config.min_tiles:
        reason = f"te weinig tiles (n={n}, min={config.min_tiles})"
    elif median is None:
        reason = "geen median"
    elif abs(median) < config.deadzone_us:
        reason = f"median in deadzone (|{median:.1f}| < {config.deadzone_us:.1f})"
    elif median > 0:
        verdict = "A_LEADS_B"
        reason = f"median={median:.1f}µs > 0"
    else:
        verdict = "B_LEADS_A"
        reason = f"median={median:.1f}µs < 0"
    
    # Confidence berekening (canonical)
    confidence = 0.0
    if verdict != "ONBEPAALD" and median is not None:
        m = abs(float(median))
        deadzone = float(config.deadzone_us)
        min_tiles = float(config.min_tiles)
        
        margin = max(0.0, m - deadzone)
        s1 = min(1.0, margin / (deadzone * 3.0)) if deadzone > 0 else 1.0
        s2 = min(1.0, max(0.0, (n - min_tiles) / min_tiles)) if min_tiles > 0 else 1.0
        confidence = 0.6 * s1 + 0.4 * s2
        confidence = max(0.0, min(1.0, confidence))
    
    global_direction = config.mapping.get(verdict, "UNDECIDED")
    
    meta = {
        "phase_class": config.phase_class,
        "deadzone_us": config.deadzone_us,
        "min_tiles": config.min_tiles,
        "n_tiles_total": len(tiles),
        "n_tiles_with_dt": n,
        "dt_stats": dt_stats,
        "reason": reason,
    }
    
    return CompassSignResult(
        global_direction=global_direction,
        confidence=confidence,
        verdict_tiles=verdict,
        meta=meta,
    )


# === Compass Adapter v1.9 (Canonical) ========================================

@dataclass
class CompassSnapshot:
    """Snapshot van compass state."""
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
    """
    v1.9: Canonical Compass met CompassSign v3.
    
    - Sliding window van tiles
    - CompassSign v3 voor window direction + confidence
    - EMA global score met window signaal
    """
    
    def __init__(self, profile: PipelineProfile):
        self.profile = profile
        self._window_tiles = profile.compass_window_tiles
        self._alpha = profile.compass_alpha
        self._threshold_high = profile.compass_threshold_high
        self._threshold_low = profile.compass_threshold_low
        
        # CompassSign v3 config
        self._compass_config = CompassSignConfig(
            deadzone_us=profile.compass_deadzone_us,
            min_tiles=profile.compass_min_tiles,
            max_abs_dt_us=profile.compass_max_abs_dt_us,
        )
        
        # Sliding window van tiles (niet alleen dt_ab)
        self._tiles_window: deque = deque(maxlen=self._window_tiles)
        
        self._global_score: float = 0.0
        self._global_direction: str = "UNDECIDED"
        self._tile_count: int = 0

    def _direction_to_signed_score(self, direction: str, confidence: float) -> float:
        """Convert direction + confidence naar signed score."""
        if direction == "CW":
            return +confidence
        elif direction == "CCW":
            return -confidence
        return 0.0

    def feed_tile(self, tile: Dict[str, Any], tile_duration_us: Optional[float]) -> CompassSnapshot:
        """
        Verwerk tile met CompassSign v3 (canonical).
        """
        self._tile_count += 1
        
        # Voeg tile toe aan window
        self._tiles_window.append(tile)
        
        # CompassSign v3 op hele window
        window_tiles = list(self._tiles_window)
        compass_result = _compute_compass_sign_from_tiles(
            tiles=window_tiles,
            config=self._compass_config,
            tile_duration_us=tile_duration_us,
        )
        
        window_dir = compass_result.global_direction
        window_conf = compass_result.confidence
        
        # EMA update met window signaal (canonical v1.5 gedrag)
        signed = self._direction_to_signed_score(window_dir, window_conf)
        self._global_score = self._alpha * self._global_score + (1.0 - self._alpha) * signed
        
        # Direction decision met hysterese
        if self._global_score >= self._threshold_high:
            self._global_direction = "CW"
        elif self._global_score <= -self._threshold_high:
            self._global_direction = "CCW"
        elif abs(self._global_score) < self._threshold_low:
            self._global_direction = "UNDECIDED"
        # Else: behoud huidige richting (hysterese zone)
        
        return CompassSnapshot(
            t_tile_index=self._tile_count,
            n_tiles_window=compass_result.meta.get("n_tiles_with_dt", 0),
            global_direction=self._global_direction,
            global_score=self._global_score,
            window_direction=window_dir,
            window_confidence=window_conf,
            window_verdict_tiles=compass_result.verdict_tiles,
            window_meta=compass_result.meta,
            global_meta={
                "alpha": self._alpha,
                "threshold_high": self._threshold_high,
                "threshold_low": self._threshold_low,
            },
        )


# === Movement State v1.9 =====================================================

@dataclass
class MovementState:
    """Complete rotor state met hiërarchie."""
    t_us: int = 0
    cycles_per_rot: float = 12.0
    
    # Position tracking
    cycle_index: float = 0.0
    rotations: float = 0.0
    theta_deg: float = 0.0
    theta_wrap_count: int = 0
    
    # Hiërarchische state (v1.9)
    rotor_state: str = ROTOR_STATE_STILL  # STILL of MOVEMENT
    direction_lock_state: str = LOCK_STATE_UNLOCKED  # Binnen MOVEMENT
    direction_locked_dir: str = ""
    direction_locked_conf: float = 0.0
    direction_global_effective: str = "UNDECIDED"
    direction_global_conf: float = 0.0
    
    # RPM tracking
    rpm_inst: float = 0.0
    rpm_est: float = 0.0
    rpm_jitter: float = 0.0
    cadence_ok: bool = False
    
    # Motion state (derived from rotor_state)
    motion_state: str = "STATIC"
    motion_conf: float = 0.0
    
    # Awareness
    flow_state: str = "IDLE"
    flow_score: float = 0.0
    resist_score: float = 0.0
    awareness_conf: float = 0.1


class MovementBodyV3:
    """
    v1.9: MovementBody met hiërarchische state machine.
    
    State Machine:
    - STILL: default, rpm < threshold
    - MOVEMENT: rpm >= threshold
      - UNLOCKED: geen richting bekend
      - SOFT_LOCK: tentative richting
      - LOCKED: committed richting (kan unlocken!)
    """
    
    def __init__(self, profile: PipelineProfile):
        self.profile = profile
        self._state = MovementState(cycles_per_rot=profile.cycles_per_rot)
        self._rpm_window: List[float] = []
        
        # Thresholds
        self.rpm_alpha = profile.rpm_alpha
        self.jitter_window_size = profile.jitter_window_size
        self.jitter_max_rel = profile.jitter_max_rel
        self.rpm_move_thresh = profile.rpm_move_thresh
        self.rpm_slow_thresh = profile.rpm_slow_thresh
        self.rpm_still_thresh = profile.rpm_still_thresh
        self.lock_confidence_threshold = profile.lock_confidence_threshold
        self.lock_soft_threshold = profile.lock_soft_threshold
        self.unlock_tiles_threshold = profile.unlock_tiles_threshold
        self.threshold_high = profile.compass_threshold_high
        self.threshold_low = profile.compass_threshold_low
        
        # Lock accumulators
        self._lock_confidence_accum = 0.0
        self._tiles_below_threshold = 0  # Voor unlock
        
    def _direction_sign(self) -> int:
        d = self._state.direction_locked_dir or self._state.direction_global_effective
        if d == "CW": return 1
        if d == "CCW": return -1
        return 0
    
    def _update_rotor_state(self) -> None:
        """Update top-level rotor state: STILL vs MOVEMENT."""
        rpm = self._state.rpm_est
        
        if self._state.rotor_state == ROTOR_STATE_STILL:
            # Transitie naar MOVEMENT als rpm stijgt
            if rpm >= self.rpm_move_thresh:
                self._state.rotor_state = ROTOR_STATE_MOVEMENT
                self._state.motion_state = "MOVING"
                self._state.motion_conf = 1.0
        else:  # MOVEMENT
            # Transitie naar STILL als rpm daalt
            if rpm < self.rpm_still_thresh:
                self._transition_to_still()
    
    def _transition_to_still(self) -> None:
        """Transitie naar STILL state."""
        self._state.rotor_state = ROTOR_STATE_STILL
        self._state.direction_lock_state = LOCK_STATE_UNLOCKED
        self._state.direction_global_effective = "UNDECIDED"
        self._state.motion_state = "STATIC"
        self._state.motion_conf = 0.0
        # Reset lock accumulators
        self._lock_confidence_accum = 0.0
        self._tiles_below_threshold = 0
    
    def set_compass_realtime(self, compass_data: Dict[str, Any]) -> str:
        """
        Update direction state met compass data.
        Returns: vorige lock_state (voor Claim-at-Lock detectie)
        """
        prev_lock_state = self._state.direction_lock_state
        
        gd = compass_data.get("global_direction", "UNDECIDED")
        gs = compass_data.get("global_score", 0.0)
        
        self._state.direction_global_effective = gd
        self._state.direction_global_conf = abs(gs)
        
        # Skip als we in STILL zijn
        if self._state.rotor_state == ROTOR_STATE_STILL:
            return prev_lock_state
        
        # === UNLOCK MECHANISME ===
        if self._state.direction_lock_state == LOCK_STATE_LOCKED:
            # Check of we moeten unlocken
            if abs(gs) < self.threshold_low:
                self._tiles_below_threshold += 1
                if self._tiles_below_threshold >= self.unlock_tiles_threshold:
                    # LOCKED → SOFT_LOCK
                    self._state.direction_lock_state = LOCK_STATE_SOFT_LOCK
                    self._tiles_below_threshold = 0
            else:
                self._tiles_below_threshold = 0
            return prev_lock_state
        
        # === SOFT_LOCK → UNLOCKED ===
        if self._state.direction_lock_state == LOCK_STATE_SOFT_LOCK:
            if abs(gs) < self.threshold_low:
                self._tiles_below_threshold += 1
                if self._tiles_below_threshold >= self.unlock_tiles_threshold:
                    # SOFT_LOCK → UNLOCKED
                    self._state.direction_lock_state = LOCK_STATE_UNLOCKED
                    self._state.direction_locked_dir = ""
                    self._state.direction_locked_conf = 0.0
                    self._tiles_below_threshold = 0
                    self._lock_confidence_accum = 0.0
            elif abs(gs) >= self.threshold_high and gd in ("CW", "CCW"):
                # SOFT_LOCK → LOCKED (terug naar locked)
                self._state.direction_lock_state = LOCK_STATE_LOCKED
                self._state.direction_locked_dir = gd
                self._state.direction_locked_conf = abs(gs)
                self._tiles_below_threshold = 0
            else:
                self._tiles_below_threshold = 0
            return prev_lock_state
        
        # === UNLOCKED → SOFT_LOCK → LOCKED ===
        if gd in ("CW", "CCW") and abs(gs) >= self.threshold_high:
            self._lock_confidence_accum += abs(gs) * 0.15
            
            if self._lock_confidence_accum > self.lock_confidence_threshold:
                # UNLOCKED → LOCKED (direct lock)
                self._state.direction_lock_state = LOCK_STATE_LOCKED
                self._state.direction_locked_dir = gd
                self._state.direction_locked_conf = abs(gs)
            elif self._lock_confidence_accum > self.lock_soft_threshold:
                # UNLOCKED → SOFT_LOCK
                self._state.direction_lock_state = LOCK_STATE_SOFT_LOCK
                self._state.direction_locked_dir = gd
                self._state.direction_locked_conf = abs(gs)
        else:
            self._lock_confidence_accum = max(0, self._lock_confidence_accum - 0.05)
        
        return prev_lock_state
            
    def _update_motion_state_from_rpm(self) -> None:
        """Update motion state binnen MOVEMENT."""
        if self._state.rotor_state == ROTOR_STATE_STILL:
            self._state.motion_state = "STATIC"
            self._state.motion_conf = 0.0
            return
            
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
            
    def _update_awareness_conf(self) -> None:
        """Update awareness confidence."""
        motion = self._state.motion_conf
        lock = self._state.direction_locked_conf if self._state.direction_lock_state == LOCK_STATE_LOCKED else 0.0
        cadence = 1.0 if self._state.cadence_ok else 0.3
        
        self._state.awareness_conf = 0.4 * motion + 0.4 * lock + 0.2 * cadence
        self._state.flow_score = lock
        
    def snapshot(self) -> MovementState:
        return self._state


# === Realtime Snapshot =======================================================

@dataclass
class RealtimeSnapshot:
    """Complete realtime pipeline snapshot."""
    t_us: Optional[int]
    cycles_emitted: List[Dict[str, Any]]
    tiles_emitted: List[Dict[str, Any]]
    compass_snapshot: Optional[CompassSnapshot]
    movement_state: Dict[str, Any]
    
    cycles_physical_this_batch: float = 0.0
    total_cycles_physical: float = 0.0
    cycles_unsigned: float = 0.0
    cycles_claimed_at_lock: float = 0.0
    
    boot_tiles_skipped: int = 0
    boot_cycles_skipped: float = 0.0
    
    profile_name: str = ""
    pure_stereo_tiles: int = 0


# === RealtimePipeline v1.9 ===================================================

class RealtimePipeline:
    """
    S02-Realtime-States v1.9 — Canonical Architecture
    
    - CompassSign v3 (canonical)
    - Hiërarchische RotorState (STILL → MOVEMENT)
    - Unlock mechanisme
    - Claim-at-Lock
    - BootWarmup
    """

    def __init__(self, profile: PipelineProfile = None):
        if profile is None:
            profile = PROFILE_PRODUCTION
        
        self.profile = profile
        
        self.cycles_state = CyclesState()
        self.tiles_state = TilesState(profile=profile)
        self.compass_adapter = CompassAdapter(profile=profile)
        self.movement_body = MovementBodyV3(profile=profile)

        self._last_t_us: Optional[int] = None
        self._last_compass_snapshot: Optional[CompassSnapshot] = None
        
        self._total_cycles_physical: float = 0.0
        self._cycles_unsigned: float = 0.0
        self._cycles_claimed_at_lock: float = 0.0
        self._lock_claimed: bool = False
        self._last_cycle_t_us: Optional[int] = None
        
        self._boot_tiles_skipped: int = 0
        self._boot_cycles_skipped: float = 0.0

    def _get_direction_sign(self) -> int:
        return self.movement_body._direction_sign()

    def _check_and_claim_at_lock(self, prev_lock_state: str) -> None:
        """Claim unsigned cycles bij transitie naar LOCKED."""
        st = self.movement_body._state
        current_lock = st.direction_lock_state
        
        # Claim alleen bij transitie NAAR LOCKED (niet bij re-lock)
        if (current_lock == LOCK_STATE_LOCKED and 
            prev_lock_state != LOCK_STATE_LOCKED and 
            self._cycles_unsigned > 0):
            
            sign = self._get_direction_sign()
            if sign != 0:
                claimed = self._cycles_unsigned * sign
                
                C = st.cycles_per_rot if st.cycles_per_rot > 0 else 12.0
                st.cycle_index += claimed
                st.rotations = st.cycle_index / C
                
                theta_new = (st.rotations * 360.0) % 360.0
                st.theta_deg = theta_new
                
                self._cycles_claimed_at_lock += self._cycles_unsigned
                self._cycles_unsigned = 0.0
                self._lock_claimed = True

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
            
            # Boot tiles: alleen cycles tellen, geen compass/lock
            if tile_state == TILE_STATE_BOOT:
                self._boot_tiles_skipped += 1
                self._boot_cycles_skipped += cycles_physical
                if has_data:
                    self._cycles_unsigned += cycles_physical
                continue
            
            # Normale tiles met data
            if has_data:
                # Compass update
                compass_snap = self.compass_adapter.feed_tile(
                    tile,
                    tile_duration_us=self.tiles_state.tile_duration_us,
                )
                self._last_compass_snapshot = compass_snap

                # Movement state update (returns prev_lock_state)
                prev_lock_state = self.movement_body.set_compass_realtime({
                    "global_direction": compass_snap.global_direction,
                    "global_score": compass_snap.global_score,
                    "window_direction": compass_snap.window_direction,
                    "window_confidence": compass_snap.window_confidence,
                })

                t_center = tile.get("tA_us") or tile.get("t_center_us") or self._last_t_us or 0

                st = self.movement_body._state
                
                # Update rotor state (STILL vs MOVEMENT)
                self.movement_body._update_rotor_state()
                
                current_lock = st.direction_lock_state
                sign = self._get_direction_sign()
                
                if current_lock == LOCK_STATE_LOCKED and sign != 0:
                    self._update_cycle_index_signed(cycles_physical, sign, t_center)
                else:
                    self._cycles_unsigned += cycles_physical
                    self._update_rpm_unsigned(t_center, cycles_physical)
                
                # Check for Claim-at-Lock
                self._check_and_claim_at_lock(prev_lock_state)

        movement_state_obj = self.movement_body.snapshot()
        movement_state = movement_state_obj.__dict__.copy()
        movement_state.update({
            "total_cycles_physical": self._total_cycles_physical,
            "cycles_unsigned": self._cycles_unsigned,
            "cycles_claimed_at_lock": self._cycles_claimed_at_lock,
            "boot_tiles_skipped": self._boot_tiles_skipped,
            "boot_cycles_skipped": self._boot_cycles_skipped,
            "profile_name": self.profile.name,
            "pure_stereo_tiles": self.tiles_state._pure_stereo_tiles,
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
            pure_stereo_tiles=self.tiles_state._pure_stereo_tiles,
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
        
        # Update rotor state based on rpm
        self.movement_body._update_rotor_state()
        self.movement_body._update_motion_state_from_rpm()

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
            "pure_stereo_tiles": self.tiles_state._pure_stereo_tiles,
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
            "pure_stereo_tiles": self.tiles_state._pure_stereo_tiles,
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
            pure_stereo_tiles=self.tiles_state._pure_stereo_tiles,
        )
