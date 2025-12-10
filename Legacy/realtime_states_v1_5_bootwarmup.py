"""
S02 — Realtime-States v1.5 BootWarmup
=====================================

NIEUWE FEATURE: BootWarmup fase isoleert tile 0 "burst" artefacten.

Architectuur (conform S02.RealtimeStates_v1_4_ClaimAtLock_Architectuur.md):
--------------------------------------------------------------------------
Bij boot/init kunnen er "burst" artefacten optreden:
- Buffer-flush van opgestapelde events
- Abnormaal hoge cycle counts in eerste tile(s)
- Onbetrouwbare dt/rpm metingen

BootWarmup oplossing:
- tile_state ∈ {BOOT, NORMAL}
- tile_index < MIN_NORMAL_TILE → tile_state = BOOT
- In BOOT:
  * cycles_physical wordt WEL geteld (fysieke waarheid behouden)
  * maar GEEN compass updates
  * GEEN lock beslissingen
  * GEEN rpm/cadence updates
  * GEEN awareness updates
- Pas bij tile_index ≥ MIN_NORMAL_TILE → tile_state = NORMAL

Kernwijzigingen t.o.v. v1.4:
---------------------------
- TilesState: tile krijgt "tile_state" veld (BOOT/NORMAL)
- RealtimePipeline: skip compass/lock/awareness voor BOOT tiles
- cycles_physical blijft correct (fysieke waarheid)
- Configureerbaar: min_normal_tile parameter (default: 1)

Verwachte verbeteringen:
-----------------------
- Compass EMA niet "vervuild" door boot burst
- RPM init niet vertekend door abnormale dt
- Awareness start vanaf schone baseline
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
DEFAULT_MIN_NORMAL_TILE = 1  # Tile 0 is BOOT, tile 1+ is NORMAL


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
    """v1.5: Tiles met cycles_physical en tile_state (BOOT/NORMAL)."""

    def __init__(self,
                 tile_span_cycles: float = 0.6,
                 boot_cycles_for_median: int = 24,
                 stereo_fusion: str = "mean",
                 min_normal_tile: int = DEFAULT_MIN_NORMAL_TILE):
        self.tile_span_cycles = float(tile_span_cycles)
        self.boot_cycles_for_median = int(boot_cycles_for_median)
        self.stereo_fusion = stereo_fusion
        self.min_normal_tile = int(min_normal_tile)  # v1.5: BootWarmup threshold

        self._boot_dt_samples: List[float] = []
        self._tile_duration_us: Optional[float] = None

        self._t0_us: Optional[float] = None
        self._current_tile_index: Optional[int] = None
        self._current_tile_cycles = {"A": [], "B": []}

        self._tiles_emitted = 0
        self._empty_tiles_emitted = 0
        self._data_tiles_emitted = 0
        self._boot_tiles_emitted = 0  # v1.5: track boot tiles
        self._total_cycles_physical = 0.0
        self._boot_cycles_physical = 0.0  # v1.5: cycles in boot phase

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
        """v1.5: Bepaal tile_state op basis van tile_index."""
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
        
        # v1.5: Bepaal tile_state
        tile_state = self._get_tile_state(idx)

        tile = {
            "tile_index": idx,
            "tile_state": tile_state,  # v1.5: BOOT of NORMAL
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

        # Statistieken
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
            "boot_tiles_emitted": self._boot_tiles_emitted,  # v1.5
            "total_cycles_physical": self._total_cycles_physical,
            "boot_cycles_physical": self._boot_cycles_physical,  # v1.5
            "stereo_fusion": self.stereo_fusion,
            "min_normal_tile": self.min_normal_tile,  # v1.5
        }


# === Compass Adapter (L2.5) — unchanged ======================================

from sym_cycles.realtime_compass import InertialCompassState
from sym_cycles.movement_body_v3 import MovementBodyV3


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
    def __init__(self,
                 window_tiles: int = 20,
                 phase_class: str = "any",
                 alpha: float = 0.95,
                 threshold_high: float = 0.6,
                 threshold_low: float = 0.25):
        self._ics = InertialCompassState(
            window_tiles=window_tiles,
            phase_class=phase_class,
            alpha=alpha,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
        )

    @property
    def inertial_state(self) -> InertialCompassState:
        return self._ics

    def feed_tile(self, tile: Dict[str, Any], tile_duration_us: Optional[float]) -> CompassSnapshot:
        snap = self._ics.feed_tile(tile, tile_duration_us=tile_duration_us)

        return CompassSnapshot(
            t_tile_index=snap.t_tile_index,
            n_tiles_window=snap.n_tiles_window,
            window_direction=snap.window_direction,
            window_confidence=snap.window_confidence,
            window_verdict_tiles=snap.window_verdict_tiles,
            window_meta=snap.window_meta,
            global_direction=snap.global_direction,
            global_score=snap.global_score,
            global_meta=snap.global_meta,
        )


# === Realtime snapshot (L3) ==================================================

@dataclass
class RealtimeSnapshot:
    t_us: Optional[int]
    cycles_emitted: List[Dict[str, Any]]
    tiles_emitted: List[Dict[str, Any]]
    compass_snapshot: Optional[CompassSnapshot]
    movement_state: Dict[str, Any]
    
    # v1.4: Cycles tracking
    cycles_physical_this_batch: float = 0.0
    total_cycles_physical: float = 0.0
    cycles_unsigned: float = 0.0
    cycles_claimed_at_lock: float = 0.0
    
    # v1.5: BootWarmup tracking
    boot_tiles_skipped: int = 0
    boot_cycles_skipped: float = 0.0


# === RealtimePipeline v1.5 — BootWarmup ======================================

class RealtimePipeline:
    """
    S02-Realtime-States v1.5 — BootWarmup
    
    NIEUW: Tiles in BOOT state worden niet gebruikt voor compass/lock/awareness.
    
    Flow:
    1. BOOT tiles (index < min_normal_tile):
       - cycles_physical wordt geteld (fysieke waarheid)
       - GEEN compass updates
       - GEEN lock beslissingen
       - GEEN rpm/awareness updates
    
    2. NORMAL tiles (index >= min_normal_tile):
       - Volledige v1.4 Claim-at-Lock logica
       - Compass, lock, rpm, awareness updates
    
    Dit beschermt tegen "tile 0 burst" artefacten.
    """

    def __init__(self,
                 cycles_per_rot: float = 12.0,
                 compass_window_tiles: int = 20,
                 compass_phase_class: str = "any",
                 compass_alpha: float = 0.95,
                 compass_threshold_high: float = 0.6,
                 compass_threshold_low: float = 0.25,
                 tile_span_cycles: float = 0.6,
                 stereo_fusion: str = "mean",
                 min_normal_tile: int = DEFAULT_MIN_NORMAL_TILE):  # v1.5
        
        self.tile_span_cycles = tile_span_cycles
        self.cycles_per_rot = cycles_per_rot
        self.stereo_fusion = stereo_fusion
        self.min_normal_tile = min_normal_tile  # v1.5
        
        self.cycles_state = CyclesState()
        self.tiles_state = TilesState(
            tile_span_cycles=tile_span_cycles,
            boot_cycles_for_median=24,
            stereo_fusion=stereo_fusion,
            min_normal_tile=min_normal_tile,  # v1.5
        )
        self.compass_adapter = CompassAdapter(
            window_tiles=compass_window_tiles,
            phase_class=compass_phase_class,
            alpha=compass_alpha,
            threshold_high=compass_threshold_high,
            threshold_low=compass_threshold_low,
        )
        self.movement_body = MovementBodyV3(cycles_per_rot=cycles_per_rot)

        self._last_t_us: Optional[int] = None
        self._last_compass_snapshot: Optional[CompassSnapshot] = None
        
        # v1.4: Cycles tracking
        self._total_cycles_physical: float = 0.0
        self._cycles_unsigned: float = 0.0
        self._cycles_claimed_at_lock: float = 0.0
        self._lock_claimed: bool = False
        self._last_cycle_t_us: Optional[int] = None
        
        # Track vorige lock state voor transitie-detectie
        self._prev_lock_state: str = "UNLOCKED"
        
        # v1.5: BootWarmup tracking
        self._boot_tiles_skipped: int = 0
        self._boot_cycles_skipped: float = 0.0

    def _get_direction_sign(self) -> int:
        """Haal direction sign uit MovementBody."""
        return self.movement_body._direction_sign()

    def _check_and_claim_at_lock(self) -> None:
        """
        v1.4 KERNLOGICA: Check of we net LOCKED zijn geworden.
        Zo ja: claim de unsigned cycles met de juiste sign.
        """
        st = self.movement_body._state
        current_lock = st.direction_lock_state
        
        # Transitie naar LOCKED?
        if (current_lock == "LOCKED" and 
            self._prev_lock_state != "LOCKED" and 
            not self._lock_claimed and
            self._cycles_unsigned > 0):
            
            # CLAIM MOMENT!
            sign = self._get_direction_sign()
            if sign != 0:
                claimed = self._cycles_unsigned * sign
                
                # Update cycle_index met de geclaimde cycles
                C = st.cycles_per_rot if st.cycles_per_rot > 0 else 12.0
                st.cycle_index += claimed
                st.rotations = st.cycle_index / C
                
                # Update theta
                theta_new = (st.rotations * 360.0) % 360.0
                st.theta_deg = theta_new
                
                # Administreer de claim
                self._cycles_claimed_at_lock = self._cycles_unsigned
                self._cycles_unsigned = 0.0
                self._lock_claimed = True
        
        self._prev_lock_state = current_lock

    def feed_event(self, ev: Dict[str, Any]) -> RealtimeSnapshot:
        """v1.5: Verwerk één EVENT24 met BootWarmup en Claim-at-Lock."""
        
        if isinstance(ev.get("t_abs_us"), (int, float)):
            self._last_t_us = int(ev["t_abs_us"])

        # L1: EVENT24 → 3-punts cycles
        cycles = self.cycles_state.feed_event(ev)

        # L2: cycles → tiles met cycles_physical en tile_state
        tiles = self.tiles_state.feed_cycles(cycles)

        compass_snap: Optional[CompassSnapshot] = self._last_compass_snapshot
        cycles_physical_batch: float = 0.0

        for tile in tiles:
            cycles_physical = tile.get("cycles_physical", 0.0)
            tile_state = tile.get("tile_state", TILE_STATE_NORMAL)
            
            cycles_physical_batch += cycles_physical
            self._total_cycles_physical += cycles_physical
            
            has_data = cycles_physical > 0
            
            # v1.5: BOOT tiles worden anders behandeld
            if tile_state == TILE_STATE_BOOT:
                # Tel cycles maar skip compass/lock/awareness updates
                self._boot_tiles_skipped += 1
                self._boot_cycles_skipped += cycles_physical
                
                # Cycles gaan WEL naar unsigned buffer (fysieke waarheid)
                if has_data:
                    self._cycles_unsigned += cycles_physical
                
                # Skip alle andere updates voor BOOT tiles
                continue
            
            # NORMAL tiles: volledige v1.4 logica
            if has_data:
                # Update kompas
                compass_snap = self.compass_adapter.feed_tile(
                    tile,
                    tile_duration_us=self.tiles_state.tile_duration_us,
                )
                self._last_compass_snapshot = compass_snap

                # Update MovementBody kompas state
                self.movement_body.set_compass_realtime({
                    "global_direction": compass_snap.global_direction,
                    "global_score": compass_snap.global_score,
                    "window_direction": compass_snap.window_direction,
                    "window_confidence": compass_snap.window_confidence,
                })

                t_center = tile.get("tA_us") or tile.get("t_center_us") or self._last_t_us or 0

                # Bepaal huidige lock state
                st = self.movement_body._state
                current_lock = st.direction_lock_state
                sign = self._get_direction_sign()
                
                if current_lock == "LOCKED" and sign != 0:
                    # NA lock: tel cycles direct bij cycle_index
                    self._update_cycle_index_signed(cycles_physical, sign, t_center)
                else:
                    # VOOR lock: accumuleer in unsigned buffer
                    self._cycles_unsigned += cycles_physical
                    # Update RPM tracking (unsigned, voor motion detection)
                    self._update_rpm_unsigned(t_center, cycles_physical)
                
                # Check voor lock transitie en claim
                self._check_and_claim_at_lock()

        # Bouw movement state
        movement_state_obj = self.movement_body.snapshot()
        movement_state = movement_state_obj.__dict__.copy()
        movement_state.update({
            "total_cycles_physical": self._total_cycles_physical,
            "cycles_unsigned": self._cycles_unsigned,
            "cycles_claimed_at_lock": self._cycles_claimed_at_lock,
            "boot_tiles_skipped": self._boot_tiles_skipped,  # v1.5
            "boot_cycles_skipped": self._boot_cycles_skipped,  # v1.5
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
            boot_tiles_skipped=self._boot_tiles_skipped,  # v1.5
            boot_cycles_skipped=self._boot_cycles_skipped,  # v1.5
        )

    def _update_cycle_index_signed(self, cycles_physical: float, sign: int, t_us: int) -> None:
        """Update cycle_index met signed cycles (NA lock)."""
        st = self.movement_body._state
        C = st.cycles_per_rot if st.cycles_per_rot > 0 else 12.0
        
        # Update cycle_index
        st.cycle_index += sign * cycles_physical
        st.rotations = st.cycle_index / C
        
        # Update theta
        theta_prev = st.theta_deg
        theta_new = (st.rotations * 360.0) % 360.0
        
        delta = theta_new - theta_prev
        if delta > 180.0:
            st.theta_wrap_count -= 1
        elif delta < -180.0:
            st.theta_wrap_count += 1
        
        st.theta_deg = theta_new
        st.t_us = t_us
        
        # Update RPM
        self._update_rpm_signed(t_us, cycles_physical)
        
        # Update motion state
        self.movement_body._update_motion_state_from_rpm()
        self.movement_body._update_awareness_conf()

    def _update_rpm_signed(self, t_us: int, cycles_physical: float) -> None:
        """Update RPM gebaseerd op werkelijke cycles (NA lock)."""
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
        
        # RPM = (cycles / dt_s) × (60 / C)
        rpm_inst = (cycles_physical / dt_s) * (60.0 / C)
        
        # EMA smoothing
        alpha = self.movement_body.rpm_alpha
        if st.rpm_est <= 0:
            rpm_est = rpm_inst
        else:
            rpm_est = (1.0 - alpha) * st.rpm_est + alpha * rpm_inst
        
        st.rpm_inst = rpm_inst
        st.rpm_est = rpm_est
        
        # Jitter tracking
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
        """Update RPM tracking VOOR lock (voor motion state)."""
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
        
        # Simpele EMA voor motion detection
        alpha = 0.3
        if st.rpm_est <= 0:
            st.rpm_est = rpm_inst
        else:
            st.rpm_est = (1.0 - alpha) * st.rpm_est + alpha * rpm_inst
        
        st.rpm_inst = rpm_inst
        
        # Update motion state op basis van rpm
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
            # v1.5
            "boot_tiles_skipped": self._boot_tiles_skipped,
            "boot_cycles_skipped": self._boot_cycles_skipped,
            "min_normal_tile": self.min_normal_tile,
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
        )
