"""
S02 — Realtime-States v1.3 Cycles Physical
===========================================

ARCHITECTUUR FIX: cycle_index telt nu FYSIEKE CYCLES, niet tiles.

Kernwijziging t.o.v. v1.2:
--------------------------
- TilesState voegt `cycles_physical` toe aan elke tile
- MovementBody doet `cycle_index += cycles_physical` (niet += 1 of += tile_span)
- `tile_span_cycles` blijft ALLEEN voor kompas-window sizing

Verwachte resultaten:
--------------------
- 60 gedetecteerde cycles → cycle_index ≈ 60
- rotations = 60 / 12 = 5.0 (fysiek correct!)
- rpm gebaseerd op werkelijke cycle-intervallen

Incremental realtime chain:
    EVENT24 → CyclesState → TilesState → InertialCompass → MovementBodyV3
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


# === Cycle detection (L1) — unchanged ========================================

class CyclesState:
    """
    Incrementele 3-punts cycle-detectie per sensor.
    Ongewijzigd — dit is de bron van fysieke waarheid.
    """

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
        """Retourneer totaal aantal gedetecteerde cycles per sensor."""
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


# === Time-based tiles with cycles_physical (L2) — v1.3 ======================

class TilesState:
    """
    v1.3: Tiles met cycles_physical.
    
    Kernwijziging:
    - Elke tile bevat nu `cycles_physical`: het aantal fysieke cycles in die tile
    - Dit is de ENIGE waarde die MovementBody gebruikt voor cycle_index
    - tile_span_cycles blijft ALLEEN voor tijdvenster-sizing
    
    Stereo fusion strategie:
    - cycles_physical = min(nA, nB) als beide > 0 (conservatief, stereo-bevestigd)
    - cycles_physical = 0 als geen stereo-pair (mono-cycles tellen niet voor rotatie)
    """

    def __init__(self,
                 tile_span_cycles: float = 0.6,
                 boot_cycles_for_median: int = 24,
                 stereo_fusion: str = "min"):  # "min", "mean", "max"
        self.tile_span_cycles = float(tile_span_cycles)
        self.boot_cycles_for_median = int(boot_cycles_for_median)
        self.stereo_fusion = stereo_fusion

        self._boot_dt_samples: List[float] = []
        self._tile_duration_us: Optional[float] = None

        self._t0_us: Optional[float] = None
        self._current_tile_index: Optional[int] = None
        self._current_tile_cycles = {"A": [], "B": []}  # cycles met timestamps

        self._tiles_emitted = 0
        self._empty_tiles_emitted = 0
        self._data_tiles_emitted = 0
        self._total_cycles_physical = 0.0

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

    def _fuse_cycles(self, nA: int, nB: int) -> float:
        """
        Stereo fusion: bepaal cycles_physical uit nA en nB.
        
        Strategie:
        - "min": conservatief, alleen stereo-bevestigde cycles
        - "mean": gemiddelde van beide sensoren
        - "max": liberaal, hoogste telling
        """
        if nA == 0 and nB == 0:
            return 0.0
        
        if self.stereo_fusion == "min":
            # Conservatief: alleen cycles die door beide sensoren bevestigd zijn
            # Als één sensor 0 heeft, tellen we 0 (geen stereo-bevestiging)
            if nA == 0 or nB == 0:
                return 0.0
            return float(min(nA, nB))
        
        elif self.stereo_fusion == "mean":
            # Gemiddelde: compromis
            return (nA + nB) / 2.0
        
        elif self.stereo_fusion == "max":
            # Liberaal: hoogste telling
            return float(max(nA, nB))
        
        else:
            # Fallback: min
            return float(min(nA, nB)) if (nA > 0 and nB > 0) else 0.0

    def _make_tile(self, idx: int, 
                   cycles_A: List[Dict], 
                   cycles_B: List[Dict]) -> Dict[str, Any]:
        """Maak een tile met cycles_physical."""
        
        t_start = (self._t0_us +
                   idx * self._tile_duration_us) if self._tile_duration_us else self._t0_us or 0.0
        t_end = t_start + (self._tile_duration_us or 0.0)
        t_center = (t_start + t_end) / 2

        nA = len(cycles_A)
        nB = len(cycles_B)
        
        # Bereken tijdgemiddeldes
        ts_A = [c["t_center_us"] for c in cycles_A]
        ts_B = [c["t_center_us"] for c in cycles_B]
        tA = sum(ts_A) / len(ts_A) if ts_A else None
        tB = sum(ts_B) / len(ts_B) if ts_B else None
        dt_ab = (tB - tA) if (tA is not None and tB is not None) else None

        # KERNWIJZIGING v1.3: cycles_physical
        cycles_physical = self._fuse_cycles(nA, nB)
        has_data = cycles_physical > 0

        tile = {
            "tile_index": idx,
            "t_start_us": t_start,
            "t_end_us": t_end,
            "t_center_us": t_center,
            "tA_us": tA,
            "tB_us": tB,
            "dt_ab_us": dt_ab,
            "nA": nA,
            "nB": nB,
            "cycles_physical": cycles_physical,  # v1.3: DE NIEUWE SLEUTEL
            "has_data": has_data,
            "tile_span_cycles": self.tile_span_cycles,  # alleen voor referentie
        }

        self._tiles_emitted += 1
        self._total_cycles_physical += cycles_physical
        if has_data:
            self._data_tiles_emitted += 1
        else:
            self._empty_tiles_emitted += 1

        return tile

    def _flush_tiles_up_to(self, target_idx: int) -> List[Dict[str, Any]]:
        """Emit alle tiles van current tot target, inclusief lege tussenliggende."""
        tiles: List[Dict[str, Any]] = []
        
        if self._current_tile_index is None:
            return tiles
        
        # Flush current tile met data
        tile = self._make_tile(
            self._current_tile_index,
            self._current_tile_cycles["A"],
            self._current_tile_cycles["B"]
        )
        tiles.append(tile)
        
        # Emit lege tiles voor tussenliggende indices
        for idx in range(self._current_tile_index + 1, target_idx):
            empty_tile = self._make_tile(idx, [], [])
            tiles.append(empty_tile)
        
        # Reset cycle verzameling
        self._current_tile_cycles = {"A": [], "B": []}
        
        return tiles

    def feed_cycles(self, cycles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """v1.3: Feed cycles en emit tiles met cycles_physical."""
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

            # Sla de hele cycle op (niet alleen timestamp)
            self._current_tile_cycles[sensor].append(c)

        return new_tiles

    def finalize(self) -> List[Dict[str, Any]]:
        """Flush de laatste tile."""
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
            "total_cycles_physical": self._total_cycles_physical,
            "stereo_fusion": self.stereo_fusion,
        }


# === Compass Snapshot adapter (L2.5) — unchanged =============================

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
    """Adapter boven InertialCompassState — ongewijzigd."""

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


# === Realtime snapshot struct (L3) ===========================================

@dataclass
class RealtimeSnapshot:
    t_us: Optional[int]
    cycles_emitted: List[Dict[str, Any]]
    tiles_emitted: List[Dict[str, Any]]
    compass_snapshot: Optional[CompassSnapshot]
    movement_state: Dict[str, Any]
    
    # v1.3: Cycles tracking
    cycles_physical_this_batch: float = 0.0
    total_cycles_physical: float = 0.0


# === RealtimePipeline (public API) — v1.3 Cycles Physical ====================

class RealtimePipeline:
    """
    S02-Realtime-States v1.3 Cycles Physical
    
    KERNWIJZIGING: cycle_index telt nu fysieke cycles, niet tiles.
    
    - TilesState berekent cycles_physical per tile (stereo fusion)
    - MovementBody krijgt cycles_physical door
    - cycle_index += cycles_physical (niet += 1 of += tile_span)
    """

    def __init__(self,
                 cycles_per_rot: float = 12.0,
                 compass_window_tiles: int = 20,
                 compass_phase_class: str = "any",
                 compass_alpha: float = 0.95,
                 compass_threshold_high: float = 0.6,
                 compass_threshold_low: float = 0.25,
                 tile_span_cycles: float = 0.6,
                 stereo_fusion: str = "min"):  # "min", "mean", "max"
        
        self.tile_span_cycles = tile_span_cycles
        self.cycles_per_rot = cycles_per_rot
        self.stereo_fusion = stereo_fusion
        
        self.cycles_state = CyclesState()
        self.tiles_state = TilesState(
            tile_span_cycles=tile_span_cycles,
            boot_cycles_for_median=24,
            stereo_fusion=stereo_fusion,
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
        
        # v1.3: Track totale fysieke cycles
        self._total_cycles_physical: float = 0.0
        self._last_cycle_t_us: Optional[int] = None

    def feed_event(self, ev: Dict[str, Any]) -> RealtimeSnapshot:
        """
        v1.3: Verwerk één EVENT24.
        
        Kernwijziging: cycle_index += cycles_physical per tile.
        """
        if isinstance(ev.get("t_abs_us"), (int, float)):
            self._last_t_us = int(ev["t_abs_us"])

        # L1: EVENT24 → 3-punts cycles
        cycles = self.cycles_state.feed_event(ev)

        # L2: cycles → tiles met cycles_physical
        tiles = self.tiles_state.feed_cycles(cycles)

        compass_snap: Optional[CompassSnapshot] = self._last_compass_snapshot
        cycles_physical_batch: float = 0.0

        for tile in tiles:
            cycles_physical = tile.get("cycles_physical", 0.0)
            cycles_physical_batch += cycles_physical
            self._total_cycles_physical += cycles_physical
            
            has_data = cycles_physical > 0
            
            if has_data:
                # Update kompas
                compass_snap = self.compass_adapter.feed_tile(
                    tile,
                    tile_duration_us=self.tiles_state.tile_duration_us,
                )
                self._last_compass_snapshot = compass_snap

                # Update MovementBody met kompas
                self.movement_body.set_compass_realtime({
                    "global_direction": compass_snap.global_direction,
                    "global_score": compass_snap.global_score,
                    "window_direction": compass_snap.window_direction,
                    "window_confidence": compass_snap.window_confidence,
                })

                # Bepaal t_center
                t_center = tile.get("tA_us") or tile.get("t_center_us") or self._last_t_us or 0

                # v1.3: KERNWIJZIGING — geef cycles_physical door
                node = {
                    "t_center_us": t_center,
                    "cycle_type": "tile_cycle",
                    "tile_index": tile.get("tile_index"),
                    "cycles_physical": cycles_physical,  # v1.3: DE SLEUTEL
                    "has_data": True,
                }
                
                # We moeten MovementBody aanpassen om cycles_physical te gebruiken
                # Voor nu: roep feed_cycle_node aan met cycles_physical keer
                # TODO: Beter zou zijn om MovementBody.feed_tile() te maken
                self._feed_movement_body_with_cycles(node, cycles_physical, t_center)

        # Bouw movement state
        movement_state_obj = self.movement_body.snapshot()
        movement_state = movement_state_obj.__dict__.copy()
        movement_state.update({
            "total_cycles_physical": self._total_cycles_physical,
        })

        return RealtimeSnapshot(
            t_us=self._last_t_us,
            cycles_emitted=cycles,
            tiles_emitted=tiles,
            compass_snapshot=compass_snap,
            movement_state=movement_state,
            cycles_physical_this_batch=cycles_physical_batch,
            total_cycles_physical=self._total_cycles_physical,
        )

    def _feed_movement_body_with_cycles(self, 
                                         node: Dict[str, Any], 
                                         cycles_physical: float,
                                         t_center: int) -> None:
        """
        v1.3: Feed MovementBody met het juiste aantal cycles.
        
        Aangezien MovementBody.feed_cycle_node() intern += 1 doet,
        moeten we ofwel:
        1. MovementBody patchen om cycles_physical te accepteren
        2. Of: meerdere keren aanroepen (niet ideaal maar werkt)
        
        Voor nu: we patchen de state direct (schonere oplossing).
        """
        st = self.movement_body._state
        
        # Haal direction sign
        s = self.movement_body._direction_sign()
        
        if s != 0:
            # v1.3: cycle_index += sign × cycles_physical
            C = st.cycles_per_rot if st.cycles_per_rot > 0 else 12.0
            
            st.cycle_index += s * cycles_physical
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
            st.t_us = t_center
            
            # Update RPM
            self._update_rpm_for_cycles(t_center, cycles_physical)
            
            # Update motion state
            self.movement_body._update_motion_state_from_rpm()
        
        self.movement_body._update_awareness_conf()

    def _update_rpm_for_cycles(self, t_us: int, cycles_physical: float) -> None:
        """
        v1.3: Update RPM gebaseerd op werkelijke cycles.
        
        RPM = (cycles / dt_seconds) × (60 / cycles_per_rot)
        """
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

    def debug_tiles_and_cycles(self) -> Dict[str, Any]:
        return {
            "cycles": self.cycles_state.debug_summary(),
            "tiles": self.tiles_state.debug_summary(),
            "total_cycles_physical": self._total_cycles_physical,
        }

    def snapshot(self) -> RealtimeSnapshot:
        movement_state_obj = self.movement_body.snapshot()
        movement_state = movement_state_obj.__dict__.copy()
        movement_state.update({
            "total_cycles_physical": self._total_cycles_physical,
        })
        
        return RealtimeSnapshot(
            t_us=self._last_t_us,
            cycles_emitted=[],
            tiles_emitted=[],
            compass_snapshot=self._last_compass_snapshot,
            movement_state=movement_state,
            total_cycles_physical=self._total_cycles_physical,
        )
