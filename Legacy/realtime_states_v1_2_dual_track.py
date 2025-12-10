"""
S02 — Realtime-States v1.2 Dual Track
======================================

Incremental realtime chain:
    EVENT24 → CyclesState → TilesState → InertialCompass → MovementBodyV3

DUAL TRACK ARCHITECTUUR:
========================

Track 1: MOTION (bewegings-awareness) — UNSIGNED
    - Update bij ELKE tile (ook lege tiles)
    - Houdt bij: tile_index, tijd, unsigned rotaties, rpm
    - Onafhankelijk van direction lock

Track 2: COMPASS (richting-awareness) — SIGNED  
    - Update alleen bij tiles met stereo-data
    - Houdt bij: direction, lock_state, signed rotaties
    - Afhankelijk van direction lock voor sign

Wijzigingen t.o.v. v1.0:
-----------------------
1. Tijdgebaseerde tile-emissie (inclusief lege tiles)
2. Dual-track MovementBody integratie
3. Aparte tracking voor unsigned en signed values
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from collections import deque
import statistics

# === Constants ===============================================================

POOL_NEU = 0
POOL_N   = 1
POOL_S   = 2


# === Cycle detection (L1) ====================================================

class CyclesState:
    """
    Incrementele 3-punts cycle-detectie per sensor.
    Ongewijzigd t.o.v. v1.0.
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


# === Time-based tiles (L2) — v1.2 Dual Track ================================

class TilesState:
    """
    v1.2 Dual Track: Tijdgebaseerde tile-emissie.
    
    Alle tiles worden geëmit op basis van TIJD, inclusief lege tiles.
    Elke tile bevat:
    - tile_index: opeenvolgend (geen gaps)
    - t_center_us: altijd aanwezig
    - has_data: True als nA > 0 or nB > 0
    - nA, nB: aantal samples per sensor
    """

    def __init__(self,
                 tile_span_cycles: float = 0.6,
                 boot_cycles_for_median: int = 24):
        self.tile_span_cycles = float(tile_span_cycles)
        self.boot_cycles_for_median = int(boot_cycles_for_median)

        self._boot_dt_samples: List[float] = []
        self._tile_duration_us: Optional[float] = None

        self._t0_us: Optional[float] = None
        self._current_tile_index: Optional[int] = None
        self._current_tile_data = {"A": [], "B": []}

        self._tiles_emitted = 0
        self._empty_tiles_emitted = 0
        self._data_tiles_emitted = 0

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

    def _make_tile(self, idx: int, ts_A: List[float], ts_B: List[float]) -> Dict[str, Any]:
        """Maak een tile met expliciete data arrays."""
        t_start = (self._t0_us +
                   idx * self._tile_duration_us) if self._tile_duration_us else self._t0_us or 0.0
        t_end = t_start + (self._tile_duration_us or 0.0)
        t_center = (t_start + t_end) / 2

        tA = sum(ts_A) / len(ts_A) if ts_A else None
        tB = sum(ts_B) / len(ts_B) if ts_B else None
        dt_ab = (tB - tA) if (tA is not None and tB is not None) else None

        has_data = len(ts_A) > 0 or len(ts_B) > 0

        tile = {
            "tile_index": idx,
            "t_start_us": t_start,
            "t_end_us": t_end,
            "t_center_us": t_center,
            "tA_us": tA,
            "tB_us": tB,
            "dt_ab_us": dt_ab,
            "nA": len(ts_A),
            "nB": len(ts_B),
            "has_data": has_data,
            "tile_span_cycles": self.tile_span_cycles,
        }

        self._tiles_emitted += 1
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
            self._current_tile_data["A"],
            self._current_tile_data["B"]
        )
        tiles.append(tile)
        
        # Emit lege tiles voor tussenliggende indices
        for idx in range(self._current_tile_index + 1, target_idx):
            empty_tile = self._make_tile(idx, [], [])
            tiles.append(empty_tile)
        
        # Reset data verzameling
        self._current_tile_data = {"A": [], "B": []}
        
        return tiles

    def feed_cycles(self, cycles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """v1.2: Tijdgebaseerde tile-emissie met lege tiles."""
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

            self._current_tile_data[sensor].append(t_center)

        return new_tiles

    def finalize(self) -> List[Dict[str, Any]]:
        """Flush de laatste tile."""
        tiles: List[Dict[str, Any]] = []
        if self._current_tile_index is not None:
            tile = self._make_tile(
                self._current_tile_index,
                self._current_tile_data["A"],
                self._current_tile_data["B"]
            )
            tiles.append(tile)
            self._current_tile_data = {"A": [], "B": []}
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
        }


# === Compass Snapshot adapter (L2.5) =========================================

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
    """Adapter boven InertialCompassState."""

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
    
    # v1.2: Extra tracking
    data_tiles_this_batch: int = 0
    empty_tiles_this_batch: int = 0


# === RealtimePipeline (public API) — v1.2 Dual Track ========================

class RealtimePipeline:
    """
    S02-Realtime-States v1.2 Dual Track
    
    Implementeert dual-track architectuur:
    - Track 1 (MOTION): Update bij elke tile
    - Track 2 (COMPASS): Update alleen bij tiles met data
    """

    def __init__(self,
                 cycles_per_rot: float = 12.0,
                 compass_window_tiles: int = 20,
                 compass_phase_class: str = "any",
                 compass_alpha: float = 0.95,
                 compass_threshold_high: float = 0.6,
                 compass_threshold_low: float = 0.25,
                 tile_span_cycles: float = 0.6):
        
        self.tile_span_cycles = tile_span_cycles
        self.cycles_per_rot = cycles_per_rot
        
        self.cycles_state = CyclesState()
        self.tiles_state = TilesState(
            tile_span_cycles=tile_span_cycles,
            boot_cycles_for_median=24,
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
        
        # v1.2: Track voor unsigned movement (los van MovementBody)
        self._motion_track = {
            "cycles_unsigned": 0.0,
            "rotations_unsigned": 0.0,
            "theta_unsigned_deg": 0.0,
            "tiles_processed": 0,
            "data_tiles_processed": 0,
            "rpm_unsigned_est": 0.0,
            "last_tile_t_us": None,
        }

    def _update_motion_track(self, tile: Dict[str, Any]) -> None:
        """v1.2: Update unsigned motion tracking voor elke tile."""
        mt = self._motion_track
        C = self.cycles_per_rot
        tile_span = tile.get("tile_span_cycles", self.tile_span_cycles)
        t_us = tile.get("t_center_us", 0)
        has_data = tile.get("has_data", False)
        
        # Accumuleer cycles en rotaties
        mt["cycles_unsigned"] += tile_span
        mt["rotations_unsigned"] = mt["cycles_unsigned"] / C
        mt["theta_unsigned_deg"] = (mt["rotations_unsigned"] * 360.0) % 360.0
        mt["tiles_processed"] += 1
        
        if has_data:
            mt["data_tiles_processed"] += 1
        
        # RPM berekening
        if mt["last_tile_t_us"] is not None and t_us > mt["last_tile_t_us"]:
            dt_us = t_us - mt["last_tile_t_us"]
            dt_s = dt_us * 1e-6
            if dt_s > 0:
                rpm_inst = (tile_span * 60.0) / (dt_s * C)
                # EMA smoothing
                alpha = 0.3
                if mt["rpm_unsigned_est"] <= 0:
                    mt["rpm_unsigned_est"] = rpm_inst
                else:
                    mt["rpm_unsigned_est"] = (1 - alpha) * mt["rpm_unsigned_est"] + alpha * rpm_inst
        
        mt["last_tile_t_us"] = t_us

    def feed_event(self, ev: Dict[str, Any]) -> RealtimeSnapshot:
        """
        v1.2 Dual Track: Verwerk één EVENT24.
        
        - Track 1 (MOTION): Update bij elke tile
        - Track 2 (COMPASS): Update alleen bij tiles met stereo-data
        """
        if isinstance(ev.get("t_abs_us"), (int, float)):
            self._last_t_us = int(ev["t_abs_us"])

        # L1: EVENT24 → 3-punts cycles
        cycles = self.cycles_state.feed_event(ev)

        # L2: cycles → tiles (inclusief lege)
        tiles = self.tiles_state.feed_cycles(cycles)

        compass_snap: Optional[CompassSnapshot] = self._last_compass_snapshot
        data_tiles = 0
        empty_tiles = 0

        for tile in tiles:
            has_data = tile.get("has_data", False)
            
            if has_data:
                data_tiles += 1
            else:
                empty_tiles += 1
            
            # === TRACK 1: MOTION (altijd) ===
            self._update_motion_track(tile)
            
            # === TRACK 2: COMPASS (alleen bij data) ===
            if has_data:
                # Update kompas
                compass_snap = self.compass_adapter.feed_tile(
                    tile,
                    tile_duration_us=self.tiles_state.tile_duration_us,
                )
                self._last_compass_snapshot = compass_snap

                # Update MovementBody met kompas + cycle node
                self.movement_body.set_compass_realtime({
                    "global_direction": compass_snap.global_direction,
                    "global_score": compass_snap.global_score,
                    "window_direction": compass_snap.window_direction,
                    "window_confidence": compass_snap.window_confidence,
                })

                # Bepaal t_center
                t_center = tile.get("tA_us") or tile.get("t_center_us") or self._last_t_us or 0

                node = {
                    "t_center_us": t_center,
                    "cycle_type": "tile_cycle",
                    "tile_index": tile.get("tile_index"),
                    "tile_span_cycles": self.tile_span_cycles,
                    "has_data": True,
                }
                self.movement_body.feed_cycle_node(node)

        # Combineer movement state met motion track
        # NB: snapshot() retourneert MovementBodyStateV3 dataclass, converteer naar dict
        movement_state_obj = self.movement_body.snapshot()
        movement_state = movement_state_obj.__dict__.copy()
        movement_state.update({
            # v1.2: Voeg unsigned motion track toe
            "cycles_unsigned": self._motion_track["cycles_unsigned"],
            "rotations_unsigned": self._motion_track["rotations_unsigned"],
            "theta_unsigned_deg": self._motion_track["theta_unsigned_deg"],
            "tiles_processed": self._motion_track["tiles_processed"],
            "data_tiles_processed": self._motion_track["data_tiles_processed"],
            "rpm_unsigned_est": self._motion_track["rpm_unsigned_est"],
        })

        return RealtimeSnapshot(
            t_us=self._last_t_us,
            cycles_emitted=cycles,
            tiles_emitted=tiles,
            compass_snapshot=compass_snap,
            movement_state=movement_state,
            data_tiles_this_batch=data_tiles,
            empty_tiles_this_batch=empty_tiles,
        )

    def debug_tiles_and_cycles(self) -> Dict[str, Any]:
        return {
            "cycles": self.cycles_state.debug_summary(),
            "tiles": self.tiles_state.debug_summary(),
            "motion_track": self._motion_track.copy(),
        }

    def snapshot(self) -> RealtimeSnapshot:
        movement_state_obj = self.movement_body.snapshot()
        movement_state = movement_state_obj.__dict__.copy()
        movement_state.update({
            "cycles_unsigned": self._motion_track["cycles_unsigned"],
            "rotations_unsigned": self._motion_track["rotations_unsigned"],
            "theta_unsigned_deg": self._motion_track["theta_unsigned_deg"],
            "tiles_processed": self._motion_track["tiles_processed"],
            "data_tiles_processed": self._motion_track["data_tiles_processed"],
            "rpm_unsigned_est": self._motion_track["rpm_unsigned_est"],
        })
        
        return RealtimeSnapshot(
            t_us=self._last_t_us,
            cycles_emitted=[],
            tiles_emitted=[],
            compass_snapshot=self._last_compass_snapshot,
            movement_state=movement_state,
        )
