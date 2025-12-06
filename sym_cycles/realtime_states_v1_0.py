"""
S02 — Realtime-States v1.0
==========================

Incremental realtime chain:
    EVENT24 → CyclesState → TilesState → InertialCompass → MovementBodyV3

Doel:
- Lichtgewicht incrementele cycles & tiles
- Compatible met sym_cycles.realtime_compass en movement_body_v3
- Uniforme feed_event() → snapshot() pipeline API
"""

from __future__ import annotations
from dataclasses import dataclass
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
    Input: losse EVENT24 events (dict).
    Output: lijst Cycle-dicts bij elke feed_event().
    """

    def __init__(self):
        # per sensor: schuivend venster van de laatste 3 events
        self._windows = {
            "A": deque(maxlen=3),
            "B": deque(maxlen=3),
        }
        # --- DEBUG / statistiek ---
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
        """
        Verwerk één EVENT24 event en probeer ermee cycles te detecteren.
        """
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

        # als we drie samples hebben → mogelijke 3-punts cycle
        if len(win) == 3:
            p0, p1, p2 = (w["to_pool"] for w in win)
            t0, t1, t2 = (w["t_us"] for w in win)
            unique = {p0, p1, p2}

            if unique == {POOL_NEU, POOL_N, POOL_S}:
                # detectie in lijn met builder_v1_0
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

                # --- DEBUG / statistiek bijhouden ---
                self._cycle_counts[s_label] += 1
                try:
                    self._dt_samples[s_label].append(float(t2 - t0))
                except Exception:
                    pass

        return cycles

    # --- Debug helpers -------------------------------------------------

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


# === Time-based tiles (L2) ===================================================

class TilesState:
    """
    Incrementele PhaseTiles v3-achtige tile-bouw.

    Strategie:
      1) Verzamel vroege cycles → mediane dt_us → tile_duration_us
      2) Na lock-in: elke cycle → tile_index → samples toevoegen
      3) Bij tile-index wissel → flush vorige tile
    """

    def __init__(self,
                 tile_span_cycles: float = 1.0,
                 boot_cycles_for_median: int = 24):
        self.tile_span_cycles = float(tile_span_cycles)
        self.boot_cycles_for_median = int(boot_cycles_for_median)

        self._boot_dt_samples: List[float] = []
        self._tile_duration_us: Optional[float] = None

        self._t0_us: Optional[float] = None
        self._current_tile_index: Optional[int] = None
        self._current_tile_data = {"A": [], "B": []}

        # --- DEBUG / statistiek ---
        self._tiles_emitted = 0

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

    def _flush_current_tile(self) -> Optional[Dict[str, Any]]:
        if self._current_tile_index is None:
            return None

        idx = self._current_tile_index
        t_start = (self._t0_us +
                   idx * self._tile_duration_us) if self._tile_duration_us else self._t0_us or 0.0
        t_end = t_start + (self._tile_duration_us or 0.0)

        ts_A = self._current_tile_data["A"]
        ts_B = self._current_tile_data["B"]

        tA = sum(ts_A) / len(ts_A) if ts_A else None
        tB = sum(ts_B) / len(ts_B) if ts_B else None
        dt = (tB - tA) if (tA is not None and tB is not None) else None

        tile = {
            "tile_index": idx,
            "t_start_us": t_start,
            "t_end_us": t_end,
            "tA_us": tA,
            "tB_us": tB,
            "dt_ab_us": dt,
            "nA": len(ts_A),
            "nB": len(ts_B),
        }

        self._tiles_emitted += 1  # DEBUG: tel tiles

        self._current_tile_data = {"A": [], "B": []}
        return tile

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
                flushed = self._flush_current_tile()
                if flushed:
                    new_tiles.append(flushed)
                self._current_tile_index = idx

            self._current_tile_data[sensor].append(t_center)

        return new_tiles

    def finalize(self) -> List[Dict[str, Any]]:
        """
        Flush eventueel de laatste tile aan het einde van een run.
        RealtimePipeline kan deze aanroepen na het laatste event.
        """
        tiles: List[Dict[str, Any]] = []
        t = self._flush_current_tile()
        if t is not None:
            tiles.append(t)
        return tiles

    def debug_summary(self) -> Dict[str, Any]:
        """
        Kleine samenvatting: hoeveel dt-samples, mediane dt, tile_duration, #tiles.
        """
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
    """
    Adapter boven InertialCompassState.
    Maakt CompassSnapshot dat MovementBody begrijpt.
    """

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


# === RealtimePipeline (public API) ===========================================

class RealtimePipeline:
    """
    S02-Realtime-States v1.0 — volledige realtime keten.
    """

    def __init__(self,
                 cycles_per_rot: float = 12.0,
                 compass_window_tiles: int = 20,
                 compass_phase_class: str = "any",
                 compass_alpha: float = 0.95,
                 compass_threshold_high: float = 0.6,
                 compass_threshold_low: float = 0.25,
                 tile_span_cycles: float = 0.6):
        self.cycles_state = CyclesState()
        self.tiles_state = TilesState(
            # tile_span_cycles=0.6 is S02.M082 (TileSpan-Resonantie)
            # → 12 / 0.6 = 20 tiles per omwenteling
            tile_span_cycles=tile_span_cycles,   # ← canon: 0.6
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


    def feed_event(self, ev: Dict[str, Any]) -> RealtimeSnapshot:
        """
        Hoofd-update entrypoint.

        - Verwerk één EVENT24
        - Update cycles / tiles / compass / movement body
        - Return een consistente snapshot van L0..L3
        """
        # Laatste bekende ruwe timestamp bewaren (niet per se 1-op-1 met tile-center)
        if isinstance(ev.get("t_abs_us"), (int, float)):
            self._last_t_us = int(ev["t_abs_us"])

        # L1: EVENT24 → 3-punts cycles (voor debugging / logging / future use)
        cycles = self.cycles_state.feed_event(ev)

        # L2: cycles → tiles
        tiles = self.tiles_state.feed_cycles(cycles)

        compass_snap: Optional[CompassSnapshot] = self._last_compass_snapshot

        # L2.5 + L3: per tile: kompas + 1 virtuele "cycle-node" (zoals offline)
        for tile in tiles:
            compass_snap = self.compass_adapter.feed_tile(
                tile,
                tile_duration_us=self.tiles_state.tile_duration_us,
            )
            self._last_compass_snapshot = compass_snap

            # 1) kompas naar MovementBody
            self.movement_body.set_compass_realtime({
                "global_direction": compass_snap.global_direction,
                "global_score": compass_snap.global_score,
                "window_direction": compass_snap.window_direction,
                "window_confidence": compass_snap.window_confidence,
            })

            # 2) 1 "cycle-node" per tile met t_center_us
            t_center = tile.get("tA_us")
            tB = tile.get("tB_us")
            if not isinstance(t_center, (int, float)):
                # fallback: midden tussen t_start_us en t_end_us
                t_start = tile.get("t_start_us")
                t_end = tile.get("t_end_us")
                if isinstance(t_start, (int, float)) and isinstance(t_end, (int, float)):
                    t_center = 0.5 * (float(t_start) + float(t_end))
                else:
                    # laatste redmiddel: gebruik _last_t_us als approximate center
                    t_center = float(self._last_t_us or 0)

            node = {
                "t_center_us": t_center,
                "cycle_type": "tile_cycle",
                "tile_index": tile.get("tile_index"),
            }
            self.movement_body.feed_cycle_node(node)

        movement_state = self.movement_body.snapshot()

        return RealtimeSnapshot(
            t_us=self._last_t_us,
            cycles_emitted=cycles,
            tiles_emitted=tiles,
            compass_snapshot=compass_snap,
            movement_state=movement_state,
        )


    def debug_tiles_and_cycles(self) -> Dict[str, Any]:
        """
        Geef een compacte debug-samenvatting voor cycles + tiles.
        Handig om offline vs realtime te vergelijken.
        """
        return {
            "cycles": self.cycles_state.debug_summary(),
            "tiles": self.tiles_state.debug_summary(),
        }


    def snapshot(self) -> RealtimeSnapshot:
        return RealtimeSnapshot(
            t_us=self._last_t_us,
            cycles_emitted=[],
            tiles_emitted=[],
            compass_snapshot=self._last_compass_snapshot,
            movement_state=self.movement_body.snapshot(),
        )
