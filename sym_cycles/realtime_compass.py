#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sym_cycles.realtime_compass

Realtime-achtige simulatie van CompassSign v3.0 door tiles één voor één
door een state-machine te voeren, met een schuivend window van tiles.

Doel:
- laten zien hoe richting + confidence zich ontwikkelen over de tijd,
- zonder grote JSON logs in realtime,
- volledig inline met PhaseTiles v3 + CompassSign v3.

Gebruik:
  1) Run je bestaande offline pipeline (events → tiles):
       python3 -m sym_cycles.pipeline_offline full core0_events.jsonl out_sym_cycles

  2) Start de realtime compass simulatie op tiles_v3:
       ppython3 -m sym_cycles.realtime_compass from-tiles out_sym_cycles/tiles_v3.json \
				--mode inertial \
				--window-tiles 20 \
				--quiet \
				--plot
  3) Of via CLI tunen
			python3 -m sym_cycles.realtime_compass from-tiles out_sym_cycles/tiles_v3.json \
				--mode inertial \
				--window-tiles 20 \
				--alpha 0.95 \
				--threshold-high 0.5 \
				--threshold-low 0.25 \
				--quiet \
				--plot
"""

from __future__ import annotations

import argparse
import json
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Any, Iterable, Optional, List

# Relatieve imports binnen het pakket, met fallback als script
try:
    from . import compass_sign_v3
except ImportError:  # stand-alone fallback
    import compass_sign_v3  # type: ignore


@dataclass
class CompassWindowSnapshot:
    """Instantane snapshot van het realtime kompasvenster."""
    t_tile_index: int
    n_tiles: int
    global_direction: str
    confidence: float
    verdict_tiles: str
    meta: Dict[str, Any]


class CompassRealtimeState:
    """
    Realtime-like CompassSign state.

    Houdt een schuivend venster met tiles bij (laatste N tiles) en
    herberekent bij iedere nieuwe tile de richting + confidence met
    CompassSign v3.0.
    """

    def __init__(
        self,
        window_tiles: int = 20,
        config: Optional[compass_sign_v3.CompassSignConfig] = None,
        phase_class: str = "any",
    ) -> None:
        if config is None:
            config = compass_sign_v3.CompassSignConfig(
                deadzone_us=100.0,
                min_tiles=3,
                max_abs_dt_us=500_000.0,
                phase_class=phase_class,
            )
        self.window_tiles: int = window_tiles
        self.config: compass_sign_v3.CompassSignConfig = config
        self.phase_class: str = phase_class

        self.tiles: Deque[Dict[str, Any]] = deque(maxlen=window_tiles)
        self.tile_duration_us: Optional[float] = None
        self.tile_index_counter: int = 0  # hoeveel tiles tot nu gezien (monotone teller)

    def feed_tile(
        self,
        tile: Dict[str, Any],
        tile_duration_us: Optional[float] = None,
    ) -> CompassWindowSnapshot:
        """
        Voeg één tile toe aan het venster en bereken een nieuwe snapshot.

        tile: verwacht minstens 'dt_ab_us' en optioneel andere velden.
        """
        self.tiles.append(tile)
        self.tile_index_counter += 1

        # tile_duration_us updaten indien gegeven
        if tile_duration_us is not None:
            self.tile_duration_us = tile_duration_us

        # Geen zinvolle compass als te weinig tiles
        if len(self.tiles) < self.config.min_tiles:
            # UNDECIDED snapshot met confidence 0
            return CompassWindowSnapshot(
                t_tile_index=self.tile_index_counter - 1,
                n_tiles=len(self.tiles),
                global_direction="UNDECIDED",
                confidence=0.0,
                verdict_tiles="ONBEPAALD",
                meta={"reason": "NOT_ENOUGH_TILES"},
            )

        # Bereken CompassSign op het huidige venster
        tiles_list: List[Dict[str, Any]] = list(self.tiles)
        result = compass_sign_v3.compute_compass_sign_from_tiles(
            tiles=tiles_list,
            config=self.config,
            tile_duration_us=self.tile_duration_us,
        )

        return CompassWindowSnapshot(
            t_tile_index=self.tile_index_counter - 1,
            n_tiles=len(self.tiles),
            global_direction=result.global_direction,
            confidence=result.confidence,
            verdict_tiles=result.verdict_tiles,
            meta=result.meta,
        )

@dataclass
class InertialCompassSnapshot:
    """
    Snapshot van zowel het snelle window-kompas als het trage globale kompas.
    """
    t_tile_index: int
    n_tiles_window: int

    window_direction: str
    window_confidence: float
    window_verdict_tiles: str
    window_meta: Dict[str, Any]

    global_direction: str
    global_score: float
    global_meta: Dict[str, Any]


class InertialCompassState:
    """
    Inertiaal/global compass bovenop CompassRealtimeState.

    - CompassRealtimeState berekent per venster een snelle richting + confidence.
    - InertialCompassState houdt een exponentieel gemiddelde bij van deze
      window-signalen, en bepaalt daaruit een trage globale richting met hysterese.
    """

    def __init__(
        self,
        window_tiles: int = 20,
        phase_class: str = "any",
        alpha: float = 0.97,       # was 0.9  → trager
        threshold_high: float = 0.6,  # was 0.4 → moeilijker switchen
        threshold_low: float = 0.3,   # was 0.2 → bredere dode zone
    ) -> None:
        """
        Parameters:
        - window_tiles   : aantal tiles in het sliding window (voor het snelle kompas)
        - alpha          : traagheid (0..1), hoe hoger, hoe trager het globale kompas reageert
        - threshold_high : drempel voor "zekere" CW/CCW
        - threshold_low  : drempel voor undecided-zone
        """
        self.window_state = CompassRealtimeState(
            window_tiles=window_tiles,
            phase_class=phase_class,
        )

        self.alpha = alpha
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low

        self.global_score: float = 0.0
        self.global_direction: str = "UNDECIDED"

    def _dir_to_signed_score(self, direction: str, confidence: float) -> float:
        if direction == "CW":
            return +confidence
        if direction == "CCW":
            return -confidence
        return 0.0

    def _update_global_direction(self) -> None:
        g = self.global_score
        if abs(g) < self.threshold_low:
            self.global_direction = "UNDECIDED"
        elif g >= self.threshold_high:
            self.global_direction = "CW"
        elif g <= -self.threshold_high:
            self.global_direction = "CCW"
        # tussen threshold_low en threshold_high laten we de huidige richting staan

    def feed_tile(
        self,
        tile: Dict[str, Any],
        tile_duration_us: Optional[float] = None,
    ) -> InertialCompassSnapshot:
        """
        Verwerk één tile:
        - update snelle window-state,
        - update globale score,
        - bepaal globale richting.
        """
        w_snap = self.window_state.feed_tile(tile, tile_duration_us=tile_duration_us)

        # update globale score op basis van window-direction + confidence
        signed = self._dir_to_signed_score(
            w_snap.global_direction,
            w_snap.confidence,
        )
        self.global_score = self.alpha * self.global_score + (1.0 - self.alpha) * signed
        self._update_global_direction()

        return InertialCompassSnapshot(
            t_tile_index=w_snap.t_tile_index,
            n_tiles_window=w_snap.n_tiles,
            window_direction=w_snap.global_direction,
            window_confidence=w_snap.confidence,
            window_verdict_tiles=w_snap.verdict_tiles,
            window_meta=w_snap.meta,
            global_direction=self.global_direction,
            global_score=self.global_score,
            global_meta={
                "alpha": self.alpha,
                "threshold_high": self.threshold_high,
                "threshold_low": self.threshold_low,
            },
        )

# -------------------- helpers voor tiles_v3.json -----------------------------


def iter_tiles_from_tiles_v3_doc(doc: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Yield afzonderlijke tiles uit een tiles_v3-document.
    Verwacht veld 'tiles_v3' als lijst.
    """
    tiles = doc.get("tiles_v3", [])
    for tile in tiles:
        yield tile


def load_tiles_v3(path: str) -> (List[Dict[str, Any]], Optional[float]):
    """
    Laad tiles_v3.json en return (tiles, tile_duration_us).
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    tiles = list(iter_tiles_from_tiles_v3_doc(doc))
    tile_duration_us = None
    meta = doc.get("tiles_v3_meta", {})
    if isinstance(meta, dict):
        if "tile_duration_us" in meta:
            try:
                tile_duration_us = float(meta["tile_duration_us"])
            except (ValueError, TypeError):
                tile_duration_us = None

    return tiles, tile_duration_us


# ------------------------------ CLI -----------------------------------------


def cmd_from_tiles(args: argparse.Namespace) -> None:
    """
    CLI subcommand: simulate realtime compass from tiles_v3.json.

    Modes:
    - window   : alleen snel sliding window kompas
    - inertial : sliding window + globaal kompas met traagheid
    """
    tiles, tile_duration_us = load_tiles_v3(args.tiles_json)

    print(
        f"[i] Realtime compass simulatie vanaf tiles_v3: {args.tiles_json}\n"
        f"    window_tiles = {args.window_tiles}, "
        f"phase_class = {args.phase_class}, "
        f"tile_duration_us = {tile_duration_us}, "
        f"mode = {args.mode}"
    )

    snapshots_window: List[CompassWindowSnapshot] = []
    snapshots_inertial: List[InertialCompassSnapshot] = []

    last_dir_window: Optional[str] = None
    last_dir_global: Optional[str] = None

    if args.mode == "window":
        state = CompassRealtimeState(
            window_tiles=args.window_tiles,
            phase_class=args.phase_class,
        )

        for i, tile in enumerate(tiles):
            snap = state.feed_tile(tile, tile_duration_us=tile_duration_us)
            snapshots_window.append(snap)

            if args.quiet:
                if snap.n_tiles >= state.config.min_tiles:
                    if snap.global_direction != last_dir_window:
                        print(
                            f"{i:5d}  n_tiles={snap.n_tiles:3d}  "
                            f"dir_window={snap.global_direction:<9}  "
                            f"conf={snap.confidence:5.3f}"
                        )
                        last_dir_window = snap.global_direction
            else:
                print(
                    f"{i:5d}  n_tiles={snap.n_tiles:3d}  "
                    f"dir_window={snap.global_direction:<9}  "
                    f"conf={snap.confidence:5.3f}  "
                    f"verdict={snap.verdict_tiles}  "
                    f"meta={snap.meta}"
                )

    else:  # inertial mode
        state = InertialCompassState(
            window_tiles=args.window_tiles,
            phase_class=args.phase_class,
            alpha=args.alpha,
            threshold_high=args.threshold_high,
            threshold_low=args.threshold_low,
        )

        for i, tile in enumerate(tiles):
            snap = state.feed_tile(tile, tile_duration_us=tile_duration_us)
            snapshots_inertial.append(snap)

            if args.quiet:
                # Log alleen bij genoeg tiles + verandering in globale richting
                if snap.n_tiles_window >= state.window_state.config.min_tiles:
                    if snap.global_direction != last_dir_global:
                        print(
                            f"{i:5d}  n_tiles={snap.n_tiles_window:3d}  "
                            f"dir_window={snap.window_direction:<9}  "
                            f"conf_window={snap.window_confidence:5.3f}  "
                            f"dir_global={snap.global_direction:<9}  "
                            f"score_global={snap.global_score:5.3f}"
                        )
                        last_dir_global = snap.global_direction
            else:
                print(
                    f"{i:5d}  n_tiles={snap.n_tiles_window:3d}  "
                    f"dir_window={snap.window_direction:<9}  "
                    f"conf_window={snap.window_confidence:5.3f}  "
                    f"dir_global={snap.global_direction:<9}  "
                    f"score_global={snap.global_score:5.3f}  "
                    f"meta_window={snap.window_meta}  "
                    f"meta_global={snap.global_meta}"
                )

    # Plotten
    if args.plot:
        if args.mode == "window" and snapshots_window:
            xs = [s.t_tile_index for s in snapshots_window]
            confs = [s.confidence for s in snapshots_window]

            def dir_to_num(d: str) -> int:
                if d == "CW":
                    return 1
                if d == "CCW":
                    return -1
                return 0

            dirs = [dir_to_num(s.global_direction) for s in snapshots_window]

            fig, ax = plt.subplots()
            ax.set_title("Realtime CompassSign v3 – window confidence & direction")
            ax.set_xlabel("tile index")
            ax.set_ylabel("window confidence / direction code")
            ax.plot(xs, confs, label="window confidence")
            ax.step(xs, dirs, where="post", label="window direction (+1=CW,-1=CCW,0=UNDEC)")
            ax.grid(True, which="both", linestyle=":", linewidth=0.5)
            ax.legend(loc="best")
            plt.tight_layout()
            try:
                plt.show()
            except KeyboardInterrupt:
                print("\n[i] Plot afgebroken met Ctrl+C (KeyboardInterrupt genegeerd).")

        elif args.mode == "inertial" and snapshots_inertial:
            xs = [s.t_tile_index for s in snapshots_inertial]
            confs_window = [s.window_confidence for s in snapshots_inertial]
            scores_global = [s.global_score for s in snapshots_inertial]

            def dir_to_num(d: str) -> int:
                if d == "CW":
                    return 1
                if d == "CCW":
                    return -1
                return 0

            dirs_window = [dir_to_num(s.window_direction) for s in snapshots_inertial]
            dirs_global = [dir_to_num(s.global_direction) for s in snapshots_inertial]

            # Eén figuur, twee rijen: boven confidence/score, onder directions
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

            # Bovenste subplot: confidence + global_score
            ax1.set_title("Realtime CompassSign v3 – window confidence & global_score")
            ax1.set_ylabel("value")
            ax1.plot(xs, confs_window, label="window confidence")
            ax1.plot(xs, scores_global, label="global_score")
            ax1.grid(True, which="both", linestyle=":", linewidth=0.5)
            ax1.legend(loc="best")

            # Onderste subplot: directions
            ax2.set_title("window vs global direction")
            ax2.set_xlabel("tile index")
            ax2.set_ylabel("dir (+1=CW, -1=CCW, 0=UNDEC)")
            ax2.step(xs, dirs_window, where="post", label="window dir", alpha=0.7)
            ax2.step(xs, dirs_global, where="post", label="global dir", linewidth=2.0)
            ax2.grid(True, which="both", linestyle=":", linewidth=0.5)
            ax2.legend(loc="best")
            plt.tight_layout()
            try:
                plt.show()
            except KeyboardInterrupt:
                print("\n[i] Plot afgebroken met Ctrl+C (KeyboardInterrupt genegeerd).")



def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Realtime-achtige simulatie van CompassSign v3.0 uit tiles_v3.json"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_from_tiles = sub.add_parser(
        "from-tiles",
        help="Simuleer realtime kompas uit een tiles_v3 JSON bestand.",
    )
    p_from_tiles.add_argument(
        "tiles_json",
        help="Pad naar tiles_v3 JSON bestand (output van pipeline_offline).",
    )
    p_from_tiles.add_argument(
        "--window-tiles",
        type=int,
        default=20,
        help="Aantal tiles in het schuivend venster (default: 20).",
    )
    p_from_tiles.add_argument(
        "--phase-class",
        default="any",
        choices=["any", "TOP", "BOTTOM"],
        help="Phase-filter (zoals in CompassSign-config).",
    )
    p_from_tiles.add_argument(
        "--mode",
        default="inertial",
        choices=["window", "inertial"],
        help="Kompasmodus: 'window' = puur sliding window, "
             "'inertial' = globaal kompas met traagheid (default).",
    )
    p_from_tiles.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Traagheid voor globaal kompas (0..1, default 0.9).",
    )
    p_from_tiles.add_argument(
        "--threshold-high",
        type=float,
        default=0.4,
        help="Drempel voor zekere CW/CCW (default 0.4).",
    )
    p_from_tiles.add_argument(
        "--threshold-low",
        type=float,
        default=0.2,
        help="Drempel voor undecided-zone (default 0.2).",
    )
    p_from_tiles.add_argument(
        "--quiet",
        action="store_true",
        help="Kortere output (zonder volledige meta).",
    )
    p_from_tiles.add_argument(
        "--plot",
        action="store_true",
        help="Toon een grafiek van confidence en richting over tile-index.",
    )
    p_from_tiles.set_defaults(func=cmd_from_tiles)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
