#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sym_cycles.pipeline_offline

Offline glue-pipeline voor Symphonia S02:

events (EVENT24 JSONL) → cycles → phase → PhaseTiles v3 → CompassSign v3.0

Deze module hergebruikt de bestaande v1.x scripts:
- builder_v1_0.py
- phase_universe_v1_0.py
- phase_tiles_v3_0.py
- compass_sign_v3.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Probeer eerst package-relative imports (voor python -m sym_cycles.pipeline_offline)
try:
    from . import builder_v1_0 as builder
    from . import phase_universe_v1_0 as phase_universe
    from . import phase_tiles_v3_0 as phase_tiles
    from . import compass_sign_v3 as compass_sign
except ImportError:
    # Fallback: direct imports als het script stand-alone gedraaid wordt
    import builder_v1_0 as builder
    import phase_universe_v1_0 as phase_universe
    import phase_tiles_v3_0 as phase_tiles
    import compass_sign_v3 as compass_sign


def events_to_cycles(events_jsonl: str, cycles_json: str) -> None:
    """
    Stap 1: EVENT24 JSONL → cycles per sensor (A/B).

    - gebruikt builder.load_events_from_event24(...)
    - gebruikt builder.build_cycles(...)
    - schrijft een JSON-structuur met "sensors" naar cycles_json
    """
    events_by_sensor = builder.load_events_from_event24(events_jsonl)
    cycles_by_sensor = builder.build_cycles(events_by_sensor)

    out: Dict[str, Any] = {
        "meta": {
            "source_events": events_jsonl,
            "builder_version": "1.0",
        },
        "sensors": cycles_by_sensor,
    }

    with open(cycles_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)


def cycles_to_phase(cycles_json: str, cycles_phase_json: str, tiles: int = 24) -> None:
    """
    Stap 2: cycles → Phase Universe tags (phase_class, run_frac, phase_bin).

    - gebruikt phase_universe.load_cycles(...)
    - gebruikt phase_universe.tag_cycles_with_phase(...)
    - gebruikt phase_universe.save_cycles_with_phase(...)
    """
    data = phase_universe.load_cycles(cycles_json)
    cycles = data.get("cycles", [])
    phase_universe.tag_cycles_with_phase(cycles, M=tiles)
    phase_universe.save_cycles_with_phase(data, cycles_phase_json)


def phase_to_tiles(
    cycles_phase_json: str,
    tiles_json: str,
    phase_class: str = "any",
    tile_span_cycles: float = 1.0,
) -> None:
    """
    Stap 3: Phase-tagged cycles → PhaseTiles v3.

    - gebruikt phase_tiles.load_cycles_generic(...)
    - gebruikt phase_tiles.filter_cycles_phase_class(...)
    - gebruikt phase_tiles.build_time_tiles_by_period(...)
    - gebruikt phase_tiles.save_tiles_v3(...)
    """
    doc = phase_tiles.load_cycles_generic(cycles_phase_json)
    cycles = doc.get("cycles", [])
    cycles_f = phase_tiles.filter_cycles_phase_class(
        cycles, phase_class_filter=phase_class
    )

    tiles, tile_duration_us = phase_tiles.build_time_tiles_by_period(
        cycles_f,
        tile_span_cycles=tile_span_cycles,
    )

    phase_tiles.save_tiles_v3(doc, tiles, tile_duration_us, tiles_json)


def tiles_to_compass(
    tiles_json: str,
    cfg: Optional[compass_sign.CompassSignConfig] = None,
    phase_class: Optional[str] = None,
) -> compass_sign.CompassSignResult:
    """
    Stap 4: PhaseTiles v3 → CompassSign v3.0 resultaat.

    - gebruikt compass_sign.compute_compass_sign_from_tiles_v3_doc(...)
    """
    with open(tiles_json, "r", encoding="utf-8") as f:
        doc = json.load(f)

    result = compass_sign.compute_compass_sign_from_tiles_v3_doc(
        doc,
        config=cfg,
        phase_class=phase_class,
    )
    return result


def run_full_pipeline(
    events_jsonl: str,
    out_dir: str,
    phase_class: str = "any",
    tile_span_cycles: float = 1.0,
    deadzone_us: float = 100.0,
    min_tiles: int = 3,
    max_abs_dt_us: float = 500_000.0,
) -> Dict[str, Any]:
    """
    Convenience: volledige pipeline events → compass in één call.

    Maakt (indien nodig) out_dir en schrijft:
      - cycles.json
      - cycles_phase.json
      - tiles_v3.json

    Retourneert een dict met paden en CompassSignResult.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cycles_json = str(out_path / "cycles.json")
    cycles_phase_json = str(out_path / "cycles_phase.json")
    tiles_json = str(out_path / "tiles_v3.json")

    events_to_cycles(events_jsonl, cycles_json)
    cycles_to_phase(cycles_json, cycles_phase_json, tiles=24)
    phase_to_tiles(
        cycles_phase_json,
        tiles_json,
        phase_class=phase_class,
        tile_span_cycles=tile_span_cycles,
    )

    cfg = compass_sign.CompassSignConfig(
        deadzone_us=deadzone_us,
        min_tiles=min_tiles,
        max_abs_dt_us=max_abs_dt_us,
        phase_class=phase_class,
    )

    result = tiles_to_compass(
        tiles_json,
        cfg=cfg,
        phase_class=phase_class,
    )

    return {
        "events_jsonl": events_jsonl,
        "out_dir": str(out_path),
        "cycles_json": cycles_json,
        "cycles_phase_json": cycles_phase_json,
        "tiles_json": tiles_json,
        "compass": {
            "global_direction": result.global_direction,
            "confidence": result.confidence,
            "verdict_tiles": result.verdict_tiles,
            "meta": result.meta,
        },
    }


# ------------------------- CLI ----------------------------------------------


def _add_common_phase_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--phase-class",
        default="any",
        choices=["any", "TOP", "BOTTOM"],
        help="Filter voor phase_class ('any' = geen filter).",
    )
    p.add_argument(
        "--tile-span-cycles",
        type=float,
        default=1.0,
        help="Aantal cycle-periodes per tile (default: 1.0).",
    )


def _add_common_compass_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--deadzone-us",
        type=float,
        default=100.0,
        help="Deadzone voor |median dt_ab_us| (µs).",
    )
    p.add_argument(
        "--min-tiles",
        type=int,
        default=3,
        help="Minimaal aantal tiles met geldige dt_ab_us.",
    )
    p.add_argument(
        "--max-abs-dt-us",
        type=float,
        default=500_000.0,
        help="Filter tiles met |dt_ab_us| > deze waarde weg.",
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Offline pipeline: EVENT24 → cycles → phase → tiles → compass."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # events → cycles
    p_e2c = sub.add_parser(
        "events-to-cycles",
        help="Converteer EVENT24 JSONL naar cycles.json.",
    )
    p_e2c.add_argument("events_jsonl", help="Input EVENT24 JSONL bestand.")
    p_e2c.add_argument("cycles_json", help="Output cycles JSON bestand.")

    # cycles → phase
    p_c2p = sub.add_parser(
        "cycles-to-phase",
        help="Voeg phase-tags (Phase Universe) toe aan cycles JSON.",
    )
    p_c2p.add_argument("cycles_json", help="Input cycles JSON bestand.")
    p_c2p.add_argument("cycles_phase_json", help="Output cycles+phase JSON bestand.")
    p_c2p.add_argument(
        "--tiles",
        type=int,
        default=24,
        help="Aantal phase-bins (default: 24).",
    )

    # phase → tiles
    p_p2t = sub.add_parser(
        "phase-to-tiles",
        help="Bouw PhaseTiles v3 uit cycles+phase JSON.",
    )
    p_p2t.add_argument("cycles_phase_json", help="Input cycles+phase JSON bestand.")
    p_p2t.add_argument("tiles_json", help="Output tiles_v3 JSON bestand.")
    _add_common_phase_args(p_p2t)

    # tiles → compass
    p_t2c = sub.add_parser(
        "tiles-to-compass",
        help="Bepaal CompassSign v3.0 richting uit tiles_v3 JSON.",
    )
    p_t2c.add_argument("tiles_json", help="Input tiles_v3 JSON bestand.")
    _add_common_phase_args(p_t2c)
    _add_common_compass_args(p_t2c)

    # full pipeline
    p_full = sub.add_parser(
        "full",
        help="Volledige pipeline: events → cycles → phase → tiles → compass.",
    )
    p_full.add_argument("events_jsonl", help="Input EVENT24 JSONL bestand.")
    p_full.add_argument("out_dir", help="Output directory voor alle artefacten.")
    _add_common_phase_args(p_full)
    _add_common_compass_args(p_full)

    args = parser.parse_args(argv)

    if args.cmd == "events-to-cycles":
        events_to_cycles(args.events_jsonl, args.cycles_json)

    elif args.cmd == "cycles-to-phase":
        cycles_to_phase(args.cycles_json, args.cycles_phase_json, tiles=args.tiles)

    elif args.cmd == "phase-to-tiles":
        phase_to_tiles(
            args.cycles_phase_json,
            args.tiles_json,
            phase_class=args.phase_class,
            tile_span_cycles=args.tile_span_cycles,
        )

    elif args.cmd == "tiles-to-compass":
        cfg = compass_sign.CompassSignConfig(
            deadzone_us=args.deadzone_us,
            min_tiles=args.min_tiles,
            max_abs_dt_us=args.max_abs_dt_us,
            phase_class=args.phase_class,
        )
        result = tiles_to_compass(
            args.tiles_json,
            cfg=cfg,
            phase_class=args.phase_class,
        )
        print(
            f"global_direction={result.global_direction} "
            f"confidence={result.confidence:.3f} "
            f"verdict_tiles={result.verdict_tiles}"
        )

    elif args.cmd == "full":
        summary = run_full_pipeline(
            events_jsonl=args.events_jsonl,
            out_dir=args.out_dir,
            phase_class=args.phase_class,
            tile_span_cycles=args.tile_span_cycles,
            deadzone_us=args.deadzone_us,
            min_tiles=args.min_tiles,
            max_abs_dt_us=args.max_abs_dt_us,
        )
        result = summary["compass"]
        print(f"[i] Artefacten geschreven naar: {summary['out_dir']}")
        print(
            f"[i] CompassSign v3.0: direction={result['global_direction']} "
            f"confidence={result['confidence']:.3f} "
            f"verdict_tiles={result['verdict_tiles']}"
        )

    else:
        parser.error(f"Onbekend commando: {args.cmd!r}")


if __name__ == "__main__":
    main()
