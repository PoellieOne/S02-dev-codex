#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sym_cycles.phase_tiles_v3_0

PhaseTiles v3.0 — Cycle-period-based time tiling

# TileSpan default (0.6) per S02.M082

Deze versie gebruikt GEEN vaste run-frac bins meer, maar bouwt tijd-tiles
op basis van de typische cycle-duur:

  1) Schat median dt_us over alle cycles (na filtering).
  2) tile_duration_us = tile_span_cycles * median_dt_us.
  3) Deel de run op in tiles van tile_duration_us breed.
  4) Groepeer per tile de cycles van sensor A en B.
  5) Bepaal per tile tA_us, tB_us en dt_ab_us = tB_us - tA_us.
  6) Filter optioneel tiles met |dt_ab_us| > max_abs_dt_us (verdacht).
  7) Bepaal median dt_ab_us en geef verdict:
       A_LEADS_B (median > 0),
       B_LEADS_A (median < 0),
       ONBEPAALD  (te weinig tiles of median in deadzone).

Input:
  - *_cycles_phase.json (met top-level "cycles": [...])
    of sensors-formaat (A/B full_cycles), dat wordt geflattened.
"""

import json
import argparse
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers voor laden, flattenen en statistiek
# ---------------------------------------------------------------------------

def load_cycles_generic(path: str) -> Dict[str, Any]:
    """
    Laad een cycles/phase JSON bestand en zorg dat er een top-level "cycles"
    lijst aanwezig is.

    Ondersteunt:
      1) {"cycles": [ ... ]}
      2) {"sensors": {"A": {"full_cycles": [...]}, "B": {...}}}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data.get("cycles"), list):
        return data

    sensors = data.get("sensors")
    if isinstance(sensors, dict):
        all_cycles: List[Dict[str, Any]] = []
        for sensor_name in ("A", "B"):
            s = sensors.get(sensor_name)
            if not isinstance(s, dict):
                continue
            full = s.get("full_cycles", [])
            if isinstance(full, list):
                for c in full:
                    if not isinstance(c, dict):
                        continue
                    c.setdefault("sensor", sensor_name)
                    all_cycles.append(c)
        data["cycles"] = all_cycles
        return data

    # Fallback
    data["cycles"] = []
    return data


def summarize_sorted(values: List[float]) -> Dict[str, Optional[float]]:
    """
    Neem een gesorteerde lijst en geef min/p25/median/p75/max terug.
    """
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }

    def pick(p: float) -> float:
        if n == 1:
            return values[0]
        idx = int(round((n - 1) * p))
        if idx < 0:
            idx = 0
        if idx >= n:
            idx = n - 1
        return values[idx]

    return {
        "n": n,
        "min": values[0],
        "p25": pick(0.25),
        "median": pick(0.5),
        "p75": pick(0.75),
        "max": values[-1],
    }


# ---------------------------------------------------------------------------
# Kern: tile-bouw op basis van cycle-periode
# ---------------------------------------------------------------------------

def filter_cycles_phase_class(
    cycles: List[Dict[str, Any]],
    phase_class_filter: str = "any",
) -> List[Dict[str, Any]]:
    """
    Filter cycles op phase_class (indien aanwezig) of op cycle_type -> TOP/BOTTOM mapping.

    phase_class_filter:
      "any"     -> geen filter
      "TOP"     -> alleen TOP
      "BOTTOM"  -> alleen BOTTOM
    """
    if phase_class_filter.lower() == "any":
        return [c for c in cycles if c.get("sensor") in ("A", "B")]

    result: List[Dict[str, Any]] = []
    for c in cycles:
        sensor = c.get("sensor")
        if sensor not in ("A", "B"):
            continue

        pc = c.get("phase_class")
        if pc is not None:
            if pc == phase_class_filter:
                result.append(c)
            continue

        # fallback: map cycle_type -> phase_class
        ct = c.get("cycle_type")
        mapped = None
        if ct == "cycle_down":
            mapped = "TOP"
        elif ct == "cycle_up":
            mapped = "BOTTOM"
        else:
            mapped = "UNKNOWN"

        if mapped == phase_class_filter:
            result.append(c)

    return result


def estimate_median_dt_us(
    cycles: List[Dict[str, Any]],
    max_dt_for_guess_us: float = 2_000_000.0,
) -> Optional[float]:
    """
    Schat median dt_us over alle cycles, met filtering op dt_us > 0 en dt_us < max_dt_for_guess_us.
    """
    dt_values: List[float] = []
    for c in cycles:
        dt = c.get("dt_us")
        if not isinstance(dt, (int, float)):
            continue
        dt_f = float(dt)
        if dt_f <= 0:
            continue
        if dt_f > max_dt_for_guess_us:
            continue
        dt_values.append(dt_f)

    dt_values.sort()
    stats = summarize_sorted(dt_values)
    if stats["n"] == 0:
        return None
    return stats["median"]


def build_time_tiles_by_period(
    cycles: List[Dict[str, Any]],
    tile_span_cycles: float,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Bouw tiles op basis van median cycle-periode:
      - schat median dt_us
      - tile_duration_us = tile_span_cycles * median_dt_us
      - maak tiles [t_min + k*tile_duration, t_min + (k+1)*tile_duration)
      - groepeer times per tile en per sensor

    Return:
      (tiles, tile_duration_us)

    tiles: lijst van dicts:
      {
        "tile": <int>,
        "t_start_us": <float>,
        "t_end_us": <float>,
        "tA_us": <float|None>,
        "tB_us": <float|None>,
        "dt_ab_us": <float|None>,
        "nA": <int>,
        "nB": <int>,
      }
    """
    # Eerst tijdrange vaststellen
    t_values: List[float] = []
    for c in cycles:
        t = c.get("t_center_us")
        if isinstance(t, (int, float)):
            t_values.append(float(t))

    if not t_values:
        return [], 0.0

    t_min = min(t_values)
    t_max = max(t_values)
    span = max(1.0, t_max - t_min)

    # Schat median cycle dt_us
    median_dt = estimate_median_dt_us(cycles)
    if median_dt is None or median_dt <= 0:
        # fallback: deel run grofweg in 8 tiles
        tile_duration_us = span / 8.0
    else:
        tile_duration_us = tile_span_cycles * median_dt

    if tile_duration_us <= 0:
        tile_duration_us = span / 8.0

    # Maak buckets per tile index
    tiles_tmp: Dict[int, Dict[str, List[float]]] = {}

    for c in cycles:
        sensor = c.get("sensor")
        if sensor not in ("A", "B"):
            continue
        t = c.get("t_center_us")
        if not isinstance(t, (int, float)):
            continue
        t_f = float(t)
        rel = (t_f - t_min) / tile_duration_us
        if rel < 0:
            idx = 0
        else:
            idx = int(rel)
        bucket = tiles_tmp.setdefault(idx, {"A": [], "B": []})
        bucket[sensor].append(t_f)

    # Zet om naar gestructureerde tiles
    tiles: List[Dict[str, Any]] = []
    for idx in sorted(tiles_tmp.keys()):
        bucket = tiles_tmp[idx]
        ts_A = bucket.get("A", [])
        ts_B = bucket.get("B", [])
        tA: Optional[float] = None
        tB: Optional[float] = None
        dt: Optional[float] = None

        if ts_A:
            tA = sum(ts_A) / len(ts_A)
        if ts_B:
            tB = sum(ts_B) / len(ts_B)
        if tA is not None and tB is not None:
            dt = tB - tA

        t_start = t_min + idx * tile_duration_us
        t_end = t_start + tile_duration_us

        tiles.append(
            {
                "tile": idx,
                "t_start_us": t_start,
                "t_end_us": t_end,
                "tA_us": tA,
                "tB_us": tB,
                "dt_ab_us": dt,
                "nA": len(ts_A),
                "nB": len(ts_B),
            }
        )

    return tiles, tile_duration_us


def direction_from_tiles(
    tiles: List[Dict[str, Any]],
    deadzone_us: float = 100.0,
    min_tiles: int = 3,
    max_abs_dt_us: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Bepaal richting op basis van dt_ab_us per tile.

    - dt_ab_us = tB_us - tA_us
    - optioneel: filter tiles met |dt| > max_abs_dt_us (vermijden van outliers)
    - deadzone_us: |median_dt| < deadzone_us => ONBEPAALD
    - min_tiles: minimale number of tiles met geldige dt_ab_us
    """
    dt_values: List[float] = []
    for t in tiles:
        dt = t.get("dt_ab_us")
        if not isinstance(dt, (int, float)):
            continue
        dt_f = float(dt)
        if max_abs_dt_us is not None and abs(dt_f) > max_abs_dt_us:
            continue
        dt_values.append(dt_f)

    dt_values.sort()
    stats = summarize_sorted(dt_values)

    verdict = "ONBEPAALD"
    reason = ""

    n = stats["n"]
    median = stats["median"]

    if n < min_tiles:
        verdict = "ONBEPAALD"
        reason = f"te weinig tiles met geldige dt_ab_us (n={n}, min={min_tiles})"
    elif median is None:
        verdict = "ONBEPAALD"
        reason = "kan median dt_ab_us niet bepalen"
    else:
        if abs(median) < deadzone_us:
            verdict = "ONBEPAALD"
            reason = f"median dt_ab_us te dicht bij 0 (|{median:.1f}| < {deadzone_us:.1f} µs)"
        elif median > 0:
            verdict = "A_LEADS_B"
            reason = f"median dt_ab_us > 0 (A ziet tiles gemiddeld eerder, median={median:.1f} µs)"
        else:
            verdict = "B_LEADS_A"
            reason = f"median dt_ab_us < 0 (B ziet tiles gemiddeld eerder, median={median:.1f} µs)"

    extra = {
        "n_tiles_with_dt": n,
        "dt_stats": stats,
        "deadzone_us": deadzone_us,
        "min_tiles": min_tiles,
        "max_abs_dt_us": max_abs_dt_us,
        "reason": reason,
    }
    return verdict, extra


def save_tiles_v3(
    data: Dict[str, Any],
    tiles: List[Dict[str, Any]],
    tile_duration_us: float,
    path: str,
) -> None:
    out = dict(data)
    out["tiles_v3"] = tiles
    out["tiles_v3_meta"] = {
        "tile_duration_us": tile_duration_us,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)


def print_summary(
    tiles: List[Dict[str, Any]],
    verdict: str,
    extra: Dict[str, Any],
    phase_class_filter: str,
    input_path: str,
    tile_duration_us: float,
) -> None:
    print("=============================================")
    print(" PhaseTiles analyzer v3.0")
    print("=============================================")
    print(f"  bestand        : {input_path}")
    print(f"  phase_class    : {phase_class_filter}")
    print(f"  tiles totaal   : {len(tiles)}")
    print(f"  tile_duration  : {tile_duration_us:.1f} µs (typisch ~tile_span_cycles * median dt_us)")

    both = [t for t in tiles if t.get("tA_us") is not None and t.get("tB_us") is not None]
    only_A = [t for t in tiles if t.get("tA_us") is not None and t.get("tB_us") is None]
    only_B = [t for t in tiles if t.get("tA_us") is None and t.get("tB_us") is not None]

    print(f"  tiles met A&B  : {len(both)}")
    print(f"  tiles alleen A : {len(only_A)}")
    print(f"  tiles alleen B : {len(only_B)}")
    print("---------------------------------------------")

    stats = extra.get("dt_stats", {})
    n = stats.get("n", 0)
    print("  dt_ab_us statistiek over tiles met geldige A&B (+ dt-filter):")
    print(f"    n     : {n}")
    print(f"    min   : {stats.get('min')}")
    print(f"    p25   : {stats.get('p25')}")
    print(f"    median: {stats.get('median')}")
    print(f"    p75   : {stats.get('p75')}")
    print(f"    max   : {stats.get('max')}")
    print()
    print(f"  deadzone_us   : {extra.get('deadzone_us')}")
    print(f"  min_tiles     : {extra.get('min_tiles')}")
    print(f"  max_abs_dt_us : {extra.get('max_abs_dt_us')}")
    print()
    print("=== PhaseTiles v3.0 — Globaal verdict ===")
    print(f"  direction : {verdict}")
    print(f"  reason    : {extra.get('reason')}")
    print()
    print("  Interpretatie:")
    print("    A_LEADS_B  -> A ziet gemiddeld eerder dan B")
    print("    B_LEADS_A  -> B ziet gemiddeld eerder dan A")
    print("    ONBEPAALD  -> geen duidelijke lead/lag of te weinig tiles")
    print("=============================================")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Symphonia PhaseTiles v3.0 — Cycle-period-based time tiling"
    )
    ap.add_argument(
        "cycles_phase_json",
        help="input *_cycles_phase.json (of cycles.json met sensors-structuur)",
    )
    ap.add_argument(
        "--phase-class",
        choices=["any", "TOP", "BOTTOM"],
        default="any",
        help="filter op phase_class (any/TOP/BOTTOM) voor tile-berekening",
    )
    ap.add_argument(
        "--tile-span-cycles",
        type=float,
        default=3.0,
        help="hoeveel cycles per tile (default 3.0, dus tile_duration = 3 * median dt_us)",
    )
    ap.add_argument(
        "--deadzone-us",
        type=float,
        default=100.0,
        help="deadzone rond 0 voor median dt_ab_us (default 100 µs)",
    )
    ap.add_argument(
        "--min-tiles",
        type=int,
        default=3,
        help="minimaal aantal tiles met geldige dt_ab_us voor verdict (default 3)",
    )
    ap.add_argument(
        "--max-abs-dt-us",
        type=float,
        default=500_000.0,
        help="maximale |dt_ab_us| per tile; grotere |dt| wordt genegeerd als outlier (default 500000 µs = 0.5 s, gebruik 0 of negatief om uit te zetten)",
    )
    ap.add_argument(
        "--output",
        help="output JSON (default: vervang .json door _tiles_v3.json)",
    )

    args = ap.parse_args()

    data = load_cycles_generic(args.cycles_phase_json)
    cycles_all = data.get("cycles", [])

    print(f"[i] PhaseTiles v3.0 — input: {args.cycles_phase_json}")
    print(f"[i]  cycles (raw): {len(cycles_all)}, phase_class filter: {args.phase_class}")

    cycles = filter_cycles_phase_class(cycles_all, phase_class_filter=args.phase_class)
    print(f"[i]  cycles (na filter): {len(cycles)}")

    tiles, tile_duration_us = build_time_tiles_by_period(
        cycles=cycles,
        tile_span_cycles=args.tile_span_cycles,
    )

    out_path = args.output
    if not out_path:
        if args.cycles_phase_json.endswith(".json"):
            out_path = args.cycles_phase_json.replace(".json", "_tiles_v3.json")
        else:
            out_path = args.cycles_phase_json + "_tiles_v3.json"

    save_tiles_v3(data, tiles, tile_duration_us, out_path)
    print(f"[i] tiles_v3: geschreven naar {out_path}")

    max_abs_dt = args.max_abs_dt_us
    if max_abs_dt is not None and max_abs_dt <= 0:
        max_abs_dt = None

    verdict, extra = direction_from_tiles(
        tiles,
        deadzone_us=args.deadzone_us,
        min_tiles=args.min_tiles,
        max_abs_dt_us=max_abs_dt,
    )

    print_summary(
        tiles=tiles,
        verdict=verdict,
        extra=extra,
        phase_class_filter=args.phase_class,
        input_path=args.cycles_phase_json,
        tile_duration_us=tile_duration_us,
    )


if __name__ == "__main__":
    main()
