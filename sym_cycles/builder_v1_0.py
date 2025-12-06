#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sym_cycles.builder

Cycle Builder v1.0 voor Symphonia L0B:
- Laadt event24 JSONL
- Bouwd per sensor volledige 3-punts cycles (N/NEU/S)
- Detecteert partial cycles (2-punts)
- Berekent timing-statistieken per cycle_type

Gebaseerd op de bestaande stereo-cycle analyse logica.
"""

import json
from collections import defaultdict, Counter
import math
from typing import List, Dict, Tuple, Any, Optional

# pool_t mapping uit firmware / EVENT24:
# 0 = POOL_NEU, 1 = POOL_N, 2 = POOL_S, 3 = POOL_UNK
POOL_NEU = 0
POOL_N   = 1
POOL_S   = 2
POOL_UNK = 3


Event = Dict[str, Any]
Cycle = Dict[str, Any]
PartialCycle = Dict[str, Any]


# ------------- helpers -------------------------------------------------------

def _quantile(sorted_vals: List[float], q: float) -> float:
    """Lineaire interpolatie tussen punten."""
    if not sorted_vals:
        raise ValueError("Empty list for quantile")
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    idx = (n - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    f = idx - lo
    return sorted_vals[lo] * (1.0 - f) + sorted_vals[hi] * f


def timing_stats_from_cycles(cycles: List[Cycle]) -> Dict[str, Dict[str, float]]:
    """
    Geeft per cycle_type (cycle_up/down/mixed) timing-statistieken:
    min / p25 / median / mean / p75 / max over dt_us.
    """
    by_type: Dict[str, List[float]] = defaultdict(list)
    for c in cycles:
        ctype = c.get("cycle_type", "unknown")
        dt = c.get("dt_us")
        if dt is None:
            continue
        by_type[ctype].append(float(dt))

    stats: Dict[str, Dict[str, float]] = {}
    for ctype, vals in by_type.items():
        vals = sorted(vals)
        n = len(vals)
        if n == 0:
            continue
        stats[ctype] = {
            "min": vals[0],
            "p25": _quantile(vals, 0.25),
            "median": _quantile(vals, 0.50),
            "mean": sum(vals) / n,
            "p75": _quantile(vals, 0.75),
            "max": vals[-1],
            "count": float(n),
        }
    return stats


# ------------- loading -------------------------------------------------------

def load_events_from_event24(path: str) -> Dict[int, List[Event]]:
    """
    Laadt event24 JSONL bestand en groepeert events per sensor.

    Verwacht records met:
      kind == "event24"
      sensor (0=A,1=B)
      to_pool, from_pool, t_abs_us, mono_q8, snr_q8, fit_err_q8
    """
    events_by_sensor: Dict[int, List[Event]] = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("kind") != "event24":
                continue

            sensor = obj.get("sensor")
            to_pool = obj.get("to_pool")
            t_abs_us = obj.get("t_abs_us")

            if sensor is None or to_pool is None or t_abs_us is None:
                continue
            if to_pool == POOL_UNK:
                continue

            e: Event = {
                "t_us": t_abs_us,
                "to_pool": to_pool,
                "from_pool": obj.get("from_pool"),
                "mono_q8": obj.get("mono_q8"),
                "snr_q8": obj.get("snr_q8"),
                "fit_err_q8": obj.get("fit_err_q8"),
            }
            events_by_sensor[sensor].append(e)

    for s in events_by_sensor:
        events_by_sensor[s].sort(key=lambda e: e["t_us"])

    return events_by_sensor


# ------------- core detection -----------------------------------------------

def detect_cycles_for_sensor(events: List[Event],
                             sensor_name: str = "") -> Tuple[List[Cycle], List[PartialCycle]]:
    """
    Detecteert 3-punts cycles (volledig) en 2-punts partial cycles voor één sensor.

    Volledige cycles:
      - {POOL_NEU, POOL_N, POOL_S} moet voorkomen
      - N->NEU->S  => cycle_up
      - S->NEU->N  => cycle_down
      - andere permutatie => cycle_mixed

    Partial cycles:
      - {POOL_NEU, POOL_N} => partial_N_neutral
      - {POOL_NEU, POOL_S} => partial_S_neutral
    """
    full_cycles: List[Cycle] = []
    partial_cycles: List[PartialCycle] = []

    pools = [e["to_pool"] for e in events]
    times = [e["t_us"] for e in events]

    # 3-punts windows
    for i in range(len(events) - 2):
        p0, p1, p2 = pools[i], pools[i+1], pools[i+2]
        t0, t1, t2 = times[i], times[i+1], times[i+2]

        unique = {p0, p1, p2}
        if unique == {POOL_NEU, POOL_N, POOL_S}:
            if   [p0, p1, p2] == [POOL_N, POOL_NEU, POOL_S]:
                ctype = "cycle_up"
            elif [p0, p1, p2] == [POOL_S, POOL_NEU, POOL_N]:
                ctype = "cycle_down"
            else:
                ctype = "cycle_mixed"

            cycle: Cycle = {
                "sensor": sensor_name,
                "cycle_type": ctype,
                "idx_start": i,
                "idx_end": i+2,
                "t_start_us": t0,
                "t_end_us": t2,
                "t_center_us": 0.5 * (t0 + t2),
                "dt_us": t2 - t0,
                "p0": p0,
                "p1": p1,
                "p2": p2,
                "mono_q8": [events[i+k].get("mono_q8") for k in range(3)],
                "snr_q8":  [events[i+k].get("snr_q8")  for k in range(3)],
            }
            full_cycles.append(cycle)

    # 2-punts partial cycles
    for i in range(len(events) - 1):
        p0, p1 = pools[i], pools[i+1]
        t0, t1 = times[i], times[i+1]
        unique = {p0, p1}

        if unique == {POOL_NEU, POOL_N}:
            ptype = "partial_N_neutral"
        elif unique == {POOL_NEU, POOL_S}:
            ptype = "partial_S_neutral"
        else:
            continue

        partial_cycles.append({
            "sensor": sensor_name,
            "type": ptype,
            "idx_start": i,
            "idx_end": i+1,
            "t_start_us": t0,
            "t_end_us": t1,
            "dt_us": t1 - t0,
            "p0": p0,
            "p1": p1,
        })

    return full_cycles, partial_cycles


def build_cycles(events_by_sensor: Dict[int, List[Event]]) -> Dict[str, Dict[str, Any]]:
    """
    Convenience: draait detectie voor alle sensoren en levert een dict:

    {
      "A": {
        "full_cycles": [...],
        "partial_cycles": [...],
        "timing_stats": {...}
      },
      "B": {
        ...
      }
    }
    """
    result: Dict[str, Dict[str, Any]] = {}
    for sensor_id, evs in events_by_sensor.items():
        if sensor_id == 0:
            name = "A"
        elif sensor_id == 1:
            name = "B"
        else:
            name = str(sensor_id)

        full_c, partial_c = detect_cycles_for_sensor(evs, sensor_name=name)
        stats = timing_stats_from_cycles(full_c)
        result[name] = {
            "full_cycles": full_c,
            "partial_cycles": partial_c,
            "timing_stats": stats,
        }

    return result


# ------------- CLI -----------------------------------------------------------

def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Symphonia Cycle Builder v1.0 (L0B)")
    ap.add_argument("events_jsonl", help="Pad naar core0 event24 JSONL")
    ap.add_argument("--out", default=None, help="Output JSON bestand (default: zelfde naam met _cycles.json)")
    args = ap.parse_args()

    events_by_sensor = load_events_from_event24(args.events_jsonl)
    all_cycles = build_cycles(events_by_sensor)

    if args.out is None:
        base, _ = os.path.splitext(args.events_jsonl)
        args.out = base + "_cycles.json"

    out_obj = {
        "meta": {
            "version": "cycle_builder_v1.0",
            "source_events": args.events_jsonl,
        },
        "sensors": all_cycles,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"[i] Cycle Builder v1.0: geschreven naar {args.out}")


if __name__ == "__main__":
    main()

