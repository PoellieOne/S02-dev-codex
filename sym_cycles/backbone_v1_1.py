#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sym_cycles.backbone

Cycle Backbone v1.1:
- Neemt cycles van A en B (output van sym_cycles.builder)
- Berekent stereo Δt-statistieken per cycle_type
- Koppel A-cycle en B-cycle alleen als ze naar hetzelfde magnetische segment kijken
- Bouwt een globale backbone (nodes gesorteerd op t_center_us)
"""

import json
import math
from typing import List, Dict, Any, Optional, Tuple


Cycle = Dict[str, Any]
BackboneNode = Dict[str, Any]


# ------------- helpers -------------------------------------------------------

def _quantile(sorted_vals: List[float], q: float) -> float:
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


def _basic_stats(values: List[float]) -> Dict[str, float]:
    vals = sorted(values)
    n = len(vals)
    return {
        "min": vals[0],
        "p25": _quantile(vals, 0.25),
        "median": _quantile(vals, 0.50),
        "mean": sum(vals) / n,
        "p75": _quantile(vals, 0.75),
        "max": vals[-1],
        "count": float(n),
    }


# ------------- stereo analyse ------------------------------------------------

def compute_stereo_deltas(cycles_A: List[Cycle],
                          cycles_B: List[Cycle],
                          max_lag_us: int = 300_000) -> Dict[str, Dict[str, Any]]:
    """
    Berekent Δt-statistieken per cycle_type op basis van centers van A/B.

    Return:
      {
        "cycle_up":   {"deltas": [...], "stats": {...}},
        "cycle_down": {...}
      }
    """
    result: Dict[str, Dict[str, Any]] = {}

    for ctype in ("cycle_up", "cycle_down"):
        centers_A = [c for c in cycles_A if c.get("cycle_type") == ctype]
        centers_B = [c for c in cycles_B if c.get("cycle_type") == ctype]

        centers_A_sorted = sorted(centers_A, key=lambda c: c["t_center_us"])
        centers_B_sorted = sorted(centers_B, key=lambda c: c["t_center_us"])

        deltas: List[float] = []

        j = 0
        for ca in centers_A_sorted:
            tA = ca["t_center_us"]

            # schuif j totdat B[j] niet te ver achterloopt
            while j < len(centers_B_sorted) and centers_B_sorted[j]["t_center_us"] < tA - max_lag_us:
                j += 1

            best_dt: Optional[float] = None
            for k in range(j, min(j + 5, len(centers_B_sorted))):
                cb = centers_B_sorted[k]
                tB = cb["t_center_us"]
                dt = tB - tA
                if abs(dt) > max_lag_us:
                    continue
                if best_dt is None or abs(dt) < abs(best_dt):
                    best_dt = dt

            if best_dt is not None:
                deltas.append(best_dt)

        if deltas:
            result[ctype] = {
                "deltas": deltas,
                "stats": _basic_stats(deltas),
            }
        else:
            result[ctype] = {
                "deltas": [],
                "stats": {},
            }

    return result


def match_same_picture_cycles(A, B, max_dt_us=20_000):
    """
    Vind alleen echte A<->B matches voor same-picture cycles.

    Criteria:
      - zelfde cycle_type
      - cycle_duration vergelijkbaar
      - beide voldoende quality (indien cycle_quality bestaat)
      - dt binnen fysisch venster

    Return: lijst van dicts met:
        {
            "A": <cycle>,
            "B": <cycle>,
            "stereo_dt_us": <float>
        }
    """

    matches = []
    used_B = set()

    for a in A:
        # Veilige quality lookup
        qa = float(a.get("cycle_quality", 1.0))
        # if qa < 0.5: continue

        best = None
        best_dt = None
        best_idx = None

        for i, b in enumerate(B):
            if i in used_B:
                continue

            qb = float(b.get("cycle_quality", 1.0))
            # if qb < 0.5: continue

            # zelfde cycle_type?
            if a.get("cycle_type") != b.get("cycle_type"):
                continue

            # vergelijkbare duur?
            da = a.get("dt_us")
            db = b.get("dt_us")
            if da is None or db is None:
                continue
            if not (0.5 * da <= db <= 1.5 * da):
                continue

            # check stereo-delta
            dt_ab = float(b["t_center_us"] - a["t_center_us"])
            if abs(dt_ab) > max_dt_us:
                continue

            # kies de beste (kleinste |dt|)
            if best is None or abs(dt_ab) < abs(best_dt):
                best = b
                best_dt = dt_ab
                best_idx = i

        # einde inner loop

        if best is not None:
            used_B.add(best_idx)
            matches.append(
                {
                    "A": a,
                    "B": best,
                    "stereo_dt_us": best_dt,
                }
            )

    return matches


# ------------- backbone node synthese ---------------------------------------

def _make_backbone_node(node_id: int,
                        cyc_A: Optional[Cycle],
                        cyc_B: Optional[Cycle],
                        timing_stats: Dict[str, Dict[str, float]],
                        stereo_stats_for_type: Optional[Dict[str, float]]) -> BackboneNode:
    """
    Bouwt één BackboneNode uit 0/1/2 cycles.
    """
    if cyc_A is not None and cyc_B is not None:
        t_center = 0.5 * (cyc_A["t_center_us"] + cyc_B["t_center_us"])
        dt_us = int(0.5 * (cyc_A["dt_us"] + cyc_B["dt_us"]))
        stereo_dt = cyc_B["t_center_us"] - cyc_A["t_center_us"]
        typ = cyc_A.get("cycle_type") or cyc_B.get("cycle_type")
    else:
        cyc = cyc_A or cyc_B
        if cyc is None:
            raise ValueError("At least one cycle must be non-None")
        t_center = cyc["t_center_us"]
        dt_us = cyc["dt_us"]
        stereo_dt = None
        typ = cyc.get("cycle_type")

    # Timing-range uit globale timing-stats (fall back op dt_us +/- 0)
    ts = timing_stats.get(typ, {})
    tmin = ts.get("min", dt_us)
    tmax = ts.get("max", dt_us)

    if stereo_stats_for_type:
        smin = stereo_stats_for_type.get("min")
        smax = stereo_stats_for_type.get("max")
    else:
        smin = None
        smax = None

    node: BackboneNode = {
        "node_id": node_id,
        "cycle_type": typ,
        "t_center_us": t_center,
        "dt_us": dt_us,
        "sensor_A": cyc_A,
        "sensor_B": cyc_B,
        "stereo_dt_us": stereo_dt,
        "timing_range_us": {
            "min": tmin,
            "max": tmax,
        },
        "stereo_range_us": {
            "min": smin,
            "max": smax,
        },
        "sequence_expected_next": None,  # later ingevuld
    }
    return node


def build_backbone(cycles_A: List[Cycle],
                   cycles_B: List[Cycle],
                   timing_stats_global: Dict[str, Dict[str, float]],
                   stereo_stats: Dict[str, Dict[str, Any]],
                   max_lag_us: int = 300_000) -> List[BackboneNode]:
    """
    Bouwt een lijst BackboneNodes uit full cycles van A en B.
    """
    remaining_A = list(cycles_A)
    remaining_B = list(cycles_B)

    # --------------------------------------------------------------
    # 1. NEW: Same-picture stereo matches via v1.1 strict matching
    # --------------------------------------------------------------

    same_picture_matches = match_same_picture_cycles(
        remaining_A, remaining_B,
        max_dt_us=min(max_lag_us, 40_000)   # fysisch venster
    )

    nodes: List[BackboneNode] = []
    node_id = 0
    used_A = set()
    used_B = set()

    for m in same_picture_matches:
        a = m["A"]
        b = m["B"]
        dt_ab = m["stereo_dt_us"]

        # markeer als gebruikt
        used_A.add(id(a))
        used_B.add(id(b))

        ctype = a.get("cycle_type") or b.get("cycle_type")
        stereo_stats_for_type = stereo_stats.get(ctype, {}).get("stats", {})

        node = _make_backbone_node(
            node_id,
            a, b,
            timing_stats_global,
            stereo_stats_for_type
        )

        # vlag toevoegen voor CompassSign v1.1
        node["stereo_match"] = True
        node["stereo_dt_us"] = dt_ab

        nodes.append(node)
        node_id += 1

    # verwijder matched cycles
    remaining_A = [c for c in remaining_A if id(c) not in used_A]
    remaining_B = [c for c in remaining_B if id(c) not in used_B]


    # Single nodes voor alles wat overblijft
    for a in remaining_A:
        ctype = a.get("cycle_type")
        stereo_stats_for_type = stereo_stats.get(ctype, {}).get("stats", {})
        node = _make_backbone_node(node_id, a, None,
                                   timing_stats_global,
                                   stereo_stats_for_type)
        nodes.append(node)
        node_id += 1

    for b in remaining_B:
        ctype = b.get("cycle_type")
        stereo_stats_for_type = stereo_stats.get(ctype, {}).get("stats", {})
        node = _make_backbone_node(node_id, None, b,
                                   timing_stats_global,
                                   stereo_stats_for_type)
        nodes.append(node)
        node_id += 1

    # sorteer nodes op tijd
    nodes.sort(key=lambda n: n["t_center_us"])

    # expected_next invullen: up→down→up, down→up→down
    for i, node in enumerate(nodes):
        ctype = node.get("cycle_type")
        if ctype == "cycle_up":
            node["sequence_expected_next"] = "cycle_down"
        elif ctype == "cycle_down":
            node["sequence_expected_next"] = "cycle_up"
        else:
            node["sequence_expected_next"] = None

    return nodes

# ------------- CLI -----------------------------------------------------------

def main():
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Symphonia Cycle Backbone v1.0 (L0B)")
    ap.add_argument("cycles_json", help="Output JSON van sym_cycles.builder ( *_cycles.json )")
    ap.add_argument("--out", default=None, help="Output backbone JSON (default: *_backbone.json)")
    ap.add_argument("--max_lag_us", type=int, default=300_000,
                    help="Maximale stereo Δt bij matching A/B (default: 300000 us)")
    args = ap.parse_args()

    with open(args.cycles_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    sensors = data.get("sensors", {})
    A = sensors.get("A", {})
    B = sensors.get("B", {})

    full_A = A.get("full_cycles", [])
    full_B = B.get("full_cycles", [])

    # globale timing-stats: combineer A en B
    timing_stats_global = {}
    for name, info in sensors.items():
        ts = info.get("timing_stats", {})
        for ctype, s in ts.items():
            lst = timing_stats_global.setdefault(ctype, {"values": []})
            lst["values"].append(s["min"])
            lst["values"].append(s["max"])

    # maak een simpele min/max per type uit de verzamelde waarden
    for ctype, st in timing_stats_global.items():
        vals = sorted(st["values"])
        timing_stats_global[ctype] = {
            "min": vals[0],
            "max": vals[-1],
        }

    stereo_stats = compute_stereo_deltas(full_A, full_B, max_lag_us=args.max_lag_us)

    backbone_nodes = build_backbone(full_A, full_B, timing_stats_global, stereo_stats,
                                    max_lag_us=args.max_lag_us)

    if args.out is None:
        base, _ = os.path.splitext(args.cycles_json)
        args.out = base + "_backbone.json"

    out_obj = {
        "meta": {
            "version": "cycle_backbone_v1.1",
            "source_cycles": args.cycles_json,
            "max_lag_us": args.max_lag_us,
        },
        "timing_stats_global": timing_stats_global,
        "stereo_stats": stereo_stats,
        "nodes": backbone_nodes,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"[i] Cycle Backbone v1.1: geschreven naar {args.out}")


if __name__ == "__main__":
    main()

