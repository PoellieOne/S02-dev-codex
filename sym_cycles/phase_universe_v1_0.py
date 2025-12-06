#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sym_cycles.phase_universe_v1_0

Phase Universe v1.0 — diagnostische laag bovenop Cycle Builder v1.0.

Doel:
  - Per cycle een grove rotor-gecentreerde fase-tag geven:
      phase_class: "TOP" / "BOTTOM" / "UNKNOWN"
      run_frac   : 0..1 (relatieve tijd in de run)
      phase_bin  : 0..M-1 (tile index)
  - Resultaat wegschrijven naar *_cycles_phase.json
"""

import json
from typing import List, Dict, Any


def load_cycles(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Als er al een top-level "cycles" lijst is, dan gebruiken we die direct
    if isinstance(data.get("cycles"), list):
        return data

    # Anders kijken we naar het "sensors" formaat:
    sensors = data.get("sensors")
    if isinstance(sensors, dict):
        all_cycles = []
        for sensor_name in ("A", "B"):
            s = sensors.get(sensor_name)
            if not isinstance(s, dict):
                continue
            full = s.get("full_cycles", [])
            for c in full:
                # Zorg dat het sensorveld er zeker op staat
                c.setdefault("sensor", sensor_name)
                all_cycles.append(c)

        # Schrijf de geflatteerde lijst weg in data["cycles"]
        data["cycles"] = all_cycles
        return data

    # Fallback: niets bruikbaars gevonden
    data["cycles"] = []
    return data


def save_cycles_with_phase(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def infer_phase_class(cycle_type: str) -> str:
    """
    Map cycle_type -> phase_class in ons TOP/BOTTOM universum.
    """
    if cycle_type == "cycle_down":
        return "TOP"
    if cycle_type == "cycle_up":
        return "BOTTOM"
    return "UNKNOWN"


def tag_cycles_with_phase(cycles: List[Dict[str, Any]], M: int = 24) -> None:
    """
    Verrijkt elke cycle in-place met:
      - phase_class
      - run_frac
      - phase_bin
    """

    # 1) bepaal globale tijdrange
    t_values = [c.get("t_center_us") for c in cycles if isinstance(c.get("t_center_us"), (int, float))]
    if not t_values:
        return

    t_min = min(t_values)
    t_max = max(t_values)
    span = max(1.0, float(t_max - t_min))

    for c in cycles:
        t = c.get("t_center_us")
        if not isinstance(t, (int, float)):
            c["phase_class"] = "UNKNOWN"
            c["run_frac"] = None
            c["phase_bin"] = None
            continue

        # 2) phase_class uit cycle_type
        phase_class = infer_phase_class(c.get("cycle_type", ""))

        # 3) run_frac 0..1
        run_frac = (float(t) - t_min) / span
        if run_frac < 0.0:
            run_frac = 0.0
        if run_frac > 1.0:
            run_frac = 1.0

        # 4) phase_bin (0..M-1)
        bin_idx = int(run_frac * M)
        if bin_idx >= M:
            bin_idx = M - 1

        c["phase_class"] = phase_class
        c["run_frac"] = run_frac
        c["phase_bin"] = bin_idx


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Symphonia Phase Universe v1.0 (cycle-phase tagger)")
    ap.add_argument("cycles_json", help="input cycles JSON (output van Cycle Builder v1.0)")
    ap.add_argument("--tiles", type=int, default=24,
                    help="aantal phase-bins (tiles) over de run (default 24)")
    ap.add_argument("--output", help="output JSON (default: *_cycles_phase.json)")

    args = ap.parse_args()

    data = load_cycles(args.cycles_json)
    cycles = data.get("cycles", [])

    print(f"[i] Phase Universe v1.0 — input cycles: {args.cycles_json}")
    print(f"[i]  cycles: {len(cycles)}, tiles: {args.tiles}")

    tag_cycles_with_phase(cycles, M=args.tiles)

    out_path = args.output
    if not out_path:
        if args.cycles_json.endswith(".json"):
            out_path = args.cycles_json.replace(".json", "_phase.json")
        else:
            out_path = args.cycles_json + "_phase.json"

    save_cycles_with_phase(data, out_path)
    print(f"[i] Phase Universe: geschreven naar {out_path}")


if __name__ == "__main__":
    main()
