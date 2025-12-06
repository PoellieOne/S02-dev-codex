#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sym_cycles.demo_check

End-to-end demo/sanity-check voor de L0B-pipeline:

  Event24 JSONL  →  Cycle Builder (A/B)
                  →  Cycle Backbone v1.0
                  →  PureStep → Backbone projectie

Gebruik:
  python3 -m sym_cycles.demo_check \
      --events core0_events_CW_1.jsonl \
      --pure   pure_steps_offline_CW_1.jsonl

Output:
  - _cycles.json         (tussenbestand met per-sensor cycles)
  - _backbone.json       (cycle-backbone)
  - _projected_steps.jsonl (PureSteps + projection scores)
  - stdout samenvatting
"""

import argparse
import json
import os
from collections import Counter
from . import builder_v1_0 as builder
from . import backbone_v1_0 as backbone
from . import projector_v1_0 as projector


def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def run_cycle_builder(events_path: str, cycles_out: str):
    print(f"[i] Cycle Builder v1.0 — input: {events_path}")

    events_by_sensor = builder.load_events_from_event24(events_path)
    all_cycles = builder.build_cycles(events_by_sensor)

    out_obj = {
        "meta": {
            "version": "cycle_builder_v1.0",
            "source_events": events_path,
        },
        "sensors": all_cycles,
    }
    ensure_dir(cycles_out)
    with open(cycles_out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"[i] Cycle Builder: geschreven naar {cycles_out}")
    return all_cycles


def summarize_cycles(all_cycles: dict):
    print("\n=== Cycle Builder Samenvatting ===")
    for name, info in all_cycles.items():
        full_c = info.get("full_cycles", [])
        partial_c = info.get("partial_cycles", [])
        stats = info.get("timing_stats", {})

        type_counts = Counter(c.get("cycle_type", "unknown") for c in full_c)

        print(f"\nSensor {name}:")
        print(f"  Volledige cycles : {len(full_c)}")
        for t, n in type_counts.items():
            print(f"    {t:12s}: {n:5d}")
        print(f"  Partial cycles   : {len(partial_c)}")
        if stats:
            print("  Timing-stats per type (dt_us):")
            for ctype, s in stats.items():
                print(
                    f"    {ctype:12s}: "
                    f"min={s['min']:.0f}, p50={s['median']:.0f}, max={s['max']:.0f}, "
                    f"n={s['count']:.0f}"
                )
        else:
            print("  (Geen timing-stats beschikbaar)")


def run_backbone(cycles_json: str, backbone_out: str, max_lag_us: int = 300_000):
    print(f"\n[i] Backbone v1.0 — input cycles: {cycles_json}")

    with open(cycles_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    sensors = data.get("sensors", {})
    A = sensors.get("A", {})
    B = sensors.get("B", {})

    full_A = A.get("full_cycles", [])
    full_B = B.get("full_cycles", [])

    # globale timing-stats: combineer A en B min/max
    timing_stats_global = {}
    for name, info in sensors.items():
        ts = info.get("timing_stats", {})
        for ctype, s in ts.items():
            lst = timing_stats_global.setdefault(ctype, {"values": []})
            lst["values"].append(s["min"])
            lst["values"].append(s["max"])

    for ctype, st in timing_stats_global.items():
        vals = sorted(st["values"])
        timing_stats_global[ctype] = {
            "min": vals[0],
            "max": vals[-1],
        }

    stereo_stats = backbone.compute_stereo_deltas(full_A, full_B, max_lag_us=max_lag_us)
    backbone_nodes = backbone.build_backbone(
        full_A, full_B, timing_stats_global, stereo_stats, max_lag_us=max_lag_us
    )

    out_obj = {
        "meta": {
            "version": "cycle_backbone_v1.0",
            "source_cycles": cycles_json,
            "max_lag_us": max_lag_us,
        },
        "timing_stats_global": timing_stats_global,
        "stereo_stats": stereo_stats,
        "nodes": backbone_nodes,
    }

    ensure_dir(backbone_out)
    with open(backbone_out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)

    print(f"[i] Backbone: geschreven naar {backbone_out}")
    return backbone_nodes, stereo_stats


def summarize_backbone(nodes, stereo_stats):
    print("\n=== Backbone Samenvatting ===")
    print(f"  Totaal nodes : {len(nodes)}")

    type_counts = Counter(n.get("cycle_type", "unknown") for n in nodes)
    for t, n in type_counts.items():
        print(f"    {t:12s}: {n:5d}")

    print("\n  Stereo Δt-statistiek per type:")
    for ctype, info in stereo_stats.items():
        stats = info.get("stats", {})
        if not stats:
            print(f"    {ctype:12s}: (geen matches)")
            continue
        print(
            f"    {ctype:12s}: "
            f"min={stats['min']:.0f}, p50={stats['median']:.0f}, "
            f"max={stats['max']:.0f}, n={stats['count']:.0f}"
        )


def run_projector(backbone_json: str, pure_steps_jsonl: str, out_path: str):
    print(f"\n[i] Projector v1.0 — backbone: {backbone_json}")
    print(f"                       pure : {pure_steps_jsonl}")

    with open(backbone_json, "r", encoding="utf-8") as f:
        backbone_data = json.load(f)
    nodes = backbone_data.get("nodes", [])

    pure_steps = projector.load_pure_steps(pure_steps_jsonl)
    projected = projector.project_pure_steps(pure_steps, nodes)

    ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in projected:
            f.write(json.dumps(p) + "\n")

    print(f"[i] Projector: {len(projected)} steps geprojecteerd → {out_path}")
    return projected


def summarize_projection(projected):
    print("\n=== Projectie Samenvatting ===")
    print(f"  Geprojecteerde steps : {len(projected)}")

    # scores
    totals = []
    by_state = Counter()
    by_dir = Counter()

    for p in projected:
        proj = p.get("projection") or {}
        total = proj.get("total_score")
        if isinstance(total, (int, float)):
            totals.append(float(total))

        step_dir = p.get("step_dir", "NONE")
        by_dir[step_dir] += 1

    print("  step_dir verdeling:")
    for d, n in by_dir.items():
        print(f"    {d:6s}: {n:5d}")

    if not totals:
        print("  (Geen total_score waarden gevonden)")
        return

    totals_sorted = sorted(totals)
    n = len(totals_sorted)

    def q(frac: float) -> float:
        if n == 1:
            return totals_sorted[0]
        idx = (n - 1) * frac
        lo = int(idx)
        hi = min(n - 1, lo + 1)
        f = idx - lo
        return totals_sorted[lo] * (1 - f) + totals_sorted[hi] * f

    print("  total_score verdeling (0..1):")
    print(f"    min   : {totals_sorted[0]:.3f}")
    print(f"    p25   : {q(0.25):.3f}")
    print(f"    median: {q(0.50):.3f}")
    print(f"    p75   : {q(0.75):.3f}")
    print(f"    max   : {totals_sorted[-1]:.3f}")

    # ruwe indruk van "trusted" subset
    high = sum(1 for v in totals if v >= 0.8)
    mid = sum(1 for v in totals if 0.5 <= v < 0.8)
    low = sum(1 for v in totals if v < 0.5)
    print("\n  Heuristische kwaliteitsverdeling (op basis van total_score):")
    print(f"    high (>=0.80): {high:5d}")
    print(f"    mid  (0.5–0.8): {mid:5d}")
    print(f"    low  (<0.5)  : {low:5d}")


def main():
    ap = argparse.ArgumentParser(description="Symphonia L0B demo-check (builder + backbone + projector)")
    ap.add_argument("--events", required=True, help="Event24 JSONL (core0_events_*.jsonl)")
    ap.add_argument("--pure", required=True, help="PureSteps JSONL (pure_steps*.jsonl)")
    ap.add_argument("--prefix", default=None,
                    help="Prefix voor output-bestanden (default: afgeleid van --events)")
    ap.add_argument("--max_lag_us", type=int, default=300_000,
                    help="Max stereo Δt voor A/B matching (default: 300000)")
    args = ap.parse_args()

    events_path = args.events
    pure_path = args.pure

    if args.prefix is None:
        base, _ = os.path.splitext(events_path)
        prefix = base
    else:
        prefix = args.prefix

    cycles_path = prefix + "_cycles.json"
    backbone_path = prefix + "_backbone.json"
    projected_path = prefix + "_projected_steps.jsonl"

    # 1) Cycle Builder
    all_cycles = run_cycle_builder(events_path, cycles_path)
    summarize_cycles(all_cycles)

    # 2) Backbone
    backbone_nodes, stereo_stats = run_backbone(cycles_path, backbone_path, max_lag_us=args.max_lag_us)
    summarize_backbone(backbone_nodes, stereo_stats)

    # 3) Projector
    projected = run_projector(backbone_path, pure_path, projected_path)
    summarize_projection(projected)

    print("\n[i] Demo-check voltooid.")
    print(f"    cycles   → {cycles_path}")
    print(f"    backbone → {backbone_path}")
    print(f"    projected→ {projected_path}")


if __name__ == "__main__":
    main()

