#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sym_cycles.projector

PureStep → Cycle-Backbone Projector v1.0

- Laadt backbone JSON (nodes)
- Laadt PureSteps JSONL
- Voor elke PureStep: zoek dichtstbijzijnde backbone-node
- Berekent een aantal deel-scores:
    - timing_score
    - direction_score (op basis van step_dir / phase_tag)
    - cycle_consistency_score
    - stereo_consistency_score
- Schrijft geprojecteerde steps naar JSONL
"""

import json
import bisect
from typing import List, Dict, Any, Optional


BackboneNode = Dict[str, Any]
PureStep = Dict[str, Any]
ProjectedStep = Dict[str, Any]


# ------------- backbone indexing --------------------------------------------

def index_backbone(nodes: List[BackboneNode]):
    """
    Maakt een aparte lijst met t_center_us zodat we snel
    dichtstbijzijnde node kunnen zoeken via bisect.
    """
    times = [float(n["t_center_us"]) for n in nodes]
    return times, nodes


def find_closest_node(nodes_times: List[float],
                      nodes: List[BackboneNode],
                      t_us: float) -> Optional[BackboneNode]:
    """
    Binaire search naar dichtstbijzijnde t_center_us voor gegeven t_us.
    """
    if not nodes:
        return None
    idx = bisect.bisect_left(nodes_times, t_us)
    candidates = []
    if 0 <= idx < len(nodes):
        candidates.append(nodes[idx])
    if 0 <= idx - 1 < len(nodes):
        candidates.append(nodes[idx - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda n: abs(float(n["t_center_us"]) - t_us))
    return best


# ------------- scores --------------------------------------------------------

def compute_timing_score(step_t_us: float, node: BackboneNode) -> float:
    t_center = float(node["t_center_us"])
    dt_to_center = abs(step_t_us - t_center)
    tmin = float(node["timing_range_us"]["min"])
    tmax = float(node["timing_range_us"]["max"])
    span = max(1.0, (tmax - tmin))
    # hoe dichter bij center, hoe beter – normaliseer tegen halve span
    half_span = 0.5 * span
    return max(0.0, 1.0 - dt_to_center / half_span)


def compute_direction_score(step: PureStep, node: BackboneNode) -> float:
    """
    Heel eenvoudige eerste versie:
      - step_dir + phase_tag gebruiken als primaire indicator.
      - cycle_type van node nog niet meegecalibreerd naar CW/CCW (dat is hardware-afhankelijk),
        dus hier nog neutraal. Je kunt dit later verfijnen.
    """
    step_dir = step.get("step_dir", "NONE")
    phase_tag = step.get("phase_tag", "")
    if step_dir == "CW" and phase_tag == "A_then_B":
        return 1.0
    if step_dir == "CCW" and phase_tag == "B_then_A":
        return 1.0
    if step_dir in ("CW", "CCW"):
        return 0.5
    return 0.2  # NONE / onbekend


def compute_cycle_consistency_score(step: PureStep, node: BackboneNode) -> float:
    """
    Gebruikt cycle_pair_consistency uit PureStep als bron.
    """
    cpc = step.get("cycle_pair_consistency", "none")
    if cpc == "both_full_same_type":
        return 1.0
    if cpc == "one_full":
        return 0.7
    if cpc == "both_full_diff_type":
        return 0.3
    return 0.2  # none / onbekend


def compute_stereo_consistency_score(step: PureStep, node: BackboneNode) -> float:
    """
    Vergelijkt delta_t_AB_us (step) met stereo_range_us (node), als beschikbaar.
    """
    delta = step.get("delta_t_AB_us")
    srange = node.get("stereo_range_us") or {}
    smin = srange.get("min")
    smax = srange.get("max")

    if delta is None or smin is None or smax is None:
        return 0.5  # neutrale score als we niks weten

    delta = float(delta)
    if smin <= delta <= smax:
        # hoe dichter bij midden, hoe beter
        mid = 0.5 * (smin + smax)
        span = max(1.0, (smax - smin))
        return max(0.0, 1.0 - abs(delta - mid) / (0.5 * span))
    else:
        # buiten range
        return 0.1


def compute_total_score(timing_score: float,
                        direction_score: float,
                        cycle_score: float,
                        stereo_score: float) -> float:
    return (
        0.35 * timing_score +
        0.30 * direction_score +
        0.20 * cycle_score +
        0.15 * stereo_score
    )


# ------------- projection ----------------------------------------------------

def project_pure_steps(pure_steps: List[PureStep],
                       backbone_nodes: List[BackboneNode]) -> List[ProjectedStep]:
    times, nodes = index_backbone(backbone_nodes)

    projected: List[ProjectedStep] = []

    for step in pure_steps:
        t_us = step.get("t_us")
        if t_us is None:
            continue
        node = find_closest_node(times, nodes, float(t_us))
        if node is None:
            # zonder backbone-node heeft projectie weinig zin
            continue

        timing_score = compute_timing_score(float(t_us), node)
        direction_score = compute_direction_score(step, node)
        cycle_score = compute_cycle_consistency_score(step, node)
        stereo_score = compute_stereo_consistency_score(step, node)
        total_score = compute_total_score(timing_score, direction_score,
                                          cycle_score, stereo_score)

        proj: ProjectedStep = {
            **step,
            "backbone_node_id": node["node_id"],
            "backbone_cycle_type": node.get("cycle_type"),
            "projection": {
                "timing_score": timing_score,
                "direction_score": direction_score,
                "cycle_consistency_score": cycle_score,
                "stereo_consistency_score": stereo_score,
                "total_score": total_score,
            }
        }
        projected.append(proj)

    return projected


def load_pure_steps(path: str) -> List[PureStep]:
    steps: List[PureStep] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if "t_us" not in obj:
                continue
            steps.append(obj)
    return steps


# ------------- CLI -----------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(description="PureStep → Backbone Projector v1.0")
    ap.add_argument("backbone_json", help="Backbone JSON (output van sym_cycles.backbone)")
    ap.add_argument("pure_steps_jsonl", help="PureSteps JSONL (output van capture_core0_pure_v1_3)")
    ap.add_argument("--out", default="pure_projected_steps_v1.0.jsonl",
                    help="Output JSONL met geprojecteerde steps")
    args = ap.parse_args()

    with open(args.backbone_json, "r", encoding="utf-8") as f:
        backbone_data = json.load(f)
    backbone_nodes: List[BackboneNode] = backbone_data.get("nodes", [])

    pure_steps = load_pure_steps(args.pure_steps_jsonl)
    projected = project_pure_steps(pure_steps, backbone_nodes)

    with open(args.out, "w", encoding="utf-8") as f:
        for p in projected:
            f.write(json.dumps(p) + "\n")

    print(f"[i] Projector v1.0: {len(projected)} steps geprojecteerd → {args.out}")


if __name__ == "__main__":
    main()

