#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compass_sign_v3

CompassSign v3.0 – kernmodule voor globale draairichting uit PhaseTiles v3.

Rol:
- Neemt PhaseTiles v3 tiles (met dt_ab_us per tile).
- Past een eenvoudige deadzone/min_tiles/max_abs_dt_us logica toe.
- Maakt een raw tiles-verdict: {"A_LEADS_B","B_LEADS_A","ONBEPAALD"}.
- Mapt dit via een configurabele mapping naar globale richting: {"CW","CCW","UNDECIDED"}.
- Berekent een confidence 0..1 op basis van mediane |dt_ab_us| en aantal tiles.
- Geeft een compact resultaatobject terug, geschikt voor Core-1 of bench-rapporten.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CompassSignConfig:
    """
    Configuratie voor CompassSign v3.0.

    - deadzone_us:  |median dt_ab_us| < deadzone_us -> ONBEPAALD
    - min_tiles   :  minimaal aantal tiles met geldige dt_ab_us
    - max_abs_dt_us: filter tiles met |dt_ab_us| > max_abs_dt_us weg (None = uit)
    - phase_class : puur label voor meta ("any", "TOP", "BOTTOM")
    - mapping     : mapping van tiles-verdict naar globale richting
    """
    deadzone_us: float = 100.0
    min_tiles: int = 3
    max_abs_dt_us: Optional[float] = None
    phase_class: str = "any"  # "any" | "TOP" | "BOTTOM"
    mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "A_LEADS_B": "CW",
            "B_LEADS_A": "CCW",
            "ONBEPAALD": "UNDECIDED",
        }
    )


@dataclass
class CompassSignResult:
    """
    Resultaat van een CompassSign v3.0 evaluatie.
    """
    global_direction: str            # "CW" | "CCW" | "UNDECIDED"
    confidence: float                # 0.0 .. 1.0
    verdict_tiles: str               # "A_LEADS_B" | "B_LEADS_A" | "ONBEPAALD"
    meta: Dict[str, Any]             # thresholds, dt_stats, counts, reasons, etc.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize_sorted(values: List[float]) -> Dict[str, Optional[float]]:
    """
    Neem een gesorteerde lijst en geef min/p25/median/p75/max terug.
    Bij n == 0 wordt alles None en n=0.
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
        "median": pick(0.50),
        "p75": pick(0.75),
        "max": values[-1],
    }


def _dt_stats_from_tiles(
    tiles: Sequence[Dict[str, Any]],
    max_abs_dt_us: Optional[float],
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Extraheer dt_ab_us uit tiles, pas optioneel een |dt|-filter toe, en geef
    zowel de lijst dt_values als een statistiek-dict terug.
    """
    dt_values: List[float] = []

    for t in tiles:
        dt = t.get("dt_ab_us")
        if not isinstance(dt, (int, float)):
            continue
        dt_f = float(dt)
        if (max_abs_dt_us is not None) and (abs(dt_f) > max_abs_dt_us):
            continue
        dt_values.append(dt_f)

    dt_values.sort()
    stats = _summarize_sorted(dt_values)
    return dt_values, stats


def _compute_tiles_verdict(
    dt_stats: Dict[str, Any],
    cfg: CompassSignConfig,
) -> Tuple[str, str]:
    """
    Bepaal tiles-verdict ("A_LEADS_B"/"B_LEADS_A"/"ONBEPAALD") en redenstring
    op basis van dt-statistiek en config.
    """
    n = dt_stats.get("n") or 0
    median = dt_stats.get("median")

    deadzone = cfg.deadzone_us
    min_tiles = cfg.min_tiles

    verdict = "ONBEPAALD"
    reason = ""

    if n < min_tiles:
        reason = f"te weinig tiles met geldige dt_ab_us (n={n}, min={min_tiles})"
        return verdict, reason

    if median is None:
        reason = "kan median dt_ab_us niet bepalen"
        return verdict, reason

    m = float(median)
    if abs(m) < deadzone:
        reason = f"median dt_ab_us te dicht bij 0 (|{m:.1f}| < {deadzone:.1f} µs)"
        return verdict, reason

    if m > 0:
        verdict = "A_LEADS_B"
        reason = f"median dt_ab_us > 0 (A ziet gemiddeld eerder, median={m:.1f} µs)"
    else:
        verdict = "B_LEADS_A"
        reason = f"median dt_ab_us < 0 (B ziet gemiddeld eerder, median={m:.1f} µs)"

    return verdict, reason


def _compute_confidence(
    verdict_tiles: str,
    dt_stats: Dict[str, Any],
    cfg: CompassSignConfig,
) -> float:
    """
    Heuristische confidence 0..1 uit tiles-verdict + dt-statistiek.
    """
    if verdict_tiles == "ONBEPAALD":
        return 0.0

    n = dt_stats.get("n") or 0
    median = dt_stats.get("median")
    if median is None or n < cfg.min_tiles:
        return 0.0

    m = abs(float(median))
    deadzone = float(cfg.deadzone_us)
    min_tiles = float(cfg.min_tiles)

    margin = max(0.0, m - deadzone)

    # Hoeveel buiten de deadzone zitten we?
    # ~1.0 bij >= 4 * deadzone
    if deadzone <= 0.0:
        s1 = 1.0
    else:
        s1 = min(1.0, margin / (deadzone * 3.0))

    # Hoeveel extra tiles hebben we t.o.v. minimum?
    if min_tiles <= 0.0:
        s2 = 1.0
    else:
        s2 = min(1.0, max(0.0, (n - min_tiles) / min_tiles))

    conf = 0.6 * s1 + 0.4 * s2
    # clamp defensief
    if conf < 0.0:
        conf = 0.0
    if conf > 1.0:
        conf = 1.0
    return float(conf)


# ---------------------------------------------------------------------------
# Kern-API
# ---------------------------------------------------------------------------

def compute_compass_sign_from_tiles(
    tiles: Sequence[Dict[str, Any]],
    config: Optional[CompassSignConfig] = None,
    tile_duration_us: Optional[float] = None,
) -> CompassSignResult:
    """
    Kernfunctie: bepaal globale kompasrichting uit een lijst PhaseTiles v3 tiles.

    Parameters
    ----------
    tiles : sequence van dicts
        Verwacht keys zoals:
          - "dt_ab_us" : float of int
          - overige velden worden genegeerd in deze kernel.
    config : CompassSignConfig, optioneel
        Thresholds en mapping. Indien None, wordt een default-config gebruikt.
    tile_duration_us : float, optioneel
        Alleen voor meta/log; de kernel heeft dit niet nodig voor de beslissing.

    Returns
    -------
    CompassSignResult
        global_direction: "CW"/"CCW"/"UNDECIDED"
        confidence: 0.0..1.0
        verdict_tiles: "A_LEADS_B"/"B_LEADS_A"/"ONBEPAALD"
        meta: dict met dt_stats, thresholds, counts, etc.
    """
    cfg = config or CompassSignConfig()

    # 1) dt-statistiek uit tiles
    dt_values, dt_stats = _dt_stats_from_tiles(
        tiles=tiles,
        max_abs_dt_us=cfg.max_abs_dt_us,
    )

    # 2) Tiles-verdict
    verdict_tiles, reason = _compute_tiles_verdict(dt_stats, cfg)

    # 3) Confidence
    confidence = _compute_confidence(verdict_tiles, dt_stats, cfg)

    # 4) Globale richting via mapping
    global_direction = cfg.mapping.get(verdict_tiles, "UNDECIDED")

    # 5) Meta opbouwen
    meta: Dict[str, Any] = {
        "phase_class": cfg.phase_class,
        "deadzone_us": cfg.deadzone_us,
        "min_tiles": cfg.min_tiles,
        "max_abs_dt_us": cfg.max_abs_dt_us,
        "n_tiles_total": len(tiles),
        "n_tiles_with_dt": dt_stats.get("n", 0),
        "dt_stats": dt_stats,
        "reason": reason,
        "mapping_used": dict(cfg.mapping),
    }
    if tile_duration_us is not None:
        meta["tile_duration_us"] = tile_duration_us

    return CompassSignResult(
        global_direction=global_direction,
        confidence=confidence,
        verdict_tiles=verdict_tiles,
        meta=meta,
    )


def compute_compass_sign_from_tiles_v3_doc(
    doc: Dict[str, Any],
    config: Optional[CompassSignConfig] = None,
    phase_class: Optional[str] = None,
) -> CompassSignResult:
    """
    Convenience: neem een volledige PhaseTiles v3 JSON-structuur.

    Verwacht structuur zoals aangemaakt door phase_tiles_v3_0:
      {
        ...
        "tiles_v3": [...],
        "tiles_v3_meta": {
          "tile_duration_us": ...
        }
      }

    phase_class kan gebruikt worden om het label in de config te overschrijven,
    bijv. wanneer dit uit de bestandsnaam / CLI-optie komt.
    """
    tiles = doc.get("tiles_v3") or doc.get("tiles") or []
    meta = doc.get("tiles_v3_meta") or {}
    tile_duration_us = None
    if isinstance(meta, dict):
        td = meta.get("tile_duration_us")
        if isinstance(td, (int, float)):
            tile_duration_us = float(td)

    cfg = config or CompassSignConfig()
    if phase_class is not None:
        cfg.phase_class = phase_class

    return compute_compass_sign_from_tiles(
        tiles=tiles,
        config=cfg,
        tile_duration_us=tile_duration_us,
    )
