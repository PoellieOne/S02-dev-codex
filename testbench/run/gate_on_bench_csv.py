#!/usr/bin/env python3
import csv
import sys
from pathlib import Path
from collections import Counter

from sym_cycles.action_gate_v0_1 import ActionGateV0_1, GateInput

def f(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def i(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

def s(x):
    return (x or "").strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: gate_on_bench_csv.py <core0_events_v1_9_bench.csv> [output.csv]")
        sys.exit(2)

    inp = Path(sys.argv[1]).expanduser().resolve()
    out = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) >= 3 else inp.with_name(inp.stem + "__gate.csv")

    gate = ActionGateV0_1()

    counts_dec = Counter()
    counts_state = Counter()
    counts_reason = Counter()

    cnt_lock = Counter()
    cnt_rotor = Counter()

    stats = {
        "awareness_conf": {"n":0,"sum":0.0,"min":1e9,"max":-1e9},
        "motion_conf": {"n":0,"sum":0.0,"min":1e9,"max":-1e9},
        "direction_locked_conf": {"n":0,"sum":0.0,"min":1e9,"max":-1e9},
        "direction_global_conf": {"n":0,"sum":0.0,"min":1e9,"max":-1e9},
        "compass_global_score": {"n":0,"sum":0.0,"min":1e9,"max":-1e9},
    }

    def bump(name, val):
        st = stats[name]
        st["n"] += 1
        st["sum"] += val
        st["min"] = min(st["min"], val)
        st["max"] = max(st["max"], val)

    with inp.open("r", encoding="utf-8", newline="") as f_in:
        rdr = csv.DictReader(f_in)
        fields = rdr.fieldnames or []
        if not fields:
            raise RuntimeError("No CSV header found")

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f_out:
            w_fields = fields + ["gate_state", "gate_decision", "gate_reason"]
            w = csv.DictWriter(f_out, fieldnames=w_fields)
            w.writeheader()

            last_t_us = None
            row_idx = 0

            for row in rdr:
                row_idx += 1

                cnt_lock[s(row.get("direction_lock_state")) or ""] += 1
                cnt_rotor[s(row.get("rotor_state")) or ""] += 1

                bump("awareness_conf", f(row.get("awareness_conf"), 0.0))
                bump("motion_conf", f(row.get("motion_conf"), 0.0))
                bump("direction_locked_conf", f(row.get("direction_locked_conf"), 0.0))
                bump("direction_global_conf", f(row.get("direction_global_conf"), 0.0))
                bump("compass_global_score", f(row.get("compass_global_score"), 0.0))

                # --- time (ms) and derived data_age_ms
                t_us = f(row.get("t_us"), 0.0)
                now_ms = int(t_us / 1000.0) if t_us > 0 else row_idx * 10

                if last_t_us is None:
                    data_age_ms = 0
                else:
                    dt_us = t_us - last_t_us
                    data_age_ms = 0 if dt_us > 0 else 9999
                last_t_us = t_us

                # --- lock state (execution-neutral)
                lock_state = s(row.get("direction_lock_state")) or "UNLOCKED"

                # --- rotor moving (simple)
                rotor_state = s(row.get("rotor_state")).upper()
                rotor_moving = (rotor_state == "MOVEMENT")

                # --- coherence score (robust fallback)
                # prefer awareness_conf, then compass_global_score, then direction_global_conf
                coherence = f(row.get("awareness_conf"), 0.0)
                if coherence <= 0.0:
                    coherence = f(row.get("compass_global_score"), 0.0)
                if coherence <= 0.0:
                    coherence = f(row.get("direction_global_conf"), 0.0)

                # Clamp (defensive)
                if coherence < 0.0:
                    coherence = 0.0
                if coherence > 1.0:
                    coherence = 1.0

                gi = GateInput(
                    now_ms=now_ms,
                    coherence_score=coherence,
                    lock_state=lock_state,
                    data_age_ms=i(data_age_ms),
                    rotor_active=bool(rotor_moving),
                    fields={
                        "event_index": row.get("event_index"),
                        "tile_index": row.get("tile_index"),
                        "rotor_state": row.get("rotor_state"),
                        "direction_lock_state": row.get("direction_lock_state"),
                        "compass_global_score": row.get("compass_global_score"),
                    }
                )

                go = gate.evaluate(gi)

                row["gate_state"] = go.state.value
                row["gate_decision"] = go.decision.value
                row["gate_reason"] = go.reason

                counts_dec[go.decision.value] += 1
                counts_state[go.state.value] += 1
                counts_reason[go.reason] += 1

                w.writerow(row)

    print("Wrote:", out)
    print("Decisions:", dict(counts_dec))
    print("States   :", dict(counts_state))
    print("Reasons  :", dict(counts_reason))

    print("\nField stats:")
    for k, st in stats.items():
        n = st["n"] or 1
        avg = st["sum"] / n
        print(f"  {k}: n={st['n']} min={st['min']:.3f} max={st['max']:.3f} avg={avg:.3f}")

    print("\nCounts:")
    print("  direction_lock_state:", dict(cnt_lock))
    print("  rotor_state:", dict(cnt_rotor))


if __name__ == "__main__":
    main()
