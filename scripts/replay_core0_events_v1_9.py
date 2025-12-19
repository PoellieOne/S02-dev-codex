#!/usr/bin/env python3
"""
replay_core0_events_v1_9.py — Canonical Architecture

v1.9 Features:
- CompassSign v3 (canonical confidence berekening)
- Hiërarchische RotorState (STILL → MOVEMENT)
- Unlock mechanisme (LOCKED → SOFT_LOCK → UNLOCKED)
- Claim-at-Lock
- BootWarmup
- xram JSON configuratie support
"""

import sys
import json
import csv
import argparse
from pathlib import Path

try:
    from sym_cycles.realtime_states_v1_9_canonical import (
        RealtimePipeline, PipelineProfile,
        PROFILE_PRODUCTION, PROFILE_BENCH, PROFILE_BENCH_TOLERANT,
        load_profile_from_xram,
        ROTOR_STATE_STILL, ROTOR_STATE_MOVEMENT,
        LOCK_STATE_UNLOCKED, LOCK_STATE_SOFT_LOCK, LOCK_STATE_LOCKED,
    )
except ImportError:
    exec(open("/mnt/user-data/outputs/realtime_states_v1_9_canonical.py").read())


def main():
    parser = argparse.ArgumentParser(description='Replay core0_events v1.9 Canonical')
    parser.add_argument('input', help='Input JSONL bestand')
    parser.add_argument('output', nargs='?', help='Output CSV (optioneel)')
    parser.add_argument('--profile', choices=['production', 'bench', 'bench_tolerant'], default='bench')
    parser.add_argument('--xram', help='Path to xram JSON configuration file')
    parser.add_argument('--min-normal-tile', type=int, default=None)
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Load profile
    if args.xram:
        print(f"[i] Loading profile '{args.profile}' from xram: {args.xram}")
        profile = load_profile_from_xram(args.xram, args.profile)
    elif args.profile == 'bench_tolerant':
        profile = PROFILE_BENCH_TOLERANT
    elif args.profile == 'bench':
        profile = PROFILE_BENCH
    else:
        profile = PROFILE_PRODUCTION
    
    if args.min_normal_tile is not None:
        profile = PipelineProfile(
            name=f"{profile.name}-custom",
            compass_alpha=profile.compass_alpha,
            compass_threshold_high=profile.compass_threshold_high,
            compass_threshold_low=profile.compass_threshold_low,
            compass_window_tiles=profile.compass_window_tiles,
            compass_deadzone_us=profile.compass_deadzone_us,
            compass_min_tiles=profile.compass_min_tiles,
            compass_max_abs_dt_us=profile.compass_max_abs_dt_us,
            lock_confidence_threshold=profile.lock_confidence_threshold,
            lock_soft_threshold=profile.lock_soft_threshold,
            unlock_tiles_threshold=profile.unlock_tiles_threshold,
            rpm_alpha=profile.rpm_alpha,
            jitter_max_rel=profile.jitter_max_rel,
            jitter_window_size=profile.jitter_window_size,
            rpm_move_thresh=profile.rpm_move_thresh,
            rpm_slow_thresh=profile.rpm_slow_thresh,
            rpm_still_thresh=profile.rpm_still_thresh,
            tile_span_cycles=profile.tile_span_cycles,
            min_normal_tile=args.min_normal_tile,
            stereo_fusion=profile.stereo_fusion,
            cycles_per_rot=profile.cycles_per_rot,
        )
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_v1_9_{profile.name}.csv")
    
    print(f"[i] Input: {input_path}")
    print(f"[i] Output: {output_path}")
    print(f"[i] Profile: {profile.name}")
    print(f"[i] v1.9 Canonical Architecture:")
    print(f"      - CompassSign v3 (canonical)")
    print(f"      - Hiërarchische RotorState (STILL → MOVEMENT)")
    print(f"      - Unlock mechanisme (LOCKED → SOFT_LOCK → UNLOCKED)")
    print(f"[i] Parameters:")
    print(f"      compass_alpha: {profile.compass_alpha}")
    print(f"      threshold_high: {profile.compass_threshold_high}")
    print(f"      threshold_low: {profile.compass_threshold_low}")
    print(f"      compass_deadzone_us: {profile.compass_deadzone_us}")
    print(f"      compass_min_tiles: {profile.compass_min_tiles}")
    print(f"      min_normal_tile: {profile.min_normal_tile}")
    print(f"      unlock_tiles_threshold: {profile.unlock_tiles_threshold}")
    
    events = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                if ev.get("kind") == "event24":
                    events.append(ev)
            except json.JSONDecodeError:
                continue
    
    print(f"[i] Loaded {len(events)} EVENT24 records")
    
    pipeline = RealtimePipeline(profile=profile)
    
    headers = [
        "event_index", "tile_index", "tile_state", "t_us", "has_data",
        "nA", "nB", "is_pure_stereo", "cycles_physical",
        # Compass
        "compass_global_direction", "compass_global_score",
        "compass_window_direction", "compass_window_score",
        # Hiërarchische state (v1.9)
        "rotor_state", "direction_lock_state",
        "direction_locked_dir", "direction_locked_conf",
        "direction_global_effective", "direction_global_conf",
        # Cycles
        "cycles_unsigned", "cycles_claimed_at_lock",
        "boot_tiles_skipped", "boot_cycles_skipped",
        "cycle_index", "rotations", "theta_deg",
        "total_cycles_physical", "pure_stereo_tiles",
        # RPM
        "rpm_inst", "rpm_est", "rpm_jitter", "cadence_ok",
        # Motion
        "motion_state", "motion_conf",
        "awareness_conf", "profile_name",
    ]
    
    rows = []
    
    for i, ev in enumerate(events):
        snap = pipeline.feed_event(ev)
        
        for tile in snap.tiles_emitted:
            cycles_physical = tile.get("cycles_physical", 0)
            if cycles_physical <= 0:
                continue
            
            mv = snap.movement_state
            cs = snap.compass_snapshot
            
            row = {
                "event_index": i,
                "tile_index": tile.get("tile_index", 0),
                "tile_state": tile.get("tile_state", "NORMAL"),
                "t_us": tile.get("t_center_us", 0),
                "has_data": tile.get("has_data", False),
                "nA": tile.get("nA", 0),
                "nB": tile.get("nB", 0),
                "is_pure_stereo": tile.get("is_pure_stereo", False),
                "cycles_physical": cycles_physical,
                
                "compass_global_direction": cs.direction if cs else "",
                "compass_global_score": cs.global_score if cs else 0,
                "compass_window_direction": cs.direction if cs else "",
                "compass_window_score": cs.conf if cs else 0,
                
                # v1.9: Hiërarchische state
                "rotor_state": mv.get("rotor_state", "STILL"),
                "direction_lock_state": mv.get("direction_lock_state", "UNLOCKED"),
                "direction_locked_dir": mv.get("direction_locked_dir", ""),
                "direction_locked_conf": mv.get("direction_locked_conf", 0),
                "direction_global_effective": mv.get("direction_global_effective", ""),
                "direction_global_conf": mv.get("direction_global_conf", 0),
                
                "cycles_unsigned": mv.get("cycles_unsigned", 0),
                "cycles_claimed_at_lock": mv.get("cycles_claimed_at_lock", 0),
                "boot_tiles_skipped": mv.get("boot_tiles_skipped", 0),
                "boot_cycles_skipped": mv.get("boot_cycles_skipped", 0),
                "cycle_index": mv.get("cycle_index", 0),
                "rotations": mv.get("rotations", 0),
                "theta_deg": mv.get("theta_deg", 0),
                "total_cycles_physical": mv.get("total_cycles_physical", 0),
                "pure_stereo_tiles": mv.get("pure_stereo_tiles", 0),
                
                "rpm_inst": mv.get("rpm_inst", 0),
                "rpm_est": mv.get("rpm_est", 0),
                "rpm_jitter": mv.get("rpm_jitter", 0),
                "cadence_ok": mv.get("cadence_ok", False),
                
                "motion_state": mv.get("motion_state", ""),
                "motion_conf": mv.get("motion_conf", 0),
                "awareness_conf": mv.get("awareness_conf", 0),
                
                "profile_name": mv.get("profile_name", profile.name),
            }
            
            rows.append(row)
    
    # Finalize
    # v1.9 canonical pipeline emits tiles during feed_event(); there is no TilesState.finalize()
    # and no public tiles_state on RealtimePipeline. Nothing to flush here.
    final_tiles = []

    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"[✓] Wrote {len(rows)} rows to {output_path}")
    
    # Debug summary
    # Optional debug: not available in all pipeline variants
    if hasattr(pipeline, "debug_tiles_and_cycles"):
        debug = pipeline.debug_tiles_and_cycles()
        cycles_debug = debug["cycles"]["per_sensor"]
        tiles_debug = debug["tiles"]

        print()
        print(f"=== DEBUG SUMMARY v1.9 ({profile.name}) ===")
        print(f"Cycles detected:")
        print(f"  - Sensor A: {cycles_debug['A']['cycles']}")
        print(f"  - Sensor B: {cycles_debug['B']['cycles']}")
        print(f"Total cycles_physical: {debug['total_cycles_physical']:.1f}")
        print()

        print(f"CompassSign v3:")
        print(f"  - pure_stereo_tiles: {tiles_debug['pure_stereo_tiles']}")
        print(f"  - deadzone_us: {profile.compass_deadzone_us}")
        print(f"  - min_tiles: {profile.compass_min_tiles}")
        print()

        print(f"BootWarmup:")
        print(f"  - boot_tiles_skipped: {debug['boot_tiles_skipped']}")
        print(f"  - boot_cycles_skipped: {debug['boot_cycles_skipped']:.1f}")
        print()

        print(f"Claim at Lock:")
        print(f"  - cycles_claimed_at_lock: {debug['cycles_claimed_at_lock']:.1f}")
        print(f"  - cycles_unsigned remaining: {debug['cycles_unsigned']:.1f}")
        print(f"  - lock_claimed: {debug['lock_claimed']}")
        print()
    
        final = pipeline.snapshot()
        mv = final.movement_state
        print("=== FINAL STATE (v1.9 Hiërarchisch) ===")
        print(f"Profile: {profile.name}")
        print(f"RotorState: {mv.get('rotor_state', 'STILL')}")
        print(f"  └── DirectionLock: {mv.get('direction_lock_state', 'UNLOCKED')}")
        print(f"       └── Direction: {mv.get('direction_global_effective', 'UNDECIDED')}")
        print(f"cycle_index: {mv.get('cycle_index', 0):.1f}")
        print(f"rotations: {mv.get('rotations', 0):.2f}")
        print(f"theta_deg: {mv.get('theta_deg', 0):.1f}°")
        print(f"rpm_est: {mv.get('rpm_est', 0):.1f}")
        print()

        # State transitions summary
        if rows:
            print("=== STATE TRANSITIONS ===")
            prev_rotor = None
            prev_lock = None
            for row in rows:
                rotor = row.get("rotor_state")
                lock = row.get("direction_lock_state")
                tile = row.get("tile_index")

                if rotor != prev_rotor:
                    print(f"  Tile {tile}: RotorState → {rotor}")
                    prev_rotor = rotor
                if lock != prev_lock:
                    print(f"  Tile {tile}: DirectionLock → {lock}")
                    prev_lock = lock

        print()

        # Verificatie
        expected_rotations = debug['total_cycles_physical'] / 12.0
        actual_rotations = abs(mv.get('rotations', 0))
        direction = mv.get('direction_global_effective', 'UNDECIDED')
        lock_state = mv.get('direction_lock_state', 'UNLOCKED')
        rotor_state = mv.get('rotor_state', 'STILL')

        print("=== VERIFICATIE ===")
        print(f"Verwacht (fysiek): {expected_rotations:.2f} rotaties")
        print(f"Gemeten: {mv.get('rotations', 0):.2f} rotaties ({direction})")

        if lock_state == 'LOCKED':
            if abs(expected_rotations - actual_rotations) < 0.5:
                print("✅ LOCKED en rotaties correct!")
            else:
                print(f"⚠️ LOCKED maar rotaties verschil: {expected_rotations - actual_rotations:.2f}")
        elif lock_state == 'SOFT_LOCK':
            print(f"⚠️ SOFT_LOCK (tentative direction: {direction})")
        else:
            print(f"⚠️ {lock_state} binnen {rotor_state}")
            if debug['cycles_unsigned'] > 0:
                print(f"   cycles_unsigned wacht op lock: {debug['cycles_unsigned']:.1f}")
    else:
        print("[i] debug_tiles_and_cycles: not available (skipped)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
