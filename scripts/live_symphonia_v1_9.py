#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_symphonia_v1_9.py — Live ESP32 Koppeling met Realtime Pipeline

Combineert:
- binlink.py: Frame parsing van ESP32
- capture_core0.py: Serial communicatie
- realtime_states_v1_9_canonical.py: Awareness pipeline

Functionaliteit:
- Live EVENT24 frames van ESP32 ontvangen
- Direct door RealtimePipeline v1.9 verwerken
- Realtime terminal UI met state updates
- Optioneel: JSONL logging voor replay
- Optioneel: CSV export voor visualisatie

Gebruik:
    python3 live_symphonia_v1_9.py [--port /dev/ttyUSB0] [--baud 115200]
    python3 live_symphonia_v1_9.py --profile bench_tolerant --log
    python3 live_symphonia_v1_9.py --help

Vereisten:
    pip install pyserial --break-system-packages
"""
import os
import sys
import json
import time
import argparse
import struct
from pathlib import Path
from datetime import datetime
from collections import deque

# === Symlink proof =================================================

# Altijd beginnen bij de "echte" file-locatie (symlink-proof)
HERE = Path(__file__).resolve()

# 1) Optioneel: expliciete override via env var, bv:
#    export SYMPHONIA_ROOT=/home/ralph/PoellieOne/symphonia-core0-pc
env_root = os.getenv("SYMPHONIA_ROOT")
PROJECT_ROOT = None

if env_root:
    PROJECT_ROOT = Path(env_root).expanduser().resolve()
else:
    # 2) Automatisch: loop omhoog totdat we een 'sym_cycles' map vinden
    for parent in [HERE.parent, *HERE.parents]:
        if (parent / "sym_cycles").exists():
            PROJECT_ROOT = parent
            break

if PROJECT_ROOT is None:
    raise RuntimeError(
        "Kon 'sym_cycles' niet vinden. "
        "Zet SYMPHONIA_ROOT of zorg dat er ergens boven deze file een 'sym_cycles/' map staat."
    )

# 3) Zorg dat Python dit als import-root ziet
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# === BINLINK MODULE (inline) =================================================

SYNC = 0xA5
TYPE_EVENT16      = 0x0
TYPE_EVENT24      = 0x1
TYPE_SUMMARY16    = 0x2
TYPE_SUMMARY24    = 0x3
TYPE_FILTER_STATS = 0x4
TYPE_LINK_STATS   = 0x5


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for ch in data:
        crc ^= (ch << 8) & 0xFFFF
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else ((crc << 1) & 0xFFFF)
    return crc


class FrameStream:
    def __init__(self, ser):
        self.ser = ser
        self.buf = bytearray()

    def read_frames(self):
        while True:
            chunk = self.ser.read(256)
            if not chunk:
                yield from ()
                continue
            self.buf.extend(chunk)
            
            while True:
                idx = self.buf.find(bytes([SYNC]))
                if idx < 0:
                    self.buf.clear()
                    break
                if idx > 0:
                    del self.buf[:idx]
                if len(self.buf) < 4:
                    break
                typever = self.buf[1]
                plen = self.buf[2]
                need = 1 + 1 + 1 + plen + 2
                if len(self.buf) < need:
                    break
                frame = bytes(self.buf[:need])
                del self.buf[:need]
                
                crc_rx = struct.unpack('<H', frame[-2:])[0]
                crc_tx = crc16_ccitt_false(frame[1:3+plen])
                if crc_rx != crc_tx:
                    continue
                    
                t = (typever >> 4) & 0x0F
                v = typever & 0x0F
                payload = frame[3:-2]
                yield (t, v, payload)


def parse_event24(p):
    dt_us, = struct.unpack_from('<H', p, 0)
    tabs, = struct.unpack_from('<I', p, 2)
    flags0 = p[6]
    flags1 = p[7]
    dvdt_q15, = struct.unpack_from('<h', p, 8)
    mono_q8 = p[10]
    snr_q8 = p[11]
    fit_err_q8 = p[12]
    rpm_hint_q, = struct.unpack_from('<H', p, 13)
    score_q8 = p[15]
    seq = p[16]
    return {
        "kind": "event24",
        "dt_us": dt_us,
        "t_abs_us": tabs,
        "flags0": flags0,
        "flags1": flags1,
        "dvdt_q15": dvdt_q15,
        "mono_q8": mono_q8,
        "snr_q8": snr_q8,
        "fit_err_q8": fit_err_q8,
        "rpm_hint_q": rpm_hint_q,
        "score_q8": score_q8,
        "seq": seq
    }


def parse_event16(p):
    dt_us, = struct.unpack_from('<H', p, 0)
    flags0 = p[2]
    flags1 = p[3]
    dvdt_q15, = struct.unpack_from('<h', p, 4)
    mono_q8 = p[6]
    snr_q8 = p[7]
    score_q8 = p[8]
    seq = p[9]
    return {
        "kind": "event16",
        "dt_us": dt_us,
        "flags0": flags0,
        "flags1": flags1,
        "dvdt_q15": dvdt_q15,
        "mono_q8": mono_q8,
        "snr_q8": snr_q8,
        "score_q8": score_q8,
        "seq": seq,
    }


def decode_flags(flags0, flags1):
    pair = (flags0 >> 7) & 1
    qlevel = (flags0 >> 5) & 0x3
    polarity = (flags0 >> 4) & 1
    sensor = (flags0 >> 3) & 1
    from_pool = (flags1 >> 6) & 0x3
    to_pool = (flags1 >> 4) & 0x3
    dir_hint = (flags1 >> 2) & 0x3
    edge_kind = (flags1 >> 0) & 0x3
    return dict(
        pair=pair, qlevel=qlevel, polarity=polarity, sensor=sensor,
        from_pool=from_pool, to_pool=to_pool, dir_hint=dir_hint, edge_kind=edge_kind
    )


# === TERMINAL UI =============================================================

class TerminalUI:
    """Simple terminal UI voor realtime status display."""
    
    # ANSI escape codes
    CLEAR_LINE = "\033[K"
    MOVE_UP = "\033[A"
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    
    def __init__(self, num_lines=12):
        self.num_lines = num_lines
        self.initialized = False
        
    def init(self):
        """Initialize display area."""
        print(self.HIDE_CURSOR, end='')
        # Print empty lines to reserve space
        for _ in range(self.num_lines):
            print()
        self.initialized = True
        
    def update(self, lines: list):
        """Update display with new content."""
        if not self.initialized:
            self.init()
        
        # Move cursor up
        print(f"\033[{self.num_lines}A", end='')
        
        # Print each line
        for i, line in enumerate(lines[:self.num_lines]):
            print(f"{self.CLEAR_LINE}{line}")
        
        # Clear remaining lines
        for _ in range(self.num_lines - len(lines)):
            print(self.CLEAR_LINE)
            
    def cleanup(self):
        """Restore terminal state."""
        print(self.SHOW_CURSOR, end='')


def format_state_display(state: dict, events_per_sec: float, elapsed: float) -> list:
    """Format pipeline state for terminal display."""
    ui = TerminalUI
    
    lines = []
    
    # Header
    lines.append(f"{ui.BOLD}═══════════════════════════════════════════════════════════════{ui.RESET}")
    lines.append(f"{ui.BOLD}  SYMPHONIA S02 v1.9 — LIVE ESP32{ui.RESET}")
    lines.append(f"═══════════════════════════════════════════════════════════════")
    
    # Rotor state
    rotor = state.get('rotor_state', 'STILL')
    rotor_color = ui.GREEN if rotor == 'MOVEMENT' else ui.YELLOW
    lines.append(f"  RotorState:     {rotor_color}{rotor:<12}{ui.RESET}")
    
    # Lock state
    lock = state.get('direction_lock_state', 'UNLOCKED')
    if lock == 'LOCKED':
        lock_color = ui.GREEN
    elif lock == 'SOFT_LOCK':
        lock_color = ui.YELLOW
    else:
        lock_color = ui.RED
    lines.append(f"  └─ LockState:   {lock_color}{lock:<12}{ui.RESET}")
    
    # Direction
    direction = state.get('direction_global_effective', 'UNDECIDED')
    dir_color = ui.CYAN if direction in ('CW', 'CCW') else ui.RESET
    lines.append(f"     └─ Direction: {dir_color}{direction:<12}{ui.RESET}")
    
    lines.append(f"───────────────────────────────────────────────────────────────")
    
    # Metrics
    rotations = state.get('rotations', 0)
    theta = state.get('theta_deg', 0)
    rpm = state.get('rpm_est', 0)
    score = state.get('compass_global_score', 0) if 'compass_global_score' in state else 0
    
    # Get compass score from snapshot if available
    if hasattr(state, 'compass_snapshot') and state.compass_snapshot:
        score = state.compass_snapshot.global_score
    
    lines.append(f"  Rotaties:   {ui.BOLD}{rotations:+8.2f}{ui.RESET}     θ: {theta:6.1f}°")
    lines.append(f"  RPM:        {rpm:8.1f}       Score: {score:+.3f}")
    lines.append(f"  Events/s:   {events_per_sec:8.1f}       Tijd: {elapsed:.1f}s")
    
    lines.append(f"═══════════════════════════════════════════════════════════════")
    
    return lines


# === MAIN APPLICATION ========================================================

def main():
    parser = argparse.ArgumentParser(
        description='Live ESP32 koppeling met Symphonia v1.9 pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Voorbeelden:
  python3 live_symphonia_v1_9.py
  python3 live_symphonia_v1_9.py --port /dev/ttyUSB0 --profile bench_tolerant
  python3 live_symphonia_v1_9.py --log --csv
  python3 live_symphonia_v1_9.py --simple  # Geen fancy UI
        """
    )
    
    parser.add_argument('--port', '-p', default='/dev/ttyUSB0',
                       help='Serial port (default: /dev/ttyUSB0)')
    parser.add_argument('--baud', '-b', type=int, default=115200,
                       help='Baud rate (default: 115200)')
    parser.add_argument('--profile', choices=['production', 'bench', 'bench_tolerant'],
                       default='bench_tolerant',
                       help='Pipeline profile (default: bench_tolerant)')
    parser.add_argument('--min-normal-tile', type=int, default=2,
                       help='Minimum normal tile index (default: 2)')
    parser.add_argument('--log', '-l', action='store_true',
                       help='Log events to JSONL file')
    parser.add_argument('--csv', '-c', action='store_true',
                       help='Export tiles to CSV file')
    parser.add_argument('--simple', '-s', action='store_true',
                       help='Simple output mode (no fancy terminal UI)')
    parser.add_argument('--xram', help='Path to xram JSON configuration')
    
    args = parser.parse_args()
    
    # Import serial
    try:
        import serial
    except ImportError:
        print("❌ pyserial niet geïnstalleerd!")
        print("   Installeer met: pip install pyserial --break-system-packages")
        return 1
    
    # Import pipeline
    try:
        from sym_cycles.realtime_states_v1_9_canonical import (
            RealtimePipeline, PipelineProfile,
            PROFILE_PRODUCTION, PROFILE_BENCH, PROFILE_BENCH_TOLERANT,
            load_profile_from_xram,
        )
    except ImportError:
        # Try loading from same directory
        import importlib.util
        base_dir = Path(__file__).resolve().parent  # volg symlink naar de échte file
        spec = importlib.util.spec_from_file_location(
            "realtime_states",
            base_dir / "realtime_states_v1_9_canonical.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            RealtimePipeline = module.RealtimePipeline
            PipelineProfile = module.PipelineProfile
            PROFILE_PRODUCTION = module.PROFILE_PRODUCTION
            PROFILE_BENCH = module.PROFILE_BENCH
            PROFILE_BENCH_TOLERANT = module.PROFILE_BENCH_TOLERANT
            load_profile_from_xram = module.load_profile_from_xram
        else:
            print("❌ realtime_states_v1_9_canonical.py niet gevonden!")
            return 1
    
    # Select profile
    if args.xram:
        print(f"[i] Loading profile '{args.profile}' from xram: {args.xram}")
        profile = load_profile_from_xram(args.xram, args.profile)
    elif args.profile == 'bench_tolerant':
        profile = PROFILE_BENCH_TOLERANT
    elif args.profile == 'bench':
        profile = PROFILE_BENCH
    else:
        profile = PROFILE_PRODUCTION
    
    # Apply min_normal_tile override
    if args.min_normal_tile != profile.min_normal_tile:
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
    
    # Open serial port
    print(f"[i] Opening {args.port} @ {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"❌ Kan serial port niet openen: {e}")
        return 1
    
    # Create frame stream
    fs = FrameStream(ser)
    
    # Create pipeline
    pipeline = RealtimePipeline(profile=profile)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = None
    csv_file = None
    csv_writer = None
    
    if args.log:
        log_path = f"live_events_{timestamp}.jsonl"
        log_file = open(log_path, 'w')
        print(f"[i] Logging to: {log_path}")
    
    if args.csv:
        csv_path = f"live_tiles_{timestamp}.csv"
        csv_file = open(csv_path, 'w')
        # Write header
        csv_headers = [
            "tile_index", "t_us", "nA", "nB", "cycles_physical",
            "rotor_state", "direction_lock_state", "direction_global_effective",
            "compass_global_score", "cycle_index", "rotations", "theta_deg",
            "rpm_est", "awareness_conf"
        ]
        csv_file.write(",".join(csv_headers) + "\n")
        print(f"[i] CSV output: {csv_path}")
    
    # Setup terminal UI
    ui = None
    if not args.simple:
        ui = TerminalUI(num_lines=12)
    
    # Statistics
    events_seen = 0
    tiles_emitted = 0
    t0 = time.time()
    last_update = time.time()
    events_window = deque(maxlen=100)  # For events/sec calculation
    
    print(f"[i] Profile: {profile.name}")
    print(f"[i] Listening... (Ctrl+C to stop)")
    print()
    
    if ui:
        ui.init()
    
    try:
        for frame_type, ver, payload in fs.read_frames():
            now = time.time()
            
            # Parse event
            ev = None
            if frame_type == TYPE_EVENT24:
                ev = parse_event24(payload)
                ev.update(decode_flags(ev["flags0"], ev["flags1"]))
            elif frame_type == TYPE_EVENT16:
                ev = parse_event16(payload)
                ev.update(decode_flags(ev["flags0"], ev["flags1"]))
                # EVENT16 heeft geen t_abs_us, we moeten die simuleren
                ev["t_abs_us"] = int((now - t0) * 1e6)
            
            if ev is None:
                continue
            
            events_seen += 1
            events_window.append(now)
            
            # Log event
            if log_file:
                log_file.write(json.dumps(ev) + "\n")
            
            # Feed to pipeline
            snap = pipeline.feed_event(ev)
            
            # Process emitted tiles
            for tile in snap.tiles_emitted:
                if tile.get('cycles_physical', 0) > 0:
                    tiles_emitted += 1
                    
                    # Write to CSV
                    if csv_file:
                        mv = snap.movement_state
                        cs = snap.compass_snapshot
                        row = [
                            tile.get('tile_index', 0),
                            tile.get('t_center_us', 0),
                            tile.get('nA', 0),
                            tile.get('nB', 0),
                            tile.get('cycles_physical', 0),
                            mv.get('rotor_state', ''),
                            mv.get('direction_lock_state', ''),
                            mv.get('direction_global_effective', ''),
                            cs.global_score if cs else 0,
                            mv.get('cycle_index', 0),
                            mv.get('rotations', 0),
                            mv.get('theta_deg', 0),
                            mv.get('rpm_est', 0),
                            mv.get('awareness_conf', 0),
                        ]
                        csv_file.write(",".join(str(x) for x in row) + "\n")
            
            # Update display periodically
            if now - last_update > 0.1:  # 10 Hz update
                elapsed = now - t0
                
                # Calculate events/sec from window
                recent = [t for t in events_window if now - t < 1.0]
                events_per_sec = len(recent)
                
                # Get current state
                state = snap.movement_state
                state['compass_snapshot'] = snap.compass_snapshot
                
                if ui:
                    lines = format_state_display(state, events_per_sec, elapsed)
                    ui.update(lines)
                elif args.simple:
                    # Simple one-line output
                    mv = snap.movement_state
                    rotor = mv.get('rotor_state', 'STILL')
                    lock = mv.get('direction_lock_state', 'UNLOCKED')
                    direction = mv.get('direction_global_effective', '?')
                    rotations = mv.get('rotations', 0)
                    rpm = mv.get('rpm_est', 0)
                    
                    print(f"\r[{elapsed:6.1f}s] {rotor:9} | {lock:10} | {direction:9} | "
                          f"rot={rotations:+6.2f} | rpm={rpm:5.1f} | ev/s={events_per_sec:3.0f}",
                          end='', flush=True)
                
                last_update = now
    
    except KeyboardInterrupt:
        print("\n\n[i] Interrupted by user")
    
    finally:
        # Cleanup
        if ui:
            ui.cleanup()
        
        ser.close()
        
        if log_file:
            log_file.close()
        if csv_file:
            csv_file.close()
        
        # Final summary
        elapsed = time.time() - t0
        final_snap = pipeline.snapshot()
        mv = final_snap.movement_state
        
        print()
        print("=" * 65)
        print("SESSION SUMMARY")
        print("=" * 65)
        print(f"  Duration:        {elapsed:.1f}s")
        print(f"  Events received: {events_seen} ({events_seen/elapsed:.1f}/s)")
        print(f"  Tiles processed: {tiles_emitted}")
        print()
        print(f"  Final RotorState:   {mv.get('rotor_state', '?')}")
        print(f"  Final LockState:    {mv.get('direction_lock_state', '?')}")
        print(f"  Final Direction:    {mv.get('direction_global_effective', '?')}")
        print(f"  Total Rotations:    {mv.get('rotations', 0):.2f}")
        print(f"  Total Cycles:       {mv.get('total_cycles_physical', 0):.1f}")
        print("=" * 65)
        
        if args.log:
            print(f"\n[i] Events saved to: live_events_{timestamp}.jsonl")
        if args.csv:
            print(f"[i] Tiles saved to: live_tiles_{timestamp}.csv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
