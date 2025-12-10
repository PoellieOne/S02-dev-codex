#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_symphonia_v1_9_poc.py — Proof of Concept: Cycle-Based Awareness

EXPERIMENTEEL: Test implementatie van het "cycle heartbeat" principe.

Kernprincipes:
- CYCLE = Hartslag = Echte beweging (N→NEU→S progressie)
- Raw EVENT = Mogelijk ruis (edge oscillatie)
- Gap > threshold zonder cycles = STILL + HARD RESET
- Stereo cycles (beide sensoren) = MOVEMENT
- Mono cycles (één sensor) = TREMOR

Dit onderscheidt echte rotatie van edge-oscillatie:
- Edge oscillatie: events maar GEEN cycles (N→NEU→N→NEU...)
- Echte rotatie: events MET cycles (N→NEU→S = complete cycle)

Gebruik:
    python3 live_symphonia_v1_9_poc.py [--port /dev/ttyUSB0] [--gap-ms 500]
"""

import sys
import json
import time
import argparse
import struct
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import statistics

# === BINLINK (inline) ========================================================

SYNC = 0xA5
TYPE_EVENT24 = 0x1
TYPE_EVENT16 = 0x0

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

    def read_frames(self, timeout_ms=100):
        """Yield frames, maar return ook als timeout zonder data."""
        chunk = self.ser.read(256)
        if chunk:
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
    return {
        "kind": "event24",
        "dt_us": dt_us,
        "t_abs_us": tabs,
        "flags0": flags0,
        "flags1": flags1,
        "sensor": (flags0 >> 3) & 1,
        "to_pool": (flags1 >> 4) & 0x3,
    }


def parse_event16(p):
    dt_us, = struct.unpack_from('<H', p, 0)
    flags0 = p[2]
    flags1 = p[3]
    return {
        "kind": "event16",
        "dt_us": dt_us,
        "flags0": flags0,
        "flags1": flags1,
        "sensor": (flags0 >> 3) & 1,
        "to_pool": (flags1 >> 4) & 0x3,
    }


# === HEARTBEAT & GAP DETECTION ===============================================

@dataclass
class Heartbeat:
    """
    Detecteert "leven" op basis van event aanwezigheid.
    
    Events zijn heartbeats. Geen events = dood = STILL.
    """
    gap_threshold_ms: float = 1500.0
    last_beat_time: Optional[float] = None  # wall clock time
    last_beat_t_us: Optional[int] = None    # event timestamp
    alive: bool = False
    
    def beat(self, t_us: int, wall_time: float) -> bool:
        """
        Registreer een hartslag (event ontvangen).
        Returns: True als dit een "revival" is (was dood, nu levend)
        """
        was_dead = not self.alive
        self.last_beat_time = wall_time
        self.last_beat_t_us = t_us
        self.alive = True
        return was_dead
    
    def check(self, wall_time: float) -> bool:
        """
        Check of we nog leven.
        Returns: True als nog levend, False als gap threshold overschreden
        """
        if self.last_beat_time is None:
            return False
        
        gap_ms = (wall_time - self.last_beat_time) * 1000.0
        
        if gap_ms > self.gap_threshold_ms:
            self.alive = False
        
        return self.alive
    
    def get_gap_ms(self, wall_time: float) -> float:
        """Huidige gap in ms sinds laatste beat."""
        if self.last_beat_time is None:
            return float('inf')
        return (wall_time - self.last_beat_time) * 1000.0


@dataclass
class StereoDetector:
    """
    Detecteert stereo activiteit (beide sensoren) vs mono (edge oscillatie).
    
    MOVEMENT vereist dat BEIDE sensoren recent events hebben gezien.
    Mono activiteit = TREMOR (edge ruis, geen echte rotatie).
    """
    stereo_window_ms: float = 300.0  # Tijdvenster voor "recent"
    
    last_event_a: Optional[float] = None  # wall clock time
    last_event_b: Optional[float] = None
    
    # Cycle detection (meer betrouwbaar dan raw events)
    last_cycle_a: Optional[float] = None
    last_cycle_b: Optional[float] = None
    
    def event(self, sensor: int, wall_time: float):
        """Registreer een event van sensor A (0) of B (1)."""
        if sensor == 0:
            self.last_event_a = wall_time
        else:
            self.last_event_b = wall_time
    
    def cycle(self, sensor: int, wall_time: float):
        """Registreer een gedetecteerde cycle van sensor A (0) of B (1)."""
        if sensor == 0:
            self.last_cycle_a = wall_time
        else:
            self.last_cycle_b = wall_time
    
    def is_stereo(self, wall_time: float) -> bool:
        """
        Check of BEIDE sensoren recent cycles hebben gezien.
        Returns: True = stereo (echte beweging), False = mono (edge ruis)
        """
        if self.last_cycle_a is None or self.last_cycle_b is None:
            return False
        
        age_a_ms = (wall_time - self.last_cycle_a) * 1000.0
        age_b_ms = (wall_time - self.last_cycle_b) * 1000.0
        
        # Beide moeten recent zijn
        return age_a_ms < self.stereo_window_ms and age_b_ms < self.stereo_window_ms
    
    def get_activity_type(self, wall_time: float) -> str:
        """
        Bepaal type activiteit.
        Returns: "STEREO", "MONO_A", "MONO_B", of "NONE"
        """
        has_a = (self.last_cycle_a is not None and 
                 (wall_time - self.last_cycle_a) * 1000.0 < self.stereo_window_ms)
        has_b = (self.last_cycle_b is not None and 
                 (wall_time - self.last_cycle_b) * 1000.0 < self.stereo_window_ms)
        
        if has_a and has_b:
            return "STEREO"
        elif has_a:
            return "MONO_A"
        elif has_b:
            return "MONO_B"
        else:
            return "NONE"
    
    def reset(self):
        """Reset bij gap."""
        self.last_event_a = None
        self.last_event_b = None
        self.last_cycle_a = None
        self.last_cycle_b = None


# === SIMPLIFIED AWARENESS STATE ==============================================

POOL_NEU = 0
POOL_N = 1
POOL_S = 2

@dataclass
class AwarenessState:
    """
    Vereenvoudigde awareness state met gap-reset semantiek.
    
    Bij gap: ALLES wordt gereset. Schone lei.
    
    Activity states:
    - STILL: geen events
    - TREMOR: mono events (1 sensor, edge oscillatie)  
    - MOVEMENT: stereo events (beide sensoren, echte rotatie)
    """
    # Activity state (nieuw!)
    activity_state: str = "STILL"  # STILL, TREMOR, MOVEMENT
    
    # Rotor state (afgeleid van activity)
    rotor_state: str = "STILL"
    direction_lock: str = "UNLOCKED"
    direction: str = "UNDECIDED"
    
    # Cycles (reset bij gap)
    cycles_a: int = 0
    cycles_b: int = 0
    cycle_index: float = 0.0
    rotations: float = 0.0
    
    # Compass (reset bij gap)
    dt_ab_window: deque = field(default_factory=lambda: deque(maxlen=15))
    compass_score: float = 0.0
    
    # RPM (reset bij gap)
    last_cycle_time: Optional[float] = None
    rpm_window: deque = field(default_factory=lambda: deque(maxlen=10))
    rpm_est: float = 0.0
    
    # Cycle detection windows
    window_a: deque = field(default_factory=lambda: deque(maxlen=3))
    window_b: deque = field(default_factory=lambda: deque(maxlen=3))
    
    # Tile aggregation
    tile_cycles_a: list = field(default_factory=list)
    tile_cycles_b: list = field(default_factory=list)
    tile_start_time: Optional[float] = None
    tile_duration_ms: float = 50.0  # ~0.6 cycles at 12 cycles/rot, 60rpm
    
    # Session stats (niet gereset)
    total_events: int = 0
    total_resets: int = 0
    tremor_episodes: int = 0  # Nieuw: tel TREMOR episodes
    session_start: Optional[float] = None
    
    def reset(self):
        """HARD RESET - terug naar schone lei."""
        self.activity_state = "STILL"
        self.rotor_state = "STILL"
        self.direction_lock = "UNLOCKED"
        self.direction = "UNDECIDED"
        
        self.cycles_a = 0
        self.cycles_b = 0
        self.cycle_index = 0.0
        self.rotations = 0.0
        
        self.dt_ab_window.clear()
        self.compass_score = 0.0
        
        self.last_cycle_time = None
        self.rpm_window.clear()
        self.rpm_est = 0.0
        
        self.window_a.clear()
        self.window_b.clear()
        
        self.tile_cycles_a.clear()
        self.tile_cycles_b.clear()
        self.tile_start_time = None
        
        self.total_resets += 1


class AwarenessPipeline:
    """
    Vereenvoudigde pipeline met gap-reset en stereo detectie.
    
    Kernprincipes:
    - Events = activiteit
    - Stereo (beide sensoren) = MOVEMENT
    - Mono (één sensor) = TREMOR (edge ruis)
    - Gap = STILL = reset
    """
    
    def __init__(self, gap_threshold_ms: float = 1500.0, stereo_window_ms: float = 300.0):
        self.heartbeat = Heartbeat(gap_threshold_ms=gap_threshold_ms)
        self.stereo = StereoDetector(stereo_window_ms=stereo_window_ms)
        self.state = AwarenessState()
        
        # Config
        self.compass_alpha = 0.85
        self.threshold_high = 0.45
        self.threshold_low = 0.15
        self.cycles_per_rot = 12.0
        
    def check_heartbeat(self, wall_time: float) -> bool:
        """
        Periodieke check - roep dit aan ook als er geen events zijn!
        Returns: True als we net gestorven zijn (gap detected)
        """
        was_alive = self.heartbeat.alive
        is_alive = self.heartbeat.check(wall_time)
        
        if was_alive and not is_alive:
            # We zijn net "gestorven" - gap detected!
            self._on_gap_detected(wall_time)
            return True
        
        # Update activity state ook bij check (voor TREMOR → STILL transitie)
        if is_alive:
            self._update_activity_state(wall_time)
        
        return False
    
    def _on_gap_detected(self, wall_time: float):
        """Handler voor gap detectie."""
        self.state.reset()
        self.stereo.reset()
    
    def _update_activity_state(self, wall_time: float):
        """Update activity state op basis van stereo detector."""
        activity_type = self.stereo.get_activity_type(wall_time)
        
        prev_state = self.state.activity_state
        
        if activity_type == "STEREO":
            self.state.activity_state = "MOVEMENT"
            self.state.rotor_state = "MOVEMENT"
        elif activity_type in ("MONO_A", "MONO_B"):
            self.state.activity_state = "TREMOR"
            self.state.rotor_state = "STILL"  # TREMOR is geen echte beweging
            if prev_state != "TREMOR":
                self.state.tremor_episodes += 1
        else:
            self.state.activity_state = "STILL"
            self.state.rotor_state = "STILL"
    
    def feed_event(self, ev: Dict[str, Any], wall_time: float) -> Dict[str, Any]:
        """
        Verwerk een event.
        Returns: huidige state snapshot
        """
        self.state.total_events += 1
        if self.state.session_start is None:
            self.state.session_start = wall_time
        
        t_us = ev.get("t_abs_us", int(wall_time * 1e6))
        sensor = ev.get("sensor", 0)
        to_pool = ev.get("to_pool", 0)
        
        # NIET hier heartbeat.beat() - alleen bij cycle detectie!
        
        # Track event voor stereo detector (raw events)
        self.stereo.event(sensor, wall_time)
        
        # Start tile timing als nodig
        if self.state.tile_start_time is None:
            self.state.tile_start_time = wall_time
        
        # Process event voor cycle detection
        sensor_label = "A" if sensor == 0 else "B"
        window = self.state.window_a if sensor == 0 else self.state.window_b
        
        window.append({"t_us": t_us, "to_pool": to_pool})
        
        # 3-point cycle detection
        if len(window) == 3:
            pools = [w["to_pool"] for w in window]
            if set(pools) == {POOL_NEU, POOL_N, POOL_S}:
                cycle_time = wall_time
                
                # *** HEARTBEAT HIER - alleen bij ECHTE cycle! ***
                self.heartbeat.beat(t_us, wall_time)
                
                # Registreer cycle bij stereo detector
                self.stereo.cycle(sensor, wall_time)
                
                if sensor == 0:
                    self.state.cycles_a += 1
                    self.state.tile_cycles_a.append(cycle_time)
                else:
                    self.state.cycles_b += 1
                    self.state.tile_cycles_b.append(cycle_time)
                
                # RPM update (alleen bij MOVEMENT)
                if self.state.last_cycle_time is not None and self.state.activity_state == "MOVEMENT":
                    dt_s = cycle_time - self.state.last_cycle_time
                    if dt_s > 0:
                        rpm_inst = (1.0 / dt_s) * (60.0 / self.cycles_per_rot)
                        self.state.rpm_window.append(rpm_inst)
                        if len(self.state.rpm_window) >= 2:
                            self.state.rpm_est = statistics.mean(self.state.rpm_window)
                        else:
                            self.state.rpm_est = rpm_inst
                
                self.state.last_cycle_time = cycle_time
        
        # Update activity state
        self._update_activity_state(wall_time)
        
        # Tile processing (simplified)
        if self.state.tile_start_time is not None:
            tile_elapsed_ms = (wall_time - self.state.tile_start_time) * 1000.0
            
            if tile_elapsed_ms >= self.state.tile_duration_ms:
                self._process_tile(wall_time)
                self.state.tile_start_time = wall_time
        
        return self.get_snapshot()
    
    def _process_tile(self, wall_time: float):
        """Process accumulated tile data."""
        nA = len(self.state.tile_cycles_a)
        nB = len(self.state.tile_cycles_b)
        
        if nA == 0 and nB == 0:
            self.state.tile_cycles_a.clear()
            self.state.tile_cycles_b.clear()
            return
        
        # Cycles physical (mean fusion)
        cycles_phys = (nA + nB) / 2.0
        
        # dt_ab voor compass (alleen bij pure stereo)
        if nA == 1 and nB == 1:
            tA = self.state.tile_cycles_a[0]
            tB = self.state.tile_cycles_b[0]
            dt_ab = (tB - tA) * 1e6  # naar microseconden
            self.state.dt_ab_window.append(dt_ab)
        
        # Compass update
        if len(self.state.dt_ab_window) >= 2:
            valid_dts = list(self.state.dt_ab_window)
            cw_count = sum(1 for d in valid_dts if d > 50)  # deadzone 50us
            ccw_count = sum(1 for d in valid_dts if d < -50)
            
            if cw_count + ccw_count > 0:
                if cw_count > ccw_count:
                    window_dir = 1
                    window_conf = cw_count / len(valid_dts)
                elif ccw_count > cw_count:
                    window_dir = -1
                    window_conf = ccw_count / len(valid_dts)
                else:
                    window_dir = 0
                    window_conf = 0
                
                signed = window_dir * window_conf
                self.state.compass_score = (
                    self.compass_alpha * self.state.compass_score +
                    (1 - self.compass_alpha) * signed
                )
        
        # Direction decision
        if self.state.compass_score >= self.threshold_high:
            self.state.direction = "CW"
        elif self.state.compass_score <= -self.threshold_high:
            self.state.direction = "CCW"
        elif abs(self.state.compass_score) < self.threshold_low:
            self.state.direction = "UNDECIDED"
        
        # Lock state
        if abs(self.state.compass_score) >= self.threshold_high:
            if self.state.direction_lock == "UNLOCKED":
                self.state.direction_lock = "SOFT_LOCK"
            elif self.state.direction_lock == "SOFT_LOCK":
                self.state.direction_lock = "LOCKED"
        elif abs(self.state.compass_score) < self.threshold_low:
            self.state.direction_lock = "UNLOCKED"
        
        # Update cycle_index en rotations
        sign = 1 if self.state.direction == "CW" else (-1 if self.state.direction == "CCW" else 0)
        if self.state.direction_lock == "LOCKED" and sign != 0:
            self.state.cycle_index += sign * cycles_phys
        else:
            # Unsigned accumulation wanneer niet locked
            pass  # cycles_phys niet toegevoegd aan cycle_index
        
        self.state.rotations = self.state.cycle_index / self.cycles_per_rot
        
        # Clear tile buffers
        self.state.tile_cycles_a.clear()
        self.state.tile_cycles_b.clear()
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get current state as dict."""
        return {
            "activity_state": self.state.activity_state,
            "rotor_state": self.state.rotor_state,
            "direction_lock": self.state.direction_lock,
            "direction": self.state.direction,
            "cycles_a": self.state.cycles_a,
            "cycles_b": self.state.cycles_b,
            "cycle_index": self.state.cycle_index,
            "rotations": self.state.rotations,
            "compass_score": self.state.compass_score,
            "rpm_est": self.state.rpm_est,
            "total_events": self.state.total_events,
            "total_resets": self.state.total_resets,
            "tremor_episodes": self.state.tremor_episodes,
            "alive": self.heartbeat.alive,
            "gap_ms": self.heartbeat.get_gap_ms(time.time()),
            "stereo_type": self.stereo.get_activity_type(time.time()),
        }


# === TERMINAL UI =============================================================

class TerminalUI:
    CLEAR_LINE = "\033[K"
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    
    def __init__(self, num_lines=14):
        self.num_lines = num_lines
        self.initialized = False
        
    def init(self):
        print(self.HIDE_CURSOR, end='')
        for _ in range(self.num_lines):
            print()
        self.initialized = True
        
    def update(self, lines: list):
        if not self.initialized:
            self.init()
        print(f"\033[{self.num_lines}A", end='')
        for line in lines[:self.num_lines]:
            print(f"{self.CLEAR_LINE}{line}")
        for _ in range(self.num_lines - len(lines)):
            print(self.CLEAR_LINE)
            
    def cleanup(self):
        print(self.SHOW_CURSOR, end='')


def format_display(state: dict, events_per_sec: float, elapsed: float, gap_threshold_ms: float) -> list:
    ui = TerminalUI
    lines = []
    
    # Header
    lines.append(f"{ui.BOLD}═══════════════════════════════════════════════════════════════{ui.RESET}")
    lines.append(f"{ui.BOLD}  SYMPHONIA v1.9 POC — Cycle-Based Heartbeat{ui.RESET}")
    lines.append(f"═══════════════════════════════════════════════════════════════")
    
    # Heartbeat status
    alive = state.get('alive', False)
    gap_ms = state.get('gap_ms', 0)
    
    if alive:
        heart = f"{ui.GREEN}♥ ALIVE{ui.RESET}"
        gap_str = f"{gap_ms:6.0f}ms"
    else:
        heart = f"{ui.RED}♡ DEAD{ui.RESET}"
        gap_str = f"{ui.RED}{gap_ms:6.0f}ms{ui.RESET}"
    
    lines.append(f"  Heartbeat: {heart}  Gap: {gap_str}")
    lines.append(f"───────────────────────────────────────────────────────────────")
    
    # Activity state (nieuw!)
    activity = state.get('activity_state', 'STILL')
    stereo_type = state.get('stereo_type', 'NONE')
    
    if activity == "MOVEMENT":
        act_color = ui.GREEN
        act_icon = "◉"
    elif activity == "TREMOR":
        act_color = ui.YELLOW
        act_icon = "◎"
    else:
        act_color = ui.DIM
        act_icon = "○"
    
    lines.append(f"  Activity:  {act_color}{act_icon} {activity:<10}{ui.RESET} (sensor: {stereo_type})")
    
    # Rotor state
    rotor = state.get('rotor_state', 'STILL')
    rotor_color = ui.GREEN if rotor == 'MOVEMENT' else ui.DIM
    lines.append(f"  RotorState:     {rotor_color}{rotor:<12}{ui.RESET}")
    
    # Lock state
    lock = state.get('direction_lock', 'UNLOCKED')
    if lock == 'LOCKED':
        lock_color = ui.GREEN
    elif lock == 'SOFT_LOCK':
        lock_color = ui.YELLOW
    else:
        lock_color = ui.DIM
    lines.append(f"  └─ LockState:   {lock_color}{lock:<12}{ui.RESET}")
    
    # Direction
    direction = state.get('direction', 'UNDECIDED')
    dir_color = ui.CYAN if direction in ('CW', 'CCW') else ui.DIM
    lines.append(f"     └─ Direction: {dir_color}{direction:<12}{ui.RESET}")
    
    lines.append(f"───────────────────────────────────────────────────────────────")
    
    # Metrics
    rotations = state.get('rotations', 0)
    rpm = state.get('rpm_est', 0)
    score = state.get('compass_score', 0)
    cycles_a = state.get('cycles_a', 0)
    cycles_b = state.get('cycles_b', 0)
    resets = state.get('total_resets', 0)
    tremors = state.get('tremor_episodes', 0)
    
    lines.append(f"  Rotaties:   {ui.BOLD}{rotations:+8.2f}{ui.RESET}     Cycles: A={cycles_a}, B={cycles_b}")
    lines.append(f"  RPM:        {rpm:8.1f}       Score: {score:+.3f}")
    lines.append(f"  Events/s:   {events_per_sec:8.1f}       Resets: {resets}  Tremors: {tremors}")
    
    lines.append(f"═══════════════════════════════════════════════════════════════")
    
    return lines


# === MAIN ====================================================================

def main():
    parser = argparse.ArgumentParser(description='POC: Gap-Reset + Stereo Detection')
    parser.add_argument('--port', '-p', default='/dev/ttyUSB0')
    parser.add_argument('--baud', '-b', type=int, default=115200)
    parser.add_argument('--gap-ms', type=float, default=500.0,
                       help='Gap threshold in ms (default: 500)')
    parser.add_argument('--stereo-ms', type=float, default=300.0,
                       help='Stereo window in ms (default: 300)')
    parser.add_argument('--simple', '-s', action='store_true')
    
    args = parser.parse_args()
    
    try:
        import serial
    except ImportError:
        print("❌ pyserial niet geïnstalleerd!")
        print("   pip install pyserial")
        return 1
    
    print(f"[i] Opening {args.port} @ {args.baud}...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.05)  # Korte timeout!
    except Exception as e:
        print(f"❌ {e}")
        return 1
    
    fs = FrameStream(ser)
    pipeline = AwarenessPipeline(
        gap_threshold_ms=args.gap_ms,
        stereo_window_ms=args.stereo_ms
    )
    
    ui = None if args.simple else TerminalUI(num_lines=15)
    
    print(f"[i] Gap threshold: {args.gap_ms}ms")
    print(f"[i] Stereo window: {args.stereo_ms}ms")
    print(f"[i] Listening... (Ctrl+C to stop)")
    print()
    print(f"[i] Heartbeat = CYCLES (niet raw events)")
    print(f"    Edge oscillatie (events zonder cycles) → blijft STILL")
    print(f"    Echte rotatie (N→NEU→S cycles) → MOVEMENT")
    print()
    print(f"[i] Activity states:")
    print(f"    ◉ MOVEMENT = stereo cycles (beide sensoren)")
    print(f"    ◎ TREMOR   = mono cycles (één sensor)")
    print(f"    ○ STILL    = geen cycles (of gap timeout)")
    print()
    
    if ui:
        ui.init()
    
    t0 = time.time()
    last_display = time.time()
    events_window = deque(maxlen=100)
    
    try:
        while True:
            now = time.time()
            
            # Check heartbeat ELKE iteratie (ook zonder events!)
            gap_detected = pipeline.check_heartbeat(now)
            
            # Process frames
            for frame_type, ver, payload in fs.read_frames():
                if frame_type == TYPE_EVENT24:
                    ev = parse_event24(payload)
                elif frame_type == TYPE_EVENT16:
                    ev = parse_event16(payload)
                    ev["t_abs_us"] = int((now - t0) * 1e6)
                else:
                    continue
                
                events_window.append(now)
                state = pipeline.feed_event(ev, now)
            
            # Update display
            if now - last_display > 0.1:
                elapsed = now - t0
                recent = [t for t in events_window if now - t < 1.0]
                events_per_sec = len(recent)
                
                state = pipeline.get_snapshot()
                
                if ui:
                    lines = format_display(state, events_per_sec, elapsed, args.gap_ms)
                    ui.update(lines)
                elif args.simple:
                    rotor = state['rotor_state']
                    lock = state['direction_lock']
                    direction = state['direction']
                    rot = state['rotations']
                    gap = state['gap_ms']
                    alive = "♥" if state['alive'] else "♡"
                    
                    print(f"\r[{elapsed:6.1f}s] {alive} {rotor:9} | {lock:10} | {direction:9} | "
                          f"rot={rot:+6.2f} | gap={gap:5.0f}ms   ",
                          end='', flush=True)
                
                last_display = now
    
    except KeyboardInterrupt:
        print("\n\n[i] Stopped")
    
    finally:
        if ui:
            ui.cleanup()
        ser.close()
        
        state = pipeline.get_snapshot()
        print()
        print("=" * 60)
        print("SESSION SUMMARY")
        print("=" * 60)
        print(f"  Total events:    {state['total_events']}")
        print(f"  Total resets:    {state['total_resets']}")
        print(f"  Tremor episodes: {state['tremor_episodes']}")
        print(f"  Final activity:  {state['activity_state']}")
        print(f"  Final state:     {state['rotor_state']} / {state['direction_lock']} / {state['direction']}")
        print(f"  Rotations:       {state['rotations']:.2f}")
        print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
