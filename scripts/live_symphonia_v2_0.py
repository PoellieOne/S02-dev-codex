#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_symphonia_v2_0.py — Live ESP32 met L1 OriginTracker v0.4.7c

S02.OriginTracker v0.4.7c — Tile Flow Observability.

Changelog v0.4.7c:
- NEW: Tile flow metrics (tiles_emitted, tiles_with_cycles, cycles_in_tiles, last_tile_index)
- IMPROVED: Rate-limited mismatch detection (grace period 0.75s or 8 CB cycles)
- IMPROVED: "awaiting tile coherence" instead of "⚠ MISMATCH" spam
- IMPROVED: Clearer "latch_samples_confirmed" label in summary
- NO BEHAVIOR CHANGE: Only observability/logging improvements

Changelog v0.4.8 (Action Gate integration):
- NEW: ActionGateV0_1 hook for execution-layer state machine
- LOGS: Gate decisions logged to JSONL (observability only)
- NO EXECUTION: Gate does not trigger motor actions
"""

import os, sys, json, time, argparse, struct
from pathlib import Path
from datetime import datetime
from collections import deque

DEBUG = bool(os.getenv("S02_DEBUG_CYCLES", ""))
INF = float('inf')

HERE = Path(__file__).resolve()
for p in [HERE.parent, *HERE.parents]:
    if (p / "sym_cycles").exists():
        sys.path.insert(0, str(p)); break

def wrap_deg_signed(x): return ((x + 180.0) % 360.0) - 180.0
def fmt_age(a): return "∞" if a == INF or a > 9999 else f"{a:.2f}"

# === BINLINK ===
SYNC = 0xA5
TYPE_EVENT16 = 0x0
TYPE_EVENT24 = 0x1

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
        chunk = self.ser.read(256)
        if not chunk:
            return
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

def parse_ev24(p):
    if len(p) < 8:
        return None
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
        "from_pool": (flags1 >> 6) & 0x3,
    }

def parse_ev16(p):
    if len(p) < 4:
        return None
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
        "from_pool": (flags1 >> 6) & 0x3,
    }

def decode_flags(flags0, flags1):
    return {
        "pair": (flags0 >> 7) & 1,
        "qlevel": (flags0 >> 5) & 0x3,
        "polarity": (flags0 >> 4) & 1,
        "sensor": (flags0 >> 3) & 1,
        "from_pool": (flags1 >> 6) & 0x3,
        "to_pool": (flags1 >> 4) & 0x3,
        "dir_hint": (flags1 >> 2) & 0x3,
        "edge_kind": flags1 & 0x3,
    }

def parse_action_intent(value: str) -> "ActionIntent":
    if not value:
        return ActionIntent.INTENT_NONE
    v = value.strip().upper()
    if v in ("NONE", "INTENT_NONE", "0"):
        return ActionIntent.INTENT_NONE
    if v in ("ACTIVATE", "INTENT_ACTIVATE", "1"):
        return ActionIntent.INTENT_ACTIVATE
    if v in ("HOLD", "INTENT_HOLD", "2"):
        return ActionIntent.INTENT_HOLD
    if v in ("RELEASE", "INTENT_RELEASE", "3"):
        return ActionIntent.INTENT_RELEASE
    return ActionIntent.INTENT_NONE

class IntentInput:
    def __init__(self, path: str | None, source: str | None):
        self.path = Path(path) if path else None
        if source:
            self.source = source
        elif self.path:
            self.source = f"file:{self.path}"
        else:
            self.source = "none"

    def read(self) -> tuple["ActionIntent", str]:
        if not self.path:
            return (ActionIntent.INTENT_NONE, self.source)
        try:
            raw = self.path.read_text(encoding="utf-8")
        except Exception:
            return (ActionIntent.INTENT_NONE, self.source)
        return (parse_action_intent(raw), self.source)

class TimestampedTee:
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file
        self._pending = ""

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        if not self.log_file:
            return
        self._pending += data
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            ts = datetime.now().isoformat(timespec="milliseconds")
            self.log_file.write(f"{ts} {line}\n")
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        if self.log_file:
            self.log_file.flush()

    def drain(self):
        if self.log_file and self._pending:
            ts = datetime.now().isoformat(timespec="milliseconds")
            self.log_file.write(f"{ts} {self._pending}\n")
            self.log_file.flush()
            self._pending = ""

class NullStream:
    def write(self, data): pass
    def flush(self): pass

# === L1 Import ===
try:
    from sym_cycles.l1_physical_activity import L1PhysicalActivity, L1Config, L1State, L1Snapshot, L1Reason, AwState, INF
except ImportError:
    from l1_physical_activity import L1PhysicalActivity, L1Config, L1State, L1Snapshot, L1Reason, AwState, INF

# === Action Gate Import (v0.4.8) ===
try:
    from sym_cycles.action_gate_v0_2 import ActionGateV0_2, GateInput, GateConfig, ActionIntent
    ACTION_GATE_AVAILABLE = True
except ImportError:
    ACTION_GATE_AVAILABLE = False

# === UI ===
class UI:
    CLR, HIDE, SHOW = "\033[K", "\033[?25l", "\033[?25h"
    B, R, G, Y, C, M, D = "\033[1m", "\033[0m", "\033[92m", "\033[93m", "\033[96m", "\033[95m", "\033[2m"
    def __init__(self, n=32): self.n, self.ok = n, False
    def init(self): print(self.HIDE, end=''); [print() for _ in range(self.n)]; self.ok = True
    def update(self, lines):
        if not self.ok: self.init()
        print(f"\033[{self.n}A", end='')
        for l in lines[:self.n]: print(f"{self.CLR}{l}")
        for _ in range(self.n - len(lines)): print(self.CLR)
    def cleanup(self): print(self.SHOW, end='')

def fmt_display(s, l2, eps, elapsed, gate_info=None):
    u = UI
    aw = s.aw_state.value
    aw_r = s.aw_reason.value
    aw_col = {"STILL": u.D, "NOISE": u.Y, "PRE_MOVEMENT": u.C, "PRE_ROTATION": u.M, "MOVEMENT": u.G}.get(aw, u.R)
    aw_ico = {"STILL": "○", "NOISE": "◎", "PRE_MOVEMENT": "◑", "PRE_ROTATION": "◐", "MOVEMENT": "◉"}.get(aw, "?")
    
    lines = [f"{u.B}═══════════════════════════════════════════════════════════════════{u.R}",
             f"{u.B}  SYMPHONIA v2.6.0 — OriginTracker v0.4.5 (MDI + Two-Phase){u.R}",
             "═══════════════════════════════════════════════════════════════════"]
    
    stale = f" {u.Y}[STALE]{u.R}" if s.l2_stale else ""
    lines.append(f"  {u.B}AWARENESS:{u.R} {aw_col}{aw_ico} {aw:<12}{u.R}{stale}")
    lines.append(f"  {u.D}reason:{u.R} {aw_r}")
    
    # Ages
    lines.append(f"  {u.B}Ages:{u.R} ageE={fmt_age(s.ageE_s):>5}s  ageC={fmt_age(s.ageC_s):>5}s  act={s.activity_score:.2f}")
    
    lines.append("───────────────────────────────────────────────────────────────────")
    
    # MDI (v0.4.5) — with mode and latch
    mdi_col = u.G if s.mdi_disp_micro_deg >= 20 else (u.Y if s.mdi_disp_micro_deg >= 10 else u.D)
    trem_col = u.Y if s.mdi_tremor_score > 0.3 else u.D
    mdi_uniq = ",".join(str(x) for x in sorted(s.mdi_unique_pools)) if s.mdi_unique_pools else "-"
    micro_t0 = f"{s.micro_t0_s:.2f}s" if s.micro_t0_s else "-"
    mode = getattr(s, 'mdi_mode', 'C')
    ev_win = getattr(s, 'mdi_ev_win', 0)
    step = getattr(s, 'mdi_micro_deg_per_step_used', 10.0)
    lines.append(f"  {u.B}MDI[{mode}]:{u.R} {mdi_col}μ={s.mdi_disp_micro_deg:5.1f}°{u.R}  "
                 f"conf={s.mdi_conf:.2f}/{s.mdi_conf_acc:.2f}  {trem_col}trem={s.mdi_tremor_score:.2f}{u.R}")
    lines.append(f"       ev={ev_win} step={step:.0f}° chg={s.mdi_pool_changes} uniq={{{mdi_uniq}}} t0={micro_t0}")
    # Latch info (Mode C)
    latch_set = getattr(s, 'mdi_latch_set', False)
    latch_age = getattr(s, 'mdi_latch_age_s', None)
    confirmed = getattr(s, 'mdi_confirmed', False)
    chg_latch = getattr(s, 'mdi_changes_since_latch', 0)
    if mode == "C":
        latch_ico = u.G + "●" + u.R if confirmed else (u.Y + "◐" + u.R if latch_set else u.D + "○" + u.R)
        age_str = f"{latch_age:.2f}s" if latch_age else "-"
        lines.append(f"       latch={latch_ico} age={age_str} chgL={chg_latch}")
    
    lines.append("───────────────────────────────────────────────────────────────────")
    
    # Pool window (V0.4)
    uniq_str = ",".join(str(x) for x in sorted(s.pool_unique_win)) if s.pool_unique_win else "-"
    pool_col = u.G if s.pool_valid_rate_win >= 0.7 else (u.Y if s.pool_valid_rate_win >= 0.4 else u.D)
    lines.append(f"  {u.B}POOL win:{u.R} chg={s.pool_changes_win}  uniq={{{uniq_str}}}  "
                 f"{pool_col}valid={s.pool_valid_rate_win*100:.0f}%{u.R}")

    # Gate (v0.2) — display only existing fields
    gi = gate_info or {}
    g_state = gi.get("state", "-")
    g_decision = gi.get("decision", "-")
    g_intent = gi.get("intent", "-")
    g_trans = gi.get("transitions", 0)
    g_coh = gi.get("coherence", 0.0)
    g_lock = gi.get("lock", "-")
    g_rotor = gi.get("rotor", "-")
    g_age = gi.get("data_age_ms", 0)
    lines.append(f"  {u.B}GATE:{u.R} state={g_state} decision={g_decision} intent={g_intent} trans={g_trans}")
    lines.append(f"       coherence={g_coh:.2f} lock={g_lock} rotor={g_rotor} data_age_ms={g_age}")

    # Candidate (V0.4)
    if s.origin_candidate_set:
        lines.append(f"  {u.C}CAND:{u.R} t0={s.origin_candidate_time_s:.2f}s  "
                     f"conf={s.origin_candidate_conf:.2f}  reason={s.origin_candidate_reason}")
    else:
        lines.append(f"  {u.D}CAND: not set{u.R}")
    
    # Commit (V0.4)
    if s.origin_commit_set:
        delay = (s.origin_time_s - s.origin_time0_s) if s.origin_time_s and s.origin_time0_s else 0
        lines.append(f"  {u.G}COMMIT:{u.R} t0={s.origin_time0_s:.2f}s  tC={s.origin_time_s:.2f}s  "
                     f"θ0={s.origin_theta_deg or 0:.1f}°  delay={delay:.2f}s")
    else:
        lines.append(f"  {u.D}COMMIT: not set{u.R}")
    
    lines.append("───────────────────────────────────────────────────────────────────")
    
    # Displacement
    acc_col = u.G if abs(s.disp_acc_deg) >= 20 else (u.Y if abs(s.disp_acc_deg) >= 10 else u.D)
    dir_col = u.C if s.early_dir != "UNDECIDED" else u.D
    lines.append(f"  {u.B}ACC:{u.R} {acc_col}{s.disp_acc_deg:+6.1f}°{u.R}  "
                 f"Δorigin: {s.disp_from_origin_deg:+6.1f}°  ω: {s.speed_deg_s:5.1f}°/s  "
                 f"dir: {dir_col}{s.early_dir}{u.R}")
    
    lines.append("───────────────────────────────────────────────────────────────────")
    
    # L1 tactile
    l1c = {"STILL": u.D, "FEELING": u.C, "SCRAPE": u.Y, "DISPLACEMENT": u.M, "MOVING": u.G}.get(s.state.value, u.R)
    lines.append(f"  {u.B}L1:{u.R} {l1c}{s.state.value:<12}{u.R}  θ̂={s.theta_hat_deg:6.1f}°  "
                 f"Δθ={s.delta_theta_deg_signed:+5.1f}°  cy={s.total_cycles:.0f}")
    cb = int(s.encoder_conf * 20)
    lines.append(f"  conf:[{'█'*cb}{'░'*(20-cb)}] {s.encoder_conf:.2f}")
    
    lines.append("───────────────────────────────────────────────────────────────────")
    
    # L2
    rotor = l2.get("rotor_state", "?")
    lock = l2.get("direction_lock_state", "?")
    dir_ = l2.get("direction_global_effective", "?")
    rc = u.G if rotor == "MOVEMENT" else u.D
    lc = u.G if lock == "LOCKED" else (u.Y if lock == "SOFT_LOCK" else u.D)
    lines.append(f"  {u.B}L2:{u.R} {rc}{rotor:<10}{u.R} {lc}{lock:<10}{u.R} {dir_:<8}")
    
    lines += ["───────────────────────────────────────────────────────────────────",
              f"  Events/s: {eps:6.1f}    Tijd: {elapsed:.1f}s",
              "═══════════════════════════════════════════════════════════════════"]
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', '-p', default='/dev/ttyUSB0')
    ap.add_argument('--baud', '-b', type=int, default=115200)
    ap.add_argument('--profile', choices=['production', 'bench', 'bench_tolerant'], default='bench_tolerant')
    ap.add_argument('--origin-step', type=float, default=30.0)
    ap.add_argument('--pool-win-ms', type=float, default=250.0)
    # v0.4.5 MDI mode
    ap.add_argument('--mdi-mode', choices=['A', 'B', 'C'], default='C',
                    help='MDI mode: A=aggressive, B=adaptive step, C=latch/confirm (default)')
    ap.add_argument('--log', '-l', action='store_true')
    ap.add_argument('--simple', '-s', action='store_true')
    ap.add_argument('--scoreboard', action='store_true')
    ap.add_argument('--scoreboard-hz', type=float, default=2.0,
                    help='Scoreboard refresh rate in Hz (default: 2.0)')
    ap.add_argument('--quiet-console', action='store_true',
                    help='Suppress non-scoreboard stdout output')
    ap.add_argument('--intent-file', default=None,
                    help='Path to file with ActionIntent value (per tick)')
    ap.add_argument('--intent-source', default=None,
                    help='ActionIntent source label (defaults to file:PATH)')
    ap.add_argument('--gate-log', action='store_true',
                    help='Print Action Gate v0.2 log entries')
    ap.add_argument('--log-file', default=None,
                    help='Write all console output to a timestamped log file')
    ap.add_argument('--log-dir', nargs='?', const='testbench/out', default=None,
                    help='Directory for timestamped log files (default: testbench/out)')
    ap.add_argument('--gate-smoke', action='store_true',
                    help='Run Action Gate smoke loop without ESP')
    ap.add_argument('--gate-smoke-ticks', type=int, default=20,
                    help='Ticks for --gate-smoke')
    ap.add_argument('--gate-smoke-interval-ms', type=int, default=100,
                    help='Tick interval for --gate-smoke')
    args = ap.parse_args()

    log_fh = None
    tee_out = None
    tee_err = None
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_requested = bool(args.log_file or args.log_dir)
    stdout_stream = orig_stdout if not args.quiet_console else NullStream()
    if log_requested:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.log_file:
            log_path = Path(args.log_file)
        else:
            log_dir = Path(args.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"live_symphonia_{ts}.log"
        log_fh = open(log_path, "w", encoding="utf-8")
        tee_out = TimestampedTee(stdout_stream, log_fh)
        tee_err = TimestampedTee(sys.stderr, log_fh)
        sys.stdout = tee_out
        sys.stderr = tee_err
        print(f"[i] Console log file: {log_path}")
    elif args.quiet_console:
        sys.stdout = stdout_stream

    scoreboard_stream = sys.stdout
    if args.quiet_console:
        scoreboard_stream = TimestampedTee(orig_stdout, log_fh) if log_fh else orig_stdout

    # --- v0.4.5a run-wide summary semantics ---
    saw_pre_movement = False
    first_micro_t0 = None
    peak_micro_deg = 0.0
    peak_mdi_conf_acc = 0.0

    saw_candidate = False
    saw_commit = False
    first_candidate_t = None
    first_commit_t = None

    peak_cb_cycles = 0
    peak_cb_events = 0
    
    # Latch tracking
    latch_episodes = 0
    latch_dropped = 0
    latch_confirmed = 0
    
    gate_log_enabled = args.gate_log or log_requested
    if args.gate_smoke:
        if not ACTION_GATE_AVAILABLE:
            print("[i] Action Gate not available (module not found)")
            if tee_out:
                tee_out.drain()
            if tee_err:
                tee_err.drain()
            if log_fh:
                log_fh.close()
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            return 1
        action_gate = ActionGateV0_2()
        intent_input = IntentInput(args.intent_file, args.intent_source)
        print(f"[i] Action Gate v{ActionGateV0_2.VERSION} initialized (smoke mode)")
        for _ in range(args.gate_smoke_ticks):
            now = time.time()
            intent, intent_source = intent_input.read()
            gate_input = GateInput(
                now_ms=int(now * 1000),
                coherence_score=0.9,
                lock_state="LOCKED",
                data_age_ms=0,
                rotor_active=False,
                force_fallback=False,
                arm_signal=True,
                activate_signal=True,
                action_intent=intent,
                intent_source=intent_source,
            )
            gate_output = action_gate.evaluate(gate_input)
            if gate_log_enabled:
                for entry in gate_output.log_entries:
                    print(entry)
            time.sleep(args.gate_smoke_interval_ms / 1000.0)
        print(f"[i] Gate smoke done; transitions={action_gate.transition_count}")
        if tee_out:
            tee_out.drain()
        if tee_err:
            tee_err.drain()
        if log_fh:
            log_fh.close()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        return 0

    try: import serial
    except:
        print("❌ pyserial!")
        if tee_out:
            tee_out.drain()
        if tee_err:
            tee_err.drain()
        if log_fh:
            log_fh.close()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        return 1
    
    try:
        from sym_cycles.realtime_states_v1_9_canonical import RealtimePipeline, PROFILE_PRODUCTION, PROFILE_BENCH, PROFILE_BENCH_TOLERANT
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("rs", Path(__file__).parent / "realtime_states_v1_9_canonical.py")
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            RealtimePipeline, PROFILE_PRODUCTION, PROFILE_BENCH, PROFILE_BENCH_TOLERANT = mod.RealtimePipeline, mod.PROFILE_PRODUCTION, mod.PROFILE_BENCH, mod.PROFILE_BENCH_TOLERANT
        else: print("❌ realtime_states!"); return 1
    
    l2_prof = {"bench_tolerant": PROFILE_BENCH_TOLERANT, "bench": PROFILE_BENCH}.get(args.profile, PROFILE_PRODUCTION)
    
    # v0.4.5: Wire CLI mdi_mode to L1Config
    l1_cfg = L1Config(
        cycles_per_rot=l2_prof.cycles_per_rot,
        origin_step_deg=args.origin_step,
        pool_win_ms=args.pool_win_ms,
        mdi_mode=args.mdi_mode,  # CRITICAL: Wire CLI → Config
    )
    
    print(f"[i] MDI mode: {args.mdi_mode} | origin_step: {args.origin_step}° | profile: {args.profile}")
    print(f"[i] Opening {args.port}...")
    try: ser = serial.Serial(args.port, args.baud, timeout=0.01)
    except Exception as e: print(f"❌ {e}"); return 1
    
    fs = FrameStream(ser)
    l2 = RealtimePipeline(profile=l2_prof)
    l1 = L1PhysicalActivity(config=l1_cfg)
    
    log_file = None
    if args.log:
        lp = f"live_origin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        log_file = open(lp, 'w'); print(f"[i] Log: {lp}")
    
    ui = None if args.simple or args.scoreboard or args.quiet_console else UI(32)
    print(f"[i] Profile: {l2_prof.name}, origin_step={args.origin_step}°, pool_win={args.pool_win_ms}ms")
    print("[i] Running...")
    if ui: ui.init()
    
    t0 = session_t0 = last_disp = last_tick = time.time()
    events_win, total_ev = deque(maxlen=100), 0
    prev_cy, cy_total = 0.0, 0.0
    snap = L1Snapshot(state=L1State.STILL, reason=L1Reason.INIT)
    l2_snap = {}
    
    # v0.4.6: Initialize dual cycle counter variables
    cb_cycles_tick = 0
    cb_cycles_total = 0
    cb_last_reason = "-"
    cycles_source_key = "unknown"
    claim_path_ok = False
    claim_path_mismatch = False
    dcy = 0.0
    last_logged_cb_total = -1  # v0.4.7: For mismatch dedup
    
    # v0.4.7c: Tile flow tracking (observability)
    tiles_emitted_total = 0
    tiles_with_cycles_total = 0
    cycles_in_tiles_total = 0.0
    last_tile_index = None
    
    # v0.4.7c: Rate-limited mismatch detection
    MISMATCH_GRACE_S = 0.75
    MISMATCH_CB_MIN = 8
    mismatch_first_seen_t = None
    mismatch_last_logged_t = 0.0
    last_scoreboard_t = 0.0
    scoreboard_interval_s = 1.0 / args.scoreboard_hz if args.scoreboard_hz > 0 else 0.5

    # v0.4.8: Action Gate (execution layer, observability only)
    action_gate = None
    gate_output = None
    gate_transitions = 0
    intent_input = None
    gate_info = {
        "state": "-",
        "decision": "-",
        "intent": ActionIntent.INTENT_NONE.value if ACTION_GATE_AVAILABLE else "-",
        "transitions": 0,
        "coherence": 0.0,
        "lock": "-",
        "rotor": "-",
        "data_age_ms": 0,
    }
    if ACTION_GATE_AVAILABLE:
        action_gate = ActionGateV0_2()
        intent_input = IntentInput(args.intent_file, args.intent_source)
        print(f"[i] Action Gate v{ActionGateV0_2.VERSION} initialized (observability mode)")
    else:
        print(f"[i] Action Gate not available (module not found)")
    
    try:
        while True:
            now = time.time()
            elapsed = now - session_t0
            ev_batch = 0
            
            for ft, ver, pay in fs.read_frames():
                ev = None
                if ft == TYPE_EVENT24:
                    ev = parse_ev24(pay)
                    if ev:
                        ev.update(decode_flags(ev["flags0"], ev["flags1"]))
                elif ft == TYPE_EVENT16:
                    ev = parse_ev16(pay)
                    if ev:
                        ev.update(decode_flags(ev["flags0"], ev["flags1"]))
                        ev["t_abs_us"] = int((now - t0) * 1e6)
                
                if ev is None:
                    continue
                
                events_win.append(now); total_ev += 1; ev_batch += 1
                
                # Record pool (V0.4)
                l1.record_pool(ev.get("to_pool"), ev.get("sensor", 0), now)
                
                res = l2.feed_event(ev)
                l2_snap = res.movement_state
                l2_snap["compass_snapshot"] = res.compass_snapshot
                
                # v0.4.7c: Tile flow metrics (observability)
                tiles_this_tick = getattr(res, 'tiles_emitted', []) or []
                for tile in tiles_this_tick:
                    tiles_emitted_total += 1
                    cyc_phys = tile.get("cycles_physical", 0)
                    if cyc_phys > 0:
                        tiles_with_cycles_total += 1
                        cycles_in_tiles_total += cyc_phys
                    tidx = tile.get("tile_index")
                    if tidx is not None:
                        last_tile_index = tidx
                
                # v0.4.7b: Canonical cycle truth from v1.9
                # MovementBody.snapshot() provides "total_cycles_physical" (float)
                cy_total = l2_snap.get("total_cycles_physical", 0.0)
                cycles_source_key = "total_cycles_physical"
                
                # CB internal counter (for diagnostics only)
                cb_debug = l2_snap.get("_cb", {})
                cb_cycles_total = cb_debug.get("cycles_total", 0)
                cb_last_reason = cb_debug.get("last_reason", "-") or "-"
                
                # Cycles emitted this tick (from debug)
                debug = l2_snap.get("_debug", {})
                cb_cycles_tick = debug.get("cycles_emitted_n", 0)
                
                # v0.4.7c: Rate-limited mismatch detection
                # Mismatch = CB has cycles but MovementBody hasn't received them yet
                potential_mismatch = (cb_cycles_total > 0 and cy_total == 0)
                
                if potential_mismatch:
                    if mismatch_first_seen_t is None:
                        mismatch_first_seen_t = now
                    # Only flag after grace period OR enough CB cycles
                    grace_elapsed = (now - mismatch_first_seen_t) >= MISMATCH_GRACE_S
                    cb_threshold_met = cb_cycles_total >= MISMATCH_CB_MIN
                    claim_path_mismatch = grace_elapsed or cb_threshold_met
                else:
                    mismatch_first_seen_t = None
                    claim_path_mismatch = False
            
            dcy = cy_total - prev_cy
            if dcy > 0: prev_cy = cy_total
            
            if now - last_tick >= 0.05 or ev_batch > 0:
                last_tick = now
                cs = l2_snap.get("compass_snapshot")
                dc = abs(getattr(cs, 'global_score', 0.0)) if cs else 0.0
                lk = l2_snap.get("direction_lock_state", "UNLOCKED")
                de = l2_snap.get("direction_global_effective", "UNDECIDED")
                
                snap = l1.update(wall_time=now, cycles_physical_total=cy_total, events_this_batch=ev_batch,
                                 direction_conf=dc, lock_state=lk, direction_effective=de)

                # --- v0.4.8: Action Gate evaluation (observability only) ---
                if action_gate is not None:
                    # Compute data age from last event time
                    data_age_ms = int((now - t0) * 1000) if ev_batch == 0 else 0
                    if snap.ageE_s != INF:
                        data_age_ms = int(snap.ageE_s * 1000)

                    intent = ActionIntent.INTENT_NONE
                    intent_source = "none"
                    if intent_input is not None:
                        intent, intent_source = intent_input.read()

                    # Build gate input from live pipeline snapshot fields
                    gate_input = GateInput(
                        now_ms=int(now * 1000),
                        coherence_score=dc,  # compass confidence as coherence
                        lock_state=lk,       # direction_lock_state from L2
                        data_age_ms=data_age_ms,
                        rotor_active=(l2_snap.get("rotor_state", "STILL") == "MOVEMENT"),
                        force_fallback=False,
                        arm_signal=(lk in ("SOFT_LOCK", "LOCKED")),
                        activate_signal=(lk == "LOCKED" and dc >= 0.5),
                        action_intent=intent,
                        intent_source=intent_source,
                    )

                    prev_transitions = action_gate.transition_count
                    gate_output = action_gate.evaluate(gate_input)
                    if action_gate.transition_count > prev_transitions:
                        gate_transitions += 1
                    if gate_log_enabled and gate_output:
                        for entry in gate_output.log_entries:
                            print(entry)
                    if gate_output:
                        gate_info.update({
                            "state": gate_output.state.value,
                            "decision": gate_output.decision.value,
                            "intent": intent.value,
                            "transitions": action_gate.transition_count,
                            "coherence": dc,
                            "lock": lk,
                            "rotor": l2_snap.get("rotor_state", "STILL"),
                            "data_age_ms": data_age_ms,
                        })

                # --- v0.4.5: run-wide tracking ---
                aw = getattr(snap, "aw_state", None)
                aw_name = aw.value if hasattr(aw, "value") else aw
                aw_reason = getattr(snap, "aw_reason", None)
                reason_name = aw_reason.value if hasattr(aw_reason, "value") else str(aw_reason)
                
                if aw_name == "PRE_MOVEMENT":
                    saw_pre_movement = True
                    mt0 = getattr(snap, "micro_t0_s", None)
                    if first_micro_t0 is None and mt0 is not None:
                        first_micro_t0 = mt0

                # Latch tracking (Mode C)
                if reason_name == "MDI_LATCH":
                    latch_episodes += 1
                elif reason_name == "MDI_LATCH_DROPPED":
                    latch_dropped += 1
                elif reason_name == "MDI_TRIGGER" and getattr(snap, "mdi_confirmed", False):
                    latch_confirmed += 1

                peak_micro_deg = max(peak_micro_deg, float(getattr(snap, "mdi_disp_micro_deg", 0.0) or 0.0))
                peak_mdi_conf_acc = max(peak_mdi_conf_acc, float(getattr(snap, "mdi_conf_acc", getattr(snap, "mdi_conf", 0.0)) or 0.0))

                if getattr(snap, "origin_candidate_set", False):
                    if not saw_candidate:
                        saw_candidate = True
                        first_candidate_t = getattr(snap, "origin_candidate_time_s", None)

                if getattr(snap, "origin_commit_set", False):
                    if not saw_commit:
                        saw_commit = True
                        first_commit_t = getattr(snap, "origin_time_s", None)

                # CycleBuilder telemetry truth (if present)
                cb = l2_snap.get("_cb", {})
                if isinstance(cb, dict):
                    peak_cb_cycles = max(peak_cb_cycles, int(cb.get("cb_cycles_emitted_total", cb.get("cycles_total", 0)) or 0))
                    peak_cb_events = max(peak_cb_events, int(cb.get("cb_events_seen_total", cb.get("events_total", 0)) or 0))
                
                if log_file:
                    entry = {
                        "t_s": round(elapsed, 4), "ev": ev_batch, "cy": cy_total, "dcy": dcy,
                        "θ̂": round(snap.theta_hat_deg, 2), "Δθ_signed": round(snap.delta_theta_deg_signed, 2),
                        "l1": {"state": snap.state.value, "reason": snap.reason.value,
                               "act": round(snap.activity_score, 3), "conf": round(snap.encoder_conf, 3)},
                        "ages": {"ageE_s": round(snap.ageE_s, 3) if snap.ageE_s != INF else "INF",
                                 "ageC_s": round(snap.ageC_s, 3) if snap.ageC_s != INF else "INF"},
                        "pool_win": {"chg": snap.pool_changes_win, 
                                     "uniq": list(snap.pool_unique_win),
                                     "valid_rate": round(snap.pool_valid_rate_win, 3)},
                        "pools_total": {"hist": snap.to_pool_hist},
                        # MDI v0.4.5 — FULL OBSERVABILITY
                        "mdi": {"mode": getattr(snap, "mdi_mode", "C"),
                                "ev_win": getattr(snap, "mdi_ev_win", 0),
                                "step_used": getattr(snap, "mdi_micro_deg_per_step_used", 10.0),
                                "micro_acc": round(snap.mdi_micro_acc, 2),
                                "disp_deg": round(snap.mdi_disp_micro_deg, 1),
                                "conf": round(snap.mdi_conf, 3),
                                "conf_acc": round(getattr(snap, "mdi_conf_acc", snap.mdi_conf), 3),
                                "conf_used": round(getattr(snap, "mdi_conf_used", snap.mdi_conf), 3),  # v0.4.5
                                "tremor": round(snap.mdi_tremor_score, 3),
                                "pool_chg": snap.mdi_pool_changes,
                                "unique": list(snap.mdi_unique_pools),
                                "valid_rate": round(snap.mdi_valid_rate, 3),
                                "t0": round(snap.micro_t0_s, 3) if snap.micro_t0_s else None,
                                "dir_hint": snap.micro_dir_hint,
                                # Latch (Mode C) — FULL STATE
                                "latch_set": getattr(snap, "mdi_latch_set", False),
                                "latch_age": round(snap.mdi_latch_age_s, 3) if getattr(snap, "mdi_latch_age_s", None) else None,
                                "chg_latch": getattr(snap, "mdi_changes_since_latch", 0),
                                "confirmed": getattr(snap, "mdi_confirmed", False),
                                "latch_reason": getattr(snap, "mdi_latch_reason", "")},
                        "candidate": {"set": snap.origin_candidate_set,
                                      "t0": round(snap.origin_candidate_time_s, 3) if snap.origin_candidate_time_s else None,
                                      "conf": round(getattr(snap, "origin_candidate_conf", 0), 3)},
                        "commit": {"set": snap.origin_commit_set,
                                   "t0": round(snap.origin_time0_s, 3) if snap.origin_time0_s else None,
                                   "tC": round(snap.origin_time_s, 3) if snap.origin_time_s else None,
                                   "θ0": round(snap.origin_theta_deg, 2) if snap.origin_theta_deg else None},
                        "disp": {"acc": round(snap.disp_acc_deg, 2), 
                                 "from_O": round(snap.disp_from_origin_deg, 2),
                                 "speed": round(snap.speed_deg_s, 1), 
                                 "dir": snap.early_dir},
                        "aw": {"state": snap.aw_state.value, "reason": snap.aw_reason.value},
                        # v0.4.7b: l2_snap is a dict in v1.9 canonical
                        "l2": {"rotor": l2_snap.get("rotor_state"),
                               "lock": lk, "dir": de,
                               "total_cycles_physical": l2_snap.get("total_cycles_physical"),
                               "cycle_index": l2_snap.get("cycle_index"),
                               "rotations": l2_snap.get("rotations")},
                        # v0.4.7b: Canonical Cycle Truth
                        "_cycle_truth": {
                            "used_total": cy_total,
                            "cb_total": cb_cycles_total,
                            "cb_tick": cb_cycles_tick,
                            "cb_last_reason": cb_last_reason,
                            "l2_delta": dcy,
                            "source_key": cycles_source_key,
                            "mismatch": claim_path_mismatch,
                        },
                        # v0.4.7c: Tile flow metrics
                        "_tile_flow": {
                            "tiles_emitted": tiles_emitted_total,
                            "tiles_with_cycles": tiles_with_cycles_total,
                            "cycles_in_tiles": cycles_in_tiles_total,
                            "last_tile_index": last_tile_index,
                        },
                        # v0.4.8: Action Gate (execution layer observability)
                        "_gate": {
                            "state": gate_output.state.value if gate_output else None,
                            "decision": gate_output.decision.value if gate_output else None,
                            "reason": gate_output.reason if gate_output else None,
                            "allowed": gate_output.allowed if gate_output else None,
                        } if gate_output else None,
                    }
                    log_file.write(json.dumps(entry) + "\n")
                
                if args.scoreboard and (now - last_scoreboard_t >= scoreboard_interval_s):
                    last_scoreboard_t = now
                    aw = snap.aw_state.value
                    cand = "C" if snap.origin_candidate_set else "-"
                    comm = "O" if snap.origin_commit_set else "-"
                    stale = "[S]" if snap.l2_stale else ""
                    
                    # v0.4.5 MDI mode + latch
                    mode = getattr(snap, "mdi_mode", "C")
                    ev_win = getattr(snap, "mdi_ev_win", 0)
                    latch = "L" if getattr(snap, "mdi_latch_set", False) else "-"
                    conf_stat = "✓" if getattr(snap, "mdi_confirmed", False) else "-"
                    
                    # v0.4.7c: CycleTruth with tile coherence status
                    if cy_total > 0:
                        coherence_status = "✓"
                    elif claim_path_mismatch:
                        coherence_status = "(awaiting tile coherence)"
                    else:
                        coherence_status = ""
                    
                    # v0.4.7c: Rate-limit mismatch logging (max 1/sec)
                    show_mismatch_line = claim_path_mismatch and (now - mismatch_last_logged_t >= 1.0)
                    if show_mismatch_line:
                        mismatch_last_logged_t = now
                    
                    print(f"{aw:12} {snap.aw_reason.value:24} cand={cand} comm={comm} "
                          f"[{mode}] μ={snap.mdi_disp_micro_deg:4.0f}° ev={ev_win} latch={latch}{conf_stat} {stale}",
                          file=scoreboard_stream)
                    print(f"  CycleTruth: used={cy_total:.0f} cb={cb_cycles_total} src={cycles_source_key} {coherence_status}",
                          file=scoreboard_stream)
                    
                    # v0.4.7c: Tile flow metrics
                    tile_idx_str = str(last_tile_index) if last_tile_index is not None else "-"
                    print(f"  Flow: tiles={tiles_emitted_total} w/cycles={tiles_with_cycles_total} "
                          f"cycles_in_tiles={cycles_in_tiles_total:.1f} last_tile={tile_idx_str}",
                          file=scoreboard_stream)
                    print(f"  Gate: state={gate_info['state']} decision={gate_info['decision']} "
                          f"intent={gate_info['intent']} trans={gate_info['transitions']}",
                          file=scoreboard_stream)
                    print(f"  GateBasis: coherence={gate_info['coherence']:.2f} lock={gate_info['lock']} "
                          f"rotor={gate_info['rotor']} data_age_ms={gate_info['data_age_ms']}",
                          file=scoreboard_stream)
            
            if ui and now - last_disp > 0.1:
                eps = len([t for t in events_win if now - t < 1.0])
                ui.update(fmt_display(snap, l2_snap, eps, elapsed, gate_info))
                last_disp = now
            elif args.simple and now - last_disp > 0.1:
                stale = "[S]" if snap.l2_stale else ""
                mode = getattr(snap, "mdi_mode", "C")
                print(f"\r[{elapsed:5.1f}s] {snap.aw_state.value:12} [{mode}] μ={snap.mdi_disp_micro_deg:4.0f}° "
                      f"cand={snap.origin_candidate_set} cy={cy_total:.0f} {stale}", 
                      end='', flush=True)
                last_disp = now
    except KeyboardInterrupt:
        print("\n[i] Stopped")
    finally:
        if ui: ui.cleanup()
        ser.close()
        if log_file: log_file.close()
        if tee_out:
            tee_out.drain()
        if tee_err:
            tee_err.drain()
        if log_fh:
            log_fh.close()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        
        # === SUMMARY v0.4.7c ===
        print(f"\n{'='*70}")
        print(f"SUMMARY: {time.time()-session_t0:.1f}s, {total_ev} events")
        print(f"MDI Mode: {args.mdi_mode}")
        print(f"Final: {snap.aw_state.value}, reason: {snap.aw_reason.value}")
        
        # v0.4.7c: Tile Flow Metrics
        print("-"*70)
        print("TILE FLOW (v0.4.7c):")
        print(f"  Tiles emitted:        {tiles_emitted_total}")
        print(f"  Tiles with cycles:    {tiles_with_cycles_total}")
        print(f"  Cycles in tiles:      {cycles_in_tiles_total:.1f}")
        print(f"  Last tile index:      {last_tile_index if last_tile_index is not None else '-'}")
        
        # v0.4.7c: Canonical Cycle Truth
        print("-"*70)
        print("CANONICAL CYCLE TRUTH (v0.4.7c):")
        print(f"  total_cycles_physical: {cy_total:.1f}")
        print(f"  CB cycles_total:       {cb_cycles_total}")
        print(f"  Source key:            {cycles_source_key}")
        if cb_cycles_total > 0 and cy_total == 0:
            print(f"  ℹ️  CB detected {cb_cycles_total} cycles; tiles not yet flushed to MovementBody")
        elif cb_cycles_total == 0 and cy_total == 0:
            print(f"  ℹ️  No cycles (normal for short hand-burst)")
        elif cy_total > 0:
            print(f"  ✅ Cycles flowing: {cy_total:.0f} cycles reached MovementBody")
        
        # MDI v0.4.5
        test1_pass = saw_pre_movement
        print("-"*70)
        print("MDI v0.4.5 SUMMARY (run-wide)")
        print(f"Test-1: {'✅ PASS' if test1_pass else '❌ FAIL'}  (PRE_MOVEMENT seen during run)")
        print(f"Peak: μ={peak_micro_deg:.1f}°  conf_acc={peak_mdi_conf_acc:.2f}  first t0μ={first_micro_t0}")
        if args.mdi_mode == 'C':
            # v0.4.7c: Clearer label for confirmed counter
            print(f"Latch: episodes={latch_episodes}  dropped={latch_dropped}  latch_samples_confirmed={latch_confirmed}")
        print(f"Origin: candidate_seen={saw_candidate} t_cand={first_candidate_t}  commit_seen={saw_commit} t_commit={first_commit_t}")

        # v0.4.8: Action Gate summary
        if action_gate is not None:
            print("-"*70)
            print(f"ACTION GATE v{ActionGateV0_2.VERSION} (execution layer observability):")
            gate_state = action_gate.state.value
            print(f"  Final state:        {gate_state}")
            print(f"  Total transitions:  {action_gate.transition_count}")
            if gate_output:
                print(f"  Last decision:      {gate_output.decision.value}")
                print(f"  Last reason:        {gate_output.reason}")
                print(f"  Action allowed:     {gate_output.allowed}")

        print("="*70)
    return 0

if __name__ == "__main__": sys.exit(main())
