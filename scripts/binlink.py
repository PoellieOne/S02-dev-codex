# binlink.py
import struct, time, io

SYNC = 0xA5
# TYPE upper nibble, VER lower nibble
TYPE_EVENT16      = 0x0
TYPE_EVENT24      = 0x1
TYPE_SUMMARY16    = 0x2  # Legacy
TYPE_SUMMARY24    = 0x3  # Legacy
TYPE_FILTER_STATS = 0x4  # NEW: Filter layer statistics
TYPE_LINK_STATS   = 0x5  # NEW: Transport layer statistics

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
        # generator: yields (type, ver, payload:bytes)
        while True:
            chunk = self.ser.read(256)  # tune me
            if not chunk:
                yield from ()
            self.buf.extend(chunk)
            # parse
            while True:
                # zoeken naar SYNC
                idx = self.buf.find(bytes([SYNC]))
                if idx < 0:
                    self.buf.clear()
                    break
                if idx > 0:
                    del self.buf[:idx]
                if len(self.buf) < 4:
                    break
                typever = self.buf[1]
                plen    = self.buf[2]
                need = 1 + 1 + 1 + plen + 2
                if len(self.buf) < need:
                    break
                frame = bytes(self.buf[:need])
                del self.buf[:need]
                # CRC check over TYPE|VER + LEN + PAYLOAD
                crc_rx = struct.unpack('<H', frame[-2:])[0]
                crc_tx = crc16_ccitt_false(frame[1:3+plen])  # typever+len+payload
                if crc_rx != crc_tx:
                    continue
                t = (typever >> 4) & 0x0F
                v = typever & 0x0F
                payload = frame[3:-2]
                yield (t, v, payload)

def parse_event16(p):
    dt_us, = struct.unpack_from('<H', p, 0)
    flags0 = p[2]; flags1 = p[3]
    dvdt_q15, = struct.unpack_from('<h', p, 4)
    mono_q8 = p[6]; snr_q8 = p[7]; score_q8 = p[8]; seq = p[9]
    return {
        "kind":"event16",
        "dt_us":dt_us, "flags0":flags0, "flags1":flags1,
        "dvdt_q15":dvdt_q15, "mono_q8":mono_q8, "snr_q8":snr_q8,
        "score_q8":score_q8, "seq":seq,
    }

def parse_event24(p):
    dt_us,  = struct.unpack_from('<H', p, 0)
    tabs,   = struct.unpack_from('<I', p, 2)
    flags0  = p[6]; flags1 = p[7]
    dvdt_q15, = struct.unpack_from('<h', p, 8)
    mono_q8 = p[10]; snr_q8 = p[11]; fit_err_q8 = p[12]
    rpm_hint_q, = struct.unpack_from('<H', p, 13)
    score_q8 = p[15]; seq = p[16]
    return {
        "kind":"event24",
        "dt_us":dt_us, "t_abs_us":tabs,
        "flags0":flags0, "flags1":flags1,
        "dvdt_q15":dvdt_q15, "mono_q8":mono_q8, "snr_q8":snr_q8,
        "fit_err_q8":fit_err_q8, "rpm_hint_q":rpm_hint_q,
        "score_q8":score_q8, "seq":seq
    }

def parse_summary16(p):
    (win_ms, ev_em, ev_dr, lvl_s, lvl_n, lvl_w, txb) = struct.unpack('<HHHBBBH', p)
    return {"kind":"summary16","window_ms":win_ms,"emitted":ev_em,"dropped":ev_dr,
            "lvl_strong":lvl_s,"lvl_normal":lvl_n,"lvl_weak":lvl_w,"tx_bytes":txb}

def parse_summary24(p):
    # 19 bytes: HHH (6) + 13×B (13) = 19 bytes
    fields = struct.unpack('<HHHBBBBBBBBBBBBB', p + b'\x00'* (19-len(p)))

    return {"kind":"summary24",
            "window_ms":fields[0],"emitted":fields[1],"dropped":fields[2],
            "lvl_strong":fields[3],"lvl_normal":fields[4],"lvl_weak":fields[5],
            "t_cross_rms_us_q":fields[6],"ab_skew_p95_us_q":fields[7],
            "queue_depth_max":fields[8],"utilization_q":fields[9],
            "rej_lowdvdt_u8":fields[10],"rej_nonmono_u8":fields[11],
            "rej_lowsnr_u8":fields[12],
            # fields[13], fields[14], fields[15] zijn de 3 gereserveerde bytes.
            }

def parse_filter_stats(p):
    """
    PKT_FILTER_STATS (0x4) - Filter layer statistieken (19 bytes)

    Payload layout:
      [0-1]   window_ms         - Measurement window in milliseconds
      [2-3]   events_emitted    - Events successfully emitted
      [4-5]   events_dropped    - Events dropped by backpressure
      [6-7]   events_considered - Total candidates offered (low 16 bits)
      [8-9]   events_rejected   - Candidates rejected by quality gates
      [10]    pct_strong        - Percentage strong quality (0-255 = 0-100%)
      [11]    pct_normal        - Percentage normal quality
      [12]    pct_weak          - Percentage weak quality
      [13]    drops_strong      - Count of strong events dropped
      [14]    drops_normal      - Count of normal events dropped
      [15]    drops_weak        - Count of weak events dropped
      [16]    tokens_q8         - Current token bucket level (signed int8, scale×10)
      [17]    coalesce_win_ms   - Current coalescing window size
      [18]    reserved          - Future use
    """
    if len(p) < 19:
        p = p + b'\x00' * (19 - len(p))

    fields = struct.unpack('<HHHHHBBBBBBBBB', p)

    # Decode tokens (signed int8, scaled by 10)
    tokens_raw = fields[11]
    if tokens_raw > 127:  # Handle as signed
        tokens_raw = tokens_raw - 256
    tokens = tokens_raw / 10.0

    return {
        "kind": "filter_stats",
        "window_ms": fields[0],
        "events_emitted": fields[1],
        "events_dropped": fields[2],
        "events_considered": fields[3],
        "events_rejected": fields[4],
        "pct_strong": fields[5],      # 0-255 scale
        "pct_normal": fields[6],      # 0-255 scale
        "pct_weak": fields[7],        # 0-255 scale
        "drops_strong": fields[8],
        "drops_normal": fields[9],
        "drops_weak": fields[10],
        "tokens": tokens,             # Float: -12.8 to +12.7
        "coalesce_win_ms": fields[12],
        # Derived metrics for convenience
        "total_drops": fields[2],
        "drop_rate_pct": (fields[2] * 100.0 / fields[1]) if fields[1] > 0 else 0.0,
        "reject_rate_pct": (fields[4] * 100.0 / fields[3]) if fields[3] > 0 else 0.0,
        # Convert 0-255 scale to 0-100%
        "pct_strong_normalized": fields[5] * 100.0 / 255.0,
        "pct_normal_normalized": fields[6] * 100.0 / 255.0,
        "pct_weak_normalized": fields[7] * 100.0 / 255.0,
    }

def parse_link_stats(p):
    """
    PKT_LINK_STATS (0x5) - Transport layer statistieken (19 bytes)

    Payload layout:
      [0-1]   window_ms          - Measurement window in milliseconds
      [2-3]   frames_sent        - UART frames successfully written
      [4-5]   frames_failed      - UART write failures
      [6-9]   bytes_sent         - Total bytes transmitted (32-bit)
      [10-11] avg_write_us       - Average UART write latency in microseconds
      [12]    uart_blocked_count - Number of partial writes
      [13]    queue_fill_pct     - TX queue fill percentage (0-255 = 0-100%)
      [14-15] queue_high_water   - Maximum queue depth seen
      [16]    event16_count      - Number of EVENT16 packets sent
      [17]    event24_count      - Number of EVENT24 packets sent
      [18]    reserved           - Future use
    """
    if len(p) < 19:
        p = p + b'\x00' * (19 - len(p))

    fields = struct.unpack('<HHHIHBBHBBB', p)

    return {
        "kind": "link_stats",
        "window_ms": fields[0],
        "frames_sent": fields[1],
        "frames_failed": fields[2],
        "bytes_sent": fields[3],
        "avg_write_us": fields[4],
        "uart_blocked_count": fields[5],
        "queue_fill_pct": fields[6],     # 0-255 scale
        "queue_high_water": fields[7],
        "event16_count": fields[8],
        "event24_count": fields[9],
        # Derived metrics for convenience
        "throughput_bps": (fields[3] * 8000.0 / fields[0]) if fields[0] > 0 else 0.0,
        "throughput_kbps": (fields[3] * 8.0 / fields[0]) if fields[0] > 0 else 0.0,
        "frame_success_rate": (fields[1] * 100.0 / (fields[1] + fields[2])) if (fields[1] + fields[2]) > 0 else 100.0,
        "frames_per_sec": (fields[1] * 1000.0 / fields[0]) if fields[0] > 0 else 0.0,
        # Convert 0-255 scale to 0-100%
        "queue_fill_normalized": fields[6] * 100.0 / 255.0,
    }

def decode_flags(flags0, flags1):
    pair      = (flags0 >> 7) & 1
    qlevel    = (flags0 >> 5) & 0x3
    polarity  = (flags0 >> 4) & 1
    sensor    = (flags0 >> 3) & 1
    from_pool = (flags1 >> 6) & 0x3
    to_pool   = (flags1 >> 4) & 0x3
    dir_hint  = (flags1 >> 2) & 0x3
    edge_kind = (flags1 >> 0) & 0x3
    return dict(pair=pair,qlevel=qlevel,polarity=polarity,sensor=sensor,
                from_pool=from_pool,to_pool=to_pool,dir_hint=dir_hint,edge_kind=edge_kind)

# --- Batch writer: schrijft in batches i.p.v. per regel
class BatchWriter:
    def __init__(self, path, batch_size=256, flush_ms=500):
        self.path = path
        self.batch_size = batch_size
        self.flush_ms = flush_ms
        self.buf = []
        self.last_flush = time.time()
        self.f = open(path, "a", buffering=1024*1024)  # grote buffer

    def add(self, line: str):
        self.buf.append(line)
        now = time.time()
        if len(self.buf) >= self.batch_size or (now - self.last_flush) * 1000.0 >= self.flush_ms:
            self.flush(now)

    def flush(self, now=None):
        if not self.buf: return
        self.f.write("\n".join(self.buf) + "\n")
        self.buf.clear()
        self.last_flush = now if now else time.time()

    def close(self):
        self.flush()
        self.f.close()
