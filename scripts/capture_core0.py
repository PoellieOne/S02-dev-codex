# capture_core0.py
import json, time, serial
from binlink import (
    FrameStream,
    TYPE_EVENT16, TYPE_EVENT24,
    TYPE_SUMMARY16, TYPE_SUMMARY24,
    TYPE_FILTER_STATS, TYPE_LINK_STATS,
    parse_event16, parse_event24,
    parse_summary16, parse_summary24,
    parse_filter_stats, parse_link_stats,
    decode_flags,
    BatchWriter
)

PORT = "/dev/ttyUSB0"
BAUD = 115200
#BAUD = 460800

def format_filter_stats(fs):
    """Format filter stats voor console output"""
    return (f"[FILTER] emitted={fs['events_emitted']:4d} "
            f"dropped={fs['events_dropped']:3d} ({fs['drop_rate_pct']:5.1f}%) "
            f"rejected={fs['events_rejected']:4d} ({fs['reject_rate_pct']:5.1f}%) "
            f"tokens={fs['tokens']:+5.1f} "
            f"strong={fs['pct_strong_normalized']:4.1f}% "
            f"normal={fs['pct_normal_normalized']:4.1f}% "
            f"weak={fs['pct_weak_normalized']:4.1f}%")

def format_link_stats(ls):
    """Format link stats voor console output"""
    return (f"[LINK]   frames={ls['frames_sent']:4d} "
            f"failed={ls['frames_failed']:2d} "
            f"throughput={ls['throughput_kbps']:6.1f}kb/s "
            f"latency={ls['avg_write_us']:4d}µs "
            f"queue={ls['queue_fill_normalized']:4.1f}% "
            f"(peak={ls['queue_high_water']:3d}) "
            f"blocked={ls['uart_blocked_count']:2d}")

def main():
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    fs  = FrameStream(ser)

    # Separate files voor verschillende data types
    evs = BatchWriter("core0_events.jsonl",       batch_size=256, flush_ms=400)
    flt = BatchWriter("core0_filter_stats.jsonl", batch_size=32,  flush_ms=800)
    lnk = BatchWriter("core0_link_stats.jsonl",   batch_size=32,  flush_ms=800)
    leg = BatchWriter("core0_legacy.jsonl",       batch_size=32,  flush_ms=800)  # Voor legacy summaries

    print(f"[capture] listening on {PORT} @ {BAUD} … (Ctrl+C to stop)")
    print(f"[capture] writing to:")
    print(f"  - events:       core0_events.jsonl")
    print(f"  - filter stats: core0_filter_stats.jsonl")
    print(f"  - link stats:   core0_link_stats.jsonl")
    print(f"  - legacy:       core0_legacy.jsonl")
    print()

    events_seen = 0
    filter_stats_seen = 0
    link_stats_seen = 0
    t0 = time.time()
    last_status = time.time()

    try:
        for t, v, payload in fs.read_frames():
            now = time.time()

            # === EVENT PACKETS ===
            if t == TYPE_EVENT24:
                ev = parse_event24(payload)
                ev |= decode_flags(ev["flags0"], ev["flags1"])
                evs.add(json.dumps(ev))
                events_seen += 1

            elif t == TYPE_EVENT16:
                ev = parse_event16(payload)
                ev |= decode_flags(ev["flags0"], ev["flags1"])
                evs.add(json.dumps(ev))
                events_seen += 1

            # === NEW: FILTER STATS ===
            elif t == TYPE_FILTER_STATS:
                fs_data = parse_filter_stats(payload)
                flt.add(json.dumps(fs_data))
                filter_stats_seen += 1
                print(format_filter_stats(fs_data))

            # === NEW: LINK STATS ===
            elif t == TYPE_LINK_STATS:
                ls_data = parse_link_stats(payload)
                lnk.add(json.dumps(ls_data))
                link_stats_seen += 1
                print(format_link_stats(ls_data))

            # === LEGACY SUMMARIES ===
            elif t == TYPE_SUMMARY24:
                sm = parse_summary24(payload)
                leg.add(json.dumps(sm))
                print(f"[LEGACY24] emitted={sm['emitted']}, dropped={sm['dropped']}")

            elif t == TYPE_SUMMARY16:
                sm = parse_summary16(payload)
                leg.add(json.dumps(sm))
                print(f"[LEGACY16] emitted={sm['emitted']}, dropped={sm['dropped']}")

            # === UNKNOWN PACKETS ===
            else:
                print(f"[debug] unknown frame type={t} ver={v} len={len(payload)}")

            # Status update elke 5 seconden
            if now - last_status > 5.0:
                elapsed = now - t0
                evps = events_seen / elapsed if elapsed > 0 else 0
                print(f"[status] events: {events_seen} ({evps:.1f}/s)  "
                      f"filter_stats: {filter_stats_seen}  "
                      f"link_stats: {link_stats_seen}  "
                      f"buffers: ev={len(evs.buf)} flt={len(flt.buf)} lnk={len(lnk.buf)}")
                last_status = now

    except KeyboardInterrupt:
        print("\n[capture] interrupted by user")
    except Exception as e:
        print(f"\n[capture] error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close all files
        evs.close()
        flt.close()
        lnk.close()
        leg.close()
        ser.close()

        # Final statistics
        elapsed = time.time() - t0
        print("\n" + "="*70)
        print("[capture] Session summary:")
        print(f"  Duration:      {elapsed:.1f}s")
        print(f"  Events:        {events_seen} ({events_seen/elapsed:.1f}/s)")
        print(f"  Filter stats:  {filter_stats_seen}")
        print(f"  Link stats:    {link_stats_seen}")
        print("="*70)
        print("[capture] closed.")

if __name__ == "__main__":
    main()
