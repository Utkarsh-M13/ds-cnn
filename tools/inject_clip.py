"""
Stream a 2 s audio clip into the Thingy:53 over the serial console and run it
through the on-device int8 pipeline, bypassing the (too quiet) microphone.

Pair with the firmware `inject` console command (serial-pcm-inject branch). The
clip can be recorded anywhere, e.g. in Audacity; this tool resamples it to
16 kHz mono and normalizes it to a loud reference level (-20 dBFS RMS), close to
the level the training clips naturally sit at, so it reaches the model loud rather
than at the mic's level. (Training itself does not loudness-normalize; the clips
are just naturally loud.) That host-side normalization is fix A applied on the
host, which is why injected clips do not suffer the quiet-mic problem.

Usage:
    # real run against the board
    python tools/inject_clip.py my_word.wav --port /dev/tty.usbmodemXXXX

    # dry run: just prepare the clip and save it, no serial (for host-side checks)
    python tools/inject_clip.py my_word.wav --out-pcm prepared.npy

Requires: numpy, scipy. Real (non dry-run) also needs pyserial.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE * 1          # 16000, must match firmware RAW_SAMPLE_COUNT (v2, 1 s clips)
TARGET_DBFS = -20.0                       # loud reference level (near the training clips' natural level)


def load_and_prepare(path, target_dbfs, normalize=True):
    """wav (any rate/channels) -> (32000,) int16 at 16 kHz, normalized to target."""
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.ndim > 1:                        # stereo -> mono
        x = x.mean(axis=1)
    # scipy returns int16 for PCM16 wavs; bring to [-1, 1] like soundfile/firmware
    if np.issubdtype(np.dtype(x.dtype), np.floating) is False:
        pass
    if np.max(np.abs(x)) > 1.5:           # integer PCM, scale down
        x = x / 32768.0
    if sr != SAMPLE_RATE:                 # resample to 16 kHz
        g = np.gcd(int(sr), SAMPLE_RATE)
        x = resample_poly(x, SAMPLE_RATE // g, sr // g)
    if normalize:
        rms = np.sqrt(np.mean(x ** 2)) + 1e-12
        cur_dbfs = 20.0 * np.log10(rms)
        gain = 10.0 ** ((target_dbfs - cur_dbfs) / 20.0)
        x = x * gain
    # pad / trim to exactly 2 s
    if len(x) < CLIP_SAMPLES:
        x = np.concatenate([x, np.zeros(CLIP_SAMPLES - len(x))])
    else:
        x = x[:CLIP_SAMPLES]
    pcm = np.round(x * 32768.0).clip(-32768, 32767).astype(np.int16)
    return pcm


def measured_dbfs(pcm):
    f = pcm.astype(np.float64) / 32768.0
    return 20.0 * np.log10(np.sqrt(np.mean(f ** 2)) + 1e-12)


def _ack_count(ack):
    """Pull the running sample count out of '[INJECT] ok N' or '[INJECT] FULL N'."""
    m = re.search(r"(?:ok|FULL)\s+(\d+)", ack)
    return int(m.group(1)) if m else None


def stream_to_device(pcm, port, baud, per_line, read_timeout):
    import serial  # pyserial, imported lazily so dry runs need no install
    checksum = int(np.sum(pcm.astype(np.int64)))   # firmware sums as a long
    capacity = len(pcm)
    ser = serial.Serial(port, baud, timeout=read_timeout)
    time.sleep(0.3)
    ser.reset_input_buffer()

    def wait_for(substr, limit=10.0):
        end = time.time() + limit
        while time.time() < end:
            ln = ser.readline().decode(errors="replace").strip()
            if ln:
                if "[INJECT]" in ln or "OUT," in ln:
                    print("  <", ln)
                if substr in ln:
                    return ln
        return None

    def cancel():
        # Always leave the device OUT of inject mode. A run that never fills the
        # buffer leaves the firmware treating the next typed line as PCM data
        # (the stuck-inject bug), so we injectcancel on every exit path.
        try:
            ser.write(b"\ninjectcancel\n")
            time.sleep(0.3)
            ser.reset_input_buffer()
        except Exception:
            pass

    # The device drops characters when a line arrives faster than it can take them
    # in (UART RX overrun while the console thread is busy), so a long line loses
    # most of its samples. We verify the device's running count after every line;
    # if it falls behind what we sent, we abort and retry the whole clip at a
    # smaller per_line. A mismatch is caught on the first bad line, so a failed
    # attempt costs ~1 line, not a full upload. Reliability over speed: a parity
    # clip can be slow. (Once the firmware pauses inference during inject, large
    # per_line values work and this falls through on the first try.)
    candidates = sorted({per_line, 64, 32, 16}, reverse=True)
    candidates = [c for c in candidates if c <= per_line] or [per_line]

    try:
        for pl in candidates:
            print(f"[host] inject attempt at {pl} samples/line ...")
            ser.write(b"\nstop\n")          # halt inference so the console isn't starved
            time.sleep(0.4)
            ser.write(b"injectcancel\n")    # clear any stuck inject mode, then start fresh
            time.sleep(0.4)
            ser.reset_input_buffer()
            ser.write(b"inject\n")
            if not wait_for("READY", 10.0):
                cancel()
                sys.exit("[host] device did not acknowledge 'inject' (no READY). Is the v2 firmware flashed?")

            t0 = time.time()
            expected = 0
            full_line = None
            dropped = False
            for i in range(0, capacity, pl):
                chunk = pcm[i:i + pl]
                expected = min(expected + len(chunk), capacity)
                ser.write((",".join(str(int(v)) for v in chunk) + "\n").encode())
                ack = wait_for("[INJECT]", 10.0)
                if ack is None:
                    print(f"[host] no ack after sample {i}; serial stalled.")
                    dropped = True
                    break
                if "FULL" in ack:
                    full_line = ack
                    break
                count = _ack_count(ack)
                if count is None or count != expected:
                    print(f"[host] device dropped samples (sent {expected}, device has {count}); "
                          f"retrying at a smaller line size.")
                    dropped = True
                    break
            dt = time.time() - t0

            if dropped:
                cancel()
                continue  # retry smaller

            if full_line is None:
                full_line = wait_for("checksum=", 10.0)
            if not (full_line and "checksum=" in full_line):
                print("[host] never got the FULL/checksum line; retrying smaller.")
                cancel()
                continue

            dev_sum = int(full_line.split("checksum=")[1].split()[0])
            ok = (dev_sum == checksum)
            print(f"[host] upload done in {dt:.1f}s ({pl}/line); "
                  f"checksum host={checksum} device={dev_sum} -> {'MATCH' if ok else 'MISMATCH'}")
            if not ok:
                print("[host] checksum mismatch; retrying at a smaller line size.")
                cancel()
                continue

            # success: the clip is loaded and the ML thread runs it. Collect OUT lines.
            outs = []
            end = time.time() + 30.0
            while time.time() < end:
                ln = ser.readline().decode(errors="replace").strip()
                if not ln:
                    continue
                if ln.startswith("OUT,") and "ERROR" not in ln:
                    parts = ln.split(",")
                    outs.append((parts[1], float(parts[3])))
                if "CAPTURE COMPLETE" in ln:
                    break
            return outs

        sys.exit("[host] inject kept dropping data down to the smallest line size. "
                 "The device is losing serial characters -- needs the firmware fix "
                 "(pause inference during inject) flashed first.")
    finally:
        cancel()
        ser.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", help="audio clip to inject (any sample rate / channels)")
    ap.add_argument("--port", help="serial port (e.g. /dev/tty.usbmodemXXXX). Omit for a dry run.")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dbfs", type=float, default=TARGET_DBFS, help="target loudness (default -20)")
    ap.add_argument("--no-normalize", action="store_true", help="skip loudness normalization")
    ap.add_argument("--per-line", type=int, default=200,
                    help="samples per serial line (firmware caps at 256 / "
                         "CONSOLE_INPUT_MAX_LINE_LEN=2048; lower it if a line gets truncated)")
    ap.add_argument("--read-timeout", type=float, default=1.0)
    ap.add_argument("--out-pcm", help="save the prepared int16 clip to this .npy and exit if no --port")
    args = ap.parse_args()

    pcm = load_and_prepare(args.wav, args.dbfs, normalize=not args.no_normalize)
    checksum = int(np.sum(pcm.astype(np.int64)))
    print(f"[host] prepared {len(pcm)} samples @ {SAMPLE_RATE} Hz, "
          f"{measured_dbfs(pcm):+.1f} dBFS, checksum={checksum}")

    if args.out_pcm:
        np.save(args.out_pcm, pcm)
        print(f"[host] saved prepared clip -> {args.out_pcm}")

    if not args.port:
        print("[host] no --port given: dry run only, nothing sent.")
        return

    outs = stream_to_device(pcm, args.port, args.baud, args.per_line, args.read_timeout)
    if outs:
        outs.sort(key=lambda t: t[1], reverse=True)
        print("\n[host] device top-5:")
        for label, pct in outs[:5]:
            print(f"  {label:<14} {pct:6.1f}%")


if __name__ == "__main__":
    main()
