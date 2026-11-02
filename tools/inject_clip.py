"""
Stream a 2 s audio clip into the Thingy:53 over the serial console and run it
through the on-device int8 pipeline, bypassing the (too quiet) microphone.

Pair with the firmware `inject` console command (serial-pcm-inject branch). The
clip can be recorded anywhere, e.g. in Audacity; this tool resamples it to
16 kHz mono and normalizes it to the training loudness (-20 dBFS RMS) so it
reaches the model at the level it was trained on. That normalization is fix A
applied on the host, which is why injected clips do not suffer the quiet-mic
problem.

Usage:
    # real run against the board
    python tools/inject_clip.py my_word.wav --port /dev/tty.usbmodemXXXX

    # dry run: just prepare the clip and save it, no serial (for host-side checks)
    python tools/inject_clip.py my_word.wav --out-pcm prepared.npy

Requires: numpy, scipy. Real (non dry-run) also needs pyserial.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE * 2          # 32000, must match firmware RAW_SAMPLE_COUNT
TARGET_DBFS = -20.0                       # training loudness


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


def stream_to_device(pcm, port, baud, per_line, read_timeout):
    import serial  # pyserial, imported lazily so dry runs need no install
    checksum = int(np.sum(pcm.astype(np.int64)))   # firmware sums as a long
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

    print(f"[host] entering inject mode on {port} @ {baud}")
    ser.write(b"inject\n")
    if not wait_for("READY", 10.0):
        ser.close()
        sys.exit("[host] device did not acknowledge 'inject' (no READY). Is the firmware flashed and inference not spamming? Try 'stop' first.")

    print(f"[host] streaming {len(pcm)} samples, {per_line}/line ...")
    t0 = time.time()
    for i in range(0, len(pcm), per_line):
        chunk = pcm[i:i + per_line]
        line = ",".join(str(int(v)) for v in chunk) + "\n"
        ser.write(line.encode())
        # wait for the per-line ack so we never outrun the device's RX buffer
        ack = wait_for("[INJECT]", 10.0)
        if ack and "FULL" in ack:
            break
        if ack is None:
            ser.close()
            sys.exit(f"[host] no ack after sample {i}; aborting (serial stalled).")
    dt = time.time() - t0
    print(f"[host] upload done in {dt:.1f}s")

    # device prints "[INJECT] FULL N samples checksum=X" then runs the clip
    full = wait_for("checksum=", 10.0)
    if full:
        dev_sum = int(full.split("checksum=")[1].split()[0])
        ok = (dev_sum == checksum)
        print(f"[host] checksum host={checksum} device={dev_sum} -> {'MATCH' if ok else 'MISMATCH (data corrupted in transit)'}")
        if not ok:
            print("[host] WARNING: predictions below are not trustworthy.")

    # collect OUT lines until capture completes
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
    ser.close()
    return outs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", help="audio clip to inject (any sample rate / channels)")
    ap.add_argument("--port", help="serial port (e.g. /dev/tty.usbmodemXXXX). Omit for a dry run.")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dbfs", type=float, default=TARGET_DBFS, help="target loudness (default -20)")
    ap.add_argument("--no-normalize", action="store_true", help="skip loudness normalization")
    ap.add_argument("--per-line", type=int, default=16, help="samples per serial line (keep small)")
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
