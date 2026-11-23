#!/usr/bin/env python3
"""Score int8 models on the test split with background noise at fixed SNRs.

For each run directory (a QAT run: kws_tf.json + its int8 tflite), this scores
the real int8 tflite file on the test clips (split from file names, test_*):
clean, then mixed with noise at each SNR. It uses train_tf's exact pipeline:
waveform loading, mix_at_snr, the configured MFCC backend, and the run's own
feature_norm and input quantization.

Every model gets the same noisy clips: per clip, one noise file and offset are
picked from a fixed seed. Only the level changes between SNR columns.

Held-out noise: training mixes in every wav in its NOISE_DIR. Scoring with the
same files is optimistic. Use --noise-files to pick files the model did not
train on, or point --noise-dir at a different noise set.

Usage:
    python tools/eval_noisy.py \
        --runs runs/acc_k5_cos200_noise_qat runs/acc_k5_cos400_qat \
        --noise-dir downloads/speech_commands_v0.02/_background_noise_ \
        --snrs 20 10 5 0 --csv noisy_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import PROCESSED_DIR, load_config  # noqa: E402
from mfcc import backend_name, make_frontend  # noqa: E402
from mfcc_tf import MFCCParams  # noqa: E402
from train_tf import (_list_clips, _load_waveform, _split_from_name,  # noqa: E402
                      mix_at_snr)
from eval_heldout import run_tflite  # noqa: E402


def load_run(run_dir: Path) -> dict:
    meta_path = run_dir / "kws_tf.json"
    if not meta_path.exists():
        raise SystemExit(f"[error] {meta_path} not found. Pass a run directory "
                         "that train_tf.py finished (use the *_qat run).")
    meta = json.loads(meta_path.read_text())
    tfl = run_dir / meta.get("int8_tflite", "kws_tf_int8.tflite")
    if not tfl.exists():
        raise SystemExit(f"[error] int8 model not found: {tfl}")
    return {"name": run_dir.name, "meta": meta, "tflite": tfl}


def feature_key(meta: dict) -> tuple:
    m = meta["mfcc"]
    return (int(m["sample_rate"]), float(m["frame_stride_ms"]), int(m["num_mfcc"]),
            int(m["num_frames"]), str(meta["dsp"]["backend"]), tuple(meta["labels"]))


def noise_segments(bank: list[np.ndarray], n_clips: int, n: int,
                   seed: int) -> list[np.ndarray]:
    """One fixed noise slice per clip, so all SNRs and models share it."""
    segs = []
    for i in range(n_clips):
        rng = np.random.default_rng([seed, i])
        noise = bank[int(rng.integers(len(bank)))]
        if len(noise) > n:
            start = int(rng.integers(0, len(noise) - n + 1))
            segs.append(noise[start:start + n])
        else:
            segs.append(np.tile(noise, int(np.ceil(n / len(noise))))[:n])
    return segs


def load_bank(noise_dir: Path, names: list[str] | None, sr: int) -> tuple[list, list]:
    import soundfile as sf
    if not noise_dir.is_dir():
        raise SystemExit(f"[error] noise dir not found: {noise_dir}")
    files = sorted(noise_dir.glob("*.wav"))
    if names:
        missing = [n for n in names if not (noise_dir / n).exists()]
        if missing:
            raise SystemExit(f"[error] not in {noise_dir}: {', '.join(missing)}")
        files = [noise_dir / n for n in names]
    if not files:
        raise SystemExit(f"[error] no .wav noise files in {noise_dir}")
    bank = []
    for f in files:
        y, fsr = sf.read(str(f), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if fsr != sr:
            n_new = int(round(len(y) * sr / fsr))
            y = np.interp(np.linspace(0, len(y), n_new, endpoint=False),
                          np.arange(len(y)), y).astype(np.float32)
        bank.append(y.astype(np.float32))
    return bank, [f.name for f in files]


def features(frontend, wavs: list[np.ndarray], batch: int = 256) -> np.ndarray:
    out = [frontend(np.stack(wavs[i:i + batch])) for i in range(0, len(wavs), batch)]
    return np.concatenate(out).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run directories with kws_tf.json and the int8 tflite.")
    ap.add_argument("--noise-dir", required=True)
    ap.add_argument("--noise-files", nargs="+", default=None,
                    help="Use only these wav names from --noise-dir (held-out noise).")
    ap.add_argument("--snrs", nargs="+", type=float, default=[20, 10, 5, 0],
                    help="SNR levels in dB (default 20 10 5 0). Lower = louder noise.")
    ap.add_argument("--seed", type=int, default=1234,
                    help="Picks the noise file and offset per clip (default 1234).")
    ap.add_argument("--data-dir", default=str(PROCESSED_DIR))
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only the first N test clips (quick check).")
    ap.add_argument("--csv", default=None, help="Also write the table to this CSV.")
    args = ap.parse_args()

    runs = [load_run(Path(r)) for r in args.runs]
    root = Path(args.data_dir)
    cols = ["clean"] + [f"{s:g}dB" for s in args.snrs]
    results: dict[str, dict[str, float]] = {}
    noise_used: list[str] = []

    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault(feature_key(r["meta"]), []).append(r)

    for key, members in groups.items():
        sr, stride, n_mfcc, n_frames, backend, labels = key
        cfg = load_config()
        cfg["audio"]["sample_rate"] = sr
        cfg["mfcc_reference"]["frame_stride_ms"] = stride
        if backend_name(cfg) != backend:
            raise SystemExit(f"[error] config.yaml backend is {backend_name(cfg)}, "
                             f"but {members[0]['name']} was trained with {backend}.")
        p = MFCCParams(cfg, sample_rate=sr, frame_stride_ms=stride, num_mfcc=n_mfcc)
        p, frontend = make_frontend(cfg, p=p)
        if p.num_frames != n_frames:
            raise SystemExit(f"[error] frame count mismatch: {p.num_frames} vs {n_frames}")

        files, y = _list_clips(root, list(labels))
        if not files:
            raise SystemExit(f"[error] no clips for these labels in {root}")
        split = [_split_from_name(f) for f in files]
        if any(s is None for s in split):
            raise SystemExit("[error] clip names have no train_/val_/test_ prefix; "
                             "cannot find the test split.")
        test = [i for i, s in enumerate(split) if s == 2]
        if args.limit:
            test = test[:args.limit]
        y_test = y[test]
        print(f"[data] {len(test)} test clips, {len(labels)} classes, "
              f"sr={sr} stride={stride:g}ms mfcc={n_mfcc} backend={backend}")

        wavs = [_load_waveform(files[i], p.n_samples, tgt_sr=sr) for i in test]
        bank, noise_used = load_bank(Path(args.noise_dir), args.noise_files, sr)
        segs = noise_segments(bank, len(wavs), p.n_samples, args.seed)
        print(f"[noise] {len(bank)} files: {', '.join(noise_used)}")

        feats = {"clean": features(frontend, wavs)}
        for snr, col in zip(args.snrs, cols[1:]):
            print(f"[mfcc] {col} ...", flush=True)
            feats[col] = features(frontend, [mix_at_snr(w, s, snr)
                                             for w, s in zip(wavs, segs)])

        for r in members:
            norm = r["meta"]["feature_norm"]
            results[r["name"]] = {}
            for col in cols:
                Xs = ((feats[col] - norm["mean"]) / norm["std"])[..., None]
                preds, _ = run_tflite(str(r["tflite"]), Xs.astype(np.float32))
                results[r["name"]][col] = float((preds == y_test).mean())
            results[r["name"]]["keras"] = float(r["meta"].get("test_acc", float("nan")))

    print("\nint8 tflite accuracy (%) on the test split. 'keras' = the run's own"
          "\nreported test accuracy (fake-quant model), for comparison with 'clean'.")
    w = max(len(n) for n in results) + 2
    print("model".ljust(w) + "".join(c.rjust(9) for c in ["keras"] + cols))
    for name, res in results.items():
        print(name.ljust(w) + "".join(f"{res[c] * 100:9.2f}" for c in ["keras"] + cols))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["model", "keras"] + cols + ["noise_files", "seed"])
            for name, res in results.items():
                wr.writerow([name] + [f"{res[c] * 100:.2f}" for c in ["keras"] + cols]
                            + [";".join(noise_used), args.seed])
        print(f"[csv] wrote {args.csv}")


if __name__ == "__main__":
    main()
