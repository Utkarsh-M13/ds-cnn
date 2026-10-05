"""Download the snowballlab/30-keywords dataset from HuggingFace and arrange it as
processed/<label>/*.wav (16 kHz mono) -- the layout src/train_tf.py expects -- so the QAT
fine-tune can run without Bisti's prebuilt mfcc_cache.npz.

Usage (on Anvil):
    pip install --user datasets soundfile numpy
    python fetch_dataset.py                      # writes ./processed/<label>/*.wav
    python fetch_dataset.py --limit-per-label 50 # small subset for a quick smoke test

Notes:
- This is a first-pass arranger and has not been run against the real dataset yet. It prints the
  dataset's splits, columns, and label names first, so if the schema differs we can adjust.
- The 30 label folders must be named exactly as the model's labels (underscores, e.g.
  volume_down, set_a_timer). The script replaces spaces with underscores to match.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="snowballlab/30-keywords")
    ap.add_argument("--out", default="processed")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--limit-per-label", type=int, default=0, help="0 = all clips")
    args = ap.parse_args()

    from datasets import load_dataset, Audio, concatenate_datasets

    ds = load_dataset(args.dataset)
    print("[info] splits:", list(ds.keys()))
    parts = [ds[s] for s in ds.keys()]
    data = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    print("[info] columns:", data.column_names)

    feats = data.features
    audio_col = next((c for c in data.column_names
                      if feats[c].__class__.__name__ == "Audio"), None)
    if audio_col is None and "audio" in data.column_names:
        audio_col = "audio"
    label_col = next((c for c in data.column_names
                      if c.lower() in ("label", "labels", "keyword", "word", "class")), None)
    if audio_col is None or label_col is None:
        raise SystemExit(f"[error] could not find audio/label columns in {data.column_names}")
    print(f"[info] audio column: {audio_col}   label column: {label_col}")

    names = getattr(feats[label_col], "names", None)  # ClassLabel -> list of names
    if names is not None:
        print(f"[info] {len(names)} labels:", names)

    def label_name(v):
        if names is not None and isinstance(v, int):
            return names[v]
        return str(v)

    data = data.cast_column(audio_col, Audio(sampling_rate=args.sr))

    out = Path(args.out)
    counts: dict[str, int] = {}
    for i, ex in enumerate(data):
        lbl = label_name(ex[label_col]).strip().replace(" ", "_")
        if args.limit_per_label and counts.get(lbl, 0) >= args.limit_per_label:
            continue
        a = ex[audio_col]
        wav = np.asarray(a["array"], dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        d = out / lbl
        d.mkdir(parents=True, exist_ok=True)
        sf.write(str(d / f"{counts.get(lbl, 0):05d}.wav"), wav, args.sr, subtype="PCM_16")
        counts[lbl] = counts.get(lbl, 0) + 1
        if (i + 1) % 2000 == 0:
            print(f"  wrote {i + 1} clips ...", flush=True)

    print("[done] per-label counts:")
    for k in sorted(counts):
        print(f"  {k:16s} {counts[k]}")
    print(f"[done] {sum(counts.values())} clips -> {out}/   ({len(counts)} labels)")


if __name__ == "__main__":
    main()
