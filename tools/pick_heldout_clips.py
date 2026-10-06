#!/usr/bin/env python3
"""Tier-2 prep: copy out HELD-OUT raw wavs (train_tf's test split) for the device test.

The oracle before/after used TRAINING clips, and the two headline words saturated the
int8 output (both chip and PC hit 127, so a "match" proves nothing). This pulls clips
the QAT model never trained on, including mid-confidence words (search, five) that do
NOT saturate, so the device-vs-PC gap is actually visible.

It reproduces train_tf's exact clip-level split (same labels order, same seed, same
0.15/0.15 fractions) and copies, for each requested word, the first few clips that land
in the TEST split. scp the output folder to the Mac, then bake them into the oracle.

Run on Anvil in ~/ds-cnn with qatenv:
    python tools/pick_heldout_clips.py --out heldout_clips \
        --words set_a_timer volume_down yes search five --per-word 1
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from config import load_config, PROCESSED_DIR


def split_indices(n: int, seed: int, val_frac=0.15, test_frac=0.15):
    # identical to src/train_tf.py:split_indices
    idx = np.random.default_rng(seed).permutation(n)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    return (idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(PROCESSED_DIR))
    ap.add_argument("--out", default="heldout_clips")
    ap.add_argument("--words", nargs="+",
                    default=["set_a_timer", "volume_down", "yes", "search", "five"])
    ap.add_argument("--per-word", type=int, default=1)
    args = ap.parse_args()

    root = Path(args.root)
    seed = load_config()["sampling"]["seed"]

    # same ordering train_tf used: sorted label dirs, then sorted wavs per label
    labels = sorted(d.name for d in root.iterdir() if d.is_dir())
    files, ys = [], []
    for i, label in enumerate(labels):
        for wav in sorted((root / label).glob("*.wav")):
            files.append(wav); ys.append(i)
    ys = np.array(ys)
    n = len(files)
    _, _, test_idx = split_indices(n, seed)
    test_set = set(int(i) for i in test_idx)
    print(f"[split] n={n} clips, test split has {len(test_set)} clips (seed={seed})")

    out = Path(args.out); out.mkdir(exist_ok=True)
    picked = []
    for w in args.words:
        if w not in labels:
            print(f"[warn] label '{w}' not found; skipping"); continue
        c = labels.index(w)
        # global indices of this word that are in the TEST split
        word_test = [i for i in range(n) if ys[i] == c and i in test_set]
        if not word_test:
            print(f"[warn] no TEST-split clips for '{w}'"); continue
        for k, gi in enumerate(word_test[:args.per_word]):
            dst = out / f"{w}_{k}.wav"
            shutil.copy(files[gi], dst)
            picked.append((w, files[gi].name, dst.name))
            print(f"[pick] {w}: {files[gi].name}  (held-out) -> {dst}")

    print(f"\n[done] {len(picked)} clips in {out}/  (all from the TEST split, never trained on)")
    print("Next: scp this folder to the Mac, e.g.")
    print(f"  scp -r x-umajithia@anvil.rcac.purdue.edu:ds-cnn/{args.out} "
          f"/Users/utkarsh_m/Work/Snowball-Labs/ds-cnn/")


if __name__ == "__main__":
    main()
