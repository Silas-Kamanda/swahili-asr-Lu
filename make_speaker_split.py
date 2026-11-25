#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the four-way evaluation split for the speaker-familiarity study.

THE DESIGN
----------
One trained model, three test sets that differ ONLY in how familiar the speakers are:

    train        75%   clips from training speakers
    val           5%   held-out clips, training speakers   (model selection only)
    seen_test    10%   held-out clips, TRAINING speakers    <- what a clip-level split
                                                               actually measures
    unseen_test  10%   every clip from held-out speakers    <- speaker-disjoint

plus FLEURS as a fourth, out-of-corpus condition scored separately.

Because the model is held constant, the difference between seen_test and unseen_test
is attributable to speaker familiarity and nothing else. Training two models on two
different splits would confound that with ordinary run-to-run variation.

TWO PROPERTIES THAT ARE DELIBERATE, NOT BUGS
--------------------------------------------
1. `seen_test` is drawn proportionally from every training speaker, so it inherits the
   corpus's skew — the top speaker supplies ~13% of it. That is exactly what a
   clip-level split does, and reproducing it faithfully is the point: seen_test IS the
   conventional protocol, measured honestly.

2. `unseen_test` speakers are capped so no single voice can dominate, which means they
   are drawn from lower-volume contributors. On this corpus 10 of 111 speakers hold 61%
   of the clips, so any uncapped draw risks a test set that measures one person. The
   consequence — held-out speakers are systematically the less prolific ones — is forced
   by the corpus and belongs in the limitations, not hidden.

Usage:
    python make_speaker_split.py --csv swahili_with_speakers.csv --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd


def build_split(df: pd.DataFrame, seed: int = 42,
                val_frac: float = 0.05, seen_frac: float = 0.10,
                unseen_frac: float = 0.10,
                max_speaker_share: float = 0.15) -> pd.DataFrame:
    counts = df["client_id"].value_counts()
    total = int(counts.sum())
    unseen_target = int(round(total * unseen_frac))

    # --- choose held-out speakers -----------------------------------------------
    # Cap by share of the EVALUATION SPLIT, not of the corpus: a 400-clip speaker is
    # only 4.4% of a 9000-clip corpus but would be 44% of a 900-clip test set.
    cap = unseen_target * max_speaker_share
    eligible = [s for s, c in counts.items() if c <= cap]
    if not eligible:
        raise SystemExit(
            f"No speaker holds {int(cap)} clips or fewer — a speaker-disjoint test set "
            f"cannot be built without one voice dominating it. Raise "
            f"--max-speaker-share and say so in the paper, or drop this condition."
        )

    rng = np.random.default_rng(seed)
    order = list(rng.permutation(eligible))
    held_out, filled = [], 0
    for spk in order:
        if filled >= unseen_target:
            break
        held_out.append(spk)
        filled += int(counts[spk])

    assign = pd.Series("train", index=df.index, dtype=object)
    is_held = df["client_id"].isin(held_out)
    assign[is_held] = "unseen_test"

    # --- carve val and seen_test out of the TRAINING speakers --------------------
    # Stratified per speaker, so both inherit the training speaker distribution and
    # seen_test faithfully reproduces what a clip-level split would have produced.
    train_rows = df[~is_held]
    train_clips = len(train_rows)
    val_within = (total * val_frac) / train_clips
    seen_within = (total * seen_frac) / train_clips

    for speaker, rows in train_rows.groupby("client_id"):
        idx = rng.permutation(rows.index.to_numpy())
        n = len(idx)
        n_val = int(round(n * val_within))
        n_seen = int(round(n * seen_within))
        if n_val + n_seen >= n:            # tiny speakers stay wholly in train
            continue
        assign[idx[:n_val]] = "val"
        assign[idx[n_val:n_val + n_seen]] = "seen_test"

    out = df.copy()
    out["split"] = assign.values
    print(f"{len(eligible)}/{len(counts)} speakers small enough to hold out "
          f"(<= {int(cap)} clips); {len(held_out)} actually held out")
    return out


def report(out: pd.DataFrame) -> Dict:
    stats: Dict = {}
    total = len(out)
    print(f"\n{'split':<13}{'clips':>8}{'%':>8}{'speakers':>11}{'top speaker':>14}")
    print("-" * 54)
    for name in ("train", "val", "seen_test", "unseen_test"):
        part = out[out["split"] == name]
        if part.empty:
            continue
        vc = part["client_id"].value_counts()
        share = 100 * vc.iloc[0] / len(part)
        print(f"{name:<13}{len(part):>8}{100*len(part)/total:>7.1f}%"
              f"{len(vc):>11}{share:>13.1f}%")
        stats[name] = {"clips": int(len(part)), "speakers": int(len(vc)),
                       "top_speaker_pct": round(float(share), 1)}
    print("-" * 54)

    train_spk = set(out[out["split"].isin(["train", "val", "seen_test"])]["client_id"])
    unseen_spk = set(out[out["split"] == "unseen_test"]["client_id"])
    leak = train_spk & unseen_spk
    seen_spk = set(out[out["split"] == "seen_test"]["client_id"])
    ok = not leak and seen_spk.issubset(train_spk)
    print("unseen_test speakers disjoint from training:",
          "VERIFIED" if not leak else f"FAILED ({len(leak)} leaked)")
    print("seen_test speakers all appear in training:  ",
          "VERIFIED" if seen_spk.issubset(train_spk) else "FAILED")
    stats["valid"] = bool(ok)
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--csv", default="/content/drive/MyDrive/Colab/swahili_with_speakers.csv")
    p.add_argument("--out", default=None,
                   help="Defaults to splits_seed<N>.csv beside the input.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seen-frac", type=float, default=0.10)
    p.add_argument("--unseen-frac", type=float, default=0.10)
    p.add_argument("--max-speaker-share", type=float, default=0.15,
                   help="Cap on one speaker's share of the held-out-speaker test set.")
    args = p.parse_args(argv if argv is not None else
                        ([] if "ipykernel" in sys.modules else sys.argv[1:]))

    df = pd.read_csv(args.csv)
    missing = [c for c in ("path", "transcript", "client_id") if c not in df.columns]
    if missing:
        raise SystemExit(f"{args.csv} is missing {missing}")
    df = df[df["client_id"].notna()].reset_index(drop=True)

    out = build_split(df, args.seed, args.val_frac, args.seen_frac,
                      args.unseen_frac, args.max_speaker_share)
    stats = report(out)
    if not stats.get("valid"):
        raise SystemExit("Split failed its own integrity checks; not written.")

    path = args.out or os.path.join(os.path.dirname(args.csv) or ".",
                                    f"splits_seed{args.seed}.csv")
    out.to_csv(path, index=False)
    with open(os.path.splitext(path)[0] + "_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\nwritten -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
