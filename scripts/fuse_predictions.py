#!/usr/bin/env python
"""
fuse_predictions.py -- prediction-level ("decision") fusion of two arms.

Embedding-level fusion (concatenate the two embeddings, fit one model) is what the
"combined" arm does. This script does the other kind: fit one model per arm, then
combine their PREDICTIONS. Both are late merges, but they can behave differently.

Input: the out-of-fold predictions that `evaluate_embeddings.py --cv-frozen-encoder K
--save-predictions` writes for each arm (<root>/<name>/cv_oof_predictions.csv).
Every prediction comes from a model that never saw that slide's patient, so the
fused predictions are also out-of-fold.

Two fusions are produced, per factor, with weight w on arm B and (1 - w) on arm A:

  avg    w = 0.5 for every factor (no tuning at all)
  stack  w chosen from a grid by mean per-slide Pearson on the slides of the OTHER
         folds only, then applied to the held-out fold (so no slide chooses its own weight)

Output folders (readable by compare_cv.py):
  <root>/<out-name>_avg/cv_per_sample.csv
  <root>/<out-name>_stack/cv_per_sample.csv
and a printout with the mean per-slide Pearson of A, B, avg and stack, plus the
weights stacking chose.

    python scripts/fuse_predictions.py --a omiclip_image_oof --b omiclip_st_oof \
        --out-name omiclip_predfuse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
GRID = np.linspace(0.0, 1.0, 11)


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


def load_arm(root, name):
    df = pd.read_csv(root / name / "cv_oof_predictions.csv")
    meta = pd.read_csv(root / name / "cv_per_sample.csv")
    cols = ["sample_id", "patient_id", "fold"] + (["technology"] if "technology" in meta.columns else [])
    info = meta[cols].drop_duplicates("sample_id").set_index("sample_id")
    factors = [c[len("pred_"):] for c in df.columns if c.startswith("pred_")]
    return df, info, factors


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=PROJECT_DIR / "outputs" / "eval")
    p.add_argument("--a", required=True, help="arm A (weight 1-w), e.g. the H&E arm")
    p.add_argument("--b", required=True, help="arm B (weight w), e.g. the ST arm")
    p.add_argument("--out-name", required=True)
    args = p.parse_args()

    da, info_a, factors = load_arm(args.root, args.a)
    db, info_b, factors_b = load_arm(args.root, args.b)
    if factors != factors_b:
        raise SystemExit(f"Factor lists differ: {factors} vs {factors_b}")

    keys = ["sample_id", "barcode_key"]
    m = da.merge(db, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    print(f"Spots predicted by both arms: {len(m):,} (A {len(da):,}, B {len(db):,})")
    if len(m) != len(da) or len(m) != len(db):
        raise SystemExit("The two arms do not cover the same spots; re-run both with the same --restrict-to.")
    for f in factors:
        if np.abs(m[f"y_{f}_a"] - m[f"y_{f}_b"]).max() > 1e-4:
            raise SystemExit(f"Targets for {f} differ between the two arms.")

    slides = sorted(m["sample_id"].unique())
    fold_of = info_a["fold"].to_dict()
    if info_a["fold"].to_dict() != info_b["fold"].to_dict():
        raise SystemExit("The two arms used different CV folds (different split file or seed).")

    # S[slide, factor, w] = per-slide Pearson of (1-w)*A + w*B
    S = np.full((len(slides), len(factors), len(GRID)), np.nan)
    for i, s in enumerate(slides):
        g = m[m["sample_id"] == s]
        for j, f in enumerate(factors):
            y = g[f"y_{f}_a"].to_numpy(float)
            pa = g[f"pred_{f}_a"].to_numpy(float)
            pb = g[f"pred_{f}_b"].to_numpy(float)
            for k, w in enumerate(GRID):
                S[i, j, k] = pearson(y, (1 - w) * pa + w * pb)

    # self-check: w = 0 and w = 1 must reproduce each arm's own per-slide Pearson from evaluate_embeddings.py
    for arm, k in ((args.a, 0), (args.b, len(GRID) - 1)):
        per = pd.read_csv(args.root / arm / "cv_per_sample.csv")
        for j, f in enumerate(factors):
            ref = per[per["factor"] == f].set_index("sample_id")["pearson"].reindex(slides).to_numpy(float)
            gap = float(np.nanmax(np.abs(ref - S[:, j, k])))
            if gap > 1e-6:
                raise SystemExit(f"Self-check failed for {arm}, {f}: per-slide Pearson differs by {gap:.2e} from cv_per_sample.csv.")
    print("Self-check passed: the fusion reproduces each arm's own per-slide Pearson at w = 0 and w = 1.")

    k_half = int(np.argmin(np.abs(GRID - 0.5)))
    slide_fold = np.array([fold_of[s] for s in slides])
    avg = S[:, :, k_half]
    stack = np.empty_like(avg)
    chosen = {}
    for fold in sorted(set(slide_fold)):
        rest = slide_fold != fold
        for j, f in enumerate(factors):
            k_best = int(np.nanargmax(np.nanmean(S[rest, j, :], axis=0)))
            chosen[(fold, f)] = float(GRID[k_best])
            stack[~rest, j] = S[~rest, j, k_best]

    def table(values, model):
        rows = []
        for i, s in enumerate(slides):
            for j, f in enumerate(factors):
                row = dict(model=model, sample_id=s, patient_id=info_a.loc[s, "patient_id"], factor=f,
                           pearson=float(values[i, j]), fold=int(fold_of[s]))
                if "technology" in info_a.columns:
                    row["technology"] = info_a.loc[s, "technology"]
                rows.append(row)
        return pd.DataFrame(rows)

    for suffix, values in (("avg", avg), ("stack", stack)):
        name = f"{args.out_name}_{suffix}"
        out = args.root / name
        out.mkdir(parents=True, exist_ok=True)
        table(values, name).to_csv(out / "cv_per_sample.csv", index=False)
        print(f"Wrote {out / 'cv_per_sample.csv'}")

    print(f"\nmean per-slide Pearson over {len(slides)} slides (arm A = {args.a}, arm B = {args.b}):")
    print("  " + f"{'factor':<22}{'A only':>10}{'B only':>10}{'avg 0.5':>10}{'stack':>10}")
    for j, f in enumerate(factors):
        print("  " + f"{f:<22}{S[:, j, 0].mean():>10.3f}{S[:, j, -1].mean():>10.3f}"
              f"{avg[:, j].mean():>10.3f}{stack[:, j].mean():>10.3f}")
    print("  " + f"{'MEAN_OF_FACTORS':<22}{S[:, :, 0].mean():>10.3f}{S[:, :, -1].mean():>10.3f}"
          f"{avg.mean():>10.3f}{stack.mean():>10.3f}")

    print("\nweight on arm B chosen by stacking, per held-out fold (0 = all A, 1 = all B):")
    for f in factors:
        print("  " + f"{f:<22}" + "  ".join(f"{chosen[(fold, f)]:.1f}" for fold in sorted(set(slide_fold))))
    print("\nNext: python scripts/compare_cv.py --reference <embedding-level arm> --models <that arm> "
          f"{args.out_name}_avg {args.out_name}_stack")


if __name__ == "__main__":
    main()
