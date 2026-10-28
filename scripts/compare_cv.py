#!/usr/bin/env python
"""
compare_cv.py -- paired comparison of representations evaluated with
`evaluate_embeddings.py --cv-frozen-encoder K`.

In that mode every slide is predicted by a model that never saw its patient, for
every representation, so per-slide Pearson values are PAIRED across
representations. The bootstrap resamples PATIENTS (the independent unit), not
spots, so the intervals are not inflated by spatial autocorrelation.

  python scripts/compare_cv.py --reference virchow2 \
      --models virchow2 omiclip_pretrained

Reads  <root>/<name>/cv_per_sample.csv   (root defaults to outputs/eval)
Writes <root>/compare_cv.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_wide(root, name):
    df = pd.read_csv(root / name / "cv_per_sample.csv")
    index = ["sample_id", "patient_id"]
    if "technology" in df.columns and df["technology"].notna().all():
        index.append("technology")
    wide = df.pivot_table(index=index, columns="factor", values="pearson")
    wide["MEAN_OF_FACTORS"] = wide.mean(axis=1)
    return wide


def patient_bootstrap(diff, patients, n_boot, seed):
    rng = np.random.RandomState(seed)
    uniq, inverse = np.unique(patients, return_inverse=True)
    sums = np.bincount(inverse, weights=diff, minlength=len(uniq))
    counts = np.bincount(inverse, minlength=len(uniq)).astype(float)
    draws = rng.randint(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return np.percentile(boot, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=PROJECT_DIR / "outputs" / "eval")
    p.add_argument("--models", nargs="+", required=True, help="names (folders under --root)")
    p.add_argument("--reference", required=True, help="model the others are compared against")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    names = list(dict.fromkeys(args.models + [args.reference]))
    wides = {n: load_wide(args.root, n) for n in names}
    common = wides[names[0]].index
    for n in names[1:]:
        common = common.intersection(wides[n].index)
    wides = {n: w.loc[common] for n, w in wides.items()}
    patients = common.get_level_values("patient_id").to_numpy()
    print(f"{len(common)} slides from {len(set(patients))} patients are present for every model.\n")

    columns = [c for c in wides[args.reference].columns if c != "MEAN_OF_FACTORS"] + ["MEAN_OF_FACTORS"]

    print("mean per-slide Pearson (higher is better):")
    print("  " + f"{'factor':<22}" + "".join(f"{n[:16]:>18}" for n in names))
    for c in columns:
        print("  " + f"{c:<22}" + "".join(f"{wides[n][c].mean():>18.3f}" for n in names))

    rows = []
    ref = wides[args.reference]
    print(f"\npaired difference vs {args.reference} (model - reference), 95% CI by patient bootstrap:")
    print("  " + f"{'model':<18}{'factor':<22}{'diff':>8}{'CI low':>9}{'CI high':>9}{'slides better':>15}")
    for n in names:
        if n == args.reference:
            continue
        for c in columns:
            diff = (wides[n][c] - ref[c]).to_numpy()
            lo, hi = patient_bootstrap(diff, patients, args.boot, args.seed)
            wins = int((diff > 0).sum())
            rows.append(dict(model=n, reference=args.reference, factor=c, mean_diff=diff.mean(),
                             ci_low=lo, ci_high=hi, slides_better=wins, n_slides=len(diff)))
            flag = "  *" if (lo > 0 or hi < 0) else ""
            print(f"  {n[:17]:<18}{c:<22}{diff.mean():>+8.3f}{lo:>+9.3f}{hi:>+9.3f}{wins:>9d}/{len(diff):<5d}{flag}")
    print("\n  * = interval excludes zero. With one slide per patient and a handful of "
          "factors tested, treat a lone star as suggestive, not conclusive.")

    if "technology" in common.names:
        tech = common.get_level_values("technology").to_numpy()
        print("\nby technology, mean of the factors only (model - reference, mean per-slide Pearson):")
        print("  " + f"{'model':<18}{'technology':<12}{'slides':>7}{'diff':>8}{'CI low':>9}{'CI high':>9}")
        for n in names:
            if n == args.reference:
                continue
            diff_all = (wides[n]["MEAN_OF_FACTORS"] - ref["MEAN_OF_FACTORS"]).to_numpy()
            for t in sorted(set(tech)):
                m = tech == t
                lo, hi = patient_bootstrap(diff_all[m], patients[m], args.boot, args.seed)
                print(f"  {n[:17]:<18}{t:<12}{int(m.sum()):>7d}{diff_all[m].mean():>+8.3f}{lo:>+9.3f}{hi:>+9.3f}")
                rows.append(dict(model=n, reference=args.reference, factor=f"MEAN_OF_FACTORS[{t}]",
                                 mean_diff=diff_all[m].mean(), ci_low=lo, ci_high=hi,
                                 slides_better=int((diff_all[m] > 0).sum()), n_slides=int(m.sum())))

    out = args.root / "compare_cv.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
