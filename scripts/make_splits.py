#!/usr/bin/env python
"""
make_splits.py -- technology-balanced splits for the 24 ccRCC slides.

Why: the original split (train INT1-18, val INT19-21, test INT22-24) follows the
HEST table order, and that order is grouped by Visium chemistry. INT1-12 are
fresh-frozen (36,601 genes) and INT13-24 are FFPE (17,943 genes), so val and test
ended up all-FFPE while training was two-thirds frozen.

This script writes ONE file that every script should read:

  datasets/splits/ccrcc_splits_v1.csv
      sample_id, patient, technology, fixed_split, cv_fold

  fixed_split : balanced train/val/test for the slow fine-tuning runs
                (default 16 / 4 / 4, half frozen and half FFPE in each)
  cv_fold     : balanced K-fold for patient-grouped CV (default 6 folds,
                each holding out 2 frozen + 2 FFPE slides)

Technology comes from the `subseries` column ("... frozen_a_1", "... ffpe_c_2"),
because `preservation_method` is empty in the HEST metadata. `nb_genes` is used
only as a cross-check.

    python scripts/make_splits.py
    python scripts/make_splits.py --seed 1 --folds 4

Same seed -> same file on every machine (numpy legacy RandomState).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_table(path):
    d = pd.read_csv(path)
    ids = d["id"].astype(str).str.strip().str.upper()
    d = d[ids.str.match(r"^INT\d+$")].copy()
    d["sample_id"] = ids[d.index]
    if len(d) == 0:
        raise SystemExit(f"No INT<number> rows found in {path}.")
    sub = d["subseries"].astype(str).str.lower()
    tech = np.where(sub.str.contains("frozen"), "frozen", np.where(sub.str.contains("ffpe"), "ffpe", "unknown"))
    if (tech == "unknown").any():
        bad = d.loc[tech == "unknown", ["sample_id", "subseries"]]
        raise SystemExit(f"Cannot tell frozen from FFPE for:\n{bad.to_string(index=False)}")
    d["technology"] = tech
    if "nb_genes" in d.columns:
        counts = d.groupby("technology")["nb_genes"].nunique()
        if (counts > 1).any():
            print("WARNING: nb_genes is not constant within a technology:\n", counts)
        else:
            print("Cross-check passed: nb_genes is constant within each technology "
                  + str(d.groupby("technology")["nb_genes"].first().to_dict()))
    d["n"] = d["sample_id"].str.extract(r"(\d+)")[0].astype(int)
    if d["patient"].isna().any():
        raise SystemExit("Some INT rows have no patient value.")
    return d.sort_values("n").reset_index(drop=True)


def build(d, folds, n_val, n_test, seed):
    out = d[["sample_id", "patient", "technology"]].copy()
    out["fixed_split"] = ""
    out["cv_fold"] = -1
    split_rng = np.random.RandomState(seed)
    fold_rng = np.random.RandomState(seed + 1)
    for tech in sorted(d["technology"].unique()):
        ids = d.loc[d["technology"] == tech, "sample_id"].tolist()
        if n_val + n_test >= len(ids):
            raise SystemExit(f"Not enough {tech} slides ({len(ids)}) for {n_val} val + {n_test} test.")
        order = [ids[i] for i in split_rng.permutation(len(ids))]
        for j, s in enumerate(order):
            part = "test" if j >= len(ids) - n_test else "val" if j >= len(ids) - n_test - n_val else "train"
            out.loc[out["sample_id"] == s, "fixed_split"] = part
        order = [ids[i] for i in fold_rng.permutation(len(ids))]
        for j, s in enumerate(order):
            out.loc[out["sample_id"] == s, "cv_fold"] = j % folds
        if len(ids) % folds:
            print(f"NOTE: {len(ids)} {tech} slides do not divide into {folds} folds; folds differ by one slide.")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample-table", type=Path, default=PROJECT_DIR / "datasets" / "hest_metadata" / "ccrcc_samples.csv")
    p.add_argument("--out", type=Path, default=PROJECT_DIR / "datasets" / "splits" / "ccrcc_splits_v1.csv")
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--n-val-per-tech", type=int, default=2)
    p.add_argument("--n-test-per-tech", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d = load_table(args.sample_table)
    print(f"{len(d)} INT slides: " + ", ".join(f"{k} {v}" for k, v in d['technology'].value_counts().items()))
    out = build(d, args.folds, args.n_val_per_tech, args.n_test_per_tech, args.seed)

    print("\nfixed split (slides per technology):")
    print(pd.crosstab(out["fixed_split"], out["technology"]).to_string())
    print("\nCV folds (slides per technology):")
    print(pd.crosstab(out["cv_fold"], out["technology"]).to_string())
    for part in ("train", "val", "test"):
        ids = out.loc[out["fixed_split"] == part, "sample_id"].tolist()
        print(f"\n{part}: " + " ".join(sorted(ids, key=lambda s: int(s[3:]))))
    for f in range(args.folds):
        ids = out.loc[out["cv_fold"] == f, "sample_id"].tolist()
        print(f"fold {f}: " + " ".join(sorted(ids, key=lambda s: int(s[3:]))))

    if out["patient"].duplicated().any():
        print("\nNOTE: some patients have several slides; both CV and fixed split must keep a patient in one part. "
              "Check the file before use.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()