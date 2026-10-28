#!/usr/bin/env python
"""
jitc_indomain.py -- how well can each arm recover the factor scores INSIDE the kidney cohort?

response_test.py trains the ridge heads on the 24 HEST slides and applies them to the JITC slides.
That mixes two problems: the heads may not transfer (different assay, much blurrier images), and the
arm may simply not carry the signal. This script removes the transfer problem: the heads are trained
and tested on JITC itself, leave-patients-out (6 folds over the 12 patients; both slides of a patient
always stay together; alpha chosen by an inner 3-fold patient-grouped CV on within-slide Pearson).

    python scripts/jitc_indomain.py

The result is a ceiling for the transfer numbers in response_test.py:
  * image arm in-domain  ~ 0   -> the blurry SpaceRanger image carries no usable signal at all
  * image arm in-domain  > 0 but transfer ~ 0   -> the image carries signal, the HEST heads just do not transfer
  * st arm in-domain far above its transfer     -> the sentence/assay shift hurts the HEST heads

Outputs go to outputs/jitc_indomain: indomain_pearson.csv (one row per slide x arm) and a printed table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from response_test import choose_alphas, colwise_pearson, fit_ridge

PROJECT_DIR = Path(__file__).resolve().parents[1]
FACTORS = ["t_cell_infiltration", "cytotoxicity", "tgfb_caf", "proliferation"]


def load_jitc(h5_path, pairs_path, factors_path):
    pairs = pd.read_csv(pairs_path)
    pairs["sample_id"] = pairs["sample_id"].astype(str)
    with h5py.File(h5_path, "r") as h:
        img, txt = h["image_embeddings"][...], h["st_embeddings"][...]
        if "sample_id" in h and "barcode" in h:
            dec = lambda a: np.array([v.decode() if isinstance(v, bytes) else str(v) for v in a[...]], dtype=object)
            if len(img) != len(pairs) or not (dec(h["sample_id"]) == pairs["sample_id"].to_numpy()).all() \
                    or not (dec(h["barcode"]) == pairs["barcode"].astype(str).to_numpy()).all():
                raise SystemExit("The embedding file does not line up with the pairs file (sample_id/barcode differ).")
        elif len(img) != len(pairs):
            raise SystemExit("Row count of the embedding file and the pairs file differ.")
    measured = pd.read_csv(factors_path)
    measured = pairs[["sample_id", "barcode"]].merge(measured, on=["sample_id", "barcode"], how="left")
    if len(measured) != len(pairs):
        raise SystemExit("Measured factor scores do not match the pairs file.")
    factors = [f for f in FACTORS if f in measured.columns]
    y = measured[factors].to_numpy(dtype=np.float32)
    if not np.isfinite(y).all():
        raise SystemExit("Measured factor scores contain NaN or inf.")
    return pairs, {"image": img, "st": txt, "both": np.concatenate([img, txt], axis=1)}, y, factors


def patient_folds(patients, k, seed):
    order = np.array(sorted(patients))
    order = order[np.random.RandomState(seed).permutation(len(order))]
    return {p: i % k for i, p in enumerate(order)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jitc-h5", type=Path, default=PROJECT_DIR / "outputs" / "jitc" / "jitc_omiclip_embeddings.h5")
    p.add_argument("--pairs", type=Path, default=PROJECT_DIR / "datasets" / "jitc_mrcc" / "jitc_pairs.csv")
    p.add_argument("--factors-csv", type=Path, default=PROJECT_DIR / "datasets" / "jitc_mrcc" / "jitc_factor_scores.csv")
    p.add_argument("--transfer-csv", type=Path, default=PROJECT_DIR / "outputs" / "jitc_response" / "transfer_pearson.csv",
                   help="transfer table from response_test.py, printed next to the in-domain numbers if present")
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alphas", default="0.1,1,10,100,1000,10000,100000,1000000,10000000,100000000")
    p.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "outputs" / "jitc_indomain")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]

    pairs, arms, y, factors = load_jitc(args.jitc_h5, args.pairs, args.factors_csv)
    slide = pairs["sample_id"].to_numpy()
    patient = pairs["patient"].astype(str).to_numpy()
    fold_of = patient_folds(set(patient), args.folds, args.seed)
    fold = np.array([fold_of[p_] for p_ in patient])
    slide_patient = dict(zip(slide, patient))
    print(f"{len(pairs):,} spots, {len(set(slide))} slides, {len(set(patient))} patients, {args.folds}-fold leave-patients-out")

    rows = []
    for arm, X in arms.items():
        oof = np.full(y.shape, np.nan)
        chosen = []
        for k in range(args.folds):
            te, tr = fold == k, fold != k
            tr_patients = sorted(set(patient[tr]))
            inner = patient_folds(tr_patients, args.inner_folds, args.seed + 1 + k)
            slide_fold = {s: inner[slide_patient[s]] for s in set(slide[tr])}
            alpha_best, _ = choose_alphas(X[tr], y[tr], slide[tr], slide_fold, alphas)
            coef, intercept = fit_ridge(X[tr], y[tr], alpha_best)
            oof[te] = X[te].astype(np.float64) @ coef + intercept
            chosen.append(alpha_best)
        print(f"{arm}: median chosen alpha per factor {dict(zip(factors, [float(a) for a in np.median(np.array(chosen), axis=0)]))}", flush=True)
        for s in sorted(set(slide)):
            ix = np.flatnonzero(slide == s)
            r = colwise_pearson(oof[ix], y[ix])
            rows.append(dict(sample_id=s, tissue=pairs["tissue"].iloc[ix[0]], patient=slide_patient[s], arm=arm,
                             **{f: float(v) for f, v in zip(factors, r)}))
    res = pd.DataFrame(rows)
    res["MEAN_OF_FACTORS"] = res[factors].mean(axis=1)
    res.to_csv(args.out_dir / "indomain_pearson.csv", index=False)

    cols = factors + ["MEAN_OF_FACTORS"]
    pd.set_option("display.width", 200)
    print("\nIn-domain (leave-patients-out inside JITC): mean within-slide Pearson, predicted vs measured")
    print(res.groupby(["tissue", "arm"])[cols].mean().round(3).to_string())
    print("\nAll 18 slides together:")
    print(res.groupby("arm")[cols].mean().round(3).to_string())

    if args.transfer_csv.exists():
        tr = pd.read_csv(args.transfer_csv)
        tr["MEAN_OF_FACTORS"] = tr[factors].mean(axis=1)
        print("\nFor comparison, TRANSFER from the HEST heads (response_test.py):")
        print(tr.groupby(["tissue", "arm"])[cols].mean().round(3).to_string())
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
