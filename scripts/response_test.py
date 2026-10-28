#!/usr/bin/env python
"""
response_test.py -- exploratory response test on the JITC mRCC cohort, three arms.

Stage 1 ridge heads (trained on the 24 HEST ccRCC slides, alpha chosen per factor by 6-fold
patient-grouped CV on within-slide Pearson) are applied to the JITC spots. Per slide the mean
predicted score of each factor is compared between responders and non-responders.

    python scripts/response_test.py --out-dir outputs/jitc_response

Arms (the same three as Stage 1, all from the frozen OmiCLIP towers):
    image  H&E tower only       st  text tower on the gene sentence       both  the two concatenated
plus `measured`: the factor scores computed directly from the JITC counts (the ceiling, and the
paper's own immune signal).

What is reported, per tissue (PRIM = primary tumour slides, MET = metastasis slides):
  * transfer: spot-level within-slide Pearson of predicted vs measured score on the JITC slides
    (all 18 slides, no labels needed). It shows how well each arm survives the move to this cohort
    and how much the blurry SpaceRanger image costs the H&E arm.
  * response: for each arm x factor, the difference of means (R minus NR), the AUC for R above NR, and
    the exact two-sided permutation p-value over every way of assigning the labels.
  Pre-specified primary contrast (written before any result was seen): T-cell score on metastasis slides,
  because the cohort paper reports its immune niche in metastases only. Everything else is descriptive.

Labels default to the six patients named in the paper (responders W3, W33, W38; non-responders W10, W47,
W55). For the other six the paper gives no per-patient label; pass --labels labels.csv (columns patient,response
with R or NR) if you obtain them.
"""

from __future__ import annotations

import argparse
import itertools
from math import comb
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE = PROJECT_DIR / "outputs" / "omiclip_factor_baseline"
DEFAULT_LABELS = {"W3": "R", "W33": "R", "W38": "R", "W10": "NR", "W47": "NR", "W55": "NR"}
PRIMARY = ("t_cell_infiltration", "MET")


# ----------------------------------------------------------------------------
# ridge with the alpha chosen by grouped CV (closed form, eigen-decomposition)
# ----------------------------------------------------------------------------

def colwise_pearson(a, b):
    a = a - a.mean(0)
    b = b - b.mean(0)
    den = np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0))
    return np.where(den > 0, (a * b).sum(0) / np.where(den > 0, den, 1), np.nan)


def choose_alphas(X, y, slide, slide_fold, alphas):
    """Per-factor alpha maximising the mean within-slide Pearson of out-of-fold predictions."""
    folds = np.array([slide_fold[s] for s in slide])
    d, f = X.shape[1], y.shape[1]
    stats = {}
    for k in np.unique(folds):
        m = folds == k
        Xk = X[m].astype(np.float64)
        stats[k] = (Xk.T @ Xk, Xk.T @ y[m].astype(np.float64), Xk.sum(0), y[m].astype(np.float64).sum(0), int(m.sum()))
    total = [sum(stats[k][i] for k in stats) for i in range(5)]
    per_slide = {}
    for k in stats:
        xtx, xty, sx, sy, n = (total[i] - stats[k][i] for i in range(5))
        mu_x, mu_y = sx / n, sy / n
        xtx_c = xtx - n * np.outer(mu_x, mu_x)
        xty_c = xty - n * np.outer(mu_x, mu_y)
        lam, V = np.linalg.eigh(xtx_c)
        lam = np.clip(lam, 0, None)
        vt_xty = V.T @ xty_c
        m = folds == k
        A = (X[m].astype(np.float64) - mu_x) @ V
        yk, sk = y[m], slide[m]
        for ai, a in enumerate(alphas):
            pred = A @ (vt_xty / (lam + a)[:, None]) + mu_y
            for s in np.unique(sk):
                ms = sk == s
                per_slide.setdefault(s, np.full((len(alphas), f), np.nan))[ai] = colwise_pearson(pred[ms], yk[ms])
    mean_r = np.nanmean(np.stack(list(per_slide.values())), axis=0)  # alphas x factors
    best = mean_r.argmax(0)
    return np.array([alphas[b] for b in best]), mean_r


def fit_ridge(X, y, alpha_per_factor):
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    mu_x, mu_y = X.mean(0), y.mean(0)
    Xc = X - mu_x
    lam, V = np.linalg.eigh(Xc.T @ Xc)
    lam = np.clip(lam, 0, None)
    vt_xty = V.T @ (Xc.T @ (y - mu_y))
    coef = np.stack([V @ (vt_xty[:, j] / (lam + alpha_per_factor[j])) for j in range(y.shape[1])], axis=1)
    return coef, mu_y - mu_x @ coef


# ----------------------------------------------------------------------------
# statistics on six or twelve patients
# ----------------------------------------------------------------------------

def auc_r_above_nr(v, resp):
    r, nr = v[resp], v[~resp]
    gt = (r[:, None] > nr[None, :]).sum() + 0.5 * (r[:, None] == nr[None, :]).sum()
    return gt / (len(r) * len(nr))


def exact_permutation_p(v, resp, max_exact=200_000, n_random=100_000, seed=0):
    n, k = len(v), int(resp.sum())
    obs = v[resp].mean() - v[~resp].mean()
    total = comb(n, k)
    if total <= max_exact:
        idx = np.arange(n)
        diffs = np.empty(total)
        for i, c in enumerate(itertools.combinations(range(n), k)):
            m = np.zeros(n, bool)
            m[list(c)] = True
            diffs[i] = v[m].mean() - v[~m].mean()
        p = float((np.abs(diffs) >= abs(obs) - 1e-12).mean())
        return p, 2.0 / total, total, "exact"
    rng = np.random.RandomState(seed)
    hits = 0
    for _ in range(n_random):
        m = np.zeros(n, bool)
        m[rng.choice(n, k, replace=False)] = True
        hits += abs(v[m].mean() - v[~m].mean()) >= abs(obs) - 1e-12
    return (hits + 1) / (n_random + 1), 2.0 / total, total, "random"


# ----------------------------------------------------------------------------

def load_stage1(npz, meta, split_csv):
    d = np.load(npz, allow_pickle=True)
    X, y = d["X"].astype(np.float32), d["y"].astype(np.float32)
    factors = [str(f) for f in d["factor_names"]]
    m = pd.read_csv(meta)
    slide = m["sample_id"].astype(str).str.upper().to_numpy()
    if not (len(X) == len(y) == len(slide)):
        raise SystemExit(f"{npz}: X {len(X)}, y {len(y)}, metadata {len(slide)} rows differ.")
    sp = pd.read_csv(split_csv)
    sp["sample_id"] = sp["sample_id"].astype(str).str.upper()
    return X, y, factors, slide, dict(zip(sp["sample_id"], sp["cv_fold"])), dict(zip(sp["sample_id"], sp["technology"]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jitc-h5", type=Path, default=PROJECT_DIR / "outputs" / "jitc" / "jitc_omiclip_embeddings.h5")
    p.add_argument("--pairs", type=Path, default=PROJECT_DIR / "datasets" / "jitc_mrcc" / "jitc_pairs.csv")
    p.add_argument("--factors-csv", type=Path, default=PROJECT_DIR / "datasets" / "jitc_mrcc" / "jitc_factor_scores.csv")
    p.add_argument("--stage1-image", type=Path, default=BASE / "omiclip_factor_dataset.npz")
    p.add_argument("--stage1-st", type=Path, default=BASE / "omiclip_st_factor_dataset.npz")
    p.add_argument("--stage1-both", type=Path, default=BASE / "omiclip_both_factor_dataset.npz")
    p.add_argument("--stage1-meta", type=Path, default=BASE / "omiclip_factor_metadata.csv")
    p.add_argument("--split-csv", type=Path, default=PROJECT_DIR / "datasets" / "splits" / "ccrcc_splits_v1.csv")
    p.add_argument("--train-tech", choices=["all", "ffpe", "frozen"], default="all",
                   help="Stage 1 slides used to fit the ridge heads (JITC is FFPE; 'ffpe' is the sensitivity run)")
    p.add_argument("--alphas", default="0.1,1,10,100,1000,10000,100000,1000000,10000000,100000000")
    p.add_argument("--labels", type=Path, default=None, help="CSV with columns patient,response (R or NR)")
    p.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "outputs" / "jitc_response")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]

    pairs = pd.read_csv(args.pairs)
    pairs["sample_id"] = pairs["sample_id"].astype(str)
    with h5py.File(args.jitc_h5, "r") as h:
        img_j, txt_j = h["image_embeddings"][...], h["st_embeddings"][...]
        if "sample_id" in h and "barcode" in h:
            hs = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in h["sample_id"][...]], dtype=object)
            hb = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in h["barcode"][...]], dtype=object)
            if len(hs) != len(pairs) or not (hs == pairs["sample_id"].to_numpy()).all() or not (hb == pairs["barcode"].astype(str).to_numpy()).all():
                raise SystemExit("The JITC embedding file does not line up with the pairs file (sample_id/barcode differ).")
        elif len(img_j) != len(pairs):
            raise SystemExit("Row count of the embedding file and the pairs file differ.")
    jitc = {"image": img_j, "st": txt_j, "both": np.concatenate([img_j, txt_j], axis=1)}
    measured = pd.read_csv(args.factors_csv)
    measured = pairs[["sample_id", "barcode"]].merge(measured, on=["sample_id", "barcode"], how="left")
    if len(measured) != len(pairs):
        raise SystemExit("Measured factor scores do not match the pairs file.")

    stage1_paths = {"image": args.stage1_image, "st": args.stage1_st, "both": args.stage1_both}
    predictions, chosen = {}, []
    for arm, path in stage1_paths.items():
        X, y, factors, slide, slide_fold, slide_tech = load_stage1(path, args.stage1_meta, args.split_csv)
        if X.shape[1] != jitc[arm].shape[1]:
            raise SystemExit(f"{arm}: Stage 1 has {X.shape[1]} features, JITC has {jitc[arm].shape[1]}.")
        alpha_best, mean_r = choose_alphas(X, y, slide, slide_fold, alphas)
        sel = np.ones(len(slide), bool) if args.train_tech == "all" else np.array([slide_tech[s] == args.train_tech for s in slide])
        coef, intercept = fit_ridge(X[sel], y[sel], alpha_best)
        predictions[arm] = jitc[arm].astype(np.float64) @ coef + intercept
        for j, name in enumerate(factors):
            chosen.append(dict(arm=arm, factor=name, alpha=alpha_best[j], stage1_cv_within_slide_r=float(mean_r[:, j].max())))
        print(f"{arm}: alphas {dict(zip(factors, [float(a) for a in alpha_best]))}; Stage 1 CV within-slide r "
              f"{dict(zip(factors, [round(float(v), 3) for v in mean_r.max(0)]))}", flush=True)
    pd.DataFrame(chosen).to_csv(args.out_dir / "stage1_alphas.csv", index=False)

    # ---- transfer: predicted vs measured on every JITC slide
    meas = measured[factors].to_numpy()
    rows = []
    for s, g in pairs.groupby("sample_id"):
        ix = g.index.to_numpy()
        for arm, pr in predictions.items():
            r = colwise_pearson(pr[ix], meas[ix])
            rows.append(dict(sample_id=s, tissue=g["tissue"].iloc[0], patient=g["patient"].iloc[0], arm=arm,
                             **{f: float(v) for f, v in zip(factors, r)}))
    transfer = pd.DataFrame(rows)
    transfer.to_csv(args.out_dir / "transfer_pearson.csv", index=False)
    print("\nTransfer: mean within-slide Pearson of predicted vs measured score on the JITC slides")
    print(transfer.groupby(["tissue", "arm"])[factors].mean().round(3).to_string())

    # ---- slide-level scores
    scores = []
    for s, g in pairs.groupby("sample_id"):
        ix = g.index.to_numpy()
        base = dict(sample_id=s, tissue=g["tissue"].iloc[0], patient=g["patient"].iloc[0], spots=len(ix))
        for arm, pr in predictions.items():
            scores.append(dict(base, arm=arm, **{f: float(pr[ix, j].mean()) for j, f in enumerate(factors)}))
        scores.append(dict(base, arm="measured", **{f: float(np.nanmean(meas[ix, j])) for j, f in enumerate(factors)}))
    scores = pd.DataFrame(scores)
    labels = DEFAULT_LABELS
    if args.labels:
        lab = pd.read_csv(args.labels)
        labels = dict(zip(lab["patient"].astype(str), lab["response"].astype(str)))
    scores["response"] = scores["patient"].map(labels)
    scores.to_csv(args.out_dir / "slide_scores.csv", index=False)

    # ---- response tests
    out = []
    for tissue in ("PRIM", "MET"):
        for arm in list(predictions) + ["measured"]:
            sub = scores[(scores.tissue == tissue) & (scores.arm == arm) & scores.response.notna()]
            sub = sub.groupby("patient").agg({**{f: "mean" for f in factors}, "response": "first"}).reset_index()
            resp = (sub["response"] == "R").to_numpy()
            if resp.sum() < 2 or (~resp).sum() < 2:
                continue
            for f in factors:
                v = sub[f].to_numpy()
                pval, min_p, n_assign, kind = exact_permutation_p(v, resp)
                out.append(dict(tissue=tissue, arm=arm, factor=f, n_R=int(resp.sum()), n_NR=int((~resp).sum()),
                                mean_R=v[resp].mean(), mean_NR=v[~resp].mean(), diff_R_minus_NR=v[resp].mean() - v[~resp].mean(),
                                auc_R_above_NR=auc_r_above_nr(v, resp), p_two_sided=pval, min_possible_p=min_p,
                                assignments=n_assign, p_method=kind, primary=bool((f, tissue) == PRIMARY)))
    tests = pd.DataFrame(out)
    tests.to_csv(args.out_dir / "response_tests.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\nResponse tests ({int(tests.n_R.iloc[0])} R vs {int(tests.n_NR.iloc[0])} NR patients; min possible two-sided p "
          f"{tests.min_possible_p.iloc[0]:.3f}). PRE-SPECIFIED PRIMARY = T-cell on MET:")
    cols = ["tissue", "arm", "factor", "mean_R", "mean_NR", "auc_R_above_NR", "p_two_sided"]
    print(tests[tests.primary][cols].round(3).to_string(index=False))
    print("\nAll tests (descriptive; no multiplicity correction is meaningful at this n):")
    print(tests[cols].round(3).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arms = list(predictions) + ["measured"]
        fig, axes = plt.subplots(2, len(arms), figsize=(3.0 * len(arms), 6), sharex=True)
        for r_i, tissue in enumerate(("PRIM", "MET")):
            for c_i, arm in enumerate(arms):
                ax = axes[r_i, c_i]
                sub = scores[(scores.tissue == tissue) & (scores.arm == arm) & scores.response.notna()]
                for k, (lab_name, colour) in enumerate((("NR", "#C0504D"), ("R", "#2E7D6B"))):
                    s2 = sub[sub.response == lab_name]
                    ax.scatter(np.full(len(s2), k) + np.linspace(-0.08, 0.08, max(len(s2), 1))[:len(s2)],
                               s2["t_cell_infiltration"], color=colour, s=28)
                    for xv, (_, row) in zip(np.full(len(s2), k) + np.linspace(-0.08, 0.08, max(len(s2), 1))[:len(s2)], s2.iterrows()):
                        ax.annotate(row["patient"], (xv, row["t_cell_infiltration"]), fontsize=6, xytext=(3, 2), textcoords="offset points")
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["NR", "R"])
                ax.set_title(f"{arm} | {tissue}", fontsize=9)
                if c_i == 0:
                    ax.set_ylabel("mean T-cell score")
        fig.tight_layout()
        fig.savefig(args.out_dir / "tcell_by_response.png", dpi=150)
        print(f"Figure: {args.out_dir / 'tcell_by_response.png'}")
    except Exception as exc:  # the figure is a convenience
        print(f"(no figure: {exc})")
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
