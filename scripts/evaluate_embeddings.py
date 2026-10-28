#!/usr/bin/env python
"""
evaluate_embeddings.py -- one evaluation harness for every representation.

Replaces train_*_factor_baseline.py / train_*_factor_mlp.py. Same task as the
pilots (embedding -> four ST factor scores) and the same sample split, with the
measurement fixes agreed after the pilots:

  * Ridge alpha and MLP learning rate are chosen on the VALIDATION samples only.
  * Selection / early-stopping criterion = mean validation Pearson over the four
    factors (methodology doc, section 13.2).
  * Pooled AND per-sample Pearson are reported. Pooled mixes between-slide
    offsets with within-slide signal; per-sample is the within-slide signal.
  * The sample split is checked for patient leakage.
  * Any representation can be restricted to an identical spot set (--restrict-to)
    and its targets are checked against a reference file.
  * --cv-frozen-encoder K: patient-grouped K-fold CV (Ridge, nested alpha
    selection). Only valid for encoders that were never adapted on ANY evaluated
    patient (Virchow2, pretrained OmiCLIP). Do NOT use it on fine-tuned
    encoders: they saw the gene sentences of their own training patients.

Typical use, from the vcell folder:

  python scripts/evaluate_embeddings.py --name omiclip_pretrained \
      --npz  outputs/omiclip_factor_baseline/omiclip_factor_dataset.npz \
      --meta outputs/omiclip_factor_baseline/omiclip_factor_metadata.csv

  python scripts/evaluate_embeddings.py --name virchow2 \
      --npz  features/hest_ccrcc_joined/hest_ccrcc_virchow2_factor_dataset.npz \
      --meta features/hest_ccrcc_joined/hest_ccrcc_virchow2_factor_metadata.csv \
      --restrict-to outputs/omiclip_factor_baseline/omiclip_factor_metadata.csv

Add --cv-frozen-encoder 6 to either command for the patient-grouped CV.
Outputs go to outputs/eval/<name>/. With --save-predictions the CV also writes
cv_oof_predictions.csv (used by fuse_predictions.py to average two arms).

Technology-balanced splits (recommended): run scripts/make_splits.py once, then add

  --split-csv datasets/splits/ccrcc_splits_v1.csv

to any command. That file replaces the built-in split above (which puts every
frozen slide in train and only FFPE slides in val and test), makes the CV folds
hold out 2 frozen + 2 FFPE slides each, and adds by-technology tables.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT_DIR = Path(__file__).resolve().parents[1]

TRAIN_SAMPLES = [f"INT{i}" for i in range(1, 19)]
VAL_SAMPLES = ["INT19", "INT20", "INT21"]
TEST_SAMPLES = ["INT22", "INT23", "INT24"]
ALL_SAMPLES = TRAIN_SAMPLES + VAL_SAMPLES + TEST_SAMPLES

# Filled in by main() when --split-csv is given: sample_id -> "frozen" / "ffpe".
TECHNOLOGY_OF_SAMPLE = {}


# ----------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------

def norm_sample(values):
    return values.astype(str).str.strip().str.upper()


def norm_barcode(values):
    return (
        values.astype(str).str.strip().str.upper().str.replace(r"-1$", "", regex=True)
    )


def parse_float_list(text):
    return [float(t) for t in str(text).split(",") if t.strip()]


def fmt(x):
    return "   nan" if x is None or not np.isfinite(x) else f"{x:+.3f}"


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def pearson_cols(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    den = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    num = (a * b).sum(axis=0)
    out = np.full(a.shape[1], np.nan)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def rank_cols(a):
    a = np.asarray(a, dtype=np.float64)
    return np.column_stack([rankdata(a[:, j]) for j in range(a.shape[1])])


def spearman_cols(a, b):
    return pearson_cols(rank_cols(a), rank_cols(b))


def r2_cols(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ss_res = ((y - p) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    out = np.full(y.shape[1], np.nan)
    ok = ss_tot > 0
    out[ok] = 1.0 - ss_res[ok] / ss_tot[ok]
    return out


def sample_groups(samples):
    return [np.flatnonzero(samples == s) for s in pd.unique(samples)]


def selection_score(y, p, groups, mode):
    """Mean over factors of Pearson. 'pooled' = over all spots of the split,
    'within' = mean of per-sample Pearson. Pearson is invariant to per-column
    affine maps, so this can be computed in standardised target space."""
    if mode == "pooled":
        return float(np.nanmean(pearson_cols(y, p)))
    per_sample = [np.nanmean(pearson_cols(y[g], p[g])) for g in groups if len(g) >= 3]
    return float(np.nanmean(per_sample))


def per_sample_table(y, p, samples, factors):
    rows = []
    for s in pd.unique(samples):
        idx = np.flatnonzero(samples == s)
        if len(idx) < 3:
            continue
        pe = pearson_cols(y[idx], p[idx])
        sp = spearman_cols(y[idx], p[idx])
        for j, f in enumerate(factors):
            rows.append(
                dict(sample_id=s, factor=f, n=len(idx), pearson=pe[j], spearman=sp[j])
            )
    return pd.DataFrame(rows)


def metric_rows(head, split, y, p, samples, factors):
    pe = pearson_cols(y, p)
    sp = spearman_cols(y, p)
    r2 = r2_cols(y, p)
    rmse = np.sqrt(((y - p) ** 2).mean(axis=0))
    per = per_sample_table(y, p, samples, factors)
    within = per.groupby("factor")["pearson"].mean().reindex(factors)
    rows = []
    for j, f in enumerate(factors):
        rows.append(
            dict(
                head=head, split=split, factor=f, n=len(y),
                pearson=pe[j], spearman=sp[j], r2=r2[j], rmse=rmse[j],
                pearson_within=float(within.iloc[j]),
            )
        )
    per.insert(0, "split", split)
    per.insert(0, "head", head)
    return rows, per


# ----------------------------------------------------------------------------
# standardisation and closed-form ridge
# ----------------------------------------------------------------------------

class Standardiser:
    def fit(self, a):
        self.mu = a.mean(axis=0, dtype=np.float64)
        sd = a.std(axis=0, dtype=np.float64)
        sd[sd < 1e-8] = 1.0
        self.sd = sd
        return self

    def transform(self, a, dtype=np.float32):
        return (a.astype(dtype, copy=False) - self.mu.astype(dtype)) / self.sd.astype(dtype)

    def inverse(self, a):
        return np.asarray(a, dtype=np.float64) * self.sd + self.mu


class RidgePath:
    """Closed-form ridge for many alphas from ONE eigendecomposition.

    X and Y must be standardised on the training rows (zero mean), so the
    intercept is zero. Equivalent to sklearn Ridge(alpha) on the same data."""

    def __init__(self, xs, ys):
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        evals, vecs = np.linalg.eigh(x.T @ x)
        self.evals = np.clip(evals, 0.0, None)
        self.vecs = vecs
        self.vt_b = vecs.T @ (x.T @ y)

    def coef(self, alpha):
        return self.vecs @ (self.vt_b / (self.evals + alpha)[:, None])


def predict_linear(xs, w):
    return (xs @ w.astype(np.float32)).astype(np.float64)


# ----------------------------------------------------------------------------
# data loading and checks
# ----------------------------------------------------------------------------

def load_dataset(npz_path, meta_path, restrict_to, atol=1e-4):
    data = np.load(npz_path, allow_pickle=True)
    x = data["X"]
    y = data["y"]
    factors = [str(f) for f in data["factor_names"]]
    meta = pd.read_csv(meta_path)

    if not (len(x) == len(y) == len(meta)):
        raise RuntimeError(
            f"Row mismatch: X={len(x)}, y={len(y)}, metadata={len(meta)}"
        )

    meta = meta.copy()
    meta["sample_id"] = norm_sample(meta["sample_id"])
    meta["barcode_key"] = norm_barcode(meta["barcode"])

    if meta.duplicated(["sample_id", "barcode_key"]).any():
        raise RuntimeError("Duplicate (sample_id, barcode) rows in metadata.")

    # Alignment check against identifiers stored inside the npz (Virchow files).
    if "sample_id" in data.files and "barcode" in data.files:
        npz_s = pd.Series(data["sample_id"]).astype(str).str.strip().str.upper()
        npz_b = norm_barcode(pd.Series(data["barcode"]))
        if not (
            np.array_equal(npz_s.to_numpy(), meta["sample_id"].to_numpy())
            and np.array_equal(npz_b.to_numpy(), meta["barcode_key"].to_numpy())
        ):
            raise RuntimeError(
                "npz sample_id/barcode arrays do not match the metadata CSV row-for-row."
            )
        print("Alignment check: npz identifiers match metadata row-for-row.")

    info = {"rows_loaded": int(len(meta))}
    keep = np.ones(len(meta), dtype=bool)

    if restrict_to:
        ref = pd.read_csv(restrict_to)
        ref["sample_id"] = norm_sample(ref["sample_id"])
        ref["barcode_key"] = norm_barcode(ref["barcode"])
        ref_idx = pd.MultiIndex.from_arrays([ref["sample_id"], ref["barcode_key"]])
        cur_idx = pd.MultiIndex.from_arrays([meta["sample_id"], meta["barcode_key"]])
        keep = np.asarray(cur_idx.isin(ref_idx))
        missing_from_this = int((~ref_idx.isin(cur_idx)).sum())
        print(
            f"Restricting to reference spots: keeping {keep.sum():,} of "
            f"{len(meta):,} rows; {missing_from_this:,} reference spots absent here."
        )
        info.update(
            reference_rows=int(len(ref)),
            rows_kept=int(keep.sum()),
            reference_spots_missing_here=missing_from_this,
        )

        shared = [f for f in factors if f in ref.columns]
        if shared and keep.any():
            mine = meta.loc[keep, ["sample_id", "barcode_key"]].copy()
            for j, f in enumerate(factors):
                mine[f"mine_{f}"] = y[keep][:, j]
            merged = mine.merge(
                ref[["sample_id", "barcode_key"] + shared],
                on=["sample_id", "barcode_key"],
                how="inner",
            )
            worst = max(
                float(np.abs(merged[f"mine_{f}"] - merged[f]).max()) for f in shared
            )
            print(f"Target integrity vs reference: max |y - y_ref| = {worst:.2e}")
            info["target_max_abs_diff_vs_reference"] = worst
            if worst > atol:
                print("  WARNING: factor targets differ between this file and the reference.")

    x = x[keep].astype(np.float32)
    y = y[keep].astype(np.float32)
    meta = meta.loc[keep].reset_index(drop=True)
    return x, y, meta, factors, info


def load_patient_map(sample_table, samples):
    mapping = {}
    source = "sample ids used as patient ids (no sample table found)"
    if sample_table and Path(sample_table).exists():
        table = pd.read_csv(sample_table)
        id_col = next((c for c in ["id", "sample_id", "sample"] if c in table.columns), None)
        pat_col = next(
            (c for c in ["patient", "patient_id", "subject_id", "donor_id"] if c in table.columns),
            None,
        )
        if id_col and pat_col:
            for sid, pid in zip(norm_sample(table[id_col]), table[pat_col]):
                if pd.notna(pid) and str(pid).strip():
                    mapping[sid] = str(pid).strip()
            source = f"{sample_table} ({id_col} -> {pat_col})"
    for s in samples:
        mapping.setdefault(s, f"UNKNOWN_{s}")
    return mapping, source


def load_split_csv(path):
    """Read the file written by make_splits.py.
    Returns (train, val, test sample lists, technology map, cv-fold map)."""
    s = pd.read_csv(path)
    for col in ("sample_id", "fixed_split"):
        if col not in s.columns:
            raise SystemExit(f"{path} has no '{col}' column. Columns: {s.columns.tolist()}")
    s["sample_id"] = norm_sample(s["sample_id"])
    if s["sample_id"].duplicated().any():
        raise SystemExit(f"{path}: duplicate sample_id rows.")
    parts = {k: s.loc[s["fixed_split"] == k, "sample_id"].tolist() for k in ("train", "val", "test")}
    for k, v in parts.items():
        if not v:
            raise SystemExit(f"{path}: no slides with fixed_split == '{k}'.")
    tech = dict(zip(s["sample_id"], s["technology"].astype(str))) if "technology" in s.columns else {}
    folds = {}
    if "cv_fold" in s.columns:
        folds = {sid: int(f) for sid, f in zip(s["sample_id"], s["cv_fold"]) if int(f) >= 0}
    return parts["train"], parts["val"], parts["test"], tech, folds


def technology_table(per_sample):
    """Mean per-slide Pearson (mean of factors) by head / split / technology."""
    if not TECHNOLOGY_OF_SAMPLE or "technology" not in per_sample.columns:
        return None
    ps = per_sample.dropna(subset=["technology"])
    per_slide = ps.groupby(["head", "split", "technology", "sample_id"], sort=False)["pearson"].mean().reset_index()
    return (
        per_slide.groupby(["head", "split", "technology"], sort=False)
        .agg(n_slides=("sample_id", "nunique"), mean_per_slide_pearson=("pearson", "mean"))
        .reset_index()
    )


def check_patient_leakage(patient_of_sample):
    split_of = {}
    for name, group in (("train", TRAIN_SAMPLES), ("val", VAL_SAMPLES), ("test", TEST_SAMPLES)):
        for s in group:
            split_of[s] = name
    by_patient = defaultdict(lambda: defaultdict(list))
    for s, split in split_of.items():
        by_patient[patient_of_sample.get(s, s)][split].append(s)
    leaks = {p: dict(v) for p, v in by_patient.items() if len(v) > 1}
    n_patients = {
        split: len({patient_of_sample.get(s, s) for s, sp in split_of.items() if sp == split})
        for split in ("train", "val", "test")
    }
    return leaks, n_patients


# ----------------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------------

def make_mlp(d_in, d_out, dropout):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(d_in, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, d_out),
    )


def mlp_predict(model, x, device, batch=4096):
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i : i + batch]).to(device)
            out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def train_mlp(xtr, ytr, xva, yva, groups_va, lr, seed, args, device):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_mlp(xtr.shape[1], ytr.shape[1], args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    xt = torch.from_numpy(xtr).to(device)
    yt = torch.from_numpy(ytr.astype(np.float32)).to(device)
    n = len(xt)
    gen = torch.Generator().manual_seed(seed)

    best = {"monitor": -np.inf, "score": np.nan, "loss": np.nan, "epoch": 0, "state": None}
    bad = 0
    epochs_run = 0

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        epochs_run = epoch

        pv = mlp_predict(model, xva, device)
        val_loss = float(((pv - yva) ** 2).mean())
        score = selection_score(yva, pv, groups_va, args.select)
        monitor = score if args.es_metric == "pearson" else -val_loss

        if monitor > best["monitor"] + 1e-9:
            best.update(
                monitor=monitor, score=score, loss=val_loss, epoch=epoch,
                state=copy.deepcopy(model.state_dict()),
            )
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break

    model.load_state_dict(best["state"])
    model.eval()
    return model, dict(
        best_epoch=best["epoch"], val_score=best["score"],
        val_loss=best["loss"], epochs_run=epochs_run,
    )


# ----------------------------------------------------------------------------
# standard protocol (train INT1-18 / val INT19-21 / test INT22-24)
# ----------------------------------------------------------------------------

def run_standard(x, y, meta, factors, args, out_dir):
    split = meta["split"].to_numpy()
    samples = meta["sample_id"].to_numpy()
    idx = {k: np.flatnonzero(split == k) for k in ("train", "val", "test")}
    for k, v in idx.items():
        if len(v) == 0:
            raise RuntimeError(f"No rows in the {k} split.")

    x_std = Standardiser().fit(x[idx["train"]])
    y_std = Standardiser().fit(y[idx["train"]])
    xs = {k: x_std.transform(x[v]) for k, v in idx.items()}
    ys = {k: y_std.transform(y[v], np.float64) for k, v in idx.items()}
    y_raw = {k: y[v].astype(np.float64) for k, v in idx.items()}
    smp = {k: samples[v] for k, v in idx.items()}
    groups_va = sample_groups(smp["val"])

    sweep, all_metrics, all_per_sample = [], [], []
    predictions = {}
    chosen = {}

    # ---- Ridge --------------------------------------------------------------
    t0 = time.time()
    path = RidgePath(xs["train"], ys["train"])
    for alpha in args.alphas:
        pv = predict_linear(xs["val"], path.coef(alpha))
        sweep.append(
            dict(
                head="ridge", param="alpha", value=alpha, seed=-1,
                val_pooled=selection_score(ys["val"], pv, groups_va, "pooled"),
                val_within=selection_score(ys["val"], pv, groups_va, "within"),
                best_epoch=np.nan,
            )
        )
    col = "val_pooled" if args.select == "pooled" else "val_within"
    best_row = max((r for r in sweep if r["head"] == "ridge"), key=lambda r: r[col])
    alpha = best_row["value"]
    chosen["ridge_alpha"] = alpha
    print(f"\nRidge sweep ({time.time() - t0:.0f}s), selection = val mean Pearson ({args.select}):")
    for r in sweep:
        mark = "  <-- chosen" if r["value"] == alpha else ""
        print(f"  alpha={r['value']:<9g} val pooled {fmt(r['val_pooled'])}  within {fmt(r['val_within'])}{mark}")

    w = path.coef(alpha)
    preds = {k: y_std.inverse(predict_linear(xs[k], w)) for k in idx}
    predictions["ridge"] = preds
    for k in ("train", "val", "test"):
        rows, per = metric_rows("ridge", k, y_raw[k], preds[k], smp[k], factors)
        all_metrics += rows
        all_per_sample.append(per)

    # ---- MLP ----------------------------------------------------------------
    if not args.no_mlp:
        import torch

        device = torch.device(
            args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"\nMLP sweep on {device}: lrs={args.mlp_lrs}, seeds={args.seeds}, es={args.es_metric}")
        models = {}
        t0 = time.time()
        for lr in args.mlp_lrs:
            for seed in range(args.seeds):
                model, info = train_mlp(
                    xs["train"], ys["train"], xs["val"], ys["val"], groups_va, lr, seed, args, device
                )
                pv = mlp_predict(model, xs["val"], device)
                sweep.append(
                    dict(
                        head="mlp", param="lr", value=lr, seed=seed,
                        val_pooled=selection_score(ys["val"], pv, groups_va, "pooled"),
                        val_within=selection_score(ys["val"], pv, groups_va, "within"),
                        best_epoch=info["best_epoch"],
                    )
                )
                models[(lr, seed)] = copy.deepcopy(model).cpu()
                print(
                    f"  lr={lr:<7g} seed={seed}  best epoch {info['best_epoch']:>3d}/{info['epochs_run']:<3d}"
                    f"  val pooled {fmt(sweep[-1]['val_pooled'])}  within {fmt(sweep[-1]['val_within'])}"
                )
        mlp_rows = [r for r in sweep if r["head"] == "mlp"]
        by_lr = defaultdict(list)
        for r in mlp_rows:
            by_lr[r["value"]].append(r[col])
        best_lr = max(by_lr, key=lambda v: float(np.mean(by_lr[v])))
        chosen["mlp_lr"] = best_lr
        print(f"  ({time.time() - t0:.0f}s) mean val score per lr: "
              + ", ".join(f"{v:g}: {fmt(float(np.mean(s)))}" for v, s in sorted(by_lr.items()))
              + f"  -> chosen lr={best_lr:g}")

        seed_metrics, seed_per = [], []
        for seed in range(args.seeds):
            model = models[(best_lr, seed)].to(device)
            preds_s = {k: y_std.inverse(mlp_predict(model, xs[k], device)) for k in idx}
            if seed == 0:
                predictions["mlp"] = preds_s
            for k in ("train", "val", "test"):
                rows, per = metric_rows("mlp", k, y_raw[k], preds_s[k], smp[k], factors)
                for r in rows:
                    r["seed"] = seed
                per["seed"] = seed
                seed_metrics += rows
                seed_per.append(per)
        sm = pd.DataFrame(seed_metrics)
        agg = (
            sm.groupby(["head", "split", "factor"], sort=False)
            .agg(
                n=("n", "first"), pearson=("pearson", "mean"), pearson_sd=("pearson", "std"),
                spearman=("spearman", "mean"), r2=("r2", "mean"), rmse=("rmse", "mean"),
                pearson_within=("pearson_within", "mean"),
            )
            .reset_index()
        )
        all_metrics += agg.to_dict("records")
        sp = pd.concat(seed_per, ignore_index=True)
        sp = (
            sp.groupby(["head", "split", "sample_id", "factor"], sort=False)
            .agg(n=("n", "first"), pearson=("pearson", "mean"), spearman=("spearman", "mean"))
            .reset_index()
        )
        all_per_sample.append(sp)

    metrics = pd.DataFrame(all_metrics)
    per_sample = pd.concat(all_per_sample, ignore_index=True)
    sweep_df = pd.DataFrame(sweep)
    if TECHNOLOGY_OF_SAMPLE:
        per_sample["technology"] = per_sample["sample_id"].map(TECHNOLOGY_OF_SAMPLE)
    tech_tab = technology_table(per_sample)

    metrics.to_csv(out_dir / "metrics.csv", index=False)
    per_sample.to_csv(out_dir / "per_sample_metrics.csv", index=False)
    sweep_df.to_csv(out_dir / "sweep.csv", index=False)
    if tech_tab is not None:
        tech_tab.to_csv(out_dir / "metrics_by_technology.csv", index=False)

    if args.save_predictions:
        for head, preds in predictions.items():
            frames = []
            for k in ("train", "val", "test"):
                f = meta.iloc[idx[k]].copy()
                f["split"] = k
                for j, name in enumerate(factors):
                    f[f"true_{name}"] = y_raw[k][:, j]
                    f[f"pred_{name}"] = preds[k][:, j]
                frames.append(f)
            pd.concat(frames, ignore_index=True).to_csv(out_dir / f"predictions_{head}.csv", index=False)

    print_standard_summary(args.name, metrics, per_sample, factors, chosen, tech_tab)
    return chosen


def print_standard_summary(name, metrics, per_sample, factors, chosen, tech_tab=None):
    print(f"\n==== {name}: standard protocol (everything chosen on validation) ====")
    print("mean Pearson over the 4 factors:   train    val   test  | within-slide: val   test")
    for head in metrics["head"].unique():
        m = metrics[metrics["head"] == head]
        pooled = m.groupby("split")["pearson"].mean()
        within = m.groupby("split")["pearson_within"].mean()
        print(
            f"  {head:<6}                       {fmt(pooled['train'])} {fmt(pooled['val'])} {fmt(pooled['test'])}"
            f"  |          {fmt(within['val'])} {fmt(within['test'])}"
        )
    print("\ntest, pooled Pearson per factor (within-slide in brackets):")
    for head in metrics["head"].unique():
        m = metrics[(metrics["head"] == head) & (metrics["split"] == "test")].set_index("factor")
        cells = "  ".join(f"{f[:9]:<9} {fmt(m.loc[f, 'pearson'])} ({fmt(m.loc[f, 'pearson_within'])})" for f in factors)
        print(f"  {head:<6} {cells}")
    print("\nper-sample Pearson, mean of the 4 factors (val + test slides):")
    for head in per_sample["head"].unique():
        ps = per_sample[(per_sample["head"] == head) & (per_sample["split"].isin(["val", "test"]))]
        table = ps.groupby(["split", "sample_id"])["pearson"].mean()
        print("  " + f"{head:<6} " + "  ".join(f"{s}:{fmt(v)}" for (_, s), v in table.items()))
    if tech_tab is not None:
        print("\nmean per-slide Pearson (mean of the 4 factors) by technology; n = slides:")
        for head in tech_tab["head"].unique():
            for split in ("val", "test"):
                sub = tech_tab[(tech_tab["head"] == head) & (tech_tab["split"] == split)]
                cells = "  ".join(
                    f"{r.technology}: {fmt(r.mean_per_slide_pearson)} (n={int(r.n_slides)})" for r in sub.itertuples()
                )
                print(f"  {head:<6} {split:<5} {cells}")
    print(f"\nchosen: {chosen}")


# ----------------------------------------------------------------------------
# patient-grouped CV (frozen encoders only)
# ----------------------------------------------------------------------------

def assign_folds(patients, k, seed, strata=None):
    """Random patient-to-fold assignment. With `strata` (patient -> label, e.g.
    technology) each stratum is shuffled and dealt round-robin, so every fold gets a
    near-equal share of each stratum."""
    patients = sorted(set(patients))
    rng = np.random.RandomState(seed)
    if not strata:
        perm = rng.permutation(len(patients))
        return {patients[i]: rank % k for rank, i in enumerate(perm)}
    out, counter = {}, 0
    for label in sorted({str(strata.get(p, "?")) for p in patients}):
        members = [p for p in patients if str(strata.get(p, "?")) == label]
        for i in rng.permutation(len(members)):
            out[members[i]] = counter % k
            counter += 1
    return out


def run_cv(x, y, meta, factors, patient_of_sample, args, out_dir, fold_of_sample=None):
    k = args.cv_frozen_encoder
    patients_all = [patient_of_sample[s] for s in ALL_SAMPLES]
    tech_of_patient = (
        {patient_of_sample[s]: t for s, t in TECHNOLOGY_OF_SAMPLE.items() if s in patient_of_sample}
        if TECHNOLOGY_OF_SAMPLE else None
    )
    if fold_of_sample:
        fold_of_patient = {}
        for s, f in fold_of_sample.items():
            pid = patient_of_sample.get(s)
            if pid is None:
                continue
            if fold_of_patient.setdefault(pid, f) != f:
                raise RuntimeError(f"Patient {pid} is assigned to more than one CV fold in the split file.")
        k_file = max(fold_of_patient.values()) + 1
        if k != k_file:
            print(f"NOTE: --cv-frozen-encoder {k} overridden by the {k_file} folds in the split file.")
        k = k_file
    else:
        fold_of_patient = assign_folds(patients_all, k, args.cv_seed, tech_of_patient)

    patient = meta["sample_id"].map(patient_of_sample).to_numpy()
    rows_ok = np.array([p in fold_of_patient for p in patient])
    fold = np.array([fold_of_patient.get(p, -1) for p in patient])
    samples = meta["sample_id"].to_numpy()
    n_ok = int(rows_ok.sum())
    print(f"\n==== {args.name}: patient-grouped {k}-fold CV (Ridge, nested alpha) ====")
    print(f"rows used: {n_ok:,} of {len(meta):,}; patients per fold: "
          + ", ".join(f"{f}:{sum(1 for v in fold_of_patient.values() if v == f)}" for f in range(k)))

    oof = np.full(y.shape, np.nan, dtype=np.float64)
    fold_rows, chosen_alphas = [], {}
    t0 = time.time()

    for f in range(k):
        te = np.flatnonzero(rows_ok & (fold == f))
        tr = np.flatnonzero(rows_ok & (fold != f))
        tr_patients = sorted(set(patient[tr]))
        n_in = min(args.cv_inner, len(tr_patients))
        inner_fold = assign_folds(tr_patients, n_in, args.cv_seed + 1 + f, tech_of_patient)
        inner_of_row = np.array([inner_fold[p] for p in patient[tr]])

        score = defaultdict(list)
        for g in range(n_in):
            a_idx, b_idx = tr[inner_of_row != g], tr[inner_of_row == g]
            xs_, ys_ = Standardiser().fit(x[a_idx]), Standardiser().fit(y[a_idx])
            path = RidgePath(xs_.transform(x[a_idx]), ys_.transform(y[a_idx], np.float64))
            xb, yb = xs_.transform(x[b_idx]), ys_.transform(y[b_idx], np.float64)
            grp = sample_groups(samples[b_idx])
            for a in args.alphas:
                score[a].append(
                    selection_score(yb, predict_linear(xb, path.coef(a)), grp, args.select)
                )
        alpha = max(args.alphas, key=lambda a: float(np.nanmean(score[a])))
        chosen_alphas[f] = alpha

        xs_, ys_ = Standardiser().fit(x[tr]), Standardiser().fit(y[tr])
        path = RidgePath(xs_.transform(x[tr]), ys_.transform(y[tr], np.float64))
        pred = ys_.inverse(predict_linear(xs_.transform(x[te]), path.coef(alpha)))
        oof[te] = pred

        pe = pearson_cols(y[te], pred)
        for j, name in enumerate(factors):
            fold_rows.append(
                dict(fold=f, patients="|".join(sorted(set(patient[te]))), n_test=len(te),
                     alpha=alpha, factor=name, pearson_pooled=pe[j])
            )
        print(f"  fold {f}: test patients={len(set(patient[te]))}, spots={len(te):,}, alpha={alpha:g}, "
              f"mean pooled r={fmt(float(np.nanmean(pe)))}  ({time.time() - t0:.0f}s)")

    ok = rows_ok & np.isfinite(oof).all(axis=1)
    per = per_sample_table(y[ok].astype(np.float64), oof[ok], samples[ok], factors)
    per["patient_id"] = per["sample_id"].map(patient_of_sample)
    per["fold"] = per["patient_id"].map(fold_of_patient)
    if TECHNOLOGY_OF_SAMPLE:
        per["technology"] = per["sample_id"].map(TECHNOLOGY_OF_SAMPLE)
    per.insert(0, "model", args.name)

    summary = []
    oof_pe = pearson_cols(y[ok], oof[ok])
    for j, name in enumerate(factors):
        v = per[per["factor"] == name]["pearson"]
        summary.append(
            dict(factor=name, oof_pooled_pearson=oof_pe[j], mean_per_sample_pearson=v.mean(),
                 sd_per_sample=v.std(), sem_per_sample=v.std() / np.sqrt(max(len(v), 1)),
                 n_samples=len(v))
        )
    mean_row = {
        "factor": "MEAN_OF_FACTORS",
        "oof_pooled_pearson": float(np.nanmean(oof_pe)),
        "mean_per_sample_pearson": float(np.mean([r["mean_per_sample_pearson"] for r in summary])),
        "sd_per_sample": np.nan, "sem_per_sample": np.nan, "n_samples": summary[0]["n_samples"],
    }
    summary.append(mean_row)
    summary_df = pd.DataFrame(summary)

    tech_summary = None
    if TECHNOLOGY_OF_SAMPLE and "technology" in per.columns:
        g = (
            per.dropna(subset=["technology"]).groupby(["technology", "factor"])["pearson"]
            .agg(mean_per_sample_pearson="mean", n_samples="count").reset_index()
        )
        means = g.groupby("technology").agg(
            mean_per_sample_pearson=("mean_per_sample_pearson", "mean"), n_samples=("n_samples", "max")
        ).reset_index()
        means.insert(1, "factor", "MEAN_OF_FACTORS")
        tech_summary = pd.concat([g, means], ignore_index=True)

    if args.save_predictions:
        # out-of-fold predictions, so arms can be fused at the prediction level (fuse_predictions.py)
        pred_df = pd.DataFrame({"sample_id": samples[ok], "barcode_key": meta["barcode_key"].to_numpy()[ok]})
        for j, name in enumerate(factors):
            pred_df[f"y_{name}"] = y[ok][:, j]
            pred_df[f"pred_{name}"] = oof[ok][:, j]
        pred_df.to_csv(out_dir / "cv_oof_predictions.csv", index=False)
        print(f"Wrote out-of-fold predictions: {out_dir / 'cv_oof_predictions.csv'}")

    per.to_csv(out_dir / "cv_per_sample.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "cv_summary.csv", index=False)
    if tech_summary is not None:
        tech_summary.to_csv(out_dir / "cv_summary_by_technology.csv", index=False)

    print("\nout-of-fold results (every slide predicted by a model that never saw its patient):")
    print("  factor                 pooled r   mean per-slide r   (sd, sem)")
    for r in summary:
        sd = "" if not np.isfinite(r["sd_per_sample"]) else f"({r['sd_per_sample']:.3f}, {r['sem_per_sample']:.3f})"
        print(f"  {r['factor']:<22} {fmt(r['oof_pooled_pearson'])}     {fmt(r['mean_per_sample_pearson'])}       {sd}")
    if tech_summary is not None:
        print("\nmean per-slide r by technology (factors in columns):")
        wide = tech_summary.pivot(index="technology", columns="factor", values="mean_per_sample_pearson")
        n_by = tech_summary[tech_summary["factor"] == "MEAN_OF_FACTORS"].set_index("technology")["n_samples"]
        for tech, row in wide.iterrows():
            cells = "  ".join(f"{c[:9]} {fmt(row[c])}" for c in factors + ["MEAN_OF_FACTORS"])
            print(f"  {tech:<7} (n={int(n_by[tech])})  {cells}")
    return dict(cv_alphas=chosen_alphas)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="label; outputs go to outputs/eval/<name>/")
    p.add_argument("--npz", required=True, type=Path, help="npz with X, y, factor_names")
    p.add_argument("--meta", required=True, type=Path, help="metadata CSV, same row order as the npz")
    p.add_argument("--restrict-to", type=Path, default=None,
                   help="reference metadata CSV; keep only (sample_id, barcode) spots present in it")
    p.add_argument("--sample-table", type=Path,
                   default=PROJECT_DIR / "datasets" / "hest_metadata" / "ccrcc_samples.csv")
    p.add_argument("--split-csv", type=Path, default=None,
                   help="split file from make_splits.py (fixed_split, cv_fold, technology). Replaces the "
                        "built-in INT1-18 / 19-21 / 22-24 split and enables by-technology reporting.")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--alphas", type=parse_float_list, default=parse_float_list("0.1,1,10,100,1000,10000,100000"))
    p.add_argument("--mlp-lrs", type=parse_float_list, default=parse_float_list("1e-4,3e-4,1e-3"))
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--no-mlp", action="store_true")
    p.add_argument("--select", choices=["pooled", "within"], default="pooled",
                   help="validation criterion: mean over factors of pooled or per-sample Pearson")
    p.add_argument("--es-metric", choices=["pearson", "loss"], default="pearson",
                   help="MLP early stopping: val Pearson (default) or val MSE (pilot behaviour)")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--cv-frozen-encoder", type=int, default=0, metavar="K",
                   help="also run K-fold patient-grouped CV (Ridge). ONLY for encoders never adapted "
                        "on any evaluated patient (Virchow2, pretrained OmiCLIP).")
    p.add_argument("--cv-inner", type=int, default=3)
    p.add_argument("--cv-seed", type=int, default=0)
    p.add_argument("--skip-standard", action="store_true")
    return p.parse_args()


def main():
    global TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES, ALL_SAMPLES, TECHNOLOGY_OF_SAMPLE
    args = parse_args()
    out_dir = args.out_dir or (PROJECT_DIR / "outputs" / "eval" / args.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_of_sample = {}
    if args.split_csv:
        TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES, TECHNOLOGY_OF_SAMPLE, fold_of_sample = load_split_csv(args.split_csv)
        ALL_SAMPLES = TRAIN_SAMPLES + VAL_SAMPLES + TEST_SAMPLES
        print(f"Split file: {args.split_csv} -> train {len(TRAIN_SAMPLES)}, val {len(VAL_SAMPLES)}, "
              f"test {len(TEST_SAMPLES)} slides"
              + (f"; technologies: {pd.Series(TECHNOLOGY_OF_SAMPLE).value_counts().to_dict()}" if TECHNOLOGY_OF_SAMPLE else ""))

    x, y, meta, factors, info = load_dataset(args.npz, args.meta, args.restrict_to)
    print(f"\n{args.name}: X {x.shape}, y {y.shape}, factors {factors}")

    patient_of_sample, source = load_patient_map(args.sample_table, ALL_SAMPLES)
    meta["patient_id"] = meta["sample_id"].map(patient_of_sample).fillna("UNKNOWN")
    leaks, n_patients = check_patient_leakage(patient_of_sample)
    print(f"Patient mapping: {source}")
    print(f"Patients per split: {n_patients}")
    if leaks:
        print("  WARNING: patients with samples in more than one split (leakage across splits):")
        for pid, parts in leaks.items():
            print(f"    {pid}: {parts}")
    else:
        print("  No patient appears in more than one split.")

    split = np.full(len(meta), "excluded", dtype=object)
    for name, group in (("train", TRAIN_SAMPLES), ("val", VAL_SAMPLES), ("test", TEST_SAMPLES)):
        split[meta["sample_id"].isin(group).to_numpy()] = name
    meta["split"] = split
    print("Rows per split:", meta["split"].value_counts().to_dict())
    info.update(patients_per_split=n_patients, patient_leaks=leaks, patient_source=source,
                rows_per_split=meta["split"].value_counts().to_dict())

    chosen = {}
    if not args.skip_standard:
        keep = (meta["split"] != "excluded").to_numpy()
        chosen.update(run_standard(x[keep], y[keep], meta.loc[keep].reset_index(drop=True), factors, args, out_dir))
    if args.cv_frozen_encoder > 0:
        chosen.update(run_cv(x, y, meta, factors, patient_of_sample, args, out_dir, fold_of_sample))

    (out_dir / "summary.json").write_text(
        json.dumps(dict(name=args.name, npz=str(args.npz), meta=str(args.meta),
                        args={k: str(v) for k, v in vars(args).items()}, info=info, chosen=chosen),
                   indent=2, default=str)
    )
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
