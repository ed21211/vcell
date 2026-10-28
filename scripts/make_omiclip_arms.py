#!/usr/bin/env python
"""
make_omiclip_arms.py -- build the ST-only and combined (H&E + ST) datasets from
OmiCLIP embeddings that already exist on disk.

extract_omiclip_embeddings.py saved BOTH towers for every spot
(`image_embeddings` and `st_embeddings`, L2-normalised, 768-d). The join script
join_omiclip_factor_scores.py kept only the image tower. This script reads
`st_embeddings` for the same rows, in the same order, and writes

    <out-dir>/<prefix>_st_factor_dataset.npz     X = text tower on the gene sentence  (ST-only arm)
    <out-dir>/<prefix>_both_factor_dataset.npz   X = [image | text], 1536-d           (combined arm)

The metadata CSV does not change: pass omiclip_factor_metadata.csv as --meta to
evaluate_embeddings.py, because the row order is identical.

    python scripts/make_omiclip_arms.py                       # native top-50 sentences
    python scripts/make_omiclip_arms.py --h5 <text-only.h5> --image-h5 outputs/omiclip/ccrcc_omiclip_embeddings.h5 \
        --prefix omiclip_sigmasked

Checks before writing: the image embeddings in the existing npz must equal the
h5 image embeddings at the same rows (so the text rows are the right spots), all
values must be finite, and the text embeddings must be unit length.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE = PROJECT_DIR / "outputs" / "omiclip_factor_baseline"


def read_rows(dataset, rows):
    """h5py needs strictly increasing indices; read sorted, return in the requested order."""
    rows = np.asarray(rows, dtype=np.int64)
    if len(np.unique(rows)) != len(rows):
        raise SystemExit("embedding_row has duplicates; the metadata file is not one row per spot.")
    order = np.argsort(rows)
    out = np.empty((len(rows), dataset.shape[1]), dtype=np.float32)
    out[order] = dataset[rows[order]].astype(np.float32)
    return out


def x_st_rows_total(path):
    with h5py.File(path, "r") as handle:
        return handle["st_embeddings"].shape[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", type=Path, default=PROJECT_DIR / "outputs" / "omiclip" / "ccrcc_omiclip_embeddings.h5",
                   help="h5 holding st_embeddings (and image_embeddings unless --image-h5 is given)")
    p.add_argument("--image-h5", type=Path, default=None,
                   help="h5 holding the image embeddings, for the alignment check, when --h5 has only st_embeddings")
    p.add_argument("--meta", type=Path, default=BASE / "omiclip_factor_metadata.csv")
    p.add_argument("--image-npz", type=Path, default=BASE / "omiclip_factor_dataset.npz")
    p.add_argument("--out-dir", type=Path, default=BASE)
    p.add_argument("--prefix", default="omiclip")
    p.add_argument("--rebuild-image", action="store_true",
                   help="take the image embeddings from the h5 too (fine-tuned checkpoint) and also write "
                        "<prefix>_image_factor_dataset.npz; y and factor names still come from --image-npz")
    args = p.parse_args()

    meta = pd.read_csv(args.meta)
    if "embedding_row" not in meta.columns:
        raise SystemExit(f"{args.meta} has no embedding_row column.")
    rows = meta["embedding_row"].to_numpy(dtype=np.int64)

    base = np.load(args.image_npz, allow_pickle=True)
    x_img, y, factors = base["X"], base["y"], base["factor_names"]
    if not (len(x_img) == len(y) == len(meta)):
        raise SystemExit(f"Row mismatch: image npz {len(x_img)}, y {len(y)}, metadata {len(meta)}.")

    image_path = args.image_h5 or args.h5
    with h5py.File(args.h5, "r") as handle:
        if "st_embeddings" not in handle:
            raise SystemExit(f"{args.h5} has no 'st_embeddings'. Keys: {list(handle.keys())}")
        if rows.max() >= handle["st_embeddings"].shape[0]:
            raise SystemExit("embedding_row points past the end of the h5 file.")
        x_st = read_rows(handle["st_embeddings"], rows)
    # alignment check on a random subset of rows, against the file that holds the image embeddings
    with h5py.File(image_path, "r") as handle:
        if "image_embeddings" not in handle:
            raise SystemExit(f"{image_path} has no 'image_embeddings'. Keys: {list(handle.keys())}. "
                             "Pass --image-h5 <the original embeddings h5>.")
        if handle["image_embeddings"].shape[0] != x_st_rows_total(args.h5):
            raise SystemExit(f"{image_path} and {args.h5} have different numbers of rows; "
                             "the sentence file does not line up with the image file.")
        if args.rebuild_image:
            # fine-tuned image tower: the old npz image embeddings no longer apply, take them from the h5
            x_img = read_rows(handle["image_embeddings"], rows)
            if "sample_id" in handle and "barcode" in handle and {"sample_id", "barcode"} <= set(meta.columns):
                def decode(arr):
                    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in arr[...]], dtype=object)[rows]
                def clean_id(values):
                    return pd.Series(values).astype(str).str.strip().str.upper().to_numpy()

                def clean_bc(values):
                    return pd.Series(values).astype(str).str.strip().str.upper().str.replace(r"-1$", "", regex=True).to_numpy()

                same_sample = float((clean_id(decode(handle["sample_id"])) == clean_id(meta["sample_id"])).mean())
                same_barcode = float((clean_bc(decode(handle["barcode"])) == clean_bc(meta["barcode"])).mean())
                print(f"Row-order check against the h5: sample_id match {same_sample:.4f}, barcode match {same_barcode:.4f}")
                if same_sample < 1.0 or same_barcode < 0.999:
                    raise SystemExit("The h5 rows do not line up with the metadata file; not writing anything.")
            else:
                print("WARNING: no sample_id/barcode to verify the row order against (h5 or metadata lacks them).")
        else:
            rng = np.random.RandomState(0)
            probe = np.sort(rng.choice(len(rows), size=min(2000, len(rows)), replace=False))
            img_h5 = read_rows(handle["image_embeddings"], rows[probe])
    if not args.rebuild_image:
        max_diff = float(np.abs(img_h5 - x_img[probe]).max())
        print(f"Alignment check: max |image npz - image h5| on {len(probe)} rows = {max_diff:.2e}")
        if max_diff > 1e-4:
            raise SystemExit("Image embeddings in the npz do not match the h5 at these rows; the row order is wrong.")
    if not np.isfinite(x_img).all():
        raise SystemExit("Non-finite values in the image embeddings.")

    if not np.isfinite(x_st).all():
        raise SystemExit("Non-finite values in st_embeddings.")
    norms = np.linalg.norm(x_st, axis=1)
    print(f"Text embedding norms: min {norms.min():.4f}, max {norms.max():.4f} (expect 1.0)")
    if np.abs(norms - 1).max() > 1e-3:
        raise SystemExit("Text embeddings are not unit length; check the extraction.")

    x_both = np.concatenate([x_img.astype(np.float32), x_st], axis=1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    st_path = args.out_dir / f"{args.prefix}_st_factor_dataset.npz"
    both_path = args.out_dir / f"{args.prefix}_both_factor_dataset.npz"
    np.savez_compressed(st_path, X=x_st, y=y, factor_names=factors)
    np.savez_compressed(both_path, X=x_both, y=y, factor_names=factors)
    print(f"Wrote {st_path}  X {x_st.shape}")
    print(f"Wrote {both_path}  X {x_both.shape}")
    if args.rebuild_image:
        img_path = args.out_dir / f"{args.prefix}_image_factor_dataset.npz"
        np.savez_compressed(img_path, X=x_img.astype(np.float32), y=y, factor_names=factors)
        print(f"Wrote {img_path}  X {x_img.shape}")
    print(f"Use --meta {args.meta} for both (same row order).")


if __name__ == "__main__":
    main()
