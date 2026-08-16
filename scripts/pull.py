"""
Pull presentation-ready example images from your HEST ccRCC data:
  1. A few raw H&E patches (just the tissue image)
  2. A full-slide H&E image with ST spot locations overlaid
  3. A full-slide H&E image with one factor score (e.g. TGF-B/CAF) overlaid as a heatmap

Run this ON YOUR MACHINE, from your project root (e.g. /media/rokny/DATA2/Elsa/vcell),
inside your Virchow/HEST venv (needs h5py, scanpy/anndata, numpy, matplotlib, Pillow).

Usage:
    python pull_hest_examples.py --sample INT1 --n-patches 6

Adjust SAMPLE_ID and paths below if your folder layout differs from the SKILL.md paths.
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    import anndata as ad
except ImportError:
    ad = None


def load_patches(patch_h5_path, n=6):
    """Grab the first n H&E patches from the HEST patch file and save them individually."""
    with h5py.File(patch_h5_path, "r") as f:
        # HEST patch files typically store images under a key like 'img' or 'images'
        # print the keys so you can confirm/adjust if this differs
        print("Patch file keys:", list(f.keys()))
        img_key = "img" if "img" in f else list(f.keys())[0]
        imgs = f[img_key][:n]
        barcode_key = "barcode" if "barcode" in f else None
        barcodes = f[barcode_key][:n] if barcode_key else [f"patch_{i}" for i in range(n)]
    return imgs, barcodes


def save_patch_grid(imgs, barcodes, out_path):
    """Save a labeled grid of individual H&E patches — good for a 'what the data looks like' slide."""
    n = len(imgs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for i, (img, bc) in enumerate(zip(imgs, barcodes)):
        axes[i].imshow(img)
        label = bc.decode() if isinstance(bc, bytes) else str(bc)
        axes[i].set_title(label, fontsize=10)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved patch grid: {out_path}")


def save_spot_overlay(st_h5ad_path, out_path, sample_id):
    """Full tissue image with ST spot coordinates overlaid — shows what 'spatial' means visually."""
    if ad is None:
        print("anndata not installed — skipping spot overlay. pip install anndata scanpy")
        return
    adata = ad.read_h5ad(st_h5ad_path)
    coords = adata.obsm.get("spatial")
    if coords is None:
        print("No 'spatial' key in obsm — check your .h5ad structure.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    # If a full-res H&E image is stored in adata.uns['spatial'], plot it under the spots
    img = None
    try:
        lib_key = list(adata.uns["spatial"].keys())[0]
        img = adata.uns["spatial"][lib_key]["images"].get("hires")
    except Exception:
        pass

    if img is not None:
        ax.imshow(img)
    ax.scatter(coords[:, 0], coords[:, 1], s=4, c="red", alpha=0.5)
    ax.set_title(f"{sample_id}: ST spots over H&E ({coords.shape[0]} spots)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved spot overlay: {out_path}")


def save_factor_overlay(factor_csv_path, st_h5ad_path, out_path, sample_id, factor_col="tgfb_caf"):
    """Full tissue image with one factor score as a color heatmap — good for showing true biology
    before/alongside your predicted maps."""
    import pandas as pd

    if ad is None:
        print("anndata not installed — skipping factor overlay.")
        return

    df = pd.read_csv(factor_csv_path)
    df = df[df["sample_id"] == sample_id]
    if df.empty or factor_col not in df.columns:
        print(f"No rows for {sample_id} or column '{factor_col}' not found. "
              f"Available columns: {df.columns.tolist()}")
        return

    # your factor-score script stores coords as spatial_x / spatial_y
    x_col = "spatial_x" if "spatial_x" in df.columns else "x"
    y_col = "spatial_y" if "spatial_y" in df.columns else "y"

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(df[x_col], df[y_col],
                     c=df[factor_col], cmap="viridis", s=8)
    plt.colorbar(sc, ax=ax, label=factor_col)
    ax.set_title(f"{sample_id}: {factor_col} spatial distribution")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved factor overlay: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="INT1", help="HEST sample id, e.g. INT1")
    parser.add_argument("--n-patches", type=int, default=6)
    parser.add_argument("--patches-dir", default="datasets/hest_ccrcc/patches")
    parser.add_argument("--st-dir", default="datasets/hest_ccrcc/st")
    parser.add_argument("--factor-csv", default="features/hest_factor_scores/all_hest_ccrcc_factor_scores.csv")
    parser.add_argument("--out-dir", default="outputs/presentation_examples")
    parser.add_argument("--factor-col", default="tgfb_caf",
                         help="one of: t_cell_infiltration, cytotoxicity, tgfb_caf, proliferation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    patch_path = os.path.join(args.patches_dir, f"{args.sample}.h5")
    st_path = os.path.join(args.st_dir, f"{args.sample}.h5ad")

    if os.path.exists(patch_path):
        imgs, barcodes = load_patches(patch_path, n=args.n_patches)
        save_patch_grid(imgs, barcodes, os.path.join(args.out_dir, f"{args.sample}_patch_grid.png"))
    else:
        print(f"Patch file not found: {patch_path} — adjust --patches-dir")

    if os.path.exists(st_path):
        save_spot_overlay(st_path, os.path.join(args.out_dir, f"{args.sample}_spot_overlay.png"), args.sample)
    else:
        print(f"ST file not found: {st_path} — adjust --st-dir")

    if os.path.exists(args.factor_csv) and os.path.exists(st_path):
        save_factor_overlay(
            args.factor_csv, st_path,
            os.path.join(args.out_dir, f"{args.sample}_{args.factor_col}_overlay.png"),
            args.sample, factor_col=args.factor_col,
        )
    else:
        print(f"Factor CSV or ST file not found — skipping factor overlay")

    print(f"\nDone. Check: {args.out_dir}/")