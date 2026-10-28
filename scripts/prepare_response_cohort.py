#!/usr/bin/env python
"""
prepare_response_cohort.py -- turn the SpaceRanger folders of the JITC mRCC cohort
(Kalogirou et al. 2025, Zenodo 10.5281/zenodo.16833780, Visium_Objects.zip) into the same
objects the HEST pipeline uses: H&E patches, gene sentences, a pairs CSV, measured factor scores.

Input: the unzipped Visium_Objects folder, one sub-folder per slide, e.g. RCC_W47_LYM/ with
    filtered_feature_bc_matrix.h5
    spatial/tissue_hires_image.png, spatial/scalefactors_json.json, spatial/tissue_positions.csv

    python scripts/prepare_response_cohort.py --root /media/rokny/DATA2/Elsa/response_cohorts/jitc/Visium_Objects

Writes (under --out-dir, default datasets/jitc_mrcc):
    patches/<slide>.h5          img (N,224,224,3) uint8, the key the HEST extractors read
    jitc_pairs.csv              sample_id, patient_id, barcode, patch_h5_path, patch_index, gene_sentence, gene_count,
                                tissue (PRIM or MET), patient, x, y   (feeds extract_omiclip_ckpt.py unchanged)
    jitc_factor_scores.csv      sample_id, barcode, the four measured factor scores (Stage 1 definition)
    slides.csv                  one row per slide: spots, resolution, genes

IMPORTANT, resolution: SpaceRanger only ships the downsized image (largest side about 2000 px), about 3 um
per pixel, so one 55 um spot is about 17 px wide. HEST patches are 224 px at about 0.5 um/px (112 um). Here
a 112 um field is cut from the small image and enlarged to 224 px, which makes it blurry. The H&E arm is
expected to be weaker on this cohort because of that. The resolution of every slide is printed and saved.

Stage 1 definitions reused: gene sentence = top 50 genes by raw count (prepare_omiclip_pairs.make_gene_sentence),
spots with no expressed gene are dropped; factor scores = library-size normalise to 10,000, log1p, scanpy
score_genes with the four marker sets (a close re-implementation is used when scanpy is not installed).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse

from prepare_omiclip_pairs import get_expression_matrix, get_gene_names, make_gene_sentence

PROJECT_DIR = Path(__file__).resolve().parents[1]
SPOT_DIAMETER_UM = 55.0
PATCH_PX = 224

FACTORS = {
    "t_cell_infiltration": ["CD3D", "CD3E", "CD2", "CD8A", "CD8B"],
    "cytotoxicity": ["GZMB", "PRF1", "NKG7", "GNLY"],
    "tgfb_caf": ["TGFB1", "COL1A1", "COL1A2", "FAP", "ACTA2"],
    "proliferation": ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM5"],
}

NAME_RE = re.compile(r"^RCC_(W\d+)_(PRIM|MET|LYM)(?:_([A-Z0-9]+))?$")


def parse_slide_name(name):
    m = NAME_RE.match(name)
    if not m:
        return None
    patient, kind, _ = m.groups()
    return patient, ("PRIM" if kind == "PRIM" else "MET")


def read_10x_h5(path):
    """10x filtered_feature_bc_matrix.h5 -> (counts CSR spots x genes, barcodes, gene symbols)."""
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        shape = tuple(g["shape"][:])  # genes x barcodes
        mat = sparse.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
        barcodes = np.array([b.decode() if isinstance(b, bytes) else str(b) for b in g["barcodes"][:]])
        names = np.array([n.decode() if isinstance(n, bytes) else str(n) for n in g["features"]["name"][:]])
    return mat.T.tocsr().astype(np.float32), barcodes, names


def read_positions(path):
    cols = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    df = pd.read_csv(path)
    if "barcode" not in df.columns:  # old SpaceRanger: no header
        df = pd.read_csv(path, header=None, names=cols)
    return df.set_index("barcode")


def cut_patches(image, centres_xy, half_px):
    """Square crops of side 2*half_px around (x, y) in image pixels, padded with white, resized to 224."""
    pad = half_px + 2
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), constant_values=255)
    out = np.empty((len(centres_xy), PATCH_PX, PATCH_PX, 3), dtype=np.uint8)
    for i, (x, y) in enumerate(centres_xy):
        cx, cy = int(round(x)) + pad, int(round(y)) + pad
        crop = padded[cy - half_px:cy + half_px, cx - half_px:cx + half_px]
        out[i] = np.asarray(Image.fromarray(crop).resize((PATCH_PX, PATCH_PX), Image.BICUBIC))
    return out


def score_genes_fallback(x_log, gene_names, genes, n_bins=25, ctrl_size=50, seed=0):
    """Close re-implementation of scanpy.tl.score_genes (mean of the genes minus mean of binned control genes)."""
    rng = np.random.RandomState(seed)
    name_to_idx = {n: i for i, n in enumerate(gene_names)}
    idx = [name_to_idx[g] for g in genes if g in name_to_idx]
    if not idx:
        return np.full(x_log.shape[0], np.nan, dtype=np.float32)
    mean_expr = np.asarray(x_log.mean(axis=0)).ravel()
    order = np.argsort(mean_expr, kind="stable")
    bins = np.array_split(order, n_bins)
    gene_bin = np.empty(len(gene_names), dtype=int)
    for b, members in enumerate(bins):
        gene_bin[members] = b
    ctrl = set()
    for g in idx:
        members = [m for m in bins[gene_bin[g]] if m not in idx]
        if members:
            ctrl.update(rng.choice(members, size=min(ctrl_size, len(members)), replace=False).tolist())
    ctrl = sorted(ctrl - set(idx))
    gene_mean = np.asarray(x_log[:, idx].mean(axis=1)).ravel()
    ctrl_mean = np.asarray(x_log[:, ctrl].mean(axis=1)).ravel() if ctrl else 0.0
    return (gene_mean - ctrl_mean).astype(np.float32)


def factor_scores(counts, gene_names):
    """Stage 1 definition: normalise to 10,000, log1p, score_genes per marker set."""
    totals = np.asarray(counts.sum(axis=1)).ravel()
    totals[totals == 0] = 1.0
    norm = sparse.diags(1e4 / totals) @ counts
    norm = norm.tocsr()
    norm.data = np.log1p(norm.data)
    try:
        import scanpy as sc

        adata = ad.AnnData(X=norm.copy(), var=pd.DataFrame(index=pd.Index(gene_names, name="gene")))
        adata.var_names_make_unique()
        out = {}
        for name, genes in FACTORS.items():
            present = [g for g in genes if g in set(adata.var_names)]
            sc.tl.score_genes(adata, gene_list=present, score_name=name)
            out[name] = adata.obs[name].to_numpy(dtype=np.float32)
        return out, "scanpy"
    except ImportError:
        return {name: score_genes_fallback(norm, gene_names, genes) for name, genes in FACTORS.items()}, "fallback"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True, help="the unzipped Visium_Objects folder")
    p.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "datasets" / "jitc_mrcc")
    p.add_argument("--fov-um", type=float, default=112.0,
                   help="field of view of one patch; 112 um = 224 px at 0.5 um/px, the HEST setting (check pixel_size_um_estimated)")
    p.add_argument("--top-genes", type=int, default=50)
    p.add_argument("--max-slides", type=int, default=None, help="test on the first N slides")
    args = p.parse_args()

    folders = sorted(d for d in args.root.iterdir() if d.is_dir() and d.name.startswith("RCC_"))
    if not folders:
        raise SystemExit(f"No RCC_* folders under {args.root}")
    if args.max_slides:
        folders = folders[:args.max_slides]
    patch_dir = args.out_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    pairs_rows, factor_rows, slide_rows = [], [], []
    scoring = None
    for folder in folders:
        parsed = parse_slide_name(folder.name)
        if parsed is None:
            print(f"skip {folder.name}: name not recognised")
            continue
        patient, tissue = parsed
        spatial = folder / "spatial"
        scale_json = json.loads((spatial / "scalefactors_json.json").read_text())
        scale = float(scale_json["tissue_hires_scalef"])
        spot_px_full = float(scale_json["spot_diameter_fullres"])
        um_per_px = SPOT_DIAMETER_UM / (spot_px_full * scale)
        half_px = max(4, int(round(args.fov_um / um_per_px / 2)))

        counts, barcodes, gene_names = read_10x_h5(folder / "filtered_feature_bc_matrix.h5")
        pos = read_positions(spatial / "tissue_positions.csv")
        keep = [b in pos.index for b in barcodes]
        if not all(keep):
            print(f"  {folder.name}: {int((~np.array(keep)).sum())} barcodes without a position, dropped")
        keep = np.array(keep)
        counts, barcodes = counts[keep], barcodes[keep]
        pos = pos.loc[barcodes]

        adata = ad.AnnData(X=counts, var=pd.DataFrame(index=pd.Index(gene_names, name="gene")))
        names = get_gene_names(adata)
        matrix = get_expression_matrix(adata)
        sentences, n_genes = [], []
        for i in range(len(barcodes)):
            s, n = make_gene_sentence(matrix, i, names, args.top_genes)
            sentences.append(s)
            n_genes.append(n)
        sentences = np.array(sentences, dtype=object)
        usable = np.array([s != "" for s in sentences])

        image = np.asarray(Image.open(spatial / "tissue_hires_image.png").convert("RGB"))
        xy = np.column_stack([pos["pxl_col_in_fullres"].to_numpy() * scale, pos["pxl_row_in_fullres"].to_numpy() * scale])
        patches = cut_patches(image, xy[usable], half_px)
        patch_path = patch_dir / f"{folder.name}.h5"
        with h5py.File(patch_path, "w") as f:
            f.create_dataset("img", data=patches, chunks=(1, PATCH_PX, PATCH_PX, 3), compression="lzf")

        scores, scoring = factor_scores(counts, names)
        n_ok = int(usable.sum())
        for j, i in enumerate(np.flatnonzero(usable)):
            pairs_rows.append(dict(
                sample_id=folder.name, patient_id=patient, barcode=barcodes[i], patch_h5_path=str(patch_path.resolve()),
                patch_index=j, gene_sentence=sentences[i], gene_count=n_genes[i], tissue=tissue, patient=patient,
                x=float(xy[i, 0]), y=float(xy[i, 1])))
            factor_rows.append(dict(sample_id=folder.name, barcode=barcodes[i], **{k: float(v[i]) for k, v in scores.items()}))
        slide_rows.append(dict(sample_id=folder.name, patient=patient, tissue=tissue, spots_in_matrix=len(barcodes),
                               spots_used=n_ok, genes=len(names), um_per_px_hires=round(um_per_px, 3),
                               spot_px_hires=round(spot_px_full * scale, 1), crop_px=2 * half_px,
                               image_h=image.shape[0], image_w=image.shape[1]))
        print(f"{folder.name}: {n_ok:,}/{len(barcodes):,} spots | {um_per_px:.2f} um/px | spot {spot_px_full * scale:.1f} px | "
              f"crop {2 * half_px} px -> {PATCH_PX} px | image {image.shape[1]}x{image.shape[0]}", flush=True)

    pairs = pd.DataFrame(pairs_rows)
    pairs.to_csv(args.out_dir / "jitc_pairs.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(args.out_dir / "jitc_factor_scores.csv", index=False)
    slides = pd.DataFrame(slide_rows)
    slides.to_csv(args.out_dir / "slides.csv", index=False)
    print(f"\nFactor scores via {scoring}. {len(pairs):,} spots on {len(slides)} slides "
          f"({(slides.tissue == 'PRIM').sum()} primary, {(slides.tissue == 'MET').sum()} metastasis).")
    print(f"Median resolution {slides.um_per_px_hires.median():.2f} um/px (HEST: about 0.5). Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
