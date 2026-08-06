from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


PROJECT_DIR = Path(__file__).resolve().parent.parent
ST_DIR = PROJECT_DIR / "datasets" / "hest_ccrcc" / "st"
OUT_DIR = PROJECT_DIR / "features" / "hest_factor_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)


GENE_SETS = {
    "t_cell_infiltration": ["CD3D", "CD3E", "CD2", "CD8A", "CD8B"],
    "cytotoxicity": ["GZMB", "PRF1", "NKG7", "GNLY"],
    "tgfb_caf": ["TGFB1", "COL1A1", "COL1A2", "FAP", "ACTA2", "CTGF"],
    "proliferation": ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM5"],
}


def main():
    h5ad_files = sorted(
        ST_DIR.glob("INT*.h5ad"),
        key=lambda path: int(path.stem.replace("INT", "")),
    )

    print(f"Found {len(h5ad_files)} ST files", flush=True)

    completed_files = []

    for path in h5ad_files:
        sample_id = path.stem
        out_csv = OUT_DIR / f"{sample_id}_factor_scores.csv"

        # Avoid recomputing completed samples.
        if out_csv.exists():
            print(f"\nSkipping {sample_id}: output already exists", flush=True)
            completed_files.append(out_csv)
            continue

        print(f"\nLoading {sample_id}: {path}", flush=True)

        # Reading the sparse expression matrix can take some time.
        adata = ad.read_h5ad(path)

        print(
            f"Loaded {sample_id}: "
            f"{adata.n_obs:,} spots × {adata.n_vars:,} genes",
            flush=True,
        )

        adata.var_names_make_unique()

        # Process in place instead of making a full copy.
        print("Normalising counts...", flush=True)
        sc.pp.normalize_total(adata, target_sum=1e4)

        print("Applying log1p...", flush=True)
        sc.pp.log1p(adata)

        rows = pd.DataFrame(
            {
                "sample_id": sample_id,
                "barcode": adata.obs_names.astype(str),
            }
        )

        if "spatial" in adata.obsm:
            rows["spatial_x"] = adata.obsm["spatial"][:, 0]
            rows["spatial_y"] = adata.obsm["spatial"][:, 1]

        for score_name, genes in GENE_SETS.items():
            present = [
                gene for gene in genes
                if gene in adata.var_names
            ]

            print(
                f"{score_name}: "
                f"{len(present)}/{len(genes)} genes present: "
                f"{present}",
                flush=True,
            )

            if not present:
                rows[score_name] = np.nan
                continue

            sc.tl.score_genes(
                adata,
                gene_list=present,
                score_name=score_name,
                use_raw=False,
                random_state=0,
            )

            rows[score_name] = adata.obs[score_name].to_numpy()

        rows.to_csv(out_csv, index=False)
        completed_files.append(out_csv)

        print(f"Saved: {out_csv}", flush=True)

        # Release memory before loading the next sample.
        del rows
        del adata

        import gc
        gc.collect()

    print("\nCombining completed sample files...", flush=True)

    all_rows = [
        pd.read_csv(path)
        for path in completed_files
        if path.exists()
    ]

    if not all_rows:
        raise RuntimeError("No factor-score files were produced.")

    combined = pd.concat(all_rows, ignore_index=True)

    combined_out = (
        OUT_DIR
        / "all_hest_ccrcc_factor_scores.csv"
    )

    combined.to_csv(combined_out, index=False)

    print(f"Saved combined factor scores: {combined_out}")
    print(f"Combined shape: {combined.shape}")

if __name__ == "__main__":
    main()