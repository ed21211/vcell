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
    h5ad_files = sorted(ST_DIR.glob("INT*.h5ad"))

    print(f"Found {len(h5ad_files)} ST files")

    all_rows = []

    for path in h5ad_files:
        sample_id = path.stem
        print(f"\nProcessing {sample_id}: {path}")

        adata = ad.read_h5ad(path)

        # ensure gene names are unique just in case
        adata.var_names_make_unique()

        # Normalise/log
        adata_pp = adata.copy()
        sc.pp.normalize_total(adata_pp, target_sum=1e4)
        sc.pp.log1p(adata_pp)

        rows = pd.DataFrame(index=adata_pp.obs_names)
        rows["sample_id"] = sample_id
        rows["barcode"] = adata_pp.obs_names.astype(str)

        if "spatial" in adata_pp.obsm:
            rows["spatial_x"] = adata_pp.obsm["spatial"][:, 0]
            rows["spatial_y"] = adata_pp.obsm["spatial"][:, 1]

        for score_name, genes in GENE_SETS.items():
            present = [g for g in genes if g in adata_pp.var_names]

            print(f"{score_name}: {len(present)}/{len(genes)} genes present: {present}")

            if len(present) == 0:
                rows[score_name] = np.nan
                continue

            sc.tl.score_genes(
                adata_pp,
                gene_list=present,
                score_name=score_name,
                use_raw=False,
            )

            rows[score_name] = adata_pp.obs[score_name].values

        out_csv = OUT_DIR / f"{sample_id}_factor_scores.csv"
        rows.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

        all_rows.append(rows.reset_index(drop=True))

    combined = pd.concat(all_rows, ignore_index=True)
    combined_out = OUT_DIR / "all_hest_ccrcc_factor_scores.csv"
    combined.to_csv(combined_out, index=False)

    print(f"\nSaved combined factor scores: {combined_out}")
    print(combined.shape)


if __name__ == "__main__":
    main()