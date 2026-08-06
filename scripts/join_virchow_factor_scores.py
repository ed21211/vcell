from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent

VIRCHOW_DIR = PROJECT_DIR / "features" / "virchow_hest_ccrcc"
FACTOR_CSV = PROJECT_DIR / "features" / "hest_factor_scores" / "all_hest_ccrcc_factor_scores.csv"

OUT_DIR = PROJECT_DIR / "features" / "hest_ccrcc_joined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ = OUT_DIR / "hest_ccrcc_virchow2_factor_dataset.npz"
OUT_META = OUT_DIR / "hest_ccrcc_virchow2_factor_metadata.csv"


FACTOR_COLUMNS = [
    "t_cell_infiltration",
    "cytotoxicity",
    "tgfb_caf",
    "proliferation",
]


def main():
    print("Loading factor scores...")
    factors = pd.read_csv(FACTOR_CSV)

    print("Factor table:", factors.shape)
    print(factors.head())

    all_embeddings = []
    all_targets = []
    all_metadata = []

    npz_files = sorted(VIRCHOW_DIR.glob("*_virchow.npz"))

    print(f"\nFound {len(npz_files)} Virchow files")

    for npz_path in npz_files:
        sample_id = npz_path.name.replace("_virchow.npz", "")
        print(f"\nProcessing {sample_id}")

        data = np.load(npz_path, allow_pickle=True)

        embeddings = data["embeddings"]
        barcodes = data["barcodes"].astype(str)
        coords = data["coords"]

        emb_df = pd.DataFrame({
            "sample_id": sample_id,
            "barcode": barcodes,
            "emb_index": np.arange(len(barcodes)),
            "patch_x": coords[:, 0],
            "patch_y": coords[:, 1],
        })

        sample_factors = factors[factors["sample_id"].astype(str) == sample_id].copy()

        merged = emb_df.merge(
            sample_factors,
            on=["sample_id", "barcode"],
            how="inner",
        )

        print("Embeddings:", embeddings.shape)
        print("Factor rows:", sample_factors.shape)
        print("Matched rows:", merged.shape)

        if merged.empty:
            print(f"WARNING: no matched rows for {sample_id}")
            continue

        emb_indices = merged["emb_index"].values

        X = embeddings[emb_indices]
        y = merged[FACTOR_COLUMNS].values.astype(np.float32)

        all_embeddings.append(X.astype(np.float32))
        all_targets.append(y)

        meta_cols = [
            "sample_id",
            "barcode",
            "patch_x",
            "patch_y",
            "spatial_x",
            "spatial_y",
        ] + FACTOR_COLUMNS

        all_metadata.append(merged[meta_cols])

    X_all = np.concatenate(all_embeddings, axis=0)
    y_all = np.concatenate(all_targets, axis=0)
    meta_all = pd.concat(all_metadata, ignore_index=True)

    np.savez_compressed(
        OUT_NPZ,
        X=X_all,
        y=y_all,
        factor_names=np.array(FACTOR_COLUMNS),
        sample_id=meta_all["sample_id"].values.astype(str),
        barcode=meta_all["barcode"].values.astype(str),
    )

    meta_all.to_csv(OUT_META, index=False)

    print("\nDONE")
    print("X:", X_all.shape)
    print("y:", y_all.shape)
    print("metadata:", meta_all.shape)
    print("Saved:", OUT_NPZ)
    print("Saved:", OUT_META)


if __name__ == "__main__":
    main()
    