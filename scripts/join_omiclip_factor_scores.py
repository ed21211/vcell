from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

OMICLIP_H5 = (
    PROJECT_DIR
    / "outputs"
    / "omiclip"
    / "ccrcc_omiclip_embeddings.h5"
)

OMICLIP_METADATA = (
    PROJECT_DIR
    / "outputs"
    / "omiclip"
    / "ccrcc_omiclip_metadata.csv"
)

FACTOR_SCORES = (
    PROJECT_DIR
    / "features"
    / "hest_factor_scores"
    / "all_hest_ccrcc_factor_scores.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "omiclip_factor_baseline"
)

FACTOR_COLUMNS = [
    "t_cell_infiltration",
    "cytotoxicity",
    "tgfb_caf",
    "proliferation",
]


def clean_sample_id(values):
    return (
        values.astype(str)
        .str.strip()
        .str.upper()
    )


def clean_barcode(values):
    return (
        values.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"-1$", "", regex=True)
    )


def main():
    required_files = [
        OMICLIP_H5,
        OMICLIP_METADATA,
        FACTOR_SCORES,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading OmiCLIP metadata...")
    omiclip_metadata = pd.read_csv(OMICLIP_METADATA)

    print("Loading factor scores...")
    factors = pd.read_csv(FACTOR_SCORES)

    required_metadata_columns = [
        "sample_id",
        "barcode",
        "embedding_row",
    ]

    for column in required_metadata_columns:
        if column not in omiclip_metadata.columns:
            raise KeyError(
                f"OmiCLIP metadata is missing '{column}'. "
                f"Available columns: {list(omiclip_metadata.columns)}"
            )

    required_factor_columns = [
        "sample_id",
        "barcode",
        *FACTOR_COLUMNS,
    ]

    for column in required_factor_columns:
        if column not in factors.columns:
            raise KeyError(
                f"Factor-score file is missing '{column}'. "
                f"Available columns: {list(factors.columns)}"
            )

    # Create consistent matching keys.
    omiclip_metadata["sample_key"] = clean_sample_id(
        omiclip_metadata["sample_id"]
    )

    omiclip_metadata["barcode_key"] = clean_barcode(
        omiclip_metadata["barcode"]
    )

    factors["sample_key"] = clean_sample_id(
        factors["sample_id"]
    )

    factors["barcode_key"] = clean_barcode(
        factors["barcode"]
    )

    # Check for duplicates before joining.
    omiclip_duplicates = omiclip_metadata.duplicated(
        ["sample_key", "barcode_key"]
    ).sum()

    factor_duplicates = factors.duplicated(
        ["sample_key", "barcode_key"]
    ).sum()

    print(f"OmiCLIP duplicate keys: {omiclip_duplicates}")
    print(f"Factor duplicate keys:   {factor_duplicates}")

    if omiclip_duplicates:
        raise ValueError(
            "Duplicate sample/barcode pairs found in OmiCLIP metadata."
        )

    if factor_duplicates:
        raise ValueError(
            "Duplicate sample/barcode pairs found in factor scores."
        )

    factor_subset = factors[
        [
            "sample_key",
            "barcode_key",
            *FACTOR_COLUMNS,
            *[
                column
                for column in ["spatial_x", "spatial_y"]
                if column in factors.columns
            ],
        ]
    ].copy()

    joined = omiclip_metadata.merge(
        factor_subset,
        on=["sample_key", "barcode_key"],
        how="inner",
        validate="one_to_one",
    )

    print("\nJoin summary")
    print(f"OmiCLIP rows: {len(omiclip_metadata):,}")
    print(f"Factor rows:  {len(factors):,}")
    print(f"Matched rows: {len(joined):,}")

    match_rate = len(joined) / len(omiclip_metadata)

    print(f"Match rate:   {match_rate:.2%}")

    if match_rate < 0.95:
        print(
            "Warning: fewer than 95% of OmiCLIP rows matched "
            "factor scores."
        )

    # Remove rows with missing biological targets.
    before_drop = len(joined)

    joined = joined.dropna(
        subset=FACTOR_COLUMNS
    ).copy()

    print(
        "Rows removed for missing factor scores: "
        f"{before_drop - len(joined):,}"
    )

    # Sort by embedding row so we can retrieve embeddings correctly.
    joined["embedding_row"] = joined["embedding_row"].astype(int)

    joined = joined.sort_values(
        "embedding_row"
    ).reset_index(drop=True)

    embedding_rows = joined["embedding_row"].to_numpy()

    print("\nLoading OmiCLIP image embeddings...")

    with h5py.File(OMICLIP_H5, "r") as handle:
        image_dataset = handle["image_embeddings"]

        print(
            "Full image embedding shape:",
            image_dataset.shape,
        )

        if embedding_rows.max() >= image_dataset.shape[0]:
            raise IndexError(
                "An embedding_row value exceeds the HDF5 "
                "embedding count."
            )

        # h5py requires increasing row indices for indexed reading.
        X = image_dataset[embedding_rows].astype(np.float32)

    y = joined[FACTOR_COLUMNS].to_numpy(
        dtype=np.float32
    )

    print("\nFinal dataset")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Samples:", joined["sample_id"].nunique())

    dataset_path = (
        OUTPUT_DIR
        / "omiclip_factor_dataset.npz"
    )

    metadata_path = (
        OUTPUT_DIR
        / "omiclip_factor_metadata.csv"
    )

    np.savez_compressed(
        dataset_path,
        X=X,
        y=y,
        factor_names=np.asarray(FACTOR_COLUMNS),
    )

    metadata_columns = [
        column
        for column in [
            "sample_id",
            "patient_id",
            "barcode",
            "embedding_row",
            "matched_similarity",
            "spatial_x",
            "spatial_y",
            *FACTOR_COLUMNS,
        ]
        if column in joined.columns
    ]

    joined[metadata_columns].to_csv(
        metadata_path,
        index=False,
    )

    print(f"\nSaved dataset:  {dataset_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()