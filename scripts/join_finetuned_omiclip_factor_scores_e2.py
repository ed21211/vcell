from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

EMBEDDINGS_H5 = (
    PROJECT_DIR
    / "outputs"
    / "omiclip_finetuned_e2_lr1e6"
    / "ccrcc_omiclip_finetuned_embeddings.h5"
)

EMBEDDING_METADATA = (
    PROJECT_DIR
    / "outputs"
    / "omiclip_finetuned_e2_lr1e6"
    / "ccrcc_omiclip_finetuned_metadata.csv"
)

FACTOR_SCORES = (
    PROJECT_DIR
    / "features"
    / "hest_factor_scores"
    / "all_hest_ccrcc_factor_scores.csv"
)

OUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "omiclip_finetuned_e2_lr1e6_factor_baseline"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_NPZ = OUT_DIR / "omiclip_finetuned_factor_dataset.npz"
OUTPUT_METADATA = OUT_DIR / "omiclip_finetuned_factor_metadata.csv"


FACTOR_COLUMNS = [
    "t_cell_infiltration",
    "cytotoxicity",
    "tgfb_caf",
    "proliferation",
]


def main():

    print("Loading fine-tuned embedding metadata...")

    metadata = pd.read_csv(
        EMBEDDING_METADATA
    )

    print(
        "Embedding metadata:",
        metadata.shape
    )

    print("\nLoading fine-tuned embeddings...")

    with h5py.File(
        EMBEDDINGS_H5,
        "r",
    ) as handle:

        print(
            "H5 keys:",
            list(handle.keys())
        )

        X = handle[
            "image_embeddings"
        ][:].astype(
            np.float32
        )

    print(
        "Embeddings:",
        X.shape
    )

    if len(metadata) != len(X):
        raise RuntimeError(
            f"Metadata rows ({len(metadata)}) "
            f"!= embedding rows ({len(X)})"
        )

    print("\nLoading factor scores...")

    factors = pd.read_csv(
        FACTOR_SCORES
    )

    print(
        "Factor scores:",
        factors.shape
    )

    print(
        "Factor columns:",
        list(factors.columns)
    )

    missing_factors = [
        column
        for column in FACTOR_COLUMNS
        if column not in factors.columns
    ]

    if missing_factors:
        raise ValueError(
            f"Missing factor columns: "
            f"{missing_factors}"
        )

    # --------------------------------------------------
    # Normalise identifiers
    # --------------------------------------------------

    metadata[
        "sample_id"
    ] = (
        metadata["sample_id"]
        .astype(str)
        .str.upper()
    )

    metadata[
        "barcode"
    ] = (
        metadata["barcode"]
        .astype(str)
    )

    factors[
        "sample_id"
    ] = (
        factors["sample_id"]
        .astype(str)
        .str.upper()
    )

    factors[
        "barcode"
    ] = (
        factors["barcode"]
        .astype(str)
    )

    # Preserve exact embedding row order.
    metadata[
        "_embedding_index"
    ] = np.arange(
        len(metadata)
    )

    print("\nJoining by sample_id + barcode...")

    joined = metadata.merge(
        factors[
            [
                "sample_id",
                "barcode",
            ]
            + FACTOR_COLUMNS
        ],
        on=[
            "sample_id",
            "barcode",
        ],
        how="left",
        validate="one_to_one",
    )

    joined = joined.sort_values(
        "_embedding_index"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------
    # Check missing targets
    # --------------------------------------------------

    missing_mask = (
        joined[
            FACTOR_COLUMNS
        ]
        .isna()
        .any(axis=1)
    )

    n_missing = int(
        missing_mask.sum()
    )

    print(
        "Rows missing factor scores:",
        n_missing
    )

    if n_missing > 0:

        print(
            joined.loc[
                missing_mask,
                [
                    "sample_id",
                    "barcode",
                ],
            ].head(20)
        )

        raise RuntimeError(
            "Some embeddings could not "
            "be matched to factor scores."
        )

    # --------------------------------------------------
    # Create targets
    # --------------------------------------------------

    y = (
        joined[
            FACTOR_COLUMNS
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    print("\nFinal dataset:")

    print(
        "X:",
        X.shape
    )

    print(
        "y:",
        y.shape
    )

    print(
        "metadata:",
        joined.shape
    )

    print(
        "Samples:",
        joined[
            "sample_id"
        ].nunique()
    )

    # --------------------------------------------------
    # Safety checks
    # --------------------------------------------------

    if not np.isfinite(
        X
    ).all():
        raise RuntimeError(
            "Non-finite values found "
            "in embeddings."
        )

    if not np.isfinite(
        y
    ).all():
        raise RuntimeError(
            "Non-finite values found "
            "in targets."
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        y=y,
        factor_names=np.array(
            FACTOR_COLUMNS
        ),
    )

    joined.drop(
        columns=[
            "_embedding_index"
        ],
        errors="ignore",
    ).to_csv(
        OUTPUT_METADATA,
        index=False,
    )

    print(
        "\nSaved:",
        OUTPUT_NPZ
    )

    print(
        "Saved:",
        OUTPUT_METADATA
    )

    print(
        "\nFine-tuned OmiCLIP "
        "factor dataset ready."
    )


if __name__ == "__main__":
    main()