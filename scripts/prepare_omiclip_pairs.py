import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


SAMPLE_COLUMN_CANDIDATES = [
    "id",
    "sample_id",
    "sample",
]

PATIENT_COLUMN_CANDIDATES = [
    "patient_id",
    "patient",
    "subject_id",
    "donor_id",
    "individual_id",
    "case_id",
]

GENE_COLUMN_CANDIDATES = [
    "gene_name",
    "gene_symbol",
    "symbol",
    "feature_name",
]


def find_first_column(df, candidates):
    return next(
        (column for column in candidates if column in df.columns),
        None,
    )


def decode_value(value):
    """Convert HDF5 byte/string values to a normal Python string."""
    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")

    if isinstance(value, np.ndarray) and value.size == 1:
        return decode_value(value.item())

    return str(value)


def normalise_barcode(barcode):
    """
    Normalise barcodes for matching.

    This removes a trailing '-1' because some files retain the Visium
    suffix while others omit it.
    """
    barcode = decode_value(barcode).strip()

    if barcode.endswith("-1"):
        barcode = barcode[:-2]

    return barcode


def get_gene_names(adata):
    """Return gene symbols where available, otherwise use var_names."""
    for column in GENE_COLUMN_CANDIDATES:
        if column in adata.var.columns:
            names = adata.var[column].fillna("").astype(str).to_numpy()
            print(f"  Gene names taken from adata.var['{column}']")
            return names

    print("  Gene names taken from adata.var_names")
    return adata.var_names.astype(str).to_numpy()


def get_expression_matrix(adata):
    """
    Prefer a raw-count layer when available.

    Ranking is unaffected by library-size normalisation, but raw counts
    provide a simple and interpretable input for selecting top genes.
    """
    for layer_name in ["counts", "raw_counts", "count"]:
        if layer_name in adata.layers:
            print(f"  Expression source: adata.layers['{layer_name}']")
            return adata.layers[layer_name]

    print("  Expression source: adata.X")
    return adata.X


def make_gene_sentence(matrix, row_index, gene_names, top_genes):
    """Create a space-separated sentence of the top expressed genes."""
    row = matrix[row_index]

    if sparse.issparse(row):
        row = row.tocsr()
        values = row.data
        indices = row.indices

        positive = np.isfinite(values) & (values > 0)
        values = values[positive]
        indices = indices[positive]

        if values.size == 0:
            return "", 0

        order = np.argsort(values)[::-1]
        selected_indices = indices[order[:top_genes]]

    else:
        values = np.asarray(row).reshape(-1)

        positive_indices = np.flatnonzero(
            np.isfinite(values) & (values > 0)
        )

        if positive_indices.size == 0:
            return "", 0

        positive_values = values[positive_indices]

        if positive_indices.size <= top_genes:
            order = np.argsort(positive_values)[::-1]
        else:
            candidate_order = np.argpartition(
                positive_values,
                -top_genes,
            )[-top_genes:]

            order = candidate_order[
                np.argsort(positive_values[candidate_order])[::-1]
            ]

        selected_indices = positive_indices[order[:top_genes]]

    selected_genes = []
    seen = set()

    for index in selected_indices:
        gene = str(gene_names[index]).strip()

        if (
            not gene
            or gene.lower() == "nan"
            or gene in seen
        ):
            continue

        seen.add(gene)
        selected_genes.append(gene)

    return " ".join(selected_genes), len(selected_genes)


def load_patch_information(patch_path):
    """Read patch barcodes, coordinates and patch count from HDF5."""
    with h5py.File(patch_path, "r") as handle:
        required_keys = ["img", "barcode"]

        missing_keys = [
            key for key in required_keys if key not in handle
        ]

        if missing_keys:
            raise KeyError(
                f"{patch_path} is missing HDF5 keys: {missing_keys}. "
                f"Available keys: {list(handle.keys())}"
            )

        patch_count = handle["img"].shape[0]

        barcodes = [
            decode_value(value).strip()
            for value in handle["barcode"][:]
        ]

        if len(barcodes) != patch_count:
            raise ValueError(
                f"{patch_path}: {patch_count} images but "
                f"{len(barcodes)} barcodes."
            )

        if "coords" in handle:
            coordinates = np.asarray(handle["coords"][:])

            if len(coordinates) != patch_count:
                print(
                    "  Warning: coordinate count does not match patch "
                    "count; coordinates will be omitted."
                )
                coordinates = None
        else:
            coordinates = None

    return barcodes, coordinates, patch_count


def get_adata_barcodes(adata):
    """Use an explicit barcode column when present, else obs_names."""
    for column in ["barcode", "spot_barcode"]:
        if column in adata.obs.columns:
            print(f"  ST barcodes taken from adata.obs['{column}']")
            return adata.obs[column].astype(str).tolist()

    print("  ST barcodes taken from adata.obs_names")
    return adata.obs_names.astype(str).tolist()


def process_sample(
    sample_id,
    patient_id,
    patch_path,
    st_path,
    top_genes,
):
    print(f"\nProcessing {sample_id}")
    print(f"  Patient: {patient_id}")
    print(f"  Patches: {patch_path}")
    print(f"  ST:      {st_path}")

    patch_barcodes, coordinates, patch_count = (
        load_patch_information(patch_path)
    )

    adata = ad.read_h5ad(st_path)

    print(f"  Patch count: {patch_count}")
    print(f"  ST shape:    {adata.shape}")

    st_barcodes = get_adata_barcodes(adata)
    gene_names = get_gene_names(adata)
    expression_matrix = get_expression_matrix(adata)

    if expression_matrix.shape != adata.shape:
        raise ValueError(
            f"Expression matrix shape {expression_matrix.shape} does "
            f"not match AnnData shape {adata.shape}."
        )

    # Map normalised ST barcode to its row number.
    st_barcode_to_index = {}

    for row_index, barcode in enumerate(st_barcodes):
        key = normalise_barcode(barcode)

        if key in st_barcode_to_index:
            print(
                f"  Warning: duplicate normalised ST barcode: {key}"
            )
            continue

        st_barcode_to_index[key] = row_index

    records = []
    unmatched_patch_barcodes = 0
    empty_gene_sentences = 0

    for patch_index, patch_barcode in enumerate(patch_barcodes):
        barcode_key = normalise_barcode(patch_barcode)
        st_row_index = st_barcode_to_index.get(barcode_key)

        if st_row_index is None:
            unmatched_patch_barcodes += 1
            continue

        gene_sentence, gene_count = make_gene_sentence(
            expression_matrix,
            st_row_index,
            gene_names,
            top_genes,
        )

        if not gene_sentence:
            empty_gene_sentences += 1
            continue

        record = {
            "sample_id": sample_id,
            "patient_id": patient_id,
            "barcode": patch_barcode,
            "patch_h5_path": str(patch_path.resolve()),
            "patch_index": patch_index,
            "st_h5ad_path": str(st_path.resolve()),
            "st_row_index": st_row_index,
            "gene_sentence": gene_sentence,
            "gene_count": gene_count,
        }

        if (
            coordinates is not None
            and coordinates.ndim == 2
            and coordinates.shape[1] >= 2
        ):
            record["coord_x"] = coordinates[patch_index, 0]
            record["coord_y"] = coordinates[patch_index, 1]

        records.append(record)

    print(f"  Matched pairs:          {len(records)}")
    print(f"  Unmatched patches:      {unmatched_patch_barcodes}")
    print(f"  Empty gene sentences:   {empty_gene_sentences}")

    if patch_count:
        match_rate = len(records) / patch_count
        print(f"  Usable match rate:      {match_rate:.2%}")

    return records


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create barcode-matched HEST H&E/ST pairs for OmiCLIP."
        )
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "datasets/hest_metadata/ccrcc_samples.csv"
        ),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/hest_ccrcc"),
        help=(
            "Folder containing patches/<sample>.h5 and "
            "st/<sample>.h5ad."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/omiclip/ccrcc_pairs.csv"
        ),
    )

    parser.add_argument(
        "--top-genes",
        type=int,
        default=50,
        help="Maximum number of genes in each gene sentence.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.top_genes < 1:
        raise ValueError("--top-genes must be at least 1.")

    if not args.metadata.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {args.metadata}"
        )

    metadata = pd.read_csv(args.metadata)

    sample_column = find_first_column(
        metadata,
        SAMPLE_COLUMN_CANDIDATES,
    )

    patient_column = find_first_column(
        metadata,
        PATIENT_COLUMN_CANDIDATES,
    )

    if sample_column is None:
        raise KeyError(
            "Could not find a sample column. Expected one of: "
            f"{SAMPLE_COLUMN_CANDIDATES}. "
            f"Available columns: {list(metadata.columns)}"
        )

    print(f"Sample column:  {sample_column}")
    print(f"Patient column: {patient_column}")
    print(f"Top genes:      {args.top_genes}")

    all_records = []
    missing_samples = []

    for _, metadata_row in metadata.iterrows():
        sample_id = str(metadata_row[sample_column]).strip()

        if not sample_id or sample_id.lower() == "nan":
            continue

        if (
            patient_column
            and pd.notna(metadata_row[patient_column])
        ):
            patient_id = str(
                metadata_row[patient_column]
            ).strip()
        else:
            # Do not pretend sample ID is a true patient ID.
            patient_id = "UNKNOWN"

        patch_path = (
            args.data_root
            / "patches"
            / f"{sample_id}.h5"
        )

        st_path = (
            args.data_root
            / "st"
            / f"{sample_id}.h5ad"
        )

        missing = []

        if not patch_path.exists():
            missing.append(str(patch_path))

        if not st_path.exists():
            missing.append(str(st_path))

        if missing:
            print(f"\nSkipping {sample_id}; missing:")
            for path in missing:
                print(f"  {path}")

            missing_samples.append(sample_id)
            continue

        records = process_sample(
            sample_id=sample_id,
            patient_id=patient_id,
            patch_path=patch_path,
            st_path=st_path,
            top_genes=args.top_genes,
        )

        all_records.extend(records)

    if not all_records:
        raise RuntimeError(
            "No matched H&E/ST pairs were produced. Check the "
            "file paths and barcode formats."
        )

    output_df = pd.DataFrame(all_records)

    output_df = output_df.sort_values(
        ["sample_id", "patch_index"]
    ).reset_index(drop=True)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(args.output, index=False)

    print("\n========================================")
    print("OmiCLIP pair preparation complete")
    print("========================================")
    print(f"Samples represented: {output_df['sample_id'].nunique()}")
    print(f"Patients represented: "
          f"{output_df['patient_id'].nunique()}")
    print(f"Total matched pairs:  {len(output_df)}")
    print(f"Output:               {args.output.resolve()}")

    if missing_samples:
        print(
            f"Samples skipped for missing files: "
            f"{len(missing_samples)}"
        )
        print(missing_samples)

    if (output_df["patient_id"] == "UNKNOWN").any():
        print(
            "\nWarning: patient IDs were unavailable in the metadata. "
            "Do not perform patient-level splitting until the true "
            "patient mapping is added."
        )

    print("\nOutput columns:")
    print(list(output_df.columns))

    print("\nFirst rows:")
    print(
        output_df[
            [
                "sample_id",
                "patient_id",
                "barcode",
                "patch_index",
                "gene_count",
            ]
        ]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()