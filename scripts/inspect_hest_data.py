import os
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "MahmoodLab/hest"
METADATA_FILE = "HEST_v1_3_0.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "datasets", "hest_metadata")


def load_metadata(filename):
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=filename,
    )

    df = pd.read_csv(path)

    print(f"Loaded metadata: {df.shape[0]} rows")
    print("Columns:", list(df.columns))

    return df


def find_first_column(df, candidates):
    return next((column for column in candidates if column in df.columns), None)


def classify_species(value):
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().lower()

    if any(term in value for term in ["human", "homo sapiens"]):
        return "Human"

    if any(
        term in value
        for term in [
            "mouse",
            "mus musculus",
            "murine",
            "rat",
            "rattus",
            "canine",
            "dog",
            "pig",
            "porcine",
            "monkey",
            "macaque",
        ]
    ):
        return "Animal"

    return "Unknown"


def summarize_all_cancers(df):
    if "oncotree_code" not in df.columns:
        raise KeyError("Column 'oncotree_code' was not found.")

    sample_column = find_first_column(
        df,
        ["id", "sample_id", "sample"],
    )

    patient_column = find_first_column(
        df,
        [
            "patient_id",
            "subject_id",
            "patient",
            "donor_id",
            "individual_id",
            "case_id",
        ],
    )

    species_column = find_first_column(
        df,
        ["species", "organism", "species_name"],
    )

    technology_column = find_first_column(
        df,
        [
            "st_technology",
            "technology",
            "spatial_technology",
            "platform",
            "assay",
        ],
    )

    cancer_name_column = find_first_column(
        df,
        [
            "cancer_type",
            "disease_state",
            "disease",
            "diagnosis",
        ],
    )

    print("\nDetected columns:")
    print(f"  Sample:      {sample_column}")
    print(f"  Patient:     {patient_column}")
    print(f"  Species:     {species_column}")
    print(f"  Cancer name: {cancer_name_column}")
    print(f"  Technology:  {technology_column}")

    working_df = df[df["oncotree_code"].notna()].copy()

    working_df["oncotree_code"] = (
        working_df["oncotree_code"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if species_column:
        working_df["species_group"] = working_df[species_column].apply(
            classify_species
        )
    else:
        working_df["species_group"] = "Unknown"

    # Count metadata rows when no explicit sample ID exists.
    working_df["_sample_key"] = (
        working_df[sample_column]
        if sample_column
        else working_df.index.astype(str)
    )

    if patient_column:
        working_df["_patient_key"] = working_df[patient_column]
    else:
        working_df["_patient_key"] = pd.NA

    group_columns = ["oncotree_code", "species_group"]

    summary_df = (
        working_df.groupby(group_columns, dropna=False)
        .agg(
            sample_count=("_sample_key", "nunique"),
            patient_count=("_patient_key", "nunique"),
        )
        .reset_index()
    )

    # Add a readable cancer label when available.
    if cancer_name_column:
        cancer_names = (
            working_df[
                working_df[cancer_name_column].notna()
            ]
            .groupby("oncotree_code")[cancer_name_column]
            .agg(lambda values: values.value_counts().index[0])
            .rename("cancer_type")
            .reset_index()
        )

        summary_df = summary_df.merge(
            cancer_names,
            on="oncotree_code",
            how="left",
        )

        summary_df = summary_df[
            [
                "oncotree_code",
                "cancer_type",
                "species_group",
                "sample_count",
                "patient_count",
            ]
        ]

    summary_df = summary_df.sort_values(
        ["oncotree_code", "species_group"]
    ).reset_index(drop=True)

    # Wide table for easier comparison.
    pivot_df = summary_df.pivot_table(
        index=[
            column
            for column in ["oncotree_code", "cancer_type"]
            if column in summary_df.columns
        ],
        columns="species_group",
        values=["sample_count", "patient_count"],
        fill_value=0,
        aggfunc="sum",
    )

    pivot_df.columns = [
        f"{species.lower()}_{metric}"
        for metric, species in pivot_df.columns
    ]

    pivot_df = pivot_df.reset_index()

    # Add total samples and patients.
    total_df = (
        working_df.groupby("oncotree_code")
        .agg(
            total_sample_count=("_sample_key", "nunique"),
            total_patient_count=("_patient_key", "nunique"),
        )
        .reset_index()
    )

    pivot_df = pivot_df.merge(
        total_df,
        on="oncotree_code",
        how="left",
    )

    print("\n=== All cancer types ===")
    print(pivot_df.to_string(index=False))

    if technology_column:
        technology_summary = (
            working_df.groupby(
                ["oncotree_code", "species_group", technology_column],
                dropna=False,
            )
            .agg(
                sample_count=("_sample_key", "nunique"),
                patient_count=("_patient_key", "nunique"),
            )
            .reset_index()
            .sort_values(
                ["oncotree_code", "species_group", technology_column]
            )
        )
    else:
        technology_summary = pd.DataFrame()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary_path = os.path.join(
        OUTPUT_DIR,
        "HEST_all_cancers_species_summary.csv",
    )

    detailed_path = os.path.join(
        OUTPUT_DIR,
        "HEST_all_cancers_detailed_metadata.csv",
    )

    technology_path = os.path.join(
        OUTPUT_DIR,
        "HEST_all_cancers_technology_summary.csv",
    )

    pivot_df.to_csv(summary_path, index=False)
    working_df.drop(
        columns=["_sample_key", "_patient_key"],
        errors="ignore",
    ).to_csv(detailed_path, index=False)

    if not technology_summary.empty:
        technology_summary.to_csv(technology_path, index=False)

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved metadata: {detailed_path}")

    if not technology_summary.empty:
        print(f"Saved technology summary: {technology_path}")

    return pivot_df, technology_summary


if __name__ == "__main__":
    metadata = load_metadata(METADATA_FILE)

    cancer_summary, technology_summary = summarize_all_cancers(
        metadata
    )