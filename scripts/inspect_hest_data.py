import os
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "MahmoodLab/hest"
METADATA_FILE = "HEST_v1_3_0.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "datasets", "hest_metadata")


def load_metadata(filename):
    path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename)
    df = pd.read_csv(path)
    print(f"\nLoaded metadata: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", list(df.columns))
    return df


def summarize_organs(df):
    for col in ["organ", "tissue", "oncotree_code", "disease_state", "cancer_type", "dataset_title"]:
        if col in df.columns:
            print(f"\n--- value_counts for '{col}' ---")
            print(df[col].value_counts())


def filter_by_oncotree(df, codes):
    """Filter rows by exact oncotree_code match (precise, unlike keyword search
    on organ/tissue which pulls in non-cancer samples)."""
    codes_lower = [c.lower() for c in codes]
    mask = df["oncotree_code"].astype(str).str.lower().isin(codes_lower)
    filtered = df[mask]
    print(f"\nFound {len(filtered)} rows matching oncotree_code in {codes}")
    return filtered


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_metadata(METADATA_FILE)
    summarize_organs(df)

    # Change to just ["SCCRCC"] if want strict ccRCC-only (no papillary RCC)
    print("\n=== ccRCC (SCCRCC + PRCC) ===")
    ccrcc_df = filter_by_oncotree(df, ["SCCRCC", "PRCC"])
    print(ccrcc_df[["id", "organ", "tissue", "oncotree_code", "st_technology", "patient", "species"]])

    ccrcc_path = os.path.join(OUTPUT_DIR, "ccrcc_samples.csv")
    ccrcc_df.to_csv(ccrcc_path, index=False)
    print(f"Saved to {ccrcc_path}")
    print(f"\nSUMMARY: ccRCC={len(ccrcc_df)} samples")