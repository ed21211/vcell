import os
import pandas as pd
from huggingface_hub import snapshot_download

REPO_ID = "MahmoodLab/hest"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "..", "datasets")
LOCAL_DIR = os.path.join(DATASET_DIR, "hest_ccrcc")
SAMPLE_CSV = os.path.join(DATASET_DIR, "hest_metadata", "ccrcc_samples.csv")

TEST_SINGLE_SAMPLE = False  # Set to True for sample testing. doenload only ONE sample.
DOWNLOAD_XENIUM_EXTRAS = True
DOWNLOAD_VISUALS = True


def build_allow_patterns(df: pd.DataFrame) -> list[str]:
    patterns = []

    for _, row in df.iterrows():
        sid = str(row["id"])
        tech = str(row.get("st_technology", "")).lower()

        # Core files needed for H&E + ST modelling
        patterns.extend([
            f"metadata/{sid}.json",
            f"wsis/{sid}.tif",
            f"st/{sid}.h5ad",
            f"patches/{sid}.h5",
        ])

        # Useful for inspection/debugging, not always needed for training
        if DOWNLOAD_VISUALS:
            patterns.extend([
                f"thumbnails/{sid}_downscaled_fullres.jpeg",
                f"spatial_plots/{sid}_spatial_plots.png",
                f"patches_vis/{sid}_patch_vis.jpg",
                f"tissue_seg/{sid}_vis.jpg",
                f"tissue_seg/{sid}_contours.geojson",
                f"pixel_size_vis/{sid}_pixel_size_vis.png",
            ])

        # Xenium samples have transcript-level and segmentation files
        if DOWNLOAD_XENIUM_EXTRAS and "xenium" in tech:
            patterns.extend([
                f"transcripts/{sid}_transcripts.parquet",
                f"xenium_seg/{sid}_xenium_cell_seg.parquet",
                f"xenium_seg/{sid}_xenium_nucleus_seg.parquet",
                f"xenium_seg/{sid}_xenium_cell_seg.geojson.zip",
                f"xenium_seg/{sid}_xenium_nucleus_seg.geojson.zip",
                f"cellvit_seg/{sid}_cellvit_seg.parquet",
                f"cellvit_seg/{sid}_cellvit_seg.geojson.zip",
            ])

    return sorted(set(patterns))


def main():
    df = pd.read_csv(SAMPLE_CSV)

    # if TEST_SINGLE_SAMPLE: // TAKES ORST LINE BUT IS NULL AT TENX105
    #     df = df.head(1)
    #     print(f"TEST MODE: downloading single sample only: {df['id'].tolist()}")
    # else:
    #     print(f"Downloading {len(df)} samples: {df['id'].tolist()}")

    if TEST_SINGLE_SAMPLE:
        TEST_SAMPLE_ID = "INT1"
        df = df[df["id"].astype(str) == TEST_SAMPLE_ID]

        if df.empty:
            raise ValueError(f"Sample {TEST_SAMPLE_ID} not found in {SAMPLE_CSV}")

        print(f"TEST MODE: downloading single sample only: {df['id'].tolist()}")
    else:
        # optional: skip TENX105 so you only download Visium ccRCC samples
        df = df[df["st_technology"].astype(str).str.lower() == "visium"]
        print(f"Downloading {len(df)} Visium ccRCC samples: {df['id'].tolist()}")

    patterns = build_allow_patterns(df)

    print("\nAllow patterns:")
    for p in patterns:
        print("  ", p)

    os.makedirs(LOCAL_DIR, exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
    )

    print(f"\nDone. Data downloaded to: {LOCAL_DIR}")


if __name__ == "__main__":
    main()