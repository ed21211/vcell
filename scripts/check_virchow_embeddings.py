from pathlib import Path
import numpy as np

FEATURE_DIR = Path("features/virchow_hest_ccrcc")

npz_files = sorted(FEATURE_DIR.glob("*_virchow.npz"))

print(f"Found {len(npz_files)} embedding files")

for path in npz_files:
    data = np.load(path, allow_pickle=True)

    sample_id = str(data["sample_id"])
    embeddings = data["embeddings"]
    barcodes = data["barcodes"]
    coords = data["coords"]

    print("\n", path.name)
    print("sample_id:", sample_id)
    print("embeddings:", embeddings.shape)
    print("barcodes:", barcodes.shape)
    print("coords:", coords.shape)

    assert embeddings.shape[0] == barcodes.shape[0] == coords.shape[0], "Row mismatch"

print("\nAll embedding files look consistent.")