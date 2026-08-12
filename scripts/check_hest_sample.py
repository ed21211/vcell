import os
import h5py
import anndata as ad
import pandas as pd
import tifffile

# SAMPLE_ID = "TENX105"
SAMPLE_ID = "INT1"
DATA_DIR = "datasets/hest_ccrcc"

paths = {
    "wsi": f"{DATA_DIR}/wsis/{SAMPLE_ID}.tif",
    "st": f"{DATA_DIR}/st/{SAMPLE_ID}.h5ad",
    "patches": f"{DATA_DIR}/patches/{SAMPLE_ID}.h5",
    "metadata": f"{DATA_DIR}/metadata/{SAMPLE_ID}.json",
}

print(f"Checking sample: {SAMPLE_ID}\n")

# 1. Check files exist
for name, path in paths.items():
    print(f"{name}: {path}")
    print("  exists:", os.path.exists(path))
    if os.path.exists(path):
        print("  size MB:", round(os.path.getsize(path) / 1e6, 2))
    print()

# 2. Check WSI opens
print("Checking WSI...")
with tifffile.TiffFile(paths["wsi"]) as tif:
    print("  number of pages:", len(tif.pages))
    print("  first page shape:", tif.pages[0].shape)
    print("  dtype:", tif.pages[0].dtype)

# 3. Check h5ad opens
print("\nChecking ST h5ad...")
adata = ad.read_h5ad(paths["st"])
print("  adata shape:", adata.shape)
print("  obs columns:", list(adata.obs.columns)[:20])
print("  var columns:", list(adata.var.columns)[:20])
print("  obsm keys:", list(adata.obsm.keys()))
print("  uns keys:", list(adata.uns.keys()))

# Common useful fields
if "spatial" in adata.obsm:
    print("  spatial coordinates shape:", adata.obsm["spatial"].shape)
    print("  first 5 spatial coords:")
    print(adata.obsm["spatial"][:5])

# 4. Check patches h5
print("\nChecking patches h5...")
with h5py.File(paths["patches"], "r") as f:
    print("  keys:", list(f.keys()))

    def show_h5(name, obj):
        if hasattr(obj, "shape"):
            print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")

    f.visititems(show_h5)

# 5. Basic alignment sanity check
print("\nBasic alignment check...")
print("  number of ST spots/cells:", adata.n_obs)

with h5py.File(paths["patches"], "r") as f:
    for key in f.keys():
        obj = f[key]
        if hasattr(obj, "shape"):
            print(f"  patches file dataset '{key}' shape:", obj.shape)

# 6. Try to identify patch-to-ST matching fields
print("\nChecking possible patch/ST matching fields...")

with h5py.File(paths["patches"], "r") as f:
    keys = list(f.keys())

    for possible_key in ["barcode", "barcodes", "coords", "coordinates", "spatial"]:
        if possible_key in keys:
            data = f[possible_key][()]
            print(f"  Found {possible_key}: shape={data.shape}")

            if len(data) == adata.n_obs:
                print(f"  {possible_key} count matches adata.n_obs")

print("  adata obs_names example:")
print(adata.obs_names[:5].tolist())

# 7. Try to identify patch-to-ST matching fields
print("\nChecking barcode overlap...")

with h5py.File(paths["patches"], "r") as f:
    patch_barcodes = f["barcode"][:]

# decode bytes / object values cleanly
patch_barcodes = [
    b[0].decode("utf-8") if isinstance(b[0], bytes) else str(b[0])
    for b in patch_barcodes
]

adata_barcodes = set(adata.obs_names.astype(str))
overlap = [b for b in patch_barcodes if b in adata_barcodes]

print("  number of patch barcodes:", len(patch_barcodes))
print("  number of adata obs names:", len(adata_barcodes))
print("  overlap:", len(overlap))
print("  first 10 patch barcodes:", patch_barcodes[:10])
print("  first 10 overlapping barcodes:", overlap[:10])


print("\nDone.")