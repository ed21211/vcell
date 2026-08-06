from pathlib import Path
import tempfile

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image

import loki.utils


PROJECT_DIR = Path(__file__).resolve().parents[1]

PAIRS_PATH = (
    PROJECT_DIR
    / "datasets"
    / "omiclip"
    / "ccrcc_pairs.csv"
)

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "models"
    / "omiclip"
    / "checkpoint.pt"
)


def prepare_patch(array):
    """Convert a HEST patch into an RGB uint8 image."""
    array = np.asarray(array)

    # Convert channel-first to channel-last if needed.
    if (
        array.ndim == 3
        and array.shape[0] in [1, 3, 4]
        and array.shape[-1] not in [1, 3, 4]
    ):
        array = np.moveaxis(array, 0, -1)

    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)

    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)

    if array.shape[-1] == 4:
        array = array[..., :3]

    if np.issubdtype(array.dtype, np.floating):
        if array.max() <= 1:
            array = array * 255

    array = np.clip(array, 0, 255).astype(np.uint8)

    return Image.fromarray(array).convert("RGB")


def main():
    if not PAIRS_PATH.exists():
        raise FileNotFoundError(f"Pairs file missing: {PAIRS_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint missing: {CHECKPOINT_PATH}"
        )

    pairs = pd.read_csv(PAIRS_PATH)

    if pairs.empty:
        raise RuntimeError("The OmiCLIP pairs CSV is empty.")

    row = pairs.iloc[0]

    patch_path = Path(row["patch_h5_path"])
    patch_index = int(row["patch_index"])
    gene_sentence = str(row["gene_sentence"])

    print("Sample:", row["sample_id"])
    print("Barcode:", row["barcode"])
    print("Patch index:", patch_index)
    print("Gene sentence:", gene_sentence[:200])

    with h5py.File(patch_path, "r") as handle:
        patch_array = handle["img"][patch_index]

    image = prepare_patch(patch_array)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Patch size:", image.size)

    model, preprocess, tokenizer = loki.utils.load_model(
        str(CHECKPOINT_PATH),
        device,
    )
    model.eval()

    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    image.save(temporary_path)

    try:
        with torch.no_grad():
            image_embedding = loki.utils.encode_images(
                model,
                preprocess,
                [str(temporary_path)],
                device,
            )

            text_inputs = tokenizer([gene_sentence]).to(device)

            text_embedding = model.encode_text(text_inputs)

            text_embedding = torch.nn.functional.normalize(
                text_embedding,
                p=2,
                dim=-1,
            )

            similarity = image_embedding @ text_embedding.T

        print("Image embedding:", image_embedding.shape)
        print("Text embedding:", text_embedding.shape)
        print("Matched similarity:", similarity.item())

    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()