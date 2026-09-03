from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from loki.utils import load_model


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

PAIRS_CSV = (
    PROJECT_DIR
    / "datasets"
    / "omiclip"
    / "ccrcc_pairs.csv"
)

PATCH_DIR = (
    PROJECT_DIR
    / "datasets"
    / "hest_ccrcc"
    / "patches"
)

# Needed only to construct the original OmiCLIP architecture
PRETRAINED_CHECKPOINT = (
    PROJECT_DIR
    / "models"
    / "omiclip"
    / "checkpoint.pt"
)

# Your ccRCC-fine-tuned model
FINETUNED_CHECKPOINT = (
    PROJECT_DIR
    / "models"
    / "omiclip_ccrcc_full"
    / "best.pt"
)

OUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "omiclip_finetuned"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_H5 = (
    OUT_DIR
    / "ccrcc_omiclip_finetuned_embeddings.h5"
)

OUTPUT_METADATA = (
    OUT_DIR
    / "ccrcc_omiclip_finetuned_metadata.csv"
)


# ============================================================
# CONFIG
# ============================================================

BATCH_SIZE = 64
NUM_WORKERS = 8


# ============================================================
# DATASET
# ============================================================

class HESTImageDataset(Dataset):

    def __init__(
        self,
        dataframe,
        patch_dir,
        preprocess,
    ):
        self.df = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.patch_dir = Path(
            patch_dir
        )

        self.preprocess = preprocess

        # Each worker creates its own HDF5 handles.
        self._handles = {}

    def __len__(self):
        return len(self.df)

    def _get_handle(
        self,
        sample_id,
    ):
        if sample_id not in self._handles:

            path = (
                self.patch_dir
                / f"{sample_id}.h5"
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing patch file: {path}"
                )

            self._handles[
                sample_id
            ] = h5py.File(
                path,
                "r",
            )

        return self._handles[
            sample_id
        ]

    def __getitem__(
        self,
        index,
    ):
        row = self.df.iloc[
            index
        ]

        sample_id = str(
            row["sample_id"]
        )

        patch_index = int(
            row["patch_index"]
        )

        handle = self._get_handle(
            sample_id
        )

        image_array = (
            handle["img"][
                patch_index
            ]
        )

        # Handle CHW if necessary.
        if (
            image_array.ndim == 3
            and image_array.shape[0] == 3
            and image_array.shape[-1] != 3
        ):
            image_array = np.transpose(
                image_array,
                (1, 2, 0),
            )

        # Ensure uint8.
        if image_array.dtype != np.uint8:

            if (
                np.nanmax(
                    image_array
                )
                <= 1.0
            ):
                image_array = (
                    image_array
                    * 255.0
                )

            image_array = np.clip(
                image_array,
                0,
                255,
            ).astype(
                np.uint8
            )

        image = (
            Image.fromarray(
                image_array
            )
            .convert("RGB")
        )

        image = self.preprocess(
            image
        )

        return (
            image,
            index,
        )


# ============================================================
# LOAD FINE-TUNED MODEL
# ============================================================

def load_finetuned_model(
    device,
):
    print(
        "Constructing pretrained OmiCLIP architecture...",
        flush=True,
    )

    start = time.time()

    model, preprocess, tokenizer = load_model(
        str(
            PRETRAINED_CHECKPOINT
        ),
        device,
    )

    print(
        f"Base model loaded in "
        f"{time.time() - start:.1f} sec",
        flush=True,
    )

    print(
        "\nLoading fine-tuned checkpoint:",
        FINETUNED_CHECKPOINT,
        flush=True,
    )

    checkpoint = torch.load(
        FINETUNED_CHECKPOINT,
        map_location="cpu",
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Fine-tuned checkpoint does not contain "
            "'model_state_dict'. "
            f"Keys: {list(checkpoint.keys())}"
        )

    missing, unexpected = (
        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=False,
        )
    )

    print(
        "Missing keys:",
        len(missing),
    )

    print(
        "Unexpected keys:",
        len(unexpected),
    )

    if missing:
        print(
            "First missing keys:",
            missing[:10],
        )

    if unexpected:
        print(
            "First unexpected keys:",
            unexpected[:10],
        )

    if missing or unexpected:
        raise RuntimeError(
            "Fine-tuned checkpoint did not "
            "match OmiCLIP architecture exactly."
        )

    model = model.to(
        device
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    epoch = checkpoint.get(
        "epoch",
        "unknown",
    )

    val_loss = checkpoint.get(
        "val_loss",
        "unknown",
    )

    print(
        "\nFine-tuned checkpoint loaded."
    )

    print(
        "Best epoch:",
        epoch,
    )

    print(
        "Validation loss:",
        val_loss,
    )

    return (
        model,
        preprocess,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=============================================="
    )
    print(
        "EXTRACT FINE-TUNED OmiCLIP IMAGE EMBEDDINGS"
    )
    print(
        "=============================================="
    )

    if not PAIRS_CSV.exists():
        raise FileNotFoundError(
            PAIRS_CSV
        )

    if not PRETRAINED_CHECKPOINT.exists():
        raise FileNotFoundError(
            PRETRAINED_CHECKPOINT
        )

    if not FINETUNED_CHECKPOINT.exists():
        raise FileNotFoundError(
            FINETUNED_CHECKPOINT
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0),
    )

    # --------------------------------------------------------
    # Load pair metadata
    # --------------------------------------------------------

    df = pd.read_csv(
        PAIRS_CSV
    )

    print(
        "\nPair table:",
        df.shape,
    )

    print(
        "Samples:",
        df["sample_id"].nunique(),
    )

    print(
        "Rows:",
        f"{len(df):,}",
    )

    required = [
        "sample_id",
        "barcode",
        "patch_index",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Load fine-tuned OmiCLIP
    # --------------------------------------------------------

    (
        model,
        preprocess,
    ) = load_finetuned_model(
        device
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = HESTImageDataset(
        dataframe=df,
        patch_dir=PATCH_DIR,
        preprocess=preprocess,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        drop_last=False,
    )

    # --------------------------------------------------------
    # Mixed precision
    # --------------------------------------------------------

    if torch.cuda.is_bf16_supported():

        amp_dtype = torch.bfloat16

        print(
            "\nUsing BF16 inference"
        )

    else:

        amp_dtype = torch.float16

        print(
            "\nUsing FP16 inference"
        )

    # --------------------------------------------------------
    # Determine embedding dimension
    # --------------------------------------------------------

    first_batch = next(
        iter(loader)
    )

    images, indices = first_batch

    images = images.to(
        device,
        non_blocking=True,
    )

    with torch.inference_mode():

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
        ):

            features = (
                model.encode_image(
                    images
                )
            )

            features = torch.nn.functional.normalize(
                features,
                dim=-1,
            )

    embedding_dim = (
        features.shape[1]
    )

    print(
        "Embedding dimension:",
        embedding_dim,
    )

    # --------------------------------------------------------
    # Create output HDF5
    # --------------------------------------------------------

    n_rows = len(
        dataset
    )

    print(
        "\nWriting:",
        OUTPUT_H5,
    )

    start_time = time.time()

    with h5py.File(
        OUTPUT_H5,
        "w",
    ) as output:

        embedding_dataset = (
            output.create_dataset(
                "image_embeddings",
                shape=(
                    n_rows,
                    embedding_dim,
                ),
                dtype="float32",
                chunks=(
                    min(
                        1024,
                        n_rows,
                    ),
                    embedding_dim,
                ),
                compression="gzip",
            )
        )

        cursor = 0

        for (
            batch_index,
            batch,
        ) in enumerate(
            loader,
            start=1,
        ):

            images, indices = batch

            images = images.to(
                device,
                non_blocking=True,
            )

            with torch.inference_mode():

                with torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                ):

                    embeddings = (
                        model.encode_image(
                            images
                        )
                    )

                    embeddings = (
                        torch.nn.functional.normalize(
                            embeddings,
                            dim=-1,
                        )
                    )

            embeddings = (
                embeddings
                .float()
                .cpu()
                .numpy()
            )

            batch_size = (
                embeddings.shape[0]
            )

            embedding_dataset[
                cursor:
                cursor + batch_size
            ] = embeddings

            cursor += (
                batch_size
            )

            if (
                batch_index % 100
                == 0
                or batch_index
                == len(loader)
            ):

                elapsed = (
                    time.time()
                    - start_time
                )

                rows_per_second = (
                    cursor
                    / elapsed
                )

                remaining = (
                    n_rows
                    - cursor
                )

                eta_minutes = (
                    remaining
                    / max(
                        rows_per_second,
                        1e-9,
                    )
                    / 60
                )

                print(
                    f"batch "
                    f"{batch_index:4d}/"
                    f"{len(loader):4d} | "
                    f"{cursor:,}/"
                    f"{n_rows:,} rows | "
                    f"{rows_per_second:.1f} rows/s | "
                    f"ETA {eta_minutes:.1f} min",
                    flush=True,
                )

        if cursor != n_rows:
            raise RuntimeError(
                f"Expected {n_rows} embeddings, "
                f"wrote {cursor}."
            )

    # --------------------------------------------------------
    # Save metadata in EXACT same order
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_METADATA,
        index=False,
    )

    total_minutes = (
        time.time()
        - start_time
    ) / 60

    print(
        "\n=============================================="
    )
    print(
        "FINE-TUNED EMBEDDING EXTRACTION COMPLETE"
    )
    print(
        "=============================================="
    )

    print(
        "Embeddings:",
        OUTPUT_H5,
    )

    print(
        "Metadata:",
        OUTPUT_METADATA,
    )

    print(
        "Shape:",
        (
            n_rows,
            embedding_dim,
        ),
    )

    print(
        f"Time: {total_minutes:.1f} min"
    )


if __name__ == "__main__":
    main()