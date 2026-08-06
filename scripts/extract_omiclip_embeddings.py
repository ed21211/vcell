import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import loki.utils


PROJECT_DIR = Path(__file__).resolve().parents[1]


def prepare_patch(array):
    """Convert a HEST HDF5 patch into an RGB PIL image."""
    array = np.asarray(array)

    # Convert channel-first to channel-last.
    if (
        array.ndim == 3
        and array.shape[0] in (1, 3, 4)
        and array.shape[-1] not in (1, 3, 4)
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


def load_batch_images(batch_df, h5_handles):
    """Load H&E patches while keeping each HDF5 file open."""
    images = []

    for row in batch_df.itertuples(index=False):
        patch_path = str(Path(row.patch_h5_path).resolve())

        if patch_path not in h5_handles:
            h5_handles[patch_path] = h5py.File(
                patch_path,
                "r",
            )

        handle = h5_handles[patch_path]
        patch_array = handle["img"][int(row.patch_index)]

        images.append(prepare_patch(patch_array))

    return images


def encode_batch(
    batch_df,
    h5_handles,
    model,
    preprocess,
    tokenizer,
    device,
):
    images = load_batch_images(
        batch_df,
        h5_handles,
    )

    image_inputs = torch.stack(
        [preprocess(image) for image in images]
    ).to(device)

    gene_sentences = (
        batch_df["gene_sentence"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    text_inputs = tokenizer(gene_sentences).to(device)

    with torch.no_grad():
        image_embeddings = model.encode_image(
            image_inputs
        )

        text_embeddings = model.encode_text(
            text_inputs
        )

        image_embeddings = F.normalize(
            image_embeddings,
            p=2,
            dim=-1,
        )

        text_embeddings = F.normalize(
            text_embeddings,
            p=2,
            dim=-1,
        )

        # Cosine similarity for each matched image–ST pair.
        matched_similarity = (
            image_embeddings * text_embeddings
        ).sum(dim=1)

    return (
        image_embeddings.float().cpu().numpy(),
        text_embeddings.float().cpu().numpy(),
        matched_similarity.float().cpu().numpy(),
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Extract OmiCLIP image and ST embeddings "
            "for matched HEST spots."
        )
    )

    parser.add_argument(
        "--pairs",
        type=Path,
        default=(
            PROJECT_DIR
            / "datasets"
            / "omiclip"
            / "ccrcc_pairs.csv"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            PROJECT_DIR
            / "models"
            / "omiclip"
            / "checkpoint.pt"
        ),
    )

    parser.add_argument(
        "--output-h5",
        type=Path,
        default=(
            PROJECT_DIR
            / "outputs"
            / "omiclip"
            / "ccrcc_omiclip_embeddings.h5"
        ),
    )

    parser.add_argument(
        "--output-metadata",
        type=Path,
        default=(
            PROJECT_DIR
            / "outputs"
            / "omiclip"
            / "ccrcc_omiclip_metadata.csv"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional small test subset.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if not args.pairs.exists():
        raise FileNotFoundError(
            f"Pairs CSV missing: {args.pairs}"
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint missing: {args.checkpoint}"
        )

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    pairs = pd.read_csv(args.pairs)

    required_columns = [
        "sample_id",
        "patient_id",
        "barcode",
        "patch_h5_path",
        "patch_index",
        "gene_sentence",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in pairs.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Pairs CSV is missing: {missing_columns}"
        )

    pairs = pairs.dropna(
        subset=[
            "patch_h5_path",
            "patch_index",
            "gene_sentence",
        ]
    ).reset_index(drop=True)

    if args.max_rows is not None:
        pairs = pairs.head(args.max_rows).copy()

    if pairs.empty:
        raise RuntimeError("No usable pairs were found.")

    args.output_h5.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_metadata.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Pairs:       {len(pairs):,}")
    print(f"Samples:     {pairs['sample_id'].nunique()}")
    print(f"Patients:    {pairs['patient_id'].nunique()}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Device:      {device}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Output HDF5: {args.output_h5}")

    model, preprocess, tokenizer = loki.utils.load_model(
        str(args.checkpoint),
        str(device),
    )

    model.eval()

    h5_handles = {}
    similarities = np.empty(
        len(pairs),
        dtype=np.float32,
    )

    output_file = None

    try:
        for start in tqdm(
            range(0, len(pairs), args.batch_size),
            desc="Extracting OmiCLIP embeddings",
        ):
            end = min(
                start + args.batch_size,
                len(pairs),
            )

            batch_df = pairs.iloc[start:end]

            (
                image_embeddings,
                st_embeddings,
                batch_similarity,
            ) = encode_batch(
                batch_df=batch_df,
                h5_handles=h5_handles,
                model=model,
                preprocess=preprocess,
                tokenizer=tokenizer,
                device=device,
            )

            # Create output datasets after discovering
            # the checkpoint's embedding dimension.
            if output_file is None:
                embedding_dim = image_embeddings.shape[1]

                if st_embeddings.shape[1] != embedding_dim:
                    raise ValueError(
                        "Image and ST embedding dimensions differ."
                    )

                output_file = h5py.File(
                    args.output_h5,
                    "w",
                )

                output_file.attrs["embedding_dim"] = (
                    embedding_dim
                )

                output_file.attrs["number_of_pairs"] = (
                    len(pairs)
                )

                output_file.attrs["checkpoint"] = str(
                    args.checkpoint.resolve()
                )

                chunk_rows = min(
                    max(args.batch_size, 1),
                    len(pairs),
                )

                output_file.create_dataset(
                    "image_embeddings",
                    shape=(
                        len(pairs),
                        embedding_dim,
                    ),
                    dtype="float32",
                    chunks=(
                        chunk_rows,
                        embedding_dim,
                    ),
                    compression="lzf",
                )

                output_file.create_dataset(
                    "st_embeddings",
                    shape=(
                        len(pairs),
                        embedding_dim,
                    ),
                    dtype="float32",
                    chunks=(
                        chunk_rows,
                        embedding_dim,
                    ),
                    compression="lzf",
                )

                output_file.create_dataset(
                    "matched_similarity",
                    shape=(len(pairs),),
                    dtype="float32",
                    chunks=(chunk_rows,),
                    compression="lzf",
                )

                print(
                    f"\nEmbedding dimension: {embedding_dim}"
                )

            output_file[
                "image_embeddings"
            ][start:end] = image_embeddings

            output_file[
                "st_embeddings"
            ][start:end] = st_embeddings

            output_file[
                "matched_similarity"
            ][start:end] = batch_similarity

            similarities[start:end] = batch_similarity

    finally:
        for handle in h5_handles.values():
            handle.close()

        if output_file is not None:
            output_file.close()

    pairs["embedding_row"] = np.arange(
        len(pairs),
        dtype=int,
    )

    pairs["matched_similarity"] = similarities

    pairs.to_csv(
        args.output_metadata,
        index=False,
    )

    print("\n====================================")
    print("OmiCLIP embedding extraction complete")
    print("====================================")
    print(f"Pairs processed: {len(pairs):,}")
    print(
        "Mean matched similarity: "
        f"{similarities.mean():.4f}"
    )
    print(
        "Median matched similarity: "
        f"{np.median(similarities):.4f}"
    )
    print(f"Embeddings: {args.output_h5.resolve()}")
    print(f"Metadata:   {args.output_metadata.resolve()}")


if __name__ == "__main__":
    main()