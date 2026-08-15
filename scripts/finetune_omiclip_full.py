from pathlib import Path
import math
import random
import time

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

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

PRETRAINED_CHECKPOINT = (
    PROJECT_DIR
    / "models"
    / "omiclip"
    / "checkpoint.pt"
)

OUT_DIR = (
    PROJECT_DIR
    / "models"
    / "omiclip_ccrcc_full"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATA SPLIT
# ============================================================

TRAIN_SAMPLES = [
    f"INT{i}"
    for i in range(1, 19)
]

VAL_SAMPLES = [
    "INT19",
    "INT20",
    "INT21",
]

TEST_SAMPLES = [
    "INT22",
    "INT23",
    "INT24",
]


# ============================================================
# TRAINING CONFIG
# ============================================================

SEED = 42

EPOCHS = 5

BATCH_SIZE = 8
NUM_WORKERS = 8
USE_GRAD_CHECKPOINTING = True

LEARNING_RATE = 2e-6
WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.05

GRAD_CLIP_NORM = 1.0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TEXT COLUMN
# ============================================================

def detect_text_column(df):
    candidates = [
        "gene_sentence",
        "text",
        "sentence",
        "st_text",
        "caption",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not find gene-sentence column.\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# VALIDATE PAIR TABLE
# ============================================================

def validate_pair_indices(df):
    print(
        "\nValidating patch indices...",
        flush=True,
    )

    required_columns = [
        "sample_id",
        "barcode",
        "patch_index",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    problems = []

    for sample_id, sample_df in df.groupby(
        "sample_id"
    ):
        path = (
            PATCH_DIR
            / f"{sample_id}.h5"
        )

        if not path.exists():
            problems.append(
                f"{sample_id}: missing {path}"
            )
            continue

        with h5py.File(
            path,
            "r",
        ) as handle:

            if "img" not in handle:
                problems.append(
                    f"{sample_id}: "
                    f"'img' missing. "
                    f"Keys={list(handle.keys())}"
                )
                continue

            n_patches = len(
                handle["img"]
            )

        indices = (
            sample_df["patch_index"]
            .astype(int)
            .to_numpy()
        )

        min_index = int(
            indices.min()
        )

        max_index = int(
            indices.max()
        )

        valid = (
            min_index >= 0
            and max_index < n_patches
        )

        print(
            f"{sample_id}: "
            f"pairs={len(sample_df):,}, "
            f"patches={n_patches:,}, "
            f"indices={min_index}..{max_index}, "
            f"OK={valid}",
            flush=True,
        )

        if not valid:
            problems.append(
                f"{sample_id}: "
                f"indices {min_index}..{max_index} "
                f"outside 0..{n_patches - 1}"
            )

    if problems:
        raise RuntimeError(
            "\nPatch index validation failed:\n"
            + "\n".join(problems)
        )

    print(
        "All patch indices valid.",
        flush=True,
    )


# ============================================================
# DATASET
# ============================================================

class HESTOmiCLIPDataset(Dataset):

    def __init__(
        self,
        dataframe,
        preprocess,
        patch_dir,
        text_column,
    ):
        self.df = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

        self.preprocess = preprocess
        self.patch_dir = Path(
            patch_dir
        )
        self.text_column = (
            text_column
        )

        # Each DataLoader worker lazily
        # creates its own HDF5 handles.
        self._handles = {}

        samples = sorted(
            self.df["sample_id"]
            .astype(str)
            .unique()
        )

        print(
            f"Preparing patch access for "
            f"{len(samples)} samples...",
            flush=True,
        )

        for sample_id in samples:

            path = (
                self.patch_dir
                / f"{sample_id}.h5"
            )

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing patch file: {path}"
                )

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

        barcode = str(
            row["barcode"]
        )

        sentence = str(
            row[self.text_column]
        )

        # IMPORTANT:
        # Use the pairing already created by
        # prepare_omiclip_pairs.py.
        patch_index = int(
            row["patch_index"]
        )

        handle = self._get_handle(
            sample_id
        )

        if "img" not in handle:
            raise KeyError(
                f"{sample_id}.h5 does not "
                f"contain 'img'. "
                f"Keys: {list(handle.keys())}"
            )

        image_array = (
            handle["img"][
                patch_index
            ]
        )

        # Handle CHW images if needed.
        if (
            image_array.ndim == 3
            and image_array.shape[0] == 3
            and image_array.shape[-1] != 3
        ):
            image_array = (
                np.transpose(
                    image_array,
                    (1, 2, 0),
                )
            )

        # Convert to uint8.
        if (
            image_array.dtype
            != np.uint8
        ):
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

            image_array = (
                np.clip(
                    image_array,
                    0,
                    255,
                )
                .astype(
                    np.uint8
                )
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
            sentence,
            sample_id,
            barcode,
        )


# ============================================================
# TOKEN HANDLING
# ============================================================

def move_tokens_to_device(
    tokens,
    device,
):
    if isinstance(
        tokens,
        dict,
    ):
        return {
            key: (
                value.to(
                    device,
                    non_blocking=True,
                )
                if torch.is_tensor(
                    value
                )
                else value
            )
            for key, value
            in tokens.items()
        }

    if torch.is_tensor(
        tokens
    ):
        return tokens.to(
            device,
            non_blocking=True,
        )

    raise TypeError(
        f"Unsupported tokenizer "
        f"output type: {type(tokens)}"
    )


def encode_text(
    model,
    tokens,
):
    # Standard OpenCLIP tokenizer.
    if torch.is_tensor(
        tokens
    ):
        return model.encode_text(
            tokens
        )

    # HuggingFace-style tokenizer.
    if isinstance(
        tokens,
        dict,
    ):
        input_ids = tokens.get(
            "input_ids"
        )

        attention_mask = (
            tokens.get(
                "attention_mask"
            )
        )

        if input_ids is None:
            raise RuntimeError(
                "Tokenizer dictionary "
                "does not contain input_ids."
            )

        try:
            return model.encode_text(
                input_ids,
                text_valid=attention_mask,
            )

        except TypeError:
            return model.encode_text(
                input_ids
            )

    raise TypeError(
        f"Unsupported tokens: "
        f"{type(tokens)}"
    )


# ============================================================
# CONTRASTIVE LOSS
# ============================================================

def contrastive_loss(
    image_features,
    text_features,
    model,
):
    image_features = (
        F.normalize(
            image_features,
            dim=-1,
        )
    )

    text_features = (
        F.normalize(
            text_features,
            dim=-1,
        )
    )

    logit_scale = (
        model.logit_scale
        .exp()
        .clamp(max=100)
    )

    logits = (
        logit_scale
        * image_features
        @ text_features.T
    )

    labels = torch.arange(
        logits.shape[0],
        device=logits.device,
    )

    image_to_text_loss = (
        F.cross_entropy(
            logits,
            labels,
        )
    )

    text_to_image_loss = (
        F.cross_entropy(
            logits.T,
            labels,
        )
    )

    loss = (
        image_to_text_loss
        + text_to_image_loss
    ) / 2.0

    matched_similarity = (
        image_features
        * text_features
    ).sum(
        dim=-1
    ).mean()

    image_accuracy = (
        (
            logits.argmax(
                dim=1
            )
            == labels
        )
        .float()
        .mean()
    )

    text_accuracy = (
        (
            logits.argmax(
                dim=0
            )
            == labels
        )
        .float()
        .mean()
    )

    retrieval_accuracy = (
        image_accuracy
        + text_accuracy
    ) / 2.0

    return (
        loss,
        matched_similarity,
        retrieval_accuracy,
    )


# ============================================================
# OPTIMIZER
# ============================================================

def build_optimizer(
    model,
):
    decay = []
    no_decay = []

    for (
        name,
        parameter,
    ) in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if (
            parameter.ndim < 2
            or "bias" in name.lower()
            or "norm" in name.lower()
            or "logit_scale" in name
        ):
            no_decay.append(
                parameter
            )

        else:
            decay.append(
                parameter
            )

    optimizer = (
        torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "weight_decay":
                        WEIGHT_DECAY,
                },
                {
                    "params": no_decay,
                    "weight_decay":
                        0.0,
                },
            ],
            lr=LEARNING_RATE,
            foreach=False,
        )
    )

    return optimizer


# ============================================================
# LR SCHEDULER
# ============================================================

def build_scheduler(
    optimizer,
    total_steps,
):
    warmup_steps = max(
        1,
        int(
            total_steps
            * WARMUP_RATIO
        ),
    )

    def lr_lambda(
        step,
    ):
        if step < warmup_steps:

            return (
                step + 1
            ) / warmup_steps

        progress = (
            step
            - warmup_steps
        ) / max(
            1,
            total_steps
            - warmup_steps,
        )

        return (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * progress
                )
            )
        )

    return (
        torch.optim.lr_scheduler
        .LambdaLR(
            optimizer,
            lr_lambda,
        )
    )


# ============================================================
# RUN ONE EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    tokenizer,
    device,
    amp_dtype,
    optimizer=None,
    scheduler=None,
    scaler=None,
):
    training = (
        optimizer is not None
    )

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_similarity = 0.0
    total_accuracy = 0.0
    total_rows = 0

    epoch_start = time.time()

    for (
        batch_index,
        batch,
    ) in enumerate(
        loader,
        start=1,
    ):
        (
            images,
            sentences,
            sample_ids,
            barcodes,
        ) = batch

        images = images.to(
            device,
            non_blocking=True,
        )

        tokens = tokenizer(
            list(sentences)
        )

        tokens = (
            move_tokens_to_device(
                tokens,
                device,
            )
        )

        current_batch_size = (
            images.shape[0]
        )

        if training:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):
                image_features = (
                    model.encode_image(
                        images
                    )
                )

                text_features = (
                    encode_text(
                        model,
                        tokens,
                    )
                )

                (
                    loss,
                    similarity,
                    accuracy,
                ) = contrastive_loss(
                    image_features,
                    text_features,
                    model,
                )

            if training:

                if scaler is not None:

                    scaler.scale(
                        loss
                    ).backward()

                    scaler.unscale_(
                        optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        GRAD_CLIP_NORM,
                    )

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        GRAD_CLIP_NORM,
                    )

                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

        total_loss += (
            loss.item()
            * current_batch_size
        )

        total_similarity += (
            similarity.item()
            * current_batch_size
        )

        total_accuracy += (
            accuracy.item()
            * current_batch_size
        )

        total_rows += (
            current_batch_size
        )

        if (
            training
            and batch_index % 100 == 0
        ):
            elapsed = (
                time.time()
                - epoch_start
            )

            batches_per_second = (
                batch_index
                / elapsed
            )

            remaining_batches = (
                len(loader)
                - batch_index
            )

            eta_minutes = (
                remaining_batches
                / max(
                    batches_per_second,
                    1e-9,
                )
                / 60.0
            )

            lr = (
                optimizer
                .param_groups[0]["lr"]
            )

            print(
                f"batch "
                f"{batch_index:5d}/"
                f"{len(loader):5d} | "
                f"loss "
                f"{loss.item():.4f} | "
                f"sim "
                f"{similarity.item():.4f} | "
                f"retrieval "
                f"{accuracy.item():.4f} | "
                f"lr "
                f"{lr:.2e} | "
                f"ETA "
                f"{eta_minutes:.1f} min",
                flush=True,
            )

    return {
        "loss":
            total_loss
            / total_rows,

        "similarity":
            total_similarity
            / total_rows,

        "retrieval_accuracy":
            total_accuracy
            / total_rows,
    }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    epoch,
    val_loss,
):
    print(
        f"Saving checkpoint: {path}",
        flush=True,
    )

    torch.save(
        {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "val_loss":
                val_loss,

            "train_samples":
                TRAIN_SAMPLES,

            "val_samples":
                VAL_SAMPLES,

            "test_samples":
                TEST_SAMPLES,

            "learning_rate":
                LEARNING_RATE,

            "batch_size":
                BATCH_SIZE,
        },
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(
        SEED
    )

    print(
        "================================"
    )
    print(
        "FULL OmiCLIP ccRCC FINE-TUNING"
    )
    print(
        "================================"
    )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if not PAIRS_CSV.exists():
        raise FileNotFoundError(
            f"Missing: {PAIRS_CSV}"
        )

    if not (
        PRETRAINED_CHECKPOINT.exists()
    ):
        raise FileNotFoundError(
            f"Missing: "
            f"{PRETRAINED_CHECKPOINT}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for "
            "full OmiCLIP fine-tuning."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0),
    )

    gpu_memory = (
        torch.cuda
        .get_device_properties(0)
        .total_memory
        / 1024**3
    )

    print(
        f"GPU VRAM: "
        f"{gpu_memory:.2f} GB"
    )

    # --------------------------------------------------------
    # Load pairs
    # --------------------------------------------------------

    print(
        "\nLoading pair table..."
    )

    df = pd.read_csv(
        PAIRS_CSV
    )

    print(
        "Pair table:",
        df.shape,
    )

    print(
        "Columns:",
        list(df.columns),
    )

    text_column = (
        detect_text_column(
            df
        )
    )

    print(
        "Gene sentence column:",
        text_column,
    )

    df["sample_id"] = (
        df["sample_id"]
        .astype(str)
        .str.upper()
    )

    df["barcode"] = (
        df["barcode"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Validate pair table
    # --------------------------------------------------------

    validate_pair_indices(
        df
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train_df = df[
        df["sample_id"].isin(
            TRAIN_SAMPLES
        )
    ].copy()

    val_df = df[
        df["sample_id"].isin(
            VAL_SAMPLES
        )
    ].copy()

    test_df = df[
        df["sample_id"].isin(
            TEST_SAMPLES
        )
    ].copy()

    print(
        "\nSplit sizes:"
    )

    print(
        "Train:",
        len(train_df),
        TRAIN_SAMPLES,
    )

    print(
        "Validation:",
        len(val_df),
        VAL_SAMPLES,
    )

    print(
        "Held-out test:",
        len(test_df),
        TEST_SAMPLES,
    )

    if len(train_df) == 0:
        raise RuntimeError(
            "Training split is empty."
        )

    if len(val_df) == 0:
        raise RuntimeError(
            "Validation split is empty."
        )

    if len(test_df) == 0:
        raise RuntimeError(
            "Test split is empty."
        )

    print(
        "\nIMPORTANT:"
    )

    print(
        "INT22–INT24 are NOT used "
        "during fine-tuning."
    )

    # --------------------------------------------------------
    # Load OmiCLIP
    # --------------------------------------------------------

    print(
        "\nLoading pretrained OmiCLIP...",
        flush=True,
    )

    load_start = time.time()

    (
        model,
        preprocess,
        tokenizer,
    ) = load_model(
        str(
            PRETRAINED_CHECKPOINT
        ),
        device,
    )

    print(
        f"OmiCLIP loaded in "
        f"{(time.time() - load_start) / 60:.2f} minutes",
        flush=True,
    )

    # --------------------------------------------------------
    # Full fine-tuning
    # --------------------------------------------------------

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = (
            True
        )

    if (
        USE_GRAD_CHECKPOINTING
        and hasattr(
            model,
            "set_grad_checkpointing",
        )
    ):
        print(
            "Enabling gradient "
            "checkpointing...",
            flush=True,
        )

        model.set_grad_checkpointing(
            True
        )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    if (
        trainable_parameters
        != total_parameters
    ):
        raise RuntimeError(
            "Not all parameters are "
            "trainable. Full fine-tuning "
            "is not enabled."
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        "\nPreparing datasets..."
    )

    train_dataset = (
        HESTOmiCLIPDataset(
            train_df,
            preprocess,
            PATCH_DIR,
            text_column,
        )
    )

    val_dataset = (
        HESTOmiCLIPDataset(
            val_df,
            preprocess,
            PATCH_DIR,
            text_column,
        )
    )

    train_loader = (
        DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(
                NUM_WORKERS > 0
            ),
        )
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(
                NUM_WORKERS > 0
            ),
        )
    )

    print(
        "Training batches:",
        len(train_loader),
    )

    print(
        "Validation batches:",
        len(val_loader),
    )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    bf16_supported = (
        torch.cuda
        .is_bf16_supported()
    )

    if bf16_supported:

        amp_dtype = (
            torch.bfloat16
        )

        scaler = None

        print(
            "Using BF16 mixed precision"
        )

    else:

        amp_dtype = (
            torch.float16
        )

        scaler = (
            torch.cuda.amp
            .GradScaler()
        )

        print(
            "Using FP16 mixed precision"
        )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = build_optimizer(
        model
    )

    total_steps = (
        len(train_loader)
        * EPOCHS
    )

    scheduler = build_scheduler(
        optimizer,
        total_steps,
    )

    print(
        "\nTraining configuration:"
    )

    print(
        "Epochs:",
        EPOCHS,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Weight decay:",
        WEIGHT_DECAY,
    )

    print(
        "Total optimizer steps:",
        total_steps,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    history = []

    training_start = (
        time.time()
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print(
            "\n"
            "=============================="
        )

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            "==============================",
            flush=True,
        )

        train_metrics = (
            run_epoch(
                model=model,
                loader=train_loader,
                tokenizer=tokenizer,
                device=device,
                amp_dtype=amp_dtype,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )
        )

        print(
            "\nValidation...",
            flush=True,
        )

        val_metrics = (
            run_epoch(
                model=model,
                loader=val_loader,
                tokenizer=tokenizer,
                device=device,
                amp_dtype=amp_dtype,
                optimizer=None,
                scheduler=None,
                scaler=None,
            )
        )

        row = {
            "epoch":
                epoch,

            "train_loss":
                train_metrics[
                    "loss"
                ],

            "train_similarity":
                train_metrics[
                    "similarity"
                ],

            "train_retrieval_accuracy":
                train_metrics[
                    "retrieval_accuracy"
                ],

            "val_loss":
                val_metrics[
                    "loss"
                ],

            "val_similarity":
                val_metrics[
                    "similarity"
                ],

            "val_retrieval_accuracy":
                val_metrics[
                    "retrieval_accuracy"
                ],
        }

        history.append(
            row
        )

        print(
            "\nEpoch results:"
        )

        print(
            f"Train loss: "
            f"{row['train_loss']:.6f}"
        )

        print(
            f"Val loss: "
            f"{row['val_loss']:.6f}"
        )

        print(
            f"Train similarity: "
            f"{row['train_similarity']:.6f}"
        )

        print(
            f"Val similarity: "
            f"{row['val_similarity']:.6f}"
        )

        print(
            f"Train retrieval: "
            f"{row['train_retrieval_accuracy']:.6f}"
        )

        print(
            f"Val retrieval: "
            f"{row['val_retrieval_accuracy']:.6f}"
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history_path = (
            OUT_DIR
            / "training_history.csv"
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )

        # ----------------------------------------------------
        # Save latest checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            path=(
                OUT_DIR
                / "last.pt"
            ),
            model=model,
            epoch=epoch,
            val_loss=row[
                "val_loss"
            ],
        )

        # ----------------------------------------------------
        # Best checkpoint based ONLY on validation
        # ----------------------------------------------------

        if (
            row["val_loss"]
            < best_val_loss
        ):
            best_val_loss = (
                row["val_loss"]
            )

            save_checkpoint(
                path=(
                    OUT_DIR
                    / "best.pt"
                ),
                model=model,
                epoch=epoch,
                val_loss=best_val_loss,
            )

            print(
                "\nNEW BEST MODEL",
                flush=True,
            )

        print(
            f"Best validation loss: "
            f"{best_val_loss:.6f}"
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    total_hours = (
        time.time()
        - training_start
    ) / 3600.0

    print(
        "\n"
        "================================"
    )

    print(
        "FULL OmiCLIP FINE-TUNING DONE"
    )

    print(
        "================================"
    )

    print(
        f"Training time: "
        f"{total_hours:.2f} hours"
    )

    print(
        "Best checkpoint:",
        OUT_DIR / "best.pt",
    )

    print(
        "Last checkpoint:",
        OUT_DIR / "last.pt",
    )

    print(
        "History:",
        OUT_DIR
        / "training_history.csv",
    )

    print(
        "\nINT22–INT24 remained "
        "completely held out."
    )


if __name__ == "__main__":
    main()