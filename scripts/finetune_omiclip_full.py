from pathlib import Path
import math
import random

import h5py
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from loki.utils import load_model


# ============================================================
# Paths
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
# Split
# ============================================================

TRAIN_SAMPLES = [f"INT{i}" for i in range(1, 19)]
VAL_SAMPLES = ["INT19", "INT20", "INT21"]
TEST_SAMPLES = ["INT22", "INT23", "INT24"]


# ============================================================
# Training configuration
# ============================================================

SEED = 42

EPOCHS = 5

# Start conservative.
# Increase if the larger GPU permits.
BATCH_SIZE = 8

LEARNING_RATE = 2e-6
WEIGHT_DECAY = 0.01

NUM_WORKERS = 4

WARMUP_RATIO = 0.05

GRAD_CLIP_NORM = 1.0

USE_GRAD_CHECKPOINTING = True


# ============================================================
# Helpers
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_barcode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")

    return str(value)


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
# Dataset
# ============================================================

class HESTOmiCLIPDataset(Dataset):

    def __init__(
        self,
        dataframe,
        preprocess,
        patch_dir,
        text_column,
    ):
        self.df = dataframe.reset_index(drop=True).copy()

        self.preprocess = preprocess
        self.patch_dir = Path(patch_dir)
        self.text_column = text_column

        self.barcode_lookup = {}

        # HDF5 handles are opened lazily inside worker processes.
        self._handles = {}

        samples = sorted(
            self.df["sample_id"]
            .astype(str)
            .unique()
        )

        print(
            f"Building barcode lookup for "
            f"{len(samples)} samples..."
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

            with h5py.File(path, "r") as handle:

                if "barcode" not in handle:
                    raise KeyError(
                        f"{path} does not contain "
                        f"'barcode'. Keys: "
                        f"{list(handle.keys())}"
                    )

                barcodes = [
                    decode_barcode(x)
                    for x in handle["barcode"][:]
                ]

            self.barcode_lookup[sample_id] = {
                barcode: index
                for index, barcode
                in enumerate(barcodes)
            }

    def __len__(self):
        return len(self.df)

    def _get_handle(self, sample_id):

        if sample_id not in self._handles:

            path = (
                self.patch_dir
                / f"{sample_id}.h5"
            )

            self._handles[sample_id] = h5py.File(
                path,
                "r",
            )

        return self._handles[sample_id]

    def __getitem__(self, index):

        row = self.df.iloc[index]

        sample_id = str(
            row["sample_id"]
        )

        barcode = str(
            row["barcode"]
        )

        sentence = str(
            row[self.text_column]
        )

        lookup = self.barcode_lookup[
            sample_id
        ]

        if barcode not in lookup:
            raise KeyError(
                f"Barcode {barcode} "
                f"not found in {sample_id}.h5"
            )

        patch_index = lookup[barcode]

        handle = self._get_handle(
            sample_id
        )

        if "img" not in handle:
            raise KeyError(
                f"{sample_id}.h5 does not "
                f"contain 'img'. "
                f"Keys: {list(handle.keys())}"
            )

        image_array = handle[
            "img"
        ][patch_index]

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

        if image_array.dtype != np.uint8:

            if image_array.max() <= 1.0:
                image_array = (
                    image_array * 255
                )

            image_array = np.clip(
                image_array,
                0,
                255,
            ).astype(np.uint8)

        image = Image.fromarray(
            image_array
        ).convert("RGB")

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
# Token handling
# ============================================================

def move_tokens_to_device(
    tokens,
    device,
):

    if isinstance(tokens, dict):

        return {
            key: value.to(device)
            if torch.is_tensor(value)
            else value

            for key, value
            in tokens.items()
        }

    return tokens.to(device)


def encode_text(
    model,
    tokens,
):

    # Normal OpenCLIP tokenizer output.
    if torch.is_tensor(tokens):
        return model.encode_text(tokens)

    # HF-style tokenizer dictionary.
    if isinstance(tokens, dict):

        input_ids = tokens.get(
            "input_ids"
        )

        attention_mask = tokens.get(
            "attention_mask"
        )

        if input_ids is None:
            raise RuntimeError(
                "Tokenizer dictionary did not "
                "contain input_ids."
            )

        # Newer OpenCLIP.
        try:
            return model.encode_text(
                input_ids,
                text_valid=attention_mask,
            )

        # Older Loki/OpenCLIP.
        except TypeError:
            return model.encode_text(
                input_ids
            )

    raise TypeError(
        f"Unsupported token type: "
        f"{type(tokens)}"
    )


# ============================================================
# Contrastive loss
# ============================================================

def contrastive_loss(
    image_features,
    text_features,
    model,
):

    image_features = F.normalize(
        image_features,
        dim=-1,
    )

    text_features = F.normalize(
        text_features,
        dim=-1,
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

    image_to_text = F.cross_entropy(
        logits,
        labels,
    )

    text_to_image = F.cross_entropy(
        logits.T,
        labels,
    )

    loss = (
        image_to_text
        + text_to_image
    ) / 2

    matched_similarity = (
        image_features
        * text_features
    ).sum(dim=-1).mean()

    image_accuracy = (
        logits.argmax(dim=1)
        == labels
    ).float().mean()

    text_accuracy = (
        logits.argmax(dim=0)
        == labels
    ).float().mean()

    retrieval_accuracy = (
        image_accuracy
        + text_accuracy
    ) / 2

    return (
        loss,
        matched_similarity,
        retrieval_accuracy,
    )


# ============================================================
# Optimizer
# ============================================================

def build_optimizer(model):

    decay = []
    no_decay = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        if (
            parameter.ndim < 2
            or "bias" in name.lower()
            or "norm" in name.lower()
            or "logit_scale" in name
        ):
            no_decay.append(parameter)

        else:
            decay.append(parameter)

    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=LEARNING_RATE,
    )


# ============================================================
# Scheduler
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

    def lr_lambda(step):

        if step < warmup_steps:

            return (
                step + 1
            ) / warmup_steps

        progress = (
            step - warmup_steps
        ) / max(
            1,
            total_steps
            - warmup_steps,
        )

        return (
            0.5
            * (
                1
                + math.cos(
                    math.pi
                    * progress
                )
            )
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda,
    )


# ============================================================
# One epoch
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

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_similarity = 0.0
    total_accuracy = 0.0
    total_rows = 0

    for batch_index, batch in enumerate(
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

        tokens = move_tokens_to_device(
            tokens,
            device,
        )

        batch_size = images.shape[0]

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
                    device.type == "cuda"
                ),
            ):

                image_features = (
                    model.encode_image(
                        images
                    )
                )

                text_features = encode_text(
                    model,
                    tokens,
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

                scheduler.step()

        total_loss += (
            loss.item()
            * batch_size
        )

        total_similarity += (
            similarity.item()
            * batch_size
        )

        total_accuracy += (
            accuracy.item()
            * batch_size
        )

        total_rows += batch_size

        if (
            training
            and batch_index % 100 == 0
        ):

            print(
                f"  batch "
                f"{batch_index:5d}/"
                f"{len(loader):5d} | "
                f"loss "
                f"{loss.item():.4f} | "
                f"sim "
                f"{similarity.item():.4f} | "
                f"retrieval "
                f"{accuracy.item():.4f}",
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
# Main
# ============================================================

def main():

    set_seed(SEED)

    if not PAIRS_CSV.exists():
        raise FileNotFoundError(
            PAIRS_CSV
        )

    if not PRETRAINED_CHECKPOINT.exists():
        raise FileNotFoundError(
            PRETRAINED_CHECKPOINT
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
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    # --------------------------------------------------------
    # Load pair table
    # --------------------------------------------------------

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

    text_column = detect_text_column(
        df
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
    )
    print(
        "Validation:",
        len(val_df),
    )
    print(
        "Held-out test:",
        len(test_df),
    )

    print(
        "\nTEST SAMPLES ARE NOT "
        "USED FOR FINE-TUNING."
    )

    # --------------------------------------------------------
    # Load OmiCLIP
    # --------------------------------------------------------

    print(
        "\nLoading pretrained OmiCLIP..."
    )

    model, preprocess, tokenizer = (
        load_model(
            str(
                PRETRAINED_CHECKPOINT
            ),
            device,
        )
    )

    # Full fine-tuning.
    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = True

    if (
        USE_GRAD_CHECKPOINTING
        and hasattr(
            model,
            "set_grad_checkpointing",
        )
    ):
        print(
            "Enabling gradient "
            "checkpointing..."
        )

        model.set_grad_checkpointing(
            True
        )

    model.train()

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
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
            "trainable. This is not "
            "full fine-tuning."
        )

    # --------------------------------------------------------
    # Datasets
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

    train_loader = DataLoader(
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

    val_loader = DataLoader(
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

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    bf16_supported = (
        torch.cuda.is_bf16_supported()
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
            torch.cuda.amp.GradScaler()
        )

        print(
            "Using FP16 mixed precision"
        )

    # --------------------------------------------------------
    # Optimizer / scheduler
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

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print(
            f"\n"
            f"=============================="
        )
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )
        print(
            f"=============================="
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            tokenizer=tokenizer,
            device=device,
            amp_dtype=amp_dtype,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        print(
            "\nValidation..."
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            amp_dtype=amp_dtype,
            optimizer=None,
            scheduler=None,
            scaler=None,
        )

        row = {
            "epoch":
                epoch,

            "train_loss":
                train_metrics["loss"],

            "train_similarity":
                train_metrics[
                    "similarity"
                ],

            "train_retrieval_accuracy":
                train_metrics[
                    "retrieval_accuracy"
                ],

            "val_loss":
                val_metrics["loss"],

            "val_similarity":
                val_metrics[
                    "similarity"
                ],

            "val_retrieval_accuracy":
                val_metrics[
                    "retrieval_accuracy"
                ],
        }

        history.append(row)

        print(
            "\nEpoch results:"
        )

        print(
            f"Train loss: "
            f"{row['train_loss']:.5f}"
        )

        print(
            f"Val loss:   "
            f"{row['val_loss']:.5f}"
        )

        print(
            f"Train sim:  "
            f"{row['train_similarity']:.5f}"
        )

        print(
            f"Val sim:    "
            f"{row['val_similarity']:.5f}"
        )

        print(
            f"Val retrieval: "
            f"{row['val_retrieval_accuracy']:.5f}"
        )

        # Save history each epoch.
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

        # Last checkpoint.
        last_path = (
            OUT_DIR
            / "last.pt"
        )

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "val_loss":
                    row["val_loss"],

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
            last_path,
        )

        # Best checkpoint determined only from validation.
        if (
            row["val_loss"]
            < best_val_loss
        ):

            best_val_loss = (
                row["val_loss"]
            )

            best_path = (
                OUT_DIR
                / "best.pt"
            )

            torch.save(
                {
                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "val_loss":
                        best_val_loss,

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
                best_path,
            )

            print(
                f"\nNEW BEST MODEL: "
                f"{best_path}"
            )

        print(
            f"Best validation loss: "
            f"{best_val_loss:.5f}"
        )

    print(
        "\n================================"
    )
    print(
        "FULL OmiCLIP FINE-TUNING DONE"
    )
    print(
        "================================"
    )

    print(
        "Best checkpoint:",
        OUT_DIR / "best.pt",
    )

    print(
        "History:",
        OUT_DIR
        / "training_history.csv",
    )


if __name__ == "__main__":
    main()