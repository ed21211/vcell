from pathlib import Path
import copy
import random

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_NPZ = (
    PROJECT_DIR
    / "features"
    / "hest_ccrcc_joined"
    / "hest_ccrcc_virchow2_factor_dataset.npz"
)

META_CSV = (
    PROJECT_DIR
    / "features"
    / "hest_ccrcc_joined"
    / "hest_ccrcc_virchow2_factor_metadata.csv"
)

OUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "virchow_factor_mlp"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SPLIT
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
# CONFIG
# ============================================================

SEED = 42

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 100
PATIENCE = 10

DROPOUT = 0.25


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# MODEL
# ============================================================

class FactorMLP(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_dim,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                256,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                output_dim,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(x)


# ============================================================
# METRICS
# ============================================================

def safe_pearson(
    y_true,
    y_pred,
):
    try:
        return pearsonr(
            y_true,
            y_pred,
        )[0]
    except Exception:
        return np.nan


def safe_spearman(
    y_true,
    y_pred,
):
    try:
        return spearmanr(
            y_true,
            y_pred,
        )[0]
    except Exception:
        return np.nan


def evaluate_metrics(
    y_true,
    y_pred,
    factor_names,
    split_name,
):
    rows = []

    for i, factor in enumerate(
        factor_names
    ):

        true = y_true[:, i]
        pred = y_pred[:, i]

        mse = mean_squared_error(
            true,
            pred,
        )

        rmse = np.sqrt(
            mse
        )

        r2 = r2_score(
            true,
            pred,
        )

        pearson = safe_pearson(
            true,
            pred,
        )

        spearman = safe_spearman(
            true,
            pred,
        )

        rows.append(
            {
                "split":
                    split_name,

                "factor":
                    factor,

                "mse":
                    mse,

                "rmse":
                    rmse,

                "r2":
                    r2,

                "pearson":
                    pearson,

                "spearman":
                    spearman,
            }
        )

    return rows


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    X,
    device,
):
    model.eval()

    tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    loader = DataLoader(
        TensorDataset(
            tensor
        ),
        batch_size=1024,
        shuffle=False,
    )

    outputs = []

    with torch.no_grad():

        for (batch,) in loader:

            batch = batch.to(
                device
            )

            prediction = model(
                batch
            )

            outputs.append(
                prediction
                .cpu()
                .numpy()
            )

    return np.concatenate(
        outputs,
        axis=0,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "================================"
    )
    print(
        "OmiCLIP + MLP FACTOR PREDICTION"
    )
    print(
        "================================"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device,
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    data = np.load(
        DATA_NPZ,
        allow_pickle=True,
    )

    X = data["X"].astype(
        np.float32
    )

    y = data["y"].astype(
        np.float32
    )

    factor_names = data[
        "factor_names"
    ].astype(str)

    metadata = pd.read_csv(
        META_CSV
    )

    print(
        "X:",
        X.shape,
    )

    print(
        "y:",
        y.shape,
    )

    print(
        "metadata:",
        metadata.shape,
    )

    print(
        "Factors:",
        factor_names,
    )

    if (
        len(X)
        != len(y)
        or len(X)
        != len(metadata)
    ):
        raise RuntimeError(
            "X/y/metadata row mismatch."
        )

    metadata[
        "sample_id"
    ] = (
        metadata["sample_id"]
        .astype(str)
        .str.upper()
    )

    # --------------------------------------------------------
    # Split masks
    # --------------------------------------------------------

    train_mask = (
        metadata[
            "sample_id"
        ].isin(
            TRAIN_SAMPLES
        )
    ).to_numpy()

    val_mask = (
        metadata[
            "sample_id"
        ].isin(
            VAL_SAMPLES
        )
    ).to_numpy()

    test_mask = (
        metadata[
            "sample_id"
        ].isin(
            TEST_SAMPLES
        )
    ).to_numpy()

    X_train = X[
        train_mask
    ]

    X_val = X[
        val_mask
    ]

    X_test = X[
        test_mask
    ]

    y_train = y[
        train_mask
    ]

    y_val = y[
        val_mask
    ]

    y_test = y[
        test_mask
    ]

    print(
        "\nSplit sizes:"
    )

    print(
        "Train:",
        len(X_train),
        TRAIN_SAMPLES,
    )

    print(
        "Val:",
        len(X_val),
        VAL_SAMPLES,
    )

    print(
        "Test:",
        len(X_test),
        TEST_SAMPLES,
    )

    # --------------------------------------------------------
    # Standardise using TRAIN ONLY
    # --------------------------------------------------------

    print(
        "\nStandardising using training split only..."
    )

    x_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(
        X_train
    ).astype(
        np.float32
    )

    X_val = x_scaler.transform(
        X_val
    ).astype(
        np.float32
    )

    X_test = x_scaler.transform(
        X_test
    ).astype(
        np.float32
    )

    y_scaler = StandardScaler()

    y_train_scaled = (
        y_scaler.fit_transform(
            y_train
        )
        .astype(
            np.float32
        )
    )

    y_val_scaled = (
        y_scaler.transform(
            y_val
        )
        .astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train_scaled,
            dtype=torch.float32,
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = FactorMLP(
        input_dim=X.shape[1],
        output_dim=y.shape[1],
    ).to(
        device
    )

    print(
        "\nModel:"
    )

    print(
        model
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.MSELoss()

    # --------------------------------------------------------
    # Training with early stopping
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    best_state = None

    patience_counter = 0

    history = []

    print(
        "\nTraining..."
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        train_loss_sum = 0.0
        train_rows = 0

        for (
            batch_X,
            batch_y,
        ) in train_loader:

            batch_X = batch_X.to(
                device,
                non_blocking=True,
            )

            batch_y = batch_y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(
                batch_X
            )

            loss = criterion(
                pred,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            train_loss_sum += (
                loss.item()
                * len(batch_X)
            )

            train_rows += len(
                batch_X
            )

        train_loss = (
            train_loss_sum
            / train_rows
        )

        # --------------------------------------------
        # Validation
        # --------------------------------------------

        model.eval()

        with torch.no_grad():

            val_tensor = torch.tensor(
                X_val,
                dtype=torch.float32,
                device=device,
            )

            val_target = torch.tensor(
                y_val_scaled,
                dtype=torch.float32,
                device=device,
            )

            val_pred = model(
                val_tensor
            )

            val_loss = criterion(
                val_pred,
                val_target,
            ).item()

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "val_loss":
                    val_loss,
            }
        )

        print(
            f"Epoch "
            f"{epoch:03d} | "
            f"train "
            f"{train_loss:.6f} | "
            f"val "
            f"{val_loss:.6f}",
            flush=True,
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if (
            patience_counter
            >= PATIENCE
        ):
            print(
                f"\nEarly stopping at "
                f"epoch {epoch}."
            )

            break

    # --------------------------------------------------------
    # Restore best validation model
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    print(
        "\nBest validation loss:",
        best_val_loss,
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print(
        "\nPredicting..."
    )

    train_pred_scaled = predict(
        model,
        X_train,
        device,
    )

    val_pred_scaled = predict(
        model,
        X_val,
        device,
    )

    test_pred_scaled = predict(
        model,
        X_test,
        device,
    )

    train_pred = (
        y_scaler.inverse_transform(
            train_pred_scaled
        )
    )

    val_pred = (
        y_scaler.inverse_transform(
            val_pred_scaled
        )
    )

    test_pred = (
        y_scaler.inverse_transform(
            test_pred_scaled
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    rows = []

    rows += evaluate_metrics(
        y_train,
        train_pred,
        factor_names,
        "train",
    )

    rows += evaluate_metrics(
        y_val,
        val_pred,
        factor_names,
        "val",
    )

    rows += evaluate_metrics(
        y_test,
        test_pred,
        factor_names,
        "test",
    )

    metrics = pd.DataFrame(
        rows
    )

    print(
        "\nResults:"
    )

    print(
        metrics
    )

    # --------------------------------------------------------
    # Predictions dataframe
    # --------------------------------------------------------

    prediction_frames = []

    for (
        split_name,
        mask,
        true_values,
        predicted_values,
    ) in [
        (
            "train",
            train_mask,
            y_train,
            train_pred,
        ),
        (
            "val",
            val_mask,
            y_val,
            val_pred,
        ),
        (
            "test",
            test_mask,
            y_test,
            test_pred,
        ),
    ]:

        frame = metadata.loc[
            mask
        ].copy()

        frame[
            "split"
        ] = split_name

        for index, factor in enumerate(
            factor_names
        ):

            frame[
                f"true_{factor}"
            ] = true_values[
                :,
                index
            ]

            frame[
                f"pred_{factor}"
            ] = predicted_values[
                :,
                index
            ]

        prediction_frames.append(
            frame
        )

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    metrics_path = (
        OUT_DIR
        / "mlp_metrics.csv"
    )

    predictions_path = (
        OUT_DIR
        / "mlp_predictions.csv"
    )

    model_path = (
        OUT_DIR
        / "mlp_model.pt"
    )

    history_path = (
        OUT_DIR
        / "training_history.csv"
    )

    scaler_path = (
        OUT_DIR
        / "mlp_scalers.joblib"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_path,
        index=False,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_dim":
                X.shape[1],

            "output_dim":
                y.shape[1],

            "factor_names":
                factor_names.tolist(),

            "best_val_loss":
                best_val_loss,
        },
        model_path,
    )

    joblib.dump(
        {
            "x_scaler":
                x_scaler,

            "y_scaler":
                y_scaler,
        },
        scaler_path,
    )

    print(
        "\nSaved metrics:",
        metrics_path,
    )

    print(
        "Saved predictions:",
        predictions_path,
    )

    print(
        "Saved model:",
        model_path,
    )

    print(
        "Saved history:",
        history_path,
    )


if __name__ == "__main__":
    main()