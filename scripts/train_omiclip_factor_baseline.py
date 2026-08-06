from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import joblib


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_NPZ = PROJECT_DIR / "outputs" / "omiclip_factor_baseline" / "omiclip_factor_dataset.npz"
META_CSV = PROJECT_DIR / "outputs" / "omiclip_factor_baseline" / "omiclip_factor_metadata.csv"

OUT_DIR = PROJECT_DIR / "outputs" / "omiclip_factor_baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_SAMPLES = [
    "INT1", "INT2", "INT3", "INT4", "INT5", "INT6",
    "INT7", "INT8", "INT9", "INT10", "INT11", "INT12",
    "INT13", "INT14", "INT15", "INT16", "INT17", "INT18",
]

VAL_SAMPLES = ["INT19", "INT20", "INT21"]
TEST_SAMPLES = ["INT22", "INT23", "INT24"]


def safe_corr(fn, y_true, y_pred):
    try:
        value, _ = fn(y_true, y_pred)
        return value
    except Exception:
        return np.nan


def evaluate_split(split_name, y_true, y_pred, factor_names):
    rows = []

    for i, factor in enumerate(factor_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        rows.append({
            "split": split_name,
            "factor": factor,
            "mse": mean_squared_error(yt, yp),
            "rmse": np.sqrt(mean_squared_error(yt, yp)),
            "r2": r2_score(yt, yp),
            "pearson": safe_corr(pearsonr, yt, yp),
            "spearman": safe_corr(spearmanr, yt, yp),
        })

    return rows


def main():
    print("Loading joined OmiCLIP ccRCC dataset...")

    data = np.load(DATA_NPZ, allow_pickle=True)
    meta = pd.read_csv(META_CSV)

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    factor_names = data["factor_names"].astype(str)

    print("X:", X.shape)
    print("y:", y.shape)
    print("metadata:", meta.shape)
    print("factors:", factor_names)

    sample_ids = meta["sample_id"].astype(str).values

    train_mask = np.isin(sample_ids, TRAIN_SAMPLES)
    val_mask = np.isin(sample_ids, VAL_SAMPLES)
    test_mask = np.isin(sample_ids, TEST_SAMPLES)

    print("\nSplit sizes:")
    print("Train:", train_mask.sum(), TRAIN_SAMPLES)
    print("Val:", val_mask.sum(), VAL_SAMPLES)
    print("Test:", test_mask.sum(), TEST_SAMPLES)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print("\nStandardising X and y using training split only...")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s = y_scaler.transform(y_val)
    y_test_s = y_scaler.transform(y_test)

    print("Training Ridge regression baseline...")

    model = Ridge(alpha=100.0)
    model.fit(X_train_s, y_train_s)

    print("Predicting...")

    y_train_pred_s = model.predict(X_train_s)
    y_val_pred_s = model.predict(X_val_s)
    y_test_pred_s = model.predict(X_test_s)

    y_train_pred = y_scaler.inverse_transform(y_train_pred_s)
    y_val_pred = y_scaler.inverse_transform(y_val_pred_s)
    y_test_pred = y_scaler.inverse_transform(y_test_pred_s)

    rows = []
    rows += evaluate_split("train", y_train, y_train_pred, factor_names)
    rows += evaluate_split("val", y_val, y_val_pred, factor_names)
    rows += evaluate_split("test", y_test, y_test_pred, factor_names)

    results = pd.DataFrame(rows)
    results_path = OUT_DIR / "ridge_baseline_metrics.csv"
    results.to_csv(results_path, index=False)

    print("\nResults:")
    print(results)

    joblib.dump(
        {
            "model": model,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "train_samples": TRAIN_SAMPLES,
            "val_samples": VAL_SAMPLES,
            "test_samples": TEST_SAMPLES,
            "factor_names": factor_names,
        },
        OUT_DIR / "ridge_baseline_model.joblib",
    )

    pred_rows = []

    for split_name, mask, y_true_split, y_pred_split in [
        ("train", train_mask, y_train, y_train_pred),
        ("val", val_mask, y_val, y_val_pred),
        ("test", test_mask, y_test, y_test_pred),
    ]:
        split_meta = meta[mask].copy().reset_index(drop=True)
        split_meta["split"] = split_name

        for i, factor in enumerate(factor_names):
            split_meta[f"true_{factor}"] = y_true_split[:, i]
            split_meta[f"pred_{factor}"] = y_pred_split[:, i]

        pred_rows.append(split_meta)

    pred_df = pd.concat(pred_rows, ignore_index=True)
    pred_path = OUT_DIR / "ridge_baseline_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print("\nSaved metrics:", results_path)
    print("Saved predictions:", pred_path)
    print("Saved model:", OUT_DIR / "ridge_baseline_model.joblib")


if __name__ == "__main__":
    main()