from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]

VIRCHOW_METRICS = (
    PROJECT_DIR
    / "outputs"
    / "hest_factor_baseline"
    / "ridge_baseline_metrics.csv"
)

OMICLIP_METRICS = (
    PROJECT_DIR
    / "outputs"
    / "omiclip_factor_baseline"
    / "ridge_baseline_metrics.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "outputs"
    / "virchow_vs_omiclip_test_metrics.csv"
)


def main():
    virchow = pd.read_csv(VIRCHOW_METRICS)
    omiclip = pd.read_csv(OMICLIP_METRICS)

    # Compare only the held-out test samples INT22–INT24.
    virchow = virchow[virchow["split"] == "test"].copy()
    omiclip = omiclip[omiclip["split"] == "test"].copy()

    virchow = virchow.rename(
        columns={
            "mse": "virchow_mse",
            "rmse": "virchow_rmse",
            "r2": "virchow_r2",
            "pearson": "virchow_pearson",
            "spearman": "virchow_spearman",
        }
    )

    omiclip = omiclip.rename(
        columns={
            "mse": "omiclip_mse",
            "rmse": "omiclip_rmse",
            "r2": "omiclip_r2",
            "pearson": "omiclip_pearson",
            "spearman": "omiclip_spearman",
        }
    )

    comparison = virchow.merge(
        omiclip,
        on=["split", "factor"],
        how="inner",
        validate="one_to_one",
    )

    # Positive correlation/R² differences favour OmiCLIP.
    comparison["pearson_difference"] = (
        comparison["omiclip_pearson"]
        - comparison["virchow_pearson"]
    )

    comparison["spearman_difference"] = (
        comparison["omiclip_spearman"]
        - comparison["virchow_spearman"]
    )

    comparison["r2_difference"] = (
        comparison["omiclip_r2"]
        - comparison["virchow_r2"]
    )

    # Negative RMSE difference favours OmiCLIP.
    comparison["rmse_difference"] = (
        comparison["omiclip_rmse"]
        - comparison["virchow_rmse"]
    )

    comparison["better_pearson"] = comparison.apply(
        lambda row: (
            "OmiCLIP"
            if row["omiclip_pearson"] > row["virchow_pearson"]
            else "Virchow2"
        ),
        axis=1,
    )

    columns = [
        "factor",
        "virchow_pearson",
        "omiclip_pearson",
        "pearson_difference",
        "virchow_spearman",
        "omiclip_spearman",
        "spearman_difference",
        "virchow_r2",
        "omiclip_r2",
        "r2_difference",
        "virchow_rmse",
        "omiclip_rmse",
        "rmse_difference",
        "better_pearson",
    ]

    comparison = comparison[columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False)

    print("\nVirchow2 versus pretrained OmiCLIP")
    print("Held-out test samples: INT22–INT24\n")
    print(comparison.to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()