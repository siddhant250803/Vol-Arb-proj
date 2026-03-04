#!/usr/bin/env python3
"""
Balanced logistic bucket model for Friday-expiry return distribution.

Workflow
--------
1. Use lagged returns, rolling return features, and calendar features as predictors.
2. Build the cumulative log return from each date through that week's Friday.
3. Standardize that target by sqrt(days_to_friday) so weekdays are pooled on a
   comparable scale.
4. Create quantile buckets from the standardized target.
5. Fit one balanced logistic regression per bucket.
6. Convert predicted bucket probabilities into expected realised volatility.

This is intended to work directly off the project's feature table:

    python logistic.py

By default it reads `output/data/feature_table.csv` and writes
`output/data/logistic_expected_rvol.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_OUTPUT_DIR, TRADING_DAYS_PER_YEAR


DEFAULT_LAGS = (1, 2, 3, 5, 10, 22)


def build_return_features(df: pd.DataFrame, return_col: str = "log_return") -> pd.DataFrame:
    """Create lagged and rolling features from log returns."""
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["weekday"] = out["date"].dt.weekday
    out["days_to_friday"] = (4 - out["weekday"]).clip(lower=0) + 1

    for lag in DEFAULT_LAGS:
        out[f"{return_col}_lag_{lag}"] = out[return_col].shift(lag)
        out[f"abs_{return_col}_lag_{lag}"] = out[return_col].abs().shift(lag)

    out["rolling_mean_5"] = out[return_col].shift(1).rolling(5, min_periods=5).mean()
    out["rolling_mean_22"] = out[return_col].shift(1).rolling(22, min_periods=22).mean()
    out["rolling_std_5"] = out[return_col].shift(1).rolling(5, min_periods=5).std()
    out["rolling_std_22"] = out[return_col].shift(1).rolling(22, min_periods=22).std()
    out["rolling_abs_mean_5"] = out[return_col].abs().shift(1).rolling(5, min_periods=5).mean()
    out["rolling_abs_mean_22"] = out[return_col].abs().shift(1).rolling(22, min_periods=22).mean()

    return out


def add_weekly_friday_target(
    df: pd.DataFrame,
    return_col: str = "log_return",
) -> pd.DataFrame:
    """
    Add cumulative return through Friday and its horizon-standardized version.

    `days_to_friday` is inclusive:
      Monday -> 5, Tuesday -> 4, ..., Friday -> 1
    """
    out = df.copy()
    target_forward_return = np.full(len(out), np.nan)

    returns = out[return_col].to_numpy(dtype=float)
    horizons = out["days_to_friday"].to_numpy(dtype=int)

    for i, horizon in enumerate(horizons):
        end = i + horizon
        if end <= len(out):
            target_forward_return[i] = returns[i:end].sum()

    out["target_forward_return"] = target_forward_return
    out["target_horizon_scaled_return"] = (
        out["target_forward_return"] / np.sqrt(out["days_to_friday"])
    )
    return out


def add_quantile_targets(
    df: pd.DataFrame,
    target_col: str = "target_horizon_scaled_return",
    n_quantiles: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bucket standardized Friday-expiry returns into quantile categories.

    Returns
    -------
    modeled_df : pd.DataFrame
        Original data plus `target_value`, `quantile_category`,
        and one dummy column per quantile.
    quantile_info : pd.DataFrame
        Interval edges, bucket median, empirical share, and width per quantile.
    """
    out = df.copy()
    out["target_value"] = out[target_col]

    valid_target = out["target_value"].dropna()
    bucketed, edges = pd.qcut(
        valid_target,
        q=n_quantiles,
        labels=False,
        retbins=True,
        duplicates="drop",
    )

    out.loc[valid_target.index, "quantile_category"] = bucketed.astype(int)
    out["quantile_category"] = out["quantile_category"].astype("Int64")

    realized_quantiles = len(edges) - 1
    shares = out["quantile_category"].value_counts(normalize=True).sort_index()
    bucket_medians = out.groupby("quantile_category")["target_value"].median()

    quantile_rows = []
    for q in range(realized_quantiles):
        left_edge = float(edges[q])
        right_edge = float(edges[q + 1])
        midpoint = 0.5 * (left_edge + right_edge)
        bucket_median = float(bucket_medians.get(q, midpoint))
        width = right_edge - left_edge
        share = float(shares.get(q, 0.0))
        quantile_rows.append(
            {
                "quantile_category": q,
                "left_edge": left_edge,
                "right_edge": right_edge,
                "midpoint": midpoint,
                "bucket_median": bucket_median,
                "quantile_share": share,
                "quantile_width": width,
            }
        )

    quantile_info = pd.DataFrame(quantile_rows)

    for q in quantile_info["quantile_category"]:
        out[f"quantile_{q}_dummy"] = (
            out["quantile_category"].eq(q).fillna(False).astype(int)
        )

    return out, quantile_info


def fit_balanced_bucket_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    quantile_info: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, object]]:
    """Fit one balanced logistic regression per quantile bucket."""
    out = df.copy()
    model_mask = out[feature_cols + ["quantile_category"]].notna().all(axis=1)
    train = out.loc[model_mask].copy()

    X = train[feature_cols]
    models: dict[int, object] = {}
    raw_prob_cols: list[str] = []

    for q in quantile_info["quantile_category"]:
        y = (train["quantile_category"] == q).astype(int)
        prob_col = f"quantile_{q}_raw_prob"
        if y.nunique() < 2:
            train[prob_col] = 0.0
            raw_prob_cols.append(prob_col)
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                solver="lbfgs",
                random_state=42,
            ),
        )
        model.fit(X, y)
        models[int(q)] = model

        train[prob_col] = model.predict_proba(X)[:, 1]
        raw_prob_cols.append(prob_col)

    row_prob_sum = train[raw_prob_cols].sum(axis=1).replace(0, np.nan)
    for q in quantile_info["quantile_category"]:
        raw_col = f"quantile_{q}_raw_prob"
        norm_col = f"quantile_{q}_prob"
        train[norm_col] = train[raw_col] / row_prob_sum

    out = out.merge(
        train[["date"] + [f"quantile_{q}_prob" for q in quantile_info["quantile_category"]]],
        on="date",
        how="left",
    )
    return out, models


def compute_expected_realised_volatility(
    df: pd.DataFrame,
    quantile_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert predicted quantile probabilities into expected realised volatility.

    Each quantile bucket is represented by its median horizon-standardized
    return z = R / sqrt(h), where h is days_to_friday. Then:

        E[R^2 | h] = h * E[z^2]
        E[RV_annual | h] = (252 / h) * E[R^2 | h] = 252 * E[z^2]

    so the annualized expected variance depends only on the predicted second
    moment of the standardized target.
    """
    out = df.copy()
    expected_scaled_return_sq_plain = np.zeros(len(out), dtype=float)
    expected_scaled_return_sq_weighted = np.zeros(len(out), dtype=float)
    prob_mass = np.zeros(len(out), dtype=float)

    for row in quantile_info.itertuples(index=False):
        prob_col = f"quantile_{row.quantile_category}_prob"
        if prob_col not in out.columns:
            continue

        prob = out[prob_col].fillna(0.0).to_numpy()
        median_return_sq = float(row.bucket_median) ** 2
        quantile_share = float(row.quantile_share)

        expected_scaled_return_sq_plain += prob * median_return_sq
        expected_scaled_return_sq_weighted += prob * median_return_sq * quantile_share
        prob_mass += prob

    out["expected_scaled_return_sq_plain"] = np.where(
        prob_mass > 0,
        expected_scaled_return_sq_plain,
        np.nan,
    )
    out["expected_scaled_return_sq"] = np.where(
        prob_mass > 0,
        expected_scaled_return_sq_weighted,
        np.nan,
    )
    out["expected_realised_variance_plain"] = TRADING_DAYS_PER_YEAR * out["expected_scaled_return_sq_plain"]
    out["expected_realised_variance"] = TRADING_DAYS_PER_YEAR * out["expected_scaled_return_sq"]
    out["expected_realised_volatility_plain"] = np.sqrt(out["expected_realised_variance_plain"].clip(lower=0))
    out["expected_realised_volatility"] = np.sqrt(out["expected_realised_variance"].clip(lower=0))
    return out


def run_logistic_expected_vol(
    feature_df: pd.DataFrame,
    return_col: str = "log_return",
    n_quantiles: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, object]]:
    """End-to-end helper."""
    modeled = build_return_features(feature_df, return_col=return_col)
    modeled = add_weekly_friday_target(modeled, return_col=return_col)
    modeled, quantile_info = add_quantile_targets(
        modeled,
        target_col="target_horizon_scaled_return",
        n_quantiles=n_quantiles,
    )

    feature_cols = [
        c
        for c in modeled.columns
        if c.startswith(f"{return_col}_lag_")
        or c.startswith(f"abs_{return_col}_lag_")
        or c.startswith("rolling_")
        or c in {"weekday", "days_to_friday"}
    ]

    modeled, models = fit_balanced_bucket_models(
        modeled,
        feature_cols=feature_cols,
        quantile_info=quantile_info,
    )
    modeled = compute_expected_realised_volatility(modeled, quantile_info)
    return modeled, quantile_info, models


def run_logistic_expected_vol_oos(
    feature_df: pd.DataFrame,
    return_col: str = "log_return",
    n_quantiles: int = 5,
    train_window: int = 252,
) -> pd.DataFrame:
    """
    Expanding-window out-of-sample logistic forecast.

    For each date t after `train_window`, fit the quantile classifiers on data
    strictly before t, then predict the forward-return bucket probabilities at t.
    """
    modeled = build_return_features(feature_df, return_col=return_col)
    modeled = add_weekly_friday_target(modeled, return_col=return_col)

    feature_cols = [
        c
        for c in modeled.columns
        if c.startswith(f"{return_col}_lag_")
        or c.startswith(f"abs_{return_col}_lag_")
        or c.startswith("rolling_")
        or c in {"weekday", "days_to_friday"}
    ]

    results = []
    n_rows = len(modeled)

    for t in range(train_window, n_rows):
        test_row = modeled.iloc[[t]].copy()
        if test_row[feature_cols].notna().all(axis=1).iloc[0] is False:
            continue

        train_df = modeled.iloc[:t].copy()
        train_df = train_df.dropna(subset=feature_cols + ["target_horizon_scaled_return"])
        if len(train_df) < train_window:
            continue

        train_labeled, quantile_info = add_quantile_targets(
            train_df,
            target_col="target_horizon_scaled_return",
            n_quantiles=n_quantiles,
        )
        train_labeled, models = fit_balanced_bucket_models(
            train_labeled,
            feature_cols=feature_cols,
            quantile_info=quantile_info,
        )

        X_test = test_row[feature_cols]
        pred_record = {
            "date": test_row["date"].iloc[0],
            "log_return": test_row[return_col].iloc[0],
            "weekday": test_row["weekday"].iloc[0],
            "days_to_friday": test_row["days_to_friday"].iloc[0],
            "target_forward_return": test_row["target_forward_return"].iloc[0],
            "target_horizon_scaled_return": test_row["target_horizon_scaled_return"].iloc[0],
            "fwd_rvol": test_row["fwd_rvol"].iloc[0] if "fwd_rvol" in test_row.columns else np.nan,
            "atm_iv_30d": test_row["atm_iv_30d"].iloc[0] if "atm_iv_30d" in test_row.columns else np.nan,
        }

        raw_probs = {}
        for q in quantile_info["quantile_category"]:
            model = models.get(int(q))
            if model is None:
                raw_probs[int(q)] = 0.0
            else:
                raw_probs[int(q)] = float(model.predict_proba(X_test)[:, 1][0])
        prob_sum = sum(raw_probs.values())
        if prob_sum <= 0:
            continue

        for q, raw_prob in raw_probs.items():
            pred_record[f"quantile_{q}_prob"] = raw_prob / prob_sum

        one_row = pd.DataFrame([pred_record])
        one_row = compute_expected_realised_volatility(one_row, quantile_info)
        pred_record.update(
            one_row[
                [
                    "expected_scaled_return_sq_plain",
                    "expected_scaled_return_sq",
                    "expected_realised_variance_plain",
                    "expected_realised_variance",
                    "expected_realised_volatility_plain",
                    "expected_realised_volatility",
                ]
            ].iloc[0].to_dict()
        )
        results.append(pred_record)

    out = pd.DataFrame(results).sort_values("date").reset_index(drop=True)
    print(f"[logistic] OOS forecasts: {len(out)} dates (train_window={train_window}).")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balanced logistic bucket model for Friday-expiry return distribution.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_OUTPUT_DIR / "feature_table.csv",
        help="Path to the feature table CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_OUTPUT_DIR / "logistic_expected_rvol.csv",
        help="Path for the modeled output CSV.",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=5,
        help="Number of quantile buckets for standardized Friday-expiry returns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_df = pd.read_csv(args.input, parse_dates=["date"])

    modeled, quantile_info, _ = run_logistic_expected_vol(
        feature_df,
        return_col="log_return",
        n_quantiles=args.quantiles,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    modeled.to_csv(args.output, index=False)

    quantile_output = args.output.with_name(f"{args.output.stem}_quantiles.csv")
    quantile_info.to_csv(quantile_output, index=False)

    preview_cols = [
        "date",
        "log_return",
        "weekday",
        "days_to_friday",
        "target_forward_return",
        "target_horizon_scaled_return",
        "target_value",
        "quantile_category",
        "expected_scaled_return_sq",
        "expected_realised_volatility",
        "expected_realised_variance",
    ]
    print(modeled[preview_cols].dropna().head().to_string(index=False))
    print(f"\nSaved modeled output to {args.output}")
    print(f"Saved quantile metadata to {quantile_output}")


if __name__ == "__main__":
    main()
