"""Shared pipeline helpers: load data, build features, run RV models."""

from src.data_loader import load_all_data
from src.feature_engineering import build_feature_table
from src.rv_models import run_all_rv_models


def load_data_and_augment(nrows=None, train_window=252):
    """
    Load data, build features, run RV models, merge forecasts.
    Returns (data dict, augmented feature DataFrame).
    """
    data = load_all_data(nrows=nrows)
    features = build_feature_table(
        data["options"], data["spx"], rf_series=data["rf"]
    )
    forecasts = run_all_rv_models(features, train_window=train_window)
    augmented = features.merge(forecasts, on="date", how="left")
    return data, augmented
