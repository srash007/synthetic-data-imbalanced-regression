"""High-level evaluation API for regression and imbalance metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from metrics import imbalance, regression


def _validate_evaluation_inputs(
    y_true: np.ndarray | list[float],
    predictions: dict[str, np.ndarray | list[float]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Validate inputs for model evaluation.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    predictions : dict[str, np.ndarray | list[float]]
        Mapping from model names to predicted values.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        Validated `y_true` and standardized prediction arrays.

    Raises
    ------
    ValueError
        If inputs are empty, malformed, non-finite, or shape-mismatched.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    if yt.size == 0:
        raise ValueError("`y_true` must be non-empty.")
    if not np.all(np.isfinite(yt)):
        raise ValueError("`y_true` must contain only finite values.")

    if not predictions:
        raise ValueError("`predictions` must contain at least one model prediction.")

    pred_arrays: dict[str, np.ndarray] = {}
    for model_name, y_pred in predictions.items():
        yp = np.asarray(y_pred, dtype=float).ravel()
        if yp.shape != yt.shape:
            raise ValueError(
                f"Prediction shape mismatch for model '{model_name}': "
                f"expected {yt.shape}, got {yp.shape}."
            )
        if not np.all(np.isfinite(yp)):
            raise ValueError(
                f"Prediction values for model '{model_name}' must be finite."
            )
        pred_arrays[model_name] = yp

    return yt, pred_arrays


def _default_metric_registry() -> dict[str, Callable[..., float]]:
    """
    Build default metric registry from regression and imbalance modules.

    Returns
    -------
    dict[str, Callable[..., float]]
        Metric name to function mapping for all default metrics.
    """
    reg_metrics: dict[str, Callable[..., float]] = {
        "mae": regression.mae,
        "mse": regression.mse,
        "rmse": regression.rmse,
        "r2": regression.r2,
        "median_absolute_error": regression.median_absolute_error,
        "mape": regression.mape,
        "smape": regression.smape,
        "max_error": regression.max_error,
    }

    imb_metrics: dict[str, Callable[..., float]] = {
        "precision_phi": imbalance.precision_phi,
        "recall_phi": imbalance.recall_phi,
        "f1_phi": imbalance.f1_phi,
        "weighted_mae": imbalance.weighted_mae,
        "weighted_rmse": imbalance.weighted_rmse,
        "tail_mae": imbalance.tail_mae,
        "tail_rmse": imbalance.tail_rmse,
        "tail_r2": imbalance.tail_r2,
        "tail_coverage": imbalance.tail_coverage,
        "rare_region_error": imbalance.rare_region_error,
    }

    return {**reg_metrics, **imb_metrics}


def evaluate_models(
    y_true: np.ndarray | list[float],
    predictions: dict[str, np.ndarray | list[float]],
    metrics: dict[str, Callable[..., float]] | list[str] | None = None,
    imbalance_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Evaluate multiple models using regression and imbalance metrics.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    predictions : dict[str, np.ndarray | list[float]]
        Mapping from model names to predicted values.
    metrics : dict[str, Callable[..., float]] | list[str] | None, default=None
        Metrics to compute.
        - If None, all available metrics from `regression` and `imbalance` are used.
        - If list[str], names must exist in default registry.
        - If dict, keys are output metric names and values are callables.
    imbalance_kwargs : dict[str, Any] | None, default=None
        Optional keyword arguments passed only to imbalance metrics.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by model names with one column per metric.

    Raises
    ------
    ValueError
        If requested metric names are unknown.
    """
    yt, preds = _validate_evaluation_inputs(y_true, predictions)
    registry = _default_metric_registry()
    imbalance_kwargs = imbalance_kwargs or {}

    if metrics is None:
        selected_metrics = registry
    elif isinstance(metrics, list):
        unknown = [m for m in metrics if m not in registry]
        if unknown:
            raise ValueError(f"Unknown metric names: {unknown}")
        selected_metrics = {name: registry[name] for name in metrics}
    else:
        selected_metrics = metrics

    imbalance_metric_names = {
        "precision_phi",
        "recall_phi",
        "f1_phi",
        "weighted_mae",
        "weighted_rmse",
        "tail_mae",
        "tail_rmse",
        "tail_r2",
        "tail_coverage",
        "rare_region_error",
    }

    rows: list[dict[str, float | str]] = []

    for model_name, y_pred in preds.items():
        row: dict[str, float | str] = {"model": model_name}
        for metric_name, metric_fn in selected_metrics.items():
            if metric_name in imbalance_metric_names:
                value = metric_fn(yt, y_pred, **imbalance_kwargs)
            else:
                value = metric_fn(yt, y_pred)
            row[metric_name] = float(value)
        rows.append(row)

    result = pd.DataFrame(rows).set_index("model")
    return result
