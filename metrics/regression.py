"""Regression metrics for continuous targets."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import (
    max_error as sk_max_error,
    mean_absolute_error as sk_mean_absolute_error,
    mean_absolute_percentage_error as sk_mean_absolute_percentage_error,
    mean_squared_error as sk_mean_squared_error,
    median_absolute_error as sk_median_absolute_error,
    r2_score as sk_r2_score,
)


def _validate_regression_inputs(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and standardize regression metric inputs.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Flattened arrays of `y_true` and `y_pred`.

    Raises
    ------
    ValueError
        If shapes are inconsistent, arrays are empty, or contain non-finite values.
    """
    y_true_arr = np.asarray(y_true, dtype=float).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=float).ravel()

    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        raise ValueError("`y_true` and `y_pred` must be non-empty.")
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "`y_true` and `y_pred` must have the same shape after flattening."
        )
    if not np.all(np.isfinite(y_true_arr)) or not np.all(np.isfinite(y_pred_arr)):
        raise ValueError("`y_true` and `y_pred` must contain only finite values.")

    return y_true_arr, y_pred_arr


def mae(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """
    Compute Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.

    Returns
    -------
    float
        Mean absolute error.
    """
    yt, yp = _validate_regression_inputs(y_true, y_pred)
    return float(sk_mean_absolute_error(yt, yp))


def mse(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """
    Compute Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.

    Returns
    -------
    float
        Mean squared error.
    """
    yt, yp = _validate_regression_inputs(y_true, y_pred)
    return float(sk_mean_squared_error(yt, yp))


def rmse(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """
    Compute Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.

    Returns
    -------
    float
        Root mean squared error.
    """
    yt, yp = _validate_regression_inputs(y_true, y_pred)
    return float(np.sqrt(sk_mean_squared_error(yt, yp)))


def r2(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """
    Compute coefficient of determination (R² score).

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.

    Returns
    -------
    float
        R² score.
    """
    yt, yp = _validate_regression_inputs(y_true, y_pred)
    return float(sk_r2_score(yt, yp))


def median_absolute_error(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> float:
    """
    Compute Median Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.

    Returns
    -------
    float
        Median absolute error.
    """
    yt, yp = _validate_regression_inputs(y_true, y_pred)
    return float(sk_median_absolute_error(yt, yp))


def mape(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """
    Compute Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.

    Returns
    -------
    float
        Mean absolute percentage error as a ratio (not percentage points).
    """
    yt, yp = _validate_regression_inputs(y_true, y_pred)
    return float(sk_mean_absolute_percentage_error(yt, yp))


def smape(y_true: np.ndarray | list[float], y_pred: np.ndarray | list[float]) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error (sMAPE).

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.

    Returns
    -------
    float
        Symmetric mean absolute percentage error as a ratio in [0, 2].
    """
    yt, yp = _validate_regression_inputs(y_true, y_pred)
    denominator = np.abs(yt) + np.abs(yp)
    diff = np.abs(yp - yt)

    smape_values = np.zeros_like(diff, dtype=float)
    non_zero = denominator > 0.0
    smape_values[non_zero] = 2.0 * diff[non_zero] / denominator[non_zero]
    return float(np.mean(smape_values))


def max_error(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> float:
    """
    Compute maximum residual error.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.

    Returns
    -------
    float
        Maximum residual error.
    """
    yt, yp = _validate_regression_inputs(y_true, y_pred)
    return float(sk_max_error(yt, yp))
