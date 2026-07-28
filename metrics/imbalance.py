"""Imbalanced regression metrics and relevance (phi)-based utilities."""

from __future__ import annotations

from typing import Callable, Literal, Tuple

import numpy as np
from sklearn.metrics import r2_score


Tail = Literal["lower", "upper", "both"]


def _validate_targets(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float] | None = None,
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Validate target arrays for imbalance metrics.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float] | None, default=None
        Predicted values. If provided, shape must match `y_true`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        Flattened `y_true` and optional flattened `y_pred`.

    Raises
    ------
    ValueError
        If arrays are empty, shape-mismatched, or non-finite.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    if yt.size == 0:
        raise ValueError("`y_true` must be non-empty.")
    if not np.all(np.isfinite(yt)):
        raise ValueError("`y_true` must contain only finite values.")

    if y_pred is None:
        return yt, None

    yp = np.asarray(y_pred, dtype=float).ravel()
    if yp.size == 0:
        raise ValueError("`y_pred` must be non-empty.")
    if yt.shape != yp.shape:
        raise ValueError("`y_true` and `y_pred` must have the same shape.")
    if not np.all(np.isfinite(yp)):
        raise ValueError("`y_pred` must contain only finite values.")

    return yt, yp


def _validate_threshold(threshold: float) -> float:
    """
    Validate relevance threshold.

    Parameters
    ----------
    threshold : float
        Relevance threshold in [0, 1].

    Returns
    -------
    float
        Validated threshold.

    Raises
    ------
    ValueError
        If threshold is outside [0, 1].
    """
    thr = float(threshold)
    if not 0.0 <= thr <= 1.0:
        raise ValueError("`threshold` must be between 0 and 1.")
    return thr


def _validate_tail(tail: Tail) -> Tail:
    """
    Validate tail selection.

    Parameters
    ----------
    tail : {"lower", "upper", "both"}
        Tail selection strategy.

    Returns
    -------
    {"lower", "upper", "both"}
        Validated tail string.

    Raises
    ------
    ValueError
        If tail value is invalid.
    """
    if tail not in {"lower", "upper", "both"}:
        raise ValueError("`tail` must be one of {'lower', 'upper', 'both'}.")
    return tail


def _automatic_phi(y: np.ndarray, tail: Tail = "both") -> np.ndarray:
    """
    Compute automatic relevance scores (phi) from empirical tails.

    Parameters
    ----------
    y : np.ndarray
        Target values.
    tail : {"lower", "upper", "both"}, default="both"
        Tail(s) considered rare and therefore highly relevant.

    Returns
    -------
    np.ndarray
        Relevance scores in [0, 1], where higher values indicate rarer targets.
    """
    tail = _validate_tail(tail)
    q1, q3 = np.quantile(y, [0.25, 0.75])
    iqr = q3 - q1
    if iqr <= 0:
        return np.ones_like(y, dtype=float)

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    phi = np.zeros_like(y, dtype=float)

    if tail in {"lower", "both"}:
        lower_span = max(q1 - lower_fence, np.finfo(float).eps)
        lower_mask = y < q1
        lower_rel = (q1 - y[lower_mask]) / lower_span
        phi[lower_mask] = np.maximum(phi[lower_mask], np.clip(lower_rel, 0.0, 1.0))

    if tail in {"upper", "both"}:
        upper_span = max(upper_fence - q3, np.finfo(float).eps)
        upper_mask = y > q3
        upper_rel = (y[upper_mask] - q3) / upper_span
        phi[upper_mask] = np.maximum(phi[upper_mask], np.clip(upper_rel, 0.0, 1.0))

    return np.clip(phi, 0.0, 1.0)


def _resolve_phi(
    y: np.ndarray,
    phi_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    tail: Tail = "both",
) -> np.ndarray:
    """
    Resolve relevance scores from custom or automatic phi function.

    Parameters
    ----------
    y : np.ndarray
        Target values.
    phi_fn : callable | None, default=None
        Custom callable mapping values to relevance scores.
        If None, automatic phi is used.
    tail : {"lower", "upper", "both"}, default="both"
        Tail selection used by automatic phi.

    Returns
    -------
    np.ndarray
        Relevance scores in [0, 1].

    Raises
    ------
    ValueError
        If custom phi output has invalid shape or values.
    """
    if phi_fn is None:
        phi = _automatic_phi(y, tail=tail)
    else:
        phi = np.asarray(phi_fn(y), dtype=float).ravel()
        if phi.shape != y.shape:
            raise ValueError("Custom `phi_fn` must return an array matching `y` shape.")
        if not np.all(np.isfinite(phi)):
            raise ValueError("Custom `phi_fn` output must be finite.")
        phi = np.clip(phi, 0.0, 1.0)
    return phi


def _tail_mask(y: np.ndarray, tail: Tail, quantile: float = 0.1) -> np.ndarray:
    """
    Build a boolean mask for selected tail regions.

    Parameters
    ----------
    y : np.ndarray
        Target values.
    tail : {"lower", "upper", "both"}
        Tail(s) to include.
    quantile : float, default=0.1
        Fraction defining each tail boundary.

    Returns
    -------
    np.ndarray
        Boolean mask identifying tail samples.

    Raises
    ------
    ValueError
        If quantile is outside (0, 0.5).
    """
    _validate_tail(tail)
    q = float(quantile)
    if not 0.0 < q < 0.5:
        raise ValueError("`quantile` must be in (0, 0.5).")

    q_low = np.quantile(y, q)
    q_high = np.quantile(y, 1.0 - q)

    if tail == "lower":
        return y <= q_low
    if tail == "upper":
        return y >= q_high
    return (y <= q_low) | (y >= q_high)


def precision_phi(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    threshold: float = 0.8,
    phi_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    tail: Tail = "both",
) -> float:
    """
    Compute phi-based precision for rare-region detection.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.
    threshold : float, default=0.8
        Relevance threshold for positive (rare) assignment.
    phi_fn : callable | None, default=None
        Custom phi function. If None, automatic phi is used.
    tail : {"lower", "upper", "both"}, default="both"
        Tail selection for automatic phi.

    Returns
    -------
    float
        Precision of predicted rare targets against true rare targets.
    """
    yt, yp = _validate_targets(y_true, y_pred)
    thr = _validate_threshold(threshold)

    phi_true = _resolve_phi(yt, phi_fn=phi_fn, tail=tail)
    phi_pred = _resolve_phi(yp, phi_fn=phi_fn, tail=tail)

    pred_pos = phi_pred >= thr
    true_pos = phi_true >= thr

    predicted_positives = np.sum(pred_pos)
    if predicted_positives == 0:
        return 0.0

    tp = np.sum(pred_pos & true_pos)
    return float(tp / predicted_positives)


def recall_phi(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    threshold: float = 0.8,
    phi_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    tail: Tail = "both",
) -> float:
    """
    Compute phi-based recall for rare-region detection.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.
    threshold : float, default=0.8
        Relevance threshold for positive (rare) assignment.
    phi_fn : callable | None, default=None
        Custom phi function. If None, automatic phi is used.
    tail : {"lower", "upper", "both"}, default="both"
        Tail selection for automatic phi.

    Returns
    -------
    float
        Recall of predicted rare targets against true rare targets.
    """
    yt, yp = _validate_targets(y_true, y_pred)
    thr = _validate_threshold(threshold)

    phi_true = _resolve_phi(yt, phi_fn=phi_fn, tail=tail)
    phi_pred = _resolve_phi(yp, phi_fn=phi_fn, tail=tail)

    pred_pos = phi_pred >= thr
    true_pos = phi_true >= thr

    actual_positives = np.sum(true_pos)
    if actual_positives == 0:
        return 0.0

    tp = np.sum(pred_pos & true_pos)
    return float(tp / actual_positives)


def f1_phi(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    threshold: float = 0.8,
    phi_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    tail: Tail = "both",
) -> float:
    """
    Compute phi-based F1 score for rare-region detection.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.
    threshold : float, default=0.8
        Relevance threshold for positive (rare) assignment.
    phi_fn : callable | None, default=None
        Custom phi function. If None, automatic phi is used.
    tail : {"lower", "upper", "both"}, default="both"
        Tail selection for automatic phi.

    Returns
    -------
    float
        F1 score from phi-based precision and recall.
    """
    p = precision_phi(y_true, y_pred, threshold=threshold, phi_fn=phi_fn, tail=tail)
    r = recall_phi(y_true, y_pred, threshold=threshold, phi_fn=phi_fn, tail=tail)
    if p + r == 0.0:
        return 0.0
    return float(2.0 * p * r / (p + r))


def weighted_mae(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    phi_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    tail: Tail = "both",
) -> float:
    """
    Compute relevance-weighted Mean Absolute Error.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.
    phi_fn : callable | None, default=None
        Custom relevance function. If None, automatic phi is used.
    tail : {"lower", "upper", "both"}, default="both"
        Tail selection for automatic phi.

    Returns
    -------
    float
        Weighted MAE value.
    """
    yt, yp = _validate_targets(y_true, y_pred)
    phi = _resolve_phi(yt, phi_fn=phi_fn, tail=tail)
    weights = 1.0 + phi
    errors = np.abs(yp - yt)
    return float(np.average(errors, weights=weights))


def weighted_rmse(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    phi_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    tail: Tail = "both",
) -> float:
    """
    Compute relevance-weighted Root Mean Squared Error.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.
    phi_fn : callable | None, default=None
        Custom relevance function. If None, automatic phi is used.
    tail : {"lower", "upper", "both"}, default="both"
        Tail selection for automatic phi.

    Returns
    -------
    float
        Weighted RMSE value.
    """
    yt, yp = _validate_targets(y_true, y_pred)
    phi = _resolve_phi(yt, phi_fn=phi_fn, tail=tail)
    weights = 1.0 + phi
    sq_errors = (yp - yt) ** 2
    return float(np.sqrt(np.average(sq_errors, weights=weights)))


def tail_mae(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    tail: Tail = "both",
    quantile: float = 0.1,
) -> float:
    """
    Compute MAE restricted to selected target tails.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.
    tail : {"lower", "upper", "both"}, default="both"
        Tail(s) to evaluate.
    quantile : float, default=0.1
        Tail quantile size.

    Returns
    -------
    float
        Tail-restricted MAE.

    Raises
    ------
    ValueError
        If no samples fall in selected tail region.
    """
    yt, yp = _validate_targets(y_true, y_pred)
    mask = _tail_mask(yt, tail=tail, quantile=quantile)
    if not np.any(mask):
        raise ValueError("No samples found in the selected tail region.")
    return float(np.mean(np.abs(yp[mask] - yt[mask])))


def tail_rmse(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    tail: Tail = "both",
    quantile: float = 0.1,
) -> float:
    """
    Compute RMSE restricted to selected target tails.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.
    tail : {"lower", "upper", "both"}, default="both"
        Tail(s) to evaluate.
    quantile : float, default=0.1
        Tail quantile size.

    Returns
    -------
    float
        Tail-restricted RMSE.

    Raises
    ------
    ValueError
        If no samples fall in selected tail region.
    """
    yt, yp = _validate_targets(y_true, y_pred)
    mask = _tail_mask(yt, tail=tail, quantile=quantile)
    if not np.any(mask):
        raise ValueError("No samples found in the selected tail region.")
    return float(np.sqrt(np.mean((yp[mask] - yt[mask]) ** 2)))


def tail_r2(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    tail: Tail = "both",
    quantile: float = 0.1,
) -> float:
    """
    Compute R² restricted to selected target tails.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.
    tail : {"lower", "upper", "both"}, default="both"
        Tail(s) to evaluate.
    quantile : float, default=0.1
        Tail quantile size.

    Returns
    -------
    float
        Tail-restricted R² score.

    Raises
    ------
    ValueError
        If fewer than 2 samples fall in selected tail region.
    """
    yt, yp = _validate_targets(y_true, y_pred)
    mask = _tail_mask(yt, tail=tail, quantile=quantile)
    if np.sum(mask) < 2:
        raise ValueError("At least 2 samples are required in tail region for R².")
    return float(r2_score(yt[mask], yp[mask]))


def tail_coverage(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    threshold: float = 0.8,
    phi_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    tail: Tail = "both",
) -> float:
    """
    Compute coverage of true rare targets by predicted rare targets.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.
    threshold : float, default=0.8
        Relevance threshold for rarity.
    phi_fn : callable | None, default=None
        Custom relevance function. If None, automatic phi is used.
    tail : {"lower", "upper", "both"}, default="both"
        Tail selection for automatic phi.

    Returns
    -------
    float
        Fraction of truly rare samples also predicted as rare.
    """
    return recall_phi(y_true, y_pred, threshold=threshold, phi_fn=phi_fn, tail=tail)


def rare_region_error(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    threshold: float = 0.8,
    phi_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    tail: Tail = "both",
) -> float:
    """
    Compute MAE over rare region samples only.

    Parameters
    ----------
    y_true : np.ndarray | list[float]
        Ground-truth target values.
    y_pred : np.ndarray | list[float]
        Predicted target values.
    threshold : float, default=0.8
        Relevance threshold defining rare samples.
    phi_fn : callable | None, default=None
        Custom relevance function. If None, automatic phi is used.
    tail : {"lower", "upper", "both"}, default="both"
        Tail selection for automatic phi.

    Returns
    -------
    float
        MAE on rare region samples.

    Raises
    ------
    ValueError
        If no rare samples are found under the chosen threshold.
    """
    yt, yp = _validate_targets(y_true, y_pred)
    thr = _validate_threshold(threshold)
    phi = _resolve_phi(yt, phi_fn=phi_fn, tail=tail)
    mask = phi >= thr
    if not np.any(mask):
        raise ValueError("No rare samples found for the provided threshold.")
    return float(np.mean(np.abs(yp[mask] - yt[mask])))
