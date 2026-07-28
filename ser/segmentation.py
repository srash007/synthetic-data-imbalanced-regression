"""
Segmentation strategies for the Segmentation Expert-Mixture
Regularization (SER) framework.

This module implements all segmentation algorithms proposed
in the SER paper.

Available methods
-----------------
A : MAD-based segmentation
B : Quantile-based segmentation
C : Iterative trimming
D : Bias-Variance optimization

Author
------
Sarah Elyane Rashiwa
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .utils import mad


# =============================================================================
# Helper Functions
# =============================================================================

def tail_mean_check(
    values: np.ndarray,
    mask_tail: np.ndarray,
    mask_center: np.ndarray,
    alpha: float = 0.15,
) -> bool:
    """
    Determine whether the tail has a significantly different
    mean from the center region.

    Parameters
    ----------
    values : np.ndarray
        Target values.

    mask_tail : np.ndarray
        Boolean mask identifying the tail region.

    mask_center : np.ndarray
        Boolean mask identifying the center region.

    alpha : float
        Sensitivity coefficient.

    Returns
    -------
    bool
        True if the tail should be preserved.
    """

    eps = 1e-12

    tail_mean = values[mask_tail].mean() if mask_tail.any() else 0.0
    center_mean = values[mask_center].mean() if mask_center.any() else 0.0

    return abs(tail_mean - center_mean) > alpha * (values.std() + eps)


# =============================================================================
# Basic Target Partition
# =============================================================================

def split_by_target_thresholds(
    y,
    low_thr: float,
    up_thr: float,
):
    """
    Split the target into Lower, Center and Upper regions.

    Returns
    -------
    tuple
        groups,
        idxL,
        idxC,
        idxU
    """

    y = np.asarray(y)

    mask_lower = y < low_thr
    mask_center = (y >= low_thr) & (y <= up_thr)
    mask_upper = y > up_thr

    idx_lower = np.where(mask_lower)[0]
    idx_center = np.where(mask_center)[0]
    idx_upper = np.where(mask_upper)[0]

    groups = np.zeros(len(y), dtype=int)

    groups[idx_lower] = 1
    groups[idx_upper] = 2
    groups[idx_center] = 3

    return groups, idx_lower, idx_center, idx_upper


# =============================================================================
# SER-A
# =============================================================================

def segmentation_3conditions_target(
    y,
    k_tau: float = 1.5,
    alpha_tail: float = 0.15,
):
    """
    SER-A.

    Median Absolute Deviation (MAD)-based segmentation.
    """

    y = np.asarray(y)

    median = np.median(y)

    tau = k_tau * mad(y)

    low_thr = median - tau
    up_thr = median + tau

    groups, idxL, idxC, idxU = split_by_target_thresholds(
        y,
        low_thr,
        up_thr,
    )

    keep_lower = tail_mean_check(
        y,
        np.isin(np.arange(len(y)), idxL),
        np.isin(np.arange(len(y)), idxC),
        alpha=alpha_tail,
    )

    keep_upper = tail_mean_check(
        y,
        np.isin(np.arange(len(y)), idxU),
        np.isin(np.arange(len(y)), idxC),
        alpha=alpha_tail,
    )

    if not keep_lower:

        groups[idxL] = 0

        idxC = np.union1d(idxC, idxL)

        idxL = np.array([], dtype=int)

    if not keep_upper:

        groups[idxU] = 0

        idxC = np.union1d(idxC, idxU)

        idxU = np.array([], dtype=int)

    return {
        "low_thr": low_thr,
        "up_thr": up_thr,
        "groups": groups,
        "idxL": idxL,
        "idxC": idxC,
        "idxU": idxU,
    }


# =============================================================================
# SER-B
# =============================================================================

def segmentation_target_quantiles(
    y,
    q_low: float = 0.10,
    q_up: float = 0.90,
):
    """
    SER-B.

    Fixed quantile segmentation.
    """

    y = np.asarray(y)

    low_thr = np.quantile(y, q_low)
    up_thr = np.quantile(y, q_up)

    groups, idxL, idxC, idxU = split_by_target_thresholds(
        y,
        low_thr,
        up_thr,
    )

    return {
        "low_thr": low_thr,
        "up_thr": up_thr,
        "groups": groups,
        "idxL": idxL,
        "idxC": idxC,
        "idxU": idxU,
    }


# =============================================================================
# SER-C
# =============================================================================

def iterative_target_trimming(
    X,
    y,
    side: str = "left",
    step: int = 50,
    stop_delta: float = 1e-3,
):
    """
    SER-C.

    Iteratively removes observations from one tail until
    no meaningful improvement is obtained.
    """

    y = np.asarray(y)

    order = np.argsort(y)

    performances = []

    n = len(y)

    best = None

    iterator = range(0, n // 3, step)

    for i in iterator:

        if side == "left":

            kept = order[i:]

            threshold = y[order[i]]

        else:

            kept = order[: len(order) - i]

            j = len(order) - i - 1

            threshold = y[order[j]]

        if len(kept) <= X.shape[1] + 1:
            break

        X_subset = X.iloc[kept] if hasattr(X, "iloc") else X[kept]

        y_subset = y[kept]

        model = sm.OLS(
            y_subset,
            sm.add_constant(X_subset),
        ).fit()

        performances.append(
            (
                i,
                model.rsquared,
                np.mean(np.abs(model.resid)),
                threshold,
            )
        )

        if (
            len(performances) > 1
            and abs(
                performances[-1][1] - performances[-2][1]
            )
            < stop_delta
        ):
            best = performances[-1]
            break

    if best is None and performances:

        best = max(
            performances,
            key=lambda x: x[1],
        )

    if best is None:
        return None

    if side == "left":

        return {
            "low_thr": best[3],
            "impr_points": best[0],
            "r2": best[1],
            "mae": best[2],
        }

    return {
        "up_thr": best[3],
        "impr_points": best[0],
        "r2": best[1],
        "mae": best[2],
    }


# =============================================================================
# SER-D
# =============================================================================

def optimize_bias_variance_target(
    X,
    y,
    q_low_grid=np.linspace(0.05, 0.30, 12),
    q_up_grid=np.linspace(0.70, 0.95, 12),
):
    """
    SER-D.

    Search for the segmentation thresholds minimizing the
    bias-variance criterion.
    """

    y = np.asarray(y)

    rows = []

    for q_low in q_low_grid:

        for q_up in q_up_grid:

            if q_low >= q_up:
                continue

            low_thr = np.quantile(y, q_low)

            up_thr = np.quantile(y, q_up)

            _, idxL, idxC, idxU = split_by_target_thresholds(
                y,
                low_thr,
                up_thr,
            )

            if (
                len(idxC) <= X.shape[1] + 1
                or min(len(idxL), len(idxU)) < 20
            ):
                continue

            X_center = X.iloc[idxC]

            y_center = y[idxC]

            model = sm.OLS(
                y_center,
                sm.add_constant(X_center),
            ).fit()

            residuals = model.resid

            variance = np.var(residuals)

            bias = np.mean(residuals)

            vb_score = variance + bias ** 2

            balance = (
                min(len(idxL), len(idxC), len(idxU))
                /
                max(len(idxL), len(idxC), len(idxU))
            )

            score = vb_score * (
                1
                + 0.1 * np.log(
                    1.0 / max(balance, 1e-6)
                )
            )

            rows.append({

                "q_low": q_low,

                "q_up": q_up,

                "low_thr": low_thr,

                "up_thr": up_thr,

                "nL": len(idxL),

                "nC": len(idxC),

                "nU": len(idxU),

                "r2C": model.rsquared,

                "vb": vb_score,

                "bal": balance,

                "score": score,

            })

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("score")

    best = df.iloc[0].to_dict()

    return best, df