"""
Expert selection module for the Segmentation Expert-Mixture
Regularization (SER) framework.

This module trains all candidate local experts,
evaluates their predictive performance,
and selects the best expert for each segment.

Author
------
Sarah Elyane Rashiwa
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from .utils import to_numpy

from .experts import (
    fit_ols,
    fit_wls,
    fit_polynomial,
    fit_huber,
    fit_poisson,
    fit_quantile,
    fit_ridge,
    fit_lasso,
)


# =============================================================================
# Expert Selection
# =============================================================================

def select_best_local_model(
    X_group,
    y_group,
    residuals=None,
    quantile_tau: float = 0.50,
):
    """
    Train all candidate experts on one segment and
    select the best model according to R².

    Parameters
    ----------
    X_group : pandas.DataFrame

    y_group : pandas.Series

    residuals : array-like, optional
        Residuals from the global model.
        Required only for WLS.

    quantile_tau : float, default=0.50

    Returns
    -------
    best_model : dict

    performance_table : pandas.DataFrame
    """

    X_group = pd.DataFrame(X_group).reset_index(drop=True)

    y_group = pd.Series(y_group).reset_index(drop=True)

    candidates = []

    # -------------------------------------------------------
    # OLS
    # -------------------------------------------------------

    candidates.append(
        fit_ols(
            X_group,
            y_group,
        )
    )

    # -------------------------------------------------------
    # WLS
    # -------------------------------------------------------

    if residuals is not None:

        try:

            candidates.append(

                fit_wls(
                    X_group,
                    y_group,
                    residuals,
                )

            )

        except Exception:

            pass

    # -------------------------------------------------------
    # Polynomial
    # -------------------------------------------------------

    candidates.append(

        fit_polynomial(
            X_group,
            y_group,
            degree=2,
        )

    )

    # -------------------------------------------------------
    # Huber
    # -------------------------------------------------------

    candidates.append(

        fit_huber(
            X_group,
            y_group,
        )

    )

    # -------------------------------------------------------
    # Poisson
    # -------------------------------------------------------

    try:

        candidates.append(

            fit_poisson(
                X_group,
                y_group,
            )

        )

    except Exception:

        pass

    # -------------------------------------------------------
    # Quantile
    # -------------------------------------------------------

    candidates.append(

        fit_quantile(
            X_group,
            y_group,
            tau=quantile_tau,
        )

    )

    # -------------------------------------------------------
    # Ridge
    # -------------------------------------------------------

    candidates.append(

        fit_ridge(
            X_group,
            y_group,
        )

    )

    # -------------------------------------------------------
    # Lasso
    # -------------------------------------------------------

    candidates.append(

        fit_lasso(
            X_group,
            y_group,
        )

    )

    # -------------------------------------------------------
    # Evaluate every candidate
    # -------------------------------------------------------

    results = []

    y_true = to_numpy(y_group)

    for candidate in candidates:

        prediction = candidate["y_pred"]

        results.append(

            {

                "name": candidate["name"],

                "R2": r2_score(
                    y_true,
                    prediction,
                ),

                "MAE": mean_absolute_error(
                    y_true,
                    prediction,
                ),

                "RMSE": np.sqrt(

                    mean_squared_error(
                        y_true,
                        prediction,
                    )

                ),

                "model": candidate["model"],

            }

        )

    performance_table = (

        pd.DataFrame(results)

        .sort_values(
            "R2",
            ascending=False,
        )

        .reset_index(drop=True)

    )

    best = performance_table.iloc[0]

    best_model = {

        "name": best["name"],

        "model": best["model"],

        "R2": best["R2"],

        "MAE": best["MAE"],

        "RMSE": best["RMSE"],

    }

    return best_model, performance_table


# =============================================================================
# Safe Wrapper
# =============================================================================

def train_local_expert(
    X_group,
    y_group,
    label: str,
    tau: float,
    residuals=None,
    min_samples: int = 10,
    verbose: bool = False,
):
    """
    Train a local expert only if enough observations
    are available.

    Parameters
    ----------
    min_samples : int
        Minimum number of observations required
        to train a local model.
    """

    n = 0 if X_group is None else len(X_group)

    if X_group is None or n < min_samples:

        if verbose:

            print(

                f"{label} skipped "

                f"(n={n})"

            )

        return None, None

    return select_best_local_model(

        X_group=X_group,

        y_group=y_group,

        residuals=residuals,

        quantile_tau=tau,

    )