"""
Expert models for the Segmentation Expert-Mixture Regularization (SER)
framework.

This module implements all candidate regression models used
inside each SER segment.

Author
------
Sarah Elyane Rashiwa
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import (
    HuberRegressor,
    Lasso,
    PoissonRegressor,
    QuantileRegressor,
    Ridge,
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.preprocessing import PolynomialFeatures

from .utils import (
    add_constant,
    safe_weights,
    to_numpy,
)


# =============================================================================
# Ordinary Least Squares
# =============================================================================

def fit_ols(
    X_group: pd.DataFrame,
    y_group: pd.Series,
    cov_type: str = "HC3",
) -> dict:
    """
    Fit an Ordinary Least Squares (OLS) regression model.

    Parameters
    ----------
    X_group : pandas.DataFrame
        Feature matrix.

    y_group : pandas.Series
        Target vector.

    cov_type : str, default="HC3"
        Robust covariance estimator.

    Returns
    -------
    dict
        Dictionary containing the fitted model,
        predictions and model name.
    """

    X_design = add_constant(X_group)

    model = sm.OLS(
        y_group,
        X_design,
    ).fit(cov_type=cov_type)

    predictions = model.predict(X_design)

    return {
        "model": model,
        "y_pred": to_numpy(predictions),
        "name": "OLS",
    }


# =============================================================================
# Weighted Least Squares
# =============================================================================

def fit_wls(
    X_group,
    y_group,
    residuals,
    cov_type: str = "HC3",
):
    """
    Fit a Weighted Least Squares (WLS) model.
    """

    X_design = add_constant(X_group)

    weights = safe_weights(residuals)

    model = sm.WLS(
        y_group,
        X_design,
        weights=weights,
    ).fit(cov_type=cov_type)

    predictions = model.predict(X_design)

    return {
        "model": model,
        "y_pred": to_numpy(predictions),
        "name": "WLS",
    }


# =============================================================================
# Polynomial Regression
# =============================================================================

def fit_polynomial(
    X_group,
    y_group,
    degree: int = 2,
):
    """
    Fit a polynomial regression model.
    """

    transformer = PolynomialFeatures(degree=degree)

    X_poly = transformer.fit_transform(X_group)

    model = sm.OLS(
        y_group,
        X_poly,
    ).fit()

    predictions = model.predict(X_poly)

    return {
        "model": model,
        "y_pred": to_numpy(predictions),
        "name": f"Poly ({degree})",
    }


# =============================================================================
# Huber Regression
# =============================================================================

def fit_huber(
    X_group,
    y_group,
    epsilon: float = 1.35,
    alpha: float = 0.0,
):
    """
    Fit a Huber robust regression model.
    """

    model = HuberRegressor(
        epsilon=epsilon,
        alpha=alpha,
    )

    model.fit(
        to_numpy(X_group),
        to_numpy(y_group),
    )

    predictions = model.predict(
        to_numpy(X_group)
    )

    return {
        "model": model,
        "y_pred": to_numpy(predictions),
        "name": "Huber",
    }


# =============================================================================
# Poisson Regression
# =============================================================================

def fit_poisson(
    X_group,
    y_group,
):
    """
    Fit a Poisson regression model.
    """

    model = PoissonRegressor()

    model.fit(
        to_numpy(X_group),
        to_numpy(y_group),
    )

    predictions = model.predict(
        to_numpy(X_group)
    )

    return {
        "model": model,
        "y_pred": to_numpy(predictions),
        "name": "Poisson",
    }


# =============================================================================
# Ridge Regression
# =============================================================================

def fit_ridge(
    X_group,
    y_group,
    alpha: float = 1.0,
):
    """
    Fit a Ridge regression model.
    """

    model = Ridge(alpha=alpha)

    model.fit(
        to_numpy(X_group),
        to_numpy(y_group),
    )

    predictions = model.predict(
        to_numpy(X_group)
    )

    return {
        "model": model,
        "y_pred": to_numpy(predictions),
        "name": "Ridge",
    }


# =============================================================================
# Lasso Regression
# =============================================================================

def fit_lasso(
    X_group,
    y_group,
    alpha: float = 1.0,
):
    """
    Fit a Lasso regression model.
    """

    model = Lasso(alpha=alpha)

    model.fit(
        to_numpy(X_group),
        to_numpy(y_group),
    )

    predictions = model.predict(
        to_numpy(X_group)
    )

    return {
        "model": model,
        "y_pred": to_numpy(predictions),
        "name": "Lasso",
    }


# =============================================================================
# Quantile Regression
# =============================================================================

def fit_quantile(
    X_group,
    y_group,
    tau: float = 0.10,
    alpha: float = 0.0,
):
    """
    Fit a Quantile regression model.
    """

    model = QuantileRegressor(
        quantile=tau,
        alpha=alpha,
        solver="highs",
    )

    model.fit(
        to_numpy(X_group),
        to_numpy(y_group),
    )

    predictions = model.predict(
        to_numpy(X_group)
    )

    return {
        "model": model,
        "y_pred": to_numpy(predictions),
        "name": f"Quantile (τ={tau})",
    }