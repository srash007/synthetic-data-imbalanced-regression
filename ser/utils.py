"""
Utility functions for the Segmentation Expert-Mixture Regularization (SER)
framework.

These utilities are shared across the segmentation, experts,
blending, and pipeline modules.

Author: Sarah Elyane Rashiwa
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import PolynomialFeatures


# =============================================================================
# Data Utilities
# =============================================================================

def to_numpy(x) -> np.ndarray:
    """
    Convert an input object to a NumPy array while preserving its order.

    Parameters
    ----------
    x : array-like
        Pandas DataFrame, Series, list or NumPy array.

    Returns
    -------
    np.ndarray
        NumPy representation of the input.
    """
    return x.values if hasattr(x, "values") else np.asarray(x)


def add_constant(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add an intercept column for StatsModels regression.

    Parameters
    ----------
    X : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame
        Design matrix including an intercept.
    """
    return sm.add_constant(X, has_constant="add")


def safe_weights(residuals, epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute stable inverse-squared residual weights for
    Weighted Least Squares (WLS).

    Parameters
    ----------
    residuals : array-like
        Residual vector.

    epsilon : float, default=1e-6
        Small constant preventing division by zero.

    Returns
    -------
    np.ndarray
        Weight vector.
    """
    residuals = to_numpy(residuals)
    return 1.0 / (residuals ** 2 + epsilon)


# =============================================================================
# Statistics
# =============================================================================

def mad(x: np.ndarray) -> float:
    """
    Compute the Median Absolute Deviation (MAD).

    A robust estimator of statistical dispersion.

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    float
        Median Absolute Deviation scaled by 1.4826.
    """
    median = np.median(x)
    return 1.4826 * np.median(np.abs(x - median))


# =============================================================================
# Prediction Utilities
# =============================================================================

def predict_with_model(model, X_row: pd.DataFrame, model_name: str) -> float:
    """
    Generate a prediction for a single observation using
    either a StatsModels estimator, a Scikit-learn estimator,
    or a polynomial regression model.

    Parameters
    ----------
    model :
        Trained regression model.

    X_row : pandas.DataFrame
        Single observation.

    model_name : str
        Name of the regression model.

    Returns
    -------
    float
        Predicted value.

    Raises
    ------
    ValueError
        If the model type is unsupported.
    """

    # -------------------------------------------------------------------------
    # Polynomial regression
    # -------------------------------------------------------------------------
    if "Poly" in model_name:

        try:
            degree = int(model_name.split("(")[1].split(")")[0])

        except (IndexError, ValueError):
            degree = 2

        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X_row)

        return float(model.predict(X_poly)[0])

    # -------------------------------------------------------------------------
    # Scikit-learn estimators
    # -------------------------------------------------------------------------
    if hasattr(model, "coef_"):

        return float(model.predict(X_row.to_numpy())[0])

    # -------------------------------------------------------------------------
    # StatsModels estimators
    # -------------------------------------------------------------------------
    if hasattr(model, "predict"):

        X_row = add_constant(X_row)

        return float(model.predict(X_row)[0])

    # -------------------------------------------------------------------------
    # Unknown estimator
    # -------------------------------------------------------------------------
    raise ValueError(
        f"Unsupported model type: {type(model).__name__}"
    )