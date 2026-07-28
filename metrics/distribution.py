"""Distribution comparison metrics for original versus synthetic targets."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import cramervonmises_2samp, entropy, ks_2samp, wasserstein_distance as sp_wasserstein
from scipy.stats import energy_distance as sp_energy_distance


def _validate_samples(
    original_target: np.ndarray | list[float],
    synthetic_target: np.ndarray | list[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate two target sample arrays.

    Parameters
    ----------
    original_target : np.ndarray | list[float]
        Reference/original target values.
    synthetic_target : np.ndarray | list[float]
        Synthetic/generated target values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Flattened and validated arrays.

    Raises
    ------
    ValueError
        If arrays are empty or contain non-finite values.
    """
    x = np.asarray(original_target, dtype=float).ravel()
    y = np.asarray(synthetic_target, dtype=float).ravel()

    if x.size == 0 or y.size == 0:
        raise ValueError("Both input arrays must be non-empty.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Inputs must contain only finite values.")

    return x, y


def _hist_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 30,
    epsilon: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build normalized histogram probabilities on shared bins.

    Parameters
    ----------
    x : np.ndarray
        First sample array.
    y : np.ndarray
        Second sample array.
    bins : int, default=30
        Number of histogram bins.
    epsilon : float, default=1e-12
        Small additive smoothing value.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Smoothed and normalized probability vectors.
    """
    all_values = np.concatenate([x, y])
    min_val = float(np.min(all_values))
    max_val = float(np.max(all_values))

    if np.isclose(min_val, max_val):
        min_val -= 0.5
        max_val += 0.5

    bin_edges = np.linspace(min_val, max_val, bins + 1)

    px, _ = np.histogram(x, bins=bin_edges, density=False)
    py, _ = np.histogram(y, bins=bin_edges, density=False)

    px = px.astype(float) + epsilon
    py = py.astype(float) + epsilon

    px /= np.sum(px)
    py /= np.sum(py)

    return px, py


def wasserstein_distance(
    original_target: np.ndarray | list[float],
    synthetic_target: np.ndarray | list[float],
) -> float:
    """
    Compute first Wasserstein distance between two target distributions.

    Parameters
    ----------
    original_target : np.ndarray | list[float]
        Reference/original target values.
    synthetic_target : np.ndarray | list[float]
        Synthetic/generated target values.

    Returns
    -------
    float
        Wasserstein distance.
    """
    x, y = _validate_samples(original_target, synthetic_target)
    return float(sp_wasserstein(x, y))


def kl_divergence(
    original_target: np.ndarray | list[float],
    synthetic_target: np.ndarray | list[float],
    bins: int = 30,
    epsilon: float = 1e-12,
) -> float:
    """
    Compute Kullback-Leibler divergence KL(P || Q) from histograms.

    Parameters
    ----------
    original_target : np.ndarray | list[float]
        Reference/original target values.
    synthetic_target : np.ndarray | list[float]
        Synthetic/generated target values.
    bins : int, default=30
        Number of histogram bins.
    epsilon : float, default=1e-12
        Additive smoothing for zero-probability handling.

    Returns
    -------
    float
        KL divergence value.
    """
    x, y = _validate_samples(original_target, synthetic_target)
    px, py = _hist_probabilities(x, y, bins=bins, epsilon=epsilon)
    return float(entropy(px, py))


def jensen_shannon_divergence(
    original_target: np.ndarray | list[float],
    synthetic_target: np.ndarray | list[float],
    bins: int = 30,
    epsilon: float = 1e-12,
) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.

    Parameters
    ----------
    original_target : np.ndarray | list[float]
        Reference/original target values.
    synthetic_target : np.ndarray | list[float]
        Synthetic/generated target values.
    bins : int, default=30
        Number of histogram bins.
    epsilon : float, default=1e-12
        Additive smoothing for zero-probability handling.

    Returns
    -------
    float
        Jensen-Shannon divergence value.
    """
    x, y = _validate_samples(original_target, synthetic_target)
    px, py = _hist_probabilities(x, y, bins=bins, epsilon=epsilon)
    m = 0.5 * (px + py)
    js = 0.5 * entropy(px, m) + 0.5 * entropy(py, m)
    return float(js)


def kolmogorov_smirnov(
    original_target: np.ndarray | list[float],
    synthetic_target: np.ndarray | list[float],
) -> float:
    """
    Compute Kolmogorov-Smirnov two-sample statistic.

    Parameters
    ----------
    original_target : np.ndarray | list[float]
        Reference/original target values.
    synthetic_target : np.ndarray | list[float]
        Synthetic/generated target values.

    Returns
    -------
    float
        KS statistic.
    """
    x, y = _validate_samples(original_target, synthetic_target)
    return float(ks_2samp(x, y).statistic)


def cramer_von_mises(
    original_target: np.ndarray | list[float],
    synthetic_target: np.ndarray | list[float],
) -> float:
    """
    Compute Cramér-von Mises two-sample statistic.

    Parameters
    ----------
    original_target : np.ndarray | list[float]
        Reference/original target values.
    synthetic_target : np.ndarray | list[float]
        Synthetic/generated target values.

    Returns
    -------
    float
        Cramér-von Mises statistic.
    """
    x, y = _validate_samples(original_target, synthetic_target)
    return float(cramervonmises_2samp(x, y).statistic)


def energy_distance(
    original_target: np.ndarray | list[float],
    synthetic_target: np.ndarray | list[float],
) -> float:
    """
    Compute energy distance between two sample distributions.

    Parameters
    ----------
    original_target : np.ndarray | list[float]
        Reference/original target values.
    synthetic_target : np.ndarray | list[float]
        Synthetic/generated target values.

    Returns
    -------
    float
        Energy distance.
    """
    x, y = _validate_samples(original_target, synthetic_target)
    return float(sp_energy_distance(x, y))
