"""Statistical comparison utilities for method benchmarking."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy.stats import friedmanchisquare, norm, ttest_rel, wilcoxon


def _validate_paired_samples(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate two paired sample arrays.

    Parameters
    ----------
    x : np.ndarray | list[float]
        First paired sample.
    y : np.ndarray | list[float]
        Second paired sample.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Flattened and validated arrays.

    Raises
    ------
    ValueError
        If arrays are empty, not same length, or non-finite.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()

    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError("Both paired samples must be non-empty.")
    if x_arr.shape != y_arr.shape:
        raise ValueError("Paired samples must have the same shape.")
    if not np.all(np.isfinite(x_arr)) or not np.all(np.isfinite(y_arr)):
        raise ValueError("Paired samples must contain only finite values.")

    return x_arr, y_arr


def paired_ttest(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
) -> dict[str, Any]:
    """
    Run paired Student's t-test.

    Parameters
    ----------
    x : np.ndarray | list[float]
        First paired sample.
    y : np.ndarray | list[float]
        Second paired sample.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - `statistic` : float
        - `p_value` : float
        - `confidence_interval` : None
        - `effect_size` : float (Cohen's d for paired differences)
    """
    x_arr, y_arr = _validate_paired_samples(x, y)
    test = ttest_rel(x_arr, y_arr)
    return {
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "confidence_interval": None,
        "effect_size": cohens_d(x_arr, y_arr)["effect_size"],
    }


def wilcoxon_test(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    zero_method: str = "wilcox",
) -> dict[str, Any]:
    """
    Run Wilcoxon signed-rank test on paired samples.

    Parameters
    ----------
    x : np.ndarray | list[float]
        First paired sample.
    y : np.ndarray | list[float]
        Second paired sample.
    zero_method : str, default="wilcox"
        Zero handling mode passed to `scipy.stats.wilcoxon`.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - `statistic` : float
        - `p_value` : float
        - `confidence_interval` : None
        - `effect_size` : float (Cliff's delta)
    """
    x_arr, y_arr = _validate_paired_samples(x, y)
    test = wilcoxon(x_arr, y_arr, zero_method=zero_method)
    return {
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "confidence_interval": None,
        "effect_size": cliffs_delta(x_arr, y_arr)["effect_size"],
    }


def friedman_test(*samples: np.ndarray | list[float]) -> dict[str, Any]:
    """
    Run Friedman test across multiple related samples.

    Parameters
    ----------
    *samples : np.ndarray | list[float]
        Related samples (e.g., per-model scores across datasets/folds).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - `statistic` : float
        - `p_value` : float
        - `confidence_interval` : None
        - `effect_size` : float (Kendall's W estimate)

    Raises
    ------
    ValueError
        If fewer than 3 samples are provided or lengths mismatch.
    """
    if len(samples) < 3:
        raise ValueError("Friedman test requires at least 3 related samples.")

    arrays = [np.asarray(s, dtype=float).ravel() for s in samples]
    n = arrays[0].size
    if n == 0:
        raise ValueError("Samples must be non-empty.")
    if any(arr.size != n for arr in arrays):
        raise ValueError("All samples must have equal length.")
    if any(not np.all(np.isfinite(arr)) for arr in arrays):
        raise ValueError("All sample values must be finite.")

    test = friedmanchisquare(*arrays)
    k = len(arrays)
    kendalls_w = float(test.statistic / (n * (k - 1)))
    return {
        "statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "confidence_interval": None,
        "effect_size": kendalls_w,
    }


def nemenyi_test(*samples: np.ndarray | list[float]) -> dict[str, Any]:
    """
    Approximate pairwise Nemenyi-style post-hoc comparisons via rank differences.

    Parameters
    ----------
    *samples : np.ndarray | list[float]
        Related samples, one array per method.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - `statistic` : float (maximum absolute z among pairs)
        - `p_value` : float (minimum pairwise two-sided p-value)
        - `confidence_interval` : None
        - `effect_size` : dict mapping pair names to average-rank difference
        - `pairwise` : dict mapping pair names to z and p values
    """
    if len(samples) < 3:
        raise ValueError("Nemenyi test requires at least 3 related samples.")

    arrays = [np.asarray(s, dtype=float).ravel() for s in samples]
    n = arrays[0].size
    if n == 0:
        raise ValueError("Samples must be non-empty.")
    if any(arr.size != n for arr in arrays):
        raise ValueError("All samples must have equal length.")
    if any(not np.all(np.isfinite(arr)) for arr in arrays):
        raise ValueError("All sample values must be finite.")

    data = np.vstack(arrays).T
    ranks = np.apply_along_axis(
        lambda row: np.argsort(np.argsort(row)).astype(float) + 1.0,
        1,
        data,
    )
    avg_ranks = np.mean(ranks, axis=0)
    k = len(arrays)
    se = np.sqrt(k * (k + 1) / (6.0 * n))

    pairwise: dict[str, dict[str, float]] = {}
    effects: dict[str, float] = {}
    z_values = []
    p_values = []

    for i, j in combinations(range(k), 2):
        diff = float(avg_ranks[i] - avg_ranks[j])
        z = abs(diff) / se if se > 0 else 0.0
        p = float(2.0 * (1.0 - norm.cdf(z)))
        key = f"method_{i}_vs_method_{j}"
        pairwise[key] = {"z": float(z), "p_value": p}
        effects[key] = diff
        z_values.append(z)
        p_values.append(p)

    return {
        "statistic": float(max(z_values) if z_values else 0.0),
        "p_value": float(min(p_values) if p_values else 1.0),
        "confidence_interval": None,
        "effect_size": effects,
        "pairwise": pairwise,
    }


def bootstrap_confidence_interval(
    values: np.ndarray | list[float],
    statistic: str = "mean",
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: int | None = None,
) -> dict[str, Any]:
    """
    Compute bootstrap confidence interval for a univariate sample statistic.

    Parameters
    ----------
    values : np.ndarray | list[float]
        Sample values.
    statistic : {"mean", "median"}, default="mean"
        Statistic for bootstrap resampling.
    confidence_level : float, default=0.95
        Confidence level in (0, 1).
    n_bootstrap : int, default=1000
        Number of bootstrap replicates.
    random_state : int | None, default=None
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - `statistic` : float (original sample statistic)
        - `p_value` : None
        - `confidence_interval` : tuple[float, float]
        - `effect_size` : None
    """
    vals = np.asarray(values, dtype=float).ravel()
    if vals.size == 0:
        raise ValueError("`values` must be non-empty.")
    if not np.all(np.isfinite(vals)):
        raise ValueError("`values` must contain only finite values.")
    if statistic not in {"mean", "median"}:
        raise ValueError("`statistic` must be either 'mean' or 'median'.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("`confidence_level` must be in (0, 1).")
    if n_bootstrap <= 0:
        raise ValueError("`n_bootstrap` must be > 0.")

    rng = np.random.default_rng(random_state)

    if statistic == "mean":
        stat_fn = np.mean
    else:
        stat_fn = np.median

    observed = float(stat_fn(vals))
    boot = np.empty(n_bootstrap, dtype=float)

    n = vals.size
    for i in range(n_bootstrap):
        sample = rng.choice(vals, size=n, replace=True)
        boot[i] = float(stat_fn(sample))

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(boot, alpha / 2.0))
    upper = float(np.quantile(boot, 1.0 - alpha / 2.0))

    return {
        "statistic": observed,
        "p_value": None,
        "confidence_interval": (lower, upper),
        "effect_size": None,
    }


def cohens_d(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
) -> dict[str, Any]:
    """
    Compute Cohen's d effect size between two paired samples.

    Parameters
    ----------
    x : np.ndarray | list[float]
        First sample.
    y : np.ndarray | list[float]
        Second sample.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - `statistic` : float (mean difference)
        - `p_value` : None
        - `confidence_interval` : None
        - `effect_size` : float (Cohen's d)

    Notes
    -----
    For paired samples, Cohen's d is computed on paired differences.
    """
    x_arr, y_arr = _validate_paired_samples(x, y)
    diff = x_arr - y_arr
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    d = mean_diff / std_diff if std_diff > 0 else 0.0
    return {
        "statistic": mean_diff,
        "p_value": None,
        "confidence_interval": None,
        "effect_size": float(d),
    }


def cliffs_delta(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
) -> dict[str, Any]:
    """
    Compute Cliff's delta effect size between two samples.

    Parameters
    ----------
    x : np.ndarray | list[float]
        First sample.
    y : np.ndarray | list[float]
        Second sample.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - `statistic` : float (delta value)
        - `p_value` : None
        - `confidence_interval` : None
        - `effect_size` : float (Cliff's delta)
    """
    x_arr, y_arr = _validate_paired_samples(x, y)
    gt = np.sum(x_arr[:, None] > y_arr[None, :])
    lt = np.sum(x_arr[:, None] < y_arr[None, :])
    n_pairs = x_arr.size * y_arr.size
    delta = float((gt - lt) / n_pairs) if n_pairs > 0 else 0.0
    return {
        "statistic": delta,
        "p_value": None,
        "confidence_interval": None,
        "effect_size": delta,
    }
