# -*- coding: utf-8 -*-
"""
Bridge between SMOGN's phi relevance function and SER's region-based
segmentation.

Goal
----
Extract the exact y-value boundaries where SMOGN's relevance function
phi(y) crosses `rel_thres` -- i.e. the boundary SMOGN itself uses to
separate over-sampled (rare) from under-sampled (normal) observations --
and package them as a `RegionDefinition` that can be handed directly to
`SERRegressor.fit(region_definition=...)`. This lets SER's Lower/Center/
Upper segmentation agree exactly with whatever SMOGN would consider
"rare" for a given rel_thres / rel_method / rel_xtrm_type / rel_coef.

Why this file exists instead of just calling smogn.phi_ctrl_pts directly
--------------------------------------------------------------------------
1. numpy>=2.0 compatibility: smogn.box_plot_stats() (used internally by
   phi_ctrl_pts() in "auto" mode) calls
   `np.quantile(x, q, interpolation="midpoint")`. The `interpolation`
   kwarg was removed in NumPy 2.0 (replaced by `method`), so on any
   environment with numpy>=2.0 (yours: 2.4.4), `rel_method="auto"` raises
   a TypeError deep inside the smogn package. `_box_plot_stats` below is
   a numpy-version-safe reimplementation of the exact same logic.
2. smogn.smoter() only exposes the *result* of the rare/normal split
   (the resampled dataframe) -- it never returns the actual y-value
   boundaries. This module recovers them by replicating the same control
   point / phi computation smoter() performs internally, then walking the
   sorted phi curve to find where it crosses rel_thres.

Only rel_method="auto" is implemented (mirrors phi_ctrl_pts.phi_extremes,
the box-plot-based method, which is also smogn's own default). If you use
rel_method="manual" in your own pipeline, you already know your control
points explicitly, so there's nothing to "extract" in that case.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
import smogn

from segmentation.regions import RegionDefinition


# ================================================================
# numpy>=2.0-safe reimplementation of smogn's box-plot relevance logic
# ================================================================
def _box_plot_stats(x: np.ndarray, coef: float = 1.5) -> dict:
    """Identical logic to smogn.box_plot_stats, but numpy>=2.0-safe
    (uses `method=` instead of the removed `interpolation=` kwarg)."""
    x = np.asarray(x, dtype=float)

    median = np.quantile(x, 0.50, method="midpoint")
    q1 = np.quantile(x, 0.25, method="midpoint")
    q3 = np.quantile(x, 0.75, method="midpoint")
    iqr = q3 - q1

    lower = q1 - coef * iqr
    lower_whisk_obs = x[x >= lower].min()

    upper = q3 + coef * iqr
    upper_whisk_obs = x[x <= upper].max()

    return {
        "stats": np.array([lower_whisk_obs, q1, median, q3, upper_whisk_obs]),
        "xtrms": x[(x < lower_whisk_obs) | (x > upper_whisk_obs)],
    }


def _phi_extremes_ctrl_pts(y: np.ndarray, xtrm_type: str = "both", coef: float = 1.5) -> dict:
    """Identical logic to smogn.phi_ctrl_pts.phi_extremes, built on the
    numpy-safe `_box_plot_stats` above. Produces the same control-point
    dict format expected by smogn.phi()."""
    y = np.asarray(y, dtype=float)
    ctrl_pts = []
    bx = _box_plot_stats(y, coef)
    rng = [y.min(), y.max()]

    if xtrm_type in ("both", "low") and (bx["xtrms"] < bx["stats"][0]).any():
        ctrl_pts.extend([bx["stats"][0], 1, 0])
    else:
        ctrl_pts.extend([rng[0], 0, 0])

    if bx["stats"][2] != rng[0]:
        ctrl_pts.extend([bx["stats"][2], 0, 0])

    if xtrm_type in ("both", "high") and (bx["xtrms"] > bx["stats"][4]).any():
        ctrl_pts.extend([bx["stats"][4], 1, 0])
    else:
        if bx["stats"][2] != rng[1]:
            ctrl_pts.extend([rng[1], 0, 0])

    return {"method": "auto", "num_pts": round(len(ctrl_pts) / 3), "ctrl_pts": ctrl_pts}


# ================================================================
# Public API
# ================================================================
def extract_smogn_thresholds(
    y,
    rel_thres: float = 0.8,
    rel_xtrm_type: str = "both",
    rel_coef: float = 1.5,
) -> Dict[str, Optional[float]]:
    """
    Replicates SMOGN's internal ("auto") relevance computation and returns
    the y-value boundaries where phi(y) crosses rel_thres -- the same
    boundaries SMOGN uses to decide which observations get over-sampled
    vs under-sampled.

    Parameters
    ----------
    y : array-like
        Target variable (same values you'd pass to SMOGNPreprocessor.fit_resample).
    rel_thres : float, default 0.8
        Relevance threshold above which an observation is "rare". Must match
        the rel_thres you use in SMOGNPreprocessor for the boundaries to be
        meaningful for that specific SMOGN run.
    rel_xtrm_type : {"both", "high", "low"}, default "both"
        Which tail(s) of the distribution are treated as rare.
    rel_coef : float, default 1.5
        Box-plot whisker coefficient (same meaning as Tukey's IQR fences).

    Returns
    -------
    dict with keys:
        low_thr    : float or None -- boundary of the low tail
                     (None if the low end isn't flagged as rare).
        up_thr     : float or None -- boundary of the high tail
                     (None if the high end isn't flagged as rare).
        y_sorted   : np.ndarray -- y sorted ascending.
        y_phi      : np.ndarray -- phi relevance value for every point in y_sorted.
        phi_params : dict -- raw control points used (for inspection/plotting).
    """
    y_sorted_series = pd.Series(np.sort(np.asarray(y, dtype=float)))

    phi_params = _phi_extremes_ctrl_pts(y_sorted_series.to_numpy(), xtrm_type=rel_xtrm_type, coef=rel_coef)
    y_phi = np.asarray(smogn.phi(y=y_sorted_series, ctrl_pts=phi_params), dtype=float)
    y_sorted = y_sorted_series.to_numpy()

    rare_mask = y_phi >= rel_thres
    if not rare_mask.any():
        raise ValueError(
            f"No observation reaches rel_thres={rel_thres} with rel_xtrm_type="
            f"'{rel_xtrm_type}', rel_coef={rel_coef}. Lower rel_thres or check "
            f"these settings."
        )
    if rare_mask.all():
        raise ValueError(
            f"Every observation exceeds rel_thres={rel_thres} -- raise rel_thres "
            f"or check rel_xtrm_type/rel_coef."
        )

    n = len(y_sorted)
    rare_indices = np.where(rare_mask)[0]

    # Low tail: boundary is the last y-value still flagged rare, walking up
    # from the smallest value. None if the low end was never flagged.
    low_thr = None
    if rare_indices[0] == 0:
        i = 0
        while i < n and rare_mask[i]:
            i += 1
        low_thr = float(y_sorted[i - 1])

    # High tail: boundary is the first y-value (from the top) still flagged
    # rare, walking down from the largest value. None if the high end was
    # never flagged.
    up_thr = None
    if rare_indices[-1] == n - 1:
        i = n - 1
        while i >= 0 and rare_mask[i]:
            i -= 1
        up_thr = float(y_sorted[i + 1])

    return {
        "low_thr": low_thr,
        "up_thr": up_thr,
        "y_sorted": y_sorted,
        "y_phi": y_phi,
        "phi_params": phi_params,
    }


def region_from_smogn_relevance(
    y,
    rel_thres: float = 0.8,
    rel_xtrm_type: str = "both",
    rel_coef: float = 1.5,
) -> RegionDefinition:
    """
    Convenience wrapper: computes SMOGN's implied rare/normal boundary and
    packages it directly as a RegionDefinition, ready to hand to
    `SERRegressor.fit(region_definition=...)`.

    This is the direct answer to "extract SMOGN's relevance threshold and
    apply it as SER's segmentation boundary": SER's Lower/Upper zones then
    exactly match what SMOGN itself would treat as rare, given the same
    rel_thres/rel_xtrm_type/rel_coef.
    """
    thresholds = extract_smogn_thresholds(
        y=y, rel_thres=rel_thres, rel_xtrm_type=rel_xtrm_type, rel_coef=rel_coef,
    )
    low_thr, up_thr = thresholds["low_thr"], thresholds["up_thr"]

    if low_thr is None or up_thr is None:
        raise ValueError(
            "SMOGN's relevance function only flags one tail as rare "
            f"(low_thr={low_thr}, up_thr={up_thr}) with rel_xtrm_type="
            f"'{rel_xtrm_type}'. RegionDefinition needs both bounds -- use "
            "rel_xtrm_type='both', or adjust rel_thres/rel_coef so both "
            "tails get flagged, or build the RegionDefinition manually for "
            "the single-tail case."
        )

    region = RegionDefinition.from_thresholds(
        y=np.asarray(y, dtype=float), lower=low_thr, upper=up_thr,
    )
    # Kept for traceability / debugging -- lets you re-plot the phi curve
    # that produced these bounds later without recomputing it.
    region.smogn_phi_params = thresholds["phi_params"]
    region.smogn_rel_thres = rel_thres
    return region
