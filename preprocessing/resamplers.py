# -*- coding: utf-8 -*-
"""
Resampling baselines for imbalanced regression: SmoteR, Gaussian Noise
resampling, and Random UnderSampling -- built to share the same target
balance logic as SMOGN (samp_method="balance"), and the same
RegionDefinition-based rare/normal split as SER, so all methods in the
Q1 comparison agree on what "rare" means and how much rebalancing to aim
for. Only the *mechanism* used to add/remove points differs between
methods -- that isolates the actual research question.

Definitions (see Branco, Torgo & Ribeiro; Torgo et al. 2013 SmoteR;
UBL package 'GaussNoiseRegress' / 'RandUnderRegress' for the reference
algorithms these approximate):

  SmoteR                : oversample the rare region by k-NN interpolation
                           (always -- no safe/unsafe branching, unlike
                           SMOGN), undersample the normal region randomly.
  Gaussian Noise         : oversample the rare region by duplicating a
                           random rare point and perturbing every feature
                           (and the target) with Gaussian noise scaled by
                           that feature's std * `pert`, undersample the
                           normal region randomly.
  Random UnderSampling   : rare region is left untouched (no synthetic
                           points at all), only the normal region is
                           randomly undersampled.

All three expose `.fit_resample(X, y, region_definition) -> (X_res, y_res)`,
matching preprocessing.smogn.SMOGNPreprocessor's interface.
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from segmentation.regions import RegionDefinition


# ================================================================
# Shared target-size logic (mirrors smogn.smoter's samp_method="balance"
# formula for the 2-bin rare/normal case, so every method aims for the
# same final composition).
# ================================================================
def _target_sizes(n_rare: int, n_normal: int, samp_method: str = "balance"):
    """Returns (target_rare_size, target_normal_size)."""
    n = n_rare + n_normal
    b = round(n / 2)

    if samp_method == "balance":
        s_perc_rare = b / n_rare
        s_perc_normal = b / n_normal
    elif samp_method == "extreme":
        scale_rare = b ** 2 / n_rare
        scale_normal = b ** 2 / n_normal
        scale = 2 * b / (scale_rare + scale_normal)
        s_perc_rare = round((scale_rare * scale) / n_rare, 1)
        s_perc_normal = round((scale_normal * scale) / n_normal, 1)
    else:
        raise ValueError("samp_method must be 'balance' or 'extreme'")

    target_rare = max(n_rare, round(n_rare * s_perc_rare))
    target_normal = min(n_normal, round(n_normal * s_perc_normal))
    return target_rare, target_normal


def _rare_normal_indices(region_definition: RegionDefinition):
    rare_idx = np.where(region_definition.rare_mask)[0]
    normal_idx = np.where(region_definition.normal_mask)[0]
    if len(rare_idx) == 0:
        raise ValueError("region_definition has an empty rare region -- nothing to resample.")
    return rare_idx, normal_idx


def _undersample_normal(normal_idx: np.ndarray, target_normal: int, rng: np.random.Generator):
    if target_normal >= len(normal_idx):
        return normal_idx
    return rng.choice(normal_idx, size=target_normal, replace=False)


# ================================================================
# SmoteR: pure k-NN interpolation oversampling (no Gaussian fallback)
# ================================================================
class SmoteRPreprocessor:
    """
    Torgo et al. (2013)'s SmoteR: for each synthetic point, pick a random
    rare observation, interpolate towards one of its k nearest neighbors
    *within the rare region only*, weight the synthetic target by inverse
    distance to each endpoint (same formula SMOGN itself uses for its
    "safe" case -- SmoteR is SMOGN without the safe/unsafe branching).

    Uses sklearn's NearestNeighbors (vectorized) rather than the
    O(n^2) pure-Python distance loop the `smogn` package uses internally
    -- much faster for the same neighbor logic.
    """

    def __init__(self, k: int = 5, samp_method: str = "balance", random_state: Optional[int] = None):
        self.k = k
        self.samp_method = samp_method
        self.random_state = random_state

    def fit_resample(self, X, y, region_definition: RegionDefinition):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        rng = np.random.default_rng(self.random_state)

        rare_idx, normal_idx = _rare_normal_indices(region_definition)
        target_rare, target_normal = _target_sizes(len(rare_idx), len(normal_idx), self.samp_method)
        n_synth = max(0, target_rare - len(rare_idx))

        X_rare = X.iloc[rare_idx].reset_index(drop=True)
        y_rare = y.iloc[rare_idx].reset_index(drop=True)

        synth_X, synth_y = [], []
        if n_synth > 0 and len(X_rare) >= 2:
            k = min(self.k, len(X_rare) - 1)
            nn = NearestNeighbors(n_neighbors=k + 1).fit(X_rare.to_numpy())
            _, knn_idx = nn.kneighbors(X_rare.to_numpy())
            knn_idx = knn_idx[:, 1:]  # drop self-match (first column)

            anchors = rng.integers(0, len(X_rare), size=n_synth)
            for anchor in anchors:
                neigh = knn_idx[anchor, rng.integers(0, k)]
                x1, x2 = X_rare.iloc[anchor].to_numpy(dtype=float), X_rare.iloc[neigh].to_numpy(dtype=float)
                y1, y2 = float(y_rare.iloc[anchor]), float(y_rare.iloc[neigh])

                w = rng.random()
                x_new = x1 + w * (x2 - x1)

                d1 = np.linalg.norm(x1 - x_new)
                d2 = np.linalg.norm(x2 - x_new)
                y_new = y1 if (d1 + d2) == 0 else (d2 * y1 + d1 * y2) / (d1 + d2)

                synth_X.append(x_new)
                synth_y.append(y_new)

        kept_normal = _undersample_normal(normal_idx, target_normal, rng)

        X_parts = [X.iloc[rare_idx], X.iloc[kept_normal]]
        y_parts = [y.iloc[rare_idx], y.iloc[kept_normal]]
        if synth_X:
            X_parts.append(pd.DataFrame(synth_X, columns=X.columns))
            y_parts.append(pd.Series(synth_y))

        X_res = pd.concat(X_parts, ignore_index=True).to_numpy()
        y_res = pd.concat(y_parts, ignore_index=True).to_numpy()
        return X_res, y_res


# ================================================================
# Gaussian Noise resampling (UBL's GaussNoiseRegress)
# ================================================================
class GaussianNoisePreprocessor:
    """
    Oversamples the rare region by duplicating a random rare observation
    and perturbing every feature AND the target with Gaussian noise,
    scaled by that column's std within the rare region times `pert`.
    Same perturbation formula SMOGN's own "unsafe" fallback uses --
    Gaussian Noise resampling is what SMOGN reduces to when every rare
    point is treated as unsafe. Undersamples the normal region randomly.
    """

    def __init__(self, pert: float = 0.02, samp_method: str = "balance", random_state: Optional[int] = None):
        self.pert = pert
        self.samp_method = samp_method
        self.random_state = random_state

    def fit_resample(self, X, y, region_definition: RegionDefinition):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        rng = np.random.default_rng(self.random_state)

        rare_idx, normal_idx = _rare_normal_indices(region_definition)
        target_rare, target_normal = _target_sizes(len(rare_idx), len(normal_idx), self.samp_method)
        n_synth = max(0, target_rare - len(rare_idx))

        X_rare = X.iloc[rare_idx].reset_index(drop=True)
        y_rare = y.iloc[rare_idx].reset_index(drop=True)

        feat_std = np.array(X_rare.std(ddof=0), dtype=float, copy=True)
        feat_std[feat_std == 0] = 1e-6  # avoid zero-noise on constant columns
        y_std = float(y_rare.std(ddof=0)) or 1e-6

        synth_X, synth_y = [], []
        if n_synth > 0 and len(X_rare) >= 1:
            anchors = rng.integers(0, len(X_rare), size=n_synth)
            for anchor in anchors:
                x0 = X_rare.iloc[anchor].to_numpy(dtype=float)
                y0 = float(y_rare.iloc[anchor])

                x_new = x0 + rng.normal(loc=0.0, scale=feat_std * self.pert)
                y_new = y0 + rng.normal(loc=0.0, scale=y_std * self.pert)

                synth_X.append(x_new)
                synth_y.append(y_new)

        kept_normal = _undersample_normal(normal_idx, target_normal, rng)

        X_parts = [X.iloc[rare_idx], X.iloc[kept_normal]]
        y_parts = [y.iloc[rare_idx], y.iloc[kept_normal]]
        if synth_X:
            X_parts.append(pd.DataFrame(synth_X, columns=X.columns))
            y_parts.append(pd.Series(synth_y))

        X_res = pd.concat(X_parts, ignore_index=True).to_numpy()
        y_res = pd.concat(y_parts, ignore_index=True).to_numpy()
        return X_res, y_res


# ================================================================
# Random UnderSampling (UBL's RandUnderRegress)
# ================================================================
class RandomUnderSamplingPreprocessor:
    """
    No synthetic points at all -- the rare region is kept exactly as-is.
    Only the normal region is randomly undersampled toward the same
    target balance the other methods use. This isolates the effect of
    "just remove majority data" as a baseline against actually
    synthesizing rare observations.
    """

    def __init__(self, samp_method: str = "balance", random_state: Optional[int] = None):
        self.samp_method = samp_method
        self.random_state = random_state

    def fit_resample(self, X, y, region_definition: RegionDefinition):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        rng = np.random.default_rng(self.random_state)

        rare_idx, normal_idx = _rare_normal_indices(region_definition)
        _, target_normal = _target_sizes(len(rare_idx), len(normal_idx), self.samp_method)

        kept_normal = _undersample_normal(normal_idx, target_normal, rng)

        X_res = pd.concat([X.iloc[rare_idx], X.iloc[kept_normal]], ignore_index=True).to_numpy()
        y_res = pd.concat([y.iloc[rare_idx], y.iloc[kept_normal]], ignore_index=True).to_numpy()
        return X_res, y_res