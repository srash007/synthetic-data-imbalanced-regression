import sys
from typing import Optional

import numpy as np
import pandas as pd
import smogn

from segmentation.regions import RegionDefinition

# =====================================================================
# Compatibility patches for the installed `smogn` package
# =====================================================================
# The public `smogn` package (last released years ago) has confirmed bugs
# that surface as soon as region_definition-based (manual) relevance is
# used, and separately whenever real over-sampling runs. Two are fixed
# here; a third cannot be fixed by monkeypatching and requires an
# environment change -- see the warning below.
#
# BUG 1 (pure Python logic bug, independent of numpy version):
#   smogn.phi_ctrl_pts.phi_range() builds the derivative list for
#   rel_method="manual" with `m.insert(len(sx), 0)` instead of
#   `m.insert(len(m), 0)`. For our 3-point control scheme
#   ([low_thr,1], [median,0], [up_thr,1]) this inserts at the wrong
#   position and produces a mis-shaped control-point array, which then
#   crashes deeper inside phi_range() with a ValueError. Patched below by
#   `_fixed_manual_ctrl_pts`, which rebuilds a correct (and still
#   monotonicity-safe, since smogn.phi() re-adjusts these slopes anyway)
#   set of control points.
#
# BUG 2 (numpy>=2.0 incompatibility):
#   smogn.box_plot_stats() calls `np.quantile(x, q, interpolation=...)`.
#   The `interpolation` kwarg was removed in NumPy 2.0 (replaced by
#   `method`). box_plot_stats() is called both by phi_ctrl_pts() in
#   "auto" mode AND by over_sampling() (to size the Gaussian-noise
#   perturbation), so this breaks almost any real SMOGN run on numpy>=2.0.
#   Patched below with a numpy-version-safe reimplementation.
#
# BUG 3 (numpy>=2.0 incompatibility, NOT patchable here):
#   Inside over_sampling(), `int(np.random.choice(a=..., size=1))` relies
#   on implicitly converting a 1-element ndarray to a Python scalar --
#   removed in NumPy 2.0 ("only 0-dimensional arrays can be converted to
#   Python scalars"). This is inline code inside a large function body,
#   not a swappable helper, so it can't be fixed with a targeted
#   monkeypatch here. If fit_resample() below raises that TypeError, pin
#   numpy below 2.0 in this environment:
#       pip install "numpy<2"
#   (safe: this only affects the numpy version used at runtime, not any
#   of the boundary-sharing logic itself.)

_SMOGN_PATCHED = False


def _patch_smogn_for_this_environment():
    """Idempotent: applies the fixes above once per process."""
    global _SMOGN_PATCHED
    if _SMOGN_PATCHED:
        return

    def _box_plot_stats_safe(x, coef=1.5):
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

    def _fixed_manual_ctrl_pts(ctrl_pts):
        ctrl_pts = np.array(ctrl_pts, dtype=float)
        ctrl_pts = ctrl_pts[np.argsort(ctrl_pts[:, 0])]
        num_pts = ctrl_pts.shape[0]

        dx = ctrl_pts[1:, 0] - ctrl_pts[:-1, 0]
        dy = ctrl_pts[1:, 1] - ctrl_pts[:-1, 1]
        sx = dy / dx

        m = np.empty(num_pts)
        m[0] = sx[0]
        m[-1] = sx[-1]
        if num_pts > 2:
            m[1:-1] = (sx[:-1] + sx[1:]) / 2

        flat = []
        for i in range(num_pts):
            flat.extend([ctrl_pts[i, 0], ctrl_pts[i, 1], m[i]])
        return {"method": "manual", "num_pts": num_pts, "ctrl_pts": flat}

    # Import submodules explicitly so sys.modules holds the real module
    # objects -- `import smogn.X as x` binds to the smogn package's
    # attribute namespace instead (which smogn/__init__.py re-exports as
    # functions), not to the actual submodule.
    import smogn.smoter
    import smogn.over_sampling
    import smogn.phi_ctrl_pts
    import smogn.box_plot_stats

    smoter_mod = sys.modules["smogn.smoter"]
    over_sampling_mod = sys.modules["smogn.over_sampling"]
    phi_ctrl_pts_mod = sys.modules["smogn.phi_ctrl_pts"]
    box_plot_stats_mod = sys.modules["smogn.box_plot_stats"]

    box_plot_stats_mod.box_plot_stats = _box_plot_stats_safe
    phi_ctrl_pts_mod.box_plot_stats = _box_plot_stats_safe
    over_sampling_mod.box_plot_stats = _box_plot_stats_safe

    original_phi_ctrl_pts = phi_ctrl_pts_mod.phi_ctrl_pts

    def _patched_phi_ctrl_pts(y, method="auto", xtrm_type="both", coef=1.5, ctrl_pts=None):
        if method == "manual":
            return _fixed_manual_ctrl_pts(ctrl_pts)
        return original_phi_ctrl_pts(y=y, method=method, xtrm_type=xtrm_type, coef=coef, ctrl_pts=ctrl_pts)

    smoter_mod.phi_ctrl_pts = _patched_phi_ctrl_pts

    _SMOGN_PATCHED = True


_patch_smogn_for_this_environment()


class SMOGNPreprocessor:
    """
    Wrapper around the SMOGN algorithm.

    Parameters
    ----------
    rel_thres : float, default=0.8
        Relevance threshold defining rare observations.

    k : int, default=5
        Number of nearest neighbors.

    pert : float, default=0.02
        Gaussian noise perturbation.

    rel_method : str, default="auto"
        Method used to compute the relevance function.
    """

    def __init__(
        self,
        rel_thres=0.8,
        rel_method="auto",
        samp_method="balance",
        k=5,
        pert=0.02,
        
    ):

        self.rel_thres = rel_thres
        self.k = k
        self.pert = pert
        self.rel_method = rel_method
        self.samp_method = samp_method

    def fit_resample(
        self,
        X,
        y,
        region_definition: Optional[RegionDefinition] = None,
    ):
        """
        Apply SMOGN resampling.

        Parameters
        ----------
        X : ndarray or DataFrame
        y : ndarray or Series

        region_definition : RegionDefinition, optional
            If provided, forces SMOGN's relevance function to be anchored
            at region_definition.lower_threshold / .upper_threshold
            (rel_method="manual"), instead of computing its own "auto"
            box-plot-based relevance. This is what lets SMOGN and SER
            agree on (approximately) the same rare/normal boundary -- see
            the note in fit_resample's body for the precision caveat.

        Returns
        -------
        X_resampled
        y_resampled
        """

        # Build DataFrame expected by smogn
        X = pd.DataFrame(X).copy()
        X["target"] = y

        # ---------------------------------------------------------------
        # Prepare keyword arguments for smogn.smoter()
        # ---------------------------------------------------------------
        # NOTE: the installed `smogn` package (PyPI) does NOT accept
        # rare_index/normal_index kwargs -- smogn.smoter()'s signature has
        # no such parameters (verified: data, y, k, pert, samp_method,
        # under_samp, drop_na_col, drop_na_row, replace, rel_thres,
        # rel_method, rel_xtrm_type, rel_coef, rel_ctrl_pts_rg). Passing
        # rare_index/normal_index raises a TypeError -- there is no way to
        # inject an arbitrary rare/normal index partition directly through
        # the public API.
        #
        # Instead, when a RegionDefinition is supplied, we switch SMOGN to
        # rel_method="manual" and anchor its relevance control points
        # exactly at region_definition's lower/upper thresholds (phi=1
        # there) with phi=0 at the median. This is the closest the public
        # API allows to forcing SMOGN to agree with an externally shared
        # boundary. Caveat: phi is a smooth monotone spline between control
        # points (see smogn/phi.py), so the *exact* rare/normal crossing at
        # rel_thres will sit slightly inside these anchors rather than
        # exactly on low_thr/up_thr -- if you need bit-for-bit identical
        # partitions between SER and SMOGN, compare
        # region_definition.rare_mask against the resampled target's range
        # empirically rather than assuming exact equality.
        smoter_kwargs = dict(
            data=X,
            y="target",
            rel_thres=self.rel_thres,
            k=self.k,
            pert=self.pert,
            samp_method=self.samp_method,
        )

        if region_definition is not None:
            low_thr = region_definition.lower_threshold
            up_thr = region_definition.upper_threshold
            median_y = float(np.median(np.asarray(y, dtype=float)))

            smoter_kwargs["rel_method"] = "manual"
            smoter_kwargs["rel_ctrl_pts_rg"] = [
                [low_thr, 1],
                [median_y, 0],
                [up_thr, 1],
            ]
        else:
            smoter_kwargs["rel_method"] = self.rel_method

        # Apply SMOGN
        data_resampled = smogn.smoter(**smoter_kwargs)

        # Split predictors and target
        X_resampled = data_resampled.drop(columns="target").to_numpy()
        y_resampled = data_resampled["target"].to_numpy()

        return X_resampled, y_resampled
