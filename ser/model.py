# -*- coding: utf-8 -*-
"""
SER (Segmented Ensemble Regression) — Object-Oriented Version
===============================================================
Object-oriented refactor of the SER algorithm: segments the target
variable into Lower / Center / Upper zones, trains one expert model
per zone, then produces a continuous prediction via alpha-blending
across zone boundaries.

Architecture:
  - BaseRegressionModel (+ subclasses): common fit/predict interface
    for each candidate model type (OLS, WLS, Poly, Huber, Poisson,
    Quantile). Each class encapsulates its own prediction logic
    (no more manual dispatch based on model type).
  - ModelSelector: trains every candidate on a group and keeps the
    best one according to R^2.
  - BaseSegmenter (+ subclasses A/B/C/D): each target-segmentation
    method is an interchangeable strategy.
  - BlendedPredictor: applies the Lower <-> Center <-> Upper
    alpha-blending.
  - SERPipeline: orchestrator (replaces the old run_pipeline function).
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import HuberRegressor, PoissonRegressor, QuantileRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ================================================================
# UTILITIES
# ================================================================
class Utils:
    @staticmethod
    def to_numpy(x):
        """Ensure np.array (preserves order)."""
        return x.values if hasattr(x, "values") else np.asarray(x)

    @staticmethod
    def add_const(X):
        """Add a constant column (statsmodels)."""
        return sm.add_constant(X, has_constant="add")

    @staticmethod
    def safe_weights(resid, eps=1e-6):
        """WLS weights = 1 / (resid^2 + eps) to avoid division by zero."""
        r = Utils.to_numpy(resid)
        return 1.0 / (r ** 2 + eps)

    @staticmethod
    def mad(x):
        med = np.median(x)
        return 1.4826 * np.median(np.abs(x - med))

    @staticmethod
    def metrics_table(y_true, y_pred, label):
        y_true = Utils.to_numpy(y_true)
        y_pred = Utils.to_numpy(y_pred)
        return pd.DataFrame([{
            "Model": label,
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        }])


# ================================================================
# CANDIDATE MODELS (Strategy Pattern)
# ================================================================
class BaseRegressionModel(ABC):
    """Common interface: fit(X, y) -> self ; predict(X) -> np.ndarray."""

    name: str = "Base"

    def __init__(self):
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X, y, **kwargs):
        ...

    @abstractmethod
    def predict(self, X):
        ...


class OLSModel(BaseRegressionModel):
    name = "OLS"

    def __init__(self, cov_type="HC3"):
        super().__init__()
        self.cov_type = cov_type

    def fit(self, X, y, **kwargs):
        Xc = Utils.add_const(X)
        self.model = sm.OLS(y, Xc).fit(cov_type=self.cov_type)
        self.is_fitted = True
        return self

    def predict(self, X):
        return Utils.to_numpy(self.model.predict(Utils.add_const(X)))


class WLSModel(BaseRegressionModel):
    name = "WLS"

    def __init__(self, cov_type="HC3"):
        super().__init__()
        self.cov_type = cov_type

    def fit(self, X, y, resid_from_ols=None, **kwargs):
        Xc = Utils.add_const(X)
        w = Utils.safe_weights(resid_from_ols)
        self.model = sm.WLS(y, Xc, weights=w).fit(cov_type=self.cov_type)
        self.is_fitted = True
        return self

    def predict(self, X):
        return Utils.to_numpy(self.model.predict(Utils.add_const(X)))


class PolynomialModel(BaseRegressionModel):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree
        self.name = f"Poly ({degree})"
        self.poly = PolynomialFeatures(degree=degree)

    def fit(self, X, y, **kwargs):
        Xp = self.poly.fit_transform(X)
        self.model = sm.OLS(y, Xp).fit()
        self.is_fitted = True
        return self

    def predict(self, X):
        Xp = self.poly.transform(X)
        return Utils.to_numpy(self.model.predict(Xp))


class HuberModel(BaseRegressionModel):
    name = "Huber"

    def __init__(self, epsilon=1.35, alpha=0.0):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha

    def fit(self, X, y, **kwargs):
        self.model = HuberRegressor(epsilon=self.epsilon, alpha=self.alpha)
        self.model.fit(Utils.to_numpy(X), Utils.to_numpy(y))
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.model.predict(Utils.to_numpy(X))


class PoissonModel(BaseRegressionModel):
    name = "Poisson"

    def fit(self, X, y, **kwargs):
        self.model = PoissonRegressor()
        self.model.fit(Utils.to_numpy(X), Utils.to_numpy(y))
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.model.predict(Utils.to_numpy(X))


class QuantileModel(BaseRegressionModel):
    def __init__(self, tau=0.5, alpha=0.0):
        super().__init__()
        self.tau = tau
        self.alpha = alpha
        self.name = f"Quantile(tau={tau})"

    def fit(self, X, y, **kwargs):
        self.model = QuantileRegressor(quantile=self.tau, alpha=self.alpha, solver="highs")
        self.model.fit(Utils.to_numpy(X), Utils.to_numpy(y))
        self.is_fitted = True
        return self

    def predict(self, X):
        return self.model.predict(Utils.to_numpy(X))


# ================================================================
# MODEL SELECTOR (replaces fit_group_all_models + safe_fit_group)
# ================================================================
class ModelSelector:
    """Trains several candidate models on a group and keeps the best one (R^2)."""

    def __init__(self, quantile_tau=0.5, min_n=10, verbose=False):
        self.quantile_tau = quantile_tau
        self.min_n = min_n
        self.verbose = verbose
        self.best_model_: Optional[BaseRegressionModel] = None
        self.best_metrics_: Optional[dict] = None
        self.perf_df_: Optional[pd.DataFrame] = None

    def _candidates(self, resid_from_ols):
        candidates = [OLSModel()]
        if resid_from_ols is not None:
            candidates.append(WLSModel())
        candidates += [PolynomialModel(degree=2), HuberModel(), PoissonModel(),
                       QuantileModel(tau=self.quantile_tau)]
        return candidates

    def fit(self, X, y, label="Group", resid_from_ols=None):
        X = pd.DataFrame(X).copy()
        X.index = range(len(X))
        y = pd.Series(y).copy()
        y.index = range(len(y))

        if len(X) < self.min_n:
            if self.verbose:
                print(f"Group {label} skipped (n={len(X)})")
            return self

        if self.verbose:
            print(f"\nTraining group {label} (n={len(X)})")

        results = []
        y_true = Utils.to_numpy(y)
        for cand in self._candidates(resid_from_ols):
            try:
                cand.fit(X, y, resid_from_ols=resid_from_ols)
                y_pred = cand.predict(X)
                results.append({
                    "name": cand.name,
                    "R2": r2_score(y_true, y_pred),
                    "MAE": mean_absolute_error(y_true, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                    "model": cand,
                })
            except Exception:
                continue  # silently skip a candidate that fails to fit

        if not results:
            return self

        df_perf = pd.DataFrame(results).sort_values("R2", ascending=False)
        self.perf_df_ = df_perf
        best_row = df_perf.iloc[0]
        self.best_model_ = best_row["model"]
        self.best_metrics_ = {k: best_row[k] for k in ("name", "R2", "MAE", "RMSE")}

        
        return self


# ================================================================
# SEGMENTATION (Strategy Pattern) — methods A / B / C / D
# ================================================================
class SegmentationResult:
    def __init__(self, low_thr, up_thr, groups, idxL, idxC, idxU):
        self.low_thr = low_thr
        self.up_thr = up_thr
        self.groups = groups
        self.idxL = idxL
        self.idxC = idxC
        self.idxU = idxU


class BaseSegmenter(ABC):
    name = "Base"

    @staticmethod
    def split_by_thresholds(y, low_thr, up_thr):
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

    @abstractmethod
    def segment(self, X, y) -> SegmentationResult:
        ...


class MADSegmenter(BaseSegmenter):
    """Method A: segmentation around the median based on the MAD."""
    name = "A"

    def __init__(self, k_tau=1.5, alpha_tail=0.15):
        self.k_tau = k_tau
        self.alpha_tail = alpha_tail

    @staticmethod
    def _tail_mean_check(res, mask_tail, mask_center, alpha):
        """True if the tail's mean error is distinct from the center's."""
        eps = 1e-12
        mu_tail = res[mask_tail].mean() if mask_tail.any() else 0.0
        mu_center = res[mask_center].mean() if mask_center.any() else 0.0
        return abs(mu_tail - mu_center) > alpha * (res.std() + eps)

    def segment(self, X, y) -> SegmentationResult:
        y = np.asarray(y)
        med = np.median(y)
        tau_c = self.k_tau * Utils.mad(y)
        low_thr, up_thr = med - tau_c, med + tau_c

        groups, idxL, idxC, idxU = self.split_by_thresholds(y, low_thr, up_thr)
        n = len(y)
        keepL = self._tail_mean_check(y, np.isin(np.arange(n), idxL), np.isin(np.arange(n), idxC), self.alpha_tail)
        keepU = self._tail_mean_check(y, np.isin(np.arange(n), idxU), np.isin(np.arange(n), idxC), self.alpha_tail)

        if not keepL:
            groups[idxL] = 0
            idxC = np.union1d(idxC, idxL)
            idxL = np.array([], dtype=int)
        if not keepU:
            groups[idxU] = 0
            idxC = np.union1d(idxC, idxU)
            idxU = np.array([], dtype=int)

        return SegmentationResult(low_thr, up_thr, groups, idxL, idxC, idxU)


class QuantileSegmenter(BaseSegmenter):
    """Method B: segmentation using fixed quantiles of the target."""
    name = "B"

    def __init__(self, q_low=0.10, q_up=0.90):
        self.q_low = q_low
        self.q_up = q_up

    def segment(self, X, y) -> SegmentationResult:
        y = np.asarray(y)
        low_thr = np.quantile(y, self.q_low)
        up_thr = np.quantile(y, self.q_up)
        groups, idxL, idxC, idxU = self.split_by_thresholds(y, low_thr, up_thr)
        return SegmentationResult(low_thr, up_thr, groups, idxL, idxC, idxU)


class IterativeSegmenter(BaseSegmenter):
    """Method C: iterative removal of target extremes, maximizing R^2."""
    name = "C"

    def __init__(self, pas=50, stop_delta=1e-3, default_q=(0.10, 0.90)):
        self.pas = pas
        self.stop_delta = stop_delta
        self.default_q = default_q

    def _search_bound(self, X, y, cote):
        y_arr = np.asarray(y)
        order = np.argsort(y_arr)
        perf = []
        best = None
        n = len(y_arr)

        for i in range(0, n // 3, self.pas):
            if cote == "gauche":
                kept = order[i:]
                bound = y_arr[order[i]] if i < len(order) else y_arr.min()
            else:
                kept = order[: len(order) - i]
                j = len(order) - i - 1
                bound = y_arr[order[j]] if j >= 0 else y_arr.max()

            if len(kept) <= X.shape[1] + 1:
                break  # not enough data left to fit a model

            Xs = X.iloc[kept] if hasattr(X, "iloc") else X[kept]
            ys = y.iloc[kept] if hasattr(y, "iloc") else y[kept]
            model = sm.OLS(ys, sm.add_constant(Xs)).fit()
            perf.append((i, model.rsquared, np.mean(np.abs(model.resid)), bound))

            if len(perf) > 1 and abs(perf[-1][1] - perf[-2][1]) < self.stop_delta:
                best = perf[-1]
                break

        if not best and perf:
            best = max(perf, key=lambda t: t[1])
        return best

    def segment(self, X, y) -> SegmentationResult:
        best_left = self._search_bound(X, y, "gauche")
        best_right = self._search_bound(X, y, "droite")
        low_thr = best_left[3] if best_left else np.quantile(y, self.default_q[0])
        up_thr = best_right[3] if best_right else np.quantile(y, self.default_q[1])
        groups, idxL, idxC, idxU = self.split_by_thresholds(y, low_thr, up_thr)
        return SegmentationResult(low_thr, up_thr, groups, idxL, idxC, idxU)


class BiasVarianceSegmenter(BaseSegmenter):
    """Method D: bias-variance optimization of the center over a quantile grid."""
    name = "D"

    def __init__(self, q_low_grid=None, q_up_grid=None, min_tail_n=20):
        self.q_low_grid = q_low_grid if q_low_grid is not None else np.linspace(0.05, 0.30, 12)
        self.q_up_grid = q_up_grid if q_up_grid is not None else np.linspace(0.70, 0.95, 12)
        self.min_tail_n = min_tail_n
        self.grid_: Optional[pd.DataFrame] = None  # full search history

    def segment(self, X, y) -> SegmentationResult:
        y_arr = np.asarray(y)
        rows = []

        for ql in self.q_low_grid:
            for qu in self.q_up_grid:
                if ql >= qu:
                    continue

                low_thr = np.quantile(y_arr, ql)
                up_thr = np.quantile(y_arr, qu)
                _, idxL, idxC, idxU = self.split_by_thresholds(y_arr, low_thr, up_thr)

                if len(idxC) <= X.shape[1] + 1 or min(len(idxL), len(idxU)) < self.min_tail_n:
                    continue

                Xc = X.iloc[idxC] if hasattr(X, "iloc") else X[idxC]
                yc = y.iloc[idxC] if hasattr(y, "iloc") else y[idxC]
                mdl = sm.OLS(yc, sm.add_constant(Xc)).fit()
                res_c = mdl.resid

                var_c = np.var(res_c)
                bias_c = np.mean(res_c)
                score_vb = var_c + bias_c ** 2
                bal = min(len(idxL), len(idxC), len(idxU)) / max(len(idxL), len(idxC), len(idxU))
                score_final = score_vb * (1 + 0.1 * np.log(1.0 / max(bal, 1e-6)))

                rows.append({
                    "q_low": ql, "q_up": qu, "low_thr": low_thr, "up_thr": up_thr,
                    "nL": len(idxL), "nC": len(idxC), "nU": len(idxU),
                    "r2C": mdl.rsquared, "vb": score_vb, "bal": bal, "score": score_final,
                })

        if not rows:
            raise RuntimeError("No valid threshold combination found for method D.")

        df = pd.DataFrame(rows).sort_values("score")
        self.grid_ = df
        best = df.iloc[0].to_dict()
        groups, idxL, idxC, idxU = self.split_by_thresholds(y_arr, best["low_thr"], best["up_thr"])
        return SegmentationResult(best["low_thr"], best["up_thr"], groups, idxL, idxC, idxU)


# ================================================================
# ALPHA-BLENDING PREDICTION
# ================================================================
class BlendedPredictor:
    """
    Continuous prediction with alpha-blending across Lower <-> Center <-> Upper.
    Zones are decided ONLY from the global model's prediction (y_hat_global)
    -> no information leakage.
    """

    def __init__(self, global_model: OLSModel, seg: SegmentationResult,
                 group_models: Dict[str, Optional[BaseRegressionModel]], alpha: float):
        self.global_model = global_model
        self.seg = seg
        self.group_models = group_models
        self.alpha = alpha

    def predict(self, X_test):
        low_thr, up_thr, alpha = self.seg.low_thr, self.seg.up_thr, self.alpha
        y_hat_global = np.asarray(self.global_model.predict(X_test))

        mdl_C = self.group_models["Center"]
        mdl_L = self.group_models["Lower"] or mdl_C
        mdl_U = self.group_models["Upper"] or mdl_C

        y_preds = []
        for i, yg in enumerate(y_hat_global):
            x_i = X_test.iloc[i:i + 1]
            try:
                if yg < low_thr - alpha:                                   # 1) Pure Lower zone
                    pred = mdl_L.predict(x_i)[0]
                elif low_thr - alpha <= yg <= low_thr + alpha:             # 2) Lower-Center transition
                    w = (low_thr + alpha - yg) / (2 * alpha)
                    pred = w * mdl_L.predict(x_i)[0] + (1 - w) * mdl_C.predict(x_i)[0]
                elif low_thr + alpha < yg < up_thr - alpha:                # 3) Pure Center zone
                    pred = mdl_C.predict(x_i)[0]
                elif up_thr - alpha <= yg <= up_thr + alpha:               # 4) Center-Upper transition
                    w = (up_thr + alpha - yg) / (2 * alpha)
                    pred = w * mdl_C.predict(x_i)[0] + (1 - w) * mdl_U.predict(x_i)[0]
                else:                                                       # 5) Pure Upper zone
                    pred = mdl_U.predict(x_i)[0]
            except Exception as e:
                print(f"Error on observation {i}: {e}")
                pred = np.nan
            y_preds.append(pred)

        return np.array(y_preds)


# ================================================================
# MAIN PIPELINE (replaces run_pipeline)
# ================================================================
class SERRegressor:
    """
    Full SER pipeline:
      1) naive global model (OLS)
      2) target segmentation (method A/B/C/D)
      3) one expert model trained per segment (Lower/Center/Upper)
      4) continuous prediction via alpha-blending
      5) evaluation (Naive vs SER vs global Huber)

    Usage:
        pipeline = SERPipeline(methode="A", verbose=True)
        results = pipeline.run(X_train, y_train, X_test, y_test)
    """

    SEGMENTERS = {
        "A": MADSegmenter,
        "B": QuantileSegmenter,
        "C": IterativeSegmenter,
        "D": BiasVarianceSegmenter,
    }

    def __init__(self, methode="A", min_group_n=50, verbose=True, segmenter_kwargs=None):
        if methode not in self.SEGMENTERS:
            raise ValueError("Unknown method. Choose from: 'A', 'B', 'C', 'D'.")
        self.methode = methode
        self.min_group_n = min_group_n
        self.verbose = verbose
        self.segmenter = self.SEGMENTERS[methode](**(segmenter_kwargs or {}))

        self.global_model_: Optional[OLSModel] = None
        self.huber_global_: Optional[HuberModel] = None
        self.seg_: Optional[SegmentationResult] = None
        self.group_models_: Dict[str, Optional[BaseRegressionModel]] = {}
        self.group_perf_: Dict[str, Optional[pd.DataFrame]] = {}
        self.alpha_: Optional[float] = None

    # ---------------------------------------------------------
    def fit(self, X_train, y_train):
        if self.verbose:
            print(f"\n{'=' * 30}\nPIPELINE - method {self.methode}\n{'=' * 30}")
            print(f"Train: {len(X_train)}\n")

        X_train = pd.DataFrame(X_train).reset_index(drop=True)
        y_train = pd.Series(y_train).reset_index(drop=True)

        # 1) Naive global model
        self.global_model_ = OLSModel().fit(X_train, y_train)
        

        y_hat_train_global = pd.Series(self.global_model_.predict(X_train))
        resid_global = (y_train - y_hat_train_global).reset_index(drop=True)

        

        # 2) Segmentation
        self.seg_ = self.segmenter.segment(X_train, y_train)
        self.alpha_ = 0.05 * (self.seg_.up_thr - self.seg_.low_thr)

        idxL, idxC, idxU = self.seg_.idxL, self.seg_.idxC, self.seg_.idxU
        if self.verbose:
            print(f"\nGroup sizes: Lower={len(idxL)} / Center={len(idxC)} / Upper={len(idxU)}")

        # 3) Train experts per group
        groups_config = {"Lower": (idxL, 0.10), "Center": (idxC, 0.50), "Upper": (idxU, 0.90)}
        for name, (idx, tau) in groups_config.items():
            Xg, yg = X_train.iloc[idx], y_train.iloc[idx]
            resid_g = resid_global.iloc[idx].reset_index(drop=True)
            selector = ModelSelector(quantile_tau=tau, min_n=10, verbose=self.verbose)
            selector.fit(Xg, yg, label=name, resid_from_ols=resid_g)
            self.group_models_[name] = selector.best_model_
            self.group_perf_[name] = selector.perf_df_

        # Fall back to the Center model if a group is absent/too small
        if self.group_models_["Lower"] is None or len(idxL) < self.min_group_n:
            self.group_models_["Lower"] = self.group_models_["Center"]
        if self.group_models_["Upper"] is None or len(idxU) < self.min_group_n:
            self.group_models_["Upper"] = self.group_models_["Center"]

        return self

    # ---------------------------------------------------------
    def predict(self, X_test):
        X_test = pd.DataFrame(X_test).reset_index(drop=True)
        blender = BlendedPredictor(self.global_model_, self.seg_, self.group_models_, self.alpha_)
        return blender.predict(X_test)

    # ---------------------------------------------------------
    def evaluate(self, X_test, y_test):
        X_test = pd.DataFrame(X_test).reset_index(drop=True)
        y_test = pd.Series(y_test).reset_index(drop=True)

        y_pred_naive = self.global_model_.predict(X_test)
        y_pred_blend = self.predict(X_test)
        y_pred_huber = self.huber_global_.predict(X_test)

        df_all = pd.concat([
            Utils.metrics_table(y_test, y_pred_naive, "Naive"),
            Utils.metrics_table(y_test, y_pred_blend, "SER"),
            Utils.metrics_table(y_test, y_pred_huber, "Huber"),
        ], ignore_index=True)

        if self.verbose:
            print(f"\n{'=' * 30}\nFinal summary (test set) - method: {self.methode}\n{'=' * 30}")
            print(df_all)

        return {
            "global_model": self.global_model_,
            "seg": self.seg_,
            "group_models": self.group_models_,
            "group_perf_dfs": self.group_perf_,
            "pred_naive": y_pred_naive,
            "pred_blend": y_pred_blend,
            "pred_huber": y_pred_huber,
            "metrics": df_all,
            "X_test": X_test,
            "y_test": y_test,
        }

    # ---------------------------------------------------------
    def run(self, X_train, y_train, X_test, y_test):
        """Shortcut for fit + evaluate (equivalent to the old run_pipeline() function)."""
        self.fit(X_train, y_train)
        return self.evaluate(X_test, y_test)


