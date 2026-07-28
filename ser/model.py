"""
SERRegressor

Object-Oriented implementation of the
Segmentation Expert-Mixture Regularization (SER) framework.

Author
------
Sarah Elyane Rashiwa
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .utils import add_constant

from .segmentation import (
    segmentation_3conditions_target,
    segmentation_target_quantiles,
    iterative_target_trimming,
    optimize_bias_variance_target,
    split_by_target_thresholds,
)

from .selection import train_local_expert

from .blending import blended_predict

from segmentation.regions import RegionDefinition

class SERRegressor:
    """
    Segmentation Expert-Mixture Regularization.

    Parameters
    ----------
    segmentation : {"A","B","C","D"}

        A : MAD segmentation

        B : Quantile segmentation

        C : Iterative trimming

        D : Bias-Variance optimization

    alpha_ratio : float

        Width of the blending region.

    quantile_bounds : tuple(float,float)

        Quantiles used by method B.

    verbose : bool
    """

    def __init__(
        self,
        segmentation="A",
        alpha_ratio=0.05,
        quantile_bounds=(0.15, 0.85),
        verbose=True,
    ):

        self.segmentation_method = segmentation

        self.alpha_ratio = alpha_ratio

        self.quantile_bounds = quantile_bounds

        self.verbose = verbose

        # Learned during fit()

        self.global_model = None

        self.segmentation = None

        self.local_experts = None

        self.group_performance = None

        self.alpha = None
        
        self.summary = {}

    # =====================================================
    # Global Model
    # =====================================================

    def _fit_global_model(
        self,
        X_train,
        y_train,
    ):

        X_design = add_constant(X_train)

        model = sm.OLS(
            y_train,
            X_design,
        ).fit(cov_type="HC3")

        prediction = model.predict(X_design)

        residuals = (

            pd.Series(y_train).reset_index(drop=True)

            -

            pd.Series(prediction).reset_index(drop=True)

        )

        return model, residuals

    # =====================================================
    # Segmentation
    # =====================================================

    def _segment(
        self,
        X_train,
        y_train,
    ):

        method = self.segmentation_method

        if method == "A":

            return segmentation_3conditions_target(y_train)

        elif method == "B":

            q_low, q_up = self.quantile_bounds

            return segmentation_target_quantiles(
                y_train,
                q_low=q_low,
                q_up=q_up,
            )

        elif method == "C":

            left = iterative_target_trimming(
                X_train,
                y_train,
                side="left",
            )

            right = iterative_target_trimming(
                X_train,
                y_train,
                side="right",
            )

            low_thr = (

                left["low_thr"]

                if left

                else np.quantile(y_train, 0.10)

            )

            up_thr = (

                right["up_thr"]

                if right

                else np.quantile(y_train, 0.90)

            )

            groups, idxL, idxC, idxU = split_by_target_thresholds(

                y_train,

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

        elif method == "D":

            best, _ = optimize_bias_variance_target(

                X_train,

                y_train,

            )

            groups, idxL, idxC, idxU = split_by_target_thresholds(

                y_train,

                best["low_thr"],

                best["up_thr"],

            )

            return {

                "low_thr": best["low_thr"],

                "up_thr": best["up_thr"],

                "groups": groups,

                "idxL": idxL,

                "idxC": idxC,

                "idxU": idxU,

            }

        raise ValueError(

            "Unknown segmentation strategy."

        )

    # =====================================================
    # FIT
    # =====================================================

    def fit(
        self,
        X_train,
        y_train,
        region_definition: Optional[RegionDefinition] = None,
    ):

        if self.verbose:

            print(

                f"\n===== SER ({self.segmentation_method}) ====="

            )
            
        if isinstance(X_train, np.ndarray) : X_train = pd.DataFrame(X_train)
        if isinstance(y_train, np.ndarray): y_train = pd.Series(y_train)

        self.global_model, residuals = self._fit_global_model(

            X_train,

            y_train,

        )

        # ---------------------------------------------------------------
        # Segmentation: use provided RegionDefinition or compute internally
        # ---------------------------------------------------------------
        if region_definition is not None:

            # Convert boolean masks to index arrays
            idxL = np.where(region_definition.lower_mask)[0]
            idxC = np.where(region_definition.center_mask)[0]
            idxU = np.where(region_definition.upper_mask)[0]

            # Build groups array
            groups = np.zeros(len(y_train), dtype=int)
            groups[idxL] = 1
            groups[idxU] = 2
            groups[idxC] = 3

            self.segmentation = {
                "low_thr": region_definition.lower_threshold,
                "up_thr": region_definition.upper_threshold,
                "groups": groups,
                "idxL": idxL,
                "idxC": idxC,
                "idxU": idxU,
            }

        else:

            self.segmentation = self._segment(

                X_train,

                y_train,

            )

        self.alpha = (

            self.alpha_ratio

            *

            (

                self.segmentation["up_thr"]

                -

                self.segmentation["low_thr"]

            )

        )

        idxL = self.segmentation["idxL"]

        idxC = self.segmentation["idxC"]

        idxU = self.segmentation["idxU"]

        experts = {}

        performances = {}

        for label, idx, tau in [

            ("Lower", idxL, 0.10),

            ("Center", idxC, 0.50),

            ("Upper", idxU, 0.90),

        ]:

            X_group = X_train.iloc[idx]

            y_group = y_train.iloc[idx]

            group_residuals = residuals.iloc[idx].reset_index(drop=True)

            best, perf = train_local_expert(

                X_group,

                y_group,

                label=label,

                tau=tau,

                residuals=group_residuals,

                verbose=self.verbose,

            )

            experts[label] = best

            performances[label] = perf

        if experts["Lower"] is None:

            experts["Lower"] = experts["Center"]

        if experts["Upper"] is None:

            experts["Upper"] = experts["Center"]

        self.local_experts = experts

        self.group_performance = performances

        self._build_summary(
            X_train,
            y_train,
        )

        if self.verbose:
            self.print_summary()

        return self

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(
        self,
        X_test,
    ):
        
        if isinstance(X_test,np.ndarray): X_test = pd.DataFrame(X_test)
        return blended_predict(

            X_test=X_test,

            global_model=self.global_model,

            segmentation=self.segmentation,

            local_experts=self.local_experts,

            alpha=self.alpha,

        )

# =====================================================
# FIT + PREDICT
# =====================================================

    def fit_predict(
        self,
        X_train,
        y_train,
        X_test,
    ):
        """
        Fit the SER model and return predictions.

        Parameters
        ----------
        X_train : pandas.DataFrame
            Training predictors.

        y_train : pandas.Series
            Training target.

        X_test : pandas.DataFrame
            Test predictors.

        Returns
        -------
        numpy.ndarray
            Predictions on the test set.
        """

        self.fit(
            X_train,
            y_train,
        )

        return self.predict(
            X_test,
        )
        
    # =====================================================
# Build Summary
# =====================================================

    def _build_summary(
        self,
        X_train,
        y_train,
    ):

        idxL = self.segmentation["idxL"]
        idxC = self.segmentation["idxC"]
        idxU = self.segmentation["idxU"]

        self.summary = {

            "segmentation": self.segmentation_method,

            "alpha_ratio": self.alpha_ratio,

            "alpha": self.alpha,

            "n_samples": len(y_train),

            "n_features": X_train.shape[1],

            "lower_threshold": self.segmentation["low_thr"],

            "upper_threshold": self.segmentation["up_thr"],

            "lower_size": len(idxL),

            "center_size": len(idxC),

            "upper_size": len(idxU),

            "lower_expert": (
                self.local_experts["Lower"]["name"]
                if self.local_experts["Lower"] is not None
                else "Skipped"
            ),

            "center_expert": (
                self.local_experts["Center"]["name"]
                if self.local_experts["Center"] is not None
                else "Skipped"
            ),

            "upper_expert": (
                self.local_experts["Upper"]["name"]
                if self.local_experts["Upper"] is not None
                else "Skipped"
            ),
        }
        # =====================================================
    # Summary
    # =====================================================

    def print_summary(self):

        s = self.summary

        print("\n" + "=" * 50)
        print("              SER SUMMARY")
        print("=" * 50)

        print("\nModel")
        print("-" * 50)
        print(f"Segmentation      : SER-{s['segmentation']}")
        print(f"Alpha ratio       : {s['alpha_ratio']:.3f}")
        print(f"Alpha             : {s['alpha']:.4f}")

        print("\nDataset")
        print("-" * 50)
        print(f"Samples           : {s['n_samples']}")
        print(f"Features          : {s['n_features']}")

        print("\nSegmentation")
        print("-" * 50)
        print(f"Lower threshold   : {s['lower_threshold']:.4f}")
        print(f"Upper threshold   : {s['upper_threshold']:.4f}")

        print()

        print(f"Lower region      : {s['lower_size']}")
        print(f"Center region     : {s['center_size']}")
        print(f"Upper region      : {s['upper_size']}")

        print("\nExperts")
        print("-" * 50)
        print(f"Lower             : {s['lower_expert']}")
        print(f"Center            : {s['center_expert']}")
        print(f"Upper             : {s['upper_expert']}")

        print("=" * 50)