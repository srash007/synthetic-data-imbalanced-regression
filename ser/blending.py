"""
Prediction blending for the Segmentation Expert-Mixture
Regularization (SER) framework.

This module implements the α-blending strategy proposed
in the SER paper.

Author
------
Sarah Elyane Rashiwa
"""

from __future__ import annotations

import numpy as np
import statsmodels.api as sm

from .utils import (
    add_constant,
    predict_with_model,
)


# =============================================================================
# Alpha Blending Prediction
# =============================================================================

def blended_predict(
    X_test,
    global_model,
    segmentation,
    local_experts,
    alpha: float,
):
    """
    Perform continuous α-blending prediction without information leakage.

    Region assignment is performed exclusively using the predictions
    of the global model rather than the true target values.

    Parameters
    ----------
    X_test : pandas.DataFrame
        Test feature matrix.

    global_model :
        Global regression model.

    segmentation : dict
        Segmentation information returned by the segmentation module.

    local_experts : dict
        Dictionary containing the selected expert for
        Lower, Center and Upper regions.

    alpha : float
        Blending transition width.

    Returns
    -------
    np.ndarray
        Final blended predictions.
    """

    low_thr = segmentation["low_thr"]
    up_thr = segmentation["up_thr"]

    # ---------------------------------------------------------------------
    # Global prediction
    # ---------------------------------------------------------------------

    X_design = add_constant(X_test)

    global_prediction = np.asarray(
        global_model.predict(X_design)
    )

    # ---------------------------------------------------------------------
    # Retrieve local experts
    # ---------------------------------------------------------------------

    center_expert = local_experts["Center"]

    lower_expert = (
        local_experts["Lower"]
        if local_experts["Lower"] is not None
        else center_expert
    )

    upper_expert = (
        local_experts["Upper"]
        if local_experts["Upper"] is not None
        else center_expert
    )

    predictions = []

    # ---------------------------------------------------------------------
    # Alpha blending
    # ---------------------------------------------------------------------

    for index, prediction in enumerate(global_prediction):

        sample = X_test.iloc[index:index + 1]

        try:

            # =============================================================
            # Lower region
            # =============================================================

            if prediction < low_thr - alpha:

                final_prediction = predict_with_model(
                    lower_expert["model"],
                    sample,
                    lower_expert["name"],
                )

            # =============================================================
            # Lower ↔ Center transition
            # =============================================================

            elif low_thr - alpha <= prediction <= low_thr + alpha:

                weight = (
                    (low_thr + alpha - prediction)
                    /
                    (2 * alpha)
                )

                lower_prediction = predict_with_model(
                    lower_expert["model"],
                    sample,
                    lower_expert["name"],
                )

                center_prediction = predict_with_model(
                    center_expert["model"],
                    sample,
                    center_expert["name"],
                )

                final_prediction = (

                    weight * lower_prediction

                    +

                    (1 - weight) * center_prediction

                )

            # =============================================================
            # Center region
            # =============================================================

            elif low_thr + alpha < prediction < up_thr - alpha:

                final_prediction = predict_with_model(
                    center_expert["model"],
                    sample,
                    center_expert["name"],
                )

            # =============================================================
            # Center ↔ Upper transition
            # =============================================================

            elif up_thr - alpha <= prediction <= up_thr + alpha:

                weight = (
                    (up_thr + alpha - prediction)
                    /
                    (2 * alpha)
                )

                center_prediction = predict_with_model(
                    center_expert["model"],
                    sample,
                    center_expert["name"],
                )

                upper_prediction = predict_with_model(
                    upper_expert["model"],
                    sample,
                    upper_expert["name"],
                )

                final_prediction = (

                    weight * center_prediction

                    +

                    (1 - weight) * upper_prediction

                )

            # =============================================================
            # Upper region
            # =============================================================

            else:

                final_prediction = predict_with_model(
                    upper_expert["model"],
                    sample,
                    upper_expert["name"],
                )

        except Exception as error:

            print(

                f"Prediction failed "

                f"for sample {index}: "

                f"{error}"

            )

            final_prediction = np.nan

        predictions.append(final_prediction)

    return np.asarray(predictions)