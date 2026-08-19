from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RegionDefinition:
    """
    Store a common segmentation of the target variable.

    This object is intended to be shared between different
    imbalanced regression algorithms (SER, SMOGN, future methods)
    so that they operate on exactly the same definition of
    rare and normal observations.

    Attributes
    ----------
    lower_threshold : float
        Lower boundary of the rare region.

    upper_threshold : float
        Upper boundary of the rare region.

    lower_mask : ndarray
        Boolean mask selecting observations in the lower region.

    center_mask : ndarray
        Boolean mask selecting observations in the central region.

    upper_mask : ndarray
        Boolean mask selecting observations in the upper region.

    rare_mask : ndarray
        Boolean mask selecting all rare observations.

    normal_mask : ndarray
        Boolean mask selecting normal observations.
    """

    lower_threshold: float
    upper_threshold: float

    lower_mask: np.ndarray
    center_mask: np.ndarray
    upper_mask: np.ndarray

    rare_mask: np.ndarray
    normal_mask: np.ndarray
    
    
    @classmethod
    def from_thresholds(
        cls,
        y: np.ndarray,
        lower: float,
        upper: float,
    ):
        """
        Build a RegionDefinition from two thresholds.

        Parameters
        ----------
        y : ndarray
            Target values.

        lower : float
            Lower threshold.

        upper : float
            Upper threshold.

        Returns
        -------
        RegionDefinition
            Complete segmentation object.
        """

        lower_mask = y <= lower

        upper_mask = y >= upper

        center_mask = (~lower_mask) & (~upper_mask)

        rare_mask = lower_mask | upper_mask

        normal_mask = center_mask

        return cls(
            lower_threshold=lower,
            upper_threshold=upper,
            lower_mask=lower_mask,
            center_mask=center_mask,
            upper_mask=upper_mask,
            rare_mask=rare_mask,
            normal_mask=normal_mask,
        )
        
    @property
    def n_lower(self):
        """Number of lower-tail observations."""
        return int(self.lower_mask.sum())


    @property
    def n_center(self):
        """Number of central observations."""
        return int(self.center_mask.sum())


    @property
    def n_upper(self):
        """Number of upper-tail observations."""
        return int(self.upper_mask.sum())


    @property
    def n_rare(self):
        """Number of rare observations."""
        return int(self.rare_mask.sum())
    
    def summary(self):
        """
        Print a summary of the segmentation.
        """

        print("=" * 50)
        print("Region Definition")
        print("=" * 50)

        print(f"Lower threshold : {self.lower_threshold:.4f}")
        print(f"Upper threshold : {self.upper_threshold:.4f}")

        print()

        print(f"Lower samples   : {self.n_lower}")
        print(f"Center samples  : {self.n_center}")
        print(f"Upper samples   : {self.n_upper}")
        print(f"Rare samples    : {self.n_rare}")
