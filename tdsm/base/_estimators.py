from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class TDSMBaseEstimator(ABC, RegressorMixin, BaseEstimator):
    """
    Abstract base estimator for Data-Driven Sparse Modeling (DDSM).
    """

    def __init__(self) -> None:
        """Initialize the TDBaseEstimator instance."""
        super(TDSMBaseEstimator, self).__init__()

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> TDSMBaseEstimator:
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training vectors.
        y : ndarray of shape (n_samples, n_targets)
            Target values.

        Returns
        -------
        TDSMBaseEstimator
            The fitted estimator.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression targets for X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples, n_targets)
            Predicted values.
        """
        pass

    def __sklearn_tags__(self) -> dict[str, Any]:
        """
        Specify Scikit-learn estimator tags.

        Returns
        -------
        dict
            Dictionary of boolean tags.
        """
        tags = super(TDSMBaseEstimator, self).__sklearn_tags__()
        tags.target_tags.multi_output = True
        return tags
