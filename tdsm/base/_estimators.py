from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Self

import numpy as np


class Estimator(object, metaclass=ABCMeta):
    def __init__(self) -> None:
        """Initialize the estimator."""
        super(Estimator, self).__init__()

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        """Fit the estimator to training data.

        Args:
            x: Input features used for training.
            y: Target values used for training.

        Returns:
            The fitted estimator.
        """
        ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict outputs for the given input data.

        Args:
            x: Input features to predict on.

        Returns:
            Predicted values.
        """
        ...
