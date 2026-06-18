"""状態ベクトルを TT 特徴表現へ lift する辞書と builder を提供する。"""

from operator import index
from typing import override

import numpy as np

from ..base._dicts import CoreDictionary


class Monomials4TT(CoreDictionary):
    """TT 用の単項式辞書。"""

    degree: int
    _degree_list: np.ndarray

    def __init__(self, degree: int) -> None:
        """単項式辞書を初期化する。

        Args:
            degree: 単項式の次数。
        """
        self.degree = degree
        self._degree_list = np.arange(self.degree + 1)

    @property
    def output_dim(self) -> int:
        """特徴ベクトルの次元を返す。"""
        return int(self._degree_list.size)

    @property
    @override
    def constant_index(self) -> int:
        """定数項に対応する特徴 index を返す。"""
        return 0

    def s2i(self, degree: int) -> int:
        """単項式の次数を特徴 index に変換する。

        Args:
            degree: 変換する単項式の次数。

        Returns:
            対応する特徴 index。

        Raises:
            TypeError: ``degree`` が整数でない場合。
            ValueError: ``degree`` が辞書に含まれない場合。
        """
        if isinstance(degree, (bool, np.bool_)):
            raise ValueError("degree must be an integer")
        try:
            degree_int = index(degree)
        except TypeError as exc:
            raise TypeError("degree must be an integer") from exc
        if degree_int < 0 or degree_int > self.degree:
            raise ValueError(f"degree must satisfy 0 <= degree <= {self.degree}")
        return degree_int

    @override
    def lift_point(self, x: np.number) -> np.ndarray:
        """`1, x, x^2, ..., x^degree` を返す。

        Args:
            x: 入力スカラー。

        Returns:
            単項式特徴ベクトル。
        """
        return np.power(x, self._degree_list)

    @override
    def lift_batch(self, x_1d: np.ndarray) -> np.ndarray:
        """1 次元サンプル列を単項式辞書で一括評価する。

        Args:
            x_1d: 形状 ``(n_samples,)`` の 1 次元サンプル列。

        Returns:
            形状 ``(degree + 1, n_samples)`` の特徴行列。

        Raises:
            ValueError: 入力が空の場合。
        """
        values = np.asarray(x_1d).reshape(-1)
        if values.size == 0:
            raise ValueError("x_1d must contain at least one sample")
        return np.power(values[np.newaxis, :], self._degree_list[:, np.newaxis])

    @override
    def reconstruct(self, features: np.ndarray) -> np.generic:
        """単項式特徴ベクトルから 1 次項を取り出す。

        Args:
            features: 形状 ``(degree + 1,)`` の単項式特徴ベクトル。

        Returns:
            1 次項の値。

        Raises:
            ValueError: 1 次項を持たない場合、または入力長が不正な場合。
        """
        if self.degree < 1:
            raise ValueError("first-order terms are not available when degree < 1")
        values = np.asarray(features).reshape(-1)
        if values.size != self.output_dim:
            raise ValueError("features must have length output_dim")
        return values[1]
