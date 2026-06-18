"""TT core を表す薄いラッパークラス群を提供する。"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Self

import numpy as np

Scalar = int | float | complex | np.generic


class BaseTTCore:
    """TT core の shape 管理を担う基底クラス。"""

    core_ndim: ClassVar[int]
    _data: np.ndarray

    def __init__(
        self,
        data: 'np.ndarray | BaseTTCore',
        copy: bool = True,
    ) -> None:
        """TT core を初期化する。

        Parameters
        ----------
        data : np.ndarray or BaseTTCore
            保持する core 配列、または別の TT core。
        copy : bool, default True
            True の場合は配列をコピーして保持する。

        Raises
        ------
        ValueError
            次元数、rank、または mode 次元が不正な場合。
        """
        if isinstance(data, BaseTTCore):
            array = data.as_array(copy=copy)
        else:
            array = np.asarray(data)
            if copy:
                array = array.copy()
        self._data = np.asarray(array)
        self._validate()

    def __repr__(self) -> str:
        """core の概要を表す文字列を返す。"""
        return f"{type(self).__name__}(shape={self.shape}, dtype={self.dtype})"

    def __array__(
        self,
        dtype: np.dtype[np.generic] | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        """NumPy 配列として返す。"""
        array = self._data if dtype is None else np.asarray(self._data, dtype=dtype)
        if copy:
            return array.copy()
        return array

    def __getitem__(self, key: Any) -> Any:
        """内部配列の要素または slice を返す。

        Parameters
        ----------
        key : Any
            NumPy 配列に渡す index または slice。

        Returns
        -------
        Any
            内部配列から取得した値。
        """
        return self._data[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        """内部配列の要素または slice を更新する。

        Parameters
        ----------
        key : Any
            NumPy 配列に渡す index または slice。
        value : Any
            代入する値。
        """
        self._data[key] = value

    def __mul__(self, scalar: Scalar) -> Self:
        """右から scalar 倍した同型の core を返す。

        Parameters
        ----------
        scalar : Scalar
            内部配列へ掛ける scalar。

        Returns
        -------
        Self
            scalar 倍した core。

        Raises
        ------
        TypeError
            ``scalar`` が scalar でない場合。
        """
        return self.scale(scalar)

    def __rmul__(self, scalar: Scalar) -> Self:
        """左から scalar 倍した同型の core を返す。

        Parameters
        ----------
        scalar : Scalar
            内部配列へ掛ける scalar。

        Returns
        -------
        Self
            scalar 倍した core。

        Raises
        ------
        TypeError
            ``scalar`` が scalar でない場合。
        """
        return self.scale(scalar)

    def _validate(self) -> None:
        """core shape の整合性を検証する。

        Raises
        ------
        ValueError
            次元数、rank、または mode 次元が不正な場合。
        """
        if self._data.ndim != self.core_ndim:
            raise ValueError(f"Each core must be {self.core_ndim}-dimensional.")
        if self.left_rank <= 0 or self.right_rank <= 0:
            raise ValueError("TT ranks must be positive.")
        if any(size <= 0 for size in self.mode_dims):
            raise ValueError("Mode dimensions must be positive.")

    @property
    def shape(self) -> tuple[int, ...]:
        """配列 shape を返す。"""
        return self._data.shape

    @property
    def ndim(self) -> int:
        """配列の次元数を返す。"""
        return self._data.ndim

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """配列 dtype を返す。"""
        return self._data.dtype

    @property
    def left_rank(self) -> int:
        """左 rank を返す。"""
        return int(self.shape[0])

    @property
    def right_rank(self) -> int:
        """右 rank を返す。"""
        return int(self.shape[-1])

    @property
    def mode_dims(self) -> tuple[int, ...]:
        """rank 軸を除いた mode 次元を返す。"""
        return tuple(int(size) for size in self.shape[1:-1])

    @property
    def mode_size(self) -> int:
        """rank 軸を除いた mode 次元の積を返す。"""
        return math.prod(self.mode_dims)

    def as_array(self, copy: bool = False) -> np.ndarray:
        """内部配列を返す。

        Parameters
        ----------
        copy : bool, default False
            True の場合はコピーを返す。

        Returns
        -------
        np.ndarray
            内部配列。
        """
        return self._data.copy() if copy else self._data

    def astype(
        self,
        dtype: np.dtype[np.generic],
        copy: bool = True,
    ) -> np.ndarray:
        """内部配列の dtype を変換した結果を返す。

        Parameters
        ----------
        dtype : np.dtype
            変換後の dtype。
        copy : bool, default True
            True の場合はコピーを返す。

        Returns
        -------
        np.ndarray
            dtype 変換後の配列。
        """
        return self._data.astype(dtype, copy=copy)

    def copy(self) -> Self:
        """deep copy を返す。"""
        return type(self)(self._data, copy=True)

    def reverse_ranks(self) -> Self:
        """左右 rank 軸を入れ替えた同型の core を返す。

        Returns
        -------
        Self
            mode 次元の順序を保ち、先頭と末尾の rank 軸だけを
            入れ替えた core。
        """
        axes = (self.ndim - 1, *range(1, self.ndim - 1), 0)
        return type(self)(self._data.transpose(axes), copy=False)

    def left_unfold(self) -> np.ndarray:
        """左から見た行列化を返す。"""
        return self._data.reshape(self.left_rank * self.mode_size, self.right_rank)

    def right_unfold(self) -> np.ndarray:
        """右から見た行列化を返す。"""
        return self._data.reshape(self.left_rank, self.mode_size * self.right_rank)

    def from_left_unfold(self, matrix: np.ndarray) -> Self:
        """左行列化から同型の core を復元する。

        Parameters
        ----------
        matrix : np.ndarray
            形状 ``(left_rank * mode_size, new_right_rank)`` の行列。

        Returns
        -------
        Self
            復元した core。
        """
        return type(self)(
            matrix.reshape(self.left_rank, *self.mode_dims, matrix.shape[1]),
            copy=False,
        )

    def from_right_unfold(self, matrix: np.ndarray) -> Self:
        """右行列化から同型の core を復元する。

        Parameters
        ----------
        matrix : np.ndarray
            形状 ``(new_left_rank, mode_size * right_rank)`` の行列。

        Returns
        -------
        Self
            復元した core。
        """
        return type(self)(
            matrix.reshape(matrix.shape[0], *self.mode_dims, self.right_rank),
            copy=False,
        )

    def scale(self, scalar: Scalar) -> Self:
        """scalar 倍した同型の core を返す。

        Parameters
        ----------
        scalar : Scalar
            内部配列へ掛ける scalar。

        Returns
        -------
        Self
            scalar 倍した core。

        Raises
        ------
        TypeError
            ``scalar`` が scalar でない場合。
        """
        if not np.isscalar(scalar):
            raise TypeError("scalar must be a scalar.")
        return type(self)(self._data * scalar, copy=False)

    def scale_right_rank(self, weights: np.ndarray) -> Self:
        """右 rank 方向に 1 次元の重みを掛けた同型の core を返す。

        右 rank 軸だけを対角スケーリングするための専用 helper です。
        概念的には ``apply_right_factor(np.diag(weights))`` と同じですが、
        対角行列を作らず、``weights`` の次元と長さを明示的に検証します。
        broadcast に依存した要素積と異なり、右 rank 軸をスケールする
        操作であることを API として表します。

        Parameters
        ----------
        weights : np.ndarray
            右 rank と同じ長さを持つ 1 次元配列。

        Returns
        -------
        Self
            右 rank 方向に重みを掛けた core。

        Raises
        ------
        ValueError
            ``weights`` が 1 次元でない、または長さが右 rank と一致しない場合。
        """
        weights_array = np.asarray(weights)
        if weights_array.ndim != 1:
            raise ValueError("weights must be 1-dimensional.")
        if weights_array.size != self.right_rank:
            raise ValueError("weights size must match the right rank.")

        broadcast_shape = (1,) * (self.ndim - 1) + (weights_array.size,)
        return type(self)(self._data * weights_array.reshape(broadcast_shape), copy=False)

    def apply_left_factor(self, factor: np.ndarray) -> Self:
        """左 rank 方向から行列を掛けた core を返す。

        左 rank 軸と ``factor`` の列方向を縮約する線形変換です。
        要素積ではなく、左 rank を ``factor.shape[0]`` に変換します。

        Parameters
        ----------
        factor : np.ndarray
            形状 ``(new_left_rank, left_rank)`` の行列。

        Returns
        -------
        Self
            左から変換を掛けた core。
        """
        return type(self)(np.tensordot(factor, self._data, axes=(1, 0)), copy=False)

    def apply_right_factor(self, factor: np.ndarray) -> Self:
        """右 rank 方向へ行列を掛けた core を返す。

        右 rank 軸と ``factor`` の行方向を縮約する線形変換です。
        要素積ではなく、右 rank を ``factor.shape[1]`` に変換します。

        Parameters
        ----------
        factor : np.ndarray
            形状 ``(right_rank, new_right_rank)`` の行列。

        Returns
        -------
        Self
            右へ変換を掛けた core。
        """
        return type(self)(np.tensordot(self._data, factor, axes=(-1, 0)), copy=False)


class TTTensorCore(BaseTTCore):
    """3 次元 TT tensor core。"""

    core_ndim = 3

    @classmethod
    def from_factor(
        cls,
        factor: np.ndarray,
        dtype: np.dtype[np.generic] | None = None,
    ) -> Self:
        """rank-one factor から TT tensor core を生成する。

        Parameters
        ----------
        factor : np.ndarray
            1 次元 factor 配列。
        dtype : np.dtype or None, optional
            core に使う dtype。省略時は入力から推定する。

        Returns
        -------
        Self
            形状 ``(1, factor.size, 1)`` の TT tensor core。
        """
        factor_array = np.asarray(factor, dtype=dtype).reshape(-1)
        return cls(factor_array[np.newaxis, :, np.newaxis], copy=False)

    @property
    def mode_dim(self) -> int:
        """mode 次元を返す。"""
        return self.mode_dims[0]

    def as_operator_row_core(self) -> "TTOperatorCore":
        """mode を row 側へ置いた TT operator core を返す。

        Returns
        -------
        TTOperatorCore
            形状 ``(left_rank, mode_dim, 1, right_rank)`` の core。
        """
        return TTOperatorCore(self._data[:, :, np.newaxis, :], copy=False)

    def as_operator_col_core(self) -> "TTOperatorCore":
        """mode を column 側へ置いた TT operator core を返す。

        Returns
        -------
        TTOperatorCore
            形状 ``(left_rank, 1, mode_dim, right_rank)`` の core。
        """
        return TTOperatorCore(self._data[:, np.newaxis, :, :], copy=False)


class TTOperatorCore(BaseTTCore):
    """4 次元 TT operator core。"""

    core_ndim = 4

    @classmethod
    def from_rank_matrix(cls, matrix: np.ndarray) -> Self:
        """rank 間の小行列を row/column mode 次元 1 の operator core に変換する。

        Parameters
        ----------
        matrix : np.ndarray
            形状 ``(left_rank, right_rank)`` の行列。

        Returns
        -------
        Self
            形状 ``(left_rank, 1, 1, right_rank)`` の TT operator core。

        Raises
        ------
        ValueError
            ``matrix`` が 2 次元でない場合。
        """
        array = np.asarray(matrix)
        if array.ndim != 2:
            raise ValueError("matrix must be 2-dimensional.")
        return cls(array[:, np.newaxis, np.newaxis, :], copy=False)

    @classmethod
    def from_factor_as_row(
        cls,
        factor: np.ndarray,
        dtype: np.dtype[np.generic] | None = None,
    ) -> Self:
        """rank-one factor を row 側 operator core に変換する。

        Parameters
        ----------
        factor : np.ndarray
            1 次元 factor 配列。
        dtype : np.dtype or None, optional
            core に使う dtype。省略時は入力から推定する。

        Returns
        -------
        Self
            形状 ``(1, factor.size, 1, 1)`` の TT operator core。
        """
        factor_array = np.asarray(factor, dtype=dtype).reshape(-1)
        return cls(factor_array[np.newaxis, :, np.newaxis, np.newaxis], copy=False)

    @classmethod
    def from_factor_as_col(
        cls,
        factor: np.ndarray,
        dtype: np.dtype[np.generic] | None = None,
    ) -> Self:
        """rank-one factor を column 側 operator core に変換する。

        Parameters
        ----------
        factor : np.ndarray
            1 次元 factor 配列。
        dtype : np.dtype or None, optional
            core に使う dtype。省略時は入力から推定する。

        Returns
        -------
        Self
            形状 ``(1, 1, factor.size, 1)`` の TT operator core。
        """
        factor_array = np.asarray(factor, dtype=dtype).reshape(-1)
        return cls(factor_array[np.newaxis, np.newaxis, :, np.newaxis], copy=False)

    @property
    def row_dim(self) -> int:
        """row 次元を返す。"""
        return self.mode_dims[0]

    @property
    def col_dim(self) -> int:
        """column 次元を返す。"""
        return self.mode_dims[1]

    def transpose_modes(self) -> Self:
        """row/column mode 次元を入れ替えた core を返す。"""
        return type(self)(self._data.transpose(0, 2, 1, 3), copy=False)

    def adjoint_modes(self) -> Self:
        """row/column を入れ替えて複素共役した core を返す。"""
        return type(self)(np.conj(self._data.transpose(0, 2, 1, 3)), copy=False)
