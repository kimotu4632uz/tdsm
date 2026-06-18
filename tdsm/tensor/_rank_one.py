"""rank-1 tensor と TT 表現との変換・内積処理を提供する。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from types import NotImplementedType
from typing import Self, overload

import numpy as np

from ._base import Scalar
from ._tt_core import TTTensorCore
from ._tt_tensor import TTChainTensor, TTTensor


class RankOneTensor(Sequence[np.ndarray]):
    """rank-1 tensor を factor ベクトル列として表すクラス。"""

    _factors: list[np.ndarray]
    _dtype: np.dtype[np.generic]

    def __init__(
        self,
        factors: Sequence[np.ndarray] | "RankOneTensor",
        dtype: np.dtype[np.generic] | None = None,
    ) -> None:
        """rank-1 tensor を初期化する。

        Args:
            factors: 1 次元 factor 配列の列、または別の RankOneTensor。
            dtype: factor の保存に使う dtype。

        Raises:
            ValueError: factor が空、または空の factor を含む場合。
        """
        source = (
            factors.to_list(copy=False)
            if isinstance(factors, RankOneTensor)
            else list(factors)
        )
        if len(source) == 0:
            raise ValueError("factors must contain at least one element")

        if dtype is None:
            dtype_obj = np.dtype(
                np.result_type(*[np.asarray(factor).dtype for factor in source])
            )
        else:
            dtype_obj = np.dtype(dtype)

        self._factors: list[np.ndarray] = []
        for factor in source:
            factor_array = np.asarray(factor, dtype=dtype_obj).reshape(-1)
            if factor_array.size == 0:
                raise ValueError("each factor must be non-empty")
            self._factors.append(factor_array.copy())
        self._dtype = dtype_obj

    def __repr__(self) -> str:
        """rank-1 tensor の概要を表す文字列を返す。"""
        return (
            "[rank-1 tensor]\n"
            f"ndim      = {self.ndim},\n"
            f"mode_dims = {self.mode_dims},\n"
            f"dtype     = {self.dtype}"
        )

    def __len__(self) -> int:
        """factor の個数を返す。"""
        return len(self._factors)

    @overload
    def __getitem__(self, index: int) -> np.ndarray:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[np.ndarray]:
        ...

    def __getitem__(self, index: int | slice) -> np.ndarray | list[np.ndarray]:
        """指定 index または slice の factor を返す。

        Args:
            index: 取得する factor の index または slice。

        Returns:
            指定された factor 配列、または factor 配列の list。
        """
        return self._factors[index]

    @property
    def ndim(self) -> int:
        """mode 数を返す。"""
        return len(self._factors)

    @property
    def mode_dims(self) -> tuple[int, ...]:
        """各 mode の次元を返す。"""
        return tuple(factor.size for factor in self._factors)

    @property
    def ranks(self) -> tuple[int, ...]:
        """rank-1 TT としての TT rank 列を返す。"""
        return (1,) * (self.ndim + 1)

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """factor の dtype を返す。"""
        return self._dtype

    @property
    def memory_size(self) -> int:
        """factor 表現の要素数を返す。"""
        return sum(factor.size for factor in self._factors)

    @property
    def dense_size(self) -> int:
        """dense tensor 表現の要素数を返す。"""
        return math.prod(self.mode_dims)

    @property
    def is_operator(self) -> bool:
        """TT operator かどうかを返す。"""
        return False

    @property
    def is_open_boundary(self) -> bool:
        """open boundary 型の TT tensor かどうかを返す。"""
        return False

    def copy(self) -> Self:
        """factor を deep copy した RankOneTensor を返す。"""
        return type(self)(self, dtype=self.dtype)

    def astype(self, dtype: np.dtype[np.generic]) -> Self:
        """factor の dtype を変換した RankOneTensor を返す。"""
        return type(self)(self, dtype=dtype)

    def _multiply_scalar(self, scalar: Scalar) -> Self | NotImplementedType:
        """scalar 倍した RankOneTensor を返す内部 helper。"""
        if not np.isscalar(scalar):
            return NotImplemented
        factors = self.to_list(copy=True)
        factors[0] *= scalar
        return type(self)(factors, dtype=np.result_type(self.dtype, scalar))

    def __mul__(self, scalar: Scalar) -> Self | NotImplementedType:
        """scalar 倍した RankOneTensor を返す。"""
        return self._multiply_scalar(scalar)

    def __rmul__(self, scalar: Scalar) -> Self | NotImplementedType:
        """scalar 倍した RankOneTensor を返す。"""
        return self._multiply_scalar(scalar)

    def to_list(self, copy: bool = True) -> list[np.ndarray]:
        """factor 配列を list として返す。

        Args:
            copy: True の場合、各 factor をコピーして返す。

        Returns:
            factor 配列の list。
        """
        if copy:
            return [factor.copy() for factor in self._factors]
        return list(self._factors)

    def factor_norms(self) -> np.ndarray:
        """各 factor の Euclidean norm を返す。"""
        return np.asarray(
            [np.linalg.norm(factor) for factor in self._factors],
            dtype=np.float64,
        )

    def tensor_norm(self) -> float:
        """表現している rank-1 tensor の norm を返す。"""
        return float(np.prod(self.factor_norms()))

    def normalized(self) -> tuple[Self, np.generic]:
        """各 factor を正規化し、全体の scale を分離する。

        Returns:
            ``(正規化済み RankOneTensor, 全体 scale)``。
        """
        normalized_factors: list[np.ndarray] = []
        total_scale = self.dtype.type(1.0)
        for factor in self._factors:
            norm = float(np.linalg.norm(factor))
            if norm > 0.0:
                normalized = factor / norm
            else:
                normalized = np.zeros_like(factor, dtype=self.dtype)
                normalized[0] = 1.0
            normalized_factors.append(normalized)
            total_scale = total_scale * norm
        return type(self)(normalized_factors, dtype=self.dtype), total_scale

    def to_tt(self) -> TTTensor:
        """同じ mode 次元を持つ一般の `TTTensor` へ変換する。

        Returns:
            rank-1 core 列で構成した `TTTensor`。
        """
        return TTTensor(self._to_tensor_cores(), copy=False)

    def to_chain_tensor(self) -> TTChainTensor:
        """同じ mode 次元を持つ `TTChainTensor` へ変換する。

        Returns:
            rank-1 core 列で構成した `TTChainTensor`。
        """
        return TTChainTensor(self._to_tensor_cores(), copy=False)

    def _to_tensor_cores(self) -> list[TTTensorCore]:
        """factor 列を rank-1 TT core 列へ変換する。"""
        return [TTTensorCore.from_factor(factor) for factor in self._factors]

    def to_dense(self) -> np.ndarray:
        """dense tensor を返す。"""
        tensor = self._factors[0].copy()
        for factor in self._factors[1:]:
            tensor = np.multiply.outer(tensor, factor)
        return tensor

    def vectorize_l2r(self) -> np.ndarray:
        """左から右の Kronecker 順で dense vector に変換する。"""
        return self.to_dense().reshape(self.dense_size, order="C")

    def vectorize_r2l(self) -> np.ndarray:
        """右から左の Kronecker 順で dense vector に変換する。"""
        axes = tuple(range(self.ndim - 1, -1, -1))
        return self.to_dense().transpose(axes).reshape(self.dense_size, order="C")

    def get_element(self, idx: Sequence[int]) -> np.generic:
        """指定 index の tensor 要素を返す。

        Args:
            idx: ``[i1, ..., id]`` 形式の index。

        Returns:
            指定された tensor 要素。

        Raises:
            ValueError: index の個数または範囲が不正な場合。
        """
        if len(idx) != self.ndim:
            raise ValueError("The number of indices must match the dimension.")
        value = self.dtype.type(1.0)
        for d, (index, factor) in enumerate(zip(idx, self._factors, strict=True)):
            if index < 0 or index >= factor.size:
                raise ValueError(f"Index {d} is out of bounds.")
            value = value * factor[index]
        return value

    def as_scalar(self) -> np.generic:
        """全 mode の次元が 1 の場合に scalar として返す。

        Raises:
            ValueError: dense tensor が scalar でない場合。
        """
        if self.dense_size != 1:
            raise ValueError("mode_dims must all be 1.")
        return self.get_element([0] * self.ndim)

    def sum_entries(self) -> np.generic:
        """全要素の総和を返す。"""
        value = self.dtype.type(1.0)
        for factor in self._factors:
            value = value * np.sum(factor)
        return value

    def l1_norm_nonnegative(self) -> np.generic:
        """全要素が非負である仮定のもとで L1 norm を返す。"""
        return self.sum_entries()

    def l2_norm(self) -> float:
        """rank-1 tensor の Euclidean norm を返す。"""
        return self.tensor_norm()

    def save_npz(self, file: str) -> None:
        """rank-1 core 形式で factor を npz 形式に保存する。

        Args:
            file: 保存先ファイルパス。
        """
        np.savez_compressed(
            file,
            *[core.as_array(copy=False) for core in self._to_tensor_cores()],
        )

    @classmethod
    def load_npz(cls, file: str) -> Self:
        """npz ファイルから RankOneTensor を読み込む。

        Args:
            file: 読み込み元ファイルパス。

        Returns:
            読み込んだ RankOneTensor。

        Raises:
            ValueError: 保存された core が rank-1 tensor を表さない場合。
        """
        with np.load(file) as npz:
            cores = [npz[k] for k in npz.files]

        factors: list[np.ndarray] = []
        for core_index, core in enumerate(cores):
            if core.ndim == 1:
                factors.append(core)
                continue
            if core.ndim != 3:
                raise ValueError(f"Core {core_index} must be 1- or 3-dimensional.")
            if core.shape[0] != 1 or core.shape[2] != 1:
                raise ValueError("Saved cores do not represent a rank-1 tensor.")
            factors.append(core.reshape(core.shape[1]))
        return cls(factors)
