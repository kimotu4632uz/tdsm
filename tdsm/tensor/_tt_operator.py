"""TT operator 表現と基本演算を提供する。"""

import math
from collections.abc import Sequence
from typing import Self, cast

import numpy as np

from ._base import BaseTT
from ._tt_core import BaseTTCore, TTOperatorCore


class BaseTTOperator(BaseTT):
    """TT operator の共通基底クラス。"""

    core_ndim = 4
    core_type = TTOperatorCore
    display_name = "TT operator"

    @property
    def cores(self) -> tuple[TTOperatorCore, ...]:
        """TT operator core の読み取り専用 tuple を返す。"""
        return tuple(cast(TTOperatorCore, core) for core in self._cores)

    @property
    def mode_dims(self) -> tuple[tuple[int, int], ...]:
        """各 mode の row/column 次元を返す。"""
        return tuple((core.row_dim, core.col_dim) for core in self.cores)

    @property
    def row_dims(self) -> tuple[int, ...]:
        """各 row mode の次元を返す。"""
        return tuple(core.row_dim for core in self.cores)

    @property
    def col_dims(self) -> tuple[int, ...]:
        """各 column mode の次元を返す。"""
        return tuple(core.col_dim for core in self.cores)

    @property
    def dense_size(self) -> int:
        """dense matrix 表現の要素数を返す。"""
        return int(self.ranks[0] * math.prod(self.row_dims) * math.prod(self.col_dims) * self.ranks[-1])

    def __repr__(self) -> str:
        """TT operator の概要を表す文字列を返す。"""
        return (
            f"[{self.display_name}]\n"
            f"ndim     = {self.ndim},\n"
            f"row_dims = {self.row_dims},\n"
            f"col_dims = {self.col_dims},\n"
            f"ranks    = {self.ranks}"
        )

    def print_memory_size(self) -> None:
        """TT 表現と matrix 表現の要素数を表示する。"""
        print(f"TT memory size:     {self.memory_size}")
        print(f"matrix memory size: {math.prod(self.row_dims) * math.prod(self.col_dims)}")

    @property
    def is_operator(self) -> bool:
        """TT operator かどうかを返す。"""
        return True

    def transpose(self) -> Self:
        """転置した TT operator を返す。"""
        cores = [core.transpose_modes() for core in self.cores]
        return self._new_like(cores)

    def adjoint(self) -> Self:
        """随伴 TT operator を返す。"""
        cores = [core.adjoint_modes() for core in self.cores]
        return self._new_like(cores)


class TTChainOperator(BaseTTOperator):
    """open boundary の TT operator。"""

    display_name = "TT chain operator"

    def _validate_boundary_cores(self, cores: Sequence[BaseTTCore]) -> None:
        """境界 rank 条件を検証する。"""
        return None

    @property
    def is_open_boundary(self) -> bool:
        """open boundary 型の TT operator かどうかを返す。"""
        return True


class TTOperator(BaseTTOperator):
    """境界 rank が 1 の TT operator。"""

    display_name = "TT operator"

    def _validate_boundary_cores(self, cores: Sequence[BaseTTCore]) -> None:
        """境界 rank が 1 であることを検証する。"""
        if cores[0].left_rank != 1:
            raise ValueError("The first TT rank must be 1.")
        if cores[-1].right_rank != 1:
            raise ValueError("The final TT rank must be 1.")

    def __init__(self, cores: Sequence[np.ndarray | BaseTTCore], copy: bool = True) -> None:
        """TT operator を初期化する。

        Args:
            cores: 4 次元 TT core の列。
            copy: True の場合、core をコピーして保持する。

        Raises:
            ValueError: 境界 rank が 1 でない場合。
        """
        super().__init__(cores, copy=copy)
        if self.ranks[0] != 1:
            raise ValueError("The first TT rank must be 1.")
        if self.ranks[-1] != 1:
            raise ValueError("The final TT rank must be 1.")

    @property
    def is_open_boundary(self) -> bool:
        """open boundary 型の TT operator かどうかを返す。"""
        return False

    def _matricize_from_cores(
        self,
        cores: Sequence[np.ndarray | BaseTTCore],
        row_dims: Sequence[int],
        col_dims: Sequence[int],
        ranks: Sequence[int],
    ) -> np.ndarray:
        """指定 core 順序で左から右に縮約して行列化する。

        Args:
            cores: 縮約する TT operator core の列。
            row_dims: ``cores`` に対応する row mode 次元。
            col_dims: ``cores`` に対応する column mode 次元。
            ranks: ``cores`` に対応する TT rank。

        Returns:
            行列化した 2 次元配列。
        """
        mat = np.asarray(cores[0]).reshape(row_dims[0], col_dims[0], ranks[1]).copy()
        for d in range(1, self.ndim):
            mat = np.tensordot(mat, cores[d], axes=(2, 0))
            mat = mat.transpose([0, 2, 1, 3, 4]).reshape(
                (
                    math.prod(row_dims[: d + 1]),
                    math.prod(col_dims[: d + 1]),
                    ranks[d + 1],
                )
            )

        row_dim = math.prod(row_dims)
        col_dim = math.prod(col_dims)
        return mat.reshape(row_dim, col_dim)

    def matricize_r2l(self) -> np.ndarray:
        """右から左に縮約して dense vector または dense matrix に変換する。

        Returns:
            row/column mode を ``(id, ..., i1)`` / ``(jd, ..., j1)`` の順に
            並べた dense vector または dense matrix。
        """
        cores = [core.reverse_ranks() for core in reversed(self.cores)]
        return self._matricize_from_cores(
            cores,
            self.row_dims[::-1],
            self.col_dims[::-1],
            self.ranks[::-1],
        )

    def matricize_l2r(self) -> np.ndarray:
        """左から右に縮約して dense vector または dense matrix に変換する。

        Returns:
            row/column mode を ``(i1, ..., id)`` / ``(j1, ..., jd)`` の順に
            並べた dense vector または dense matrix。
        """
        return self._matricize_from_cores(
            self.cores,
            self.row_dims,
            self.col_dims,
            self.ranks,
        )

    def get_element(self, idx: Sequence[int]) -> np.generic:
        """指定 index の operator 要素を返す。

        Args:
            idx: ``[x1, ..., xd, y1, ..., yd]`` 形式の index。

        Returns:
            指定された要素。

        Raises:
            ValueError: index の個数または範囲が不正な場合。
        """
        if len(idx) != (2 * self.ndim):
            raise ValueError("The number of indices must be twice the dimension.")
        for d in range(self.ndim):
            row_index = idx[d]
            col_index = idx[self.ndim + d]
            if row_index < 0 or row_index >= self.row_dims[d]:
                raise ValueError(f"Row index {d} is out of bounds.")
            if col_index < 0 or col_index >= self.col_dims[d]:
                raise ValueError(f"Column index {d} is out of bounds.")

        comp = np.squeeze(self.cores[0][:, idx[0], idx[self.ndim], :]).reshape(1, self.ranks[1])
        for d in range(1, self.ndim):
            comp = comp.dot(
                np.squeeze(self.cores[d][:, idx[d], idx[self.ndim + d], :]).reshape(
                    self.ranks[d],
                    self.ranks[d + 1],
                )
            )
        return comp[0, 0]

    def as_scalar(self) -> np.generic:
        """全 row/column 次元が 1 の場合に scalar として返す。

        Raises:
            ValueError: dense operator が scalar でない場合。
        """
        if np.prod(self.row_dims) == 1 and np.prod(self.col_dims) == 1:
            return self.get_element([0] * (2 * self.ndim))
        raise ValueError("row_dims and col_dims must all be 1.")


def filled_operator(
    val: int | float | complex,
    row_dims: Sequence[int],
    col_dims: Sequence[int],
    ranks: int | Sequence[int] = 1,
) -> TTOperator:
    """指定値で埋めた TT operator を返す。"""
    if len(row_dims) != len(col_dims):
        raise ValueError("row_dims and col_dims must have the same length.")
    if not isinstance(ranks, Sequence):
        rank_list = [1] + [int(ranks) for _ in range(len(row_dims) - 1)] + [1]
    else:
        rank_list = list(ranks)
    cores = [
        val * np.ones((rank_list[d], row_dims[d], col_dims[d], rank_list[d + 1]))
        for d in range(len(row_dims))
    ]
    return TTOperator(cores)


def eye(dim: Sequence[int]) -> TTOperator:
    """恒等 TT operator を返す。"""
    cores = [np.zeros((1, dim[d], dim[d], 1)) for d in range(len(dim))]
    for d, size in enumerate(dim):
        cores[d][0, :, :, 0] = np.eye(size)
    return TTOperator(cores)
