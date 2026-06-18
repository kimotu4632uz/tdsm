"""TT tensor 表現と基本演算を提供する。"""

import math
from collections.abc import Sequence
from typing import Self, cast

import numpy as np

from ddsm.utils.svd import trunc_svd

from ._base import BaseTT, _combine_max_rank
from ._tt_core import BaseTTCore, TTTensorCore
from ._tt_operator import TTChainOperator, TTOperator


class BaseTTTensor(BaseTT):
    """TT tensor の共通基底クラス。"""

    core_ndim = 3
    core_type = TTTensorCore
    display_name = "TT tensor"

    @property
    def cores(self) -> tuple[TTTensorCore, ...]:
        """TT tensor core の読み取り専用 tuple を返す。"""
        return tuple(cast(TTTensorCore, core) for core in self._cores)

    @property
    def is_operator(self) -> bool:
        """TT operator かどうかを返す。"""
        return False

    def __repr__(self) -> str:
        """TT tensor の概要を表す文字列を返す。"""
        return (
            f"[{self.display_name}]\n"
            f"ndim      = {self.ndim},\n"
            f"mode_dims = {self.mode_dims},\n"
            f"ranks     = {self.ranks}"
        )

    @property
    def mode_dims(self) -> tuple[int, ...]:
        """各 mode の次元を返す。"""
        return tuple(core.mode_dim for core in self.cores)

    @property
    def dense_size(self) -> int:
        """dense tensor 表現の要素数を返す。"""
        return int(self.ranks[0] * math.prod(self.mode_dims) * self.ranks[-1])

    def print_memory_size(self) -> None:
        """TT 表現と dense tensor 表現の要素数を表示する。"""
        print(f"TT memory size:     {self.memory_size}")
        print(f"dense memory size:  {self.dense_size}")

    def to_dense_chain(self) -> np.ndarray:
        """境界 rank を残した dense tensor を返す。

        Returns:
            形状 ``(r0, n1, ..., nd, rd)`` の dense tensor。
        """
        tensor = self.cores[0].as_array().copy()
        for core in self.cores[1:]:
            tensor = np.tensordot(tensor, core.as_array().copy(), axes=(-1, 0))
        return tensor

    def _vectorize_from_cores(
        self,
        cores: Sequence[BaseTTCore],
        mode_dims: Sequence[int],
        ranks: Sequence[int],
    ) -> np.ndarray:
        """指定 core 順序で左から右に縮約して行列化する。

        Args:
            cores: 縮約する TT core の列。
            mode_dims: ``cores`` に対応する mode 次元。
            ranks: ``cores`` に対応する TT rank。

        Returns:
            左境界 rank と mode index を行方向にまとめ、右境界 rank を
            列方向に残した 2 次元配列。
        """
        vector = np.asarray(cores[0]).reshape(
            ranks[0] * mode_dims[0],
            ranks[1],
            order="C",
        ).copy()
        for d in range(1, len(cores)):
            vector = np.tensordot(vector, cores[d], axes=(1, 0))
            vector = vector.reshape(
                ranks[0] * math.prod(mode_dims[: d + 1]),
                ranks[d + 1],
                order="C",
            )
        return vector

    def _vectorize_l2r_with_boundary(self) -> np.ndarray:
        """左から右の Kronecker 順で境界 rank 付き dense 行列に変換する。"""
        return self._vectorize_from_cores(self.cores, self.mode_dims, self.ranks)

    def _vectorize_r2l_with_boundary(self) -> np.ndarray:
        """右から左の Kronecker 順で境界 rank 付き dense 行列に変換する。"""
        cores = [core.reverse_ranks() for core in reversed(self.cores)]
        reversed_vector = self._vectorize_from_cores(
            cores,
            self.mode_dims[::-1],
            self.ranks[::-1],
        )
        reversed_dense = reversed_vector.reshape(
            self.ranks[-1],
            *self.mode_dims[::-1],
            self.ranks[0],
            order="C",
        )
        dense = reversed_dense.transpose(
            self.ndim + 1,
            *range(1, self.ndim + 1),
            0,
        )
        return dense.reshape(
            self.ranks[0] * math.prod(self.mode_dims),
            self.ranks[-1],
            order="C",
        )

    def global_svd(
        self,
        threshold: float | None = 0,
        max_rank: int | None = None,
        rank_list: list[int] | None = None,
    ) -> tuple[Self, np.ndarray, np.ndarray]:
        """TT tensor 全体を左直交 TT と末尾の SVD 因子へ分解する。

        Args:
            threshold: 相対特異値による打ち切り閾値。``None`` の場合は打ち切らない。
            max_rank: 各中間 rank の最大値。
            rank_list: core ごとの rank 上限。

        Returns:
            ``(左直交化した TT tensor, 特異値, 右特異ベクトル行列)``。
        """
        ortho = self.left_ortho(threshold=threshold, max_rank=max_rank, rank_list=rank_list)
        last_core = ortho.cores[-1]
        u, s, v = trunc_svd(
            last_core.left_unfold(),
            criterion="relative",
            threshold=threshold,
            max_rank=_combine_max_rank(
                max_rank,
                rank_list[-1] if rank_list else None,
            ),
        )
        ortho.update_cores([(-1, last_core.from_left_unfold(u))])
        return ortho, s, v

    def left_qr(self) -> tuple[Self, np.ndarray]:
        """TT tensor 全体を左直交 TT と右端の QR 因子へ分解する。

        Returns:
            ``(左直交化した TT tensor, R 行列)``。
        """
        ortho = self.left_ortho(method='qr')
        last_core = ortho.cores[-1]
        q, r = np.linalg.qr(
            last_core.left_unfold(),
            mode="reduced",
        )
        ortho.update_cores([(-1, last_core.from_left_unfold(q))])
        return ortho, r


class TTChainTensor(BaseTTTensor):
    """open boundary の TT tensor。"""

    display_name = "TT chain tensor"

    def _validate_boundary_cores(self, cores: Sequence[BaseTTCore]) -> None:
        """境界 rank 条件を検証する。"""
        return None

    @property
    def is_open_boundary(self) -> bool:
        """open boundary 型の TT tensor かどうかを返す。"""
        return True

    def as_operator_rows(self) -> TTChainOperator:
        """各 mode を row 側に置いた TT operator chain に変換する。

        Returns:
            row 次元が ``mode_dims``、column 次元がすべて 1 の operator chain。
        """
        return TTChainOperator([core.as_operator_row_core() for core in self.cores])

    def as_operator_cols(self) -> TTChainOperator:
        """各 mode を column 側に置いた TT operator chain に変換する。

        Returns:
            row 次元がすべて 1、column 次元が ``mode_dims`` の operator chain。
        """
        return TTChainOperator([core.as_operator_col_core() for core in self.cores])

    def vectorize_l2r(self) -> np.ndarray:
        """左から右に縮約して境界 rank 付き dense 行列に変換する。

        Returns:
            mode index を ``(i1, ..., id)`` の順に並べた、形状
            ``(r0 * prod(mode_dims), rd)`` の 2 次元配列。
        """
        return self._vectorize_l2r_with_boundary()

    def vectorize_r2l(self) -> np.ndarray:
        """右から左の mode 順で境界 rank 付き dense 行列に変換する。

        Returns:
            mode index を ``(id, ..., i1)`` の順に並べた、形状
            ``(r0 * prod(mode_dims), rd)`` の 2 次元配列。
        """
        return self._vectorize_r2l_with_boundary()


class TTTensor(BaseTTTensor):
    """境界 rank が 1 の TT tensor。"""

    display_name = "TT tensor"

    def _validate_boundary_cores(self, cores: Sequence[BaseTTCore]) -> None:
        """境界 rank が 1 であることを検証する。"""
        if cores[0].left_rank != 1 or cores[-1].right_rank != 1:
            raise ValueError("The first and final TT ranks must be 1.")

    def __init__(self, cores: Sequence[np.ndarray | BaseTTCore], copy: bool = True) -> None:
        """TT tensor を初期化する。

        Args:
            cores: 3 次元 TT core の列。
            copy: True の場合、core をコピーして保持する。

        Raises:
            ValueError: 境界 rank が 1 でない場合。
        """
        super().__init__(cores, copy=copy)
        if self.ranks[0] != 1 or self.ranks[-1] != 1:
            raise ValueError("The first and final TT ranks must be 1.")

    @property
    def is_open_boundary(self) -> bool:
        """open boundary 型の TT tensor かどうかを返す。"""
        return False

    def to_dense(self) -> np.ndarray:
        """dense tensor を返す。"""
        return self.to_dense_chain().reshape(self.mode_dims)

    def as_operator_rows(self) -> TTOperator:
        """各 mode を row 側に置いた TT operator に変換する。

        Returns:
            row 次元が ``mode_dims``、column 次元がすべて 1 の TT operator。
        """
        return TTOperator([core.as_operator_row_core() for core in self.cores])

    def as_operator_cols(self) -> TTOperator:
        """各 mode を column 側に置いた TT operator に変換する。

        Returns:
            row 次元がすべて 1、column 次元が ``mode_dims`` の TT operator。
        """
        return TTOperator([core.as_operator_col_core() for core in self.cores])

    def vectorize_r2l(self) -> np.ndarray:
        """右から左に縮約して dense vector に変換する。

        Returns:
            mode index を ``(id, ..., i1)`` の順に並べた 1 次元配列。
        """
        return self._vectorize_r2l_with_boundary().reshape(math.prod(self.mode_dims))

    def vectorize_l2r(self) -> np.ndarray:
        """左から右に縮約して dense vector に変換する。

        Returns:
            mode index を ``(i1, ..., id)`` の順に並べた 1 次元配列。
        """
        return self._vectorize_l2r_with_boundary().reshape(math.prod(self.mode_dims))

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
        for d, (index, size) in enumerate(zip(idx, self.mode_dims, strict=True)):
            if index < 0 or index >= size:
                raise ValueError(f"Index {d} is out of bounds.")

        comp = self.cores[0][:, idx[0], :].reshape(1, self.ranks[1])
        for d in range(1, self.ndim):
            comp = comp.dot(self.cores[d][:, idx[d], :].reshape(self.ranks[d], self.ranks[d + 1]))
        return comp[0, 0]

    def as_scalar(self) -> np.generic:
        """全 mode の次元が 1 の場合に scalar として返す。

        Raises:
            ValueError: dense tensor が scalar でない場合。
        """
        if math.prod(self.mode_dims) != 1:
            raise ValueError("mode_dims must all be 1.")
        return self.get_element([0] * self.ndim)

    def sum_entries(self) -> np.generic:
        """全要素の総和を返す。"""
        comp = np.sum(self.cores[0], axis=1).reshape(1, self.ranks[1])
        for d in range(1, self.ndim):
            comp = comp.dot(
                np.sum(self.cores[d], axis=1).reshape(
                    self.ranks[d],
                    self.ranks[d + 1],
                )
            )
        return comp[0, 0]

    def l1_norm_nonnegative(self) -> np.generic:
        """全要素が非負である仮定のもとで L1 norm を返す。"""
        return self.sum_entries()

    def l2_norm(self) -> float:
        """TT tensor の Euclidean norm を返す。"""
        tt = self.right_ortho()
        return float(np.linalg.norm(tt.cores[0]))


def filled_tensor(
    val: int | float | complex,
    mode_dims: Sequence[int],
    ranks: int | Sequence[int] = 1,
) -> TTTensor:
    """指定値で埋めた TT tensor を返す。

    Args:
        val: core の各要素に入れる値。
        mode_dims: 各 mode の次元。
        ranks: TT rank。整数の場合は全中間 rank に同じ値を使う。

    Returns:
        指定値で埋めた TT tensor。
    """
    if not isinstance(ranks, Sequence):
        rank_list = [1] + [int(ranks) for _ in range(len(mode_dims) - 1)] + [1]
    else:
        rank_list = list(ranks)
    cores = [
        val * np.ones((rank_list[d], mode_dims[d], rank_list[d + 1]))
        for d in range(len(mode_dims))
    ]
    return TTTensor(cores)
