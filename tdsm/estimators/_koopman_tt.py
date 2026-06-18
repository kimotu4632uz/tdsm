"""TT-EDMD の推定結果として使う Koopman 作用素表現を提供する。"""

from collections.abc import Sequence
from typing import Literal, Self

import numpy as np

from ..tenalg import inner_product
from ..tensor import RankOneTensor, TTChainTensor, TTTensor


class TTKoopmanOperator:
    """2 つの TT chain で表す Koopman 作用素表現。

    `TTEDMD` が推定した作用素を、出力側の `left_chain` と入力側の
    `right_chain` の組として保持します。両方とも左端 rank が 1、
    右端 rank が共通の Koopman rank になる向きで保持します。
    """

    left_chain: TTChainTensor
    right_chain: TTChainTensor
    display_name = "TT Koopman operator"

    def __init__(
        self,
        *,
        left_chain: TTChainTensor,
        right_chain: TTChainTensor,
    ) -> None:
        """Koopman 作用素表現を初期化する。

        Parameters
        ----------
        left_chain : TTChainTensor
            出力側の TT chain。左端 rank は 1。
        right_chain : TTChainTensor
            入力側の TT chain。左端 rank は 1。

        Raises
        ------
        ValueError
            境界 rank または Koopman rank が不整合な場合。
        """
        if left_chain.ranks[0] != 1:
            raise ValueError("left_chain must have left boundary rank 1.")
        if right_chain.ranks[0] != 1:
            raise ValueError("right_chain must have left boundary rank 1.")
        if left_chain.ranks[-1] != right_chain.ranks[-1]:
            raise ValueError("left_chain and right_chain must have the same right boundary rank.")

        self.left_chain = left_chain
        self.right_chain = right_chain

    def __repr__(self) -> str:
        """TT Koopman operator の概要を表す文字列を返す。"""
        return (
            f"[{self.display_name}]\n"
            f"row_ndim    = {self.left_chain.ndim},\n"
            f"row_dims    = {self.row_dims},\n"
            f"left_ranks  = {self.left_chain.ranks},\n\n"
            f"mid_rank    = {self.mid_rank},\n\n"
            f"col_ndim    = {self.right_chain.ndim},\n"
            f"col_dims    = {self.col_dims},\n"
            f"right_ranks = {self.right_chain.ranks}"
        )

    @property
    def ndim(self) -> int:
        """TT core の個数を返す。"""
        return self.left_chain.ndim + self.right_chain.ndim

    @property
    def row_dims(self) -> tuple[int, ...]:
        """出力側の mode 次元を返す。"""
        return self.left_chain.mode_dims

    @property
    def col_dims(self) -> tuple[int, ...]:
        """入力側の mode 次元を返す。"""
        return self.right_chain.mode_dims

    @property
    def mid_rank(self) -> int:
        """左右 chain を結ぶ中間 rank を返す。"""
        return self.left_chain.ranks[-1]

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """作用素の dtype を返す。"""
        return np.dtype(np.result_type(self.left_chain.dtype, self.right_chain.dtype))

    def left_ortho(
        self,
        *,
        threshold: float | None = 0,
        max_rank: int | None = None,
        rank_list: list[int] | None = None,
        method: Literal["svd", "qr"] = "svd",
    ) -> Self:
        """左直交化した TT Koopman operator を返す。

        `left_chain` を左から因子分解し、接続部で得られる因子を
        `right_chain` の右端 rank 側へ吸収します。

        Parameters
        ----------
        threshold : float or None, default 0
            相対特異値による打ち切り閾値。``None`` の場合は打ち切らない。
        max_rank : int or None, optional
            各中間 rank の最大値。
        rank_list : list[int] or None, optional
            合成済み TT operator の中間 rank ごとの上限。
        method : {"svd", "qr"}, default "svd"
            局所分解に使う手法。

        Returns
        -------
        Self
            左直交化した TT Koopman operator。
        """
        left_rank_list, right_rank_list = self._split_left_ortho_rank_list(rank_list)
        left_chain, factor = _left_factorize_chain(
            self.left_chain,
            threshold=threshold,
            max_rank=max_rank,
            rank_list=left_rank_list,
            method=method,
        )

        right_cores = list(self.right_chain.cores)
        right_cores[-1] = right_cores[-1].apply_right_factor(factor.T)
        right_chain = TTChainTensor(right_cores).right_ortho(
            threshold=threshold,
            max_rank=max_rank,
            rank_list=right_rank_list,
            method=method,
        )
        return type(self)(left_chain=left_chain, right_chain=right_chain)

    def right_ortho(
        self,
        *,
        threshold: float | None = 0,
        max_rank: int | None = None,
        rank_list: list[int] | None = None,
        method: Literal["svd", "qr"] = "svd",
    ) -> Self:
        """右直交化した TT Koopman operator を返す。

        `right_chain` を左から因子分解し、接続部で得られる因子を
        `left_chain` の右端 rank 側へ吸収します。

        Parameters
        ----------
        threshold : float or None, default 0
            相対特異値による打ち切り閾値。``None`` の場合は打ち切らない。
        max_rank : int or None, optional
            各中間 rank の最大値。
        rank_list : list[int] or None, optional
            合成済み TT operator の中間 rank ごとの上限。
        method : {"svd", "qr"}, default "svd"
            局所分解に使う手法。

        Returns
        -------
        Self
            右直交化した TT Koopman operator。
        """
        left_rank_list, right_rank_list = self._split_right_ortho_rank_list(rank_list)
        right_chain, factor = _left_factorize_chain(
            self.right_chain,
            threshold=threshold,
            max_rank=max_rank,
            rank_list=right_rank_list,
            method=method,
        )

        left_cores = list(self.left_chain.cores)
        left_cores[-1] = left_cores[-1].apply_right_factor(factor.T)
        left_chain = TTChainTensor(left_cores).right_ortho(
            threshold=threshold,
            max_rank=max_rank,
            rank_list=left_rank_list,
            method=method,
        )
        return type(self)(left_chain=left_chain, right_chain=right_chain)

    def _split_left_ortho_rank_list(
        self,
        rank_list: list[int] | None,
    ) -> tuple[list[int] | None, list[int] | None]:
        """左直交化用に合成 operator の rank 上限を左右 chain へ分配する。"""
        if rank_list is None:
            return None, None

        ranks = self._validate_operator_rank_list(rank_list)
        left_ndim = self.left_chain.ndim
        right_ndim = self.right_chain.ndim
        left_rank_list = ranks[:left_ndim]
        right_rank_list = [
            ranks[left_ndim + right_ndim - 2 - index]
            for index in range(right_ndim - 1)
        ]
        return left_rank_list, right_rank_list

    def _split_right_ortho_rank_list(
        self,
        rank_list: list[int] | None,
    ) -> tuple[list[int] | None, list[int] | None]:
        """右直交化用に合成 operator の rank 上限を左右 chain へ分配する。"""
        if rank_list is None:
            return None, None

        ranks = self._validate_operator_rank_list(rank_list)
        left_ndim = self.left_chain.ndim
        right_ndim = self.right_chain.ndim
        left_rank_list = ranks[: left_ndim - 1]
        right_rank_list = [
            ranks[left_ndim + right_ndim - 2 - index]
            for index in range(right_ndim - 1)
        ]
        right_rank_list.append(ranks[left_ndim - 1])
        return left_rank_list, right_rank_list

    def _validate_operator_rank_list(self, rank_list: list[int]) -> list[int]:
        """合成済み TT operator の中間 rank 上限列を検証する。"""
        expected_size = self.left_chain.ndim + self.right_chain.ndim - 1
        if len(rank_list) != expected_size:
            raise ValueError(
                "rank_list must have one entry for each intermediate rank of "
                "the composed TT operator."
            )
        return list(rank_list)

    def apply_rank_one(self, tensor: RankOneTensor) -> TTTensor:
        """rank-one 入力へ Koopman 作用素表現を作用させる。

        Parameters
        ----------
        tensor : RankOneTensor
            入力側辞書で lift 済みの rank-one tensor。

        Returns
        -------
        TTTensor
            出力側のリフト状態を表す `TTTensor`。
        """
        coeff = inner_product(self.right_chain, tensor)[0, 0, :, 0]
        dtype = np.dtype(np.result_type(self.left_chain.dtype, coeff.dtype))
        last_core = np.tensordot(
            np.asarray(self.left_chain.cores[-1], dtype=dtype),
            np.asarray(coeff, dtype=dtype),
            axes=(-1, 0),
        )[:, :, np.newaxis]
        return TTTensor([*self.left_chain.cores[:-1], last_core])

    def pick_dx(self, *, real: bool = True) -> np.ndarray:
        """1 階の出力 observable に対応する入力側係数を取り出す。

        出力側 mode index が ``(0, ..., 1, ..., 0)`` となる行を
        `matricize_r2l()` と同じ column 順序で取り出します。返す行列の
        第 `i` 行が `dx_i` に対応します。

        Parameters
        ----------
        real : bool, default True
            True の場合、実部だけを返す。

        Returns
        -------
        np.ndarray
            形状 ``(prod(col_dims), row_ndim)`` の係数行列。

        Raises
        ------
        ValueError
            出力側 mode 次元に 1 階成分が存在しない場合。
        """
        for mode_index, mode_dim in enumerate(self.row_dims):
            if mode_dim <= 1:
                raise ValueError(
                    f"row mode {mode_index} must have at least 2 entries "
                    "to pick the first-order observable."
                )

        right = self.right_chain.vectorize_r2l()
        derivatives = np.empty(
            (self.left_chain.ndim, right.shape[0]),
            dtype=self.dtype,
        )
        for deriv_index in range(self.left_chain.ndim):
            left_rank_vector = self._left_rank_vector_for_first_order(deriv_index)
            derivatives[deriv_index, :] = right.dot(left_rank_vector)

        if real:
            return derivatives.real
        return derivatives

    def _left_rank_vector_for_first_order(self, deriv_index: int) -> np.ndarray:
        """指定した 1 階出力 observable の右端 rank ベクトルを返す。"""
        comp = self.left_chain.cores[0][
            :,
            1 if deriv_index == 0 else 0,
            :,
        ].reshape(1, self.left_chain.ranks[1])
        for core_index in range(1, self.left_chain.ndim):
            mode_index = 1 if deriv_index == core_index else 0
            comp = comp.dot(
                self.left_chain.cores[core_index][
                    :,
                    mode_index,
                    :,
                ].reshape(
                    self.left_chain.ranks[core_index],
                    self.left_chain.ranks[core_index + 1],
                )
            )
        return comp.reshape(self.mid_rank)

    def get_element(self, idx: Sequence[int]) -> np.generic:
        """指定 index の Koopman operator 要素を返す。

        Parameters
        ----------
        idx : Sequence[int]
            ``[x1, ..., xd, y1, ..., ye]`` 形式の index。
            前半は出力側 `left_chain`、後半は入力側 `right_chain` の
            自然な mode 順序に対応する。

        Returns
        -------
        np.generic
            指定された要素。

        Raises
        ------
        ValueError
            index の個数または範囲が不正な場合。
        """
        row_ndim = self.left_chain.ndim
        col_ndim = self.right_chain.ndim
        if len(idx) != row_ndim + col_ndim:
            raise ValueError("The number of indices must match row_ndim + col_ndim.")

        row_idx = tuple(idx[:row_ndim])
        col_idx = tuple(idx[row_ndim:])
        self._validate_mode_indices(row_idx, self.row_dims, "Row")
        self._validate_mode_indices(col_idx, self.col_dims, "Column")

        left = self._rank_vector_for_indices(self.left_chain, row_idx)
        right = self._rank_vector_for_indices(self.right_chain, col_idx)
        return np.dot(left, right)

    def _validate_mode_indices(
        self,
        idx: Sequence[int],
        mode_dims: Sequence[int],
        label: str,
    ) -> None:
        """mode index の範囲を検証する。"""
        for mode_index, (index, size) in enumerate(zip(idx, mode_dims, strict=True)):
            if index < 0 or index >= size:
                raise ValueError(f"{label} index {mode_index} is out of bounds.")

    def _rank_vector_for_indices(
        self,
        chain: TTChainTensor,
        idx: Sequence[int],
    ) -> np.ndarray:
        """指定 index で TT chain を縮約し、右端 rank ベクトルを返す。"""
        comp = chain.cores[0][:, idx[0], :].reshape(1, chain.ranks[1])
        for core_index in range(1, chain.ndim):
            comp = comp.dot(
                chain.cores[core_index][:, idx[core_index], :].reshape(
                    chain.ranks[core_index],
                    chain.ranks[core_index + 1],
                )
            )
        return comp.reshape(chain.ranks[-1])

    def matricize_l2r(self) -> np.ndarray:
        """自然な mode 順序で dense Koopman 行列表現を返す。

        Returns
        -------
        np.ndarray
            row/column mode をそれぞれ ``(i1, ..., id)`` /
            ``(j1, ..., je)`` の順に並べた dense 行列。
        """
        left = self.left_chain.vectorize_l2r()
        right = self.right_chain.vectorize_l2r()
        return left @ right.T

    def matricize_r2l(self) -> np.ndarray:
        """自然な mode 順序で dense Koopman 行列表現を返す。

        Returns
        -------
        np.ndarray
            row/column mode をそれぞれ ``(id, ..., i1)`` /
            ``(je, ..., j1)`` の順に並べた dense 行列。
        """
        left = self.left_chain.vectorize_r2l()
        right = self.right_chain.vectorize_r2l()
        return left @ right.T


def _left_factorize_chain(
    chain: TTChainTensor,
    *,
    threshold: float | None,
    max_rank: int | None,
    rank_list: list[int] | None,
    method: Literal["svd", "qr"],
) -> tuple[TTChainTensor, np.ndarray]:
    """TT chain を左直交 chain と右端因子へ分解する。"""
    if method == "svd":
        ortho, singular_values, right_vectors = chain.global_svd(
            threshold=threshold,
            max_rank=max_rank,
            rank_list=rank_list,
        )
        return ortho, np.diag(singular_values).dot(right_vectors)

    if method == "qr":
        return chain.left_qr()

    raise ValueError("method must be 'svd' or 'qr'.")
