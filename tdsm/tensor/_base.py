"""TT 表現の共通処理を提供する基底クラス群。"""

from __future__ import annotations

import math
from abc import ABCMeta, abstractmethod
from numbers import Integral
from types import NotImplementedType
from typing import ClassVar, Literal, Self, Sequence

import numpy as np

from ddsm.utils.svd import trunc_svd

from ._tt_core import BaseTTCore

Scalar = int | float | complex | np.generic
CoreUpdate = tuple[int, np.ndarray | BaseTTCore]


class BaseTT(metaclass=ABCMeta):
    """TT 表現の共通基底クラス。"""

    core_ndim: ClassVar[int]
    core_type: ClassVar[type[BaseTTCore]]
    display_name: ClassVar[str] = "TT"

    _cores: list[BaseTTCore]
    _ndim: int
    _ranks: list[int]
    _dtype: np.dtype[np.generic]

    def __init__(
        self,
        cores: Sequence[np.ndarray | BaseTTCore],
        copy: bool = True,
    ) -> None:
        """TT 表現を初期化する。

        Args:
            cores: TT core の列。
            copy: True の場合、core をコピーして保持する。

        Raises:
            ValueError: core が空、次元数が不正、または rank が接続しない場合。
        """
        if len(cores) == 0:
            raise ValueError("cores must contain at least one core.")

        wrapped_cores = [
            self._as_core_type(core, copy=copy)
            for core in cores
        ]
        self._cores = wrapped_cores
        self._validate_cores()
        self._refresh_metadata()

    @abstractmethod
    def __repr__(self) -> str:
        """TT 表現の概要を表す文字列を返す。"""

    @property
    def ndim(self) -> int:
        """TT core の個数を返す。"""
        return self._ndim

    @property
    @abstractmethod
    def cores(self) -> tuple[BaseTTCore, ...]:
        """TT core の読み取り専用 tuple を返す。"""

    @property
    def ranks(self) -> tuple[int, ...]:
        """TT rank 列を返す。"""
        return tuple(self._ranks)

    @property
    def dtype(self) -> np.dtype[np.generic]:
        """core の dtype を返す。"""
        return self._dtype

    @property
    def memory_size(self) -> int:
        """TT 表現の要素数を返す。"""
        return sum(math.prod(core.shape) for core in self._cores)

    @property
    @abstractmethod
    def is_operator(self) -> bool:
        """TT operator かどうかを返す。"""

    @property
    @abstractmethod
    def is_open_boundary(self) -> bool:
        """open boundary 型の TT operator かどうかを返す。"""

    def _as_core_type(
        self,
        core: np.ndarray | BaseTTCore,
        copy: bool,
    ) -> BaseTTCore:
        """入力 core を現在の TT 表現用 core 型へ正規化する。"""
        if isinstance(core, self.core_type):
            return core.copy() if copy else core
        return self.core_type(core, copy=copy)

    def _validate_cores(self, cores: Sequence[BaseTTCore] | None = None) -> None:
        """core の次元数と rank 接続を検証する。"""
        core_sequence = self._cores if cores is None else cores
        for core in core_sequence:
            if core.ndim != self.core_ndim:
                raise ValueError(f"Each core must be {self.core_ndim}-dimensional.")

        for left_core, right_core in zip(core_sequence[:-1], core_sequence[1:], strict=True):
            if left_core.right_rank != right_core.left_rank:
                raise ValueError("TT ranks do not match.")
        self._validate_boundary_cores(core_sequence)

    @abstractmethod
    def _validate_boundary_cores(self, cores: Sequence[BaseTTCore]) -> None:
        """具象クラス固有の境界 rank 条件を検証する。"""

    def _refresh_metadata(self) -> None:
        """core からメタデータを更新する。"""
        self._ndim = len(self._cores)
        self._ranks = [core.left_rank for core in self._cores] + [self._cores[-1].right_rank]
        self._dtype = np.dtype(np.result_type(*[core.dtype for core in self._cores]))

    def _new_like(self, cores: Sequence[np.ndarray | BaseTTCore]) -> Self:
        """同じ具象型の TT 表現を生成する。"""
        return type(self)(cores)

    def update_cores(self, updates: Sequence[CoreUpdate]) -> None:
        """複数の core をまとめて更新する。

        更新候補全体の rank 接続を検証してから反映します。

        Args:
            updates: ``(core_index, core)`` の列。

        Raises:
            IndexError: core index が範囲外の場合。
            ValueError: 更新後の core 列が TT 表現として不正な場合。
        """
        new_cores = list(self._cores)
        for index, core in updates:
            new_cores[index] = self._as_core_type(core, copy=False)

        self._validate_cores(new_cores)
        self._cores = new_cores
        self._refresh_metadata()

    def copy(self) -> Self:
        """自身の deep copy を返す。"""
        return self._new_like([core.copy() for core in self.cores])

    def reversed(self) -> Self:
        """core の順序と rank 接続の向きを反転した TT 表現を返す。

        Returns:
            mode 次元の順序を逆にし、左右 rank 軸を入れ替えた TT 表現。
        """
        return self._new_like([core.reverse_ranks() for core in reversed(self.cores)])

    def astype(self, dtype: np.dtype[np.generic]) -> Self:
        """core の dtype を変換した copy を返す。

        Args:
            dtype: 変換後の dtype。

        Returns:
            dtype を変換した TT 表現。
        """
        return self._new_like([np.asarray(core, dtype=dtype).copy() for core in self.cores])

    def _multiply_scalar(self, scalar: Scalar) -> Self | NotImplementedType:
        """scalar 倍した TT 表現を返す内部 helper。"""
        if not np.isscalar(scalar):
            return NotImplemented
        cores = [core.copy() for core in self.cores]
        cores[0] = cores[0].scale(scalar)
        return self._new_like(cores)

    def __mul__(self, scalar: Scalar) -> Self | NotImplementedType:
        """scalar 倍した TT 表現を返す。"""
        return self._multiply_scalar(scalar)

    def __rmul__(self, scalar: Scalar) -> Self | NotImplementedType:
        """scalar 倍した TT 表現を返す。"""
        return self._multiply_scalar(scalar)

    def left_ortho(
        self,
        *,
        threshold: float | None = 0,
        max_rank: int | None = None,
        rank_list: list[int] | None = None,
        method: Literal['svd', 'qr'] = 'svd',
        start_idx: int = 0,
        end_idx: int | None = None,
    ) -> Self:
        """左直交化した TT 表現を返す。

        Args:
            threshold: 相対特異値による打ち切り閾値。``None`` の場合は打ち切らない。
            max_rank: 各中間 rank の最大値。
            rank_list: core ごとの rank 上限。
            method: 局所分解に使う手法。
            start_idx: 左直交化を開始する core index。
            end_idx: 左直交化する最後の core index。省略時は最後から 2 番目。

        Returns:
            左直交化した TT 表現。
        """
        if not isinstance(start_idx, Integral):
            raise TypeError("start_idx must be an integer.")
        if method not in {"svd", "qr"}:
            raise ValueError("method must be 'svd' or 'qr'.")
        start_idx = int(start_idx)

        if self.ndim == 1 and start_idx == 0 and end_idx is None:
            return self.copy()

        if end_idx is None:
            end_idx = self.ndim - 2
        if not isinstance(end_idx, Integral):
            raise TypeError("end_idx must be an integer.")
        end_idx = int(end_idx)
        if not 0 <= start_idx <= end_idx < self.ndim - 1:
            raise ValueError(
                "start_idx and end_idx must satisfy "
                "0 <= start_idx <= end_idx < dim - 1."
            )

        tt = self.copy()
        for d in range(start_idx, end_idx + 1):
            current_core = tt.cores[d]
            if method == 'svd':
                u, s, v = trunc_svd(
                    current_core.left_unfold(),
                    criterion="relative",
                    threshold=threshold,
                    max_rank=_combine_max_rank(
                        max_rank,
                        rank_list[d] if rank_list else None,
                    ),
                )

                tt.update_cores([
                    (d, current_core.from_left_unfold(u)),
                    (d + 1, tt.cores[d + 1].apply_left_factor(np.diag(s).dot(v))),
                ])
            else:
                q, r = np.linalg.qr(
                    current_core.left_unfold(),
                    mode="reduced",
                )
                tt.update_cores([
                    (d, current_core.from_left_unfold(q)),
                    (d + 1, tt.cores[d + 1].apply_left_factor(r)),
                ])
        return tt

    def right_ortho(
        self,
        *,
        threshold: float | None = 0,
        max_rank: int | None = None,
        rank_list: list[int] | None = None,
        method: Literal['svd', 'qr'] = 'svd',
        start_idx: int | None = None,
        end_idx: int = 1,
    ) -> Self:
        """右直交化した TT 表現を返す。

        Args:
            threshold: 相対特異値による打ち切り閾値。``None`` の場合は打ち切らない。
            max_rank: 各中間 rank の最大値。
            rank_list: core ごとの rank 上限。
            method: 局所分解に使う手法。
            start_idx: 右直交化を開始する core index。省略時は最後の core。
            end_idx: 右直交化する最後の core index。

        Returns:
            右直交化した TT 表現。
        """
        if start_idx is not None and not isinstance(start_idx, Integral):
            raise TypeError("start_idx must be an integer.")
        if not isinstance(end_idx, Integral):
            raise TypeError("end_idx must be an integer.")
        if method not in {"svd", "qr"}:
            raise ValueError("method must be 'svd' or 'qr'.")
        end_idx = int(end_idx)

        if self.ndim == 1 and start_idx is None and end_idx == 1:
            return self.copy()

        if start_idx is None:
            start_idx = self.ndim - 1
        else:
            start_idx = int(start_idx)
        if not 0 < end_idx <= start_idx < self.ndim:
            raise ValueError(
                "start_idx and end_idx must satisfy "
                "0 < end_idx <= start_idx < dim."
            )

        tt = self.copy()
        for d in range(start_idx, end_idx - 1, -1):
            current_core = tt.cores[d]
            if method == 'svd':
                u, s, v = trunc_svd(
                    current_core.right_unfold(),
                    criterion="relative",
                    threshold=threshold,
                    max_rank=_combine_max_rank(
                        max_rank,
                        rank_list[d - 1] if rank_list else None,
                    ),
                )

                tt.update_cores([
                    (d - 1, tt.cores[d - 1].apply_right_factor(u.dot(np.diag(s)))),
                    (d, current_core.from_right_unfold(v)),
                ])
            else:
                q, r = np.linalg.qr(
                    current_core.right_unfold().T,
                    mode="reduced",
                )
                tt.update_cores([
                    (d - 1, tt.cores[d - 1].apply_right_factor(r.T)),
                    (d, current_core.from_right_unfold(q.T)),
                ])
        return tt

    def save_npz(self, file: str) -> None:
        """TT core を npz 形式で保存する。

        Args:
            file: 保存先ファイルパス。
        """
        np.savez_compressed(file, *[core.as_array(copy=False) for core in self.cores])

    @classmethod
    def load_npz(cls, file: str) -> Self:
        """npz ファイルから TT 表現を読み込む。

        Args:
            file: 読み込み元ファイルパス。

        Returns:
            読み込んだ TT 表現。
        """
        with np.load(file) as npz:
            cores = [npz[k] for k in npz.files]
        return cls(cores)


def _combine_max_rank(
    max_rank: int | None,
    local_max_rank: int | None,
) -> int | None:
    """共通 rank 上限と局所的な rank 上限を合成する。"""
    if local_max_rank is None:
        return max_rank
    if max_rank is None:
        return local_max_rank
    return min(max_rank, local_max_rank)
