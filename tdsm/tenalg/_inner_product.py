from __future__ import annotations

from typing import overload

import numpy as np

from ..tensor import RankOneTensor, TTChainTensor, TTTensor
from ..tensor.types import BaseTTTensor


def _contract_tensor_inner_product(
    left: BaseTTTensor,
    right: BaseTTTensor,
    dtype: np.dtype[np.generic] | None = None,
) -> np.ndarray:
    """物理 index を縮約し、境界 rank を残した内積を返す。"""
    if left.mode_dims != right.mode_dims:
        raise ValueError("Mode dimensions must match.")

    if dtype is None:
        dtype_obj = np.dtype(np.result_type(left.dtype, right.dtype))
    else:
        dtype_obj = np.dtype(dtype)

    left_core = np.conjugate(np.asarray(left.cores[0], dtype=dtype_obj))
    right_core = np.asarray(right.cores[0], dtype=dtype_obj)
    contracted = np.einsum(
        "anr,bns->abrs",
        left_core,
        right_core,
        optimize=True,
    )

    for left_core, right_core in zip(left.cores[1:], right.cores[1:], strict=True):
        contracted = np.einsum(
            "abij,inr,jns->abrs",
            contracted,
            np.conjugate(np.asarray(left_core, dtype=dtype_obj)),
            np.asarray(right_core, dtype=dtype_obj),
            optimize=True,
        )
    return contracted


def _rank_one_inner_product(
    left: RankOneTensor,
    right: RankOneTensor,
    dtype: np.dtype[np.generic] | None = None,
) -> np.generic:
    """rank-1 tensor 同士の内積を factor ごとの内積の積として返す。"""
    if left.mode_dims != right.mode_dims:
        raise ValueError("Mode dimensions must match.")

    if dtype is None:
        dtype_obj = np.dtype(np.result_type(left.dtype, right.dtype))
    else:
        dtype_obj = np.dtype(dtype)

    value = dtype_obj.type(1.0)
    for left_factor, right_factor in zip(left, right, strict=True):
        value = value * np.vdot(
            np.asarray(left_factor, dtype=dtype_obj),
            np.asarray(right_factor, dtype=dtype_obj),
        )
    return value


def _contract_tt_with_rank_one(
    left: BaseTTTensor,
    right: RankOneTensor,
    dtype: np.dtype[np.generic] | None = None,
) -> np.ndarray:
    """TT tensor と rank-1 tensor の内積を境界 rank 付きで返す。"""
    if left.mode_dims != right.mode_dims:
        raise ValueError("Mode dimensions must match.")

    if dtype is None:
        dtype_obj = np.dtype(np.result_type(left.dtype, right.dtype))
    else:
        dtype_obj = np.dtype(dtype)

    contracted = np.einsum(
        "anr,n->ar",
        np.conjugate(np.asarray(left.cores[0], dtype=dtype_obj)),
        np.asarray(right[0], dtype=dtype_obj),
        optimize=True,
    )
    for core, factor in zip(left.cores[1:], right[1:], strict=True):
        contracted = np.einsum(
            "ar,rns,n->as",
            contracted,
            np.conjugate(np.asarray(core, dtype=dtype_obj)),
            np.asarray(factor, dtype=dtype_obj),
            optimize=True,
        )
    return contracted.reshape(left.ranks[0], 1, left.ranks[-1], 1)


def _contract_rank_one_with_tt(
    left: RankOneTensor,
    right: BaseTTTensor,
    dtype: np.dtype[np.generic] | None = None,
) -> np.ndarray:
    """rank-1 tensor と TT tensor の内積を境界 rank 付きで返す。"""
    if left.mode_dims != right.mode_dims:
        raise ValueError("Mode dimensions must match.")

    if dtype is None:
        dtype_obj = np.dtype(np.result_type(left.dtype, right.dtype))
    else:
        dtype_obj = np.dtype(dtype)

    contracted = np.einsum(
        "n,bns->bs",
        np.conjugate(np.asarray(left[0], dtype=dtype_obj)),
        np.asarray(right.cores[0], dtype=dtype_obj),
        optimize=True,
    )
    for factor, core in zip(left[1:], right.cores[1:], strict=True):
        contracted = np.einsum(
            "br,rns,n->bs",
            contracted,
            np.asarray(core, dtype=dtype_obj),
            np.conjugate(np.asarray(factor, dtype=dtype_obj)),
            optimize=True,
        )
    return contracted.reshape(1, right.ranks[0], 1, right.ranks[-1])


@overload
def inner_product(
    left: TTChainTensor,
    right: TTChainTensor,
    dtype: np.dtype[np.generic] | None = None,
) -> np.ndarray:
    ...


@overload
def inner_product(
    left: TTChainTensor,
    right: TTTensor | RankOneTensor,
    dtype: np.dtype[np.generic] | None = None,
) -> np.ndarray:
    ...


@overload
def inner_product(
    left: TTTensor | RankOneTensor,
    right: TTChainTensor,
    dtype: np.dtype[np.generic] | None = None,
) -> np.ndarray:
    ...


@overload
def inner_product(
    left: TTTensor,
    right: TTTensor | RankOneTensor,
    dtype: np.dtype[np.generic] | None = None,
) -> np.generic:
    ...


@overload
def inner_product(
    left: RankOneTensor,
    right: TTTensor | RankOneTensor,
    dtype: np.dtype[np.generic] | None = None,
) -> np.generic:
    ...


def inner_product(
    left: BaseTTTensor | RankOneTensor,
    right: BaseTTTensor | RankOneTensor,
    dtype: np.dtype[np.generic] | None = None,
) -> np.ndarray | np.generic:
    """TT tensor または rank-1 tensor 同士の内積を返す。

    少なくとも一方が open boundary の場合は境界 rank を残した配列を返し、
    両方が閉境界の場合は scalar を返します。

    Args:
        left: 左側の tensor。
        right: 右側の tensor。
        dtype: 内部計算に使う dtype。省略時は入力から推定します。

    Returns:
        open boundary を含む場合は内積配列、閉境界同士の場合は内積 scalar。

    Raises:
        TypeError: 入力が TT tensor または rank-1 tensor でない場合。
        ValueError: mode 次元が一致しない場合。
    """
    if not isinstance(left, (BaseTTTensor, RankOneTensor)) or not isinstance(
        right,
        (BaseTTTensor, RankOneTensor),
    ):
        raise TypeError("inner_product requires TT tensors or rank-1 tensors.")
    if left.mode_dims != right.mode_dims:
        raise ValueError("Mode dimensions must match.")

    if isinstance(left, RankOneTensor) and isinstance(right, RankOneTensor):
        return _rank_one_inner_product(left, right, dtype=dtype)

    if isinstance(left, BaseTTTensor) and isinstance(right, RankOneTensor):
        contracted = _contract_tt_with_rank_one(left, right, dtype=dtype)
        if left.is_open_boundary:
            return contracted
        return contracted[0, 0, 0, 0]

    if isinstance(left, RankOneTensor) and isinstance(right, BaseTTTensor):
        contracted = _contract_rank_one_with_tt(left, right, dtype=dtype)
        if right.is_open_boundary:
            return contracted
        return contracted[0, 0, 0, 0]

    assert isinstance(left, BaseTTTensor)
    assert isinstance(right, BaseTTTensor)
    if left.is_open_boundary or right.is_open_boundary:
        return _contract_tensor_inner_product(
            left,
            right,
            dtype=dtype,
        )

    contracted = _contract_tensor_inner_product(
        left,
        right,
        dtype=dtype,
    )
    return contracted[0, 0, 0, 0]
