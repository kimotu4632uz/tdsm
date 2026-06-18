"""TT 表現の和・積・内積などの演算を提供する。"""

from __future__ import annotations

import numpy as np

from ..tensor import TTOperator, TTTensor


def add_tensor(tt1: TTTensor, tt2: TTTensor) -> TTTensor:
    """2 つの TT tensor の和を返す。

    Args:
        tt1: 1 つ目の TT tensor。
        tt2: 2 つ目の TT tensor。

    Returns:
        和を表す TT tensor。

    Raises:
        ValueError: mode 次元が一致しない場合。
    """
    if tt1.mode_dims != tt2.mode_dims:
        raise ValueError("Two TT tensors must have the same mode dimensions.")

    dim = tt1.ndim
    ranks = [1] + [tt1.ranks[d] + tt2.ranks[d] for d in range(1, dim)] + [1]
    dtype = np.result_type(*[core.dtype for core in tt1.cores + tt2.cores])
    cores: list[np.ndarray] = []
    for d in range(dim):
        core = np.zeros((ranks[d], tt1.mode_dims[d], ranks[d + 1]), dtype=dtype)
        core[: tt1.ranks[d], :, : tt1.ranks[d + 1]] = tt1.cores[d]
        core[
            ranks[d] - tt2.ranks[d] : ranks[d],
            :,
            ranks[d + 1] - tt2.ranks[d + 1] : ranks[d + 1],
        ] = tt2.cores[d]
        cores.append(core)
    return tt1._new_like(cores)


def add_operator(tt1: TTOperator, tt2: TTOperator) -> TTOperator:
    """2 つの TT operator の和を返す。

    Args:
        tt1: 1 つ目の TT operator。
        tt2: 2 つ目の TT operator。

    Returns:
        和を表す TT operator。

    Raises:
        ValueError: row 次元または column 次元が一致しない場合。
    """
    if tt1.row_dims != tt2.row_dims or tt1.col_dims != tt2.col_dims:
        raise ValueError("Two TT operators must have the same row and column dimensions.")
    dim = tt1.ndim
    ranks = [1] + [tt1.ranks[d] + tt2.ranks[d] for d in range(1, dim)] + [1]
    dtype = np.result_type(*[core.dtype for core in tt1.cores + tt2.cores])
    cores: list[np.ndarray] = []
    for d in range(dim):
        core = np.zeros((ranks[d], tt1.row_dims[d], tt1.col_dims[d], ranks[d + 1]), dtype=dtype)
        core[: tt1.ranks[d], :, :, : tt1.ranks[d + 1]] = tt1.cores[d]
        core[
            ranks[d] - tt2.ranks[d] : ranks[d],
            :,
            :,
            ranks[d + 1] - tt2.ranks[d + 1] : ranks[d + 1],
        ] = tt2.cores[d]
        cores.append(core)
    return type(tt1)(cores)


def mul_operator_core(core1: np.ndarray, core2: np.ndarray) -> np.ndarray:
    """2 つの TT operator core の積を返す。

    Args:
        core1: 左側の TT operator core。
        core2: 右側の TT operator core。

    Returns:
        積を表す TT operator core。
    """
    core1 = np.asarray(core1)
    core2 = np.asarray(core2)
    r1, n1, _m1, s1 = core1.shape
    r2, _n2, m2, s2 = core2.shape
    reshaped_core1 = core1.transpose(0, 3, 1, 2)[:, np.newaxis, :, np.newaxis, :, :]
    broadcasted_core1 = np.broadcast_to(reshaped_core1, (r1, r2, s1, s2, n1, core1.shape[2]))
    reshaped_core2 = core2.transpose(0, 3, 1, 2)[np.newaxis, :, np.newaxis, :, :, :]
    broadcasted_core2 = np.broadcast_to(reshaped_core2, (r1, r2, s1, s2, core2.shape[1], m2))

    contracted_core = broadcasted_core1 @ broadcasted_core2
    core = contracted_core.reshape(
        core1.shape[0] * core2.shape[0],
        core1.shape[3] * core2.shape[3],
        core1.shape[1],
        core2.shape[2],
    ).transpose(0, 2, 3, 1)
    return core


def mul_operator(tt1: TTOperator, tt2: TTOperator) -> TTOperator:
    """2 つの TT operator の積を返す。

    Args:
        tt1: 左側の TT operator。
        tt2: 右側の TT operator。

    Returns:
        積を表す TT operator。

    Raises:
        ValueError: 次元数、または左 column 次元と右 row 次元が一致しない場合。
    """
    if tt1.ndim != tt2.ndim:
        raise ValueError("Two TT operators must have the same dimension.")
    if tt1.col_dims != tt2.row_dims:
        raise ValueError("Left column dimensions must match right row dimensions.")
    cores = [mul_operator_core(np.asarray(tt1.cores[d]), np.asarray(tt2.cores[d])) for d in range(tt1.ndim)]
    return type(tt1)(cores)


def apply_operator(op: TTOperator, tensor: TTTensor) -> TTTensor:
    """TT operator を TT tensor に作用させる。

    Args:
        op: 作用させる TT operator。
        tensor: 入力 TT tensor。

    Returns:
        ``op`` を ``tensor`` に作用させた TT tensor。

    Raises:
        ValueError: 次元数、または operator の column 次元と tensor の mode 次元が一致しない場合。
    """
    if op.ndim != tensor.ndim:
        raise ValueError("TT operator and TT tensor must have the same dimension.")
    if op.col_dims != tensor.mode_dims:
        raise ValueError("Operator column dimensions must match tensor mode dimensions.")

    cores: list[np.ndarray] = []
    for operator_core, tensor_core in zip(op.cores, tensor.cores, strict=True):
        contracted_core = np.einsum(
            "aijb,cjd->acibd",
            operator_core,
            tensor_core,
            optimize=True,
        )
        cores.append(
            contracted_core.reshape(
                operator_core.left_rank * tensor_core.left_rank,
                operator_core.row_dim,
                operator_core.right_rank * tensor_core.right_rank,
            )
        )
    return TTTensor(cores)
