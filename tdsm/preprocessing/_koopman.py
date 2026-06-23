"""スケール座標で同定した TT 形式の Koopman 作用素を元座標へ戻すユーティリティ。"""

from __future__ import annotations

import numpy as np

from ..estimators import TTKoopmanOperator
from ..tensor import TTChainTensor


def unscale_koopman(
    operator: TTKoopmanOperator,
    scale: np.ndarray | float,
) -> TTKoopmanOperator:
    """単項式基底の TT Koopman 作用素をスケール座標から元座標へ戻す。

    スケール変数 ``y_k = scale_k * x_k`` の下で同定した作用素を、元の座標 ``x`` に
    対応する作用素へ相似変換 ``D^{-1} A D`` で戻す。``MonomialsCoreDict`` を用いた TT
    表現では、各コアの mode index がその変数の単項式次数に一致するため、出力側
    ``left_chain`` の各コアを ``scale_k ** i`` で除算し、入力側 ``right_chain`` の各コアを
    ``scale_k ** i`` で乗算することで、dense 表現に対する相似変換と等価な変換になる。
    全次元共通の係数(スカラー)でも次元ごとに異なる係数(``PowerOfTenScaler.scale_``)でも
    正しく戻せる。

    Notes
    -----
    各コアの mode index を単項式次数とみなすため、``MonomialsCoreDict`` のように
    "mode index = その変数の次数" となる辞書で同定した作用素にのみ適用できる。

    Parameters
    ----------
    operator : TTKoopmanOperator
        スケール座標で同定した TT 形式の Koopman 作用素。
    scale : np.ndarray or float
        データのスケーリング係数。``PowerOfTenScaler.scale_`` を渡す。
        スカラーを渡した場合は全次元共通として扱う。配列を渡す場合は形状
        ``(dim,)`` で、各 chain のコア順序に対応する。

    Returns
    -------
    TTKoopmanOperator
        元座標に対応する TT Koopman 作用素(直交化はしない)。

    Raises
    ------
    ValueError
        ``scale`` が配列の場合に、その長さが各 chain のコア数と一致しないとき。
    """
    scale = np.asarray(scale, dtype=float)

    left_ndim = operator.left_chain.ndim
    right_ndim = operator.right_chain.ndim

    if scale.ndim == 0:
        left_scales = np.full(left_ndim, scale)
        right_scales = np.full(right_ndim, scale)
    else:
        if scale.shape[0] != left_ndim or scale.shape[0] != right_ndim:
            raise ValueError(
                "scale array length must match the number of cores in both chains"
            )
        left_scales = scale
        right_scales = scale

    left_cores = []
    for core, s in zip(operator.left_chain.cores, left_scales, strict=True):
        scaled = core.as_array(copy=True)
        powers = s ** np.arange(core.mode_dim)
        scaled /= powers[None, :, None]
        left_cores.append(scaled)

    right_cores = []
    for core, s in zip(operator.right_chain.cores, right_scales, strict=True):
        scaled = core.as_array(copy=True)
        powers = s ** np.arange(core.mode_dim)
        scaled *= powers[None, :, None]
        right_cores.append(scaled)

    return type(operator)(
        left_chain=TTChainTensor(left_cores),
        right_chain=TTChainTensor(right_cores),
    )
