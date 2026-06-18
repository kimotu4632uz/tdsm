"""TT-EDMD による Koopman 作用素の推定器を提供する。"""

from typing import Self

import numpy as np
from ddsm.utils.svd import truncated_pinv_values

from ..base import TDSMBaseEstimator
from ..dicts import SVDTTBuilder, TensorProductDictionary
from ..tensor import TTChainTensor, TTTensor
from ._koopman_tt import TTKoopmanOperator


class TTEDMD(TDSMBaseEstimator):
    """TT-EDMD によって Koopman 作用素を推定する推定器。

    スナップショット対から lifted observable 空間上の Koopman 作用素を
    `TTKoopmanOperator` として推定し、学習済み作用素を用いて予測します。
    """

    psi: TensorProductDictionary
    threshold_for_svd: float | None
    threshold_for_pinv: float | None
    K: TTKoopmanOperator | None

    def __init__(
        self,
        *,
        psi: TensorProductDictionary,
        threshold_for_svd: float | None,
        threshold_for_pinv: float | None = 1.0e-3,
    ) -> None:
        """TTEDMD 推定器を初期化する。

        Args:
            psi: 各状態成分へ適用する core dictionary。
            threshold_for_svd: lifted 行列の SVD 打ち切り閾値。
            threshold_for_pinv: 擬似逆で無視する特異値の閾値。``None`` の場合は
                0 でない特異値をすべて使う。
        """
        self.psi = psi
        self.threshold_for_svd = threshold_for_svd
        self.threshold_for_pinv = threshold_for_pinv
        self.K = None


    def fit(self, x: np.ndarray, y: np.ndarray) -> Self:
        """スナップショット列から TT 形式の Koopman 作用素を推定する。

        Args:
            x: スナップショット列の入力側。shape は ``(dim, n_samples)``。
            y: スナップショット列の出力側。shape は ``(dim, n_samples)``。

        Returns:
            推定済みの自身。
        """
        builder = SVDTTBuilder(
            psi=self.psi,
            threshold_for_svd=self.threshold_for_svd,
        )
        basis_x, singular_values_x, right_vectors_x = builder.factorize(x)
        basis_y, singular_values_y, right_vectors_y = builder.factorize(y)
        if right_vectors_y.shape[1] != right_vectors_x.shape[1]:
            raise ValueError("x and y must have the same number of samples.")

        cross = right_vectors_y @ right_vectors_x.T

        dtype = np.dtype(
            np.result_type(
                singular_values_y,
                singular_values_x,
                right_vectors_y,
                right_vectors_x,
            )
        )
        inv_sx = truncated_pinv_values(
            singular_values_x,
            pinv_tol=self.threshold_for_pinv,
        ).astype(dtype, copy=False)
        middle = (singular_values_y[:, None] * cross) * inv_sx[None, :]

        expected_middle_shape = (basis_y.ranks[-1], basis_x.ranks[-1])
        if middle.shape != expected_middle_shape:
            raise ValueError(f"middle must have shape {expected_middle_shape}.")
        left_chain = TTChainTensor([
            *basis_y.cores[:-1],
            basis_y.cores[-1].apply_right_factor(middle),
        ])
        self.K = TTKoopmanOperator(
            left_chain=left_chain,
            right_chain=basis_x,
        )
        return self


    def predict(self, x: np.ndarray) -> np.ndarray:
        """推定済み TT Koopman 作用素で 1 ステップ先の状態を予測する。

        Args:
            x: 初期状態。形状 ``(dim,)``。

        Returns:
            予測した次時刻の状態ベクトル。

        Raises:
            ValueError: ``fit`` がまだ呼ばれておらず Koopman 作用素が未推定の場合。
        """
        return self.psi.reconstruct(self.predict_tt(x))

    def predict_tt(self, x: np.ndarray) -> TTTensor:
        """推定済み TT Koopman 作用素で 1 ステップ先のリフト状態を予測する。

        Args:
            x: 初期状態。形状 ``(dim,)``。

        Returns:
            予測したリフト状態の `TTTensor`。

        Raises:
            ValueError: ``fit`` がまだ呼ばれておらず Koopman 作用素が未推定の場合。
        """
        if self.K is None:
            raise ValueError("TTEDMD: K is not computed yet. Call fit() first.")
        return self.K.apply_rank_one(self.psi.lift_point(x))
