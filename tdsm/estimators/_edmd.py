"""TT-EDMD による Koopman 作用素の推定器を提供する。"""

from typing import Any, Self

import numpy as np
from ddsm.utils.svd import truncated_pinv_values
from sklearn.utils.validation import check_is_fitted, validate_data

from ..base import TDSMBaseEstimator
from ..dicts import MonomialsCoreDict, SVDTTBuilder, TensorProductDict
from ..tensor import TTChainTensor, TTTensor
from ._koopman_tt import TTKoopmanOperator


class TTEDMD(TDSMBaseEstimator):
    """TT-EDMD によって Koopman 作用素を推定する推定器。

    スナップショット対から lifted observable 空間上の Koopman 作用素を
    `TTKoopmanOperator` として推定し、学習済み作用素を用いて予測します。
    入力側 ``X`` と出力側 ``y`` で別々の積辞書を使えます。
    """

    psix_cls: type[TensorProductDict]
    psix_kwargs: dict[str, Any] | None
    psiy_cls: type[TensorProductDict]
    psiy_kwargs: dict[str, Any] | None
    threshold_for_svd: float | None
    threshold_for_pinv: float | None
    psix_: TensorProductDict
    psiy_: TensorProductDict
    K_: TTKoopmanOperator

    def __init__(
        self,
        psix_cls: type[TensorProductDict] = TensorProductDict,
        psix_kwargs: dict[str, Any] | None = None,
        psiy_cls: type[TensorProductDict] = TensorProductDict,
        psiy_kwargs: dict[str, Any] | None = None,
        threshold_for_svd: float | None = None,
        threshold_for_pinv: float | None = 1.0e-3,
    ) -> None:
        """TTEDMD 推定器を初期化する。

        Parameters
        ----------
        psix_cls : type[TensorProductDict], default TensorProductDict
            入力側 ``X`` を lift する積辞書クラス。`fit` 時に
            ``psix_cls(**psix_kwargs)`` で構築する。
        psix_kwargs : dict[str, Any] or None, optional
            ``psix_cls`` へ渡すキーワード引数。``core_dicts`` に単一の
            core dictionary を渡し ``ndim`` を省略すると、mode 数は `fit` 時に
            ``X`` の特徴次元へ合わせる(遅延テンプレート)。``None`` の場合は
            ``{"core_dicts": Monomials4TT(degree=2)}`` を使う。
        psiy_cls : type[TensorProductDict], default TensorProductDict
            出力側 ``y`` を lift する積辞書クラス。
        psiy_kwargs : dict[str, Any] or None, optional
            ``psiy_cls`` へ渡すキーワード引数。``None`` の場合は
            ``psix_kwargs`` と同じ既定値を使う。
        threshold_for_svd : float or None, optional
            lifted 行列の SVD 打ち切り閾値。
        threshold_for_pinv : float or None, optional
            擬似逆で無視する特異値の閾値。``None`` の場合は
            0 でない特異値をすべて使う。
        """
        super().__init__()
        self.psix_cls = psix_cls
        self.psix_kwargs = psix_kwargs
        self.psiy_cls = psiy_cls
        self.psiy_kwargs = psiy_kwargs
        self.threshold_for_svd = threshold_for_svd
        self.threshold_for_pinv = threshold_for_pinv

    def _build_psi(
        self,
        psi_cls: type[TensorProductDict],
        psi_kwargs: dict[str, Any] | None,
    ) -> TensorProductDict:
        """ハイパーパラメータから積辞書を構築する。

        ``psi_kwargs`` が ``None`` の場合は ``Monomials4TT(degree=2)`` の
        遅延テンプレートを使う。
        """
        kwargs = (
            psi_kwargs
            if psi_kwargs is not None
            else {"core_dicts": MonomialsCoreDict(degree=2)}
        )
        return psi_cls(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """スナップショット列から TT 形式の Koopman 作用素を推定する。

        Parameters
        ----------
        X : np.ndarray
            スナップショット列の入力側。shape は ``(n_samples, n_features)``。
        y : np.ndarray
            スナップショット列の出力側。shape は ``(n_samples, n_features)`` か
            ``(n_samples,)``。

        Returns
        -------
        Self
            推定済みの自身。
        """
        X, y = validate_data(self, X, y, reset=True, multi_output=True)
        self._y_ndim_1d = y.ndim == 1
        if self._y_ndim_1d:
            y = y[:, np.newaxis]

        # 入力側 X と出力側 y で別々の積辞書を構築し、それぞれを fit する。
        self.psix_ = self._build_psi(self.psix_cls, self.psix_kwargs)
        self.psiy_ = self._build_psi(self.psiy_cls, self.psiy_kwargs)
        builder_x = SVDTTBuilder(
            psi=self.psix_,
            threshold_for_svd=self.threshold_for_svd,
        )
        builder_y = SVDTTBuilder(
            psi=self.psiy_,
            threshold_for_svd=self.threshold_for_svd,
        )
        builder_x.fit(X)  # psix_.fit(X): mode 数の確定/検証
        builder_y.fit(y)  # psiy_.fit(y): mode 数の確定/検証

        basis_x, singular_values_x, right_vectors_x = builder_x.factorize(X)
        basis_y, singular_values_y, right_vectors_y = builder_y.factorize(y)
        if right_vectors_y.shape[1] != right_vectors_x.shape[1]:
            raise ValueError("X and y must have the same number of samples.")

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
        self.K_ = TTKoopmanOperator(
            left_chain=left_chain,
            right_chain=basis_x,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """推定済み TT Koopman 作用素で 1 ステップ先の状態を予測する。

        Parameters
        ----------
        X : np.ndarray
            初期状態のバッチ。形状 ``(n_samples, n_features)``。

        Returns
        -------
        np.ndarray
            予測した次時刻の状態。形状 ``(n_samples, n_targets)``。`fit` 時の
            ``y`` が 1 次元だった場合は ``(n_samples,)``。
        """
        check_is_fitted(self, "K_")
        X = validate_data(self, X, reset=False)
        # 各サンプル(行)を 1 点ずつ予測して積み上げる。出力は psiy_ 空間で復元する。
        predictions = np.stack(
            [self.psiy_.reconstruct(self.predict_tt(sample)) for sample in X]
        )
        if self._y_ndim_1d:
            return predictions[:, 0]
        return predictions

    def predict_tt(self, x: np.ndarray) -> TTTensor:
        """推定済み TT Koopman 作用素で 1 ステップ先のリフト状態を予測する。

        Parameters
        ----------
        x : np.ndarray
            初期状態。形状 ``(n_features,)``。

        Returns
        -------
        TTTensor
            予測したリフト状態の `TTTensor`(出力側 ``psiy_`` 空間)。
        """
        check_is_fitted(self, "K_")
        return self.K_.apply_rank_one(self.psix_.lift_point(x))
