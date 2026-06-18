from collections.abc import Sequence
from typing import override

import numpy as np

from ddsm.utils.svd import trunc_svd

from ..base._dicts import TensorProductDictionary, TTBuilder
from ..tensor import TTChainTensor, TTTensor


class SVDTTBuilder(TTBuilder):
    """SVD によってバッチデータを TT へ圧縮する builder。"""

    threshold_for_svd: float | None

    def __init__(
        self,
        psi: TensorProductDictionary,
        threshold_for_svd: float | None,
    ) -> None:
        """SVD builder を初期化する。

        Args:
            psi: 積辞書、または各 mode の core dictionary の列。
            threshold_for_svd: 特異値の累積寄与率による打ち切り閾値。
        """
        super().__init__(psi)
        self.threshold_for_svd = threshold_for_svd

    @override
    def _build_from_core_features(self, core_features: Sequence[np.ndarray]) -> TTTensor:
        """辞書評価済み特徴から sample mode を含む `TTTensor` を構築する。

        Args:
            core_features: 各 mode ごとの特徴行列列。

        Returns:
            空間 mode と sample mode を持つ `TTTensor`。
        """
        basis_tt, singular_values, right_vectors = self._factorize_from_core_features(core_features)
        residual = np.diag(singular_values).dot(right_vectors)
        sample_core = residual.reshape(residual.shape[0], residual.shape[1], 1)
        return TTTensor([*basis_tt.cores, sample_core])

    def factorize(self, x: np.ndarray) -> tuple[TTChainTensor, np.ndarray, np.ndarray]:
        """バッチデータを TT 基底と右側行列へ分解する。

        Args:
            x: 形状 ``(dim, n_samples)`` のデータ行列。

        Returns:
            ``(TT 基底, 特異値, 右特異ベクトル行列)``。
        """
        return self._factorize_from_core_features(self.psi._lift_cores_batch(x))

    def _factorize_from_core_features(
        self,
        core_features: Sequence[np.ndarray],
    ) -> tuple[TTChainTensor, np.ndarray, np.ndarray]:
        """辞書評価済み特徴を TT 基底と右側行列へ分解する。

        Args:
            core_features: 各 mode ごとに形状 ``(feature_dim, n_samples)`` を持つ配列列。

        Returns:
            ``(TT 基底, 特異値, 右特異ベクトル行列)``。

        Raises:
            ValueError: 入力が空、shape が不整合、またはサンプル数が一致しない場合。
        """
        if len(core_features) == 0:
            raise ValueError("core_features must contain at least one mode")

        num_data = core_features[0].shape[1]
        for mode_index, features in enumerate(core_features):
            if features.ndim != 2:
                raise ValueError(f"core_features[{mode_index}] must be 2-dimensional")
            if features.shape[0] == 0 or features.shape[1] == 0:
                raise ValueError(f"core_features[{mode_index}] must be non-empty")
            if features.shape[1] != num_data:
                raise ValueError("All core_features must share the same number of samples")

        cores: list[np.ndarray] = [np.empty(0)] * len(core_features)

        residual = np.ones((1, num_data), dtype=core_features[0].dtype)
        singular_values = np.empty(0, dtype=core_features[0].dtype)
        right_vectors = np.empty((0, num_data), dtype=core_features[0].dtype)

        for mode_index, features in enumerate(core_features):
            core_tmp = residual[:, None, :] * features[None, :, :]
            reshaped = core_tmp.reshape(core_tmp.shape[0] * core_tmp.shape[1], core_tmp.shape[2])
            u, singular_values, right_vectors = trunc_svd(
                reshaped,
                criterion="cumulative",
                threshold=self.threshold_for_svd,
            )
            cores[mode_index] = u.reshape(core_tmp.shape[0], core_tmp.shape[1], u.shape[1])
            residual = np.diag(singular_values).dot(right_vectors)

        return TTChainTensor(cores), singular_values, right_vectors
