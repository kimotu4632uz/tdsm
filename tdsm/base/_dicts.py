from abc import ABC, abstractmethod
from collections.abc import Sequence
from operator import index

import numpy as np

from ..tensor._rank_one import RankOneTensor
from ..tensor._tt_tensor import TTTensor


class BaseCoreDict(ABC):
    """1 変数入力を特徴ベクトルへ変換する辞書の抽象基底クラス。"""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """出力特徴ベクトルの次元を返す。"""
        ...

    @property
    @abstractmethod
    def constant_index(self) -> int:
        """定数関数 1 に対応する特徴 index を返す。"""
        ...

    @abstractmethod
    def lift_point(self, x: np.number) -> np.ndarray:
        """1 点の入力を特徴ベクトルへ変換する。

        Args:
            x: 1 変数入力。

        Returns:
            1 次元特徴ベクトル。
        """
        ...

    @abstractmethod
    def lift_batch(self, x_1d: np.ndarray) -> np.ndarray:
        """1 次元サンプル列を特徴行列へ変換する。

        Args:
            x_1d: 形状 ``(n_samples,)`` の 1 次元サンプル列。

        Returns:
            形状 ``(output_dim, n_samples)`` の特徴行列。
        """
        ...

    @abstractmethod
    def reconstruct(self, features: np.ndarray) -> np.generic:
        """1 mode 分の特徴ベクトルから状態値を復元する。

        Args:
            features: 形状 ``(output_dim,)`` の特徴ベクトル。

        Returns:
            復元した状態スカラー。
        """
        ...


class TensorProductDict:
    """mode ごとの `BaseCoreDict` を束ねる積辞書。"""

    core_dicts: list[BaseCoreDict]

    def __init__(
        self,
        core_dicts: Sequence[BaseCoreDict] | BaseCoreDict,
        ndim: int | None = None,
    ) -> None:
        """積辞書を初期化する。

        Args:
            core_dicts: 各 mode に対応する core dictionary の列。
                `ndim` を指定した場合は、全 mode に使う単一の core dictionary。
            ndim: 単一の core dictionary を複製する mode 数。

        Raises:
            ValueError: 辞書列が空の場合。
            TypeError: `CoreDictionary` でない要素を含む場合。
        """
        if ndim is None:
            if isinstance(core_dicts, BaseCoreDict):
                raise ValueError("ndim must be provided when core_dicts is a single CoreDictionary")
            dictionaries = list(core_dicts)
        else:
            if not isinstance(core_dicts, BaseCoreDict):
                raise TypeError("core_dicts must be a CoreDictionary when ndim is provided")
            if isinstance(ndim, bool):
                raise ValueError("ndim must be a positive integer")
            try:
                dim_int = index(ndim)
            except TypeError as exc:
                raise TypeError("ndim must be an integer") from exc
            if dim_int <= 0:
                raise ValueError("ndim must be a positive integer")
            dictionaries = [core_dicts for _ in range(dim_int)]
        if len(dictionaries) == 0:
            raise ValueError("core_dicts must contain at least one dictionary")
        if any(not isinstance(dictionary, BaseCoreDict) for dictionary in dictionaries):
            raise TypeError("core_dicts must contain only CoreDictionary instances")
        self.core_dicts = dictionaries

    @property
    def ndim(self) -> int:
        """mode 数を返す。"""
        return len(self.core_dicts)

    @property
    def mode_dims(self) -> tuple[int, ...]:
        """各 mode の特徴次元を返す。"""
        return tuple(dictionary.output_dim for dictionary in self.core_dicts)

    def lift_point(self, x: np.ndarray) -> RankOneTensor:
        """1 点の状態を rank-1 TT tensor へ変換する。

        Args:
            x: 形状 ``(ndim,)`` の状態ベクトル。

        Returns:
            lift 後の `RankOneTensor`。

        Raises:
            ValueError: 入力長が辞書数と一致しない場合。
        """
        values = np.asarray(x).reshape(-1)
        if values.size != self.ndim:
            raise ValueError("The number of state entries must match the number of core dictionaries.")
        factors = [
            np.asarray(dictionary.lift_point(value)).reshape(-1)
            for dictionary, value in zip(self.core_dicts, values, strict=True)
        ]
        dtype = np.dtype(np.result_type(*[factor.dtype for factor in factors]))
        return RankOneTensor(factors, dtype=dtype)

    def _lift_cores_batch(self, x: np.ndarray) -> list[np.ndarray]:
        """バッチデータを mode ごとの特徴行列列へ変換する。

        Args:
            x: 形状 ``(ndim, n_samples)`` のデータ行列。

        Returns:
            各 mode ごとに形状 ``(feature_dim_d, n_samples)`` を持つ配列列。

        Raises:
            ValueError: 入力 shape が不正な場合。
        """
        data = np.asarray(x)
        if data.ndim != 2:
            raise ValueError("x must be a 2D array")
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("x must contain at least one mode and one sample")
        if data.shape[0] != self.ndim:
            raise ValueError("The number of rows must match the number of core dictionaries.")
        return [
            dictionary.lift_batch(data[mode_index, :])
            for mode_index, dictionary in enumerate(self.core_dicts)
        ]

    def reconstruct(self, tt: TTTensor) -> np.ndarray:
        """TT 特徴表現から状態ベクトルを復元する。

        各状態成分について、対象 mode 以外を各 core dictionary の
        ``constant_index`` で固定し、対象 mode の特徴ベクトルを作ってから
        ``CoreDictionary.reconstruct`` に委譲します。

        Args:
            tt: 積辞書の特徴空間上の `TTTensor`。

        Returns:
            復元した状態ベクトル。

        Raises:
            ValueError: TT の mode 数または mode 次元が辞書と一致しない場合。
        """
        if tt.ndim != self.ndim:
            raise ValueError("tt must have the same number of modes as the dictionary.")
        if tt.mode_dims != self.mode_dims:
            raise ValueError("tt mode dimensions must match the dictionary.")

        constant_indices = [dictionary.constant_index for dictionary in self.core_dicts]
        for mode_index, (constant_index, dictionary) in enumerate(
            zip(constant_indices, self.core_dicts, strict=True)
        ):
            if constant_index < 0 or constant_index >= dictionary.output_dim:
                raise ValueError(f"constant_index for mode {mode_index} is out of bounds.")

        values: list[np.generic] = []
        for mode_index, dictionary in enumerate(self.core_dicts):
            # features_j[k] = TT[constant_index_0, ..., k, ..., constant_index_d]
            # という特徴ベクトルを作り、reconstruct して状態値を得る
            features = np.empty(dictionary.output_dim, dtype=tt.dtype)
            state_indices = list(constant_indices)
            for feature_index in range(dictionary.output_dim):
                state_indices[mode_index] = feature_index
                features[feature_index] = tt.get_element(state_indices)
            values.append(dictionary.reconstruct(features))
        return np.asarray(values)


class TTBuilder(ABC):
    """積辞書から TT 特徴を構築する基底クラス。"""

    psi: TensorProductDict

    def __init__(
        self,
        psi: TensorProductDict,
    ) -> None:
        """builder を初期化する。

        Args:
            psi: 積辞書、または各 mode の core dictionary の列。
        """
        self.psi = psi

    def lift_point(self, x: np.ndarray) -> RankOneTensor:
        """1 点の状態を rank-1 TT tensor へ変換する。

        Args:
            x: 形状 ``(dim,)`` の状態ベクトル。

        Returns:
            lift 後の `RankOneTensor`。
        """
        return self.psi.lift_point(x)

    def lift_batch(self, x: np.ndarray) -> TTTensor:
        """バッチデータを `TTTensor` へ変換する。

        Args:
            x: 形状 ``(dim, n_samples)`` のデータ行列。

        Returns:
            sample mode を含む `TTTensor`。
        """
        return self._build_from_core_features(self.psi._lift_cores_batch(x))

    @abstractmethod
    def _build_from_core_features(self, core_features: Sequence[np.ndarray]) -> TTTensor:
        """辞書評価済み特徴から `TTTensor` を構築する。

        Args:
            core_features: 各 mode ごとの特徴行列列。

        Returns:
            構築した `TTTensor`。
        """
        ...
