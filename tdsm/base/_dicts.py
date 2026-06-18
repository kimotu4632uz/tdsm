from abc import ABC, abstractmethod
from collections.abc import Sequence
from operator import index
from typing import Self

import numpy as np
from sklearn.utils.validation import check_array

from ..tensor._rank_one import RankOneTensor
from ..tensor._tt_tensor import TTTensor


class BaseCoreDict(ABC):
    """1 変数入力を特徴ベクトルへ変換する辞書の抽象基底クラス。"""

    @abstractmethod
    def __len__(self) -> int:
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

        Parameters
        ----------
        x : np.number
            1 変数入力。

        Returns
        -------
        np.ndarray
            1 次元特徴ベクトル。
        """
        ...

    @abstractmethod
    def lift_batch(self, x_1d: np.ndarray) -> np.ndarray:
        """1 次元サンプル列を特徴行列へ変換する。

        Parameters
        ----------
        x_1d : np.ndarray
            形状 ``(n_samples,)`` の 1 次元サンプル列。

        Returns
        -------
        np.ndarray
            形状 ``(n_samples, output_dim)`` の特徴行列。
        """
        ...

    @abstractmethod
    def reconstruct(self, features: np.ndarray) -> np.generic:
        """1 mode 分の特徴ベクトルから状態値を復元する。

        Parameters
        ----------
        features : np.ndarray
            形状 ``(output_dim,)`` の特徴ベクトル。

        Returns
        -------
        np.generic
            復元した状態スカラー。
        """
        ...


class TensorProductDict:
    """mode ごとの `BaseCoreDict` を束ねる積辞書。"""

    core_dicts: list[BaseCoreDict] | None
    _template: BaseCoreDict | None

    def __init__(
        self,
        core_dicts: Sequence[BaseCoreDict] | BaseCoreDict,
        ndim: int | None = None,
    ) -> None:
        """積辞書を初期化する。

        Parameters
        ----------
        core_dicts : Sequence[BaseCoreDict] or BaseCoreDict
            各 mode に対応する core dictionary の列。
            単一の core dictionary を渡した場合は全 mode で同じ辞書を使う。
        ndim : int or None, optional
            単一の core dictionary を複製する mode 数。単一辞書を渡し
            `ndim` を省略した場合は遅延テンプレートとなり、mode 数は
            `fit` 時にデータの特徴次元から決まる。

        Raises
        ------
        ValueError
            辞書列が空の場合。
        TypeError
            `BaseCoreDict` でない要素を含む場合。
        """
        if ndim is None:
            if isinstance(core_dicts, BaseCoreDict):
                # 遅延テンプレート: mode 数は fit() でデータ次元から決める。
                self._template = core_dicts
                self.core_dicts = None
                return
            self._template = None
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
            self._template = core_dicts
            dictionaries = [core_dicts for _ in range(dim_int)]
        if len(dictionaries) == 0:
            raise ValueError("core_dicts must contain at least one dictionary")
        if any(not isinstance(dictionary, BaseCoreDict) for dictionary in dictionaries):
            raise TypeError("core_dicts must contain only CoreDictionary instances")
        self.core_dicts = dictionaries

    def _require_fitted(self) -> list[BaseCoreDict]:
        """確定済みの core dictionary 列を返す。未確定なら例外を送出する。"""
        if self.core_dicts is None:
            raise ValueError(
                "TensorProductDict is a deferred template; call fit() to fix its modes."
            )
        return self.core_dicts

    def fit(self, X: np.ndarray, **kwargs) -> Self:
        """データの特徴次元に合わせて積辞書の mode 構成を確定する。

        Parameters
        ----------
        X : np.ndarray
            形状 ``(n_samples, n_features)`` の学習データ。
        **kwargs
            追加のハイパーパラメータ(未使用)。

        Returns
        -------
        Self
            確定済みの自身。

        Raises
        ------
        ValueError
            入力が不正、または確定済み(mode 数固定)の辞書で
            特徴次元が一致しない場合。
        """
        X = check_array(X)
        n_features = X.shape[1]
        if self.core_dicts is None:
            # 遅延テンプレート: データ次元に合わせて mode を構成する。
            assert self._template is not None
            self.core_dicts = [self._template for _ in range(n_features)]
        elif len(self.core_dicts) != n_features:
            raise ValueError(
                f"TensorProductDict has {len(self.core_dicts)} modes "
                f"but X has {n_features} features."
            )
        return self

    @property
    def ndim(self) -> int:
        """mode 数を返す。"""
        return len(self._require_fitted())

    @property
    def mode_dims(self) -> tuple[int, ...]:
        """各 mode の特徴次元を返す。"""
        return tuple(len(dictionary) for dictionary in self._require_fitted())

    def lift_point(self, x: np.ndarray) -> RankOneTensor:
        """1 点の状態を rank-1 TT tensor へ変換する。

        Parameters
        ----------
        x : np.ndarray
            形状 ``(ndim,)`` の状態ベクトル。

        Returns
        -------
        RankOneTensor
            lift 後の `RankOneTensor`。

        Raises
        ------
        ValueError
            入力長が辞書数と一致しない場合。
        """
        core_dicts = self._require_fitted()
        values = check_array(x, ensure_2d=False).reshape(-1)
        if values.size != self.ndim:
            raise ValueError("The number of state entries must match the number of core dictionaries.")
        factors = [
            np.asarray(dictionary.lift_point(value)).reshape(-1)
            for dictionary, value in zip(core_dicts, values, strict=True)
        ]
        dtype = np.dtype(np.result_type(*[factor.dtype for factor in factors]))
        return RankOneTensor(factors, dtype=dtype)

    def _lift_cores_batch(self, x: np.ndarray) -> list[np.ndarray]:
        """バッチデータを mode ごとの特徴行列列へ変換する。

        Parameters
        ----------
        x : np.ndarray
            形状 ``(n_samples, n_features)`` のデータ行列。

        Returns
        -------
        list[np.ndarray]
            各 mode ごとに形状 ``(n_samples, feature_dim_d)`` を持つ配列列。

        Raises
        ------
        ValueError
            入力 shape が不正な場合。
        """
        core_dicts = self._require_fitted()
        data = check_array(x)
        if data.shape[1] != self.ndim:
            raise ValueError("The number of features must match the number of core dictionaries.")
        # SVD 機構が「サンプル=列」を前提とするため、ここで各 mode 列を取り出す。
        return [
            dictionary.lift_batch(data[:, mode_index])
            for mode_index, dictionary in enumerate(core_dicts)
        ]

    def reconstruct(self, tt: TTTensor) -> np.ndarray:
        """TT 特徴表現から状態ベクトルを復元する。

        各状態成分について、対象 mode 以外を各 core dictionary の
        ``constant_index`` で固定し、対象 mode の特徴ベクトルを作ってから
        ``CoreDictionary.reconstruct`` に委譲します。

        Parameters
        ----------
        tt : TTTensor
            積辞書の特徴空間上の `TTTensor`。

        Returns
        -------
        np.ndarray
            復元した状態ベクトル。

        Raises
        ------
        ValueError
            TT の mode 数または mode 次元が辞書と一致しない場合。
        """
        core_dicts = self._require_fitted()
        if tt.ndim != self.ndim:
            raise ValueError("tt must have the same number of modes as the dictionary.")
        if tt.mode_dims != self.mode_dims:
            raise ValueError("tt mode dimensions must match the dictionary.")

        constant_indices = [dictionary.constant_index for dictionary in core_dicts]
        for mode_index, (constant_index, dictionary) in enumerate(
            zip(constant_indices, core_dicts, strict=True)
        ):
            if constant_index < 0 or constant_index >= len(dictionary):
                raise ValueError(f"constant_index for mode {mode_index} is out of bounds.")

        values: list[np.generic] = []
        for mode_index, dictionary in enumerate(core_dicts):
            # features_j[k] = TT[constant_index_0, ..., k, ..., constant_index_d]
            # という特徴ベクトルを作り、reconstruct して状態値を得る
            features = np.empty(len(dictionary), dtype=tt.dtype)
            state_indices = list(constant_indices)
            for feature_index in range(len(dictionary)):
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

        Parameters
        ----------
        psi : TensorProductDict
            積辞書、または各 mode の core dictionary の列。
        """
        self.psi = psi

    def lift_point(self, x: np.ndarray) -> RankOneTensor:
        """1 点の状態を rank-1 TT tensor へ変換する。

        Parameters
        ----------
        x : np.ndarray
            形状 ``(n_features,)`` の状態ベクトル。

        Returns
        -------
        RankOneTensor
            lift 後の `RankOneTensor`。
        """
        x = check_array(x, ensure_2d=False)
        return self.psi.lift_point(x)

    def lift_batch(self, x: np.ndarray) -> TTTensor:
        """バッチデータを `TTTensor` へ変換する。

        Parameters
        ----------
        x : np.ndarray
            形状 ``(n_samples, n_features)`` のデータ行列。

        Returns
        -------
        TTTensor
            sample mode を含む `TTTensor`。
        """
        x = check_array(x)
        return self._build_from_core_features(self.psi._lift_cores_batch(x))

    @abstractmethod
    def _build_from_core_features(self, core_features: Sequence[np.ndarray]) -> TTTensor:
        """辞書評価済み特徴から `TTTensor` を構築する。

        Parameters
        ----------
        core_features : Sequence[np.ndarray]
            各 mode ごとの特徴行列列。

        Returns
        -------
        TTTensor
            構築した `TTTensor`。
        """
        ...

    def fit(self, X: np.ndarray, **kwargs) -> Self:
        """積辞書をデータに合わせて確定する。

        Parameters
        ----------
        X : np.ndarray
            形状 ``(n_samples, n_features)`` の学習データ。
        **kwargs
            追加のハイパーパラメータ(`psi.fit` へ転送)。

        Returns
        -------
        Self
            確定済みの自身。
        """
        X = check_array(X)
        self.psi.fit(X, **kwargs)
        return self
