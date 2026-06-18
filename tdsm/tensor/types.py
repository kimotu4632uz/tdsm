"""TT tensor/operator の型判定・拡張実装向け公開 API を提供する。

このモジュールでは、`isinstance` 判定や独自実装の継承先として使う
基底クラスと core 型を公開します。通常利用する具象クラスは
`lib_tt.tensor` から import してください。
"""

from ._base import BaseTT
from ._tt_core import BaseTTCore, TTOperatorCore, TTTensorCore
from ._tt_operator import BaseTTOperator
from ._tt_tensor import BaseTTTensor
