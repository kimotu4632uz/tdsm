"""TT tensor/operator の通常利用向け公開 API を提供する。

このモジュールでは、利用者が直接生成・操作する具象クラスと factory
関数を公開します。型判定や拡張実装で使う基底クラスは
`lib_tt.tensor.types` から import してください。
"""

from ._rank_one import (
    RankOneTensor,
)
from ._tt_operator import (
    TTChainOperator,
    TTOperator,
    eye,
    filled_operator,
)
from ._tt_tensor import (
    TTChainTensor,
    TTTensor,
    filled_tensor,
)
