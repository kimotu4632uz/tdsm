"""TT tensor/operator の和や積、内積などの線形代数演算を提供するサブパッケージ。"""

from ._einsum import (
    CachedEinsum,
)
from ._inner_product import (
    inner_product,
)
from ._tenalg import (
    add_operator,
    add_tensor,
    apply_operator,
    mul_operator,
    mul_operator_core,
)
