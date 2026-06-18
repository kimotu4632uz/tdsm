from __future__ import annotations

from functools import lru_cache

import numpy as np
import opt_einsum as oe
from opt_einsum.contract import ContractExpression
from opt_einsum.typing import OptimizeKind


class CachedEinsum:
    """`opt_einsum` の contraction 式を cache して効率的に縮約するためのクラス。"""

    optimize: OptimizeKind

    def __init__(self) -> None:
        """contraction 最適化設定を初期化する。"""
        self.optimize: OptimizeKind = 'greedy'

    @lru_cache(maxsize=1024)
    def cached_expr(self, equation: str, *shapes: tuple) -> ContractExpression:
        """shape に対応する contraction 式を cache 付きで返す。

        Parameters
        ----------
        equation : str
            einsum の添字式。
        *shapes : tuple
            各 operand の shape。

        Returns
        -------
        ContractExpression
            再利用可能な contraction 式。
        """
        return oe.contract_expression(equation, *shapes, optimize=self.optimize)

    def expr(self, equation: str, *shapes: tuple) -> ContractExpression:
        """shape に対応する contraction 式を返す。

        Parameters
        ----------
        equation : str
            einsum の添字式。
        *shapes : tuple
            各 operand の shape。

        Returns
        -------
        ContractExpression
            再利用可能な contraction 式。
        """
        return self.cached_expr(equation, *shapes)

    def contract(self, equation: str, *operands: np.ndarray, cache: bool = True) -> np.ndarray:
        """指定した einsum 式で operand を縮約する。

        Parameters
        ----------
        equation : str
            einsum の添字式。
        *operands : np.ndarray
            縮約対象の配列。
        cache : bool, default True
            True の場合、shape ごとに構築済みの式を再利用する。

        Returns
        -------
        np.ndarray
            縮約結果。
        """
        if cache:
            return self.cached_expr(equation, *[op.shape for op in operands])(*operands)
        else:
            return oe.contract(equation, *operands, optimize=self.optimize)
