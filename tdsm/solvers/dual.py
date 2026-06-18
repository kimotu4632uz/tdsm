"""TT 形式で双対方程式を解く solver を提供する。"""

from typing import Literal, Sequence

import numpy as np
import sympy as sp
from ddsm.solvers.dual import StatsTrajectory
from ddsm.solvers.model import BaseModel
from ddsm.solvers.time_grid import TimeGrid

from ..dicts import Monomials4TT, TensorProductDict
from ..tenalg import add_operator, inner_product
from ..tensor import TTOperator, TTTensor, filled_tensor
from .ode import CrankNicolson, LinearSolverOptions


def create_generator(model: BaseModel, degree: int) -> TTOperator:
    """SDE モデルの Koopman 生成子を TT operator として構築する。

    Args:
        model: 随伴作用素を構築する SDE モデル。
        degree: 各 mode の最大次数。

    Returns:
        辞書として Monomials4TT を使って得られる Koopman 生成子を表す TT operator。
    """
    sym_derivs = sp.symarray(r'\partial', (model.dim,))
    L = model.build_adjoint_operator(sym_derivs)

    variables: list[sp.Symbol] = []
    variables.extend(model.sym_xs.flat)
    variables.extend(sym_derivs.flat)

    n_trunc = degree + 1
    dim = model.dim
    TT_op: TTOperator = None  # type: ignore

    # 式に具体的なパラメータを代入し、オペレータを求める。
    expand = L.subs(model.param_pairs)
    for e, arg in enumerate(expand.args):   # 各項（イベント）ごとに処理
        coeff = sp.LC(arg)  # 係数を引っ張り出す
        degree_list = np.array(sp.degree_list(
            arg, gens=variables))  # 次数のリストを得る
        state_change = degree_list[:dim] - degree_list[dim:]  # 状態変化のベクトルを得る
        state_rate = degree_list[dim:]  # 微分演算子の部分の次数を得る（これが係数に反映される）
        cores = []
        for d in range(dim):
            if state_rate[d] == 0:
                comp = np.eye(n_trunc, k=-state_change[d])
            elif state_rate[d] == 1:
                comp = np.eye(
                    n_trunc, k=-state_change[d])@np.diag(np.arange(n_trunc))
            else:
                comp = np.eye(
                    n_trunc, k=-state_change[d])@np.diag(np.arange(n_trunc)*(np.arange(n_trunc)-1))
            core = np.zeros([1, n_trunc, n_trunc, 1])

            # TTは2次元以上でないと定義できない
            # なので必ずdim >= 2
            if d == 0:
                core[0, :, :, 0] = coeff*comp
            else:
                core[0, :, :, 0] = comp
            cores.append(core)

        if e == 0:  # 最初だけ作成
            TT_op = TTOperator(cores)
        else:
            TT_op = add_operator(TT_op, TTOperator(cores))

    return TT_op


class TTDual:
    """TT 形式の Crank-Nicolson 法で双対方程式を解くソルバー。"""

    degree: int
    tt_rank: int
    threshold: float | None
    method: Literal['als', 'mals']
    _core_psi: Monomials4TT

    def __init__(
        self,
        *,
        degree: int,
        tt_rank: int,
        threshold: float | None,
        method: Literal['als', 'mals'] = 'als',
    ) -> None:
        """TT dual ソルバーを初期化する。

        Args:
            degree: 各 mode の最大次数。
            tt_rank: Crank-Nicolson で使う最大 TT rank。
            threshold: TT rank 打ち切り閾値。``None`` の場合は打ち切らない。
            method: 線形方程式ソルバーの種類。
        """
        self.degree = degree
        self.tt_rank = tt_rank
        self.threshold = threshold
        self.method = method
        self._core_psi = Monomials4TT(degree=degree)

    def _calc_stat(self, tt: TTTensor, x0: np.ndarray) -> float:
        """初期点で評価した TT tensor の期待値を計算する。

        Args:
            tt: 評価対象の TT tensor。
            x0: 初期点。

        Returns:
            初期点で評価した期待値。
        """

        psi = TensorProductDict(
            self._core_psi,
            ndim=x0.size,
        )
        lifted = psi.lift_point(x0)
        return float(inner_product(lifted, tt))


    def solve_ode(self, *, model: BaseModel, time_grid: TimeGrid, comp_index: list[int]) -> Sequence[TTTensor]:
        """双対方程式を解き、観測時刻ごとの統計量と状態を返す。

        Args:
            model: 対象とする BaseModel。
            time_grid: 時間離散化と保存時刻の設定。
            comp_index: 初期 TT tensor で 1 を置く各 mode の単項式次数。

        Returns:
            統計量列と TT tensor 状態列を含む軌道。

        Raises:
            ValueError: ``comp_index`` の長さが model の次元と一致しない場合、
                または次数が辞書に含まれない場合。
        """
        if len(comp_index) != model.dim:
            raise ValueError("comp_index must have the same length as model dimension.")

        TT_op = create_generator(model, self.degree)

        p_ini = filled_tensor(0, TT_op.col_dims)
        for p, degree in enumerate(comp_index):
            p_ini.cores[p][0, self._core_psi.s2i(degree), 0] = 1

        solver = CrankNicolson(
            solver_type=self.method,
            options=LinearSolverOptions(
                alg_decomp='svd',
                atol=1e-9,
                threshold=self.threshold,
                max_rank=self.tt_rank,
            ))
        states = solver.solve(TT_op, p_ini, time_grid)
        return states


    def solve(self, *, model: BaseModel, x0: np.ndarray, time_grid: TimeGrid, comp_index: list[int]) -> StatsTrajectory[TTTensor]:
        """双対方程式を解き、観測時刻ごとの統計量と状態を返す。

        Args:
            model: 対象とする BaseModel。
            x0: 統計量を評価する初期点。
            time_grid: 時間離散化と保存時刻の設定。
            comp_index: 初期 TT tensor で 1 を置く各 mode の単項式次数。
        Returns:
            統計量列と TT tensor 状態列を含む軌道。
        """
        p_list = self.solve_ode(model=model, time_grid=time_grid, comp_index=comp_index)
        stats = [self._calc_stat(p, x0) for p in p_list]

        return StatsTrajectory(time_grid=time_grid, stats=np.asarray(stats), states=p_list)
