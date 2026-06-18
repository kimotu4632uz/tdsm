"""TT 形式の Crank-Nicolson 法と ALS/MALS 線形方程式 solver を提供する。

参考文献:
    Patrick Gelß, "The Tensor-Train Format and Its Applications,"
    Ph.D. Thesis, Freie Universität Berlin (2017).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg as lin
from ddsm.solvers.time_grid import TimeGrid
from ddsm.utils.svd import trunc_svd
from scipy.sparse.linalg import LinearOperator, lgmres

from ..tenalg import CachedEinsum, add_operator, apply_operator
from ..tensor import TTOperator, TTTensor, eye, filled_tensor


@dataclass(frozen=True, kw_only=True)
class LinearSolverOptions:
    """ALS/MALS 系 solver に共通するオプション。"""

    repeats: int = 1
    threshold: float | None = 1e-12
    max_rank: int | None = None
    atol: float = 0.0
    rtol: float = 1e-4
    alg_decomp: Literal['svd', 'qr'] = 'svd'

    def __post_init__(self) -> None:
        """オプション値を検証する。"""
        if self.repeats < 1:
            raise ValueError("`repeats` must be greater than or equal to 1.")

        if self.max_rank is not None and self.max_rank < 1:
            raise ValueError("`max_rank` must be greater than or equal to 1.")

        if self.threshold is not None and not np.isfinite(self.threshold):
            raise ValueError("`threshold` must be finite or None.")

        if self.threshold is not None and (self.threshold < 0 or self.threshold >= 1):
            raise ValueError("`threshold` must satisfy 0 <= threshold < 1.")

        if self.alg_decomp not in {'svd', 'qr'}:
            raise ValueError("`alg_decomp` must be 'svd' or 'qr'.")


class CrankNicolson:
    """TT operator で表した線形 ODE を Crank-Nicolson 法で時間発展させる。"""

    solver_type: Literal['als', 'mals']
    options: LinearSolverOptions

    def __init__(
        self,
        solver_type: Literal['als', 'mals'],
        options: LinearSolverOptions,
    ) -> None:
        """Crank-Nicolson solver を初期化する。

        Parameters
        ----------
        solver_type : {"als", "mals"}
            ALS/MALS のどちらの solver を使うか。
        options : LinearSolverOptions
            solver オプション。
        """
        if solver_type not in {'als', 'mals'}:
            raise ValueError("solver_type must be 'als' or 'mals'.")

        self.options = options
        self.solver_type = solver_type

    def solve(self, op: TTOperator, p_ini: TTTensor, time_grid: TimeGrid) -> tuple[TTTensor, ...]:
        """指定された時間グリッド上で TT tensor 状態を時間発展させる。

        Parameters
        ----------
        op : TTOperator
            線形 ODE の生成作用素。
        p_ini : TTTensor
            初期状態を表す TT tensor。
        time_grid : TimeGrid
            時間刻みと保存時刻の設定。

        Returns
        -------
        tuple[TTTensor, ...]
            保存対象時刻における TT tensor 状態の tuple。

        Raises
        ------
        ValueError
            operator と初期状態の次元が一致しない場合。
        """
        if op.col_dims != p_ini.mode_dims:
            raise ValueError("Operator column dimensions must match the initial state mode dimensions.")

        pre_sol = p_ini.copy()
        sol = filled_tensor(1, op.col_dims, ranks=op.ranks)
        sol = sol.right_ortho()
        states: list[TTTensor] = []

        solver: 'ALS | MALS'
        if self.solver_type == 'mals':
            solver = MALS(self.options)
        else:
            solver = ALS(self.options)

        identity = eye(op.row_dims)
        op_lhs = add_operator(identity, (-0.5) * time_grid.dt * op)
        op_rhs = add_operator(identity, (+0.5) * time_grid.dt * op)

        for step in time_grid.iter_steps():
            U = apply_operator(op_rhs, pre_sol)
            pre_sol = solver.solve(op_lhs, sol, U)
            if step.should_save:
                states.append(pre_sol.copy())

        return tuple(states)


class _ALSEnvironment:
    """ALS/MALS に共通する環境テンソルと作業バッファを管理する。

    sweep 中の解は、局所 core 更新によって隣接 core との rank 接続が
    一時的に崩れることがあります。そのため、内部では ``TTTensor`` や
    ``TTTensorCore`` として管理せず、``ndarray`` の core 列と rank 列を
    可変な作業バッファとして保持します。最終的な解を返す段階でのみ
    ``TTTensor`` を構築し、TT としての整合性を検証します。
    """

    contractor: CachedEinsum
    op: TTOperator
    sol_cores: list[np.ndarray]
    sol_ranks: list[int]
    sol_mode_dims: list[int]
    rhs: TTTensor
    LA: list[np.ndarray]
    RA: list[np.ndarray]
    LU: list[np.ndarray]
    RU: list[np.ndarray]

    def __init__(self, contractor: CachedEinsum) -> None:
        """環境テンソルの縮約に使う contractor を保持する。

        Parameters
        ----------
        contractor : CachedEinsum
            einsum 縮約を実行する補助オブジェクト。
        """
        self.contractor = contractor

    def init(self, op: TTOperator, sol: TTTensor, rhs: TTTensor) -> None:
        """局所問題を解くための環境テンソルを初期化する。

        Parameters
        ----------
        op : TTOperator
            左辺の正方 TT operator。
        sol : TTTensor
            更新対象の初期解 TT tensor。
        rhs : TTTensor
            右辺 TT tensor。

        Raises
        ------
        ValueError
            operator、解、右辺の次元が整合しない場合。
        """
        if op.ndim != sol.ndim or op.ndim != rhs.ndim:
            raise ValueError("Operator, solution, and right-hand side must have the same dimension.")
        if op.row_dims != op.col_dims:
            raise ValueError("ALS/MALS requires a square TT operator.")
        if op.col_dims != sol.mode_dims:
            raise ValueError("Operator column dimensions must match solution mode dimensions.")
        if op.row_dims != rhs.mode_dims:
            raise ValueError("Operator row dimensions must match right-hand side mode dimensions.")

        self.op = op
        self.sol_cores = [core.as_array(copy=True) for core in sol.cores]
        self.sol_ranks = list(sol.ranks)
        self.sol_mode_dims = list(sol.mode_dims)
        self.rhs = rhs.copy()

        self.LA: list[np.ndarray] = [None]*op.ndim  # type: ignore
        self.RA: list[np.ndarray] = [None]*op.ndim  # type: ignore
        self.LU: list[np.ndarray] = [None]*op.ndim  # type: ignore
        self.RU: list[np.ndarray] = [None]*op.ndim  # type: ignore

        self.LA[0] = np.array([1], ndmin=3)
        self.LU[0] = np.array([1], ndmin=2)
        self.RA[-1] = np.array([1], ndmin=3)
        self.RU[-1] = np.array([1], ndmin=2)

    def solution(self) -> TTTensor:
        """作業用バッファから TT tensor を構築する。"""
        return TTTensor(self.sol_cores)

    def update_LA(self, d: int) -> None:
        """左側の operator 環境 ``LA[d]`` を更新する。

        Gelß (2017) の式 (4.2.16) に対応します。

        Parameters
        ----------
        d : int
            更新先の core index。``d - 1`` 番目の core を縮約に使う。
        """
        solv = self.sol_cores[d-1].reshape(self.sol_ranks[d-1],
                                           self.sol_mode_dims[d-1], self.sol_ranks[d])
        self.LA[d] = self.contractor.contract(
            'ijk,iln,jmlo,kmp->nop', self.LA[d-1], solv, np.array(self.op.cores[d-1]), solv)

    def update_RA(self, d: int) -> None:
        """右側の operator 環境 ``RA[d]`` を更新する。

        Gelß (2017) の式 (4.2.17) に対応します。

        Parameters
        ----------
        d : int
            更新先の core index。``d + 1`` 番目の core を縮約に使う。
        """
        solv = self.sol_cores[d+1].reshape(self.sol_ranks[d+1],
                                           self.sol_mode_dims[d+1], self.sol_ranks[d+2])
        self.RA[d] = self.contractor.contract(
            'kmp,nop,jmlo,iln->ijk', solv, self.RA[d+1], np.array(self.op.cores[d+1]), solv)

    def update_LU(self, d: int) -> None:
        """左側の右辺環境 ``LU[d]`` を更新する。

        Gelß (2017) の式 (4.2.18) に対応します。

        Parameters
        ----------
        d : int
            更新先の core index。``d - 1`` 番目の core を縮約に使う。
        """
        rhsv = self.rhs.cores[d-1].as_array().reshape(self.rhs.ranks[d-1],
                                           self.rhs.mode_dims[d-1], self.rhs.ranks[d])
        solv = self.sol_cores[d-1].reshape(self.sol_ranks[d-1],
                                           self.sol_mode_dims[d-1], self.sol_ranks[d])
        self.LU[d] = self.contractor.contract(
            'ij,ikl,jkm->lm', self.LU[d-1], rhsv, solv)

    def update_RU(self, d: int) -> None:
        """右側の右辺環境 ``RU[d]`` を更新する。

        Gelß (2017) の式 (4.2.19) に対応します。

        Parameters
        ----------
        d : int
            更新先の core index。``d + 1`` 番目の core を縮約に使う。
        """
        solv = self.sol_cores[d+1].reshape(self.sol_ranks[d+1],
                                           self.sol_mode_dims[d+1], self.sol_ranks[d+2])
        rhsv = self.rhs.cores[d+1].as_array().reshape(self.rhs.ranks[d+1],
                                           self.rhs.mode_dims[d+1], self.rhs.ranks[d+2])
        self.RU[d] = self.contractor.contract(
            'jkm,lm,ikl->ij', solv, self.RU[d+1], rhsv)


class ALS:
    """1 core ずつ更新する Alternating Linear Scheme solver。"""

    options: LinearSolverOptions
    contractor: CachedEinsum
    _env: _ALSEnvironment

    def __init__(
        self,
        options: LinearSolverOptions,
    ) -> None:
        """ALS solver を初期化する。

        Parameters
        ----------
        options : LinearSolverOptions
            ALS solver オプション。
        """
        self.options = options

        self.contractor = CachedEinsum()
        self._env = _ALSEnvironment(self.contractor)

    def solve(self, op: TTOperator, initial_sol: TTTensor, rhs: TTTensor) -> TTTensor:
        """ALS で TT 線形方程式を近似的に解く。

        Gelß (2017) の第 4 章および付録 A.2.2 Algorithm 11 に対応します。

        Parameters
        ----------
        op : TTOperator
            左辺の正方 TT operator。
        initial_sol : TTTensor
            初期解。
        rhs : TTTensor
            右辺 TT tensor。

        Returns
        -------
        TTTensor
            更新後の解 TT tensor。
        """
        self._env.init(op, initial_sol, rhs)
        for d in range(op.ndim-2, -1, -1):
            self._env.update_RA(d)
            self._env.update_RU(d)

        for iter in range(self.options.repeats):
            # line 5-13
            for d in range(op.ndim-1):
                # line 6-8
                if d != 0:
                    self._env.update_LA(d)
                    self._env.update_LU(d)
                # line 9 (preparetion for local_A and local_u)
                local_A = self.__calculate_local_A(d)
                local_u = self.__calculate_local_u(d)
                # line 9-12
                self.__update_core_forward(d, local_A, local_u)
            # line 14
            self._env.update_LA(op.ndim-1)
            self._env.update_LU(op.ndim-1)
            # line 15-23
            for d in range(op.ndim-1, -1, -1):
                # line 16-18
                if d != op.ndim-1:
                    self._env.update_RA(d)
                    self._env.update_RU(d)
                # line 19 (preparetion for local_A and local_u)
                local_A = self.__calculate_local_A(d)
                local_u = self.__calculate_local_u(d)
                # line 19-26
                self.__update_core_backward(d, local_A, local_u)
        return self._env.solution()

    def __calculate_local_A(self, d: int) -> LinearOperator:
        """
        scipy.sparse.linalg.lgmres に渡すための LinearOperator を返す (Matrix-free)。
        """
        env = self._env

        # 必要なテンソル
        L = env.LA[d]          # (r_prev, op_dim_L, r_curr)
        Core = env.op.cores[d] # (op_rank_L, row, col, op_rank_R)
        R = env.RA[d]          # (r_curr, op_dim_R, r_next)

        # 次元の定義
        r_prev = env.sol_ranks[d]
        r_next = env.sol_ranks[d+1]
        rows = env.op.row_dims[d]
        cols = env.op.col_dims[d]

        # 入力ベクトルと出力ベクトルの形状
        in_shape_t = (r_prev, cols, r_next)  # iln
        out_shape_t = (r_prev, rows, r_next) # kmp
        dim_in, dim_out = np.prod(in_shape_t), np.prod(out_shape_t)

        expr = self.contractor.expr(
            'ijk,jmlo,nop,iln->kmp', L.shape, Core.shape, R.shape, in_shape_t)

        def matvec(v: np.ndarray) -> np.ndarray:
            y_tensor = expr(L, Core, R, v.reshape(in_shape_t))
            return y_tensor.reshape(-1)

        return LinearOperator((dim_out, dim_in), matvec=matvec)

    def __calculate_local_u(self, d: int) -> np.ndarray:
        """
        右辺ベクトル local_u を計算する
        """
        env = self._env
        rhsv = env.rhs.cores[d].as_array().reshape(
            env.rhs.ranks[d], env.rhs.mode_dims[d], env.rhs.ranks[d+1])
        local_u = self.contractor.contract(
            'ij,ikl,lm->jkm', env.LU[d], rhsv, env.RU[d])
        return local_u.reshape(-1)

    def __update_core_forward(self, d: int, local_A: LinearOperator, local_u: np.ndarray) -> None:
        """前進 sweep で 1 core の局所問題を解き、右 rank を更新する。

        Parameters
        ----------
        d : int
            更新する core index。
        local_A : LinearOperator
            局所線形作用素。
        local_u : np.ndarray
            局所右辺ベクトル。

        Raises
        ------
        ValueError
            LGMRES が収束しない場合。
        """
        opts = self.options
        x, exitCode = lgmres(local_A, local_u, atol=opts.atol, rtol=opts.rtol)

        if exitCode != 0:
            raise ValueError("[__update_core_forward] non-convergence in the LGMRES algorithm.")
        env = self._env
        if opts.alg_decomp == 'svd':
            u, s, _ = trunc_svd(
                x.reshape(env.sol_ranks[d] * env.sol_mode_dims[d], env.sol_ranks[d + 1]),
                criterion="relative",
                threshold=opts.threshold,
                max_rank=opts.max_rank,
            )
            env.sol_ranks[d + 1] = s.shape[0]
            env.sol_cores[d] = u.reshape(
                env.sol_ranks[d], env.sol_mode_dims[d], env.sol_ranks[d + 1])
        else:
            # line 11
            [q, _] = lin.qr(
                x.reshape(env.sol_ranks[d] * env.sol_mode_dims[d], env.sol_ranks[d + 1]),
                overwrite_a=True,
                mode='economic',
                check_finite=False,
            )
            env.sol_ranks[d + 1] = q.shape[1]
            env.sol_cores[d] = q.reshape(
                env.sol_ranks[d], env.sol_mode_dims[d], env.sol_ranks[d + 1])

    def __update_core_backward(self, d: int, local_A: LinearOperator, local_u: np.ndarray) -> None:
        """後退 sweep で 1 core の局所問題を解き、左 rank を更新する。

        Parameters
        ----------
        d : int
            更新する core index。
        local_A : LinearOperator
            局所線形作用素。
        local_u : np.ndarray
            局所右辺ベクトル。

        Raises
        ------
        ValueError
            LGMRES が収束しない場合。
        """
        opts = self.options
        x, exitCode = lgmres(local_A, local_u, atol=opts.atol, rtol=opts.rtol)
        if exitCode != 0:
            raise ValueError("[__update_core_backward] non-convergence in the LGMRES algorithm.")
        env = self._env
        if opts.alg_decomp == 'svd':
            if d != 0:
                _, s, vt = trunc_svd(
                    x.reshape(env.sol_ranks[d], env.sol_mode_dims[d] * env.sol_ranks[d + 1]),
                    criterion="relative",
                    threshold=opts.threshold,
                    max_rank=opts.max_rank,
                )
                env.sol_ranks[d] = s.shape[0]
                env.sol_cores[d] = vt.reshape(
                    env.sol_ranks[d], env.sol_mode_dims[d], env.sol_ranks[d + 1])
            else:
                env.sol_cores[d] = x.reshape(
                    env.sol_ranks[d], env.sol_mode_dims[d], env.sol_ranks[d + 1])
        else:
            # line 21
            if d != 0:
                [_, q] = lin.rq(
                    x.reshape(env.sol_ranks[d], env.sol_mode_dims[d] * env.sol_ranks[d + 1]),
                    overwrite_a=True,
                    mode='economic',
                    check_finite=False,
                )
                env.sol_ranks[d] = q.shape[0]
                env.sol_cores[d] = q.reshape(
                    env.sol_ranks[d], env.sol_mode_dims[d], env.sol_ranks[d + 1])
            else:
                env.sol_cores[d] = x.reshape(
                    env.sol_ranks[d], env.sol_mode_dims[d], env.sol_ranks[d + 1])


class MALS:
    """隣接する 2 core を同時に更新する Modified ALS solver。"""

    options: LinearSolverOptions
    contractor: CachedEinsum
    _env: _ALSEnvironment

    def __init__(
        self,
        options: LinearSolverOptions,
    ) -> None:
        """MALS solver を初期化する。

        Parameters
        ----------
        options : LinearSolverOptions
            MALS solver オプション。
        """
        if options.alg_decomp != 'svd':
            raise ValueError("MALS only supports SVD-based decomposition.")
        self.options = options

        self.contractor = CachedEinsum()
        self._env = _ALSEnvironment(self.contractor)

    def solve(self, op: TTOperator, initial_sol: TTTensor, rhs: TTTensor) -> TTTensor:
        """MALS で TT 線形方程式を近似的に解く。

        Gelß (2017) の第 4 章および付録 A.2.3 Algorithm 12 に対応します。

        Parameters
        ----------
        op : TTOperator
            左辺の正方 TT operator。
        initial_sol : TTTensor
            初期解。
        rhs : TTTensor
            右辺 TT tensor。

        Returns
        -------
        TTTensor
            更新後の解 TT tensor。
        """
        self._env.init(op, initial_sol, rhs)
        # line 2-4
        for d in range(op.ndim-2, 0, -1):
            self._env.update_RA(d)
            self._env.update_RU(d)
        for iter in range(self.options.repeats):
            # line 5-13
            for d in range(op.ndim-1):
                # line 6-8
                if d != 0:
                    self._env.update_LA(d)
                    self._env.update_LU(d)
                # line 9 (preparetion for local_A and local_u)
                local_A = self.__calculate_local_A(d)
                local_u = self.__calculate_local_u(d)
                # line 9-12
                self.__update_core_forward(d, local_A, local_u)
            # line 14-23
            for d in range(op.ndim-2, -1, -1):
                # line 15-17
                if d != (op.ndim-2):
                    self._env.update_RA(d+1)
                    self._env.update_RU(d+1)
                # line 18 (preparetion for local_A and local_u)
                local_A = self.__calculate_local_A(d)
                local_u = self.__calculate_local_u(d)
                # line 18-23
                self.__update_core_backward(d, local_A, local_u)
        return self._env.solution()

    def __calculate_local_A(self, d: int) -> LinearOperator:
        """
        scipy.sparse.linalg.lgmres に渡すための LinearOperator を返す (Matrix-free)。
        """
        env = self._env

        # 必要なテンソル
        L = env.LA[d]           # (r_in, left_dim, r_out)
        Op1 = env.op.cores[d]   # (r_left, out_dim, in_dim, r_right)
        Op2 = env.op.cores[d+1] # (r_left, out_dim, in_dim, r_right)
        R = env.RA[d+1]         # (r_in, right_dim, r_out)

        # 次元の定義
        r_prev = env.sol_ranks[d]
        r_next = env.sol_ranks[d+2]
        row1 = env.op.row_dims[d]
        row2 = env.op.row_dims[d+1]
        col1 = env.op.col_dims[d]
        col2 = env.op.col_dims[d+1]

        # 入力ベクトルと出力ベクトルの形状
        in_shape_t = (r_prev, col1, col2, r_next)  # iloq
        out_shape_t = (r_prev, row1, row2, r_next) # kmps
        dim_in, dim_out = np.prod(in_shape_t), np.prod(out_shape_t)

        expr = self.contractor.expr(
            'ijk,jmln,npor,qrs,iloq->kmps', L.shape, Op1.shape, Op2.shape, R.shape, in_shape_t)

        def matvec(v: np.ndarray) -> np.ndarray:
            y_tensor = expr(L, Op1, Op2, R, v.reshape(in_shape_t))
            return y_tensor.reshape(-1)

        return LinearOperator((dim_out, dim_in), matvec=matvec)

    def __calculate_local_u(self, d: int) -> np.ndarray:
        """
        右辺ベクトル local_u を計算する
        """
        env = self._env
        rhsv1 = env.rhs.cores[d].as_array().reshape(
            env.rhs.ranks[d], env.rhs.mode_dims[d], env.rhs.ranks[d+1])
        rhsv2 = env.rhs.cores[d+1].as_array().reshape(
            env.rhs.ranks[d+1], env.rhs.mode_dims[d+1], env.rhs.ranks[d+2])
        local_u = self.contractor.contract(
            'ij,ikl,lmn,no->jkmo', env.LU[d], rhsv1, rhsv2, env.RU[d+1])
        return local_u.reshape(-1)

    def __update_core_forward(self, d: int, local_A: LinearOperator, local_u: np.ndarray) -> None:
        """前進 sweep で隣接 2 core の局所問題を解き、左 core を更新する。

        Parameters
        ----------
        d : int
            更新対象となる左側 core の index。
        local_A : LinearOperator
            局所線形作用素。
        local_u : np.ndarray
            局所右辺ベクトル。

        Raises
        ------
        ValueError
            LGMRES が収束しない場合。
        """
        opts = self.options
        x, exitCode = lgmres(local_A, local_u, atol=opts.atol, rtol=opts.rtol)
        if exitCode != 0:
            raise ValueError("[__update_core_mals_forward] non-convergence in the LGMRES algorithm.")
        env = self._env
        # line 11
        u, s, _ = trunc_svd(
            x.reshape(
                env.sol_ranks[d] * env.sol_mode_dims[d],
                env.sol_mode_dims[d + 1] * env.sol_ranks[d + 2],
            ),
            criterion="relative",
            threshold=opts.threshold,
            max_rank=opts.max_rank,
        )
        env.sol_ranks[d + 1] = s.shape[0]
        # line 12
        env.sol_cores[d] = u.reshape(
            env.sol_ranks[d], env.sol_mode_dims[d], env.sol_ranks[d + 1])

    def __update_core_backward(self, d: int, local_A: LinearOperator, local_u: np.ndarray) -> None:
        """後退 sweep で隣接 2 core の局所問題を解き、右 core を更新する。

        Parameters
        ----------
        d : int
            更新対象となる左側 core の index。
        local_A : LinearOperator
            局所線形作用素。
        local_u : np.ndarray
            局所右辺ベクトル。

        Raises
        ------
        ValueError
            LGMRES が収束しない場合。
        """
        opts = self.options
        x, exitCode = lgmres(local_A, local_u, atol=opts.atol, rtol=opts.rtol)
        if exitCode != 0:
            raise ValueError("[__update_core_mals_backward] non-convergence in the LGMRES algorithm.")
        env = self._env
        # line 20
        u, s, vt = trunc_svd(
            x.reshape(
                env.sol_ranks[d] * env.sol_mode_dims[d],
                env.sol_mode_dims[d + 1] * env.sol_ranks[d + 2],
            ),
            criterion="relative",
            threshold=opts.threshold,
            max_rank=opts.max_rank,
        )
        env.sol_ranks[d + 1] = s.shape[0]
        # line 21
        env.sol_cores[d + 1] = vt.reshape(
            env.sol_ranks[d + 1],
            env.sol_mode_dims[d + 1],
            env.sol_ranks[d + 2],
        )
        if d == 0:
            # line 23
            env.sol_cores[d] = (u.dot(np.diag(s))).reshape(
                env.sol_ranks[d],
                env.sol_mode_dims[d],
                env.sol_ranks[d + 1],
            )
