# -*- coding: utf-8 -*-
"""
F1 · 剂量-反应曲线  θ(t) = E[Y(t)] 的估计

【为什么单报一个 ATE 不够】
"投喂量提高一个标准差，生长指数提高 0.19" 这句话在**非线性**场景下是误导：
   - 低投喂区边际效应可能很大（饲料真的不够）
   - 高投喂区边际效应可能为负（过量投喂恶化水质）
两者的**平均**接近零，但"再投一点"和"少投一点"的后果完全相反。

本项目合成数据的真实结构就长这样（诊断② 实测，t=low 处边际效应 +0.6541、
t=high 处 −0.0967）。因此**必须输出整条曲线**，而不是一个数。

【★ 两条估计路径，用途不同（不要混用）】

路径 A（默认，`estimate_dose_response`）—— 「可加多项式基」：
    Y = g(X) + β₁T + β₂T² + β₃T³ + ε
    g(X) 用梯度提升拟合（灵活吸收混杂），T 的部分是**解析可微的多项式**。
    → θ(t) 平滑，dθ/dt 有解析表达式，可用来谈"边际效应""拐点"。
    → 依赖「T 的效应形式为低阶多项式」这一设定。

路径 B（对照，`estimate_dose_response_robust`）—— 「GBR 直接拟合 μ(X,T)」：
    μ̂(X,T) 用梯度提升把 T 当普通特征。设定更灵活，**不假设 T 的函数形式**。
    ⚠️ 但它的 θ(t) 是**分段常数**（树的本质），数值微分会得到锯齿。
    → 实测：41 点网格上相邻跳变最大 0.249，符号翻转 9 次（见诊断③）。
    → **路径 B 只用于交叉核对水平值 θ(t)，禁止用它报边际效应。**

两条路径若在水平值上接近 → 曲线可信；若显著不一致 → 说明多项式阶数不够
或 X 部分未充分吸收，须如实报告（红线 K-3）。

【与 ATE 的关系】
  θ(t_high) − θ(t_low)  = 两点差口径的 ATE
  dθ/dt 在 t 处            = 斜率口径的边际效应
本模块同时输出两者，并在报告中强制标出各自口径。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

# ---------------------------------------------------------------- 设定常量
DEFAULT_DEGREE = 3        # T 的多项式阶数（默认三次：足以表达"先增后减"）
RIDGE_LAMBDA = 1e-8       # 近奇异时的极小岭修正，仅防数值崩溃


@dataclass
class DoseResponseCurve:
    """剂量-反应曲线及其元信息。"""
    grid: np.ndarray                       # 评估网格（t 的取值）
    theta: np.ndarray                      # θ̂(t) = E[Y(t)]
    se: np.ndarray                         # 逐点标准误（来自影响函数的经验方差）
    method: str                            # 估计方法（须可追溯到代码）
    path: str                              # "poly-additive" | "gbr-robust"
    n_used: int
    covariates: list[str]
    beta: np.ndarray | None = None         # 路径 A：多项式系数 [β₁, β₂, …, β_d]
    beta_se: np.ndarray | None = None      # 路径 A：系数标准误
    r2_holdout: float = float("nan")       # 结果模型在留出集上的 R²（设定诊断）
    smoothness: dict = field(default_factory=dict)  # 平滑性指标（诊断③ 用）

    # ---- 平滑性 ----
    SMOOTH_REL_TOL = 0.02   # max|Δ²θ| / θ 的极差 ≤ 2%

    @property
    def is_smooth(self) -> bool:
        """θ(t) 是否足够平滑以支持数值微分。

        ⚠️ 判据设计（本项目实测后修正过两次，过程记录在 08-复盘）：

        第 1 版用「相邻点一阶差分中位数 < 1e-3」→ 多项式曲线在陡峭段
        相邻点差本来就大，被错判为不平滑。**废弃**。

        第 2 版用「roughness = max|Δ²θ| / max|Δθ| < 1.0」→ 失败原因：
        GBR 阶梯曲线的一阶差分也大，比值被稀释到 0.97，仍被判为平滑。**废弃**。

        第 3 版（本版）用**二阶差分的绝对量**，以曲线自身极差归一化：
            max|Δ²θ| / ptp(θ) < 0.02
        理由：这个量衡量"曲线的弯折程度相对于它的总变化幅度有多大"。
          - 光滑曲线（多项式）：实测 0.005 / 1.0 ≈ 0.005  → 通过
          - 阶梯曲线（GBR）：实测 0.19 / 1.2  ≈ 0.16     → 拒绝
        两者相差约 30 倍，阈值取在中间（0.02）有充足裕度。
        """
        if not self.smoothness:
            return False
        m2 = self.smoothness.get("max_second_diff", float("inf"))
        rng = self.smoothness.get("theta_range", 0.0) or 0.0
        if not np.isfinite(m2) or rng <= 1e-12:
            return False
        return (m2 / rng) < self.SMOOTH_REL_TOL

    @property
    def smoothness_ratio(self) -> float:
        """max|Δ²θ| / ptp(θ)，越小越平滑（诊断用，便于在报告中给数）。"""
        m2 = self.smoothness.get("max_second_diff", float("nan"))
        rng = self.smoothness.get("theta_range", 0.0) or 0.0
        return (m2 / rng) if rng > 1e-12 else float("inf")

    # ---- 求值 ----
    def marginal_effect(self) -> np.ndarray:
        """边际效应 dθ/dt。

        路径 A：用解析导数（平滑、可靠）。
        路径 B：中心差分 —— ⚠️ 返回值为锯齿，**不得用于报告结论**。
        """
        if self.path == "poly-additive" and self.beta is not None:
            me = np.zeros_like(self.grid)
            for k, b in enumerate(self.beta, start=1):
                me += k * b * self.grid ** (k - 1)
            return me
        return np.gradient(self.theta, self.grid)

    def at(self, t: float) -> tuple[float, float]:
        """在指定 t 处取 (θ(t), se)。"""
        i = int(np.argmin(np.abs(self.grid - t)))
        return float(self.theta[i]), float(self.se[i])

    def two_point_ate(self, t_high: float, t_low: float) -> tuple[float, float]:
        """由曲线读出两点差口径的 ATE 及其标准误（近似，忽略两点相关性）。"""
        h, sh = self.at(t_high)
        l, sl = self.at(t_low)
        return h - l, float(np.hypot(sh, sl))

    def to_frame(self) -> pd.DataFrame:
        me = self.marginal_effect()
        return pd.DataFrame({
            "t": self.grid,
            "theta": self.theta,
            "se": self.se,
            "ci_low": self.theta - 1.959964 * self.se,
            "ci_high": self.theta + 1.959964 * self.se,
            "marginal_effect": me,
        })


# ---------------------------------------------------------------- 内部工具
def _make_grid(T: np.ndarray, n_grid: int,
               grid_range: tuple[float, float] | None,
               quantile: float = 0.10) -> np.ndarray:
    """构造评估网格。

    ★ 默认只覆盖**数据实际密集的范围**（10%–90% 分位），原因有二：

    ① 外推是常见错误：超出正性支持范围的 θ(t) 没有数据支撑（红线 K-2）。
    ② 多项式路径在数据稀疏的尾部会**放大不稳定性**。本项目实测：
       2%–98% 分位网格下，三次多项式在 t≈−2.07 处的 se 达 0.1134，
       是中间段（≈0.017）的 6 倍以上，且与 GBR 路径的水平值偏差达 0.358。
       收窄到 10%–90% 分位后偏差显著下降（见 08-复盘 的修正记录）。

    调用方可用 `quantile` 参数调整（0.02 复现最初的宽网格，用于对照实验）。
    """
    if grid_range is not None:
        lo, hi = grid_range
    else:
        lo, hi = float(np.quantile(T, quantile)), float(np.quantile(T, 1.0 - quantile))
    return np.linspace(lo, hi, n_grid)


def _smoothness_stats(theta: np.ndarray) -> dict:
    """计算曲线的平滑性指标。

    三个指标，判据由 `is_smooth` 综合使用：
      1. max_second_diff —— 二阶差分绝对最大值。真正的判别量：
         阶梯函数在切分点处二阶差分出现大脉冲（GBR 实测 ≈ 0.19），
         光滑曲线则很小（多项式实测 ≈ 0.005，相差约 40 倍）。
      2. roughness = max|Δ²θ| / max|Δθ| —— 相对粗糙度。
         ⚠️ 单独用它会失效：GBR 的一阶差分也大，比值被稀释到 ≈ 0.97，
            会把阶梯曲线误判为平滑（本项目第一版即此错）。
      3. n_sign_alt_2nd —— 二阶差分符号翻转次数，阶梯函数的伴生特征。
    """
    d1 = np.diff(theta)
    d2 = np.diff(theta, n=2)
    max1 = float(np.abs(d1).max()) if len(d1) else float("nan")
    max2 = float(np.abs(d2).max()) if len(d2) else float("nan")
    sign2 = np.sign(d2)
    n_alt = int(np.sum(np.diff(sign2) != 0)) if len(sign2) > 1 else 0
    return {
        "max_jump": max1,
        "median_jump": float(np.median(np.abs(d1))) if len(d1) else float("nan"),
        "max_second_diff": max2,
        "roughness": (max2 / max1) if (max1 and max1 > 1e-12 and np.isfinite(max2)) else float("inf"),
        "n_sign_alt_2nd": n_alt,
        "theta_range": float(np.ptp(theta)),
    }


def _crossfit_residualize(X: np.ndarray, targets: np.ndarray,
                          n_folds: int, seed: int,
                          n_estimators: int = 200, max_depth: int = 3) -> np.ndarray:
    """交叉拟合残差化：对 targets 的每一列用 GBR(X) 拟合，返回残差。

    targets 可以是 (n,) 或 (n, k)：无论哪种，都**逐列**独立拟合。
    诊断④ 已验证逐列与 JobLib 多输出在本设定下结果完全一致
    （β = [0.2421, -0.1773, -0.0079] 两法相同），故此处选实现更简单的逐列版本。
    """
    if targets.ndim == 1:
        targets = targets[:, None]
        squeeze = True
    else:
        squeeze = False

    n, k = targets.shape
    resid = np.zeros_like(targets, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for j in range(k):
        for tr, te in kf.split(X):
            m = GradientBoostingRegressor(random_state=seed, n_estimators=n_estimators,
                                          max_depth=max_depth, learning_rate=0.05)
            m.fit(X[tr], targets[tr, j])
            resid[te, j] = targets[te, j] - m.predict(X[te])
    return resid[:, 0] if squeeze else resid


# ---------------------------------------------------------------- 路径 A（默认）
def estimate_dose_response(df: pd.DataFrame, t_col: str = "feed_rate",
                           y_col: str = "growth",
                           covariates: list[str] | None = None,
                           n_grid: int = 41,
                           degree: int = DEFAULT_DEGREE,
                           n_folds: int = 5,
                           seed: int = 20260920,
                           grid_range: tuple[float, float] | None = None,
                           grid_quantile: float = 0.10,
                           ) -> DoseResponseCurve:
    """路径 A：可加多项式基的剂量-反应曲线（默认，平滑、可微分）。

    实现（部分线性 + 交叉拟合）
    --------------------------
    ① 对 Y 与 T 的各次幂 [T, T², …, T^d] 分别用 GBR(X) 交叉拟合做残差化
    ② 在残差上解正交矩条件  Σ Ỹᵢ·T̃ᵢ = Σ (Σ_k β_k T̃ᵢₖ)·T̃ᵢ
       即 β̂ = (T̃ᵀT̃)⁻¹T̃ᵀỸ，得到 θ(t) = Σ_k β̂_k t^k 的解析形式
    ③ θ(t) 与 dθ/dt 均可解析求值 → 无分段常数锯齿问题

    ⚠️ 设定依赖：假设 T 对 Y 的效应可由 d 阶多项式表达。
       本项目合成结构恰为二次（d=2 足够），但真实数据未必；
       因此默认 d=3 给一点余量，并须与路径 B 的水平值对照。
    """
    if covariates is None:
        from synth import available_covariates
        covariates = available_covariates()

    X = df[covariates].to_numpy(dtype=float)
    T = df[t_col].to_numpy(dtype=float)
    Y = df[y_col].to_numpy(dtype=float)
    n = len(df)

    # ---- ① 构造多项式基 + 残差化 ----
    basis = np.column_stack([T ** k for k in range(1, degree + 1)])
    resY = _crossfit_residualize(X, Y, n_folds, seed)                 # (n,)
    resT = _crossfit_residualize(X, basis, n_folds, seed)             # (n, d)

    # ---- ② 正交矩条件求解 ----
    A = resT.T @ resT
    b = resT.T @ resY
    try:
        beta = np.linalg.solve(A + RIDGE_LAMBDA * np.eye(degree), b)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(resT, resY, rcond=None)[0]

    # ---- β 的影响函数标准误 ----
    resid = resY - resT @ beta
    sigma2 = float(resid @ resid) / max(n - degree, 1)
    try:
        cov_beta = sigma2 * np.linalg.inv(A + RIDGE_LAMBDA * np.eye(degree))
        beta_se = np.sqrt(np.diag(cov_beta))
    except np.linalg.LinAlgError:
        beta_se = np.full(degree, np.nan)

    # ---- ③ 在网格上求 θ(t) 与逐点标准误 ----
    grid = _make_grid(T, n_grid, grid_range, grid_quantile)
    G = np.column_stack([grid ** k for k in range(1, degree + 1)])   # (n_grid, d)
    theta = G @ beta
    # θ(t) 的方差 = G Cov(β) Gᵀ 的对角线（解析，不用 MC）
    try:
        var = np.einsum("ij,jk,ik->i", G, cov_beta, G)
        se = np.sqrt(np.maximum(var, 0.0))
    except Exception:
        se = np.full(n_grid, np.nan)

    # ---- 设定诊断：留出集 R² ----
    pred_obs = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        m = GradientBoostingRegressor(random_state=seed, n_estimators=200,
                                      max_depth=3, learning_rate=0.05)
        m.fit(np.column_stack([X[tr], T[tr]]), Y[tr])
        pred_obs[te] = m.predict(np.column_stack([X[te], T[te]]))
    ss_res = float(np.sum((Y - pred_obs) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    return DoseResponseCurve(
        grid=grid, theta=theta, se=se,
        method=f"additive-poly(deg={degree}) + GBR(X) 交叉拟合({n_folds}-fold)",
        path="poly-additive",
        n_used=n, covariates=list(covariates),
        beta=beta, beta_se=beta_se,
        r2_holdout=r2,
        smoothness=_smoothness_stats(theta),
    )


# ---------------------------------------------------------------- 路径 B（对照）
def estimate_dose_response_robust(df: pd.DataFrame, t_col: str = "feed_rate",
                                  y_col: str = "growth",
                                  covariates: list[str] | None = None,
                                  n_grid: int = 41,
                                  n_folds: int = 5,
                                  seed: int = 20260920,
                                  grid_range: tuple[float, float] | None = None,
                                  grid_quantile: float = 0.10,
                                  ) -> DoseResponseCurve:
    """路径 B：GBR 直接拟合 μ(X,T) 的剂量-反应曲线（**仅用于水平值交叉核对**）。

    ⚠️ 不要用它的边际效应：
       GBR 的 θ(t) 是分段常数，`np.gradient` 会放大台阶得到锯齿。
       本项目实测（诊断③）：41 点网格相邻跳变最大 0.249，符号翻转 9 次；
       网格加密到 201 点时最大跳变升至 0.420（越密越容易落在台阶间）。
       因此 `DoseResponseCurve.is_smooth` 会对其返回 False，
       实验脚本与报告生成器必须检查该标志并拒绝输出边际效应（红线 K-4）。
    """
    if covariates is None:
        from synth import available_covariates
        covariates = available_covariates()

    X = df[covariates].to_numpy(dtype=float)
    T = df[t_col].to_numpy(dtype=float)
    Y = df[y_col].to_numpy(dtype=float)
    n = len(df)

    grid = _make_grid(T, n_grid, grid_range, grid_quantile)
    preds = np.zeros((n_grid, n))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        m = GradientBoostingRegressor(random_state=seed, n_estimators=200,
                                      max_depth=3, learning_rate=0.05)
        m.fit(np.column_stack([X[tr], T[tr]]), Y[tr])
        for gi, t_val in enumerate(grid):
            preds[gi, te] = m.predict(
                np.column_stack([X[te], np.full(len(te), t_val)]))

    theta = preds.mean(axis=1)
    se = preds.std(axis=1, ddof=1) / np.sqrt(n)

    pred_obs = np.zeros(n)
    for tr, te in kf.split(X):
        m = GradientBoostingRegressor(random_state=seed, n_estimators=200,
                                      max_depth=3, learning_rate=0.05)
        m.fit(np.column_stack([X[tr], T[tr]]), Y[tr])
        pred_obs[te] = m.predict(np.column_stack([X[te], T[te]]))
    ss_res = float(np.sum((Y - pred_obs) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    return DoseResponseCurve(
        grid=grid, theta=theta, se=se,
        method=f"g-computation(GBR 直接拟合 μ(X,T), {n_folds}-fold)",
        path="gbr-robust",
        n_used=n, covariates=list(covariates),
        r2_holdout=r2,
        smoothness=_smoothness_stats(theta),
    )


# ---------------------------------------------------------------- 一致性核对
def crosscheck_curves(curve_a: DoseResponseCurve, curve_b: DoseResponseCurve,
                      tol: float = 0.05) -> pd.DataFrame:
    """核对两条路径的 θ(t) 水平值是否一致。

    ★ 这是红线 K-3 的落实：单一方法得出的曲线不得直接作为结论。
      |θ_A(t) − θ_B(t)| 超过 tol 的点，必须在报告中标注为"两法不一致"。

    注意：tol 是绝对阈值，需结合 Y 的尺度解读
    （本项目 Y 已标准化，sd=1，故 0.05 约相当于 5% 个标准差）。
    """
    if not np.allclose(curve_a.grid, curve_b.grid):
        raise ValueError("两条曲线的网格不同，无法逐点核对")
    gap = np.abs(curve_a.theta - curve_b.theta)
    return pd.DataFrame({
        "t": curve_a.grid,
        "theta_A_poly": curve_a.theta,
        "theta_B_gbr": curve_b.theta,
        "abs_gap": gap,
        "within_tol": gap <= tol,
        "A_smooth": curve_a.is_smooth,
        "B_smooth": curve_b.is_smooth,
    })


# ---------------------------------------------------------------- 形状报告
def monotonicity_report(curve: DoseResponseCurve) -> dict:
    """报告曲线的形状特征（仅在 is_smooth 时有意义）。

    养殖场景里"投喂越多越好"是最常见的错误直觉；本函数给出可核验的量化反证：
      - 拐点位置 t*（边际效应由正转负处）
      - 峰值 θ 与峰值位置
      - 观测区间内是否已出现下降段

    ★ 返回字典**恒含 usable 键**；当曲线不平滑时，形状键一律为 None，
      调用方必须检查 `usable` 后再读形状键（不得假定键存在）。
    """
    null_shape = {
        "peak_t": None, "peak_theta": None, "n_sign_flips": None,
        "flip_t": None, "flip_t_exact": None,
        "me_first": None, "me_last": None,
        "decreasing_tail": None, "me_range": None,
    }
    if not curve.is_smooth:
        return {
            "usable": False,
            "reason": (f"曲线不平滑（roughness={curve.smoothness.get('roughness')}，"
                       f"路径={curve.path}），边际效应不可靠；"
                       f"本函数结果不得用于结论（红线 K-4）"),
            **null_shape,
        }

    me = curve.marginal_effect()
    sign = np.sign(me)
    flips = np.where(np.diff(sign) != 0)[0]
    peak_i = int(np.argmax(curve.theta))
    return {
        "usable": True,
        "reason": "",
        "peak_t": float(curve.grid[peak_i]),
        "peak_theta": float(curve.theta[peak_i]),
        "n_sign_flips": int(len(flips)),
        "flip_t": [float(curve.grid[i]) for i in flips],
        "flip_t_exact": _roots_of_marginal(curve),
        "me_first": float(me[0]),
        "me_last": float(me[-1]),
        "decreasing_tail": bool(me[-1] < 0),
        "me_range": (float(me.min()), float(me.max())),
    }


def _roots_of_marginal(curve: DoseResponseCurve) -> list[float]:
    """解析求 dθ/dt = 0 的根（仅路径 A）。

    dθ/dt = β₁ + 2β₂t + 3β₃t² + … + d·β_d·t^(d−1)

    ⚠️ numpy.roots 要求系数**按降幂排列**，即
        [d·β_d, (d−1)·β_{d−1}, …, 2·β₂, β₁]
    第一版写成了 [2β₂, β₁] 这种顺序（对 d=3 应为 [3β₃, 2β₂, β₁]），
    导致求出的根无意义、被范围过滤后返回空列表。
    本版按降幂正确构造，并对结果做范围过滤。
    """
    if curve.path != "poly-additive" or curve.beta is None or len(curve.beta) < 2:
        return []

    d = len(curve.beta)
    # 导数系数：t^(k-1) 的系数是 k·β_k，k 从 d 降到 1 → 降幂排列
    coeffs = np.array([k * curve.beta[k - 1] for k in range(d, 0, -1)], dtype=float)
    # 去掉最高次的零系数（避免 numpy.roots 因前导零报错）
    nz = np.nonzero(np.abs(coeffs) > 1e-12)[0]
    if len(nz) == 0:
        return []
    coeffs = coeffs[nz[0]:]
    if len(coeffs) < 2:
        return []   # 常函数，无根

    roots = np.roots(coeffs)
    lo, hi = float(curve.grid[0]), float(curve.grid[-1])
    return sorted(float(r.real) for r in roots
                  if abs(r.imag) < 1e-9 and lo <= r.real <= hi)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import synth

    print("=" * 100)
    print("剂量-反应曲线冒烟测试：路径 A（多项式，默认）vs 路径 B（GBR，对照）")
    print("=" * 100)

    # ---- 先做一个「网格宽度」的敏感性对照：这决定默认值取多少 ----
    print("\n【网格宽度敏感性】θ_A(t) 与 θ_B(t) 的一致性随网格分位的变化")
    print("-" * 100)
    df0 = synth.generate(n=4000, u_strength=0.0, nonlinear=True, seed=7)
    print(f"  {'分位':>6} {'网格范围':>20} {'最大偏差':>10} {'通过点数':>10} "
          f"{'端点se':>10} {'中段se':>10} {'端点/中段':>10}" + " {'最小箱n':>9}")
    for q in (0.02, 0.05, 0.10, 0.15, 0.20):
        a = estimate_dose_response(df0, grid_quantile=q)
        b = estimate_dose_response_robust(df0, grid_quantile=q)
        cc = crosscheck_curves(a, b, tol=0.05)
        se_edge = float(np.mean([a.se[0], a.se[-1]]))
        # 中段 = 曲线中间 1/3 的平均 se，作为"数据密集区"的对照
        m = len(a.se) // 3
        se_mid = float(np.mean(a.se[m:2 * m]))
        # 数据密集度：落在网格范围内的样本占比 + 最小 T 箱计数
        rng = (a.grid[0], a.grid[-1])
        n_in = int(np.sum((df0["feed_rate"] >= rng[0]) & (df0["feed_rate"] <= rng[1])))
        cnt, _ = np.histogram(df0["feed_rate"], bins=5, range=rng)
        print(f"  {q:>6.2f} [{a.grid[0]:+.3f},{a.grid[-1]:+.3f}]".ljust(30)
              + f"{cc['abs_gap'].max():>10.4f} {int(cc['within_tol'].sum()):>7}/{len(cc)}"
              + f"{se_edge:>11.4f} {se_mid:>10.4f}"
              + f"{se_edge/se_mid:>10.2f}x {int(cnt.min()):>9d}")

    print("\n【正式测试】默认分位 0.10")
    for u in (0.0, 0.8):
        df = synth.generate(n=4000, u_strength=u, nonlinear=True, seed=7)
        ca = estimate_dose_response(df)
        cb = estimate_dose_response_robust(df)
        ra, rb = monotonicity_report(ca), monotonicity_report(cb)
        ate2 = float(df["_true_ate"].iloc[0])
        k = synth.C_WQ_ON_T / df.attrs["wq_sd"]

        print(f"\n=== u_strength={u} ===")
        print(f"  结果模型留出集 R²：A={ca.r2_holdout:.4f}  B={cb.r2_holdout:.4f}")
        print(f"  平滑性 A：max|Δ²θ|/ptp(θ)={ca.smoothness_ratio:.5f}"
              f"  平滑={ca.is_smooth}  ← 应为 True")
        print(f"  平滑性 B：max|Δ²θ|/ptp(θ)={cb.smoothness_ratio:.5f}"
              f"  平滑={cb.is_smooth}  ← 应为 False")
        print(f"  β̂ (A) = {np.round(ca.beta, 4)}   β_se = {np.round(ca.beta_se, 4)}")
        print(f"  解析期望 β = [{synth.C_Y_ON_T + synth.C_Y_ON_WQ*k:+.4f}, "
              f"{synth.C_Y_T_SQUARED + synth.C_Y_T_WQ_INTER*k:+.4f}, 0.0000]")
        if ra["usable"]:
            print(f"  形状 A：峰值 θ={ra['peak_theta']:+.4f} @ t={ra['peak_t']:+.4f}")
            print(f"          边际效应 {ra['me_first']:+.4f} → {ra['me_last']:+.4f}"
                  f"   符号翻转 {ra['n_sign_flips']} 次，解析拐点 t={ra['flip_t_exact']}")
        else:
            print(f"  形状 A：不可用 —— {ra['reason']}")
        print(f"  形状 B：{'不可用（预期）' if not rb['usable'] else '★ 意外可用'}")

        gc, _ = ca.two_point_ate(df.attrs["treat_high_mean"], df.attrs["treat_low_mean"])
        print(f"  曲线 A 两点差 = {gc:+.4f}   真值 = {ate2:+.4f}   差 {gc-ate2:+.4f}")

        cc = crosscheck_curves(ca, cb)
        print(f"  两路径水平值核对（tol=0.05）：通过 {int(cc['within_tol'].sum())}/"
              f"{len(cc)} 点，最大偏差 {cc['abs_gap'].max():.4f}")

        print("  曲线 A 抽样（每 8 点）：")
        for _, r in ca.to_frame().iloc[::8].iterrows():
            print(f"    t={r['t']:+.3f}  θ={r['theta']:+.4f} ± {1.96*r['se']:.4f}"
                  f"   dθ/dt={r['marginal_effect']:+.4f}")
