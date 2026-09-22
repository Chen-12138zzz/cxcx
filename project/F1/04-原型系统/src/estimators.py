# -*- coding: utf-8 -*-
"""
F1 · 主估计器：双重机器学习（DML）与对照估计器

【本模块回答什么】
给定观测数据 (W, S, T, Y)，估计干预 T 对结果 Y 的**总效应**。
估计目标（见 03-技术方案/核心算法设计说明.md 第 2 节）：
    - ATE（平均处理效应）
    - 连续干预下的剂量-反应曲线  θ(t) = E[Y(t)]
（CATE 在 cate.py，诊断在 diagnostics/）

【为什么用 DML 而不是直接回归】
直接回归 Y ~ T + W 在两种情况下不一致：
  ① T 由 W 与未测 U 共同决定 ⇒ T 与误差项相关（内生性）
  ② Y 对 T 的关系非线性（阈值、饱和）⇒ 线性模型设定误差被误当成"效应"

DML 的做法（Chernozhukov et al. 2018）：
  ① 用机器学习分别拟合  E[T|X]  与  E[Y|X]     （X = 可用协变量）
  ② 取残差  T̃ = T − Ê[T|X]， Ỹ = Y − Ê[Y|X]
  ③ 在残差上做正交矩条件：  解  Σ ψ(W; θ, η̂) = 0
  ④ 交叉拟合（cross-fitting）消除"用同一批数据既拟合又估计"的过拟合偏倚

【DML 说清了什么、没说清什么 —— 必须记住】
DML 解决的是 **①**（用灵活的 nuisance 函数吸收混杂），
**不解决**未测混杂（U 未被观测就不可能被调整）。
因此本项目把 **敏感性分析** 作为一等公民（diagnostics/sensitivity.py），
并把"无未测混杂"列为**可检验但不可证明**的假设（台账 A1）。

【实现说明】
主实现依赖 doubleml 包（PartiallyLinearRegression / IRM）。
同时提供**自实现版本** `_dml_plr_manual`，用于：
  - doubleml 不可用时降级
  - 交叉验证两者结果一致（防止"调包却不知其行为"）
自实现版本不自称"更优"，仅作为对照。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:  # 主实现
    from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
    HAS_DOUBLEML = True
except Exception:  # pragma: no cover
    HAS_DOUBLEML = False

try:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import KFold
    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    HAS_SKLEARN = False


# ------------------------------------------------------------------ 结果容器
@dataclass
class CausalEstimate:
    """一次因果估计的结果与元信息。

    ★ 纪律：任何对外报告的效应值必须携带本容器中的
      `assumptions`（依赖的识别假设）、`n_used`、`method` 字段，
      禁止只报一个孤立的数字（红线 K-1）。
    """
    target: str                       # "ATE(slope)" | "ATE(two-point)" | "theta(t)" | "CATE"
    point: float                      # 点估计
    se: float | None = None           # 标准误
    ci_low: float | None = None
    ci_high: float | None = None
    method: str = ""                  # 估计器名称（须可追溯到代码）
    n_used: int = 0                   # 实际进入估计的样本数
    covariates: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # ★ 口径元信息（防止"斜率"与"两点差"被混为一谈）
    contrast_kind: str = "slope"      # "slope" = 每单位 T 的边际效应
                                      # "two-point" = 两个 T 取值上反事实结果的差
    t_ref: tuple[float, float] | None = None  # two-point 口径下的 (high, low)
    t_at: float | None = None         # slope 口径下求导/局部线性化的参考点

    @property
    def ci95(self) -> tuple[float, float] | None:
        if self.ci_low is None or self.ci_high is None:
            return None
        return (self.ci_low, self.ci_high)

    def rescale_to_twopoint(self, d_t: float) -> "CausalEstimate":
        """把【斜率口径】的结果换算为【两点差口径】：θ_slope × (high − low)。

        ⚠️ 这个换算**只在效应近似线性的区间内成立**。
        本项目合成数据在高投喂区效应已转负（边际效应 −0.0967 < 0），
        因此换算值与真实两点差会有系统性偏差 —— 该偏差本身就是
        "为什么不能只报一个数"的证据，必须随结果一起报告，
        不得只报换算后的值。
        """
        return CausalEstimate(
            target=self.target.replace("(slope)", "(two-point, rescaled)"),
            point=self.point * d_t,
            se=None if self.se is None else self.se * abs(d_t),
            ci_low=None if self.ci_low is None else self.ci_low * d_t,
            ci_high=None if self.ci_high is None else self.ci_high * d_t,
            method=self.method + f" | ×d_t({d_t:+.4f})",
            n_used=self.n_used, covariates=list(self.covariates),
            assumptions=list(self.assumptions) + ["效应在区间内近似线性（本项目不成立）"],
            notes=list(self.notes) + ["★ 换算值仅在效应近线性时有效；高投喂区效应已转负，换算有偏"],
            contrast_kind="two-point",
            t_ref=self.t_ref, t_at=self.t_at,
        )

    def to_row(self) -> dict:
        return {
            "target": self.target,
            "contrast_kind": self.contrast_kind,
            "point": self.point,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "t_ref": "" if self.t_ref is None else f"{self.t_ref[0]:.4f}|{self.t_ref[1]:.4f}",
            "t_at": self.t_at,
            "method": self.method,
            "n_used": self.n_used,
            "covariates": "|".join(self.covariates),
            "assumptions": "|".join(self.assumptions),
        }


# 本估计流程恒定的识别假设声明（写死在代码里，防止遗忘）
BASE_ASSUMPTIONS = [
    "A1 无未测混杂（不可检验，仅能做敏感性分析）",
    "A2 正性/重叠（可诊断，见 diagnostics/positivity.py）",
    "A3 一致性 / SUTVA（假设塘口间无干扰）",
    "A4 调整集不含中介（water_degrad）",
    "A5 调整集不含碰撞（fcr）",
    "A6 投喂记录为实际值而非计划值（待现场核实）",
]


def _nuisance_models(kind: str = "flex", seed: int = 20260920):
    """返回 (Y 的 nuisance 模型, T 的 nuisance 模型)。

    kind="flex"  → 梯度提升（捕捉非线性，本项目默认）
    kind="linear" → 线性/逻辑回归（用于对照：验证"非线性设定是否重要"）
    """
    if kind == "linear":
        return LinearRegression(), LinearRegression()
    if not HAS_SKLEARN:
        raise RuntimeError("sklearn 不可用，无法构造 nuisance 模型")
    return (
        GradientBoostingRegressor(random_state=seed, n_estimators=200,
                                  max_depth=3, learning_rate=0.05),
        GradientBoostingRegressor(random_state=seed, n_estimators=200,
                                  max_depth=3, learning_rate=0.05),
    )


# ------------------------------------------------------- 自实现 DML（PLR，连续 T）
def _dml_plr_manual(df: pd.DataFrame, t_col: str, y_col: str,
                    covariates: list[str], n_folds: int = 5,
                    kind: str = "flex", seed: int = 20260920) -> CausalEstimate:
    """手写 DML-PLR（部分线性模型），用于与 doubleml 结果交叉验证。

    部分线性模型：  Y = θ·T + g(X) + ε,   E[ε|T,X] = 0

    正交矩条件（Neyman 正交，对 nuisance 的一阶误差不敏感）：
        ψ = (Ỹ − θ·T̃)·T̃
        θ̂ = Σ T̃ᵢỸᵢ / Σ T̃ᵢ²

    交叉拟合：把样本分成 K 折，第 k 折的残差用**其余 K−1 折**训练的模型预测，
    避免过拟合导致的残差人为变小（进而 θ̂ 偏倚）。
    """
    X = df[covariates].to_numpy(dtype=float)
    T = df[t_col].to_numpy(dtype=float)
    Y = df[y_col].to_numpy(dtype=float)
    n = len(df)

    t_res = np.zeros(n)
    y_res = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        _, m_t = _nuisance_models(kind, seed)
        m_y, _ = _nuisance_models(kind, seed)
        m_t.fit(X[tr], T[tr])
        m_y.fit(X[tr], Y[tr])
        t_res[te] = T[te] - m_t.predict(X[te])
        y_res[te] = Y[te] - m_y.predict(X[te])

    denom = float(np.sum(t_res ** 2))
    if abs(denom) < 1e-12:
        raise RuntimeError("正交化后干预残差方差为 0：调整集可能已完全解释干预（违反正性）")
    theta = float(np.sum(t_res * y_res) / denom)

    # 影响函数标准误（PLR 的解析形式）
    psi = (y_res - theta * t_res) * t_res
    J = float(np.mean(t_res ** 2))
    se = float(np.sqrt(np.mean(psi ** 2)) / (abs(J) * np.sqrt(n)))
    # 用影响函数更稳的写法：se = sd(psi)/ (sqrt(n) * J)
    se = float(np.std(psi, ddof=1) / (np.sqrt(n) * abs(J)))

    return CausalEstimate(
        target="ATE(slope)", point=theta, se=se,
        ci_low=theta - 1.959964 * se, ci_high=theta + 1.959964 * se,
        method=f"manual_DML_PLR({kind}, {n_folds}-fold)",
        n_used=n, covariates=list(covariates),
        assumptions=list(BASE_ASSUMPTIONS),
        notes=["部分线性设定：假定 T 对 Y 的效应以加性 θ·T 进入",
               "★ 口径 = 斜率（每单位 T 的边际效应），不是两点差"],
        contrast_kind="slope", t_at=0.0,
    )


# ------------------------------------------------------- doubleml 主实现
def estimate_ate_continuous(df: pd.DataFrame, t_col: str = "feed_rate",
                            y_col: str = "growth",
                            covariates: list[str] | None = None,
                            n_folds: int = 5, kind: str = "flex",
                            seed: int = 20260920,
                            engine: str = "auto") -> CausalEstimate:
    """估计连续干预的 ATE（以干预的 1 个标准差为单位）。

    engine:
      "auto"     → 优先 doubleml，失败则自实现
      "doubleml" → 强制 doubleml
      "manual"   → 强制自实现（用于对照验证）
    """
    if covariates is None:
        from synth import available_covariates
        covariates = available_covariates()

    if engine == "manual":
        return _dml_plr_manual(df, t_col, y_col, covariates, n_folds, kind, seed)

    if engine in ("auto", "doubleml"):
        if not HAS_DOUBLEML:
            if engine == "doubleml":
                raise RuntimeError("doubleml 不可用；请改用 engine='manual'")
            warnings.warn("doubleml 不可用，降级到自实现版本", RuntimeWarning)
            return _dml_plr_manual(df, t_col, y_col, covariates, n_folds, kind, seed)

        data = DoubleMLData.from_arrays(
            x=df[covariates].to_numpy(dtype=float),
            y=df[y_col].to_numpy(dtype=float),
            d=df[t_col].to_numpy(dtype=float),
        )
        ml_l, ml_m = _nuisance_models(kind, seed)
        # ⚠️ doubleml ≥ 0.9 的签名已改为 ml_l（线性部分 E[Y|X]）/ ml_m（干预部分 E[D|X]），
        #    旧文档里的 ml_g 会报 missing positional argument 'ml_l'。
        dml = DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m,
                          n_folds=n_folds, score="partialling out")
        dml.fit()
        theta = float(dml.coef[0])
        se = float(dml.se[0])
        return CausalEstimate(
            target="ATE(slope)", point=theta, se=se,
            ci_low=float(dml.confint().iloc[0, 0]),
            ci_high=float(dml.confint().iloc[0, 1]),
            method=f"doubleml_PLR(partialling out, {kind}, {n_folds}-fold)",
            n_used=len(df), covariates=list(covariates),
            assumptions=list(BASE_ASSUMPTIONS),
            notes=["部分线性设定；score=partialling out",
                   "★ 口径 = 斜率（每单位 T 的边际效应），不是两点差"],
            contrast_kind="slope", t_at=0.0,
        )

    raise ValueError(f"未知 engine: {engine}")


# ------------------------------------------------------- 二值化干预（IRM/AIPW）
def _aipw_binary(df: pd.DataFrame, d_col: str, y_col: str,
                 covariates: list[str], n_folds: int = 5,
                 kind: str = "flex", seed: int = 20260920) -> CausalEstimate:
    """手写 AIPW（增广逆概率加权），二值干预的双稳健估计。

    AIPW 的影响函数（对结果回归与倾向得分都只要求一阶正确 → 双稳健）：
        ψᵢ = [ Dᵢ(Yᵢ−μ̂₁(Xᵢ))/ê(Xᵢ) + μ̂₁(Xᵢ) ]
             − [ (1−Dᵢ)(Yᵢ−μ̂₀(Xᵢ))/(1−ê(Xᵢ)) + μ̂₀(Xᵢ) ]  − θ
    """
    if not HAS_SKLEARN:
        raise RuntimeError("sklearn 不可用")
    X = df[covariates].to_numpy(dtype=float)
    D = df[d_col].to_numpy(dtype=int)
    Y = df[y_col].to_numpy(dtype=float)
    n = len(df)

    mu1 = np.zeros(n)
    mu0 = np.zeros(n)
    e = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        if kind == "linear":
            m1, m0 = LogisticRegression(max_iter=1000), LogisticRegression(max_iter=1000)
        else:
            m1 = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3)
            m0 = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3)
        if len(np.unique(D[tr])) < 2:
            # 训练折中只有一个处理水平 → 该折无法估倾向得分（正性严重违背）
            warnings.warn("训练折中处理变量无变异，跳过该折", RuntimeWarning)
            mu1[te] = np.nan
            mu0[te] = np.nan
            e[te] = np.nan
            continue
        m1.fit(X[tr][D[tr] == 1], Y[tr][D[tr] == 1])
        m0.fit(X[tr][D[tr] == 0], Y[tr][D[tr] == 0])
        idx_ = tr
        if kind == "linear":
            ps = LogisticRegression(max_iter=1000).fit(X[idx_], D[idx_])
        else:
            ps = GradientBoostingClassifier(random_state=seed, n_estimators=200, max_depth=3)
            ps.fit(X[idx_], D[idx_])
        mu1[te] = m1.predict(X[te])
        mu0[te] = m0.predict(X[te])
        e[te] = ps.predict_proba(X[te])[:, 1]

    ok = ~np.isnan(e)
    if ok.sum() < n * 0.5:
        raise RuntimeError("超过一半样本因正性违背被丢弃，估计不可信")
    Xo, Do, Yo, mu1o, mu0o, eo = X[ok], D[ok], Y[ok], mu1[ok], mu0[ok], e[ok]

    e_clip = np.clip(eo, 1e-3, 1 - 1e-3)   # ★ 裁剪：未裁剪时权重爆炸、方差无穷
    psi_i = (Do * (Yo - mu1o) / e_clip + mu1o
             - ((1 - Do) * (Yo - mu0o) / (1 - e_clip) + mu0o))
    theta = float(np.mean(psi_i))
    se = float(np.std(psi_i - theta, ddof=1) / np.sqrt(len(psi_i)))

    return CausalEstimate(
        target="ATE(two-point)", point=theta, se=se,
        ci_low=theta - 1.959964 * se, ci_high=theta + 1.959964 * se,
        method=f"manual_AIPW({kind}, {n_folds}-fold)",
        n_used=int(ok.sum()), covariates=list(covariates),
        assumptions=list(BASE_ASSUMPTIONS),
        notes=[f"倾向得分裁剪到 [{1e-3:.3f}, {1-1e-3:.3f}]",
               f"因正性/无变异丢弃 {int((~ok).sum())} 个样本",
               "★ 口径 = 两点差（高投喂组均值 vs 低投喂组均值）"],
        contrast_kind="two-point",
    )


def estimate_ate_binary(df: pd.DataFrame, d_col: str = "feed_high",
                        y_col: str = "growth",
                        covariates: list[str] | None = None,
                        n_folds: int = 5, kind: str = "flex",
                        seed: int = 20260920,
                        engine: str = "auto") -> CausalEstimate:
    """估计二值干预（高投喂组 vs 低投喂组）的 ATE。

    ★ 与连续版的区别（必须在报告中说明）：
      连续版估的是"投喂提高 1 个标准差"的效应，二值版估的是
      "实际高投喂组均值 vs 低投喂组均值"的效应。两者**不是同一个量**，
      因为真实存在的组均值差 ≠ 1 个标准差（见 05-验证 的尺度对照表）。
    """
    if covariates is None:
        from synth import available_covariates
        covariates = available_covariates()

    if engine == "manual" or not HAS_DOUBLEML:
        return _aipw_binary(df, d_col, y_col, covariates, n_folds, kind, seed)

    data = DoubleMLData.from_arrays(
        x=df[covariates].to_numpy(dtype=float),
        y=df[y_col].to_numpy(dtype=float),
        d=df[d_col].to_numpy(dtype=int),
    )
    ml_l = (GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3)
            if kind != "linear" else LinearRegression())
    ml_m = (GradientBoostingClassifier(random_state=seed, n_estimators=200, max_depth=3)
            if kind != "linear" else LogisticRegression(max_iter=1000))
    dml = DoubleMLIRM(data, ml_g=ml_l, ml_m=ml_m, n_folds=n_folds, score="ATE")
    dml.fit()
    theta = float(dml.coef[0])
    se = float(dml.se[0])
    return CausalEstimate(
        target="ATE(two-point)", point=theta, se=se,
        ci_low=float(dml.confint().iloc[0, 0]),
        ci_high=float(dml.confint().iloc[0, 1]),
        method=f"doubleml_IRM(ATE, {kind}, {n_folds}-fold)",
        n_used=len(df), covariates=list(covariates),
        assumptions=list(BASE_ASSUMPTIONS),
        notes=["★ 口径 = 两点差（高投喂组均值 vs 低投喂组均值）"],
        contrast_kind="two-point",
    )


# ------------------------------------------------------- 朴素对照（用于展示偏倚）
def estimate_naive(df: pd.DataFrame, t_col: str = "feed_rate",
                   y_col: str = "growth") -> CausalEstimate:
    """不做任何调整的朴素估计（回归 Y ~ T），用于**展示偏倚方向与量级**。

    ★ 它存在的唯一目的是作为"错误做法的对照项"进入实验结果表，
      不得作为研究结论报告。
    """
    T = df[t_col].to_numpy(dtype=float)
    Y = df[y_col].to_numpy(dtype=float)
    A = np.column_stack([np.ones(len(T)), T])
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ beta
    n, k = A.shape
    s2 = float(resid @ resid) / (n - k)
    cov = s2 * np.linalg.inv(A.T @ A)
    se = float(np.sqrt(cov[1, 1]))
    return CausalEstimate(
        target="ATE(naive)", point=float(beta[1]), se=se,
        ci_low=float(beta[1]) - 1.959964 * se, ci_high=float(beta[1]) + 1.959964 * se,
        method="naive_OLS(Y~T)",
        n_used=n, covariates=[],
        assumptions=["无（未做任何混杂调整；未测混杂全额进入估计）"],
        notes=["★ 仅作偏倚对照，不得作为研究结论",
               "★ 口径 = 斜率（回归系数），与 DML 同为斜率口径才能公平对比"],
        contrast_kind="slope", t_at=0.0,
    )


def estimate_adjusted_ols(df: pd.DataFrame, t_col: str = "feed_rate",
                          y_col: str = "growth",
                          covariates: list[str] | None = None) -> CausalEstimate:
    """线性回归调整所有协变量（Y ~ T + X），作为 DML 的第二个对照。

    与 DML 的差别：
      - 不交叉拟合 ⇒ nuisance 过拟合会带来正则化偏倚
      - 线性设定 ⇒ 若真实关系非线性，效应估计被设定误差污染
    两者结果差多少，本身就是"为什么要用 DML"的量化证据。
    """
    if covariates is None:
        from synth import available_covariates
        covariates = available_covariates()
    X = df[covariates].to_numpy(dtype=float)
    T = df[t_col].to_numpy(dtype=float)
    Y = df[y_col].to_numpy(dtype=float)
    A = np.column_stack([np.ones(len(T)), T, X])
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ beta
    n, k = A.shape
    s2 = float(resid @ resid) / (n - k)
    cov = s2 * np.linalg.inv(A.T @ A)
    se = float(np.sqrt(cov[1, 1]))
    return CausalEstimate(
        target="ATE(OLS-adjusted)", point=float(beta[1]), se=se,
        ci_low=float(beta[1]) - 1.959964 * se, ci_high=float(beta[1]) + 1.959964 * se,
        method=f"OLS(Y~T+{'+'.join(covariates)})",
        n_used=n, covariates=list(covariates),
        assumptions=list(BASE_ASSUMPTIONS) + ["线性设定（T 与 Y 关系为线性）"],
        notes=["★ 对照项：用于量化『不做正交化/不交叉拟合』的代价",
               "★ 口径 = 斜率，但为【线性设定下的恒定斜率】，与真实随 t 变化的边际效应不等价"],
        contrast_kind="slope", t_at=0.0,
    )


if __name__ == "__main__":  # 手动冒烟测试
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import synth

    print("=" * 92)
    print("估计器冒烟测试：四种估计器 × 两个未测混杂档位")
    print("★ 注意：DML/OLS/naive 报的是【斜率口径】，真值 _true_ate 是【两点差口径】")
    print("  两者不可直接比 —— 需先除以 d_t = treat_high_mean − treat_low_mean")
    print("=" * 92)

    for u in (0.0, 0.8):
        df = synth.generate(n=2000, u_strength=u, nonlinear=True, seed=7)
        ate2 = float(df["_true_ate"].iloc[0])
        d_t = df.attrs["treat_high_mean"] - df.attrs["treat_low_mean"]
        print(f"\n=== u_strength={u} ===")
        print(f"  真值（两点差口径）= {ate2:+.4f}      d_t = {d_t:+.4f}")
        print(f"  真值（斜率口径）  = {ate2/d_t:+.4f}   ← 这才是与下面各条可比的量")
        rows = []
        for name, fn in (("naive", estimate_naive),
                         ("ols_adj", estimate_adjusted_ols),
                         ("dml_doubleml", lambda d: estimate_ate_continuous(d, engine="doubleml")),
                         ("dml_manual", lambda d: estimate_ate_continuous(d, engine="manual")),
                         ("aipw_binary", lambda d: estimate_ate_binary(d, engine="manual"))):
            try:
                r = fn(df)
                rows.append((name, r.point, r.se, r.contrast_kind, r.method))
            except Exception as e:
                rows.append((name, None, None, "-", f"失败 {type(e).__name__}: {e}"))
        for name, pt, se, ck, m in rows:
            if pt is None:
                print(f"  {name:14s} {m}")
            else:
                # 斜率口径 → 换算成两点差以便与真值同尺度
                conv = pt * d_t if ck == "slope" else pt
                print(f"  {name:14s} {pt:+.4f} (se={se:.4f}) → 两点差口径 {conv:+.4f}"
                      f"  [{ck}]  {m}")
