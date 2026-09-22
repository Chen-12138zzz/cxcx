# -*- coding: utf-8 -*-
"""
F1 · 未测混杂敏感性分析

【为什么这是本项目最关键的一个模块】
DML 能处理"可观测混杂"（把 X 放进 nuisance 模型就够了），
但它**完全无法处理未测混杂**。而养殖数据的未测混杂是最要命的一类：

    未测混杂 U = 虾本身的健康/摄食状态
    U → 投喂决策（摄食强 → 多投）
    U → 生长结果（健康的虾本来就长得快）
    U 无法观测（需要逐一取样、称重、镜检，散户做不到）

只要 U 存在，"投喂多 → 长得好"这个观测关联里就混了"虾本来就好"的成分，
任何只调整观测变量 X 的方法都消除不掉。

**因此唯一诚实的做法是：量化"要多强的未测混杂才能推翻结论"。**
本模块提供三种互补的量化方式：

  ① **E-value**（VanderWeele & Ding 2017）
     问题："未测混杂要与 T 和 Y 的关联强度**至少**多大，
            才能把观测到的效应完全解释掉（拉到零）？"
     输出一个数字，越大越稳健。优点：直观、可比、只需点估计。
     保守做法：用 CI 的下界（靠近零的那一端）算，"推翻"的定义更严格。

  ② **Cinelli–Hazlett 偏 R² 型敏感性分析**（Cinelli & Hazlett 2020）
     把未测混杂的强度参数化为两个偏 R²：
        ρ²_{T~U|X}  未测混杂解释了"T 的残差"多少
        ρ²_{Y~U|X,D} 未测混杂解释了"Y 的残差"多少
     输出一条等值线：在这些 (ρ²) 组合下，效应会被调整到 0 / 被减半。
     优点：与已观测变量的解释力可直接对比
     （"未测混杂要比最强观测变量还强 X 倍才能推翻"）。

  ③ **合成数据压力测试**（本项目自有，因为真值已知）
     在 synth.py 里把 u_strength 从 0 逐级调到 1，看估计的偏倚如何增长。
     这是前两种方法**无法提供**的：真值已知，可以实测"偏倚到底多大"。
     ⚠️ 但结论仅适用于**合成结构**，不得外推到真实塘口（红线 K-5）。

【使用纪律（红线 K-1 / K-3）】
  - 任何效应估计的报告**必须**附至少 ①（E-value），
    只报点估计与 p 值视为不合格报告。
  - 敏感性分析的结论表述必须是"推翻结论需要多强的未测混杂"，
    **不是**"未测混杂很弱"或"结论稳健"——后者是过度解读。
  - ② 的 ρ² 与已观测变量的 ρ² 的比较，只说明"相对强度"，
    不构成"未测混杂不存在"的任何证据。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ E-value
def e_value(est: float, se: float | None = None, ci_low: float | None = None,
            ci_high: float | None = None, rare: bool = True) -> dict:
    """计算 E-value（VanderWeele & Ding 2017）。

    公式（风险比尺度 RR）：
        E = RR + sqrt(RR * (RR − 1))
    对**非 RR 尺度**的效应（如本项目标准化后的连续 Y），
    需要先做一个近似转换：RR ≈ exp(0.91 × d)，其中 d 是标准化均差。
    ⚠️ 该转换是**近似**，仅在效应不太大时可用；本项目输出时标注为近似值。

    对任意尺度的通用做法（本项目采用）：
        RR_approx = exp(0.91 * |est|)，est 已按 Y 的标准差标准化。
    若调用方传入的是未标准化量，应自行先转换（本函数不做单位判断）。

    返回
    ----
    dict，含：
      e_value_point     —— 把点估计拉到 0 所需的混杂强度
      e_value_ci        —— 把 CI 下界（更靠近 0 的一端）拉到 0 所需的强度，
                           这是**应当报告**的那个（更保守）
      rr_approx         —— 转换后的近似 RR，供核对
      note              —— 尺度转换的说明（必须随结果一起展示）
    """
    def _ev(rr: float) -> float:
        rr = max(rr, 1.0)          # RR < 1 时先取其倒数（把效应方向规范化）
        return rr + np.sqrt(rr * (rr - 1.0))

    # 点估计：把效应按其符号方向取绝对值，转换为 RR
    rr_pt = float(np.exp(0.91 * abs(est)))
    ev_pt = _ev(rr_pt)

    ev_ci = None
    if ci_low is not None and ci_high is not None:
        # CI 中离 0 更近的那一端决定"推翻"的难度
        nearer = ci_low if abs(ci_low) < abs(ci_high) else ci_high
        # 若 CI 已跨 0，则无需任何混杂即可"推翻"（E-value 概念上为 1）
        if ci_low * ci_high <= 0:
            ev_ci = 1.0
        else:
            ev_ci = _ev(float(np.exp(0.91 * abs(nearer))))

    return {
        "e_value_point": float(ev_pt),
        "e_value_ci": (None if ev_ci is None else float(ev_ci)),
        "rr_approx": rr_pt,
        "note": ("E-value 基于 RR 尺度；本项目 Y 已标准化，"
                 "故用 RR ≈ exp(0.91 × |标准化效应|) 近似转换。"
                 "该转换在小效应下较准，效应很大时应改用原始 RR 计算。"
                 "报告应使用 e_value_ci（更保守）。"),
    }


# ------------------------------------------------ Cinelli–Hazlett 偏 R² 型分析
@dataclass
class SensitivityResult:
    """偏 R² 型敏感性分析结果。"""
    theta: float                    # 原始效应估计
    se: float
    dof: int
    r2_treated: float               # 已观测 X 对 T 的 R²（残差尺度）
    r2_outcome: float               # 已观测 X 对 Y 的 R²（残差尺度）
    robust_curve: pd.DataFrame      # 等值线：各 ρ_T 下，使 θ 恰好被拉到目标的 ρ_Y
    partial_r2_benchmarks: dict     # 与已观测变量的偏 R² 对比基准
    notes: list[str] = field(default_factory=list)

    def breakdown_at(self, target_ratio: float = 0.0) -> dict:
        """给定"要把效应拉到原值的 target_ratio 倍"，返回所需的 (ρ_T, ρ_Y)。

        target_ratio=0.0 → 完全拉到零（最强的推翻）
        target_ratio=0.5 → 减半
        """
        c = self.robust_curve
        col = f"ry_for_{target_ratio:.2f}"
        if col not in c.columns:
            raise KeyError(f"未计算 target_ratio={target_ratio}；可用："
                           f"{[x for x in c.columns if x.startswith('ry_for_')]}")
        i = int(np.argmin(np.abs(c["ry_for_" + f"{target_ratio:.2f}"].to_numpy()
                                 - self.r2_outcome)))
        return {
            "target_ratio": target_ratio,
            "required_rho_T": float(c["rho_T"].iloc[0]),
            "required_rho_Y_at_observed_level": float(c[col].iloc[0]),
            "note": ("二维等值线：给定未测混杂对 T 的偏 R²（rho_T），"
                     "需要多大的对 Y 的偏 R² 才能把效应拉到目标值。"
                     "rho_T 越大，所需的 rho_Y 越小（两者此消彼长）。"),
        }


def cih_sensitivity(theta: float, se: float, dof: int,
                    r2_treated: float, r2_outcome: float,
                    ratios: tuple[float, ...] = (0.0, 0.5, 0.8, 1.0),
                    n_grid: int = 60) -> SensitivityResult:
    """Cinelli–Hazlett 偏 R² 型敏感性分析。

    核心关系（Cinelli & Hazlett 2020, JRSS-B，简化形式）：
        设未测混杂带来的偏 R² 为 ρ_T（对 T）与 ρ_Y（对 Y），
        则调整后的效应近似为
            θ_adj ≈ θ − bias(ρ_T, ρ_Y)
        其中 bias 与 ρ_T、ρ_Y 的乘积成正比，比例系数由观测数据的
        "已解释变异的量级"决定（本项目用已观测 R² 与 se 反推该系数）。

    ⚠️ 这是一个**近似实现**：原论文给出了严格的界（bounding）公式，
    本实现用"与已观测变量同尺度"的简化参数化，目的是给出
    "未测混杂要比可观测因素强多少倍"这种**可解读**的输出，
    而**不是**给出严格的置信区间外推。
    因此本模块的输出必须在报告中标注为「近似敏感性指标」，
    不得表述为"调整后的置信区间"（红线 K-3）。
    """
    # ---- 用观测 R² 反推偏倚的比例系数 ----
    # 思路：已观测 X 已解释了 r2_treated / r2_outcome 的残差方差；
    #       未测混杂若达到同等解释力，造成的偏倚量级可由 se 与 dof 估计。
    #       系数 k 取 se * sqrt(dof) 的一个保守上界（越大 → 越保守的敏感性）。
    if se <= 0 or dof <= 0:
        raise ValueError("se 与 dof 必须为正")
    k = se * np.sqrt(dof)

    # 把 R² 解释为"残差方差的比例"，粗略对应强度：
    #   强度 ~ R² / (1 − R²)
    def strength(r2: float) -> float:
        r2 = float(np.clip(r2, 0.0, 0.999))
        return r2 / (1.0 - r2) if r2 < 1.0 else float("inf")

    s_t = strength(r2_treated)
    s_y = strength(r2_outcome)

    rho_T_grid = np.linspace(0.0, min(1.0, max(0.3, 2.5 * r2_treated)), n_grid)
    rows = []
    for rt in rho_T_grid:
        row = {"rho_T": float(rt)}
        st = strength(rt)
        for ratio in ratios:
            # 目标：θ_adj = ratio * θ  ⇒ bias = (1 − ratio) * θ
            target_bias = (1.0 - ratio) * theta
            # bias ≈ k * (已观测强度 + 未测强度) 的相对增量
            # 归一化：当未测强度与已观测同量级时，bias ≈ k
            denom = (s_t + s_y + 1e-12)
            # 需要的未测强度乘积（归一到已观测尺度）
            if abs(k) < 1e-15 or abs(denom) < 1e-15:
                needed_sy = float("nan")
            else:
                needed_sy = target_bias / k * denom / (st + 1e-12) - s_y
            # 强度 → R²
            ry = needed_sy / (1.0 + needed_sy) if np.isfinite(needed_sy) and needed_sy > -1 else float("nan")
            row[f"ry_for_{ratio:.2f}"] = float(np.clip(ry, 0.0, 1.0)) if np.isfinite(ry) else float("nan")
        rows.append(row)

    curve = pd.DataFrame(rows)

    notes = [
        "★ 本实现是对 Cinelli–Hazlett 的**近似参数化**，不是原论文的严格界；",
        "  输出应表述为『近似敏感性指标』，不得表述为调整后的置信区间。",
        f"已观测因素对 T 的解释力 R²={r2_treated:.4f}（强度 {s_t:.4f}）",
        f"已观测因素对 Y 的解释力 R²={r2_outcome:.4f}（强度 {s_y:.4f}）",
        "解读方式：看『未测混杂的强度需要达到已观测因素的几倍』，",
        "  而不是看 ρ² 的绝对值大小。",
    ]

    benchmarks = {
        "r2_treated_observed": r2_treated,
        "r2_outcome_observed": r2_outcome,
        "strength_treated": s_t,
        "strength_outcome": s_y,
        "k_scale": k,
    }

    return SensitivityResult(
        theta=theta, se=se, dof=dof,
        r2_treated=r2_treated, r2_outcome=r2_outcome,
        robust_curve=curve, partial_r2_benchmarks=benchmarks, notes=notes,
    )


# ------------------------------------------------ 汇总
def summarize_sensitivity(est, sensitivity: SensitivityResult | None,
                          e_val: dict | None) -> pd.DataFrame:
    """把敏感性分析的可用输出汇成一张表（供报告直接引用）。

    est : estimators.CausalEstimate 实例（取其 point/se/ci）
    """
    rows: list[dict] = []
    if e_val is not None:
        rows.append({"指标": "E-value（点估计）",
                     "取值": f"{e_val['e_value_point']:.3f}",
                     "含义": "把点估计拉到 0 所需的未测混杂强度（RR 尺度）"})
        if e_val["e_value_ci"] is not None:
            rows.append({"指标": "E-value（CI 近零端）★ 应报告此值",
                         "取值": f"{e_val['e_value_ci']:.3f}",
                         "含义": "把 CI 中离 0 更近的一端拉到 0 所需的强度"})
    if sensitivity is not None:
        b = sensitivity.partial_r2_benchmarks
        rows.append({"指标": "已观测因素对 T 的 R²",
                     "取值": f"{b['r2_treated_observed']:.4f}",
                     "含义": "未测混杂强度的比较基准"})
        rows.append({"指标": "已观测因素对 Y 的 R²",
                     "取值": f"{b['r2_outcome_observed']:.4f}",
                     "含义": "同上"})
        rows.append({"指标": "拉到零所需的未测混杂强度（相对基准倍数）",
                     "取值": "见 robust_curve 等值线",
                     "含义": "倍数越大 → 结论对未测混杂越不敏感"})
    return pd.DataFrame(rows)


# ------------------------------------------------ 合成数据压力测试
def stress_test_bias(est_fn, u_levels: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                     n: int = 2000, seed: int = 20260920,
                     contrast: str = "slope") -> pd.DataFrame:
    """未测混杂压力测试：逐级调高 u_strength，实测估计偏倚如何增长。

    ★ 这是本项目相对真实数据研究的**结构性优势**：
      真值已知 → 可以实测"偏倚到底多大"，而不只是给一个抽象的敏感性指标。

    ⚠️ 结论**仅适用于 synth.py 中写死的合成结构**，
      不得外推为"真实塘口未测混杂的偏倚就是这么多"（红线 K-5）。

    est_fn : 接受 DataFrame 返回 CausalEstimate 的函数
    contrast : "slope" 用真值斜率口径；"two-point" 用两点差口径
    """
    import synth

    rows = []
    for u in u_levels:
        df = synth.generate(n=n, u_strength=u, nonlinear=True, seed=seed)
        d_t = df.attrs["treat_high_mean"] - df.attrs["treat_low_mean"]
        ate2 = float(df["_true_ate"].iloc[0])
        true_val = ate2 / d_t if contrast == "slope" else ate2

        try:
            r = est_fn(df)
            est_val = r.point
            se = r.se
        except Exception as e:
            rows.append({"u_strength": u, "true": true_val, "est": np.nan,
                         "bias": np.nan, "se": np.nan, "note": f"{type(e).__name__}: {e}"})
            continue

        rows.append({
            "u_strength": u,
            "true": true_val,
            "est": est_val,
            "bias": est_val - true_val,
            "abs_bias": abs(est_val - true_val),
            "rel_bias": (est_val - true_val) / true_val if abs(true_val) > 1e-12 else np.nan,
            "se": se,
            "bias_in_se": (abs(est_val - true_val) / se) if se and se > 1e-12 else np.nan,
            "note": "",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import synth
    from estimators import estimate_ate_continuous, estimate_naive

    print("=" * 96)
    print("未测混杂敏感性分析冒烟测试")
    print("=" * 96)

    # ---- ① E-value 演示 ----
    print("\n【E-value】不同效应量级下的 E-value")
    for est in (0.05, 0.15, 0.30, 0.60):
        ev = e_value(est, ci_low=est * 0.5, ci_high=est * 1.5)
        print(f"  效应 {est:+.2f} → E-value(点)={ev['e_value_point']:.2f}"
              f"   E-value(CI近零端)={ev['e_value_ci']:.2f}")

    # ---- ② 偏 R² 型敏感性 ----
    print("\n【偏 R² 型敏感性分析】")
    df = synth.generate(n=3000, u_strength=0.0, nonlinear=True, seed=5)
    r = estimate_ate_continuous(df, engine="manual")
    cov = synth.available_covariates()

    # 计算已观测 X 对 T 与 Y 的解释力（线性近似，作为比较基准）
    from diagnostics.bad_control import _partial_r2
    Xn = df[cov].to_numpy(float)
    r2_t = _partial_r2(df["pond_area"].to_numpy(float),
                       df["feed_rate"].to_numpy(float),
                       df[[c for c in cov if c != "pond_area"]].to_numpy(float))
    # 更直接：整体 X 对 T 的 R²
    def full_r2(Xcols, target):
        A = np.column_stack([np.ones(len(target))] + [df[c].to_numpy(float) for c in Xcols])
        y = df[target].to_numpy(float)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ beta
        return 1 - float(res @ res) / float(np.sum((y - y.mean()) ** 2))
    r2_t_full = full_r2(cov, "feed_rate")
    r2_y_full = full_r2(cov, "growth")
    print(f"  已观测 X 对 T 的 R² = {r2_t_full:.4f}")
    print(f"  已观测 X 对 Y 的 R² = {r2_y_full:.4f}")

    sr = cih_sensitivity(r.point, r.se, len(df) - 5, r2_t_full, r2_y_full)
    print(f"  估计效应 θ = {r.point:+.4f} (se={r.se:.4f})")
    print("  等值线（rho_T → 把效应拉到 0/50%/80%/100% 所需的 rho_Y）：")
    for _, row in sr.robust_curve.iloc[::12].iterrows():
        print(f"    rho_T={row['rho_T']:.3f}  →  "
              + "  ".join(f"{c.split('_')[-1]}: {row[c]:.3f}"
                          for c in sr.robust_curve.columns if c.startswith("ry_for_")))
    for n_ in sr.notes:
        print(f"  {n_}")

    # ---- ③ 压力测试 ----
    print("\n【未测混杂压力测试】DML vs 朴素 OLS（斜率口径，真值固定）")
    print("  ⚠️ 结论仅适用于合成结构，不得外推真实塘口")
    st_dml = stress_test_bias(lambda d: estimate_ate_continuous(d, engine="manual"))
    st_naive = stress_test_bias(estimate_naive)
    m = st_dml.merge(st_naive, on="u_strength", suffixes=("_dml", "_naive"))
    print(f"  {'u':>5} {'真值':>9} {'DML估计':>9} {'DML偏倚':>9} {'偏倚/se':>8}"
          f" {'OLS估计':>9} {'OLS偏倚':>9}")
    for _, row in m.iterrows():
        print(f"  {row['u_strength']:>5.1f} {row['true_dml']:>+9.4f}"
              f" {row['est_dml']:>+9.4f} {row['bias_dml']:>+9.4f}"
              f" {row['bias_in_se_dml']:>8.2f}"
              f" {row['est_naive']:>+9.4f} {row['bias_naive']:>+9.4f}")
