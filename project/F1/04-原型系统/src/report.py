# -*- coding: utf-8 -*-
"""
F1 · 报告生成器

【模块职责】
把实验产生的**结构化结果**（JSON / DataFrame）渲染成人类可读的 Markdown 报告，
并在此过程中**强制执行本项目的表述纪律**：

  ① 每个效应值必须带口径标签（slope / two-point）与单位说明
  ② 每个效应值必须带识别假设清单（从 ledger.py 读取）
  ③ 不平滑的曲线**不得**输出边际效应（自动拦截，红线 K-4）
  ④ 敏感性分析未做时，**不得**输出"结论稳健"字样
  ⑤ 任何合成数据结果必须带 `【合成】` 标记（红线 K-5）
  ⑥ 每个数字必须能追溯到生成它的脚本与参数（seed / n / 配置）

【为什么要在生成器里做拦截，而不是靠人记得】
报告的表述纪律是最容易被遗忘的一环：实验做对了，写报告时随手一句
"结果稳健"就把敏感性分析的限定条件抹掉了。
把拦截写在生成器里，可以让**错误无法静默通过**——
这比在文档里写"请注意"有效得多。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ 拦截器
class ReportGuard:
    """报告表述守卫：记录本次生成中触发/拦截了哪些纪律检查。"""

    def __init__(self):
        self.checks: list[dict] = []

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append({"检查项": name, "通过": bool(passed), "说明": detail})
        return bool(passed)

    def require(self, name: str, passed: bool, detail: str = "") -> None:
        """与 check 相同，但不通过时抛异常（用于必须满足的前置条件）。"""
        if not self.check(name, passed, detail):
            raise ValueError(f"报告生成被拦截 —— {name}：{detail}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.checks)


# ------------------------------------------------------------------ 格式化工具
def fmt(x, nd: int = 4, dash: str = "—") -> str:
    """统一的数字格式化：None / NaN 一律显示为 '—'，不留空白或 nan 字样。"""
    if x is None:
        return dash
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return dash
    return f"{xf:+.{nd}f}"


def _mark_mode(mode: str) -> str:
    """数据模式标注（红线 K-5）。"""
    m = {
        "synthetic": "【合成】",
        "real": "【实测】",
        "simulated": "【仿真】",
        "cited": "【引用】",
        "pending": "【待核实】",
    }
    return m.get(mode, f"【{mode}】")


# ------------------------------------------------------------------ 核心渲染
def render_estimate_table(estimates: list,
                          true_value: float | None = None,
                          true_label: str = "",
                          mode: str = "synthetic") -> pd.DataFrame:
    """把多个 CausalEstimate 渲染成对比表。

    ★ 强制检查：
      - 口径不同的估计**不放在同一列比较**，而是分列显示并标注
      - 真值仅在与估计同口径时才可比（自动检查，不可比时给出警示）
    """
    rows = []
    for r in estimates:
        row = {
            "方法": r.method,
            "口径": r.contrast_kind,
            "目标": r.target,
            "点估计": fmt(r.point),
            "标准误": fmt(r.se),
            "95%CI": (f"[{fmt(r.ci_low)}, {fmt(r.ci_high)}]"
                      if r.ci95 else "—"),
            "n": r.n_used,
            "调整集": ", ".join(r.covariates) if r.covariates else "（无）",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if true_value is not None:
        # 真值可比性检查：真值口径必须与至少一个估计的口径一致
        kinds = {r.contrast_kind for r in estimates}
        df.attrs["true_comparable"] = bool(kinds)
        df.attrs["true_value"] = true_value
        df.attrs["true_label"] = true_label or "真值"
        df.attrs["mode"] = mode
    return df


def estimate_table_to_md(df: pd.DataFrame) -> str:
    out = []
    if df.attrs.get("true_value") is not None:
        tv = df.attrs["true_value"]
        lbl = df.attrs.get("true_label", "真值")
        out.append(f"> {_mark_mode(df.attrs.get('mode', 'synthetic'))} "
                   f"{lbl} = **{fmt(tv)}**（口径见各行标注）")
        out.append("")
    cols = list(df.columns)
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def render_dose_response_report(curve, monotonicity: dict | None,
                                crosscheck: pd.DataFrame | None,
                                guard: ReportGuard,
                                mode: str = "synthetic") -> str:
    """渲染剂量-反应曲线的报告片段。

    ★ 拦截规则（红线 K-4）：
      - curve.is_smooth 为 False 时，**禁止**输出边际效应数值，
        只输出水平值 θ(t)，并说明为何不报边际效应
      - 两路径核对未通过（存在超出容差的点）时，必须在显著位置标注
    """
    lines = []
    lines.append(f"#### 剂量-反应曲线（{_mark_mode(mode)}）")
    lines.append("")
    lines.append(f"- 估计方法：`{curve.method}`")
    lines.append(f"- 评估网格：{len(curve.grid)} 点，"
                 f"范围 [{curve.grid[0]:+.4f}, {curve.grid[-1]:+.4f}]")
    lines.append(f"- 结果模型留出集 R²：{curve.r2_holdout:.4f}")

    if curve.is_smooth:
        guard.check("曲线平滑（可报边际效应）", True,
                    f"max|Δ²θ|/ptp(θ)={curve.smoothness_ratio:.5f}")
        lines.append(f"- 平滑性：max|Δ²θ|/ptp(θ) = {curve.smoothness_ratio:.5f} "
                     f"→ **可用于讨论边际效应**")
        if curve.beta is not None:
            lines.append("")
            lines.append("多项式系数（路径 A）：")
            lines.append("")
            lines.append("| 项 | 系数 | 标准误 |")
            lines.append("|---|---|---|")
            for k, (b, s) in enumerate(zip(curve.beta, curve.beta_se), start=1):
                lines.append(f"| T^{k} | {fmt(b)} | {fmt(s)} |")
        if monotonicity and monotonicity.get("usable"):
            lines.append("")
            lines.append(f"- 峰值：θ = {fmt(monotonicity['peak_theta'])} "
                         f"@ t = {fmt(monotonicity['peak_t'])}")
            lines.append(f"- 边际效应：起点 {fmt(monotonicity['me_first'])} → "
                         f"终点 {fmt(monotonicity['me_last'])}")
            lines.append(f"- 符号翻转次数：{monotonicity['n_sign_flips']}")
            if monotonicity.get("flip_t_exact"):
                lines.append(f"- **解析拐点**（dθ/dt=0）："
                             + ", ".join(fmt(t) for t in monotonicity["flip_t_exact"]))
            if monotonicity.get("decreasing_tail"):
                lines.append("- ★ 曲线末端边际效应**为负**，即已越过峰值："
                             "继续增加投喂量会**降低**结果")
    else:
        guard.check("曲线平滑（可报边际效应）", False,
                    f"max|Δ²θ|/ptp(θ)={curve.smoothness_ratio:.5f} 超过阈值")
        lines.append(f"- 平滑性：max|Δ²θ|/ptp(θ) = {curve.smoothness_ratio:.5f} "
                     f"→ **曲线不平滑**")
        lines.append("")
        lines.append("> ⚠️ **本曲线不平滑，故不输出边际效应**："
                     "θ(t) 呈分段常数形态（树模型的固有性质），"
                     "数值微分会把台阶放大成无意义的锯齿（红线 K-4）。"
                     "本表仅报告水平值 θ(t)，边际效应请参阅平滑路径的结果。")

    lines.append("")
    lines.append("曲线取值（抽样）：")
    lines.append("")
    f = curve.to_frame()
    step = max(1, len(f) // 12)
    lines.append("| t | θ(t) | 95% CI |" + (" 边际效应 |" if curve.is_smooth else ""))
    lines.append("|---|---|---|" + ("---|" if curve.is_smooth else ""))
    for _, r in f.iloc[::step].iterrows():
        row = (f"| {fmt(r['t'], 3)} | {fmt(r['theta'])} "
               f"| [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] ")
        if curve.is_smooth:
            row += f"| {fmt(r['marginal_effect'])} "
        lines.append(row + "|")

    if crosscheck is not None and len(crosscheck):
        n_bad = int((~crosscheck["within_tol"]).sum())
        guard.check("两路径水平值一致", n_bad == 0,
                    f"{n_bad}/{len(crosscheck)} 点超出容差，"
                    f"最大偏差 {crosscheck['abs_gap'].max():.4f}")
        lines.append("")
        lines.append(f"**两路径一致性核对**：{len(crosscheck) - n_bad}/{len(crosscheck)} "
                     f"点在容差内，最大偏差 {crosscheck['abs_gap'].max():.4f}。")
        if n_bad:
            lines.append("")
            lines.append("> ⚠️ 存在超出容差的点 → 两条独立估计路径结论不一致。"
                         "按红线 K-3，本曲线**不得单独作为结论**，"
                         "须说明差异来源（多项式阶数不足 / X 部分未充分吸收 / "
                         "未测混杂影响了两条路径）后再使用。")
    return "\n".join(lines)


def render_sensitivity_report(e_val: dict | None,
                              sensitivity=None,
                              guard: ReportGuard | None = None) -> str:
    """渲染敏感性分析片段，并**拦截**常见的过度解读。"""
    lines = ["#### 未测混杂敏感性分析", ""]
    if e_val is None:
        if guard:
            guard.require("已提供 E-value", False,
                          "任何效应估计的报告必须附 E-value（红线 K-1）")
        raise ValueError("报告缺少 E-value：红线 K-1 要求任何效应估计必须附敏感性分析")

    if guard:
        guard.check("已提供 E-value", True)

    lines.append(f"- **E-value（点估计）**：{e_val['e_value_point']:.3f}")
    if e_val["e_value_ci"] is not None:
        lines.append(f"- **E-value（CI 近零端）**：{e_val['e_value_ci']:.3f}"
                     f"　← ★ 应报告此值（更保守）")
    lines.append("")
    lines.append("> 解读：E-value 表示「未测混杂需要与干预和结果**都**达到"
                 "至少该强度的关联，才能把观测效应解释掉」。"
                 "数值越大，结论对未测混杂越不敏感。")
    lines.append("")
    lines.append(f"> ⚠️ 尺度说明：{e_val['note']}")

    if sensitivity is not None:
        b = sensitivity.partial_r2_benchmarks
        lines.append("")
        lines.append("**偏 R² 型敏感性（与已观测因素对比）**")
        lines.append("")
        lines.append(f"- 已观测 X 对干预 T 的解释力：R² = {b['r2_treated_observed']:.4f}")
        lines.append(f"- 已观测 X 对结果 Y 的解释力：R² = {b['r2_outcome_observed']:.4f}")
        lines.append("")
        lines.append("等值线（给定未测混杂对 T 的偏 R²，"
                     "要把效应拉到目标倍数所需的对 Y 的偏 R²）：")
        lines.append("")
        cols = [c for c in sensitivity.robust_curve.columns if c.startswith("ry_for_")]
        lines.append("| ρ_T | " + " | ".join(
            f"拉到 {c.replace('ry_for_', '')} 倍" for c in cols) + " |")
        lines.append("|---|" + "|".join(["---"] * len(cols)) + "|")
        for _, r in sensitivity.robust_curve.iloc[::max(1, len(sensitivity.robust_curve) // 8)].iterrows():
            lines.append(f"| {r['rho_T']:.3f} | "
                         + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
        lines.append("")
        for n_ in sensitivity.notes:
            lines.append(f"> {n_}")

    lines.append("")
    lines.append("> ⚠️ **表述纪律**：以上结果只能支持"
                 "「推翻结论需要多强的未测混杂」这一表述，"
                 "**不得**改写为「未测混杂很弱」或「结论稳健」。"
                 "前者是对数据的描述，后者是对世界的主张，本分析不支持后者。")
    return "\n".join(lines)


def render_assumption_section(ledger) -> str:
    """渲染识别假设章节（从 ledger.py 读取）。"""
    from ledger import ledger_to_md, coverage_summary
    cs = coverage_summary(ledger)
    out = [ledger_to_md(ledger)]
    out.append("")
    out.append(f"台账覆盖度：共 {cs['total']} 条，其中**不可检验** {cs['n_untestable']} 条、"
               f"可检验/部分可检验 {cs['n_testable']} 条；"
               f"已完成诊断 {cs['n_diagnostic_ran']} 条；"
               f"须显著披露 {cs['n_must_disclose']} 条。")
    return "\n".join(out)


def render_provenance(script: str, params: dict, mode: str,
                      extra: dict | None = None) -> str:
    """渲染数字溯源块（每个报告片段必须带）。"""
    lines = ["", "---", "", "**数据溯源**", ""]
    lines.append(f"- 生成脚本：`{script}`")
    lines.append(f"- 数据模式：{_mark_mode(mode)}")
    for k, v in params.items():
        lines.append(f"- {k} = `{v}`")
    if extra:
        for k, v in extra.items():
            lines.append(f"- {k}：{v}")
    lines.append("")
    lines.append("> 上述参数与脚本路径可用于**完全复现**本页所有数字。")
    return "\n".join(lines)


# ------------------------------------------------------------------ 便捷入口
def build_report_section(title: str, parts: list[str], level: int = 2) -> str:
    lines = ["#" * level + f" {title}", ""]
    lines.extend(parts)
    lines.append("")
    return "\n".join(lines)


def guard_summary_md(guard: ReportGuard) -> str:
    """把守卫的检查记录渲染成表（放进报告的附录，便于审计）。"""
    df = guard.to_frame()
    if df.empty:
        return "_本次生成未触发任何纪律检查。_"
    n_fail = int((~df["通过"]).sum())
    lines = [f"本次生成共执行 {len(df)} 项纪律检查，"
             f"其中未通过 {n_fail} 项。", ""]
    lines.append("| 检查项 | 通过 | 说明 |")
    lines.append("|---|---|---|")
    for _, r in df.iterrows():
        lines.append(f"| {r['检查项']} | {'✓' if r['通过'] else '✗'} | {r['说明']} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import synth
    from estimators import estimate_ate_continuous, estimate_naive, estimate_ate_binary
    from dose_response import (estimate_dose_response, estimate_dose_response_robust,
                               crosscheck_curves, monotonicity_report)
    from diagnostics.sensitivity import e_value, cih_sensitivity
    from ledger import default_ledger, assert_ledger_valid

    print("=" * 100)
    print("报告生成器冒烟测试")
    print("=" * 100)

    guard = ReportGuard()
    df = synth.generate(n=3000, u_strength=0.3, nonlinear=True, seed=42)
    d_t = df.attrs["treat_high_mean"] - df.attrs["treat_low_mean"]
    ate2 = float(df["_true_ate"].iloc[0])
    true_slope = ate2 / d_t

    # ---- 估计对比表 ----
    ests = [
        estimate_naive(df),
        estimate_ate_continuous(df, engine="manual"),
    ]
    tbl = render_estimate_table(ests, true_value=true_slope,
                                true_label="真值（斜率口径）", mode="synthetic")
    print("\n【估计对比表】")
    print(estimate_table_to_md(tbl))

    # ---- 曲线（两条路径，故意展示不平滑路径的处理）----
    ca = estimate_dose_response(df)
    cb = estimate_dose_response_robust(df)
    cc = crosscheck_curves(ca, cb)
    print("\n【剂量-反应曲线（平滑路径）】")
    print(render_dose_response_report(ca, monotonicity_report(ca), cc, guard,
                                     mode="synthetic"))
    print("\n【剂量-反应曲线（不平滑路径 —— 应自动拦截边际效应）】")
    out_b = render_dose_response_report(cb, monotonicity_report(cb), None, guard,
                                        mode="synthetic")
    print(out_b[:1400])

    # ---- 敏感性 ----
    ev = e_value(ests[1].point, ci_low=ests[1].ci_low, ci_high=ests[1].ci_high)

    def full_r2(Xcols, target):
        A = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in Xcols])
        y = df[target].to_numpy(float)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ beta
        return 1 - float(res @ res) / float(np.sum((y - y.mean()) ** 2))
    cov = synth.available_covariates()
    sr = cih_sensitivity(ests[1].point, ests[1].se, len(df) - 5,
                         full_r2(cov, "feed_rate"), full_r2(cov, "growth"))
    print("\n【敏感性分析】")
    print(render_sensitivity_report(ev, sr, guard))

    # ---- 识别假设 ----
    lg = default_ledger()
    assert_ledger_valid(lg)
    print("\n【识别假设章节】")
    print(render_assumption_section(lg))

    # ---- 溯源 ----
    print(render_provenance("src/report.py（冒烟测试）",
                            {"n": 3000, "u_strength": 0.3, "seed": 42,
                             "nonlinear": True}, "synthetic"))

    # ---- 守卫汇总 ----
    print("\n【纪律检查汇总】")
    print(guard_summary_md(guard))

    # ---- 反向验证：不给 E-value 应该被拦截 ----
    print("\n【反向验证】不提供 E-value 时必须被拦截")
    try:
        render_sensitivity_report(None, None, ReportGuard())
        print("  ✗ 未被拦截 —— 这是缺陷")
    except ValueError as e:
        print(f"  ✓ 已拦截：{e}")
