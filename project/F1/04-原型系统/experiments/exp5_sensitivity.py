# -*- coding: utf-8 -*-
"""
F1 · 实验 5：未测混杂的敏感性分析与压力测试

【实验目的】
因果推断最脆弱的地方是「无未测混杂」（A1）—— 它**不可检验**。
本实验不假装能检验它，而是回答三个**可以**回答的问题：

  ① 要让结论翻转为零，未测混杂需要多强？（E-value）
  ② 未测混杂要比**已观测**的混杂强多少倍才能翻转结论？（偏 R² 型）
  ③ 真值已知时，偏倚**实测**随混杂强度如何增长？（压力测试）

第 ③ 点是我们相对真实数据研究的**结构性优势**：
真实数据只能给抽象的敏感性指标，我们可以**直接测出偏倚**。

【★ 跑前写死的通过标准】
    F1（E-value 有限）：点估计的 E-value 落在 (1, 10) 区间
        —— 下界 1 排除"零效应"，上界 10 排除"荒谬地稳健"
        （若 E > 10，说明该效应强到任何现实混杂都翻不动，这本身可疑）
    F2（CI 端 E-value 更保守）：CI 近零端的 E-value ≤ 点估计的 E-value
        —— 这是 E-value 的**定义性质**，若不成立则实现有误
    F3（压力测试单调）：实测偏倚随 u_strength 单调上升，且 Spearman 相关 ≥ 0.9
    F4（偏 R² 尺度合理）：使结论翻转所需的 ρ_Y（在 ρ_T 取"已观测混杂强度"时）
        落在 (0, 1) 区间且 > 0 —— 即翻转**可能但不易**
        ⚠️ 本判据在 2026-09-20 被**修正过一次**，见 05-验证/修正记录.md R-13：
        初版从 `breakdown_at(0.0)` 的返回中读 `rho_Y_at_observed_level`，
        而该字段**根本不存在**（真实字段名是
        `required_rho_Y_at_observed_level`）⇒ `ry` 恒为空 Series ⇒
        F4 **每次都判不通过**，且原因被误显为"未能从 CIH 结果中抽取值"。
        修正：改为直接读 `robust_curve` 的 `ry_for_0.00` 列在 ρ_T 网格
        **中位行**处的取值（该处不饱和，实测 0.22–0.24）。
        为什么不继续用 `breakdown_at`：它挑"`ry_for_0.00` 最接近
        `r2_outcome` 的那一行"，而该列在参数区间两端被 `np.clip(·,0,1)`
        顶到 1.0 饱和 ⇒ 恒返回第 0 行、给出 ρ_Y ≈ 0.9999999999。
        浮点上它 < 1，"必须 <1"的判据会**假通过** —— 这是本项目抓到的
        第二个"判据被数值边界愚弄"的例子（第一个见 R-4）。
    F5（稳健性对照）：在 u=0（无未测混杂）下，调整后估计的 CI 应覆盖真值

【数据模式：合成】
⚠️ 全部数值来自 synth.py 的合成结构。E-value 本身是**对任意数据都可算**的
   通用指标，但本实验报出的具体数值**不具备**对真实塘口的外推性（红线 K-5）。

【输出】
    results_exp5_sensitivity.json / .csv
    results_exp5_stress.csv
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
RES = os.path.join(HERE, "..", "..", "05-验证")
sys.path.insert(0, SRC)

import synth                                            # noqa: E402
from estimators import estimate_ate_continuous          # noqa: E402
from diagnostics.sensitivity import (                   # noqa: E402
    e_value, cih_sensitivity, summarize_sensitivity, stress_test_bias,
)

PASS_CRITERIA = {
    "F1": "点估计 E-value 落在 (1, 10)",
    "F2": "CI 近零端 E-value ≤ 点估计 E-value（定义性质）",
    "F3": "实测偏倚随 u_strength 单调上升，Spearman 相关 ≥ 0.9",
    "F4": "翻转所需的 ρ_Y（ρ_T 固定为已观测水平）落在 (0, 1) 且 > 0",
    "F5": "u=0 时调整后估计的 95%CI 覆盖真值",
}

SEEDS = (11, 22, 33, 44, 55, 66)
N = 3000
COV = ["pond_area", "pond_depth", "pond_type", "season_temp"]

# 压力测试档位（比 exp1 更细，用于看偏倚的增长形状）
U_STRESS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2)


def main():
    print("=" * 100)
    print("实验 5：未测混杂的敏感性分析与压力测试（【合成】数据，真值已知）")
    print("=" * 100)
    print("  ★ 通过标准已写死：")
    for k, v in PASS_CRITERIA.items():
        print(f"     {k}: {v}")

    t0 = time.time()
    rows = []

    # ---------------- 逐种子：E-value 与偏 R² 敏感性 ----------------
    print(f"\n--- 主分析（n={N}，{len(SEEDS)} 个种子，u=0.6 模拟实质未测混杂）---")
    for s in SEEDS:
        df = synth.generate(n=N, u_strength=0.6, nonlinear=True, seed=s)
        d_t = df.attrs["treat_high_mean"] - df.attrs["treat_low_mean"]
        true_slope = float(df["_true_ate"].iloc[0]) / d_t

        r = estimate_ate_continuous(df, covariates=COV, engine="manual")

        # ---- E-value ----
        # ⚠️ Y 已标准化，est 即为标准化均差，可直接用 exp(0.91*d) 近似转换
        ev = e_value(r.point, se=r.se, ci_low=r.ci_low, ci_high=r.ci_high,
                     rare=True)

        # ---- Cinelli-Hazlett 偏 R² 型 ----
        # ρ_T 取"已观测混杂对 T 的解释力"作为参照基准
        cih = cih_sensitivity(
            theta=r.point, se=r.se, dof=len(df) - len(COV) - 2,
            r2_treated=0.15,      # 观测协变量对 T 的解释力（保守估计）
            r2_outcome=0.20,      # 观测协变量对 Y 的解释力
            ratios=(0.0, 0.5, 0.8, 1.0, 1.2, 1.5),
            n_grid=60,
        )

        rec = {
            "seed": s, "n": N, "u_strength": 0.6,
            "true_slope": true_slope,
            "theta_hat": r.point, "se": r.se,
            "ci_low": r.ci_low, "ci_high": r.ci_high,
            "bias": r.point - true_slope,
            "covers_true": bool(r.ci_low <= true_slope <= r.ci_high),
            "e_value_point": ev.get("e_value_point"),
            "e_value_ci": ev.get("e_value_ci"),
            "rr_approx": ev.get("rr_approx"),
        }
        # 从 CIH 的稳健性曲线上抽「把效应拉到零所需的 ρ_Y」
        # ⚠️ 口径说明（2026-09-20 修正）：**不要**用 breakdown_at(0.0) ——
        #    该访问器挑「ry_for_0.00 最接近 r2_outcome 的那一行」，
        #    但 ry_for_0.00 在参数区间上被 np.clip(·, 0, 1) 顶到 1.0 而饱和，
        #    于是它恒返回第 0 行，且给出 ρ_Y ≈ 0.9999999999（浮点上 < 1，
        #    会让"必须 <1"的判据**假通过**）。这是本项目抓到的第二个
        #    "判据被数值边界愚弄"的例子（第一个见修正记录 R-4）。
        #    改用 ρ_T 取网格中位处的 ry_for_0.00，它取值 0.22–0.24，有意义。
        try:
            cv = cih.robust_curve
            col = "ry_for_0.00"
            if col in cv.columns:
                mid = cv.iloc[len(cv) // 2]
                rec["rho_T_at_mid"] = float(mid["rho_T"])
                rec["rho_Y_required_to_zero"] = float(mid[col])
            bd = cih.breakdown_at(0.0)
            rec["rho_Y_from_breakdown_at"] = bd[
                "required_rho_Y_at_observed_level"]
            rec["breakdown_at_saturated"] = bool(
                bd["required_rho_Y_at_observed_level"] > 0.99)
        except Exception as e:
            rec["cih_error"] = f"{type(e).__name__}: {e}"
        rec["cih_note"] = " ".join(getattr(cih, "notes", [])[:2])
        rows.append(rec)
        print(f"  seed={s:>3}  θ̂={r.point:+.4f}  真值={true_slope:+.4f}  "
              f"E(点)={rec['e_value_point']:.3f}  E(CI)={rec['e_value_ci']:.3f}  "
              f"[{time.time()-t0:.0f}s]")

    raw = pd.DataFrame(rows)

    # ---------------- 压力测试 ----------------
    print(f"\n--- 压力测试（n={N}，seed=11，档位 {U_STRESS}）---")
    st = stress_test_bias(
        lambda d: estimate_ate_continuous(d, covariates=COV, engine="manual"),
        u_levels=U_STRESS, n=N, seed=11, contrast="slope")
    print(st.to_string(index=False))

    # ---------------- 判定 ----------------
    print("\n" + "=" * 100)
    print("判定（对照跑数前写死的标准）")
    print("=" * 100)
    verdicts = {}

    # F1
    evp = raw["e_value_point"].dropna()
    f1 = bool(len(evp) and evp.min() > 1.0 and evp.max() < 10.0)
    verdicts["F1"] = {"value": f"E 范围 [{evp.min():.3f}, {evp.max():.3f}]",
                      "pass": f1}
    print(f"F1 E-value 有限性：范围 [{evp.min():.3f}, {evp.max():.3f}]"
          f"（须落在 (1,10)）→ {'通过' if f1 else '**不通过**'}")

    # F2
    evc = raw["e_value_ci"].dropna()
    f2 = bool(len(evc) and (evc <= evp.reindex(evc.index) + 1e-9).all())
    verdicts["F2"] = {"value": f"CI 端 [{evc.min():.3f}, {evc.max():.3f}]",
                      "pass": f2}
    print(f"F2 CI 端更保守：CI 端 E 范围 [{evc.min():.3f}, {evc.max():.3f}]"
          f" ≤ 点估计 → {'通过' if f2 else '**不通过**'}")

    # F3
    biases = st["bias"].to_numpy(dtype=float)
    mono = bool(all(biases[i] < biases[i + 1] for i in range(len(biases) - 1)))
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(st["u_strength"].to_numpy(dtype=float),
                              biases).statistic)
    except Exception:
        rho = float("nan")
    f3 = bool(mono and np.isfinite(rho) and rho >= 0.9)
    verdicts["F3"] = {"value": f"单调={mono}  Spearman={rho:.3f}", "pass": f3}
    print(f"F3 压力测试单调：单调={mono}  Spearman={rho:.3f}"
          f" → {'通过' if f3 else '**不通过**'}")

    # F4：把效应拉到零所需的 ρ_Y（在 ρ_T 与已观测同量级处）
    # ⚠️ 口径说明（2026-09-20 第二次修正，见修正记录 R-13）：
    #    初版读 `rho_Y_at_observed_level` —— 该字段名**不存在**，
    #    `breakdown_at()` 返回的是 `required_rho_Y_at_observed_level`，
    #    于是 `ry` 恒为空 Series，F4 每次都落到"未能抽取"分支。
    #    这属于**接口字段名写错**，不是数据问题。
    #    现改为直接读 `_r7` 探针确认过的真实字段。
    _F4_COL = "ry_for_0.00"
    if "rho_Y_required_to_zero" in raw.columns:
        ry = raw["rho_Y_required_to_zero"].dropna().astype(float)
    else:
        ry = pd.Series([], dtype=float)
    # 该列在 rho_T 接近 0 或 1 的两端会被 np.clip(·,0,1) 顶到边界而饱和，
    # 因此额外报告"取值落在 (0,1) 内部"的比例，作为判据可信度的自检。
    ry_inner = ry[(ry > 1e-6) & (ry < 1 - 1e-6)]
    f4 = bool(len(ry) and (ry_inner == ry).all() and (ry > 0).all() and (ry < 1).all())
    if len(ry):
        n_sat = int(len(ry) - len(ry_inner))
        verdicts["F4"] = {
            "value": (f"ρ_Y 范围 [{ry.min():.3f}, {ry.max():.3f}]"
                      f"（字段 {_F4_COL}，取自 ρ_T 网格中位行）；"
                      f"落在 (0,1) 内部 {len(ry_inner)}/{len(ry)}，"
                      f"边界饱和 {n_sat} 个"),
            "pass": f4}
        print(f"F4 翻转所需 ρ_Y 合理：范围 [{ry.min():.3f}, {ry.max():.3f}]"
              f"（内部 {len(ry_inner)}/{len(ry)}，饱和 {n_sat} 个）"
              f"（须在 (0,1)）→ {'通过' if f4 else '**不通过**'}")
        # 作为对照，一并报告「饱和访问器」的取值，说明为何弃用它
        if "rho_Y_from_breakdown_at" in raw.columns:
            sat = raw["rho_Y_from_breakdown_at"].dropna().astype(float)
            if len(sat):
                print(f"   [对照] breakdown_at(0.0) 返回 "
                      f"[{sat.min():.6f}, {sat.max():.6f}] —— 被 clip 顶到 1.0 饱和，"
                      f"故弃用（否则会误判为通过）")
    else:
        verdicts["F4"] = {"value": f"未能从 CIH 结果中抽取值（字段 {_F4_COL} 缺失）",
                          "pass": False}
        print(f"F4 翻转所需 ρ_Y 合理：**未能抽取**"
              f"（缺字段 {_F4_COL}，可用列 "
              f"{[c for c in cih.robust_curve.columns if c.startswith('ry_for_')]}）"
              f"→ 判不通过")

    # F5（u=0 的覆盖，单独跑）
    cov0 = 0
    for s in SEEDS:
        df0 = synth.generate(n=N, u_strength=0.0, nonlinear=True, seed=s)
        d_t0 = df0.attrs["treat_high_mean"] - df0.attrs["treat_low_mean"]
        ts0 = float(df0["_true_ate"].iloc[0]) / d_t0
        r0 = estimate_ate_continuous(df0, covariates=COV, engine="manual")
        if r0.ci_low <= ts0 <= r0.ci_high:
            cov0 += 1
    f5 = cov0 >= 5
    verdicts["F5"] = {"value": f"{cov0}/6 覆盖", "pass": bool(f5)}
    print(f"F5 u=0 时 CI 覆盖真值：{cov0}/6 → {'通过' if f5 else '**不通过**'}")

    # ---------------- 落地 ----------------
    os.makedirs(RES, exist_ok=True)
    csv1 = os.path.join(RES, "results_exp5_sensitivity.csv")
    csv2 = os.path.join(RES, "results_exp5_stress.csv")
    raw.to_csv(csv1, index=False, encoding="utf-8-sig")
    st.to_csv(csv2, index=False, encoding="utf-8-sig")

    payload = {
        "experiment": "exp5_sensitivity",
        "data_mode": "synthetic",
        "data_mode_note": (
            "E-value 与偏 R² 敏感性是**通用指标**，其定义不依赖数据来源；"
            "但本实验报出的**具体数值**仅适用于 synth.py 的合成结构，"
            "不得表述为真实养殖系统所需的混杂强度（红线 K-5）。"
            "另外：偏 R² 敏感性为**近似实现**，不是原论文的严格界，"
            "报告中须标注为「近似敏感性指标」（红线 K-3）。"),
        "script": "experiments/exp5_sensitivity.py",
        "n": N, "seeds": list(SEEDS), "u_levels": list(U_STRESS),
        "covariates": COV,
        "pass_criteria_defined_before_running": PASS_CRITERIA,
        "verdicts": verdicts,
        "wall_seconds": round(time.time() - t0, 1),
    }
    json_path = os.path.join(RES, "results_exp5_sensitivity.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": payload,
                   "rows": raw.to_dict(orient="records"),
                   "stress": st.to_dict(orient="records")},
                  f, ensure_ascii=False, indent=2)

    print(f"\n已写出：\n  {os.path.normpath(csv1)}\n  {os.path.normpath(csv2)}"
          f"\n  {os.path.normpath(json_path)}")
    n_fail = sum(1 for v in verdicts.values() if not v["pass"])
    print(f"\n总判定：{len(verdicts)} 项，未通过 {n_fail} 项"
          + ("（全部通过）" if n_fail == 0 else "（须在报告中如实说明）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
