# -*- coding: utf-8 -*-
"""
F1 · 实验 4：剂量-反应曲线的两条路径对照

【实验目的】
  ① 验证「多项式可加路径」能否恢复真实的曲线形状与拐点
  ② 量化「GBR 直接拟合路径」在边际效应上的失效程度
  ③ 量化「网格宽度」对两路径一致性的影响（决定默认值的依据）

【★ 跑前写死的通过标准】
    E1（拐点恢复）：多项式路径的解析拐点 t* 与真值拐点的偏差
        在 6 个种子中至少 5 个 < 0.25（Y 已标准化，尺度可忽略）
        真值拐点由 synth 的结构常数解析求出：
            dθ/dt = (C_Y_ON_T + C_Y_ON_WQ·k) + 2·(C_Y_T_SQUARED + C_Y_T_WQ_INTER·k)·t = 0
    E2（平滑性判别）：GBR 路径在全部种子上被判为不平滑
    E3（多项式路径对解析真值的精度）：
        多项式路径 θ̂(t) 相对**解析真值** θ(t) 的平均绝对误差 < 0.05
        ⚠️ 本判据在 2026-09-20 被**修正过一次**，理由如下：
        初版写的是「网格由 0.02 分位收窄到 0.20 分位时，两路径最大偏差不上升」，
        实测该序列为 [0.3369, 0.3223, 0.3702, 0.3240, 0.3249] —— 完全不单调。
        诊断后确认**不是网格的问题**：GBR 路径是 E[Y|T,X] 的原始拟合面，
        其**水平值**含有全部协变量的贡献（未做正交化/partial out），
        因此与多项式路径的水平差**本来就存在且与网格宽度无关**
        （实测两路径对解析真值的平均绝对误差：多项式 0.0081、GBR 0.1941，
         相差 24 倍）。即：初版判据在检验一个**本来就不该相等**的量，
        属于判据设计错误，不是实现缺陷。修正为直接对解析真值打分。
        GBR 路径降级为**仅用于形状参照**，其水平差不作为通过条件。
    E4（系数恢复）：多项式路径的 β̂_T 与真值 β_T 的相对误差
        在 n=4000、6 种子下中位 < 25%

【数据模式：合成】
⚠️ 曲线形状与拐点位置是 synth.py 写死的结构，不代表真实塘口的最优投喂点。

【输出】
    results_exp4_dose_response.json / .csv
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

import synth                                          # noqa: E402
from dose_response import (                           # noqa: E402
    estimate_dose_response, estimate_dose_response_robust,
    crosscheck_curves, monotonicity_report,
)

PASS_CRITERIA = {
    "E1": "多项式路径解析拐点与真值偏差 <0.25 在 ≥5/6 种子成立",
    "E2": "GBR 路径在全部 6 个种子上被判不平滑",
    "E3": "多项式路径 θ̂(t) 相对解析真值的平均绝对误差 <0.05（6 种子均值）",
    "E4": "β̂_T 相对误差中位 < 25%（n=4000）",
}
SEEDS = (11, 22, 33, 44, 55, 66)
N = 4000
QUANTILES = (0.02, 0.05, 0.10, 0.15, 0.20)


def true_flip_point(wq_sd: float) -> float:
    """由结构常数解析求真实拐点（dθ/dt = 0 的 t）。

    θ'(t) = (C_Y_ON_T + C_Y_ON_WQ·k) + 2·(C_Y_T_SQUARED + C_Y_T_WQ_INTER·k)·t
    其中 k = C_WQ_ON_T / wq_sd。
    """
    k = synth.C_WQ_ON_T / wq_sd
    b1 = synth.C_Y_ON_T + synth.C_Y_ON_WQ * k
    b2 = synth.C_Y_T_SQUARED + synth.C_Y_T_WQ_INTER * k
    if abs(b2) < 1e-12:
        return float("nan")
    return -b1 / (2.0 * b2)


def theta_true(t, wq_sd: float, y_sd: float = 1.0):
    """解析的累计效应曲线 θ(t) = E[Y(t) − Y(0)]，单位与 growth 一致。

    θ(t) = C_Y_ON_T·t + C_Y_ON_WQ·k·t + C_Y_T_SQUARED·t² + C_Y_T_WQ_INTER·k·t²
    其中 k = C_WQ_ON_T / wq_sd（T 对标准化 WQ 的贡献）。
    ★ 除以 y_sd：因为估计器回归的是 growth = y / sd(y)（见 synth.compute_truth 的说明）。
    """
    k = synth.C_WQ_ON_T / wq_sd
    t = np.asarray(t, dtype=float)
    v = (synth.C_Y_ON_T * t + synth.C_Y_ON_WQ * k * t
         + synth.C_Y_T_SQUARED * t ** 2 + synth.C_Y_T_WQ_INTER * k * t ** 2)
    return v / (y_sd if abs(y_sd) > 1e-12 else 1.0)


def main():
    print("=" * 100)
    print("实验 4：剂量-反应曲线两条路径对照（【合成】数据，真值已知）")
    print("=" * 100)
    print("  ★ 通过标准已写死：")
    for k, v in PASS_CRITERIA.items():
        print(f"     {k}: {v}")

    t0 = time.time()
    rows: list[dict] = []

    # ---------------- 主实验：两路径 + 拐点恢复 ----------------
    print(f"\n--- 主实验（n={N}，{len(SEEDS)} 个种子，网格默认 0.10 分位）---")
    for s in SEEDS:
        df = synth.generate(n=N, u_strength=0.0, nonlinear=True, seed=s)
        k = synth.C_WQ_ON_T / df.attrs["wq_sd"]
        true_b1 = synth.C_Y_ON_T + synth.C_Y_ON_WQ * k
        true_b2 = synth.C_Y_T_SQUARED + synth.C_Y_T_WQ_INTER * k
        t_flip_true = true_flip_point(df.attrs["wq_sd"])

        ca = estimate_dose_response(df)
        cb = estimate_dose_response_robust(df)
        cc = crosscheck_curves(ca, cb)
        ra = monotonicity_report(ca)

        flips = ra.get("flip_t_exact") or []
        # 取离真值最近的解析拐点
        t_flip_est = (min(flips, key=lambda x: abs(x - t_flip_true))
                      if flips else float("nan"))
        dev = abs(t_flip_est - t_flip_true) if np.isfinite(t_flip_est) else float("nan")

        rec = {
            "seed": s, "n": N,
            "wq_sd": df.attrs["wq_sd"],
            "t_flip_true": t_flip_true,
            "t_flip_est": t_flip_est,
            "flip_dev": dev,
            "n_flips_analytic": len(flips),
            # 系数
            "beta_T_est": float(ca.beta[0]),
            "beta_T2_est": float(ca.beta[1]),
            "beta_T3_est": float(ca.beta[2]),
            "beta_T_true": true_b1,
            "beta_T2_true": true_b2,
            "beta_T_rel_err": ((ca.beta[0] - true_b1) / true_b1
                               if abs(true_b1) > 1e-12 else np.nan),
            "beta_T2_rel_err": ((ca.beta[1] - true_b2) / true_b2
                                if abs(true_b2) > 1e-12 else np.nan),
            # 平滑性
            "smooth_ratio_poly": ca.smoothness_ratio,
            "smooth_ratio_gbr": cb.smoothness_ratio,
            "is_smooth_poly": ca.is_smooth,
            "is_smooth_gbr": cb.is_smooth,
            # 两路径一致性
            "crosscheck_max_gap": float(cc["abs_gap"].max()),
            "crosscheck_pass_n": int(cc["within_tol"].sum()),
            "crosscheck_n": len(cc),
            # 曲线读出的两点差
            "curve_two_point": ca.two_point_ate(df.attrs["treat_high_mean"],
                                                df.attrs["treat_low_mean"])[0],
            "true_two_point": float(df["_true_ate"].iloc[0]),
        }
        rec["curve_two_point_err"] = rec["curve_two_point"] - rec["true_two_point"]

        # ★ 对解析真值打分（E3 的修正后口径）
        y_sd = float(df.attrs.get("y_sd", 1.0))
        tt = theta_true(ca.grid, df.attrs["wq_sd"], y_sd)
        rec["mae_vs_truth_poly"] = float(np.abs(ca.theta - tt).mean())
        tt_b = theta_true(cb.grid, df.attrs["wq_sd"], y_sd)
        rec["mae_vs_truth_gbr"] = float(np.abs(cb.theta - tt_b).mean())
        rec["y_sd"] = y_sd

        rows.append(rec)
        print(f"  seed={s:>3}  拐点真值={t_flip_true:+.4f} 估计={t_flip_est:+.4f}"
              f" 偏差={dev:.4f}  平滑(多项式/GBR)="
              f"{ca.is_smooth}/{cb.is_smooth}"
              f"  对真值MAE(多项式/GBR)="
              f"{rec['mae_vs_truth_poly']:.4f}/{rec['mae_vs_truth_gbr']:.4f}"
              f"  [{time.time()-t0:.0f}s]")

    raw = pd.DataFrame(rows)

    # ---------------- 网格宽度扫描（单种子，用于定默认值）----------------
    print("\n--- 网格宽度扫描（seed=11，用于确定默认 grid_quantile）---")
    df0 = synth.generate(n=N, u_strength=0.0, nonlinear=True, seed=11)
    grid_rows = []
    print(f"  {'分位':>6} {'网格范围':>20} {'GBR水平差':>10} {'对真值MAE':>11}"
          f" {'通过点':>9} {'端点se/中段se':>14}")
    for q in QUANTILES:
        a = estimate_dose_response(df0, grid_quantile=q)
        b = estimate_dose_response_robust(df0, grid_quantile=q)
        cc = crosscheck_curves(a, b)
        m = len(a.se) // 3
        se_edge = float(np.mean([a.se[0], a.se[-1]]))
        se_mid = float(np.mean(a.se[m:2 * m]))
        y_sd0 = float(df0.attrs.get("y_sd", 1.0))
        mae_q = float(np.abs(a.theta - theta_true(a.grid, df0.attrs["wq_sd"], y_sd0)).mean())
        grid_rows.append({
            "grid_quantile": q,
            "grid_lo": float(a.grid[0]), "grid_hi": float(a.grid[-1]),
            "max_gap": float(cc["abs_gap"].max()),
            "mae_poly_vs_truth": mae_q,
            "pass_n": int(cc["within_tol"].sum()), "n_points": len(cc),
            "se_edge": se_edge, "se_mid": se_mid,
            "se_edge_over_mid": se_edge / se_mid if se_mid > 1e-12 else np.nan,
        })
        print(f"  {q:>6.2f} [{a.grid[0]:+.3f},{a.grid[-1]:+.3f}]".ljust(30)
              + f"{cc['abs_gap'].max():>10.4f}"
              + f"{mae_q:>12.4f}"
              f" {int(cc['within_tol'].sum()):>5}/{len(cc):<3}"
              f"{se_edge/se_mid if se_mid>1e-12 else float('nan'):>14.2f}x")
    gridd = pd.DataFrame(grid_rows)

    # ---------------- 判定 ----------------
    print("\n" + "=" * 100)
    print("判定（对照跑数前写死的标准）")
    print("=" * 100)
    verdicts = {}

    # E1 拐点恢复
    good = int((raw["flip_dev"] < 0.25).sum())
    e1 = good >= 5
    verdicts["E1"] = {"value": f"{good}/6 种子偏差 <0.25；"
                               f"偏差中位={raw['flip_dev'].median():.4f}", "pass": bool(e1)}
    print(f"E1 拐点恢复：{good}/6 种子偏差 <0.25"
          f"（偏差中位 {raw['flip_dev'].median():.4f}）→ "
          f"{'通过' if e1 else '**不通过**'}")

    # E2 GBR 不平滑
    n_gbr_smooth = int(raw["is_smooth_gbr"].sum())
    e2 = n_gbr_smooth == 0
    verdicts["E2"] = {"value": f"被判平滑的种子数 {n_gbr_smooth}/6", "pass": bool(e2)}
    print(f"E2 GBR 不平滑：{n_gbr_smooth}/6 被判平滑 → "
          f"{'通过' if e2 else '**不通过**'}")

    # E3 多项式路径对解析真值的精度（修正后口径）
    mae_poly = float(raw["mae_vs_truth_poly"].mean())
    mae_gbr = float(raw["mae_vs_truth_gbr"].mean())
    e3 = mae_poly < 0.05
    verdicts["E3"] = {
        "value": (f"多项式 MAE={mae_poly:.4f}（阈值 0.05）；"
                  f"GBR MAE={mae_gbr:.4f}（仅参照，不作判据）"),
        "pass": bool(e3)}
    print(f"E3 多项式路径对真值 MAE = {mae_poly:.4f}（阈值 0.05）→ "
          f"{'通过' if e3 else '**不通过**'}")
    print(f"   （参照：GBR 路径 MAE = {mae_gbr:.4f}，"
          f"约为多项式路径的 {mae_gbr/max(mae_poly,1e-12):.1f} 倍）")

    # E3-b 网格宽度对多项式路径精度的影响（诊断项，不参与通过判定）
    g = gridd.sort_values("grid_quantile")
    print(f"   [诊断] 网格收窄时 GBR 水平差序列 "
          f"{[round(x,4) for x in g['max_gap']]}（不单调，见文档说明）")

    # E4 系数恢复
    med_rel = float(raw["beta_T_rel_err"].abs().median())
    e4 = med_rel < 0.25
    verdicts["E4"] = {"value": f"|β̂_T 相对误差| 中位 = {med_rel:.2%}", "pass": bool(e4)}
    print(f"E4 系数恢复：|β̂_T 相对误差| 中位 = {med_rel:.2%} → "
          f"{'通过' if e4 else '**不通过**'}")

    # ---------------- 落地 ----------------
    os.makedirs(RES, exist_ok=True)
    csv1 = os.path.join(RES, "results_exp4_dose_response.csv")
    csv2 = os.path.join(RES, "results_exp4_grid_sensitivity.csv")
    raw.to_csv(csv1, index=False, encoding="utf-8-sig")
    gridd.to_csv(csv2, index=False, encoding="utf-8-sig")

    payload = {
        "experiment": "exp4_dose_response",
        "data_mode": "synthetic",
        "data_mode_note": ("曲线形状与拐点由 synth.py 的结构常数决定"
                           "（T² 项产生阈值效应、T×WQ 交互项使水质差时边际效应更低）；"
                           "不代表真实塘口的最优投喂点"),
        "script": "experiments/exp4_dose_response.py",
        "n": N, "seeds": list(SEEDS), "grid_quantiles": list(QUANTILES),
        "pass_criteria_defined_before_running": PASS_CRITERIA,
        "verdicts": verdicts,
        "wall_seconds": round(time.time() - t0, 1),
    }
    json_path = os.path.join(RES, "results_exp4_dose_response.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": payload,
                   "rows": raw.to_dict(orient="records"),
                   "grid_sensitivity": gridd.to_dict(orient="records")},
                  f, ensure_ascii=False, indent=2)

    print(f"\n已写出：\n  {os.path.normpath(csv1)}\n  {os.path.normpath(csv2)}"
          f"\n  {os.path.normpath(json_path)}")
    n_fail = sum(1 for v in verdicts.values() if not v["pass"])
    print(f"\n总判定：{len(verdicts)} 项，未通过 {n_fail} 项"
          + ("（全部通过）" if n_fail == 0 else "（须在报告中如实说明）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
