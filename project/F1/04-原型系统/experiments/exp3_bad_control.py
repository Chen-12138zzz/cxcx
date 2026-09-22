# -*- coding: utf-8 -*-
"""
F1 · 实验 3：坏控制（中介 / 碰撞）的量化代价

【实验目的】
"不要把中介和碰撞放进调整集"是一条方法论常识，但在真实研究里它常常被违反，
因为违反的代价**不会在模型指标上体现**（R² 反而可能更高、系数更"显著"）。
本实验把这个代价**量化**出来，使"为什么不能这样写"从口号变成数字。

要回答三个问题：
  ① 调整中介（水质）会把效应压低多少？（对总效应的屏蔽比例）
  ② 调整碰撞（FCR）会把效应推高多少？（后门通路的放大倍数）
  ③ 把"全变量回归"（把所有能拿到的变量都放进去）当作常规做法，
     代价有多大？

【★ 跑前写死的通过标准】
    D1（中介方向）：加入 water_degrad 后，|θ̂| 相对基线的变化显著（> 5%），
        且方向符合"屏蔽部分中介效应"的预期（总效应 → 直接效应，绝对值下降）
        ⚠️ 注意：本项目合成结构中 T→WQ 的系数为正、WQ→Y 为负，
           总效应 = 直接 + 中介（中介项为负）；调整中介后**去掉中介项**，
           因此 θ̂ 应当**升高**（去掉负的中介项）。
           判定标准据此写为"变化显著"，方向预期记录在案，不作硬性通过条件——
           因为方向的符号依赖结构的正负设定，真实数据里方向无法预先确定。
    D2（碰撞破坏力）：加入 fcr 后，|Δθ| 至少是加入中介后 |Δθ| 的 2 倍
    D3（全变量回归代价）：把中介与碰撞同时放入，|Δθ| > 单独放碰撞的 |Δθ|

【数据模式：合成】
⚠️ 效应量仅适用于 synth.py 的合成结构，不得外推真实塘口（红线 K-5）。

【输出】
    results_exp3_bad_control.json / .csv
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
from estimators import estimate_ate_continuous        # noqa: E402
from diagnostics.bad_control import (                 # noqa: E402
    audit_adjustment_set, check_bad_control,
)

PASS_CRITERIA = {
    "D1": "加入中介后 |Δθ|/|θ_base| > 5%（变化显著）",
    "D2": "|Δθ(fcr)| ≥ 2 × |Δθ(water_degrad)|（碰撞破坏力至少是中介的 2 倍）",
    "D3": "|Δθ(两者都加)| > |Δθ(仅加 fcr)|",
}
SEEDS = (11, 22, 33, 44, 55, 66)
N = 3000
U_LEVELS = (0.0, 0.3, 0.6)

# 调整集的四种配置（实验的核心自变量）
CONFIGS = {
    "base_correct": ["pond_area", "pond_depth", "pond_type", "season_temp"],
    "plus_mediator": ["pond_area", "pond_depth", "pond_type", "season_temp",
                      "water_degrad"],
    "plus_collider": ["pond_area", "pond_depth", "pond_type", "season_temp", "fcr"],
    "plus_both": ["pond_area", "pond_depth", "pond_type", "season_temp",
                  "water_degrad", "fcr"],
}


def main():
    print("=" * 100)
    print("实验 3：坏控制的量化代价（【合成】数据，真值已知）")
    print("=" * 100)
    print("  ★ 通过标准已写死：")
    for k, v in PASS_CRITERIA.items():
        print(f"     {k}: {v}")

    # ---------------- 静态审计（不需数据）----------------
    print("\n--- 静态审计：四种调整集的角色合规性 ---")
    audit_rows = []
    for name, cs in CONFIGS.items():
        a = audit_adjustment_set(cs)
        audit_rows.append({
            "config": name, "n_vars": a["n"],
            "pass": a["pass"],
            "violations": "; ".join(v["变量"] for v in a["violations"]) or "（无）",
        })
        print(f"  {name:16s} {'通过' if a['pass'] else '**不通过**':>12s}"
              f"  违规：{audit_rows[-1]['violations']}")
    audit_df = pd.DataFrame(audit_rows)

    # ---------------- 动态对照 ----------------
    print(f"\n--- 动态对照：四种调整集的效应估计（n={N}，{len(SEEDS)} 个种子）---")
    t0 = time.time()
    rows = []
    for u in U_LEVELS:
        for s in SEEDS:
            df = synth.generate(n=N, u_strength=u, nonlinear=True, seed=s)
            d_t = df.attrs["treat_high_mean"] - df.attrs["treat_low_mean"]
            ate2 = float(df["_true_ate"].iloc[0])
            true_slope = ate2 / d_t

            rec = {"u_strength": u, "seed": s,
                   "true_slope": true_slope, "true_two_point": ate2, "d_t": d_t}
            for cname, cs in CONFIGS.items():
                try:
                    r = estimate_ate_continuous(df, covariates=cs, engine="manual")
                    rec[f"theta_{cname}"] = r.point
                    rec[f"se_{cname}"] = r.se
                    rec[f"bias_{cname}"] = r.point - true_slope
                    rec[f"absbias_{cname}"] = abs(r.point - true_slope)
                except Exception as e:
                    rec[f"theta_{cname}"] = None
                    rec[f"err_{cname}"] = f"{type(e).__name__}: {e}"
            rows.append(rec)
        print(f"  u={u:.1f} 全部 {len(SEEDS)} 个种子完成（累计 {time.time()-t0:.0f}s）")

    raw = pd.DataFrame(rows)

    # ---- 位移量 ----
    for cname in ("plus_mediator", "plus_collider", "plus_both"):
        raw[f"delta_{cname}"] = raw[f"theta_{cname}"] - raw["theta_base_correct"]
        raw[f"absdelta_{cname}"] = raw[f"delta_{cname}"].abs()
    raw["ratio_collider_over_mediator"] = (
        raw["absdelta_plus_collider"] / raw["absdelta_plus_mediator"])

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 100)
    print("结果汇总（按未测混杂档位）")
    print("=" * 100)
    summary_rows = []
    for u in U_LEVELS:
        sub = raw[raw["u_strength"] == u]
        print(f"\n--- u_strength={u} ---")
        print(f"  真值斜率口径 = {sub['true_slope'].mean():+.4f}")
        print(f"  {'调整集':<18} {'θ̂ 均值':>10} {'|偏倚|均值':>12} {'Δθ 均值':>10}"
              f" {'|Δθ| 均值':>10}")
        for cname in CONFIGS:
            th = sub[f"theta_{cname}"].mean()
            ab = sub[f"absbias_{cname}"].mean()
            if cname == "base_correct":
                dl = ad = 0.0
            else:
                dl = sub[f"delta_{cname}"].mean()
                ad = sub[f"absdelta_{cname}"].mean()
            print(f"  {cname:<18} {th:>+10.4f} {ab:>12.4f} {dl:>+10.4f} {ad:>10.4f}")
            summary_rows.append({
                "u_strength": u, "config": cname,
                "theta_mean": float(th), "absbias_mean": float(ab),
                "delta_mean": float(dl), "absdelta_mean": float(ad),
                "n_seeds": len(sub),
            })
    summ = pd.DataFrame(summary_rows)

    # ---------------- 判定 ----------------
    print("\n" + "=" * 100)
    print("判定（对照跑数前写死的标准）")
    print("=" * 100)
    verdicts = {}

    # D1：中介的位移显著性（在所有 u 档上都检查，报告最弱的一档）
    d1_ok, d1_detail = True, []
    for u in U_LEVELS:
        sub = raw[raw["u_strength"] == u]
        rel = (sub["absdelta_plus_mediator"] / sub["theta_base_correct"].abs()).mean()
        d1_detail.append(f"u={u}: {rel:.2%}")
        if not (rel > 0.05):
            d1_ok = False
    verdicts["D1"] = {"value": "; ".join(d1_detail), "pass": bool(d1_ok)}
    print(f"D1 中介位移显著（>5%）：{'; '.join(d1_detail)} → "
          f"{'通过' if d1_ok else '**不通过**'}")

    # D2：碰撞破坏力 ≥ 中介的 2 倍
    r_all = raw["ratio_collider_over_mediator"].dropna()
    d2_ok = bool(r_all.mean() >= 2.0)
    verdicts["D2"] = {"value": f"平均倍数 {r_all.mean():.2f}（范围 "
                               f"{r_all.min():.2f}–{r_all.max():.2f}）", "pass": d2_ok}
    print(f"D2 碰撞/中介 破坏力倍数：平均 {r_all.mean():.2f}"
          f"（范围 {r_all.min():.2f}–{r_all.max():.2f}）→ "
          f"{'通过' if d2_ok else '**不通过**'}")

    # D3：两者都加 比 仅加碰撞 位移更大
    d3_ok = bool((raw["absdelta_plus_both"] > raw["absdelta_plus_collider"]).mean() > 0.5)
    frac = float((raw["absdelta_plus_both"] > raw["absdelta_plus_collider"]).mean())
    verdicts["D3"] = {"value": f"成立的种子比例 {frac:.0%}", "pass": d3_ok}
    print(f"D3 两变量同放的额外位移：成立比例 {frac:.0%} → "
          f"{'通过' if d3_ok else '**不通过**'}")

    # ---------------- 角色诊断（偏 R² 信号）----------------
    print("\n--- 偏 R² 信号（用于提示人工复核角色判定）---")
    sig_rows = []
    for u in U_LEVELS:
        df = synth.generate(n=N, u_strength=u, nonlinear=True, seed=SEEDS[0])
        dgs = check_bad_control(df, covariates=CONFIGS["base_correct"],
                                candidates=["water_degrad", "fcr"],
                                run_contrast=False)
        for d in dgs:
            sig_rows.append({
                "u_strength": u, "candidate": d.candidate, "role": d.role,
                "partial_r2_with_t": d.partial_r2_with_t,
                "partial_r2_with_y": d.partial_r2_with_y,
                "compliant": d.compliant,
            })
            print(f"  u={u:.1f}  {d.candidate:13s} 角色={d.role:10s}"
                  f" 偏R²(T)={d.partial_r2_with_t:+.4f}"
                  f" 偏R²(Y)={d.partial_r2_with_y:+.4f}"
                  f" 合规={d.compliant}")
    sig_df = pd.DataFrame(sig_rows)

    # ---------------- 落地 ----------------
    os.makedirs(RES, exist_ok=True)
    csv1 = os.path.join(RES, "results_exp3_bad_control_raw.csv")
    csv2 = os.path.join(RES, "results_exp3_bad_control_summary.csv")
    csv3 = os.path.join(RES, "results_exp3_bad_control_audit.csv")
    csv4 = os.path.join(RES, "results_exp3_bad_control_signal.csv")
    raw.to_csv(csv1, index=False, encoding="utf-8-sig")
    summ.to_csv(csv2, index=False, encoding="utf-8-sig")
    audit_df.to_csv(csv3, index=False, encoding="utf-8-sig")
    sig_df.to_csv(csv4, index=False, encoding="utf-8-sig")

    payload = {
        "experiment": "exp3_bad_control",
        "data_mode": "synthetic",
        "data_mode_note": ("中介为 water_degrad（受 T 影响且影响 Y）、"
                           "碰撞为 fcr（分子含 T、分母含 Y）；"
                           "合成结构中 T→WQ 系数为正、WQ→Y 为负，"
                           "故调整中介后 θ̂ 方向为升，该方向**依赖结构设定**，"
                           "真实数据中不可预先确定"),
        "script": "experiments/exp3_bad_control.py",
        "n": N, "seeds": list(SEEDS), "u_levels": list(U_LEVELS),
        "configs": {k: v for k, v in CONFIGS.items()},
        "pass_criteria_defined_before_running": PASS_CRITERIA,
        "verdicts": verdicts,
        "wall_seconds": round(time.time() - t0, 1),
    }
    json_path = os.path.join(RES, "results_exp3_bad_control.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": payload,
            "summary": summ.to_dict(orient="records"),
            "audit": audit_df.to_dict(orient="records"),
            "signal": sig_df.to_dict(orient="records"),
            "raw_head": raw.head(6).to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n已写出：\n  {os.path.normpath(csv1)}\n  {os.path.normpath(csv2)}"
          f"\n  {os.path.normpath(csv3)}\n  {os.path.normpath(csv4)}"
          f"\n  {os.path.normpath(json_path)}")
    n_fail = sum(1 for v in verdicts.values() if not v["pass"])
    print(f"\n总判定：{len(verdicts)} 项，未通过 {n_fail} 项"
          + ("（全部通过）" if n_fail == 0 else "（须在报告中如实说明）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
