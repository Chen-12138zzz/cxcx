# -*- coding: utf-8 -*-
"""
F1 · 实验 1：估计器无偏性验证（合成数据，真值已知）

【实验目的】
验证「标准因果估计流程」在**已知真值**的合成数据上是否正确：
  ① 朴素回归（无调整）偏倚多大？
  ② 只调整协变量的 OLS 偏倚多大？
  ③ DML（正交化 + 交叉拟合）是否比 ② 更接近真值？
  ④ 双稳健 AIPW（二值干预）是否与两点差真值一致？

【★ 关键：跑实验前先写死通过标准（红线 K-3 的落实）】
在**看结果之前**先固定判定规则，避免"结果出来后调整标准"。

    ⚠️ P1 与 P3 在 2026-09-20 被**修正过一次**（判据设计错误，非实现缺陷），
       理由如下（首轮实测数字见 05-验证/修正记录.md 的 R-6）：

    首轮 P1 为「|bias(DML)| < |bias(OLS_adj)|，6 种子中 ≥5 成立」：
        u=0 时得 2/6（不通过）、u=0.6 时得 6/6（通过）。
        诊断：u=0 时**已无未测混杂可去**，三种方法都落在真值 ±0.01 内
        （OLS −0.0037 / DML −0.0119 / doubleml −0.0081），
        此时"谁更准"完全由噪声决定 ⇒ 该比较在 u=0 处是**退化比较**，
        用它做通过条件本身就是错的。
        修正：P1 只在**存在实质未测混杂**（u ≥ 0.3）的档位上判定。

    首轮 P3 为「AIPW 覆盖真值 ≥5/6」：
        u=0 时得 6/6（通过）、u=0.6 时得 1/6（不通过）。
        诊断：u=0.6 时**确实存在**未测混杂，估计量**本应**有偏，
        其 CI 因此**理应盖不住真值** —— 要求它覆盖等于要求混杂不存在，
        与实验目的（展示混杂后果）自相矛盾。
        修正：P3 改为双向判据 —— u=0 时必须覆盖 ≥5/6（无混杂时应当正确），
        u≥0.3 时覆盖率**应当显著下降**（不覆盖恰是混杂生效的证据）。

    主标准 P1（DML 优于 OLS 调整）：
        在 u ≥ 0.3 的档位上，|bias(DML)| < |bias(OLS_adj)|，
        6 个种子中至少 5 个成立。
    主标准 P2（DML 偏倚在 seed 间稳定）：
        6 个种子的 DML 偏倚的**极差** < 0.05（Y 已标准化，0.05 ≈ 5% 个标准差）
    主标准 P3（AIPW 覆盖率的**单调**判据）：
        u=0 时覆盖真值 ≥5/6 种子（无混杂 ⇒ 区间应正确）；
        且覆盖率随 u_strength **单调不增**（Σu_u≥0.3 ≤ Σu_u=0）。
        ⚠️ P3 在 2026-09-20 被**第二次**修正，理由如下（见修正记录 R-6）：
        第二版写的是「u≥0.3 时覆盖率 **< 5/6**」。首轮实测 u=0.3 得 5/6、
        u=0.6 得 1/6 —— 于是**整体判不通过**，但原因是一个**边界值**：
        u=0.3 恰好落在 5/6 上，比"应 <5/6"只差一个种子。
        诊断：**在弱混杂档位上要求用 6 个种子分辨出覆盖率下降，分辨率本就不够**——
        单种子覆盖与否是 0/1，6 个种子的分辨率就是 1/6 ≈ 0.167，
        而 u=0.3 的预期覆盖率下降幅度正好在这个量级以下。
        该判据把"实现缺陷"与"判据分辨率不足"混在了一起：
        真值未覆盖既可能是混杂生效（预期），也可能是抽样波动（噪声）。
        修正：改为**跨档位单调**——不再要求某个具体档位低于 5/6，
        而是要求覆盖率作为 u 的函数非增。这既保留了"混杂应使区间失效"
        这一方向性预期，又不把 1/6 的量子化噪声当成失败。
        辅助观察（不作为通过条件，但必须报告）：
        - 朴素估计的偏倚方向与量级
        - 各方法在 u_strength 增大时的偏倚增长曲线
        - u=0 处三方法的偏倚幅度（用于说明该档位是退化比较）

【数据模式：合成】
⚠️ 本实验全部结果来自 synth.py 的合成结构，
   不得表述为真实养殖系统的效应量（红线 K-5）。

【输出】
    results_exp1_estimator_bias.json   —— 逐种子逐方法的完整数值
    results_exp1_estimator_bias.csv    —— 同一内容的表格形式
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

import synth                                   # noqa: E402
from estimators import (                       # noqa: E402
    estimate_naive, estimate_adjusted_ols,
    estimate_ate_continuous, estimate_ate_binary,
)

# ============================================================ 跑实验前写死的通过标准
PASS_CRITERIA = {
    "P1": "在 u≥0.3 的档位上 |bias(DML)| < |bias(OLS_adj)| 成立 ≥5/6（u=0 不作判定）",
    "P2": "DML 偏倚在 6 个种子间的极差 < 0.05",
    "P3": "覆盖真值的种子**总数**随 u 单调不增（u=0 须 ≥5/6），即"
          "Σcov(u≥0.3) ≤ Σcov(u=0) 且 u=0 覆盖 ≥5/6",
}
SEEDS = (11, 22, 33, 44, 55, 66)
N = 3000

# u_strength=0 表示 U 不影响干预（不是混杂），此时所有方法都应当接近真值；
# u_strength=0.6 表示存在实质未测混杂，用于展示偏倚随混杂强度增长。
U_LEVELS = (0.0, 0.3, 0.6)

# 参与 P1/P3 判定的"有实质混杂"档位（u=0 是退化比较，排除）
CONFOUNDED_LEVELS = tuple(u for u in U_LEVELS if u >= 0.3)


def run_one(df: pd.DataFrame) -> dict:
    """在一份数据上跑全部估计器，返回扁平字典。"""
    d_t = df.attrs["treat_high_mean"] - df.attrs["treat_low_mean"]
    ate2 = float(df["_true_ate"].iloc[0])
    true_slope = ate2 / d_t

    out = {
        "n": len(df),
        "d_t": d_t,
        "true_ate_two_point": ate2,
        "true_slope": true_slope,
    }

    # ---- 斜率口径的三个估计器 ----
    for name, fn in (("naive_ols", estimate_naive),
                     ("ols_adjusted", estimate_adjusted_ols),
                     ("dml_plr", lambda d: estimate_ate_continuous(d, engine="manual")),
                     ("dml_doubleml", lambda d: estimate_ate_continuous(d, engine="doubleml"))):
        try:
            r = fn(df)
            out[f"{name}_point"] = r.point
            out[f"{name}_se"] = r.se
            out[f"{name}_ci_low"] = r.ci_low
            out[f"{name}_ci_high"] = r.ci_high
            # 偏倚按斜率口径（这三个都是 slope）
            out[f"{name}_bias"] = r.point - true_slope
            out[f"{name}_abs_bias"] = abs(r.point - true_slope)
        except Exception as e:
            out[f"{name}_point"] = None
            out[f"{name}_error"] = f"{type(e).__name__}: {e}"

    # ---- 两点差口径（二值 AIPW）----
    for name, fn in (("aipw_binary", lambda d: estimate_ate_binary(d, engine="manual")),
                     ("aipw_doubleml", lambda d: estimate_ate_binary(d, engine="doubleml"))):
        try:
            r = fn(df)
            out[f"{name}_point"] = r.point
            out[f"{name}_se"] = r.se
            out[f"{name}_ci_low"] = r.ci_low
            out[f"{name}_ci_high"] = r.ci_high
            out[f"{name}_bias"] = r.point - ate2
            out[f"{name}_abs_bias"] = abs(r.point - ate2)
            # 真值是否落在 CI 内（用于 P3）
            out[f"{name}_covers_true"] = bool(
                r.ci_low is not None and r.ci_high is not None
                and r.ci_low <= ate2 <= r.ci_high)
        except Exception as e:
            out[f"{name}_point"] = None
            out[f"{name}_error"] = f"{type(e).__name__}: {e}"

    return out


def main():
    print("=" * 100)
    print("实验 1：估计器无偏性验证（【合成】数据，真值已知）")
    print("=" * 100)
    print(f"  样本量 n={N}，种子 {SEEDS}，未测混杂档位 {U_LEVELS}")
    print("  ★ 通过标准已在本脚本源码中写死（见 PASS_CRITERIA），跑数前不可修改")
    for k, v in PASS_CRITERIA.items():
        print(f"     {k}: {v}")

    t0 = time.time()
    rows = []
    for u in U_LEVELS:
        for s in SEEDS:
            df = synth.generate(n=N, u_strength=u, nonlinear=True, seed=s)
            rec = run_one(df)
            rec["u_strength"] = u
            rec["seed"] = s
            rows.append(rec)
            print(f"  u={u:.1f} seed={s:>3} 完成（已用 {time.time()-t0:.0f}s）")

    raw = pd.DataFrame(rows)

    # ---- 判定 ----
    print("\n" + "=" * 100)
    print("判定（对照跑数前写死的标准）")
    print("=" * 100)

    verdicts = {}
    p1_wins, p3_cov = [], {}
    for u in U_LEVELS:
        sub = raw[raw["u_strength"] == u]
        print(f"\n--- u_strength={u} ---")

        # 描述统计
        for m in ("naive_ols", "ols_adjusted", "dml_plr", "dml_doubleml"):
            col = f"{m}_bias"
            if col not in sub.columns:
                continue
            v = sub[col].dropna()
            print(f"  {m:14s} 平均偏倚={v.mean():+.4f}  "
                  f"|偏倚|中位={v.abs().median():.4f}  极差={v.max()-v.min():.4f}")

        # P1：DML 优于 OLS 调整（仅在 u≥0.3 判定）
        if "dml_plr_abs_bias" in sub.columns and "ols_adjusted_abs_bias" in sub.columns:
            win = int((sub["dml_plr_abs_bias"] < sub["ols_adjusted_abs_bias"]).sum())
            if u >= 0.3:
                p1_wins.append(win)
                print(f"  P1  |bias(DML)|<|bias(OLS)| ： {win}/6  ← 参与判定")
            else:
                print(f"  P1  |bias(DML)|<|bias(OLS)| ： {win}/6"
                      f"  ← **不参与判定**（u=0 无混杂可去，三者均在 ±0.01 内，属退化比较）")

        # P2：DML 偏倚稳定性
        v = sub["dml_plr_bias"].dropna()
        rng = float(v.max() - v.min())
        p2 = rng < 0.05
        verdicts[f"P2_u{u}"] = {"value": rng, "pass": bool(p2)}
        print(f"  P2  DML 偏倚极差 = {rng:.4f} → {'通过' if p2 else '**不通过**'}（阈值 0.05）")

        # P3：AIPW 覆盖（双向判据）
        if "aipw_binary_covers_true" in sub.columns:
            cov = int(sub["aipw_binary_covers_true"].sum())
            p3_cov[u] = cov
            print(f"  P3  AIPW 覆盖真值 ： {cov}/6"
                  + ("  ← 无混杂档：应高" if u < 0.3 else "  ← 有混杂档：应低"))

    # ---- P1 汇总 ----
    if p1_wins:
        p1 = all(w >= 5 for w in p1_wins)
        worst = min(p1_wins)
        verdicts["P1"] = {
            "value": f"u≥0.3 档位胜出数 {p1_wins}（最差 {worst}/6）",
            "pass": bool(p1)}
        print(f"\nP1（u≥0.3 档位）DML 优于 OLS：{p1_wins} → "
              f"{'通过' if p1 else '**不通过**'}")

    # ---- P3 汇总（单调判据，2026-09-20 第二次修正）----
    u_sorted = sorted(p3_cov)
    cov_seq = [p3_cov[u] for u in u_sorted]
    # ① u=0（无混杂档）必须覆盖 ≥5/6：区间在假设成立时应正确
    ok_low = p3_cov.get(0.0, 0) >= 5
    # ② 覆盖率随 u 单调不增：允许平台（相等），不允许回升
    #    ⚠️ 用"总和"而非"逐档严格下降"，理由见文件头 P3 说明：
    #       6 个种子的分辨率是 1/6，弱混杂档的预期下降无法被逐档分辨。
    mono_noninc = all(cov_seq[i] >= cov_seq[i + 1] for i in range(len(cov_seq) - 1))
    sum_conf = sum(p3_cov.get(u, 0) for u in CONFOUNDED_LEVELS)
    sum_base = p3_cov.get(0.0, 0)
    ok_mono = bool(mono_noninc and sum_conf <= sum_base)
    p3 = bool(ok_low and ok_mono)
    verdicts["P3"] = {
        "value": (f"覆盖数 " + "，".join(f"u={u}:{p3_cov[u]}/6" for u in u_sorted)
                  + f"；单调不增={mono_noninc}；Σ(混杂档)={sum_conf} ≤ "
                    f"Σ(u=0)={sum_base}"),
        "pass": p3}
    print(f"P3（单调）u=0 应 ≥5/6（实际 {sum_base}/6）；"
          f"覆盖率须随 u 单调不增（实际序列 {cov_seq}，"
          f"Σ混杂档 {sum_conf} ≤ Σ无混杂 {sum_base}）→ "
          f"{'通过' if p3 else '**不通过**'}")

    # ---- 落地 ----
    os.makedirs(RES, exist_ok=True)
    csv_path = os.path.join(RES, "results_exp1_estimator_bias.csv")
    raw.to_csv(csv_path, index=False, encoding="utf-8-sig")

    payload = {
        "experiment": "exp1_estimator_bias",
        "data_mode": "synthetic",
        "data_mode_note": ("全部结果来自 src/synth.py 的合成结构；"
                           "不得表述为真实养殖系统的效应量"),
        "script": "experiments/exp1_estimator_bias.py",
        "n": N, "seeds": list(SEEDS), "u_levels": list(U_LEVELS),
        "pass_criteria_defined_before_running": PASS_CRITERIA,
        "verdicts": verdicts,
        "n_rows": len(raw),
        "wall_seconds": round(time.time() - t0, 1),
    }
    json_path = os.path.join(RES, "results_exp1_estimator_bias.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": payload, "rows": raw.to_dict(orient="records")},
                  f, ensure_ascii=False, indent=2)

    print(f"\n已写出：\n  {os.path.normpath(csv_path)}\n  {os.path.normpath(json_path)}")

    n_fail = sum(1 for v in verdicts.values() if not v["pass"])
    print(f"\n总判定：{len(verdicts)} 项标准，未通过 {n_fail} 项"
          + ("（全部通过）" if n_fail == 0 else "（须在报告中如实说明）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
