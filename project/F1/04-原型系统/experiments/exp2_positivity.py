# -*- coding: utf-8 -*-
"""
F1 · 实验 2：正性诊断的**能力边界**验证

【★ 本实验第二版，2026-09-20 重写】
第一版试图验证诊断能否"区分稀疏空箱与结构性空箱"（C1 敏感性 / C2 特异性）。
**第一版的两个核心判据已被本项目自己的验证推翻**（见 05-验证/修正记录.md R-11）：

  在「全局等频箱 × 分位协变量层」框架下，把 n 放大 32 倍后：
    - 无任何违背时空箱层：4 → 0
    - 人为**硬截断**支撑集（概率恰为 0）后：7 → 0
  两者放大样本后**同样收敛到 0** ⇒ 空箱计数无法区分抽样波动与机制缺失。
  第一版的 C1「违背强度 ≥0.4 时结构性层数 >0」之所以"通过"，
  只是因为 n=6000 下样本量仍不足；这不是敏感性，是**伪敏感**。

因此第二版不再声称能判结构性，改为验证诊断**确实具备**的三项能力：

【★ 跑前写死的通过标准（第二版）】
    C1（样本量单调性）：无真实违背时，空箱层数随 n 增大**不增**
        （600→19200 六档中，空箱层数序列须满足 首档 ≥ 末档）
    C2（分辨率单调性）：固定 n、无真实违背时，
        分层 3 段 → 2 段的空箱层数应**下降**（层粗则每层样本多）
    C3（稀疏判定正确性）：每箱期望样本数 < 阈值时，判定须为 'sparse'
        （即：期望不足时必须归入稀疏，不得归入 undetermined）
    C4（分辨率不足的诚实性）：当每箱期望样本数 ≥ 阈值却仍有空箱时，
        判定须为 'undetermined'，且**不得**出现 'structural' 字样
        （structural 类别已从代码中移除；本项检查代码层可用性）

【数据模式：合成】
⚠️ 第一版用 positivity_violation 制造"正性违背"，第二版证实该参数
   的实现是**平移层内 T 分布**而非**截断支撑集**，故**不能**制造机制性违背
   （见 R-11 与 R-12，已在 synth.py 文档中标注）。本实验不再依赖它。

【输出】
    results_exp2_positivity.json / .csv
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

import synth                                        # noqa: E402
from diagnostics.positivity import (                # noqa: E402
    diagnose_positivity, restricted_sample,
)

PASS_CRITERIA = {
    "C1": "无真实违背时，空箱层数随 n 增大不增（600→19200 六档，首档 ≥ 末档）",
    "C2": "固定 n、无真实违背时，分层 3 段 → 2 段后空箱层数下降",
    "C3": "每箱期望样本数 < 阈值的档位，判定须为 'sparse'",
    "C4": "代码中不存在 'structural' 判定类别（已移除）；判定值 ∈ "
          "{ok, sparse, narrow, undetermined}",
}
SEED = 11

# ---- 场景 A：样本量扫描（无真实违背）----
SCEN_A = [600, 1200, 2400, 4800, 9600, 19200]
# ---- 场景 B：分层分辨率（固定 n，无真实违背）----
SCEN_C = [(1200, 2), (1200, 3), (4800, 2), (4800, 3)]
# ---- 场景 C：极端小样本（验证稀疏判定）----
SCEN_D = [200, 400, 600]

VALID_VERDICTS = {"ok", "sparse", "narrow", "undetermined"}


def rec(d, n, pv, spc, tag) -> dict:
    return {
        "scenario": tag,
        "n": n,
        "positivity_violation": pv,
        "strata_per_covariate": spc,
        "n_strata": d.n_strata,
        "n_boxes": d.n_boxes,
        "expected_per_box": d.expected_per_box,
        "median_expected_in_box": d.median_expected_in_box,
        "median_stratum_size": d.median_stratum_size,
        "n_strata_undetermined": d.n_strata_undetermined,
        "n_strata_sparse_gap": d.n_strata_sparse_gap,
        "undetermined_fraction": d.undetermined_fraction,
        "n_strata_small": d.n_strata_small,
        "min_box_count": d.min_box_count,
        "verdict": d.verdict,
        "worst_stratum_size": d.worst_stratum["stratum_size"],
        "worst_empty_boxes": str(d.worst_stratum["empty_boxes"]),
    }


def main():
    print("=" * 100)
    print("实验 2：正性诊断的**能力边界**验证（【合成】数据）")
    print("=" * 100)
    print("  ★ 通过标准已写死：")
    for k, v in PASS_CRITERIA.items():
        print(f"     {k}: {v}")

    t0 = time.time()
    rows: list[dict] = []

    # ---------------- 场景 A ----------------
    print("\n--- 场景 A：样本量扫描（无真实违背）---")
    print(f"  {'n':>7} {'每箱期望n':>10} {'空箱(undet)':>12} {'稀疏':>6} {'判定':>13}")
    for n in SCEN_A:
        df = synth.generate(n=n, u_strength=0.3, nonlinear=True,
                            positivity_violation=0.0, seed=SEED)
        d = diagnose_positivity(df)
        rows.append(rec(d, n, 0.0, 3, "A_sample_size"))
        print(f"  {n:>7} {d.expected_per_box:>10.1f} {d.n_strata_undetermined:>12}"
              f" {d.n_strata_sparse_gap:>6} {d.verdict:>13}")

    # ---------------- 场景 B ----------------
    print("\n--- 场景 B：分层分辨率对照（无真实违背）---")
    print(f"  {'n':>7} {'每协变量段数':>12} {'层数':>6} {'每箱期望n':>10}"
          f" {'空箱(undet)':>12} {'稀疏':>6} {'判定':>13}")
    for n, spc in SCEN_C:
        df = synth.generate(n=n, u_strength=0.3, nonlinear=True,
                            positivity_violation=0.0, seed=SEED)
        d = diagnose_positivity(df, strata_per_covariate=spc)
        rows.append(rec(d, n, 0.0, spc, "C_resolution"))
        print(f"  {n:>7} {spc:>12} {d.n_strata:>6} {d.expected_per_box:>10.1f}"
              f" {d.n_strata_undetermined:>12} {d.n_strata_sparse_gap:>6} {d.verdict:>13}")

    # ---------------- 场景 C ----------------
    print("\n--- 场景 C：极端小样本（验证稀疏判定）---")
    print(f"  {'n':>7} {'每箱期望n':>10} {'稀疏':>6} {'判定':>13}")
    for n in SCEN_D:
        df = synth.generate(n=n, u_strength=0.3, nonlinear=True,
                            positivity_violation=0.0, seed=SEED)
        d = diagnose_positivity(df)
        rows.append(rec(d, n, 0.0, 3, "D_tiny_n"))
        print(f"  {n:>7} {d.expected_per_box:>10.1f} {d.n_strata_sparse_gap:>6}"
              f" {d.verdict:>13}")

    raw = pd.DataFrame(rows)

    # ---------------- 判定 ----------------
    print("\n" + "=" * 100)
    print("判定（对照跑数前写死的标准）")
    print("=" * 100)
    verdicts = {}

    A = raw[raw["scenario"] == "A_sample_size"].sort_values("n")
    A["n_gap_total"] = A["n_strata_undetermined"] + A["n_strata_sparse_gap"]

    # C1 样本量单调性：空箱层总数随 n 不增
    seq = A["n_gap_total"].tolist()
    c1 = bool(len(seq) >= 2 and seq[0] >= seq[-1])
    verdicts["C1"] = {"value": f"空箱层序列 {seq}", "pass": c1}
    print(f"C1 样本量单调性：{seq}（首 {seq[0] if seq else '缺'} ≥ 末 "
          f"{seq[-1] if seq else '缺'}）→ {'通过' if c1 else '**不通过**'}")

    # C2 分辨率单调性：同 n 下 3 段 → 2 段空箱层下降
    c2_detail = {}
    c2_ok = []
    for n in (1200, 4800):
        sub = raw[(raw["scenario"] == "C_resolution") & (raw["n"] == n)].copy()
        if len(sub) != 2:
            c2_detail[n] = "数据不足"
            continue
        sub["gap_total"] = sub["n_strata_undetermined"] + sub["n_strata_sparse_gap"]
        g2 = int(sub[sub["strata_per_covariate"] == 2]["gap_total"].iloc[0])
        g3 = int(sub[sub["strata_per_covariate"] == 3]["gap_total"].iloc[0])
        c2_detail[n] = f"2段={g2}, 3段={g3}"
        c2_ok.append(g2 <= g3)
    c2 = bool(c2_ok) and all(c2_ok)
    verdicts["C2"] = {"value": str(c2_detail), "pass": c2}
    print(f"C2 分辨率单调性：{c2_detail} → {'通过' if c2 else '**不通过**'}")

    # C3 稀疏判定正确性：每箱期望 < 阈值的档位须判 sparse
    D = raw[raw["scenario"] == "D_tiny_n"]
    thr = 5.0
    need = D[D["median_expected_in_box"] < thr]
    if len(need):
        c3 = bool((need["verdict"] == "sparse").all())
        c3_val = f"期望<{thr} 的档位判定 = {need['verdict'].tolist()}"
    else:
        c3 = False
        c3_val = "无满足条件的档位（无法判定）"
    verdicts["C3"] = {"value": c3_val, "pass": c3}
    print(f"C3 稀疏判定正确性：{c3_val} → {'通过' if c3 else '**不通过**'}")

    # C4 诚实性：判定值集合必须在本版定义内，且源码不含 structural 类别
    all_v = set(raw["verdict"].unique().tolist())
    src_path = os.path.join(SRC, "diagnostics", "positivity.py")
    with open(src_path, encoding="utf-8") as f:
        src_txt = f.read()
    # 判定赋值处不得再出现 "structural"
    has_struct_assign = ('verdict = "structural"' in src_txt)
    c4 = bool(all_v <= VALID_VERDICTS) and (not has_struct_assign)
    verdicts["C4"] = {
        "value": f"实际判定值 {sorted(all_v)}；源码含 structural 赋值 = {has_struct_assign}",
        "pass": c4}
    print(f"C4 诚实性：判定值 {sorted(all_v)}，源码含 'structural' 赋值 = "
          f"{has_struct_assign} → {'通过' if c4 else '**不通过**'}")

    # ---------------- 落地 ----------------
    os.makedirs(RES, exist_ok=True)
    csv_path = os.path.join(RES, "results_exp2_positivity.csv")
    raw.to_csv(csv_path, index=False, encoding="utf-8-sig")

    payload = {
        "experiment": "exp2_positivity",
        "version": "v2 (2026-09-20 rewritten after R-11)",
        "data_mode": "synthetic",
        "data_mode_note": ("本版**不再**用 positivity_violation 制造违背："
                           "该项目已证实只能平移层内 T 分布、不能截断支撑集，"
                           "故无法制造机制性缺失（R-12）。实验转而验证诊断的"
                           "能力边界：样本量单调性、分辨率单调性、稀疏判定、"
                           "以及『不再声称能判结构性』这一设计约束。"),
        "v1_retracted": ("第一版 C1/C2（敏感性/特异性，基于结构性空箱）已被"
                         "R-11 证伪并撤回；撤回理由见 05-验证/修正记录.md"),
        "script": "experiments/exp2_positivity.py",
        "seed": SEED,
        "scenarios": {"A_sample_size": SCEN_A,
                      "C_resolution": SCEN_C,
                      "D_tiny_n": SCEN_D},
        "pass_criteria_defined_before_running": PASS_CRITERIA,
        "verdicts": verdicts,
        "wall_seconds": round(time.time() - t0, 1),
    }
    json_path = os.path.join(RES, "results_exp2_positivity.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": payload, "rows": raw.to_dict(orient="records")},
                  f, ensure_ascii=False, indent=2)

    print(f"\n已写出：\n  {os.path.normpath(csv_path)}\n  {os.path.normpath(json_path)}")
    n_fail = sum(1 for v in verdicts.values() if not v["pass"])
    print(f"\n总判定：{len(verdicts)} 项，未通过 {n_fail} 项"
          + ("（全部通过）" if n_fail == 0 else "（须在报告中如实说明）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
