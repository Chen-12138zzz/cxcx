# -*- coding: utf-8 -*-
"""
实验 3 · 把"报告缺失"变成"可计算后果"（Q3，D 组主输出）

【要审计的对象 —— 引用他已完成的工作，本项目不重复统计】
Silverthorn et al. (2025, PLOS Sustain. Transform. 4(1):e0000155) 系统分析 71 篇
GIS-MCE 养殖选址文献，报告缺失统计：
  • 空间分辨率  49/71 未报告（仅 33.3% 报告最终输出分辨率）
  • 权重赋值    13/71 未报告
  • 重分类分值  24/71 未报告
  • 数据来源    10/71 未识别、28/71 仅部分识别
  • 建模软件     9/71 未报告
  • 时点        59/71 为单一时点；仅 1/71 考虑气候变化情景

【本实验做什么】
**不重新统计这些缺失**（那是上述论文已完成的工作），
而是把三条**计数**各自映射到一个**可计算后果**：

  | 缺失项     | 原统计    | 本实验的对应后果                        |
  |-----------|----------|---------------------------------------|
  | 空间分辨率 | 49/71    | 尺度失配代价（见实验 1：比值最高 2.29×） |
  | 权重赋值   | 13/71    | 排序翻转概率（本实验主体）               |
  | 重分类分值 | 24/71    | 方案互换重叠率（本实验主体）             |

【红线 G-5（必须写在最前面）】
本实验产出的所有"后果数字"都**建立在示范性设定之上**（权重扰动幅度 rel_sd、
重分类的解释边界 lo/hi）。它们的作用是**说明这类缺失的影响量级**，
**不是**对任何具体已发表研究的评价。
⇒ **不得**用本实验的输出指责任何具体论文"不稳健"。
   本实验只在**我们自己的合成场**上运行，从不接收真实研究的内部参数。

【判据（预注册）】
  A1  退化自检：rel_sd = 0 时前 k% 保留率**恰为** 1.0
      （恒不通过 / 恒通过都是可疑信号 —— 来自 F1 方向 R-13 教训）
  A2  权重扰动后果随 rel_sd 单调不增（保留率）且恒等式 kept+changed == k
  A3  重分类方案互换产生可测后果（前 k 重叠 < 1），但也不至于完全无关
  A4  所有参与层的极差 > 0（无退化层混入 WLC）
  A5  时效性检查清单**不含任何数字**（纪律 8：无数据不产数）
  A6  k 的口径一致：k == round(n_ranked × k_frac)，且 k >= 1
  A7  多子验证退化点：rel_sd=0 时 200 次抽样全部为 1.0（不是平均为 1）
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from src.synth import make_multiscale_field, make_pond_layout
from src.mce import (minmax_normalize, wlc_score, normalized_weights,
                     assert_not_degenerate, rank_descending)
from src.fusion.gain import weight_perturbation, weight_perturbation_curve
from src.audit import reclass_choice_sensitivity, timeliness_flags, audit_summary


def main() -> int:
    out = {"experiment": "exp3_audit",
           "question": "把三条'报告缺失'计数各自变成什么样的可计算后果？",
           "data_mode": "【合成】",
           "quoted_statistics": {
               "source": ("Silverthorn et al. 2025, PLOS Sustain. Transform. "
                          "4(1):e0000155"),
               "n_papers": 71,
               "missing_spatial_resolution": 49,
               "missing_weights": 13,
               "missing_reclass": 24,
               "missing_data_source": 10,
               "partial_data_source": 28,
               "missing_software": 9,
               "single_timepoint": 59,
               "note": "以上为他文已完成的统计，本实验不重复统计，仅引用其计数",
           }}

    f = make_multiscale_field(n=128, length_m=3200.0, seed=0)
    pond = make_pond_layout(n=128, length_m=3200.0)
    mask = pond["pond_id"] >= 0
    L = f["layers"]

    raw_stack = np.stack([L["L1_800m"], L["L2_200m"], L["L3_50m"]], axis=0)  # 原始物理量
    rel_stack = np.stack([minmax_normalize(x) for x in raw_stack], axis=0)   # 已重分类
    w = normalized_weights([1.0, 0.6, 0.35])
    k_frac = 0.10

    # ---- A4 无退化层 ----
    degen_ok = True
    degen_info = []
    for i in range(raw_stack.shape[0]):
        try:
            assert_not_degenerate(raw_stack[i], f"layer_{i+1}")
            degen_info.append({"layer": i + 1, "range": float(
                raw_stack[i].max() - raw_stack[i].min()), "degenerate": False})
        except ValueError as e:
            degen_ok = False
            degen_info.append({"layer": i + 1, "error": str(e), "degenerate": True})
    out["layer_degeneracy_check"] = degen_info

    # ---- 审计项 2：权重扰动 ----
    rel_sds = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50)
    curve = weight_perturbation_curve(rel_stack, w, mask, rel_sds=rel_sds,
                                      n_draws=200, seed=0, k_frac=k_frac)
    out["weight_perturbation_curve"] = curve

    # 退化自检（多次抽样，逐次都要为 1.0，不是平均为 1.0）
    r0 = weight_perturbation(rel_stack, w, mask, n_draws=200, rel_sd=0.0,
                             seed=1, k_frac=k_frac)
    a1 = (abs(r0["topk_kept_share"]["min"] - 1.0) < 1e-12
          and abs(r0["topk_kept_share"]["max"] - 1.0) < 1e-12)
    out["degeneracy_at_rel_sd_0"] = {
        "kept_share_min": r0["topk_kept_share"]["min"],
        "kept_share_max": r0["topk_kept_share"]["max"],
        "spearman_min": r0["spearman_vs_base"]["min"],
        "spearman_max": r0["spearman_vs_base"]["max"],
        "changed_count_mean": r0["topk_changed_count"]["mean"],
        "n_draws": r0["n_draws"], "k": r0["k"],
    }

    shares = [c["topk_kept_share_mean"] for c in curve]
    a2a = all(shares[i] >= shares[i + 1] - 1e-12 for i in range(len(shares) - 1))
    # 恒等式 kept + changed == k
    ident_ok = all(abs(c["topk_kept_mean"] + c["topk_changed_mean"] - c["k"]) < 1e-9
                   for c in curve)
    a2 = a2a and ident_ok

    # ---- 审计项 3：重分类方案互换 ----
    # lo/hi 是**示范性设定**：取合成层的分位数（并显式标注，红线 G-5）
    lo = float(np.percentile(raw_stack, 25))
    hi = float(np.percentile(raw_stack, 75))
    rcs = reclass_choice_sensitivity(raw_stack, w, mask, lo, hi,
                                     schemes=("minmax", "fuzzy"), k_frac=k_frac)
    out["reclass_sensitivity"] = rcs
    pair = rcs["pairs"][0]
    a3 = 0.0 < pair["topk_overlap"] < 1.0

    # ---- 审计项 2'：时效性（刻意不产数）----
    tf = timeliness_flags()
    a5 = all(not any(ch.isdigit() for ch in it["item"]) for it in tf)
    out["timeliness_flags"] = tf

    # ---- A6 k 口径 ----
    n_ranked = int(mask.sum())
    k_expect = max(1, int(round(n_ranked * k_frac)))
    a6 = all(c["k"] == k_expect for c in curve) and k_expect >= 1
    out["k_consistency"] = {"n_ranked": n_ranked, "k_frac": k_frac,
                            "k_expected": k_expect,
                            "k_observed": [c["k"] for c in curve]}

    # ---- 汇总表：三条缺失 → 三个后果 ----
    summ = audit_summary(rel_stack, w, mask, lo, hi,
                         scale_curve=None, rel_sds=rel_sds,
                         n_draws=200, seed=2, k_frac=k_frac)
    a7 = summ["degeneracy_check"]["ok"] is True
    out["audit_summary"] = {
        "degeneracy_check": summ["degeneracy_check"],
        "k_frac": summ["k_frac"],
        "n_weight_levels": len(summ["weight_sensitivity"]),
    }
    # 把实验 1 的尺度后果并进来（跨实验引用，证明三条缺失可并列）
    exp1_path = os.path.join(os.path.dirname(_PKG), "05-验证",
                             "results_exp1_scale.json")
    if os.path.exists(exp1_path):
        with io.open(exp1_path, encoding="utf-8") as fh:
            e1 = json.load(fh)
        csr = e1.get("cross_seed_robustness", {})
        out["scale_consequence_from_exp1"] = {
            "note": "引用实验 1 的结果作为'空间分辨率缺失'的后果",
            "max_ratio_decision_over_info": (
                max(csr.get("mean_ratio_per_scale", [float("nan")]))
                if csr.get("mean_ratio_per_scale") else float("nan")),
            "mean_ratio_per_scale": csr.get("mean_ratio_per_scale"),
            "scales_m": csr.get("scales_m"),
        }

    missing_table = [
        {"missing_item": "空间分辨率", "count": "49/71",
         "consequence": "尺度失配代价：决策相对损失 / 信息相对损失 最高 2.28×",
         "source": "实验 1"},
        {"missing_item": "权重赋值", "count": "13/71",
         "consequence": (f"rel_sd=0.30 时前 {int(k_frac*100)}% 平均换掉 "
                         f"{[c['topk_changed_mean'] for c in curve if c['rel_sd']==0.30][0]:.1f} 个"
                         f"（共 {k_expect} 个）"),
         "source": "本实验"},
        {"missing_item": "重分类分值", "count": "24/71",
         "consequence": (f"min-max 与 fuzzy 互换时前 {int(k_frac*100)}% 换掉 "
                         f"{pair['topk_changed_mean']:.1f} 个，秩相关 "
                         f"{pair['spearman']:.4f}"),
         "source": "本实验"},
        {"missing_item": "数据来源/年份", "count": "10/71 未报 + 28/71 部分",
         "consequence": "**不产数** —— 只输出应检查项清单（纪律 8）",
         "source": "本实验 timeliness_flags()"},
    ]
    out["missing_to_consequence_table"] = missing_table

    checks = [
        {"id": "A1", "desc": "[恒等自检] rel_sd=0 时前k保留率逐次恰为 1.0",
         "pass": bool(a1),
         "observed": (f"min={r0['topk_kept_share']['min']:.12f}, "
                      f"max={r0['topk_kept_share']['max']:.12f}"),
         "expected": "min=max=1.0"},
        {"id": "A2", "desc": "保留率随 rel_sd 单调不增 且 kept+changed==k",
         "pass": bool(a2),
         "observed": (" → ".join(f"{s:.4f}" for s in shares)
                      + f" | 恒等式={'成立' if ident_ok else '不成立'}"),
         "expected": "单调不增 且 恒等式成立"},
        {"id": "A3", "desc": "重分类方案互换产生可测后果（0<重叠<1）",
         "pass": bool(a3),
         "observed": f"spearman={pair['spearman']:.4f}, "
                     f"topk_overlap={pair['topk_overlap']:.4f}",
         "expected": "0 < topk_overlap < 1"},
        {"id": "A4", "desc": "所有参与层极差>0（无退化层混入 WLC）",
         "pass": bool(degen_ok),
         "observed": f"{len(degen_info)} 层全部通过", "expected": "全部通过"},
        {"id": "A5", "desc": "[纪律8] 时效性清单不含任何数字（无数据不产数）",
         "pass": bool(a5), "observed": f"{len(tf)} 项均不含数字",
         "expected": "不含数字"},
        {"id": "A6", "desc": "k 口径一致：k == round(n×k_frac) 且 k>=1",
         "pass": bool(a6), "observed": f"n={n_ranked}, k={k_expect}",
         "expected": f"{k_expect}"},
        {"id": "A7", "desc": "审计摘要的 degeneracy_check 通过",
         "pass": bool(a7), "observed": summ["degeneracy_check"]["note"],
         "expected": "ok=True"},
    ]
    out["checks"] = checks
    out["n_pass"] = sum(1 for c in checks if c["pass"])
    out["n_total"] = len(checks)
    out["verdict"] = "PASS" if out["n_pass"] == out["n_total"] else "FAIL"
    out["disclaimer"] = ("红线 G-5：rel_sd 与 lo/hi 均为示范性设定，"
                         "不代表实测的专家分歧度或领域阈值；"
                         "不得用于评价任何具体已发表研究。")

    resdir = os.path.abspath(os.path.join(_PKG, "..", "05-验证"))
    os.makedirs(resdir, exist_ok=True)
    jp = os.path.join(resdir, "results_exp3_audit.json")
    with io.open(jp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("=" * 78)
    print("实验 3 · 把『报告缺失』变成『可计算后果』")
    print("=" * 78)
    print(f"数据模式：{out['data_mode']}  |  池体像元 {n_ranked}  |  k={k_expect}"
          f"（前 {int(k_frac*100)}%）")
    print("引用的他文统计（不重复统计）：49/71 未报分辨率、13/71 未报权重、"
          "24/71 未报重分类")
    print("-" * 78)
    print("审计项 2 · 权重扰动 → 排序翻转：")
    print(f"  {'rel_sd':>7}{'ρ均值':>9}{'保留率均值':>11}{'换掉个数均值':>13}"
          f"{'换掉占比':>10}")
    for c in curve:
        print(f"  {c['rel_sd']:>7.2f}{c['spearman_mean']:>9.4f}"
              f"{c['topk_kept_share_mean']:>11.4f}"
              f"{c['topk_changed_mean']:>13.2f}"
              f"{c['topk_changed_mean']/c['k']:>10.4f}")
    print(f"  [退化自检] rel_sd=0 时保留率 min={r0['topk_kept_share']['min']:.12f} "
          f"max={r0['topk_kept_share']['max']:.12f}（应均为 1.0）")
    print("-" * 78)
    print("审计项 3 · 重分类方案互换（lo/hi 为示范性设定，红线 G-5）：")
    print(f"  lo={lo:.4f}  hi={hi:.4f}（合成层 25%/75% 分位数）")
    print(f"  min-max vs fuzzy：spearman={pair['spearman']:.4f}  "
          f"前k重叠={pair['topk_overlap']:.4f}  "
          f"换掉 {pair['topk_changed_mean']:.1f}/{pair['k']} 个")
    print("-" * 78)
    print("审计项 2' · 时效性（**刻意不产数**，纪律 8）：")
    for it in tf:
        print(f"  • {it['item']}")
    print("-" * 78)
    print("★ 三条缺失 → 三个可计算后果：")
    print(f"  {'缺失项':<16}{'原统计':<20}{'对应后果'}")
    for m in missing_table:
        print(f"  {m['missing_item']:<16}{m['count']:<20}{m['consequence']}")
    print("-" * 78)
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['id']} {c['desc']}")
        print(f"          实测={c['observed']}")
    print("-" * 78)
    print(f"结果：{out['n_pass']}/{out['n_total']} → {out['verdict']}")
    print(f"已写入 {jp}")
    print("=" * 78)
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
