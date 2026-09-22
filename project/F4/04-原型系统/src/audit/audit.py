# -*- coding: utf-8 -*-
"""
审计模块 · 把"报告缺失"变成"可计算后果"

【要审计的对象】
Silverthorn et al. (2025, PLOS Sustain. Transform. 4(1):e0000155) 系统分析了 71 篇
GIS-MCE 养殖选址文献，统计出报告缺失：
  • 49/71 篇（>2/3）未报告空间分辨率；仅 33.3% 报告最终输出分辨率
  • 13/71 篇未报告权重赋值；24/71 篇未报告重分类分值
  • 10/71 篇未识别数据来源；28/71 篇仅部分识别
  • 9/71 篇未报告建模软件
  • 59/71 篇输出为单一时点；仅 1 篇考虑气候变化情景

【本模块做什么，不做什么】
**它不重新统计这些缺失**（那是上述论文已完成的工作）。
**它把这些缺失转化为可计算的后果**：
  • "未报告分辨率" ⇒ 读者无法判断结论与决策单元的**尺度比**，故后果 = 尺度失配代价
  • "未报告权重"   ⇒ 读者无法判断结论对权重的**依赖程度**，故后果 = 排序翻转概率
  • "未报告重分类" ⇒ 读者无法判断结论对**归一化选择**的敏感性
  • "未报告来源/年份" ⇒ 读者无法判断结论的**时效性**（本模块只做规则性判定，不产数）

【关键纪律（红线 G-5）】
本模块产出的所有"后果数字"都**建立在示范性设定之上**（权重扰动的幅度、
重分类方案的选择）。它们的作用是**说明这类缺失的影响量级**，
**不是**对任何具体已发表研究的评价。
⇒ 因此**不得**用本模块的输出指责任何具体论文"不稳健"。
   本模块只在**我们自己的合成场上**运行，从不接收真实研究的内部参数。
"""
from __future__ import annotations

import numpy as np

from ..mce import (minmax_normalize, reclass_fuzzy, wlc_score,
                   rank_descending, spearman_rho)
from ..fusion.gain import weight_perturbation_curve


# ----------------------------------------------------------------------------
# 审计项 1：重分类选择的后果
# ----------------------------------------------------------------------------
def reclass_choice_sensitivity(layer_stack: list[np.ndarray] | np.ndarray,
                               weights: np.ndarray,
                               mask: np.ndarray,
                               lo: float,
                               hi: float,
                               schemes: tuple[str, ...] = ("minmax", "fuzzy"),
                               k_frac: float = 0.10) -> dict:
    """
    重分类方案（min-max 线性 vs 模糊隶属度）对最终排序的影响。

    【为什么要审计这一项】
    Silverthorn 2025 统计出 **24/71 篇对重分类分值报告不足或缺失**。
    重分类决定了"某个温度值算多适宜"，不同方案会把同一片水域推到不同名次。
    本函数量化：**换一个同样合理、同样常见的重分类方案，前 k% 换掉几个。**

    参数
    ----
    layer_stack : (K,H,W) **原始物理量**层（未重分类）
    weights     : 长度 K，归一化权重
    mask        : (n,n) bool，参与排名的像元
    lo, hi      : 模糊重分类的物理解释边界。**必须由调用方给出**
                  （本项目不自行编造领域阈值；示范运行时使用合成层的分位数并显式标注）
    schemes     : 要对比的重分类方案

    返回
    ----
    dict: per_scheme 得分场摘要、两两间的 Spearman ρ 与前 k% 重叠率
    """
    stack = np.asarray(layer_stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError("layer_stack 应为 (K,H,W)")

    scores = {}
    for sch in schemes:
        rel = []
        for k in range(stack.shape[0]):
            x = stack[k]
            if sch == "minmax":
                rel.append(minmax_normalize(x))
            elif sch == "fuzzy":
                rel.append(reclass_fuzzy(x, lo, hi, edge="both"))
            else:
                raise ValueError(f"未知重分类方案：{sch!r}")
        scores[sch] = wlc_score(rel, weights)["score"]

    names = list(scores.keys())
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = scores[names[i]], scores[names[j]]
            ra = rank_descending(a, mask)
            rb = rank_descending(b, mask)
            n = min(ra.size, rb.size)
            k = max(1, int(round(n * k_frac)))
            ov = len(set(ra[:k].tolist()) & set(rb[:k].tolist())) / k
            pairs.append({
                "pair": f"{names[i]}__vs__{names[j]}",
                "spearman": spearman_rho(a[mask], b[mask]),
                "topk_overlap": float(ov),
                "k": int(k),
                "topk_changed_mean": float(k - ov * k),
            })

    return {
        "schemes": names,
        "n_units_ranked": int(np.asarray(mask, dtype=bool).sum()),
        "pairs": pairs,
        "lo": float(lo), "hi": float(hi),
        "note": ("lo/hi 为示范性设定的解释边界，不是任何具体研究使用的阈值；"
                 "见红线 G-5"),
    }


# ----------------------------------------------------------------------------
# 审计项 2：时效性规则判定（**不产数**）
# ----------------------------------------------------------------------------
def timeliness_flags(consider_typo: bool = False) -> list[dict]:
    """
    输出"时效性"这一类缺失的**检查项清单**（规则，不含任何数字）。

    【为什么这个函数不产数】
    唯一事实来源 §5.4 纪律 8：**无数据不产数**。
    我们不知道任何具体研究的数据采集年份，因此**不能**给出"过期年数"这类数字，
    只能给出"应当检查哪些项"的清单。
    这是刻意设计，不是未完成 —— 见 workspace-scaffold 的要求：
    "若某个模块无真实数据，**让它不产数**并写明这是刻意设计，而不是填占位值。"
    """
    return [
        {"item": "数据层所属年份是否逐层给出",
         "why": "同一模型混用不同年份图层会使结论无法对应任何一个真实时点",
         "source_of_concern": "Silverthorn et al. 2025：Reporting Template 表5 要求逐数据层说明年份"},
        {"item": "数据采集—建模—发表之间的时间差是否说明",
         "why": "发表年份不能可靠指示数据时效",
         "source_of_concern": "Silverthorn et al. 2025 原文关于 longevity 的论述"},
        {"item": "是否使用『典型』条件而非年际变异",
         "why": "仅 14/71 篇考虑自然灾害与年际气候变异（如季风、ENSO）",
         "source_of_concern": "Silverthorn et al. 2025 统计"},
        {"item": "是否考虑未来情景",
         "why": "仅 1/71 篇考虑气候变化情景",
         "source_of_concern": "Silverthorn et al. 2025 统计"},
    ]


# ----------------------------------------------------------------------------
# 审计项 3：完整审计摘要（把若干后果汇总成一张表）
# ----------------------------------------------------------------------------
def audit_summary(layer_stack: list[np.ndarray] | np.ndarray,
                  weights: np.ndarray,
                  mask: np.ndarray,
                  lo: float,
                  hi: float,
                  scale_curve: list[dict] | None = None,
                  rel_sds: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30),
                  n_draws: int = 200,
                  seed: int = 0,
                  k_frac: float = 0.10) -> dict:
    """
    审计摘要：把三类缺失（分辨率 / 权重 / 重分类）的后果放在同一张表里。

    这是本项目 D 组的主输出。它把 Silverthorn 2025 的三条**计数**
    （49/71、13/71、24/71）各自对应到一个**可计算后果**：

      | 缺失项 | 原统计 | 对应的可计算后果 |
      |---|---|---|
      | 空间分辨率 | 49/71 未报 | 尺度失配曲线（信息量保留率 + 决策一致性） |
      | 权重赋值 | 13/71 未报 | 权重扰动 → 前 k% 重叠率 |
      | 重分类分值 | 24/71 未报 | 重分类方案互换 → 前 k% 重叠率 |
    """
    w = np.asarray(weights, dtype=float).ravel()
    if abs(w.sum() - 1.0) > 1e-9:
        raise ValueError("权重须归一化（和为 1）")

    wcurve = weight_perturbation_curve(layer_stack, w, mask, rel_sds=rel_sds,
                                       n_draws=n_draws, seed=seed, k_frac=k_frac)
    rcs = reclass_choice_sensitivity(layer_stack, w, mask, lo, hi,
                                     schemes=("minmax", "fuzzy"), k_frac=k_frac)

    # 退化自检：rel_sd=0 时重叠率必须为 1.0
    base = [r for r in wcurve if r["rel_sd"] == 0.0]
    degeneracy_ok = True
    degeneracy_note = ""
    if base:
        degeneracy_ok = abs(base[0]["topk_kept_share_mean"] - 1.0) < 1e-9
        degeneracy_note = (f"rel_sd=0 时前 k% 保留率 = "
                           f"{base[0]['topk_kept_share_mean']:.12f}（应为 1.0）")
    else:
        degeneracy_ok = False
        degeneracy_note = "未提供 rel_sd=0 基线，无法验证实现正确性"

    return {
        "weight_sensitivity": wcurve,
        "reclass_sensitivity": rcs,
        "scale_sensitivity": scale_curve,
        "degeneracy_check": {"ok": bool(degeneracy_ok), "note": degeneracy_note},
        "k_frac": float(k_frac),
    }
