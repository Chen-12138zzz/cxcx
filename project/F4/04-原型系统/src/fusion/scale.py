# -*- coding: utf-8 -*-
"""
诊断 1 · 尺度失配（scale mismatch）

【要回答的问题（Q2）】
当输出网格**粗于**决策单元（高位池单池 ~50 m，而输出网格可能 100 m–1 km）时，
任何"精度"数字还站得住吗？失配的代价如何随尺度比增长？

【为什么这是真问题，而不是工程细节】
Silverthorn et al. (2025, PLOS Sustain. Transform. 4(1):e0000155) 统计：
71 篇 GIS-MCE 养殖选址文献中 **49 篇（>2/3）未报告空间分辨率**，
仅 33.3% 报告了最终输出分辨率，且**重采样方法常未说明**。
⇒ 读者无法判断"这个适宜性结论作用于多大地块"，也无法判断它是否与决策单元匹配。

【核心方法论：两个必须分开的量】
本项目把"网格变粗的后果"拆成**两个不同性质**的量，刻意不合并：

  (A) `variance_retention` —— **信息量**保留率。
      真值场在粗化前后的方差比。它只反映"高频结构丢了多少"，
      **与"是否排序正确"无关**。
  (B) `rank_agreement` —— **决策**一致性。
      把所有池体像元的真值降序排名，与粗网格打分的排名做 Spearman ρ。
      它反映"决策顺序是否被打乱"。

⚠️ 这不是学术洁癖。把这两者混起来的直接后果是：
   一个方案可以"保留了 95% 的方差"却"把前 10 名换了 7 个" ——
   因为**小尺度结构的绝对方差占比可以很低，却恰好决定池与池之间的排序**。
   本项目的主结果之一即是要量化这种分离。

【与非池体像元的处理】
粗化后每个网格可能跨越池体与塘埂。本诊断提供两种聚合口径并**要求显式选择**：
  - 'pond_mean'  ：只在池体像元上取平均（语义上对应"这一片池的平均适宜性"）
  - 'all_mean'   ：全部像元平均（对应"这一片区域的适宜性"）
两种口径在池体占比高时接近、在池体占比低时显著不同 —— 该差异本身要被报告。
"""
from __future__ import annotations

import numpy as np

from ..mce import rank_descending, spearman_rho, top_k_overlap


def aggregate_to_coarse(field: np.ndarray, factor: int,
                        mask: np.ndarray | None = None,
                        mode: str = "all_mean") -> np.ndarray:
    """
    把高分辨率场聚合到粗网格。

    参数
    ----
    field  : (n, n) 场
    factor : 粗化倍数（n 必须能被整除）
    mask   : (n, n) bool，True 表示参与聚合（如池体）。mode='all_mean' 时忽略。
    mode   : 'all_mean' 或 'pond_mean'
             'pond_mean' 下，若某粗格内**无任何**有效像元，结果为 np.nan
             （**刻意不填 0** —— 填 0 会让下游把"无数据"当成"最不适宜"）
    """
    if mode not in ("all_mean", "pond_mean"):
        raise ValueError(f"未知 mode：{mode!r}")
    n = field.shape[0]
    if field.shape[0] != field.shape[1]:
        raise ValueError("要求方阵")
    if n % factor != 0:
        raise ValueError(f"边长 {n} 不能被 factor={factor} 整除")
    m = n // factor

    if mode == "all_mean":
        return field.reshape(m, factor, m, factor).mean(axis=(1, 3))

    if mask is None:
        raise ValueError("mode='pond_mean' 需要提供 mask")
    w = mask.astype(float)
    num = (field * w).reshape(m, factor, m, factor).sum(axis=(1, 3))
    den = w.reshape(m, factor, m, factor).sum(axis=(1, 3))
    out = np.full((m, m), np.nan, dtype=float)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def _coarse_cell_truth(field: np.ndarray, factor: int,
                       mask: np.ndarray | None, mode: str) -> np.ndarray:
    """粗网格上的"真值"：用同一聚合口径对**真值场**做聚合（真值必须先聚合到同一尺度）。"""
    return aggregate_to_coarse(field, factor, mask=mask, mode=mode)


def scale_mismatch_curve(truth: np.ndarray,
                         mask: np.ndarray,
                         cell_m: float,
                         factors: tuple[int, ...] = (1, 2, 4, 8, 16),
                         mode: str = "pond_mean",
                         k_frac: float = 0.10) -> list[dict]:
    """
    尺度失配曲线：逐个 factor 计算信息量保留率与决策一致性。

    【两个量测的是什么，以及为什么必须用两个不同基准】
    粗化后，**每个粗格的能力上限**是"它覆盖区域内池体真值的均值" ——
    这是粗格能提供的、关于该区域内池体的**最充分统计量**，无法再改进。
    因此本函数把粗格得分与该上限对比，得到的 ρ 是**粗化本身造成的决策损失**
    （不含估计误差、不含权重误差）：

      (A) `variance_retention` = Var(粗格聚合值) / Var(高分辨率真值)
          量测"高频结构丢了多少" —— 纯信息量口径。
          ⚠️ **分母必须取 `truth[mask]` 的方差，不能取全像元方差**：
             分子只用池体像元，分母若用全像元就是两个不同总体相除，
             factor=1 时保留率不会等于 1（这是本项目实测踩到并修掉的一个缺陷）。
      (B) `rank_agreement` = Spearman(粗格得分, 粗格基准真值均值)
          量测"粗格之间的排序是否被打乱" —— 决策口径。
          基准 = `pond_mean`（只在池体像元上平均）；得分 = `all_mean`（含塘埂）。
          用两个**不同**口径是为了避免"自己跟自己比"（那会恒得 1，毫无信息）。
      (C) `pooled_rank_agreement` = Spearman(高分辨率真值排名, 粗格排名**展开回像元后**)
          量测"把粗格结论贴回每个像元时，像元级排名被破坏多少"（受 (A) 的低频限制）。

    参数
    ----
    truth   : (n, n) 真值场
    mask    : (n, n) bool，池体像元
    cell_m  : 高分辨率网格间距（米）
    factors : 待测粗化倍数（不能整除的会被跳过并记录）
    mode    : 聚合口径（见 aggregate_to_coarse）
    k_frac  : 前 k% 重叠率的 k 比例

    返回
    ----
    list[dict]，每项见下方 out.append。
    """
    n = truth.shape[0]
    if mask.shape != truth.shape:
        raise ValueError("mask 与 truth 形状不一致")
    if int(mask.sum()) < 2:
        raise ValueError("池体像元数 < 2，无法排名")

    # 【基准方差的正确口径 —— 必须与分子同一个总体】
    # ⚠️ 这里曾经写错：分子是"池体像元的聚合值"的方差，分母却用了**全像元**方差。
    #    两个不同的总体相除，使得 factor=1 时保留率不等于 1（实测 1.0221），
    #    这会让整条曲线的**水平**整体失真。正确做法是分母取
    #    **参与评估的池体像元自身的方差**，即 var(truth[mask])。
    #    （见修正记录 R-2 同类问题：F1 方向也踩过"分子分母不同尺度"。）
    v0 = float(np.var(truth[mask]))
    if v0 <= 0:
        raise ValueError("池体像元上的真值方差为 0，无法计算保留率（退化输入）")

    out = []
    skipped = []
    for f in factors:
        if f < 1:
            raise ValueError("factor 必须 >= 1")
        if n % f != 0:
            skipped.append(int(f))
            continue

        m = n // f
        # 粗格上的池体像元占比（用于说明聚合口径的影响）
        cnt = mask.astype(float).reshape(m, f, m, f).sum(axis=(1, 3))
        coarse = aggregate_to_coarse(truth, f, mask=mask, mode=mode)
        valid = ~np.isnan(coarse) & (cnt > 0)
        n_valid = int(valid.sum())

        vr = float(np.var(coarse[valid]) / v0) if n_valid > 0 else float("nan")

        # (B) 粗格之间：得分 vs 基准真值（此处"得分"= 该粗格聚合真值，
        #     故 f=1 时 ρ=1；随 f 增大，ρ 因粗格数减少而下降）
        rho_cells = float("nan")
        if n_valid >= 2:
            # 用两个**不同**的粗化口径构造"得分"与"基准"，避免自比：
            # 基准 = pond_mean（只在池体像元上平均）；得分 = all_mean（含塘埂）。
            # 两者在池体占比高时接近，占比低时分离 —— 这正是重采样口径的影响。
            alt = aggregate_to_coarse(truth, f, mask=None, mode="all_mean")
            sc = alt[valid]
            bs = coarse[valid]
            rho_cells = spearman_rho(bs, sc)

        # (C) 像元级：粗格结论展开回像元（每个池体像元取其所在粗格的值），
        #     与高分辨率真值比排名。这是最贴近"读者实际怎么用它"的口径。
        rho_pooled = float("nan")
        if mode == "pond_mean":
            expanded = np.repeat(np.repeat(coarse, f, axis=0), f, axis=1)
            pm = mask & ~np.isnan(expanded)
            if int(pm.sum()) >= 2:
                rho_pooled = spearman_rho(truth[pm], expanded[pm])

        # 前 k 重叠率（像元级，基于 (C) 的两套排名）
        topk = float("nan")
        if mode == "pond_mean":
            expanded = np.repeat(np.repeat(coarse, f, axis=0), f, axis=1)
            pm = mask & ~np.isnan(expanded)
            n_pm = int(pm.sum())
            if n_pm >= 1:
                k = max(1, int(round(n_pm * k_frac)))
                ids = np.flatnonzero(pm.ravel())
                va = truth.ravel()[ids]
                vb = expanded.ravel()[ids]
                oa = ids[np.argsort(-va, kind="stable")][:k]
                ob = ids[np.argsort(-vb, kind="stable")][:k]
                topk = float(len(set(oa.tolist()) & set(ob.tolist())) / k)

        out.append({
            "factor": int(f),
            "coarse_cell_m": float(cell_m * f),
            "grid_vs_pond_ratio": float((cell_m * f) / 50.0),
            "variance_retention": vr,
            "rank_agreement_cells": rho_cells,
            "rank_agreement_pooled": rho_pooled,
            "topk_overlap_pooled": topk,
            "n_valid_cells": n_valid,
            "n_pooled_pixels": int((mask & ~np.isnan(
                np.repeat(np.repeat(coarse, f, axis=0), f, axis=1))).sum())
            if mode == "pond_mean" else int(mask.sum()),
            "mean_pond_fraction_in_cell": float(np.mean(cnt[valid])) if n_valid else float("nan"),
        })

    if skipped:
        # 不静默丢弃：把跳过的 factor 写进结果，避免"看起来全跑了"
        out.append({
            "factor": None,
            "note": f"以下 factor 因不能整除边长 {n} 被跳过：{skipped}",
            "skipped_factors": skipped,
        })
    return out



def scale_mismatch_curve_perturbed(truth: np.ndarray,
                                   mask: np.ndarray,
                                   cell_m: float,
                                   factors: tuple[int, ...] = (1, 2, 4, 8, 16),
                                   mode: str = "pond_mean",
                                   seed: int = 0,
                                   k_frac: float = 0.10) -> list[dict]:
    """
    ⚠️ 备用口径（默认不用于主结论，仅供对照）。

    与 `scale_mismatch_curve` 的区别：在粗网格上叠加一个观测噪声，
    模拟"粗网格估计本身也有误差"。主结论**不使用**本函数，
    因为它把"尺度失配"与"估计噪声"两个因素混在一起（违反本项目"逐项分离"的纪律）。
    """
    rng = np.random.default_rng(seed)
    n = truth.shape[0]
    v0 = float(np.var(truth))
    out = []
    for f in factors:
        if n % f != 0:
            continue
        coarse = _coarse_cell_truth(truth, f, mask, mode)
        valid = ~np.isnan(coarse)
        # 用与粗化后场同量级的噪声
        sd = float(np.nanstd(coarse[valid])) if valid.any() else 0.0
        noisy = coarse + rng.normal(0.0, 0.25 * sd, size=coarse.shape)
        vr = float(np.var(coarse[valid]) / v0) if valid.any() else float("nan")
        rho = spearman_rho(coarse[valid], noisy[valid]) if int(valid.sum()) >= 2 else float("nan")
        out.append({
            "factor": int(f),
            "coarse_cell_m": float(cell_m * f),
            "grid_vs_pond_ratio": float((cell_m * f) / 50.0),
            "variance_retention": vr,
            "rank_agreement": rho,
            "n_valid_cells": int(valid.sum()),
            "note": "含观测噪声的对照口径，不作为主结论",
        })
    return out


# ----------------------------------------------------------------------------
# 与池体的几何关系（不依赖任何场，纯几何量）
# ----------------------------------------------------------------------------
def pond_vs_grid_geometry(n: int, length_m: float, pond_side_m: float,
                          gap_m: float, factors: tuple[int, ...]) -> list[dict]:
    """
    纯几何诊断：一个输出网格覆盖几口池？一个池占几个网格？

    这个量**不需要任何数据**，因此它是本组结论中**最稳**的一个 ——
    它只取决于池体边长与输出网格分辨率的相对关系。
    """
    cell_m = length_m / n
    out = []
    for f in factors:
        g = cell_m * f
        out.append({
            "factor": int(f),
            "grid_m": float(g),
            "cells_per_pond": float(pond_side_m / g),
            "ponds_per_cell": float((g / (pond_side_m + gap_m)) ** 2),
            "grid_coarser_than_pond": bool(g > pond_side_m),
            "ratio_grid_to_pond": float(g / pond_side_m),
        })
    return out
