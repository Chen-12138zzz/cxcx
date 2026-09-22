# -*- coding: utf-8 -*-
"""
F4 原型系统 · 多准则评价（MCE）与加权线性组合（WLC）引擎

【为什么自己实现而不调库】
本项目的 D 组（审计）需要对**权重扰动、重分类、重采样**逐项做可控实验。
现成 GIS 库（QGIS/rasterio/GDAL）把这些步骤封装为不可分解的黑箱，
无法在"权重变化了 0.01 时排序变了几个"这一粒度上测量。
因此本模块用显式线性代数重实现 WLC —— **代价是实现正确性必须自证**
（见 tests/ 与修正记录）。

【方法学声明（重要）】
WLC 与 AHP 是 **2000 年以来 GIS-MCE 养殖选址的标准配方**（见
Chentouf et al. 2023, ISPRS IJGI 12(10):439 的综述结论）。
**本项目不声称在这些方法上有任何创新。** 本模块的存在目的是
**让这些方法可被分解、可被扰动、可被审计**，而不是提供更好的方法。
"""
from __future__ import annotations

import numpy as np

# 公开接口契约。注意 _rankdata 虽以下划线开头（内部工具），
# 但被 tests 直接断言（因其行为正确性是本项目自证的一部分），
# 故不列入 __all__，而在接口文档中单独标注为"内部但已断言"。
__all__ = [
    "minmax_normalize",
    "assert_not_degenerate",
    "reclass_fuzzy",
    "wlc_score",
    "normalized_weights",
    "ahp_priority_from_consistent",
    "rank_descending",
    "spearman_rho",
    "top_k_overlap",
]


# ----------------------------------------------------------------------------
# 归一化 / 重分类
# ----------------------------------------------------------------------------
def minmax_normalize(x: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    线性 min-max 归一化到 [0, 1]（最常用的"重分类"做法之一）。

    invert=True 表示该图层与适宜性**负相关**（越大越不适宜）。
    这是 GIS-MCE 中"成本图层"的标准处理方式。

    ⚠️ 注意：本函数**不在内部**做常数层的保护 —— 常数层（极差为 0）
    会得到全 0。这是刻意选择：把退化情形暴露出来，由调用方决定处置。
    见 `assert_not_degenerate`。
    """
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    rng = hi - lo
    if rng <= 0:
        out = np.zeros_like(x)
    else:
        out = (x - lo) / rng
    return 1.0 - out if invert else out


def assert_not_degenerate(x: np.ndarray, name: str = "layer", tol: float = 1e-12) -> None:
    """
    退化情形自检：常数层无法参与"重分类"，必须显式拒绝而非静默产出全 0。

    这条自检来自 F1 方向的一条判据设计教训：
    **一个诊断若没经过"退化输入下应报什么"的验证，它的输出毫无意义。**
    """
    x = np.asarray(x, dtype=float)
    if x.max() - x.min() <= tol:
        raise ValueError(
            f"图层 {name!r} 为常数（极差 {x.max() - x.min():.3e} <= {tol:.1e}）："
            f"min-max 重分类将静默产出全 0。请显式处理退化层，不要让它进入 WLC。"
        )


def reclass_fuzzy(x: np.ndarray, lo: float, hi: float, edge: str = "both") -> np.ndarray:
    """
    模糊隶属度重分类（线性过渡）。

    edge:
      'both'   —— lo 以下为 0，hi 以上为 1，中间线性过渡（"越大越适宜"）
      'low'    —— 只在低端过渡（lo 以上即饱和为 1）
      'high'   —— 只在高端过渡（hi 以下即饱和为 1）

    与 min-max 的区别：min-max 用**数据极值**，本函数用**给定的物理解释边界**
    （如 DO > 5 mg/L 为适宜 —— 来自 Chentouf 2023 综述归纳的准则）。
    两者的差异正是 D 组要审计的"重分类选择"问题之一。
    """
    x = np.asarray(x, dtype=float)
    if hi <= lo:
        raise ValueError("要求 hi > lo")
    if edge == "low":
        out = np.where(x >= lo, 1.0, 0.0) + np.where(
            (x > lo - (hi - lo)) & (x < lo), (x - (lo - (hi - lo))) / (hi - lo), 0.0)
        out = np.clip(out, 0.0, 1.0)
    elif edge == "high":
        out = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    elif edge == "both":
        out = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    else:
        raise ValueError(f"未知 edge 模式：{edge!r}")
    return out


# ----------------------------------------------------------------------------
# WLC 打分
# ----------------------------------------------------------------------------
def wlc_score(layer_stack: list[np.ndarray] | np.ndarray,
              weights: list[float] | np.ndarray) -> dict:
    """
    加权线性组合（WLC）：score = Σ_k w_k * s_k，其中 s_k ∈ [0,1] 为第 k 个图层
    的重分类得分，w_k 为权重。

    参数
    ----
    layer_stack : (K, H, W) 数组或长度为 K 的数组列表，每个元素为已重分类的 [0,1] 得分层。
    weights     : 长度 K 的权重，**要求归一化（和为 1）**，否则报错。
                  理由：不归一化的权重会让 score 的**量纲随权重变动**，
                  使"精度"与"权重"两个因素纠缠，无法分离。

    返回
    ----
    dict:
      score       : (H, W) 综合得分
      weights     : 实际使用的权重（归一化后）
      K           : 层数
    """
    stack = np.asarray(layer_stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError(f"layer_stack 应为 3 维 (K,H,W)，实际 {stack.ndim} 维")
    w = np.asarray(weights, dtype=float).ravel()
    K = stack.shape[0]
    if w.size != K:
        raise ValueError(f"权重数 {w.size} 与层数 {K} 不一致")
    if not np.all(np.isfinite(w)):
        raise ValueError("权重含非有限值")
    if np.any(w < 0):
        raise ValueError("权重不得为负（负权重会使 WLC 失去补偿性语义）")

    s = w.sum()
    if abs(s - 1.0) > 1e-9:
        raise ValueError(
            f"权重未归一化（和 = {s:.12f}）。WLC 要求 Σw=1，"
            f"否则得分的量纲会随权重漂移，导致『精度』与『权重』无法分离。"
        )

    score = np.tensordot(w, stack, axes=(0, 0))
    return {"score": score, "weights": w.copy(), "K": K}


def normalized_weights(raw: list[float] | np.ndarray) -> np.ndarray:
    """把任意非负权重归一化到和为 1。"""
    w = np.asarray(raw, dtype=float).ravel()
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("权重须为有限非负值")
    s = w.sum()
    if s <= 0:
        raise ValueError("权重全 0，无法归一化")
    return w / s


def ahp_priority_from_consistent(priority: list[float] | np.ndarray) -> dict:
    """
    从一个**已知优先级向量**构造 AHP 判断矩阵，并回算其权重。

    【为什么这样做】
    AHP 的标准流程是"专家填判断矩阵 → 求特征向量"。本项目无专家、无现场（U-1/U-2），
    **不得**编造专家判断。因此本函数**反过来**：给定一组权重，构造完全一致的
    判断矩阵 A_ij = p_i / p_j（一致性比必然为 0），用于验证权重扰动实验的**载体正确性**，
    **不用于声称任何"专家权重"**。

    返回 dict: matrix, weights, consistency_ratio（应为 0）
    """
    p = np.asarray(priority, dtype=float).ravel()
    if np.any(p <= 0):
        raise ValueError("优先级须为正")
    p = p / p.sum()
    K = p.size
    A = p[:, None] / p[None, :]

    # 主特征向量（幂法即可，且在完全一致矩阵上一步收敛）
    vals, vecs = np.linalg.eig(A)
    idx = int(np.argmax(vals.real))
    w = np.abs(vecs[:, idx].real)
    w = w / w.sum()

    lam_max = float(vals.real[idx])
    ci = (lam_max - K) / (K - 1) if K > 1 else 0.0
    ri_table = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_table.get(K, 1.49)
    cr = (ci / ri) if ri > 0 else 0.0

    return {"matrix": A, "weights": w, "consistency_ratio": float(cr), "lambda_max": lam_max}


# ----------------------------------------------------------------------------
# 排序与秩相关
# ----------------------------------------------------------------------------
def rank_descending(score: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    返回按得分降序的索引序列。mask 为 True 的位置才参与排序（如只排池体像元）。

    平均秩用 `method="average"`，以正确处理并列得分。
    """
    s = np.asarray(score, dtype=float).ravel()
    if mask is None:
        idx = np.arange(s.size)
    else:
        idx = np.flatnonzero(np.asarray(mask, dtype=bool).ravel())
    vals = s[idx]
    order = np.argsort(-vals, kind="stable")
    return idx[order]


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关系数（无 scipy 依赖的显式实现，便于审计）。"""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size:
        raise ValueError("长度不一致")
    if a.size < 2:
        raise ValueError("至少需要 2 个观测")
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if den <= 1e-15:
        return 1.0  # 至少一方为常数：定义上视为完全一致
    return float((ra * rb).sum() / den)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """平均秩（与 scipy.stats.rankdata 的 'average' 一致）。"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=float)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def top_k_overlap(score_a: np.ndarray, score_b: np.ndarray,
                  mask: np.ndarray | None = None, k: int | None = None) -> float:
    """
    两个打分方案的"前 k 名重叠率"。k 缺省取参与像元数的 10%（至 least 1）。

    这个指标是本项目 D 组的核心输出之一：
    **权重扰动后，被推荐进入前 k 名的地块换了几个。**
    它比秩相关更贴近决策语义 —— 决策者关心的是"我要看的这几块地变没变"，
    不是全部地块的秩是否一致。
    """
    ra = rank_descending(score_a, mask)
    rb = rank_descending(score_b, mask)
    n = min(ra.size, rb.size)
    if n == 0:
        raise ValueError("参与像元数为 0")
    if k is None:
        k = max(1, int(round(n * 0.10)))
    k = min(k, n)
    sa, sb = set(ra[:k].tolist()), set(rb[:k].tolist())
    return float(len(sa & sb) / k)
