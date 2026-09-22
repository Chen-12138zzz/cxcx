# -*- coding: utf-8 -*-
"""
诊断 2 · 融合的代价结构（fusion gain vs cost）

【要回答的问题（Q1）】
把多源数据融合进适宜性评估，**融合带来的精度增益**与**引入的方差代价**各是多少？

【核心纪律：不得预设方向（红线 G-3）】
本模块**不假定**"融合一定更好"。它输出的是一个**账本**，
允许出现"融合在该条件下反而更差"的结果，并把该结果如实报出。
这正是本项目与"融合是好的"这一类默认立场拉开距离的地方。

【为什么"增益"必须相对一个明确定义的基线】
若不指定基线，"融合提升了精度"这句话没有内容。本模块要求调用方
显式给出**基线方案**（单源或最粗源），并把所有增益表达为**相对该基线的差**。
默认基线 = 只用第 0 个图层（通常在语义上对应"最容易拿到的那一路数据"）。

【误差的三种分解（必须分开报）】
对一个由 K 个图层融合得到的得分场 S 与真值 T：

  1. `bias_hat`  —— 把 S 对 T 做最优线性拟合（含截距）后的**系统性偏离**。
                    它回答"会不会整体偏高/偏低"。
  2. `var_hat`   —— 拟合残差的方差。回答"围绕趋势抖得厉害吗"。
  3. `mse_hat`   —— 总均方误差（未经任何标定）。回答"直接用能不能用"。

⚠️ 把这三者分开的必要性：一个方案可以 MSE 很大但 bias 很小（纯粹方差问题），
也可以 bias 很大但 var 很小（纯粹系统性偏移，**可用一个常数修正掉**）。
只看 MSE 会把这两类完全不同的问题混为一谈，导致"该修哪个"无从判断。

【标定后的误差（可选）】
`mse_calibrated` = 允许用真值做一次**仿射标定**（a + b·S）后的残差 MSE。
它的作用是把"**结构误差**"与"**尺度/偏移误差**"分开：
若 mse_calibrated 远小于 mse_hat，说明**误差主要是尺度问题**（可由标定消除）；
若两者接近，说明**误差是结构性的**（标定救不了）。
"""
from __future__ import annotations

import numpy as np


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """一元线性最小二乘 y ≈ a + b x（显式实现，便于审计）。"""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size < 2:
        raise ValueError("样本数须一致且 >= 2")
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 0:
        raise ValueError("x 为常数（退化输入），无法拟合")
    b = float(((x - xm) * (y - ym)).sum() / sxx)
    a = float(ym - b * xm)
    resid = y - (a + b * x)
    return {"a": a, "b": b, "resid": resid,
            "var_resid": float(np.var(resid)),
            "bias_mean": float(np.mean(resid))}


def error_decomposition(pred: np.ndarray, truth: np.ndarray,
                        mask: np.ndarray | None = None) -> dict:
    """
    误差三分解 + 标定后误差。只在 mask 为 True 的位置上计算。

    返回 dict 字段见模块 docstring。
    """
    p = np.asarray(pred, dtype=float).ravel()
    t = np.asarray(truth, dtype=float).ravel()
    if p.size != t.size:
        raise ValueError("pred 与 truth 尺寸不一致")
    if mask is not None:
        m = np.asarray(mask, dtype=bool).ravel()
        p, t = p[m], t[m]
    if p.size < 3:
        raise ValueError("有效样本 < 3，无法做误差分解")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(t))):
        raise ValueError("输入含非有限值；请先显式处理缺失，不要静默跳过")

    vt = float(np.var(t))
    if vt <= 0:
        raise ValueError("真值为常数（退化输入）")

    fit = _linear_fit(p, t)
    mse_raw = float(np.mean((p - t) ** 2))
    mse_cal = float(np.mean(fit["resid"] ** 2))

    # 相关系数（判别力口径：排序是否被抓住）
    rho = float(np.corrcoef(p, t)[0, 1]) if p.size >= 2 else float("nan")

    return {
        "n": int(p.size),
        "mse_raw": mse_raw,
        "rmse_raw": float(np.sqrt(mse_raw)),
        "var_raw": float(np.var(p)),
        "bias_mean_raw": float(np.mean(p - t)),
        "mse_calibrated": mse_cal,
        "rmse_calibrated": float(np.sqrt(mse_cal)),
        "var_resid_calibrated": fit["var_resid"],
        "calib_a": fit["a"],
        "calib_b": fit["b"],
        "pearson_r": rho,
        "var_truth": vt,
        # 相对量：以真值方差为标尺，便于跨设置比较
        "nmse_raw": mse_raw / vt,
        "nmse_calibrated": mse_cal / vt,
        # 尺度误差占比：1 - 标定后MSE/原始MSE
        "share_of_error_from_scale": (1.0 - mse_cal / mse_raw) if mse_raw > 0 else float("nan"),
    }


def fusion_ledger(source_scores: dict[str, np.ndarray],
                  truth: np.ndarray,
                  mask: np.ndarray,
                  baseline: str | None = None) -> dict:
    """
    融合账本：对每个**方案**（单源 / 多源融合）算出误差分解，并给出相对基线的增益。

    参数
    ----
    source_scores : {方案名: (n,n) 得分场}
                    必须包含至少一个单源方案与（若要做融合分析）一个融合方案。
                    命名建议：`single_<layer>` 与 `fused_<layers>`。
    truth : (n,n) 真值
    mask  : (n,n) bool，参与评估的像元（如池体）
    baseline : 基线方案名。缺省取**键名排序后的第一个**，并显式报出选了谁。

    返回
    ----
    dict:
      baseline         : 实际使用的基线名
      per_scheme       : {方案名: error_decomposition}
      gain_vs_baseline : {方案名: {'nmse_reduction', 'rmse_reduction', 'r_reduction'}}
                         —— 正数表示**优于**基线
      verdict          : 'fusion_helps' / 'fusion_hurts' / 'inconclusive'
                         **不做方向预设**；由数据决定
    """
    if not source_scores:
        raise ValueError("source_scores 为空")
    names = sorted(source_scores.keys())
    if baseline is None:
        baseline = names[0]
    if baseline not in source_scores:
        raise ValueError(f"基线 {baseline!r} 不在方案中：{names}")

    per = {}
    for nm, sc in source_scores.items():
        if np.shape(sc) != np.shape(truth):
            raise ValueError(f"方案 {nm!r} 形状 {np.shape(sc)} 与真值 {np.shape(truth)} 不一致")
        per[nm] = error_decomposition(sc, truth, mask)

    b = per[baseline]
    gain = {}
    for nm, e in per.items():
        gain[nm] = {
            # 正 = 该方案误差更小（更好）
            "nmse_reduction": float(b["nmse_raw"] - e["nmse_raw"]),
            "rmse_reduction": float(b["rmse_raw"] - e["rmse_raw"]),
            "r_increase": float(e["pearson_r"] - b["pearson_r"]),
            "mse_calibrated_reduction": float(b["mse_calibrated"] - e["mse_calibrated"]),
        }

    # 判定：只看**融合类方案**（键名含 'fused'）相对基线的 NMSE 变化
    fused = [n for n in names if n.startswith("fused")]
    verdict = "inconclusive"
    detail = "未提供融合方案（键名须以 'fused' 开头），无法判定融合效果"
    if fused:
        ds = [gain[n]["nmse_reduction"] for n in fused]
        all_pos = all(d > 0 for d in ds)
        all_neg = all(d < 0 for d in ds)
        if all_pos:
            verdict = "fusion_helps"
        elif all_neg:
            verdict = "fusion_hurts"
        else:
            verdict = "inconclusive"
        detail = ("各融合方案相对基线的 NMSE 降幅（正=更好）："
                  + ", ".join(f"{n}:{gain[n]['nmse_reduction']:+.4f}" for n in fused))

    return {
        "baseline": baseline,
        "per_scheme": per,
        "gain_vs_baseline": gain,
        "verdict": verdict,
        "verdict_detail": detail,
    }


def weight_perturbation(scores_by_layer: list[np.ndarray] | np.ndarray,
                        base_weights: np.ndarray,
                        mask: np.ndarray,
                        n_draws: int = 200,
                        rel_sd: float = 0.25,
                        seed: int = 0,
                        k_frac: float = 0.10) -> dict:
    """
    权重扰动实验（这是 D 组"把缺失计数变成可计算后果"的具体手段）。

    【它回答什么】
    Silverthorn et al. (2025) 统计出 **13/71 篇未报告权重赋值**。
    本函数把这句话变成：**在合理扰动范围内，前 k% 的推荐地块平均换掉几个？**

    扰动模型：对数正态扰动再归一化（保证权重恒正、和为 1）。
      w'_k ∝ w_k * exp(N(0, rel_sd^2))
    rel_sd=0.25 对应"专家给出的权重可有约 25% 的相对不确定" ——
    **这是一个示范性设定，不是实测的专家分歧度**（见红线 G-5）。

    返回
    ----
    dict:
      spearman_vs_base   : 扰动后得分排序与基准的 Spearman ρ（各次抽样的统计）
      topk_kept_share    : 前 k% **保持率**（各次抽样的统计）
                           1.0 = 前 k 名一个没变；0.0 = 前 k 名全换
      topk_kept_count    : 前 k 名里**留住**的个数（= share × k）
      topk_changed_count : 前 k 名里**换掉**的个数（= k − kept），决策语义最直观
      ...的 mean / median / p05 / p95 / min / max
      n_draws, rel_sd, k
    """
    stack = np.asarray(scores_by_layer, dtype=float)
    if stack.ndim != 3:
        raise ValueError("scores_by_layer 应为 (K,H,W)")
    K = stack.shape[0]
    w0 = np.asarray(base_weights, dtype=float).ravel()
    if w0.size != K:
        raise ValueError("权重数与层数不一致")
    if abs(w0.sum() - 1.0) > 1e-9:
        raise ValueError("基准权重须归一化（和为 1）")

    from ..mce import wlc_score, spearman_rho, rank_descending

    base_score = wlc_score(stack, w0)["score"]
    base_rank = rank_descending(base_score, mask)
    n_used = base_rank.size
    k = max(1, int(round(n_used * k_frac)))

    rng = np.random.default_rng(seed)
    rhos, tov = [], []
    for _ in range(n_draws):
        logs = rng.normal(0.0, rel_sd, size=K)
        w = w0 * np.exp(logs)
        w = w / w.sum()
        sc = wlc_score(stack, w)["score"]
        rhos.append(spearman_rho(base_score[mask], sc[mask]))
        ra = rank_descending(sc, mask)
        tov.append(len(set(base_rank[:k].tolist()) & set(ra[:k].tolist())) / k)

    r = np.array(rhos, dtype=float)
    t = np.array(tov, dtype=float)

    def stat(a):
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
                "min": float(a.min()), "max": float(a.max())}

    # 【"换掉了几个" 的正确写法】
    # ⚠️ 这里曾经写错：把 `topk_kept_count` 实现为 `share * k`。
    #    那是**保留个数**（rel_sd=0 时必然等于 k，即"一个都没换"），
    #    但字段名读起来像"保留的个数"而决策者关心的是"**换掉**了几个"，
    #    两种读法在 rel_sd=0 这件事上会给出完全相反的心理预期（一个是 k，一个是 0）。
    #    为避免歧义，现在同时给出两个**语义无歧义**的字段：
    #      topk_changed_count = k - kept  ——"前 k 名里换掉几个"（rel_sd=0 → 0）
    #      topk_kept_count    = kept      ——"前 k 名里留住几个"（rel_sd=0 → k）
    #    并把两者的恒等式写成注释，便于核对。
    return {
        "n_draws": int(n_draws), "rel_sd": float(rel_sd), "k": int(k),
        "n_units_ranked": int(n_used),
        "base_weights": w0.tolist(),
        "spearman_vs_base": stat(r),
        # 前 k 名里**换掉**的个数（决策语义上最直观的"后果"）
        "topk_changed_count": stat(k - t * k),
        # 前 k 名里**留住**的个数（恒等式：changed + kept == k）
        "topk_kept_count": stat(t * k),
        "topk_kept_share": stat(t),
        "note": ("rel_sd 为示范性设定（对数正态扰动），不代表实测的专家分歧度；"
                 "见唯一事实来源 §5.3 红线 G-5。恒等式："
                 "topk_kept_count + topk_changed_count == k（逐次抽样成立）"),
    }


def weight_perturbation_curve(scores_by_layer: list[np.ndarray] | np.ndarray,
                              base_weights: np.ndarray,
                              mask: np.ndarray,
                              rel_sds: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30),
                              n_draws: int = 200,
                              seed: int = 0,
                              k_frac: float = 0.10) -> list[dict]:
    """
    权重不确定度 → 结论不稳度的**曲线**（D 组的核心图）。

    ⚠️ 必须有一条 rel_sd=0 的基线：若 rel_sd=0 时重合率不是 1.0，
    说明实现有问题（**恒不通过 / 恒通过都是可疑信号**，
    这条纪律来自 F1 方向 R-13 的教训）。
    """
    out = []
    for sd in rel_sds:
        r = weight_perturbation(scores_by_layer, base_weights, mask,
                                n_draws=n_draws, rel_sd=sd, seed=seed, k_frac=k_frac)
        out.append({
            "rel_sd": float(sd),
            "k": r["k"],
            "n_units_ranked": r["n_units_ranked"],
            "spearman_mean": r["spearman_vs_base"]["mean"],
            "spearman_p05": r["spearman_vs_base"]["p05"],
            "topk_kept_mean": r["topk_kept_count"]["mean"],
            "topk_changed_mean": r["topk_changed_count"]["mean"],
            "topk_kept_share_mean": r["topk_kept_share"]["mean"],
            "topk_kept_share_p05": r["topk_kept_share"]["p05"],
        })
    return out
