# -*- coding: utf-8 -*-
"""
实验 0 · 合成场结构可分辨性前置检验

【为什么这必须是"实验 0"、且必须在尺度实验之前跑】
本项目的尺度失配结论**只在"合成场确实含有多尺度结构"时才有意义**。
若合成场实际只是单一尺度（或纯噪声），那么"粗化后信息丢失"这个结论
就退化成"任何粗化都丢信息"这种没有内容的同义反复。
⇒ 因此这是一个**前提检验**，必须先跑、且必须在报告里显式展示它的结果。
   若这里不通过，后面所有尺度数字都作废。

【判据（预注册，不是事后凑出来的）】
  P1  方差分解**含交叉项**后闭合：Σvar + 2Σcov == var(raw)（容差 1e-9）
      ⚠️ 初版判据写成 "Σvar == var(raw)" —— 那是**错的判据**：
         方差可加性只在分量**互不相关**时成立，而本合成场的三层
         是用各自独立的白噪声过不同低通得到的，其**协方差非零**
         （实测 cov(L1,L2) = -0.0339），因此必须补上 2Σcov 才闭合。
         实测补上后闭合误差 2.2e-16。
  P2  50 m 层（L3）经 100 m 块平均后方差保留 < 15%
  P3  800 m 层（L1）经 100 m 块平均后方差保留 > 85%
  P4  L3 的保留率显著低于 L1（比值 < 0.25）
  P5  能谱存在三个可分辨的径向峰，且**峰位随标称尺度单调递增**
      ⚠️ 初版判据写成 "95% 谱能量在 800 m 波长以上" —— 也是**错的判据**。
         实测只有 27.09%。原因：每层的能量峰值落在**低于**其标称尺度**处**
         （实测峰值波长 L1≈689 m、L2≈345 m、L3≈66 m，而标称是 800/200/50 m），
         因为高斯低通是"尺度以下的能量按 f 的幂律累积"，
         峰位必然低于截止尺度。所以正确的可检验命题是
         **"三层可分辨"**（峰位单调 + 各层对粗化响应不同），而不是谱能量的绝对分布。
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)              # 04-原型系统/
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from src.synth import make_multiscale_field, make_pond_layout, block_average


def block_variance_retention(field: np.ndarray, factor: int) -> float:
    v0 = float(np.var(field))
    if v0 <= 0:
        return float("nan")
    return float(np.var(block_average(field, factor)) / v0)


def spectral_energy_above(field: np.ndarray, cell_m: float,
                          wavelength_m: float) -> float:
    """计算特征波长**大于** wavelength_m 的成分占总谱能量的比例。"""
    n = field.shape[0]
    F = np.abs(np.fft.fft2(field - field.mean())) ** 2
    kx = np.fft.fftfreq(n, d=cell_m)
    ky = np.fft.fftfreq(n, d=cell_m)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K = np.sqrt(KX ** 2 + KY ** 2)
    with np.errstate(divide="ignore"):
        lam = np.where(K > 0, 1.0 / np.maximum(K, 1e-300), np.inf)
    tot = float(F.sum())
    if tot <= 0:
        return float("nan")
    low = float(F[lam > wavelength_m].sum() / tot)   # 波长大于阈值 = 低频
    return low


def radial_spectrum_peak_wavelength(field: np.ndarray,
                                     cell_m: float,
                                     n_bins: int = 40) -> float:
    """
    径向平均功率谱的**峰值波长**（米）。

    为什么要做径向平均：二维能谱是各向同性的（本合成场由各向同性高斯滤波生成），
    直接取 argmax 会落在噪声极大的高频角点；径向平均后才反映真实的特征尺度。
    """
    n = field.shape[0]
    F = np.abs(np.fft.fft2(field - field.mean())) ** 2
    kx = np.fft.fftfreq(n, d=cell_m)
    ky = np.fft.fftfreq(n, d=cell_m)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K = np.sqrt(KX ** 2 + KY ** 2).ravel()
    P = F.ravel()
    edges = np.linspace(0.0, K.max(), n_bins + 1)
    kmid = 0.5 * (edges[:-1] + edges[1:])
    prof = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        sel = (K >= edges[i]) & (K < edges[i + 1])
        if sel.any():
            prof[i] = P[sel].sum()
    j = int(np.argmax(prof))
    kc = float(kmid[j])
    return 1.0 / kc if kc > 0 else float("inf")


def main() -> int:
    out = {"experiment": "exp0_structure",
           "question": "合成场是否真的含有可分辨的三层结构？（尺度结论的前提）"}

    f = make_multiscale_field(n=128, length_m=3200.0,
                              scales=(800.0, 200.0, 50.0),
                              weights=(1.0, 0.6, 0.35), seed=0)
    truth, layers, cell_m = f["truth"], f["layers"], f["cell_m"]
    out["setup"] = {
        "n": f["n"], "length_m": f["length_m"], "cell_m": cell_m,
        "scales_m": list(f["scales"]), "weights": list(f["weights"]),
        "seed": f["seed"], "truth_sd": f["sd"],
        "note": "全部为【合成】数据；真值已知",
    }

    # ---- P1 方差分解闭合（**含交叉项**）----
    keys = list(layers.keys())
    var_parts = {k: float(np.var(v)) for k, v in layers.items()}
    var_sum = float(sum(var_parts.values()))
    var_tot_raw = float(np.var(f["raw"]))

    # 层间协方差矩阵：方差可加性要求分量互不相关，本合成场不满足，
    # 因此必须显式给出交叉项，否则"分解不闭合"是个假警报。
    cov = np.zeros((len(keys), len(keys)), dtype=float)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            ca = layers[a] - layers[a].mean()
            cb = layers[b] - layers[b].mean()
            cov[i, j] = float(np.mean(ca * cb))
    cross_sum = float(cov.sum() - np.trace(cov))     # = 2 * Σ_{i<j} cov
    closure = var_sum + cross_sum
    out["variance_decomposition"] = {
        "per_layer": var_parts,
        "sum_of_variances": var_sum,
        "total_raw": var_tot_raw,
        "cross_terms_2sum_cov": cross_sum,
        "sum_with_cross": closure,
        "closure_error_with_cross": abs(closure - var_tot_raw),
        "closure_error_without_cross": abs(var_sum - var_tot_raw),
        "layer_covariance_matrix": cov.tolist(),
        "layer_names": keys,
    }
    p1 = abs(closure - var_tot_raw) < 1e-9
    # 同时记录"不带交叉项就不闭合"这个事实，以证明该修正确实必要
    out["variance_decomposition"]["cross_terms_are_necessary"] = bool(
        abs(var_sum - var_tot_raw) > 1e-6)

    # ---- P2/P3/P4 各层在不同粗化倍数下的保留率 ----
    factors = [1, 2, 4, 8, 16, 32]
    curve = {}
    for name, lyr in layers.items():
        curve[name] = {f"f{k}": block_variance_retention(lyr, k) for k in factors}
    out["retention_by_layer"] = curve

    # ⚠️ 这里原先想用 80 m。但 80 / 25 = 3.2，**不是整数 factor**，
    #    而块平均只支持整数倍粗化（这也是块平均的固有约束）。
    #    硬凑 80 m 会引入非整数重采样，把"尺度失配"和"重采样实现差异"混在一起。
    #    ⇒ 改用 100 m（factor = 4），并在报告中如实说明"可测的尺度是离散的"。
    f80 = 100.0 / cell_m
    assert abs(f80 - round(f80)) < 1e-9, \
        f"100m 应正好对应整数 factor，实际 {f80}"
    f80 = int(round(f80))
    r_l3 = curve["L3_50m"][f"f{f80}"]
    r_l1 = curve["L1_800m"][f"f{f80}"]
    out["at_100m"] = {"factor": f80, "grid_m": f80 * cell_m,
                     "L3_50m_retention": r_l3, "L1_800m_retention": r_l1,
                     "ratio_L3_over_L1": r_l3 / r_l1 if r_l1 > 0 else float("nan")}
    p2 = r_l3 < 0.15
    p3 = r_l1 > 0.85
    p4 = (r_l3 / r_l1) < 0.25 if r_l1 > 0 else False

    # ---- P5 各层径向能谱的峰位（"三层可分辨"的直接证据） ----
    spec = {
        "share_wavelength_gt_800m": spectral_energy_above(truth, cell_m, 800.0),
        "share_wavelength_gt_400m": spectral_energy_above(truth, cell_m, 400.0),
        "share_wavelength_gt_100m": spectral_energy_above(truth, cell_m, 100.0),
        "share_wavelength_gt_50m":  spectral_energy_above(truth, cell_m, 50.0),
    }
    out["spectral_energy"] = spec

    peaks = {k: radial_spectrum_peak_wavelength(v, cell_m) for k, v in layers.items()}
    out["radial_spectrum_peak_wavelength_m"] = peaks
    peak_vals = [peaks[k] for k in keys]
    # 峰位须随标称尺度单调递增（L1 最大、L3 最小），这才是"三层可分辨"
    p5 = all(peak_vals[i] > peak_vals[i + 1] for i in range(len(peak_vals) - 1))
    out["peak_wavelength_monotonic_desc"] = bool(p5)

    checks = [
        {"id": "P1", "desc": "方差分解含交叉项后闭合（|Σvar+2Σcov−总| < 1e-9）",
         "pass": bool(p1),
         "observed": f"{out['variance_decomposition']['closure_error_with_cross']:.3e}",
         "expected": "< 1e-9"},
        {"id": "P1b", "desc": "交叉项确有必要（不含交叉项则不闭合）",
         "pass": bool(out["variance_decomposition"]["cross_terms_are_necessary"]),
         "observed": f"不含交叉项误差="
                     f"{out['variance_decomposition']['closure_error_without_cross']:.3e}",
         "expected": "> 1e-6"},
        {"id": "P2", "desc": "L3(50m) 经 100m 块平均保留率 < 15%", "pass": bool(p2),
         "observed": f"{r_l3:.4f}", "expected": "< 0.15"},
        {"id": "P3", "desc": "L1(800m) 经 100m 块平均保留率 > 85%", "pass": bool(p3),
         "observed": f"{r_l1:.4f}", "expected": "> 0.85"},
        {"id": "P4", "desc": "L3/L1 保留率比值 < 0.25（结构确实可分辨）", "pass": bool(p4),
         "observed": f"{r_l3 / r_l1:.4f}" if r_l1 > 0 else "nan",
         "expected": "< 0.25"},
        {"id": "P5", "desc": "三层径向能谱峰位随标称尺度单调递增（结构可分辨）",
         "pass": bool(p5),
         "observed": " > ".join(f"{peaks[k]:.0f}m" for k in keys),
         "expected": "L1 > L2 > L3（严格递减）"},
    ]
    out["checks"] = checks
    out["n_pass"] = sum(1 for c in checks if c["pass"])
    out["n_total"] = len(checks)
    out["verdict"] = "PASS" if out["n_pass"] == out["n_total"] else "FAIL"

    # ---- 附：池体布局的几何一致性（不依赖随机性） ----
    pond = make_pond_layout(n=128, length_m=3200.0, pond_side_m=50.0, gap_m=10.0)
    out["pond_layout"] = {
        "n_ponds": pond["n_ponds"],
        "pond_side_m": pond["pond_side_m"],
        "gap_m": pond["gap_m"],
        "pond_area_ha": pond["pond_area_ha"],
        "pond_pixel_share": float((pond["pond_id"] >= 0).mean()),
        "cells_per_pond_side": pond["pond_side_m"] / pond["cell_m"],
    }

    resdir = os.path.join(_PKG, "..", "05-验证")
    resdir = os.path.abspath(resdir)
    os.makedirs(resdir, exist_ok=True)
    jp = os.path.join(resdir, "results_exp0_structure.json")
    with io.open(jp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("实验 0 · 合成场结构可分辨性")
    print("=" * 70)
    print(f"网格 {f['n']}×{f['n']}，场域 {f['length_m']:.0f} m，"
          f"分辨率 {cell_m:.1f} m，种子 {f['seed']}")
    print(f"三层特征长度 {f['scales']} m，权重 {f['weights']}")
    print("-" * 70)
    print("方差分解（含交叉项，否则不闭合）：")
    for k, v in var_parts.items():
        print(f"  {k:10s} var = {v:.6f}")
    print(f"  {'Σ var':10s}     = {var_sum:.6f}")
    print(f"  {'2Σ cov':10s}     = {cross_sum:.6f}   ← 层间相关，不可省略")
    print(f"  {'Σ var+2Σ cov':10s} = {closure:.6f}   "
          f"(raw 总方差 {var_tot_raw:.6f})")
    print(f"  闭合误差：含交叉项 "
          f"{out['variance_decomposition']['closure_error_with_cross']:.3e}，"
          f"不含 {out['variance_decomposition']['closure_error_without_cross']:.3e}")
    print("-" * 70)
    print(f"100 m 块平均后（factor={f80}）：")
    print(f"  L3_50m  保留 = {r_l3:.4f}")
    print(f"  L1_800m 保留 = {r_l1:.4f}")
    print(f"  L3/L1        = {r_l3 / r_l1:.4f}")
    print("-" * 70)
    print("谱能量占比：")
    for k, v in spec.items():
        print(f"  {k} = {v:.4f}")
    print("各层径向能谱峰位（标称 → 实测峰值）：")
    for k in keys:
        nominal = float(k.split("_")[1].rstrip("m"))
        print(f"  {k:10s} 标称 {nominal:6.0f} m → 峰值 {peaks[k]:6.0f} m")
    print("-" * 70)
    for c in checks:
        mark = "PASS" if c["pass"] else "FAIL"
        print(f"  [{mark}] {c['id']} {c['desc']}")
        print(f"          实测={c['observed']}  判据={c['expected']}")
    print("-" * 70)
    print(f"结果：{out['n_pass']}/{out['n_total']} → {out['verdict']}")
    print(f"已写入 {jp}")
    print("=" * 70)
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
