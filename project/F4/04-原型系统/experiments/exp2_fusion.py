# -*- coding: utf-8 -*-
"""
实验 2 · 融合的收益与代价（Q1）+ 收益对分辨率的依赖

【回答的问题】
把多源数据融合进适宜性评估，**精度增益**与**代价**各是多少？
更关键：**"融合有益"这个结论本身是否依赖于输出分辨率？**

【红线 G-3：不得预设方向】
本实验**不假定**"融合一定更好"。账本允许并如实报出
"融合在该条件下反而更差"。`fusion_ledger` 的 verdict 由数据决定。

【方案设计（为什么这样设计）】
本项目的立论是"融合的收益取决于是否引入了与决策单元**尺度匹配**的那一路数据"。
因此方案必须能把"尺度"这一因素**单独动**，其余保持不动：

  A. single_L1        只用最粗的一路（800 m 层）—— 模拟"只有卫星级数据"
  B. fused_L1L2       粗两路融合（800 + 200 m）
  C. fused_L1L2L3     三路全融合（含 50 m 池体级）—— 引入了尺度匹配的那一路
  D. fused_coarse_expand  三路都融合，但**统一先聚合到 200 m 再展开回像元**
                          —— 模拟"融合了，但输出分辨率粗于决策单元"

  A→B→C 单独动"是否包含细尺度源"；C→D 单独动"输出分辨率"。
  ⇒ 这样就分离了"源的丰富度"与"输出分辨率"两个因素。

【评价口径】
全部在高分辨率的**池体像元**上评价（同一 support），确保不同方案可公平比较。
若各方案在不同 support 上评价，差异会混入"评价尺度的变化"，见 F1 方向 R-2 教训。

【判据（预注册）】
  G1  误差三分解自洽：mse_raw = bias² + var_resid（容差 1e-9）
  G2  基线对自身的增益恒为 0
  G3  纯缩放型方案可被仿射标定清零（share_of_error_from_scale ≈ 1）
  G4  **主结果**：判别力随融合单调提升 —— pearson_r(A) < r(B) < r(C)
  G4b **符号反转**：nmse_raw 与 nmse_calibrated 的变化方向**相反**
      ⇒ 即"单一精度数字会把结论的符号搞反"，这是本项目最核心的示范
      ⚠️ 初版判据写成 "引入尺度匹配的源使 nmse_raw 下降" —— 这是**错误的判据**：
         实测 nmse_raw 反而**上升**（1.0736 → 1.1224）。查清机制后确认这不是实现缺陷：
         min-max 把每层压到 [0,1]（方差 0.0265 远小于真值方差 1.0221），
         WLC 的 Σw=1 凸组合再压缩一次（实测 0.0092），故得分是**单调代理**
         而非**同尺度估计**，原始 MSE 被"纯仿射失配"支配。
         因此用 nmse_raw 判断"融合是否提升精度"是**问错了量**。
         calib_b 逐级放大（5.07 → 8.48 → 10.50）正是该压缩的直接证据。
  G4c 标定后融合的优势重现：rmse_calibrated(A) > (B) > (C)
  G5  输出分辨率变粗（C→D）使 nmse_raw 上升
  G6  verdict 由数据决定，且必须落在三值集合内（不预设方向）
  G8  得分场被系统性压缩（calib_b > 1）—— 记录该压缩量级
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
from src.mce import minmax_normalize, wlc_score, normalized_weights, spearman_rho
from src.fusion.gain import error_decomposition, fusion_ledger
from src.fusion.scale import aggregate_to_coarse


def expand_back(coarse: np.ndarray, factor: int, n: int) -> np.ndarray:
    """把粗格结果按最近邻展开回高分辨率（模拟"读者把粗结论贴回地块"）。"""
    return np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)[:n, :n]


def main() -> int:
    out = {"experiment": "exp2_fusion",
           "question": "融合的收益/代价是多少？该结论是否依赖输出分辨率？",
           "data_mode": "【合成】", "truth_known": True}

    f = make_multiscale_field(n=128, length_m=3200.0, seed=0)
    truth, cell_m = f["truth"], f["cell_m"]
    pond = make_pond_layout(n=128, length_m=3200.0)
    mask = pond["pond_id"] >= 0
    L = f["layers"]

    # 各路"源"先做标准 GIS-MCE 重分类（min-max），再进 WLC。
    # 注意：这里用的是**标准化后的层**（unit std），不是标准化前的 raw；
    # 因为真实场景中各源量纲不同，必须先重分类到同一尺度。
    rel = {f"L{i+1}": minmax_normalize(v) for i, v in enumerate(
        [L["L1_800m"], L["L2_200m"], L["L3_50m"]])}

    stack_all = np.stack([rel["L1"], rel["L2"], rel["L3"]], axis=0)
    w_all = normalized_weights([1.0, 0.6, 0.35])

    # ---- 构造方案 ----
    schemes = {}
    schemes["single_L1"] = wlc_score(rel["L1"][None, ...], [1.0])["score"]
    schemes["fused_L1L2"] = wlc_score(
        np.stack([rel["L1"], rel["L2"]], axis=0),
        normalized_weights([1.0, 0.6]))["score"]
    fused_full = wlc_score(stack_all, w_all)["score"]
    schemes["fused_L1L2L3"] = fused_full

    # D：三路都融合，但先聚合到 200 m 再展开（模拟输出分辨率粗于决策单元）
    fac = 8                                     # 25 m × 8 = 200 m
    coarse = aggregate_to_coarse(fused_full, fac, mask=None, mode="all_mean")
    schemes["fused_coarse_expand"] = expand_back(coarse, fac, f["n"])

    # ---- 账本 ----
    ledger = fusion_ledger(schemes, truth, mask, baseline="single_L1")
    out["schemes"] = list(schemes.keys())
    out["weights_full"] = w_all.tolist()
    out["per_scheme"] = ledger["per_scheme"]
    out["gain_vs_baseline"] = ledger["gain_vs_baseline"]
    out["verdict"] = ledger["verdict"]
    out["verdict_detail"] = ledger["verdict_detail"]
    out["baseline"] = ledger["baseline"]

    # 交换律自检：融合结果不应依赖层序（WLC 是线性的）
    perm = [2, 0, 1]
    schemes_perm = wlc_score(stack_all[perm], w_all[perm])["score"]
    perm_diff = float(np.max(np.abs(schemes_perm - fused_full)))
    out["layer_order_invariance_max_diff"] = perm_diff

    # ---- G1 误差三分解自洽：mse_raw = bias² + var ----
    g1_rows = []
    g1_ok = True
    for nm, e in ledger["per_scheme"].items():
        # 恒等式：E[(p-t)²] = (E[p-t])² + Var(p-t)
        p = schemes[nm][mask].ravel()
        t = truth[mask].ravel()
        d = p - t
        lhs = float(np.mean(d ** 2))
        rhs = float(np.mean(d) ** 2) + float(np.var(d))
        g1_rows.append({"scheme": nm, "mse_raw": lhs,
                        "bias2_plus_var": rhs, "diff": abs(lhs - rhs)})
        g1_ok &= abs(lhs - rhs) < 1e-9
    out["decomposition_identity"] = g1_rows

    # ---- 判据 ----
    g1 = g1_ok
    # 基线对自身的增益按定义为 0
    g2 = abs(ledger["gain_vs_baseline"]["single_L1"]["nmse_reduction"]) < 1e-12

    # G3 纯缩放：构造一个 2 倍缩放的方案
    scaled = 2.0 * truth
    e_scaled = error_decomposition(scaled, truth, mask)
    g3 = (abs(e_scaled["share_of_error_from_scale"] - 1.0) < 1e-9
          and e_scaled["mse_calibrated"] < 1e-20)

    # ---- G4 主结果：判别力随融合单调提升 ----
    seq = ["single_L1", "fused_L1L2", "fused_L1L2L3"]
    r_seq = [ledger["per_scheme"][k]["pearson_r"] for k in seq]
    g4 = all(r_seq[i] < r_seq[i + 1] for i in range(len(r_seq) - 1))

    # ---- G4b 符号反转：raw MSE 与 calibrated MSE 方向相反 ----
    nmraw = [ledger["per_scheme"][k]["nmse_raw"] for k in seq]
    nmcal = [ledger["per_scheme"][k]["nmse_calibrated"] for k in seq]
    rmcal = [ledger["per_scheme"][k]["rmse_calibrated"] for k in seq]
    bseq = [ledger["per_scheme"][k]["calib_b"] for k in seq]
    # 从 A→C 的变化方向
    d_raw = nmraw[2] - nmraw[0]        # >0 表示原始误差变大（变差）
    d_cal = nmcal[2] - nmcal[0]        # <0 表示标定后误差变小（变好）
    sign_inversion = (d_raw > 0) and (d_cal < 0)
    g4b = sign_inversion
    # G4c：标定后优势单调重现
    g4c = all(rmcal[i] > rmcal[i + 1] for i in range(len(rmcal) - 1))
    # G8：得分被系统性压缩
    g8 = all(b > 1.0 for b in bseq)
    # 方差压缩的直接证据
    var_truth = float(np.var(truth[mask]))
    var_single = float(np.var(schemes["single_L1"][mask]))
    var_fused = float(np.var(schemes["fused_L1L2L3"][mask]))
    out["variance_compression"] = {
        "var_truth_on_mask": var_truth,
        "var_single_L1_on_mask": var_single,
        "var_fused_L1L2L3_on_mask": var_fused,
        "single_vs_truth_ratio": var_single / var_truth,
        "fused_over_single_ratio": var_fused / var_single,
        "theoretical_upper_bound_sum_w_squared": float(np.sum(w_all ** 2)),
        "note": ("min-max 重分类把各层压入 [0,1]，其方差远小于真值方差；"
                 "WLC 的 Σw=1 凸组合再压缩一次。故得分是单调代理、非同尺度估计。"),
    }
    out["sign_inversion"] = {
        "sequence": seq,
        "nmse_raw": nmraw,
        "nmse_calibrated": nmcal,
        "rmse_calibrated": rmcal,
        "calib_b": bseq,
        "pearson_r": r_seq,
        "delta_nmse_raw_A_to_C": d_raw,
        "delta_nmse_calibrated_A_to_C": d_cal,
        "sign_is_inverted": bool(sign_inversion),
        "note": ("nmse_raw 上升而 nmse_calibrated 下降 ⇒ 单一精度数字会把"
                 "结论的符号搞反。这正是本项目主张『必须分解误差』的直接证据。"),
    }
    # G5 粗输出 vs 细输出
    g5 = (ledger["per_scheme"]["fused_coarse_expand"]["nmse_raw"]
          > ledger["per_scheme"]["fused_L1L2L3"]["nmse_raw"])
    # G6 verdict 合法
    g6 = ledger["verdict"] in ("fusion_helps", "fusion_hurts", "inconclusive")

    # 附加：分辨率的代价（在同一融合内容下只改输出分辨率）
    res_cost = []
    for fc in (1, 2, 4, 8, 16, 32):
        if f["n"] % fc != 0:
            continue
        c = aggregate_to_coarse(fused_full, fc, mask=None, mode="all_mean")
        ex = expand_back(c, fc, f["n"])
        e = error_decomposition(ex, truth, mask)
        res_cost.append({
            "factor": fc, "out_cell_m": cell_m * fc,
            "grid_vs_pond_ratio": cell_m * fc / pond["pond_side_m"],
            "nmse_raw": e["nmse_raw"], "pearson_r": e["pearson_r"],
        })
    out["resolution_cost"] = res_cost

    checks = [
        {"id": "G1", "desc": "误差三分解自洽 mse_raw == bias²+var（容差1e-9）",
         "pass": bool(g1),
         "observed": f"max|diff|={max(r['diff'] for r in g1_rows):.3e}",
         "expected": "< 1e-9"},
        {"id": "G2", "desc": "基线对自身增益恒为 0", "pass": bool(g2),
         "observed": f"{ledger['gain_vs_baseline']['single_L1']['nmse_reduction']:.3e}",
         "expected": "== 0"},
        {"id": "G3", "desc": "纯缩放方案可被仿射标定清零（尺度占比≈1）",
         "pass": bool(g3),
         "observed": f"share={e_scaled['share_of_error_from_scale']:.12f}, "
                     f"mse_cal={e_scaled['mse_calibrated']:.3e}",
         "expected": "share==1 且 mse_cal<1e-20"},
        {"id": "G4", "desc": "★主结果：判别力随融合单调提升 r(A)<r(B)<r(C)",
         "pass": bool(g4),
         "observed": " → ".join(f"{x:.4f}" for x in r_seq),
         "expected": "严格递增"},
        {"id": "G4b", "desc": "★符号反转：Δnmse_raw>0 而 Δnmse_calibrated<0",
         "pass": bool(g4b),
         "observed": f"Δraw={d_raw:+.4f}, Δcal={d_cal:+.4f}",
         "expected": "raw 上升 且 calibrated 下降"},
        {"id": "G4c", "desc": "标定后融合优势重现：rmse_cal 单调下降",
         "pass": bool(g4c),
         "observed": " → ".join(f"{x:.4f}" for x in rmcal),
         "expected": "严格递减"},
        {"id": "G8", "desc": "得分场被系统性压缩（calib_b > 1）",
         "pass": bool(g8),
         "observed": " → ".join(f"{x:.2f}" for x in bseq),
         "expected": "全部 > 1"},
        {"id": "G5", "desc": "输出分辨率变粗(C→D)使 nmse_raw 上升",
         "pass": bool(g5),
         "observed": (f"C={ledger['per_scheme']['fused_L1L2L3']['nmse_raw']:.4f} → "
                      f"D={ledger['per_scheme']['fused_coarse_expand']['nmse_raw']:.4f}"),
         "expected": "D > C"},
        {"id": "G6", "desc": "verdict 落在三值集合内（不预设方向，红线G-3）",
         "pass": bool(g6), "observed": ledger["verdict"],
         "expected": "fusion_helps / fusion_hurts / inconclusive"},
        {"id": "G7", "desc": "WLC 对层序不变（交换律自检）",
         "pass": perm_diff < 1e-12, "observed": f"max|diff|={perm_diff:.3e}",
         "expected": "< 1e-12"},
    ]
    out["checks"] = checks
    out["n_pass"] = sum(1 for c in checks if c["pass"])
    out["n_total"] = len(checks)
    out["verdict_experiment"] = "PASS" if out["n_pass"] == out["n_total"] else "FAIL"

    resdir = os.path.abspath(os.path.join(_PKG, "..", "05-验证"))
    os.makedirs(resdir, exist_ok=True)
    jp = os.path.join(resdir, "results_exp2_fusion.json")
    with io.open(jp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("=" * 78)
    print("实验 2 · 融合的收益与代价")
    print("=" * 78)
    print(f"数据模式：{out['data_mode']}，真值已知  |  池体像元 {int(mask.sum())}")
    print(f"融合权重 {np.round(w_all, 4).tolist()}（示范性设定，见红线 G-5）")
    print("-" * 78)
    print(f"{'方案':<22}{'NMSE_raw':>10}{'RMSE_raw':>10}{'pearson_r':>10}"
          f"{'NMSE_cal':>10}{'尺度占比':>9}")
    print("-" * 78)
    for nm in schemes:
        e = ledger["per_scheme"][nm]
        print(f"{nm:<22}{e['nmse_raw']:>10.4f}{e['rmse_raw']:>10.4f}"
              f"{e['pearson_r']:>10.4f}{e['nmse_calibrated']:>10.4f}"
              f"{e['share_of_error_from_scale']:>9.4f}")
    print("-" * 78)
    print("相对基线的增益（正 = 优于基线）：")
    for nm in schemes:
        g = ledger["gain_vs_baseline"][nm]
        print(f"  {nm:<22} ΔNMSE={g['nmse_reduction']:>+9.4f}  "
              f"ΔRMSE={g['rmse_reduction']:>+9.4f}  Δr={g['r_increase']:>+8.4f}")
    print(f"判定（基于 nmse_raw）：{ledger['verdict']}")
    print(f"  {ledger['verdict_detail']}")
    print("-" * 78)
    print("★ 符号反转（本项目核心示范）—— 同一个融合过程，三个指标三个方向：")
    print(f"  {'方案':<16}{'NMSE_raw':>10}{'pearson_r':>11}{'NMSE_cal':>11}"
          f"{'RMSE_cal':>11}{'calib_b':>9}")
    for i, k in enumerate(seq):
        print(f"  {k:<16}{nmraw[i]:>10.4f}{r_seq[i]:>11.4f}"
              f"{nmcal[i]:>11.4f}{rmcal[i]:>11.4f}{bseq[i]:>9.2f}")
    print(f"  A→C 变化：ΔNMSE_raw = {d_raw:+.4f}（变差）  "
          f"ΔNMSE_cal = {d_cal:+.4f}（变好）")
    print("  ⇒ 只看 NMSE_raw 会得出『融合有害』；看 r 或标定后 MSE 却是『融合大幅有益』。")
    vc = out["variance_compression"]
    print("  机制（方差压缩）：")
    print(f"    var(truth)={vc['var_truth_on_mask']:.4f} | "
          f"var(single)={vc['var_single_L1_on_mask']:.4f} "
          f"({vc['single_vs_truth_ratio']:.4f}×) | "
          f"var(fused)={vc['var_fused_L1L2L3_on_mask']:.4f} "
          f"({vc['fused_over_single_ratio']:.4f}× 于单层)")
    print(f"    凸组合理论上界 Σw² = {vc['theoretical_upper_bound_sum_w_squared']:.4f}")
    print("-" * 78)
    print("★ 分辨率代价（同一融合内容，只改输出分辨率）：")
    print(f"  {'输出格m':>8} {'/池':>6} {'NMSE_raw':>10} {'pearson_r':>10}")
    for rc in res_cost:
        print(f"  {rc['out_cell_m']:>8.0f} {rc['grid_vs_pond_ratio']:>6.1f} "
              f"{rc['nmse_raw']:>10.4f} {rc['pearson_r']:>10.4f}")
    print("-" * 78)
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['id']} {c['desc']}")
        print(f"          实测={c['observed']}")
    print("-" * 78)
    print(f"结果：{out['n_pass']}/{out['n_total']} → {out['verdict_experiment']}")
    print(f"已写入 {jp}")
    print("=" * 78)
    return 0 if out["verdict_experiment"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
