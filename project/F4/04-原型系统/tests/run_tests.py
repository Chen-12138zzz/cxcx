# -*- coding: utf-8 -*-
"""
F4 原型系统 · 自动化不变量测试

【为什么用自写的 runner 而不是 pytest】
本项目的测试要能在**最简环境**下复现（只需 numpy），
且每一条断言都要求能打印出**它实际观测到的值** —— 因为本项目的核心风险
不是"程序跑不起来"，而是"跑起来了但静默算错"。
因此这里不用断言宏，而是显式记录 (期望, 实测) 并在末尾汇总。

【测试的两类价值（skill 阶段 4 要求）】
  1. 防回归：改动后再跑一次，确认没有破坏既有性质。
  2. **把"我以为成立的性质"变成可执行断言** —— 这是更重要的价值。
     凡是项目文档里写下的"性质"（如"块平均保均值""rel_sd=0 时保留率为 1"），
     这里都有一条对应断言。若某条性质没被断言，它在文档里就只是**声称**。

【运行】
    python tests/run_tests.py            # 全部
    python tests/run_tests.py --list     # 只列测试名
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# 导入路径设置
# ---------------------------------------------------------------------------
# ⚠️ 这里必须把 **src 的父目录**（即 04-原型系统/）加入 sys.path，
#    然后以 `src.xxx` 形式导入 —— 不能把 `src/` 本身加入 sys.path。
#    原因：src/fusion/scale.py 里用的是相对导入 `from ..mce import ...`，
#    而 `..` 需要回退到 `src` 这个包。若把 `src/` 当 sys.path 根来 `import fusion`，
#    `fusion` 就成了顶层包，`..mce` 会越过顶层包边界报：
#      ImportError: attempted relative import beyond top-level package
#    （这条是实测踩到的坑，已记录在修正记录中。）
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)          # 04-原型系统/
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from src.synth import (make_multiscale_field, make_pond_layout,
                       block_average, target_factor_for_cell, effective_cell_m)
from src.mce import (minmax_normalize, assert_not_degenerate, reclass_fuzzy,
                     wlc_score, normalized_weights, ahp_priority_from_consistent,
                     rank_descending, spearman_rho, _rankdata, top_k_overlap)
from src.fusion import (aggregate_to_coarse, scale_mismatch_curve,
                        pond_vs_grid_geometry, error_decomposition,
                        fusion_ledger, weight_perturbation,
                        weight_perturbation_curve)
from src.audit import (reclass_choice_sensitivity, timeliness_flags,
                       audit_summary)


# ---------------------------------------------------------------------------
# 极简测试框架
# ---------------------------------------------------------------------------
_RESULTS: list[tuple[str, str, bool, str, str]] = []


def check(group: str, name: str, cond: bool, expected: str = "", got: str = "") -> None:
    """记录一条断言。cond 必须已是 bool（调用方负责比较）。"""
    _RESULTS.append((group, name, bool(cond), expected, got))
    mark = "PASS" if cond else "FAIL"
    line = f"  [{mark}] {name}"
    if expected or got:
        line += f"   期望={expected}  实测={got}"
    print(line)


def expect_raises(group: str, name: str, fn, exc: type = Exception,
                  expected: str = "") -> None:
    """断言 fn() 必须抛出 exc 类型。不抛 = FAIL（静默通过是缺陷）。"""
    try:
        fn()
    except exc as e:
        check(group, name, True, expected or f"抛 {exc.__name__}",
              f"抛 {type(e).__name__}: {str(e)[:70]}")
    except Exception as e:  # noqa: BLE001
        check(group, name, False, expected or f"抛 {exc.__name__}",
              f"抛了别的类型 {type(e).__name__}: {str(e)[:70]}")
    else:
        check(group, name, False, expected or f"抛 {exc.__name__}", "没有抛异常")


def near(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# Group A · 合成场与几何（synth.py）
# ---------------------------------------------------------------------------
def group_A() -> None:
    G = "A synth/几何"
    print("\n[A] 合成场与池体布局")

    f = make_multiscale_field(n=128, length_m=3200.0, seed=0)
    check(G, "A1 真值标准化后 std==1", near(np.std(f["truth"]), 1.0, 1e-12),
          "1.0", f"{np.std(f['truth']):.12f}")
    check(G, "A2 真值标准化后 mean==0", near(np.mean(f["truth"]), 0.0, 1e-12),
          "0.0", f"{np.mean(f['truth']):.3e}")
    check(G, "A3 sd 字段被保留且 >0", f["sd"] > 0, ">0", f"{f['sd']:.6f}")
    check(G, "A4 grid=3200/128=25m", near(f["cell_m"], 25.0, 1e-12),
          "25.0", f"{f['cell_m']}")

    # 还原关系：truth * sd + mean_raw == raw  —— 这是 R-2 教训的可执行化
    reco = f["truth"] * f["sd"] + f["mean_raw"]
    err = float(np.max(np.abs(reco - f["raw"])))
    check(G, "A5 truth*sd+mean_raw 能精确还原 raw", err < 1e-9,
          "max|err| < 1e-9", f"{err:.3e}")

    # 三层结构可分辨性（这是整个尺度诊断成立的**前提**）
    layers = f["layers"]
    v_hi = float(np.var(layers["L3_50m"]))
    v_lo = float(np.var(layers["L1_800m"]))
    chk = block_average(layers["L3_50m"], 4)
    chk2 = block_average(layers["L1_800m"], 4)
    r_hi = float(np.var(chk) / v_hi)
    r_lo = float(np.var(chk2) / v_lo)
    check(G, "A6 50m层经100m块平均后方差大幅丢失(<0.3)", r_hi < 0.3,
          "<0.3", f"{r_hi:.4f}")
    check(G, "A7 800m层经100m块平均后方差基本保留(>0.8)", r_lo > 0.8,
          ">0.8", f"{r_lo:.4f}")
    check(G, "A8 高层丢失率 > 低层丢失率（结构确实可分辨）", r_hi < r_lo,
          "r_hi < r_lo", f"{r_hi:.4f} < {r_lo:.4f}")

    # 复现性：同种子结果完全一致
    f2 = make_multiscale_field(n=128, length_m=3200.0, seed=0)
    check(G, "A9 同种子结果逐元素一致", np.array_equal(f["truth"], f2["truth"]),
          "完全相同", "相同" if np.array_equal(f["truth"], f2["truth"]) else "不同")
    f3 = make_multiscale_field(n=128, length_m=3200.0, seed=1)
    check(G, "A10 不同种子结果不同", not np.array_equal(f["truth"], f3["truth"]),
          "不同", "不同" if not np.array_equal(f["truth"], f3["truth"]) else "相同")

    expect_raises(G, "A11 scales/weights 长度不一致须报错",
                  lambda: make_multiscale_field(scales=(800.0, 200.0),
                                                weights=(1.0,)), ValueError,
                  "ValueError")

    # 块平均的均值守恒（最重要的一条）
    x = np.arange(64 * 64, dtype=float).reshape(64, 64)
    ba = block_average(x, 8)
    check(G, "A12 块平均保均值（float64 容差内）", near(ba.mean(), x.mean(), 1e-9),
          f"{x.mean()}", f"{ba.mean()}")
    check(G, "A13 块平均后形状正确 64/8=8", ba.shape == (8, 8),
          "(8, 8)", f"{ba.shape}")
    # 常数场块平均后仍是同一常数
    c = np.full((32, 32), 7.5)
    check(G, "A14 常数场块平均后不变", near(block_average(c, 4).mean(), 7.5, 1e-12),
          "7.5", f"{block_average(c, 4).mean()}")
    expect_raises(G, "A15 不整除须报错",
                  lambda: block_average(np.zeros((10, 10)), 4), ValueError,
                  "ValueError")
    expect_raises(G, "A16 非方阵须报错",
                  lambda: block_average(np.zeros((10, 20)), 2), ValueError,
                  "ValueError")

    check(G, "A17 target_factor 向下取整为 1",
          target_factor_for_cell(25.0, 20.0) == 1, "1",
          f"{target_factor_for_cell(25.0, 20.0)}")
    check(G, "A18 target_factor 100m 目标 → 4",
          target_factor_for_cell(25.0, 100.0) == 4, "4",
          f"{target_factor_for_cell(25.0, 100.0)}")
    check(G, "A19 effective_cell 25*4=100",
          near(effective_cell_m(25.0, 4), 100.0), "100.0",
          f"{effective_cell_m(25.0, 4)}")

    # 池体布局
    pond = make_pond_layout(n=128, length_m=3200.0, pond_side_m=50.0, gap_m=10.0)
    check(G, "A20 单池边长50m → 面积0.25ha",
          near(pond["pond_area_ha"], 0.25, 1e-12), "0.25",
          f"{pond['pond_area_ha']}")
    check(G, "A21 池数 >0", pond["n_ponds"] > 0, ">0", f"{pond['n_ponds']}")
    # 每口池的像元数应等于 side_cells^2 = (50/25)^2 = 4
    pid = pond["pond_id"]
    check(G, "A22 每池恰好4个像元(50m/25m=2 → 2x2)", int((pid == 0).sum()) == 4,
          "4", f"{int((pid == 0).sum())}")
    expect_raises(G, "A23 单池+间隔超过边长须报错",
                  lambda: make_pond_layout(n=128, length_m=100.0,
                                           pond_side_m=80.0, gap_m=30.0),
                  ValueError, "ValueError")


# ---------------------------------------------------------------------------
# Group B · MCE / WLC（mce.py）
# ---------------------------------------------------------------------------
def group_B() -> None:
    G = "B MCE/WLC"
    print("\n[B] MCE / WLC / 秩相关")

    x = np.array([1.0, 2.0, 3.0, 4.0])
    mn = minmax_normalize(x)
    check(G, "B1 minmax 映射到 [0,1]", near(mn.min(), 0.0) and near(mn.max(), 1.0),
          "[0,1]", f"[{mn.min()}, {mn.max()}]")
    inv = minmax_normalize(x, invert=True)
    check(G, "B2 invert 后顺序反转", near(inv[0], 1.0) and near(inv[-1], 0.0),
          "首=1 末=0", f"首={inv[0]} 末={inv[-1]}")
    # 常数层：minmax 静默产全 0（这是刻意的），但 assert_not_degenerate 必须拦住
    z = minmax_normalize(np.full(5, 3.0))
    check(G, "B3 常数层 minmax 产全0（已文档化的退化行为）", near(z.max(), 0.0),
          "全0", f"max={z.max()}")
    expect_raises(G, "B4 assert_not_degenerate 必须拒常数层",
                  lambda: assert_not_degenerate(np.full(5, 3.0), "const_layer"),
                  ValueError, "ValueError")
    # 非常数层必须放行
    try:
        assert_not_degenerate(np.array([1.0, 2.0, 3.0]), "ok_layer")
        check(G, "B5 assert_not_degenerate 放行非常数层", True, "不抛", "不抛")
    except Exception as e:  # noqa: BLE001
        check(G, "B5 assert_not_degenerate 放行非常数层", False, "不抛",
              f"抛了 {type(e).__name__}")

    # 模糊重分类
    fz = reclass_fuzzy(np.array([0.0, 5.0, 10.0]), lo=0.0, hi=10.0, edge="both")
    check(G, "B6 fuzzy both 端点内线性", near(fz[0], 0.0) and near(fz[2], 1.0),
          "0 与 1", f"{fz}")
    fz_out = reclass_fuzzy(np.array([-5.0, 15.0]), lo=0.0, hi=10.0, edge="both")
    check(G, "B7 fuzzy 越界被 clip 到 [0,1]",
          near(fz_out[0], 0.0) and near(fz_out[1], 1.0), "[0,1]", f"{fz_out}")
    expect_raises(G, "B8 fuzzy hi<=lo 须报错",
                  lambda: reclass_fuzzy(np.array([1.0]), lo=5.0, hi=5.0),
                  ValueError, "ValueError")

    # WLC 的核心契约
    stack = np.random.default_rng(0).random((3, 8, 8))
    w = np.array([0.5, 0.3, 0.2])
    r = wlc_score(stack, w)
    check(G, "B9 WLC 结果形状 (8,8)", r["score"].shape == (8, 8), "(8, 8)",
          f"{r['score'].shape}")
    check(G, "B10 WLC 是线性组合（手算校验）",
          near(r["score"][0, 0], 0.5 * stack[0, 0, 0] + 0.3 * stack[1, 0, 0]
               + 0.2 * stack[2, 0, 0], 1e-12),
          "手算值", f"{r['score'][0, 0]:.12f}")
    # Σw=1 时，若所有层同为常数 c，则得分恒为 c
    cc = np.full((2, 4, 4), 0.7)
    check(G, "B11 Σw=1 时常数层组合得分保持不变",
          near(wlc_score(cc, np.array([0.6, 0.4]))["score"].mean(), 0.7, 1e-12),
          "0.7", f"{wlc_score(cc, np.array([0.6, 0.4]))['score'].mean()}")

    expect_raises(G, "B12 权重未归一化必须报错（不静默缩放）",
                  lambda: wlc_score(stack, np.array([1.0, 1.0, 1.0])),
                  ValueError, "ValueError")
    expect_raises(G, "B13 权重数不匹配须报错",
                  lambda: wlc_score(stack, np.array([0.5, 0.5])), ValueError,
                  "ValueError")
    expect_raises(G, "B14 负权重须报错",
                  lambda: wlc_score(stack, np.array([1.2, -0.1, -0.1])),
                  ValueError, "ValueError")
    expect_raises(G, "B15 权重含 nan 须报错",
                  lambda: wlc_score(stack, np.array([0.5, 0.5, np.nan])),
                  ValueError, "ValueError")
    expect_raises(G, "B16 二维输入须报错",
                  lambda: wlc_score(np.zeros((4, 4)), np.array([1.0])),
                  ValueError, "ValueError")

    check(G, "B17 normalized_weights 和为1",
          near(normalized_weights([2, 3, 5]).sum(), 1.0), "1.0",
          f"{normalized_weights([2, 3, 5]).sum()}")
    expect_raises(G, "B18 全0权重须报错",
                  lambda: normalized_weights([0, 0, 0]), ValueError, "ValueError")

    # AHP：从已知权重反构，回算必须复原且 CR=0
    p = np.array([0.5, 0.3, 0.2])
    ahp = ahp_priority_from_consistent(p)
    check(G, "B19 AHP 回算权重复原原优先级",
          np.allclose(np.sort(ahp["weights"]), np.sort(p), atol=1e-9),
          f"{np.sort(p)}", f"{np.sort(ahp['weights'])}")
    check(G, "B20 AHP 完全一致矩阵 CR==0", near(ahp["consistency_ratio"], 0.0, 1e-12),
          "0.0", f"{ahp['consistency_ratio']:.3e}")
    # ⚠️ AHP 判断矩阵的第三条性质是**互反** A_ij = 1/A_ji，**不是对称**。
    #    （写测试时一开始按"对称"断言，实测 max|A-A.T| = 2.10 才发现写错了：
    #      0.5/0.2 = 2.5，而 0.2/0.5 = 0.4，两者互为倒数但绝不相等。
    #      互相成倒数 → A·A.T 的对角为 1。）
    A = ahp["matrix"]
    recip_err = float(np.max(np.abs(A * A.T - 1.0)))
    check(G, "B21 AHP 矩阵互反(A_ij*A_ji=1)且对角线为1",
          recip_err < 1e-12 and np.allclose(np.diag(A), 1.0, atol=1e-12),
          "互反 + 对角1", f"max|A*A.T-1|={recip_err:.2e}")
    check(G, "B21b AHP 矩阵一般**不**对称（澄清性质，防后人再写错）",
          not np.allclose(A, A.T, atol=1e-9),
          "不对称", f"max|A-A.T|={np.max(np.abs(A - A.T)):.2e}")
    expect_raises(G, "B22 AHP 非正优先级须报错",
                  lambda: ahp_priority_from_consistent([0.5, -0.1, 0.6]),
                  ValueError, "ValueError")

    # 秩与秩相关
    a = np.array([1.0, 2.0, 3.0, 4.0])
    check(G, "B23 spearman 完全正相关 == 1", near(spearman_rho(a, a), 1.0, 1e-12),
          "1.0", f"{spearman_rho(a, a)}")
    check(G, "B24 spearman 完全反相关 == -1",
          near(spearman_rho(a, a[::-1]), -1.0, 1e-12), "-1.0",
          f"{spearman_rho(a, a[::-1])}")
    check(G, "B25 spearman 单调非线性变换后仍为1",
          near(spearman_rho(a, np.exp(a)), 1.0, 1e-12), "1.0",
          f"{spearman_rho(a, np.exp(a))}")
    check(G, "B26 spearman 对并列取平均秩（不虚高）",
          spearman_rho(np.array([1.0, 2.0, 2.0, 3.0]),
                       np.array([1.0, 3.0, 3.0, 2.0])) < 1.0,
          "<1.0",
          f"{spearman_rho(np.array([1.0, 2.0, 2.0, 3.0]), np.array([1.0, 3.0, 3.0, 2.0])):.6f}")
    check(G, "B27 _rankdata 平均秩正确",
          np.allclose(_rankdata(np.array([10.0, 20.0, 20.0, 30.0])),
                      np.array([1.0, 2.5, 2.5, 4.0])),
          "[1, 2.5, 2.5, 4]", f"{_rankdata(np.array([10.0, 20.0, 20.0, 30.0]))}")
    expect_raises(G, "B28 spearman 样本不足须报错",
                  lambda: spearman_rho(np.array([1.0]), np.array([2.0])),
                  ValueError, "ValueError")
    expect_raises(G, "B29 spearman 长度不一致须报错",
                  lambda: spearman_rho(np.array([1.0, 2.0]), np.array([1.0])),
                  ValueError, "ValueError")

    # 排名与前 k 重叠
    sc = np.array([5.0, 9.0, 1.0, 7.0])
    rk = rank_descending(sc)
    check(G, "B30 rank_descending 按降序返回索引",
          rk.tolist() == [1, 3, 0, 2], "[1,3,0,2]", f"{rk.tolist()}")
    m = np.array([True, True, False, True])
    rkm = rank_descending(sc, m)
    check(G, "B31 rank_descending 尊重 mask",
          rkm.tolist() == [1, 3, 0], "[1,3,0]", f"{rkm.tolist()}")
    check(G, "B32 top_k_overlap 自身 == 1",
          near(top_k_overlap(sc, sc, k=2), 1.0), "1.0",
          f"{top_k_overlap(sc, sc, k=2)}")
    check(G, "B33 top_k_overlap 完全反序时前k重叠为0",
          near(top_k_overlap(np.array([4.0, 3.0, 2.0, 1.0]),
                             np.array([1.0, 2.0, 3.0, 4.0]), k=2), 0.0),
          "0.0",
          f"{top_k_overlap(np.array([4.0, 3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0, 4.0]), k=2)}")
    expect_raises(G, "B34 mask 全 False 须报错",
                  lambda: top_k_overlap(sc, sc, np.zeros(4, dtype=bool)),
                  ValueError, "ValueError")


# ---------------------------------------------------------------------------
# Group C · 尺度失配（fusion/scale.py）
# ---------------------------------------------------------------------------
def group_C() -> None:
    G = "C 尺度失配"
    print("\n[C] 尺度失配诊断")

    f = make_multiscale_field(n=128, length_m=3200.0, seed=0)
    truth = f["truth"]
    mask = np.ones_like(truth, dtype=bool)   # 全像元（几何诊断不依赖池体）

    # aggregate_to_coarse 的两种口径
    ac = aggregate_to_coarse(truth, 4, mask=mask, mode="pond_mean")
    aa = aggregate_to_coarse(truth, 4, mask=None, mode="all_mean")
    check(G, "C1 pond_mean 全掩码时等于 all_mean",
          np.allclose(ac, aa, atol=1e-12), "相等",
          f"max|diff|={np.max(np.abs(ac - aa)):.2e}")

    # ★ 关键：mask 只覆盖左下角 1/4 时，右上粗格必须返回 NaN，不能填 0
    m2 = np.zeros_like(truth, dtype=bool)
    m2[:64, :64] = True
    pm = aggregate_to_coarse(truth, 64, mask=m2, mode="pond_mean")  # 128/64 = 2
    n_nan = int(np.isnan(pm).sum())
    check(G, "C2 无有效像元的粗格返回 NaN（不填0）", n_nan == 3,
          "3 个 NaN", f"{n_nan} 个 NaN")
    check(G, "C3 有有效像元的粗格为有限值", np.isfinite(pm[0, 0]),
          "有限", f"{pm[0, 0]:.6f}")
    check(G, "C4 NaN 格不是 0（填0会把'无数据'当'最不适宜'）",
          not near(pm[1, 1], 0.0) if np.isfinite(pm[1, 1]) else True,
          "非0或NaN", f"{pm[1, 1]}")

    expect_raises(G, "C5 pond_mean 缺 mask 须报错",
                  lambda: aggregate_to_coarse(truth, 4, mask=None, mode="pond_mean"),
                  ValueError, "ValueError")
    expect_raises(G, "C6 未知 mode 须报错",
                  lambda: aggregate_to_coarse(truth, 4, mode="nope"), ValueError,
                  "ValueError")
    expect_raises(G, "C7 不整除须报错",
                  lambda: aggregate_to_coarse(np.zeros((10, 10)), 4), ValueError,
                  "ValueError")

    # 尺度失配曲线
    pond = make_pond_layout(n=128, length_m=3200.0)
    pmask = pond["pond_id"] >= 0
    curve = scale_mismatch_curve(truth, pmask, f["cell_m"],
                                 factors=(1, 2, 4, 8, 16), mode="pond_mean")
    rows = [r for r in curve if r.get("factor") is not None]
    check(G, "C8 曲线含 5 个 factor", len(rows) == 5, "5", f"{len(rows)}")
    check(G, "C9 factor=1 时信息量保留率应为 1",
          near(rows[0]["variance_retention"], 1.0, 1e-9), "1.0",
          f"{rows[0]['variance_retention']:.12f}")

    # ★ 关键：信息量必须随粗化单调下降（这是"尺度失配"命题的直接检验）
    vr = [r["variance_retention"] for r in rows]
    mono = all(vr[i] >= vr[i + 1] - 1e-12 for i in range(len(vr) - 1))
    check(G, "C10 信息量保留率随粗化单调不增", mono, "单调不增",
          " → ".join(f"{v:.4f}" for v in vr))

    # ★ 关键：本项目的核心命题 —— 方差保留高 ≠ 决策一致
    # 在本合成场上，应能观察到"保留率仍不低但前 k 重叠已明显下降"
    topk = [r["topk_overlap_pooled"] for r in rows]
    check(G, "C11 前k重叠率随粗化单调不增（不恒为1）",
          all(topk[i] >= topk[i + 1] - 1e-12 for i in range(len(topk) - 1))
          and not all(near(t, 1.0) for t in topk),
          "单调不增且至少一个 <1",
          " → ".join(f"{t:.4f}" for t in topk))

    # ★ 回归测试：修复前的实现里 topk_overlap 恒为 1.0（自比缺陷）
    check(G, "C12 [回归] 非最细尺度 topk 不恒为 1（修自比缺陷）",
          not all(near(t, 1.0) for t in topk[1:]),
          "至少一个 <1.0",
          " → ".join(f"{t:.4f}" for t in topk))

    # ★ 回归测试：rank_agreement_cells 在 f=1 时应为 1（两个口径此时等价）
    check(G, "C13 [回归] f=1 时粗格排序一致性为1",
          near(rows[0]["rank_agreement_cells"], 1.0, 1e-9), "1.0",
          f"{rows[0]['rank_agreement_cells']:.12f}")

    # 不整除的 factor 必须被记录，不能静默丢弃
    curve2 = scale_mismatch_curve(truth, pmask, f["cell_m"],
                                  factors=(1, 3, 6), mode="pond_mean")
    tail = [r for r in curve2 if r.get("factor") is None]
    check(G, "C14 不整除的 factor 被显式记录为 skipped",
          len(tail) == 1 and set(tail[0]["skipped_factors"]) == {3, 6},
          "skipped=[3,6]",
          f"{tail[0]['skipped_factors'] if tail else '未记录'}")

    # 退化输入必须被拒
    expect_raises(G, "C15 真值方差为0须报错",
                  lambda: scale_mismatch_curve(np.zeros((32, 32)),
                                               np.ones((32, 32), dtype=bool), 25.0),
                  ValueError, "ValueError")
    expect_raises(G, "C16 mask 与 truth 形状不一致须报错",
                  lambda: scale_mismatch_curve(truth, np.ones((8, 8), dtype=bool),
                                               f["cell_m"]), ValueError, "ValueError")
    # ⚠️ 这里一开始写成了 np.eye(128)，那有 128 个 True（不是 1 个），
    #    所以"必须报错"当然不成立 —— 是测试写错，不是实现漏检。
    #    正确构造"只有 1 个池体像元"的方式是显式布尔索引赋值。
    m_one = np.zeros((128, 128), dtype=bool)
    m_one[0, 0] = True
    check(G, "C17a 单像元 mask 计数确为 1（先自检构造正确）",
          int(m_one.sum()) == 1, "1", f"{int(m_one.sum())}")
    expect_raises(G, "C17 池体像元<2 须报错",
                  lambda: scale_mismatch_curve(truth, m_one, f["cell_m"]),
                  ValueError, "ValueError")

    # 纯几何诊断（不依赖任何数据，因此最稳）
    geo = pond_vs_grid_geometry(128, 3200.0, 50.0, 10.0, (1, 2, 4, 8, 40))
    g25 = geo[0]
    check(G, "C18 25m 网格内单池占2格边(=50/25)", near(g25["cells_per_pond"], 2.0),
          "2.0", f"{g25['cells_per_pond']}")
    check(G, "C19 25m 网格不比池粗", g25["grid_coarser_than_pond"] is False,
          "False", f"{g25['grid_coarser_than_pond']}")
    g1000 = [r for r in geo if r["factor"] == 40][0]
    check(G, "C20 1000m 网格比池粗", g1000["grid_coarser_than_pond"] is True,
          "True", f"{g1000['grid_coarser_than_pond']}")
    check(G, "C21 1000m 网格每个约覆盖2.78口池",
          near(g1000["ponds_per_cell"], (1000.0 / 60.0) ** 2, 1e-9),
          f"{(1000.0/60.0)**2:.6f}", f"{g1000['ponds_per_cell']:.6f}")
    check(G, "C22 每个几何输出的比值为 grid/pond",
          near(g1000["ratio_grid_to_pond"], 20.0), "20.0",
          f"{g1000['ratio_grid_to_pond']}")


# ---------------------------------------------------------------------------
# Group D · 融合账本与误差分解（fusion/gain.py）
# ---------------------------------------------------------------------------
def group_D() -> None:
    G = "D 融合账本"
    print("\n[D] 融合账本 / 误差分解 / 权重扰动")

    rng = np.random.default_rng(7)
    n = 64
    truth = rng.normal(0, 1, (n, n))
    mask = np.ones((n, n), dtype=bool)

    # 完美预测：所有误差项应为 0，标定系数应为 a=0, b=1
    ed = error_decomposition(truth, truth, mask)
    check(G, "D1 完美预测 MSE_raw==0", near(ed["mse_raw"], 0.0, 1e-24), "0.0",
          f"{ed['mse_raw']:.3e}")
    check(G, "D2 完美预测 pearson_r==1", near(ed["pearson_r"], 1.0, 1e-12), "1.0",
          f"{ed['pearson_r']:.12f}")
    check(G, "D3 完美预测标定 b==1", near(ed["calib_b"], 1.0, 1e-9), "1.0",
          f"{ed['calib_b']:.12f}")
    check(G, "D4 完美预测标定 a==0", near(ed["calib_a"], 0.0, 1e-9), "0.0",
          f"{ed['calib_a']:.3e}")

    # ★ 纯尺度误差：pred = 2*truth → 标定应完全消除（mse_calibrated ≈ 0）
    pred_scale = 2.0 * truth
    ed2 = error_decomposition(pred_scale, truth, mask)
    check(G, "D5 纯缩放误差 MSE_raw>0", ed2["mse_raw"] > 0.1, ">0.1",
          f"{ed2['mse_raw']:.4f}")
    check(G, "D6 纯缩放误差可被标定完全消除",
          ed2["mse_calibrated"] < 1e-20, "<1e-20",
          f"{ed2['mse_calibrated']:.3e}")
    check(G, "D7 纯缩放误差的尺度占比≈1",
          near(ed2["share_of_error_from_scale"], 1.0, 1e-9), "1.0",
          f"{ed2['share_of_error_from_scale']:.12f}")
    check(G, "D8 纯缩放误差 pearson_r 仍为1（排序没坏）",
          near(ed2["pearson_r"], 1.0, 1e-9), "1.0", f"{ed2['pearson_r']:.12f}")

    # ★ 纯结构误差：pred = 随机 → 标定救不了
    pred_noise = rng.normal(0, 1, (n, n))
    ed3 = error_decomposition(pred_noise, truth, mask)
    check(G, "D9 纯结构误差（随机）标定后仍然很大",
          ed3["mse_calibrated"] > 0.5, ">0.5",
          f"{ed3['mse_calibrated']:.4f}")
    # ⚠️ 这里一开始期望 "<0.2"，实测 0.5007。想清楚后判定是**测试写错**：
    #    share = 1 - MSE_cal/MSE_raw。当 pred 与 truth 相互独立时，
    #    最优线性拟合的 b→0、a→mean(truth)，残差就是 truth 自身波动，
    #    故 MSE_cal → Var(truth)，而 MSE_raw → Var(truth)+Var(pred) = 2·Var(truth)。
    #    ⇒ 渐近上 share → 1 - 1/2 = **恰好 0.5**。
    #    单次实现（seed=7）得到 0.5007，偏差 0.0007 —— 这是有限样本波动，不是缺陷。
    #    实测 60 个种子：mean=0.5001、sd=0.0104、极差 [0.4732, 0.5215]。
    #    因此正确做法是**对多种子做统计断言**，而不是把单次容忍放宽。
    shares = []
    for _sd in range(24):
        _r = np.random.default_rng(1000 + _sd)
        _t = _r.normal(0, 1, (n, n))
        _p = _r.normal(0, 1, (n, n))
        shares.append(error_decomposition(_p, _t, mask)["share_of_error_from_scale"])
    sh = np.array(shares, dtype=float)
    check(G, "D10 纯结构误差的尺度占比收敛到 0.5（多子均值）",
          abs(float(sh.mean()) - 0.5) < 0.01, "|mean-0.5| < 0.01",
          f"mean={sh.mean():.4f}  (24 子)")
    check(G, "D10b 24 个子全部落在 0.5±0.05（无离群）",
          bool(np.all(np.abs(sh - 0.5) < 0.05)), "全部 |share-0.5|<0.05",
          f"max|dev|={np.max(np.abs(sh - 0.5)):.4f}")
    check(G, "D10c 单次实现 0.5007 位于 0.5±0.05 内（澄清原测试的容忍过严）",
          abs(ed3["share_of_error_from_scale"] - 0.5) < 0.05,
          "|0.5007-0.5| < 0.05",
          f"{ed3['share_of_error_from_scale']:.4f}")

    expect_raises(G, "D11 pred/truth 尺寸不一致须报错",
                  lambda: error_decomposition(np.zeros((4, 4)), np.zeros((5, 5))),
                  ValueError, "ValueError")
    expect_raises(G, "D12 有效样本<3 须报错",
                  lambda: error_decomposition(np.zeros(2), np.zeros(2)), ValueError,
                  "ValueError")
    expect_raises(G, "D13 真值为常数须报错",
                  lambda: error_decomposition(np.arange(10.0),
                                              np.full(10, 3.0)), ValueError,
                  "ValueError")
    expect_raises(G, "D14 含 NaN 须报错（不静默跳过）",
                  lambda: error_decomposition(
                      np.array([1.0, np.nan, 3.0, 4.0]), np.arange(4.0)),
                  ValueError, "ValueError")

    # 融合账本：融合方案优于基线时应判 fusion_helps
    base = truth + rng.normal(0, 1.0, (n, n))
    fused = truth + rng.normal(0, 0.3, (n, n))
    led = fusion_ledger({"single_a": base, "fused_ab": fused}, truth, mask,
                        baseline="single_a")
    check(G, "D15 账本记录指定基线", led["baseline"] == "single_a", "single_a",
          led["baseline"])
    check(G, "D16 融合优于基线 → fusion_helps",
          led["verdict"] == "fusion_helps", "fusion_helps", led["verdict"])
    check(G, "D17 增益为正", led["gain_vs_baseline"]["fused_ab"]["nmse_reduction"] > 0,
          ">0", f"{led['gain_vs_baseline']['fused_ab']['nmse_reduction']:.4f}")
    check(G, "D18 基线对自身的增益为 0",
          near(led["gain_vs_baseline"]["single_a"]["nmse_reduction"], 0.0, 1e-12),
          "0.0", f"{led['gain_vs_baseline']['single_a']['nmse_reduction']:.3e}")

    # ★ 红线 G-3 的可执行化：融合**更差**时也必须如实报出（不预设方向）
    fused_bad = truth + rng.normal(0, 3.0, (n, n))
    led_bad = fusion_ledger({"single_a": base, "fused_bad": fused_bad}, truth, mask,
                            baseline="single_a")
    check(G, "D19 [G-3] 融合更差时如实判 fusion_hurts",
          led_bad["verdict"] == "fusion_hurts", "fusion_hurts", led_bad["verdict"])
    check(G, "D20 [G-3] 此时增益为负",
          led_bad["gain_vs_baseline"]["fused_bad"]["nmse_reduction"] < 0, "<0",
          f"{led_bad['gain_vs_baseline']['fused_bad']['nmse_reduction']:.4f}")

    # 无融合方案时不预设结论
    led_none = fusion_ledger({"single_a": base}, truth, mask)
    check(G, "D21 无融合方案 → inconclusive（不预设）",
          led_none["verdict"] == "inconclusive", "inconclusive", led_none["verdict"])

    expect_raises(G, "D22 空方案集须报错",
                  lambda: fusion_ledger({}, truth, mask), ValueError, "ValueError")
    expect_raises(G, "D23 基线名不存在须报错",
                  lambda: fusion_ledger({"single_a": base}, truth, mask,
                                        baseline="nope"), ValueError, "ValueError")
    expect_raises(G, "D24 方案形状不匹配须报错",
                  lambda: fusion_ledger({"single_a": np.zeros((4, 4))}, truth, mask),
                  ValueError, "ValueError")

    # ---- 权重扰动 ----
    stack = rng.random((3, 32, 32))
    w0 = np.array([0.5, 0.3, 0.2])
    m32 = np.ones((32, 32), dtype=bool)

    # ★★ 最关键的一条：rel_sd=0 时保留率必须**恰为** 1.0
    #    恒不通过 / 恒通过 都是可疑信号（F1 R-13 教训的可执行化）
    r0 = weight_perturbation(stack, w0, m32, n_draws=20, rel_sd=0.0, seed=0)
    check(G, "D25 [恒等自检] rel_sd=0 时 spearman 恒为1",
          near(r0["spearman_vs_base"]["min"], 1.0, 1e-12)
          and near(r0["spearman_vs_base"]["max"], 1.0, 1e-12),
          "min=max=1.0",
          f"min={r0['spearman_vs_base']['min']:.12f} max={r0['spearman_vs_base']['max']:.12f}")
    check(G, "D26 [恒等自检] rel_sd=0 时前k保留=1.0",
          near(r0["topk_kept_share"]["mean"], 1.0, 1e-12), "1.0",
          f"{r0['topk_kept_share']['mean']:.12f}")
    check(G, "D27 [恒等自检] rel_sd=0 时**换掉**的个数为 0",
          near(r0["topk_changed_count"]["mean"], 0.0, 1e-9), "0.0",
          f"{r0['topk_changed_count']['mean']:.3e}")
    check(G, "D27b [恒等式] kept_count + changed_count == k（rel_sd=0）",
          near(r0["topk_kept_count"]["mean"] + r0["topk_changed_count"]["mean"],
               float(r0["k"]), 1e-9),
          f"{r0['k']}",
          f"{r0['topk_kept_count']['mean']:.1f} + "
          f"{r0['topk_changed_count']['mean']:.1f}")

    # 扰动加大 → 保留率不应升高（方向性自检）
    r_small = weight_perturbation(stack, w0, m32, n_draws=200, rel_sd=0.05, seed=1)
    r_large = weight_perturbation(stack, w0, m32, n_draws=200, rel_sd=0.60, seed=1)
    check(G, "D28 扰动越大秩相关越低（方向性）",
          r_large["spearman_vs_base"]["mean"] < r_small["spearman_vs_base"]["mean"],
          "large < small",
          f"{r_large['spearman_vs_base']['mean']:.4f} < "
          f"{r_small['spearman_vs_base']['mean']:.4f}")
    check(G, "D29 扰动越大前k保留越低（方向性）",
          r_large["topk_kept_share"]["mean"] < r_small["topk_kept_share"]["mean"],
          "large < small",
          f"{r_large['topk_kept_share']['mean']:.4f} < "
          f"{r_small['topk_kept_share']['mean']:.4f}")
    check(G, "D30 保留率恒在 [0,1]", 0.0 <= r_large["topk_kept_share"]["min"]
          and r_large["topk_kept_share"]["max"] <= 1.0, "[0,1]",
          f"[{r_large['topk_kept_share']['min']:.4f}, "
          f"{r_large['topk_kept_share']['max']:.4f}]")

    expect_raises(G, "D31 权数不匹配须报错",
                  lambda: weight_perturbation(stack, np.array([0.5, 0.5]), m32),
                  ValueError, "ValueError")
    expect_raises(G, "D32 基准权重未归一化须报错",
                  lambda: weight_perturbation(stack, np.array([1.0, 1.0, 1.0]), m32),
                  ValueError, "ValueError")
    expect_raises(G, "D33 层堆叠非3维须报错",
                  lambda: weight_perturbation(np.zeros((4, 4)), np.array([1.0]), m32),
                  ValueError, "ValueError")

    # 曲线
    curve = weight_perturbation_curve(stack, w0, m32,
                                      rel_sds=(0.0, 0.1, 0.3), n_draws=50, seed=2)
    check(G, "D34 曲线含 3 个 rel_sd", len(curve) == 3, "3", f"{len(curve)}")
    check(G, "D35 [恒等自检] 曲线第一条(rel_sd=0)保留率为1",
          near(curve[0]["topk_kept_share_mean"], 1.0, 1e-12), "1.0",
          f"{curve[0]['topk_kept_share_mean']:.12f}")
    check(G, "D36 曲线保留率随 rel_sd 单调不增",
          all(curve[i]["topk_kept_share_mean"] >= curve[i + 1]["topk_kept_share_mean"]
              - 1e-12 for i in range(len(curve) - 1)),
          "单调不增",
          " → ".join(f"{c['topk_kept_share_mean']:.4f}" for c in curve))


# ---------------------------------------------------------------------------
# Group E · 审计模块（audit/audit.py）
# ---------------------------------------------------------------------------
def group_E() -> None:
    G = "E 审计模块"
    print("\n[E] 审计：把缺失变成可计算后果")

    rng = np.random.default_rng(11)
    stack = rng.random((3, 32, 32))
    w = np.array([0.5, 0.3, 0.2])
    mask = np.ones((32, 32), dtype=bool)
    lo, hi = 0.2, 0.8

    rcs = reclass_choice_sensitivity(stack, w, mask, lo, hi,
                                     schemes=("minmax", "fuzzy"), k_frac=0.10)
    check(G, "E1 重分类审计输出 1 个方案对", len(rcs["pairs"]) == 1, "1",
          f"{len(rcs['pairs'])}")
    p = rcs["pairs"][0]
    check(G, "E2 两方案间的 spearman 在 [-1,1]",
          -1.0 - 1e-12 <= p["spearman"] <= 1.0 + 1e-12, "[-1,1]",
          f"{p['spearman']:.6f}")
    check(G, "E3 前k重叠率在 [0,1]", 0.0 <= p["topk_overlap"] <= 1.0, "[0,1]",
          f"{p['topk_overlap']:.4f}")
    check(G, "E4 换掉的个数 = k - 重叠*k",
          near(p["topk_changed_mean"], p["k"] - p["topk_overlap"] * p["k"], 1e-9),
          "一致", f"{p['topk_changed_mean']:.4f}")
    check(G, "E5 lo/hi 被如实记录（可追溯）", near(rcs["lo"], lo) and near(rcs["hi"], hi),
          f"{lo}/{hi}", f"{rcs['lo']}/{rcs['hi']}")
    expect_raises(G, "E6 未知重分类方案须报错",
                  lambda: reclass_choice_sensitivity(stack, w, mask, lo, hi,
                                                     schemes=("nope",)),
                  ValueError, "ValueError")

    # ★ 审计项 2：刻意不产数
    tf = timeliness_flags()
    check(G, "E7 时效性检查清单非空", len(tf) > 0, ">0", f"{len(tf)} 项")
    no_digits = all(not any(ch.isdigit() for ch in item["item"]) for item in tf)
    check(G, "E8 [纪律8] 清单项不含数字（无数据不产数）", no_digits,
          "不含数字", "不含数字" if no_digits else "含数字")
    check(G, "E9 每项都给出 why 与来源",
          all(item.get("why") and item.get("source_of_concern") for item in tf),
          "齐备", "齐备")

    # 审计摘要
    summ = audit_summary(stack, w, mask, lo, hi, scale_curve=None,
                         rel_sds=(0.0, 0.1, 0.3), n_draws=50, seed=3, k_frac=0.10)
    check(G, "E10 摘要含权重敏感度曲线",
          len(summ["weight_sensitivity"]) == 3, "3",
          f"{len(summ['weight_sensitivity'])}")
    check(G, "E11 摘要含重分类敏感度",
          "pairs" in summ["reclass_sensitivity"], "有 pairs",
          "有" if "pairs" in summ["reclass_sensitivity"] else "无")
    check(G, "E12 [恒等自检] degeneracy_check 通过",
          summ["degeneracy_check"]["ok"] is True, "True",
          f"{summ['degeneracy_check']['ok']} | {summ['degeneracy_check']['note']}")

    # ★ 退化自检必须能失败：若去掉 rel_sd=0 基线，degeneracy_check 应报 False
    summ_no0 = audit_summary(stack, w, mask, lo, hi,
                             rel_sds=(0.1, 0.3), n_draws=50, seed=3)
    check(G, "E13 [恒不通过自检] 缺 rel_sd=0 基线时 degeneracy_check 报 False",
          summ_no0["degeneracy_check"]["ok"] is False, "False",
          f"{summ_no0['degeneracy_check']['ok']}")

    expect_raises(G, "E14 摘要权重未归一化须报错",
                  lambda: audit_summary(stack, np.array([1.0, 1.0, 1.0]), mask,
                                        lo, hi), ValueError, "ValueError")


# ---------------------------------------------------------------------------
# Group F · 端到端一致性（跨模块口径是否对得上）
# ---------------------------------------------------------------------------
def group_F() -> None:
    G = "F 端到端"
    print("\n[F] 端到端：跨模块口径一致性")

    f = make_multiscale_field(n=128, length_m=3200.0, seed=0)
    pond = make_pond_layout(n=128, length_m=3200.0)
    truth = f["truth"]
    pmask = pond["pond_id"] >= 0

    # 几何诊断中"100m 网格比池粗"必须与失配曲线一致
    geo = pond_vs_grid_geometry(128, 3200.0, 50.0, 10.0, (1, 2, 4, 8))
    curve = scale_mismatch_curve(truth, pmask, f["cell_m"],
                                 factors=(1, 2, 4, 8), mode="pond_mean")
    rows = [r for r in curve if r.get("factor") is not None]
    consistent = True
    detail = []
    for g, r in zip(geo, rows):
        same = (near(g["grid_m"], r["coarse_cell_m"], 1e-9)
                and near(g["ratio_grid_to_pond"], r["grid_vs_pond_ratio"], 1e-9))
        consistent &= same
        detail.append(f"f={g['factor']}:{g['grid_m']:.0f}m")
    check(G, "F1 几何诊断与失配曲线的网格尺度一致", consistent, "完全一致",
          " ".join(detail))

    # 端到端主链路：合成场 → 重分类 → WLC → 账本
    _wlc = wlc_score
    l1 = minmax_normalize(f["layers"]["L1_800m"])
    l2 = minmax_normalize(f["layers"]["L2_200m"])
    l3 = minmax_normalize(f["layers"]["L3_50m"])
    single = _wlc([l1], [1.0])["score"]
    fused = _wlc([l1, l2, l3], normalized_weights([1.0, 0.6, 0.35]))["score"]
    led = fusion_ledger({"single_l1": single, "fused_all": fused}, truth, pmask,
                        baseline="single_l1")
    check(G, "F2 端到端账本能给出判定（三选一）",
          led["verdict"] in ("fusion_helps", "fusion_hurts", "inconclusive"),
          "三选一之一", led["verdict"])
    check(G, "F3 端到端账本记录了两个方案",
          set(led["per_scheme"].keys()) == {"single_l1", "fused_all"},
          "两个方案", f"{sorted(led['per_scheme'].keys())}")
    check(G, "F4 每个方案的 n 等于池体像元数",
          all(v["n"] == int(pmask.sum()) for v in led["per_scheme"].values()),
          f"{int(pmask.sum())}",
          f"{[v['n'] for v in led['per_scheme'].values()]}")

    # 全链路可复现：同种子两次运行结果逐位相同
    f_again = make_multiscale_field(n=128, length_m=3200.0, seed=0)
    led_again = fusion_ledger(
        {"single_l1": _wlc([minmax_normalize(f_again["layers"]["L1_800m"])],
                           [1.0])["score"],
         "fused_all": _wlc([minmax_normalize(f_again["layers"]["L1_800m"]),
                            minmax_normalize(f_again["layers"]["L2_200m"]),
                            minmax_normalize(f_again["layers"]["L3_50m"])],
                           normalized_weights([1.0, 0.6, 0.35]))["score"]},
        f_again["truth"], pmask, baseline="single_l1")
    same = near(led["per_scheme"]["fused_all"]["mse_raw"],
                led_again["per_scheme"]["fused_all"]["mse_raw"], 1e-15)
    check(G, "F5 全链路同种子结果可复现（逐位）", same, "完全相同",
          f"{led['per_scheme']['fused_all']['mse_raw']:.15e} vs "
          f"{led_again['per_scheme']['fused_all']['mse_raw']:.15e}")

    # 合成数据不得外推：检查模块里没有真实地理/塘口标识
    # （此处原写 _SRC，后来把该变量改名为 _PKG_ROOT 时漏改，属测试自身缺陷）
    src_dir = os.path.join(_PKG_ROOT, "src")
    forbidden = ["真实塘口", "实测数据", "湛江", "雷州"]
    hits = []
    for dp, _, fns in os.walk(src_dir):
        for fn in fns:
            if not fn.endswith(".py"):
                continue
            txt = open(os.path.join(dp, fn), encoding="utf-8").read()
            for kw in forbidden:
                # 允许出现在"不得/禁止"等否定语境中
                for i, line in enumerate(txt.splitlines(), 1):
                    if kw in line and not any(
                            neg in line for neg in ("不得", "禁止", "非实测", "不是")):
                        hits.append(f"{fn}:{i} {kw}")
    check(G, "F6 [红线G-2] 源码中无未加否定的真实地点/实测断言", len(hits) == 0,
          "0 处", f"{len(hits)} 处 {hits[:3]}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
GROUPS = {
    "A": group_A, "B": group_B, "C": group_C,
    "D": group_D, "E": group_E, "F": group_F,
}


def main() -> int:
    args = sys.argv[1:]
    if "--list" in args:
        print("可用测试组：A B C D E F")
        return 0
    only = [a for a in args if a in GROUPS]
    print("=" * 78)
    print("F4 原型系统 · 不变量测试")
    print("=" * 78)
    for key, fn in GROUPS.items():
        if only and key not in only:
            continue
        try:
            fn()
        except Exception:  # noqa: BLE001
            print(f"\n!! 测试组 {key} 抛异常（视为整组失败）")
            traceback.print_exc()
            _RESULTS.append((f"{key}", f"组 {key} 运行", False, "无异常",
                             "抛异常"))

    n = len(_RESULTS)
    nf = sum(1 for r in _RESULTS if not r[2])
    print("\n" + "=" * 78)
    print(f"总计 {n} 项，通过 {n - nf} 项，失败 {nf} 项")
    if nf:
        print("\n失败明细：")
        for grp, name, ok, exp, got in _RESULTS:
            if not ok:
                print(f"  [{grp}] {name}\n      期望={exp}\n      实测={got}")
    print("=" * 78)
    return 1 if nf else 0


if __name__ == "__main__":
    raise SystemExit(main())
