# -*- coding: utf-8 -*-
"""真值自检：用**独立的数值反事实**验证 synth.compute_truth 的解析值。

【为什么要这个文件】
真值算错了，后面所有验证都没有意义 —— 这是本项目最基础的防线。

【做法】
把同一个体的 T 分别置为 high 与 low，其余协变量与噪声**保持不变**，
生成两份反事实结果，逐个体相减再取平均 → 蒙特卡洛真值。
与解析公式对比。

【关键实现细节（第一版曾在此出错）】
1. 干预只改变 WQ 中与 T 有关的部分；WQ 的标准化常数必须用**同一套**
   （否则 T 变化会通过重新标准化污染与 T 无关的部分）。
   做法：先按基线算出 WQ 的均值/标准差作为参考系，两个反事实共用。
2. 两个反事实必须用**完全相同的噪声**（逐个体配对），否则差值里会混入噪声方差。
3. 数值真值应逐个体计算差值再平均，而不是先各自平均再相减
   （在配对噪声下二者等价，但逐个体相减在诊断上是更好的写法）。
4. ★ 尺度必须与估计器一致（2026-09-20 新增）：`_true_ate` 现在定义在
   **标准化后**的 Y 尺度上（因为估计器回归的 `growth` 列 = y / sd(y)），
   而本文件的数值反事实是在**原始 y 尺度**上算的。
   故这里必须同样除以 `df.attrs["y_sd"]`，两个尺度才可比。
   —— 若不除，本自检会立刻报出 ≈19% 的相对差；这正是发现
   synth.compute_truth 单位不一致的那条线索（见修正记录 R-2）。
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import synth  # noqa: E402


def _wq_raw(t, df):
    """WQ 的原始（未标准化）线性部分。t 可以是标量或数组。"""
    t = np.asarray(t, dtype=float)
    return (synth.C_WQ_ON_T * t
            + synth.C_WQ_ON_S * df["season_temp"].values
            + synth.C_WQ_ON_W_DEPTH * df["pond_depth"].values)


def counterfactual_truth(df, t_value, nonlinear, wq_mu, wq_sd, noise,
                         t_as_vector: bool = False):
    """把所有人的 T 置为 t_value，用固定的 WQ 参考系与固定噪声重算 Y。

    ⚠️ 必须与 synth.generate 使用**同一套**标准化常数 (wq_mu, wq_sd)，
    否则两条路径不在同一尺度上（本文件第一版即此错）。

    t_as_vector=True 时 t_value 已是与 df 等长的数组（用于数值求斜率）。
    """
    if t_as_vector:
        t = np.asarray(t_value, dtype=float)
    else:
        t = np.full(len(df), float(t_value))
    wq = (_wq_raw(t, df) - wq_mu) / wq_sd
    y = (synth.C_Y_ON_T * t
         + synth.C_Y_ON_WQ * wq
         + synth.C_Y_ON_S * df["season_temp"].values
         + synth.C_Y_ON_U * df["shrimp_appetite"].values
         + synth.C_Y_ON_W_AREA * df["pond_area"].values
         + noise)
    if nonlinear:
        y = y + synth.C_Y_T_SQUARED * t ** 2 + synth.C_Y_T_WQ_INTER * t * wq
    return y


def main():
    print("=" * 76)
    print("真值自检：解析值 vs 数值反事实（配对噪声 + 共用 WQ 标准化常数）")
    print("=" * 76)
    ok_all = True
    for nl in (False, True):
        df = synth.generate(n=4000, u_strength=0.0, nonlinear=nl, seed=12345)
        high = df.attrs["treat_high_mean"]
        low = df.attrs["treat_low_mean"]
        wq_mu = df.attrs["wq_mu"]
        wq_sd = df.attrs["wq_sd"]
        ate_analytic = float(df["_true_ate"].iloc[0])

        # 配对噪声：同一组噪声用于 high 与 low 两个反事实
        noise = np.random.default_rng(20260920).normal(0, synth.NOISE_Y, len(df))

        y_hi = counterfactual_truth(df, high, nl, wq_mu, wq_sd, noise)
        y_lo = counterfactual_truth(df, low, nl, wq_mu, wq_sd, noise)
        mc_raw = float(np.mean(y_hi - y_lo))
        # ★ 换算到估计器实际作用的尺度（growth = y / sd(y)）
        y_sd = float(df.attrs.get("y_sd", 1.0))
        mc = mc_raw / y_sd

        rel = abs(ate_analytic - mc) / (abs(mc) if abs(mc) > 1e-9 else 1.0)
        passed = rel < 0.02
        ok_all &= passed
        print(f"\nnonlinear={nl}")
        print(f"  解析 ATE（标准化尺度）    = {ate_analytic:+.6f}")
        print(f"  数值反事实 ATE（原始尺度） = {mc_raw:+.6f}   y_sd={y_sd:.6f}")
        print(f"  数值反事实 ATE（标准化）   = {mc:+.6f}  (逐个体配对差，n=4000)")
        print(f"  相对差                    = {rel:.6%}   "
              f"→ {'通过' if passed else '**不通过**'}")

    print("\n" + "=" * 76)
    print("自检结论：", "全部通过" if ok_all else "**存在不通过项，真值不可用**")
    print("=" * 76)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
