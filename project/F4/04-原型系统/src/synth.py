# -*- coding: utf-8 -*-
"""
F4 原型系统 · 合成空间数据生成器

【设计定位】
本项目**无高位池现场准入、无实测真值**（约束 U-1），因此本模块提供
**具有已知空间结构的合成适宜性场**，作为全部精度/误差数字的真值来源。
所有由此产出的数字在报告中一律标 **【合成】**。

【为什么合成场必须"结构已知"】
若合成场只是随机噪声，则"融合是否提升精度"毫无意义 —— 任何方法都能在噪声上
表现"更好"或"更差"。因此本模块提供**三层可分辨的结构**：
  1. 大尺度趋势（低频、平滑）—— 模拟区域性的水温/盐度梯度
  2. 中尺度斑块（中频）—— 模拟局部地形与底质差异
  3. 小尺度纹理（高频）—— 模拟池体级差异（这是高位池决策真正在乎的尺度）
融合诊断的核心命题即是：**当采样分辨率粗于第 3 层时，第 3 层不可恢复**。

【纪律】
- 本模块产出的任何数值**不得**被表述为真实地理适宜性（红线 G-2）。
- 合成场的参数由调用方显式给出，**没有"默认的地理值"**。
"""
from __future__ import annotations

import numpy as np

# 公开接口契约：与 fusion/audit 两包保持对等，
# 使"哪些符号是公开 API"这件事可被机器读取，而不是靠约定。
__all__ = [
    "make_multiscale_field",
    "make_pond_layout",
    "block_average",
    "target_factor_for_cell",
    "effective_cell_m",
]


# ----------------------------------------------------------------------------
# 多尺度合成场
# ----------------------------------------------------------------------------
def make_multiscale_field(n: int = 128,
                          length_m: float = 3200.0,
                          scales: tuple[float, ...] = (800.0, 200.0, 50.0),
                          weights: tuple[float, ...] = (1.0, 0.6, 0.35),
                          seed: int = 0,
                          standardize: bool = True) -> dict:
    """
    生成一个具有三个可分辨空间尺度的合成适宜性场。

    参数
    ----
    n : 网格边长（格数）。输出为 n x n 的高分辨率"真值场"。
    length_m : 场域物理边长（米）。网格间距 = length_m / n。
    scales : 各结构层的**特征长度**（米），长度须与 weights 一致。
    weights : 各层的振幅权重。
    seed : 随机种子（真值随种子变化，但统计性质不变）。
    standardize : 是否把结果标准化到均值 0、标准差 1。

    返回
    ----
    dict:
      truth       : (n, n) 真值场（已标准化，若 standardize=True）
      raw         : (n, n) 标准化前的原始场
      sd          : 标准化前的标准差（**必须保留**，否则真值与估计量会尺度不一致）
      scales      : 各层特征长度
      weights     : 各层权重
      cell_m      : 网格间距（米）
      length_m    : 场域边长（米）
      layers      : dict，各层单独的场（用于分层误差归因）

    说明
    ----
    `sd` 的保留是本项目从 F1 方向继承的一条硬纪律：
    **结果若被标准化，真值必须同时换算到同一尺度**，
    否则会出现"不随样本量消失的常数偏倚"这类伪装成方法缺陷的尺度错误。
    """
    if len(scales) != len(weights):
        raise ValueError("scales 与 weights 长度必须一致")

    rng = np.random.default_rng(seed)
    cell_m = length_m / n

    # 物理坐标（米）
    xs = (np.arange(n) + 0.5) * cell_m
    X, Y = np.meshgrid(xs, xs, indexing="xy")

    layers = {}
    field = np.zeros((n, n), dtype=float)

    for i, (L, w) in enumerate(zip(scales, weights)):
        # 高斯随机场的谱域构造：给白噪声施加尺度相关的低通滤波。
        # 用"频率域圆截止"实现，截止波数 k_c = 1 / L。
        white = rng.normal(0.0, 1.0, size=(n, n))
        F = np.fft.fft2(white)

        kx = np.fft.fftfreq(n, d=cell_m)   # 周期/米
        ky = np.fft.fftfreq(n, d=cell_m)
        KX, KY = np.meshgrid(kx, ky, indexing="xy")
        K = np.sqrt(KX ** 2 + KY ** 2)

        # 高斯型低通：k=0 处为 1，k >> 1/L 处迅速衰减
        filt = np.exp(-(K * L) ** 2 / 2.0)
        low = np.real(np.fft.ifft2(F * filt))

        # 去掉均值后按标准差归一，使各层振幅可控、可解释
        low = low - low.mean()
        s = low.std()
        if s > 1e-12:
            low = low / s
        layers[f"L{i+1}_{int(L)}m"] = low * w
        field += low * w

    raw = field.copy()
    sd = float(raw.std())
    truth = raw.copy()
    if standardize:
        if sd > 1e-12:
            truth = (truth - truth.mean()) / sd

    return {
        "truth": truth,
        "raw": raw,
        "sd": sd,
        "mean_raw": float(raw.mean()),
        "scales": tuple(scales),
        "weights": tuple(weights),
        "cell_m": cell_m,
        "length_m": length_m,
        "n": n,
        "layers": layers,
        "seed": seed,
    }


# ----------------------------------------------------------------------------
# 池体布局（高位池格网的几何模型）
# ----------------------------------------------------------------------------
def make_pond_layout(n: int = 128,
                     length_m: float = 3200.0,
                     pond_side_m: float = 50.0,
                     gap_m: float = 10.0,
                     seed: int = 0) -> dict:
    """
    生成一个规则的"高位池连片"布局。

    参数
    ----
    pond_side_m : 单池边长（米）。
                  现实参照：高位池单池 3-5 亩 ≈ 0.2-0.33 ha；
                  方形时边长约 sqrt(2000)-sqrt(3300) ≈ 45-57 m。
                  默认 50 m 取其中值（标 **【引用】+【待核实】**，见唯一事实来源 §8 的 V-3）。
    gap_m : 池间道路/塘埂宽度（米）。默认 10 m（**示范性设定**，非实测）。

    返回
    ----
    dict:
      pond_id     : (n, n) int，-1 表示非池体（道路/塘埂），>=0 为池编号
      n_ponds     : 池数
      pond_side_m : 单池边长
      cell_m      : 网格间距
      pond_area_ha: 单池面积（公顷）
    """
    cell_m = length_m / n
    pond_id = np.full((n, n), -1, dtype=int)

    # 池体 + 间隔的周期长度
    pitch = pond_side_m + gap_m
    if pitch > length_m:
        raise ValueError("单池+间隔大于场域边长，请增大 length_m 或减小 pond_side_m")

    n_per_axis = int(length_m // pitch)
    side_cells = int(round(pond_side_m / cell_m))
    gap_cells = int(round(gap_m / cell_m))
    if side_cells < 1:
        raise ValueError("单池边长小于一个网格，请提高 n 或增大 pond_side_m")

    pid = 0
    for i in range(n_per_axis):
        for j in range(n_per_axis):
            r0 = i * (side_cells + gap_cells)
            c0 = j * (side_cells + gap_cells)
            r1, c1 = r0 + side_cells, c0 + side_cells
            if r1 <= n and c1 <= n:
                pond_id[r0:r1, c0:c1] = pid
                pid += 1

    return {
        "pond_id": pond_id,
        "n_ponds": pid,
        "pond_side_m": pond_side_m,
        "gap_m": gap_m,
        "cell_m": cell_m,
        "length_m": length_m,
        "n": n,
        "pond_area_ha": (pond_side_m ** 2) / 10000.0,
        "n_per_axis": n_per_axis,
    }


# ----------------------------------------------------------------------------
# 降采样（模拟"输出栅格粗于池体"）
# ----------------------------------------------------------------------------
def block_average(field: np.ndarray, factor: int) -> np.ndarray:
    """
    把 field 按 factor×factor 做块平均（**唯一的合法粗化方式**）。

    ⚠️ 刻意不提供 `field[::factor, ::factor]` 这种"抽点采样"路径，
    因为它会引入混叠且不是融合的平均语义；本诊断要测的是
    **块平均带来的尺度失配**，不是采样策略的差异。
    若将来确需对比，应显式命名为 `point_sample` 并单独说明。

    要求 field 的边长可被 factor 整除。
    """
    n = field.shape[0]
    if field.shape[0] != field.shape[1]:
        raise ValueError("要求方阵")
    if factor < 1:
        raise ValueError("factor 必须 >= 1")
    if n % factor != 0:
        raise ValueError(f"边长 {n} 不能被 factor={factor} 整除")
    m = n // factor
    return field.reshape(m, factor, m, factor).mean(axis=(1, 3))


def target_factor_for_cell(cell_m: float, target_m: float) -> int:
    """给定当前网格间距与目标间距（米），返回需要整除的 factor（>=1）。"""
    if target_m <= cell_m:
        return 1
    return int(round(target_m / cell_m))


def effective_cell_m(cell_m: float, factor: int) -> float:
    """粗化后的有效网格间距（米）。"""
    return cell_m * factor
