# -*- coding: utf-8 -*-
"""
F1 · 合成养殖观测数据生成器（已知真值的因果结构）

【为什么需要它】
因果推断有一个真实数据无法提供的条件：**可以构造已知真值的数据**。
真实养殖数据没有 "真实效应" 可比对，因此无法判断一条估计流程是否正确。
本模块生成一个**显式写死因果结构**的合成数据集，使得：

    ATE 真值 与 CATE 真值 均可解析计算

从而可以量化「标准方法在这个结构上偏多少」，并验证诊断模块能否发现假设违背。

【与真实数据的关系 —— 必须遵守的纪律】
本模块产出的一切均为 **【合成】** 数据，其因果结构是**本项目设定的假设**，
不是真实养殖系统的因果结构。
- 不得表述为实测结果或真实养殖结论（红线 K-5）
- 不得用本模块的结论推断真实塘口的效应量
它唯一的用途是：**验证估计流程与诊断模块的实现是否正确**。

【结构（与 03-技术方案/核心算法设计说明.md 第 5.1 节一致）】
    W  ~ 预处理协变量：塘面积、水深、塘型
    S  ~ 季节/气温（与 W 相关）
    U  ~ 未测混杂：虾的摄食状态（★ 强度可控开关）
    T  = f(W, S, U) + noise         ← 干预由协变量 + 未测混杂共同决定（内生）
    WQ = g(W, S, T) + noise         ← 水质：受 T 影响 ⇒ 是【中介】不是混杂
    Y  = h(W, S, T, WQ, U) + noise  ← 结果

【真值推导】
因为所有函数形式显式已知，ATE 与 CATE 可解析求出（见 compute_truth）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 结构常数
# 全部系数写死为模块级常量，便于测试断言与审计。
C_T_ON_W_AREA = 0.35       # 塘面积越大 → 投喂总量越高
C_T_ON_W_DEPTH = 0.20
C_T_ON_S = -0.45           # 气温越高 → 投喂越高（低温少投）
C_T_ON_U = 0.60            # ★ 未测混杂：虾的摄食状态越活跃 → 投喂越高

C_WQ_ON_T = 0.55           # 投喂 ↑ → 有机物 ↑ → 溶氧 ↓（此处用正号表示"水质退化指数"）
C_WQ_ON_S = 0.30
C_WQ_ON_W_DEPTH = -0.15

C_Y_ON_T = 0.40            # 干预对结果的真实效应（线性主效应）
C_Y_ON_WQ = -0.25          # 水质退化 → 生长下降（中介通路）
C_Y_ON_S = 0.35
C_Y_ON_U = 0.50            # ★ 未测混杂直接影响结果
C_Y_ON_W_AREA = -0.10
C_Y_T_SQUARED = -0.18      # 非线性：投喂过高反而有害（阈值效应）
C_Y_T_WQ_INTER = -0.12     # 交互：水质差时投喂的边际效应更低

NOISE_T = 1.0
NOISE_WQ = 1.0
NOISE_Y = 1.0

# 干预（投喂量）的二值化切点：按分位数确定
TREAT_QUANTILE = 0.5


def _standardize(a: np.ndarray) -> np.ndarray:
    s = a.std()
    return (a - a.mean()) / (s if s > 1e-12 else 1.0)


def generate(n: int = 800,
             u_strength: float = 0.0,
             nonlinear: bool = True,
             positivity_violation: float = 0.0,
             seed: int = 20260920) -> pd.DataFrame:
    """生成一份合成养殖观测数据集。

    参数
    ----
    n : 样本量（塘·周期 数）
    u_strength : 未测混杂强度开关。0 = 无未测混杂（U 不影响 T，但仍影响 Y）；
                 1 = 按结构常数 C_T_ON_U 生效。取值可 >1 做压力测试。
                 实现方式：T 方程中的 U 项系数 = C_T_ON_U * u_strength。
                 **注意**：U 始终影响 Y（C_Y_ON_U 不变），因此 u_strength=0
                 表示"U 影响结果但不影响干预"——此时 U 不是混杂，调整它也没必要。
    nonlinear : 是否启用 T² 项与交互项。False → 纯线性结构，用于对照。
    positivity_violation : ★ 正性违背参数（0–1），**但请注意其实际语义**。
                 实现方式：在"低温 + 大塘"层给 T **加一个常数 lift**，
                 即把该层的 T 分布**整体平移**上来。

                 ⚠️⚠️ **重要更正（2026-09-20，见 05-验证/修正记录.md R-12）**：
                 平移分布 **不等于** 截断支撑集。实测确认：
                   viol=0.0 时该层 T 的 min = −1.615；
                   viol=0.8 时该层 T 的 min = −0.790（仅抬高 0.8，仍连续）。
                 因此该层在**任何**投喂箱上的概率密度都仍 > 0，
                 低投喂箱的空缺只是**概率降低**，不是**机制排除**。
                 推论：本参数**不能**制造机制性（结构性）正性违背，
                 也不能用来验证"结构性空箱可被检出"。
                 验证证据（n: 3000 → 96000，32 倍）：
                   viol=0.8 的空箱层数 7 → 0（与无违背同样消失）。
                 若确需构造机制性违背，须**硬截断支撑集**
                 （把该层低于阈值的 T 重抽为高值），见
                 `.tmp-tools/_r7_final.py` 的 `hard_truncate`。

    seed : 随机种子（可复现性要求：实验脚本必须固定种子并记录）

    返回
    ----
    DataFrame，列：
        pond_area, pond_depth, pond_type, season_temp     ← 预处理协变量 W/S
        feed_rate                                          ← 干预 T（连续，标准化尺度）
        feed_high                                          ← 干预 T（二值，按中位数切）
        water_degrad                                       ← 时变变量（★ 中介）
        fcr                                                ← ★ 碰撞（同时依赖 T 与 Y）
        growth                                             ← 结果 Y
        shrimp_appetite                                    ← ★ 未测混杂 U（仅用于诊断评估，
                                                              **不得进入任何调整集**）
        _true_ate, _true_cate                              ← 真值（仅用于评估，不得进入模型）
    """
    rng = np.random.default_rng(seed)

    # ---- W / S：预处理协变量与季节 ----
    pond_area = _standardize(rng.normal(0, 1, n))                 # 塘面积（标准化）
    pond_depth = _standardize(rng.normal(0, 1, n))                # 水深（标准化）
    pond_type = rng.integers(0, 2, n)                             # 0 = 土塘, 1 = 高位池/铺膜
    season_temp = _standardize(rng.normal(0, 1, n) + 0.25 * pond_type)  # 气温（与塘型弱相关）

    # ---- U：未测混杂（虾的摄食状态） ----
    # 与季节弱相关（温度影响摄食），但**在数据分析中 U 不作为可用变量**
    shrimp_appetite = _standardize(rng.normal(0, 1, n) + 0.2 * season_temp)

    # ---- T：干预（投喂量）----
    t_lin = (C_T_ON_W_AREA * pond_area
             + C_T_ON_W_DEPTH * pond_depth
             + C_T_ON_S * season_temp
             + C_T_ON_U * u_strength * shrimp_appetite)
    feed_rate = t_lin + rng.normal(0, NOISE_T, n)

    # 正性违背：在"低温 + 大塘"层强制抬高投喂量下限，
    # 使这些层几乎不出现低投喂 → 正性假设在该层被破坏
    if positivity_violation > 0:
        low_temp = season_temp < np.quantile(season_temp, 0.3)
        big_pond = pond_area > np.quantile(pond_area, 0.6)
        forced = low_temp & big_pond
        lift = positivity_violation * (np.quantile(feed_rate, 0.7) - np.quantile(feed_rate, 0.3))
        feed_rate = np.where(forced, feed_rate + lift, feed_rate)

    feed_rate = _standardize(feed_rate)

    # ---- WQ：水质退化指数（★ 中介：受 T 影响，且影响 Y）----
    # ⚠️ 关键：这里先算**原始** WQ，再标准化；标准化常数（raw 的 mean/sd）被记入
    #    df.attrs，供真值计算与反事实构造共用。
    #    如果真值路径自己另算一套标准化常数，两条路径就不在同一尺度上，
    #    解析真值与数值反事实会对不上（本模块第一版曾在此出错，由
    #    tests/test_truth_selfcheck.py 抓出）。
    wq_raw = (C_WQ_ON_T * feed_rate
              + C_WQ_ON_S * season_temp
              + C_WQ_ON_W_DEPTH * pond_depth
              + rng.normal(0, NOISE_WQ, n))
    wq_mu = float(wq_raw.mean())
    wq_sd = float(wq_raw.std())
    water_degrad = (wq_raw - wq_mu) / (wq_sd if wq_sd > 1e-12 else 1.0)

    # ---- Y：结果（生长表现指数）----
    y = (C_Y_ON_T * feed_rate
         + C_Y_ON_WQ * water_degrad
         + C_Y_ON_S * season_temp
         + C_Y_ON_U * shrimp_appetite
         + C_Y_ON_W_AREA * pond_area)
    if nonlinear:
        y = y + C_Y_T_SQUARED * feed_rate ** 2 + C_Y_T_WQ_INTER * feed_rate * water_degrad
    y = y + rng.normal(0, NOISE_Y, n)
    growth = _standardize(y)

    # ---- FCR：★ 碰撞（同时依赖 T 与 Y，因此不得进入调整集）----
    fcr = _standardize(0.6 * feed_rate - 0.5 * growth + rng.normal(0, 0.5, n))

    # ---- 二值化干预（按中位数切；记录切点以便真值计算）----
    cut = float(np.quantile(feed_rate, TREAT_QUANTILE))
    feed_high = (feed_rate >= cut).astype(int)

    df = pd.DataFrame({
        "pond_area": pond_area,
        "pond_depth": pond_depth,
        "pond_type": pond_type,
        "season_temp": season_temp,
        "feed_rate": feed_rate,
        "feed_high": feed_high,
        "water_degrad": water_degrad,
        "fcr": fcr,
        "growth": growth,
        "shrimp_appetite": shrimp_appetite,
    })

    # ---- 真值 ----
    ate, cate = compute_truth(df, cut=cut, high=float(feed_rate[feed_high == 1].mean()),
                              low=float(feed_rate[feed_high == 0].mean()),
                              nonlinear=nonlinear, wq_sd=wq_sd, y_sd=float(y.std()))
    df["_true_ate"] = ate
    df["_true_cate"] = cate
    df.attrs["cut"] = cut
    df.attrs["treat_high_mean"] = float(feed_rate[feed_high == 1].mean())
    df.attrs["treat_low_mean"] = float(feed_rate[feed_high == 0].mean())
    df.attrs["wq_mu"] = wq_mu
    df.attrs["wq_sd"] = wq_sd
    df.attrs["y_sd"] = float(y.std())   # ★ 真值换算尺度，审计用
    df.attrs["nonlinear"] = nonlinear
    df.attrs["u_strength"] = u_strength
    df.attrs["positivity_violation"] = positivity_violation
    df.attrs["n"] = n
    df.attrs["seed"] = seed
    return df


def compute_truth(df: pd.DataFrame, cut: float, high: float, low: float,
                  nonlinear: bool, wq_sd: float,
                  y_sd: float = 1.0) -> tuple[float, np.ndarray]:
    """解析计算真值。

    【关键推导】结果方程中与干预 t 有关的部分有两条通路：

        ① 直接通路      T → Y       : C_Y_ON_T * t
        ② 中介通路      T → WQ → Y  : C_Y_ON_WQ * (C_WQ_ON_T * t / wq_sd)

    ⚠️ 中介通路**计入总效应**（干预确实通过它影响了结果）。
    这与「把 WQ 放进调整集会屏蔽部分效应」是同一件事的两面：
    调整中介 ⇒ 只估直接效应，不估总效应。

    ⚠️ 中介通路里的 `wq_sd` 是不可省略的：模型用的 water_degrad 是**标准化后**的量，
    因此"T 改变 1 单位 ⇒ 标准化 WQ 改变 C_WQ_ON_T / wq_sd"。
    漏掉这个因子会让解析真值与数值反事实相差 1/wq_sd 倍
    （本模块第一版即此错，由 tests/test_truth_selfcheck.py 抓出）。

    ⚠️ `y_sd` 同样不可省略（本模块第二版的错，2026-09-20 修正）：
    结果列 `growth` 是 **标准化后** 的 y（`growth = (y - mean) / sd(y)`），
    而估计器回归的正是 `growth`。因此所有估计量都作用在 y/sd(y) 尺度上。
    若这里仍按 **原始 y 单位** 给真值，则真值会被系统性放大 1/sd(y) ≈ 1.19 倍，
    表现为一个**与样本量无关的常数偏倚**（实测 -0.044，n=200000 仍存在），
    并被误判成"估计器有偏"或"非线性导致".
    定位证据：调整集 OLS 系数 0.2335 × sd(y)=1.19293 = 0.2785 ≈ true_slope 0.2775。

    非线性项：
        C_Y_T_SQUARED * t²            —— 投喂过高反而有害（阈值效应）
        C_Y_T_WQ_INTER * t * WQ(t)    —— 水质差时投喂的边际效应更低

    由于非线性项的存在，ATE 依赖于 t 的具体取值，因此本函数按**实际的组均值
    high / low** 计算，而不是取 ±1 之类的抽象值。
    """
    scale = wq_sd if abs(wq_sd) > 1e-12 else 1.0
    yscale = y_sd if abs(y_sd) > 1e-12 else 1.0

    def y_effect_part(t: np.ndarray | float) -> np.ndarray | float:
        t = np.asarray(t, dtype=float)
        wq_from_t = (C_WQ_ON_T / scale) * t        # ← T 对【标准化】WQ 的贡献
        val = C_Y_ON_T * t + C_Y_ON_WQ * wq_from_t
        if nonlinear:
            val = val + C_Y_T_SQUARED * t ** 2 + C_Y_T_WQ_INTER * t * wq_from_t
        return val

    # ★ 换算到估计器实际作用的尺度：growth = y / sd(y)（均值中心化不影响差分）
    ate_true = float((y_effect_part(high) - y_effect_part(low)) / yscale)

    # CATE：本合成结构的非线性是全局的（不含协变量交互），因此 CATE 在本结构下为常数。
    # ⚠️ 如实说明：该结构**不能**用来检验异质性检测能力。
    # 异质性验证需要在结构中加入 T × X 交互项（见 TODO-1）。
    cate = np.full(len(df), ate_true)
    return ate_true, cate


def available_covariates(include_mediator: bool = False,
                         include_bad_control: bool = False) -> list[str]:
    """返回可用作调整集的变量。

    include_mediator=True  → 加入 water_degrad（★ 会屏蔽部分效应）
    include_bad_control=True → 加入 fcr（★ 碰撞，禁止）
    shrimp_appetite 永不返回（未测混杂，不得进入调整集）。
    """
    cols = ["pond_area", "pond_depth", "pond_type", "season_temp"]
    if include_mediator:
        cols.append("water_degrad")
    if include_bad_control:
        cols.append("fcr")
    return cols
