# -*- coding: utf-8 -*-
"""
F1 · 正性 / 重叠诊断

【假设】
A2 正性（positivity / overlap）：
    对协变量 X 的任一取值，接受各处理水平的概率都严格大于 0。
    0 < P(T = t | X = x) < 1   （对连续 T：f(t|X=x) 在各处有共同支撑）

【为什么在养殖数据上特别容易违背】
投喂量不是随机分配的，而是**按塘口条件人为决定**的：
  - 大塘、深水塘天然投得更多 → 小塘几乎不会出现极高投喂量
  - 低温季节天然投得少     → 高温期几乎不会出现极低投喂量
  - 有经验的养殖户会根据水色、虾的摄食状态调整 → 某些组合从组合上都少见
结果是：**某些协变量层里，某个投喂区间完全没有观测**。
此时"该层在另一投喂水平下的反事实结果"是靠模型外推猜出来的，
不是从数据里估出来的。

【诊断方式】
本项目对连续干预采用**分位分箱 + 层内支撑度**的做法（而非仅看倾向得分）：
  ① 把 T 分成 L 个分位箱（默认 5 箱，对应"低/中低/中/中高/高"投喂）
  ② 对每个协变量层（按 X 的分位分箱交叉形成），统计各 T 箱的样本数
  ③ 报告：
     - 每层的最小箱样本数（= 0 表示该层缺失某个投喂水平 → 严重违背）
     - 有缺失的层占比
     - 各层的"有效支撑宽度"（覆盖了 T 的多少比例范围）
  ④ 连续 T 的另一个视角：对每个 X 层，看 T 的条件分布是否严重不平衡
     —— 用一个简单的可解释指标：层内 T 的 IQR 与全局 IQR 的比值。

【输出如何用】
  - 若存在完全缺失的层 → 在报告中**必须**给出"该层不可识别"的声明，
    并把结论限定在"有共同支撑的子总体"上（写明子总体是如何界定的）
  - 若缺失轻微 → 报告警示，并对结果做"仅保留有支撑层"的稳健性复算，
    两者差异一并报告
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PositivityDiagnosis:
    """正性诊断结果。

    ★★ 重要设计说明（第三版，2026-09-20 依据实测证据重写）

    【本诊断能回答什么、不能回答什么】

    本诊断**不主张**判定"结构性空箱"（即"该组合在机制上不可能出现因此
    反事实不可识别"）。第三版起**已移除 `structural` 这一判定类别**。

    理由是实测证伪，证据见 `.tmp-tools/_r7_final.py` 与
    `05-验证/修正记录.md` 的 R-11：

      在「全局等频箱 × 分位协变量层」框架下，**空箱计数是样本量依赖的**。
      把 n 从 3000 放大到 96000（32 倍）后：
        - 无任何违背时：空箱层数 4 → 0（必消失）
        - 人为**硬截断**某子群体的 T 支撑集（概率恰为 0）后：
          空箱层数 7 → 0（**同样消失**）
      两者**在样本量放大后都收敛到 0** ⇒ 空箱存在与否**无法区分**
      "抽样波动"与"机制性缺失"。分辨率被 `n / (层数 × 箱数)` 限死，
      而分层已把样本摊薄到 `n/81` 再除以 5 箱。

    【因此本诊断只报告三件事，均为**样本量相关**的有限样本陈述】

      1. **稀疏空箱**（"sparse"）—— 存在空箱，且该层每箱期望样本数不足。
         这是**有限样本问题**：扩样本或合并层/减箱数可缓解。
         报告表述须写「本样本量下细分层导致部分组合无观测」，
         **不得**写成「正性假设被违背」。

      2. **覆盖偏窄**（"narrow"）—— 无空箱，但存在层内 T 支撑区间
         明显窄于全局的层（`stratum_iqr_ratio` 低）。这是唯一
         **样本量不敏感**的信号，但它同样**不等于**正性被违背：
         它只说明该层的干预取值范围窄，外推风险高。

      3. **分辨率不足**（"undetermined"）—— 存在空箱，且该层每箱
         期望样本数高于判定阈值，因而**无法**用"期望不足"解释。
         ★ 本类别**不是**"结构性"的同义词。它只诚实地表明：
         在本样本量与本分层方案下，本诊断**没有能力**判定空箱成因。
         需要更强证据（外部情境知识、离散 T、或行政约束记录）才能定性。

    ⚠️ 判据的迭代史（保留以说明为何不再声称能判结构性）：
      第 1 版「层总样本量 ≥ 阈值 却空 ⇒ 结构性」：n=600 时每箱期望仅 2.2，
             空箱必然，却因层中位样本量 11 被误判 → 25 个假阳性。
      第 2 版「该箱期望样本数 ≥ 5 却空 ⇒ 结构性」：把期望改成
             `层样本量 / 箱数`。**仍失败** —— n=3000、54 层下
             20/20 个种子在**无任何违背**时全部报 structural。
      第 3 版（本版）不再试图判结构性，改为诚实三分类。
    """
    n: int
    n_strata: int
    n_boxes: int                       # 投喂箱数（供报告自由度和期望值解读）
    # --- 稀疏性 ---
    n_strata_sparse_gap: int           # 仅因期望样本数不足而空箱的层数
    n_strata_small: int                # 每箱期望样本数不足的层数
    median_stratum_size: float
    median_expected_in_box: float      # 每层每箱期望样本数的中位数
    expected_per_box: float            # 全局平均 = n / (n_strata * n_boxes)
    # --- 分辨率不足（★ 原 "结构性"，第三版更名，语义已变更）---
    n_strata_undetermined: int         # 空箱且期望充足 → 本诊断无法定性
    undetermined_fraction: float
    # --- 通用 ---
    min_box_count: int
    worst_stratum: dict
    box_edges: np.ndarray
    box_counts_by_stratum: np.ndarray
    stratum_sizes: np.ndarray
    global_iqr: float
    stratum_iqr_ratio: np.ndarray
    verdict: str                       # "ok"|"sparse"|"narrow"|"undetermined"
    notes: list[str] = field(default_factory=list)

    # ---- 兼容别名（第三版新增；旧字段名保留但不推荐使用）----
    # ⚠️ 别名的语义与旧版**不同**：旧 `n_strata_struct_gap` 声称"结构性"，
    #    新字段只声称"本诊断无法定性"。引用旧名的代码须重新审视其结论表述。
    @property
    def n_strata_struct_gap(self) -> int:
        """[已弃用] 旧名 → 指向 `n_strata_undetermined`。语义已变更，见类文档。"""
        return self.n_strata_undetermined

    @property
    def struct_gap_fraction(self) -> float:
        """[已弃用] 旧名 → 指向 `undetermined_fraction`。语义已变更。"""
        return self.undetermined_fraction

    def summary_rows(self) -> list[dict]:
        return [
            {"指标": "样本量 n", "取值": self.n},
            {"指标": "协变量层数 × 投喂箱数",
             "取值": f"{self.n_strata} × {self.n_boxes}"},
            {"指标": "全局平均每箱期望样本数", "取值": f"{self.expected_per_box:.1f}"},
            {"指标": "层样本量（中位）", "取值": f"{self.median_stratum_size:.1f}"},
            {"指标": "★ 每层每箱期望样本数（中位）",
             "取值": f"{self.median_expected_in_box:.1f}"},
            {"指标": "期望样本数不足的层数", "取值": self.n_strata_small},
            {"指标": "稀疏空箱层数（期望不足导致）", "取值": self.n_strata_sparse_gap},
            {"指标": "★ 空箱但期望充足的层数（本诊断无法定性）",
             "取值": self.n_strata_undetermined},
            {"指标": "★ 上述层占比", "取值": f"{self.undetermined_fraction:.1%}"},
            {"指标": "最小箱样本数", "取值": self.min_box_count},
            {"指标": "全局 T 的 IQR", "取值": f"{self.global_iqr:.4f}"},
            {"指标": "层内 IQR 比值（最小/中位/最大）",
             "取值": (f"{np.min(self.stratum_iqr_ratio):.3f} / "
                      f"{np.median(self.stratum_iqr_ratio):.3f} / "
                      f"{np.max(self.stratum_iqr_ratio):.3f}")},
            {"指标": "判定", "取值": self.verdict},
        ]

    def subgroup_definition(self) -> str:
        """返回「有共同支撑的子总体」的自然语言界定（报告须直接引用）。"""
        gap = self.n_strata_sparse_gap + self.n_strata_undetermined
        if gap == 0:
            return "全部样本均在各投喂箱上有支撑，无需限定子总体。"
        return (f"剔除 {gap} 个存在空投喂箱的协变量层后，"
                f"剩余 {self.n_strata - gap} 层（原 {self.n_strata} 层的 "
                f"{(self.n_strata - gap)/self.n_strata:.1%}）"
                f"在每个投喂箱上均有观测。结论的适用范围应限定于该子总体。")


def diagnose_positivity(df: pd.DataFrame, t_col: str = "feed_rate",
                        covariates: list[str] | None = None,
                        n_boxes: int = 5,
                        strata_per_covariate: int = 3,
                        min_count: int = 5,
                        min_stratum_size: int = 10,
                        min_expected_in_box: float | None = None,
                        seed: int = 20260920) -> PositivityDiagnosis:
    """诊断正性/重叠假设（区分稀疏空箱与结构性空箱）。

    参数
    ----
    n_boxes : T 的分位箱数（默认 5）
    strata_per_covariate : 每个协变量按分位切几段（默认 3）
    min_count : 判定"该箱支撑不足"的样本数阈值（默认 5）
    min_stratum_size : 辅助阈值，仅用于"层样本量极大"的辅助判据（默认 10）
    min_expected_in_box : ★ 判定"结构性"的**核心阈值**（默认 None → 取
        max(3.0, min_count)）：
        当某层的**每箱期望样本数**（= 层样本量 / 箱数）达到该阈值，
        却仍有空箱时，才判为结构性。

    判定规则（代码化，不靠事后自由裁量）
    --------
    "structural" ：存在结构性空箱（期望样本数充足却仍空）
    "sparse"     ：无结构性空箱，但存在稀疏空箱（期望样本数不足所致）
    "warning"    ：无空箱，但存在箱样本数 < min_count 的层
    "ok"         ：所有层的所有箱样本数 ≥ min_count
    """
    if covariates is None:
        from synth import available_covariates
        covariates = available_covariates()

    T = df[t_col].to_numpy(dtype=float)
    n = len(df)

    # ---- ① T 的分位箱 ----
    qs = np.linspace(0, 1, n_boxes + 1)
    edges = np.quantile(T, qs)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    box = np.clip(np.digitize(T, edges) - 1, 0, n_boxes - 1)

    # ---- ② 协变量层 ----
    codes = []
    for c in covariates:
        v = df[c].to_numpy(dtype=float)
        if len(np.unique(v)) <= strata_per_covariate:
            codes.append(pd.factorize(v)[0])
        else:
            q = np.linspace(0, 1, strata_per_covariate + 1)
            e = np.quantile(v, q)
            e[0] -= 1e-9
            e[-1] += 1e-9
            codes.append(np.clip(np.digitize(v, e) - 1, 0, strata_per_covariate - 1))

    strata_id = np.zeros(n, dtype=np.int64)
    for cc in codes:
        strata_id = strata_id * (int(cc.max()) + 1) + cc
    uniq, inv = np.unique(strata_id, return_inverse=True)
    n_strata = len(uniq)

    # ---- ③ 逐层逐箱计数 ----
    counts = np.zeros((n_strata, n_boxes), dtype=int)
    for s in range(n_strata):
        m = inv == s
        for b in range(n_boxes):
            counts[s, b] = int(np.sum(m & (box == b)))
    sizes = counts.sum(axis=1)

    has_gap = (counts == 0).any(axis=1)

    # ★★★ 第三版判据（2026-09-20 重写，依据 R-11 的实测证据）★★★
    #
    # 【为什么不再判"结构性"】
    #
    # 前两版都试图用"期望样本数够不够"来区分稀疏/结构性。实测证伪：
    # 在「全局等频箱 × 分位协变量层」框架下，空箱计数**必然是样本量依赖的**。
    # 证据（.tmp-tools/_r7_final.py，n: 3000 → 96000，32 倍）：
    #
    #   无任何违背时   ：gap 层 4 → 0    （必然消失）
    #   硬截断支撑集后 ：gap 层 7 → 0    （★ 概率恰为 0 也消失）
    #
    # 两者放大样本后都收敛到 0 ⇒ 空箱存在与否**无法区分**抽样波动与机制缺失。
    # 分辨率被 n/(层数×箱数) = n/405 限死；分层先把样本摊到 n/81，再除 5 箱。
    #
    # 因此第三版只做**诚实的有限样本陈述**，三分类如下：
    #
    #   "sparse"       ：有空箱，且该层每箱期望样本数 < 阈值
    #                    → 空箱可由"期望不足"充分解释（有限样本问题）
    #   "undetermined" ：有空箱，但该层每箱期望样本数 ≥ 阈值
    #                    → 本诊断**没有能力**解释成因（既不判稀疏也不判结构性）
    #   "narrow"       ：无空箱，但存在层内 T 支撑明显窄于全局的层
    #   "ok"           ：无空箱，且各层支撑宽度可接受
    #
    # ⚠️ `min_expected_in_box` 的作用已变更：它**不再**是"结构性阈值"，
    #    而是"稀疏解释是否充分"的分界。名称保留以兼容既有调用。
    min_expected_in_box = (max(3.0, float(min_count))
                           if min_expected_in_box is None else float(min_expected_in_box))
    exp_in_box = sizes / float(n_boxes)                 # 每层每箱的期望样本数
    enough = exp_in_box >= min_expected_in_box

    # "期望不足" ⇒ 空箱可由有限样本充分解释
    sparse_gap = has_gap & ~enough
    # "期望充足却空" ⇒ 本诊断无法定性（★ 不再叫 structural）
    undetermined = has_gap & enough

    n_sparse = int(sparse_gap.sum())
    n_undet = int(undetermined.sum())
    n_small = int((exp_in_box < min_expected_in_box).sum())
    min_cnt = int(counts.min())
    exp_per_box = n / (n_strata * n_boxes) if n_strata and n_boxes else float("nan")

    # ---- ④ 层内 IQR 相对全局（★ 须先于 narrow 判定计算）----
    global_iqr = float(np.subtract(*np.percentile(T, [75, 25])))
    ratios = np.full(n_strata, np.nan)
    for s in range(n_strata):
        Ts = T[inv == s]
        if len(Ts) >= 4:
            iqr_s = float(np.subtract(*np.percentile(Ts, [75, 25])))
            ratios[s] = iqr_s / global_iqr if global_iqr > 1e-12 else np.nan
    valid = ratios[np.isfinite(ratios)]

    # 覆盖偏窄的层：层内 T 的 IQR 相对全局明显偏窄（样本量不敏感信号）
    #   阈值 0.5 为经验值：低于全局 IQR 一半 ⇒ 该层干预取值范围明显受限
    narrow_strata = np.isfinite(ratios) & (ratios < 0.5)
    n_narrow = int(narrow_strata.sum())

    # 最差层：优先"无法定性"层中样本最多的；其次稀疏层；否则最小计数层
    for mask in (undetermined, sparse_gap):
        cand = np.where(mask)[0]
        if len(cand):
            worst_s = int(cand[np.argmax(sizes[cand])])
            break
    else:
        worst_s = int(np.argmin(counts.min(axis=1)))
    worst = {
        "stratum_index": worst_s,
        "stratum_size": int(sizes[worst_s]),
        "counts": counts[worst_s].tolist(),
        "empty_boxes": np.where(counts[worst_s] == 0)[0].tolist(),
        "is_undetermined": bool(undetermined[worst_s]),
        "t_range": [
            float(T[inv == worst_s].min()) if (inv == worst_s).any() else float("nan"),
            float(T[inv == worst_s].max()) if (inv == worst_s).any() else float("nan"),
        ],
    }

    global_iqr = global_iqr  # noqa: PLW0127  (kept for readability of §④/§⑤ order)

    # ---- ④ 层内 IQR 相对全局 ----
    global_iqr = float(np.subtract(*np.percentile(T, [75, 25])))
    ratios = np.full(n_strata, np.nan)
    for s in range(n_strata):
        Ts = T[inv == s]
        if len(Ts) >= 4:
            iqr_s = float(np.subtract(*np.percentile(Ts, [75, 25])))
            ratios[s] = iqr_s / global_iqr if global_iqr > 1e-12 else np.nan
    valid = ratios[np.isfinite(ratios)]

    # 覆盖偏窄的层：层内 T 的 IQR 相对全局明显偏窄（样本量不敏感信号）
    #   阈值 0.5 为经验值：低于全局 IQR 一半 ⇒ 该层干预取值范围明显受限
    narrow_strata = np.isfinite(ratios) & (ratios < 0.5)
    n_narrow = int(narrow_strata.sum())

    # ---- ⑤ 判定（第三版：诚实三分类 + narrow）----
    if n_undet > 0:
        verdict = "undetermined"
    elif n_sparse > 0:
        verdict = "sparse"
    elif n_narrow > 0:
        verdict = "narrow"
    else:
        verdict = "ok"

    notes: list[str] = []
    if verdict == "undetermined":
        notes.append(
            f"存在 {n_undet} 个**有空箱、且无法用『期望样本数不足』解释**的层"
            f"（每箱期望样本数已 ≥ {min_expected_in_box:.0f}，"
            f"中位 {np.median(exp_in_box[enough]):.1f}）。"
            f"★ 本诊断**不对其成因作判断**：既不能称其为『稀疏』，"
            f"**也不得**称其为『结构性』。"
            f"理由：本项目的验证（见 05-验证/修正记录.md R-11）显示，在"
            f"『全局等频箱 × 分位协变量层』框架下，把 n 放大 32 倍后，"
            f"**无论是否人为硬截断支撑集，空箱都同样消失** —— "
            f"空箱计数无法区分抽样波动与机制缺失。"
            f"要定性需借助本诊断之外的证据：场景知识、离散型 T、或"
            f"制度/操作约束记录。")
        notes.append(
            "报告要求：结论须限定在『有共同支撑的子总体』，"
            "并写明该子总体的界定方式（见 subgroup_definition）；"
            "成因留作待查事项，**不得**据此断言反事实不可识别")
    elif verdict == "sparse":
        notes.append(
            f"存在 {n_sparse} 个**稀疏**空箱层：这些层的每箱期望样本数不足 "
            f"{min_expected_in_box:.0f}（全局平均仅 {exp_per_box:.1f}），"
            "空箱可由抽样波动充分解释。")
        notes.append(
            "★ 这是**有限样本问题**，不是数据生成机制的问题："
            "扩大样本量、减少分层数（strata_per_covariate）或减少投喂箱数"
            "（n_boxes）都能缓解。报告表述须写"
            "『本样本量下细分层导致部分组合无观测』，"
            "**不得**写成『正性假设被违背』——后者是对机制的断言，本诊断不支持")
    elif verdict == "narrow":
        notes.append(
            f"无空箱，但存在 {n_narrow} 个层的层内 T 支撑明显窄于全局"
            f"（IQR 比值 < 0.5，最小 {np.min(valid):.3f}）。"
            "★ 这是本诊断中**唯一对样本量不敏感**的信号，"
            "但它同样**不等于**正性被违背：只说明这些层的干预取值范围窄，"
            "外推风险高。报告须提示并做剔除后复算")
    else:
        notes.append(f"未发现空箱或样本数 < {min_count} 的层；"
                     "★ 这不等于正性成立，只是在本诊断的分辨率下未发现违背")

    if n_small > 0:
        notes.append(
            f"另有 {n_small}/{n_strata} 个层的每箱期望样本数 < "
            f"{min_expected_in_box:.0f}；分层方案为 "
            f"{strata_per_covariate}^{len(covariates)} = {n_strata} 层 × {n_boxes} 箱，"
            f"对 n={n} 而言分辨率偏高。报告须说明层数/箱数的选择依据，"
            f"并建议同时报告一个更粗的分层方案作为稳健性对照。")

    return PositivityDiagnosis(
        n=n, n_strata=n_strata, n_boxes=n_boxes,
        n_strata_sparse_gap=n_sparse, n_strata_small=n_small,
        median_stratum_size=float(np.median(sizes)),
        median_expected_in_box=float(np.median(exp_in_box)),
        expected_per_box=float(exp_per_box),
        n_strata_undetermined=n_undet,
        undetermined_fraction=n_undet / n_strata if n_strata else float("nan"),
        min_box_count=min_cnt, worst_stratum=worst,
        box_edges=edges, box_counts_by_stratum=counts, stratum_sizes=sizes,
        global_iqr=global_iqr, stratum_iqr_ratio=valid,
        verdict=verdict, notes=notes,
    )


def restricted_sample(df: pd.DataFrame, diag: PositivityDiagnosis,
                      t_col: str = "feed_rate",
                      covariates: list[str] | None = None,
                      n_boxes: int = 5,
                      strata_per_covariate: int = 3) -> pd.DataFrame:
    """返回**仅保留有共同支撑层**的子样本，用于稳健性复算。

    ★ 用法：在完整样本与子样本上各估一次效应，两者差异一并报告。
      若差异很大 → 说明结论严重依赖无支撑层的外推，必须如实说明；
      若差异很小 → 可作为"结论对正性违背不敏感"的**证据之一**（不是证明）。
    """
    if covariates is None:
        from synth import available_covariates
        covariates = available_covariates()

    T = df[t_col].to_numpy(dtype=float)
    edges = diag.box_edges
    box = np.clip(np.digitize(T, edges) - 1, 0, n_boxes - 1)

    codes = []
    for c in covariates:
        v = df[c].to_numpy(dtype=float)
        if len(np.unique(v)) <= strata_per_covariate:
            codes.append(pd.factorize(v)[0])
        else:
            q = np.linspace(0, 1, strata_per_covariate + 1)
            e = np.quantile(v, q)
            e[0] -= 1e-9
            e[-1] += 1e-9
            codes.append(np.clip(np.digitize(v, e) - 1, 0, strata_per_covariate - 1))

    strata_id = np.zeros(len(df), dtype=np.int64)
    for cc in codes:
        strata_id = strata_id * (int(cc.max()) + 1) + cc
    uniq, inv = np.unique(strata_id, return_inverse=True)

    keep = np.zeros(len(df), dtype=bool)
    good = (diag.box_counts_by_stratum > 0).all(axis=1)
    for s in np.where(good)[0]:
        keep |= (inv == s)
    out = df[keep].copy()
    out.attrs.update(df.attrs)
    out.attrs["positivity_restricted"] = True
    out.attrs["n_dropped"] = int((~keep).sum())
    return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import synth

    print("=" * 90)
    print("正性诊断冒烟测试")
    print("=" * 90)
    for pv in (0.0, 0.6, 1.0):
        df = synth.generate(n=1500, u_strength=0.3, nonlinear=True,
                            positivity_violation=pv, seed=11)
        d = diagnose_positivity(df)
        print(f"\n=== positivity_violation={pv} ===")
        for r in d.summary_rows():
            print(f"  {r['指标']:34s} {r['取值']}")
        if d.n_strata_with_gap:
            w = d.worst_stratum
            print(f"  最差层 #{w['stratum_index']}（n={w['stratum_size']}）"
                  f" 各投喂箱计数 = {w['counts']}  空箱 = {w['empty_boxes']}")
        for nt in d.notes:
            print(f"  · {nt}")
        if d.n_strata_with_gap:
            r = restricted_sample(df, d)
            print(f"  子样本（仅保留有支撑层）：n={len(r)}（剔除 {r.attrs['n_dropped']}）")
