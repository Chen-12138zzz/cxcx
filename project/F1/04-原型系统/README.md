# F1 原型系统 · README

> **一句话**：这是一个**因果推断估计与诊断**的可复现原型，
> 用来回答「在养殖观测数据上做干预效应估计时，哪些假设可能失效、失效后偏多少」。
>
> **它不做什么**：不产生预警、不预测产量、不给养殖建议。
> 它只对**研究过程本身的可靠性**负责。
>
> ⚠️ **默认没有演示数据**。本目录**不含** `.csv`/`.parquet` 等数据集；
> 唯一的数据来源是 `src/synth.py` 现场生成（`synth.generate(...)`），种子固定、真值解析可算。
> 这样做是刻意的：**一个随包分发的数据文件会被误当成「某种真实数据的样本」**，
> 而合成结构只能验证流程实现，不能外推真实塘口（红线 K-5）。

---

## 1. 它解决什么问题

养殖观测数据里做因果推断，有三个**结构性陷阱**：

| 陷阱 | 在养殖场景中的具体形态 | 后果 |
|---|---|---|
| **中介被误当混杂** | 水质退化（`water_degrad`）既是投喂的结果，又是生长的原因 | 把中介放进调整集 ⇒ 只估"直接效应"，**低估总效应** |
| **碰撞被误当混杂** | 饲料系数（FCR）= 投喂量 / 增重，同时依赖干预与结果 | 把碰撞放进调整集 ⇒ 打开后门通路，**效应被严重放大** |
| **未测混杂不可检验** | 虾的摄食状态会同时影响投喂决策与生长，但**测不到** | 估计有偏，且**无任何统计量能告警** |

前两个陷阱可以**静态审计**（只看变量名就能判定）；第三个**原则上不可检验**，
只能做**敏感性分析**并如实披露。

本原型的贡献不是"给出一个新估计量"，而是把上述三类陷阱变成
**可执行的检查 + 可量化的代价 + 可审计的台账**。

---

## 2. 目录结构

```
04-原型系统/
├── src/
│   ├── synth.py                 合成数据生成器（已知真值的因果结构）
│   ├── estimators.py            主估计器（DML-PLR / AIPW / 朴素 / 调整 OLS）
│   ├── dose_response.py         剂量-反应曲线（两条路径）
│   ├── ledger.py                识别假设台账（A1–A6）
│   ├── report.py                报告生成器（含输出守卫）
│   └── diagnostics/
│       ├── __init__.py
│       ├── positivity.py        正性/重叠诊断（稀疏空箱 vs 判不了的层）
│       ├── bad_control.py       坏控制检测（中介/碰撞的角色审计）
│       └── sensitivity.py       E-value / 偏 R² 敏感性 / 压力测试
├── experiments/
│   ├── exp1_estimator_bias.py   估计器无偏性
│   ├── exp2_positivity.py       正性与样本量
│   ├── exp3_bad_control.py      坏控制的量化代价
│   ├── exp4_dose_response.py    剂量-反应曲线两条路径
│   └── exp5_sensitivity.py      未测混杂的敏感性分析
├── tests/
│   ├── test_truth_selfcheck.py  真值自检（解析 vs 数值反事实）
│   └── test_invariants.py       五项方法论不变量
└── README.md                    本文件
```

---

## 3. 快速上手

### 3.1 环境

```
Python 3.13（托管环境）
依赖：numpy / pandas / scipy / scikit-learn / statsmodels / doubleml>=0.11.4
```

本机解释器路径（托管环境）：
```
C:\Users\Chen\.workbuddy\binaries\python\envs\default\Scripts\python.exe
```

`econml` / `causalml` **未安装**，因果森林与 CATE 估计未实现（见第 7 节）。

### 3.2 跑通测试（必须先跑，确认真值可信）

```bash
# 真值自检：解析真值 vs 独立数值反事实
python tests/test_truth_selfcheck.py

# 五项方法论不变量
python tests/test_invariants.py
```

**通过标准**：两个脚本都应以退出码 0 结束。
若 `test_truth_selfcheck.py` 的相对差 > 2%，说明真值不可用，**后续实验结论全部作废**。

### 3.3 跑实验

```bash
python experiments/exp1_estimator_bias.py
python experiments/exp2_positivity.py
python experiments/exp3_bad_control.py
python experiments/exp4_dose_response.py
python experiments/exp5_sensitivity.py
```

结果落盘到 `../05-验证/results_exp*.json / .csv`。

⚠️ exp1/exp3/exp5 需要 DML 交叉拟合，**单个种子约 30 秒**，
完整跑一轮需要 5–15 分钟。建议后台运行。

---

## 4. 核心 API

### 4.1 主估计器（`estimators.py`）

```python
import synth
from estimators import estimate_ate_continuous, estimate_ate_binary

df = synth.generate(n=3000, u_strength=0.6, nonlinear=True, seed=11)
COV = ["pond_area", "pond_depth", "pond_type", "season_temp"]

r = estimate_ate_continuous(df, covariates=COV, engine="manual")
print(r.point, r.se, r.ci_low, r.ci_high)
print(r.contrast_kind)   # "slope" —— ★ 口径标记
```

| 函数 | 估计对象 | 口径 |
|---|---|---|
| `estimate_naive` | 不做任何调整 | slope |
| `estimate_adjusted_ols` | OLS + 协变量 | slope |
| `estimate_ate_continuous` | DML-PLR（正交化 + 交叉拟合） | slope |
| `estimate_ate_binary` | AIPW（二值干预，双稳健） | two-point |

### ★ 4.2 两种口径不可混用（本项目最重要的接口约定）

`CausalEstimate.contrast_kind` 明确标记每个估计量的口径：

| 口径 | 含义 | 谁用它 |
|---|---|---|
| `"slope"` | **每单位 T 的边际效应** | DML-PLR / OLS / naive |
| `"two-point"` | **两个 T 取值上反事实结果的差** | AIPW / `_true_ate` |

两者的换算关系：

```
two_point = slope × d_t     （d_t = treat_high_mean − treat_low_mean）
```

若需要把 slope 口径的估计换算成 two-point 口径：

```python
r2 = r.rescale_to_twopoint(d_t)   # ⚠️ 仅在效应近似线性时有效
```

**本项目曾在这一点上误判为 bug**（见 `05-验证/修正记录.md` 与
`03-技术方案/核心算法设计说明.md`），因此把它提升为显式接口字段。

### 4.3 诊断模块

```python
from diagnostics.positivity import diagnose_positivity
from diagnostics.bad_control import audit_adjustment_set, check_bad_control
from diagnostics.sensitivity import e_value, cih_sensitivity, stress_test_bias

# 正性诊断（区分稀疏空箱与「判不了」的层）
diag = diagnose_positivity(df, covariates=COV)
print(diag.verdict)                  # ok | sparse | narrow | undetermined
print(diag.summary_rows())

# 坏控制审计（静态：不需要数据）
a = audit_adjustment_set(COV + ["fcr"])
print(a["pass"], a["violations"])    # False, [{'变量': 'fcr', ...}]

# 敏感性分析
ev = e_value(r.point, se=r.se, ci_low=r.ci_low, ci_high=r.ci_high)
print(ev["e_value_point"], ev["e_value_ci"], ev["note"])
```

### 4.4 假设台账（`ledger.py`）

```python
from ledger import default_ledger, assert_ledger_valid, ledger_to_md

lg = default_ledger()          # A1–A6 六条识别假设
assert_ledger_valid(lg)
print(ledger_to_md(lg))
```

台账的 `status` 字段**刻意排除 `"verified"`** ——
只有 `unverified` / `partially_verified` / `violated` / `not_applicable`。
理由：识别假设（尤其 A1 无未测混杂）**不可被数据证实**，
允许写 `verified` 会诱导虚假的确定性。

守卫在 `Assumption.__post_init__` 中强制：
- 不可检验项（`evidence_level="untestable"`）**不得**标 `partially_verified`；
- 标 `partially_verified` 或 `violated` 的，**必须** `diagnostic_ran=True`。

---

## 5. 关键设计决策

### 5.1 为什么用合成数据而不是真实养殖数据

因果推断的核心困难是**真值不可得** —— 真实数据里没有"真实效应"可比对，
因此**无法判断一条估计流程是否正确**。

`synth.py` 写死一个显式因果结构，使 ATE 可**解析计算**，
从而能把"估计偏了多少"变成**可测量的数字**，而不是"看起来还行"。

**代价（必须承认）**：合成结构是**我们设定的假设**，
不是真实养殖系统的因果结构。它只能验证**流程实现是否正确**，
**不能**用来推断真实塘口的效应量。

### 5.2 真值为什么要单独自检

`compute_truth()` 是解析公式，`counterfactual_truth()` 是独立的数值反事实。
两者**互相验证**：若解析式写错，数值反事实会立刻暴露差异。

本项目在这个机制上抓出了**两个真值错误**：
- **R-1**：中介通路漏掉 `1/wq_sd` 因子（差 1.95 倍）；
- **R-2**：真值未换算到 Y 的标准化尺度（造成 −0.044 的常数偏倚）。

详见 `05-验证/修正记录.md`。

### 5.3 为什么剂量-反应曲线要两条路径

| 路径 | 方法 | 用途 |
|---|---|---|
| A（默认） | 多项式可加基（`degree=3`） | **主结果**：平滑、可解析微分 |
| B（仅参照） | GBR 直接拟合 | 只做**形状**对照 |

路径 B 拟合的是 `E[Y|T,X]` 的**原始面**，其水平值包含全部协变量的贡献，
**未做正交化**，因此与路径 A 的水平差**本来就不该相等**。

实测对解析真值的平均绝对误差：**路径 A 0.0081 / 路径 B 0.1941**（相差 24 倍）。
故路径 B 降级为形状参照，其水平差不作为通过条件。

### 5.4 正性诊断为什么**不再**声称能区分"结构性空箱"（★ 第三版重写）

**初版设计**（已被推翻）：区分"稀疏空箱"（期望样本数不足）与"结构性空箱"
（期望充足却仍空 ⇒ 机制问题）。两者处理方式不同，故分开报告。

**实测推翻了这一设计。** 在 n=3000、54 层、**无任何真实违背**时，
20/20 个种子**全部**被判 `structural`。三重证据（详见《修正记录》R-11）：

| 证据 | 结果 |
|---|---|
| 样本量放大 | n=3000 → 4 个 gap 层；n=12000 → **0**；n=48000 → **0** |
| 置换零模型（T 随机重排） | 5/5 轮 gap 层从 4/3/5/2/1 → **0** |
| 空箱位置 | 几乎全在**极端箱**（箱 0、箱 4），而 `层样本量/箱数` 是**均匀假设** |

**决定性实验**：连人为**硬截断支撑集**（低箱概率**恰为 0**）后，
gap 层也在 n=48000 时降到 0：

| n | 硬截断 gap 层 | 无截断 gap 层 |
|---|---|---|
| 3000 | 7 | 4 |
| 12000 | 1 | 0 |
| 48000 | **0** | **0** |

⇒ 在「全局等频箱 × 分位协变量层」框架下，**空箱计数无法区分抽样波动与机制缺失**
（每格期望被 `n/405` 限死）。**因此第三版删除了 `structural` 类别**，改为：

| 判定 | 含义 |
|---|---|
| `sparse` | 有空箱且期望不足 ⇒ 可由有限样本充分解释 |
| `undetermined` | 有空箱但期望充足 ⇒ ★ **本诊断判不了**（不是"结构性"） |
| `narrow` | 无空箱，但有层的 T 支撑明显窄于全局（**样本量不敏感**） |
| `ok` | 无空箱且支撑宽度可接受 |

★ 报告纪律：`undetermined` **不得**被解读为反事实不可识别；
要定性须借助本诊断之外的证据（场景知识、离散 T、制度约束记录）。

### 5.5 为什么输出要有守卫（`ReportGuard`）

报告生成器主动**拦截**不该出现的输出：

| 守卫 | 行为 |
|---|---|
| 曲线被判不平滑 | **拒绝**输出边际效应（数值微分会放大噪声） |
| 无 E-value 可算 | **抛异常**，而不是留空或填 0 |
| 无数据的指标 | 输出 `"—"`（破折号），**不允许留空或填 0** |

理由：报告里一个"看起来合理"的假数字，比一个显式的空缺危险得多。

---

## 6. 已知局限（必读）

| 编号 | 局限 | 影响 |
|---|---|---|
| **U-1** | ~~正性诊断在无真实违背时误报 `structural`~~ → **已升级为 R-11 并修复**。原「阈值邻近判定不稳」的根因描述**已被实测推翻**（实为 20/20 系统性假阳性）；第三版已移除 `structural` 类别 | 假阳性；**已修复**。新的能力上限：`undetermined` 只表明「判不了」 |
| **U-2** | 合成结构的非线性**不含 T×X 交互**，故 CATE 恒为常数 | **无法**检验异质性检测能力 |
| **U-3** | 偏 R² 敏感性是**近似参数化**，不是 Cinelli-Hazlett 的严格界 | 只能作「近似敏感性指标」解读，**不得**表述为调整后的置信区间 |
| **U-4** | E-value 对连续 Y 用 `RR ≈ exp(0.91d)` 近似转换 | 仅在效应不太大时可用；输出中已标注为近似 |
| **U-7** | 正性诊断的**能力上限**：`undetermined` 只表明「本诊断判不了」，**不是**「结构性违背」的证据 | 任何把 `undetermined` 读成「有机制问题」的表述都是过度解读（R-11） |
| **U-5** | `econml` / `causalml` 未安装 | 因果森林、CATE 估计**未实现** |
| **U-6** | 全部数值来自合成结构 | **不得**外推为真实塘口的效应量（红线 K-5） |

---

## 7. 纪律红线（对报告作者）

1. **数据模式必须标注**：本系统产出的一切都是**【合成】**，
   任何引用都必须带此标注（红线 K-5）；
2. **不得用「没人做」立项**，只能走能力差异型论证；
3. **不得把 E-value / 偏 R² 输出表述为"调整后的置信区间"**（红线 K-3）；
4. **不得把"我没检索到"写成"不存在"**；
5. **失败与局限必须留痕**，禁止只报通过的项。

---

## 8. 复现检查清单

新接手者按此顺序验证，任一步失败则停止：

- [ ] `python tests/test_truth_selfcheck.py` → 退出码 0，相对差 < 2%
- [ ] `python tests/test_invariants.py` → 退出码 0，全部不变量通过
- [ ] `python experiments/exp2_positivity.py` → 4/4 通过
- [ ] `python experiments/exp3_bad_control.py` → 3/3 通过
- [ ] `python experiments/exp4_dose_response.py` → 多项式路径 MAE < 0.05
- [ ] `python experiments/exp1_estimator_bias.py` → **3/3 通过**（P3 已改为跨档位单调不增，见 R-14）
- [ ] `python experiments/exp5_sensitivity.py` → **5/5 通过**（F4 已改用 `ry_for_0.00` 中位行，见 R-13）
- [ ] 抽查 `05-验证/results_*.json` 中 5 个数字，逐个能说明来源
- [ ] 读一遍 `05-验证/修正记录.md`，确认理解 R-2 与 R-4 的成因

---

## 9. 与文档的对应关系

| 模块 | 对应文档章节 |
|---|---|
| `synth.py` | `03-技术方案/核心算法设计说明.md` 第 5 节（因果结构） |
| `estimators.py` | 同上 第 4 节（估计量） |
| `dose_response.py` | 同上 第 6 节（剂量-反应） |
| `diagnostics/*` | 同上 第 7 节（诊断与敏感性） |
| `ledger.py` | 同上 第 3 节（识别假设） |
| `experiments/*` | `05-验证/实验与验证报告.md` |
| 全部修正 | `05-验证/修正记录.md` |

---

*本 README 中的每个数字都可在 `05-验证/` 的结果文件中复现。*
*若发现文档与结果文件不一致，以**结果文件**为准，并修正文档。*
