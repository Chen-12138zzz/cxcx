# -*- coding: utf-8 -*-
"""
F1 · 识别假设台账（identification ledger）

【模块存在的原因】
因果结论的"可信度"不来自算法，而来自**识别假设**。
假设是一串自然语言论断，永远不会出现在代码里，因此极易在项目推进中被遗忘、
被软化、或被静默改写成"已验证"。

本模块把假设写成**机器可读的台账**，并强制：
  ① 每条假设有明确的**证据等级**（哪些能用数据检验、哪些根本不能）
  ② 每条假设有**检验方式**与**检验证据的存放位置**
  ③ 生成报告时必须逐条输出当前状态，**不允许只输出结论不输出假设清单**
  ④ "诊断通过"不能升级为"假设成立"——台账里两者是不同字段

【证据等级（写死在代码里，不允许调高）】
    "testable"      可用数据做诊断，通过/不通过可自动判定
                    （但仍只能得出"未发现违背"）
    "partially"     只能部分检验，需要额外假设才能推进
                    （例：测量误差只能估衰减上界，方向可判、大小需假设）
    "untestable"    数据里没有任何信息可用（典型：无未测混杂）
                    只能做敏感性分析，**永远不能标记为"已验证"**

【红线】
  任何把 "untestable" 条目标注为 "verified" 的操作，
  都会被 `assert_ledger_valid()` 拒绝（抛异常），
  以此防止"不小心"把不可检验的假设写成结论。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

# 允许的状态取值
VALID_STATUS = {"unverified", "partially_verified", "violated", "not_applicable"}
# ★ 注意 "verified" 不在允许列表里：识别假设永远不能被"验证"，
#   最多只能说"未发现违背"（partially_verified）。
#   唯一可以标 verified 的是"可检验的**诊断**是否已运行"，
#   那是另一个字段（diagnostic_ran）。


@dataclass
class Assumption:
    """一条识别假设。"""
    id: str                       # A1, A2, ...
    name: str                     # 简短名称
    statement: str                # 假设的完整陈述
    evidence_level: str           # testable | partially | untestable
    diagnostic: str               # 检验方式（"无" 表示不可检验）
    evidence_file: str            # 检验证据存放位置（结果文件路径）
    status: str = "unverified"    # 当前状态
    diagnostic_ran: bool = False  # 检验是否已实际运行
    consequence_if_violated: str = ""   # 若被违背，结论会怎样
    notes: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.evidence_level == "untestable" and self.diagnostic != "无":
            raise ValueError(
                f"[{self.id}] evidence_level=untestable 却填了 diagnostic；"
                f"不可检验的假设不应有检验方式（若确有数据可查，请改为 partially）")
        if self.evidence_level != "untestable" and self.diagnostic == "无":
            raise ValueError(
                f"[{self.id}] evidence_level={self.evidence_level} 却没有检验方式；"
                f"要么补上 diagnostic，要么改为 untestable")
        if self.status not in VALID_STATUS:
            raise ValueError(
                f"[{self.id}] status='{self.status}' 非法。"
                f"允许值：{sorted(VALID_STATUS)}。"
                f"★ 'verified' 被刻意排除——识别假设不能被验证，"
                f"只能用 'partially_verified' 表示『已诊断且未发现违背』")

        # ★ 以下三条是跨字段一致性检查，必须放在 __post_init__ 里。
        #   第一版把它们只写在 assert_ledger_valid()，而那个函数在**构造对象时
        #   不会自动被调用** —— 于是"把不可检验项标为已验证"这种非法写法
        #   能顺利构造出对象，防线形同虚设。
        #   由 tests/test_ledger_guard.py 的反向验证抓出（见 08-复盘）。
        if self.evidence_level == "untestable" and self.status == "partially_verified":
            raise ValueError(
                f"[{self.id}] 不可检验的假设却标为 partially_verified；"
                f"只能标 unverified（或 not_applicable）。"
                f"★ 无未测混杂这类假设在任何数据上都无法检验，"
                f"标注为『已诊断』是对读者的误导")
        if self.status == "partially_verified" and not self.diagnostic_ran:
            raise ValueError(
                f"[{self.id}] 标为 partially_verified 但 diagnostic_ran=False；"
                f"未实际运行诊断不得声称已诊断")
        if self.status == "violated" and not self.diagnostic_ran:
            raise ValueError(
                f"[{self.id}] 标为 violated 但 diagnostic_ran=False；"
                f"未运行诊断不可能知道假设被违背")

    @property
    def must_disclose(self) -> bool:
        """是否必须在报告的显著位置披露（所有 untestable 与 violated 项）。"""
        return self.evidence_level == "untestable" or self.status == "violated"

    def to_row(self) -> dict:
        return {
            "编号": self.id,
            "假设": self.name,
            "证据等级": self.evidence_level,
            "检验方式": self.diagnostic,
            "已运行": "是" if self.diagnostic_ran else "否",
            "当前状态": self.status,
            "若违背": self.consequence_if_violated,
        }


# ------------------------------------------------------------------ 本项目台账
def default_ledger() -> list[Assumption]:
    """本项目（F1 养殖因果推断）的识别假设台账。

    与 03-技术方案/核心算法设计说明.md 第 4 节逐条对应。
    """
    return [
        Assumption(
            id="A1", name="无未测混杂",
            statement=("给定调整集 X = {塘面积, 水深, 塘型, 气温} 后，"
                       "干预 T（投喂量）与潜在结果 Y(t) 独立。"
                       "通俗说：所有同时影响『投了多少』与『长得怎样』的因素都已被观测到。"),
            evidence_level="untestable",
            diagnostic="无",
            evidence_file="05-验证/敏感性分析结果.md",
            consequence_if_violated=("估计值向『虾本身状态好』的方向偏移，"
                                     "效应被高估；偏倚量与 U 的强度成正比，"
                                     "无法从数据中估计，只能给出『推翻所需强度』"),
            notes=["本项目最脆弱的一条：虾的摄食/健康状态无法逐塘观测",
                   "唯一能做的事是 E-value 与偏 R² 敏感性分析，给出『需要多强才能推翻』",
                   "★ 永远不得标记为已验证"],
        ),
        Assumption(
            id="A2", name="正性 / 重叠",
            statement=("对每个协变量层，投喂量在任一取值上都有正概率密度；"
                       "即各层都有共同支撑，不会出现『某类塘口从不出现某档投喂量』。"),
            evidence_level="testable",
            diagnostic="diagnostics/positivity.py：T 分位分箱 × 协变量分层，统计空箱",
            evidence_file="05-验证/正性诊断结果.md",
            consequence_if_violated=("缺失箱的反事实结果靠模型外推，"
                                     "不是从数据估出；若缺失严重，"
                                     "结论必须限定在『有共同支撑的子总体』"),
            notes=["判定规则已代码化：存在空箱 → violated；"
                   "存在箱计数 < 5 → warning；否则 ok（ok ≠ 假设成立）"],
        ),
        Assumption(
            id="A3", name="一致性 / SUTVA",
            statement=("一个塘口的潜在结果只取决于它自己接受的投喂量，"
                       "不受其他塘口投喂决策影响（无干扰）；"
                       "且『投喂量』的定义在各塘口间一致（无版本差异）。"),
            evidence_level="partially",
            diagnostic=("空间邻近性检查：若邻近塘口存在共同水源/换水通道，"
                        "干扰假说无法排除；本项目的处理是**在报告中声明**"
                        "而非检验"),
            evidence_file="05-验证/实验与验证报告.md#SUTVA",
            consequence_if_violated=("若塘口间有水质相互影响（共用水源），"
                                     "估计的是『投喂 + 其外溢效应』，"
                                     "与个体层面的干预效应不等价"),
            notes=["散户高位池多在相邻位置、可能共用进排水渠 → 干扰不可完全排除",
                   "★ 本项目合成数据不含空间结构，因此该条在合成验证中不可检验",
                   "真实数据阶段须实地确认进排水是否独立"],
        ),
        Assumption(
            id="A4", name="调整集不含中介",
            statement="水质退化指数不在调整集中（要估的是总效应）。",
            evidence_level="testable",
            diagnostic="diagnostics/bad_control.py：audit_adjustment_set 静态审计 + Δθ 对照",
            evidence_file="05-验证/坏控制检查结果.md",
            consequence_if_violated=("估计的是控制水质后的直接效应，"
                                     "系统性小于总效应；报告若标为总效应即为错误"),
            notes=["常见错误做法：把水质当作『控制变量』放进回归 —— "
                   "看起来更严谨，实际改变了估计目标"],
        ),
        Assumption(
            id="A5", name="调整集不含碰撞",
            statement="FCR（饲料转化率）不在调整集中。",
            evidence_level="testable",
            diagnostic="diagnostics/bad_control.py：同上；FCR = 饲料量/增重，分子含 T、分母含 Y",
            evidence_file="05-验证/坏控制检查结果.md",
            consequence_if_violated=("打开 T 与 Y 之间的后门通路，"
                                     "制造本来不存在的相关；偏倚方向与大小均不可预期"),
            notes=["FCR 在养殖文献里被大量当作协变量使用 —— "
                   "本项目认为这是结构性错误，会在报告中说明理由"],
        ),
        Assumption(
            id="A6", name="干预变量为实际值",
            statement=("投喂记录反映的是**实际投喂量**，而不是计划投喂量或"
                       "『应该投』的推荐值。"),
            evidence_level="partially",
            diagnostic=("可估测量误差导致的**衰减上界**：若有少量塘口的实际投喂量"
                        "可独立核实（如称重记录），可估计测量误差方差比例，"
                        "进而给出真实效应的下界修正"),
            evidence_file="待建（依赖 B-1 数据通道）",
            status="unverified",
            consequence_if_violated=("若记录为计划值，则真实投喂与记录的偏离使"
                                     "估计效应**向零衰减**（attenuation bias）；"
                                     "真实效应的绝对值大于估计值"),
            notes=["散户常有『按经验投』而不称量 → 记录可能是估计值",
                   "★ 这是真实数据阶段必须现场核实的一条（阻断项 B-1 的一部分）",
                   "当前无法判定，报告中须显式声明"],
        ),
    ]


# ------------------------------------------------------------------ 校验与输出
def assert_ledger_valid(ledger: list[Assumption]) -> None:
    """台账自检：任何非法状态在写报告前就抛异常。

    ⚠️ 注意分工：
      - **单条**的自洽性检查在 `Assumption.__post_init__` 里，
        构造对象时立即生效（这是主要防线）。
      - 本函数负责**集合级**检查（编号重复、跨条一致性）。
      第一版把单条检查也放在这里，结果构造对象时不触发，防线失效——
      由 tests/test_ledger_guard.py 的反向验证抓出。
    """
    seen = set()
    for a in ledger:
        if a.id in seen:
            raise ValueError(f"台账中假设编号重复：{a.id}")
        seen.add(a.id)
        # 再次核对（防止有人绕过 dataclass 直接改属性）
        if a.evidence_level == "untestable" and a.status == "partially_verified":
            raise ValueError(
                f"[{a.id}] 不可检验的假设却标为 partially_verified；"
                f"只能标 unverified（或 not_applicable）")
        if a.status == "partially_verified" and not a.diagnostic_ran:
            raise ValueError(
                f"[{a.id}] 标为 partially_verified 但 diagnostic_ran=False；"
                f"未实际运行诊断不得声称已诊断")
        if a.status == "violated" and not a.diagnostic_ran:
            raise ValueError(
                f"[{a.id}] 标为 violated 但 diagnostic_ran=False；"
                f"未运行诊断不可能知道假设被违背")


def ledger_to_md(ledger: list[Assumption], title: str = "识别假设台账") -> str:
    """生成可直接嵌入报告的 Markdown 表格。"""
    lines = [f"### {title}", ""]
    lines.append("| 编号 | 假设 | 证据等级 | 检验方式 | 已运行 | 当前状态 |")
    lines.append("|---|---|---|---|---|---|")
    lvl = {"testable": "可检验", "partially": "部分可检验", "untestable": "**不可检验**"}
    st = {"unverified": "未检验", "partially_verified": "已诊断·未见违背",
          "violated": "**已违背**", "not_applicable": "不适用"}
    for a in ledger:
        lines.append(f"| {a.id} | {a.name} | {lvl.get(a.evidence_level, a.evidence_level)} "
                     f"| {a.diagnostic} | {'是' if a.diagnostic_ran else '否'} "
                     f"| {st.get(a.status, a.status)} |")
    lines.append("")
    lines.append("**须显著披露的条目**（不可检验，或已违背）：")
    lines.append("")
    for a in ledger:
        if a.must_disclose:
            lines.append(f"- **{a.id} {a.name}**（{lvl.get(a.evidence_level)}，"
                         f"{st.get(a.status)}）：若被违背，{a.consequence_if_violated}")
    lines.append("")
    lines.append("> 说明：识别假设**没有一条能被『验证』**。"
                 "表中『已诊断·未见违背』仅表示所采用的诊断程序"
                 "在当前数据分辨率下未发现问题，**不等于假设成立**。")
    return "\n".join(lines)


def ledger_to_json(ledger: list[Assumption], path: str | Path) -> None:
    """台账落地为 JSON，供一致性校验脚本与报告生成器读取。"""
    assert_ledger_valid(ledger)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([asdict(a) for a in ledger], ensure_ascii=False, indent=2),
                 encoding="utf-8")


def load_ledger(path: str | Path) -> list[Assumption]:
    """从 JSON 读回台账（并重新执行合法性校验）。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = [Assumption(**r) for r in raw]
    assert_ledger_valid(out)
    return out


def coverage_summary(ledger: list[Assumption]) -> dict:
    """台账覆盖度摘要：不可检验项数、已诊断项数等（供报告与校验脚本使用）。"""
    n = len(ledger)
    by_lvl: dict[str, int] = {}
    for a in ledger:
        by_lvl[a.evidence_level] = by_lvl.get(a.evidence_level, 0) + 1
    return {
        "total": n,
        "by_evidence_level": by_lvl,
        "n_untestable": by_lvl.get("untestable", 0),
        "n_testable": by_lvl.get("testable", 0) + by_lvl.get("partially", 0),
        "n_diagnostic_ran": sum(1 for a in ledger if a.diagnostic_ran),
        "n_must_disclose": sum(1 for a in ledger if a.must_disclose),
        "n_violated": sum(1 for a in ledger if a.status == "violated"),
        "all_diagnostics_run": all(
            a.diagnostic_ran for a in ledger if a.evidence_level != "untestable"),
    }


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    print("=" * 90)
    print("识别假设台账自检")
    print("=" * 90)
    lg = default_ledger()
    assert_ledger_valid(lg)
    print(f"\n台账共 {len(lg)} 条，合法性检查通过。\n")
    print(ledger_to_md(lg))

    cs = coverage_summary(lg)
    print("\n【覆盖度摘要】")
    for k, v in cs.items():
        print(f"  {k:24s} {v}")

    # ---- 反向验证：故意写入非法状态，必须抛异常 ----
    print("\n【反向验证】以下四种非法写法必须被拒绝：")
    tests = [
        ("把不可检验项标为已验证", dict(
            id="X1", name="t", statement="s", evidence_level="untestable",
            diagnostic="无", evidence_file="-", status="partially_verified")),
        ("不可检验项却填了检验方式", dict(
            id="X2", name="t", statement="s", evidence_level="untestable",
            diagnostic="某诊断", evidence_file="-")),
        ("声称已诊断但标记未运行", dict(
            id="X3", name="t", statement="s", evidence_level="testable",
            diagnostic="d", evidence_file="-", status="partially_verified",
            diagnostic_ran=False)),
        ("声称已违背但未运行诊断", dict(
            id="X4", name="t", statement="s", evidence_level="testable",
            diagnostic="d", evidence_file="-", status="violated",
            diagnostic_ran=False)),
    ]
    all_rejected = True
    for label, kwargs in tests:
        try:
            Assumption(**kwargs)
            print(f"  ✗ {label} —— **未被拒绝，这是缺陷**")
            all_rejected = False
        except ValueError as e:
            print(f"  ✓ {label} —— 已拒绝")
            print(f"      {str(e)[:96]}")
    print(f"\n  反向验证总体：{'全部被正确拒绝' if all_rejected else '**存在漏网的非法写法**'}")

    # ---- 落地台账 JSON ----
    # ★ 路径：src/ledger.py → 上两级到 04-原型系统 → 上一级到 F1 → _source/
    #   第一版只上了两级，把台账写到了 project/_source/（少了 F1 层），
    #   由本脚本的输出核对抓出。
    f1_root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    out = os.path.join(f1_root, "_source", "assumption_ledger.json")
    assert os.path.basename(f1_root) == "F1", (
        f"路径解析异常：f1_root 应为 .../F1，实际为 {f1_root}")
    ledger_to_json(lg, out)
    print(f"\n台账已写入：{out}")

    # 读回验证
    back = load_ledger(out)
    print(f"读回验证：{len(back)} 条，{'一致' if len(back) == len(lg) else '不一致'}")
