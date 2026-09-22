# -*- coding: utf-8 -*-
"""
F1 · 诊断模块包

【为什么诊断是本项目的核心而不是附属】
因果推断与机器学习的根本差别在于：
    ML 只要预测准就行；因果推断要**预测对反事实**，
    而反事实不可观测 → 结论完全建立在**不可检验的识别假设**上。

因此本项目的产品价值不在"能算出一个效应值"（谁都能算），
而在"**能说清这个值在什么条件下可信、在什么条件下不可信**"。
三个诊断模块对应的正是三条最容易在养殖数据上被破坏的假设：

    positivity.py   正性 / 重叠       —— "某些塘口从不出现某种投喂量"
    bad_control.py  坏控制与中介      —— "把不该调整的变量放进了调整集"
    sensitivity.py  未测混杂敏感性    —— "还有一个没测到的因素在同时影响两者"

⚠️ 三个模块的共同纪律：
  诊断结果**只用于降低结论强度或给出警示区间**，
  不允许反向用来"证明假设成立"。任何"诊断通过"的表述都必须在
  报告中写明"诊断通过 ≠ 假设成立，只是未发现违背"。
"""

from .positivity import (  # noqa: F401
    PositivityDiagnosis,
    diagnose_positivity,
)
from .bad_control import (  # noqa: F401
    BadControlDiagnosis,
    check_bad_control,
)
from .sensitivity import (  # noqa: F401
    SensitivityResult,
    e_value,
    cih_sensitivity,
    summarize_sensitivity,
)

__all__ = [
    "PositivityDiagnosis", "diagnose_positivity",
    "BadControlDiagnosis", "check_bad_control",
    "SensitivityResult", "e_value", "cih_sensitivity", "summarize_sensitivity",
]
