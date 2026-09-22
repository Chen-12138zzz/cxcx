# -*- coding: utf-8 -*-
"""
F4 原型系统 · 一键复现全部测试与实验

【为什么需要这个脚本】
本项目的核心风险不是"程序跑不起来"，而是"跑起来了但静默算错"。
因此把「单元测试 + 四个实验」串成一条命令，任何一次改动后跑一遍即可确认
没有破坏既有性质。退出码为所有子步骤的**最大值**，可直接作为门禁：
    exit 0 = 全绿；非 0 = 有失败（并会打印是哪一步）

【运行】
    python experiments/run_all.py            # 全部
    python experiments/run_all.py --skip-tests   # 只跑实验
"""
from __future__ import annotations

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)        # 04-原型系统/

PY = sys.executable

STEPS = [
    ("单元测试 · 140 条不变量断言", os.path.join(_PKG, "tests", "run_tests.py"),
     True),
    ("实验 0 · 合成场结构可分辨性", os.path.join(_HERE, "exp0_structure.py"), False),
    ("实验 1 · 尺度失配的代价", os.path.join(_HERE, "exp1_scale.py"), False),
    ("实验 2 · 融合的收益与代价", os.path.join(_HERE, "exp2_fusion.py"), False),
    ("实验 3 · 缺失→可计算后果", os.path.join(_HERE, "exp3_audit.py"), False),
]


def main() -> int:
    skip_tests = "--skip-tests" in sys.argv
    print("=" * 78)
    print("F4 原型系统 · 一键复现")
    print("=" * 78)
    print(f"Python: {PY}")
    print(f"工作目录: {_PKG}")

    codes = []
    for name, path, is_test in STEPS:
        if skip_tests and is_test:
            print(f"\n[跳过] {name}")
            continue
        if not os.path.exists(path):
            print(f"\n[缺失] {name} —— 找不到 {path}")
            codes.append(1)
            continue
        print("\n" + "-" * 78)
        print(f"[运行] {name}")
        print(f"       {path}")
        print("-" * 78)
        r = subprocess.run([PY, "-u", path], cwd=_PKG)
        codes.append(r.returncode)
        print(f"[退出码 {r.returncode}] {name}")

    worst = max(codes) if codes else 0
    print("\n" + "=" * 78)
    print(f"总退出码：{worst}  （{'全部通过' if worst == 0 else '存在失败'}）")
    print("=" * 78)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
