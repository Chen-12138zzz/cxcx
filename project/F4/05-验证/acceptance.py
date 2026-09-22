# -*- coding: utf-8 -*-
"""
F4 交付验收：依次跑通全部门禁，汇总退出码。

【门禁清单】
  1. 单元测试 + 4 个实验        `04-原型系统/experiments/run_all.py`
  2. 跨文档一致性校验            `consistency_check.py`（skill 自带，需 --baseline baseline.json）
  3. 校验器基线不死性自检        `05-验证/check_baseline_liveness.py`
  4. 接口一致性核对              `05-验证/check_api_consistency.py`
  5. 源代码清单与代码同步        `04-原型系统/experiments/gen_source_listing.py`

【用法】
    python 05-验证/acceptance.py
退出码 = 各步骤退出码的最大值（0 = 全部通过 = 交付就绪）。
"""
from __future__ import annotations

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
F4 = os.path.dirname(_HERE)
PY = sys.executable

CK = (r"C:\Users\Chen\.agents\skills\project-deliverable-workspace"
      r"\scripts\consistency_check.py")

STEPS = [
    ("门禁 1 · 单元测试 + 4 个实验",
     [PY, "-u", os.path.join(F4, "04-原型系统", "experiments", "run_all.py")]),
    ("门禁 2 · 跨文档一致性校验",
     [PY, "-u", CK, "--root", ".", "--baseline", "baseline.json"]),
    ("门禁 3 · 校验器基线不死性自检",
     [PY, "-u", os.path.join(_HERE, "check_baseline_liveness.py")]),
    ("门禁 4 · 接口一致性核对",
     [PY, "-u", os.path.join(_HERE, "check_api_consistency.py")]),
    ("门禁 5 · 源代码清单再生成",
     [PY, "-u", os.path.join(F4, "04-原型系统", "experiments",
                             "gen_source_listing.py")]),
]

KEYS = ("总计", "结果：", "总退出码", "权威值缺位", "不死性自检：",
        "公开接口符号", "合计:", "代码行", "待检查文件",
        "FAIL", "偏离", "未触发", "发现的问题")


def main() -> int:
    codes = []
    for name, cmd in STEPS:
        print("\n" + "=" * 78)
        print(f"[{name}]")
        print("=" * 78)
        r = subprocess.run(cmd, cwd=F4, capture_output=True, text=True,
                           encoding="utf-8")
        out = (r.stdout or "") + (r.stderr or "")
        for ln in out.splitlines():
            if any(k in ln for k in KEYS):
                print("   " + ln.strip())
        print(f"   → 退出码 {r.returncode}")
        codes.append(r.returncode)

    print("\n" + "=" * 78)
    print("F4 交付验收汇总")
    print("=" * 78)
    for (name, _), c in zip(STEPS, codes):
        print(f"  [{'PASS' if c == 0 else 'FAIL'}] 退出码 {c}  {name}")
    worst = max(codes) if codes else 0
    print("-" * 78)
    print(f"总退出码：{worst}  "
          f"（{'全部通过 —— 交付就绪' if worst == 0 else '存在失败'}）")
    print("=" * 78)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
