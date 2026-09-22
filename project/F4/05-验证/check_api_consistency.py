# -*- coding: utf-8 -*-
"""
接口一致性核对：从 `src/` **实际代码**抽取公开接口签名。

【为什么必须有这个脚本】
`consistency_check.py` 只做**文档之间**的校验，
**"代码签名 ↔ 接口文档"的一致性没有任何自动检查**。

本项目实测因此漏过 2 处缺陷（`修正记录.md` 的 A-6、A-7）：
  • A-6 `synth.py` / `mce.py` 缺 `__all__`（"哪些是公开 API"只存在于约定中，无法被机器读取）
  • A-7 `ahp_priority_from_consistent` 注解 `-> np.ndarray` 但实返回 `dict`（注解不参与运行）

⇒ 因此接口文档中的签名**一律由本脚本抽取**，禁止手工转录。

【用法】
    python 05-验证/check_api_consistency.py
    # 可选：与基线比对
    python 05-验证/check_api_consistency.py --expect-baseline

【接口基线】synth 5 / mce 9 / fusion 8 / audit 3 = **25**
任何偏离都视为**接口变更**，需同步更新 `03-技术方案/03-接口设计文档.md`
并登记到 `修正记录.md`。
"""
from __future__ import annotations

import inspect
import os
import sys

# 本脚本位于 project/F4/05-验证/，src 的父目录是 04-原型系统/
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.join(os.path.dirname(_HERE), "04-原型系统")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

BASELINE = {"src.synth": 5, "src.mce": 9, "src.fusion": 8, "src.audit": 3}
BASELINE_TOTAL = sum(BASELINE.values())


def main() -> int:
    import src.synth as synth
    import src.mce as mce
    import src.fusion as fusion
    import src.audit as audit

    mods = [("src.synth", synth), ("src.mce", mce),
            ("src.fusion", fusion), ("src.audit", audit)]

    print("=" * 74)
    print("F4 公开接口抽取（由实际代码生成，非手工转录）")
    print("=" * 74)

    counts = {}
    bad = []
    for name, mod in mods:
        alls = list(getattr(mod, "__all__", []))
        counts[name] = len(alls)
        if not alls:
            bad.append(f"{name} 未声明 __all__（公开契约无法被机器读取）")
        print(f"\n[{name}]  {len(alls)} 个公开符号")
        for sym in alls:
            obj = getattr(mod, sym, None)
            if obj is None:
                bad.append(f"{name}.{sym} 在 __all__ 中但不存在")
                print(f"    *** {sym}  —— __all__ 声明了但模块里没有")
                continue
            try:
                sig = str(inspect.signature(obj))
            except (TypeError, ValueError):
                sig = "(?)"
            doc = (inspect.getdoc(obj) or "").strip().split("\n")[0]
            print(f"    {sym}{sig}")
            if doc:
                print(f"        # {doc[:88]}")
            # 注解与实现一致性：返回注解声称 ndarray 但实际不是函数的常见错误
            ann = getattr(obj, "__annotations__", {}).get("return")
            if ann is not None and ("dict" in str(ann)) == ("dict" not in doc.lower()
                                                            and False):
                pass  # 占位：注解核对需要运行期样本，见下方提示

    total = sum(counts.values())
    print("\n" + "=" * 74)
    print("计数")
    print("=" * 74)
    for k in BASELINE:
        mark = "OK  " if counts.get(k) == BASELINE[k] else "*** 偏离 ***"
        print(f"  {mark} {k}: {counts.get(k)}  （基线 {BASELINE[k]}）")
    mark = "OK  " if total == BASELINE_TOTAL else "*** 偏离 ***"
    print(f"  {mark} 合计: {total}  （基线 {BASELINE_TOTAL}）")

    if bad:
        print("\n发现的问题：")
        for b in bad:
            print(f"  - {b}")

    print("\n" + "-" * 74)
    print("⚠️ 本脚本只能核对**符号数量与签名**，**不能**核对返回注解与实现是否相符")
    print("   （注解不参与运行，需要运行期样本才能验证 —— 见 A-7）。")
    print("   若改动 src/，请重跑本脚本并与 03-技术方案/03-接口设计文档.md 比对。")
    print("=" * 74)
    return 1 if (bad or total != BASELINE_TOTAL) else 0


if __name__ == "__main__":
    raise SystemExit(main())
