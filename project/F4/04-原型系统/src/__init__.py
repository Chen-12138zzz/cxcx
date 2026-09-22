# -*- coding: utf-8 -*-
"""
F4 原型系统 · 顶层包

【为什么需要这个文件】
`fusion/` 与 `audit/` 内部使用相对导入（`from ..mce import ...`），
这要求存在一个**同一个顶层包**。若把 `src/` 本身当作包根，
`..mce` 就会越过顶层包边界而报
`ImportError: attempted relative import beyond top-level package`。

【正确的调用方式】
从 `04-原型系统/` 目录（即 `src/` 的**父目录**）运行：
    python -c "from src.fusion import scale_mismatch_curve"

或直接在 `tests/run_tests.py` 中把 `src/` 的**父目录**加入 sys.path，
然后 `import src.mce`。tests/run_tests.py 已按此处理。

⚠️ 反例（会挂）：把 `src/` 本身加入 sys.path 后 `import fusion`。
   那样 `fusion` 成为顶层包，`..mce` 无父包可回，必然 ImportError。
"""
