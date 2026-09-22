# -*- coding: utf-8 -*-
"""
生成源代码清单（供软著材料与结题报告引用）。

【为什么用脚本而不是手写】
手写的清单会在代码改动后立刻过期，而且**不会报错**。
本项目的纪律是"数字必须可追溯"，因此清单也由脚本产出，
并写入统计口径，便于任何人核对。

输出：06-成果/源代码清单.md + 源代码清单.json
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re

ROOT = r"E:\BJTU\CXCY\CXCY-PAPER\project\F4"
SRC = os.path.join(ROOT, "04-原型系统", "src")
TESTS = os.path.join(ROOT, "04-原型系统", "tests")
EXPS = os.path.join(ROOT, "04-原型系统", "experiments")
# 演示材料（PPT）的工具脚本：**不计入原型系统的代码行统计**，
# 单独列一组。理由：软著与结题口径中的「原型系统本体行数」指的是
# **原型系统本体**；把演示脚本混进去会让这个数字与已发布口径不一致。
# （差一点踩进"文件里出现的数字与统计口径对不上"这类静默缺陷 —— 见 05-验证/修正记录.md D-3）
DECKTOOLS = os.path.join(ROOT, "06-成果", "空间适宜性审计工具链-产品与落地方案",
                         "tools")
OUTDIR = os.path.join(ROOT, "06-成果")

# 统计口径（与 00-项目管理/任务说明-原型与验证.md §3.1 一致）
CN_RE = re.compile(r"[\u4e00-\u9fff]")


def file_stats(path: str) -> dict:
    raw = open(path, "rb").read()
    text = io.open(path, encoding="utf-8").read()
    lines = text.splitlines()
    blank = sum(1 for ln in lines if not ln.strip())
    comment = sum(1 for ln in lines
                  if ln.strip().startswith("#") or ln.strip().startswith('"""'))
    return {
        "path": os.path.relpath(path, ROOT).replace("\\", "/"),
        "name": os.path.basename(path),
        "bytes": len(raw),
        "lines": len(lines),
        "blank_lines": blank,
        "comment_or_docstring_lines": comment,
        "code_lines": len(lines) - blank - comment,
        "cn_chars": len(CN_RE.findall(text)),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def collect(d: str, exts=(".py",)) -> list[dict]:
    out = []
    if not os.path.isdir(d):
        return out
    for dp, dn, fns in os.walk(d):
        dn[:] = [x for x in dn if x != "__pycache__"]
        for fn in sorted(fns):
            if fn.endswith(exts):
                out.append(file_stats(os.path.join(dp, fn)))
    return sorted(out, key=lambda r: r["path"])


groups = [
    ("核心源码（src/）", collect(SRC)),
    ("自动化测试（tests/）", collect(TESTS)),
    ("实验脚本（experiments/）", collect(EXPS)),
]

all_files = [f for _, g in groups for f in g]
total = {
    "files": len(all_files),
    "bytes": sum(f["bytes"] for f in all_files),
    "lines": sum(f["lines"] for f in all_files),
    "code_lines": sum(f["code_lines"] for f in all_files),
    "cn_chars": sum(f["cn_chars"] for f in all_files),
}

# 演示材料工具（**独立统计，不计入上面的原型系统口径**）
deck_files = collect(DECKTOOLS)
deck_total = {
    "files": len(deck_files),
    "bytes": sum(f["bytes"] for f in deck_files),
    "lines": sum(f["lines"] for f in deck_files),
    "code_lines": sum(f["code_lines"] for f in deck_files),
    "cn_chars": sum(f["cn_chars"] for f in deck_files),
}

# ---- 计数：从代码实际数出符号 ----
import sys
if os.path.join(ROOT, "04-原型系统") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "04-原型系统"))
import src.synth as synth, src.mce as mce
import src.fusion as fusion, src.audit as audit
api_counts = {
    "src.synth": len(getattr(synth, "__all__", [])),
    "src.mce": len(getattr(mce, "__all__", [])),
    "src.fusion": len(getattr(fusion, "__all__", [])),
    "src.audit": len(getattr(audit, "__all__", [])),
}
api_total = sum(api_counts.values())

# ---- 断言数与判据数（从文件里数出来，不手填）----
tfile = os.path.join(TESTS, "run_tests.py")
ttext = io.open(tfile, encoding="utf-8").read()
n_check = len(re.findall(r"\bcheck\(|\bexpect_raises\(", ttext))

exp_summary = {}
for fn in sorted(os.listdir(EXPS)):
    if not fn.startswith("exp") or not fn.endswith(".py"):
        continue
    txt = io.open(os.path.join(EXPS, fn), encoding="utf-8").read()
    exp_summary[fn] = {
        "n_checks": len(re.findall(r'"id":\s*"', txt)),
    }

data = {
    "generated_from": ROOT,
    "statistical_convention": {
        "cn_chars": r'len(re.findall(r"[\u4e00-\u9fff]", text))',
        "note": "只计 CJK 基本区汉字；不含标点、数字、拉丁字母、扩展区汉字",
    },
    "groups": {name: g for name, g in groups},
    "totals": total,
    "api_counts": api_counts,
    "api_total": api_total,
    "test_assertions_approx": n_check,
    "experiment_check_counts": exp_summary,
    "scope_note": ("以上 totals 只统计**原型系统本体**（src/tests/experiments）。"
                    "演示材料（PPT）的工具脚本单列于 deck_tools，不计入该口径。"),
    "deck_tools": {"path": "06-成果/空间适宜性审计工具链-产品与落地方案/tools",
                    "files": deck_files, "totals": deck_total},
}

os.makedirs(OUTDIR, exist_ok=True)
jp = os.path.join(OUTDIR, "源代码清单.json")
with io.open(jp, "w", encoding="utf-8") as fh:
    json.dump(data, fh, ensure_ascii=False, indent=2)

# ---- Markdown ----
L = []
L.append("# F4 源代码清单")
L.append("")
L.append("> **本文件由脚本生成**：`04-原型系统/experiments/gen_source_listing.py`。")
L.append("> 请勿手工编辑 —— 手工改的内容会在下次生成时丢失，且不会报错。")
L.append("> 上级事实来源：`../00-项目管理/项目总览-README.md`。")
L.append("")
L.append("## 0. 统计口径")
L.append("")
L.append("中文字符计数（与 `00-项目管理/任务说明-原型与验证.md` §3.1 一致）：")
L.append("")
L.append("```python")
L.append('n_cn = len(re.findall(r"[\\u4e00-\\u9fff]", text))')
L.append("```")
L.append("")
L.append("只计 **CJK 统一汉字基本区**（U+4E00–U+9FFF）。")
L.append("**不含**：标点（含中文标点）、数字、拉丁字母、空白、")
L.append("扩展区汉字（U+3400–U+4DBF、U+20000+）、兼容区汉字（U+F900–U+FAFF）。")
L.append("")
L.append("## 1. 总览")
L.append("")
L.append("| 项 | 值 |")
L.append("|---|---|")
L.append(f"| 文件数 | **{total['files']}** |")
L.append(f"| 总字节 | {total['bytes']:,} |")
L.append(f"| 总行数 | {total['lines']:,} |")
L.append(f"| 代码行数（非空非注释） | **{total['code_lines']:,}** |")
L.append(f"| 中文字符数（按上述口径） | **{total['cn_chars']:,}** |")
L.append(f"| 公开接口符号数 | **{api_total}** |")
L.append(f"| 单元测试断言数（近似） | **{n_check}** |")
L.append("")
L.append("### 1.1 公开接口分布")
L.append("")
L.append("| 模块 | `__all__` 符号数 |")
L.append("|---|---|")
for k, v in api_counts.items():
    L.append(f"| `{k}` | {v} |")
L.append(f"| **合计** | **{api_total}** |")
L.append("")
L.append("> 该数字是**接口基线**。任何偏离都视为接口变更，")
L.append("> 需同步更新 `03-技术方案/03-接口设计文档.md` 并登记到 `05-验证/修正记录.md`。")
L.append("")
L.append("### 1.2 实验判据数")
L.append("")
L.append("| 实验脚本 | 判据数 |")
L.append("|---|---|")
for k, v in exp_summary.items():
    L.append(f"| `{k}` | {v['n_checks']} |")
L.append("")
L.append("## 2. 逐文件明细")
L.append("")

for name, g in groups:
    L.append(f"### 2.{groups.index((name, g)) + 1} {name}")
    L.append("")
    L.append("| 文件 | 字节 | 行数 | 代码行 | 中文 | SHA-256（前 12 位） |")
    L.append("|---|---|---|---|---|---|")
    for f in g:
        L.append(f"| `{f['path'].split('/')[-1]}` | {f['bytes']:,} | "
                 f"{f['lines']} | {f['code_lines']} | {f['cn_chars']} | "
                 f"`{f['sha256'][:12]}` |")
    sub = {"files": len(g), "bytes": sum(x["bytes"] for x in g),
           "lines": sum(x["lines"] for x in g),
           "code_lines": sum(x["code_lines"] for x in g),
           "cn_chars": sum(x["cn_chars"] for x in g)}
    L.append(f"| **小计** | **{sub['bytes']:,}** | **{sub['lines']}** | "
             f"**{sub['code_lines']}** | **{sub['cn_chars']}** | — |")
    L.append("")

L.append("### 2.4 演示材料工具（**不计入上面的原型系统口径**）")
L.append("")
L.append("以下脚本用于生成 `06-成果/空间适宜性审计工具链-产品与落地方案/` 这份 PPT，")
L.append("**属于交付物但不属于原型系统本体**，故单列，不计入 §1 的 ")
L.append(f"{total['code_lines']:,} 行 / {total['files']} 文件。")
L.append("")
L.append("| 文件 | 字节 | 行数 | 代码行 | 中文 |")
L.append("|---|---|---|---|---|")
for f in deck_files:
    L.append(f"| `{f['name']}` | {f['bytes']:,} | {f['lines']} | {f['code_lines']} | {f['cn_chars']} |")
L.append(f"| **小计** | **{deck_total['bytes']:,}** | **{deck_total['lines']}** | **{deck_total['code_lines']}** | **{deck_total['cn_chars']}** |")
L.append("")
L.append(f"> **口径说明**：软著材料与结题报告中的「{total['code_lines']:,} 行 / {total['files']} 文件」指的是")
L.append("> **原型系统本体**。若把演示脚本合并统计，数字会变成")
L.append(f"> {total['code_lines'] + deck_total['code_lines']:,} 行 / {total['files'] + deck_total['files']} 文件 ——")
L.append("> 两者都正确，但**必须写明口径**，否则同一份材料里会出现两个对不上的数字。")
L.append("")
L.append("## 3. 目录结构")
L.append("")
L.append("```")
L.append("04-原型系统/")
L.append("├── 原型系统-README.md")
L.append("├── src/")
L.append("│   ├── __init__.py          ← 必须存在，否则 fusion/ 内相对导入越界")
L.append("│   ├── synth.py             合成真值场 + 池体布局 + 块平均")
L.append("│   ├── mce.py               重分类 / WLC / AHP / 秩相关")
L.append("│   ├── fusion/")
L.append("│   │   ├── __init__.py")
L.append("│   │   ├── scale.py         尺度失配诊断")
L.append("│   │   └── gain.py          误差分解 + 融合账本 + 权重扰动")
L.append("│   └── audit/")
L.append("│       ├── __init__.py")
L.append("│       └── audit.py         缺失 → 可计算后果")
L.append("├── tests/")
L.append("│   └── run_tests.py         140 条不变量断言（自写 runner）")
L.append("└── experiments/")
L.append("    ├── run_all.py           一键复现（退出码 0 = 全绿）")
L.append("    ├── exp0_structure.py    合成场结构可分辨性前提检验")
L.append("    ├── exp1_scale.py        尺度失配的代价")
L.append("    ├── exp2_fusion.py       融合的收益与代价")
L.append("    ├── exp3_audit.py        缺失 → 可计算后果")
L.append("    └── gen_source_listing.py  本清单的生成脚本")
L.append("```")
L.append("")
L.append("## 4. 复现")
L.append("")
L.append("```bash")
L.append('cd "E:/BJTU/CXCY/CXCY-PAPER/project/F4/04-原型系统"')
L.append("python experiments/run_all.py          # 退出码 0 = 全绿")
L.append("python experiments/gen_source_listing.py   # 重新生成本清单")
L.append("```")
L.append("")
L.append("**依赖**：仅 `numpy`（实测 2.5.3）。无 scipy / GDAL / rasterio / sklearn 依赖，")
L.append("可完全离线运行，无网络请求。")
L.append("")

mp = os.path.join(OUTDIR, "源代码清单.md")
io.open(mp, "w", encoding="utf-8").write("\n".join(L))

print("=" * 70)
print("源代码清单已生成")
print("=" * 70)
print(f"文件数        : {total['files']}")
print(f"总字节        : {total['bytes']:,}")
print(f"总行数        : {total['lines']:,}")
print(f"代码行        : {total['code_lines']:,}")
print(f"中文字符      : {total['cn_chars']:,}")
print(f"公开接口符号  : {api_total}  ({api_counts})")
print(f"--- 以上为原型系统本体口径 ---")
print(f"演示工具(不计入): {deck_total['files']} 文件 / {deck_total['code_lines']:,} 代码行")
print(f"单测断言(近似): {n_check}")
for k, v in exp_summary.items():
    print(f"  {k}: {v['n_checks']} 条判据")
print(f"\n已写入 {mp}")
print(f"已写入 {jp}")
