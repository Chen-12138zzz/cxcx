# -*- coding: utf-8 -*-
"""
校验器基线「不死性」自检 + 反向验证。

【为什么必须做这一步】
一条"禁止性规则"如果**永远不会触发**，它的存在毫无价值 ——
它会让使用者在 `exit=0` 时误以为"检查过了、是干净的"。
而正式运行恰恰可能"全 0 命中"，因此**必须先证明规则活着**，
才能说那个 0 是有意义的 0。

【做法】
  1. 在**工作区之外**的临时目录里放两个故意违规的探针文件。
  2. 用同一份 `baseline.json` 跑 `consistency_check.py`。
  3. 断言：每条 deprecated / banned_phrases / banned_patterns 都至少触发一次；
     authoritative 在文件缺失时必报 error。
  4. 清理探针。

【⚠️ 两个实测踩到的坑（都导致过假阴性）】
  坑 1：探针根目录**不能放在 `.tmp-tools/` 下**。
        `consistency_check.py` 的 `iter_files` 会跳过**路径中任一环节**命中
        `SKIP_DIRS` 的文件，而 `SKIP_DIRS` 含 `.tmp-tools`。
        实测把探针放那里时输出「待检查文件：0 个」，15 条规则全部表现为"未触发"。
        ⇒ 本脚本改用系统临时目录（`tempfile.gettempdir()`）。
  坑 2：核对 banned_patterns 时**不能**用 `pat["regex"] in 输出`。
        校验器打印的是**匹配到的文本**（`命中：…`），不是正则本身。
        实测该写法恒为 False，4 条正则全部假"未触发"。
        ⇒ 本脚本改为核对 `re.search(...).group(0)` 是否出现在输出中。

【运行】
    python 05-验证/check_baseline_liveness.py
    退出码 0 = 全部规则均能触发；非 0 = 有条规则是死的
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

# 本脚本位于 project/F4/05-验证/，故 F4 根 = 上两级
_HERE = os.path.dirname(os.path.abspath(__file__))
F4 = os.path.dirname(_HERE)

# 探针根目录：**必须**在工作区之外（见文件头「坑 1」）
PROBE = os.path.join(tempfile.gettempdir(), "f4_probe_root")

CHECKER = (r"C:\Users\Chen\.agents\skills\project-deliverable-workspace"
           r"\scripts\consistency_check.py")
PY = sys.executable

# ---- 探针内容：故意命中每一条规则 ----
#
# 【⚠️ 坑 3：探针字面量不能直接写在脚本里】
# 本脚本**位于工作区内**（`05-验证/`），因此它自己也在 `consistency_check.py`
# 的扫描范围内。若把违规字面量（禁用词、废弃数值）直接写成字符串常量，
# 脚本**本身**就会在正式运行中报出一堆 deprecated / info 命中 ——
# 实测把探针字面量直接写进来后，正式运行的
# 「废弃值残留」由 0 变 5、「绝对化表述」由 0 变 12。
# ⇒ 处置：把违规字面量**拆成片段再拼接**（拼接函数见下），
#   并在**会触发正则的相邻片段**处改用 `\uXXXX` 转义 ——
#   这样运行期拼出的字符串完全一样，但文件文本里既不出现完整违规词，
#   也不出现"触发词 + 后续片段"这个可被正则匹配的组合。
def _j(*parts: str) -> str:
    """把片段拼起来（片段之间无分隔），使违规字面量不出现在本文件文本中。"""
    return "".join(parts)


# 废弃值（片段化）
_DEP = [
    _j("**1.02", "2117**"),
    _j("0.92", "53"),
    _j("95.", "05"),
    _j("9.6", "6%"),
    _j("80 m ", "块平均"),
]
# 禁用词（片段化）
_BAN = [
    _j("没", "有", "人"),
    _j("无", "人研", "究"),
    _j("完全", "空白"),
    _j("尚属", "空白"),
    _j("未检索", "到同类"),
    _j("国内", "首个"),
]
# 危险正则的匹配样本（片段化；触发词之后的片段用 \u 转义，
# 否则文件文本里会出现"触发词 + 片段"这个可被正则命中的组合）
_PAT = [
    _j("均为", "\u8f93\u51fa", " 且均为", "\u4f4d\u7f6e"),
    _j("本研究为 ", "\u9996\u521b"),
    _j("文献中 仅见 ", "\u62a5\u9053"),
    _j("该方案属于 ", "\u4e1a\u754c\u9886\u5148"),
]


PROBE_MD = _j(
    "# 探针文件（故意违规，仅用于校验器不死性自检）\n\n"
    "## 1. 应命中的废弃值\n\n",
    "数值 ", _j("1.02", "2117"), " 是 A-2 的旧错误值（本行不含加粗，故只验裸串）。\n",
    "加粗形式：", _DEP[0], " 应被 deprecated 规则命中。\n",
    "factor=2 旧值 ", _DEP[1], " 应被命中。\n",
    "旧保留率 ", _DEP[2], " 与 ", _DEP[3], " 应被命中。\n",
    "旧口径 ", _DEP[4], " 应被命中。\n\n",
    "## 2. 应命中的禁用词\n\n",
    _BAN[0], "做过这个方法。该问题", _BAN[1], "。领域", _BAN[2], "且", _BAN[3], "。\n",
    "在所及范围内 ", _BAN[4], " 工作。这属于", _BAN[5], " 尝试。\n\n",
    "## 3. 应命中的危险正则\n\n",
    _PAT[0], " 一致的结论。\n",
    _PAT[1], " 的方法。\n",
    _PAT[2], " 该现象。\n",
    _PAT[3], " 水平。\n",
)

# 第二个探针：故意**不含**任何权威值，用于验证 authoritative 会报 error
PROBE_NO_AUTH = _j(
    "# 探针文件 B（不含任何权威值）\n\n"
    "本文件故意不含 ",
    _j("2.2", "91"), " / ", _j("0.99", "61"), " / ", _j("48", "9.0"), " / ",
    _j("0.10", "29"), " / ", _j("0.09", "67"), " / ", _j("0.93", "90"), " / ",
    _j("49/", "71"), " / ", _j("12/", "12"), "。\n",
)


def run(root: str, baseline: str) -> tuple[int, str]:
    r = subprocess.run([PY, "-u", CHECKER, "--root", root,
                        "--baseline", baseline, "--max-per-type", "99"],
                       capture_output=True, text=True, encoding="utf-8")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    if os.path.exists(PROBE):
        shutil.rmtree(PROBE)
    os.makedirs(PROBE, exist_ok=True)
    shutil.copy(os.path.join(F4, "baseline.json"),
                os.path.join(PROBE, "baseline.json"))
    io.open(os.path.join(PROBE, "probe_violations.md"), "w",
            encoding="utf-8").write(PROBE_MD)
    io.open(os.path.join(PROBE, "probe_no_auth.md"), "w",
            encoding="utf-8").write(PROBE_NO_AUTH)

    code, out = run(PROBE, os.path.join(PROBE, "baseline.json"))

    print("=" * 74)
    print("隔离探针运行结果")
    print("=" * 74)
    print(f"退出码：{code}（预期 1：探针 B 不含权威值，应报 error）")
    # 自检探针确实被扫描到 —— 若为 0 个，后续全部结论无效（坑 1）
    m_scan = re.search(r"待检查文件：(\d+) 个", out)
    n_scanned = int(m_scan.group(1)) if m_scan else -1
    print(f"探针被扫描文件数：{n_scanned}（应为 2；若为 0，说明探针被 SKIP_DIRS 跳过，"
          f"本次自检无效）")
    if n_scanned != 2:
        print("\n!! 探针未被扫描 ⇒ 自检无效。检查探针根目录是否落在 SKIP_DIRS 之内。")
        shutil.rmtree(PROBE, ignore_errors=True)
        return 2

    bl = json.loads(io.open(os.path.join(F4, "baseline.json"),
                            encoding="utf-8").read())

    n_fail = 0

    print("\n[A] deprecated 规则是否都能触发")
    for rule in bl["deprecated"]:
        needle = rule["old"]
        expect = needle.strip("*")
        fired = (expect in PROBE_MD) and (needle in out or expect in out)
        if not fired:
            n_fail += 1
        print(f"  {'OK  ' if fired else '*** 未触发 ***'} {needle!r}")

    print("\n[B] banned_phrases 是否都能触发")
    for ph in bl["banned_phrases"]:
        fired = (ph in PROBE_MD) and (ph in out)
        if not fired:
            n_fail += 1
        print(f"  {'OK  ' if fired else '*** 未触发 ***'} {ph!r}")

    print("\n[C] banned_patterns 是否都能触发")
    for pat in bl["banned_patterns"]:
        rx = re.compile(pat["regex"])
        sample = rx.search(PROBE_MD)
        if not sample:
            print(f"  SKIP  {pat['regex'][:44]!r}  探针无对应样本")
            continue
        # 见文件头「坑 2」：核对**匹配文本**，不是正则本身
        matched = sample.group(0)
        fired = matched in out
        if not fired:
            n_fail += 1
        print(f"  {'OK  ' if fired else '*** 未触发 ***'} 匹配文本 {matched!r}")

    print("\n[D] authoritative 缺位是否报 error")
    m = re.search(r"权威值缺位 (\d+) 处", out)
    n_err = int(m.group(1)) if m else -1
    fired = n_err > 0
    if not fired:
        n_fail += 1
    print(f"  {'OK  ' if fired else '*** 未触发 ***'} 权威值缺位 {n_err} 处（预期 >0）")

    n_rules = (len(bl["deprecated"]) + len(bl["banned_phrases"])
               + len(bl["banned_patterns"]) + 1)
    print("\n" + "=" * 74)
    if n_fail == 0:
        print(f"不死性自检：通过 —— {n_rules} 条规则均能触发")
        print("⇒ 正式运行的 0 命中是【有意义的 0】，不是死规则造成的假阴性。")
    else:
        print(f"不死性自检：{n_fail} 条规则未能触发（共 {n_rules} 条）")
    print("=" * 74)

    shutil.rmtree(PROBE, ignore_errors=True)
    print(f"\n已清理探针目录：{PROBE}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
