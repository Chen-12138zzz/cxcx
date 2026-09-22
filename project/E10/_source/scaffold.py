# -*- coding: utf-8 -*-
"""E10 项目骨架生成器（幂等：已存在的文件不覆盖）"""
import os, io, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIRS = [
    "00-项目管理",
    "01-调研",
    "02-申报材料",
    "03-技术方案",
    "04-原型系统",
    "05-验证",
    "06-成果",
    "06-成果/演示截图",
    "07-结题",
    "08-复盘",
]

# (相对路径, 初始内容)  —— 只创建空壳，不写内容；内容由后续步骤逐个填写
PLACEHOLDERS = [
    ("README.md", None),
    ("baseline.json", None),
    ("01-调研/01-竞品与近邻尽调.md", None),
    ("01-调研/02-问题调研与选题论证.md", None),
    ("01-调研/03-参考文献清单.md", None),
    ("02-申报材料/项目申报书.md", None),
    ("02-申报材料/立项论证.md", None),
    ("03-技术方案/系统设计说明书.md", None),
    ("03-技术方案/核心算法设计说明.md", None),
    ("03-技术方案/接口设计文档.md", None),
    ("04-原型系统/原型系统-README.md", None),
    ("05-验证/实验与验证报告.md", None),
    ("06-成果/论文初稿.md", None),
    ("06-成果/软件著作权申请材料.md", None),
    ("06-成果/汇报PPT大纲.md", None),
    ("07-结题/中期检查报告.md", None),
    ("07-结题/结题报告.md", None),
    ("08-复盘/复盘报告.md", None),
]

TMP_MARK = "<!-- 占位：待填写 -->\n"

def main():
    made_d, made_f = [], []
    for d in DIRS:
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            os.makedirs(p, exist_ok=True)
            made_d.append(d)
    for rel, content in PLACEHOLDERS:
        p = os.path.join(ROOT, rel.replace("/", os.sep))
        if not os.path.exists(p):
            with io.open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(content if content is not None else TMP_MARK)
            made_f.append(rel)
    print("新建目录:", len(made_d))
    for d in made_d: print("  +", d)
    print("新建文件:", len(made_f))
    for f in made_f: print("  +", f)

if __name__ == "__main__":
    main()
