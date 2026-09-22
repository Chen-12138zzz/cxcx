# -*- coding: utf-8 -*-
"""16 方向审查工作区脚手架（幂等）"""
import os, io

ROOT = os.path.join(r"E:\BJTU\CXCY\CXCY-PAPER", "project", "16-方向审查")

DIRS = [
    "00-总览",
    "01-D1-智能投喂与摄食感知",
    "02-D2-水下图像增强与目标检测",
    "03-D3-水质与环境时序预测预警",
    "04-D4-水产病害智能诊断",
    "05-D5-边缘智能与轻量化部署",
    "06-D6-多模态大模型与渔业知识",
    "07-E1-物联网感知与传感网络",
    "08-E2-数字孪生与决策支持平台",
    "09-E3-养殖数据质量治理",
    "10-E4-水下机器人与网箱巡检",
    "11-E5-精准投喂装备与机电控制",
    "12-E6-模型轻量化与边缘部署",
    "13-E7-水产品溯源与供应链",
    "14-E8-无人机与遥感工程应用",
    "15-E9-水下声学感知与通信",
    "16-E10-养殖AI可信性工程",
]

GATE_FILES = ["门禁A-竞品与近邻尽调.md", "门禁B-能力差异检验.md", "门禁C-否定性举证与表述纪律.md"]
TMP = "<!-- 占位：待填写 -->\n"

def main():
    md = []
    for d in DIRS:
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):
            os.makedirs(p, exist_ok=True); md.append(d)
    mf = []
    # 根 README
    for rel in ["README.md", "baseline.json"]:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            io.open(p, "w", encoding="utf-8", newline="\n").write(TMP); mf.append(rel)
    # 每个方向三门禁
    for d in DIRS[1:]:
        for g in GATE_FILES:
            p = os.path.join(ROOT, d, g)
            if not os.path.exists(p):
                io.open(p, "w", encoding="utf-8", newline="\n").write(TMP); mf.append(d + "/" + g)
    # 总览
    for rel in ["00-总览/16方向审查总报告.md", "00-总览/判定汇总表.md", "00-总览/改进方案与重组建议.md"]:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            io.open(p, "w", encoding="utf-8", newline="\n").write(TMP); mf.append(rel)
    print("新建目录:", len(md))
    print("新建文件:", len(mf))

if __name__ == "__main__":
    main()
