# -*- coding: utf-8 -*-
"""生成交付文件：doc/ 下的分方向文献清单 + 总表 + CSV + BibTeX，以及工作区根目录的选题 md。"""
import json, os, re, csv

BASE = r"E:\BJTU\CXCY\CXCY-PAPER"
DOC = os.path.join(BASE, "doc")
os.makedirs(DOC, exist_ok=True)

RAW = json.load(open(os.path.join(BASE, "tools", "_papers_raw.json"), encoding="utf-8"))
STORE = RAW["records"]
SEL = json.load(open(os.path.join(BASE, "tools", "_selected.json"), encoding="utf-8"))

DIRS = [
    ("D1", "水产养殖智能投喂与摄食行为智能感知"),
    ("D2", "水下图像增强与水下目标检测识别"),
    ("D3", "养殖水质与环境时序预测预警"),
    ("D4", "水产动物病害智能诊断与健康评估"),
    ("D5", "边缘智能与轻量化模型部署（养殖物联网）"),
    ("D6", "多模态大模型与海洋渔业知识智能"),
]

DIR_QUERIES = {
    "D1": '"fish feeding behavior recognition computer vision"、"intelligent feeding system aquaculture deep learning"、"appetite detection fish acoustic"、"shrimp feeding behavior detection"、"feeding intensity detection aquaculture"',
    "D2": '"underwater image enhancement deep learning"、"underwater object detection deep learning"、"degraded underwater image restoration"、"fish detection underwater video"',
    "D3": '"water quality prediction machine learning aquaculture"、"dissolved oxygen prediction model pond"、"harmful algal bloom early warning deep learning"、"aquaculture water quality monitoring IoT prediction"',
    "D4": '"fish disease detection deep learning"、"shrimp disease recognition image classification"、"aquatic animal disease diagnosis machine learning"、"fish disease detection YOLO"、"fish disease classification convolutional neural network"、"aquaculture fish anomaly detection machine vision"、"marine fish disease recognition deep learning"、"shrimp disease detection machine learning image"',
    "D5": '"edge computing smart aquaculture"、"lightweight deep learning model deployment aquaculture"、"embedded AI fish detection edge device"、"IoT aquaculture monitoring system design"、"aquaculture edge computing lightweight model deployment"、"real-time fish detection embedded device"',
    "D6": '"large language model marine science"、"multimodal large language model aquaculture"、"vision language model fish recognition"、"knowledge graph aquaculture fishery"、"ocean large language model domain adaptation"、"retrieval augmented generation domain knowledge water"、"large language model aquaculture fish farming"、"multimodal large language model underwater"、"knowledge graph fish disease aquaculture"、"foundation model marine remote sensing"',
}

DIR_NOTE = {
    "D1": "本质是「视频/音频信号 → 摄食强度 → 投喂决策」的软件流水线，可全程离线开发；数据既可自采（料罾俯拍），也可先用公开鱼类行为数据集起步。",
    "D2": "纯算法方向，公开数据集充足（UIEB、URPC、Brackish 等），不依赖任何硬件即可完成；但赛道极为拥挤，需警惕「再提一个增强网络」式的微创新。",
    "D3": "时序预测、异常检测、可解释性建模，是软件工程学生的舒适区；公开水质数据平台可直接取数。注意：与早期「赤潮预警」类选题存在重叠风险。",
    "D4": "图像分类/检测/分割任务，公开病害数据集较多，容易做出可演示系统；白斑病等病害是湛江对虾养殖的真实主要风险。",
    "D5": "模型压缩、推理优化、端云协同、系统集成——最贴近软件工程专业能力；但单独作为选题偏工程集成，学术创新点较弱，更适合作为其它方向的落地层。",
    "D6": "RAG、多模态对齐、知识图谱、领域大模型微调，都是可快速上手的工程化课题；本方向文献相对少且集中出现在 2024 年之后。",
}

DIR_RISK = {
    "D1": "需要有真实塘口或可信公开数据；若既无数据也无设备，只能停留在公开数据集复现。",
    "D2": "创新性门槛高，评审容易质疑「与已有网络差别在哪」。",
    "D3": "若与上一轮失败的赤潮预警项目同构，必须先做竞品尽调与能力差异论证，不能再用「文献空白」立项。",
    "D4": "病害图像获取与标注成本高；同类工作密集。",
    "D5": "单独立项创新性不足，建议与 D1/D4 组合。",
    "D6": "生成内容的可信性与幻觉是公开难题（见幻觉综述）；算力需求需评估。文献量少≠空白，禁止把「检索没找到」写成「没人做过」。",
}


def au(r, n=3):
    a = r["authors"]
    if not a:
        return "佚名"
    s = ", ".join(a[:n])
    return s + (", et al." if r["n_authors"] > n else "")


def org(r):
    return "国内" if r["cn_first"] else ("国内(合作)" if r["cn"] else "国外")


def stat(recs):
    n = len(recs)
    r3 = sum(1 for x in recs if (x["year"] or 0) >= 2024)
    cn = sum(1 for x in recs if x["cn"])
    cnf = sum(1 for x in recs if x["cn_first"])
    return n, r3, cn, cnf


def main():
    lines_all, csv_rows, bib = [], [], []
    stats = []
    used_bibkey = set()

    for did, dname in DIRS:
        recs = [STORE[i] for i in SEL[did]]
        n, r3, cn, cnf = stat(recs)
        stats.append((did, dname, n, r3, cn, cnf))
        L = [f"# {did} {dname}", "",
             f"> 入选 **{n}** 篇 ｜ 近三年（2024 年及以后）**{r3}** 篇（{r3/n:.0%}）｜ "
             f"含中国机构 **{cn}** 篇（{cn/n:.0%}）｜ 中国第一作者 **{cnf}** 篇（{cnf/n:.0%}）", "",
             f"**方向定位**：{DIR_NOTE[did]}", "",
             f"**主要风险**：{DIR_RISK[did]}", "",
             "**检索式（OpenAlex，2021 年起）**：" + DIR_QUERIES[did], "",
             "**文献清单**"
             "", ""]
        for k, r in enumerate(recs, 1):
            L.append(f"{k}. **{r['title']}**")
            L.append(f"   - 作者：{au(r)} ｜ 出处：{r['venue'] or '（OpenAlex 未标注出处）'} ｜ 年份：{r['year']} "
                     f"｜ 国别：{org(r)} ｜ 被引：{r['cited']}")
            L.append(f"   - DOI：{r['doi'] or '（无）'}")
            csv_rows.append(dict(方向编号=did, 方向=dname, 序号=k, 标题=r["title"],
                                 作者=au(r, 99), 出处=r["venue"], 年份=r["year"],
                                 国别=org(r), 被引=r["cited"], DOI=r["doi"],
                                 OpenAlexID=(r["id"] or "").split("/")[-1]))
            key = re.sub(r"[^A-Za-z]", "", (r["authors"][0].split()[-1] if r["authors"] else "anon")) + str(r["year"] or "") + str(k)
            while key in used_bibkey:
                key += "x"
            used_bibkey.add(key)
            et = "article" if r["type"] != "proceedings-article" else "inproceedings"
            bib.append(f"@{et}{{{key},\n  title   = {{{r['title']}}},\n  author  = {{{' and '.join(r['authors'])}}},\n"
                       f"  journal = {{{r['venue']}}},\n  year    = {{{r['year']}}},\n  doi     = {{{r['doi']}}}\n}}\n")
        L.append("")
        L.append("---")
        L.append("")
        L.append("> 本文件由 `tools/gen_outputs.py` 自动生成；全部条目来自 OpenAlex 开放学术图谱的真实元数据，"
                 "未做任何人工编造。引用前请以 DOI 原文复核。")
        open(os.path.join(DOC, f"{did}-{dname}.md"), "w", encoding="utf-8").write("\n".join(L))

        lines_all.append(f"## {did} {dname}（{n} 篇）")
        lines_all.append("")
        for k, r in enumerate(recs, 1):
            lines_all.append(f"{k}. {r['title']}（{au(r)}. {r['venue']}. {r['year']}. DOI: {r['doi'] or '—'}）")
        lines_all.append("")

    # 总表 md
    T = dict(n=sum(s[2] for s in stats), r3=sum(s[3] for s in stats),
             cn=sum(s[4] for s in stats), cnf=sum(s[5] for s in stats))
    head = ["# 大创选题文献总表（按方向分组）", "",
            "**文献来源**：OpenAlex 开放学术图谱（api.openalex.org），检索日期 2026-09-20。",
            "**入选规则**：方向相关性关键词门禁 → 按「近三年 / 国内」优先配额筛选 → 标题去重。",
            "",
            f"**合计 {T['n']} 篇** ｜ 近三年（2024 年起）**{T['r3']}** 篇（**{T['r3']/T['n']:.1%}**）"
            f" ｜ 含中国机构 **{T['cn']}** 篇（**{T['cn']/T['n']:.1%}**）"
            f" ｜ 中国第一作者 **{T['cnf']}** 篇（**{T['cnf']/T['n']:.1%}**）", ""]
    open(os.path.join(DOC, "参考文献总表.md"), "w", encoding="utf-8").write(
        "\n".join(head + lines_all))

    # CSV
    with open(os.path.join(DOC, "参考文献总表.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["方向编号", "方向", "序号", "标题", "作者", "出处",
                                          "年份", "国别", "被引", "DOI", "OpenAlexID"])
        w.writeheader()
        w.writerows(csv_rows)

    # BibTeX
    open(os.path.join(DOC, "references.bib"), "w", encoding="utf-8").write("\n".join(bib))

    print("生成完毕。")
    for s in stats:
        print(f"  {s[0]} {s[1]}: {s[2]} 篇 | 近三年 {s[3]} ({s[3]/s[2]:.0%}) | 国内 {s[4]} ({s[4]/s[2]:.0%}) | 国内一作 {s[5]} ({s[5]/s[2]:.0%})")
    print(f"  合计 {T['n']} | 近三年 {T['r3']} ({T['r3']/T['n']:.1%}) | 国内 {T['cn']} ({T['cn']/T['n']:.1%}) | 国内一作 {T['cnf']} ({T['cnf']/T['n']:.1%})")

    json.dump({"stats": stats, "total": T,
               "dirs": {d: [dict(seq=k + 1, title=r["title"], year=r["year"], venue=r["venue"],
                                 doi=r["doi"], org=org(r), au=au(r))
                            for k, r in enumerate([STORE[i] for i in SEL[d]])]
                        for d, _ in DIRS}},
              open(os.path.join(BASE, "tools", "_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
