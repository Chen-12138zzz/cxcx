# -*- coding: utf-8 -*-
"""生成工程落地向交付物：doc/工程落地补充/ 下分方向清单 + 总表 + CSV + BibTeX + 工作区根补充说明。"""
import json, os, re, csv

BASE = r"E:\BJTU\CXCY\CXCY-PAPER"
OUT = os.path.join(BASE, "doc", "工程落地补充")
os.makedirs(OUT, exist_ok=True)

RAW = json.load(open(os.path.join(BASE, "tools", "_eng_raw.json"), encoding="utf-8"))
STORE = RAW["records"]
SEL = json.load(open(os.path.join(BASE, "tools", "_eng_selected.json"), encoding="utf-8"))
SUMS = json.load(open(os.path.join(BASE, "tools", "_eng_summary.json"), encoding="utf-8"))
stats = SUMS["stats"]
T = SUMS["total"]

QUERIES = json.load(open(os.path.join(BASE, "tools", "eng_search.py"), encoding="utf-8")
                   ) if False else None

DIRS = [(s["did"], s["name"], s["support"]) for s in stats]

QUERY_TEXT = {
    "E1": '"wireless sensor network aquaculture monitoring system"、"low power IoT sensor node water quality"、"LoRa aquaculture monitoring system design"、"sensor network fish farm real-time monitoring"、"energy harvesting wireless sensor aquaculture"、"underwater wireless sensor network deployment"',
    "E2": '"digital twin aquaculture system"、"decision support system aquaculture farm management"、"smart aquaculture platform architecture design"、"fishery information system design implementation"、"precipitation aquaculture management information platform"',
    "E3": '"missing data imputation water quality time series"、"anomaly detection sensor data water monitoring"、"data quality sensor network environmental monitoring"、"outlier detection environmental time series deep learning"、"sensor fault detection water quality monitoring system"',
    "E4": '"underwater robot inspection aquaculture net cage"、"net cleaning robot aquaculture"、"ROV aquaculture inspection system"、"autonomous underwater vehicle fish farm inspection"、"biofouling removal net cage robot design"',
    "E5": '"automatic feeding machine aquaculture control system"、"feeding robot fish farm design"、"pneumatic feeding system fish cage"、"precision feeding equipment aquaculture development"、"automatic feeder control system design aquaculture"',
    "E6": '"knowledge distillation lightweight model edge deployment"、"model quantization pruning real-time inference embedded"、"edge inference optimization deep learning deployment"、"TinyML microcontroller deep learning deployment"、"neural network acceleration edge device aquaculture"',
    "E7": '"seafood traceability blockchain system"、"aquaculture supply chain traceability system design"、"food traceability IoT blockchain implementation"、"digital traceability fishery product supply chain"',
    "E8": '"UAV remote sensing aquaculture mapping"、"drone imagery fish pond monitoring"、"satellite remote sensing aquaculture site selection"、"unmanned aerial vehicle water quality monitoring"、"remote sensing aquaculture area extraction mapping"',
    "E9": '"underwater acoustic communication network"、"hydroacoustic fish biomass estimation"、"acoustic telemetry fish behavior monitoring"、"underwater acoustic sensor localization"、"fishery acoustics sonar survey method"',
    "E10": '"machine learning model monitoring drift detection production"、"MLOps deployment pipeline monitoring"、"large language model hallucination detection mitigation"、"explainable AI trustworthy machine learning deployment"、"uncertainty quantification deep learning deployment"',
}

POS = {
    "E1": "把「感知」做成可部署的硬件 + 组网工程：传感器选型、低功耗节点、LoRa/NB-IoT 组网、网关与数据回传。这是所有现场系统的地基，也是最容易做出「看得见」成果的一层。",
    "E2": "数据底座与决策支持平台：数据模型、系统架构、看板与台账、决策流程编排。软件工程学生的天然主场。",
    "E3": "数据质量治理：缺失值插补、异常/故障检测、传感器漂移校准。任何现场部署都会先撞上这堵墙——数据是脏的。",
    "E4": "水下机器人与网箱巡检装备：ROV/AUV、网衣清洗、生物附着去除、自主巡检路径。",
    "E5": "精准投喂执行装备与机电控制：投饵机结构、气力输送、计量与执行机构、控制回路。",
    "E6": "模型轻量化与边缘部署工程：蒸馏、量化、剪枝、TinyML、NPU/FPGA 加速、端侧推理流水线。",
    "E7": "水产品溯源与供应链可信数据：区块链存证、来源认证、数字护照、防篡改。",
    "E8": "无人机与遥感工程应用：多光谱/高光谱巡检、养殖区提取与选址、水面遥感反演。",
    "E9": "水下声学感知与通信：水声通信组网、声学鱼群量估计、声学行为遥测。",
    "E10": "AI 系统可信性工程：生产环境模型监控、数据漂移检测、可靠性与退化预警、大模型幻觉检测与抑制、可解释性与不确定性量化。",
}
RISK = {
    "E1": "国内智慧渔业边缘网关/AI 盒子已是成熟商品，纯硬件组网易被判为集成复现。",
    "E2": "容易被评价为「又一个管理平台」，必须绑定独特数据或独特任务，否则无差异化。",
    "E3": "若数据全部来自公开平台，「脏数据」问题不够真实，说服力下降。",
    "E4": "ROV/水下装备单价常在数万元级，1 万元预算难以覆盖；队伍无机械/水动背景。",
    "E5": "需要机械设计与加工能力，纯计算机系队伍吃紧；且投喂机是成熟工业品。",
    "E6": "属工程集成而非学术创新，宜作为其它方向的落地层组合申报。",
    "E7": "与养殖现场距离远，难以拿到真实供应链数据，易做成「纸面系统」。",
    "E8": "需要飞行与遥感处理能力，且业务化遥感服务已存在，差异化难。",
    "E9": "声学硬件门槛与成本高，水声通信实验条件（水池/海上）难获得。",
    "E10": "偏通用方法，必须绑定养殖场景（如预警文本的事实核查）才有应用价值，否则答辩讲不清。",
}


def safe(s):
    return re.sub(r'[\\/:*?"<>|]', '-', s).strip()


def au(r, n=3):
    a = r["authors"]
    if not a:
        return "佚名"
    return ", ".join(a[:n]) + (", et al." if r["n_authors"] > n else "")


def org(r):
    return "国内" if r["cn_first"] else ("国内(合作)" if r["cn"] else "国外")


def engf(t):
    t = (t or "").lower()
    K = ["system", "platform", "framework", "design and implementation", "implementation",
         "prototype", "deploy", "field trial", "field test", "case study", "application",
         "architecture", "pipeline", "tool", "device", "instrument", "apparatus",
         "practical", "industrial", "engineering", "integration", "real-world", "in-situ",
         "on-site", "experiment", "development", "construction", "monitoring station",
         "testbed", "setup", "equipment", "automatic", "control", "smart", "low-cost",
         "cost-effective"]
    return any(k in t for k in K)


def main():
    lines_all, csv_rows, bib, used = [], [], [], set()

    for did, dname, dsup in DIRS:
        recs = [STORE[i] for i in SEL[did]]
        n = len(recs)
        r3 = sum(1 for r in recs if (r["year"] or 0) >= 2024)
        cn = sum(1 for r in recs if r["cn"])
        cnf = sum(1 for r in recs if r["cn_first"])
        eng = sum(1 for r in recs if engf(r["title"]))
        L = [f"# {did} {dname}", "",
             f"> **{dsup}** ｜ 入选 **{n}** 篇 ｜ 近三年 **{r3}**（{r3/n:.0%}）｜ 含中国机构 **{cn}**（{cn/n:.0%}）"
             f" ｜ 中国第一作者 **{cnf}**（{cnf/n:.0%}）｜ 含「工程落地/V类」标题特征 **{eng}** 篇（{eng/n:.0%}）", "",
             f"**方向定位**：{POS[did]}", "",
             f"**主要风险**：{RISK[did]}", "",
             f"**检索式**：{QUERY_TEXT[did]}", "",
             "**文献清单**", ""]
        for k, r in enumerate(recs, 1):
            tag = "工程落地" if engf(r["title"]) else "方法/分析"
            L.append(f"{k}. **{r['title']}**")
            L.append(f"   - 作者：{au(r)} ｜ 出处：{r['venue'] or '（未标注出处）'} ｜ 年份：{r['year']} "
                     f"｜ 国别：{org(r)} ｜ 被引：{r['cited']} ｜ 类型：{tag}")
            L.append(f"   - DOI：{r['doi'] or '（无）'}")
            csv_rows.append(dict(方向编号=did, 方向=dname, 支撑关系=dsup, 序号=k, 标题=r["title"],
                                 作者=au(r, 99), 出处=r["venue"], 年份=r["year"], 国别=org(r),
                                 被引=r["cited"], 类型=tag, DOI=r["doi"],
                                 OpenAlexID=(r["id"] or "").split("/")[-1]))
            key = re.sub(r"[^A-Za-z]", "", (r["authors"][0].split()[-1] if r["authors"] else "anon")) + str(r["year"] or "") + str(k)
            while key in used:
                key += "x"
            used.add(key)
            et = "inproceedings" if r["type"] == "proceedings-article" else "article"
            bib.append(f"@{et}{{{key},\n  title   = {{{r['title']}}},\n  author  = {{{' and '.join(r['authors'])}}},\n"
                       f"  journal = {{{r['venue']}}},\n  year    = {{{r['year']}}},\n  doi     = {{{r['doi']}}}\n}}\n")
        L += ["", "---", "",
              "> 由 `tools/eng_gen.py` 自动生成；全部条目来自 OpenAlex 真实元数据，未做任何人工编造。引用前请以 DOI 原文复核。", ""]
        open(os.path.join(OUT, safe(f"{did}-{dname}") + ".md"), "w", encoding="utf-8").write("\n".join(L))

        lines_all += [f"## {did} {dname}（{n} 篇｜{dsup}）", ""]
        for k, r in enumerate(recs, 1):
            lines_all.append(f"{k}. {r['title']}（{au(r)}. {r['venue']}. {r['year']}. DOI: {r['doi'] or '—'}）")
        lines_all.append("")

    head = ["# 工程落地向文献总表（新增第二批）", "",
            "**数据源**：OpenAlex 开放学术图谱（api.openalex.org），检索日期 2026-09-20。",
            "**与第一批的关系**：已按 OpenAlex ID 剔除第一批 133 篇，本表**全部为新增文献**。", "",
            f"**新增合计 {T['n']} 篇** ｜ 近三年（2024 年起）**{T['rec']}**（**{T['rec']/T['n']:.1%}**）"
            f" ｜ 含中国机构 **{T['cn']}**（**{T['cn']/T['n']:.1%}**）"
            f" ｜ 中国第一作者 **{T['cnf']}**（**{T['cnf']/T['n']:.1%}**）", ""]
    open(os.path.join(OUT, "文献总表.md"), "w", encoding="utf-8").write("\n".join(head + lines_all))

    with open(os.path.join(OUT, "文献总表.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["方向编号", "方向", "支撑关系", "序号", "标题", "作者", "出处",
                                          "年份", "国别", "被引", "类型", "DOI", "OpenAlexID"])
        w.writeheader()
        w.writerows(csv_rows)

    open(os.path.join(OUT, "references-eng.bib"), "w", encoding="utf-8").write("\n".join(bib))
    print("doc/工程落地补充/ 生成完毕")
    for s in stats:
        print(f"  {s['did']} {s['name']}: {s['n']} 篇 | 2024+ {s['rec']} ({s['rec']/max(1,s['n']):.0%}) | 国内 {s['cn']} ({s['cn']/max(1,s['n']):.0%}) | 工程落地 {s['eng']} ({s['eng']/max(1,s['n']):.0%})")
    print(f"  合计 {T['n']} | 近三年 {T['rec']} ({T['rec']/T['n']:.1%}) | 国内 {T['cn']} ({T['cn']/T['n']:.1%}) | 国内一作 {T['cnf']} ({T['cnf']/T['n']:.1%}) | 工程落地 {T['eng']} ({T['eng']/T['n']:.1%})")


if __name__ == "__main__":
    main()
