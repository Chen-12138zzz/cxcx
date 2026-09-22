# -*- coding: utf-8 -*-
"""
选题文献：相关性门禁 + 配额配平 + 交付文件生成。
目标：总数 >= 100；近三年(2024+) >= 70%；国内(含中国机构) >= 50% 且 国内第一作者 >= 50%。
相关性：每条须通过该方向的关键词门禁（标题命中），杜绝"铁路CV""脑肿瘤检测"类噪声。
"""
import json, os, re

BASE = r"E:\BJTU\CXCY\CXCY-PAPER"
DOC = os.path.join(BASE, "doc")
os.makedirs(DOC, exist_ok=True)

RAW = json.load(open(os.path.join(BASE, "tools", "_papers_raw.json"), encoding="utf-8"))
STORE, PER_DIR = RAW["records"], RAW["per_dir"]

DIR_NAMES = {
    "D1": "水产养殖智能投喂与摄食行为智能感知",
    "D2": "水下图像增强与水下目标检测识别",
    "D3": "养殖水质与环境时序预测预警",
    "D4": "水产动物病害智能诊断与健康评估",
    "D5": "边缘智能与轻量化模型部署（养殖物联网）",
    "D6": "多模态大模型与海洋渔业知识智能",
}

# ---------------- 相关性门禁 ----------------
AQ = ["fish", "shrimp", "prawn", "aquacultur", "aquatic", "crab", "oyster", "mollusc",
      "shellfish", "tilapia", "salmon", "trout", "seabream", "sea bass", "seabass",
      "lobster", "sea cucumber", "seaweed", "kelp", "fishery", "fisheries", "pond",
      "cage", "mariculture", "hatchery", "shoal", "marine", "ocean"]

GATE = {
    # D1：投喂 / 摄食 / 行为 / 养殖本体
    "D1": [["feeding", "feed", "appetite", "satiet", "satiety", "satiated", "hunger",
            "food intake", "ingest", "diet", "pellet", "ration", "feeding intensity",
            "feeding decision", "behaviour", "behavior", "shoal", "schooling",
            "swarm", "activity", "biomass", "growth", "stocking", "aquacultur",
            "fish farm", "smart fishery", "intelligent fishery"]],
    "D2": [["underwater", "subsea", "sub-aqua", "sonar image", "turbid", "dehaz",
            "water image", "marine image", "ocean image", "underwater vision"]],
    "D3": [["water quality", "dissolved oxygen", "algal bloom", "algae bloom",
            "cyanobacter", "eutroph", "ammonia nitrogen", "chlorophyll", "hypoxi",
            "deoxygen", "pond water", "water environment", "water quality index",
            "aquaculture water", "wastewater", "effluent", "water parameter",
            "coastal water", "reservoir", "lake", "water temperature", "salinity"]],
    "D4": [["disease", "lesion", "pathogen", "virus", "viral", "antiviral", "bacteri",
            "parasit", "symptom", "diagnos", "infection", "ulcer", "wound", "sick",
            "abnormal", "deform", "mortalit"],
           AQ],
    "D5": [["edge", "lightweight", "light-weight", "embedded", "iot", "internet of things",
            "tinyml", "microcontroller", "quantiz", "pruning", "distillation",
            "real-time", "deploy", "low-power", "energy-efficient", "mcu",
            "on-device", "mobile device", "smart farm", "smart agriculture"],
           ["aquacultur", "fisher", "fish", "shrimp", "agricultur", "farm", "marine",
            "ocean", "pond", "crop", "livestock", "greenhouse"]],
    "D6": [["large language model", "llm", "multimodal", "multi-modal", "vision-language",
            "vision language", "knowledge graph", "foundation model", "generative ai",
            "gpt", "agent", "question answering", "retrieval-augmented", "rag",
            "vision transformer", "clip"],
           ["marine", "ocean", "aquacultur", "fisher", "fish", "shrimp", "water",
            "coastal", "maritime", "sea ", "seafood", "aquatic", "fishery"]],
}

QUOTA = dict(r_cnf=15, r_cn=3, r_in=4, o_cn=3, o_in=2)

# 明确的跨领域噪声（标题命中即剔除），与选题方向无关
EXCLUDE = ["carotenoid", "metaverse", "railway", "orb-slam", "text data augmentation",
           "optical metrology", "brain tumor", "parkinson", "alzheimer", "pothole",
           "spatial transcriptomics", "cholinergic", "protein structure prediction",
           "alphafold", "microplastic", "crop yield", "cotton", "maize", "apple disease",
           "plant disease", "shelf life", "heavy metal", "uav pest", "marketing",
           "soil", "greenhouse gas", "transcriptom", "antibiotic", "antimicrob",
           "resistance genes", "breast cancer", "tumor", "artificial skin",
           "inundated areas", "sri lanka", "haptic", "walrus", "metaheuristic",
           "apelin", "physiological response"]

# D4 只保留"计算/AI 侧"的病害诊断文献：剔除分子生物学与湿实验检测
EXCLUDE_D4 = ["crispr", "rpa", "pfa", "lamp", "assay", "genom", "transposon",
              "metabolom", "metabolite", "gene ", "genes", "protein", "rna",
              "lipid", "immune", "pathway", "nf-", "stress-induced", "escrts",
              "nanoparticle", "phage", "biocontrol", "virulence", "pathogenesis",
              "exoskeleton", "recombinase", "fluorescent", "spatial-omics",
              "apoptos", "infection and", "host ", "antiviral", "vaccine", "mrna",
              "bacterioc", "antibiotic", "antimicrob", "resistance genes",
              "microbiome", "dna extraction", "cancer", "tumor", "breast"]


def tkey(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:70]


def gate_ok(did, title):
    t = " " + (title or "").lower() + " "
    if any(k in t for k in EXCLUDE):
        return False
    if did == "D4" and any(k in t for k in EXCLUDE_D4):
        return False
    for group in GATE[did]:
        if not any(k in t for k in group):
            return False
    return True


def select(did, ids):
    seen, uniq = set(), []
    for i in ids:
        r = STORE[i]
        k = tkey(r["title"])
        if not k or k in seen:
            continue
        if not gate_ok(did, r["title"]):
            continue
        seen.add(k)
        uniq.append(r)

    rec = [r for r in uniq if (r["year"] or 0) >= 2024]
    old = [r for r in uniq if (r["year"] or 0) < 2024]
    r_cnf = [r for r in rec if r["cn_first"]]
    r_cn = [r for r in rec if r["cn"] and not r["cn_first"]]
    r_in = [r for r in rec if not r["cn"]]
    o_cn = sorted([r for r in old if r["cn"]], key=lambda x: -x["cited"])
    o_in = sorted([r for r in old if not r["cn"]], key=lambda x: -x["cited"])

    picked, s2 = [], set()
    for r in (r_cnf[:QUOTA["r_cnf"]] + r_cn[:QUOTA["r_cn"]] + r_in[:QUOTA["r_in"]]
              + o_cn[:QUOTA["o_cn"]] + o_in[:QUOTA["o_in"]]):
        k = tkey(r["title"])
        if k in s2:
            continue
        s2.add(k)
        picked.append(r)
    picked.sort(key=lambda x: (-(x["year"] or 0), -x["cited"]))
    return picked, len(uniq)


def authors_str(r):
    a = r["authors"]
    if not a:
        return "佚名"
    return ", ".join(a[:3]) + (", et al." if r["n_authors"] > 3 else "")


def origin(r):
    if r["cn_first"]:
        return "国内"
    if r["cn"]:
        return "国内(合作)"
    return "国外"


def main():
    sel_all, stat = {}, []
    for did in DIR_NAMES:
        s, pool = select(did, PER_DIR[did])
        sel_all[did] = s
        n = len(s)
        stat.append(dict(did=did, name=DIR_NAMES[did], n=n, pool=pool,
                         rec=sum(1 for r in s if (r["year"] or 0) >= 2024),
                         cn=sum(1 for r in s if r["cn"]),
                         cnf=sum(1 for r in s if r["cn_first"])))

    with open(os.path.join(BASE, "tools", "_review.txt"), "w", encoding="utf-8") as f:
        for did in DIR_NAMES:
            f.write(f"\n##### {did} {DIR_NAMES[did]} #####\n")
            for i, r in enumerate(sel_all[did], 1):
                f.write(f"{i:2d}|{r['year']}|{origin(r)}|{r['cited']:5d}| {r['title']}\n")

    json.dump({d: [r["id"] for r in v] for d, v in sel_all.items()},
              open(os.path.join(BASE, "tools", "_selected.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    for s in stat:
        print(f"{s['did']} {s['name']}: 入选 {s['n']} / 门禁后候选 {s['pool']} | "
              f"2024+ {s['rec']} ({s['rec']/s['n']:.0%}) | 国内 {s['cn']} ({s['cn']/s['n']:.0%}) "
              f"| 国内一作 {s['cnf']} ({s['cnf']/s['n']:.0%})")
    T = dict(n=sum(s["n"] for s in stat), rec=sum(s["rec"] for s in stat),
             cn=sum(s["cn"] for s in stat), cnf=sum(s["cnf"] for s in stat))
    print(f"\n合计 {T['n']} | 近三年 {T['rec']} ({T['rec']/T['n']:.1%}) "
          f"| 国内 {T['cn']} ({T['cn']/T['n']:.1%}) | 国内一作 {T['cnf']} ({T['cnf']/T['n']:.1%})")


if __name__ == "__main__":
    main()
