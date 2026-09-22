# -*- coding: utf-8 -*-
"""
第三批（F1-F16 新方向）文献筛选：
  方向关键词门禁(GATE) + 跨领域噪声排除(EXCLUDE) + 与第一批133篇/第二批332篇 DOI 去重
  + 配额配平（近三年 >=70%、国内 >=50% 双口径）+ 输出人工复核 dump。

口径声明（重要，与第二批不可直接比较）：
  第一批 OpenAlex，国别来自机构 country_code（严格）。
  第二批 Crossref，无国别字段，国别按机构名启发式推断（保守下限）。
  第三批回到 OpenAlex，国别恢复 country_code 口径 —— 与第一批同源，与第二批不同源。
"""
import json, os, re

BASE = r"E:\BJTU\CXCY\CXCY-PAPER"
RAW = json.load(open(os.path.join(BASE, "tools", "_f_raw.json"), encoding="utf-8"))
STORE, PER_DIR = RAW["records"], RAW["per_dir"]


# ---------- 与既有两批去重（按 DOI） ----------
def _collect_dois(paths):
    out = set()
    for dpath, rpath in paths:
        if not (os.path.exists(dpath) and os.path.exists(rpath)):
            continue
        try:
            sel = json.load(open(dpath, encoding="utf-8"))
            rec = json.load(open(rpath, encoding="utf-8"))["records"]
        except Exception:
            continue
        for _, ids in sel.items():
            for i in ids:
                d = (rec.get(i) or {}).get("doi") or ""
                if d:
                    out.add(d.strip().lower())
    return out


OLD_DOI = _collect_dois([
    (os.path.join(BASE, "tools", "_selected.json"), os.path.join(BASE, "tools", "_papers_raw.json")),
    (os.path.join(BASE, "tools", "_eng_selected.json"), os.path.join(BASE, "tools", "_eng_raw.json")),
])

DIR_NAMES = {
    "F1":  "水产养殖因果推断与干预效应估计",
    "F2":  "养殖系统灰箱机理—数据混合建模",
    "F3":  "养殖 AI 模型的评测基准与公平性审计",
    "F4":  "地理空间多源数据融合的养殖适宜性评估",
    "F5":  "养殖多智能体仿真与策略评估",
    "F6":  "远域迁移与基础模型适配",
    "F7":  "养殖 AI 的对抗鲁棒性与数据投毒防御",
    "F8":  "公开气象海况数据的养殖风险指数构建",
    "F9":  "公开图像库的少样本零样本病害识别",
    "F10": "水生生物基因组与表型数据的关联挖掘",
    "F11": "养殖水体的卫星水质反演与地面校核",
    "F12": "养殖环境数据的隐私保护与联邦学习",
    "F13": "公开舆情与市场数据的产业链风险感知",
    "F14": "养殖 AI 系统的可复现性与复现审计",
    "F15": "模型部署后的行为监控与自动回滚",
    "F16": "可解释性方法的场景化验证",
}

# 预研门禁 A 结论（来自 00-总览/02-门禁A预研-占位者初查.md），用于清单分组标注
GATE_A_PRE = {
    "F1": "待复核（缺口明显）", "F2": "已占位", "F3": "待复核（缺口明显）",
    "F4": "待复核（缺口明显）", "F5": "已占位", "F6": "已占位",
    "F7": "高概率占位", "F8": "已占位（成套）", "F9": "高概率占位",
    "F10": "高概率占位", "F11": "已占位（含组织级反证）", "F12": "已占位",
    "F13": "已占位（成套）", "F14": "待复核（缺口明显）", "F15": "高概率占位",
    "F16": "已占位",
}

DOMAIN = ["fish", "shrimp", "prawn", "aquacultur", "aquatic", "crab", "oyster", "mollusc",
          "shellfish", "tilapia", "salmon", "trout", "seabream", "seabass", "lobster",
          "sea cucumber", "seaweed", "kelp", "fishery", "fisheries", "pond", "cage",
          "mariculture", "hatchery", "marine", "ocean", "coastal", "water", "aquafarm",
          "seafood", "fisher", "shoal", "reef", "estuar", "lake", "reservoir",
          "agricultur", "farm", "food"]

GATE = {
    "F1": [["causal", "counterfactual", "treatment effect", "confound", "instrumental variable",
            "double machine learning", "tmle", "targeted maximum likelihood", "do-calculus",
            "dag", "directed acyclic", "effect estimation", "intervention effect"], DOMAIN],
    "F2": [["physics-informed", "physics informed", "physics-guided", "hybrid model",
            "grey-box", "grey box", "gray-box", "mechanistic", "data assimilation",
            "differentiable", "parameter identification", "hybrid modeling", "surrogate"], DOMAIN],
    "F3": [["benchmark", "evaluation framework", "fairness", "equity", "disparity",
            "subgroup", "audit", "bias", "evaluation protocol", "assessment framework",
            "performance evaluation", "comparison study", "leaderboard"], DOMAIN],
    "F4": [["gis", "spatial multi-criteria", "site selection", "site suitability",
            "suitability", "geospatial", "spatial analysis", "land suitability",
            "decision support", "ahp", "multi-criteria", "remote sensing", "geographic"], DOMAIN],
    "F5": [["agent-based", "multi-agent", "multiagent", "simulation", "abm", "digital twin",
            "policy evaluation", "scenario analysis", "system dynamics", "reinforcement learning",
            "model predictive control", "simulator"], DOMAIN],
    "F6": [["transfer learning", "domain adaptation", "domain generalization", "cross-domain",
            "cross-species", "foundation model", "pretrain", "meta-learning", "maml",
            "few-shot", "zero-shot", "adaptation"], DOMAIN],
    "F7": [["adversarial", "poisoning", "poison", "attack", "robustness", "backdoor",
            "perturbation", "security", "threat", "defense", "defence", "tamper"], DOMAIN],
    "F8": [["risk index", "weather index", "climate risk", "typhoon", "storm", "cold wave",
            "heat wave", "meteorological", "early warning", "risk assessment", "insurance",
            "extreme weather", "hazard", "sea state", "wave height"], DOMAIN],
    "F9": [["few-shot", "zero-shot", "one-shot", "meta-learning", "disease", "pathology",
            "lesion", "recognition", "classification", "public dataset", "image dataset",
            "open dataset", "data-efficient", "self-supervised"], DOMAIN],
    "F10": [["genomic", "genome", "gwas", "transcriptom", "phenotype", "phenotyp",
             "snp", "breeding", "quantitative trait", "gene expression", "sequence",
             "association study", "genetic"], DOMAIN],
    "F11": [["remote sensing", "satellite", "sentinel", "landsat", "modis", "retrieval",
             "inversion", "inversion model", "water quality", "chlorophyll", "turbidity",
             "spectral", "reflectance", "atmospheric correction"], DOMAIN],
    "F12": [["federated", "privacy", "differential privacy", "secure aggregation",
             "homomorphic", "data sharing", "confidential", "anonymi", "decentralized learning",
             "distributed learning"], DOMAIN],
    "F13": [["sentiment", "public opinion", "market price", "price volatility", "supply chain",
             "risk perception", "social media", "news", "regime", "early warning",
             "market risk", "commodity"], DOMAIN],
    "F14": [["reproducib", "repeatab", "replicab", "replication crisis", "benchmark",
             "open data", "open source", "fa:ir", "fair data", "data availability",
             "validation protocol", "reporting", "audit", "meta-research", "meta-analysis",
             "confirmatory", "preregistration"], DOMAIN],
    "F15": [["drift", "monitoring", "rollback", "degradation", "concept drift", "data drift",
             "maintenance", "predictive maintenance", "failure", "fault", "anomaly detection",
             "production deployment", "post-deployment", "model monitoring", "retrain"], DOMAIN],
    "F16": [["explainab", "interpretab", "xai", "shap", "lime", "transparen",
             "explanation", "user study", "trust", "human-centered", "human-centred",
             "decision support", "visualization", "stakeholder"], DOMAIN],
}

EXCLUDE = ["carotenoid", "metaverse", "railway", "orb-slam", "brain tumor", "parkinson",
           "alzheimer", "pothole", "spatial transcriptomics", "alphafold", "microplastic",
           "crop yield", "cotton", "maize", "apple disease", "plant disease", "shelf life",
           "heavy metal", "marketing", "soil", "greenhouse gas", "antibiotic", "walrus",
           "metaheuristic", "bitcoin", "cryptocurrency", "electric vehicle", "autonomous driving",
           "traffic prediction", "smart city", "concrete", "bridge damage", "wind turbine",
           "photovoltaic", "stock market", "covid", "fruit", "vegetable", "dairy", "poultry",
           "pig ", "cattle", "wheat", "rice", "forest fire", "landslide", "earthquake",
           "flood forecasting", "urban drainage", "drinking water treatment",
           "membrane bioreactor", "activated sludge", "hospital", "wearable", "emotion",
           "retraction notice", "structural health monitoring", "wind farm", "offshore wind",
           "tunnel", "gravity dam", "coal mine", "natural gas", "petroleum", "water meter",
           "shelf-life", "human gut", "clinical trial", "medical imaging", "cancer",
           "protein structure", "drug discovery", "social network analysis"]

# 方向专属排除（防跨领域串味）
EXCLUDE_BY_DIR = {
    "F1":  ["covid", "epidemiolog", "smoking", "education", "labor market", "wage",
            "income inequality", "political", "medicine"],
    "F2":  ["battery", "combustion", "turbine", "aerospace", "blood flow", "cardiac"],
    "F3":  ["recidivism", "hiring", "lending", "criminal justice", "loan", "insurance pricing",
            "social media", "face recognition", "facial"],
    "F4":  ["urban planning", "real estate", "highway", "municipal solid waste", "landfill",
            "solar farm", "wind energy", "mining", "road"],
    "F5":  ["epidemic model", "traffic flow", "crowd", "pedestrian", "electricity market",
            "supply chain management", "game theory"],
    "F6":  ["speech", "natural language", "machine translation", "text classification",
            "sentiment analysis", "recommendation", "question answering"],
    "F7":  ["autonomous driving", "face recognition", "malware", "network intrusion",
            "deepfake", "llm jailbreak", "jailbreak", "prompt injection"],
    "F8":  ["crop", "wheat", "maize", "rice", "drought index", "crop insurance", "hail",
            "livestock", "poultry", "forest", "heat wave mortality city"],
    "F9":  ["plant disease", "leaf disease", "crop disease", "maize", "wheat", "rice",
            "apple", "tomato", "grape", "covid", "chest x-ray", "skin lesion", "mri",
            "brain", "lung", "breast", "retinop", "histopatholog"],
    "F10": ["human", "cattle", "pig", "poultry", "maize", "wheat", "rice", "arabidopsis",
            "mouse", "drosophila", "cancer", "covid", "human genome", "personalized medicine",
            "drug"],
    "F11": ["crop", "vegetation", "soil moisture", "drought", "land cover", "urban heat",
            "air quality", "pm2.5", "aerosol", "snow", "glacier", "forest biomass",
            "terrestrial", "ndvi crop", "agricultur"],
    "F12": ["blockchain", "smart contract", "financial", "bank", "healthcare record",
            "electronic health", "smart grid", "mobile edge", "recommendation",
            "cryptocurrenc"],
    "F13": ["twitter", "stock return", "cryptocurrency", "covid", "tourism", "hotel",
            "airline", "oil price", "electricity price", "housing"],
    "F14": ["psychology", "social science", "cancer biology", "genomics reproducibility",
            "machine learning benchmark levit", "imagenet", "nlp reproducibility"],
    "F15": ["manufacturing", "wind turbine", "engine", "bearing", "industrial process",
            "semiconductor", "power grid", "structural", "automotive", "network intrusion"],
    "F16": ["recommendation", "recidivism", "hiring", "lending", "autonomous driving",
            "medical imaging", "radiology", "chatbot", "llm", "medical diagnosis",
            "education", "finance", "recruit"],
}

QUOTA = dict(r_cnf=22, r_cn=8, r_in=10, o_cn=4, o_in=4)

# 缺口证据标记（用于复核时快速定位有综述背书的条目）
GAP_EVIDENCE = {
    "F3":  ["benchmark", "evaluation", "audit", "fairness", "assessment"],
    "F14": ["reproducib", "replicab", "repeatab", "fair data", "validation protocol",
            "reporting", "data availability"],
}


def tkey(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:70]


def gate_ok(did, title):
    t = " " + (title or "").lower() + " "
    if any(k in t for k in EXCLUDE):
        return False
    if any(k in t for k in EXCLUDE_BY_DIR.get(did, [])):
        return False
    for group in GATE[did]:
        if not any(k in t for k in group):
            return False
    return True


def gap_flag(did, title):
    t = (title or "").lower()
    return 1 if any(k in t for k in GAP_EVIDENCE.get(did, [])) else 0


def select(did, ids):
    seen, uniq = set(), []
    for i in ids:
        if i not in STORE:
            continue
        r = STORE[i]
        d = (r.get("doi") or "").strip().lower()
        if d and d in OLD_DOI:
            continue
        k = tkey(r["title"])
        if not k or k in seen or not gate_ok(did, r["title"]):
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


def main():
    sel_all, stats = {}, []
    for did in DIR_NAMES:
        s, pool = select(did, PER_DIR.get(did, []))
        sel_all[did] = s
        n = len(s)
        stats.append(dict(did=did, name=DIR_NAMES[did], gate_a=GATE_A_PRE[did], n=n, pool=pool,
                          rec=sum(1 for r in s if (r["year"] or 0) >= 2024),
                          cn=sum(1 for r in s if r["cn"]),
                          cnf=sum(1 for r in s if r["cn_first"]),
                          aff=sum(1 for r in s if r.get("has_aff")),
                          gap=sum(1 for r in s if gap_flag(did, r["title"]))))

    T = dict(n=sum(s["n"] for s in stats), rec=sum(s["rec"] for s in stats),
             cn=sum(s["cn"] for s in stats), cnf=sum(s["cnf"] for s in stats),
             aff=sum(s["aff"] for s in stats), pool=sum(s["pool"] for s in stats))

    rp = os.path.join(BASE, "tools", "_f_review.txt")
    with open(rp, "w", encoding="utf-8") as f:
        for did in DIR_NAMES:
            f.write(f"\n##### {did} {DIR_NAMES[did]} | 门禁A预研={GATE_A_PRE[did]} #####\n")
            for i, r in enumerate(sel_all[did], 1):
                org = "国内" if r["cn_first"] else ("国内(合作)" if r["cn"] else "国外")
                f.write(f"{i:2d}|{r['year']}|{org}|gap={gap_flag(did, r['title'])}|"
                        f"{r['cited']:6d}| {r['title']}\n")

    json.dump({d: [r["id"] for r in v] for d, v in sel_all.items()},
              open(os.path.join(BASE, "tools", "_f_selected.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    for s in stats:
        print(f"{s['did']} {s['name']}: 入选 {s['n']} / 候选 {s['pool']} | "
              f"2024+ {s['rec']} ({s['rec']/max(1,s['n']):.0%}) | 国内 {s['cn']} "
              f"({s['cn']/max(1,s['n']):.0%}) | 国内一作 {s['cnf']} | 缺口证据 {s['gap']}"
              f"  [{s['gate_a']}]")
    print(f"\n新增合计 {T['n']} / 候选池 {T['pool']} | 近三年 {T['rec']} "
          f"({T['rec']/max(1,T['n']):.1%}) | 国内 {T['cn']} ({T['cn']/max(1,T['n']):.1%}) "
          f"| 国内一作 {T['cnf']} ({T['cnf']/max(1,T['n']):.1%})")

    json.dump({"stats": stats, "total": T},
              open(os.path.join(BASE, "tools", "_f_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("dump ->", rp)


if __name__ == "__main__":
    main()
