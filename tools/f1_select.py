# -*- coding: utf-8 -*-
"""
F1 方向文献筛选：门禁（因果方法词 x 水产域词）+ 跨领域排除 + 与既有两批 DOI 去重 + 配额配平。

口径声明：
  第一批 133 篇 = OpenAlex（国别取自机构 country_code，严格）
  第二批 332 篇 = Crossref（无国别字段，启发式推断，保守下限）
  第三批 F1    = OpenAlex（回到 country_code 严格口径，与第一批同源、与第二批不同源）

用法：python tools/f1_select.py
"""
import json, os, re

BASE = r"E:\BJTU\CXCY\CXCY-PAPER"
SRC = os.path.join(BASE, "project", "F1", "_source")
RAW = json.load(open(os.path.join(SRC, "f1_papers.json"), encoding="utf-8"))
STORE = RAW["records"]

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

# 门禁：必须同时命中【因果识别方法词】与【水产/水域域词】
CAUSAL = ["causal", "counterfactual", "treatment effect", "confound", "instrumental variable",
          "double machine learning", "debiased", "tmle", "targeted maximum likelihood",
          "do-calculus", "directed acyclic", "dag ", "effect estimation", "intervention effect",
          "potential outcome", "propensity score", "difference-in-differences",
          "synthetic control", "mediation analysis", "structural causal", "backdoor",
          "average treatment", "heterogeneous treatment", "causal discovery",
          "granger causality", "transfer entropy", "attributable effect", "path analysis"]

DOMAIN = ["aquacultur", "fish", "shrimp", "prawn", "crab", "oyster", "mollusc", "shellfish",
          "tilapia", "salmon", "trout", "seabream", "seabass", "lobster", "sea cucumber",
          "seaweed", "kelp", "mariculture", "hatchery", "pond", "cage", "fishery", "fisheries",
          "aquatic", "water quality", "eutroph", "algal bloom", "harmful algal",
          "coral", "reef", "estuar", "lake", "reservoir", "river", "stream", "watershed",
          "wetland", "coastal", "marine", "ocean", "seafood", "biofloc", "recirculating aquacultur",
          "dissolved oxygen", "chlorophyll", "nutrient load"]

# 跨领域噪声排除（防止"医学/社科因果推断"混入）
EXCLUDE = ["covid", "sars-cov", "epidemiolog", "smoking", "tobacco", "alcohol",
           "education", "school", "student", "labor market", "wage", "employment",
           "income inequality", "poverty", "political", "election", "voting",
           "cancer", "tumor", "tumour", "patient", "clinical", "hospital", "disease risk factor",
           "depression", "mental health", "obesity", "diabetes", "vaccine", "mortality rate human",
           "hiring", "recidivism", "lending", "criminal", "loan", "marketing", "advertising",
           "autonomous driving", "traffic", "social media", "twitter", "stock", "cryptocurrenc",
           "housing", "real estate", "bitcoin", "railway", "brain", "gene expression human",
           "soil carbon", "crop yield", "maize", "wheat", "rice", "cotton", "livestock",
           "poultry", "cattle", "pig ", "swine", "forest fire", "wildfire", "earthquake",
           "landslide", "hurricane damage", "flood damage human", "air pollution",
           "pm2.5", "pm10", "ozone health", "heat mortality", "urban heat",
           "greenhouse gas", "carbon emission", "energy policy", "electricity",
           "battery", "solar panel", "wind turbine", "machine translation", "nlp",
           "recommendation system", "sentiment analysis", "fake news", "misinformation",
           "protein structure", "drug discovery", "genomic human", "microbiome human",
           "sports", "tourism", "hotel", "airline", "retail", "e-commerce"]


def tkey(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())[:70]


def gate_ok(title, venue=""):
    t = " " + (title or "").lower() + " "
    v = " " + (venue or "").lower() + " "
    if any(k in t for k in EXCLUDE):
        return False
    if not any(k in t or k in v for k in CAUSAL):
        return False
    if not any(k in t or k in v for k in DOMAIN):
        return False
    return True


# 强相关加分：标题里同时出现因果词与水产词（而非靠 venue 兜底）
def strong(title):
    t = (title or "").lower()
    return (any(k in t for k in CAUSAL) and any(k in t for k in DOMAIN))


QUOTA = dict(r_cnf=20, r_cn=6, r_in=12, o_cn=4, o_in=5)


def main():
    seen, uniq, dropped_dedup, dropped_gate = set(), [], 0, 0
    for i, r in STORE.items():
        d = (r.get("doi") or "").strip().lower()
        if d and d in OLD_DOI:
            dropped_dedup += 1
            continue
        if not gate_ok(r["title"], r.get("venue", "")):
            dropped_gate += 1
            continue
        k = tkey(r["title"])
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    # 优先保留"标题同时含因果与水产词"的强相关条目
    uniq.sort(key=lambda r: (not strong(r["title"]), -(r["year"] or 0), -r["cited"]))

    rec = [r for r in uniq if (r["year"] or 0) >= 2024]
    old = [r for r in uniq if (r["year"] or 0) < 2024]
    r_cnf = [r for r in rec if r["cn_first"]]
    r_cn = [r for r in rec if r["cn"] and not r["cn_first"]]
    r_in = [r for r in rec if not r["cn"]]
    o_cn = [r for r in old if r["cn"]]
    o_in = [r for r in old if not r["cn"]]

    picked, s2 = [], set()
    for grp, n in ((r_cnf, QUOTA["r_cnf"]), (r_cn, QUOTA["r_cn"]), (r_in, QUOTA["r_in"]),
                   (o_cn, QUOTA["o_cn"]), (o_in, QUOTA["o_in"])):
        for r in grp[:n]:
            k = tkey(r["title"])
            if k in s2:
                continue
            s2.add(k)
            picked.append(r)

    picked.sort(key=lambda x: (-(x["year"] or 0), -x["cited"]))

    n = len(picked)
    T = dict(n=n, rec=sum(1 for r in picked if (r["year"] or 0) >= 2024),
             cn=sum(1 for r in picked if r["cn"]),
             cnf=sum(1 for r in picked if r["cn_first"]),
             strong=sum(1 for r in picked if strong(r["title"])),
             pool=len(uniq), raw=len(STORE),
             dropped_dedup=dropped_dedup, dropped_gate=dropped_gate)

    # 人工复核 dump
    rp = os.path.join(SRC, "_f1_review.txt")
    with open(rp, "w", encoding="utf-8") as f:
        for i, r in enumerate(picked, 1):
            org = "国内" if r["cn_first"] else ("国内合作" if r["cn"] else "国外")
            f.write(f"{i:2d}|{r['year']}|{org}|S={int(strong(r['title']))}|{r['cited']:5d}| "
                    f"{r['title'][:120]}\n      venue={r['venue'][:90]} | doi={r['doi']}\n")

    json.dump({"_meta": {"queries": RAW["queries"], "per_query": RAW["per_query"]},
               "selected": picked, "stats": T},
              open(os.path.join(SRC, "f1_selected.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print(f"原始唯一 {T['raw']} → 过门禁 {T['pool']} → 入选 {T['n']}")
    print(f"去重丢弃 {dropped_dedup} | 门禁丢弃 {dropped_gate}")
    print(f"2024+ {T['rec']} ({T['rec']/max(1,n):.1%}) | 国内机构 {T['cn']} ({T['cn']/max(1,n):.1%}) "
          f"| 国内一作 {T['cnf']} | 强相关 {T['strong']}")
    print("dump ->", rp)


if __name__ == "__main__":
    main()
