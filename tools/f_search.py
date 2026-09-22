# -*- coding: utf-8 -*-
"""
第三轮扩展：F1–F16 新候选方向的文献检索（OpenAlex）。

数据源说明（与第二批的重要差异）：
- 第二批因 OpenAlex 持久 429 改用 Crossref，**无国别字段**，国别靠机构名启发式推断。
- 第三轮**OpenAlex 已恢复**（2026-09-20 实测 HTTP 200），因此本批回到 OpenAlex，
  **机构国别来自 country_code 字段**，口径与第一批一致，强于第二批。
  → 引用时须声明：第三批与第二批的"国内占比"口径不同，不可直接比较。

双通道检索：
- 通道 A：2021 年起，相关度前 N
- 通道 B：2024 年起，中国机构，前 M
"""
import json, os, time, hashlib, urllib.request, urllib.parse, urllib.error

BASE = r"E:\BJTU\CXCY\CXCY-PAPER"
CACHE = os.path.join(BASE, "tools", "_cache_f")
os.makedirs(CACHE, exist_ok=True)
MAIL = "cxcy.research@example.com"
SEL = ("id,doi,title,publication_year,publication_date,authorships,"
       "primary_location,cited_by_count,type,language,topics,is_retracted")

DIRECTIONS = {
    "F1": {"name": "水产养殖因果推断与干预效应估计", "kind": "T1 技术迁移", "queries": [
        "causal inference observational aquaculture",
        "causal effect estimation fish growth feeding",
        "causal machine learning environmental intervention effect",
        "double machine learning treatment effect environmental data",
        "targeted maximum likelihood estimation causal effect ecology",
        "causal discovery time series environmental monitoring",
        "counterfactual prediction ecological intervention",
        "confounding adjustment observational fishery data"]},
    "F2": {"name": "养殖系统灰箱机理—数据混合建模", "kind": "T1 技术迁移", "queries": [
        "physics informed neural network water quality",
        "hybrid model mechanistic machine learning environmental system",
        "grey box model dissolved oxygen pond",
        "physics guided machine learning ecological forecasting",
        "parameter identification mechanistic model aquaculture",
        "data assimilation mechanistic model water quality",
        "differentiable simulation ecosystem model",
        "hybrid model predictive control aquaculture"]},
    "F3": {"name": "养殖 AI 模型的评测基准与公平性审计", "kind": "T3 横切工具", "queries": [
        "benchmark dataset aquaculture machine learning evaluation",
        "fairness audit machine learning subgroups performance disparity",
        "model evaluation protocol agriculture benchmark",
        "performance disparity across domains machine learning audit",
        "reproducible benchmark environmental machine learning",
        "subgroup robustness evaluation model",
        "benchmark fish detection segmentation evaluation",
        "fairness machine learning agriculture smallholder"]},
    "F4": {"name": "地理空间多源数据融合的养殖适宜性评估", "kind": "T2 新数据源", "queries": [
        "multi-source geospatial data fusion aquaculture suitability",
        "spatial multi-criteria analysis aquaculture site suitability",
        "GIS remote sensing suitability mapping mariculture",
        "spatiotemporal fusion multi-source remote sensing water",
        "scale mismatch multi-scale remote sensing modeling",
        "geospatial data fusion land suitability machine learning",
        "aquaculture zoning spatial analysis multi-source"]},
    "F5": {"name": "养殖多智能体仿真与策略评估", "kind": "T1 技术迁移", "queries": [
        "agent based model aquaculture management simulation",
        "individual based model fish farm simulation",
        "agent based simulation policy evaluation agriculture",
        "simulation policy evaluation reinforcement learning environment",
        "digital twin simulation based decision evaluation",
        "individual based model disease spread aquaculture",
        "simulation model management strategy fish pond",
        "multi agent simulation resource management fisheries"]},
    "F6": {"name": "远域迁移与基础模型适配", "kind": "T1 技术迁移", "queries": [
        "domain shift quantification transfer learning fish",
        "cross domain generalization aquaculture image",
        "foundation model adaptation domain shift agriculture",
        "domain adaptation fish species recognition cross dataset",
        "distribution shift detection transfer learning ecological",
        "domain gap underwater imagery cross dataset evaluation",
        "foundation model remote sensing domain adaptation water",
        "out of distribution generalization species recognition"]},
    "F7": {"name": "养殖 AI 的对抗鲁棒性与数据投毒防御", "kind": "T1 技术迁移", "queries": [
        "adversarial attack robustness aquaculture model",
        "data poisoning defense machine learning agriculture",
        "adversarial examples environmental monitoring model",
        "backdoor attack detection time series model",
        "adversarial robustness water quality prediction model",
        "poisoning attack sensor data detection defense",
        "certified robustness machine learning deployment",
        "adversarial attack fish detection model"]},
    "F8": {"name": "公开气象海况数据的养殖风险指数构建", "kind": "T2 新数据源", "queries": [
        "meteorological index aquaculture risk assessment",
        "typhoon storm risk index mariculture",
        "marine heatwave risk index aquaculture",
        "climate risk index coastal aquaculture modeling",
        "extreme weather risk index fish farming",
        "compound extreme event risk coastal index",
        "early warning index public meteorological data agriculture",
        "cold wave risk index pond aquaculture"]},
    "F9": {"name": "公开图像库的少样本零样本病害识别", "kind": "T2 新数据源", "queries": [
        "few shot learning aquatic species classification",
        "zero shot learning fish disease recognition",
        "cross dataset generalization plant disease classification",
        "open image dataset benchmark fish classification",
        "few shot learning animal disease image",
        "zero shot transfer crop disease open dataset",
        "public dataset generalization skin lesion cross dataset",
        "domain generalization few shot image classification benchmark"]},
    "F10": {"name": "水生生物基因组与表型数据的关联挖掘", "kind": "T2 新数据源", "queries": [
        "genomic prediction disease resistance shrimp",
        "machine learning genome wide association aquaculture species",
        "transcriptomic biomarker machine learning fish immunity",
        "genomic selection breeding value prediction aquatic",
        "public genome database machine learning phenotype prediction",
        "multi omics integration machine learning aquatic species",
        "gene expression classification stress response shrimp"]},
    "F11": {"name": "养殖水体的卫星水质反演与地面校核", "kind": "T2 新数据源", "queries": [
        "satellite retrieval chlorophyll small inland water",
        "remote sensing water quality inversion turbidity validation",
        "Sentinel-2 inland water quality retrieval accuracy",
        "small water body remote sensing algorithm validation",
        "water quality retrieval algorithm comparison validation",
        "satellite derived water quality validation in situ",
        "remote sensing retrieval aquaculture pond water quality",
        "atmospheric correction inland water retrieval uncertainty"]},
    "F12": {"name": "养殖环境数据的隐私保护与联邦学习", "kind": "T1 技术迁移", "queries": [
        "federated learning aquaculture farm data privacy",
        "federated learning environmental monitoring sensor",
        "differential privacy time series prediction agriculture",
        "privacy preserving machine learning agriculture data sharing",
        "federated learning water quality prediction",
        "secure aggregation sensor network privacy utility tradeoff",
        "federated learning heterogeneous clients non iid",
        "privacy preserving collaborative learning farm data"]},
    "F13": {"name": "公开舆情与市场数据的产业链风险感知", "kind": "T2 新数据源", "queries": [
        "news based early warning food supply chain",
        "text mining fishery market price signal",
        "social media signal detection agriculture risk",
        "web data mining agricultural commodity price prediction",
        "automated event extraction food safety news",
        "supply chain risk signal extraction text",
        "online market data analysis seafood trade",
        "public data driven fishery market analysis"]},
    "F14": {"name": "养殖 AI 系统的可复现性与复现审计", "kind": "T3 横切工具", "queries": [
        "reproducibility crisis machine learning audit",
        "reproducibility study deep learning agriculture",
        "replication study evaluation methodology machine learning",
        "code availability reproducibility environmental modeling",
        "reproducibility assessment machine learning papers",
        "replication audit empirical study deep learning",
        "reproducibility machine learning remote sensing",
        "reporting standards model documentation"]},
    "F15": {"name": "模型部署后的行为监控与自动回滚", "kind": "T3 横切工具", "queries": [
        "concept drift adaptation deployment strategy cost",
        "model rollback automation deployment monitoring",
        "retraining strategy cost benefit drift",
        "safe deployment machine learning degradation response",
        "monitoring production model degradation tradeoff",
        "online learning drift adaptation environmental prediction",
        "deployed model maintenance automation strategy",
        "failure detection deployment machine learning system"]},
    "F16": {"name": "可解释性方法的场景化验证", "kind": "T1 技术迁移", "queries": [
        "faithfulness evaluation explainable AI methods",
        "explanation evaluation benchmark attribution methods",
        "XAI evaluation user study decision making",
        "explanation faithfulness metrics comparison",
        "concept bottleneck model evaluation interpretability",
        "saliency map evaluation sanity checks",
        "explanation quality agriculture decision support evaluation",
        "evaluating interpretability methods neural network"]},
}


def fetch(url, tries=4):
    key = hashlib.md5(url.encode("utf-8")).hexdigest()[:16] + ".json"
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": f"cxcy-literature-survey/1.0 (mailto:{MAIL})",
                "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=45) as r:
                d = json.loads(r.read().decode("utf-8"))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
            time.sleep(0.25)
            return d
        except urllib.error.HTTPError as e:
            last = e
            time.sleep(3 * (2 ** i))
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    print("  !! fetch failed:", last, url[:110], flush=True)
    return None


def build_url(search, filters, per_page):
    q = {"search": search, "filter": filters, "per-page": per_page,
         "sort": "relevance_score:desc", "select": SEL, "mailto": MAIL}
    return "https://api.openalex.org/works?" + urllib.parse.urlencode(q)


def parse(w):
    auths = w.get("authorships") or []
    countries, names = set(), []
    for a in auths:
        nm = (a.get("author") or {}).get("display_name")
        if nm:
            names.append(nm)
        for inst in (a.get("institutions") or []):
            cc = inst.get("country_code")
            if cc:
                countries.add(cc)
    first_cc = ""
    for inst in ((auths[0].get("institutions") if auths else None) or []):
        if inst.get("country_code"):
            first_cc = inst["country_code"]
            break
    src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    topics = [t.get("display_name") for t in (w.get("topics") or [])[:2]]
    return {
        "id": w.get("id"),
        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
        "title": (w.get("title") or "").strip(),
        "year": w.get("publication_year"),
        "date": w.get("publication_date"),
        "type": w.get("type"),
        "venue": src,
        "cited": w.get("cited_by_count") or 0,
        "authors": names[:12],
        "n_authors": len(names),
        "first_author": names[0] if names else "",
        "first_country": first_cc,
        "countries": sorted(countries),
        "cn": ("CN" in countries),
        "cn_first": (first_cc == "CN"),
        "topics": topics,
    }


def main():
    store, per_dir, log, failed = {}, {d: [] for d in DIRECTIONS}, [], 0
    for did, spec in DIRECTIONS.items():
        for q in spec["queries"]:
            ua = build_url(q, "from_publication_date:2021-01-01,"
                              "type:article|review|proceedings-article,is_retracted:false", 25)
            ub = build_url(q, "from_publication_date:2024-01-01,institutions.country_code:CN,"
                              "type:article|review|proceedings-article,is_retracted:false", 15)
            got = 0
            for u in (ua, ub):
                d = fetch(u)
                if not d:
                    failed += 1
                    continue
                for w in d.get("results", []):
                    r = parse(w)
                    if not r["title"] or not r["id"] or r["id"] in store:
                        continue
                    store[r["id"]] = r
                    per_dir[did].append(r["id"])
                    got += 1
            log.append({"dir": did, "query": q, "new": got})
            print(f"[{did}] {q} -> +{got}", flush=True)

    print("\n=== 候选池 ===", flush=True)
    print("唯一文献:", len(store), "| 抓取失败次数:", failed, flush=True)
    for did, ids in per_dir.items():
        recs = [store[i] for i in ids]
        r3 = sum(1 for r in recs if (r["year"] or 0) >= 2024)
        cn = sum(1 for r in recs if r["cn"])
        print(f"{did} {DIRECTIONS[did]['name']}: {len(ids)} 篇 | 2024+ {r3} | 含中国机构 {cn}", flush=True)

    json.dump({"records": store, "per_dir": per_dir, "log": log, "failed": failed},
              open(os.path.join(BASE, "tools", "_f_raw.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
