# -*- coding: utf-8 -*-
"""
F1 方向专用检索（OpenAlex）：用更贴合"因果推断 x 水产养殖"的检索式，
补齐 `tools/f_search.py` 中 F1 那 8 条宽泛检索式覆盖不到的强相关文献。

与 f_search.py 的差异：
  - 检索式更窄（因果识别方法词 + 养殖干预对象词同时出现）
  - 增加"中文/中国机构"通道
  - 缓存目录沿用 _cache_f，键仍按 URL 的 md5，因此可增量续跑

输出：project/F1/_source/f1_extra_raw.json

用法：python tools/f1_fetch.py
"""
import json, os, time, hashlib, urllib.request, urllib.parse, urllib.error

BASE = r"E:\BJTU\CXCY\CXCY-PAPER"
CACHE = os.path.join(BASE, "tools", "_cache_f")
os.makedirs(CACHE, exist_ok=True)
MAIL = "cxcy.research@example.com"
SEL = ("id,doi,title,publication_year,publication_date,authorships,"
       "primary_location,cited_by_count,type,language,topics,is_retracted")

QUERIES = [
    # A. 干预效应：投喂 / 密度 / 换水 / 增氧 / 益生菌
    "causal effect feeding rate growth shrimp pond",
    "causal effect stocking density fish survival observational",
    "causal inference water exchange aquaculture pond",
    "treatment effect aeration dissolved oxygen pond",
    "causal impact probiotic supplementation shrimp",
    # B. 方法层：观测数据 + 混杂
    "double machine learning observational ecological data",
    "targeted maximum likelihood estimation environmental health",
    "causal forest heterogeneous effects environmental",
    "directed acyclic graph causal aquaculture",
    "propensity score matching aquaculture adoption",
    "difference in differences fishery policy impact",
    "instrumental variable estimation fishery yield",
    "causal discovery time series water quality",
    "Granger causality water quality nutrient dynamics",
    "counterfactual scenario simulation fishery management",
    "mediation analysis growth performance aquatic",
    "unmeasured confounding sensitivity analysis ecology",
    "time varying confounding longitudinal ecological exposure",
    # C. 中文相关（以英文检索式覆盖中国机构）
    "causal inference shrimp farming China",
    "aquaculture intervention effect China observational data",
    "causal analysis pond aquaculture production China",
    # D. 与相邻可做方向的衔接
    "causal reasoning machine learning model evaluation aquaculture",
    "causal feature selection environmental prediction model",
]


def fetch(url, tries=5):
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
            time.sleep(1.2)
            return d
        except urllib.error.HTTPError as e:
            last = e
            time.sleep(20 * (2 ** i))
        except Exception as e:
            last = e
            time.sleep(5 * (i + 1))
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
        "topics": [t.get("display_name") for t in (w.get("topics") or [])[:3]],
    }


def main():
    store, log, failed = {}, [], 0
    for q in QUERIES:
        ua = build_url(q, "from_publication_date:2020-01-01,"
                          "type:article|review|proceedings-article,is_retracted:false", 25)
        ub = build_url(q, "from_publication_date:2023-01-01,institutions.country_code:CN,"
                          "type:article|review|proceedings-article,is_retracted:false", 20)
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
                got += 1
        log.append({"query": q, "new": got})
        print(f"  +{got:3d}  {q}", flush=True)

    out = os.path.join(BASE, "project", "F1", "_source")
    os.makedirs(out, exist_ok=True)
    json.dump({"records": store, "log": log, "failed": failed, "queries": QUERIES},
              open(os.path.join(out, "f1_extra_raw.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n唯一文献 {len(store)} | 抓取失败 {failed}")


if __name__ == "__main__":
    main()
