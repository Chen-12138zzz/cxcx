# -*- coding: utf-8 -*-
"""
F1 方向专用文献抽取：从 tools/_cache_f 已缓存的 OpenAlex 响应中，
把 F1 的 8 条检索式的全部命中抽出来，写为 project/F1/_source/f1_papers.json。

**不重新联网**——只用本地缓存，保证可复现且不受 429 影响。
缓存文件是 f_search.py 按 URL 的 md5 命名的，因此这里重建同样的 URL 来定位缓存。

用法：python tools/f1_harvest.py
"""
import json, os, hashlib, urllib.parse

BASE = r"E:\BJTU\CXCY\CXCY-PAPER"
CACHE = os.path.join(BASE, "tools", "_cache_f")
MAIL = "cxcy.research@example.com"
SEL = ("id,doi,title,publication_year,publication_date,authorships,"
       "primary_location,cited_by_count,type,language,topics,is_retracted")

F1_QUERIES = [
    "causal inference observational aquaculture",
    "causal effect estimation fish growth feeding",
    "causal machine learning environmental intervention effect",
    "double machine learning treatment effect environmental data",
    "targeted maximum likelihood estimation causal effect ecology",
    "causal discovery time series environmental monitoring",
    "counterfactual prediction ecological intervention",
    "confounding adjustment observational fishery data",
]


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
    topics = [t.get("display_name") for t in (w.get("topics") or [])[:3]]
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
    store, log, missing, per_q = {}, [], [], {}
    for q in F1_QUERIES:
        ua = build_url(q, "from_publication_date:2021-01-01,"
                          "type:article|review|proceedings-article,is_retracted:false", 25)
        ub = build_url(q, "from_publication_date:2024-01-01,institutions.country_code:CN,"
                          "type:article|review|proceedings-article,is_retracted:false", 15)
        got = 0
        for tag, u in (("A", ua), ("B", ub)):
            key = hashlib.md5(u.encode("utf-8")).hexdigest()[:16] + ".json"
            path = os.path.join(CACHE, key)
            if not os.path.exists(path):
                missing.append({"query": q, "chan": tag})
                continue
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            for w in d.get("results", []):
                r = parse(w)
                if not r["title"] or not r["id"]:
                    continue
                if r["id"] not in store:
                    store[r["id"]] = r
                    got += 1
        per_q[q] = got
        log.append({"query": q, "new": got})

    out = os.path.join(BASE, "project", "F1", "_source")
    os.makedirs(out, exist_ok=True)
    json.dump({"records": store, "log": log, "missing_cache": missing,
               "queries": F1_QUERIES, "per_query": per_q},
              open(os.path.join(out, "f1_papers.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("唯一文献:", len(store))
    for q in F1_QUERIES:
        print(f"  {per_q.get(q,0):3d}  {q}")
    if missing:
        print("\n缺失缓存（需重跑 f_search.py）:")
        for m in missing:
            print("  ", m["chan"], m["query"])
    r3 = sum(1 for r in store.values() if (r["year"] or 0) >= 2024)
    cn = sum(1 for r in store.values() if r["cn"])
    cnf = sum(1 for r in store.values() if r["cn_first"])
    print(f"\n2024+ {r3} ({r3/max(1,len(store)):.1%}) | 含中国机构 {cn} | 中国一作 {cnf}")
    print("dump ->", os.path.join(out, "f1_papers.json"))


if __name__ == "__main__":
    main()
