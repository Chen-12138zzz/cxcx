# -*- coding: utf-8 -*-
"""
大创选题文献检索脚本（OpenAlex 开放学术图谱）
用途：按候选选题方向批量检索真实文献，落盘为 JSON 缓存 + 汇总表。
约束：脚本只做检索与统计，不做任何"编造"——所有字段均来自 OpenAlex 原始响应。
"""
import json, os, time, hashlib, urllib.request, urllib.parse, sys

BASE = r"E:\BJTU\CXCY\CXCY-PAPER"
CACHE = os.path.join(BASE, "tools", "_cache")
os.makedirs(CACHE, exist_ok=True)
MAIL = "cxcy.research@example.com"
SEL = ("id,doi,title,publication_year,publication_date,authorships,"
       "primary_location,cited_by_count,type,language,topics,is_retracted")

# ---------------- 候选选题方向与检索式 ----------------
DIRECTIONS = {
    "D1": {
        "name": "水产养殖智能投喂与摄食行为智能感知",
        "queries": [
            "fish feeding behavior recognition computer vision",
            "intelligent feeding system aquaculture deep learning",
            "appetite detection fish acoustic hydrophone",
            "shrimp feeding behavior detection",
            "feeding intensity detection aquaculture",
        ],
    },
    "D2": {
        "name": "水下图像增强与水下目标检测识别",
        "queries": [
            "underwater image enhancement deep learning",
            "underwater object detection deep learning",
            "degraded underwater image restoration",
            "fish detection underwater video",
        ],
    },
    "D3": {
        "name": "养殖水质与环境时序预测预警",
        "queries": [
            "water quality prediction machine learning aquaculture",
            "dissolved oxygen prediction model pond",
            "harmful algal bloom early warning deep learning",
            "aquaculture water quality monitoring IoT prediction",
        ],
    },
    "D4": {
        "name": "水产动物病害智能诊断与健康评估",
        "queries": [
            "fish disease detection deep learning",
            "shrimp disease recognition image classification",
            "aquatic animal disease diagnosis machine learning",
            "fish disease image classification transfer learning",
            "white spot syndrome virus detection shrimp",
            "fish health monitoring computer vision aquaculture",
            "fish skin lesion segmentation detection",
            "fish disease detection YOLO",
            "fish disease classification convolutional neural network",
            "aquaculture fish anomaly detection machine vision",
            "marine fish disease recognition deep learning",
            "shrimp disease detection machine learning image",
        ],
    },
    "D5": {
        "name": "边缘智能与轻量化模型部署（养殖物联网）",
        "queries": [
            "edge computing smart aquaculture",
            "lightweight deep learning model deployment aquaculture",
            "embedded AI fish detection edge device",
            "IoT aquaculture monitoring system design",
            "aquaculture edge computing lightweight model deployment",
            "real-time fish detection embedded device",
        ],
    },
    "D6": {
        "name": "多模态大模型与海洋渔业知识智能",
        "queries": [
            "large language model marine science",
            "multimodal large language model aquaculture",
            "vision language model fish recognition",
            "knowledge graph aquaculture fishery",
            "ocean large language model domain adaptation",
            "large language model agriculture question answering",
            "multimodal large language model water environment",
            "vision language model fish feeding behavior",
            "retrieval augmented generation domain knowledge water",
            "large language model aquaculture fish farming",
            "domain-specific large language model fishery",
            "multimodal large language model underwater",
            "knowledge graph fish disease aquaculture",
            "foundation model marine remote sensing",
            "large language model water quality monitoring",
        ],
    },
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
            req = urllib.request.Request(url, headers={"User-Agent": MAIL})
            with urllib.request.urlopen(req, timeout=40) as r:
                d = json.loads(r.read().decode("utf-8"))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
            time.sleep(0.15)
            return d
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    print("  !! fetch failed:", last, url[:120])
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
    store = {}       # oaid -> record
    order = []       # oaid 首次出现顺序（按方向内相关度）
    per_dir = {}     # D -> [oaid]
    runlog = []

    for did, spec in DIRECTIONS.items():
        per_dir.setdefault(did, [])
        for q in spec["queries"]:
            # 通道 A：2021 年起的相关度排序
            ua = build_url(q, "from_publication_date:2021-01-01,"
                              "type:article|review|proceedings-article,is_retracted:false", 25)
            da = fetch(ua)
            # 通道 B：2024 年起的中国机构文献（提升"近三年 + 国内"占比）
            ub = build_url(q, "from_publication_date:2024-01-01,institutions.country_code:CN,"
                              "type:article|review|proceedings-article,is_retracted:false", 15)
            db = fetch(ub)
            got = 0
            for d in (da, db):
                if not d:
                    continue
                for w in d.get("results", []):
                    r = parse(w)
                    if not r["title"] or not r["id"]:
                        continue
                    if r["id"] not in store:
                        store[r["id"]] = r
                        order.append(r["id"])
                        per_dir[did].append(r["id"])
                        got += 1
            runlog.append({"dir": did, "query": q, "new": got})
            print(f"[{did}] {q}  -> +{got}")

    out = {"records": store, "per_dir": per_dir, "log": runlog}
    with open(os.path.join(BASE, "tools", "_papers_raw.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n=== 汇总 ===")
    print("唯一文献总数:", len(store))
    for did, ids in per_dir.items():
        recs = [store[i] for i in ids]
        rec = sum(1 for r in recs if r["year"] and r["year"] >= 2024)
        cn = sum(1 for r in recs if r["cn"])
        print(f"{did} {DIRECTIONS[did]['name']}: {len(ids)} 篇 | 近三年 {rec} ({rec/max(1,len(ids)):.0%}) | 含中国机构 {cn} ({cn/max(1,len(ids)):.0%})")


if __name__ == "__main__":
    main()
