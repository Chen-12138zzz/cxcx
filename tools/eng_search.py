# -*- coding: utf-8 -*-
"""
工程落地向文献检索（第二批）：OpenAlex 真实元数据，带本地 JSON 缓存。
与第一批 133 篇按 OpenAlex ID 去重，保证"300 篇是新增的"。
"""
import json, os, time, hashlib, urllib.request, urllib.parse, urllib.error

BASE = r"E:\BJTU\CXCY\CXCY-PAPER"
CACHE = os.path.join(BASE, "tools", "_cache_eng")
os.makedirs(CACHE, exist_ok=True)
MAIL = "cxcy.research@example.com"
SEL = ("id,doi,title,publication_year,publication_date,authorships,"
       "primary_location,cited_by_count,type,language,topics,is_retracted")

DIRECTIONS = {
    "E1": {
        "name": "养殖物联网感知与低功耗传感网络",
        "support": "D1/D3/D5",
        "queries": [
            "wireless sensor network aquaculture monitoring system",
            "low power IoT sensor node water quality",
            "LoRa aquaculture monitoring system design",
            "sensor network fish farm real-time monitoring",
            "energy harvesting wireless sensor aquaculture",
            "underwater wireless sensor network deployment",
            "smart fish pond IoT node implementation",
        ],
    },
    "E2": {
        "name": "养殖数字孪生与决策支持平台",
        "support": "D3/D5",
        "queries": [
            "digital twin aquaculture system",
            "decision support system aquaculture farm management",
            "smart aquaculture platform architecture design",
            "fishery information system design implementation",
            "precision aquaculture management information platform",
            "aquaculture production management software platform",
        ],
    },
    "E3": {
        "name": "养殖数据质量治理：缺失插补与异常检测",
        "support": "D3 新方向",
        "queries": [
            "missing data imputation water quality time series",
            "anomaly detection sensor data water monitoring",
            "data quality sensor network environmental monitoring",
            "outlier detection environmental time series deep learning",
            "sensor fault detection water quality monitoring system",
            "data cleaning low-cost sensor environmental data",
        ],
    },
    "E4": {
        "name": "水下机器人与网箱巡检装备",
        "support": "D2 新方向",
        "queries": [
            "underwater robot inspection aquaculture net cage",
            "net cleaning robot aquaculture",
            "ROV aquaculture inspection system",
            "autonomous underwater vehicle fish farm inspection",
            "biofouling removal net cage robot design",
            "aquaculture cage inspection robot prototype",
        ],
    },
    "E5": {
        "name": "精准投喂装备与机电控制",
        "support": "D1",
        "queries": [
            "automatic feeding machine aquaculture control system",
            "feeding robot fish farm design",
            "pneumatic feeding system fish cage",
            "precision feeding equipment aquaculture development",
            "automatic feeder control system design aquaculture",
            "feed dispensing control system pond aquaculture",
        ],
    },
    "E6": {
        "name": "模型轻量化与边缘部署工程",
        "support": "D1/D4/D5",
        "queries": [
            "knowledge distillation lightweight model edge deployment",
            "model quantization pruning real-time inference embedded",
            "edge inference optimization deep learning deployment",
            "TinyML microcontroller deep learning deployment",
            "neural network acceleration edge device aquaculture",
            "model compression deployment pipeline edge computing",
        ],
    },
    "E7": {
        "name": "水产品溯源与供应链可信数据",
        "support": "新方向",
        "queries": [
            "seafood traceability blockchain system",
            "aquaculture supply chain traceability system design",
            "food traceability IoT blockchain implementation",
            "digital traceability fishery product supply chain",
            "aquatic product quality traceability platform",
        ],
    },
    "E8": {
        "name": "无人机与遥感在养殖中的工程应用",
        "support": "D3",
        "queries": [
            "UAV remote sensing aquaculture mapping",
            "drone imagery fish pond monitoring",
            "satellite remote sensing aquaculture site selection",
            "unmanned aerial vehicle water quality monitoring",
            "remote sensing aquaculture area extraction mapping",
        ],
    },
    "E9": {
        "name": "水下声学感知与通信工程",
        "support": "D1/D2",
        "queries": [
            "underwater acoustic communication network",
            "hydroacoustic fish biomass estimation",
            "acoustic telemetry fish behavior monitoring",
            "underwater acoustic sensor localization",
            "fishery acoustics sonar survey method",
        ],
    },
    "E10": {
        "name": "养殖 AI 系统可信性工程（MLOps / 漂移监测 / 幻觉审计）",
        "support": "D6 新方向",
        "queries": [
            "machine learning model monitoring drift detection production",
            "MLOps deployment pipeline monitoring",
            "large language model hallucination detection mitigation",
            "explainable AI trustworthy machine learning deployment",
            "uncertainty quantification deep learning deployment",
            "model reliability degradation monitoring field deployment",
        ],
    },
}


def fetch(url, tries=6):
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
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode("utf-8"))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
            time.sleep(1.0)          # 限流保护：OpenAlex polite pool
            return d
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 403, 503):
                time.sleep(8 * (2 ** i))    # 8/16/32/64/128/256 秒退避
            else:
                time.sleep(2 * (i + 1))
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    print("  !! failed:", last, url[:110])
    return None


def build_url(search, filters, per_page):
    return "https://api.openalex.org/works?" + urllib.parse.urlencode({
        "search": search, "filter": filters, "per-page": per_page,
        "sort": "relevance_score:desc", "select": SEL, "mailto": MAIL})


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
        "first_country": first_cc,
        "countries": sorted(countries),
        "cn": ("CN" in countries),
        "cn_first": (first_cc == "CN"),
        "topics": [t.get("display_name") for t in (w.get("topics") or [])[:2]],
    }


def main():
    store, per_dir, log = {}, {d: [] for d in DIRECTIONS}, []
    for did, spec in DIRECTIONS.items():
        for q in spec["queries"]:
            ua = build_url(q, "from_publication_date:2021-01-01,"
                              "type:article|review|proceedings-article,is_retracted:false", 50)
            da = fetch(ua)
            ub = build_url(q, "from_publication_date:2024-01-01,institutions.country_code:CN,"
                              "type:article|review|proceedings-article,is_retracted:false", 25)
            db = fetch(ub)
            got = 0
            for d in (da, db):
                if not d:
                    continue
                for w in d.get("results", []):
                    r = parse(w)
                    if not r["title"] or not r["id"] or r["id"] in store:
                        continue
                    store[r["id"]] = r
                    per_dir[did].append(r["id"])
                    got += 1
            log.append({"dir": did, "query": q, "new": got})
            print(f"[{did}] {q} -> +{got}")

    json.dump({"records": store, "per_dir": per_dir, "log": log},
              open(os.path.join(BASE, "tools", "_eng_raw.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n=== 候选池 ===")
    print("唯一文献:", len(store))
    for did, ids in per_dir.items():
        recs = [store[i] for i in ids]
        r3 = sum(1 for r in recs if (r["year"] or 0) >= 2024)
        cn = sum(1 for r in recs if r["cn"])
        print(f"{did} {DIRECTIONS[did]['name']}: {len(ids)} | 2024+ {r3} ({r3/max(1,len(ids)):.0%}) | 中国机构 {cn} ({cn/max(1,len(ids)):.0%})")


if __name__ == "__main__":
    main()
