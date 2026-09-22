# -*- coding: utf-8 -*-
"""
工程落地向文献检索（第二批）——数据源：Crossref REST API。

为什么换源：OpenAlex 与本机出口 IP 之间触发持久 HTTP 429（单请求也 429），
Semantic Scholar 同样 429；Crossref 正常可用。

国别口径差异（重要，须在交付物中声明）：
- OpenAlex 直接给机构 country_code；Crossref 只给机构名称字符串，**无国别字段**。
- 本脚本按机构名称字符串做国别推断（见 CN_MARKERS），属于**启发式**，
  且部分记录完全没有机构信息 → 覆盖率无法达到 100%。因此本批的"国内"占比
  只在**有机构信息的记录**上统计，并在交付物中同时给出覆盖率。
"""
import json, os, re, time, hashlib, urllib.request, urllib.parse, urllib.error

BASE = r"E:\BJTU\CXCY\CXCY-PAPER"
CACHE = os.path.join(BASE, "tools", "_cache_cr")
os.makedirs(CACHE, exist_ok=True)
MAIL = "cxcy.research@example.com"

DIRECTIONS = {
    "E1": {"name": "养殖物联网感知与低功耗传感网络", "support": "D1/D3/D5", "queries": [
        "wireless sensor network aquaculture monitoring system",
        "low power IoT sensor node water quality",
        "LoRa aquaculture monitoring system design",
        "sensor network fish farm real-time monitoring",
        "energy harvesting wireless sensor aquaculture",
        "underwater wireless sensor network deployment",
        "smart fish pond IoT node implementation",
        "aquaculture water quality monitoring buoy design",
        "remote monitoring system shrimp pond implementation",
        "IoT based smart fishery system development"]},
    "E2": {"name": "养殖数字孪生与决策支持平台", "support": "D3/D5", "queries": [
        "digital twin aquaculture system",
        "decision support system aquaculture farm management",
        "smart aquaculture platform architecture design",
        "fishery information system design implementation",
        "aquaculture production management software platform",
        "aquaculture IoT cloud platform implementation",
        "fish farm management software development",
        "smart fishery data platform construction"]},
    "E3": {"name": "养殖数据质量治理：缺失插补与异常检测", "support": "新方向（支撑 D3）", "queries": [
        "missing data imputation water quality time series",
        "anomaly detection sensor data water monitoring",
        "data quality sensor network environmental monitoring",
        "outlier detection environmental time series deep learning",
        "sensor fault detection water quality monitoring system",
        "data cleaning low-cost sensor environmental data",
        "water quality sensor data imputation machine learning",
        "aquaculture sensor data anomaly detection system",
        "sensor drift correction water quality monitoring",
        "data preprocessing water quality monitoring network"]},
    "E4": {"name": "水下机器人与网箱巡检装备", "support": "新方向（支撑 D2）", "queries": [
        "underwater robot inspection aquaculture net cage",
        "net cleaning robot aquaculture",
        "ROV aquaculture inspection system",
        "autonomous underwater vehicle fish farm inspection",
        "biofouling removal net cage robot design",
        "cage culture robot system development",
        "underwater vehicle visual inspection implementation"]},
    "E5": {"name": "精准投喂装备与机电控制", "support": "D1", "queries": [
        "automatic feeding machine aquaculture control system",
        "feeding robot fish farm design",
        "pneumatic feeding system fish cage",
        "precision feeding equipment aquaculture development",
        "automatic feeder control system design aquaculture",
        "automatic feeding machine design pond culture",
        "feed spreader design aquaculture system",
        "intelligent feeding device aquaculture control"]},
    "E6": {"name": "模型轻量化与边缘部署工程", "support": "D1/D4/D5", "queries": [
        "knowledge distillation lightweight model edge deployment",
        "model quantization pruning real-time inference embedded",
        "edge inference optimization deep learning deployment",
        "TinyML microcontroller deep learning deployment",
        "lightweight deep learning model aquaculture edge device",
        "model compression deployment pipeline edge computing",
        "lightweight fish detection model embedded deployment",
        "edge computing real-time fish detection system",
        "lightweight underwater object detection model deployment",
        "mobile device fish disease detection lightweight model"]},
    "E7": {"name": "水产品溯源与供应链可信数据", "support": "新方向", "queries": [
        "seafood traceability blockchain system",
        "aquaculture supply chain traceability system design",
        "food traceability IoT blockchain implementation",
        "digital traceability fishery product supply chain",
        "aquatic product quality traceability platform",
        "fish supply chain traceability digital system"]},
    "E8": {"name": "无人机与遥感在养殖中的工程应用", "support": "D3", "queries": [
        "UAV remote sensing aquaculture mapping",
        "drone imagery fish pond monitoring",
        "satellite remote sensing aquaculture site selection",
        "unmanned aerial vehicle water quality monitoring",
        "remote sensing aquaculture area extraction mapping",
        "multispectral imagery pond aquaculture estimation"]},
    "E9": {"name": "水下声学感知与通信工程", "support": "D1/D2", "queries": [
        "underwater acoustic communication network",
        "hydroacoustic fish biomass estimation",
        "acoustic telemetry fish behavior monitoring",
        "underwater acoustic sensor localization",
        "fishery acoustics sonar survey method",
        "acoustic monitoring fish farm feeding system"]},
    "E10": {"name": "养殖 AI 系统可信性工程（MLOps / 漂移监测 / 幻觉审计）", "support": "新方向（支撑 D6）", "queries": [
        "machine learning model monitoring drift detection production",
        "MLOps deployment pipeline monitoring",
        "large language model hallucination detection mitigation",
        "explainable AI trustworthy machine learning deployment",
        "uncertainty quantification deep learning deployment",
        "model reliability degradation monitoring field deployment",
        "explainable artificial intelligence aquaculture prediction",
        "model drift detection water quality prediction system",
        "uncertainty quantification water quality prediction model",
        "trustworthy AI environmental monitoring deployment"]},
}

CN_MARKERS = [
    "china", "chinese", "p.r. china", "pr china", "beijing", "shanghai", "tianjin",
    "chongqing", "guangdong", "zhejiang", "jiangsu", "shandong", "fujian", "hunan",
    "hubei", "henan", "hebei", "anhui", "jiangxi", "sichuan", "yunnan", "guizhou",
    "shaanxi", "gansu", "liaoning", "jilin", "heilongjiang", "shanxi", "hainan",
    "guangxi", "ningxia", "xinjiang", "tibet", "inner mongolia", "taiwan",
    "hong kong", "macao", "macau", "zhanjiang", "qingdao", "dalian", "xiamen",
    "ningbo", "shenzhen", "guangzhou", "hangzhou", "nanjing", "wuhan", "chengdu",
    "suzhou", "wuxi", "fuzhou", "haikou", "sanya", "kunming", "changsha",
    "zhengzhou", "jinan", "hefei", "nanchang", "nanning", "guiyang", "lanzhou",
    "taiyuan", "shijiazhuang", "harbin", "changchun", "shenyang", "urumqi",
    "hohhot", "yinchuan", "xining", "lhasa", "yangling", "wenzhou", "yantai",
]

KEEP_TYPES = {"journal-article", "proceedings-article", "book-chapter", "posted-content", "report"}


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
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
            time.sleep(0.35)
            return d
        except urllib.error.HTTPError as e:
            last = e
            time.sleep(4 * (2 ** i))
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    print("  !! failed:", last, url[:110])
    return None


def cr_url(query, frm, rows):
    return "https://api.crossref.org/works?" + urllib.parse.urlencode({
        "query.bibliographic": query,
        "filter": f"from-pub-date:{frm},until-pub-date:2026-12-31",
        "rows": str(rows),
        "select": "DOI,title,author,issued,container-title,is-referenced-by-count,type,abstract",
        "mailto": MAIL,
    })


def parse(it):
    doi = (it.get("DOI") or "").strip().lower()
    if not doi:
        return None
    tp = it.get("type") or ""
    if tp not in KEEP_TYPES:
        return None
    ti = (it.get("title") or [""])
    title = (ti[0] if ti else "").strip()
    title = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", title))
    if not title:
        return None
    auths = it.get("author") or []
    names = []
    for a in auths:
        nm = " ".join(x for x in [a.get("given"), a.get("family")] if x).strip() or a.get("name") or ""
        if nm:
            names.append(nm)
    affs = []
    for a in auths:
        for af in (a.get("affiliation") or []):
            s = (af.get("name") or "").strip()
            if s:
                affs.append(s)
    first_aff = ""
    for a in auths:
        got = ""
        for af in (a.get("affiliation") or []):
            if af.get("name"):
                got = af["name"].strip()
                break
        if got:
            first_aff = got
            break
    blob = " | ".join(affs).lower()
    has_aff = bool(affs)
    cn = has_aff and any(m in blob for m in CN_MARKERS)
    cn_first = has_aff and any(m in first_aff.lower() for m in CN_MARKERS)
    yr = None
    for k in ("issued", "published", "published-online", "published-print", "created"):
        dp = ((it.get(k) or {}).get("date-parts") or [[None]])
        if dp and dp[0] and dp[0][0]:
            yr = dp[0][0]
            break
    venue = ((it.get("container-title") or [""]) or [""])[0].strip()
    return {
        "id": "doi:" + doi,
        "doi": doi,
        "title": title,
        "year": yr,
        "date": "",
        "type": tp,
        "venue": venue,
        "cited": it.get("is-referenced-by-count") or 0,
        "authors": names[:12],
        "n_authors": len(names),
        "first_country": "CN" if cn_first else "",
        "countries": [],
        "cn": bool(cn),
        "cn_first": bool(cn_first),
        "has_aff": has_aff,
        "topics": [],
    }


def main():
    store, per_dir, log = {}, {d: [] for d in DIRECTIONS}, []
    for did, spec in DIRECTIONS.items():
        for q in spec["queries"]:
            urls = [(cr_url(q, "2021-01-01", 100), "A"),
                    (cr_url(q + " China", "2024-01-01", 60), "B")]
            got = 0
            for u, tag in urls:
                d = fetch(u)
                if not d:
                    continue
                for it in (d.get("message", {}).get("items") or []):
                    r = parse(it)
                    if not r or r["id"] in store:
                        continue
                    store[r["id"]] = r
                    per_dir[did].append(r["id"])
                    got += 1
            log.append({"dir": did, "query": q, "new": got})
            print(f"[{did}] {q} -> +{got}", flush=True)

    # 记录“有多少条带机构信息、其中多少判为中国”
    allrec = list(store.values())
    with_aff = [r for r in allrec if r["has_aff"]]
    cn_in_aff = [r for r in with_aff if r["cn"]]
    print("\n=== 候选池 ===")
    print("唯一文献:", len(allrec))
    print(f"带机构信息: {len(with_aff)} ({len(with_aff)/max(1,len(allrec)):.1%}) | 其中判为中国: {len(cn_in_aff)}")
    for did, ids in per_dir.items():
        recs = [store[i] for i in ids]
        r3 = sum(1 for r in recs if (r["year"] or 0) >= 2024)
        wa = [r for r in recs if r["has_aff"]]
        cn = sum(1 for r in recs if r["cn"])
        print(f"{did}: {len(ids)} | 2024+ {r3} ({r3/max(1,len(ids)):.0%}) | 带机构 {len(wa)} | 中国 {cn}")

    json.dump({"records": store, "per_dir": per_dir, "log": log},
              open(os.path.join(BASE, "tools", "_eng_raw.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
