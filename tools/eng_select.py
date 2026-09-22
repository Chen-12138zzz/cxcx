# -*- coding: utf-8 -*-
"""
工程落地向文献：相关性门禁 + 配额配平 + 与第一批 133 篇去重 + 工程落地度标记。
目标：新增 >= 300 篇；近三年 >= 70%；国内 >= 50%（含中国机构与中国第一作者双口径）。
"""
import json, os, re

BASE = r"E:\BJTU\CXCY\CXCY-PAPER"

RAW = json.load(open(os.path.join(BASE, "tools", "_eng_raw.json"), encoding="utf-8"))
STORE, PER_DIR = RAW["records"], RAW["per_dir"]

# 与第一批 133 篇去重：本批 id 形如 "doi:xxx"，故按 DOI 比对
OLD, OLD_DOI = set(), set()
_selp = os.path.join(BASE, "tools", "_selected.json")
_rawp = os.path.join(BASE, "tools", "_papers_raw.json")
if os.path.exists(_selp):
    _sel = json.load(open(_selp, encoding="utf-8"))
    for _, ids in _sel.items():
        OLD.update(ids)
    if os.path.exists(_rawp):
        _oldrec = json.load(open(_rawp, encoding="utf-8"))["records"]
        for _, ids in _sel.items():
            for i in ids:
                d = (_oldrec.get(i) or {}).get("doi") or ""
                if d:
                    OLD_DOI.add(d.strip().lower())

DIR_NAMES = {
    "E1": "养殖物联网感知与低功耗传感网络",
    "E2": "养殖数字孪生与决策支持平台",
    "E3": "养殖数据质量治理：缺失插补与异常检测",
    "E4": "水下机器人与网箱巡检装备",
    "E5": "精准投喂装备与机电控制",
    "E6": "模型轻量化与边缘部署工程",
    "E7": "水产品溯源与供应链可信数据",
    "E8": "无人机与遥感在养殖中的工程应用",
    "E9": "水下声学感知与通信工程",
    "E10": "养殖 AI 系统可信性工程（MLOps / 漂移监测 / 幻觉审计）",
}
DIR_SUPPORT = {
    "E1": "支撑 D1/D3/D5", "E2": "支撑 D3/D5", "E3": "新方向（支撑 D3）",
    "E4": "新方向（支撑 D2）", "E5": "支撑 D1", "E6": "支撑 D1/D4/D5",
    "E7": "新方向", "E8": "支撑 D3", "E9": "支撑 D1/D2",
    "E10": "新方向（支撑 D6）",
}

DOMAIN = ["fish", "shrimp", "prawn", "aquacultur", "aquatic", "crab", "oyster", "mollusc",
          "shellfish", "tilapia", "salmon", "trout", "seabream", "seabass", "lobster",
          "sea cucumber", "seaweed", "kelp", "fishery", "fisheries", "pond", "cage",
          "mariculture", "hatchery", "marine", "ocean", "coastal", "water", "aquafarm",
          "hydroacoustic", "seafood", "fisher", "shoal", "reef", "estuar", "lake",
          "reservoir", "agricultur", "farm", "food", "livestock"]

GATE = {
    "E1": [["sensor", "iot", "internet of things", "lora", "wireless", "node", "telemetry",
            "monitoring network", "zigbee", "nb-iot", "5g", "remote monitoring",
            "data acquisition", "gateway", "smart buoy", "buoy"], DOMAIN],
    "E2": [["digital twin", "decision support", "platform", "information system",
            "management system", "architecture", "dashboard", "software", "data management",
            "visualization", "enterprise resource", "system design", "framework"], DOMAIN],
    "E3": [["imputation", "missing data", "anomaly detection", "outlier", "data quality",
            "fault detection", "data cleaning", "noise reduction", "calibration",
            "uncertainty", "data-driven", "gap filling"], DOMAIN],
    "E4": [["robot", "rov", "auv", "autonomous underwater", "manipulator", "inspection",
            "crawler", "remotely operated", "net cleaning", "biofouling", "docking",
            "mechanical design", "prototype"], DOMAIN],
    "E5": [["feeder", "feeding machine", "feeding system", "feed dispenser", "actuator",
            "control system", "mechatronic", "dispensing", "dosing", "pneumatic",
            "conveyor", "vibratory", "machine design", "automatic feeding"], DOMAIN],
    "E6": [["distillation", "quantization", "quantized", "pruning", "lightweight", "tinyml",
            "microcontroller", "embedded", "edge computing", "edge device", "inference",
            "compression", "acceleration", "on-device", "fpga", "npu", "jetson",
            "model optimization", "real-time deployment"], DOMAIN],
    "E7": [["traceability", "blockchain", "supply chain", "provenance", "origin authentication",
            "food safety", "certification", "digital passport", "tamper"], DOMAIN],
    "E8": [["uav", "drone", "unmanned aerial", "satellite", "remote sensing", "multispectral",
            "hyperspectral", "sentinel-2", "landsat", "aerial imagery", "mapping",
            "site selection", "spatial analysis"], DOMAIN],
    "E9": [["acoustic", "sonar", "hydroacoustic", "echosound", "underwater communication",
            "telemetry", "passive acoustic", "spectrogram", "sound", "ultrasonic"], DOMAIN],
    "E10": [["mlops", "drift", "monitoring", "hallucination", "trustworthy", "explainable",
             "uncertainty quantification", "reliability", "degradation",
             "production deployment", "maintenance", "governance", "audit",
             "model monitoring", "robustness"], DOMAIN],
}

EXCLUDE = ["carotenoid", "metaverse", "railway", "orb-slam", "text data augmentation",
           "optical metrology", "brain tumor", "parkinson", "alzheimer", "pothole",
           "spatial transcriptomics", "alphafold", "microplastic", "crop yield", "cotton",
           "maize", "apple disease", "plant disease", "shelf life", "heavy metal",
           "marketing", "soil", "greenhouse gas", "antibiotic", "walrus", "metaheuristic",
           "bitcoin", "cryptocurrency", "electric vehicle", "autonomous driving",
           "traffic prediction", "smart city", "concrete", "bridge damage", "wind turbine",
           "photovoltaic", "stock market", "social media", "covid", "fruit", "vegetable",
           "dairy", "poultry", "pig", "cattle", "wheat", "rice", "forest fire", "landslide",
           "earthquake", "flood forecasting", "urban drainage", "drinking water treatment",
           "membrane bioreactor", "activated sludge", "hospital", "wearable", "emotion",
           # 第二批新增噪声词
           "retraction", "cold vortex", "microphysical", "structural health monitoring",
           "wind farm", "offshore wind", "tunnel", "packet-reliability", "nmpc",
           "gravity dam", "persons-in-water", "trajectory prediction", "thermal effluent",
           "elkhorn coral", "halal", "milk", "textile", "logistics", "e-commerce",
           "credit risk", "supply chain finance", "carbon emission", "coal mine",
           "natural gas", "petroleum", "water meter", "crop", "vegetation", "potato"]

# 方向专属排除
EXCLUDE_BY_DIR = {
    "E4": ["secrecy", "ofdm", "tdma", "protocol", "capacity", "relay",
           "channel estimation", "multiple access", "modulation"],
    "E2": ["structural health", "dam", "bridge", "building"],
}

QUOTA = dict(r_cnf=26, r_cn=6, r_in=12, o_cn=4, o_in=4)

ENG_STRONG = ["system", "systematic design", "platform", "framework", "design and implementation",
              "implementation", "prototype", "deploy", "field trial", "field test",
              "case study", "application", "architecture", "pipeline", "tool", "device",
              "instrument", "apparatus", "practical", "industrial", "engineering",
              "integration", "real-world", "in-situ", "on-site", "experiment",
              "development", "construction", "monitoring station", "testbed", "setup",
              "equipment", "automatic", "control", "smart", "low-cost", "cost-effective"]


BLOCK_TITLES = {         # 人工复核后剔除（标题归一化前缀）
    "areviewofdeeptransferlearning",          # E6：通用迁移学习综述，非养殖
    "scenedataaugmentationforunderwater",      # E2：与数字孪生平台无关
    "anenhancedyolo",                          # E2：与数字孪生平台无关
    "adaptiveselfsupervisedlearning",          # E2：与数字孪生平台无关
    "waterqualitymonitoringandearlywarning",   # E8：水质预警，非遥感
    "alayeredapproachforvaluecreationan",      # E8：渔业价值链，非遥感
}
BLOCK_KEYS = BLOCK_TITLES


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


def eng_flag(title):
    t = (title or "").lower()
    return 1 if any(k in t for k in ENG_STRONG) else 0


def select(did, ids):
    seen, uniq = set(), []
    for i in ids:
        if i in OLD:
            continue
        r = STORE[i]
        if r.get("doi") and r["doi"].strip().lower() in OLD_DOI:
            continue
        k = tkey(r["title"])
        if not k or k in seen or not gate_ok(did, r["title"]):
            continue
        if any(k.startswith(b) for b in BLOCK_KEYS):
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
        s, pool = select(did, PER_DIR[did])
        sel_all[did] = s
        n = len(s)
        stats.append(dict(did=did, name=DIR_NAMES[did], support=DIR_SUPPORT[did], n=n, pool=pool,
                          rec=sum(1 for r in s if (r["year"] or 0) >= 2024),
                          cn=sum(1 for r in s if r["cn"]),
                          cnf=sum(1 for r in s if r["cn_first"]),
                          aff=sum(1 for r in s if r.get("has_aff")),
                          eng=sum(1 for r in s if eng_flag(r["title"]))))

    T = dict(n=sum(s["n"] for s in stats), rec=sum(s["rec"] for s in stats),
             cn=sum(s["cn"] for s in stats), cnf=sum(s["cnf"] for s in stats),
             aff=sum(s["aff"] for s in stats),
             eng=sum(s["eng"] for s in stats))

    with open(os.path.join(BASE, "tools", "_eng_review.txt"), "w", encoding="utf-8") as f:
        for did in DIR_NAMES:
            f.write(f"\n##### {did} {DIR_NAMES[did]} #####\n")
            for i, r in enumerate(sel_all[did], 1):
                org = "国内" if r["cn_first"] else ("国内(合作)" if r["cn"] else "国外")
                f.write(f"{i:2d}|{r['year']}|{org}|eng={eng_flag(r['title'])}|{r['cited']:5d}| {r['title']}\n")

    json.dump({d: [r["id"] for r in v] for d, v in sel_all.items()},
              open(os.path.join(BASE, "tools", "_eng_selected.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    for s in stats:
        print(f"{s['did']} {s['name']} [{s['support']}]: 入选 {s['n']} / 候选 {s['pool']} | "
              f"2024+ {s['rec']} ({s['rec']/max(1,s['n']):.0%}) | 国内 {s['cn']} "
              f"({s['cn']/max(1,s['n']):.0%}) | 国内一作 {s['cnf']} | 工程落地标记 {s['eng']}")
    print(f"\n新增合计 {T['n']} | 近三年 {T['rec']} ({T['rec']/max(1,T['n']):.1%}) "
          f"| 国内 {T['cn']} ({T['cn']/max(1,T['n']):.1%}) | 国内一作 {T['cnf']} ({T['cnf']/max(1,T['n']):.1%}) "
          f"| 工程落地标记 {T['eng']} ({T['eng']/max(1,T['n']):.1%})")
    json.dump({"stats": stats, "total": T}, open(os.path.join(BASE, "tools", "_eng_summary.json"),
                                                 "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
