# tools/ · 文献检索链路与缓存

> 本目录存放**文献检索的可复现链路**：检索脚本、筛选脚本、产出生成脚本，以及 API 响应缓存。
> **本目录是工具层，不是交付物层** —— 交付物在 `../doc/` 与 `../project/`。

---

## ⚠️ 先读这一段：缓存不在项目目录里

```
tools/_cache*/        ← OpenAlex / Crossref 的原始 API 响应（几百个 json）
```

**缓存根在本目录，不在 `project/*/` 内。删掉任何 `project/` 下的项目目录，都不会清掉它。**
`project/*/_source/*.json` 只是从缓存里**抽出来的派生产物**。

**⇒ 要彻底清理检索痕迹，必须同时删 `tools/_cache*`；要复现旧结果，也依赖这些缓存。**

---

## 检索链路

脚本按**执行顺序**排列，前一个的输出是后一个的输入：

### 第一批（D1–D6，OpenAlex）

```
search_openalex.py  →  select_papers.py  →  gen_outputs.py  →  gen_topic_md.py
   检索 + 落缓存        相关性筛选            生成总表/csv/bib     生成分方向 md
```

产出落在 `../doc/`。

### 第二批（E1–E10，Crossref — 因 OpenAlex 429）

```
eng_search.py / eng_search_cr.py  →  eng_select.py  →  eng_gen.py  →  eng_topic_md.py
      检索（Crossref）                 筛选          生成汇总        生成分方向 md
```

产出落在 `../doc/工程落地补充/`。

### 第三批（F1–F16，OpenAlex 已恢复）

```
f_search.py  →  f_select.py
   检索           筛选        → 输出 _f_raw.json
```

### F1 专用（从缓存抽取）

| 脚本 | 作用 | 状态 |
|---|---|---|
| `f1_harvest.py` | 从 `_cache_f/` 抽取 290 条 → `project/F1/_source/f1_papers.json` | 已运行 |
| `f1_fetch.py` | **22 条更窄的补充检索式** | ⚠️ **一条也没跑**（2026-09-20 两次探测 OpenAlex 均 HTTP 429） |

### 辅助

| 脚本 | 作用 |
|---|---|
| `gen_competitor.py` | 生成竞品分析表（产出 `../竞品分析表.md` / `.csv`） |
| `probe_sources.py` | 探测各数据源可用性（用于判断 429 状态） |

---

## 缓存目录对照

| 目录 | 对应批次 | 数据源 |
|---|---|---|
| `_cache/` | 第一批 D1–D6 | OpenAlex |
| `_cache_cr/` | 第二批 E1–E10 | Crossref |
| `_cache_eng/` | 第二批（工程落地，早期批次） | — |
| `_cache_f/` | 第三批 F1–F16 + F1 抽取源 | OpenAlex |

---

## 复现方式

```bash
PY="C:/Users/Chen/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
cd "E:/BJTU/CXCY/CXCY-PAPER/tools"

# 探测数据源是否可用（先跑这个，避免白跑）
"$PY" -u probe_sources.py

# 第一批：检索 → 筛选 → 产出
"$PY" -u search_openalex.py && "$PY" -u select_papers.py \
  && "$PY" -u gen_outputs.py && "$PY" -u gen_topic_md.py

# 第三批：F1–F16
"$PY" -u f_search.py && "$PY" -u f_select.py
```

> **有缓存时会直接命中缓存**，不会重复请求 API —— 这也是为什么旧结果仍可复现。

---

## ⚠️ 本机环境注意（实测踩过）

- **前台跑批量网络抓取会被 SIGTERM** ⇒ 整条流水线串成**一个后台任务**（`run_in_background`）。
- **Python 必须加 `-u`**，否则进程被杀时日志为空。
- **Bash 工具 PATH 残缺**（`ls`/`cat`/`cp`/`head`/`tail`/`dirname` 全不可用）⇒ 文件操作走 Python，
  输出重定向到文件后用 Read 读。
- **⚠️ 不要在前台串行跑 17 个脚本**：超时会 auto-background，中途状态难判断。
- **不抓 CNKI** —— 中文期刊覆盖不足是**已知且必须如实声明**的边界，不是待办。

---

## 表述红线（与全工作区一致）

1. **「我没检索到」≠「不存在」** —— 禁止把检索范围当成事实边界。
2. 所有否定性结论必须给**举证范围 + 举证日期**，并附「未见披露 ≠ 不存在」。
3. **不得**把本目录的检索结果表述为"覆盖了中文期刊"。
4. 与上一轮相同的检索式**不代表相同结果** —— 数据源在变（本目录已因 429 换过一次源），
   每次引用**必须写明数据源与日期**。

---

## 相关

- 产出物说明与两批口径差异：`../doc/README.md`
- 工作区总入口：`../README.md`
