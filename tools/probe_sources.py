# -*- coding: utf-8 -*-
"""探测各学术数据源在当前网络下的可用性（第三轮扩展前的前置检查）。"""
import json, urllib.request, urllib.parse, urllib.error

MAIL = "cxcy.research@example.com"


def try_url(name, url, note=""):
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": f"cxcy-literature-survey/1.0 (mailto:{MAIL})",
            "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
        print(f"[OK  ] {name}: HTTP {r.status}  bytes={len(raw)}  {note}")
        return True, raw
    except urllib.error.HTTPError as e:
        print(f"[FAIL] {name}: HTTPError {e.code}  {note}")
        return False, None
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__} {e}  {note}")
        return False, None


def main():
    print("=== 数据源可用性探测 ===")
    try_url("OpenAlex (works, search)",
            "https://api.openalex.org/works?search=aquaculture&per-page=1&mailto=" + MAIL)
    try_url("Crossref (works)",
            "https://api.crossref.org/works?query.bibliographic=aquaculture&rows=1&mailto=" + MAIL)
    try_url("Semantic Scholar",
            "https://api.semanticscholar.org/graph/v1/paper/search?query=aquaculture&limit=1")
    try_url("arXiv API",
            "http://export.arxiv.org/api/query?search_query=all:aquaculture&max_results=1")
    try_url("DOAJ",
            "https://doaj.org/api/search/articles/aquaculture?pageSize=1")
    try_url("Europe PMC",
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=aquaculture&format=json&pageSize=1")
    print("=== 探测结束 ===")


if __name__ == "__main__":
    main()
