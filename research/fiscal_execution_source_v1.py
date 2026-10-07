"""取得财政部月度执行原文；不读取行情、不计算标签或选择策略。"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_fiscal_execution_state_20d_v1"
BASE = "https://bgt.mof.gov.cn/zhuantilanmu/rdwyh/czszsj/"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).replace("—", "-").replace("–", "-")


def acquire(key, url):
    receipt = OUT / "evidence/acquisition" / (key + ".json")
    if receipt.exists():
        return json.loads(receipt.read_text(encoding="utf-8"))
    attempts = []
    for scheme in ("https", "http"):
        target = re.sub(r"^https?", scheme, url)
        raw = OUT / "raw" / (key + "_" + scheme + ".html")
        raw.parent.mkdir(parents=True, exist_ok=True)
        call = subprocess.run(["curl.exe", "-L", "--connect-timeout", "8", "--max-time", "30", "-o", str(raw), "-w", "%{http_code}", target], capture_output=True)
        data = raw.read_bytes() if raw.exists() else b""
        text = data.decode("utf-8", "replace")
        valid = call.returncode == 0 and call.stdout.strip() == b"200" and "财政" in text and "Bad Gateway" not in text and len(data) > 4000
        attempts.append({"url": target, "returncode": call.returncode, "http_status": call.stdout.decode("ascii", "replace"), "raw": str(raw.relative_to(OUT)), "sha256": sha(raw) if raw.exists() else None, "bytes": len(data), "admitted": valid, "stderr": call.stderr.decode("utf-8", "replace")[-300:]})
        if valid:
            break
    result = {"key": key, "requested_url": url, "retrieved_at": now(), "attempts": attempts, "status": "ACQUIRED" if attempts[-1]["admitted"] else "NO_SOURCE", "raw": attempts[-1]["raw"] if attempts[-1]["admitted"] else None}
    save(receipt, result)
    return result


def period(title):
    s = norm(title)
    m = re.fullmatch(r"(20\d{2})年(.*?)财政收支情况", s)
    if not m:
        return None
    y, rest = int(m[1]), m[2]
    if rest == "":
        month = 12
    elif rest in ("一季度", "第一季度"):
        month = 3
    elif rest in ("上半年", "1-6月"):
        month = 6
    elif rest in ("前三季度", "1-9月"):
        month = 9
    else:
        mm = re.fullmatch(r"(?:1-)?(\d{1,2})月份?", rest)
        if not mm:
            return None
        month = int(mm[1])
    return f"{y:04}-{month:02}"


def catalogs():
    if not OUT.exists():
        save(OUT / "data_acquisition_plan.json", {"created_at": now(), "source": BASE, "fixed_catalog_pages": 16, "target_reference_start": "2015-02", "target_reference_end": "2026-07", "exclude_reference_january": "财政部常合并1—2月；不拆分、不用后来数值倒填1月", "source_priority": "财政部代表委员栏目当月原文，保留本页日期与正文日期，取较晚的明确日期23:59:59为保守可用上界", "features_not_frozen": True, "market_read": False, "research_duplicate_check": "research/*.py与RESEARCH_STATUS.md未发现财政收支执行独立模型；已核对政策数量、直接效用、增长新订单失败研究，不能救回或改名复试。", "not_consensus": True})
    tasks = [(f"catalog_{i:02}", BASE + ("" if i == 0 else f"index_{i}.htm")) for i in range(16)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda x: acquire(*x), tasks))
    rows = []
    for result in results:
        if result["status"] != "ACQUIRED":
            continue
        soup = BeautifulSoup((OUT / result["raw"]).read_text(encoding="utf-8"), "html.parser")
        for a in soup.find_all("a", href=True):
            title = a.get_text("", strip=True)
            p = period(title)
            if p:
                parent = a.parent.get_text(" ", strip=True)
                dates = re.findall(r"20\d{2}-\d{2}-\d{2}", parent)
                rows.append({"catalog": result["key"], "title": title, "reference_period": p, "catalog_date": dates[-1] if dates else None, "url": urljoin(result["requested_url"], a["href"])})
    frame = pd.DataFrame(rows).drop_duplicates(["url"])
    frame.to_csv(OUT / "catalog_all.csv", index=False, encoding="utf-8-sig")
    target = frame[(frame.reference_period >= "2015-02") & (frame.reference_period <= "2026-07") & ~frame.reference_period.str.endswith("-01")].sort_values(["reference_period", "catalog_date", "url"])
    if target.reference_period.duplicated().any():
        raise ValueError("同一所属期多个来源，先核对，禁止自动选值")
    target.to_csv(OUT / "source_targets.csv", index=False, encoding="utf-8-sig")
    expected = [str(p) for p in pd.period_range("2015-02", "2026-07", freq="M") if p.month != 1]
    report = {"catalogs_acquired": sum(r["status"] == "ACQUIRED" for r in results), "all_records": len(frame), "target_records": len(target), "missing_periods": sorted(set(expected) - set(target.reference_period)), "model_fits": 0}
    save(OUT / "evidence/catalog_coverage.json", report)
    print(json.dumps(report, ensure_ascii=False))


def articles():
    frame = pd.read_csv(OUT / "source_targets.csv")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda x: acquire("fiscal_" + x.reference_period, x.url), frame.itertuples()))
    print(json.dumps({"原文成功": sum(r["status"] == "ACQUIRED" for r in results), "原文失败": [r["key"] for r in results if r["status"] != "ACQUIRED"]}, ensure_ascii=False))


def supplement():
    base = "https://www.mof.gov.cn/zhengwuxinxi/redianzhuanti/quanguocaizhengshouzhiqingkuang/"
    save(OUT / "evidence/catalog_extension_plan.json", {"created_at": now(), "reason": "代表委员目录6页当前取得失败，且部分所属月目录未列出；用财政部全国财政收支专题完整9页补齐连续母集。仅来源补齐，未读取标签。", "pages": 9, "url": base, "model_fits": 0})
    tasks = [(f"topic_{i:02}", base + ("" if i == 0 else f"index_{i}.html")) for i in range(9)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda x: acquire(*x), tasks))
    rows = []
    for result in results:
        if result["status"] != "ACQUIRED":
            continue
        soup = BeautifulSoup((OUT / result["raw"]).read_text(encoding="utf-8"), "html.parser")
        for a in soup.find_all("a", href=True):
            title = a.get_text("", strip=True)
            p = period(title)
            if p:
                dates = re.findall(r"20\d{2}-\d{2}-\d{2}", a.parent.get_text(" ", strip=True))
                rows.append({"catalog": result["key"], "title": title, "reference_period": p, "catalog_date": dates[-1] if dates else None, "url": urljoin(result["requested_url"], a["href"])})
    topic = pd.DataFrame(rows).drop_duplicates(["url"])
    topic.to_csv(OUT / "catalog_topic_all.csv", index=False, encoding="utf-8-sig")
    existing = pd.read_csv(OUT / "source_targets.csv")
    target = topic[(topic.reference_period >= "2015-02") & (topic.reference_period <= "2026-07") & ~topic.reference_period.str.endswith("-01")].sort_values(["reference_period", "catalog_date", "url"])
    if target.reference_period.duplicated().any():
        raise ValueError("专题目录同期重复需核对")
    all_sources = pd.concat([existing, target], ignore_index=True).drop_duplicates(["url"])
    all_sources.to_csv(OUT / "source_candidates.csv", index=False, encoding="utf-8-sig")
    tasks = [("topic_fiscal_" + x.reference_period, x.url) for x in target.itertuples() if not (OUT / "evidence/acquisition" / ("fiscal_" + x.reference_period + ".json")).exists() or json.loads((OUT / "evidence/acquisition" / ("fiscal_" + x.reference_period + ".json")).read_text(encoding="utf-8"))["status"] != "ACQUIRED"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        acquired = list(pool.map(lambda x: acquire(*x), tasks))
    expected = [str(p) for p in pd.period_range("2015-02", "2026-07", freq="M") if p.month != 1]
    coverage = {"topic_catalogs_acquired": sum(r["status"] == "ACQUIRED" for r in results), "topic_records": len(topic), "missing_catalog_periods": sorted(set(expected) - set(all_sources.reference_period)), "supplement_acquired": sum(r["status"] == "ACQUIRED" for r in acquired), "supplement_failed": [r["key"] for r in acquired if r["status"] != "ACQUIRED"]}
    save(OUT / "evidence/supplement_coverage.json", coverage)
    print(json.dumps(coverage, ensure_ascii=False))


def parse_articles():
    sources = pd.read_csv(OUT / "source_candidates.csv")
    rows, failures = [], []
    for p in [str(x) for x in pd.period_range("2015-02", "2026-07", freq="M") if x.month != 1]:
        admitted = []
        for prefix in ("fiscal_", "topic_fiscal_", "repair_fiscal_", "search_fiscal_", "web_fiscal_", "complete_fiscal_", "alternate_fiscal_"):
            receipt = OUT / "evidence/acquisition" / (prefix + p + ".json")
            if receipt.exists():
                r = json.loads(receipt.read_text(encoding="utf-8"))
                if r["status"] == "ACQUIRED":
                    admitted.append(r)
        if not admitted:
            failures.append({"reference_period": p, "reason": "NO_SOURCE"})
            continue
        r = admitted[0]
        raw_text = (OUT / r["raw"]).read_text(encoding="utf-8")
        if r.get("format") == "WEB_TOOL_RENDERED_TEXT_NOT_RAW_HTTP_HTML":
            import html
            lines = [re.sub(r"^L\d+:\s*", "", s) for s in raw_text.splitlines() if re.match(r"^L\d+:", s)]
            if not lines:
                header = re.search(r"\n#{1,4} 20\d{2}年[^\n]+", raw_text)
                if header is None:
                    raise ValueError("网页工具结果没有完整正文起点，不能用搜索摘要填值")
                lines = raw_text[header.start():].splitlines()
            lines = [re.sub(r"cite.*?", "", s) for s in lines]
            soup = BeautifulSoup("".join("<p>" + html.escape(s) + "</p>" for s in lines), "html.parser")
        else:
            soup = BeautifulSoup(raw_text, "html.parser")
        paragraphs = [norm(t.get_text("", strip=True)).replace("全国政府性基金支出", "全国政府性基金预算支出") for t in soup.find_all("p")]
        full = norm(soup.get_text(" ", strip=True))
        # 正文单独字段，不以网址中的创建日代替公开日。
        dates = []
        for t in soup.find_all(["h2", "h3"]):
            dates.extend(re.findall(r"20\d{2}-\d{2}-\d{2}", t.get_text(" ", strip=True)))
        for t in paragraphs:
            m = re.match(r"(?:发布日期[:：]?)?(20\d{2})年(\d{1,2})月(\d{1,2})日(?:来源|$)", t)
            if m:
                dates.append(f"{int(m[1]):04}-{int(m[2]):02}-{int(m[3]):02}")
        for meta in soup.find_all("meta"):
            if meta.get("name", "").lower() in ("pubdate", "publishdate", "articledate"):
                dates.extend(re.findall(r"20\d{2}-\d{2}-\d{2}", meta.get("content", "")))
        if "ndrc.gov.cn" in r["requested_url"]:
            for node in soup.find_all(class_=re.compile(r"time|sj|date", re.I)):
                dates.extend(re.findall(r"20\d{2}-\d{2}-\d{2}", node.get_text(" ", strip=True)))
        candidates = sources[sources.url.str.replace("http://", "https://", regex=False) == r["requested_url"].replace("http://", "https://")]
        catalog_dates = [str(x) for x in candidates.catalog_date.dropna()]
        dates.extend(catalog_dates)
        dates = sorted(set(dates))
        if not dates:
            failures.append({"reference_period": p, "reason": "NO_PUBLICATION_DATE", "raw": r["raw"]})
            continue
        item = {"reference_period": p, "published_date": dates[-1], "available_at": dates[-1] + "T23:59:59+08:00", "date_precision": "DAY_UPPER_BOUND", "date_evidence": "|".join(dates), "source_url": r["requested_url"], "raw": r["raw"], "source_format": r.get("format", "RAW_HTTP_HTML"), "raw_sha256": sha(OUT / r["raw"]), "retrieved_at": r["retrieved_at"], "consensus": None, "surprise": None, "historical_first_version_authenticated": False}
        try:
            for kind, label in (("general", "全国一般公共预算支出"), ("fund", "全国政府性基金预算支出")):
                matched = [s for s in paragraphs if label in s and re.search(label + r"[\d,.]+亿元", s)]
                if not matched:
                    matched = re.findall(r"(?:1-\d{1,2}月累计[,，]|20\d{2}年[,，])" + label + r"[^。]+。", full)
                # 有单月数与累计数时，只取正文明确列明的累计段。
                cumulative = [s for s in matched if re.search(r"1[-至]\d{1,2}月(?:累计)?", s[:s.index(label)]) or re.search(r"(?:20\d{2}年|上半年|前三季度|一季度)[,，]?$", s[:s.index(label)])]
                selected = cumulative
                if len(selected) != 1:
                    raise ValueError(f"{kind}累计段数量{len(selected)}")
                paragraph = selected[0]
                m = re.search(label + r"([\d,.]+)亿元(?:\([^)]*\))?[,，](.*?)(?:。|分中央|其中)", paragraph)
                if not m:
                    raise ValueError(f"{kind}支出段格式")
                amount, clause = float(m[1].replace(",", "")), m[2]
                ys = re.findall(r"(增长|下降|减少)(\d+(?:\.\d+)?)(%|倍)", clause)
                if not ys:
                    if "持平" in clause:
                        headline = 0.
                    else:
                        raise ValueError(f"{kind}同比数缺失")
                else:
                    headline = float(ys[0][1]) * (-1 if ys[0][0] != "增长" else 1) * (100 if ys[0][2] == "倍" else 1)
                comparable = re.search(r"同口径(?:\[\d+\])?(增长|下降|减少)(\d+(?:\.\d+)?)%", clause)
                adjusted = float(comparable[2]) * (-1 if comparable[1] != "增长" else 1) if comparable else None
                item.update({kind + "_expenditure_ytd_yi": amount, kind + "_headline_yoy_pp": headline, kind + "_comparable_yoy_pp": adjusted, kind + "_model_yoy_pp": adjusted if adjusted is not None else headline, kind + "_basis": "CONTEMPORANEOUS_COMPARABLE" if adjusted is not None else "CONTEMPORANEOUS_HEADLINE", kind + "_evidence_paragraph": paragraph})
            # 正文中保留的统计调整说明便于逐条复核，不能由模型结果决定是否剔除。
            notes = [s for s in paragraphs if any(w in s for w in ["转列", "同口径", "整理期", "口径调整"]) and len(s) < 3000]
            item["scope_notes"] = "\n".join(notes)
            if not (p < item["published_date"][:7] <= str(pd.Period(p, freq="M") + 3)):
                raise ValueError("公布所属月或延迟超过三个月需人工核对")
            rows.append(item)
        except ValueError as e:
            failures.append({"reference_period": p, "reason": str(e), "raw": r["raw"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "fiscal_observations.csv", index=False, encoding="utf-8-sig")
    frame.to_parquet(OUT / "fiscal_observations.parquet", index=False)
    save(OUT / "evidence/parsing_failures.json", failures)
    print(json.dumps({"解析成功": len(frame), "缺失或需核对": failures, "同口径行数": int(frame.general_comparable_yoy_pp.notna().sum()) if len(frame) else 0}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="财政执行原文取得，不运行模型")
    parser.add_argument("action", choices=["catalogs", "articles", "supplement", "parse"])
    args = parser.parse_args()
    {"catalogs": catalogs, "articles": articles, "supplement": supplement, "parse": parse_articles}[args.action]()
