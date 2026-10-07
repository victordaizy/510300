"""取得LPR事前调查原始页面和事实台账；本阶段不读取股票收益。"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_lpr_prior_expectations_v1"
PARENT = ROOT / "reports/research/510300_lpr_joint_response_20d_v1"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def initialize():
    if (OUT / "source_plan.json").exists():
        raise FileExistsError("来源计划已存在，不能改写")
    for name in ["raw_local_only", "evidence/search", "evidence/acquisition", "facts", "code", "figures", "results", "inputs"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    actual = pd.read_parquet(PARENT / "inputs/lpr.parquet")
    actual.to_parquet(OUT / "inputs/official_actual_lpr.parquet", index=False)
    actual[["month", "event_date", "available_at", "lpr_1y_percent", "lpr_5y_percent", "source_url", "raw_sha256"]].to_csv(OUT / "facts/全部84月实际公告母集.csv", index=False, encoding="utf-8-sig")
    save(OUT / "source_plan.json", {
        "study_id": "510300_LPR_PRIOR_EXPECTATIONS_V1",
        "created_at": now(),
        "stage": "SOURCE_AND_PREANNOUNCEMENT_EXPECTATIONS_ADMISSION_BEFORE_MODEL_PROTOCOL",
        "universe": "2019-08至2026-07全部84个月新机制LPR；初次检索已看到2024-08/09及2026-07调查线索，但未计算新股票标签或拟合。",
        "question": "已知价格反应和实际政策变化之外，真正事前调查的分歧或兑现差异是否提供新增信息；不是重调旧LPR实际变化模型。",
        "source_priority": "Reuters自行实施的调查，优先其直接网页，再采用可核对Reuters署名的Investing、Yahoo、MarketScreener等正式转载；微信公众号只作可追溯线索或独立机构原发布的对照。",
        "search_rule": "按完整母集逐月固定英文Reuters China loan prime rate poll与月年检索；没有事前原文的月份最多再作一次定向来源检索。不挑利率变动月或后来出现行情的日子。所有月份保留。",
        "clock_admission": "网页有明确ISO时间优先；日期精度或时区不明则保守上界，必须早于当次官方公告。标题正文必须是事前调查预测，公布后的actual报道即使引用调查也不能冒充当时可知。网页首版的历史真实性仍须披露未独立认证。",
        "facts": "分期限记录受访人数、预计下调/不变/上调人数、明确点预测及其依据。报道只有模糊多数或区间且不可唯一推出时保持缺失。",
        "median_admission": "只用明确中位数或严格多数对应同一确切报价；多数预计降息但未给幅度不能推成10bp。",
        "cut_vote_definition": "预计降息人数/受访人数只表示调查票数比例；实际降息指示减该比例为调查兑现差异，不能等同风险中性市场概率或结构性政策冲击。",
        "selection": "按公布前最近一次含完整同口径调查内容的原文，若同一调查转载一致则按来源优先级；遇到版本冲突先保留，不按股价挑来源。",
        "permission_scope": "仅510300/CASH_CNY历史研究；20万元主、2万元成本对照；旧LPR、货币五日、新订单、财政、85/15和NBS失败或终止均保留。",
        "copyright_delivery": "新闻全文仅本地来源核对缓存，交付包保留URL、时间、内容哈希、数字事实和短摘录，不分发整篇Reuters新闻。",
        "new_equity_labels": 0, "new_models": 0, "new_accounts": 0,
        "whole_macro_objective_complete": False,
    })
    print("84个月事前调查来源计划已保存，尚未创建模型协议或股票标签。", flush=True)


def fetch(item):
    key, url = item["key"], item["url"]
    receipt_path = OUT / "evidence/acquisition" / (key + ".json")
    if receipt_path.exists():
        return {"key": key, "status": "EXISTING_REQUEST_RETAINED"}
    receipt = {"key": key, "url": url, "requested_at": now()}
    try:
        response = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/xhtml+xml,*/*"})
        raw = response.content
        path = OUT / "raw_local_only" / (key + ".html")
        path.write_bytes(raw)
        soup = BeautifulSoup(raw, "html.parser")
        ld = []
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                ld.append(json.loads(tag.string or tag.get_text()))
            except (json.JSONDecodeError, TypeError):
                continue
        meta = {tag.get("property", tag.get("name", "")): tag.get("content", "") for tag in soup.find_all("meta") if tag.get("content")}
        article = soup.find("div", class_=lambda v: v and ("article_WYSIWYG" in v or "articlePage" in v))
        if article is None:
            article = soup.find("article")
        body = (article or soup).get_text("\n", strip=True)
        text_path = OUT / "raw_local_only" / (key + ".txt")
        text_path.write_text(body, encoding="utf-8")
        receipt.update(status_code=response.status_code, final_url=response.url, bytes=len(raw),
                       sha256=hashlib.sha256(raw).hexdigest(), raw_path=path.relative_to(OUT).as_posix(),
                       text_path=text_path.relative_to(OUT).as_posix(),
                       title=soup.title.get_text(" ", strip=True) if soup.title else None,
                       metadata=meta, structured_data=ld,
                       contains_reuters="Reuters" in body, contains_survey="survey" in body.lower(),
                       catalogue_headers={k: v for k, v in response.headers.items() if k.lower() in {"x-wp-total", "x-wp-totalpages", "link", "content-type"}},
                       completed_at=now())
        save(receipt_path, receipt)
        return {"key": key, "status": response.status_code, "bytes": len(raw), "title": receipt["title"],
                "published_meta": {k: v for k, v in meta.items() if re.search("time|date|published", k, re.I)}}
    except requests.RequestException as exc:
        receipt.update(error=str(exc), completed_at=now())
        save(receipt_path, receipt)
        return {"key": key, "status": "ERROR", "error": str(exc)}


def batch(path):
    if not (OUT / "source_plan.json").exists():
        raise FileNotFoundError("尚未固定来源计划")
    items = json.loads(path.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row in pool.map(fetch, items):
            print(json.dumps(row, ensure_ascii=False), flush=True)


def catalogue():
    actual = pd.read_parquet(OUT / "inputs/official_actual_lpr.parquet").set_index("month")
    rows, detail, ids = [], [], set()
    requests_fixed = [(site, expected, page) for site, expected in [("kfgo", 256), ("wsau", 233)] for page in range(1, 4)]
    for site, expected, page in requests_fixed:
        path = OUT / "raw_local_only" / f"{site}_catalogue_page_{page}.html"
        receipt = json.loads((OUT / "evidence/acquisition" / f"{site}_catalogue_page_{page}.json").read_text(encoding="utf-8"))
        if receipt["status_code"] != 200 or receipt["catalogue_headers"]["X-WP-Total"] != str(expected):
            raise ValueError("目录范围或请求未完整")
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data:
            if (site, item["id"]) in ids:
                raise ValueError("目录分页出现重复，不能宣称完整")
            ids.add((site, item["id"]))
            published = pd.Timestamp(item["date_gmt"], tz="UTC").tz_convert("Asia/Shanghai")
            modified = pd.Timestamp(item["modified_gmt"], tz="UTC").tz_convert("Asia/Shanghai")
            month = published.strftime("%Y-%m")
            title = html.unescape(item["title"]["rendered"])
            paragraphs = [p.get_text(" ", strip=True) for p in BeautifulSoup(item["content"]["rendered"], "html.parser").find_all("p")]
            content = "\n".join(paragraphs)
            monthly_topic = bool(re.search(r"\bLPRs?\b|\b(?:loan|lending) (?:prime |benchmarks?\b|rates?\b)|\bbenchmark (?:lending |loan |rates?\b)|\bmortgage (?:reference )?rates?\b|\b(?:keep|keeps|leave|leaves|hold|holds) rates?\b", title, re.I))
            future_title = bool(re.search(r"expected|seen|likely|set to|poised|poll", title, re.I))
            quote_poll = "Reuters" in content and bool(re.search(r"survey|poll", content, re.I))
            before = month in actual.index and published < pd.Timestamp(actual.loc[month, "available_at"])
            version_before = before and modified < pd.Timestamp(actual.loc[month, "available_at"])
            status = "CANDIDATE_PRE_RELEASE_POLL" if monthly_topic and future_title and quote_poll and version_before else "NOT_SELECTED_NONMONTHLY_OR_POST_RELEASE"
            if monthly_topic and future_title and quote_poll and before and not version_before:
                status = "EXCLUDED_MODIFIED_AFTER_RELEASE"
            record = {"id": item["id"], "source_site": site, "uid": f"{site}_{item['id']}", "month": month, "published_at": published.isoformat(), "modified_at": modified.isoformat(),
                      "title": title, "source_url": item["link"], "page": page,
                      "page_sha256": receipt["sha256"], "content_sha256": hashlib.sha256(item["content"]["rendered"].encode()).hexdigest(),
                      "monthly_lpr_topic": monthly_topic, "future_title": future_title, "contains_reuters_survey": quote_poll,
                      "published_before_actual": before, "current_version_before_actual": version_before, "candidate_status": status}
            rows.append(record)
            if status == "CANDIDATE_PRE_RELEASE_POLL":
                detail.append(dict(record, paragraphs=paragraphs, focused_paragraphs=[p for p in paragraphs if re.search(r"respond|particip|survey|poll|basis point|expected|forecast|predic", p, re.I)]))
    if len(rows) != 489:
        raise ValueError("目录总数与分页宣称不一致")
    frame = pd.DataFrame(rows).sort_values(["published_at", "id"])
    frame.to_csv(OUT / "facts/全部489条公开目录.csv", index=False, encoding="utf-8-sig")
    frame.to_parquet(OUT / "facts/全部489条公开目录.parquet", index=False)
    candidates = frame[frame.candidate_status == "CANDIDATE_PRE_RELEASE_POLL"]
    candidates.to_csv(OUT / "facts/事前调查候选目录.csv", index=False, encoding="utf-8-sig")
    save(OUT / "raw_local_only/候选调查正文及重点.json", sorted(detail, key=lambda x: x["published_at"]))
    present = set(candidates.month)
    save(OUT / "evidence/目录准入汇总.json", {"created_at": now(), "total": 489,
        "candidate_articles": len(candidates), "candidate_months": len(present),
        "missing_months": [m for m in actual.index if m not in present],
        "duplicates_in_pagination": 0, "catalogue_order_note": "网站没有按查询要求的日期顺序返回，分别取得两个站点的三页、共489项并验证站内唯一后按UTC时间排序；跨站转载不算新调查。",
        "admitted_numeric_survey_months": 0, "candidate_titles_not_final_admission": True, "new_equity_labels": 0, "new_models": 0})
    print(json.dumps({"全部目录": 489, "候选文章": len(candidates), "候选月份": len(present), "缺失月份": [m for m in actual.index if m not in present],
                     "同月多个候选": candidates.month.value_counts()[lambda x: x > 1].to_dict()}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPR事前调查来源取得")
    parser.add_argument("action", choices=["initialize", "fetch", "catalogue"])
    parser.add_argument("--targets", type=Path)
    args = parser.parse_args()
    if args.action == "initialize":
        initialize()
    elif args.action == "catalogue":
        catalogue()
    else:
        batch(args.targets)
