"""官方货币政策目录与已有宣布链的因果时钟；目录事件日不能充当公告日。"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import pandas as pd

BASE = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125963/index.html"
TZ = "Asia/Shanghai"
TOOLS = {
    "准备金率": [r"准备金率", r"降准"],
    "政策及操作利率": [r"政策利率", r"逆回购.*?利率", r"公开市场.*?利率"],
    "贷款报价": [r"贷款市场报价利率", r"LPR"],
    "结构信贷": [r"再贷款", r"再贴现", r"抵押补充贷款"],
    "资本市场工具": [r"互换便利", r"股票回购", r"增持再贷款"],
    "外汇约束": [r"外汇.*?准备金", r"外汇风险准备金", r"跨境融资宏观审慎"],
    "中期借贷操作": [r"中期借贷便利", r"MLF"],
}


def clean(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def compact(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def index_rows(raw, url):
    soup = BeautifulSoup(raw, "html.parser", from_encoding="utf-8")
    rows = []
    for a in soup.select("a[istitle='true'][href]"):
        title = clean(a.get_text(" ", strip=True))
        if "中国货币政策大事记" not in title:
            raise ValueError("固定官方目录列表出现不同主题。")
        link = urljoin(url, a["href"])
        if urlparse(link).hostname != "www.pbc.gov.cn":
            raise ValueError("官方目录文章跳出原主站。")
        cell = a.find_parent("td")
        text = cell.get_text(" ", strip=True) if cell is not None else ""
        match = re.search(r"20\d{2}-\d{2}-\d{2}", text)
        if match is None:
            raise ValueError("官方列表没有文章公布日期。")
        rows.append({"title": title, "url": link, "listed_published_date": match[0]})
    if not rows:
        raise ValueError("官方目录没有固定标题链接。")
    return rows


def index_plan(raw):
    soup = BeautifulSoup(raw, "html.parser", from_encoding="utf-8")
    page = soup.find("input", attrs={"name": "article_paging_list_hidden"})
    if page is None:
        raise ValueError("官方分页元数据缺失。")
    module, pages = str(page["moduleid"]), int(page["totalpage"])
    text = soup.get_text(" ", strip=True)
    count = re.search(r"总记录数\s*[:：]\s*(\d+)", text)
    if count is None or pages < 1 or pages > 5 or module != "17134":
        raise ValueError("固定官方目录范围或模块改变。")
    urls = [BASE] + [urljoin(BASE, f"{module}-{i}.html") for i in range(2, pages+1)]
    return {"module": module, "pages": pages, "advertised_records": int(count[1]), "urls": urls}


def period(title):
    year_match = re.match(r"^(20\d{2})年", title)
    if year_match is None:
        raise ValueError("目录年份不明。")
    year = int(year_match[1])
    if "第一季度" in title:
        start, end, role = f"{year}-01-01", f"{year}-03-31", "第一季度"
    elif "第二季度" in title:
        start, end, role = f"{year}-04-01", f"{year}-06-30", "第二季度"
    elif "第三季度" in title:
        start, end, role = f"{year}-07-01", f"{year}-09-30", "第三季度"
    elif "第四季度" in title:
        start, end, role = f"{year}-10-01", f"{year}-12-31", "第四季度"
    elif "上半年" in title:
        start, end, role = f"{year}-01-01", f"{year}-06-30", "上半年"
    elif "前三季度" in title:
        start, end, role = f"{year}-01-01", f"{year}-09-30", "前三季度"
    elif title == f"{year}年中国货币政策大事记":
        start, end, role = f"{year}-01-01", f"{year}-12-31", "全年"
    else:
        raise ValueError("未固定的目录期间标题。")
    return {"catalog_year": year, "reference_start": start, "reference_end": end, "reference_role": role,
        "scope_boundary": "标题列示的回顾期间；不代表全国政策全集或逐事件首次公开。"}


def tools(text):
    return "|".join(name for name, patterns in TOOLS.items() if any(re.search(pattern, text, re.I) for pattern in patterns)) or "其他"


def parse_document(raw, item):
    soup = BeautifulSoup(raw, "html.parser", from_encoding="utf-8")
    headline = clean(soup.title.get_text()) if soup.title is not None else ""
    if item["title"] not in headline:
        raise ValueError("原文标题与目录身份不一致。")
    body = soup.find(id="zoom")
    if body is None:
        body = next((node for node in soup.find_all("div") if "content" in node.get("class", []) and re.search(r"\d+月\d+日", node.get_text())), None)
    if body is None:
        raise ValueError("原文正文结构未识别，保留失败。")
    text = unicodedata.normalize("NFKC", body.get_text("\n", strip=True))
    page_text = soup.get_text(" ", strip=True)
    clock = re.search(r"20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", page_text)
    if clock is not None:
        published = pd.Timestamp(clock[0], tz=TZ)
        precision = "SECOND_CURRENT_PAGE_NOT_FIRST_VINTAGE"
    else:
        published = pd.Timestamp(item["listed_published_date"], tz=TZ) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        precision = "LISTED_DATE_END_UPPER_NOT_FIRST_VINTAGE"
    if str(published.date()) != item["listed_published_date"]:
        raise ValueError("官方原文与目录公布日期不一致。")
    matches = list(re.finditer(r"(?:^|\n)\s*(\d{1,2})月(\d{1,2})日", text))
    if not matches:
        raise ValueError("正文没有完整日期条目。")
    prefix = text[:matches[0].start()].strip()
    records = []
    year = item["catalog_year"]
    for i, match in enumerate(matches):
        paragraph = clean(text[match.start():matches[i+1].start() if i+1 < len(matches) else len(text)])
        event_date = pd.Timestamp(year=year, month=int(match[1]), day=int(match[2]))
        key = hashlib.sha256((event_date.isoformat()+"|"+compact(paragraph)).encode("utf-8")).hexdigest()
        records.append({"catalog_id": item["catalog_id"], "paragraph_number": i+1, "catalog_event_date": event_date,
            "logical_record_id": key, "text": paragraph, "navigation_tool_tags": tools(paragraph),
            "multiple_date_tokens": len(re.findall(r"\d+月\d+日", paragraph)), "source_available_upper": published,
            "source_clock_precision": precision, "source_url": item["url"],
            "record_role": "RETROSPECTIVE_LEAD_ONLY_NOT_ORIGINAL_ANNOUNCEMENT", "historical_first_vintage_authenticated": False})
    return records, {"source_available_upper": published, "clock_precision": precision,
        "paragraphs": len(records), "body_prefix_before_first_date": prefix, "body_text": text}


def known_chain_daily(observed, nodes):
    data = observed.copy()
    data["date"] = pd.to_datetime(data.date).astype("datetime64[ns]")
    if not data.date.is_unique or not data.date.is_monotonic_increasing:
        raise ValueError("原决定日历未唯一排序。")
    chain = nodes.copy()
    chain["available"] = pd.to_datetime(chain.source_available_upper, utc=True).dt.tz_convert(TZ)
    chain = chain.sort_values(["available", "node_id"])
    if not chain.node_id.is_unique:
        raise ValueError("原政策链节点身份重复。")
    previous, rows = None, []
    for row in data.itertuples(index=False):
        at = pd.Timestamp(row.decision_time).tz_convert(TZ)
        known = chain.loc[chain.available.le(at)]
        new = known.iloc[:0] if previous is None else known.loc[known.available.gt(previous)]
        rows.append({"date": row.date, "decision_time": at, "stage_entry_type": row.stage_entry_type,
            "recorded_known_chain_nodes": len(known), "recorded_new_chain_nodes": len(new),
            "recorded_new_chain_ids": "|".join(new.node_id), "recorded_new_channels": "|".join(sorted(set(new.channel))),
            "recorded_new_stages": "|".join(sorted(set(new.stage))),
            "recorded_latest_chain_clock": known.available.iloc[-1] if len(known) else pd.NaT,
            "recorded_latest_chain_ids": "|".join(known.loc[known.available.eq(known.available.max()), "node_id"]) if len(known) else None,
            "news_completeness_status": "PARTIAL_24_MANUALLY_VERIFIED_CHAIN_NODES_NOT_ALL_POLICY_NEWS",
            "no_record_is_no_policy": False})
        previous = at
    result = pd.DataFrame(rows)
    result["recorded_latest_chain_clock"] = pd.to_datetime(result.recorded_latest_chain_clock, utc=True).dt.tz_convert(TZ).astype("datetime64[ns, Asia/Shanghai]")
    result["recorded_latest_chain_ids"] = result.recorded_latest_chain_ids.astype("string")
    return result
