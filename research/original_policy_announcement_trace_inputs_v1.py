"""原公告来源钟及共同来源关系；不把实施、转载或重复节点当新冲击。"""
from __future__ import annotations

import hashlib
import re

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

TZ = "Asia/Shanghai"


def clock(value):
    if value is None or pd.isna(value):
        return pd.NaT
    result = pd.Timestamp(value)
    return result.tz_localize(TZ) if result.tzinfo is None else result.tz_convert(TZ)


def clean(value):
    return re.sub(r"\s+", " ", str(value)).strip()


def source_document(raw):
    soup = BeautifulSoup(raw, "html.parser", from_encoding="utf-8")
    title = clean(soup.title.get_text(" ", strip=True)) if soup.title else ""
    body = soup.find(id="zoom") or soup.select_one("article.bdycont") or soup.select_one("div.pages_content")
    if body is None:
        body = soup.select_one("div.TRS_Editor")
    if body is None:
        raise ValueError("未识别官方正文结构，保留原文和未知。")
    text = clean(body.get_text(" ", strip=True))
    whole = soup.get_text(" ", strip=True)
    second = re.search(r"20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", whole)
    if second:
        available, precision = clock(second[0]), "SECOND_CURRENT_PAGE_NOT_FIRST_VINTAGE"
    else:
        label = soup.select_one(".artlabel")
        meta = soup.find("meta", attrs={"name": re.compile("PubDate|pubdate|publishdate", re.I)})
        dated = clean(label.get_text(" ", strip=True)) if label else str(meta.get("content", "")) if meta else ""
        match = re.search(r"20\d{2}-\d{2}-\d{2}", dated)
        if not match:
            match = re.search(r"20\d{2}-\d{2}-\d{2}", whole)
        if not match:
            available, precision = pd.NaT, "UNKNOWN_SOURCE_PUBLICATION_CLOCK"
        else:
            available = clock(match[0]) + pd.Timedelta(hours=23, minutes=59, seconds=59)
            precision = "DATE_END_UPPER_CURRENT_PAGE_NOT_FIRST_VINTAGE"
    return {"title": title, "text": text, "source_available_upper": available, "clock_precision": precision}


def navigation_family(text, capital=False):
    if capital:
        return "回购增持信贷" if "回购" in text else "互换便利"
    if "外汇风险准备金" in text:
        return "远期售汇风险准备金"
    if "外汇存款准备金" in text:
        return "外汇存款准备金"
    if "考核制度" in text or "平均法" in text:
        return "准备金考核制度"
    if "2016年度" in text or "2015年度" in text:
        return "定向准备金年度考核"
    if "境外人民币业务参加行" in text:
        return "境外人民币准备金制度"
    return "人民币准备金率"


def observed_rate(raw, record):
    source = source_document(raw)
    expected = clock(record["published_at"])
    value = float(record["seven_day_rate_percent"])
    before = record["previous_rate_percent"]
    tables = BeautifulSoup(raw, "html.parser", from_encoding="utf-8").find(id="zoom")
    if tables is None:
        raise ValueError("原利率操作正文不存在。")
    matched = []
    for row in tables.find_all("tr"):
        cells = [clean(cell.get_text(" ", strip=True)) for cell in row.find_all(["td", "th"])]
        if cells and re.fullmatch(r"7\s*天", cells[0]):
            rates = [float(match[1]) for cell in cells[1:] if (match := re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[%％]", re.sub(r"\s+", "", cell)))]
            matched.extend(rates)
    if len(matched) != 1 or not np.isclose(matched[0], value, rtol=0, atol=1e-12):
        raise ValueError("原公告7天中标利率与冻结记录不一致。")
    if pd.isna(source["source_available_upper"]) or source["source_available_upper"] != expected:
        raise ValueError("原公告实际页面钟与原记录不同。")
    prior_known = not pd.isna(before)
    return {**source, "observed_operation_date": pd.Timestamp(record["notice_date"]),
        "seven_day_rate_percent": value, "previous_observed_rate_percent": float(before) if prior_known else np.nan,
        "change_vs_previous_observed_bp": (value-float(before))*100 if prior_known else np.nan,
        "change_status": "PRIOR_OBSERVED_RECORD_KNOWN_NOT_FIRST_ANNOUNCEMENT_CERTIFICATION" if prior_known else "PRIOR_OPERATION_RATE_UNKNOWN",
        "event_role": "ACTUAL_OPERATION_NOTICE_NOT_ALL_EARLIER_POLICY_ANNOUNCEMENTS",
        "historical_first_vintage_authenticated": False}


def daily_known(frame, nodes):
    data = frame[["date", "decision_time"]].copy().reset_index(drop=True)
    data["date"] = pd.to_datetime(data.date).astype("datetime64[ns]")
    if not data.date.is_unique or not data.date.is_monotonic_increasing:
        raise ValueError("日历必须唯一、顺序一致。")
    required = {"node_id", "source_available_upper", "common_source_id", "economic_identity", "information_role"}
    if not required.issubset(nodes.columns) or not nodes.node_id.is_unique:
        raise ValueError("公告节点身份缺失或重复。")
    items = nodes.to_dict("records")
    previous = pd.NaT
    rows = []
    for date, decision_time in data.itertuples(index=False):
        decision = clock(decision_time)
        if pd.isna(decision) or decision.tz_localize(None).normalize() != date:
            raise ValueError("观察决定钟不属于该ETF日。")
        known = [item for item in items if not pd.isna(clock(item["source_available_upper"])) and clock(item["source_available_upper"]) <= decision]
        fresh = [item for item in known if not pd.isna(previous) and previous < clock(item["source_available_upper"]) <= decision]
        announcements = [item for item in fresh if item["information_role"] == "ANNOUNCEMENT"]
        rows.append({"date": date, "decision_time": decision, "known_recorded_nodes": len(known),
            "new_recorded_nodes": len(fresh), "new_node_ids": "|".join(item["node_id"] for item in fresh),
            "new_unique_common_sources": len({item["common_source_id"] for item in fresh}),
            "new_unique_economic_identities": len({item["economic_identity"] for item in fresh}),
            "new_recorded_announcement_nodes": len(announcements),
            "new_announcement_node_ids": "|".join(item["node_id"] for item in announcements),
            "coverage_status": "RECORDED_SOURCES_ONLY_NOT_ALL_POLICY_NEWS",
            "empty_record_means_policy_absent": False, "historical_first_vintage_authenticated": False})
        previous = decision
    return pd.DataFrame(rows)


def stable_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
