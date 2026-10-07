"""整理固定LPR完整公告系列与股债联合反应所需来源，暂不生成标签。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_lpr_joint_response_20d_v1"
URLS = {
    "cfets_lpr_history": "https://www.chinamoney.com.cn/r/cms/chinese/chinamoney/html/currency/lpr-shibor-history-download.html",
    "cfets_lpr_main": "https://www.chinamoney.com.cn/chinese/bklpr/",
    "pboc_lpr_english": "https://www.pbc.gov.cn/en/3688229/3688335/3730276/3883798/19e15ae1-4.html",
    "chinabond_clock": "https://yield.chinabond.com.cn/cbweb-cbrc-web/cbrc/showCbrc",
    "shieh_method": "https://harrisonshieh.com/research/",
    "aea_information_shock": "https://www.aeaweb.org/articles?id=10.1257/mac.20180090",
}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def initialize():
    if OUT.exists():
        raise FileExistsError("本轮来源计划已存在，不能覆盖")
    for folder in ["raw", "evidence/acquisition", "inputs", "results", "code", "figures"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    save(OUT / "source_plan.json", {
        "created_at": now(), "study_id": "510300_LPR_JOINT_RESPONSE_20D_V1",
        "research_channel": "贷款定价变化、国债市场与股票联合反应是否提供当前持有判断的新信息",
        "lpr_universe": "2019-08至2026-07新机制全部月份；包括利率不变，禁止只挑降息；首月只作初始值。",
        "sources": URLS,
        "source_budget": "先取得六个固定说明/入口，再使用官方页面明确的公开历史接口、分页或Excel；保留所有请求和缺失。如需替代，先单列替代来源范围。",
        "lpr_expectations": "实际1年与5年以上报价和各自变化分别保留；没有真正事前共识时surprise为缺失，变化不命名为意外。",
        "yield": "继承中债1年/10年日曲线，研究固定只用10年；核对官方日终17:30发布说明，不放入15:00已知信息。",
        "fx": "中间价不冒充市场成交反应；本轮无合格成交汇率输入则不加入，不能阻断独立股债通道。",
        "author_calendar": "446个公告日期仅作线索，原日历未列2024-09-24首次宣布，不能自动当完整首发事件全集；Target/Path不入模型。",
        "deduplication": "旧政策操作量/DR007、M1M2五日、新订单20日、财政20日及85/15失败或终止继续保留；本轮不是重调上述窗口或方向。",
        "future_labels_read_in_this_source_stage": False, "new_models": 0, "new_accounts": 0,
        "whole_macro_objective_complete": False,
    })
    print("LPR完整月份与股债来源计划已保存，尚无标签或拟合。", flush=True)


def fetch(item):
    key, url = item
    receipt_path = OUT / "evidence/acquisition" / (key + ".json")
    if receipt_path.exists():
        raise FileExistsError("该请求已完成；不能覆盖历史回执")
    receipt = {"key": key, "url": url, "requested_at": now(), "method": "GET"}
    try:
        response = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html,application/json,*/*"})
        data = response.content
        path = OUT / "raw" / (key + ".html")
        path.write_bytes(data)
        receipt.update(status_code=response.status_code, final_url=response.url, bytes=len(data),
                       sha256=hashlib.sha256(data).hexdigest(), raw_path=path.relative_to(OUT).as_posix(),
                       content_type=response.headers.get("Content-Type", ""), completed_at=now())
        if response.status_code == 200:
            soup = BeautifulSoup(data, "html.parser")
            scripts = [urljoin(response.url, tag.get("src")) for tag in soup.find_all("script", src=True)]
            links = [{"text": tag.get_text(" ", strip=True), "url": urljoin(response.url, tag["href"])} for tag in soup.find_all("a", href=True)]
            save(OUT / "evidence/acquisition" / (key + "_links.json"), {"scripts": scripts, "links": links})
    except requests.RequestException as exc:
        receipt.update(error=str(exc), completed_at=now())
    save(receipt_path, receipt)
    return {"key": key, "status": receipt.get("status_code", "ERROR"), "bytes": receipt.get("bytes"), "error": receipt.get("error")}


def probe():
    if not (OUT / "source_plan.json").exists():
        raise FileNotFoundError("必须先保存来源计划")
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch, URLS.items()):
            print(json.dumps(result, ensure_ascii=False), flush=True)


def catalogs():
    prefix = "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125440/3876551/"
    items = [(f"pboc_lpr_cn_page_{i}", prefix + f"de24575c-{i}.html") for i in range(2, 6)]
    save(OUT / "evidence/acquisition/catalog_batch_plan.json", {"planned_at": now(), "basis": "已取得官方首页分页明确下一页2、末页5，共91条；固定取得2至5页，不按行情挑选。", "requests": items})
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch, items):
            print(json.dumps(result, ensure_ascii=False), flush=True)
    records = []
    for key in ["pboc_lpr_cn_index"] + [item[0] for item in items]:
        raw = OUT / "raw" / (key + ".html")
        if not raw.exists():
            raise ValueError("目录页缺失，不能继续当成完整母集")
        receipt = json.loads((OUT / "evidence/acquisition" / (key + ".json")).read_text(encoding="utf-8"))
        if receipt.get("status_code") != 200:
            raise ValueError("目录页非200，必须保留缺口")
        soup = BeautifulSoup(raw.read_bytes(), "html.parser")
        for tag in soup.find_all("a", href=True):
            title = unicodedata.normalize("NFKC", tag.get("title") or tag.get_text("", strip=True))
            if "贷款市场报价利率" not in title or ("公告" not in title and "报价行" not in title):
                continue
            match = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日.*贷款市场报价利率.*公告", title)
            date = None if not match else f"{int(match[1]):04d}-{int(match[2]):02d}-{int(match[3]):02d}"
            records.append({"title": title, "event_date": date, "month": None if date is None else date[:7],
                            "source_url": urljoin(prefix, tag["href"]), "catalog_key": key,
                            "included": bool(date is not None and "2019-08" <= date[:7] <= "2026-07")})
    frame = pd.DataFrame(records).drop_duplicates("source_url").sort_values("source_url")
    frame.to_csv(OUT / "lpr_catalog_all.csv", index=False, encoding="utf-8-sig")
    selected = frame[frame.included].sort_values("event_date").reset_index(drop=True)
    expected = set(pd.period_range("2019-08", "2026-07", freq="M").astype(str))
    save(OUT / "evidence/acquisition/catalog_coverage.json", {"records": len(frame), "selected": len(selected),
         "expected_months": len(expected), "missing_months": sorted(expected - set(selected.month)),
         "duplicate_months": selected.loc[selected.month.duplicated(False), "month"].tolist()})
    if len(selected) != len(expected) or set(selected.month) != expected or selected.month.duplicated().any():
        raise ValueError("固定84个月目录不完整或重复")
    selected.to_csv(OUT / "lpr_source_targets.csv", index=False, encoding="utf-8-sig")
    print(f"官方目录固定月份完整：{len(selected)}个月；不变月份没有删除。", flush=True)


def articles():
    frame = pd.read_csv(OUT / "lpr_source_targets.csv")
    items = [("lpr_" + row.month, row.source_url) for row in frame.itertuples()]
    save(OUT / "evidence/acquisition/article_batch_plan.json", {"planned_at": now(), "requests": items, "fixed_months": 84})
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(fetch, items), start=1):
            if i % 12 == 0 or result["status"] != 200:
                print(json.dumps({"completed": i, **result}, ensure_ascii=False), flush=True)


def parse():
    targets = pd.read_csv(OUT / "lpr_source_targets.csv")
    observations, failures = [], []
    for row in targets.itertuples():
        key = "lpr_" + row.month
        receipt = json.loads((OUT / "evidence/acquisition" / (key + ".json")).read_text(encoding="utf-8"))
        if receipt.get("status_code") != 200:
            failures.append({"month": row.month, "reason": "原文请求未成功", "key": key})
            continue
        path = OUT / receipt["raw_path"]
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        text = unicodedata.normalize("NFKC", re.sub(r"\s+", "", soup.get_text("", strip=True)))
        one = re.findall(r"1年期LPR为([0-9.]+)%", text)
        five = re.findall(r"5年期以上LPR为([0-9.]+)%", text)
        times = re.findall(r"(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}(?::\d{2})?)", text)
        published = [pd.Timestamp(day + " " + clock, tz="Asia/Shanghai") for day, clock in times]
        if len(set(one)) != 1 or len(set(five)) != 1:
            failures.append({"month": row.month, "reason": "正文没有唯一两期限报价", "one": one, "five": five, "key": key})
            continue
        conservative = pd.Timestamp(row.event_date + " 23:59:59", tz="Asia/Shanghai")
        available = max([conservative] + published)
        observations.append({"month": row.month, "event_date": row.event_date,
                             "available_at": available.isoformat(), "availability_rule": "MAX_OFFICIAL_EVENT_DAY_END_AND_DISPLAYED_TIMESTAMP",
                             "displayed_timestamps": json.dumps([x.isoformat() for x in published]),
                             "lpr_1y_percent": float(one[0]), "lpr_5y_percent": float(five[0]),
                             "consensus_1y": None, "consensus_5y": None, "surprise_1y": None, "surprise_5y": None,
                             "source_url": row.source_url, "raw_path": receipt["raw_path"], "raw_sha256": receipt["sha256"],
                             "historical_first_version_authenticated": False})
    out = pd.DataFrame(observations).sort_values("month")
    out.to_csv(OUT / "lpr_observations.csv", index=False, encoding="utf-8-sig")
    out.to_parquet(OUT / "lpr_observations.parquet", index=False)
    save(OUT / "evidence/acquisition/lpr_parse_status.json", {"parsed": len(out), "expected": 84, "failures": failures})
    print(json.dumps({"已解析月份": len(out), "缺口": failures}, ensure_ascii=False), flush=True)


def refine_clocks():
    target = OUT / "lpr_observations_precise.parquet"
    if target.exists() or (OUT / "freeze.json").exists():
        raise FileExistsError("精确公告时钟已建立或研究已冻结")
    prior = pd.read_parquet(OUT / "lpr_observations.parquet")
    refined = []
    for row in prior.to_dict("records"):
        soup = BeautifulSoup((OUT / row["raw_path"]).read_bytes(), "html.parser")
        plain = unicodedata.normalize("NFKC", re.sub(r"\s+", "", soup.get_text("", strip=True)))
        matches = re.findall(r"(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}(?::\d{2})?)", plain)
        accepted = [(day, clock) for day, clock in matches if day == row["event_date"]]
        row["date_upper_bound_available_at"] = row["available_at"]
        if len(set(accepted)) == 1 and len(matches) == 1:
            day, clock = accepted[0]
            moment = pd.Timestamp(day + " " + clock, tz="Asia/Shanghai")
            if len(clock) == 5:
                moment += pd.Timedelta(seconds=59)
            row["available_at"] = moment.isoformat()
            row["availability_rule"] = "OFFICIAL_DISPLAYED_TIMESTAMP_UPPER_BOUND"
            row["display_precision"] = "MINUTE_UPPER_BOUND" if len(clock) == 5 else "SECOND_REPORTED"
        else:
            row["display_precision"] = "DATE_UPPER_BOUND_NO_UNIQUE_MATCHING_TIMESTAMP"
        refined.append(row)
    frame = pd.DataFrame(refined)
    frame.to_csv(target.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    frame.to_parquet(target, index=False)
    save(OUT / "evidence/acquisition/clock_refinement_before_model_freeze.json", {
        "created_at": now(), "before_any_new_label_or_model": True,
        "reason": "所有原文已经取得，能使用匹配公告日的唯一精确显示时间；来源初探的较粗日期上界表原样保留，不能将当前9:00时间回填至历史9:30/9:15。",
        "records": len(frame), "precision_counts": frame.display_precision.value_counts().to_dict(),
        "first_full_session_rule": "股市开盘必须严格晚于公告时间上界；9:30恰好开盘的来源保守顺延下一交易日。",
        "historical_first_version_authenticated": False})
    print(json.dumps({"已细化公告时钟": len(frame), "精度": frame.display_precision.value_counts().to_dict()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPR与股债来源准备")
    parser.add_argument("action", choices=["initialize", "probe", "catalogs", "articles", "parse", "refine-clocks"])
    {"initialize": initialize, "probe": probe, "catalogs": catalogs, "articles": articles, "parse": parse, "refine-clocks": refine_clocks}[parser.parse_args().action]()
