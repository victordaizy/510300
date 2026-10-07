"""整理官方政策目录、首次公开与实施链；不拟合预测、不运行账户。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_policy_information_clock_v1"
TZ = ZoneInfo("Asia/Shanghai")
SEEDS = {
    "pboc_2024": "https://jrj.sh.gov.cn/SCGK194/20250214/65f4da70183342d08300c459502e0b9d.html",
    "pboc_2025": "https://jrj.sh.gov.cn/SCGK194/20260211/ca47d0d49e9e4cdab07a5fd2d1f1f747.html",
    "scio_2024": "https://english.scio.gov.cn/2024ChinaSCIOPressBriefings/node_9014600.html",
    "scio_2025": "https://english.scio.gov.cn/node_9018021.html",
}
CHANNEL_WORDS = {
    "增长与消费": ["consum", "trade-in", "equipment", "economic performance", "macroeconomic", "national development and reform", "industry", "industrial", "两新", "消费", "设备更新", "科技创新"],
    "通胀": ["inflation", "consumer price", "producer price", "price level", "通胀", "物价"],
    "利率与流动性": ["financial support", "financial sector", "people's bank", "monetary", "再贷款", "再贴现", "准备金", "利率", "流动性", "MLF", "LPR", "逆回购", "货币政策"],
    "财政与债务": ["fiscal", "taxation", "ministry of finance", "财政", "债务"],
    "地产": ["housing", "property", "房地产", "住房", "房贷"],
    "资本市场": ["capital market", "securities", "股票", "互换便利", "股市"],
    "外部与贸易": ["trade", "commerce", "import", "export", "foreign exchange", "foreign investment", "贸易", "外汇", "跨境", "本币互换", "境外", "关税"],
}
CHAIN_SOURCES = {
    "rate_20240125_gov": "https://app.www.gov.cn/govdata/gov/202401/25/511522/article.html",
    "joint_20240924_csrc": "https://www.csrc.gov.cn/csrc/c106311/c7508374/content.shtml",
    "rate_20240927_govwechat": "https://app.www.gov.cn/govdata/gov/202409/27/519932/article.html",
    "joint_20250507_csrc": "https://www.csrc.gov.cn/csrc/c106311/c7555758/content.shtml",
    "sfisf_20241018_csrc": "https://www.csrc.gov.cn/csrc/c100028/c7513113/content.shtml",
    "sfisf_20241231_csrc": "https://www.csrc.gov.cn/csrc/c100028/c7529480/content.shtml",
    "sfisf_20250102_pboc": "https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/5481510/5555069/index.html",
    "fiscal_20241012_mof": "https://m.mof.gov.cn/czxw/202410/t20241012_3945410.htm",
    "debt_20241109_mof": "https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/202411/t20241109_3947230.htm",
    "debt_20250110_mof": "https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/202501/t20250110_3951525.htm",
    "housing_20240517_pboc": "https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212554091417/index.html",
    "consumer_20240725_govwechat": "https://app.www.gov.cn/govdata/gov/202407/25/517625/article.html",
    "consumer_20240725_ndrc": "https://www.ndrc.gov.cn/xwdt/wszb/jlzcdgmsbxfpyjhx/wzsl/202407/t20240725_1391949.html",
    "consumer_20250108_samr": "https://www.samr.gov.cn/xw/xwfbt/art/2025/art_0be3bc00dedb4243ae8274425e821085.html",
    "trade_20250404_mof": "https://gss.mof.gov.cn/gzdt/zhengcefabu/202504/t20250404_3961451.htm",
    "trade_20250409_mof": "https://gss.mof.gov.cn/gzdt/zhengcefabu/202504/t20250409_3961684.htm",
    "trade_20250411_mof": "https://gss.mof.gov.cn/gzdt/zhengcejiedu/202504/t20250411_3961824.htm",
    "trade_20250512_mofcom": "https://www.mofcom.gov.cn/syxwfb/art/2025/art_9748270381e54001bc291213ec6ee778.html",
}


def now() -> str:
    return datetime.now(TZ).isoformat()


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p: Path, obj: object) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def prepare() -> None:
    if (OUT / "protocol.json").exists():
        raise FileExistsError("本轮范围已固定，不覆盖已有协议")
    for d in ("inputs", "raw", "receipts", "results", "evidence", "figures", "code", "history"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {
        "study_id": "510300_POLICY_INFORMATION_CLOCK_V1", "frozen_at": now(),
        "objective": "以官方目录建立跨通道政策线索母集，并区分宣布、细则、生效与执行进度，支持持续更新研究。",
        "stage": "数据与时钟研究，不检验任何新政策收益模型；所有已观察历史均非独立留出。",
        "catalog_period": ["2024-01-01", "2025-12-31"], "catalog_sources": SEEDS,
        "catalog_inclusion": "两年SCIO英文官方年度发布会目录的全部发布会入口，以及两年PBOC年度大事记全部有日期条目。不因市场涨跌筛选。",
        "completeness_boundary": "仅证明固定官方目录提取完整；目录并非全国全部政策，回顾文章日期不是历史首次公开时点。SCIO英文翻译时间不是中文首发时间。",
        "channels": ["增长与消费", "通胀", "利率与流动性", "财政与债务", "地产", "资本市场", "外部与贸易", "其他"],
        "catalog_role": "全部记录只作追溯线索；会议开始、文件落款、目录事件日期均不能自动进入可交易信息账本。",
        "chain_role": "人工核对若干跨通道政策链，逐条保存来源定位、先前已知内容、新增内容和剩余未知；不代表全部目录已核实。",
        "expectation_rule": "只有可核对的事前原始共识才计算预期差；实际政策变化、较前次变化、政策目标、事后超预期评价均不能替代。缺失为UNKNOWN，不为零。",
        "information_clock": "有秒用秒；仅分钟用该分钟59秒；日期精度用23:59:59。确认内容可知后的首个15:00收盘复核，下一交易日开盘作为统一可成交起点。实际实施日另列。",
        "market_overlay": "只画完整日频510300价格及预先列明政策链；不计算政策事件前后收益排序，不把同日共振归因于单项政策。",
        "state_context": "当时最新已公开增长状态、最新货币口径、实际政策利率和截至复核收盘的价格状态分别记录；仅作为描述，不从失败PMI模型产生新交易。",
        "new_model_fits": 0, "new_accounts": 0, "independent_forward_events": 0,
        "capital_main": 200000, "capital_cost_comparison": 20000,
        "next_gate": "政策机制及目标另行固定；有足够可核实时点与比较基准后再检验。资金缺口不封锁其他独立通道。",
        "prior_terminal_families_preserved": ["SELECTED_MIX_BAND10_SIMPLE2", "NBS_FIXED_5MIN_FAMILY", "MONEY_CONSENSUS_INCREMENT_V2", "GROWTH_STATE_INCREMENT_20D_V1"],
    })
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print("政策目录与时钟研究范围已固定。")


def fetch(key: str, url: str) -> dict:
    receipt = OUT / "receipts" / f"{key}.json"
    target = OUT / "raw" / f"{key}.html"
    if receipt.exists():
        result = json.loads(receipt.read_text(encoding="utf-8"))
        if target.exists() and sha(target) == result.get("sha256") and result.get("http_code") == 200:
            return result
        raise RuntimeError(f"已有失败或身份不同的来源，保留原回执后使用新key：{key}")
    started = now()
    cmd = ["curl.exe", "--silent", "--show-error", "--location", "--connect-timeout", "10", "--max-time", "40", "--output", str(target), "--write-out", "%{json}", url]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)
    try:
        meta = json.loads(proc.stdout)
    except json.JSONDecodeError:
        meta = {}
    result = {"key": key, "url": url, "retrieved_at": started, "finished_at": now(), "curl_exit": proc.returncode,
              "http_code": meta.get("http_code"), "url_effective": meta.get("url_effective"), "raw_path": f"raw/{key}.html",
              "bytes": target.stat().st_size if target.exists() else 0, "sha256": sha(target) if target.exists() else None,
              "error": proc.stderr[:500], "historical_version_authenticated": False}
    save(receipt, result)
    return result


def fetch_seeds() -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(lambda kv: fetch(*kv), SEEDS.items()):
            print(json.dumps({k: r[k] for k in ("key", "http_code", "bytes", "error")}, ensure_ascii=False))


def soup_for(key: str) -> BeautifulSoup:
    receipt = json.loads((OUT / "receipts" / f"{key}.json").read_text(encoding="utf-8"))
    p = OUT / receipt["raw_path"]
    if receipt["http_code"] != 200 or sha(p) != receipt["sha256"]:
        raise ValueError(f"来源不可用或身份变化：{key}")
    return BeautifulSoup(p.read_bytes(), "html.parser", from_encoding="utf-8")


def channels(text: str) -> str:
    text = text.casefold()
    found = []
    for name, words in CHANNEL_WORDS.items():
        searched = text.replace("trade-ins", "").replace("trade-in", "") if name == "外部与贸易" else text
        if any(w.casefold() in searched for w in words):
            found.append(name)
    return "|".join(found) if found else "其他"


def parse_catalogs() -> None:
    """所有目录行均保留；词表仅用于导航，不能充当经济影响评分。"""
    rows = []
    counts = {}
    months = {name: i + 1 for i, name in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
    for year in (2024, 2025):
        key = f"pboc_{year}"
        source = soup_for(key)
        paragraphs = [unicodedata.normalize("NFKC", p.get_text("", strip=True)) for p in source.find_all("p")]
        paragraphs = [p for p in paragraphs if re.match(r"^\d+月\d+日", p)]
        for i, text in enumerate(paragraphs, 1):
            m = re.match(r"^(\d+)月(\d+)日", text)
            assert m is not None
            date = f"{year}-{int(m[1]):02}-{int(m[2]):02}"
            rows.append({"catalog_id": f"PBOC_{year}_{i:03}", "source_key": key, "catalog_year": year,
                         "catalog_month": date[:7], "catalog_event_date": date, "catalog_date_precision": "DATE_AS_RECORDED_NOT_FIRST_PUBLICATION",
                         "scheduled_start": None, "text": text, "channel_tags": channels(text), "detail_url": None,
                         "source_url": SEEDS[key], "record_status": "LEAD_ONLY_NOT_TRADABLE", "expectation": None,
                         "multiple_date_tokens": len(re.findall(r"\d+月\d+日", text)), "kind": "PBOC_CHRONOLOGY_PARAGRAPH"})
        counts[key] = len(paragraphs)
        key = f"scio_{year}_http"
        source = soup_for(key)
        selector = "div.section .list-box a[href]" if year == 2024 else "div.monthBox div.prTitle > a[href]"
        anchors = source.select(selector)
        seen = set()
        for i, a in enumerate(anchors, 1):
            link = urljoin(SEEDS[f"scio_{year}"], a["href"])
            if link in seen:
                raise ValueError("目录出现重复链接，需要保留并单独处理，不能静默删除")
            seen.add(link)
            cls = "section" if year == 2024 else "monthBox"
            parent = next(p for p in a.parents if cls in p.get("class", []))
            tag = next(c.casefold() for c in parent["class"] if c.casefold() in months)
            month = f"{year}-{months[tag]:02}"
            clock = a.parent.select_one("span.date")
            clock_text = " ".join(clock.get_text(" ", strip=True).split()) if clock else None
            event_date = None
            if clock_text:
                day = re.search(r"([A-Za-z.]+)\s+(\d{1,2}),?\s+2025", clock_text)
                if day:
                    date_month = months.get(day[1][:3].casefold())
                    if date_month != months[tag]:
                        raise ValueError("目录月份与会议日期不同")
                    event_date = f"2025-{date_month:02}-{int(day[2]):02}"
            text = " ".join(a.get_text(" ", strip=True).split())
            rows.append({"catalog_id": f"SCIO_{year}_{i:03}", "source_key": key, "catalog_year": year,
                         "catalog_month": month, "catalog_event_date": event_date,
                         "catalog_date_precision": "CONFERENCE_DATE_NOT_CONTENT_AVAILABILITY" if event_date else "MONTH_ONLY",
                         "scheduled_start": clock_text, "text": text, "channel_tags": channels(text), "detail_url": link,
                         "source_url": SEEDS[f"scio_{year}"], "record_status": "LEAD_ONLY_NOT_TRADABLE", "expectation": None,
                         "multiple_date_tokens": 0, "kind": "SCIO_DIRECTORY_ENTRY"})
        counts[f"scio_{year}"] = len(anchors)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "results/全部官方目录记录.csv", index=False, encoding="utf-8-sig")
    frame.to_parquet(OUT / "results/全部官方目录记录.parquet", index=False)
    expanded = frame.assign(channel=frame.channel_tags.str.split("|")).explode("channel")
    expanded.groupby(["catalog_month", "channel"]).size().rename("catalog_rows").reset_index().to_csv(OUT / "results/每月各通道目录覆盖.csv", index=False, encoding="utf-8-sig")
    lpr = frame[(frame.kind == "PBOC_CHRONOLOGY_PARAGRAPH") & frame.text.str.contains("贷款市场报价利率")].copy()
    lpr["one_year_percent"] = lpr.text.str.extract(r"1年期LPR为([0-9.]+)%")[0].astype(float)
    lpr["five_year_percent"] = lpr.text.str.extract(r"5年期以上LPR为([0-9.]+)%")[0].astype(float)
    lpr["stated_no_change"] = ~lpr.text.str.contains("下降")
    lpr[["catalog_id", "catalog_event_date", "one_year_percent", "five_year_percent", "stated_no_change", "source_url"]].to_csv(OUT / "results/全部24次LPR目录记录.csv", index=False, encoding="utf-8-sig")
    save(OUT / "results/catalog_summary.json", {"counts": counts, "total_catalog_rows": len(frame),
        "unique_policy_event_count": "NOT_ESTABLISHED_CROSS_SOURCE_DEDUP_AND_MULTI_ACTION_SPLIT_PENDING",
        "scio_month_only_rows": int((frame.catalog_date_precision == "MONTH_ONLY").sum()),
        "scio_dated_rows": int((frame.catalog_date_precision == "CONFERENCE_DATE_NOT_CONTENT_AVAILABILITY").sum()),
        "lpr_rows": len(lpr), "lpr_stated_no_change": int(lpr.stated_no_change.sum()),
        "source_catalog_extraction": "ALL_FIXED_SELECTOR_ENTRIES_RETAINED",
        "china_all_policy_completeness": False, "tradable_catalog_rows": 0,
        "taxonomy_use": "仅导航，可多标签，不作为利好/利空评分，不用于筛选收益。"})
    save(OUT / "evidence/catalog_acquisition_adjustment.json", {"at": now(),
        "http_fallback": "SCIO两份HTTPS证书主机名核验失败；保留失败回执，改用同一官方域名公开HTTP页面，未关闭TLS校验。",
        "selectors": {"2024": "div.section .list-box a[href]", "2025": "div.monthBox div.prTitle > a[href]"},
        "excluded_page_elements": "栏目导航、新闻导读图片与非目录推荐链接。2025页面含2026年推荐，不混入2025目录。",
        "coverage_caveat": "2024官方回顾另称超过190场，而当前固定年度目录列186项；因此不能宣称SCIO全年发布会全集。"})
    shutil.copy2(__file__, OUT / "code/catalog_builder.py")
    print(json.dumps({"counts": counts, "total": len(frame), "lpr": len(lpr), "unchanged_lpr": int(lpr.stated_no_change.sum())}, ensure_ascii=False))


def fetch_chains() -> None:
    save(OUT / "inputs/chain_source_list.json", CHAIN_SOURCES)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(lambda kv: fetch(*kv), CHAIN_SOURCES.items()):
            print(json.dumps({k: r[k] for k in ("key", "http_code", "bytes", "error")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="政策目录与首次公开时钟")
    parser.add_argument("action", choices=["prepare", "fetch-seeds", "catalogs", "fetch-chains"])
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "fetch-seeds":
        fetch_seeds()
    elif args.action == "catalogs":
        parse_catalogs()
    else:
        fetch_chains()
