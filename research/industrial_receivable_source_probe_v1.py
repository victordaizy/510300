"""保存工业企业收款期限的原文口径，不把财务指标变化直接当作股票信号。"""
from __future__ import annotations

from pathlib import Path
import re
import sys
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_industrial_receivable_source_probe_v1"
STUDY = "510300_INDUSTRIAL_RECEIVABLE_SOURCE_PROBE_V1"
SOURCES = {
    "2018_12": "https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/201901/t20190128_1768279.html",
    "2019_02": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900265.html",
    "2020_02": "https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/202003/t20200327_1767778.html",
    "2025_12": "https://www.stats.gov.cn/sj/zxfb/202601/t20260127_1962382.html",
    "2026_07": "https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/202608/t20260827_1965126.html",
    "release_catalogue": "https://www.stats.gov.cn/sj/zxfb/",
}


def fetch(url, name):
    target = OUT / "requests" / f"{name}.json"
    if target.exists():
        record = read(target)
        if record.get("raw_path"):
            assert digest(ROOT / record["raw_path"]) == record["sha256"]
        return record
    assert urlparse(url).hostname == "www.stats.gov.cn"
    assert len(list((OUT / "requests").glob("*.json"))) < 12
    if (OUT / "REQUESTS_STOPPED.json").exists():
        raise RuntimeError("来源要求停止请求。")
    record = {"requested_at": now(), "url": url, "status": "REQUEST_STARTED"}
    try:
        response = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        raw = OUT / "raw" / f"{name}.html"
        raw.write_bytes(response.content)
        record.update(received_at=now(), final_url=response.url, http_status=response.status_code,
                      status="HTTP_OK" if response.ok else "HTTP_FAILURE", bytes=len(response.content),
                      raw_path=raw.relative_to(ROOT).as_posix(), sha256=digest(raw),
                      content_type=response.headers.get("Content-Type"))
        if response.status_code in [403, 429]:
            save(OUT / "REQUESTS_STOPPED.json", {"at": now(), "url": url, "http_status": response.status_code}, True)
    except requests.RequestException as exc:
        record.update(received_at=now(), status="TRANSPORT_FAILURE", error=f"{type(exc).__name__}: {exc}")
    save(target, record, True)
    return record


def extract(record):
    assert record["status"] == "HTTP_OK"
    soup = BeautifulSoup((ROOT / record["raw_path"]).read_bytes(), "html.parser", from_encoding="utf-8")
    text = soup.get_text(" ", strip=True)
    compact = re.sub(r"\s+", "", text)
    observations = []
    pattern = r"(应收(?:票据及应收)?账款平均回收期)为(\d+(?:\.\d+)?)天[，,](?:(同比|比上年(?:末)?))?(增加|减少|延长|缩短)(\d+(?:\.\d+)?)天"
    for match in re.finditer(pattern, compact):
        observations.append({"label": match.group(1), "days": float(match.group(2)),
            "reported_reference": match.group(3), "reported_direction": match.group(4),
            "published_change_days": float(match.group(5)) * (-1 if match.group(4) in ["减少", "缩短"] else 1),
            "source_excerpt": match.group(0)})
    formulas = re.findall(r"应收(?:票据及应收)?账款平均回收期[=＝][^。；]+[天]", compact)
    metadata = {m.get("name", m.get("property", "")): m.get("content", "") for m in soup.find_all("meta") if m.get("content")}
    date_text = re.findall(r"成文日期(20\d{2}年\d{2}月\d{2}日)", compact)
    title = soup.title.get_text(strip=True) if soup.title else None
    return {"title": title, "metadata": metadata, "written_date_text": date_text,
        "cash_collection_observations": observations, "cash_collection_formulas": formulas,
        "2019_income_denominator_change": "从2019年起，用“营业收入”替代“主营业务收入”" in compact,
        "2020_receivable_numerator_change": "从2020年起，停止发布月度“应收票据及应收账款”数据，改为发布“应收账款”数据" in compact,
        "comparability_warning_present": "不能直接相比计算增速" in compact,
        "links": [{"label": a.get_text(" ", strip=True), "url": urljoin(record["url"], a["href"])}
                  for a in soup.find_all("a", href=True) if "工业企业" in a.get_text() or "下一页" in a.get_text()],
        "script_sources": [urljoin(record["url"], s["src"]) for s in soup.find_all("script", src=True)]}


def run():
    if (OUT / "result.json").exists():
        raise RuntimeError("工业应收来源探查已有终态。")
    for name in ["raw", "requests", "code"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "sources": SOURCES,
        "purpose": "核查财务汇总的收款期限能否与季度主观回款判断区分，先核对分母、分子和公布时钟。",
        "duplicate_check": "已存在个股财报应收质量研究；未发现相同的全国工业企业平均回收期月度用途。问卷收款的新结果保留，不反转方向营救。",
        "candidate_feature": "后续候选为原报告直接给出的可比同比变化天数；本阶段不构造连续历史，不读取股票收益。",
        "known_breaks_to_verify": ["2019由主营业务收入改营业收入", "2019应收含票据，2020改回应收账款"],
        "restrictions": "每个原文自己的公布时间；报告期不是公布日；不能由不同年份旧版水平直接推算官方可比同比。",
        "request_limit": 12, "timeout_seconds": 25, "transport_retry_per_url": 1,
        "new_accounts": 0, "new_strategy_returns": 0, "orders_authorized": False, "goal_achieved": False}, True)
    save(OUT / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)), "protocol_sha256": digest(OUT / "protocol.json")}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    records, failures = [], []
    for name, url in SOURCES.items():
        receipt = fetch(url, name)
        if receipt["status"] == "TRANSPORT_FAILURE":
            receipt = fetch(url, name + "_transport_retry")
        if receipt["status"] == "HTTP_OK":
            records.append({"name": name, "receipt": receipt, **extract(receipt)})
        else:
            failures.append({"name": name, "receipt": receipt})
            if (OUT / "REQUESTS_STOPPED.json").exists():
                break
        print(f"工业收款原文已保存{len(records)}份，未完成{len(failures)}份。", flush=True)
    save(OUT / "source_measurements.json", records, True)
    save(OUT / "unresolved_requests.json", failures, True)
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "METHOD_AND_CLOCK_SOURCES_SAVED_FULL_HISTORY_NOT_COLLECTED" if not failures else "PARTIAL_METHOD_AND_CLOCK_SOURCES",
        "saved_pages": len(records), "unresolved_pages": len(failures),
        "methodology_breaks": {r["name"]: {k: r[k] for k in ["2019_income_denominator_change", "2020_receivable_numerator_change", "comparability_warning_present"]} for r in records},
        "new_accounts": 0, "new_strategy_returns": 0, "full_historical_panel_ready": False,
        "current_market_view": "NO_VIEW", "historical_first_vintage_verified": False,
        "goal_status": "active", "goal_achieved": False, "orders_authorized": False}, True)
    print("工业企业收款口径来源探查完成，尚无该指标的新策略结果。", flush=True)


if __name__ == "__main__":
    run()
