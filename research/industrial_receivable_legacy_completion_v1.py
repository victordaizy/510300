"""补齐官方现行目录未覆盖的18份旧月报，原始部分结果保持不变。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import sys
from threading import Lock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.industrial_receivable_monthly_source_v1 as prior
from research.exchange_bank_funding_gap_daily_v1 import adapt
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_industrial_receivable_legacy_completion_v1"
STUDY = "510300_INDUSTRIAL_RECEIVABLE_LEGACY_COMPLETION_V1"
PANEL = OUT / "released_industrial_receivables.parquet"
LEGACY = {
    "2020-02": "https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/202003/t20200327_1767778.html",
    "2020-03": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900702.html",
    "2020-04": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900738.html",
    "2020-05": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900764.html",
    "2020-06": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900801.html",
    "2020-07": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900834.html",
    "2020-08": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900858.html",
    "2020-09": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900893.html",
    "2020-10": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900919.html",
    "2020-11": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900948.html",
    "2020-12": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900985.html",
    "2021-02": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901035.html",
    "2021-03": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901067.html",
    "2021-04": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901113.html",
    "2021-05": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901140.html",
    "2021-06": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901177.html",
    "2021-07": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1901204.html",
    "2021-08": "https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/202109/t20210928_1822582.html",
}
FETCH = adapt(prior.fetch, {"OUT": OUT, "_request_count": len(list((OUT / "requests").glob("*.json"))), "_lock": Lock()})
REQUEST = adapt(prior.request_job, {"fetch": FETCH})


def parse(month, receipt, catalogue_item=None):
    soup = prior.soup_for(receipt)
    text = soup.get_text(" ", strip=True)
    compact = re.sub(r"\s+", "", text)
    headings = [tag.get_text(" ", strip=True) for tag in soup.find_all(["title", "h1", "h2"])]
    titles = sorted({title for title in headings if prior.title_month(title) == month})
    assert titles, "原文标题未辨认出目标统计月。"
    clocks = set(re.findall(r"20\d{2}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}(?::\d{2})?", text))
    pub = soup.find("meta", attrs={"name": "PubDate"})
    if pub is not None:
        clocks.add(pub.get("content", ""))
    dates = re.findall(r"成文日期(20\d{2})年(\d{2})月(\d{2})日", compact)
    parsed_clocks = {pd.Timestamp(clock).tz_localize("Asia/Shanghai") for clock in clocks}
    if parsed_clocks:
        assert len(parsed_clocks) == 1, "原公布时间不唯一。"
        clock = next(iter(parsed_clocks))
        precision = "MINUTE_OR_SECOND"
    else:
        assert len(set(dates)) == 1, "原成文日期不唯一。"
        clock = pd.Timestamp("-".join(dates[0])).tz_localize("Asia/Shanghai")
        precision = "DAY_ONLY"
    if dates:
        assert {"-".join(d) for d in dates} == {clock.strftime("%Y-%m-%d")}, "页面时间与成文日期不一致。"
    if catalogue_item is not None:
        assert clock.strftime("%Y-%m-%d") == catalogue_item["catalogue_publication_date"]
    matches = re.findall(r"应收账款平均回收期为(\d+(?:\.\d+)?)天[，,](同比|比上年(?:末)?)(增加|减少|延长|缩短)(\d+(?:\.\d+)?)天", compact)
    values = {(float(level), reference, (-1 if direction in ["减少", "缩短"] else 1) * float(change)) for level, reference, direction, change in matches}
    flat = re.findall(r"应收账款平均回收期为(\d+(?:\.\d+)?)天[，,](同比|与上年(?:末)?)持平", compact)
    values.update((float(level), reference, 0.) for level, reference in flat)
    assert len(values) == 1, "正文回收期或原文同比不唯一。"
    days, reference, change = next(iter(values))
    formula = "应收账款平均回收期=360×平均应收账款÷营业收入×累计月数÷12"
    assert formula in compact.replace("＝", "="), "公式不属于2020年后口径。"
    table_values = []
    for table in soup.find_all("table"):
        heading = re.sub(r"\s+", "", table.get_text())
        if "应收账款平均回收期" not in heading or "营业收入利润率" not in heading:
            continue
        for row in table.find_all("tr"):
            cells = [re.sub(r"\s+", "", cell.get_text()) for cell in row.find_all(["td", "th"], recursive=False)]
            if len(cells) == 9 and cells[0] == "总计" and all(re.fullmatch(r"-?\d+(?:\.\d+)?", v) for v in cells[1:]):
                table_values.append(float(cells[-1]))
    assert set(table_values) == {days}, "正文与经济效益表总计行不一致。"
    item = {} if catalogue_item is None else dict(catalogue_item)
    return {**item, "stat_month": month, "title": titles[0], "url": receipt["url"],
        "published_at": clock.isoformat() if precision != "DAY_ONLY" else clock.strftime("%Y-%m-%d"),
        "publication_precision": precision,
        "clock_support": "ORIGINAL_PAGE_ONLY_LEGACY_ARCHIVE" if catalogue_item is None else "ORIGINAL_PAGE_AND_CATALOGUE",
        "known_at": (clock.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).isoformat(),
        "receivable_collection_days": days, "receivable_days_published_yoy_change": change, "reported_reference": reference,
        "formula": formula, "body_and_total_table_agree": True, "raw_path": receipt["raw_path"], "raw_sha256": receipt["sha256"],
        "method_group": "RECEIVABLE_EXCLUDING_NOTES_OPERATING_REVENUE_2020_ONWARD", "historical_first_vintage_verified": False}


def run():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("旧月报补全已固定，不能重复覆盖。")
    parent_result = read(prior.OUT / "result.json")
    assert read(prior.OUT / "catalogue_result.json")["missing_months"] == list(LEGACY)
    for folder in ["raw", "requests", "code"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "legacy_sources": LEGACY,
        "source_discovery": "官方站点定向检索取得18个月原文，补足现行67页目录仅覆盖2021-09之后的缺口。只有原文数值可进入面板，搜索摘要不作为数值输入。",
        "parent_result": parent_result,
        "parse_completion": "信息公开页可能用通用网页标题，实际报告标题在h2；由title/h1/h2辨认原报告。有日内时间时精确保存，无时间时按原成文日末保守可用，不能伪造09:30。",
        "original_controls": "仍要求当季定义、正文与总计表一致、直接公布同比、2020年后同名公式；原部分来源及失败输出不改。",
        "clock_limit": "旧档案没有现行目录对照的18个月，明确记为原页时钟，历史不可变首版仍未认证。",
        "network": "本次最多18个URL，已存2020-02原文复用；同时最多2个公开请求，25秒超时，传输故障普通重试一次，403/429停止。",
        "new_accounts": 0, "new_strategy_returns": 0, "goal_achieved": False, "orders_authorized": False}, True)
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"), "code_sha256": digest(Path(__file__)),
        "parent_result_sha256": digest(prior.OUT / "result.json"), "parent_records_sha256": digest(prior.OUT / "released_records.json"),
        "parent_parser_sha256": digest(Path(prior.__file__)), "probe_sources_sha256": digest(prior.probe.OUT / "source_measurements.json")}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    records = read(prior.OUT / "released_records.json")
    failures = []
    original_issues = read(prior.OUT / "unresolved_fields.json")
    for issue in original_issues:
        try:
            records.append(parse(issue["stat_month"], issue["receipt"], issue))
        except (AssertionError, ValueError, KeyError) as exc:
            failures.append({"stat_month": issue["stat_month"], "error": str(exc), "prior_issue": issue})
    cached = {r["receipt"]["url"]: r["receipt"] for r in read(prior.probe.OUT / "source_measurements.json") if r["receipt"]["status"] == "HTTP_OK"}
    items = list(LEGACY.items())
    for month, url in items:
        if url in cached:
            records.append(parse(month, cached[url]))
    todo = [(month, url) for month, url in items if url not in cached]
    with ThreadPoolExecutor(max_workers=2) as executor:
        for start in range(0, len(todo), 2):
            batch = todo[start:start + 2]
            jobs = [(url, "month_" + month) for month, url in batch]
            for (month, url), receipt in zip(batch, executor.map(REQUEST, jobs)):
                try:
                    records.append(parse(month, receipt))
                except (AssertionError, ValueError, KeyError) as exc:
                    failures.append({"stat_month": month, "url": url, "receipt": receipt, "error": str(exc)})
            print(f"工业收款合并原月报{len(records)}/72份，待核{len(failures)}份。", flush=True)
            if (OUT / "REQUESTS_STOPPED.json").exists():
                break
    records.sort(key=lambda r: r["stat_month"])
    assert len(records) == len({r["stat_month"] for r in records})
    frame = pd.DataFrame(records)
    frame["known_at"] = pd.to_datetime(frame.known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
    assert frame.known_at.is_monotonic_increasing and not frame.known_at.duplicated().any()
    frame.to_parquet(PANEL, index=False)
    save(OUT / "released_records.json", records, True)
    save(OUT / "unresolved_fields.json", failures, True)
    missing = sorted(set(prior.EXPECTED) - set(frame.stat_month))
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "COMPLETE_SAME_DEFINITION_MONTHLY_RECEIVABLE_SOURCES_READY" if not missing else "PARTIAL_SAME_DEFINITION_MONTHLY_RECEIVABLE_SOURCES",
        "expected_months": 72, "admitted_months": len(frame), "missing_months": missing, "unresolved_fields": len(failures),
        "legacy_page_only_clocks": int(frame.get("clock_support", pd.Series(dtype=str)).eq("ORIGINAL_PAGE_ONLY_LEGACY_ARCHIVE").sum()),
        "latest_source": records[-1], "new_accounts": 0, "new_strategy_returns": 0,
        "historical_first_vintage_verified": False, "current_market_view": "NO_VIEW",
        "goal_status": "active", "goal_achieved": False, "orders_authorized": False}, True)
    print(f"工业企业收款来源补全结束，{len(frame)}/72个月已准入，仍缺{len(missing)}个月。", flush=True)


if __name__ == "__main__":
    run()
