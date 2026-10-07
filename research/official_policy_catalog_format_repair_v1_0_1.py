"""只修复官方旧标题/字体拆日期；月份不补具体日，原采集与失败保持。"""
from __future__ import annotations

import argparse
import calendar
import hashlib
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import pandas as pd

from research import official_policy_announcement_catalog_intake_v1 as original

ROOT, SOURCE, parent, inputs = original.ROOT, original.OUT, original.parent, original.inputs
OUT = SOURCE / "implementation_v1_0_1"
REPAIRS = ("PBOC_015", "PBOC_016", "PBOC_017", "PBOC_018", "PBOC_035", "PBOC_038", "PBOC_040")


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def index_rows(raw, url):
    soup = BeautifulSoup(raw, "html.parser", from_encoding="utf-8")
    rows = []
    for anchor in soup.select("a[istitle='true'][href]"):
        title = inputs.clean(anchor.get_text(" ", strip=True))
        if re.fullmatch(r"20\d{2}年.*(?:中国)?货币政策大事记", title) is None:
            raise ValueError("目录题目超出原货币政策大事记主题。")
        link = urljoin(url, anchor["href"])
        if urlparse(link).hostname != "www.pbc.gov.cn":
            raise ValueError("目录文章不在原官方主站。")
        cell = anchor.find_parent("td")
        date = re.search(r"20\d{2}-\d{2}-\d{2}", cell.get_text(" ", strip=True) if cell else "")
        if date is None:
            raise ValueError("旧标题对应目录日期缺失。")
        rows.append({"title": title, "url": link, "listed_published_date": date[0]})
    return rows


def parse_document(raw, doc):
    soup = BeautifulSoup(raw, "html.parser", from_encoding="utf-8")
    title = inputs.clean(soup.title.get_text()) if soup.title else ""
    if doc["title"] not in title:
        raise ValueError("已保存原文标题身份不一致。")
    body = soup.find(id="zoom")
    if body is None:
        raise ValueError("固定正文结构没有找到。")
    blocks = [inputs.clean(p.get_text("", strip=True)) for p in body.find_all("p")]
    blocks = [block for block in blocks if block]
    if not blocks:
        raise ValueError("受影响源文没有段落，不扩展格式补救。")
    clock = re.search(r"20\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", soup.get_text(" ", strip=True))
    at = pd.Timestamp(clock[0], tz=inputs.TZ) if clock else pd.Timestamp(doc["listed_published_date"], tz=inputs.TZ)+pd.Timedelta(hours=23, minutes=59, seconds=59)
    if str(at.date()) != doc["listed_published_date"]:
        raise ValueError("保存原文和目录公布日期冲突。")
    rows, unknown = [], []
    year = int(doc["catalog_year"])
    for serial, text in enumerate(blocks, 1):
        dated = re.match(r"^(\d{1,2})月(\d{1,2})日", text)
        monthly = re.match(r"^(\d{1,2})月(下旬)?[,，]", text)
        if dated:
            event_date = pd.Timestamp(year=year, month=int(dated[1]), day=int(dated[2]))
            lower, upper, precision = event_date, event_date, "DATE_RECORDED_NOT_ANNOUNCEMENT_CLOCK"
            key_date = event_date.isoformat()
        elif monthly:
            month = int(monthly[1])
            lower = pd.Timestamp(year=year, month=month, day=21 if monthly[2] else 1)
            upper = pd.Timestamp(year=year, month=month, day=calendar.monthrange(year, month)[1])
            event_date = pd.NaT
            precision = "LATE_MONTH_EXACT_DAY_UNKNOWN" if monthly[2] else "MONTH_EXACT_DAY_UNKNOWN"
            key_date = f"{year}-{month:02}|{precision}"
        else:
            unknown.append({"catalog_id": doc["catalog_id"], "source_paragraph_number": serial, "text": text,
                "status": "NON_DATE_PARAGRAPH_RETAINED_NOT_MAPPED"})
            continue
        key = hashlib.sha256((key_date+"|"+inputs.compact(text)).encode("utf-8")).hexdigest()
        rows.append({"catalog_id": doc["catalog_id"], "paragraph_number": serial, "catalog_event_date": event_date,
            "catalog_event_date_lower": lower, "catalog_event_date_upper": upper, "event_date_precision": precision,
            "logical_record_id": key, "text": text, "navigation_tool_tags": inputs.tools(text),
            "multiple_date_tokens": len(re.findall(r"\d+月\d+日", text)), "source_available_upper": at,
            "source_clock_precision": "SECOND_CURRENT_PAGE_NOT_FIRST_VINTAGE" if clock else "LISTED_DATE_END_UPPER_NOT_FIRST_VINTAGE",
            "source_url": doc["url"], "record_role": "RETROSPECTIVE_LEAD_ONLY_NOT_ORIGINAL_ANNOUNCEMENT",
            "historical_first_vintage_authenticated": False, "source_path": doc["source_path"], "source_sha256": doc["source_sha256"],
            "reference_start": doc["reference_start"], "reference_end": doc["reference_end"], "reference_role": doc["reference_role"]})
    return rows, unknown


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("源文格式修复已登记，不覆盖。")
    summary = parent.read(SOURCE / "summary.json")
    tests = parent.read(OUT / "tests_receipt.json")
    if summary["decision"] != "TECH.R228" or summary["selected_documents"] != 46 or tests["passed"] != 3 or tests["exit_code"] != 0:
        raise ValueError("原实际来源或三必要修复测试不匹配。")
    manifest = parent.read(SOURCE / "source_manifest.json")
    paths = [Path(__file__), ROOT / "tests/test_official_policy_catalog_format_repair_v1_0_1.py",
        SOURCE / "summary.json", SOURCE / "source_manifest.json", SOURCE / "source_plan.json",
        SOURCE / "results/全部目录日期段落_源实例与多日期保留.parquet", SOURCE / "results/全部3488当时已记录政策链_空记录不等于无政策.parquet"]
    paths.extend(ROOT / item["source_path"] for item in [*manifest["index_review"], *manifest["documents"]] if "source_path" in item)
    parent.write(OUT / "protocol.json", {"at": parent.original.now(), "parent_decision": "TECH.R228", "implementation": "1.0.1",
        "scope": "仅INDEX_4旧标题少中国字样、2016字体拆分和六条真实月份日期独立保存；所有旧规则/24链输入/原文不改。",
        "repair_catalog_ids": list(REPAIRS), "new_network_requests": 0, "new_accounts": 0, "new_fits": 0,
        "new_daily_chain_runs": 0, "ambiguous_month_is_imputed_day": False, "necessary_repair_tests": 3,
        "all_unknowns_retained": True, "original_failure_receipts_retained": True,
        "files": [{"path": original.relative(path), "sha256": parent.digest(path)} for path in dict.fromkeys(paths)]})
    print("R228源文格式修复固定：原已保存源文、零新请求/链计算/金融；月份日期不补日。", flush=True)


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("已保存原文格式修复已经运行，不重复。")
    protocol = parent.read(OUT / "protocol.json")
    for source in protocol["files"]:
        if parent.digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("原保存来源或格式代码改变。")
    parent.write(OUT / "RUN_STARTED.json", {"at": parent.original.now(), "new_requests": 0})
    manifest = parent.read(SOURCE / "source_manifest.json")
    all_index = []
    for item in sorted(manifest["index_review"], key=lambda x: x["key"]):
        rows = index_rows((ROOT / item["source_path"]).read_bytes(), item["url"])
        all_index.extend({"index_key": item["key"], "index_source_sha256": item["source_sha256"], **row} for row in rows)
    index_frame = pd.DataFrame(all_index)
    if len(index_frame) != 88 or index_frame.url.duplicated().any():
        raise ValueError("源文格式处理未复原完整88母目录。")
    selected_urls = []
    for row in index_frame.to_dict("records"):
        match = re.match(r"^(20\d{2})年", row["title"])
        if match and 2015 <= int(match[1]) <= 2026:
            details = inputs.period(row["title"])
            if details["reference_end"] <= "2026-06-30":
                selected_urls.append(row["url"])
    if selected_urls != [doc["url"] for doc in manifest["documents"]]:
        raise ValueError("格式处理改变46固定源文集合或顺序。")
    old_records = pd.read_parquet(SOURCE / "results/全部目录日期段落_源实例与多日期保留.parquet")
    unchanged = old_records.loc[~old_records.catalog_id.isin(REPAIRS)].copy()
    unchanged["catalog_event_date_lower"] = unchanged.catalog_event_date
    unchanged["catalog_event_date_upper"] = unchanged.catalog_event_date
    unchanged["event_date_precision"] = "DATE_RECORDED_NOT_ANNOUNCEMENT_CLOCK"
    records, unknown, statuses = [], [], []
    for doc in manifest["documents"]:
        if doc["catalog_id"] in REPAIRS:
            rows, rest = parse_document((ROOT / doc["source_path"]).read_bytes(), doc)
            records.extend(rows); unknown.extend(rest)
            statuses.append({"catalog_id": doc["catalog_id"], "title": doc["title"], "original_status": doc["parse_status"],
                "new_role": "PARSED_EXPLICIT_DATE_OR_AMBIGUOUS_MONTH", "source_rows": len(rows),
                "exact_day_unknown_rows": sum(pd.isna(row["catalog_event_date"]) for row in rows), "unassigned_body_fragments": len(rest)})
        else:
            statuses.append({"catalog_id": doc["catalog_id"], "title": doc["title"], "original_status": doc["parse_status"],
                "new_role": "ORIGINAL_EXACT_ROWS_REUSED" if doc["parse_status"] == "PARSED" else "ORIGINAL_FETCH_FAILURE_KEPT",
                "source_rows": int(unchanged.catalog_id.eq(doc["catalog_id"]).sum()), "exact_day_unknown_rows": 0, "unassigned_body_fragments": 0})
    repaired = pd.DataFrame(records)
    all_records = pd.concat([unchanged, repaired], ignore_index=True).sort_values(["catalog_id", "paragraph_number"])
    for name in ("catalog_event_date", "catalog_event_date_lower", "catalog_event_date_upper"):
        all_records[name] = pd.to_datetime(all_records[name]).astype("datetime64[ns]")
    all_records["source_available_upper"] = pd.to_datetime(all_records.source_available_upper, utc=True).dt.tz_convert(inputs.TZ).astype("datetime64[ns, Asia/Shanghai]")
    retained = all_records.loc[~all_records.catalog_id.isin(REPAIRS), old_records.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(old_records.loc[~old_records.catalog_id.isin(REPAIRS)].reset_index(drop=True), retained)
    logical = all_records.groupby("logical_record_id", sort=True).agg(catalog_event_date=("catalog_event_date", "first"),
        event_date_precision=("event_date_precision", "first"), catalog_event_date_lower=("catalog_event_date_lower", "first"),
        catalog_event_date_upper=("catalog_event_date_upper", "first"), text=("text", "first"), navigation_tool_tags=("navigation_tool_tags", "first"),
        source_instances=("catalog_id", "size"), catalog_ids=("catalog_id", lambda values: "|".join(values)),
        earliest_retrospective_publication=("source_available_upper", "min")).reset_index()
    logical["record_role"] = "RETROSPECTIVE_LEAD_ONLY_NOT_ORIGINAL_ANNOUNCEMENT"
    tools = all_records.assign(tool=all_records.navigation_tool_tags.str.split("|")).explode("tool").groupby("tool").agg(
        source_instances=("catalog_id", "size"), logical_records=("logical_record_id", "nunique")).reset_index()
    for name, frame in (("全部官方88目录入口_旧标题原样", index_frame), ("全部日期或月份条目_精度未知保留", all_records),
            ("全部逻辑记录_所有源实例与日期边界", logical), ("全部工具导航_非经济方向", tools),
            ("七原文格式处理_失败与月份未知保留", pd.DataFrame(statuses)),
            ("全部未归类正文片段_未丢弃", pd.DataFrame(unknown, columns=["catalog_id", "source_paragraph_number", "text", "status"]))):
        table(name, frame)
    for source in protocol["files"]:
        if parent.digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("格式修复改变原文或失败事实。")
    summary = {"at": parent.original.now(), "parent_registration": "TECH.R227", "decision": "TECH.R228", "implementation": "1.0.1",
        "status": "OFFICIAL_CATALOG_SOURCE_FORMAT_REPAIR_COMPLETE_UNKNOWN_AND_FAILURE_KEPT",
        "all_catalog_rows": len(index_frame), "all_88_index_entries_exact": True, "fixed_selected_documents": 46,
        "source_instance_records": len(all_records), "logical_records": len(logical),
        "exact_day_unknown_source_rows": int(all_records.catalog_event_date.isna().sum()),
        "exact_day_unknown_logical_rows": int(logical.catalog_event_date.isna().sum()),
        "extra_source_instances": len(all_records)-len(logical), "repeated_logical_records": int(logical.source_instances.gt(1).sum()),
        "known_date_source_rows": int(all_records.catalog_event_date.notna().sum()), "repair_source_ids": list(REPAIRS),
        "unassigned_body_fragments": len(unknown), "unchanged_source_rows_exact": len(retained),
        "original_fetch_failure_kept": 1, "new_requests": 0, "new_accounts": 0, "new_fits": 0, "new_daily_chain_runs": 0,
        "original_six_tests_passed": 6, "necessary_repair_tests_passed": 3, "original_19_prefix_checks_unchanged": True,
        "source_files_unchanged": len(protocol["files"]), "navigation_counts": tools.to_dict("records"),
        "complete_policy_announcement_coverage": "NOT_ESTABLISHED_RETROSPECTIVE_CATALOG_NOT_FIRST_ANNOUNCEMENT",
        "financial_metrics": "NOT_COMPUTED", "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    parent.write(OUT / "summary.json", summary)
    print(f"R228格式处理完成：88母目录、{len(all_records)}源实例/{len(logical)}逻辑记录；日期不明{summary['exact_day_unknown_source_rows']}行，1原生失败保持；零新请求或金融。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="只从已保存源文登记或运行一次格式处理")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze == args.run:
        parser.error("唯一选择--freeze或--run")
    freeze() if args.freeze else run()
