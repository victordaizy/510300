"""一次建立官方完整追溯目录，连接既有核对宣布链；不运行金融模型。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re

import pandas as pd
import requests

from research import broker_stage_policy_study_v1 as parent
from research import entry_information_sequence_description_study_v1 as previous
from research import official_policy_announcement_catalog_inputs_v1 as inputs

ROOT = parent.ROOT
OUT = ROOT / "reports/research/510300_official_policy_announcement_catalog_v1"
OLD = ROOT / "reports/research/510300_policy_information_clock_v1"
OLD_CATALOG = OLD / "results/全部官方目录记录.parquet"
OLD_CHAIN = OLD / "results/跨通道政策链_完整事实与时钟.parquet"
PROBE_INDEX = OUT / "probe/official_index_first_page.html"
PROBE_DOC = OUT / "probe/official_2024_annual.html"


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def source_paths():
    chain = pd.read_parquet(OLD_CHAIN, columns=["source_path", "source_sha256"])
    raw_paths = [OLD / str(path) for path in chain.source_path.unique()]
    for row in chain.itertuples(index=False):
        if parent.digest(OLD / row.source_path) != row.source_sha256:
            raise ValueError("原核对政策链原文身份改变。")
    return list(dict.fromkeys([Path(__file__), Path(inputs.__file__),
        ROOT / "tests/test_official_policy_announcement_catalog_v1.py", ROOT / "docs/510300_OFFICIAL_POLICY_ANNOUNCEMENT_CATALOG_V1.md",
        PROBE_INDEX, PROBE_INDEX.with_name("official_index_first_page_receipt.json"),
        PROBE_DOC, PROBE_DOC.with_name("official_2024_annual_receipt.json"),
        OLD_CATALOG, OLD_CHAIN, OLD / "protocol.json", OLD / "results/result.json", *raw_paths,
        previous.OBSERVED, previous.original.SOURCEFILES["cases"], previous.OUT / "summary.json"]))


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("官方政策目录已登记，不覆盖。")
    tests = parent.read(OUT / "tests_receipt.json")
    if tests["exit_code"] != 0 or tests["passed"] != 6 or tests["inputs_sha256"] != parent.digest(Path(inputs.__file__)):
        raise ValueError("六必要测试未通过或输入代码改变。")
    plan = inputs.index_plan(PROBE_INDEX.read_bytes())
    if plan["pages"] != 5 or plan["advertised_records"] != 88:
        raise ValueError("固定官方母目录数量不同。")
    parent.write(OUT / "protocol.json", {"at": parent.original.now(), "registration": "TECH.R227", "decision": "TECH.R228",
        "purpose": "OFFICIAL_POLICY_RETROSPECTIVE_CATALOG_AND_KNOWN_CHAIN_COVERAGE_NOT_FINANCIAL",
        "native_index_plan": plan, "document_selection": "目录年份2015—2025全部入口，以及2026参考期结束<=06-30的全部入口；不选全年/季度优胜版本。",
        "native_acquisition": "复用已保存首目录/2024全文；其余每URL一次GET，10/35秒，4并行；所有失败不重试。",
        "all_original_observations": 3488, "all_original_events": 143, "all_original_case_rows": 240,
        "necessary_tests": 6, "prefix_checks": 19, "old_catalog_rows": 490, "old_manual_chain_nodes": 24,
        "old_chain_scope": "六人工核对链不是全国新闻全集；24原文身份复用，旧15点复核和次开口径不改。",
        "new_daily_clock": "仅16:00可知记录与此前ETF决定之间新钟；不是新进入规则；回顾事件日不是公布钟。",
        "duplicate_rule": "全部源实例保留；事件日期+规范全文稳定逻辑身份；多日期条目不拆独立事件。",
        "navigation_tags": inputs.TOOLS, "economic_direction_from_tags": "NOT_COMPUTED",
        "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "new_accounts": 0, "new_fits": 0, "new_predictive_targets": 0, "new_market_bars": 0,
        "financial_metrics": "NOT_COMPUTED", "goal_achieved": False,
        "probe_accounting": {"saved_native_probes": 2, "unsaved_read_only_native_index_probes": 1,
            "web_discovery_only": "原文网页检索和首目录浏览是来源发现，不是准入新闻或交易标签。",
            "pre_freeze_synthetic_failures": 2, "repairs": "显式纳秒时间与可缺失字符串，不改变钟或事件。"},
        "files": [{"path": relative(path), "sha256": parent.digest(path)} for path in source_paths()]})
    print("R227官方政策来源固定：5页88母目录、全部2015—2025及2026上半年源文、原24链/143进入；零新金融。", flush=True)


def fetch(key, url):
    path = OUT / "sources" / (key + ".html")
    receipt = OUT / "receipts" / (key + ".json")
    if receipt.exists() or path.exists():
        raise RuntimeError("该原生请求已经记录，不重复：" + key)
    record = {"key": key, "url": url, "retrieved_at": parent.original.now(), "status": "NOT_FETCHED",
        "historical_first_vintage_authenticated": False, "native_request": True}
    try:
        response = requests.get(url, timeout=(10, 35), headers={"User-Agent": "Mozilla/5.0"})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        record.update(http_status=response.status_code, final_url=response.url, bytes=len(response.content),
            source_path=relative(path), source_sha256=parent.digest(path))
        response.raise_for_status()
        record["status"] = "FETCHED"
    except requests.RequestException as error:
        record.update(status="FETCH_FAILED", error=str(error))
    record["finished_at"] = parent.original.now()
    parent.write(receipt, record)
    return record


def reuse_probe(key, path, old_receipt):
    receipt = parent.read(old_receipt)
    if receipt["http_status"] != 200 or receipt["sha256"] != parent.digest(path):
        raise ValueError("已保存结构探测原文不可复用。")
    return {"key": key, "url": receipt["url"], "status": "FETCHED_REUSED_SAVED_PROBE", "http_status": 200,
        "bytes": path.stat().st_size, "source_path": relative(path), "source_sha256": parent.digest(path),
        "native_request": False, "historical_first_vintage_authenticated": False}


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("官方目录用途已开始，不重复运行或重采失败。")
    protocol = parent.read(OUT / "protocol.json")
    for source in protocol["files"]:
        if parent.digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("冻结来源或代码改变：" + source["path"])
    parent.write(OUT / "RUN_STARTED.json", {"at": parent.original.now(), "new_accounts": 0})
    indices = [reuse_probe("INDEX_1", PROBE_INDEX, PROBE_INDEX.with_name("official_index_first_page_receipt.json"))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, f"INDEX_{i}", url): i for i, url in enumerate(protocol["native_index_plan"]["urls"][1:], 2)}
        for future in as_completed(futures):
            result = future.result()
            indices.append(result)
            print("官方分页完成：" + result["key"] + "，" + result["status"], flush=True)
    catalog, index_review = [], []
    for source in sorted(indices, key=lambda item: item["key"]):
        review = {**source, "parsed_rows": 0, "parse_status": "NOT_PARSED"}
        if source["status"].startswith("FETCHED"):
            try:
                rows = inputs.index_rows((ROOT / source["source_path"]).read_bytes(), source["url"])
                catalog.extend({"index_key": source["key"], "index_source_sha256": source["source_sha256"], **row} for row in rows)
                review.update(parsed_rows=len(rows), parse_status="PARSED")
            except ValueError as error:
                review.update(parse_status="PARSE_FAILED", parse_error=str(error))
        index_review.append(review)
    frame = pd.DataFrame(catalog)
    if frame.empty:
        raise ValueError("全部母目录没有可审查记录。")
    duplicates = frame.url.duplicated(keep=False)
    catalog_exact = len(frame) == 88 and not duplicates.any() and all(row["parse_status"] == "PARSED" for row in index_review)
    frame["catalog_url_duplicate"] = duplicates
    selected, title_failures = [], []
    for row in frame.to_dict("records"):
        match = re.match(r"^(20\d{2})年", row["title"])
        if match is None or not 2015 <= int(match[1]) <= 2026:
            continue
        try:
            details = inputs.period(row["title"])
            if details["catalog_year"] == 2026 and details["reference_end"] > "2026-06-30":
                continue
            selected.append({**row, **details, "catalog_id": f"PBOC_{len(selected)+1:03d}"})
        except ValueError as error:
            title_failures.append({**row, "status": "TITLE_PERIOD_UNKNOWN", "error": str(error)})
    if not selected or len(selected) > 64 or len(set(item["url"] for item in selected)) != len(selected):
        raise ValueError("固定源文范围为空/超限或重复URL，不能自动选择版本。")
    parent.write(OUT / "source_plan.json", {"catalog_exact_88": catalog_exact, "documents": selected, "unknown_titles": title_failures})
    table("全部官方88目录入口_来源及重复保留", frame)
    receipts = {}
    reuse_url = parent.read(PROBE_DOC.with_name("official_2024_annual_receipt.json"))["url"]
    for item in selected:
        if item["url"] == reuse_url:
            receipts[item["catalog_id"]] = reuse_probe(item["catalog_id"], PROBE_DOC, PROBE_DOC.with_name("official_2024_annual_receipt.json"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, item["catalog_id"], item["url"]): item["catalog_id"] for item in selected if item["catalog_id"] not in receipts}
        for number, future in enumerate(as_completed(futures), 1):
            receipt = future.result()
            receipts[receipt["key"]] = receipt
            if number % 5 == 0 or receipt["status"] == "FETCH_FAILED":
                print(f"官方源文完成{len(receipts)}/{len(selected)}，最新状态{receipt['status']}。", flush=True)
    records, documents, unparsed = [], [], []
    for item in selected:
        receipt = receipts[item["catalog_id"]]
        doc = {**item, **receipt, "parse_status": "NOT_PARSED", "paragraphs": 0}
        if receipt["status"].startswith("FETCHED"):
            try:
                rows, meta = inputs.parse_document((ROOT / receipt["source_path"]).read_bytes(), item)
                text_path = OUT / "sources" / (item["catalog_id"] + ".txt")
                with text_path.open("x", encoding="utf-8") as stream:
                    stream.write(meta["body_text"])
                unknown_lines = [inputs.clean(line) for line in meta["body_text"].splitlines()
                    if re.match(r"^\s*(?:20\d{2}年)?\d{1,2}月", line) and not re.match(r"^\s*\d{1,2}月\d{1,2}日", line)]
                unparsed.extend({"catalog_id": item["catalog_id"], "text": text, "status": "DATE_HEADING_NOT_PARSED_NOT_DISCARDED"} for text in unknown_lines)
                records.extend({**row, "source_path": receipt["source_path"], "source_sha256": receipt["source_sha256"],
                    "reference_start": item["reference_start"], "reference_end": item["reference_end"], "reference_role": item["reference_role"]} for row in rows)
                doc.update(parse_status="PARSED" if not unknown_lines else "PARTIAL_PARSE_DATE_HEADING_UNKNOWN",
                    source_available_upper=meta["source_available_upper"], source_clock_precision=meta["clock_precision"],
                    paragraphs=len(rows), unparsed_date_headings=len(unknown_lines), body_prefix_before_first_date=meta["body_prefix_before_first_date"],
                    body_text_path=relative(text_path), body_text_sha256=parent.digest(text_path))
            except ValueError as error:
                doc.update(parse_status="PARSE_FAILED", parse_error=str(error))
        documents.append(doc)
    parent.write(OUT / "source_manifest.json", {"index_review": index_review, "documents": documents, "unknown_titles": title_failures})
    records_frame = pd.DataFrame(records)
    if records_frame.empty:
        raise ValueError("原文没有可审查日期条目，不运行后续描述。")
    records_frame["catalog_event_date"] = pd.to_datetime(records_frame.catalog_event_date).astype("datetime64[ns]")
    records_frame["source_available_upper"] = pd.to_datetime(records_frame.source_available_upper, utc=True).dt.tz_convert(inputs.TZ).astype("datetime64[ns, Asia/Shanghai]")
    logical = records_frame.groupby("logical_record_id", sort=True).agg(catalog_event_date=("catalog_event_date", "first"),
        text=("text", "first"), navigation_tool_tags=("navigation_tool_tags", "first"), source_instances=("catalog_id", "size"),
        catalog_ids=("catalog_id", lambda values: "|".join(values)), earliest_retrospective_publication=("source_available_upper", "min")).reset_index()
    logical["record_role"] = "RETROSPECTIVE_LEAD_ONLY_NOT_ORIGINAL_ANNOUNCEMENT"
    tool_counts = records_frame.assign(tool=records_frame.navigation_tool_tags.str.split("|")).explode("tool").groupby("tool").agg(
        source_instances=("catalog_id", "size"), logical_records=("logical_record_id", "nunique")).reset_index()
    old_catalog = pd.read_parquet(OLD_CATALOG)
    chain = pd.read_parquet(OLD_CHAIN)
    if len(old_catalog) != 490 or len(chain) != 24:
        raise ValueError("原政策目录或核对链数量改变。")
    observed = pd.read_parquet(previous.OBSERVED, columns=["date", "decision_time", "stage_entry_type"])
    daily = inputs.known_chain_daily(observed, chain)
    prefix = []
    for date in previous.KEY_DATES:
        short = observed.loc[pd.to_datetime(observed.date).le(date)].copy()
        cut = inputs.known_chain_daily(short, chain)
        pd.testing.assert_frame_equal(daily.iloc[:len(short)].reset_index(drop=True), cut, rtol=0, atol=0)
        prefix.append({"cut": str(date.date()), "all_recorded_chain_prefix_rows_exact": True})
    daily["retrospective_reference_scope_has_parsed_source"] = False
    for doc in documents:
        if doc["parse_status"] == "PARSED":
            covered = daily.date.between(doc["reference_start"], doc["reference_end"])
            daily.loc[covered, "retrospective_reference_scope_has_parsed_source"] = True
    daily["reference_scope_is_same_day_announcement_coverage"] = False
    events = daily.loc[daily.date.ge(pd.Timestamp("2015-01-05")) & daily.stage_entry_type.ne("NONE")].copy()
    cases_source = pd.read_parquet(previous.original.SOURCEFILES["cases"], columns=["date", "original_episode_id"])
    cases_source["date"] = pd.to_datetime(cases_source.date).astype("datetime64[ns]")
    cases = cases_source.merge(daily, on="date", validate="one_to_one")
    keys = daily.loc[daily.date.isin(previous.KEY_DATES)].copy()
    if len(daily) != 3488 or len(events) != 143 or len(cases) != 240 or len(keys) != 19:
        raise ValueError("原日历、全部进入、四案例或19关键点改变。")
    yearly = []
    for year, group in daily.loc[daily.date.ge(pd.Timestamp("2015-01-05"))].groupby(daily.date.dt.year):
        yearly.append({"year": int(year), "original_research_days": len(group),
            "retrospective_reference_source_days": int(group.retrospective_reference_scope_has_parsed_source.sum()),
            "recorded_new_chain_days": int(group.recorded_new_chain_nodes.gt(0).sum()),
            "recorded_new_chain_nodes": int(group.recorded_new_chain_nodes.sum()),
            "complete_policy_news_coverage": "NOT_ESTABLISHED", "empty_chain_is_policy_absent": False})
    docframe = pd.DataFrame(documents)
    for name, data in (("全部源文期间与公布钟_失败未知保留", docframe), ("全部目录日期段落_源实例与多日期保留", records_frame),
            ("全部逻辑记录_所有重复来源及回顾公布钟", logical), ("全部工具导航_非利好利空评分", tool_counts),
            ("原24人工核对链_原复核与执行时钟保持", chain), ("全部3488当时已记录政策链_空记录不等于无政策", daily),
            ("全部143进入点_已记录宣布与资料覆盖区别", events), ("原四案例240日_已记录政策链与未知", cases),
            ("全部19关键点_原链可知钟与两假启动", keys), ("全部研究年份_追溯目录和宣布覆盖区别", pd.DataFrame(yearly))):
        table(name, data)
    table("全部未解析日期标题_原文仍保留", pd.DataFrame(unparsed, columns=["catalog_id", "text", "status"]))
    for source in protocol["files"]:
        if parent.digest(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("本次用途改变原来源：" + source["path"])
    native = sum(bool(item["native_request"]) for item in [*indices, *receipts.values()])
    all_parsed = catalog_exact and not title_failures and all(doc["parse_status"] == "PARSED" for doc in documents)
    summary = {"at": parent.original.now(), "registration": "TECH.R227", "decision": "TECH.R228",
        "status": "OFFICIAL_RETROSPECTIVE_CATALOG_AND_PARTIAL_KNOWN_CHAIN_COMPLETE_NOT_FINANCIAL",
        "advertised_catalog_rows": 88, "actual_catalog_rows": len(frame), "all_88_index_entries_exact": catalog_exact,
        "selected_documents": len(selected), "document_fetch_status": docframe.status.value_counts().to_dict(),
        "document_parse_status": docframe.parse_status.value_counts().to_dict(), "all_selected_documents_parsed_without_unknown": all_parsed,
        "all_source_instance_records": len(records_frame), "all_logical_records": len(logical),
        "repeated_logical_records": int(logical.source_instances.gt(1).sum()), "extra_source_instances": len(records_frame)-len(logical),
        "unparsed_date_headings": len(unparsed), "multi_date_source_records": int(records_frame.multiple_date_tokens.gt(1).sum()),
        "navigation_counts": tool_counts.to_dict("records"), "native_requests_this_registered_run": native,
        "saved_probe_reuses": 2, "earlier_native_probes": 3, "native_total_including_unsaved_probe": native+3,
        "native_fetch_failures": sum(item["status"] == "FETCH_FAILED" for item in [*indices, *receipts.values()]),
        "old_catalog_rows": 490, "old_verified_chain_nodes": 24, "original_daily_rows": len(daily), "original_events": len(events),
        "original_case_rows": len(cases), "key_rows": len(keys), "necessary_tests_passed": 6, "prefix_checks": prefix,
        "source_files_unchanged": len(protocol["files"]), "yearly_source_scope": yearly,
        "original_events_with_recorded_new_chain": int(events.recorded_new_chain_nodes.gt(0).sum()),
        "complete_policy_announcement_coverage": "NOT_ESTABLISHED_RETROSPECTIVE_CATALOG_NOT_FIRST_ANNOUNCEMENT",
        "new_accounts": 0, "new_fits": 0, "new_predictive_targets": 0, "new_market_bars": 0,
        "financial_metrics": "NOT_COMPUTED", "latest_actual_financial_decision_unchanged": "TECH.R224",
        "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}
    parent.write(OUT / "summary.json", summary)
    print(f"R228官方来源完成：{len(frame)}母目录/{len(documents)}源文/{len(records_frame)}源记录、24原链/143进入；GET失败{summary['native_fetch_failures']}，零新金融。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="登记或一次补完整官方政策追溯目录")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze == args.run:
        parser.error("唯一选择--freeze或--run")
    freeze() if args.freeze else run()
