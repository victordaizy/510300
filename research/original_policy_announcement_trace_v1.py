"""固定完整线索母集追溯官方原公告；保留来源未知，不运行新金融。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research import broker_stage_policy_study_v1 as parent
from research import original_policy_announcement_trace_inputs_v1 as inputs

ROOT = Path(__file__).absolute().parent.parent
OUT = ROOT / "reports/research/510300_original_policy_announcement_trace_v1"
RATES = ROOT / "reports/research/510300_fiscal_execution_state_20d_v1/inputs/operation_rates.parquet"
CATALOG = ROOT / "reports/research/510300_official_policy_announcement_catalog_v1/implementation_v1_0_1/results/全部逻辑记录_所有源实例与日期边界.parquet"
CHAIN_ROOT = ROOT / "reports/research/510300_policy_information_clock_v1"
CHAINS = CHAIN_ROOT / "results/跨通道政策链_完整事实与时钟.parquet"
DESCRIPTION = ROOT / "reports/research/510300_entry_information_sequence_description_v1/results"
DAILY = DESCRIPTION / "全部3488进入信息身份_公布钟与量价行业.parquet"
EVENTS = DESCRIPTION / "全部143进入事件_来源重复与既有周期上下文.parquet"
KEYS = DESCRIPTION / "原17关键日_来源钟与行业量价.parquet"
GOV_OLD = ROOT / "reports/research/510300_historical_index_liquidity_transmission_v1/sources/gov_20190104.html"


def relative(path):
    return path.absolute().relative_to(ROOT).as_posix()


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def mother():
    rate, catalog, chain = pd.read_parquet(RATES), pd.read_parquet(CATALOG), pd.read_parquet(CHAINS)
    if len(rate) != 25 or len(chain) != 24:
        raise ValueError("原固定25利率身份或24节点已改变。")
    rows = []
    for number, item in enumerate(rate.to_dict("records"), 1):
        rows.append({"reference_id": f"RATE_{number:02}", "reference_kind": "原利率操作身份", "reference_date": item["notice_date"],
            "navigation_family": "7天逆回购实际操作", "reference_text": f"原记录7天利率{item['seven_day_rate_percent']}%，前一已观察值{item['previous_rate_percent']}%。",
            "parent_source_identity": item["source_url"], "parent_source_path": item["raw_path"],
            "lookup_group_id": None, "record_role": "OPERATION_NOTICE_NOT_ALL_POLICY_ANNOUNCEMENTS"})
    for tag, expected, kind, prefix in (("准备金率", 45, "准备金导航全文", "RRR"), ("资本市场工具", 3, "资本工具导航全文", "CAP")):
        subset = catalog.loc[catalog.navigation_tool_tags.str.contains(tag, regex=False)]
        if len(subset) != expected:
            raise ValueError("固定导航全文数量改变。")
        for item in subset.to_dict("records"):
            family = inputs.navigation_family(item["text"], capital=prefix == "CAP")
            date = pd.Timestamp(item["catalog_event_date"])
            group = "LOOKUP_" + inputs.stable_id(f"{date.date()}|{family}")
            rows.append({"reference_id": prefix + "_" + item["logical_record_id"][:16], "reference_kind": kind,
                "reference_date": date, "navigation_family": family, "reference_text": item["text"],
                "parent_source_identity": item["logical_record_id"], "parent_source_path": relative(CATALOG),
                "lookup_group_id": group, "record_role": "RETROSPECTIVE_NAVIGATION_NOT_ANNOUNCEMENT_CLOCK"})
    for item in chain.to_dict("records"):
        rows.append({"reference_id": "CHAIN_" + item["node_id"], "reference_kind": "原人工核对链", "reference_date": pd.Timestamp(item["economic_event_date"]),
            "navigation_family": item["chain"], "reference_text": item["new_or_confirmed_information"],
            "parent_source_identity": item["source_key"], "parent_source_path": relative(CHAIN_ROOT / item["source_path"]),
            "lookup_group_id": None, "record_role": "ORIGINAL_VERIFIED_NODE_NOT_NEWS_UNIVERSE"})
    result = pd.DataFrame(rows)
    if len(result) != 97 or not result.reference_id.is_unique:
        raise ValueError("97固定原参考身份不完整或重复。")
    return result


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("原公告追溯已固定，不覆盖。")
    tests = parent.read(OUT / "tests_receipt.json")
    if tests["passed"] != 6 or tests["exit_code"] != 0 or tests["inputs_sha256"] != parent.digest(Path(inputs.__file__)):
        raise ValueError("六个必要来源与因果测试尚未完成。")
    records = mother()
    table("全部97原参考身份_不是97独立政策", records)
    lookups = []
    subset = records.loc[records.lookup_group_id.notna()]
    for group, frame in subset.groupby("lookup_group_id", sort=False):
        row = frame.iloc[0]
        date = pd.Timestamp(row.reference_date)
        query = f"中国人民银行 {date.year}年 {date.month}月{date.day}日 {row.navigation_family} 宣布 实施"
        lookups.append({"lookup_group_id": group, "reference_date": str(date.date()), "navigation_family": row.navigation_family,
            "reference_ids": frame.reference_id.tolist(), "fixed_query": query,
            "search_domain_scope": ["pbc.gov.cn", "gov.cn", "csrc.gov.cn"], "maximum_native_sources": 2,
            "selection_rule": "与完整参考原文所述工具相符的官方原宣布/实施或同期政府报道；优先原发机关，无盈亏筛选。未匹配保留未知。"})
    parent.write(OUT / "lookup_plan.json", {"at": parent.original.now(), "groups": lookups, "native_source_cap_per_group": 2})
    paths = [Path(__file__), Path(inputs.__file__), ROOT / "tests/test_original_policy_announcement_trace_v1.py",
        ROOT / "docs/510300_ORIGINAL_POLICY_ANNOUNCEMENT_TRACE_V1.md", RATES, CATALOG, CHAINS, DAILY, EVENTS, KEYS, GOV_OLD]
    paths.extend(ROOT / path for path in pd.read_parquet(RATES).raw_path)
    paths.extend(CHAIN_ROOT / path for path in pd.read_parquet(CHAINS).source_path)
    unique_paths = list(dict.fromkeys(paths))
    parent.write(OUT / "protocol.json", {"at": parent.original.now(), "registration": "TECH.R229", "decision": "TECH.R230",
        "purpose": "ORIGINAL_ANNOUNCEMENT_SOURCE_CLOCK_TRACE_NOT_FINANCIAL",
        "references": 97, "original_rate_identities_not_changes": 25, "rrr_navigation_records_not_cuts": 45,
        "capital_navigation_records_not_shocks": 3, "original_chain_nodes": 24,
        "lookup_groups": len(lookups), "all_reference_ids": records.reference_id.tolist(),
        "scope": "全固定母集，不按盈利日期选取；FX准备金、考核制度、人民币RRR与资本工具分别保留。",
        "rate_role": "核对25已保存实际操作原文及原钟；首次观察变化不自动等于最早宣布。",
        "old_chain_role": "原人工核对信息及原上界原样复用，共同来源保留；不扩充成全国新闻全集。",
        "new_source_clock": "秒级原页面时间，日期级用日终上界；转载公开日期不能提前到正文叙述的事件日。当前网页历史首版未认证。",
        "search_scope": "固定分组查询，官方域名，每组最多2个原生来源，无盲目重试；搜索结果只作线索。",
        "unknown": "未搜索/未匹配/获取失败/原钟未知各保留；无节点不是无政策。",
        "daily_clock": "原16:00观察，不将之后公开的内容解释成已形成收盘价格；不计算或假设同收盘成交。",
        "description": "原3488日、143原进入点、19关键点；完整成功/反例不改变，19因果前缀。",
        "known_correction": "R228所称gov_20190104为01-04来源不准确；该原文件页面标注01-05，宣布日与转载日另列。旧实验不覆盖。",
        "new_accounts": 0, "new_fits": 0, "new_market_requests": 0, "financial_admission": "NOT_ADMITTED_NOT_RUN",
        "historical_first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "files": [{"path": relative(path), "sha256": parent.digest(path)} for path in unique_paths]})
    print(f"R229固定97参考身份、{len(lookups)}查询组；原25操作/24链复用，未知不补，零金融。", flush=True)


def fetch(item):
    destination = OUT / "raw" / (item["source_id"] + ".html")
    receipt = OUT / "receipts" / (item["source_id"] + ".json")
    if receipt.exists() or destination.exists():
        raise RuntimeError("来源已尝试，禁止因观测超时重新请求。")
    result = {**item, "at": parent.original.now(), "status": "FETCH_FAILED", "http_status": None,
        "raw_path": None, "sha256": None, "historical_first_vintage_authenticated": False}
    try:
        response = requests.get(item["url"], timeout=(10, 35), headers={"User-Agent": "Mozilla/5.0"})
        result["http_status"] = response.status_code
        result["effective_url"] = response.url
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(response.content)
        result.update(status="FETCHED", raw_path=relative(destination), sha256=parent.digest(destination), bytes=len(response.content))
    except Exception as exc:
        result["error"] = str(exc)
    result["finished_at"] = parent.original.now()
    parent.write(receipt, result)
    return result


def collect():
    protocol = parent.read(OUT / "protocol.json")
    for item in protocol["files"]:
        if parent.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("原登记输入改变。")
    if (OUT / "native_collection_started.json").exists():
        raise RuntimeError("原生采集已开始，保留同一过程，不启动第二次。")
    selection = parent.read(OUT / "lookup_source_selection.json")
    all_groups = {item["lookup_group_id"]: item for item in parent.read(OUT / "lookup_plan.json")["groups"]}
    items = selection["native_sources"]
    if len({item["source_id"] for item in items}) != len(items):
        raise ValueError("新来源身份重复。")
    for group_id, group in all_groups.items():
        if sum(group_id in item["lookup_group_ids"] for item in items) > group["maximum_native_sources"]:
            raise ValueError("查询组新来源超出事先上限。")
    parent.write(OUT / "native_collection_started.json", {"at": parent.original.now(), "sources": len(items), "selection_sha256": parent.digest(OUT / "lookup_source_selection.json")})
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, items))
    parent.write(OUT / "source_manifest.json", {"at": parent.original.now(), "native_requests": len(items), "sources": results})
    print(f"原公告采集终态：{len(results)}请求，{sum(item['status']=='FETCHED' for item in results)}成功；失败保留不重试。", flush=True)


def build():
    if (OUT / "observation_started.json").exists() or (OUT / "summary.json").exists():
        raise RuntimeError("原公告描述已开始或完成，不重复运行。")
    protocol = parent.read(OUT / "protocol.json")
    for item in protocol["files"]:
        if parent.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("原登记输入改变。")
    selection = parent.read(OUT / "lookup_source_selection.json")
    manifest = parent.read(OUT / "source_manifest.json")
    parent.write(OUT / "observation_started.json", {"at": parent.original.now(), "new_accounts": 0})
    rates = pd.read_parquet(RATES)
    rate_rows, nodes = [], []
    for index, item in enumerate(rates.to_dict("records"), 1):
        path = ROOT / item["raw_path"]
        if parent.digest(path) != item["raw_sha256"]:
            raise ValueError("原利率原文身份改变。")
        record = inputs.observed_rate(path.read_bytes(), item)
        node_id = f"RATE_{index:02}"
        rate_rows.append({"node_id": node_id, **record, "source_url": item["source_url"], "raw_path": item["raw_path"], "sha256": item["raw_sha256"]})
        nodes.append({"node_id": node_id, "source_available_upper": record["source_available_upper"],
            "common_source_id": item["raw_sha256"], "economic_identity": f"操作原记录|{pd.Timestamp(item['notice_date']).date()}|{item['seven_day_rate_percent']}",
            "information_role": "OPERATION_CONFIRMATION", "source_url": item["source_url"], "source_path": item["raw_path"]})
    table("全部25利率操作原文_改变与首发不混同", pd.DataFrame(rate_rows))
    chain = pd.read_parquet(CHAINS)
    for item in chain.to_dict("records"):
        path = CHAIN_ROOT / item["source_path"]
        if parent.digest(path) != item["source_sha256"]:
            raise ValueError("原核对链原文身份改变。")
        nodes.append({"node_id": "CHAIN_"+item["node_id"], "source_available_upper": inputs.clock(item["source_available_upper"]),
            "common_source_id": item["source_sha256"], "economic_identity": "原链|"+item["chain"]+"|"+str(item["economic_event_date"]),
            "information_role": "ANNOUNCEMENT" if item["stage"] == "宣布" else "ORIGINAL_CHAIN_CONFIRMATION",
            "source_url": item["source_url"], "source_path": relative(path)})
    catalog_references = pd.read_parquet(OUT / "results/全部97原参考身份_不是97独立政策.parquet")
    source_rows = []
    native_by_id = {item["source_id"]: item for item in manifest["sources"]}
    for item in selection["native_sources"]:
        receipt = native_by_id[item["source_id"]]
        row = {**item, "status": receipt["status"], "source_available_upper": pd.NaT, "clock_precision": "UNKNOWN",
            "raw_path": receipt["raw_path"], "sha256": receipt["sha256"], "text": None, "title": None}
        if receipt["status"] == "FETCHED":
            try:
                path = ROOT / receipt["raw_path"]
                if parent.digest(path) != receipt["sha256"]:
                    raise ValueError("新原文身份不一致。")
                doc = inputs.source_document(path.read_bytes())
                evidence = item["evidence_terms"]
                if not all(term in doc["text"] for term in evidence):
                    raise ValueError("指定官方原文工具核对词未全部匹配。")
                if pd.isna(doc["source_available_upper"]):
                    raise ValueError("原文没有可核对公开钟。")
                row.update(doc)
                row["status"] = "SOURCE_CONTENT_AND_CLOCK_MATCHED_CURRENT_PAGE_NOT_FIRST_VINTAGE"
                nodes.append({"node_id": item["source_id"], "source_available_upper": doc["source_available_upper"],
                    "common_source_id": receipt["sha256"], "economic_identity": item["economic_identity"],
                    "information_role": item["information_role"], "source_url": item["url"], "source_path": receipt["raw_path"]})
            except Exception as exc:
                row["status"] = "SOURCE_PARSING_OR_MATCH_UNKNOWN"
                row["error"] = str(exc)
        source_rows.append(row)
    sources = pd.DataFrame(source_rows)
    table("全部新官方来源_钟内容与失败未知", sources)
    source_map = []
    matched_status = "SOURCE_CONTENT_AND_CLOCK_MATCHED_CURRENT_PAGE_NOT_FIRST_VINTAGE"
    for item in catalog_references.to_dict("records"):
        if item["lookup_group_id"] is None or pd.isna(item["lookup_group_id"]):
            status = "ORIGINAL_SAVED_SOURCE_REUSED"
            ids = item["reference_id"]
        else:
            hits = sources.loc[sources.lookup_group_ids.apply(lambda values: item["lookup_group_id"] in values)]
            good = hits.loc[hits.status.eq(matched_status)]
            status = "MATCHED_REFERENCE_SOURCE_NOT_GLOBAL_FIRST_PUBLICATION" if len(good) else "SOURCE_UNKNOWN_OR_NOT_MATCHED"
            ids = "|".join(good.source_id)
        source_map.append({**item, "trace_status": status, "matched_source_ids": ids,
            "complete_policy_news_coverage": "NOT_ESTABLISHED", "global_first_publication_certified": False})
    table("全部97参考身份_原公告覆盖与未知", pd.DataFrame(source_map))
    old_gov = inputs.source_document(GOV_OLD.read_bytes())
    table("原政府转载日期纠正_事件日与来源日分开", pd.DataFrame([{
        "file": relative(GOV_OLD), "sha256": parent.digest(GOV_OLD), "announced_event_date_in_text": "2019-01-04",
        "source_available_upper": old_gov["source_available_upper"], "precision": old_gov["clock_precision"],
        "correction": "R228把该原文件描述为01-04来源不准确；实际页标01-05，旧文件与旧实验不改。"}]))
    node_frame = pd.DataFrame(nodes)
    table("全部已核对来源节点_共同源与角色保留", node_frame)
    observed = pd.read_parquet(DAILY)
    data = inputs.daily_known(observed, node_frame)
    table("全部3488日_已记录公告操作与资料覆盖", data)
    event_dates = pd.read_parquet(EVENTS)[["date", "stage_entry_type"]]
    event_dates["date"] = pd.to_datetime(event_dates.date).astype("datetime64[ns]")
    event_data = event_dates.merge(data, on="date", validate="one_to_one")
    table("全部143原点位_真实来源可知及共同源", event_data)
    keys = pd.read_parquet(KEYS)
    keys["date"] = pd.to_datetime(keys.date).astype("datetime64[ns]")
    key_data = keys.merge(data.drop(columns="decision_time"), on="date", validate="one_to_one")
    table("全部19具体点位_量价行业宏观与公告钟", key_data)
    checks = []
    for cut in keys.date:
        decision = inputs.clock(cut) + pd.Timedelta(hours=16)
        prefix_nodes = node_frame.loc[node_frame.source_available_upper.map(inputs.clock).le(decision)].copy()
        prefix = inputs.daily_known(observed.loc[observed.date.le(cut)], prefix_nodes)
        target = data.loc[data.date.le(cut)].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix, target, check_exact=True)
        checks.append({"cut": str(cut.date()), "all_prior_rows_exact": True})
    unchanged = 0
    for item in protocol["files"]:
        if parent.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("运行后原输入改变。")
        unchanged += 1
    parent.write(OUT / "summary.json", {"at": parent.original.now(), "registration": "TECH.R229", "decision": "TECH.R230",
        "status": "FIXED_ORIGINAL_POLICY_SOURCE_TRACE_AND_DAILY_DESCRIPTION_COMPLETE_NOT_FINANCIAL",
        "original_reference_identities": 97, "original_rate_sources_reused_and_verified": len(rate_rows),
        "original_chain_nodes_reused": len(chain), "fixed_lookup_groups": protocol["lookup_groups"],
        "new_native_requests": manifest["native_requests"], "new_native_received": int(sum(item["status"] == "FETCHED" for item in manifest["sources"])),
        "new_sources_content_and_clock_matched": int(sources.status.eq(matched_status).sum()),
        "all_reference_trace_status": pd.DataFrame(source_map).trace_status.value_counts().to_dict(),
        "actual_source_nodes": len(node_frame), "unique_common_sources": node_frame.common_source_id.nunique(),
        "all_daily_rows": len(data), "all_original_events": len(event_data), "all_key_rows": len(key_data),
        "events_with_any_new_recorded_node": int(event_data.new_recorded_nodes.gt(0).sum()),
        "events_with_new_recorded_announcement": int(event_data.new_recorded_announcement_nodes.gt(0).sum()),
        "prefix_checks": checks, "necessary_tests_passed": 6, "input_files_unchanged": unchanged,
        "old_gov_page_corrected_source_date": str(old_gov["source_available_upper"].date()),
        "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0,
        "financial_admission": "NOT_ADMITTED_NOT_RUN", "financial_metrics": "NOT_COMPUTED",
        "global_first_publication": "NOT_CERTIFIED", "complete_policy_news_coverage": "NOT_ESTABLISHED",
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False})
    print("R230全部原参考来源和日历解释完成；政府转载日期纠正，未知保留，零金融。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="官方原宣布来源固定追溯")
    parser.add_argument("action", choices=["freeze", "collect", "build"])
    action = parser.parse_args().action
    {"freeze": freeze, "collect": collect, "build": build}[action]()


if __name__ == "__main__":
    main()
