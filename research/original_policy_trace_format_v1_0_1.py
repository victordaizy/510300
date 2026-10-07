"""读取已保存的官方页面格式，补齐两个原反例与既有来源关联；不采集或运行金融。"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

from research import original_policy_announcement_trace_v1 as original
from research import original_policy_announcement_trace_inputs_v1 as inputs

ROOT, SOURCE, parent = original.ROOT, original.OUT, original.parent
OUT = SOURCE / "implementation_v1_0_1"
NEGATIVES = original.DESCRIPTION / "两个新增亏损信号_全部进入信息与上下文.parquet"
MATCHED = "SOURCE_CONTENT_AND_CLOCK_MATCHED_CURRENT_PAGE_NOT_FIRST_VINTAGE"
FORMATS = {
    "SOURCE_004": ("div.TRS_Editor", "div.time"),
    "SOURCE_005": ("article#articleins", "article#articleins"),
    "SOURCE_007": ("article#articleins", "article#articleins"),
    "SOURCE_008": ("div.detail-news", "div.content"),
    "SOURCE_010": ("div.article_con", "div.article_item"),
    "SOURCE_011": ("div.xxleftmain", "#ivs_date"),
    "SOURCE_012": ("#article-box", ".time-box"),
    "SOURCE_014": ("#zoomcon", "span.date"),
    "SOURCE_015": ("div.TRS_Editor", "div.time"),
    "SOURCE_016": ("div.TRS_Editor", "div.time"),
    "SOURCE_018": ("div.detBox", "div.info"),
    "SOURCE_019": ("section.detail_article_content", None),
}
LOCAL_LINKS = {
    "LOOKUP_98ffc1aeb74f3f00": ["LOCAL_GOV_20190105"],
    "LOOKUP_027161d5d91045e9": ["CHAIN_R01", "CHAIN_R02"],
    "LOOKUP_a9e8489112e53ade": ["CHAIN_R03"],
    "LOOKUP_d73188e538516bf6": ["CHAIN_C02"],
    "LOOKUP_17f24141c3951afb": ["CHAIN_C03"],
}


def relative(path: Path) -> str:
    return path.absolute().relative_to(ROOT).as_posix()


def table(name: str, frame: pd.DataFrame) -> None:
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def match_text(value: str) -> str:
    """仅删除排版空格，不修改原文内容、工具方向或参考时间。"""
    return re.sub(r"\s+", "", value)


def parse_saved(raw: bytes, body_selector: str, clock_selector: str | None) -> dict:
    soup = BeautifulSoup(raw, "html.parser", from_encoding="utf-8")
    body = soup.select_one(body_selector)
    clock_body = soup.select_one(clock_selector) if clock_selector else soup
    if body is None or clock_body is None:
        raise ValueError("登记的正文或公开日期结构缺失，保持未知。")
    text = inputs.clean(body.get_text(" ", strip=True))
    markup = inputs.clean(clock_body.get_text(" ", strip=True))
    second = re.search(r"20\d{2}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2}", markup)
    dated = re.search(r"20\d{2}[-/]\d{2}[-/]\d{2}", markup)
    if second:
        visible = inputs.clock(second[0].replace("/", "-"))
        precision = "SECOND_VISIBLE_CURRENT_PAGE_NOT_FIRST_VINTAGE"
        literal = second[0]
    elif dated:
        visible = inputs.clock(dated[0].replace("/", "-")) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        precision = "DATE_END_UPPER_VISIBLE_CURRENT_PAGE_NOT_FIRST_VINTAGE"
        literal = dated[0]
    else:
        raise ValueError("正文以外的公开钟未识别，不用正文事件日或网址补。")
    meta = soup.find("meta", attrs={"name": re.compile("PubDate|pubdate|publishdate", re.I)})
    metadata = str(meta.get("content", "")) if meta else ""
    meta_day = re.search(r"20\d{2}-\d{2}-\d{2}", metadata)
    conflict = bool(meta_day and pd.Timestamp(meta_day[0]).date() != visible.date())
    return {
        "title": inputs.clean(soup.title.get_text(" ", strip=True)) if soup.title else "",
        "text": text, "source_available_upper": pd.NaT if conflict else visible,
        "visible_publication_upper": visible, "clock_precision": precision,
        "visible_publication_literal": literal, "metadata_publication_literal": metadata,
        "metadata_visible_calendar_date_conflict": conflict,
        "clock_admission": "UNKNOWN_VISIBLE_METADATA_DATE_CONFLICT" if conflict else "VISIBLE_CLOCK_NOT_FIRST_VINTAGE",
    }


def combined_keys() -> pd.DataFrame:
    keys = pd.read_parquet(original.KEYS)
    negative = pd.read_parquet(NEGATIVES)
    result = pd.concat([keys, negative], ignore_index=True).sort_values("date").reset_index(drop=True)
    result["date"] = pd.to_datetime(result.date).astype("datetime64[ns]")
    if len(keys) != 17 or len(negative) != 2 or len(result) != 19 or not result.date.is_unique:
        raise ValueError("原17日及两个原反例不完整或日期重复。")
    return result


def freeze() -> None:
    if (OUT / "protocol.json").exists():
        raise RuntimeError("格式用途已经登记，不覆盖。")
    base = parent.read(SOURCE / "summary.json")
    tests = parent.read(OUT / "tests_receipt.json")
    if (base["all_key_rows"] != 17 or base["new_sources_content_and_clock_matched"] != 6
            or tests["passed"] != 3 or tests["exit_code"] != 0
            or tests["implementation_sha256"] != parent.digest(Path(__file__))):
        raise ValueError("原解析缺口与必要格式测试不一致。")
    paths = [Path(__file__), ROOT / "tests/test_original_policy_trace_format_v1_0_1.py",
             original.DAILY, original.EVENTS, original.KEYS, NEGATIVES, original.GOV_OLD,
             SOURCE / "protocol.json", SOURCE / "summary.json", SOURCE / "source_manifest.json"]
    paths.extend(sorted((SOURCE / "results").glob("*.parquet")))
    manifest = parent.read(SOURCE / "source_manifest.json")
    paths.extend(ROOT / row["raw_path"] for row in manifest["sources"] if row["status"] == "FETCHED")
    paths = list(dict.fromkeys(paths))
    parent.write(OUT / "protocol.json", {
        "at": parent.original.now(), "registration": "TECH.R229", "decision": "TECH.R230",
        "implementation": "1.0.1_SOURCE_FORMAT_AND_ORIGINAL_CASE_COMPLETENESS",
        "purpose": "12保存格式来源、旧政府转载及原17+2反例完整连接；原六合格新源和所有冻结失败保留。",
        "body_and_publication_selectors": {key: list(value) for key, value in FORMATS.items()},
        "local_saved_links": LOCAL_LINKS,
        "clock": "只用可见公开钟；秒级保持、分级和日期级保守日终上界；元数据跨日冲突未知，不用URL或正文实施日期补。",
        "format_matching": "核对词只删除排版空格；所有经济内容、角色和源集合固定，不根据盈亏删选。",
        "description": "补两个原反例，原17实际描述保持留档；3488日、143事件和19日全部前缀；无节点不等于无政策。",
        "new_requests": 0, "new_accounts": 0, "new_fits": 0,
        "files": [{"path": relative(path), "sha256": parent.digest(path)} for path in paths],
    })
    print("R230格式实现固定：12已保存页面、旧来源关联及原19日；零请求和金融。", flush=True)


def check_inputs(protocol: dict) -> int:
    for item in protocol["files"]:
        if parent.digest(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("格式登记输入改变。")
    return len(protocol["files"])


def build() -> None:
    if (OUT / "started.json").exists() or (OUT / "summary.json").exists():
        raise RuntimeError("格式实现已开始或完成，不重复运行。")
    protocol = parent.read(OUT / "protocol.json")
    check_inputs(protocol)
    parent.write(OUT / "started.json", {"at": parent.original.now(), "new_requests": 0, "new_accounts": 0})
    sources = pd.read_parquet(SOURCE / "results/全部新官方来源_钟内容与失败未知.parquet")
    rows, additions, format_rows = [], [], []
    for item in sources.to_dict("records"):
        row = dict(item)
        if item["source_id"] in FORMATS and item["status"] == "SOURCE_PARSING_OR_MATCH_UNKNOWN":
            try:
                doc = parse_saved((ROOT / item["raw_path"]).read_bytes(), *FORMATS[item["source_id"]])
                if not all(match_text(term) in match_text(doc["text"]) for term in item["evidence_terms"]):
                    raise ValueError("原登记工具核对词仍未全部匹配，不改变核对要求。")
                row.update(doc)
                row["error"] = None
                if doc["metadata_visible_calendar_date_conflict"]:
                    row["status"] = "SOURCE_CONTENT_MATCHED_PUBLICATION_METADATA_DATE_CONFLICT"
                else:
                    row["status"] = MATCHED
                    additions.append({"node_id": item["source_id"], "source_available_upper": doc["source_available_upper"],
                        "common_source_id": item["sha256"], "economic_identity": item["economic_identity"],
                        "information_role": item["information_role"], "source_url": item["url"], "source_path": item["raw_path"]})
            except Exception as exc:
                row["status"] = "SOURCE_PARSING_OR_MATCH_UNKNOWN"
                row["error"] = str(exc)
            format_rows.append({"source_id": item["source_id"], "old_status": item["status"], "new_status": row["status"],
                                "raw_sha256": item["sha256"], "error": row.get("error")})
        rows.append(row)
    final_sources = pd.DataFrame(rows)
    table("全部22新来源_格式读取与原失败保留", final_sources)
    table("12原格式缺口_真实结果与公开钟冲突", pd.DataFrame(format_rows))
    original_nodes = pd.read_parquet(SOURCE / "results/全部已核对来源节点_共同源与角色保留.parquet")
    old_gov = inputs.source_document(original.GOV_OLD.read_bytes())
    if old_gov["source_available_upper"] != inputs.clock("2019-01-05 23:59:59"):
        raise ValueError("原2019政府转载可见公开日发生变化。")
    additions.append({"node_id": "LOCAL_GOV_20190105", "source_available_upper": old_gov["source_available_upper"],
        "common_source_id": parent.digest(original.GOV_OLD), "economic_identity": "2019-01-04|人民币RRR计划",
        "information_role": "OFFICIAL_REPRINT_REPORT", "source_url": "https://app.www.gov.cn/govdata/gov/201901/05/433817/article.html",
        "source_path": relative(original.GOV_OLD)})
    nodes = pd.concat([original_nodes, pd.DataFrame(additions)], ignore_index=True)
    pd.testing.assert_frame_equal(nodes.iloc[:len(original_nodes)].reset_index(drop=True), original_nodes, check_exact=True)
    table("全部来源节点_旧49与原六新源逐值保持", nodes)
    references = pd.read_parquet(SOURCE / "results/全部97参考身份_原公告覆盖与未知.parquet")
    trace_rows = []
    for item in references.to_dict("records"):
        row = dict(item)
        row["initial_trace_status"] = row["trace_status"]
        group = item["lookup_group_id"]
        if group is not None and not pd.isna(group):
            hits = final_sources.loc[final_sources.lookup_group_ids.apply(lambda values: group in values)]
            ids = list(hits.loc[hits.status.eq(MATCHED), "source_id"])
            ids.extend(LOCAL_LINKS.get(group, []))
            if ids:
                if not set(ids).issubset(set(nodes.node_id)):
                    raise ValueError("旧来源关联缺少实际节点。")
                row["trace_status"] = "MATCHED_REFERENCE_SOURCE_NOT_GLOBAL_FIRST_PUBLICATION"
                row["matched_source_ids"] = "|".join(dict.fromkeys(ids))
        trace_rows.append(row)
    table("全部97参考_新旧实际来源及仍未知", pd.DataFrame(trace_rows))
    table("五组原已保存来源关联_非新发现", pd.DataFrame([
        {"lookup_group_id": group, "node_ids": "|".join(ids), "new_request": False}
        for group, ids in LOCAL_LINKS.items()]))
    observed = pd.read_parquet(original.DAILY)
    daily = inputs.daily_known(observed, nodes)
    table("全部3488日_已记录来源与未知", daily)
    events = pd.read_parquet(original.EVENTS)[["date", "stage_entry_type"]]
    events["date"] = pd.to_datetime(events.date).astype("datetime64[ns]")
    joined = events.merge(daily, on="date", validate="one_to_one")
    table("全部143原进入事件_真正可知来源", joined)
    keys = combined_keys()
    final_keys = keys.merge(daily.drop(columns="decision_time"), on="date", validate="one_to_one")
    table("全部19关键日_原17与两假启动", final_keys)
    checks = []
    for cut in keys.date:
        decision = inputs.clock(cut) + pd.Timedelta(hours=16)
        prefix_nodes = nodes.loc[nodes.source_available_upper.map(inputs.clock).le(decision)].copy()
        prefix = inputs.daily_known(observed.loc[observed.date.le(cut)], prefix_nodes)
        pd.testing.assert_frame_equal(prefix, daily.loc[daily.date.le(cut)].reset_index(drop=True), check_exact=True)
        checks.append({"cut": str(cut.date()), "all_prior_rows_exact": True})
    checked = check_inputs(protocol)
    parent.write(OUT / "summary.json", {
        "at": parent.original.now(), "registration": "TECH.R229", "decision": "TECH.R230", "implementation": "1.0.1",
        "status": "SAVED_SOURCE_FORMAT_AND_ALL_ORIGINAL_CASES_COMPLETE_NOT_FINANCIAL",
        "original_reference_identities": 97, "fixed_lookup_groups": 42,
        "original_native_requests": 22, "original_native_received": 18, "original_native_failures": 4,
        "source_statuses": final_sources.status.value_counts().to_dict(),
        "new_source_content_and_clock_matched": int(final_sources.status.eq(MATCHED).sum()),
        "clock_conflicts_kept_unknown": int(final_sources.status.eq("SOURCE_CONTENT_MATCHED_PUBLICATION_METADATA_DATE_CONFLICT").sum()),
        "all_reference_trace_status": pd.DataFrame(trace_rows).trace_status.value_counts().to_dict(),
        "old_55_nodes_preserved_exact": True, "all_actual_nodes": len(nodes), "unique_common_sources": nodes.common_source_id.nunique(),
        "local_original_source_groups_reused": len(LOCAL_LINKS), "all_daily_rows": len(daily), "all_events": len(joined),
        "all_key_rows": len(final_keys), "initial_key_rows_were_17_not_19": True,
        "events_with_new_recorded_node": int(joined.new_recorded_nodes.gt(0).sum()),
        "events_with_new_announcement_node": int(joined.new_recorded_announcement_nodes.gt(0).sum()),
        "prefix_checks": checks, "necessary_format_tests_passed": 3, "input_files_unchanged": checked,
        "new_requests": 0, "new_accounts": 0, "new_fits": 0, "new_labels": 0, "new_market_bars": 0,
        "financial_admission": "NOT_ADMITTED_NOT_RUN", "financial_metrics": "NOT_COMPUTED",
        "global_first_publication": "NOT_CERTIFIED", "complete_policy_news_coverage": "NOT_ESTABLISHED",
        "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
    })
    print("R230已保存格式和原19点位完整终态；失败、元数据冲突、旧金融保持，零新请求。", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="原公告已保存格式及原案例完整用途")
    parser.add_argument("action", choices=["freeze", "build"])
    args = parser.parse_args()
    {"freeze": freeze, "build": build}[args.action]()


if __name__ == "__main__":
    main()
