"""从固定原PDF核对叙述前后值与既有预测表和旧月度记录的关系。"""
from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_eps_explicit_revision_novelty_diagnostic_v1.json"
PROTOCOL = ROOT / "docs/510300_EPS_EXPLICIT_REVISION_NOVELTY_DIAGNOSTIC_V1.md"
OUT = ROOT / "reports/research/510300_eps_explicit_revision_novelty_diagnostic_v1"
RELATED = ["research/forward_eps_monthly_policy_v1.py", "research/forward_eps_two_institution_features_v3.py",
           "research/forward_eps_revision_distribution_features_v1.py", "research/forward_eps_revision_diagnostic_v1.py",
           "docs/510300_FORWARD_EPS_REVISION_DISTRIBUTION_CONTINUATION_20260907.md",
           "research/a_share_hs_earnings_forecast_revision_pdf_table_parser_v1.py",
           "config/a_share_hs_official_nonoverlapping_upward_profit_forecast_revision_v1.yaml"]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def identity(path):
    return {"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def compact(value):
    return re.sub(r"[\s,]", "", unicodedata.normalize("NFKC", value))


def census(config):
    queue = pd.read_parquet(ROOT / config["source_queue"])
    selected = queue.loc[queue.ts_code.isin(config["symbols"]) & queue.publishDate.astype(str).str[:10].between(config["census_start"], config["census_end"])].copy()
    if sorted(selected.infoCode) != sorted(config["expected_census_ids"]) or selected.infoCode.duplicated().any():
        raise ValueError("既有选择表的固定两公司报告候选集改变")
    return selected.sort_values(["ts_code", "publishDate", "infoCode"]).reset_index(drop=True)


def source_paths(config):
    paths = [CONFIG, PROTOCOL, Path(__file__), ROOT / config["source_queue"], ROOT / config["old_institution_features"]]
    paths += [ROOT / relative for relative in RELATED]
    for aid in census(config).infoCode:
        record_path = ROOT / config["original_root"] / "document_records" / (aid + ".json")
        record = read(record_path)
        paths += [record_path, ROOT / config["fact_root"] / "document_facts" / (aid + ".json"),
                  ROOT / record["source"]["raw_pdf_path"], ROOT / record["source"]["raw_html_path"],
                  ROOT / record["directory_record"]["source_raw_path"]]
    return sorted(set(paths))


def document(aid, config, extract):
    record = read(ROOT / config["original_root"] / "document_records" / (aid + ".json"))
    facts = read(ROOT / config["fact_root"] / "document_facts" / (aid + ".json"))
    path = ROOT / record["source"]["raw_pdf_path"]
    if identity(path)["sha256"] != record["source"]["pdf_sha256"]:
        raise ValueError("原PDF哈希与采集记录不符")
    if str(record["provider_metadata"]["company_code"]) != "80000007":
        raise ValueError("原件机构身份不是国信证券")
    with pdfplumber.open(path) as pdf:
        pages = [{"page": i + 1, "text": pdf.pages[i].extract_text() or ""} for i in range(2)]
        saved = {"report_id": aid, "original_pdf": identity(path), "full_pdf_pages": len(pdf.pages), "selected_pages": pages}
    if extract is True:
        save(OUT / "pages" / (aid + ".json"), saved)
    elif extract is False and saved != read(OUT / "pages" / (aid + ".json")):
        raise ValueError("原PDF前两页重新提取不一致")
    front = compact(pages[0]["text"])
    if record["ts_code"] not in front:
        raise ValueError("原PDF首页证券代码不匹配")
    match = re.search(r"证券研究报告\|(\d{4})年(\d{2})月(\d{2})日", front)
    if match is None or "-".join(match.groups()) != facts["report_internal_date"]:
        raise ValueError("原PDF署期与保存事实不一致")
    values = {}
    table_page = facts["table"]["page"]
    if table_page not in (1, 2):
        raise ValueError("固定报告的原预测表不在前两页")
    table_text = compact(pages[table_page - 1]["text"])
    for fact in facts["facts"]:
        if compact(fact["header_raw"]) not in table_text or compact(fact["net_profit_source_raw"]) not in table_text:
            raise ValueError("原PDF表格年度或精确预测行不匹配")
        values[int(fact["target_fiscal_year"])] = str(Decimal(fact["net_profit_value_exact"]))
    metadata = record["provider_metadata"]
    date = max(record["directory_record"]["publishDate"][:10], metadata["notice_date"][:10], metadata["eitime"][:10], facts["report_internal_date"], facts["conservative_information_date"])
    return {"report_id": aid, "symbol": record["ts_code"], "information_date": date, "internal_date": facts["report_internal_date"],
            "original_pdf": identity(path), "full_pdf_pages": saved["full_pdf_pages"], "table_page": table_page,
            "reported_million_values": values, "label": facts["table"]["optional"]["net_profit"]["label"],
            "compact_front_text": front}


def parse_narrative(doc, style):
    front = doc["compact_front_text"]
    if style == "PARENTHESIS_PREVIOUS_VECTOR":
        pattern = r"2025-2027年归母净利润为([\d/]+)亿\(前值为([\d/\-]+)亿\)"
    else:
        pattern = r"2025-2027年归母净利润至([\d/]+)亿元\(2025-2026原预测([\d/]+)亿元\)"
    matches = list(re.finditer(pattern, front))
    if len(matches) != 1:
        raise ValueError("固定原文的年度及前后值叙述不是唯一匹配")
    match = matches[0]
    current = match[1].split("/")
    prior = match[2].split("/")
    if len(current) != 3 or len(prior) not in (2, 3):
        raise ValueError("叙述年度与金额数量不相等")
    if len(prior) == 3 and prior[2] != "-":
        raise ValueError("2027年前值不应静默纳入未登记比较")
    return {"source_fragment": match[0], "current_100_million": dict(zip([2025, 2026, 2027], current)),
            "prior_100_million": dict(zip([2025, 2026], prior[:2])),
            "prior_2027_status": "NOT_COMPARABLE_NEW_TARGET_YEAR_NO_PRIOR_FORECAST"}


def matches_display(amount_million, displayed_100_million):
    return abs(Decimal(amount_million) / 100 - Decimal(displayed_100_million)) < Decimal("0.5")


def safe_records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=15))


def compute(extract=None):
    config = read(CONFIG)
    selection = census(config)
    documents = {aid: document(aid, config, extract) for aid in selection.infoCode}
    old = pd.read_parquet(ROOT / config["old_institution_features"])
    old = old.loc[old.ts_code.isin(config["symbols"]) & old.institution.eq("guosen") & old.origin.between(config["old_feature_slice_start"], config["old_feature_slice_end"])].sort_values(["origin", "ts_code"]).reset_index(drop=True)
    fields = ["origin", "ts_code", "institution", "report_id", "prior_report_id", "profit_revision", "target_fiscal_year", "status"]
    old = old[fields].copy()
    target_year = config["target_fiscal_year"]
    results = []
    for target in config["targets"]:
        current = documents[target["report_id"]]
        narrative = parse_narrative(current, target["narrative_style"])
        for year, displayed in narrative["current_100_million"].items():
            if not matches_display(current["reported_million_values"][year], displayed):
                raise ValueError("叙述当前预测与原表精确数值不在相同舍入区间")
        candidates = [doc for doc in documents.values() if doc["symbol"] == target["symbol"] and doc["information_date"] < current["information_date"]]
        observations = []
        for candidate in candidates:
            compared = {year: year in candidate["reported_million_values"] and matches_display(candidate["reported_million_values"][year], narrative["prior_100_million"][year]) for year in config["joint_prior_fiscal_years"]}
            observations.append({"report_id": candidate["report_id"], "information_date": candidate["information_date"],
                                 "precise_prior_vector_million": {year: candidate["reported_million_values"].get(year) for year in config["joint_prior_fiscal_years"]},
                                 "year_matches": compared, "joint_match": all(compared.values())})
        matched = [row for row in observations if row["joint_match"]]
        if len(matched) != 1:
            raise ValueError("给定既有候选集中的联合年度对应不是唯一")
        prior = documents[matched[0]["report_id"]]
        new_value = Decimal(current["reported_million_values"][target_year])
        old_value = Decimal(prior["reported_million_values"][target_year])
        symmetric = float(2 * (new_value - old_value) / (abs(new_value) + abs(old_value)))
        rows = old.loc[old.ts_code.eq(target["symbol"])]
        exact_pair = rows.loc[rows.report_id.eq(current["report_id"]) & rows.prior_report_id.eq(prior["report_id"]) & rows.target_fiscal_year.eq(target_year)]
        finite_pair = exact_pair.loc[np.isfinite(exact_pair.profit_revision)]
        if len(finite_pair) and not np.allclose(finite_pair.profit_revision.to_numpy(), symmetric, rtol=0, atol=1e-14):
            raise ValueError("既有月度记录的同一报告对数值无法复现")
        origin = rows.loc[rows.origin.eq(pd.Timestamp(config["diagnostic_origin"]))]
        if len(origin) != 1 or origin.iloc[0].report_id != current["report_id"]:
            raise ValueError("固定原点的既有月度报告不同")
        origin_row = origin.iloc[0]
        selected_prior = documents[origin_row.prior_report_id]
        same_label = current["label"] == selected_prior["label"]
        if np.isfinite(origin_row.profit_revision) or same_label:
            raise ValueError("本轮固定案例不是所记录的旧标签缺失")
        results.append({"name": target["name"], "symbol": target["symbol"], "current_report_id": current["report_id"],
            "narrative": narrative, "candidate_observations": observations,
            "single_2026_year_candidate_count": sum(row["year_matches"][target_year] for row in observations),
            "joint_two_year_candidate_count": len(matched), "matched_prior_report_id": prior["report_id"],
            "matched_prior_information_date": prior["information_date"],
            "worldwide_unique_historical_version_proven": False, "report_explicitly_names_prior_id_proven": False,
            "both_original_tables_existed_in_previous_local_source": True,
            "both_table_dates_strictly_before_diagnostic_origin": max(prior["information_date"], current["information_date"]) < config["diagnostic_origin"],
            "new_original_numeric_forecast_values": 0, "target_fiscal_year": target_year,
            "current_exact_million": str(new_value), "matched_prior_exact_million": str(old_value),
            "relative_revision_from_existing_tables": float(new_value / old_value - 1), "symmetric_revision_from_existing_tables": symmetric,
            "old_same_pair_observations": safe_records(exact_pair), "old_same_pair_finite_count": len(finite_pair),
            "old_diagnostic_origin_observation": safe_records(origin)[0],
            "old_origin_current_label": current["label"], "old_origin_prior_label": selected_prior["label"],
            "old_origin_prior_differs_from_narrative_match": origin_row.prior_report_id != prior["report_id"],
            "timing_or_representation_difference_present": True,
            "future_price_predictive_value": "NOT_COMPUTED", "new_account": "NOT_RUN"})
    result = {"study_id": config["study_id"], "status": "TWO_NARRATIVES_REUSE_EXISTING_NUMBERS_TIMING_AND_ALIAS_GAPS_RETAINED",
              "census_reports": len(documents), "census_complete_only_within_existing_selected_queue": True,
              "full_original_pdf_pages": sum(doc["full_pdf_pages"] for doc in documents.values()),
              "source_pages_reextracted": 2 * len(documents), "diagnostic_events": len(results),
              "events": results, "old_company_month_rows": len(old),
              "new_original_numeric_forecast_values": 0, "novel_numeric_source_hypothesis_for_these_two_reports": "NOT_SUPPORTED",
              "all_possible_report_text_features_rejected": False, "timing_alternative_tested": False,
              "label_alias_admitted": False, "old_ninety_day_rule_changed": False,
              "new_network_requests": 0, "new_return_labels": 0, "new_model_fits": 0, "new_accounts_generated": 0,
              "paused_queues_resumed": False, "goal_achieved": False}
    return old, result


def verify(receipt):
    claim = read(OUT / "claim.json")
    for row in claim["files"]:
        if identity(ROOT / row["path"]) != row:
            raise ValueError("冻结输入或代码改变")
    old, result = compute(extract=False)
    saved = read(OUT / "result.json")
    # JSON中的年度字典键是字符串，比较前只规范这种无信息损失的序列化表示。
    normalized = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if normalized != {key: value for key, value in saved.items() if key != "completed_at"}:
        raise ValueError("保存结论无法由原件重现")
    existing = pd.read_parquet(OUT / "old_company_month_observations.parquet")
    for frame in (old, existing):
        frame["origin"] = frame.origin.astype("datetime64[ns]")
    pd.testing.assert_frame_equal(old, existing, check_exact=True)
    record = {"status": "PASS_SEVEN_ORIGINAL_PDFS_TWO_YEAR_MATCHES_AND_OLD_RECORDS_RECOMPUTED", "completed_at": now(),
              "frozen_files_verified": len(claim["files"]), "original_pdfs": 7, "source_pages_reextracted": 14,
              "old_company_month_rows": len(old), "new_original_forecast_numbers": 0,
              "new_network_requests": 0, "new_model_fits": 0, "new_accounts": 0, "goal_achieved": False}
    save(receipt, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "run", "verify"], required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "verify":
        if args.receipt is None:
            raise ValueError("复核必须指定新回执路径")
        verify(args.receipt)
        return
    if args.mode == "run":
        save(OUT / "claim.json", {"started_at": now(), "previews_already_seen": True, "new_strategy_preregistration": False,
                                  "files": [identity(path) for path in source_paths(read(CONFIG))]})
    old, result = compute(extract=True if args.mode == "run" else None)
    if args.mode == "run":
        old.to_parquet(OUT / "old_company_month_observations.parquet", index=False)
        old.to_csv(OUT / "既有两公司逐月修正记录.csv", index=False, encoding="utf-8-sig")
        save(OUT / "result.json", {"completed_at": now(), **result})
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
