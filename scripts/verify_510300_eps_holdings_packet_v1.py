"""离线验证年报持仓原文、已有盈利事实、覆盖边界和包内文件。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.eps_disclosed_holdings_diagnostic_v1 import OUT, computation, identity, now, read, save


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    indexed = 0
    index = ROOT / "FILE_INDEX.csv"
    if index.exists():
        with index.open(encoding="utf-8-sig", newline="") as stream:
            entries = list(csv.DictReader(stream))
        if len({x["path"] for x in entries}) != len(entries):
            raise ValueError("文件索引重复")
        for item in entries:
            p = ROOT / item["path"]
            if p.stat().st_size != int(item["size_bytes"]) or hashlib.sha256(p.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("文件索引不匹配：" + item["path"])
            indexed += 1
        print(f"已核对包内{indexed}个索引文件。", flush=True)
    for item in read(OUT / "diagnostic_claim.json")["inputs"]:
        if identity(ROOT / item["path"]) != item:
            raise ValueError("冻结诊断输入不匹配")
    result = read(OUT / "result.json")
    text_pages = 0
    for report in result["report_admissions"]:
        extracted = read(OUT / f"annual_{report['year']}_pages.json")
        targets = {1, report["fund_nav_source_page"], report["stock_total_source_page"], *report["stock_table_pages"]}
        with pdfplumber.open(ROOT / report["pdf"]["path"]) as pdf:
            if len(pdf.pages) != len(extracted["pages"]) or pdf.metadata != extracted["metadata"]:
                raise ValueError("原PDF页数或元数据不同")
            for page in sorted(targets):
                if (pdf.pages[page-1].extract_text() or "") != extracted["pages"][page-1]["text"]:
                    raise ValueError("原PDF重新提取的持仓或身份页面不同")
                text_pages += 1
        print(f"{report['year']}年原PDF的相关页面已重新提取并严格比较。", flush=True)
    details, _, reproduced = computation()
    if reproduced != {k:v for k,v in result.items() if k != "completed_at"}:
        raise ValueError("结果数字不一致")
    saved = pd.read_parquet(OUT / "holdings_and_eps_evidence.parquet")
    details["origin"] = details.origin.astype("datetime64[ns]")
    saved["origin"] = saved.origin.astype("datetime64[ns]")
    pd.testing.assert_frame_equal(details, saved, check_exact=True)
    closure = read(OUT / "selected_earnings_source_closure.json")
    for item in closure["files"]:
        if identity(ROOT / item["path"]) != item:
            raise ValueError("旧研报原件或事实哈希变化")
    selected = pd.read_parquet(OUT / "selected_institution_company_evidence.parquet")
    full = pd.read_parquet(ROOT / "reports/research/510300_forward_eps_two_institution_features_v3/institution_company_month_evidence.parquet")
    comparison = full.loc[full.origin.isin(selected.origin.unique())].reset_index(drop=True)
    selected = selected.reset_index(drop=True)
    pd.testing.assert_frame_equal(comparison, selected, check_exact=True)
    from scripts.prepare_510300_eps_holdings_review_evidence_v1 import symmetric
    docs = {(x["institution"],x["report_id"]):x for x in closure["documents"]}
    facts_checked = 0
    for row in selected.to_dict("records"):
        aid = row["report_id"]
        if pd.isna(aid):
            continue
        institution = row["institution"]
        item = docs[(institution,aid)]
        fact = read(ROOT / item["facts"])
        meta = read(ROOT / item["record"])
        clocks = [meta["directory_record"]["publishDate"][:10], meta["provider_metadata"]["notice_date"][:10],
                  meta["provider_metadata"]["eitime"][:10]]
        clocks += [fact[k] for k in ["report_internal_date","conservative_information_date"] if fact.get(k)]
        if max(clocks) != row["information_date"] or not pd.Timestamp(max(clocks)) < row["origin"]:
            raise ValueError("旧当前报告可用日期与保存事实不一致")
        by_year = {int(x["target_fiscal_year"]):x for x in fact["facts"]}
        target = row["origin"].year+1
        future, current = by_year[target], by_year[target-1]
        values = {"eps_growth":symmetric(future["eps_value_exact"],current["eps_value_exact"]),
                  "reported_earnings_yield":np.nan,"profit_revision":np.nan,"raw_eps_revision_unadjusted":np.nan}
        pe = future.get("pe_value_exact")
        if pe is not None and np.isfinite(float(pe)) and float(pe) != 0:
            values["reported_earnings_yield"] = 1/float(pe)
        if pd.notna(row["prior_report_id"]):
            prior_doc = docs[(institution,row["prior_report_id"])]
            prior_fact = read(ROOT/prior_doc["facts"])
            prior = next(x for x in prior_fact["facts"] if int(x["target_fiscal_year"]) == target)
            values["raw_eps_revision_unadjusted"] = symmetric(future["eps_value_exact"],prior["eps_value_exact"])
            if future.get("net_profit_source_label") == prior.get("net_profit_source_label"):
                values["profit_revision"] = symmetric(future.get("net_profit_value_exact"),prior.get("net_profit_value_exact"))
        for field, value in values.items():
            if not np.isclose(value,row[field],rtol=1e-12,atol=1e-12,equal_nan=True):
                raise ValueError("原盈利事实重新计算不一致")
            facts_checked += 1
    monthly = pd.read_parquet(ROOT / "reports/research/510300_forward_eps_two_institution_features_v3/monthly_two_institution_features.parquet")
    valid = monthly.loc[monthly.common_sources_valid]
    support = read(OUT / "saved_eps_support_diagnostic.json")
    for field, counts in support["statistics"].items():
        values = valid[field]
        current = {"negative_months":int(values.lt(0).sum()),"zero_months":int(values.eq(0).sum()),
                   "positive_months":int(values.gt(0).sum()),"minimum":float(values.min()),
                   "median":float(values.median()),"maximum":float(values.max())}
        if current != counts:
            raise ValueError("已有信号取值统计不一致")
    for source in support["existing_account_source_files"]:
        if identity(ROOT/source["path"]) != source:
            raise ValueError("既有账户结果来源变化")
    receipt = {"status":"PASS_OFFLINE_ORIGINAL_TABLE_EPS_FACT_AND_COVERAGE_RECOMPUTATION", "completed_at":now(),
               "indexed_files_checked":indexed,"original_fund_pdf_pages_reextracted":text_pages,
               "fund_reports":len(result["report_admissions"]),"holdings_rows":len(details),
               "old_earnings_original_documents":closure["source_documents"],"old_earnings_fact_fields_checked":facts_checked,
               "saved_institution_company_rows":len(selected),"existing_valid_months":len(valid),
               "new_network_requests":0,"new_model_fits":0,"new_accounts_generated":0,"new_random_draws":0,
               "old_full_earnings_directory_completeness_reproved":False,"old_account_simulations_repeated":False,
               "external_gpt_review":False,"goal_achieved":False}
    save(Path(args.receipt),receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
