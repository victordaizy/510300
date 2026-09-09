"""只读核对金融原始数值、来源页、单位、归属关系与每股收益算术。"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from decimal import Decimal
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text("utf-8"))


def compact(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).replace("—", "-").replace("−", "-"))


def amount(cell):
    text = str(cell).replace(",", "")
    return -Decimal(text[1:-1]) if text.startswith("(") else Decimal(text)


def verify(root=ROOT):
    out = root / "reports/research/510300_financial_original_facts_v1_1"
    sources = {}
    for folder in ["510300_financial_original_layout_inventory_v1", "510300_financial_report_subject_repair_v1", "510300_financial_ttm_dependencies_v1"]:
        for file in (root/"reports/research"/folder/"page_texts").glob("*.json"):
            d = read(file)
            sources[str(d["source"]["announcement_id"])] = d
    verified_pdfs = 0
    for aid, data in sources.items():
        s = data["source"]
        raw = root / s["raw_path"]
        if not raw.exists():
            raw = Path(r"E:\ResearchData\New project 8") / s["raw_path"]
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == s["sha256"], aid
        assert pd.Timestamp(s["event_publication_date"]).strftime("%Y-%m-%d") in s["official_pdf_url"], aid
        verified_pdfs += 1
    facts = pd.read_parquet(out / "current_original_financial_facts.parquet")
    assert len(facts) == 94 and not facts.duplicated(["announcement_id", "metric_id"]).any()
    for row in facts.to_dict("records"):
        aid = str(row["announcement_id"])
        d = sources[aid]
        assert compact(row["source_raw_row"]) in compact(d["pages"][row["source_page"] - 1]), (aid, row["metric_id"])
        assert row["source_raw_value"] == row["all_numeric_cells_for_verification"][row["selected_numeric_column_one_based"] - 1]
        calculated = amount(row["source_raw_value"]) * Decimal(str(row["source_unit_multiplier"]))
        assert abs(calculated - Decimal(str(row["metric_value"]))) < Decimal(".00001"), (aid, row["metric_id"])
        assert not row["previous_year_comparative_is_original_prior_vintage"]
        assert row["strategy_input_status"] == ("PASS_EXPLICIT_CNY_CURRENT_VINTAGE" if row["currency_explicit_in_statement"] else "NO_VIEW_CURRENCY_NOT_EXPLICIT")
    identities = 0
    max_error = Decimal(0)
    for file in (out / "document_records").glob("*.json"):
        rec = read(file)
        rows = rec.get("all_rows_for_verification")
        if not rows:
            continue
        v = {k: [amount(c) for c in row["cells"]] for k,row in rows.items()}
        revenue, expense, profit = (v[k] for k in ["OPERATING_REVENUE_YTD", "OPERATING_EXPENSE_YTD", "OPERATING_PROFIT_YTD"])
        code = rec["source"]["ts_code"]
        pairs = []
        if code == "000001.SZ":
            pre = v["PRE_IMPAIRMENT_OPERATING_PROFIT_YTD"]
            pairs += [([a+b for a,b in zip(revenue, expense)], pre)]
            pairs += [([a+b+c for a,b,c in zip(pre, v["CREDIT_IMPAIRMENT_YTD"], v["OTHER_IMPAIRMENT_YTD"])], profit)]
            pairs += [([v["PARENT_NET_PROFIT_YTD"][i] for i in [0,2]], [v["TOTAL_NET_PROFIT_YTD"][i] for i in [0,2]])]
        else:
            sign = -1 if code in {"600999.SH", "600030.SH", "000776.SZ", "601211.SH"} else 1
            pairs += [([a+sign*b for a,b in zip(revenue, expense)], profit)]
            parent = v["PARENT_NET_PROFIT_YTD"]
            pairs += [([a+b for a,b in zip(parent, v["MINORITY_NET_PROFIT_YTD"])], v["TOTAL_NET_PROFIT_YTD"][:len(parent)])]
        tolerance = Decimal("1.01") if rec["facts"][0]["source_unit_multiplier"] == 1000000 else Decimal(".02")
        for left, right in pairs:
            assert len(left) == len(right)
            error = max(abs(a-b) for a,b in zip(left,right))
            assert error <= tolerance, file.name
            max_error = max(error,max_error)
            identities += 1
    old = root / "reports/research/510300_financial_original_facts_v1/document_records"
    unchanged = 0
    for file in old.glob("*.json"):
        prev = read(file)
        if prev["facts"]:
            assert prev["facts"] == read(out / "document_records" / file.name)["facts"]
            unchanged += len(prev["facts"])
    eps = read(out / "ordinary_share_eps_reconciliation.json")["rows"]
    for rec in eps:
        r = rec["original_rows"]
        parent = Decimal(rec["parent_profit_cny"])
        ordinary = parent - amount(r["preferred_dividend"]["cells"][0]) - amount(r["perpetual_interest"]["cells"][0])
        result = ordinary/amount(r["shares"]["cells"][0])
        assert ordinary == Decimal(rec["ordinary_profit_after_other_equity_returns_cny"])
        assert result == Decimal(rec["corrected_ordinary_profit_divided_by_latest_shares"])
        assert abs(result - amount(r["reported_latest_share_eps"]["cells"][0])) <= Decimal(".005")
        assert not rec["latest_share_count_is_proven_ytd_weighted_average_share_count"]
    frozen = 0
    for filename in ["510300_financial_report_subject_repair_v1_manifest.json", "510300_financial_original_facts_v1_manifest.json", "510300_financial_original_facts_v1_1_manifest.json", "510300_financial_ttm_dependencies_v1_manifest.json"]:
        for rec in read(root/"config"/filename)["files"]:
            path = root/rec["path"]
            if not path.exists() and rec["path"].startswith("data/"):
                path = Path(r"E:\ResearchData\New project 8")/rec["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == rec["sha256"], rec["path"]
            frozen += 1
    return {"status": "PASS_READ_ONLY_ORIGINAL_NUMERICAL_AND_SOURCE_RECOMPUTATION", "checked_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "verified_original_pdfs": verified_pdfs, "current_vintage_facts_checked": len(facts), "accounting_identities_checked": identities,
            "maximum_accounting_error_in_reported_units": str(max_error), "unchanged_old_fact_rows": unchanged,
            "ordinary_eps_reconciliations": len(eps), "frozen_file_references_checked": frozen,
            "strategy_returns_read_or_recomputed": False, "new_model_fit": False, "security_audit_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="核对原始金融来源及数值，不重跑账户")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))
