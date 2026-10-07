"""在已保存输入上只读重算财务测量，不联网、不读市场收益、不形成账户。"""
import argparse
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pypdfium2 as pdfium


def verify_semantic_contradictions(out, module):
    extra = out / "anomaly_evidence"
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    review = read(extra / "review_freeze.json")
    assert module.digest(out / "result.json") == review["measurement_result_sha256"]
    assert module.digest(out / "source_invalidation_addendum.json") == review["invalidation_addendum_sha256"]
    for row in review["files"]:
        path = extra / row["path"]
        assert path.stat().st_size == row["bytes"] and module.digest(path) == row["sha256"], row["path"]
    cards = read(extra / "confirmed_field_contradictions.json")
    selected = pd.read_parquet(extra / "flagged_fields_before_source_review.parquet")
    assert len(cards) == 7 and len(set(r["announcement_id"] for r in cards)) == 6
    text = {}
    for aid in sorted(set(r["announcement_id"] for r in cards)):
        pdf = extra / "pdf" / (aid + ".pdf")
        receipt = read(extra / "receipts" / (aid + ".json"))
        assert module.digest(pdf) == receipt["sha256"] == receipt["archived_sha256"]
        document = pdfium.PdfDocument(str(pdf))
        pages = sorted({r["page"] for card in cards if card["announcement_id"] == aid for r in card["proof"]})
        for number in pages:
            page = document[number - 1]
            textpage = page.get_textpage()
            text[(aid, number)] = re.sub(r"\s+", "", textpage.get_text_range())
            textpage.close()
            page.close()
        document.close()
    for card in cards:
        row = selected[(selected.announcement_id == card["announcement_id"]) & (selected.metric_id == card["metric_id"])].iloc[0]
        assert row.verified_value == card["archived_value_cny"] != card["original_page_value_cny"]
        assert row.official_pdf_sha256 == card["pdf_sha256"]
        for anchor in card["proof"]:
            assert anchor["normalized_fragment"] in text[(card["announcement_id"], anchor["page"])]
        amount = card["original_page_value_cny"]
        token = f"{amount:,.0f}" if float(amount).is_integer() else f"{amount:,.2f}"
        assert any(token in anchor["normalized_fragment"] for anchor in card["proof"])
    dep = pd.read_parquet(out / "formula_dependencies.parquet")
    keys = pd.DataFrame([{"source_announcement_id": c["announcement_id"], "metric_id": c["metric_id"], "error_kind": c["error_kind"]} for c in cards])
    affected = dep.merge(keys, on=["source_announcement_id", "metric_id"], how="inner", validate="many_to_one")
    pd.testing.assert_frame_equal(affected, pd.read_parquet(extra / "affected_formula_dependencies.parquet"), check_dtype=False)
    measured = pd.read_parquet(out / "member_report_measurements.parquet")
    flagged = measured.loc[measured.announcement_id.isin(set(affected.target_announcement_id))]
    pd.testing.assert_frame_equal(flagged.reset_index(drop=True), pd.read_parquet(extra / "affected_report_measurements.parquet"), check_dtype=False)
    panel = pd.read_parquet(out / "daily_member_measurements.parquet")
    observed = panel.loc[panel.announcement_id.isin(set(affected.target_announcement_id)) & panel.joint_known,
                         ["date", "ts_code", "announcement_id", "report_period", "L02", "L04_change", "joint_known"]].copy()
    observed["status"] = "CONFIRMED_FIELD_DEPENDENCY_CONTAMINATED_NO_TRADING_ADMISSION"
    pd.testing.assert_frame_equal(observed.reset_index(drop=True), pd.read_parquet(extra / "affected_daily_member_measurements.parquet"), check_dtype=False)
    impact = read(out / "source_invalidation_addendum.json")
    assert impact["affected_formula_dependency_occurrences"] == len(affected) == 36
    assert impact["affected_historical_union_reports"] == len(flagged) == 17
    assert impact["affected_mechanically_joint_member_report_events"] == int((flagged.member_at_available & flagged.joint_known).sum()) == 16
    assert impact["affected_daily_joint_member_rows"] == len(observed) == 979
    assert impact["affected_calendar_sessions"] == observed.date.nunique() == 778
    assert impact["current_dataset_admission"] == "BLOCKED_CONFIRMED_SEMANTIC_EXTRACTION_ERRORS"
    assert impact["T11"] == "NOT_RUN" and not impact["corrected_feature_values_computed"]
    return {"frozen_files": len(review["files"]), "reparsed_pdf_documents": 6, "reparsed_evidence_pages": len(text),
        "confirmed_wrong_fields": 7, "proof_fragments": sum(len(c["proof"]) for c in cards),
        "affected_member_report_events": 17, "affected_daily_member_rows": 979,
        "dataset_semantic_admission": "BLOCKED_CONFIRMED_SEMANTIC_EXTRACTION_ERRORS"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-addendum-only", action="store_true")
    args = parser.parse_args()
    out = args.root.resolve()
    spec = importlib.util.spec_from_file_location("frozen_financial_measurement", out / "code/factor96_earnings_cashflow_measurement_v1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.source_addendum_only:
        receipt = verify_semantic_contradictions(out, module)
        print(json.dumps({"status": "PASS_SEVEN_SOURCE_CONTRADICTIONS_REPARSED", **receipt}, ensure_ascii=False), flush=True)
        return
    count = module.verify_inputs(out)
    frames, result = module.calculate(out)
    saved_result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result == saved_result, "保存结果与冻结输入重算不同"
    verified_rows = {}
    for name, expected in frames.items():
        if name.endswith(".parquet"):
            saved = pd.read_parquet(out / name)
            pd.testing.assert_frame_equal(saved, expected, check_dtype=False, check_exact=True)
        else:
            saved = pd.read_csv(out / name, encoding="utf-8-sig")
            pd.testing.assert_frame_equal(saved, expected, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
        verified_rows[name] = len(expected)
    measured = frames["member_report_measurements.parquet"]
    dep = frames["formula_dependencies.parquet"]
    # 独立代数检查：保存的四季流量、分母和历史同季值须直接恢复保存因子。
    joint = measured.loc[measured.joint_known]
    expected_l02 = (joint.quarter_roa - (joint.prior_same_quarter_roa + joint.prior2_same_quarter_roa) / 2) / (
        np.abs(joint.prior_same_quarter_roa - joint.prior2_same_quarter_roa) / np.sqrt(2))
    expected_l04 = (joint.ttm_cashflow - joint.ttm_profit) / joint.assets - (joint.prior_ttm_cashflow - joint.prior_ttm_profit) / joint.prior_assets
    assert np.allclose(expected_l02, joint.L02, rtol=1e-12, atol=1e-12)
    assert np.allclose(expected_l04, joint.L04_change, rtol=1e-12, atol=1e-12)
    admitted = dep.status.eq("VERIFIED_SAVED_ORIGINAL_FACT")
    assert (dep.loc[admitted, "source_available_date"] <= dep.loc[admitted, "target_available_date"]).all()
    assert dep.loc[~admitted, "value"].isna().all(), "未知或迟报依赖不得残留可用数值"
    assert not measured.loc[measured.is_financial, "joint_known"].any()
    assert (joint.seasonal_sd > 0).all() and (joint.assets > 0).all() and (joint.prior_assets > 0).all()
    assert saved_result["T11"] == "NOT_RUN" and saved_result["O02"] == "NOT_COMPUTED"
    contradictions = verify_semantic_contradictions(out, module)
    receipt = {"status": "PASS_SAVED_FINANCIAL_MEASUREMENT_AND_SOURCE_REJECTION", "frozen_files": count,
        "recomputed_tables": verified_rows, "independent_algebra_checked_rows": len(joint),
        "verified_dependency_rows": int(admitted.sum()), "future_dependency_values_used": 0,
        "new_accounts": 0, "network_requests": 0, "strategy_returns_read": False,
        "external_review": "NOT_PERFORMED", "goal_achieved": False, "semantic_contradictions": contradictions,
        "interpretation": "机械重算及错误传播核查通过；数据语义准入明确失败，不代表T11有效。"}
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
