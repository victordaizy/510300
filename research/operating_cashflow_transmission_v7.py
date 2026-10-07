"""在固定同组公司和同期间下分解现金流，保留未识别的原因。"""
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.operating_cashflow_sources_v7 import OUT, sha, save, now

RAW_FIELDS = {
    "net_operating": "NETCASH_OPERATE",
    "operating_inflow": "TOTAL_OPERATE_INFLOW",
    "operating_outflow": "TOTAL_OPERATE_OUTFLOW",
    "sales_receipts": "SALES_SERVICES",
    "purchases_paid": "BUY_SERVICES",
    "employees_paid": "PAY_STAFF_CASH",
    "taxes_paid": "PAY_ALL_TAX",
    "other_operating_paid": "PAY_OTHER_OPERATE",
    "group_net_profit": "NETPROFIT",
    "operating_receivables_reduction": "OPERATE_RECE_REDUCE",
    "operating_payables_increase": "OPERATE_PAYABLE_ADD",
    "net_operating_indirect_note": "NETCASH_OPERATENOTE",
}
LABELS = {
    "net_operating": "经营现金流净额", "operating_inflow": "经营现金流入合计", "operating_outflow": "经营现金流出合计",
    "sales_receipts": "销售商品和提供劳务收到的现金", "other_operating_receipts": "其余经营收款（含税费返还）",
    "purchases_paid": "购买商品和接受劳务支付的现金", "employees_paid": "支付职工现金", "taxes_paid": "支付税费", "other_operating_paid": "其他经营付款",
    "outflow_unallocated": "流出合计与列示四项的差额", "net_table_adjustment": "净额与流入减流出的差额",
    "group_net_profit": "净利润（包含少数股东口径）", "operating_receivables_reduction": "经营性应收项目减少（负数为增加）",
    "operating_payables_increase": "经营性应付项目增加（负数为减少）", "inventory_noncash_other_adjustments": "存货、非现金及其余调整合计", "net_operating_indirect_note": "补充资料经营现金流净额",
}
DIRECT = {"sales_receipts": 1, "other_operating_receipts": 1, "purchases_paid": -1, "employees_paid": -1, "taxes_paid": -1, "other_operating_paid": -1, "outflow_unallocated": -1, "net_table_adjustment": 1}
INDIRECT = {"group_net_profit": 1, "operating_receivables_reduction": 1, "operating_payables_increase": 1, "inventory_noncash_other_adjustments": 1}
TERMS = {
    "ttm_current": {"2025-06-30": 1, "2024-12-31": 1, "2024-06-30": -1},
    "ttm_prior": {"2024-06-30": 1, "2023-12-31": 1, "2023-06-30": -1},
    "latest_half_current": {"2025-06-30": 1}, "latest_half_prior": {"2024-06-30": 1},
    "earlier_half_current": {"2024-12-31": 1, "2024-06-30": -1},
    "earlier_half_prior": {"2023-12-31": 1, "2023-06-30": -1},
}


def csv(frame, name):
    frame.to_csv(OUT / "results" / name, index=False, encoding="utf-8-sig", float_format="%.15g")


def finite(x):
    if isinstance(x, dict):
        return {str(k): finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [finite(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def run():
    assert not (OUT / "results/build_receipt.json").exists(), "本轮已有结果，不覆盖已有研究。"
    raw = pd.read_parquet(OUT / "results/固定247家公司_五期现金流完整原字段.parquet")
    cohort = pd.read_parquet(OUT / "inputs/fixed_247_company_cohort.parquet")
    old = pd.read_parquet(OUT / "inputs/parent_cashflow.parquet")
    assert len(raw) == 1235 and raw.CURRENCY.eq("CNY").all()
    assert raw[list(RAW_FIELDS.values())].notna().all().all()
    assert set(raw.SECUCODE) == set(cohort.stock_code)
    compare = raw[["SECUCODE", "report_end", "NETCASH_OPERATE", "UPDATE_DATE", "source_sha256"]].merge(old[["stock_code", "report_end", "operating_cashflow", "update_date", "source_sha256"]], left_on=["SECUCODE", "report_end"], right_on=["stock_code", "report_end"], validate="one_to_one", suffixes=("_new", "_parent"))
    compare["cashflow_value_difference_cny"] = compare.NETCASH_OPERATE - compare.operating_cashflow
    csv(compare, "1235条新旧来源净经营现金流核对.csv")
    assert len(compare) == 1235
    # 本轮各期净额与父研究完全一致，因此可以直接解释父研究的现金流增量。
    assert compare.cashflow_value_difference_cny.abs().max() == 0
    rows = raw[["SECUCODE", "SECURITY_NAME_ABBR", "report_end", "NOTICE_DATE", "UPDATE_DATE", "source_file", "source_sha256", "retrieved_at"]].rename(columns={"SECUCODE": "stock_code", "SECURITY_NAME_ABBR": "company_name"}).copy()
    for name, original in RAW_FIELDS.items():
        rows[name] = pd.to_numeric(raw[original])
    rows["other_operating_receipts"] = rows.operating_inflow - rows.sales_receipts
    rows["outflow_unallocated"] = rows.operating_outflow - rows[["purchases_paid", "employees_paid", "taxes_paid", "other_operating_paid"]].sum(axis=1)
    rows["net_table_adjustment"] = rows.net_operating - (rows.operating_inflow - rows.operating_outflow)
    rows["inventory_noncash_other_adjustments"] = rows.net_operating - rows[["group_net_profit", "operating_receivables_reduction", "operating_payables_increase"]].sum(axis=1)
    rows["missing_inventory_detail"] = raw.INVENTORY_REDUCE.isna()
    rows["missing_tax_refund_detail"] = raw.RECEIVE_TAX_REFUND.isna()
    rows["missing_other_receipts_detail"] = raw.RECEIVE_OTHER_OPERATE.isna()
    rows["source_vintage"] = "CURRENT_SECONDARY_SOURCE_RECONSTRUCTION_NOT_STRICT_PIT"
    csv(rows, "247家公司五期_经营收付与营运资金主分解.csv")
    fields = list(LABELS)
    matrices = {field: rows.pivot(index="stock_code", columns="report_end", values=field) for field in fields}
    result = cohort.set_index("stock_code").copy()
    result["company_name"] = rows.drop_duplicates("stock_code").set_index("stock_code").company_name
    additions = {}
    for field, matrix in matrices.items():
        for period, terms in TERMS.items():
            additions[field + "_" + period] = sum(sign * matrix[pd.Timestamp(date)] for date, sign in terms.items())
        for stage in ["ttm", "latest_half", "earlier_half"]:
            additions[field + "_" + stage + "_change"] = additions[field + "_" + stage + "_current"] - additions[field + "_" + stage + "_prior"]
    result = pd.concat([result, pd.DataFrame(additions)], axis=1)
    np.testing.assert_allclose(result.net_operating_ttm_current, result.operating_cashflow_ttm, atol=1e-3, rtol=1e-13)
    np.testing.assert_allclose(result.net_operating_ttm_prior, result.operating_cashflow_ttm_prior_year, atol=1e-3, rtol=1e-13)
    for field in fields:
        np.testing.assert_allclose(result[field + "_ttm_change"], result[field + "_latest_half_change"] + result[field + "_earlier_half_change"], atol=1e-3, rtol=1e-13)
    result = result.reset_index()
    result.to_parquet(OUT / "results/固定247家公司_现金流期间与原因金额.parquet", index=False)
    ranking = result[["stock_code", "company_name", "industry_name", "diagnostic_weight", "net_operating_ttm_current", "net_operating_ttm_prior", "net_operating_ttm_change", "net_operating_latest_half_current", "net_operating_latest_half_prior", "net_operating_latest_half_change", "net_operating_earlier_half_change", "sales_receipts_latest_half_change", "purchases_paid_latest_half_change", "operating_receivables_reduction_latest_half_change", "operating_payables_increase_latest_half_change"]].sort_values("net_operating_ttm_change", ascending=False)
    csv(ranking, "247家公司_经营现金流增量完整排名.csv")
    aggregate_rows, contribution_rows = [], []
    for field in fields:
        x = {"field": field, "label": LABELS[field], "companies": len(result)}
        for period in TERMS:
            x[period + "_cny"] = float(result[field + "_" + period].sum())
        for stage in ["ttm", "latest_half", "earlier_half"]:
            current, previous = x[stage + "_current_cny"], x[stage + "_prior_cny"]
            x[stage + "_change_cny"] = current - previous
            x[stage + "_yoy"] = current / previous - 1 if previous > 0 else np.nan
        aggregate_rows.append(x)
    aggregate = pd.DataFrame(aggregate_rows).set_index("field")
    for method, components in [("直接法现金收付", DIRECT), ("间接法主桥", INDIRECT)]:
        for field, sign in components.items():
            a = aggregate.loc[field]
            contribution_rows.append({"method": method, "field": field, "label": LABELS[field], "cashflow_sign": sign, **{stage + "_contribution_cny": float(sign * a[stage + "_change_cny"]) for stage in ["ttm", "latest_half", "earlier_half"]}})
        for stage in ["ttm", "latest_half", "earlier_half"]:
            total = sum(sign * aggregate.loc[field, stage + "_change_cny"] for field, sign in components.items())
            np.testing.assert_allclose(total, aggregate.loc["net_operating", stage + "_change_cny"], atol=.1, rtol=1e-12)
    csv(aggregate.reset_index(), "247家公司汇总_现金流期间及收付原金额.csv")
    csv(pd.DataFrame(contribution_rows), "现金流增量_直接法与间接法完整金额桥.csv")
    sector = result.groupby("industry_name").agg(companies=("stock_code", "size"), net_operating_ttm_change_cny=("net_operating_ttm_change", "sum"), net_operating_latest_half_change_cny=("net_operating_latest_half_change", "sum"), net_operating_earlier_half_change_cny=("net_operating_earlier_half_change", "sum"), sales_receipts_latest_half_change_cny=("sales_receipts_latest_half_change", "sum"), purchases_paid_latest_half_change_cny=("purchases_paid_latest_half_change", "sum"), operating_receivables_latest_half_change_cny=("operating_receivables_reduction_latest_half_change", "sum"), operating_payables_latest_half_change_cny=("operating_payables_increase_latest_half_change", "sum")).reset_index().sort_values("net_operating_ttm_change_cny", ascending=False)
    csv(sector, "全部非金融行业_现金流增量与期间.csv")
    positives = ranking[ranking.net_operating_ttm_change > 0]
    negatives = ranking[ranking.net_operating_ttm_change < 0].sort_values("net_operating_ttm_change")
    selected = pd.concat([positives.head(2), negatives.head(1)], ignore_index=True)
    csv(selected, "按预定规则选择的三家原文核对对象.csv")
    a = aggregate.loc["net_operating"]
    concentration = {"at": now(), "company_count": len(result), "companies_with_ttm_cashflow_increase": len(positives), "companies_with_ttm_cashflow_decrease": len(negatives), "companies_with_latest_half_cashflow_increase": int((result.net_operating_latest_half_change > 0).sum()), "positive_ttm_increment_cny": float(positives.net_operating_ttm_change.sum()), "negative_ttm_increment_cny": float(negatives.net_operating_ttm_change.sum()), "net_ttm_increment_cny": float(a.ttm_change_cny), "top_three_positive_ttm_increment_cny": float(positives.head(3).net_operating_ttm_change.sum()), "top_three_share_of_positive_pool": float(positives.head(3).net_operating_ttm_change.sum() / positives.net_operating_ttm_change.sum()), "top_three_share_of_net_increment": float(positives.head(3).net_operating_ttm_change.sum() / a.ttm_change_cny), "latest_half_share_of_ttm_increment": float(a.latest_half_change_cny / a.ttm_change_cny), "earlier_half_share_of_ttm_increment": float(a.earlier_half_change_cny / a.ttm_change_cny), "selection_rule": "固定同组公司TTM现金流增量最大的两家与减少最多的一家，非价格收益筛选。"}
    save(OUT / "results/concentration_and_period.json", finite(concentration))
    save(OUT / "results/build_receipt.json", finite({"at": now(), "study_id": "510300_OPERATING_CASHFLOW_TRANSMISSION_V7", "companies": len(result), "source_company_periods": len(rows), "source_net_values_changed_from_v5": int((compare.cashflow_value_difference_cny != 0).sum()), "ttm_net_cashflow_yoy": a.ttm_yoy, "latest_half_net_cashflow_yoy": a.latest_half_yoy, "earlier_half_net_cashflow_yoy": a.earlier_half_yoy, "missing_inventory_details": int(raw.INVENTORY_REDUCE.isna().sum()), "missing_tax_refund_details": int(raw.RECEIVE_TAX_REFUND.isna().sum()), "no_missing_fields_filled_as_known_zero": True, "largest_main_net_table_rounding_residual_cny": float(rows.net_table_adjustment.abs().max()), "maximum_main_vs_indirect_note_difference_cny": float((rows.net_operating-rows.net_operating_indirect_note).abs().max()), "index_weight_attribution": False, "historical_first_version_authenticated": False, "new_return_models": 0, "new_accounts": 0, "goal_achieved": False}))
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print("固定247家公司现金流分解已完成。金额均为亿元：")
    print((aggregate[["label", "ttm_change_cny", "latest_half_change_cny", "earlier_half_change_cny"]].assign(ttm_change_cny=lambda x:x.ttm_change_cny/1e8,latest_half_change_cny=lambda x:x.latest_half_change_cny/1e8,earlier_half_change_cny=lambda x:x.earlier_half_change_cny/1e8)).to_string())
    print("需核对原文的固定对象：")
    print(selected[["stock_code", "company_name", "net_operating_ttm_change", "net_operating_latest_half_change"]].to_string(index=False))
    print(json.dumps(finite(concentration), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
