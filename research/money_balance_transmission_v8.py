"""从原始央行表格复算货币余额、结构和对应资产，不拟合未来收益。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.money_balance_sources_v8 import OUT, now, save, digest

DCS = {
    "nfa": "国外净资产", "domestic_credit": "国内信贷", "government_net": "对政府债权（净）",
    "nonfinancial_claims": "对非金融部门债权", "other_financial_claims": "对其他金融部门债权",
    "m2": "货币和准货币", "m1_table": "货币", "m0": "流通中货币", "quasi_table": "准货币",
    "other_deposits_table": "其他存款", "excluded_deposits": "不纳入广义货币的存款",
    "bonds": "债券", "capital": "实收资本", "other_net": "其他（净）",
}
CB = {"cb_gov_claims": "对政府债权", "cb_gov_deposits": "政府存款", "cb_payment_reserves": "非金融机构存款",
      "cb_foreign_assets": "国外资产", "cb_foreign_liabilities": "国外负债"}
ODCS = {"odcs_gov_claims": "对政府债权", "corporate_demand": "单位活期存款", "corporate_time": "单位定期存款",
        "personal_total": "个人存款", "odcs_nonfinancial_claims": "对非金融机构债权",
        "odcs_other_resident_claims": "对其他居民部门债权", "odcs_financial_deposits_m2": "其中：计入广义货币的存款",
        "odcs_foreign_assets": "国外资产", "odcs_foreign_liabilities": "国外负债"}
COUNTER = {"nfa": 1, "government_net": 1, "nonfinancial_claims": 1, "other_financial_claims": 1,
           "excluded_deposits": -1, "bonds": -1, "capital": -1, "other_net": -1, "counterpart_rounding": 1}
M1_PARTS = ["m0", "corporate_demand", "personal_demand_derived", "payment_reserves", "m1_rounding"]
M2_HOLDERS = ["m0", "corporate_demand", "corporate_time", "personal_total", "payment_reserves",
              "odcs_financial_deposits_m2", "other_deposits_residual", "holder_rounding"]


def extract(text, label, count):
    pattern = r"(?<![\u4e00-\u9fff])" + re.escape(label) + r"[^\d\u4e00-\u9fff]+((?:\d+\.\d+\s*)+)"
    matches = list(re.finditer(pattern, text))
    assert len(matches) == 1, (label, len(matches))
    values = [float(v) for v in re.findall(r"\d+\.\d+", matches[0].group(1))]
    assert len(values) == count, (label, len(values), count)
    return values


def read_table(name, year, mapping):
    text = (OUT / "sources" / name).read_text(encoding="utf-8")
    count = 12 if year == 2024 else 11
    main_table = text.split("注：", 1)[0]
    rows = {key: extract(main_table, label, count) for key, label in mapping.items()}
    rows["stat_month"] = [f"{year}-{m:02d}" for m in range(1, count + 1)]
    return pd.DataFrame(rows).set_index("stat_month"), text


def render_sources():
    binary = Path(r"E:\CodexData\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe")
    for source in sorted((OUT / "sources").glob("*.pdf")):
        target = source.with_suffix("")
        subprocess.run([str(binary), "-f", "1", "-l", "1", "-singlefile", "-r", "200", "-png", str(source), str(target)],
                       check=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)


def build_balances():
    frames = []
    for year in [2024, 2025]:
        mapping = dict(DCS)
        if year == 2024:
            mapping.update({"corporate_demand": "单位活期存款", "corporate_time": "单位定期存款", "personal_total": "个人存款"})
        else:
            mapping.update({"demand_total": "活期存款", "time_total": "定期存款", "payment_reserves": "非银行支付机构客户备付金"})
        dcs, dcs_text = read_table(f"存款性公司概览_{year}.txt", year, mapping)
        cb, _ = read_table(f"货币当局资产负债表_{year}.txt", year, CB)
        data = dcs.join(cb)
        if year == 2024:
            new_text = (OUT / "sources/存款性公司概览_2025.txt").read_text(encoding="utf-8")
            match = re.search(r"余额（亿元）\s+((?:\d+\s*){12})", new_text)
            assert match
            values = [float(v) for v in re.findall(r"\d+", match.group(1))]
            assert len(values) == 12
            data["m1_new_comparable"] = values
            data["payment_reserves"] = data.cb_payment_reserves
            data["demand_total"] = data.m1_new_comparable - data.m0 - data.payment_reserves
            data["m1_definition_in_table"] = "OLD_M1"
            data["m1_new_source_status"] = "2025年表脚注的2024回溯值，精度1亿元；不能回填2024信息集"
            data["odcs_gov_claims_derived"] = data.government_net - data.cb_gov_claims + data.cb_gov_deposits
        else:
            odcs, _ = read_table("其他存款性公司资产负债表_2025.txt", year, ODCS)
            data = data.join(odcs)
            data["m1_new_comparable"] = data.m1_table
            data["m1_definition_in_table"] = "NEW_M1"
            data["m1_new_source_status"] = "2025年表主栏原值，精度0.01亿元；仍为事后取得版本"
            data["odcs_gov_claims_derived"] = data.odcs_gov_claims
        data["personal_demand_derived"] = data.demand_total - data.corporate_demand
        data["personal_time_derived"] = data.personal_total - data.personal_demand_derived
        if year == 2024:
            data["time_total"] = data.corporate_time + data.personal_time_derived
        # 2024其他存款包括当年在M1之外的备付金，转为新M1展示时需同步移出准货币，M2总量不变。
        data["other_deposits_new_basis"] = data.other_deposits_table - (data.payment_reserves if year == 2024 else 0)
        data["quasi_new_basis"] = data.m2 - data.m1_new_comparable
        data["counterpart_rounding"] = data.m2 - (data.nfa + data.government_net + data.nonfinancial_claims + data.other_financial_claims
                                                          - data.excluded_deposits - data.bonds - data.capital - data.other_net)
        data["government_identity_residual"] = data.government_net - (data.cb_gov_claims + data.odcs_gov_claims_derived - data.cb_gov_deposits)
        data["m1_rounding"] = data.m1_new_comparable - (data.m0 + data.corporate_demand + data.personal_demand_derived + data.payment_reserves)
        data["quasi_identity_residual"] = data.quasi_new_basis - (data.time_total + data.other_deposits_new_basis)
        if year == 2025:
            data["other_deposits_residual"] = data.other_deposits_new_basis - data.odcs_financial_deposits_m2
            data["holder_rounding"] = data.m2 - data[[v for v in M2_HOLDERS if v != "holder_rounding"]].sum(axis=1)
            data["nfa_reconstruction_residual"] = data.nfa - (data.cb_foreign_assets + data.odcs_foreign_assets - data.cb_foreign_liabilities - data.odcs_foreign_liabilities)
            data["nonfinancial_reconstruction_residual"] = data.nonfinancial_claims - (data.odcs_nonfinancial_claims + data.odcs_other_resident_claims)
        frames.append(data)
    all_data = pd.concat(frames).sort_index()
    data = all_data.loc["2024-01":"2025-08"].copy()
    data["first_historical_vintage_verified"] = False
    data["admitted_to_historical_return_model"] = False
    assert len(data) == 20
    for field in ["counterpart_rounding", "government_identity_residual", "m1_rounding", "quasi_identity_residual",
                  "holder_rounding", "nfa_reconstruction_residual", "nonfinancial_reconstruction_residual"]:
        assert data[field].abs().max() < 0.051, (field, data[field].abs().max())
    data.to_csv(OUT / "results/20个月_货币与银行资产负债原金额.csv", encoding="utf-8-sig")
    data.to_parquet(OUT / "results/20个月_货币与银行资产负债原金额.parquet")
    return data


def interval_tables(data):
    rows = []
    for end in data.index:
        end_period = pd.Period(end, freq="M")
        spans = [("单月", str(end_period - 1))]
        if end.startswith("2025"):
            spans += [("三个月", str(end_period - 3)), ("年内", "2024-12")]
        for kind, start in spans:
            if start not in data.index:
                continue
            before, after = data.loc[start], data.loc[end]
            numerical = data.select_dtypes(include=[np.number]).columns
            delta = after[numerical] - before[numerical]
            record = {"interval": kind, "start_month_end": start, "end_month_end": end, "m2_delta": float(delta.m2),
                      "m1_new_delta": float(delta.m1_new_comparable)}
            for field in numerical:
                record["delta_" + field] = float(delta[field])
            for field, sign in COUNTER.items():
                record["m2_contribution_" + field] = float(sign * delta[field])
            if end.startswith("2025"):
                record["m1_current_growth_pct"] = 100 * (after.m1_new_comparable / before.m1_new_comparable - 1)
                record["m2_current_growth_pct"] = 100 * (after.m2 / before.m2 - 1)
                record["m1_m2_current_relative_log_pp"] = 100 * math.log((after.m1_new_comparable / before.m1_new_comparable) / (after.m2 / before.m2))
            previous_start, previous_end = str(pd.Period(start, freq="M") - 12), str(end_period - 12)
            if previous_start in data.index and previous_end in data.index:
                pa, pb = data.loc[previous_end], data.loc[previous_start]
                for field in COUNTER:
                    record["previous_year_delta_" + field] = float(pa[field] - pb[field])
                    record["yoy_change_of_delta_" + field] = float(delta[field] - (pa[field] - pb[field]))
                record["previous_year_m2_delta"] = float(pa.m2 - pb.m2)
                record["previous_year_m1_new_delta"] = float(pa.m1_new_comparable - pb.m1_new_comparable)
                record["m1_m2_base_relative_log_pp"] = -100 * math.log((pa.m1_new_comparable / pb.m1_new_comparable) / (pa.m2 / pb.m2))
                record["change_log_yoy_relative_pp"] = record["m1_m2_current_relative_log_pp"] + record["m1_m2_base_relative_log_pp"]
            rows.append(record)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "results/完整窗口_单月三个月年内余额增量及对应项.csv", index=False, encoding="utf-8-sig")
    bridge_rows = []
    for item in frame.to_dict("records"):
        for field, sign in COUNTER.items():
            bridge_rows.append({"interval": item["interval"], "start_month_end": item["start_month_end"], "end_month_end": item["end_month_end"],
                                "component": field, "signed_contribution_yi": item["m2_contribution_" + field], "m2_delta_yi": item["m2_delta"]})
    pd.DataFrame(bridge_rows).to_csv(OUT / "results/M2变化_全部资产负债对应项.csv", index=False, encoding="utf-8-sig")
    return frame


def component_growth(data, intervals):
    rows = []
    for item in intervals.loc[(intervals.interval == "三个月") & intervals.end_month_end.ge("2025-04")].to_dict("records"):
        start, end = item["start_month_end"], item["end_month_end"]
        prior_start, prior_end = str(pd.Period(start, freq="M") - 12), str(pd.Period(end, freq="M") - 12)
        for side, a, b, sign in [("本期三个月", start, end, 1), ("去年同期基数", prior_start, prior_end, -1)]:
            first, last = data.loc[a], data.loc[b]
            for total, components in [("M1", {k: 1 for k in M1_PARTS}), ("M2", COUNTER)]:
                total_field = "m1_new_comparable" if total == "M1" else "m2"
                lo, hi = first[total_field], last[total_field]
                factor = math.log(hi / lo) / (hi - lo) if hi != lo else 1 / lo
                for field, counter_sign in components.items():
                    change = (last[field] - first[field]) * counter_sign
                    contribution = change * factor * 100 * sign * (1 if total == "M1" else -1)
                    rows.append({"end_month": end, "side": side, "money_side": total, "component": field,
                                 "balance_delta_yi": float(change), "relative_log_contribution_pp": float(contribution)})
    pd.DataFrame(rows).to_csv(OUT / "results/三个月剪刀差_当期与基数的分项对数贡献.csv", index=False, encoding="utf-8-sig")


def join_context(data, intervals):
    original = pd.read_csv(OUT / "inputs/monthly_context.csv")
    subset = original[original.stat_month.isin(data.index)].copy()
    joined = subset.merge(data.reset_index().add_prefix("retrospective_"), left_on="stat_month", right_on="retrospective_stat_month", how="left", validate="one_to_one")
    for col in subset:
        pd.testing.assert_series_equal(subset[col].reset_index(drop=True), joined[col], check_names=False)
    joined.to_csv(OUT / "results/20个月_原公布信息市场路径与事后结构并列.csv", index=False, encoding="utf-8-sig")
    selected = original[original.stat_month.isin(["2024-08", "2025-06", "2025-07", "2025-08"])].copy()
    fields = ["stat_month", "published_at", "spread_pp", "delta3_spread_pp", "past_return20", "past_return60",
              "v_rv20", "v_upside20", "v_downside20", "downside_window_state", "orders_first_release_value",
              "internal_breadth20", "valuation_original_pe_official", "E0_20_entry_date", "E0_20_exit_date", "E0_20_return", "E1_20_return"]
    selected[fields].to_csv(OUT / "results/四个原病例_信息位置波动与原后续路径.csv", index=False, encoding="utf-8-sig")
    original_decomp = pd.read_csv(OUT / "inputs/money_decomposition.csv")
    compare_rows = []
    for month in data.loc["2025-01":"2025-08"].index:
        prior = str(pd.Period(month, freq="M") - 12)
        current, previous = data.loc[month], data.loc[prior]
        published = original.loc[original.stat_month == month].iloc[0]
        compare_rows.append({"stat_month": month, "table_m1_yoy_pct": 100 * (current.m1_new_comparable / previous.m1_new_comparable - 1),
                             "published_m1_yoy_pct": published.m1_yoy_pp,
                             "table_m2_yoy_pct": 100 * (current.m2 / previous.m2 - 1), "published_m2_yoy_pct": published.m2_yoy_pp,
                             "versions_not_overwritten": True})
    pd.DataFrame(compare_rows).to_csv(OUT / "results/表格可比同比与原公布增速_保留版本差异.csv", index=False, encoding="utf-8-sig")
    return {"original_rows": len(original), "joined_rows": len(joined), "original_columns_unchanged": len(original.columns)}


def run():
    if (OUT / "results/build_receipt.json").exists():
        raise RuntimeError("本轮已有完成结果，不覆盖。")
    data = build_balances()
    intervals = interval_tables(data)
    component_growth(data, intervals)
    scope = join_context(data, intervals)
    render_sources()
    selected = intervals[(intervals.end_month_end.isin(["2025-06", "2025-07", "2025-08"])) & intervals.interval.isin(["单月", "三个月"])]
    summary = {"at": now(), "status": "原表余额与对应项复算完成，等待图表视觉核对", "balance_months": len(data),
               "intervals": len(intervals), **scope, "new_models": 0, "new_accounts": 0, "goal_achieved": False}
    save(OUT / "results/build_receipt.json", summary)
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print(json.dumps(summary, ensure_ascii=False))
    fields = ["interval", "end_month_end", "m2_delta", "m1_new_delta", "delta_government_net", "delta_nonfinancial_claims",
              "delta_cb_gov_deposits", "delta_odcs_gov_claims_derived", "delta_corporate_demand", "delta_personal_demand_derived",
              "delta_odcs_financial_deposits_m2", "delta_other_net", "m1_m2_current_relative_log_pp", "m1_m2_base_relative_log_pp"]
    print(selected[fields].to_json(orient="records", force_ascii=False))


if __name__ == "__main__":
    run()
