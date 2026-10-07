"""重建披露时钟下的行业盈利，并与已经发生的价格和估值变化连接。"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_macro_earnings_pricing_bridge_v5"
PARENT = ROOT / "reports/research/510300_macro_transmission_context_v4"
STATE = ROOT / "data/curated/510300_structural_equity_risk_premium_engine_v1"
FILES = {
    "income.parquet": STATE / "income_point_in_time_vintages.parquet",
    "cashflow.parquet": STATE / "cashflow_point_in_time_vintages.parquet",
    "monthly_state.parquet": STATE / "csi300_component_monthly_state.parquet",
    "membership.parquet": ROOT / "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/000300_daily_pit_membership_20150101_20260814.parquet",
    "weights.parquet": ROOT / "data/raw/constituents/000300_historical_weights.parquet",
    "index_price.parquet": ROOT / "data/raw/market/000300_price_index_return_dispersion_v1.parquet",
    "pe.parquet": PARENT / "inputs/valuation.parquet",
    "market.csv": PARENT / "inputs/market.csv",
    "monthly.csv": PARENT / "results/104个月_多层证据与原后续路径.csv",
    "weekly.csv": PARENT / "results/445周_多层证据与原后续路径.csv",
}


def now():
    return datetime.now().astimezone().isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, np.bool_):
        return bool(v)
    if v is pd.NaT or v is pd.NA:
        return None
    if isinstance(v, (datetime, pd.Timestamp)):
        return v.isoformat()
    return v


def save(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv(frame, name):
    frame.to_csv(OUT / "results" / name, index=False, encoding="utf-8-sig", float_format="%.15g")


def read(name):
    p = OUT / "inputs" / name
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def prepare():
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    receipts = []
    for name, source in FILES.items():
        dest = OUT / "inputs" / name
        if dest.exists():
            assert digest(dest) == digest(source), name
        else:
            shutil.copy2(source, dest)
        receipts.append({"input": name, "source": str(source.relative_to(ROOT)), "sha256": digest(dest), "bytes": dest.stat().st_size})
    save(OUT / "input_receipts.json", {"at": now(), "items": receipts})
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)


def ttm(mapping, period, field):
    if period.month == 12:
        terms = [(period, 1)]
    else:
        terms = [(period, 1), (pd.Timestamp(period.year - 1, 12, 31), 1), (period - pd.DateOffset(years=1), -1)]
    total, dependencies = 0.0, []
    for date, sign in terms:
        if date not in mapping or pd.isna(mapping[date].get(field)):
            return np.nan, []
        value = mapping[date]
        total += sign * float(value[field])
        dependencies.append({"report_end": date, "notice_date": value["available_at"], "update_date": value["update_date"], "source_sha256": value["source_sha256"], "coefficient": sign})
    return total, dependencies


def events(raw, fields, calendar, prefix):
    x = raw.copy()
    for c in ["available_at", "report_end", "update_date"]:
        x[c] = pd.to_datetime(x[c]).dt.normalize()
    assert not x.duplicated(["stock_code", "report_end", "available_at"]).any()
    rows, lineages = [], []
    for code, company in x.groupby("stock_code", sort=True):
        mapping = {}
        for announcement, batch in company.sort_values(["available_at", "report_end"]).groupby("available_at", sort=True):
            if announcement > calendar[-1]:
                continue
            for record in batch.to_dict("records"):
                mapping[record["report_end"]] = record
            latest = max(mapping)
            next_index = int(calendar.searchsorted(announcement, side="right"))
            if next_index >= len(calendar):
                continue
            known = calendar[next_index]
            row = {"stock_code": code, "known_day": known, prefix + "_report_end": latest, prefix + "_nominal_announcement": announcement}
            dependencies = []
            for field in fields:
                current, dep = ttm(mapping, latest, field)
                previous, old_dep = ttm(mapping, latest - pd.DateOffset(years=1), field)
                row[field + "_ttm"] = current
                row[field + "_ttm_prior_year"] = previous
                dependencies += dep + old_dep
            dates = [pd.Timestamp(d["update_date"]) for d in dependencies if pd.notna(d["update_date"])]
            row[prefix + "_dependency_update_max"] = max(dates) if dates else pd.NaT
            row[prefix + "_dependency_notice_max"] = max((pd.Timestamp(d["notice_date"]) for d in dependencies), default=pd.NaT)
            row[prefix + "_dependency_count"] = len({(d["report_end"], d["source_sha256"]) for d in dependencies})
            row[prefix + "_event_id"] = len(rows)
            row[prefix + "_known_day"] = known
            rows.append(row)
            unique = {(str(d["report_end"]), d["source_sha256"], d["coefficient"]): d for d in dependencies}
            lineages.append({"event_id": row[prefix + "_event_id"], "stock_code": code, "known_day": known, "latest_report_end": latest, "dependencies": list(unique.values()), "source_vintage": "CURRENT_AGGREGATOR_VALUE_NOT_FIRST_VERSION"})
    result = pd.DataFrame(rows).sort_values(["known_day", "stock_code", prefix + "_nominal_announcement"]).drop_duplicates(["stock_code", "known_day"], keep="last")
    result.to_parquet(OUT / "results" / (prefix + "_events.parquet"), index=False)
    save(OUT / "results" / (prefix + "_dependencies.json"), lineages)
    return result


def financial_components(origins, income, cash):
    membership = read("membership.parquet")
    membership["membership_date"] = pd.to_datetime(membership.membership_date)
    states = read("monthly_state.parquet")
    states["origin"] = pd.to_datetime(states.origin)
    weights = read("weights.parquet")
    weights["trade_date"] = pd.to_datetime(weights.trade_date)
    members = {date: part.symbol.tolist() for date, part in membership.groupby("membership_date")}
    state_by_date = {date: part.set_index("stock_code") for date, part in states.groupby("origin")}
    weight_by_date = {date: part.set_index("con_code") for date, part in weights.groupby("trade_date")}
    state_dates = pd.DatetimeIndex(sorted(state_by_date))
    weight_dates = pd.DatetimeIndex(sorted(weight_by_date))
    frames, missing = [], []
    for date in sorted(pd.to_datetime(origins.observation_date.dropna()).unique()):
        date = pd.Timestamp(date)
        if date not in members:
            missing.append({"observation_date": date, "status": "NO_VIEW_DAILY_MEMBERSHIP_OUTSIDE_SOURCE_COVERAGE"})
            continue
        frame = pd.DataFrame({"stock_code": members[date], "observation_date": date})
        assert len(frame) == frame.stock_code.nunique() == 300
        spos, wpos = state_dates.searchsorted(date, side="right") - 1, weight_dates.searchsorted(date, side="left") - 1
        if spos >= 0 and (date - state_dates[spos]).days <= 35:
            selected = state_by_date[state_dates[spos]]
            frame["industry_name"] = frame.stock_code.map(selected.industry_name)
            frame["industry_snapshot_date"] = state_dates[spos]
        else:
            frame["industry_name"], frame["industry_snapshot_date"] = pd.NA, pd.NaT
        frame["industry_name"] = frame.industry_name.fillna("行业缺失")
        if wpos >= 0 and (date - weight_dates[wpos]).days <= 62:
            frame["diagnostic_weight"] = frame.stock_code.map(weight_by_date[weight_dates[wpos]].weight) / 100
            frame["weight_snapshot_date"] = weight_dates[wpos]
        else:
            frame["diagnostic_weight"], frame["weight_snapshot_date"] = np.nan, pd.NaT
        frame["weight_status"] = "UNVERSIONED_REFERENCE_ONLY_NOT_PRECISE_INDEX_ATTRIBUTION"
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True).sort_values(["observation_date", "stock_code"])
    all_rows = pd.merge_asof(all_rows, income.sort_values("known_day"), left_on="observation_date", right_on="known_day", by="stock_code", direction="backward").drop(columns="known_day")
    all_rows = pd.merge_asof(all_rows.sort_values("observation_date"), cash.sort_values("known_day"), left_on="observation_date", right_on="known_day", by="stock_code", direction="backward").drop(columns="known_day")
    all_rows["income_age_days"] = (all_rows.observation_date - all_rows.income_report_end).dt.days
    all_rows["cash_age_days"] = (all_rows.observation_date - all_rows.cash_report_end).dt.days
    all_rows["finance_group"] = np.select([all_rows.industry_name.eq("银行"), all_rows.industry_name.eq("非银金融"), all_rows.industry_name.eq("行业缺失")], ["银行", "非银金融", "行业缺失"], default="非金融")
    all_rows["income_update_after_observation"] = all_rows.income_dependency_update_max.gt(all_rows.observation_date)
    all_rows["cash_update_after_observation"] = all_rows.cash_dependency_update_max.gt(all_rows.observation_date)
    all_rows["income_current_valid"] = all_rows.income_age_days.between(0, 200) & all_rows[["parent_net_profit_ttm", "parent_net_profit_ttm_prior_year", "total_operating_revenue_ttm", "total_operating_revenue_ttm_prior_year"]].notna().all(axis=1)
    all_rows["cash_matched_valid"] = all_rows.finance_group.eq("非金融") & all_rows.income_current_valid & all_rows.cash_age_days.between(0, 200) & all_rows.cash_report_end.eq(all_rows.income_report_end) & all_rows[["operating_cashflow_ttm", "operating_cashflow_ttm_prior_year"]].notna().all(axis=1)
    all_rows["financial_vintage_status"] = "RETROSPECTIVE_CURRENT_VALUES_WITH_NOMINAL_NOTICE_CLOCK_NOT_STRICT_PIT"
    all_rows.to_parquet(OUT / "results" / "全部观察日_公司财务重建与版本标记.parquet", index=False)
    csv(pd.DataFrame(missing), "成分覆盖缺失原点.csv")
    return all_rows


def aggregate_one(part, date, kind, label):
    valid = part[part.income_current_valid]
    row = {"observation_date": date, "group_type": kind, "group": label, "members": len(part), "valid_income_members": len(valid), "diagnostic_reference_weight": part.diagnostic_weight.sum(min_count=1), "diagnostic_weight_members": part.diagnostic_weight.notna().sum(), "valid_income_reference_weight": valid.diagnostic_weight.sum(min_count=1), "financial_source_version": "CURRENT_VALUES_RECONSTRUCTED_NOT_STRICT_PIT"}
    for field in ["parent_net_profit", "total_operating_revenue", "operating_profit"]:
        paired = valid[valid[[field + "_ttm", field + "_ttm_prior_year"]].notna().all(axis=1)]
        current, previous = paired[field + "_ttm"].sum(min_count=1), paired[field + "_ttm_prior_year"].sum(min_count=1)
        row[field + "_ttm_sum"] = current
        row[field + "_prior_ttm_sum"] = previous
        row[field + "_change_sum"] = current - previous
        row[field + "_yoy"] = current / previous - 1 if previous > 0 else np.nan
    row["profit_improving_company_fraction"] = (valid.parent_net_profit_ttm > valid.parent_net_profit_ttm_prior_year).mean() if len(valid) else np.nan
    row["income_version_future_update_members"] = int(valid.income_update_after_observation.sum())
    row["income_report_periods"] = json.dumps(valid.income_report_end.dt.strftime("%Y-%m-%d").value_counts().to_dict(), ensure_ascii=False)
    row["income_median_report_age_days"] = valid.income_age_days.median()
    row["income_latest_nominal_announcement"] = valid.income_nominal_announcement.max()
    cf = part[part.cash_matched_valid]
    row["matched_nonfinancial_cash_members"] = len(cf)
    for suffix in ["ttm", "ttm_prior_year"]:
        for field in ["operating_cashflow", "parent_net_profit", "total_operating_revenue"]:
            row["cash_cohort_" + field + "_" + suffix + "_sum"] = cf[field + "_" + suffix].sum(min_count=1)
    current_cf = row["cash_cohort_operating_cashflow_ttm_sum"]
    old_cf = row["cash_cohort_operating_cashflow_ttm_prior_year_sum"]
    row["cash_cohort_cashflow_yoy"] = current_cf / old_cf - 1 if old_cf > 0 else np.nan
    profit = row["cash_cohort_parent_net_profit_ttm_sum"]
    row["cash_cohort_cashflow_to_profit"] = current_cf / profit if profit > 0 else np.nan
    row["cash_cohort_cashflow_improving_fraction"] = (cf.operating_cashflow_ttm > cf.operating_cashflow_ttm_prior_year).mean() if len(cf) else np.nan
    row["cash_version_future_update_members"] = int((cf.income_update_after_observation | cf.cash_update_after_observation).sum())
    return row


def aggregate(components):
    rows = []
    for date, part in components.groupby("observation_date", sort=True):
        rows.append(aggregate_one(part, date, "总样本", "全部"))
        for label, p in part.groupby("finance_group"):
            rows.append(aggregate_one(p, date, "金融分层", label))
        for label, p in part.groupby("industry_name"):
            rows.append(aggregate_one(p, date, "全部行业", label))
    result = pd.DataFrame(rows)
    totals = result[result.group_type.eq("总样本")].set_index("observation_date")
    den = result.observation_date.map(totals.parent_net_profit_prior_ttm_sum)
    total_now = result.observation_date.map(totals.parent_net_profit_ttm_sum)
    result["profit_change_contribution_to_all_growth_pp"] = 100 * result.parent_net_profit_change_sum / den.where(den > 0)
    result["share_of_all_group_reported_profit"] = result.parent_net_profit_ttm_sum / total_now.where(total_now > 0)
    csv(result, "逐观察日_全部行业盈利与非金融现金流.csv")
    return result


def price_bridge(origins):
    prices = read("index_price.parquet").copy()
    prices["date"] = pd.to_datetime(prices.date).dt.normalize()
    prices = prices.sort_values("date").drop_duplicates("date").set_index("date")
    pe = read("pe.parquet").copy()
    pe["observation_date"] = pd.to_datetime(pe.observation_date).dt.normalize()
    pe["available_at"] = pd.to_datetime(pe.available_at, utc=True).dt.tz_convert("Asia/Shanghai")
    pe = pe.sort_values("observation_date").drop_duplicates("observation_date")
    pe_dates = pe.set_index("observation_date")
    records = []
    for r in origins.to_dict("records"):
        t = pd.Timestamp(r["snapshot_at"])
        observed = pd.Timestamp(r["observation_date"])
        row = {"origin_key": r["origin_key"], "price_bridge_status": "NO_VIEW_NO_MATCHING_PRICE_AND_PE"}
        q = pe[pe.available_at.le(t)]
        if pd.isna(observed) or not len(q):
            records.append(row)
            continue
        end = q.iloc[-1].observation_date
        if (observed - end).days > 5 or end not in prices.index:
            row["price_bridge_status"] = "NO_VIEW_STALE_OR_MISSING_SAME_DATE_SOURCE"
            records.append(row)
            continue
        end_price, end_pe = float(prices.loc[end, "close"]), float(q.iloc[-1].original_pe_official)
        row.update(price_bridge_status="EXACT_ALGEBRA_NOT_EARNINGS_CAUSAL_ATTRIBUTION", price_bridge_end_date=end, price_bridge_end_available_at=q.iloc[-1].available_at, price_bridge_end_price=end_price, price_bridge_end_pe=end_pe, price_bridge_end_implied_denominator=end_price / end_pe)
        index = prices.index.get_loc(end)
        for h in [20, 60]:
            prefix = f"price_bridge_{h}_"
            if index < h:
                continue
            start = prices.index[index - h]
            if start not in pe_dates.index:
                continue
            begin_price, begin_pe = float(prices.loc[start, "close"]), float(pe_dates.loc[start, "original_pe_official"])
            p = 100 * np.log(end_price / begin_price)
            v = 100 * np.log(end_pe / begin_pe)
            e = 100 * np.log((end_price / end_pe) / (begin_price / begin_pe))
            row.update({prefix + "start_date": start, prefix + "start_price": begin_price, prefix + "start_pe": begin_pe, prefix + "log_price_pp": p, prefix + "log_pe_pp": v, prefix + "log_implied_denominator_pp": e, prefix + "identity_error": p - v - e, prefix + "price_return": end_price / begin_price - 1})
        records.append(row)
    result = pd.DataFrame(records)
    csv(result, "所有原点_同日期价格与PE恒等式.csv")
    return result


def write_panels(origins, summaries, bridges):
    primary = summaries[summaries.group_type.isin(["总样本", "金融分层"])].copy()
    fields = ["members", "valid_income_members", "diagnostic_reference_weight", "parent_net_profit_yoy", "total_operating_revenue_yoy", "share_of_all_group_reported_profit", "profit_change_contribution_to_all_growth_pp", "profit_improving_company_fraction", "income_version_future_update_members", "income_median_report_age_days", "income_report_periods", "matched_nonfinancial_cash_members", "cash_cohort_cashflow_yoy", "cash_cohort_cashflow_to_profit", "cash_cohort_cashflow_improving_fraction"]
    main = origins.copy()
    for label, prefix in [("全部", "all"), ("银行", "bank"), ("非银金融", "nonbank"), ("非金融", "nonfinancial"), ("行业缺失", "unknown_industry")]:
        part = primary[primary.group.eq(label)][["observation_date", *fields]].rename(columns={f: "financial_" + prefix + "_" + f for f in fields})
        main = main.merge(part, on="observation_date", how="left", validate="many_to_one")
    main = main.merge(bridges, on="origin_key", how="left", validate="one_to_one")
    main["financial_state_status"] = np.where(main.financial_all_members.notna(), "RETROSPECTIVE_CURRENT_VALUES_NOTICE_CLOCK_NOT_FIRST_VERSION", "NO_VIEW_COMPONENT_SOURCE_COVERAGE")
    main["financial_is_strict_point_in_time"] = False
    for kind, part in main.groupby("panel"):
        labels = [c for c in part if c.startswith(("E0_", "E1_", "delay_change_", "first_reaction_"))]
        csv(part.drop(columns=labels), kind + "_新增盈利定价证据_不含未来标签.csv")
        csv(part, kind + "_盈利定价与宏观波动完整连接.csv")
    cases = main[(main.panel == "104个月") & main.stat_month.isin(["2020-06", "2021-01", "2024-08", "2025-07", "2025-08"])]
    csv(cases, "五个固定病例_盈利与定价.csv")
    return main


def run():
    assert not (OUT / "results" / "build_receipt.json").exists(), "第五轮已有完成记录，不覆盖。"
    prepare()
    origins = pd.concat([read("monthly.csv").assign(panel="104个月"), read("weekly.csv").assign(panel="445周")], ignore_index=True)
    origins["origin_key"] = origins.panel + "|" + origins.origin_id.astype(str)
    origins["observation_date"] = pd.to_datetime(origins.observation_date)
    assert origins.origin_key.is_unique
    calendar = pd.DatetimeIndex(pd.to_datetime(read("market.csv").date).sort_values())
    income = events(read("income.parquet"), ["parent_net_profit", "total_operating_revenue", "operating_profit"], calendar, "income")
    cash = events(read("cashflow.parquet"), ["operating_cashflow"], calendar, "cash")
    print(f"已构造名义公告后的利润事件{len(income)}条、现金流事件{len(cash)}条；财务版本风险逐行保留。", flush=True)
    components = financial_components(origins, income, cash)
    print(f"已连接{len(components):,}个公司观察，共{components.observation_date.nunique()}个不同交易日。", flush=True)
    summary = aggregate(components)
    bridge = price_bridge(origins)
    main = write_panels(origins, summary, bridge)
    result = {"at": now(), "study_id": "510300_MACRO_EARNINGS_PRICING_BRIDGE_V5", "status": "RETROSPECTIVE_EARNINGS_AND_PRICING_BRIDGE_BUILT", "origins": len(origins), "different_company_observation_dates": components.observation_date.nunique(), "company_observations": len(components), "industry_and_group_rows": len(summary), "financial_state_coverage": main.groupby("panel").financial_state_status.value_counts().to_dict(), "financial_source": "SECONDARY_CURRENT_VALUES_MAY_INCLUDE_LATER_REVISIONS", "new_models": 0, "new_accounts": 0, "goal_status": "active", "goal_achieved": False}
    save(OUT / "results" / "build_receipt.json", result)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print("盈利、回款和同日期价格/PE连接已保存；尚未形成方向预测。", flush=True)


if __name__ == "__main__":
    run()
