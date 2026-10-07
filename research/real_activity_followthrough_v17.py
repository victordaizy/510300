"""沿既有联合状态，观察后续真实活动披露和等待确认后的剩余价格路径。"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_real_activity_followthrough_v17"
PARENT = ROOT / "reports/research/510300_macro_transmission_context_v4"
CUTOFF = pd.Timestamp("2026-09-11T21:00:00+08:00")
SUPPORTED = "剪刀差改善_信贷与订单共同支持"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save_json(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv(name, frame):
    frame.to_csv(OUT / "results" / name, index=False, encoding="utf-8-sig", float_format="%.12g")


def clock(value):
    if pd.isna(value):
        return pd.NaT
    t = pd.Timestamp(value)
    return t.tz_localize("Asia/Shanghai") if t.tzinfo is None else t.tz_convert("Asia/Shanghai")


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("观察方案已冻结，不重复覆盖。")
    for folder in ["inputs", "results", "figures", "code"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    protocol = {
        "at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "study_id": "510300_REAL_ACTIVITY_FOLLOWTHROUGH_V17",
        "previous_turn_classification": "PROGRESS：V16公司原文和完整成分贡献已完成，整体目标仍未达成。",
        "question": "原先货币、信贷与订单共同支持的背景，随后订单及回款是否继续改善；经营确认后原20日还有多少价格路径？",
        "population": "原104个月全部保留；旧新M1分开；原行情截至2026-09-11。V4既有联合状态原样继承，不改阈值、不加条件、不删失败月。",
        "selection": "原结果已见，以下为后续传导诊断而非独立样本预测检验。旧口径共同支持的11个月全部列出，并展示其他全部状态。",
        "orders_followthrough": "以起点已知新订单统计月为基准，严格取随后第1、2、3个统计月的原发布值。三期完整才算均值、相对起点变化及是否均不低于50；缺月不跳过。50仅沿PMI经济含义，不搜索阈值。",
        "industrial_followthrough": "在起点已知工业统计期之后，按公开时钟取后续三个新的披露期；1至2月合并保留，不假装每月都有报告。记录原文公布的回收期同比变化及累计利润同比；回收期同比缩短与同比拖延程度收窄分别判断。缺起点不凭后来报告补齐。",
        "survey_followthrough": "分别取起点已知企业家及银行家调查的下一季度，记录收款、资金周转、贷款需求的原值、环比及与上年同季差；未覆盖不跳至更晚季度。季度是调查名称，按实际公开时钟，不假设季末才可用。问卷感受不是实际现金流金额。",
        "clock": "未来经营数据只作为事后结果，与起点信息分列。仅使用截止日前已公开的存量原件。分别保留统计期、发布时间上界、起点和确认快照。",
        "confirmation_path": "固定采用下一期新订单的公开时钟，不按值的好坏改变时钟。第一个不早于公开上界的每日21点快照后，下一A股交易日开盘观察原终点前剩余收益；再延迟一交易日同时保存。晚于原终点记无剩余窗口，不填零。",
        "path_accounting": "把原固定股数的20日收益按确认后开盘切分为之前与之后的本金贡献。确认日除息权益归此前持有人，重新开盘买入不享当日已除息权益。分红留现金，未扣费用，非策略或可执行账户。",
        "volatility": "确认快照时计算已知的20日上行、下行及总波动，与原起点对照；后续经营或波动不回填为起点条件。",
        "summary": "按原状态和M1制度列全体分布与逐年结果，报告均值、中位数、正收益数、最差路径及E1。季度结果在各组内按不同目标季度去重均权，披露重复月数。三月订单结果窗口重叠，不把月数当独立试验数。",
        "counterfactual_limit": "共同支持组起点订单已较高、时期集中，后续高水平不能直接称增量预测能力；不比较挑选后的最好组，不计算未经设计的显著性，不重拟合V14或旧失败模型。",
        "new_models": 0, "new_accounts": 0, "independent_validation": False, "goal_achieved": False, "orders_authorized": False,
    }
    save_json("protocol.json", protocol)
    paths = {
        "monthly.csv": ROOT / "reports/research/510300_constituent_external_bridge_v16/inputs/monthly.csv",
        "pmi_orders.parquet": PARENT / "inputs/pmi_orders.parquet",
        "pmi_extra.json": PARENT / "results/新订单既有原文补全.json",
        "industrial_profit.json": PARENT / "inputs/industrial_profit.json",
        "entrepreneur.parquet": PARENT / "inputs/entrepreneur.parquet",
        "banker.parquet": PARENT / "inputs/banker.parquet",
        "market.csv": PARENT / "inputs/market.csv",
        "daily_vol.csv": PARENT / "inputs/daily_vol.csv",
        "daily_paths.csv": ROOT / "reports/research/510300_constituent_external_bridge_v16/inputs/daily_paths.csv",
        "prior_context.csv": PARENT / "results/104个月_当时可见多层证据_不含未来标签.csv",
    }
    snapshots = []
    for name, source in paths.items():
        dest = OUT / "inputs" / name
        shutil.copy2(source, dest)
        snapshots.append({"name": name, "source": str(source), "sha256": digest(dest)})
    save_json("freeze.json", {"at": protocol["at"], "protocol_sha256": digest(OUT / "protocol.json"), "inputs": snapshots})
    shutil.copy2(OUT / "protocol.json", ROOT / "config/510300_real_activity_followthrough_v17.json")
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print("第十七轮已冻结：保留104个月及原11个共同支持月；区分未来经营结果和起点信息。")


def add_future(future_records, base, channel, horizon, target_period, source, value_fields):
    row = {"stat_month": base.stat_month, "training_regime": base.training_regime, "joint_state": base.joint_credit_orders_state,
           "origin_snapshot_at": base.snapshot_at, "channel": channel, "horizon": horizon, "target_period": target_period,
           "role": "事后传导结果，禁止进入起点输入", "status": "SOURCE_NOT_COVERED"}
    if source is not None:
        known = clock(source["known_at"])
        assert known > clock(base.snapshot_at)
        row.update({"known_at": known.isoformat(), "days_after_origin": (known - clock(base.snapshot_at)).total_seconds()/86400,
                    "source_url": source.get("source_url", source.get("url", "")),
                    "source_path": source.get("raw_path", source.get("pdf_raw_path", "")),
                    "source_sha256": source.get("source_hash", source.get("raw_sha256", source.get("pdf_sha256", ""))),
                    "historical_first_vintage_verified": False})
        row["status"] = "OBSERVED_BY_CUTOFF" if known <= CUTOFF else "AFTER_FIXED_CUTOFF"
        for field in value_fields:
            row[field] = source[field] if known <= CUTOFF else np.nan
    future_records.append(row)
    return row


def build(repair_baseline=False):
    old_supported = None
    if (OUT / "results/build_receipt.json").exists():
        if not repair_baseline:
            raise RuntimeError("第十七轮结果已完成，不覆盖原结果。")
        archive = OUT / "revisions/stale_baseline_fix"
        if archive.exists():
            raise RuntimeError("起点缺失修正只允许执行一次。")
        archive.mkdir(parents=True)
        old_supported = pd.read_csv(OUT / "results/原11个共同支持月_全部后续证据.csv")
        for source in (OUT / "results").iterdir():
            if source.is_file():
                shutil.copy2(source, archive / source.name)
        save_json("revisions/stale_baseline_fix/reason.json", {
            "reason": "初版误把保留有期名但已标记NO_VIEW_STALE_SOURCE的调查当成有效起点，违反冻结方案的缺起点不回填规则；按原状态清除其派生后续比较。",
            "changed_protocol": False, "changed_groups": False, "changed_thresholds": False,
            "draft_retained": True, "repairs": "企业家及银行家各12个过期起点，工业经营6个过期起点；原11个共同支持月的原有结果要求完全不变。"})
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    for row in frozen["inputs"]:
        assert digest(OUT / "inputs" / row["name"]) == row["sha256"]
    monthly = pd.read_csv(OUT / "inputs/monthly.csv")
    prior = pd.read_csv(OUT / "inputs/prior_context.csv").set_index("stat_month")
    assert len(monthly) == 104
    assert (monthly.joint_credit_orders_state.fillna("NO_VIEW").to_numpy() == prior.loc[monthly.stat_month, "joint_credit_orders_state"].fillna("NO_VIEW").to_numpy()).all()
    assert (monthly.joint_credit_orders_state == SUPPORTED).sum() == 11
    # 先单独保存起点表；下一期及更晚的所有经营数字随后才连接。
    origin_fields = ["stat_month", "training_regime", "joint_credit_orders_state", "snapshot_at", "observation_date", "delta3_spread_pp", "current_component", "base_component",
                     "loan_corporate_long_ytd_yoy_direction", "loan_household_long_ytd_yoy_direction", "orders_period", "orders_first_release_value",
                     "profit_period", "profit_status", "profit_receivable_days_published_yoy_change", "profit_profit_ytd_reported_yoy_pct", "entrepreneur_period", "entrepreneur_status", "banker_period", "banker_status",
                     "pre_return20_pp", "pre_return60_pp", "rv20", "downside20", "price_path_description"]
    csv("104个月_原起点状态_不含新未来结果.csv", monthly[origin_fields])
    origin_hash = digest(OUT / "results/104个月_原起点状态_不含新未来结果.csv")
    pmi = pd.read_parquet(OUT / "inputs/pmi_orders.parquet").rename(columns={"available_at": "known_at"})
    pmi = pd.concat([pmi, pd.DataFrame(json.loads((OUT / "inputs/pmi_extra.json").read_text(encoding="utf-8")))], ignore_index=True)
    assert not pmi.reference_period.duplicated().any()
    pmi = pmi.set_index("reference_period")
    industrial = pd.DataFrame(json.loads((OUT / "inputs/industrial_profit.json").read_text(encoding="utf-8")))
    industrial["time"] = industrial.known_at.map(clock)
    industrial = industrial.sort_values(["time", "stat_month"])
    surveys = {name: pd.read_parquet(OUT / "inputs" / (name + ".parquet")).set_index("quarter") for name in ["entrepreneur", "banker"]}
    market = pd.read_csv(OUT / "inputs/market.csv").set_index("date").sort_index()
    daily_vol = pd.read_csv(OUT / "inputs/daily_vol.csv").set_index("observation_date")
    paths = pd.read_csv(OUT / "inputs/daily_paths.csv")
    future_records, rows, survey_rows = [], [], []
    for base in monthly.itertuples():
        row = {field: getattr(base, field) for field in origin_fields}
        row.update({"E0_return_percent": base.E0_20_return * 100, "E1_return_percent": base.E1_20_return * 100,
                    "E0_worst_path_percent": base.E0_20_worst_path * 100, "E0_entry_date": base.E0_20_entry_date, "E0_exit_date": base.E0_20_exit_date})
        t0 = clock(base.snapshot_at)
        if pd.isna(t0) or t0 > CUTOFF or pd.isna(base.observation_date):
            row["row_status"] = "NO_VIEW_AFTER_MARKET_CUTOFF"
            rows.append(row)
            continue
        row["row_status"] = "ORIGIN_OBSERVED"
        pmi_future = []
        if pd.notna(base.orders_period):
            origin_period = pd.Period(base.orders_period, freq="M")
            base_pmi = pmi.loc[base.orders_period]
            assert clock(base_pmi.known_at) <= t0 and base_pmi.first_release_value == base.orders_first_release_value
            for k in [1, 2, 3]:
                target = str(origin_period + k)
                source = pmi.loc[target] if target in pmi.index else None
                f = add_future(future_records, base, "新订单", k, target, source, ["first_release_value"])
                pmi_future.append(f)
                row[f"orders_next{k}_period"] = target
                row[f"orders_next{k}_status"] = f["status"]
                row[f"orders_next{k}_value"] = f.get("first_release_value", np.nan)
                row[f"orders_next{k}_known_at"] = f.get("known_at")
            values = [x.get("first_release_value", np.nan) for x in pmi_future]
            row["orders_complete3"] = bool(np.isfinite(values).all())
            if row["orders_complete3"]:
                row.update({"orders_next3_mean": float(np.mean(values)), "orders_next3_mean_minus_origin": float(np.mean(values) - base.orders_first_release_value),
                            "orders_all_next3_ge50": bool(min(values) >= 50), "orders_next3_ge50_count": int(np.sum(np.array(values) >= 50)),
                            "orders_three_periods_last_known_at": pmi_future[-1]["known_at"]})
            if np.isfinite(values[0]):
                row["orders_next1_ge50"] = bool(values[0] >= 50)
                row["orders_next1_minus_origin"] = values[0] - base.orders_first_release_value
        row["industrial_baseline_status"] = base.profit_status
        if pd.notna(base.profit_period) and base.profit_status == "AVAILABLE_RECONSTRUCTED":
            base_industrial = industrial[industrial.stat_month == base.profit_period]
            assert len(base_industrial) == 1 and base_industrial.iloc[0].time <= t0
            candidates = industrial[(industrial.stat_month > base.profit_period) & (industrial.time > t0)].head(3)
            for k in [1, 2, 3]:
                source = candidates.iloc[k-1] if len(candidates) >= k else None
                target = source.stat_month if source is not None else "SOURCE_NOT_COVERED"
                f = add_future(future_records, base, "工业经营", k, target, source,
                               ["receivable_days_published_yoy_change", "profit_ytd_reported_yoy_pct"])
                row[f"industrial_next{k}_period"] = target
                row[f"industrial_next{k}_status"] = f["status"]
                row[f"industrial_next{k}_known_at"] = f.get("known_at")
                row[f"industrial_next{k}_collection_yoy_days"] = f.get("receivable_days_published_yoy_change", np.nan)
                row[f"industrial_next{k}_profit_yoy_percent"] = f.get("profit_ytd_reported_yoy_pct", np.nan)
                if f["status"] == "OBSERVED_BY_CUTOFF":
                    row[f"industrial_next{k}_collection_yoy_change_from_origin"] = f["receivable_days_published_yoy_change"] - base.profit_receivable_days_published_yoy_change
        for channel, fields, prefix in [
            ("entrepreneur", ["sales_revenue_collection_index", "fund_turnover_index"], "entrepreneur"),
            ("banker", ["loan_demand_index"], "banker")]:
            baseline_period = getattr(base, prefix + "_period")
            row[channel + "_baseline_status"] = getattr(base, prefix + "_status")
            if pd.isna(baseline_period) or getattr(base, prefix + "_status") != "AVAILABLE_RECONSTRUCTED":
                continue
            survey = surveys[channel]
            baseline = survey.loc[baseline_period]
            assert clock(baseline.known_at) <= t0
            target = str(pd.Period(baseline_period, freq="Q") + 1)
            source = survey.loc[target] if target in survey.index else None
            f = add_future(future_records, base, channel, 1, target, source, fields)
            row[channel + "_next_quarter"] = target
            row[channel + "_next_status"] = f["status"]
            row[channel + "_next_known_at"] = f.get("known_at")
            if f["status"] == "OBSERVED_BY_CUTOFF":
                for field in fields:
                    row[field + "_next_value"] = f[field]
                    row[field + "_next_qoq_change"] = f[field] - baseline[field]
                    last_year = str(pd.Period(target, freq="Q") - 4)
                    yoy = f[field] - survey.loc[last_year, field] if last_year in survey.index and clock(survey.loc[last_year, "known_at"]) <= clock(f["known_at"]) else np.nan
                    row[field + "_next_yoy_change"] = yoy
                    survey_rows.append({"stat_month": base.stat_month, "training_regime": base.training_regime, "joint_state": base.joint_credit_orders_state,
                                        "channel": channel, "field": field, "base_quarter": baseline_period, "target_quarter": target,
                                        "future_value": f[field], "future_qoq_change": f[field] - baseline[field], "future_yoy_change": yoy, "known_at": f["known_at"]})
        # 同一观察规则用于所有下一期PMI，不依其结果决定何时观察价格。
        if pmi_future and pmi_future[0]["status"] == "OBSERVED_BY_CUTOFF":
            published = clock(pmi_future[0]["known_at"])
            confirmation = published.normalize() + pd.Timedelta(hours=21)
            if confirmation < published:
                confirmation += pd.Timedelta(days=1)
            date = str(confirmation.date())
            row["confirmation_snapshot_at"] = confirmation.isoformat()
            visible_days = market.index[market.index <= date]
            if len(visible_days) and confirmation <= CUTOFF:
                observation = visible_days[-1]
                row["confirmation_observation_date"] = observation
                if observation in daily_vol.index:
                    row["confirmation_rv20_percent"] = daily_vol.loc[observation, "v_rv20"] * 100
                    row["confirmation_downside20_percent"] = daily_vol.loc[observation, "v_downside20"] * 100
                    row["confirmation_upside20_percent"] = daily_vol.loc[observation, "v_upside20"] * 100
                    row["confirmation_downside_change_from_origin_pp"] = row["confirmation_downside20_percent"] - base.downside20 * 100
            future_days = market.index[market.index > date].tolist()
            for delay in [0, 1]:
                pfx = "confirmation" if delay == 0 else "confirmation_delay1"
                if len(future_days) <= delay or pd.isna(base.E0_20_exit_date) or future_days[delay] > base.E0_20_exit_date:
                    row[pfx + "_path_status"] = "NO_REMAINING_SESSION_BEFORE_ORIGINAL_EXIT"
                    continue
                start = future_days[delay]
                end = base.E0_20_exit_date
                selected = paths[(paths.stat_month == base.stat_month) & (paths.date.between(start, end))]
                assert len(selected) and start >= base.E0_20_entry_date
                original = paths[paths.stat_month == base.stat_month]
                price_start = float(market.loc[start, "open"])
                price_end = float(market.loc[end, "close"])
                dividend_before = original.loc[original.date <= start, "earned_dividend_today"].sum()
                dividend_after = original.loc[original.date > start, "earned_dividend_today"].sum()
                before = (price_start + dividend_before) / original.entry_open.iloc[0] - 1
                after = (price_end - price_start + dividend_after) / original.entry_open.iloc[0]
                assert abs(before + after - base.E0_20_return) < 1e-10
                row.update({pfx + "_path_status": "OBSERVED_WITHIN_ORIGINAL_WINDOW", pfx + "_entry_date": start,
                            pfx + "_remaining_sessions": len(selected), pfx + "_before_contribution_pp": before * 100,
                            pfx + "_after_contribution_pp": after * 100,
                            pfx + "_fresh_entry_return_percent": ((price_end + dividend_after)/price_start - 1) * 100})
        rows.append(row)
    frame = pd.DataFrame(rows)
    future = pd.DataFrame(future_records)
    survey_rows = pd.DataFrame(survey_rows)
    csv("104个月_后续经营结果与确认时钟.csv", frame)
    csv("后续经营披露_逐条来源与时间.csv", future)
    csv("后续季度调查_月份重复关系.csv", survey_rows)
    csv("原11个共同支持月_全部后续证据.csv", frame[frame.joint_credit_orders_state == SUPPORTED])
    aggregates = []
    for (regime, state), group in frame.groupby(["training_regime", "joint_credit_orders_state"], dropna=False):
        row = {"training_regime": regime, "joint_state": state, "origin_months": len(group), "years": "|".join(sorted(set(group.stat_month.str[:4])))}
        for col in ["E0_return_percent", "E1_return_percent", "E0_worst_path_percent", "pre_return60_pp", "orders_first_release_value", "orders_next3_mean",
                    "orders_next3_mean_minus_origin", "orders_next1_minus_origin", "confirmation_fresh_entry_return_percent", "confirmation_delay1_fresh_entry_return_percent",
                    "confirmation_before_contribution_pp", "confirmation_after_contribution_pp", "confirmation_remaining_sessions",
                    "industrial_next1_collection_yoy_days", "industrial_next1_collection_yoy_change_from_origin", "industrial_next1_profit_yoy_percent",
                    "industrial_next3_collection_yoy_days", "industrial_next3_collection_yoy_change_from_origin", "industrial_next3_profit_yoy_percent"]:
            values = group[col].dropna()
            row[col + "_n"] = len(values)
            row[col + "_mean"] = values.mean()
            row[col + "_median"] = values.median()
            row[col + "_positive_count"] = int((values > 0).sum())
        for col in ["orders_all_next3_ge50", "orders_next1_ge50"]:
            values = group[col].dropna()
            row[col + "_n"] = len(values)
            row[col + "_true_count"] = int(values.astype(bool).sum())
        row["worst_E0_path_percent"] = group.E0_worst_path_percent.min()
        aggregates.append(row)
    csv("全部既有状态_经营后续与价格分布.csv", pd.DataFrame(aggregates))
    years = []
    for (regime, state, year), group in frame.groupby(["training_regime", "joint_credit_orders_state", frame.stat_month.str[:4]], dropna=False):
        years.append({"training_regime": regime, "joint_state": state, "year": year, "months": len(group), "E0_mean_percent": group.E0_return_percent.mean(),
                      "E1_mean_percent": group.E1_return_percent.mean(), "orders_complete3_n": int(group.orders_complete3.fillna(False).astype(bool).sum()),
                      "all_next3_ge50_count": int(group.orders_all_next3_ge50.fillna(False).astype(bool).sum()), "prior60_mean_percent": group.pre_return60_pp.mean()})
    csv("全部既有状态_逐年结果.csv", pd.DataFrame(years))
    unique = survey_rows.drop_duplicates(["training_regime", "joint_state", "channel", "field", "target_quarter"])
    qsummary = unique.groupby(["training_regime", "joint_state", "channel", "field"], dropna=False).agg(
        unique_quarters=("target_quarter", "nunique"), future_value_mean=("future_value", "mean"),
        qoq_change_mean=("future_qoq_change", "mean"), yoy_change_mean=("future_yoy_change", "mean"),
        qoq_positive_quarters=("future_qoq_change", lambda v: int((v > 0).sum())), yoy_observed_quarters=("future_yoy_change", "count")).reset_index()
    csv("季度去重后的经营调查结果.csv", qsummary)
    supported = frame[frame.joint_credit_orders_state == SUPPORTED]
    receipt = {"status": "BUILT_REAL_ACTIVITY_FOLLOWTHROUGH", "origin_rows": len(frame), "future_source_rows": len(future),
               "supported_original_months": len(supported), "origin_only_sha256": origin_hash, "source_cutoff": CUTOFF.isoformat(),
               "new_models": 0, "new_accounts": 0, "independent_validation": False, "goal_achieved": False}
    save_json("results/build_receipt.json", receipt)
    if old_supported is not None:
        current = pd.read_csv(OUT / "results/原11个共同支持月_全部后续证据.csv")
        pd.testing.assert_frame_equal(old_supported, current[old_supported.columns], check_dtype=False, rtol=1e-9, atol=1e-10)
        save_json("results/起点缺失修正回执.json", {"status": "CORRECTED_TO_FROZEN_MISSING_POLICY", "original_11_month_results_unchanged": True,
                   "historical_draft": "revisions/stale_baseline_fix", "new_future_source_rows": len(future), "new_protocols_or_models": 0})
    shutil.copy2(Path(__file__), OUT / "code" / Path(__file__).name)
    print("已完成全部104个月的后续经营与确认时钟连接。")
    print(supported[["stat_month", "orders_first_release_value", "orders_next1_value", "orders_next2_value", "orders_next3_value", "orders_all_next3_ge50", "pre_return60_pp", "E0_return_percent", "E1_return_percent", "confirmation_entry_date", "confirmation_fresh_entry_return_percent"]].to_string(index=False))
    print(qsummary[qsummary.joint_state == SUPPORTED].to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["freeze", "build", "repair-baseline"]:
        raise SystemExit("用法：python research\\real_activity_followthrough_v17.py freeze、build 或 repair-baseline")
    freeze() if sys.argv[1] == "freeze" else build(repair_baseline=sys.argv[1] == "repair-baseline")
