"""离岸人民币额外走强的固定可行性检验，先测毛优势再决定是否运行账户。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.mechanism_odds_open_contract_v1 import save, read, digest, now, clean
from research.lpr_joint_response_v1 import directional_limit, gross_return

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_rmb_residual_state_v1"
PRIMARY = "RESIDUAL_RMB_STRONG"
GROUPS = [PRIMARY, "RAW_RMB_STRONG", "USD_WEAK", "ALL_SOURCE_ADMITTED"]
PERIODS = {"FULL": ("2019-01-01", "2025-12-31"), "EARLY": ("2019-01-01", "2021-12-31"), "LATE": ("2022-01-01", "2025-12-31")}
LABEL_COLUMNS = ["primary", "raw_rmb_strong", "usd_weak"]


def build_models(dates, sources):
    """汇率响应模型逐A股日更新；不读510300收益，也不纳入当前被解释的汇率行。"""
    dates = pd.DatetimeIndex(dates)
    source = sources.reset_index(drop=True)
    availability = pd.DatetimeIndex(source.available_at)
    x = np.column_stack([np.ones(len(source)), source.usd_change5, source.spread_change5])
    y = source.fx_change5.to_numpy(float)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    records, training = [], []
    for day in dates:
        clock = day.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=16)
        latest = int(availability.searchsorted(clock, side="right")-1)
        record = {"date": day, "decision_at": clock, "model_status": "NO_VIEW_SOURCE", "source_admitted": False,
            "source_idx": latest, "source_date": pd.NaT, "source_available_at": pd.NaT,
            "source_age_days": np.nan, "training_n": 0, "primary": None, "raw_rmb_strong": None, "usd_weak": None}
        if latest < 0:
            records.append(record)
            continue
        age = (day-source.date.iloc[latest]).days
        record.update(source_date=source.date.iloc[latest], source_available_at=source.available_at.iloc[latest], source_age_days=age)
        if age > 7 or not valid[latest]:
            record["model_status"] = "NO_VIEW_STALE_OR_INVALID_SOURCE"
            records.append(record)
            continue
        lower = day-pd.DateOffset(years=2)
        ids = np.flatnonzero(valid & (np.arange(len(source)) < latest) & source.date.ge(lower).to_numpy())
        record["training_n"] = len(ids)
        if len(ids) < 252:
            record["model_status"] = "NO_VIEW_TRAINING_HISTORY"
            records.append(record)
            continue
        beta, _, rank, _ = np.linalg.lstsq(x[ids], y[ids], rcond=None)
        if rank != 3:
            record["model_status"] = "NO_VIEW_MODEL_RANK"
            records.append(record)
            continue
        expected = float(x[latest] @ beta)
        residual = float(y[latest]-expected)
        record.update(model_status="AVAILABLE", source_admitted=True, training_start=source.date.iloc[ids[0]],
            training_end=source.date.iloc[ids[-1]], training_max_idx=int(ids[-1]),
            intercept=float(beta[0]), beta_usd=float(beta[1]), beta_spread=float(beta[2]),
            observed_fx_change5=float(y[latest]), usd_change5=float(x[latest, 1]), spread_change5=float(x[latest, 2]),
            predicted_fx_change5=expected, residual_fx_change5=residual,
            primary=residual < 0, raw_rmb_strong=y[latest] < 0, usd_weak=x[latest, 1] < 0)
        assert ids[-1] < latest and source.available_at.iloc[latest] <= clock
        training.append({"decision_date": day, "current_source_idx": latest, "training_indices": ids.tolist()})
        records.append(record)
    return pd.DataFrame(records), training


def weekly_origins(dates):
    days = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    iso = days.dt.isocalendar()
    keys = iso.year.astype(str)+"-"+iso.week.astype(str)
    return np.flatnonzero(~keys.duplicated().to_numpy())


def coverage_gate(models):
    points = models.iloc[weekly_origins(models.date)].copy()
    points["year"] = points.date.dt.year
    years = points.groupby("year").source_admitted.agg(["count", "sum", "mean"])
    halves = {key: int(points.loc[points.date.between(start, end), "source_admitted"].sum()) for key, (start, end) in PERIODS.items() if key != "FULL"}
    gates = {"overall_at_least_90pct": points.source_admitted.mean() >= .90,
        "each_year_at_least_85pct": years["mean"].ge(.85).all(), "both_subperiods_at_least_100": min(halves.values()) >= 100}
    return {"at": now(), "status": "PASS" if all(gates.values()) else "NOT_PASSED", "gates": gates,
        "registered_weekly_origins": len(points), "available_weekly_origins": int(points.source_admitted.sum()),
        "coverage_fraction": float(points.source_admitted.mean()), "yearly": years.reset_index().to_dict("records"),
        "subperiod_available": halves, "missing": points.loc[~points.source_admitted, ["date", "model_status", "source_date", "source_age_days"]].to_dict("records")}


def prepare():
    assert not (OUT / "freeze.json").exists()
    source = pd.read_parquet(OUT / "inputs/common_sources.parquet")
    market = pd.read_parquet(OUT / "inputs/market.parquet")
    dates = pd.to_datetime(market.date)
    dates = dates.loc[dates.between(*PERIODS["FULL"])]
    models, training = build_models(dates, source)
    models.to_parquet(OUT / "daily_models.parquet", index=False)
    save(OUT / "training_indices.json", training)
    receipt = coverage_gate(models)
    save(OUT / "source_model_coverage.json", receipt)
    print(f"逐日模型已形成：{len(models)}日，周原点可用{receipt['available_weekly_origins']}/{receipt['registered_weekly_origins']}，来源门{receipt['status']}。未计算ETF事件收益。", flush=True)


def freeze():
    assert not (OUT / "freeze.json").exists()
    assert read(OUT / "prefreeze_tests.json")["exit_code"] == 0
    assert read(OUT / "source_quality_receipt.json")["status"] in ["PASS_IDENTITY_AND_QUOTE_ALIGNMENT", "PASS_AVAILABLE_QUOTES_WITH_DECLARED_YEAR_GAP"]
    assert digest(ROOT / "config/510300_existing_data_training_mandate_v1.json") == digest(OUT / "inputs/authority_snapshot.json")
    protocol = {**read(OUT / "design_registration.json"), "frozen_at": now(),
        "source_revision": read(OUT / "source_revision_before_returns.json"),
        "source_gate_registration": read(OUT / "source_gate_registration.json"),
        "source_gap_policy": read(OUT / "source_gap_policy_before_returns.json"),
        "source_coverage": read(OUT / "source_model_coverage.json"),
        "source_identity_gate_update": "CNH=X仅能验证品种、没有历史；正式输入改为Dukascopy USD-CNH BID全区间，不拼接Yahoo。",
        "execution": "开盘涨停未成交记零收益且无费用代理；计划卖出开盘跌停顺延；末端未成熟保留censored；入场日除息不计权益、持有至退出开盘的除息计入。",
        "target_unchanged": "完整20万元账户压力净夏普>=1.2，CAGR>=10%，既定风险合同、稳健性与独立前向仍需成立。"}
    save(OUT / "protocol.json", protocol)
    paths = [Path(__file__), ROOT / "scripts/collect_rmb_residual_sources_v1.py", ROOT / "tests/test_rmb_residual_state_v1.py",
        ROOT / "research/mechanism_odds_open_contract_v1.py", ROOT / "research/lpr_joint_response_v1.py"]
    paths += [p for p in OUT.rglob("*") if p.is_file()]
    save(OUT / "freeze.json", {"at": now(), "new_etf_event_returns_not_computed": True,
        "files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]})
    print("模型、来源、周原点、方向和5日检验合同已冻结。", flush=True)


def frozen_check():
    for item in read(OUT / "freeze.json")["files"]:
        assert digest(ROOT / item["path"]) == item["sha256"], item["path"]


def evaluate_events(market, models):
    market = market.reset_index(drop=True)
    date_to_idx = {day: i for i, day in enumerate(market.date)}
    records = []
    for i in weekly_origins(models.date):
        row = models.iloc[i].to_dict()
        decision = date_to_idx[row["date"]]
        row.update(decision_idx=decision, entry_idx=decision+1, planned_exit_idx=decision+6,
            event_status=row["model_status"], gross_return=np.nan, stress_cost_proxy_return=np.nan)
        if not row["source_admitted"]:
            records.append(row)
            continue
        entry, end = decision+1, decision+6
        if entry >= len(market):
            row["event_status"] = "CENSORED_NO_ENTRY"
            records.append(row)
            continue
        row["entry_date"] = market.date.iloc[entry]
        if directional_limit(market, entry, 1):
            row.update(event_status="UNFILLED_UPPER_LIMIT", gross_return=0., stress_cost_proxy_return=0., exit_idx=entry, exit_date=market.date.iloc[entry])
            records.append(row)
            continue
        while end < len(market) and directional_limit(market, end, -1):
            end += 1
        if end >= len(market):
            row["event_status"] = "CENSORED_EXIT_AFTER_SAMPLE"
            records.append(row)
            continue
        value = gross_return(market, entry, end)
        row.update(event_status="MATURE", exit_idx=end, exit_date=market.date.iloc[end], gross_return=value,
            stress_cost_proxy_return=value-.0028, exit_delay_sessions=end-entry-5)
        records.append(row)
    return pd.DataFrame(records)


def mask(frame, group):
    column = {PRIMARY: "primary", "RAW_RMB_STRONG": "raw_rmb_strong", "USD_WEAK": "usd_weak", "ALL_SOURCE_ADMITTED": "source_admitted"}[group]
    return frame[column].eq(True)


def statistics(events):
    rows = []
    for period, (start, end) in PERIODS.items():
        part = events.loc[events.date.between(start, end) & events.gross_return.notna()]
        for group in GROUPS:
            chosen = part.loc[mask(part, group)]
            x = chosen.gross_return
            rows.append({"period": period, "group": group, "n": len(chosen), "mean_gross": x.mean(),
                "median_gross": x.median(), "mean_stress_cost_proxy": chosen.stress_cost_proxy_return.mean(),
                "positive_fraction": x.gt(0).mean() if len(x) else np.nan, "min_gross": x.min(), "max_gross": x.max()})
    return pd.DataFrame(rows)


def run():
    frozen_check()
    save(OUT / "run_started.json", {"at": now(), "stage": "FIXED_GROSS_SCREEN"})
    market = pd.read_parquet(OUT / "inputs/market.parquet")
    market.date = pd.to_datetime(market.date)
    market = market.loc[market.date.le(PERIODS["FULL"][1])].reset_index(drop=True)
    models = pd.read_parquet(OUT / "daily_models.parquet")
    sources = pd.read_parquet(OUT / "inputs/common_sources.parquet")
    events = evaluate_events(market, models)
    events.to_parquet(OUT / "events.parquet", index=False)
    events.to_csv(OUT / "全部周原点与事件收益.csv", index=False, encoding="utf-8-sig")
    stats = statistics(events)
    stats.to_csv(OUT / "group_statistics.csv", index=False, encoding="utf-8-sig")
    full = stats.loc[stats.period.eq("FULL")].set_index("group")
    halves = stats.loc[stats.group.eq(PRIMARY)].set_index("period")
    gates = {"mean_above_cost_proxy": full.loc[PRIMARY, "mean_gross"] > .0028,
        "mean_above_raw_fx_control": full.loc[PRIMARY, "mean_gross"] > full.loc["RAW_RMB_STRONG", "mean_gross"],
        "early_positive": halves.loc["EARLY", "mean_gross"] > 0, "late_positive": halves.loc["LATE", "mean_gross"] > 0}
    numeric_pass = all(gates.values())
    source_pass = read(OUT / "source_model_coverage.json")["status"] == "PASS"
    rng = np.random.default_rng(20260928)
    n, reps = len(events), 2000
    indices = ((rng.integers(0, n, (reps, math.ceil(n/8)))[:, :, None]+np.arange(8))%n).reshape(reps, -1)[:, :n]
    values = events.gross_return.fillna(0).to_numpy(float)
    def boot(group):
        eligible = (mask(events, group) & events.gross_return.notna()).to_numpy(bool)
        counts = eligible[indices].sum(axis=1)
        return np.divide((eligible[indices]*values[indices]).sum(axis=1), counts, out=np.full(reps, np.nan), where=counts>0)
    primary_boot = boot(PRIMARY)
    difference = primary_boot-boot("RAW_RMB_STRONG")
    uncertainty = {"method": "完整周原点8周循环区块2000次，未知/未成熟不加入均值",
        "primary_mean_95_interval": np.nanquantile(primary_boot, [.025, .975]).tolist(),
        "primary_minus_raw_fx_control_95_interval": np.nanquantile(difference, [.025, .975]).tolist(),
        "multiple_selection_adjusted": False, "historical_independence_established": False}
    save(OUT / "uncertainty.json", uncertainty)
    np.savez_compressed(OUT / "bootstrap_indices.npz", indices=indices.astype(np.int16))
    errors = []
    for row in events.loc[events.event_status.eq("MATURE")].itertuples():
        first, last = int(row.entry_idx), int(row.exit_idx)
        cash = float(market.dividend.iloc[first+1:last+1].sum())
        check = (float(market.open.iloc[last])-float(market.open.iloc[first])+cash)/float(market.open.iloc[first])
        errors.append(abs(check-row.gross_return))
    assert max(errors, default=0.) < 1e-12
    cutoff = pd.Timestamp("2022-01-01")
    changed = sources.copy()
    after = changed.available_at.gt(cutoff.tz_localize("Asia/Shanghai"))
    changed.loc[after, ["fx_change5", "usd_change5", "spread_change5"]] += [1.0, .5, 3.0]
    repeat, _ = build_models(models.date, changed)
    before = models.date.lt(cutoff)
    columns = ["model_status", "training_n", *LABEL_COLUMNS, "residual_fx_change5", "intercept", "beta_usd", "beta_spread"]
    pd.testing.assert_frame_equal(models.loc[before, columns], repeat.loc[before, columns])
    chosen = events.loc[mask(events, PRIMARY) & events.gross_return.notna()].copy()
    chosen.to_csv(OUT / "主规则事件.csv", index=False, encoding="utf-8-sig")
    largest = chosen.loc[chosen.gross_return.idxmax()] if len(chosen) else None
    save(OUT / "concentration_diagnostic.json", {"post_result_descriptive_only": True,
        "primary_events": len(chosen), "largest_event": largest.to_dict() if largest is not None else None,
        "mean_without_largest": (chosen.gross_return.sum()-largest.gross_return)/(len(chosen)-1) if len(chosen)>1 else None,
        "main_events_deleted": 0})
    status = "PASS_FIXED_SCREEN_ACCOUNT_REQUIRED" if numeric_pass and source_pass else "PARTIAL_GROSS_SCREEN_FAILED_WITH_SOURCE_GAPS" if not numeric_pass and not source_pass else "REJECTED_FIXED_GROSS_SCREEN_NO_PARAMETER_RESCUE" if not numeric_pass else "SOURCE_GATE_NOT_PASSED"
    result = {"at": now(), "study_id": "510300_RMB_RESIDUAL_STATE_V1", "status": status,
        "registered_weekly_origins": len(events), "statuses": events.event_status.value_counts().to_dict(),
        "primary": full.loc[PRIMARY].to_dict(), "raw_fx_control": full.loc["RAW_RMB_STRONG"].to_dict(),
        "numeric_gates": gates, "source_gate_passed": source_pass,
        "account_status": "REQUIRED_NEXT_STAGE" if numeric_pass and source_pass else "NOT_RUN",
        "net_sharpe": None, "net_cagr": None, "maximum_drawdown": None, "new_accounts": 0,
        "parameter_searches": 0, "goal_achieved": False, "goal_status": "active", "independent_forward_observations": 0, "orders_authorized": False}
    save(OUT / "result.json", result)
    save(OUT / "verification_receipt.json", {"at": now(), "status": "PASS_MODEL_PREFIX_AND_EVENT_RECOMPUTATION",
        "tests": 4, "future_prefix_checks": 1, "mature_event_returns_recomputed": len(errors), "max_absolute_label_error": max(errors, default=0.),
        "boundary": "实现核对通过不代表收益达标或历史来源首次可得已证明。"})
    lines = ["|固定区间|主规则事件数|主规则5日毛均值|仅人民币走强对照|", "|---|---:|---:|---:|"]
    for key, label in [("FULL", "2019—2025"), ("EARLY", "2019—2021"), ("LATE", "2022—2025")]:
        a = stats.loc[(stats.period == key) & (stats.group == PRIMARY)].iloc[0]
        b = stats.loc[(stats.period == key) & (stats.group == "RAW_RMB_STRONG")].iloc[0]
        lines.append(f"|{label}|{int(a['n'])}|{a.mean_gross:.3%}|{b.mean_gross:.3%}|")
    conclusion = "通过固定可行性初筛，必须继续完整账户检验" if numeric_pass and source_pass else "有源子样本未通过固定毛收益初筛，且来源缺口仍保留" if not numeric_pass and not source_pass else "未通过预先固定的毛收益初筛，停止本版本" if not numeric_pass else "有源子样本数值初筛通过，但来源门未通过"
    text = f"""离岸人民币额外走强：{conclusion}。夏普1.2目标尚未实现。

先用此前两日历年的离岸人民币、美元指数和中美10年利差估计汇率共同变化，当前行不参加估计。每A股交易日更新；每周首个交易日收盘作为固定观察点，残差小于零为主规则，下一开盘起持有5个开盘间隔。每条外部来源都延迟至来源日期后第2日23:59才可用；7日以上陈旧或缺失保留NO_VIEW。不同市场日终不完全同步，回归残差不等同外生冲击或真实市场预期差。

{chr(10).join(lines)}

这些是事件毛收益，不是完整账户收益。主规则相对直接观察人民币走强的平均收益差95%描述区间为[{uncertainty['primary_minus_raw_fx_control_95_interval'][0]*100:.3f}, {uncertainty['primary_minus_raw_fx_control_95_interval'][1]*100:.3f}]个百分点；未校正此前研究选择，也不构成独立留出。0.28%比例费用代理仅作初筛，不包括最低佣金、整手、风险仓位及重叠持有路径。

共有{len(events)}个注册周原点；状态分布为{json.dumps(result['statuses'], ensure_ascii=False)}。来源覆盖门：{source_pass}。未知和末端未成熟均未填零；涨停未成交按注册规则保留零收益且不扣费用。

预定Yahoo CNH历史没有可用记录，因此在新收益计算前登记改为同一币种的Dukascopy全区间BID报价，未拼接供应商、未更换币种。美元指数使用Yahoo原始收盘，两国利率分别为美国财政部par yield和现有中债估值收益率。两者方法不同，利差只作观察代理。现存历史行情不等于当时逐日first_seen；本轮只证明保守日期代理下的可行性计算。

4项必要测试及未来前缀检查通过；{len(errors)}个成熟事件标签独立复算，最大绝对差{max(errors, default=0.):.3g}。账户状态{result['account_status']}，净夏普、年化、回撤尚未计算。不能晋升对照、翻转方向、改窗口或截取年份来补救。总目标active，无订单，不制作ZIP。

来源：[Dukascopy历史数据服务](https://www.dukascopy.com/api/data/get/historical-data-export)、[ICE美元指数说明](https://www.ice.com/forex/usdx)、[美国财政部利率口径](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/)、[财政部年度XML接口](https://home.treasury.gov/treasury-daily-interest-rate-xml-feed)。原始响应、来源替换原因、算法和模型训练索引均保留在本目录。
"""
    (OUT / "研究结论.md").write_text(text, encoding="utf-8")
    print(json.dumps(clean(result), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离岸人民币条件残差固定检验")
    parser.add_argument("stage", choices=["prepare", "freeze", "run"])
    args = parser.parse_args()
    {"prepare": prepare, "freeze": freeze, "run": run}[args.stage]()
