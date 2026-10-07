"""铜金月度原版信息的固定增量检验；缺失月份保留，不生成伪造账户。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from research.mechanism_odds_open_contract_v1 import save, read, digest, now, clean
from research.lpr_joint_response_v1 import directional_limit, gross_return

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_copper_gold_monthly_vintage_v1"
PRIMARY = "CG_UP_TREND_UP"
GROUPS = [PRIMARY, "TREND_UP", "CG_UP", "ALL_RELEASES"]
HOLD = 20
PERIODS = {"FULL": ("2017-01", "2025-12"), "EARLY": ("2017-01", "2020-12"), "LATE": ("2021-01", "2025-12")}


def information_clock(dates, release_date):
    """采用注册的保守日期代理，真实历史首次获取仍未证明。"""
    dates = pd.DatetimeIndex(dates)
    cutoff = pd.Timestamp(release_date).tz_localize("Asia/Shanghai")+pd.Timedelta(days=2, hours=23, minutes=59)
    opens = dates.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=9, minutes=30)
    entry = int(opens.searchsorted(cutoff, side="right"))
    return cutoff, entry-1, entry


def ratio_change(row):
    return math.log(row["copper_latest_month"]/row["gold_latest_month"])-math.log(row["copper_previous_month"]/row["gold_previous_month"])


def build_events(market, sources):
    market = market.reset_index(drop=True)
    dates = pd.DatetimeIndex(market.date)
    growth = np.log((market.close+market.dividend)/market.close.shift()).rolling(20, min_periods=20).sum()
    by_month = {r["release_month"]: r for r in sources}
    records = []
    for period in pd.period_range("2017-01", "2025-12", freq="M"):
        month = str(period)
        row = {"release_month": month, "status": "NO_VIEW_SOURCE", "source_admitted": False,
            "cg_up": None, "trend_up": None, "primary": None, "gross_return": np.nan,
            "proportional_stress_proxy_return": np.nan, "decision_idx": np.nan, "entry_idx": np.nan}
        if month not in by_month:
            records.append(row)
            continue
        source = by_month[month]
        cutoff, decision, entry = information_clock(dates, source["release_date"])
        row.update(source_admitted=True, release_date=source["release_date"], source_url=source["url"],
            information_cutoff=cutoff, decision_idx=decision, entry_idx=entry,
            latest_statistic_month=source["latest_statistic_month"], previous_statistic_month=source["previous_statistic_month"])
        if decision < 20 or decision >= len(market):
            row["status"] = "NO_VIEW_MARKET_HISTORY"
            records.append(row)
            continue
        cg_change = ratio_change(source)
        trend = float(np.expm1(growth.iloc[decision]))
        row.update(cg_log_change=cg_change, trend20=trend, cg_up=cg_change>0, trend_up=trend>0,
            primary=cg_change>0 and trend>0, decision_price_date=dates[decision], planned_exit_idx=entry+HOLD)
        if entry >= len(market):
            row["status"] = "CENSORED_NO_ENTRY"
            records.append(row)
            continue
        row["entry_date"] = dates[entry]
        assert dates[decision].tz_localize("Asia/Shanghai")+pd.Timedelta(hours=15) < cutoff
        assert cutoff < dates[entry].tz_localize("Asia/Shanghai")+pd.Timedelta(hours=9, minutes=30)
        if directional_limit(market, entry, 1):
            row.update(status="UNFILLED_UPPER_LIMIT", gross_return=0., proportional_stress_proxy_return=0., exit_idx=entry, exit_date=dates[entry])
            records.append(row)
            continue
        end = entry+HOLD
        while end < len(market) and directional_limit(market, end, -1):
            end += 1
        if end >= len(market):
            row["status"] = "CENSORED_EXIT_AFTER_SAMPLE"
            records.append(row)
            continue
        payoff = gross_return(market, entry, end)
        row.update(status="MATURE", exit_idx=end, exit_date=dates[end], exit_delay_sessions=end-entry-HOLD,
            gross_return=payoff, proportional_stress_proxy_return=payoff-.0028)
        records.append(row)
    return pd.DataFrame(records)


def group_mask(frame, name):
    if name == PRIMARY:
        return frame.primary.eq(True)
    if name == "TREND_UP":
        return frame.trend_up.eq(True)
    if name == "CG_UP":
        return frame.cg_up.eq(True)
    return frame.source_admitted.eq(True)


def group_statistics(events):
    records = []
    for period, (start, end) in PERIODS.items():
        part = events.loc[events.release_month.between(start, end) & events.gross_return.notna()]
        for group in GROUPS:
            sample = part.loc[group_mask(part, group)]
            x = sample.gross_return
            records.append({"period": period, "group": group, "n": len(sample), "mean_gross": x.mean(),
                "median_gross": x.median(), "mean_proportional_stress_proxy": sample.proportional_stress_proxy_return.mean(),
                "positive_fraction": x.gt(0).mean() if len(x) else np.nan, "minimum": x.min(), "maximum": x.max()})
    return pd.DataFrame(records)


def frozen_check():
    for record in read(OUT / "freeze.json")["files"]:
        assert digest(ROOT / record["path"]) == record["sha256"], record["path"]


def freeze():
    assert not (OUT / "freeze.json").exists(), "版本已冻结，不覆盖"
    assert read(OUT / "prefreeze_tests.json")["exit_code"] == 0
    quality = read(OUT / "source_qa_receipt.json")
    assert quality["status"] == "PASS_SOURCE_TABLE_COLUMNS_AND_SAMPLE_VISUAL_CHECKS"
    source_status = read(OUT / "source_extraction_status.json")
    inputs = OUT / "inputs"
    inputs.mkdir(exist_ok=True)
    for source, name in [
        (ROOT / "reports/research/510300_mechanism_odds_open_contract_v1/inputs/market.parquet", "market.parquet"),
        (ROOT / "config/510300_existing_data_training_mandate_v1.json", "authority_snapshot.json")]:
        shutil.copy2(source, inputs / name)
    protocol = {**read(OUT / "design_registration.json"), "frozen_at": now(), "source_status": source_status,
        "source_gap_policy": read(OUT / "source_gap_registration.json"), "cash_dividend_accounting": "入场当天除息无权益；此前已持有的结束日除息仍计应收。",
        "execution_proxy": "开盘触及涨停不买入，事件收益零且不扣费用代理；卖出触及跌停顺延；未成熟标签不填零。",
        "strict_primary_target": "完整账户压力净夏普>=1.2、CAGR>=10%、既定回撤与风险合同、稳健性和独立前向支持，仍不因本初筛改变。"}
    save(OUT / "protocol.json", protocol)
    paths = [Path(__file__), ROOT / "research/mechanism_odds_open_contract_v1.py", ROOT / "research/lpr_joint_response_v1.py",
        ROOT / "scripts/collect_copper_gold_vintages_v1.py", ROOT / "tests/test_copper_gold_monthly_vintage_v1.py"]
    paths += [p for p in OUT.iterdir() if p.is_file()]
    paths += list(inputs.glob("*"))+list((OUT / "sources").glob("*"))+list((OUT / "source_qa").glob("*"))
    save(OUT / "freeze.json", {"at": now(), "before_event_return_computation": True,
        "files": [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths if p.is_file()]})
    print(f"固定月度定义已冻结；原版覆盖{source_status['extracted_months']}/108，缺失处理原样保留。", flush=True)


def run():
    frozen_check()
    save(OUT / "run_started.json", {"at": now(), "stage": "FIXED_GROSS_FEASIBILITY"})
    market = pd.read_parquet(OUT / "inputs/market.parquet")
    market.date = pd.to_datetime(market.date)
    market = market.loc[market.date.le("2025-12-31")].reset_index(drop=True)
    sources = read(OUT / "original_price_rows.json")
    events = build_events(market, sources)
    events.to_parquet(OUT / "events.parquet", index=False)
    events.to_csv(OUT / "全部108个月与固定事件收益.csv", index=False, encoding="utf-8-sig")
    stats = group_statistics(events)
    stats.to_csv(OUT / "group_statistics.csv", index=False, encoding="utf-8-sig")
    full = stats.loc[stats.period.eq("FULL")].set_index("group")
    halves = stats.loc[stats.group.eq(PRIMARY)].set_index("period")
    gates = {"primary_mean_above_cost_proxy": full.loc[PRIMARY, "mean_gross"] > .0028,
        "primary_mean_above_price_control": full.loc[PRIMARY, "mean_gross"] > full.loc["TREND_UP", "mean_gross"],
        "early_positive": halves.loc["EARLY", "mean_gross"] > 0,
        "late_positive": halves.loc["LATE", "mean_gross"] > 0}
    source_complete = len(sources) == 108
    partial_passed = all(gates.values())
    n, reps = len(events), 2000
    rng = np.random.default_rng(20260928)
    indices = ((rng.integers(0, n, size=(reps, math.ceil(n/4)))[:, :, None]+np.arange(4))%n).reshape(reps, -1)[:, :n]
    values = events.gross_return.fillna(0).to_numpy(float)
    def bootstrap_mean(group):
        eligible = (group_mask(events, group) & events.gross_return.notna()).to_numpy(bool)
        counts = eligible[indices].sum(axis=1)
        return np.divide((values[indices]*eligible[indices]).sum(axis=1), counts, out=np.full(reps, np.nan), where=counts>0)
    primary_draws = bootstrap_mean(PRIMARY)
    incremental = primary_draws-bootstrap_mean("TREND_UP")
    uncertainty = {"method": "完整108月序列四月循环区块；缺失月份不加入收益均值；2000次",
        "primary_mean_95_interval": np.nanquantile(primary_draws, [.025, .975]).tolist(),
        "primary_minus_price_control_95_interval": np.nanquantile(incremental, [.025, .975]).tolist(),
        "source_missingness_adjusted": False, "multiple_selection_adjusted": False}
    save(OUT / "uncertainty.json", uncertainty)
    np.savez_compressed(OUT / "bootstrap_indices.npz", indices=indices.astype(np.int16))
    # 复核未来变动不能改变过去的信号；只比较已经形成的决策。
    origin = int(events.loc[events.decision_idx.notna(), "decision_idx"].iloc[len(sources)//2])
    future = market.copy()
    future.loc[origin+1:, ["open", "high", "low", "close", "previous_close"]] *= 1.3
    future_events = build_events(future, sources)
    before = events.decision_idx.le(origin)
    pd.testing.assert_frame_equal(events.loc[before, ["primary", "cg_up", "trend_up"]], future_events.loc[before, ["primary", "cg_up", "trend_up"]])
    # 从原始开盘与分红独立重算保存标签，不运行第二套策略。
    errors = []
    for row in events.loc[events.status.eq("MATURE")].itertuples():
        entry, end = int(row.entry_idx), int(row.exit_idx)
        earned = sum(float(market.dividend.iloc[k]) for k in range(entry+1, end+1))
        reproduced = (float(market.open.iloc[end])-float(market.open.iloc[entry])+earned)/float(market.open.iloc[entry])
        errors.append(abs(reproduced-row.gross_return))
    assert max(errors, default=0.) < 1e-12
    chosen = events.loc[events.primary.eq(True) & events.gross_return.notna()].copy()
    chosen.to_csv(OUT / "主规则事件.csv", index=False, encoding="utf-8-sig")
    total = float(chosen.gross_return.sum())
    maximum = chosen.loc[chosen.gross_return.idxmax()] if len(chosen) else None
    remaining_mean = (total-float(maximum.gross_return))/(len(chosen)-1) if len(chosen)>1 else None
    concentration = {"post_result_descriptive_only": True, "primary_events": len(chosen),
        "largest_event": maximum.to_dict() if maximum is not None else None,
        "mean_without_largest": remaining_mean, "arithmetic_sum_not_account_return": total,
        "no_event_deleted_from_main_result": True}
    save(OUT / "concentration_diagnostic.json", concentration)
    state = "PASS_GROSS_SCREEN_ACCOUNT_REQUIRED" if source_complete and partial_passed else (
        "SOURCE_GAPS_WITH_PARTIAL_GROSS_SCREEN_PASS" if partial_passed else "PARTIAL_GROSS_SCREEN_FAILED_WITH_SOURCE_GAPS" if not source_complete else "REJECTED_FIXED_GROSS_SCREEN")
    result = {"at": now(), "study_id": "510300_COPPER_GOLD_MONTHLY_VINTAGE_V1", "status": state,
        "registered_months": 108, "source_months": len(sources), "missing_months": events.loc[events.status.eq("NO_VIEW_SOURCE"), "release_month"].tolist(),
        "statuses": events.status.value_counts().to_dict(), "primary_available_subsample": full.loc[PRIMARY].to_dict(),
        "price_control_available_subsample": full.loc["TREND_UP"].to_dict(), "numeric_gates_on_available_subsample": gates,
        "source_complete_gate": source_complete, "partial_numeric_screen_passed": partial_passed,
        "account_status": "REQUIRED_NEXT_STAGE" if source_complete and partial_passed else "NOT_RUN",
        "net_sharpe": None, "net_cagr": None, "maximum_drawdown": None, "new_accounts": 0,
        "parameter_searches": 0, "goal_achieved": False, "independent_forward_observations": 0, "orders_authorized": False}
    save(OUT / "result.json", result)
    save(OUT / "verification_receipt.json", {"at": now(), "status": "PASS_AVAILABLE_SOURCE_CLOCKS_AND_SAVED_EVENT_RECOMPUTATION",
        "implemented_tests": 4, "causal_trigger_prefix_checks": 1, "saved_mature_labels": len(errors),
        "max_absolute_label_error": max(errors, default=0.), "all_108_source_months_proven": source_complete,
        "boundary": "计算正确不证明缺失来源、策略达标或独立前向有效。"})
    table = ["|固定组别|成熟事件数|20日平均毛收益|减0.28%后的比例费用代理|", "|---|---:|---:|---:|"]
    labels = {PRIMARY: "铜金比上升且510300趋势为正", "TREND_UP": "只要求510300趋势为正", "CG_UP": "只要求铜金比上升", "ALL_RELEASES": "全部已准入报告"}
    for group in GROUPS:
        row = full.loc[group]
        table.append(f"|{labels[group]}|{int(row['n'])}|{row.mean_gross:.2%}|{row.mean_proportional_stress_proxy:.2%}|")
    conclusion = "已取得来源的子样本通过继续研究数值条件，但来源缺口尚未通过" if partial_passed and not source_complete else "已取得来源的子样本未通过预先固定的数值条件" if not partial_passed else "通过毛收益初筛，必须继续完整账户验证"
    text = f"""铜金月度原版信息：{conclusion}。夏普1.2目标尚未实现，账户未运行，夏普、年化和回撤未计算。

保留2017年至2025年完整108个月的研究日历。已取得{len(sources)}份当月世界银行Pink Sheet原版PDF；缺少{', '.join(result['missing_months'])}，这些月份为NO_VIEW_SOURCE，既不记为现金也不填零收益。表中的数字只代表已取得且标签成熟的子样本，不能推广成完整108个月的结论。

每份当月报告取最新已结束月和前月的铜、金月均价；比较同一份报告中的铜/金比值变化，避免混合不同版本。铜为美元/公吨、金为美元/金衡盎司，固定单位只影响比值水平，不影响同口径变化方向。信息截止设在报告表头日期之后第2个日历日23:59，再按此前最后收盘的510300含分红20日收益确认趋势，从下一开盘计算固定20个开盘间隔。这个时间是保守的历史发布日期代理，不是当时实际抓取证明。

{chr(10).join(table)}

主规则固定前段2017—2020年平均毛收益{halves.loc['EARLY','mean_gross']:.2%}（{int(halves.loc['EARLY','n'])}次），后段2021—2025年{halves.loc['LATE','mean_gross']:.2%}（{int(halves.loc['LATE','n'])}次）。相对纯趋势对照的平均收益差95%区间[{uncertainty['primary_minus_price_control_95_interval'][0]:.2%}, {uncertainty['primary_minus_price_control_95_interval'][1]:.2%}]。区间未校正此前多重研究选择，也未解决缺失月份的偏差。

0.28%只是双边比例佣金与滑点粗筛，不包含最低佣金、整手、仓位和账户路径。没有把事件收益年化或换算成夏普，也没有把未成熟事件填成零。

铜金比只表达一种跨市场假说：增长敏感需求相对避险需求可能改善。铜的供给、黄金特有需求、实际利率和指数行业构成均可能产生竞争解释；本轮不把比值变动直接认定为增长因果冲击。主组失败后不晋升其他组，不改变窗口、期限、年份或商品组合。

4项必要测试通过；{len(errors)}个成熟标签按原始开盘与实际持有分红独立复算，最大绝对差{max(errors, default=0.):.3g}。来源目录、信号和参数均在新收益计算前保存。目标继续，未产生订单，不制作交付ZIP。

来源入口：[世界银行商品价格资料](https://www.worldbank.org/en/research/commodity-markets)、[2021—2024年月度原版目录](https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/)、[2025年月度原版目录](https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/)。逐月URL、原文行和哈希在original_price_rows.json与原版铜金价格与月份.csv中。
"""
    (OUT / "研究结论.md").write_text(text, encoding="utf-8")
    print(json.dumps(clean(result), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="铜金月度原版信息固定检验")
    parser.add_argument("stage", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.stage]()
