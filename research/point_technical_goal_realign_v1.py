"""回到技术点位：将原账户全部交易对齐入场前可知指标，不产生新策略。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import upward_episode_anatomy_v1 as anatomy
from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json

OUT = ROOT / "reports/research/510300_technical_goal_realign_20261003"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
PATHS = ROOT / "reports/research/510300_point_weight_path_bottleneck_v1"
COMPARISON = ROOT / "reports/research/510300_point_second_weight_comparison_v1/results/完整权重共同账户比较.csv"
PERIODS = ["2015_2019", "2020_2026"]
INDICATORS = ["ac", "ema20", "daily_hist", "daily_hist_rising", "weekly_last_date",
              "weekly_hist", "weekly_hist_rising", "weekly_available", "relative_volume",
              "up_volume_balance5", "rv_ratio", "past20_return", "available"]


def describe(frame: pd.DataFrame) -> dict:
    """只描述保存周期的原净回报，零样本及没有亏损不伪造盈亏比。"""
    values = frame.net_return.to_numpy(float)
    positive, negative = values[values > 0], values[values < 0]
    n = len(values)
    p, q = (len(positive) / n, len(negative) / n) if n else (None, None)
    payoff = float(positive.mean() / -negative.mean()) if len(positive) and len(negative) else None
    pb = p * payoff if payoff is not None else None
    return {"completed_cycles": n, "wins": len(positive), "losses": len(negative),
            "ties": int((values == 0).sum()), "win_rate": p, "loss_rate": q,
            "actual_net_payoff": payoff, "p_times_actual_net_b": pb,
            "standard_net_expectancy_loss_units": pb - q if pb is not None else None,
            "mean_original_cycle_net_return": float(values.mean()) if n else None,
            "original_cycle_net_pnl_cny": float(frame.net_pnl.sum()) if n else None,
            "role": "HISTORICAL_DESCRIPTION_NOT_FILTERED_STRATEGY"}


def technical_context(data: pd.DataFrame) -> pd.DataFrame:
    dividends = data.loc[data.dividend.ne(0), ["date", "dividend"]].rename(
        columns={"date": "ex_date", "dividend": "cash_dividend_per_share"})
    result, _ = anatomy.features(data, dividends)
    return result


def run() -> None:
    require(not (OUT / "summary.json").exists(), "本次点位整理已完成，不覆盖。")
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [CURRENT / "inputs/candidate_features.parquet", COMPARISON,
                    ROOT / "research/upward_episode_anatomy_v1.py",
                    ROOT / "research/point_technical_goal_realign_v1.py",
                    PATHS / "interpretation.json"]
    source_paths.extend(PATHS / "results" / period / "STRESS/逐周期.parquet" for period in PERIODS)
    source_receipts = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in source_paths]
    write_json(OUT / "description_plan.json", {
        "at": now(), "role": "DESCRIPTIVE_REALIGNMENT_NOT_NEW_STRATEGY_PROTOCOL",
        "user_steering": "你做的偏了",
        "scope": "510300日线与此前完整周，多头点位，原压力成本账户的全部周期。",
        "question": "把原真实交易逐笔放回入场前的MACD、量价、波动背景，明确后续应改善哪一种交易决策。",
        "fixed_indicator_definition": "逐字复用原上涨图谱features：现金前向平移价、EMA20、MACD12/26/9、此前完整周MACD、前20日中位量及252日相对波动。周线130周预热沿用。",
        "clock": "实际入场日前一个原交易日收盘为指标原点；使用上一自然周，未完成本周不进入背景。",
        "partition": "全组及日MACD柱正/非正×上一完整周MACD柱正/非正四组；缺失背景单列，不填零。两个原时期分别描述，不按组排名选择过滤条件。",
        "outcomes": "原保存周期net_return、net_pnl原样保留；开放周期保留、不作完整胜负。分组指标不是应用过滤器后的净值或策略夏普。",
        "restrictions": "不搜索窗口、组合或阈值，不拟合、不新增入场或账户；不复活冻结失败。全部历史已用于开发，不称独立验证。",
        "sources": source_receipts,
    }, exclusive=True)
    data = pd.read_parquet(CURRENT / "inputs/candidate_features.parquet")
    require(len(data) == 3488 and data.symbol.eq("510300.SH").all()
            and data.date.is_unique and data.date.is_monotonic_increasing, "原标的、日期或日线母集变化。")
    context = technical_context(data)
    locations = {pd.Timestamp(date): i for i, date in enumerate(data.date)}
    cycle_frames, origin_indices = [], []
    for period in PERIODS:
        cycles = pd.read_parquet(PATHS / "results" / period / "STRESS/逐周期.parquet")
        cycles.insert(0, "period", period)
        local_indices = [locations[pd.Timestamp(date)] - 1 for date in cycles.entry_date]
        require(min(local_indices) >= 0, "入场前缺少原交易日。")
        snapshots = context.loc[local_indices, ["date"] + INDICATORS].reset_index(drop=True)
        snapshots = snapshots.rename(columns={"date": "indicator_origin"})
        snapshots.columns = [name if name == "indicator_origin" else "entry_known_" + name
                             for name in snapshots.columns]
        cycles = pd.concat([cycles.reset_index(drop=True), snapshots], axis=1)
        require((cycles.indicator_origin < cycles.entry_date).all(), "指标取用了入场当日未来收盘。")
        known_week = cycles.entry_known_weekly_last_date.notna()
        require((cycles.loc[known_week, "entry_known_weekly_last_date"]
                 < cycles.loc[known_week, "indicator_origin"]).all(), "未完成周进入了指标背景。")
        cycle_frames.append(cycles)
        origin_indices.extend(local_indices)
    all_cycles = pd.concat(cycle_frames, ignore_index=True)
    # 三个按时间确定的边界检查，验证历史指标不会因后来行情加入而变化。
    boundary_indices = sorted({min(origin_indices), max(origin_indices),
                               locations[pd.Timestamp("2020-01-02")] - 1})
    for index in boundary_indices:
        prefix = technical_context(data.iloc[:index + 1].copy())
        pd.testing.assert_series_equal(context.loc[index, INDICATORS], prefix.loc[index, INDICATORS],
                                       check_names=False, check_dtype=False, atol=1e-12, rtol=0)
    comparison = pd.read_csv(COMPARISON, encoding="utf-8-sig")
    fixed_accounts = comparison.loc[comparison.cost.eq("STRESS")].copy()
    groups = []
    for period in PERIODS:
        local = all_cycles.loc[all_cycles.period.eq(period) & all_cycles.status.eq("COMPLETE")].copy()
        aggregate = describe(local)
        saved = fixed_accounts.loc[fixed_accounts.period.eq(period)
                                   & fixed_accounts.policy.eq("A_SAVED_WEIGHT")].iloc[0]
        require(aggregate["completed_cycles"] == int(saved.completed_cycles)
                and abs(aggregate["p_times_actual_net_b"] - saved.p_times_b) < 1e-10
                and abs(aggregate["original_cycle_net_pnl_cny"] - saved.completed_cycle_net_pnl) < 1e-6,
                "原完整周期与账户统计没有对上。")
        groups.append({"period": period, "context_group": "ALL_ORIGINAL_COMPLETED_CYCLES", **aggregate})
        available = local.entry_known_available.eq(True)
        daily = local.entry_known_daily_hist.gt(0)
        weekly = local.entry_known_weekly_hist.gt(0)
        for daily_positive in [False, True]:
            for weekly_positive in [False, True]:
                subset = local.loc[available & daily.eq(daily_positive) & weekly.eq(weekly_positive)]
                name = ("日MACD柱正" if daily_positive else "日MACD柱非正") + "/" + (
                    "前完整周MACD柱正" if weekly_positive else "前完整周MACD柱非正")
                groups.append({"period": period, "context_group": name, **describe(subset)})
        groups.append({"period": period, "context_group": "UNKNOWN_ORIGINAL_INDICATOR_CONTEXT",
                       **describe(local.loc[~available])})
        require(sum(row["completed_cycles"] for row in groups if row["period"] == period
                    and row["context_group"] != "ALL_ORIGINAL_COMPLETED_CYCLES") == len(local),
                "原交易分组重复或遗漏。")
    tables = {"全部实际点位_入场前日周线量价": all_cycles,
              "固定MACD背景_全部胜负对照": pd.DataFrame(groups),
              "原压力成本完整账户指标": fixed_accounts}
    for name, frame in tables.items():
        frame.to_parquet(OUT / (name + ".parquet"), index=False)
        frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig", lineterminator="\n")
    result = {"at": now(), "study": "510300_TECHNICAL_GOAL_REALIGN_20261003",
              "technical_decision": "TECH.R148", "status": "COMPLETED_TECHNICAL_TRADE_MAP_DESCRIPTION",
              "saved_original_cycles_mapped": len(all_cycles),
              "completed_original_cycles": int(all_cycles.status.eq("COMPLETE").sum()),
              "unfinished_original_cycles_preserved": int(all_cycles.status.ne("COMPLETE").sum()),
              "entry_origins_before_execution": True, "prefix_boundary_checks": len(boundary_indices),
              "fixed_background_groups": groups, "new_model_fits": 0, "new_accounts": 0,
              "new_trades": 0, "new_return_labels": 0, "new_network_requests": 0,
              "return_and_sharpe_improved_this_study": False,
              "latest_actual_financial_model_decision": "TECH.R145",
              "R147_source_experiment": "NOT_REGISTERED_NOT_RUN_STOPPED_AFTER_USER_STEERING",
              "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
              "overfit_removed": False, "goal_achieved": False}
    require(all(digest(ROOT / row["path"]) == row["sha256"] for row in source_receipts),
            "描述执行期间原源变化。")
    write_json(OUT / "summary.json", result, exclusive=True)
    print(json.dumps({key: value for key, value in result.items() if key != "fixed_background_groups"}, ensure_ascii=False))


if __name__ == "__main__":
    run()
