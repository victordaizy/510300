"""从保存账户复算稀疏节点的联合质量；不重新拟合或生成交易账户。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


STUDY = "510300_sparse_node_joint_quality_v1"
RECENT = [
    "510300_lpr_expectation_exploratory_training_v1",
    "510300_private_manager_exploratory_training_v1",
    "510300_equity_support_policy_event_v1",
    "510300_new_m1_consensus_training_v1",
    "510300_absorption_available_members_training_v1",
    "510300_if_exhaustion_existing_training_v1",
    "510300_dividend_payment_event_training_v1",
]
LATEST_REQUEST = "我们只需要交易某几个节点，高盈亏比，包括变盘信号，高胜率，高夏普，这些结合起来，拒绝过拟合"
EPSILON_CNY = 1e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    return float(value) if pd.notna(value) and np.isfinite(float(value)) else None


def boolean(value):
    return str(value).strip().lower() in {"true", "1"}


def policy_frame(frame):
    frame = frame.copy()
    if "timing" in frame:
        frame = frame.rename(columns={"timing": "model"})
    if "policy" not in frame:
        frame["policy"] = "ORIGINAL_SINGLE_POLICY"
    require(frame[["model", "scenario", "policy"]].notna().all().all(), "账户分组键有缺失。")
    return frame


def cycle_statistics(pnl, debit):
    pnl, debit = np.asarray(pnl, dtype=float), np.asarray(debit, dtype=float)
    require(len(pnl) == len(debit), "交易损益和投入金额长度不同。")
    require(np.isfinite(pnl).all() and np.isfinite(debit).all(), "已成交交易出现非有限损益或投入。")
    require((debit > 0).all(), "交易投入金额必须为正。")
    win, loss = pnl > EPSILON_CNY, pnl < -EPSILON_CNY
    n, nw, nl = len(pnl), int(win.sum()), int(loss.sum())
    return_value = pnl / debit
    mean_win = float(pnl[win].mean()) if nw else None
    mean_loss = float(-pnl[loss].mean()) if nl else None
    gross_win, gross_loss = float(pnl[win].sum()), float(-pnl[loss].sum())
    total = float(pnl.sum())
    largest_win = float(pnl[win].max()) if nw else None
    state = "DEFINED" if nw and nl else "NO_TRADES" if not n else "NO_OBSERVED_LOSS" if not nl else "NO_OBSERVED_WIN"
    return {
        "cycles": n, "winning_cycles": nw, "losing_cycles": nl, "breakeven_cycles": n - nw - nl,
        "win_rate": nw / n if n else None,
        "average_win_cny": mean_win, "average_loss_magnitude_cny": mean_loss,
        "realized_cash_payoff_ratio": mean_win / mean_loss if nw and nl else None,
        "realized_return_payoff_ratio": float(return_value[win].mean() / -return_value[loss].mean()) if nw and nl else None,
        "payoff_ratio_state": state,
        "profit_factor": gross_win / gross_loss if nl else None,
        "gross_winning_profit_cny": gross_win, "gross_losing_loss_cny": gross_loss,
        "net_cycle_profit_cny": total, "average_net_cycle_profit_cny": total / n if n else None,
        "average_normalized_cycle_return": float(return_value.mean()) if n else None,
        "largest_winner_cny": largest_win,
        "largest_winner_share_of_positive_profit": largest_win / gross_win if nw else None,
        "largest_winner_share_of_total_net_profit": largest_win / total if nw and total > EPSILON_CNY else None,
        "net_profit_minus_largest_winner_cny": total - largest_win if nw else None,
        "preentry_reward_risk_evidence": "NOT_MEASURED_FROM_SAVED_CYCLE_LEDGER",
    }


def measure_daily(daily, equity_column, return_column, stored_sharpe, stored_dd, capital=200000):
    dates = pd.to_datetime(daily["date"])
    require(len(daily) > 1 and dates.is_monotonic_increasing and not dates.duplicated().any(), "每日账户必须日期有序且唯一。")
    equity = daily[equity_column].to_numpy(dtype=float)
    require(np.isfinite(equity).all() and (equity > 0).all(), "账户权益必须为正且有限。")
    returns = equity / np.r_[float(capital), equity[:-1]] - 1
    return_error = float(np.max(np.abs(returns - daily[return_column].to_numpy(dtype=float))))
    require(return_error < 1e-10, "保存收益与权益连续变动不一致。")
    volatility = float(returns.std(ddof=1))
    sharpe = float(returns.mean() / volatility * np.sqrt(242)) if volatility > 1e-15 else None
    drawdown = float(-np.min(equity / np.maximum.accumulate(np.r_[float(capital), equity])[1:] - 1))
    saved_sharpe = number(stored_sharpe)
    require((sharpe is None) == (saved_sharpe is None), "保存夏普的缺失状态不同。")
    sharpe_error = abs(sharpe - saved_sharpe) if sharpe is not None else 0.0
    drawdown_error = abs(drawdown - abs(float(stored_dd)))
    require(sharpe_error < 1e-9 and drawdown_error < 1e-9, "完整每日账户未复现原指标。")
    fingerprint_input = "\n".join(f"{d:%Y-%m-%d},{v:.6f}" for d, v in zip(dates, equity))
    return {
        "start": dates.iloc[0].strftime("%Y-%m-%d"), "end": dates.iloc[-1].strftime("%Y-%m-%d"),
        "calendar_trading_days": len(daily), "capital_cny": capital, "ending_equity_cny": float(equity[-1]),
        "full_calendar_net_sharpe": sharpe, "max_drawdown_magnitude": drawdown,
        "annualized_return": float((equity[-1] / capital) ** (242 / len(daily)) - 1),
        "current_sharpe_drawdown_numeric_pass": bool(sharpe is not None and sharpe >= 1.3 and drawdown <= 0.1),
        "zero_return_days": int((np.abs(returns) < 1e-15).sum()),
        "daily_return_identity_error": return_error, "sharpe_recomputation_error": sharpe_error,
        "drawdown_recomputation_error": drawdown_error,
        "rounded_equity_path_sha256": hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest(),
    }


def compute(root):
    records, details = [], []
    for study in RECENT:
        folder = root / "inputs/recent" / study
        metrics = policy_frame(pd.read_csv(folder / "metrics.csv"))
        trades = policy_frame(pd.read_csv(folder / "cycles.csv"))
        daily_all = policy_frame(pd.read_csv(folder / "daily.csv"))
        protocol = load(folder / "protocol.json")
        require(protocol["annual_days"] == 242, "年化交易日定义不同。")
        grouped_daily = dict(tuple(daily_all.groupby(["model", "scenario", "policy"], sort=False)))
        grouped_trades = dict(tuple(trades.groupby(["model", "scenario", "policy"], sort=False)))
        require(len(grouped_daily) == len(metrics), "每日账簿分组与指标数量不同。")
        metric_keys = set()
        for metric in metrics.to_dict("records"):
            key = (metric["model"], metric["scenario"], metric["policy"])
            require(key not in metric_keys, "重复账户指标。")
            metric_keys.add(key)
            daily = grouped_daily[key]
            cycles = grouped_trades.get(key, trades.iloc[:0])
            n, winners = len(cycles), int((cycles["net_pnl_cny"] > EPSILON_CNY).sum())
            require(n == metric["opportunities"] and winners == metric["profitable_opportunities"], "保存成交数或盈利次数不一致。")
            account_id = "|".join([study, "evaluation", *key])
            row = {"account_id": account_id, "source_family": "RECENT_EVENT_STUDY", "study": study,
                   "model": key[0], "period": "evaluation", "cost": key[1], "policy": key[2],
                   "benchmark": key[0] in {"CASH", "BUY_AND_HOLD", "BUY_HOLD"},
                   "independent_validation": "NOT_ESTABLISHED"}
            row.update(measure_daily(daily, "equity_cny", "daily_return", metric["net_sharpe"], metric["max_drawdown"]))
            row.update(cycle_statistics(cycles["net_pnl_cny"], cycles["entry_cash_debit"]))
            expected = float(metric["ending_equity_cny"]) - 200000
            row["saved_cycle_profit_identity_error_cny"] = row["net_cycle_profit_cny"] - expected
            require(abs(row["saved_cycle_profit_identity_error_cny"]) < 1e-5, "成交损益无法解释完整账户期末损益。")
            require(abs(row["ending_equity_cny"] - float(metric["ending_equity_cny"])) < 1e-5, "期末权益与原指标不同。")
            row["cycles_per_242_days"] = n * 242 / len(daily)
            row["exposure_day_fraction"] = float(metric["exposure_day_fraction"])
            row["terminal_or_censored_cycles"] = int(cycles["exit_reason"].str.contains("TERMINAL|CENSOR|END_CLOSE", regex=True, na=False).sum())
            cost = protocol["account"]
            row.update({"commission_per_side": cost["commission_per_side"],
                        "slippage_per_side": cost["slippage_per_side"][key[1]], "minimum_commission_cny": cost["minimum_commission_cny"]})
            records.append(row)
            for order, cycle in enumerate(cycles.to_dict("records"), 1):
                details.append({"account_id": account_id, "cycle": order, "entry_date": cycle["entry_date"],
                                "exit_date": cycle["exit_date"], "net_profit_cny": cycle["net_pnl_cny"],
                                "buy_debit_cny": cycle["entry_cash_debit"],
                                "normalized_cycle_return": cycle["net_pnl_cny"] / cycle["entry_cash_debit"],
                                "exit_reason": cycle["exit_reason"]})
        require(set(grouped_trades) <= metric_keys, "成交组不在账户指标中。")

    registry = pd.read_csv(root / "inputs/old_14_scope.csv")
    require(len(registry) == 14 and not registry["model"].duplicated().any(), "原14方案范围不同。")
    for candidate in registry.to_dict("records"):
        study, model = candidate["source_study"], candidate["model"]
        folder = root / "inputs/older" / study
        config = load(folder / "config.json")
        require(config["initial_capital"] == 200000 and config["annual_days"] == 242, "旧账户资本或年化定义不同。")
        require(config["cash_annual_rate_assumption"] == 0, "旧现金收益假设不同。")
        cycles_all = pd.read_csv(folder / "cycles.csv")
        if "model" in cycles_all:
            cycles_all = cycles_all.loc[cycles_all["model"] == model]
        else:
            require(config["primary"] == model and config["candidate_configurations"] == 1, "无模型列的旧成交表不是单一主方案。")
        for period, filename in [("evaluation", "metrics.csv"), ("earlier_diagnostic", "earlier_diagnostics.csv")]:
            metrics = pd.read_csv(folder / filename)
            for cost_name in ["BASE", "STRESS"]:
                selected = metrics.loc[(metrics["model"] == model) & (metrics["cost"] == cost_name)]
                require(len(selected) == 1, "旧账户指标不唯一。")
                metric = selected.iloc[0]
                daily = pd.read_parquet(folder / period / cost_name / f"{model}_ledger.parquet")
                cycles = cycles_all.loc[(cycles_all["period"] == period) & (cycles_all["cost"] == cost_name)]
                account_id = "|".join([study, period, model, cost_name, "ORIGINAL_FIXED_RULE"])
                row = {"account_id": account_id, "source_family": "HISTORIC_SELECTED_14", "study": study,
                       "model": model, "period": period, "cost": cost_name, "policy": "ORIGINAL_FIXED_RULE",
                       "benchmark": False, "independent_validation": "NOT_ESTABLISHED"}
                row.update(measure_daily(daily, "equity", "net_return", metric["net_sharpe"], metric["max_drawdown"]))
                row.update(cycle_statistics(cycles["net_profit"], cycles["buy_debit"]))
                cycle_return_error = float(np.max(np.abs(cycles["net_profit"] / cycles["buy_debit"] - cycles["cycle_net_return"])))
                require(cycle_return_error < 1e-10, "旧周期收益归一化不同。")
                row["saved_cycle_profit_identity_error_cny"] = row["net_cycle_profit_cny"] - 200000 * float(metric["cumulative_return"])
                require(abs(row["saved_cycle_profit_identity_error_cny"]) < 1e-5, "旧交易周期与账户损益不同。")
                require(abs(row["ending_equity_cny"] - 200000 - row["net_cycle_profit_cny"]) < 1e-5, "旧账簿与周期损益不同。")
                row["cycles_per_242_days"] = len(cycles) * 242 / len(daily)
                row["exposure_day_fraction"] = float((daily["shares"] > 0).mean())
                row["terminal_or_censored_cycles"] = int(cycles["terminal_exit"].map(boolean).sum())
                cost = config["costs"][cost_name]
                row.update({"commission_per_side": cost["commission"], "slippage_per_side": cost["slippage"],
                            "minimum_commission_cny": cost["minimum"]})
                records.append(row)
                for cycle in cycles.to_dict("records"):
                    details.append({"account_id": account_id, "cycle": cycle["cycle"], "entry_date": cycle["entry_date"],
                                    "exit_date": cycle["exit_date"], "net_profit_cny": cycle["net_profit"],
                                    "buy_debit_cny": cycle["buy_debit"], "normalized_cycle_return": cycle["cycle_net_return"],
                                    "exit_reason": "TERMINAL_EXIT" if boolean(cycle["terminal_exit"]) else "ORIGINAL_RULE_EXIT"})
    accounts, cycles = pd.DataFrame(records), pd.DataFrame(details)
    accounts = accounts.sort_values("account_id", ignore_index=True)
    cycles = cycles.sort_values(["account_id", "cycle"], ignore_index=True)
    recent = accounts.loc[(accounts["source_family"] == "RECENT_EVENT_STUDY") & ~accounts["benchmark"]]
    older = accounts.loc[accounts["source_family"] == "HISTORIC_SELECTED_14"]
    require(len(accounts) == 170 and len(recent) == 86 and len(older) == 56, "本轮完整账户范围不同。")
    summary = {
        "study": STUDY, "status": "JOINT_MEASUREMENT_COMPLETE_NO_CANDIDATE_PROMOTION",
        "account_metric_records": len(accounts), "recent_strategy_records": len(recent),
        "recent_benchmark_records": int(accounts["benchmark"].sum()), "older_model_records": len(older),
        "older_models": int(older["model"].nunique()), "saved_cycle_rows": len(cycles),
        "recomputed_calendar_rows": int(accounts["calendar_trading_days"].sum()),
        "recent_strategy_numeric_passes": int(recent["current_sharpe_drawdown_numeric_pass"].sum()),
        "older_main_numeric_passes": int(older.loc[older["period"] == "evaluation", "current_sharpe_drawdown_numeric_pass"].sum()),
        "older_earlier_numeric_passes": int(older.loc[older["period"] == "earlier_diagnostic", "current_sharpe_drawdown_numeric_pass"].sum()),
        "maximum_cycle_profit_identity_error_cny": float(accounts["saved_cycle_profit_identity_error_cny"].abs().max()),
        "maximum_sharpe_recomputation_error": float(accounts["sharpe_recomputation_error"].max()),
        "maximum_daily_return_identity_error": float(accounts["daily_return_identity_error"].max()),
        "target_net_sharpe": 1.3, "target_max_drawdown": 0.1,
        "minimum_win_rate": None, "minimum_payoff_ratio": None,
        "minimums_reason": "用户未给胜率和盈亏比的数值门槛；本轮联合披露，不从结果倒推门槛。",
        "new_fits": 0, "new_accounts": 0, "new_downloads": 0, "independent_validation_established": False,
        "goal_achieved": False, "orders_authorized": False,
    }
    return accounts, cycles, summary


def snapshot(workspace, root):
    require(not root.exists(), "本轮目录已存在，不覆盖冻结记录。")
    root.mkdir(parents=True)
    sources = []

    def copy(source, relative):
        target = root / relative
        require(source.is_file(), f"缺少直接来源：{source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(digest(source) == digest(target), "来源复制后字节不同。")
        sources.append({"source": source.relative_to(workspace).as_posix(), "copy": relative,
                        "bytes": target.stat().st_size, "sha256": digest(target)})

    copy(workspace / "config/510300_existing_data_training_mandate_v1.json", "inputs/mandate_before.json")
    copy(workspace / "reports/research/510300_saved_candidates_sharpe13_v1/main_pass_14_with_target_change.csv", "inputs/old_14_scope.csv")
    for study in RECENT:
        folder = workspace / "reports/research" / study
        for source, name in [("results/账户指标.csv", "metrics.csv"), ("results/机会成交账簿.csv", "cycles.csv"),
                             ("results/完整逐日账户.csv", "daily.csv"), ("protocol.json", "protocol.json"),
                             ("summary.json", "summary.json"), ("delivery_receipt.json", "parent_delivery_receipt.json")]:
            copy(folder / source, f"inputs/recent/{study}/{name}")
    registry = pd.read_csv(root / "inputs/old_14_scope.csv")
    for study in registry["source_study"].unique():
        folder = workspace / "reports/research" / study
        copy(workspace / "config" / f"{study}.json", f"inputs/older/{study}/config.json")
        for source, name in [("saved_actual_cycles.csv", "cycles.csv"), ("metrics.csv", "metrics.csv"),
                             ("earlier_diagnostics.csv", "earlier_diagnostics.csv")]:
            copy(folder / source, f"inputs/older/{study}/{name}")
        if (folder / "acceptance_outcome.json").is_file():
            copy(folder / "acceptance_outcome.json", f"inputs/older/{study}/original_acceptance_outcome.json")
    for candidate in registry.to_dict("records"):
        for period in ["evaluation", "earlier_diagnostic"]:
            for cost in ["BASE", "STRESS"]:
                suffix = f"{candidate['source_study']}/{period}/{cost}/{candidate['model']}_ledger.parquet"
                copy(workspace / "reports/research" / suffix, f"inputs/older/{suffix}")
    copy(Path(__file__).resolve(), "code/sparse_node_joint_quality_v1.py")
    protocol = {
        "study_id": STUDY, "user_instruction": LATEST_REQUEST,
        "scope": "7个已保存事件研究的全部账户，以及上一轮已经列明的14个历史方案的两时期两成本；不重新选优。",
        "information_status": "历史结果此前已被查看；本轮是联合指标的事后测量，不是假装事前注册的新验证。",
        "recent_studies": RECENT, "older_scope": "inputs/old_14_scope.csv",
        "capital_cny": 200000, "target_net_sharpe": 1.3, "max_drawdown_magnitude": 0.1,
        "annual_days": 242, "risk_free_and_cash_daily_return": 0,
        "sharpe": "每日权益连续收益的均值/样本标准差×sqrt(242)，保留原账户全部空仓交易日。",
        "cycle": "完整买入至恢复现金周期；旧规则允许期间加减仓，以原保存周期边界为准。",
        "realized_cash_payoff_ratio": "盈利周期平均净利润/亏损周期平均净亏损绝对值。",
        "realized_return_payoff_ratio": "先将每个周期净利润除以累计买入支出，再算盈利均值/亏损均值绝对值；不是风险单位R。",
        "profit_factor": "全部盈利总额/全部亏损绝对额；与平均盈亏比区分。",
        "zero_tolerance_cny": EPSILON_CNY,
        "payoff_undefined": "没有交易、没有盈利或没有亏损时，平均盈亏比为空并说明原因；没有亏损不能证明无限盈亏比。",
        "terminal_exit_policy": "保留原强制期末清仓和截尾退出，另报数量，不通过删除它们提升指标。",
        "concentration": "最大盈利贡献和扣去该笔的金额归因；不重跑、不宣称删除大行情后仍可交易。",
        "preentry_reward_risk": "本轮的已实现周期数据不能建立入场前预期收益空间/失效风险；该项单独标为未测量。",
        "cost_comparison": "沿用各来源原成本。近期STRESS单边佣金2bp、滑点10bp；旧14方案STRESS单边佣金4bp、滑点10bp，均含最低5元。不得把同名STRESS视为完全相同。",
        "dependence": "跨模型、成本和历史分支重复使用事件，不合并为独立样本，不将14个相关变体当14次独立成功。",
        "win_rate_minimum": None, "payoff_ratio_minimum": None,
        "new_fits": 0, "new_accounts": 0, "new_downloads": 0,
    }
    write_json(root / "protocol.json", protocol)
    write_json(root / "freeze.json", {"frozen_at": datetime.now().astimezone().isoformat(),
                                     "historical_results_already_seen": True, "sources": sources})


def save(root, accounts, cycles, summary):
    destination = root / "results"
    destination.mkdir(parents=True, exist_ok=True)
    accounts.to_csv(destination / "完整账户联合指标.csv", index=False, encoding="utf-8-sig")
    cycles.to_csv(destination / "统一交易周期.csv", index=False, encoding="utf-8-sig")
    selected = accounts.loc[(accounts["source_family"] == "HISTORIC_SELECTED_14") & (accounts["cost"] == "STRESS")]
    selected.to_csv(destination / "原14方案跨时期联合指标.csv", index=False, encoding="utf-8-sig")
    write_json(root / "summary.json", summary)


def verify(root):
    for source in load(root / "freeze.json")["sources"]:
        path = root / source["copy"]
        require(path.stat().st_size == source["bytes"] and digest(path) == source["sha256"], "冻结文件发生变化。")
    accounts, cycles, summary = compute(root)
    for expected, name in [(accounts, "完整账户联合指标.csv"), (cycles, "统一交易周期.csv")]:
        saved = pd.read_csv(root / "results" / name)
        pd.testing.assert_frame_equal(saved, expected, check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-8)
    require(summary == load(root / "summary.json"), "结果摘要无法复现。")
    return {"status": "PASS_SAVED_ACCOUNT_AND_JOINT_METRIC_RECOMPUTATION", **summary}


def main():
    parser = argparse.ArgumentParser(description="保存账户的节点联合质量测量。")
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    root = args.root.resolve() if args.root else workspace / "reports/research" / STUDY
    if args.command == "run":
        snapshot(workspace, root)
        accounts, cycles, summary = compute(root)
        save(root, accounts, cycles, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(verify(root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
