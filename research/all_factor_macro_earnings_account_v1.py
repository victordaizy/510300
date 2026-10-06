"""使用已固定共同评分一次测量完整账户；不再次拟合或修改旧策略。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import all_factor_macro_earnings_joint_v1 as joint
from research import point_first_passage_study_v1 as original
from research.macro_technical_first_passage_clock_adapter_v1 import account
from research.point_account_nr7_inputs_v1 import metrics
from research.point_account_nr7_complement_v1 import annual_rows, verify_account

OUT = ROOT / "reports/research/510300_all_factor_macro_earnings_account_v1"
PREDICTIONS = joint.OUT / "results/全部四模型_共同原点评分及未知.parquet"
DATA = joint.OUT / "results/全部3488共同源视图_不足保留.parquet"
PRIMARY = joint.PRIMARY


def signals(data, prediction, policy):
    p = prediction[prediction.policy.eq(policy)].copy().reset_index(drop=True)
    joint.require(len(p) == len(data) and np.array_equal(
        pd.to_datetime(p.date).to_numpy(dtype="datetime64[ns]"),
        pd.to_datetime(data.date).to_numpy(dtype="datetime64[ns]")), "评分与账户日历不一致。")
    joint.require(not (p.candidate_quality_pass & ~p.status.eq("AVAILABLE")).any(), "未知评分产生进入请求。")
    p["entry_event"] = p.candidate_quality_pass.astype(bool)
    p["event_id"] = [f"{policy}_{i}" if on else None for i, on in enumerate(p.entry_event)]
    p["atr"] = data.atr14.to_numpy(float)
    p["stop_index"], p["target_index"] = np.nan, np.nan
    return p


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def baseline(period, cost):
    folder = original.CONTROL / period / cost / "A_SAVED_WEIGHT"
    saved = {k: pd.read_parquet(folder / (k + ".parquet")) for k in ["daily", "orders", "trades"]}
    saved["terminal"] = {"stopped": bool(saved["daily"].risk_stopped.iloc[-1])}
    return saved


def metrics_with_limits(result):
    stat = metrics(result)
    daily, trades = result["daily"], result["trades"]
    complete = trades[trades.status.eq("COMPLETE")]
    winners = complete.loc[complete.net_return.gt(0), "net_pnl"].astype(float)
    flat = daily.shares.eq(0).to_numpy(bool)
    longest, run = 0, 0
    for is_flat in flat:
        run = run + 1 if is_flat else 0
        longest = max(longest, run)
    orders = result["orders"]
    turnover = float((orders.quantity * orders.raw_open).sum()) / 200000 if len(orders) else 0.
    closed_pnl = float(complete.net_pnl.sum()) if len(complete) else 0.
    stat.update(worst_day=float(daily.net_return.min()),
                worst_trade=float(complete.net_return.min()) if len(complete) else np.nan,
                largest_winner_fraction_of_all_wins=float(winners.max() / winners.sum()) if len(winners) and winners.sum() > 0 else np.nan,
                longest_flat_sessions=longest, gross_turnover_over_initial_capital=turnover,
                open_pnl_cny=float(daily.equity.iloc[-1]) - 200000 - closed_pnl,
                unpaid_dividend_cny=float(daily.receivable.iloc[-1]))
    return stat


def gate(primary, controls):
    required = [primary.get(k, np.nan) for k in ["net_cagr", "net_sharpe", "p_times_b", "standard_expectancy_loss_units"]]
    if not np.isfinite(required).all():
        return False
    return bool(primary["net_cagr"] > max(0., *(c["net_cagr"] for c in controls))
                and primary["net_sharpe"] > max(0., *(c["net_sharpe"] for c in controls))
                and primary["p_times_b"] > 1 and primary["standard_expectancy_loss_units"] > 0
                and primary["max_drawdown"] <= .1)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    source_paths = [DATA, PREDICTIONS, joint.OUT / "protocol.json", joint.OUT / "summary.json",
                    joint.OUT / "saved_models_and_training_origins.json",
                    original.CURRENT / "inputs/candidate_prices.parquet",
                    original.WEIGHT / "inputs/dividends.csv", original.WEIGHT / "inputs/risks.parquet",
                    original.WEIGHT / "inputs/parent_signals.parquet", original.WEIGHT / "inputs/earlier_signals.parquet",
                    ROOT / "research/point_first_passage_study_v1.py",
                    ROOT / "research/point_first_passage_account_v1.py",
                    ROOT / "research/macro_technical_first_passage_clock_adapter_v1.py",
                    ROOT / "research/point_account_nr7_inputs_v1.py",
                    ROOT / "research/point_account_nr7_complement_v1.py",
                    ROOT / "research/daily_supply_test_v1.py",
                    ROOT / "reports/research/510300_macro_technical_first_passage_v1_clock_adapter/control_preflight.json",
                    ROOT / "tests/test_all_factor_macro_earnings_account_v1.py"]
    for period in original.PERIODS:
        for cost in original.COSTS:
            for name in ["daily", "orders", "trades"]:
                source_paths.append(original.CONTROL / period / cost / "A_SAVED_WEIGHT" / (name + ".parquet"))
    sources = [{"path": p.absolute().relative_to(ROOT).as_posix(), "sha256": joint.sha(p)} for p in source_paths]
    joint.write(OUT / "protocol.json", {
        "study_id": "510300_ALL_FACTOR_MACRO_EARNINGS_ACCOUNT_V1", "registered_at": joint.now(),
        "registration": "TECH.R255", "decision": "TECH.R256", "primary": PRIMARY,
        "sources": sources, "code_sha256": joint.sha(Path(__file__).absolute()),
        "prediction_results_observed_before_financial_registration": True,
        "hypothesis": "固定原始盈利×宏观×技术共同评分按原首次边界执行，能否提高完整扣费净收益和Sharpe。",
        "one_candidate_and_controls": list(joint.POLICIES), "candidate_configurations": 1,
        "new_candidate_accounts": 4, "new_matched_control_accounts": 12, "reused_A_accounts": 4,
        "new_fits": 0, "new_labels": 0, "parameter_grids": 0, "new_market_requests": 0,
        "scope": ["510300.SH", "CASH_CNY"], "initial_capital": 200000,
        "periods": original.PERIODS, "costs": {"BASE": [.0002, .0005], "STRESS": [.0004, .001]},
        "execution": "原PASSAGE_ONLY独立完整账户、次开盘、100份、0.001刻度、至少5元费用、T+1；不拼接CORE。",
        "entry": "原模型估计净pB>1且净期望>0；未知保持无新进入。原共同风险预算、50%初始请求、账户回撤10%停。",
        "exit": "原上2ATR/下1ATR日收盘边界和20收盘到期，下一真实开盘退出，不再看新评分改退出。",
        "gate": "两原时期两费用：净CAGR与净Sharpe均高于0、原A和全部同池去组对照，净pB>1、标准期望>0、DD<=10%。",
        "uncertainty": "经济门未通过即拒绝此固定配置；不以误差较优对照替代主候选。若通过仍必须新独立验证，历史点估计不完成目标。",
        "old_results_preserved": True, "evidence_role": "DEVELOPMENT_CALIBRATION_NOT_INDEPENDENT",
        "goal_achieved": False, "orders_authorized": False,
    }, exclusive=True)
    print("已登记唯一联合评分的16个完整账户，保留四原A与全部同池对照。")


def run():
    p = joint.read(OUT / "protocol.json")
    joint.require(p["code_sha256"] == joint.sha(Path(__file__).absolute()), "金融登记后代码改变。")
    for s in p["sources"]:
        joint.require(joint.sha(ROOT / s["path"]) == s["sha256"], "金融冻结来源改变：" + s["path"])
    joint.write(OUT / "run_started.json", {"at": joint.now(), "new_fits": 0, "planned_new_accounts": 16}, exclusive=True)
    try:
        data = pd.read_parquet(DATA)
        prediction = pd.read_parquet(PREDICTIONS)
        raw, dividends, risks, parents = original.load()
        for col in ["date", "open", "high", "low", "close", "ac", "atr14"]:
            actual = data[col].to_numpy(dtype="datetime64[ns]" if col == "date" else float)
            saved = raw[col].to_numpy(dtype="datetime64[ns]" if col == "date" else float)
            joint.require(np.array_equal(actual, saved, equal_nan=col != "date"), "原账户价格或日期不同：" + col)
        rows, checks, years, points, contrasts = [], [], [], [], []
        for period, (start, end) in original.PERIODS.items():
            local = data[data.date.le(end)].reset_index(drop=True)
            for cost in original.COSTS:
                saved = baseline(period, cost)
                base_stat = metrics_with_limits(saved)
                rows.append({"period": period, "cost": cost, "policy": "A_SAVED_WEIGHT", **base_stat})
                stats = {}
                for policy in joint.POLICIES:
                    selected = signals(data, prediction, policy).iloc[:len(local)].reset_index(drop=True)
                    result = account(local, dividends, parents[period], risks, selected, cost, start, "PASSAGE_ONLY")
                    joint.require(pd.DatetimeIndex(result["daily"].date).equals(pd.DatetimeIndex(saved["daily"].date)), "完整账户对照日历不一致。")
                    checks.append({"period": period, "cost": cost, "policy": policy, **verify_account(result)})
                    stat = metrics_with_limits(result)
                    yearly = annual_rows(result, period, policy, cost)
                    full_counts = [z["completed_cycles"] for z in yearly if z["full_year"]]
                    stat.update(average_full_year_cycles=float(np.mean(full_counts)),
                                zero_trade_full_years=sum(c == 0 for c in full_counts))
                    stats[policy] = stat
                    rows.append({"period": period, "cost": cost, "policy": policy, **stat})
                    years.extend(yearly)
                    for name in ["daily", "orders", "trades", "decisions", "rejections"]:
                        table(f"accounts/{period}/{cost}/{policy}/{name}", result[name])
                    joint.write(OUT / f"results/accounts/{period}/{cost}/{policy}/terminal.json", result["terminal"])
                    for trade in result["trades"].to_dict("records"):
                        origin = int(np.flatnonzero(pd.to_datetime(data.date).eq(trade["entry_origin"]))[0])
                        score = selected.iloc[origin]
                        points.append({"period": period, "cost": cost, "policy": policy, **trade,
                            **{k: score.get(k) for k in ["score", "predicted_p_times_b", "predicted_net_expectation", "earnings_fields_on_path", "macro_fields_on_path"]}})
                    print(f"{period}/{cost}/{policy}：净年化{stat['net_cagr']:.3%}，净夏普{stat['net_sharpe']:.4f}，完成{stat['completed_cycles']}。", flush=True)
                controls = [base_stat] + [stats[k] for k in joint.POLICIES if k != PRIMARY]
                main = stats[PRIMARY]
                contrasts.append({"period": period, "cost": cost, "economic_gate_passed": gate(main, controls),
                    "joint_minus_A_cagr": main["net_cagr"] - base_stat["net_cagr"],
                    "joint_minus_A_sharpe": main["net_sharpe"] - base_stat["net_sharpe"],
                    "all_controls": [{"policy": k, "cagr_delta": main["net_cagr"] - stats[k]["net_cagr"],
                                       "sharpe_delta": main["net_sharpe"] - stats[k]["net_sharpe"]}
                                      for k in joint.POLICIES if k != PRIMARY]})
        table("全部20账户_四场景同口径比较", pd.DataFrame(rows))
        table("16新账户_资金库存与时钟", pd.DataFrame(checks))
        table("全部逐年收益与完成次数", pd.DataFrame(years))
        table("全部实际点位_原始盈利与宏观路径", pd.DataFrame(points))
        passed = all(c["economic_gate_passed"] for c in contrasts)
        summary = {"study_id": p["study_id"], "completed_at": joint.now(), "decision": "TECH.R256",
            "status": "HISTORICAL_JOINT_INCREMENT_REQUIRES_INDEPENDENT_VALIDATION" if passed else "REJECTED_FIXED_MACRO_EARNINGS_JOINT_FULL_ACCOUNT_TARGETS_NOT_MET",
            "new_candidate_configurations": 1, "new_candidate_accounts": 4, "new_matched_control_accounts": 12,
            "reused_A_accounts": 4, "new_fits": 0, "new_labels": 0, "new_market_requests": 0,
            "all_four_economic_gates_passed": passed, "contrasts": contrasts,
            "primary_four_scene_metrics": [r for r in rows if r["policy"] == PRIMARY],
            "necessary_account_checks": checks, "new_independent_completed_points": 0,
            "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False,
            "goal_achieved": False, "orders_authorized": False}
        joint.write(OUT / "summary.json", summary, exclusive=True)
        report = ["# 固定原始盈利、宏观与量价联合评分的完整账户结果", "",
                  "唯一主候选和三个共同池去组对照一次16新账户，四原A保存账户同资金、同日历、同成本比较；没有重新拟合、改变评分或选择更好对照替代主候选。",
                  "全账户只持有510300或现金，原两时期完整保留；早期来源不足不缩掉。旧金融与评分拒绝保留。", "",
                  "| 时期 | 成本 | 模型 | 净年化 | 净Sharpe | 回撤 | 净pB | 完成周期 |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            report.append(f"| {r['period']} | {r['cost']} | {r['policy']} | {r['net_cagr']:.3%} | {r['net_sharpe']:.4f} | {r['max_drawdown']:.3%} | {r['p_times_b']:.4f} | {r['completed_cycles']} |")
        report += ["", f"四场景全部经济门通过：{passed}；独立验证未建立，完整目标未完成。",
                   "逐年、完整和开放周期、真实订单、全部来源不足、财富恒等式及费用保存。峰值风险、最坏日/交易、赢家集中度、敞口、周转、费用、最长空仓和开放损益列入完整指标表。",
                   "评分是开发估计；原2015—2019缺共同训练支持，不能把后期覆盖外推。净期望与实际pB分别报告；计划2ATR/1ATR不替代实际盈亏比。",
                   "结果不得据已知输赢调树深、财报覆盖、阶段、权重或窗口；任何后续须不同依据、真实新信息用途或新独立样本。"]
        (OUT / "联合评分_完整收益夏普与全部对照.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        joint.write(OUT / "run_completed.json", {"at": joint.now(), "terminal": True, "new_accounts": 16}, exclusive=True)
        print("完整16新账户和全部对照已经保存；四场景经济门：" + str(passed))
    except Exception as exc:
        joint.write(OUT / "implementation_failure.json", {"at": joint.now(), "terminal": True,
            "type": type(exc).__name__, "error": str(exc)}, exclusive=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="固定联合核心评分的完整账户测量")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()


if __name__ == "__main__":
    main()
