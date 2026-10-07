"""唯一方向确认完整点位实验：先测执行器，再冻结，再一次比较完整账户。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import upward_episode_anatomy_v1 as common
from research import point_directional_confirmation_account_v1 as execution
from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.point_account_nr7_inputs_v1 import PARENT_A, metrics, trade_statistics
from research.point_account_nr7_complement_v1 import verify_account, annual_rows
from research.adaptive_allocation_v1 import normalize_dividends

OUT = ROOT / "reports/research/510300_point_directional_confirmation_study_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
WEIGHT = ROOT / "reports/research/510300_point_weight_information_diagnostic_v1"
CONTROLS = ROOT / "reports/research/510300_point_second_weight_comparison_v1/inputs/controls"
PERIODS = {"2015_2019": ("2015-01-05", "2019-12-31"), "2020_2026": ("2020-01-02", "2026-09-30")}
COSTS = ("BASE", "STRESS")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig", lineterminator="\n")


def load():
    prices = pd.read_parquet(CURRENT / "inputs/candidate_prices.parquet")
    div = normalize_dividends(pd.read_csv(WEIGHT / "inputs/dividends.csv"))
    data, _ = common.features(prices, div)
    risks = pd.read_parquet(WEIGHT / "inputs/risks.parquet")
    require(len(data) == 3488 and data.symbol.eq("510300.SH").all(), "原3488日510300范围改变。")
    require((risks.latest_label_exit_idx <= risks.idx).all(), "复用风险估计存在未成熟标签。")
    require(np.array_equal(data.date.iloc[risks.idx.to_numpy(int)].to_numpy(), risks.date.to_numpy()),
            "复用风险原点与日线错位。")
    recent = pd.read_parquet(WEIGHT / "inputs/parent_signals.parquet")
    recent = recent.pivot(index="origin", columns="candidate", values="target").reset_index()
    early = pd.read_parquet(WEIGHT / "inputs/earlier_signals.parquet")
    early = early.loc[early.period.eq("earlier_diagnostic")].pivot(index="origin", columns="model", values="target").reset_index()
    return data, div, risks, {"2015_2019": early, "2020_2026": recent}


def source_paths():
    paths = [ROOT / "research/point_directional_confirmation_account_v1.py",
             ROOT / "research/point_directional_confirmation_study_v1.py",
             ROOT / "tests/test_point_directional_confirmation_account_v1.py", OUT / "tests_receipt.json",
             ROOT / "research/point_account_nr7_inputs_v1.py", ROOT / "research/daily_supply_test_v1.py",
             ROOT / "research/upward_episode_anatomy_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/point_account_nr7_complement_v1.py", ROOT / "research/point_account_cashflow_state_v1.py",
             CURRENT / "inputs/candidate_prices.parquet", WEIGHT / "inputs/dividends.csv",
             WEIGHT / "inputs/risks.parquet", WEIGHT / "inputs/parent_signals.parquet", WEIGHT / "inputs/earlier_signals.parquet",
             OUT / "prior_and_purpose_review.json"]
    for period in PERIODS:
        for cost in COSTS:
            for name in ["daily.parquet", "trades.parquet", "orders.parquet"]:
                paths.append(CONTROLS / period / cost / "A_SAVED_WEIGHT" / name)
    return paths


def preflight():
    require(not (OUT / "protocol.json").exists() and not (OUT / "control_preflight.json").exists(),
            "执行器对照已核验或金融实验已登记，不覆盖。")
    tests = read(OUT / "tests_receipt.json")
    require(tests["passed"] and tests["tests"] == 6
            and tests["module_sha256"] == digest(ROOT / "research/point_directional_confirmation_account_v1.py")
            and tests["test_sha256"] == digest(ROOT / "tests/test_point_directional_confirmation_account_v1.py"),
            "必要执行测试版本不符。")
    data, div, risks, parents = load()
    rows = []
    for period, (start, end) in PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        empty = pd.DataFrame({"date": local.date, "entry_event": False, "event_id": None,
                              "stop_index": np.nan, "target_index": np.nan})
        for cost in COSTS:
            actual = execution.account(local, div, parents[period], risks, empty, cost, start, "A_CONTROL")
            check = verify_account(actual)
            saved = pd.read_parquet(CONTROLS / period / cost / "A_SAVED_WEIGHT/daily.parquet")
            pd.testing.assert_frame_equal(actual["daily"], saved, check_exact=True)
            saved_orders = pd.read_parquet(CONTROLS / period / cost / "A_SAVED_WEIGHT/orders.parquet")
            pd.testing.assert_frame_equal(actual["orders"][saved_orders.columns], saved_orders, check_exact=True)
            rows.append({"period": period, "cost": cost, "saved_daily_and_orders_exactly_equal": True, **check})
    write_json(OUT / "control_preflight.json", {"at": now(), "status": "PASS_ORIGINAL_A_FOUR_ACCOUNTS_EXACT",
               "actual_control_replays": 4, "new_candidate_accounts": 0, "checks": rows}, exclusive=True)
    print("原A四个完整账户的逐日金额、份额和订单精确一致；候选账户尚未运行。")


def freeze():
    require(not (OUT / "protocol.json").exists(), "方向确认实验已冻结，不覆盖。")
    control = read(OUT / "control_preflight.json")
    require(control["passed"], "原A执行对照未通过。")
    protocol = {
        "study": "510300_POINT_DIRECTIONAL_CONFIRMATION_FULL_ACCOUNT_V1", "at": now(),
        "registration_decision": "TECH.R153", "financial_decision": "TECH.R154",
        "question": "原5%上涨确认之后到反向确认之间是否仍有可执行净收益，能否补原A空仓并提高完整账户净收益与夏普？",
        "known_before_registration": "所有历史已用于开发，原49上涨图谱与九状态附加门失败已知；原A两期指标及R150失败已知。5%是原图谱已有定义，不从本轮收益选择阈值。",
        "mechanism": "价格从事前运行低点反弹5%后趋势可能延续；从运行高点回落5%时这一延续假说失效。只有当前确认日可触发，不把极值日当入场/退出日。",
        "signal": "逐日含现金分红收盘总回报指数；仅第一向上确认日为入场事件。全历史自2012-05-28顺序维护方向、运行极值和确认日期；原两时期账户从现金开始，只响应账户开始后实际发生的确认，不补入已过去事件。",
        "threshold": .05, "chosen_numeric_signal_parameters": 1, "threshold_grid": False,
        "entry": "确认收盘决定，下一开盘尝试一次；事前原风险和数量锁定，开盘仅缩减。受阻/持仓时忽略该事件，不择日重试；无计划2R、均线/MACD/量/波动过滤。",
        "exit": "新来源持仓在事前可知方向转为向下时，下一可卖开盘退出；受阻退出锁定。无固定收益目标或最长持有期，不按未来峰值退出。原风险减仓和账户回撤停止保留，新来源不加仓。",
        "combination": "原A目标强弱、10个百分点调整带及执行保持。只在空仓、原A明确为0时采用当日新向上确认；A未知不当0。各持仓按入场来源退出，不转换身份，清仓同日不再买入。",
        "distinction": "原方向变化5%/动态波动版本用于回顾标签和解释，不是这次账户策略；原三个确认族使用MACD/EMA20/20日突破、固定目标/失效/20日上限。旧供给、C04方向确认、NR7、普通趋势及原HMM/BOCPD均保留其失败。本次研究直接确认到反向确认的自然持有净现金流，不重跑或修改它们。",
        "capital": 200000, "assets": ["510300.SH", "CASH_CNY"], "long_only": True,
        "periods": PERIODS, "costs": {"BASE": [.0002, .0005], "STRESS": [.0004, .001]},
        "annual_days": 252, "cash_annual_rate": 0., "fee_lot_tick": {"minimum_fee_cny": 5, "lot": 100, "tick": .001},
        "risk": "原50%仓位上限、ES5预算2.5%、负10%跳空预算5%与半数剩余回撤余量、10%收盘回撤停止新入场；复用保存成熟风险，不重拟合。",
        "clock_and_dividends": "收盘决定、下一开盘成交，T+1；登记权益、除息应收、支付现金分开；终点保留真实持仓不人工清仓。现金日全部纳入夏普。",
        "primary": "A_PLUS_DIRECTIONAL_MINUS_A_SAVED_WEIGHT",
        "diagnostic": "DIRECTIONAL_ONLY用于解释组件，不设每个组件单独达到系统夏普或pB目标的通用门。",
        "economic_gate": "两时期两费用四场景全部：组合净CAGR和全日历净Sharpe严格高于原A且为正，回撤<=10%，实际净pB>1、标准净期望>0。频率完整逐年列为软目标。",
        "uncertainty": "固定20/252日循环配对区块，各2000次、种子510300154；同时报告两增量95%区间，双尺度双期双费用下界均正才称历史稳定性支持。点值通过但区间失败仍不晋升。不是全项目选择校正或独立证据。",
        "new_candidate_accounts_planned": 8, "original_control_replays": 4, "new_model_fits": 0, "new_training_labels": 0,
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "first_vintage_limit": "沿用原日线/现金分红历史代理；物理首版未认证，不因前缀正确称独立PIT认证。",
        "global_DSR_PBO": "NOT_COMPUTED", "overfit_removed": False, "goal_achieved": False,
        "no_rescue": "不改5%、确认时钟、入场等待、退出、目标、费用、风险、时期或组合营救；原前瞻不变。",
        "literature": "https://repository.essex.ac.uk/33750/1/s10462-022-10307-0.pdf",
        "literature_limit": "方向变化的事件时钟及确认与极值区别；论文的10分钟外汇、多阈值/优化算法与本次日线510300不同，不提供本策略收益证明。",
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in source_paths() + [OUT / "control_preflight.json"]],
    }
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    print("唯一方向确认账户规则已冻结，8候选账户尚未运行。", flush=True)


def intervals(a, b):
    result = []
    for block in [20, 252]:
        rng = np.random.default_rng(510300154)
        values = []
        for _ in range(2000):
            starts = rng.integers(0, len(a), size=(len(a) + block - 1) // block)
            ids = ((starts[:, None] + np.arange(block)) % len(a)).ravel()[:len(a)]
            x, y = a[ids], b[ids]
            sx = x.mean() / x.std(ddof=1) * np.sqrt(252) if x.std(ddof=1) > 1e-14 else np.nan
            sy = y.mean() / y.std(ddof=1) * np.sqrt(252) if y.std(ddof=1) > 1e-14 else np.nan
            cx = np.expm1(np.log1p(x).mean() * 252)
            cy = np.expm1(np.log1p(y).mean() * 252)
            values.append([cy - cx, sy - sx])
        array = np.asarray(values)
        result.append({"block": block, "resamples": 2000,
                       "cagr_delta_95": np.nanquantile(array[:, 0], [.025, .975]).tolist(),
                       "sharpe_delta_95": np.nanquantile(array[:, 1], [.025, .975]).tolist(),
                       "role": "DEVELOPMENT_DESCRIPTIVE_NOT_INDEPENDENT"})
    return result


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "本固定实验已经开始，不重复。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "冻结来源变化：" + source["path"])
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "candidate_configuration": 1}, exclusive=True)
    data, div, risks, parents = load()
    signal_rows = execution.signals(data)
    table("全部方向变化与确认事件", signal_rows)
    rows, yearly, checks, contrasts, source_trades = [], [], [], [], []
    for period, (start, end) in PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        for cost in COSTS:
            baseline = pd.read_parquet(CONTROLS / period / cost / "A_SAVED_WEIGHT/daily.parquet")
            base_stats = metrics({"daily": baseline,
                                  "trades": pd.read_parquet(CONTROLS / period / cost / "A_SAVED_WEIGHT/trades.parquet"),
                                  "orders": pd.read_parquet(CONTROLS / period / cost / "A_SAVED_WEIGHT/orders.parquet"),
                                  "terminal": {"stopped": bool(baseline.risk_stopped.iloc[-1])}})
            base_trades = pd.read_parquet(CONTROLS / period / cost / "A_SAVED_WEIGHT/trades.parquet")
            base_complete = base_trades.loc[base_trades.status.eq("COMPLETE")]
            base_counts = [int(base_complete.exit_date.dt.year.eq(year).sum())
                           for year in baseline.date.dt.year.unique() if year < 2026]
            base_stats["average_full_year_cycles"] = float(np.mean(base_counts))
            base_stats["zero_trade_full_years"] = sum(count == 0 for count in base_counts)
            rows.append({"period": period, "cost": cost, "policy": "A_SAVED_WEIGHT", **base_stats})
            results = {}
            for mode in ["DIRECTIONAL_ONLY", "A_PLUS_DIRECTIONAL"]:
                account = execution.account(local, div, parents[period], risks, signal_rows.iloc[:len(local)], cost, start, mode)
                check = verify_account(account)
                checks.append({"period": period, "cost": cost, "policy": mode, **check})
                stat = metrics(account)
                local_yearly = annual_rows(account, period, mode, cost)
                full_counts = [row["completed_cycles"] for row in local_yearly if row["full_year"]]
                stat["average_full_year_cycles"] = float(np.mean(full_counts))
                stat["zero_trade_full_years"] = sum(count == 0 for count in full_counts)
                rows.append({"period": period, "cost": cost, "policy": mode, **stat})
                yearly.extend(local_yearly)
                for name in ["daily", "trades", "orders", "decisions", "rejections"]:
                    table(f"accounts/{period}/{cost}/{mode}/{name}", account[name])
                common.save_json(OUT / f"results/accounts/{period}/{cost}/{mode}/terminal.json", account["terminal"])
                if mode == "A_PLUS_DIRECTIONAL":
                    native = account["trades"].loc[account["trades"].source.eq("DIRECTIONAL")]
                    source_trades.append({"period": period, "cost": cost, **trade_statistics(native)})
                results[mode] = (account, stat)
                print(f"{period}/{cost}/{mode}：净年化{stat['net_cagr']:.2%}，夏普{stat['net_sharpe']:.4f}，周期{stat['completed_cycles']}。", flush=True)
            combined, combined_stats = results["A_PLUS_DIRECTIONAL"]
            _, standalone_stats = results["DIRECTIONAL_ONLY"]
            require(pd.DatetimeIndex(baseline.date).equals(pd.DatetimeIndex(combined["daily"].date)), "账户比较日期不一致。")
            passed = bool(combined_stats["net_cagr"] > max(0., base_stats["net_cagr"])
                          and combined_stats["net_sharpe"] > max(0., base_stats["net_sharpe"])
                          and combined_stats["max_drawdown"] <= .1
                          and combined_stats["p_times_b"] > 1.
                          and combined_stats["standard_expectancy_loss_units"] > 0.
                          )
            contrasts.append({"period": period, "cost": cost, "economic_gate_passed": passed,
                              "cagr_delta": combined_stats["net_cagr"] - base_stats["net_cagr"],
                              "sharpe_delta": combined_stats["net_sharpe"] - base_stats["net_sharpe"],
                              "ending_equity_delta_cny": combined_stats["ending_equity"] - base_stats["ending_equity"],
                              "completed_cycle_delta": combined_stats["completed_cycles"] - base_stats["completed_cycles"],
                              "intervals": intervals(baseline.net_return.to_numpy(float), combined["daily"].net_return.to_numpy(float))})
    table("完整账户共同口径比较", pd.DataFrame(rows))
    table("逐年净收益与完整交易次数", pd.DataFrame(yearly))
    table("实际账户资金库存核对", pd.DataFrame(checks))
    table("组合内新增来源完整交易", pd.DataFrame(source_trades))
    economic_passed = all(row["economic_gate_passed"] for row in contrasts)
    stable_passed = all(item["cagr_delta_95"][0] > 0. and item["sharpe_delta_95"][0] > 0.
                        for row in contrasts for item in row["intervals"])
    passed = economic_passed and stable_passed
    result = {"at": now(), "study": protocol["study"], "technical_decision": "TECH.R154",
              "status": "HISTORICAL_ECONOMIC_INCREMENT_ONLY_NOT_INDEPENDENT" if passed else "REJECTED_FIXED_CONFIRMED_DIRECTIONAL_FULL_ACCOUNT_GATE_FAILED",
              "candidate_configurations": 1, "actual_new_candidate_accounts": 8,
              "actual_original_control_replays": 4, "new_model_fits": 0, "new_model_training_labels": 0,
              "qualified_structural_signal_events": int(signal_rows.entry_event.sum()),
              "event_status_counts": signal_rows.event_status.value_counts().to_dict(),
              "all_period_cost_economic_gates_passed": economic_passed, "historical_stability_gate_passed": stable_passed, "contrasts": contrasts,
              "new_network_requests": 0, "independent_validation": "NOT_ESTABLISHED",
              "global_DSR_PBO": "NOT_COMPUTED", "overfit_removed": False,
              "old_strategies_and_failures_preserved": True, "goal_achieved": False}
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "执行期间冻结来源变化。")
    common.save_json(OUT / "summary.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in ["contrasts", "event_status_counts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定方向确认账户：原A对照、唯一冻结及一次运行。")
    parser.add_argument("action", choices=["preflight", "freeze", "run"])
    {"preflight": preflight, "freeze": freeze, "run": run}[parser.parse_args().action]()
