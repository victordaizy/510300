"""唯一缩量回调完整点位实验：先测执行器，再冻结，再一次比较完整账户。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import upward_episode_anatomy_v1 as common
from research import point_confirmed_pullback_account_v1 as execution
from research.point_account_cashflow_state_v1 import ROOT, digest, now, require, write_json
from research.point_account_nr7_inputs_v1 import PARENT_A, metrics, trade_statistics
from research.point_account_nr7_complement_v1 import verify_account, annual_rows
from research.adaptive_allocation_v1 import normalize_dividends

OUT = ROOT / "reports/research/510300_point_confirmed_pullback_study_v1"
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
    paths = [ROOT / "research/point_confirmed_pullback_account_v1.py",
             ROOT / "research/point_confirmed_pullback_study_v1.py",
             ROOT / "tests/test_point_confirmed_pullback_account_v1.py", OUT / "tests_receipt.json",
             ROOT / "research/point_c04_information_intake_v1.py", ROOT / "research/point_account_nr7_inputs_v1.py",
             ROOT / "research/daily_supply_test_v1.py", ROOT / "research/upward_episode_anatomy_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/point_account_nr7_complement_v1.py",
             CURRENT / "inputs/candidate_prices.parquet", WEIGHT / "inputs/dividends.csv",
             WEIGHT / "inputs/risks.parquet", WEIGHT / "inputs/parent_signals.parquet", WEIGHT / "inputs/earlier_signals.parquet",
             ROOT / "reports/research/510300_daily_supply_test_v1/protocol.json",
             ROOT / "reports/research/510300_daily_supply_test_v1/result.json",
             ROOT / "reports/research/510300_point_c04_information_intake_v1/protocol.json",
             ROOT / "reports/research/510300_point_c04_optional_correction_v1/prediction_summary.json",
             ROOT / "config/510300_force_pullback_v1.json"]
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
            and tests["module_sha256"] == digest(ROOT / "research/point_confirmed_pullback_account_v1.py")
            and tests["test_sha256"] == digest(ROOT / "tests/test_point_confirmed_pullback_account_v1.py"),
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
    require(not (OUT / "protocol.json").exists(), "本固定完整点位实验已登记，不更改。")
    control = read(OUT / "control_preflight.json")
    require(control["status"] == "PASS_ORIGINAL_A_FOUR_ACCOUNTS_EXACT", "原A执行器对照尚未通过。")
    protocol = {
        "at": now(), "study": "510300_POINT_CONFIRMED_PULLBACK_FULL_ACCOUNT_V1", "technical_decision": "TECH.R149",
        "question": "已确认上涨段内，前日已知缩量回调后的首次价格转强，加事前净空间约束，能否独立提供净优势并提高原A账户收益和夏普？",
        "one_candidate": "一套完整规则，两个原时期、两成本；单独和补A两用途共8候选账户，4执行器对照只复算原策略。",
        "stage": "原C04定义原样复用：现金前向平移收盘，左右各2根严格低高确认，b+2后已知a至b上涨；当时首次处于a与b之间开e，不追溯。原阶段结束/失效和此前5日金额基准不改。",
        "signal": "同一a,b,e回调阶段在e之后首个收盘严格超过前日已知最高价，仅判断一次；前日C04均额比<1且阶段与今日相同。金额收缩须在转强前已知；转强日金额只作背景、不拿未来量填前日。",
        "geometry": "信号时e至今日已知最低报价减一刻度为失效，已确认前高b的收盘为目标，原价空间比>=2；收盘计划及实际下一开盘均以真实计划份额/报价/双边费用要求净计划比>=2。跳空破结构或挤掉空间就取消，不等更好开盘。计划比不等于实际盈亏比。",
        "exit": "入场锁定现金前向平移的失效和目标，只按收盘确认后下一可卖开盘执行；原风险减仓/回撤退出保留，候选不补仓、不按新阶段移动目标，不新增时间止损或盘中成交假设。",
        "combination": "原A逐日目标强弱/再平衡保持；只在账户空仓且原A明确为0时响应新事件，原A未知不作0。候选持仓按自身固定结构退出，不转换身份；忽略持仓期间事件，清仓同日不重新买入。",
        "distinction": "旧供给测试是跌破前20支撑后收复、10日内第二次回测、单日相对父日缩量缩幅、3日再确认及合成2R目标；旧力度回调用2/13指数平滑力度。本轮用已确认上涨段的累计回调额与事前原前高空间，不改变这些旧失败参数。C04旧九项/可选模型两合同均失败并保留；本轮不是补算失败退出模型的账户，研究单位是首次入场自然事件及完整现金账户。",
        "capital": 200000, "assets": ["510300.SH", "CASH_CNY"], "long_only": True,
        "periods": PERIODS, "costs": {"BASE": [.0002, .0005], "STRESS": [.0004, .001]},
        "fee_lot_tick": {"minimum_fee_cny": 5, "lot": 100, "tick": .001},
        "risk": "原50%仓位上限、ES5预算2.5%、负10%跳空预算5%并受半数剩余回撤预算约束，10%账户收盘回撤停止新入场。完全复用已保存成熟风险样本，不重拟合或变风险门。",
        "clock": "昨收决定、次日开盘，原风险及最大份额在昨收锁定；开盘只能削减数量和否决结构/空间。T+1，登记/除息应收/支付现金分开，末端不人工清仓。",
        "primary_economic_gate": "两时期两成本全部：A加候选净CAGR和全日历净Sharpe严格高于原A且为正，回撤<=10%，实际净pB>1且标准净期望>0；候选单独对应账户实际净pB>1及周期均值>0。任一失败结束本固定规则，不挑时期拼接。",
        "frequency": "软目标，完整年度次数/零交易年/最长空仓全列；事件、分笔订单与费用档不计独立完整交易。",
        "uncertainty": "配对日收益固定20和252日循环区块、各2000次、种子510300150，报告95%净CAGR/Sharpe增量描述区间；不是全项目多重选择修正或独立验证。",
        "parameters_searched": False, "new_model_fits": 0, "new_model_training_labels": 0,
        "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "global_DSR_PBO": "NOT_COMPUTED", "overfit_removed": False, "goal_achieved": False,
        "no_rescue": "不换确认根数/阶段/金额基准/2R/止损目标/退出/入场先后/成本/时期/组合方式或未知处理营救；旧冻结与前瞻不变。",
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)}
                    for p in source_paths() + [OUT / "control_preflight.json"]],
    }
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    print("唯一完整技术点位规则已冻结；8候选账户尚未运行。")


def intervals(a, b):
    result = []
    for block in [20, 252]:
        rng = np.random.default_rng(510300150)
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
    table("全部回调阶段与首次转强事件", signal_rows)
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
            for mode in ["PULLBACK_ONLY", "A_PLUS_PULLBACK"]:
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
                if mode == "A_PLUS_PULLBACK":
                    native = account["trades"].loc[account["trades"].source.eq("PULLBACK")]
                    source_trades.append({"period": period, "cost": cost, **trade_statistics(native)})
                results[mode] = (account, stat)
                print(f"{period}/{cost}/{mode}：净年化{stat['net_cagr']:.2%}，夏普{stat['net_sharpe']:.4f}，周期{stat['completed_cycles']}。", flush=True)
            combined, combined_stats = results["A_PLUS_PULLBACK"]
            _, standalone_stats = results["PULLBACK_ONLY"]
            require(pd.DatetimeIndex(baseline.date).equals(pd.DatetimeIndex(combined["daily"].date)), "账户比较日期不一致。")
            passed = bool(combined_stats["net_cagr"] > max(0., base_stats["net_cagr"])
                          and combined_stats["net_sharpe"] > max(0., base_stats["net_sharpe"])
                          and combined_stats["max_drawdown"] <= .1
                          and combined_stats["p_times_b"] > 1.
                          and combined_stats["standard_expectancy_loss_units"] > 0.
                          and standalone_stats["p_times_b"] > 1.
                          and standalone_stats["mean_cycle_net_return"] > 0.)
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
    passed = all(row["economic_gate_passed"] for row in contrasts)
    result = {"at": now(), "study": protocol["study"], "technical_decision": "TECH.R150",
              "status": "HISTORICAL_ECONOMIC_INCREMENT_ONLY_NOT_INDEPENDENT" if passed else "REJECTED_FIXED_CONFIRMED_PULLBACK_FULL_ACCOUNT_GATE_FAILED",
              "candidate_configurations": 1, "actual_new_candidate_accounts": 8,
              "actual_original_control_replays": 4, "new_model_fits": 0, "new_model_training_labels": 0,
              "qualified_structural_signal_events": int(signal_rows.entry_event.sum()),
              "event_status_counts": signal_rows.event_status.value_counts().to_dict(),
              "all_period_cost_economic_gates_passed": passed, "contrasts": contrasts,
              "new_network_requests": 0, "independent_validation": "NOT_ESTABLISHED",
              "global_DSR_PBO": "NOT_COMPUTED", "overfit_removed": False,
              "old_strategies_and_failures_preserved": True, "goal_achieved": False}
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "执行期间冻结来源变化。")
    common.save_json(OUT / "summary.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in ["contrasts", "event_status_counts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定完整技术点位实验：原对照、唯一冻结及一次运行。")
    parser.add_argument("action", choices=["preflight", "freeze", "run"])
    {"preflight": preflight, "freeze": freeze, "run": run}[parser.parse_args().action]()
