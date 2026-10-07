"""一次固定的券商阶段政策实验，比较真实现金账户及进场/退出归因对照。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research import broker_stage_policy_inputs_v1 as stage
from research import point_first_passage_study_v1 as original
from research import point_first_passage_account_v1 as original_account
from research import point_account_nr7_inputs_v1 as measurements
from research import daily_supply_test_v1 as budgets
from research.point_account_nr7_complement_v1 import verify_account, annual_rows
from research.point_directional_confirmation_study_v1 import intervals

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_broker_stage_policy_v1"
SOURCE = ROOT / "reports/research/510300_cnh_macro_first_passage_v1/results/全部3488当时已知技术宏观与定盘偏离_未知保留.parquet"
PERIODS = original.PERIODS
COSTS = original.COSTS
REGISTRATION, RESULT = "TECH.R211", "TECH.R212"


def read(path):
    return original.read(path)


def write(path, value):
    original.write_json(path, value, exclusive=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def load():
    base, dividends, risks, parents = original.load()
    known = pd.read_parquet(SOURCE)
    if len(known) != 3488 or not known.symbol.eq("510300.SH").all():
        raise ValueError("原输入对象或数量改变。")
    for column in ("open", "high", "low", "close", "volume", "dividend", "cash_shift", "atr14", "daily_hist_atr", "weekly_hist_atr"):
        np.testing.assert_array_equal(base[column].to_numpy(), known[column].to_numpy())
    if not np.array_equal(base.date.astype("datetime64[ns]").to_numpy(), known.date.astype("datetime64[ns]").to_numpy()):
        raise ValueError("原日线、已知背景和风险日历错位。")
    observed = stage.observations(known)
    observed["date"] = observed.date.astype(base.date.dtype)
    return base, observed, dividends, risks, parents


def source_paths():
    paths = [Path(__file__), Path(stage.__file__), Path(original.__file__), Path(original_account.__file__),
             Path(measurements.__file__), Path(budgets.__file__), ROOT / "research/point_account_nr7_complement_v1.py",
             ROOT / "research/point_directional_confirmation_study_v1.py", ROOT / "tests/test_broker_stage_policy_v1.py",
             ROOT / "docs/510300_BROKER_STAGE_POLICY_V1.md", SOURCE,
             original.CURRENT / "inputs/candidate_prices.parquet", original.WEIGHT / "inputs/dividends.csv",
             original.WEIGHT / "inputs/risks.parquet", original.WEIGHT / "inputs/parent_signals.parquet",
             original.WEIGHT / "inputs/earlier_signals.parquet",
             ROOT / "reports/research/510300_broker_cycle_inspiration_v1/phase_observation_summary.json"]
    for period in PERIODS:
        for cost in COSTS:
            paths.extend(original.CONTROL / period / cost / "A_SAVED_WEIGHT" / f"{name}.parquet" for name in ("daily", "orders", "trades"))
    return paths


def preflight():
    if (OUT / "control_preflight.json").exists() or (OUT / "protocol.json").exists():
        raise RuntimeError("本用途原控制已核验或已登记，不重复。")
    tests = read(OUT / "tests_receipt.json")
    if tests["passed"] != 9 or tests["exit_code"] != 0 or tests["inputs_sha256"] != digest(Path(stage.__file__)):
        raise ValueError("九项必要测试或版本不满足登记条件。")
    base, observed, dividends, risks, parents = load()
    checks = []
    for period, (start, end) in PERIODS.items():
        local = base.loc[base.date.le(end)].reset_index(drop=True)
        empty = pd.DataFrame({"date": local.date, "entry_event": False, "event_id": None, "atr": np.nan, "stop_index": np.nan, "target_index": np.nan})
        for cost in COSTS:
            result = original_account.account(local, dividends, parents[period], risks, empty, cost, start, "A_CONTROL")
            saved = original.CONTROL / period / cost / "A_SAVED_WEIGHT"
            pd.testing.assert_frame_equal(result["daily"], pd.read_parquet(saved / "daily.parquet"), check_exact=True)
            old_orders = pd.read_parquet(saved / "orders.parquet")
            pd.testing.assert_frame_equal(result["orders"][old_orders.columns], old_orders, check_exact=True)
            old_trades = pd.read_parquet(saved / "trades.parquet")
            pd.testing.assert_frame_equal(result["trades"][old_trades.columns], old_trades, check_exact=True)
            checks.append({"period": period, "cost": cost, "original_daily_orders_trades_exact": True, **verify_account(result)})
    selected = ["previous_week_low", "support_week_last_date", "stage_entry_type", "raw_entry_type", "trend_confirmation", "funding_distribution"]
    for date in ["2015-06-29", "2019-01-08", "2020-04-01", "2024-09-24"]:
        count = int(np.flatnonzero(base.date.eq(pd.Timestamp(date)))[0])+1
        cut = stage.observations(pd.read_parquet(SOURCE).iloc[:count])
        pd.testing.assert_frame_equal(cut[selected], observed.iloc[:count][selected], check_exact=True)
    write(OUT / "control_preflight.json", {"at": original.now(), "original_A_replays": 4, "checks": checks,
                                          "actual_original_prefix_checks": 4, "new_policy_accounts": 0, "new_fits": 0})
    print("原A四账户逐日、订单和周期精确复现；四历史关键点截断精确保持；尚无阶段账户结果。", flush=True)


def freeze():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("阶段政策已固定，不覆盖。")
    pre = read(OUT / "control_preflight.json")
    if pre["original_A_replays"] != 4:
        raise ValueError("原对照未核验。")
    protocol = {
        "study": "510300_BROKER_STAGE_POLICY_V1", "at": original.now(), "registration_decision": REGISTRATION, "result_decision": RESULT,
        "question": "宏观角色与价格阶段共同定义进入、确认和失效，能否较固定退出及未分阶段进入提高完整收益和夏普？",
        "known_before_freeze": "全部历史已用于研究，R208拒绝；R210四原案例与全部状态描述已知。不是新样本或独立验证，也不声称首次做自适应。",
        "policies": list(stage.POLICIES), "primary": stage.POLICIES[0], "parameter_grid": False,
        "entry": "三路线均只在完整资格条件由假转真时尝试次开；同日优先重新定价、回踩、修复。修复=收盘>EMA20且日柱>0、上周柱<=0、RV20/RV60<=1，且已知资金差<=0并五日下降，或PMI订单连续月改善。重新定价=严格突破前20收盘高、当日量/前20中位>=1.5、日柱>0，不要求慢宏观同时确认。回踩=昨日<=昨日EMA20、今日>EMA20、日柱回升，且上周柱>0、已知融资五日>=0。",
        "unknown": "宏观资格未知则相关路线NO_VIEW，但完全可知的重新定价可独立观察；不填零，PMI绝对水平不作为修复硬门。未知不等于宏观看空。",
        "support": "实际进入前已知上一完整自然周最低现金平移价；次开已在该位或更低取消本次，三政策共用。",
        "phase_holding": "真实成交后EARLY。收盘<=入场结构位退出；上周柱>0、收盘>EMA20、已知融资五日>=0后一次晋为TREND，不倒改进入身份；TREND结构位取原位与每个已知上一周低的最大值，只升不降。跌破结构位或上周柱<=0退出；EARLY日柱<0且收盘<EMA20退出。资金差>0并五日上升、融资五日<0且收盘<EMA20时两阶段均退出。没有固定止盈或持有期限。",
        "fixed_exit_control": "完全相同阶段资格进入与开盘结构取消，成交经济开盘下1ATR/上2ATR/20收盘到期，ATR取决定日；只改持有退出条件。机会因资金占用会不同，不声称每笔配对。",
        "unconditioned_entry_control": "去掉修复的周线/波动/宏观资格、去掉回踩的周线/融资资格；原价格与成交事件仍保留，三种原条件各自上升沿；持有退出、风险和费用与主政策完全相同。不是把三条账户收益拼起来。",
        "account": "每时期20万元，510300.SH/CASH_CNY；不补买，只允许风险减仓；最大50%、成熟ES5预算2.5%、10%跳空预算5%和半数剩余回撤余量；账户DD10%触发次可卖开盘清仓且永久停止新买入。下一开盘只减原计划可买数量，涨跌停、T+1、100份、.001报价、股息登记/应收/到账、252年化及全部现金日、终点自然持仓保持。",
        "costs": {"BASE": [.0002,.0005], "STRESS": [.0004,.001], "minimum_commission": 5}, "periods": PERIODS,
        "economic_gate": "主政策四场景均CAGR、完整Sharpe为正并严格高于原A及两归因对照，DD<=10%、实际完成pB>1且标准亏损单位期望>0；次数是软目标，零亏损时B不定义，不用预计pB或计划2R替代。",
        "stability": "原20/252日循环块各2000、种子510300154；相对A及两对照的CAGR/Sharpe增量95%下界都>0才支持历史稳定；全部区间报告，历史多次选择未校正。",
        "new_policy_accounts": 12, "original_A_replays": 4, "new_models": 0, "new_labels": 0,
        "necessary_tests": 9, "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "first_vintage": "原市场和宏观首版未认证；保存钟不等于首版认证。",
        "scope_of_novelty": "第9轮在线机制专家、第39/40轮趋势与日内强弱路由、原周日技术和量先价后均已研究。本次为宏观改善/融资确认分工、技术重新定价独立路线、真实持仓晋级及上一完整周结构失效的完整共同政策；有限查重不证明穷尽所有旧规则。",
        "no_rescue": "本配置一次运行；不按结果改资格、符号、尺度、止盈、持有、资金风险、费用或时期；不同用途另立，旧失败和原13项前瞻保持。",
        "sources": [{"path": str(path.absolute().relative_to(ROOT)), "sha256": digest(path)} for path in source_paths()+[OUT/"tests_receipt.json", OUT/"control_preflight.json"]],
    }
    write(OUT / "protocol.json", protocol)
    print("唯一阶段政策及两归因对照已固定；计划12新账户，0新模型，尚无候选收益。", flush=True)


def extra_metrics(result, yearly):
    daily, trades = result["daily"], result["trades"]
    complete = trades.loc[trades.status.eq("COMPLETE")]
    counts = [row["completed_cycles"] for row in yearly if row["full_year"]]
    profits = complete.loc[complete.net_pnl.gt(0), "net_pnl"]
    empty = daily.shares.eq(0)
    flat_runs = empty.groupby(empty.ne(empty.shift()).cumsum()).agg(["first", "size"])
    return {"average_full_year_cycles": float(np.mean(counts)), "zero_trade_full_years": sum(count == 0 for count in counts),
            "worst_day": float(daily.net_return.min()), "worst_trade": float(complete.net_return.min()) if len(complete) else np.nan,
            "largest_winner_fraction_of_all_wins": float(profits.max()/profits.sum()) if len(profits) else np.nan,
            "longest_flat_sessions": int(flat_runs.loc[flat_runs["first"], "size"].max()) if empty.any() else 0,
            "gross_turnover_over_initial_capital": float((result["orders"].quantity*result["orders"].raw_open).sum()/200000) if len(result["orders"]) else 0.,
            "open_pnl_cny": result["terminal"].get("open_pnl_cny"), "unpaid_dividend_cny": result["terminal"]["unpaid_dividend_cny"]}


def run():
    if (OUT / "RUN_STARTED.json").exists():
        raise RuntimeError("本用途已开始，不重复候选账户。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        if digest(ROOT/source["path"]) != source["sha256"]:
            raise ValueError("固定文件改变："+source["path"])
    write(OUT / "RUN_STARTED.json", {"at": original.now(), "new_policy_accounts_planned": 12, "new_fits": 0})
    base, observed, dividends, risks, parents = load()
    table("全部3488事前阶段资格与上一完整周结构位", observed)
    accounts, records, annual, checks = {}, [], [], []
    for period, (start, end) in PERIODS.items():
        local = observed.loc[observed.date.le(end)].reset_index(drop=True)
        for cost in COSTS:
            saved = original.CONTROL/period/cost/"A_SAVED_WEIGHT"
            daily = pd.read_parquet(saved/"daily.parquet")
            result = {"daily": daily, "orders": pd.read_parquet(saved/"orders.parquet"), "trades": pd.read_parquet(saved/"trades.parquet"),
                      "terminal": {"stopped": bool(daily.risk_stopped.iloc[-1]), "unpaid_dividend_cny": float(daily.receivable.iloc[-1])}}
            yearly = annual_rows(result, period, "A_SAVED_WEIGHT", cost)
            records.append({"period": period, "cost": cost, "policy": "A_SAVED_WEIGHT", **measurements.metrics(result), **extra_metrics(result, yearly)})
            annual.extend(yearly)
            accounts[(period,cost,"A_SAVED_WEIGHT")] = result
            for policy in stage.POLICIES:
                result = stage.account(local, dividends, risks, policy, cost, start)
                checks.append({"period": period, "cost": cost, "policy": policy, **verify_account(result)})
                accounts[(period,cost,policy)] = result
                folder = OUT/"accounts"/period/cost/policy
                folder.mkdir(parents=True, exist_ok=True)
                for name in ("daily", "orders", "trades", "decisions", "rejections"):
                    result[name].to_parquet(folder/f"{name}.parquet", index=False)
                write(folder/"terminal.json", result["terminal"])
                yearly = annual_rows(result, period, policy, cost)
                stats = {**measurements.metrics(result), **extra_metrics(result, yearly)}
                records.append({"period": period, "cost": cost, "policy": policy, **stats})
                annual.extend(yearly)
                print(f"{period}/{cost}/{stage.NAMES[policy]}：年化{stats['net_cagr']:.4%}，夏普{stats['net_sharpe']:.4f}，完成{stats['completed_cycles']}，pB{stats['p_times_b']:.4f}。", flush=True)
    frame = pd.DataFrame(records)
    table("十六完整账户_同资金风险成本比较", frame)
    table("全部逐年实际完成周期与账户收益", pd.DataFrame(annual))
    table("十二新账户逐日与周期复算", pd.DataFrame(checks))
    comparisons, gates = [], []
    for period in PERIODS:
        for cost in COSTS:
            group = frame.loc[frame.period.eq(period)&frame.cost.eq(cost)].set_index("policy")
            primary = group.loc[stage.POLICIES[0]]
            economic = bool(primary.net_cagr>0 and primary.net_sharpe>0 and primary.max_drawdown<=.1 and primary.p_times_b>1 and primary.standard_expectancy_loss_units>0)
            for control in ("A_SAVED_WEIGHT", stage.POLICIES[1], stage.POLICIES[2]):
                reference = group.loc[control]
                if not (primary.net_cagr>reference.net_cagr and primary.net_sharpe>reference.net_sharpe):
                    economic = False
                a = accounts[(period,cost,control)]["daily"].net_return.to_numpy(float)
                b = accounts[(period,cost,stage.POLICIES[0])]["daily"].net_return.to_numpy(float)
                comparisons.append({"period": period, "cost": cost, "control": control,
                                    "cagr_delta": primary.net_cagr-reference.net_cagr, "sharpe_delta": primary.net_sharpe-reference.net_sharpe,
                                    "intervals": intervals(a,b)})
            gates.append({"period": period, "cost": cost, "economic_passed": economic})
    stable = all(item["cagr_delta_95"][0]>0 and item["sharpe_delta_95"][0]>0 for comparison in comparisons for item in comparison["intervals"])
    passed = all(gate["economic_passed"] for gate in gates)
    write(OUT/"summary.json", {"at": original.now(), "decision": RESULT,
        "status": "HISTORICAL_CANDIDATE_NOT_INDEPENDENTLY_VALIDATED" if passed and stable else "REJECTED_FIXED_STAGE_POLICY_FULL_ACCOUNT_GATES_NOT_MET",
        "all_economic_gates_passed": passed, "historical_stability_passed": stable, "gates": gates,
        "metrics": frame.to_dict("records"), "comparisons": comparisons, "new_accounts": 12, "original_A_exact_replays": 4,
        "new_fits": 0, "new_labels": 0, "necessary_tests_passed": 9, "saved_account_checks": len(checks),
        "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False, "goal_achieved": False})
    print("一次完整阶段实验结束，全部四场景和两尺度区间已保存；不按结果修改该配置。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="券商阶段完整政策一次实验。")
    parser.add_argument("command", choices=("preflight", "freeze", "run"))
    args = parser.parse_args()
    {"preflight": preflight, "freeze": freeze, "run": run}[args.command]()
