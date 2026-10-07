"""完整日线缺口的唯一完整账户用途：测试、共同对照、冻结、一次测量和保存核对。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import full_daily_gap_inputs_v1 as rules
from research import full_daily_gap_account_v1 as execution
from research.point_fresh_repair_order_study_v1 import load
from research.point_account_nr7_inputs_v1 import metrics
from research.point_account_nr7_complement_v1 import verify_account, annual_rows
from research.point_first_passage_study_v1 import read, write_json, digest, now, require, intervals, CONTROL, PERIODS, COSTS

OUT = ROOT / "reports/research/510300_full_daily_gap_study_v1"
EXPLANATION = ROOT / "reports/research/510300_full_daily_gap_explanation_v1"
PRIOR = ROOT / "reports/research/510300_point_fresh_repair_order_study_v1"
PRIMARY = rules.PRIMARY
TESTS = ("tests/test_full_daily_gap_inputs_v1.py", "tests/test_full_daily_gap_account_v1.py")
LEDGERS = ("daily", "orders", "trades", "decisions", "rejections")
STATE_TABLE = "全部原点_完整日线缺口及整数现金坐标"


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def tests():
    require(not (OUT / "tests_receipt.json").exists(), "必要测试已记录，不自动重试。")
    inputs = read(EXPLANATION / "tests_receipt.json")
    require(inputs["passed"] == 4 and inputs["exit_code"] == 0, "四个完整缺口输入测试未通过。")
    for source in inputs["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "缺口输入不对应原测试。")
    completed = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "-q", TESTS[1]],
                               cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    output = completed.stdout + completed.stderr
    paths = [Path(rules.__file__), Path(execution.__file__), *(ROOT / p for p in TESTS)]
    ok = completed.returncode == 0 and "4 passed" in output
    write_json(OUT / "tests_receipt.json", {
        "at": now(), "exit_code": completed.returncode, "passed": 8 if ok else 0,
        "new_account_tests_passed": 4 if ok else 0, "previous_input_tests_passed": 4,
        "previous_input_receipt": (EXPLANATION / "tests_receipt.json").relative_to(ROOT).as_posix(),
        "input_tests_rerun": False, "output": output, "new_account_results_read": 0,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
    }, exclusive=True)
    print(output, end="", flush=True)
    require(ok, "账户必要测试未通过，保留首次输出。")


def frozen_explanation_check():
    protocol = read(EXPLANATION / "protocol.json")
    require(read(EXPLANATION / "summary.json")["status"] == "COMPLETED_ALL_FULL_DAILY_GAP_EXPLANATION_NO_FINANCIAL_RESULT",
            "完整日线缺口解释尚未实际完成。")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "解释冻结来源改变。")
    return protocol


def saved_account(period, cost, policy):
    path = CONTROL / period / cost / policy if policy == "A_SAVED_WEIGHT" else PRIOR / "results/accounts" / period / cost / policy
    result = {name: pd.read_parquet(path / f"{name}.parquet") for name in ("daily", "orders", "trades")}
    result["terminal"] = {"stopped": bool(result["daily"].risk_stopped.iloc[-1])}
    return result, path


def preflight():
    require(not (OUT / "control_preflight.json").exists() and not (OUT / "protocol.json").exists(),
            "共同对照已核验或已登记，不重复。")
    receipt = read(OUT / "tests_receipt.json")
    require(receipt["passed"] == 8 and receipt["exit_code"] == 0, "必要测试未通过。")
    for source in receipt["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "账户或输入不对应测试版本。")
    frozen_explanation_check()
    data, dividends, risks, parents = load()
    states = rules.gap_states(data)
    saved = pd.read_parquet(EXPLANATION / f"results/{STATE_TABLE}.parquet")
    pd.testing.assert_frame_equal(states, saved, check_exact=True)
    checks = []
    for period, (start, end) in PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        signals = rules.account_signals(local, states.iloc[:len(local)].copy(), "PRICE_CONFIRMATION")
        for cost in COSTS:
            for policy, mode in [("A_SAVED_WEIGHT", "A_CONTROL"), ("PRICE_CONFIRMATION", "PRICE_ONLY")]:
                actual = execution.account(local, dividends, parents[period], risks, signals, cost, start, mode)
                check = verify_account(actual)
                original, path = saved_account(period, cost, policy)
                for name in ("daily", "orders", "trades"):
                    expected = original[name]
                    pd.testing.assert_frame_equal(actual[name][expected.columns], expected, check_exact=True)
                checks.append({"period": period, "cost": cost, "policy": policy,
                               "saved_path": path.relative_to(ROOT).as_posix(), **check})
    write_json(OUT / "control_preflight.json", {
        "at": now(), "status": "PASS_ALL_STATES_AND_EIGHT_SAVED_CONTROL_ACCOUNTS",
        "all_structure_rows_identical": len(states), "original_A_replays": 4, "original_price_replays": 4,
        "new_primary_accounts": 0, "checks": checks,
    }, exclusive=True)
    print("3488缺口原值一致，四原A和四纯价格账户精确复现；新缺口账户尚未运行。", flush=True)


def freeze():
    require(not (OUT / "protocol.json").exists(), "完整日线缺口金融用途已登记。")
    require(read(OUT / "control_preflight.json")["status"] == "PASS_ALL_STATES_AND_EIGHT_SAVED_CONTROL_ACCOUNTS",
            "共同执行对照未通过。")
    explanation = frozen_explanation_check()
    paths = [Path(__file__), Path(rules.__file__), Path(execution.__file__), *(ROOT / p for p in TESTS),
             OUT / "tests_receipt.json", OUT / "control_preflight.json", EXPLANATION / "protocol.json",
             EXPLANATION / "summary.json", EXPLANATION / f"results/{STATE_TABLE}.parquet",
             ROOT / "research/point_fresh_repair_order_account_v1.py", ROOT / "research/point_fresh_repair_order_inputs_v1.py",
             ROOT / "research/point_account_nr7_complement_v1.py", ROOT / "research/point_directional_confirmation_study_v1.py",
             PRIOR / "protocol.json", PRIOR / "summary.json"]
    paths.extend(ROOT / source["path"] for source in explanation["sources"])
    for period in PERIODS:
        for cost in COSTS:
            for policy in ("A_SAVED_WEIGHT", "PRICE_CONFIRMATION"):
                _, path = saved_account(period, cost, policy)
                paths.extend(path / f"{name}.parquet" for name in ("daily", "orders", "trades"))
    paths = list(dict.fromkeys(paths))
    write_json(OUT / "protocol.json", {
        "at": now(), "study": "510300_FULL_DAILY_UP_GAP_STUDY_V1",
        "registration_decision": "TECH.R176", "result_decision": "TECH.R177",
        "hypothesis": "相邻完整日线区间分离可到达某些上涨修复初期；日低点回补和后续缺口提高下沿提供完整失效时钟，需要检验能否提高扣费后全账户收益和夏普。",
        "policy_intent_fixed_before_new_outcome_join": explanation["future_complete_policy_intent_fixed_before_any_new_outcome_join"],
        "complete_signal": explanation["definition"], "numeric_clock": explanation["numeric_clock"],
        "entry": "空仓完整向上日线缺口出生，次真实开一次；次开整数现金平移价<=上一完整日high或坐标未知取消不追入。原现金和风险只缩减昨晚计划份额。",
        "exit": "初始下沿为上一完整日high；持有中新的完整向上缺口只取max(原下沿,新缺口上一日high)。已知日low<=当前下沿即回补，次合法开退出，收盘在线上也不撤销回补。未知自身不退出、原风险只减；无加仓/混A/2R/20日/期末清仓。",
        "old_contract_review": explanation["old_review"], "purpose_distinction": explanation["purpose_distinction"],
        "admission_reason": "R175全3488状态、全97向上缺口及回补、原61/49段和四例先解释；完整用途在旧结果标签连接前固定，四输入/四账户测试和八共同控制通过。只准入一次开发测量。",
        "primary": PRIMARY, "controls": ["A_SAVED_WEIGHT", "PRICE_CONFIRMATION"],
        "account": "原20万元/252日/现金收益0、510300.SH/CASH_CNY、50%上限、ES5预算2.5%、负10%跳空预算min(5%NAV,半剩余DD余量)、DD10%停止新买，T+1/100份/.001刻度/最低佣金5元；原登记、除息应收及付款股息。",
        "periods": PERIODS, "costs": {"BASE": [.0002, .0005], "STRESS": [.0004, .001]},
        "primary_gate": "全部两时期两费用净CAGR和全日历净Sharpe同时高于0、原A和纯价格，DD<=10%，实际完成周期pB>1且pB-q>0；未知不填0/无穷，次数软目标。",
        "uncertainty": "固定20/252日配对循环区块，各2000、种子510300154；两尺度两比较全报告，95%差值下界均>0才过历史稳定门，仍属开发描述。",
        "purpose_gate": "完整技术规则用途，不应用原退出MSE/115成员门；原退出R145、预测R158及最新旧金融R173拒绝保持，不称原模型修复。",
        "candidate_configurations": 1, "new_primary_accounts_planned": 4, "saved_control_accounts_replayed": 8,
        "necessary_tests": 8, "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "source_first_vintage": "NOT_CERTIFIED", "historical_sample_role": "DEVELOPMENT_CALIBRATION",
        "selection_bias": "全部历史已使用，机制由已知案例启发；固定用途不创造独立样本，有限旧用途核对不声称穷尽项目。",
        "no_rescue": "原R173及所有旧终态保持；本规则不按结果改缺口大小/宽度/回补/严格越过/费用/预算/时期，不加MACD、量或RV过滤，不混A；下缺口仅观察。",
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED", "overfitting_removed": False,
        "goal_achieved": False, "orders_authorized": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
    }, exclusive=True)
    print("TECH.R176登记唯一完整日线缺口金融用途；四新账户尚未运行。", flush=True)


def computed_stats(result, period, policy, cost):
    stats = metrics(result)
    years = annual_rows(result, period, policy, cost)
    counts = [row["completed_cycles"] for row in years if row["full_year"]]
    stats.update(average_full_year_cycles=float(np.mean(counts)), zero_trade_full_years=sum(n == 0 for n in counts))
    return stats, years


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "固定用途已经开始，不重启。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "登记来源改变。")
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_primary_accounts_planned": 4}, exclusive=True)
    data, dividends, risks, parents = load()
    states = rules.gap_states(data)
    table("全部已知完整缺口_回补失效由持仓决定", rules.account_signals(data, states, PRIMARY))
    rows, years, checks, comparisons = [], [], [], []
    for period, (start, end) in PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        signals = rules.account_signals(local, states.iloc[:len(local)].copy(), PRIMARY)
        for cost in COSTS:
            primary = execution.account(local, dividends, parents[period], risks, signals, cost, start, "GAP_ONLY")
            checks.append({"period": period, "cost": cost, "policy": PRIMARY, **verify_account(primary)})
            stat, yearly = computed_stats(primary, period, PRIMARY, cost)
            rows.append({"period": period, "cost": cost, "policy": PRIMARY, **stat})
            years.extend(yearly)
            for name in LEDGERS:
                table(f"accounts/{period}/{cost}/{PRIMARY}/{name}", primary[name])
            write_json(OUT / f"results/accounts/{period}/{cost}/{PRIMARY}/terminal.json", primary["terminal"])
            controls = []
            comparison = {"period": period, "cost": cost, "comparators": []}
            for policy in ("A_SAVED_WEIGHT", "PRICE_CONFIRMATION"):
                saved, path = saved_account(period, cost, policy)
                require(pd.DatetimeIndex(saved["daily"].date).equals(pd.DatetimeIndex(primary["daily"].date)), "账户日历不同。")
                other, local_years = computed_stats(saved, period, policy, cost)
                rows.append({"period": period, "cost": cost, "policy": policy, **other})
                years.extend(local_years)
                controls.append(other)
                comparison["comparators"].append({
                    "policy": policy, "saved_control": path.relative_to(ROOT).as_posix(),
                    "cagr_delta": stat["net_cagr"]-other["net_cagr"],
                    "sharpe_delta": stat["net_sharpe"]-other["net_sharpe"],
                    "ending_equity_delta": stat["ending_equity"]-other["ending_equity"],
                    "intervals": intervals(saved["daily"].net_return.to_numpy(float), primary["daily"].net_return.to_numpy(float)),
                })
            comparison["economic_gate_passed"] = bool(
                stat["net_cagr"] > max(0., *(control["net_cagr"] for control in controls))
                and stat["net_sharpe"] > max(0., *(control["net_sharpe"] for control in controls))
                and stat["max_drawdown"] <= .1 and stat["p_times_b"] > 1.
                and stat["standard_expectancy_loss_units"] > 0.
            )
            comparisons.append(comparison)
            print(f"{period}/{cost}：净年化{stat['net_cagr']:.2%}，夏普{stat['net_sharpe']:.4f}，完成{stat['completed_cycles']}，pB={stat['p_times_b']:.4f}。", flush=True)
    table("完整账户共同口径比较", pd.DataFrame(rows))
    table("逐年净收益与实际周期次数", pd.DataFrame(years))
    table("实际账户资金库存核对", pd.DataFrame(checks))
    economic = all(comparison["economic_gate_passed"] for comparison in comparisons)
    stable = all(interval["cagr_delta_95"][0] > 0 and interval["sharpe_delta_95"][0] > 0
                 for comparison in comparisons for comparator in comparison["comparators"] for interval in comparator["intervals"])
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "运行期间登记来源改变。")
    write_json(OUT / "summary.json", {
        "at": now(), "study": protocol["study"], "technical_decision": "TECH.R177",
        "status": "HISTORICAL_FULL_DAILY_GAP_INCREMENT_NOT_INDEPENDENT" if economic and stable else "REJECTED_FIXED_FULL_DAILY_GAP_FULL_ACCOUNT_GATE_FAILED",
        "candidate_configurations": 1, "new_primary_accounts": 4, "original_A_replays": 4,
        "original_price_replays": 4, "reused_saved_control_scenarios": 8,
        "new_model_fits": 0, "new_training_labels": 0,
        "full_up_gaps_since_2015": int(states.loc[states.date.ge("2015-01-01")].gap_event.sum()),
        "all_four_economic_gates_passed": economic, "historical_stability_gate_passed": stable,
        "comparisons": comparisons, "new_market_requests": 0, "independent_validation": "NOT_ESTABLISHED",
        "source_first_vintage": "NOT_CERTIFIED", "overfitting_removed": False, "global_DSR_PBO": "NOT_COMPUTED",
        "goal_achieved": False, "old_frozen_failures_and_forward_preserved": True,
    }, exclusive=True)
    print("TECH.R177四账户一次完成；结论以全体实际结果及完整目标判断。", flush=True)


def verify():
    require(not (OUT / "saved_result_verification.json").exists(), "保存核对已完成。")
    protocol, summary = read(OUT / "protocol.json"), read(OUT / "summary.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "保存来源不对应登记。")
    saved = pd.read_parquet(OUT / "results/完整账户共同口径比较.parquet")
    checks = []
    for period in PERIODS:
        for cost in COSTS:
            path = OUT / f"results/accounts/{period}/{cost}/{PRIMARY}"
            result = {name: pd.read_parquet(path / f"{name}.parquet") for name in LEDGERS}
            result["terminal"] = read(path / "terminal.json")
            stats, _ = computed_stats(result, period, PRIMARY, cost)
            stored = saved.loc[saved.period.eq(period) & saved.cost.eq(cost) & saved.policy.eq(PRIMARY)].iloc[0]
            for key, value in stats.items():
                if isinstance(value, (float, np.floating)) and np.isnan(value):
                    require(pd.isna(stored[key]), "未知质量被替换。")
                elif isinstance(value, (float, np.floating)):
                    require(abs(value-float(stored[key])) <= 1e-12, "保存账户指标复算不同。")
                else:
                    require(value == stored[key], "保存账户计数不同。")
            checks.append({"period": period, "cost": cost, **verify_account(result)})
    require(summary["new_primary_accounts"] == len(checks), "一次账户数量不符。")
    write_json(OUT / "saved_result_verification.json", {
        "at": now(), "status": "PASS_FOUR_SAVED_ACCOUNTS_METRICS_AND_LEDGER_IDENTITIES",
        "new_accounts_in_verification": 0, "frozen_sources": len(protocol["sources"]), "checks": checks,
    }, exclusive=True)
    print("四保存账户资金、库存、时间、指标和来源核对通过；未重跑账户。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="完整日线缺口的唯一金融用途。")
    parser.add_argument("command", choices=("tests", "preflight", "freeze", "run", "verify"))
    args = parser.parse_args()
    {"tests": tests, "preflight": preflight, "freeze": freeze, "run": run, "verify": verify}[args.command]()
