"""下降结构首次突破与只上移确认失效线的一次完整账户测量。"""
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
from research import downtrend_break_inputs_v1 as rules
from research import downtrend_break_account_v1 as execution
from research.point_fresh_repair_order_study_v1 import load
from research.point_account_nr7_inputs_v1 import metrics
from research.point_account_nr7_complement_v1 import verify_account, annual_rows
from research.point_first_passage_study_v1 import read, write_json, digest, now, require, intervals, CURRENT, WEIGHT, CONTROL, PERIODS, COSTS

OUT = ROOT / "reports/research/510300_downtrend_break_study_v1"
EXPLANATION = ROOT / "reports/research/510300_downtrend_break_explanation_v1"
PRIOR = ROOT / "reports/research/510300_point_fresh_repair_order_study_v1"
PRIMARY = "FIRST_CONFIRMED_DOWNTREND_BREAK"
TESTS = ("tests/test_downtrend_break_inputs_v1.py", "tests/test_downtrend_break_account_v1.py")
LEDGERS = ("daily", "orders", "trades", "decisions", "rejections")


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def tests():
    require(not (OUT / "tests_receipt.json").exists(), "账户必要测试已记录，不自动重试。")
    inputs = read(EXPLANATION / "tests_receipt.json")
    require(inputs["passed"] == 4 and inputs["exit_code"] == 0, "四个突破输入测试未通过。")
    for s in inputs["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "结构输入不对应原测试。")
    completed = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "-q", TESTS[1]],
                               cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    text = completed.stdout + completed.stderr
    paths = [Path(rules.__file__), Path(execution.__file__), *(ROOT / p for p in TESTS)]
    ok = completed.returncode == 0 and "4 passed" in text
    write_json(OUT / "tests_receipt.json", {
        "at": now(), "exit_code": completed.returncode, "passed": 8 if ok else 0,
        "new_account_tests_passed": 4 if ok else 0, "previous_input_tests_passed": 4,
        "previous_input_receipt": (EXPLANATION / "tests_receipt.json").relative_to(ROOT).as_posix(),
        "input_tests_rerun": False, "output": text,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
    }, exclusive=True)
    print(text, end="", flush=True)
    require(ok, "四个账户必要测试未通过，保留首次输出。")


def frozen_explanation_check():
    protocol = read(EXPLANATION / "protocol.json")
    require(read(EXPLANATION / "summary.json")["status"] == "COMPLETED_ALL_DOWNTREND_BREAK_PATH_EXPLANATION_NO_FINANCIAL_RESULT",
            "全体结构解释尚未实际完成。")
    for s in protocol["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "原结构解释的冻结来源改变。")
    return protocol


def saved_account(period, cost, policy):
    path = CONTROL / period / cost / policy if policy == "A_SAVED_WEIGHT" else PRIOR / "results/accounts" / period / cost / policy
    result = {name: pd.read_parquet(path / f"{name}.parquet") for name in ("daily", "orders", "trades")}
    result["terminal"] = {"stopped": bool(result["daily"].risk_stopped.iloc[-1])}
    return result, path


def preflight():
    require(not (OUT / "control_preflight.json").exists() and not (OUT / "protocol.json").exists(),
            "对照已核验或金融用途已登记，不重复。")
    receipt = read(OUT / "tests_receipt.json")
    require(receipt["passed"] == 8 and receipt["exit_code"] == 0, "必要测试未通过。")
    for s in receipt["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "账户或阶段不再对应测试版本。")
    frozen_explanation_check()
    data, dividends, risks, parents = load()
    states, _ = rules.break_states(data)
    saved_phases = pd.read_parquet(EXPLANATION / "results/全部原点_上一确认下降结构与首次突破.parquet")
    pd.testing.assert_frame_equal(states, saved_phases, check_exact=True)
    checks = []
    for period, (start, end) in PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        signal = rules.account_signals(local, states.iloc[:len(local)].copy(), "PRICE_CONFIRMATION")
        for cost in COSTS:
            for policy, mode in [("A_SAVED_WEIGHT", "A_CONTROL"), ("PRICE_CONFIRMATION", "PRICE_ONLY")]:
                actual = execution.account(local, dividends, parents[period], risks, signal, cost, start, mode)
                check = verify_account(actual)
                saved, path = saved_account(period, cost, policy)
                for name in ("daily", "orders", "trades"):
                    expected = saved[name]
                    pd.testing.assert_frame_equal(actual[name][expected.columns], expected, check_exact=True)
                checks.append({"period": period, "cost": cost, "policy": policy,
                               "saved_path": path.relative_to(ROOT).as_posix(), **check})
    write_json(OUT / "control_preflight.json", {
        "at": now(), "status": "PASS_ALL_STATES_AND_EIGHT_SAVED_CONTROL_ACCOUNTS",
        "all_structure_rows_identical": len(states), "original_A_replays": 4, "original_price_replays": 4,
        "new_primary_accounts": 0, "checks": checks,
    }, exclusive=True)
    print("3488结构原值一致，四原A与四纯价格账户的日账、订单和周期精确复现；新首次突破账户尚未运行。", flush=True)


def freeze():
    require(not (OUT / "protocol.json").exists(), "首次下降结构突破金融用途已登记。")
    require(read(OUT / "control_preflight.json")["status"] == "PASS_ALL_STATES_AND_EIGHT_SAVED_CONTROL_ACCOUNTS",
            "共同执行对照未通过。")
    explanation = frozen_explanation_check()
    paths = [Path(__file__), Path(rules.__file__), Path(execution.__file__), *(ROOT / p for p in TESTS),
             OUT / "tests_receipt.json", OUT / "control_preflight.json", EXPLANATION / "protocol.json",
             EXPLANATION / "summary.json", EXPLANATION / "results/全部原点_上一确认下降结构与首次突破.parquet",
             ROOT / "research/point_fresh_repair_order_account_v1.py", ROOT / "research/point_fresh_repair_order_inputs_v1.py",
             ROOT / "research/point_account_nr7_complement_v1.py", ROOT / "research/point_directional_confirmation_study_v1.py",
             PRIOR / "protocol.json", PRIOR / "summary.json"]
    paths.extend(ROOT / s["path"] for s in explanation["sources"])
    for period in PERIODS:
        for cost in COSTS:
            for policy in ("A_SAVED_WEIGHT", "PRICE_CONFIRMATION"):
                _, path = saved_account(period, cost, policy)
                paths.extend(path / f"{name}.parquet" for name in ("daily", "orders", "trades"))
    for name in ("initial_account_test_failure.json", "initial_account_test_failure_explanation.json", "initial_account_tests.py", "initial_account_before_cash_coordinate_fix.py"):
        if (OUT / name).exists():
            paths.append(OUT / name)
    paths = list(dict.fromkeys(paths))
    protocol = {
        "at": now(), "study": "510300_DOWNTREND_BREAK_STUDY_V1",
        "registration_decision": "TECH.R172", "result_decision": "TECH.R173",
        "hypothesis": "价格首次突破此前已经确认的下降结构最后高点，可能比完整双抬升更早到达；被突破旧高和向上移动的确认低点提供完整失效时钟，须检验全账户收益夏普。",
        "policy_intent_fixed_before_new_outcome_join": explanation["future_complete_policy_intent_fixed_before_any_new_outcome_join"],
        "complete_signal": explanation["definition"],
        "entry": "上一已确认两高两低严格下降，上一收盘<=同一旧H2而当前收盘>该H2，同锚只首次；空仓次真实开尝试，次开现金平移价<=旧H2取消，不追入。原现金及风险只缩减昨晚计划份额。",
        "exit": "初始线为事件上一原点已确认旧H2，持有后每个已知原点仅将线取max(原线,最新已确认低点)；已知收盘<=线次合法开退出，未知不强制退出，原风险只减，无加仓/混A/固定2R/20日，期末不人工清仓。",
        "old_contract_review": explanation["old_review"],
        "admission_reason": "R171全体具体路径及失效解释、四输入测试/四真实前缀和旧完整用途核对已完成；新用途在连接旧结果标签前固定，不根据其描述选择过滤、参数或退出。八共同控制精确复现、四账户行为测试通过。只准入一次完整开发测量，不称有盈利优势。",
        "primary": PRIMARY, "controls": ["A_SAVED_WEIGHT", "PRICE_CONFIRMATION"],
        "account": "原20万元/252交易日/现金收益0、510300.SH/CASH_CNY、50%上限、ES5预算2.5%、负10%跳空预算min(5%NAV,半剩余DD余量)、DD10%停止新买，T+1/100份/.001刻度/最低佣金5元；原登记、除息应收及付款股息。",
        "periods": PERIODS, "costs": {"BASE": [.0002, .0005], "STRESS": [.0004, .001]},
        "primary_gate": "全部两时期两费用净CAGR和全日历净Sharpe同时高于0、原A和纯价格，DD<=10%，实际完成周期pB>1且pB-q>0；未知质量不填0或无穷，次数软目标。",
        "uncertainty": "固定20/252日配对循环区块，各2000、种子510300154；两尺度两比较全报告，95%差值下界均>0才过历史稳定门，仍属开发描述。",
        "purpose_gate": "完整技术规则用途，不应用原退出MSE/115成员门；原退出R145及预测R158失败保持，不称原模型修复。",
        "candidate_configurations": 1, "new_primary_accounts_planned": 4, "saved_control_accounts_replayed": 8,
        "necessary_tests": 8, "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "source_first_vintage": "NOT_CERTIFIED", "historical_sample_role": "DEVELOPMENT_CALIBRATION",
        "selection_bias": "全部历史已经使用，机制由已知案例启发；提前固定用途仍不创造独立样本，有限旧核对不声称穷尽全项目。",
        "no_rescue": "R170双抬升固定用途及所有旧终态保持；不改原2/2确认、严格越过、同锚首次、初始旧高或只上移低点、费用/时期/预算，不由结果加量价/MACD/RV过滤或混A。",
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED", "overfitting_removed": False,
        "goal_achieved": False, "orders_authorized": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
    }
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    print("TECH.R172冻结唯一首次下降结构突破金融用途；四个新账户尚未运行。", flush=True)


def computed_stats(result, period, policy, cost):
    stats = metrics(result)
    years = annual_rows(result, period, policy, cost)
    counts = [r["completed_cycles"] for r in years if r["full_year"]]
    stats.update(average_full_year_cycles=float(np.mean(counts)), zero_trade_full_years=sum(n == 0 for n in counts))
    return stats, years


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "该固定用途已经开始，不重启。")
    protocol = read(OUT / "protocol.json")
    for s in protocol["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "登记来源改变。")
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_primary_accounts_planned": 4}, exclusive=True)
    data, dividends, risks, parents = load()
    states, _ = rules.break_states(data)
    table("全部已知首次突破与确认低点_失效由持仓决定", rules.account_signals(data, states, PRIMARY))
    rows, years, checks, comparisons = [], [], [], []
    for period, (start, end) in PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        signals = rules.account_signals(local, states.iloc[:len(local)].copy(), PRIMARY)
        for cost in COSTS:
            primary = execution.account(local, dividends, parents[period], risks, signals, cost, start, "BREAK_ONLY")
            check = verify_account(primary)
            checks.append({"period": period, "cost": cost, "policy": PRIMARY, **check})
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
                stat["net_cagr"] > max(0., *(x["net_cagr"] for x in controls))
                and stat["net_sharpe"] > max(0., *(x["net_sharpe"] for x in controls))
                and stat["max_drawdown"] <= .1 and stat["p_times_b"] > 1.
                and stat["standard_expectancy_loss_units"] > 0.
            )
            comparisons.append(comparison)
            print(f"{period}/{cost}：净年化{stat['net_cagr']:.2%}，夏普{stat['net_sharpe']:.4f}，完成{stat['completed_cycles']}，pB={stat['p_times_b']:.4f}。", flush=True)
    table("完整账户共同口径比较", pd.DataFrame(rows))
    table("逐年净收益与实际周期次数", pd.DataFrame(years))
    table("实际账户资金库存核对", pd.DataFrame(checks))
    economic = all(c["economic_gate_passed"] for c in comparisons)
    stable = all(i["cagr_delta_95"][0] > 0 and i["sharpe_delta_95"][0] > 0
                 for c in comparisons for r in c["comparators"] for i in r["intervals"])
    for s in protocol["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "运行期间登记来源改变。")
    write_json(OUT / "summary.json", {
        "at": now(), "study": protocol["study"], "technical_decision": "TECH.R173",
        "status": "HISTORICAL_DOWNTREND_BREAK_INCREMENT_NOT_INDEPENDENT" if economic and stable else "REJECTED_FIXED_DOWNTREND_BREAK_FULL_ACCOUNT_GATE_FAILED",
        "candidate_configurations": 1, "new_primary_accounts": 4, "original_A_replays": 4,
        "original_price_replays": 4, "reused_saved_control_scenarios": 8,
        "new_model_fits": 0, "new_training_labels": 0, "first_break_events_since_2015": int(states.loc[states.date.ge("2015-01-01")].break_event.sum()),
        "all_four_economic_gates_passed": economic, "historical_stability_gate_passed": stable,
        "comparisons": comparisons, "new_market_requests": 0, "independent_validation": "NOT_ESTABLISHED",
        "source_first_vintage": "NOT_CERTIFIED", "overfitting_removed": False, "global_DSR_PBO": "NOT_COMPUTED",
        "goal_achieved": False, "old_frozen_failures_and_forward_preserved": True,
    }, exclusive=True)
    print("TECH.R173四账户一次完成；完整目标与独立验证仍须以实际证据判断。", flush=True)


def verify():
    require(not (OUT / "saved_result_verification.json").exists(), "保存核对已完成。")
    protocol, summary = read(OUT / "protocol.json"), read(OUT / "summary.json")
    for s in protocol["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "保存来源不对应登记。")
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
    require(summary["new_primary_accounts"] == len(checks), "一次执行的账户数量不符。")
    write_json(OUT / "saved_result_verification.json", {
        "at": now(), "status": "PASS_FOUR_SAVED_ACCOUNTS_METRICS_AND_LEDGER_IDENTITIES",
        "new_accounts_in_verification": 0, "frozen_sources": len(protocol["sources"]), "checks": checks,
    }, exclusive=True)
    print("四保存账户资金、库存、时间、指标与登记来源核对通过；未重新运行账户。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="首次下降结构突破的唯一金融政策：必要测试、对照、冻结、一次执行、保存核对。")
    parser.add_argument("command", choices=("tests", "preflight", "freeze", "run", "verify"))
    args = parser.parse_args()
    {"tests": tests, "preflight": preflight, "freeze": freeze, "run": run, "verify": verify}[args.command]()
