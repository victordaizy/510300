"""既定日周联合阶段政策：精确对照、登记、一次完整账户及保存结果核对。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import trend_expansion_phase_inputs_v1 as rules
from research import trend_expansion_phase_account_v1 as execution
from research.point_fresh_repair_order_study_v1 import load
from research.point_account_nr7_inputs_v1 import metrics
from research.point_account_nr7_complement_v1 import verify_account, annual_rows
from research.point_first_passage_study_v1 import read, write_json, digest, now, require, intervals, CURRENT, WEIGHT, CONTROL, PERIODS, COSTS

OUT = ROOT / "reports/research/510300_trend_expansion_phase_study_v1"
EXPLANATION = ROOT / "reports/research/510300_trend_expansion_phase_explanation_v1"
PRIOR = ROOT / "reports/research/510300_point_fresh_repair_order_study_v1"
PRIMARY = "JOINT_PHASE_START"
TESTS = ("tests/test_trend_expansion_phase_inputs_v1.py", "tests/test_trend_expansion_phase_account_v1.py")
LEDGERS = ("daily", "orders", "trades", "decisions", "rejections")


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def tests():
    require(not (OUT / "tests_receipt.json").exists(), "必要测试已记录，不自动重试。")
    completed = subprocess.run([sys.executable, "-X", "utf8", "-m", "pytest", "-q", *TESTS],
                               cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    text = completed.stdout + completed.stderr
    sources = [Path(rules.__file__), Path(execution.__file__), *(ROOT / p for p in TESTS)]
    write_json(OUT / "tests_receipt.json", {
        "at": now(), "exit_code": completed.returncode, "passed": 6 if completed.returncode == 0 and "6 passed" in text else 0,
        "output": text, "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in sources],
    }, exclusive=True)
    print(text, end="", flush=True)
    require(completed.returncode == 0 and "6 passed" in text, "六项必要测试未通过，保留首次输出。")


def frozen_explanation_check():
    protocol = read(EXPLANATION / "protocol.json")
    require(read(EXPLANATION / "summary.json")["status"] == "COMPLETED_FIXED_TREND_EXPANSION_EXPLANATION_NO_FINANCIAL_RESULT",
            "全体阶段解释尚未实际完成。")
    for s in protocol["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "原阶段解释的冻结来源改变。")
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
    require(receipt["passed"] == 6 and receipt["exit_code"] == 0, "必要测试未通过。")
    for s in receipt["sources"]:
        require(digest(ROOT / s["path"]) == s["sha256"], "账户或阶段不再对应测试版本。")
    frozen_explanation_check()
    data, dividends, risks, parents = load()
    phases = rules.phase_states(data)
    saved_phases = pd.read_parquet(EXPLANATION / "results/全部原点_当时日周展开阶段.parquet")
    pd.testing.assert_frame_equal(phases, saved_phases, check_exact=True)
    checks = []
    for period, (start, end) in PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        signal = rules.account_signals(local, phases.iloc[:len(local)].copy(), "PRICE_CONFIRMATION")
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
        "at": now(), "status": "PASS_ALL_PHASES_AND_EIGHT_SAVED_CONTROL_ACCOUNTS",
        "all_phase_rows_identical": len(phases), "original_A_replays": 4, "original_price_replays": 4,
        "new_primary_accounts": 0, "checks": checks,
    }, exclusive=True)
    print("3488阶段原值一致，四原A与四纯价格账户的日账、订单和周期精确复现；新阶段账户尚未运行。", flush=True)


def freeze():
    require(not (OUT / "protocol.json").exists(), "唯一日周展开用途已经登记。")
    require(read(OUT / "control_preflight.json")["status"] == "PASS_ALL_PHASES_AND_EIGHT_SAVED_CONTROL_ACCOUNTS",
            "共同执行对照未通过。")
    explanation = frozen_explanation_check()
    paths = [
        Path(__file__), Path(rules.__file__), Path(execution.__file__), *(ROOT / p for p in TESTS),
        OUT / "tests_receipt.json", OUT / "control_preflight.json", EXPLANATION / "protocol.json",
        EXPLANATION / "summary.json", EXPLANATION / "results/全部原点_当时日周展开阶段.parquet",
        CURRENT / "inputs/candidate_prices.parquet", WEIGHT / "inputs/dividends.csv",
        WEIGHT / "inputs/risks.parquet", WEIGHT / "inputs/parent_signals.parquet", WEIGHT / "inputs/earlier_signals.parquet",
        ROOT / "research/point_fresh_repair_order_study_v1.py", ROOT / "research/point_fresh_repair_order_account_v1.py",
        ROOT / "research/point_fresh_repair_order_inputs_v1.py", ROOT / "research/upward_episode_anatomy_v1.py",
        ROOT / "research/daily_supply_test_v1.py", ROOT / "research/point_account_nr7_inputs_v1.py",
        ROOT / "research/point_account_nr7_complement_v1.py", ROOT / "research/adaptive_allocation_v1.py",
        ROOT / "research/point_first_passage_study_v1.py", ROOT / "research/point_directional_confirmation_study_v1.py",
        ROOT / "research/weekly_daily_technical_v1.py", PRIOR / "protocol.json", PRIOR / "summary.json",
    ]
    for period in PERIODS:
        for cost in COSTS:
            for policy in ("A_SAVED_WEIGHT", "PRICE_CONFIRMATION"):
                _, path = saved_account(period, cost, policy)
                paths.extend(path / f"{name}.parquet" for name in ("daily", "orders", "trades"))
    paths = list(dict.fromkeys(paths))
    protocol = {
        "at": now(), "study": "510300_TREND_EXPANSION_PHASE_STUDY_V1",
        "registration_decision": "TECH.R164", "result_decision": "TECH.R165",
        "hypothesis": "价格已越过EMA20后，日DIF和上一完整周MACD转为同时正，标记从修复到展开的信息到达；只持有该已知阶段，可能改善后段参与和失败退出。",
        "policy_intent_frozen_before_any_old_label_join": explanation["future_complete_policy_intent_fixed_before_old_label_join"],
        "complete_signal": "原价EMA20上、日MACD12/26/9的DIF>0、上一完整周MACD柱>0；当前与前原点已知且前一三项未全成立，才能出生。第一已知正或未知恢复不伪造出生。",
        "entry": "仅空仓出生原点收盘决定，次真实开盘尝试；原现金及风险可减少份额，限价阻止后不延迟追入。尾部标签未成熟不影响已知资格。",
        "exit": "持有阶段任一条件在已知收盘不再成立，次合法开退出；未知本身不强制退出，风险只减，无加仓、2R或20日到期。期末不人工清仓。",
        "old_contract_review": explanation["old_review"] + "已再次核对旧T_ONLY：W_BULL且前20日高突破触发，LL10/ATR/周熊退出及逐级加仓，不是本联合阶段出生与失效。",
        "old_reference_result": "R163全85信号原20日标签平均-0.0083%、pB0.539，失败证据保留；该旧标签不是阶段失效账户的新入组门，也不选突破或相对量通道作为候选。",
        "primary": PRIMARY, "controls": ["A_SAVED_WEIGHT", "PRICE_CONFIRMATION"],
        "account": "各时期20万元、510300.SH/CASH_CNY、252年化日、现金0、原50%上限、ES5预算2.5%、负10%跳空预算5%及半剩余DD余量、DD10%停止新买、T+1/100份/.001刻度、最低佣金5元；原登记/除息应收/付款股息口径。",
        "periods": PERIODS, "costs": {"BASE": [.0002, .0005], "STRESS": [.0004, .001]},
        "primary_gate": "全部两时期两费用，主策略净CAGR和全日历净Sharpe同时高于0、原A和纯价格，最大DD<=10%，实际完成周期pB>1且pB-q>0；未定义质量不填0或无穷。频率是软目标。",
        "uncertainty": "固定20/252日配对循环区块、各2000、种子510300154；两尺度两比较均报告，95%差值下界均>0才满足历史稳定门，仍属开发描述。",
        "purpose_gate": "这是完整规则政策，与原退出预测MSE/115成员用途不同；保留原退出R145失败，不强加其MSE门，也不将技术用途成功称原模型修复。",
        "candidate_configurations": 1, "new_primary_accounts_planned": 4, "saved_control_accounts_replayed": 8,
        "necessary_tests": 6, "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
        "source_first_vintage": "NOT_CERTIFIED", "historical_sample_role": "DEVELOPMENT_CALIBRATION",
        "selection_bias": "规则由已知上涨/失败案例启发；R163前已固定用途，全部历史已用，不是独立样本。有限旧定义核对不声称全项目完全去重。",
        "no_rescue": "不改变现有窗口、阈值、出生时钟、失效条件、费用、时期；不以结果挑指标通道、加过滤或重组A。旧R161及其他失败原样保留。",
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "goal_achieved": False, "orders_authorized": False,
        "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths],
    }
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    print("TECH.R164已冻结唯一日周出生及失效政策；四个新阶段账户尚未运行。", flush=True)


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
    phases = rules.phase_states(data)
    table("全部原点的已知出生与失效请求", rules.account_signals(data, phases, PRIMARY))
    rows, years, checks, comparisons = [], [], [], []
    for period, (start, end) in PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        signals = rules.account_signals(local, phases.iloc[:len(local)].copy(), PRIMARY)
        for cost in COSTS:
            primary = execution.account(local, dividends, parents[period], risks, signals, cost, start, "PHASE_ONLY")
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
        "at": now(), "study": protocol["study"], "technical_decision": "TECH.R165",
        "status": "HISTORICAL_JOINT_PHASE_INCREMENT_NOT_INDEPENDENT" if economic and stable else "REJECTED_FIXED_JOINT_PHASE_FULL_ACCOUNT_GATE_FAILED",
        "candidate_configurations": 1, "new_primary_accounts": 4, "original_A_replays": 4,
        "original_price_replays": 4, "reused_saved_control_scenarios": 8,
        "new_model_fits": 0, "new_training_labels": 0, "joint_onsets_since_2015": int(phases.loc[phases.date.ge("2015-01-01")].joint_phase_onset.sum()),
        "all_four_economic_gates_passed": economic, "historical_stability_gate_passed": stable,
        "comparisons": comparisons, "new_market_requests": 0, "independent_validation": "NOT_ESTABLISHED",
        "source_first_vintage": "NOT_CERTIFIED", "overfitting_removed": False, "global_DSR_PBO": "NOT_COMPUTED",
        "goal_achieved": False, "old_frozen_failures_and_forward_preserved": True,
    }, exclusive=True)
    print("TECH.R165四账户一次完成；完整目标与独立验证仍须以实际证据判断。", flush=True)


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
    parser = argparse.ArgumentParser(description="日周展开阶段的唯一金融政策：必要测试、对照、冻结、一次执行、保存核对。")
    parser.add_argument("command", choices=("tests", "preflight", "freeze", "run", "verify"))
    args = parser.parse_args()
    {"tests": tests, "preflight": preflight, "freeze": freeze, "run": run, "verify": verify}[args.command]()
