"""由具体上涨解释反推唯一当前修复顺序规则，冻结后一次运行完整账户。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import point_fresh_repair_order_inputs_v1 as rules
from research import point_fresh_repair_order_account_v1 as execution
from research import upward_episode_anatomy_v1 as anatomy
from research.adaptive_allocation_v1 import normalize_dividends
from research.point_account_nr7_inputs_v1 import metrics
from research.point_account_nr7_complement_v1 import verify_account, annual_rows
from research.point_first_passage_study_v1 import read, write_json, digest, now, require, intervals, CURRENT, WEIGHT, CONTROL, PERIODS, COSTS

OUT = ROOT / "reports/research/510300_point_fresh_repair_order_study_v1"
ATLAS = ROOT / "reports/research/510300_upward_episode_anatomy_v1"
CASES = ROOT / "reports/research/510300_concrete_upward_cases_v1"
MODE = {"FRESH_ORDERED_REPAIR": "ORDERED_ONLY", "PRICE_CONFIRMATION": "PRICE_ONLY"}


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def load():
    prices = pd.read_parquet(CURRENT / "inputs/candidate_prices.parquet")
    dividends = normalize_dividends(pd.read_csv(WEIGHT / "inputs/dividends.csv"))
    data, _ = anatomy.features(prices, dividends)
    require(len(data) == 3488 and data.symbol.eq("510300.SH").all(), "日线研究对象或原日期范围改变。")
    risks = pd.read_parquet(WEIGHT / "inputs/risks.parquet")
    require(risks.latest_label_exit_idx.le(risks.idx).all(), "风险估计读取未来标签。")
    require(np.array_equal(data.date.iloc[risks.idx.to_numpy(int)].to_numpy(), risks.date.to_numpy()), "风险估计和日线错位。")
    recent = pd.read_parquet(WEIGHT / "inputs/parent_signals.parquet").pivot(index="origin", columns="candidate", values="target").reset_index()
    early = pd.read_parquet(WEIGHT / "inputs/earlier_signals.parquet")
    early = early.loc[early.period.eq("earlier_diagnostic")].pivot(index="origin", columns="model", values="target").reset_index()
    return data, dividends, risks, {"2015_2019": early, "2020_2026": recent}


def preflight():
    require(not (OUT / "protocol.json").exists() and not (OUT / "control_preflight.json").exists(), "实验已经核验或冻结。")
    tests = read(OUT / "tests_receipt.json")
    require(tests["passed"] == 5 and tests["exit_code"] == 0, "五项必要测试未通过。")
    for key, path in [("inputs_sha256", Path(rules.__file__)), ("account_sha256", Path(execution.__file__)),
                      ("test_sha256", ROOT / "tests/test_point_fresh_repair_order_v1.py")]:
        require(tests[key] == digest(path), "必要测试版本不同。")
    data, dividends, risks, parents = load()
    old = pd.read_parquet(ATLAS / "results/features.parquet")
    require(data.date.iloc[:len(old)].equals(old.date), "具体案例与账户日线前缀日期不同。")
    maximum_error = 0.
    for name in ["ac", "ema20", "daily_hist", "weekly_hist", "relative_volume", "up_volume_balance5", "rv_ratio"]:
        a, b = data[name].iloc[:len(old)].to_numpy(float), old[name].to_numpy(float)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)
        valid = np.isfinite(a) & np.isfinite(b)
        maximum_error = max(maximum_error, float(np.max(np.abs(a[valid]-b[valid]))))
    sequence = rules.signals(data)
    case_checks = []
    for date in ["2019-01-09", "2020-04-07", "2024-09-24"]:
        r = sequence.loc[sequence.date.eq(date)].iloc[0]
        require(bool(r.ordered_repair), "本规则不能用当时已知顺序解释指定案例，不能事后换起点。")
        case_checks.append({"origin": date, "volume_positive_since": data.date.iloc[int(r.current_volume_positive_start)],
                            "macd_positive_since": data.date.iloc[int(r.current_macd_positive_start)],
                            "known_under_start": data.date.iloc[int(r.known_under_start)]})
    checks = []
    for period, (start, end) in PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        empty = pd.DataFrame({"date": local.date, "entry_event": False, "event_id": None, "atr": np.nan,
                              "stop_index": np.nan, "target_index": np.nan})
        for cost in COSTS:
            actual = execution.account(local, dividends, parents[period], risks, empty, cost, start, "A_CONTROL")
            check = verify_account(actual)
            saved = CONTROL / period / cost / "A_SAVED_WEIGHT"
            pd.testing.assert_frame_equal(actual["daily"], pd.read_parquet(saved / "daily.parquet"), check_exact=True)
            orders = pd.read_parquet(saved / "orders.parquet")
            pd.testing.assert_frame_equal(actual["orders"][orders.columns], orders, check_exact=True)
            checks.append({"period": period, "cost": cost, **check})
    write_json(OUT / "control_preflight.json", {"at": now(), "status": "PASS_ORIGINAL_ATLAS_PREFIX_AND_FOUR_A_ACCOUNTS",
               "maximum_atlas_feature_error": maximum_error, "case_known_sequence_checks": case_checks,
               "original_A_replays": 4, "new_policy_accounts": 0, "checks": checks}, exclusive=True)
    print("三个案例当前修复顺序成立；原图谱指标前缀和四原A账户一致，新规则账户尚未运行。", flush=True)


def freeze():
    require(not (OUT / "protocol.json").exists(), "唯一顺序规则已冻结。")
    require(read(OUT / "control_preflight.json")["status"] == "PASS_ORIGINAL_ATLAS_PREFIX_AND_FOUR_A_ACCOUNTS", "原输入或执行对照未通过。")
    paths = [Path(__file__), Path(rules.__file__), Path(execution.__file__), ROOT / "tests/test_point_fresh_repair_order_v1.py",
             OUT / "tests_receipt.json", OUT / "control_preflight.json", CURRENT / "inputs/candidate_prices.parquet",
             WEIGHT / "inputs/dividends.csv", WEIGHT / "inputs/risks.parquet", WEIGHT / "inputs/parent_signals.parquet",
             WEIGHT / "inputs/earlier_signals.parquet", ATLAS / "results/features.parquet", ATLAS / "results/全部原点结果标签.parquet",
             CASES / "user_direction_confirmation.json", CASES / "sequence_map_definition.json",
             ROOT / "research/upward_episode_anatomy_v1.py", ROOT / "research/daily_supply_test_v1.py",
             ROOT / "research/point_account_nr7_inputs_v1.py", ROOT / "research/point_account_nr7_complement_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/point_first_passage_study_v1.py",
             ROOT / "research/point_directional_confirmation_study_v1.py", ROOT / "research/sequential_patterns_regime_v1.py",
             ROOT / "research/directional_entry_timing_v1.py", ROOT / "research/weekly_daily_entry_locations_v1.py",
             ROOT / "research/price_volume_coherence_inputs_v1.py", ROOT / "scripts/run_pattern_daily_state_learning_v3.py"]
    for period in PERIODS:
        for cost in COSTS:
            paths.extend(CONTROL / period / cost / "A_SAVED_WEIGHT" / name for name in ["daily.parquet", "orders.parquet", "trades.parquet"])
    protocol = {"at": now(), "study": "510300_FRESH_REPAIR_ORDER_V1", "registration_decision": "TECH.R160", "result_decision": "TECH.R161",
                "user_order": "先解释具体上涨，再反推当时能够识别的点位，最后完整账户收益夏普。",
                "previous_goal_turn_classification": "PROGRESS_ACTUAL_NEGATIVE_MODEL_EVIDENCE_AND_CONCRETE_CASE_SEQUENCE_MAP",
                "hypothesis": "同一均线下修复中，当前仍持续的量方向正段先出生，日MACD正段随后出生，价格最后上穿EMA20；这种新到达的联合确认比单次价格上穿更能区分延续与短反弹。",
                "old_contract_review": "旧单项条件未通过；旧动量转折/EMA上穿/区间突破采用固定结构2R和20日终点，旧三形态为压缩突破/失地收复/冲击半修复，V3是每日状态回报学习，价量相关为20日Pearson确认。此处当前正段出生顺序及其价格失效退出为新的完整用途；有限定义核对不声称全项目去重。",
                "case_selection_bias": "已知2019/2020/2024上涨与2015失败，规则由这些已用案例提出；全部历史开发，不能称独立验证。",
                "feature_recipe": "原图谱原样：前向现金平移、EMA20、MACD12/26/9柱2*(DIF-DEA)、经济收益正负签成交量5日占比，前完整周130周预热。无新指标/新窗口/波动或放量阈值筛选。",
                "current_birth": "当前严格正段的起点，之前真实已知值必须<=0；初始未知或一直正不伪造转正。期间一旦非正或未知就重置。",
                "signal": "原点价格从<=EMA20至>EMA20，原特征available，连续均线下起点<=当前量正段出生<当前日MACD正段出生<价格上穿原点。严格顺序，不接受同日、倒序或当前已失效正段。",
                "entry": "空仓时唯一上穿原点决定，次真实开盘尝试，原风险可缩小预定份额；开盘被限价阻止不延后追入。所有合格原点，包括未来标签尚未成熟者，不按结果成熟标记决定是否交易。",
                "exit": "买后价格收盘重新<=当日EMA20即确认价格修复失效，次合法开盘退出；无2R或20日到期，无模型重估；其他情况下风险只减仓。",
                "policies": rules.POLICIES, "primary": "FRESH_ORDERED_REPAIR",
                "comparison": "两个独立完整政策账户：顺序规则与完全相同执行退出的纯价格确认；另原A精确对照，不混合核心或按结果切换。",
                "account": "各时期20万元，510300.SH/CASH_CNY，现金0、252日、原50%及ES5预算2.5%/负10%跳空预算5%/半剩余DD余量/DD10停止新入场、T+1/100份/.001刻度/最低佣金5元；终点不人工清仓。",
                "periods": PERIODS, "costs": {"BASE": [.0002, .0005], "STRESS": [.0004, .001]},
                "primary_gate": "全部两时期两费用，顺序账户净CAGR和全日历Sharpe同时高于原A和纯价格并为正，DD<=10%、实际完成pB>1且标准期望>0；频率软目标。未定义质量保留，不填0或无穷。",
                "uncertainty": "原20/252日配对循环区块各2000，种子510300154，两尺度/两比较均报告。仍为开发描述。",
                "old_reference_diagnostic": "原186成熟价格事件的原20日结果标签仅用于解释顺序组/全部组，完整账户不受该诊断门阻止。不制造新训练标签、策略夏普或独立样本。",
                "necessary_tests": 5, "original_A_replays": 4, "new_policy_accounts_planned": 8,
                "new_model_fits": 0, "new_training_labels": 0, "new_market_requests": 0,
                "source_limit": "原价量/股息历史首版未认证，当前所有已使用历史DEVELOPMENT_CALIBRATION。",
                "no_rescue": "当前正段、严格顺序、EMA/MACD/量窗口、进出、执行/成本和时期固定，不按结果改序、加阈值或过滤；原失败保持。",
                "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False, "orders_authorized": False,
                "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths]}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    print("唯一当前修复顺序及完整失效退出规则已冻结，八个新政策账户尚未运行。", flush=True)


def reference_stats(frame):
    values = frame.net_reference_return.to_numpy(float)
    positive, negative = values[values > 0], values[values < 0]
    n = len(values)
    p = len(positive)/n if n else np.nan
    q = len(negative)/n if n else np.nan
    b = positive.mean()/-negative.mean() if len(positive) and len(negative) else np.nan
    return {"reference_origins": n, "net_label_win_rate": p, "net_label_payoff": b, "net_label_p_times_b": p*b,
            "net_label_mean": float(values.mean()) if n else np.nan, "standard_label_expectation": p*b-q,
            "executable_account_metrics": False}


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "唯一顺序规则已经开始，不重启。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "冻结来源改变。")
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_policy_accounts_planned": 8}, exclusive=True)
    data, dividends, risks, parents = load()
    sequence = rules.signals(data)
    table("全部原点的当时已知修复顺序", sequence)
    labels = pd.read_parquet(ATLAS / "results/全部原点结果标签.parquet")
    reference = labels.loc[labels.status.eq("MATURE")].merge(sequence[["date", "price_cross", "ordered_repair"]], on="date", validate="one_to_one")
    reference = reference.loc[reference.date.ge("2015-01-01") & reference.price_cross]
    require(len(reference) == 186, "原成熟价格事件母集改变。")
    table("原186事件的顺序状态与旧结果标签", reference)
    diagnostic = []
    for name, (start, end) in {"ALL": ("2015-01-01", "2026-09-16"), "2015_2019": ("2015-01-01", "2019-12-31"),
                              "2020_2023": ("2020-01-01", "2023-12-31"), "2024_2026": ("2024-01-01", "2026-09-16")}.items():
        local = reference.loc[reference.date.between(start, end)]
        for group, mask in [("ALL_ORIGINAL_PRICE_EVENTS", pd.Series(True, index=local.index)), ("FIXED_ORDERED_EVENTS", local.ordered_repair)]:
            diagnostic.append({"period": name, "group": group, **reference_stats(local.loc[mask])})
    table("旧20日标签的解释对照", pd.DataFrame(diagnostic))
    rows, years, checks, comparisons = [], [], [], []
    for period, (start, end) in PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        seq = sequence.iloc[:len(local)].copy()
        for cost in COSTS:
            saved = CONTROL / period / cost / "A_SAVED_WEIGHT"
            baseline = {name: pd.read_parquet(saved / f"{name}.parquet") for name in ["daily", "trades", "orders"]}
            baseline["terminal"] = {"stopped": bool(baseline["daily"].risk_stopped.iloc[-1])}
            base_stats = metrics(baseline)
            base_years = annual_rows(baseline, period, "A_SAVED_WEIGHT", cost)
            base_stats["average_full_year_cycles"] = float(np.mean([r["completed_cycles"] for r in base_years if r["full_year"]]))
            rows.append({"period": period, "cost": cost, "policy": "A_SAVED_WEIGHT", **base_stats})
            years.extend(base_years)
            actual = {}
            for policy in rules.POLICIES:
                signal = rules.account_signals(local, seq, policy)
                account = execution.account(local, dividends, parents[period], risks, signal, cost, start, MODE[policy])
                checks.append({"period": period, "cost": cost, "policy": policy, **verify_account(account)})
                stat = metrics(account)
                yearly = annual_rows(account, period, policy, cost)
                stat.update(average_full_year_cycles=float(np.mean([r["completed_cycles"] for r in yearly if r["full_year"]])))
                rows.append({"period": period, "cost": cost, "policy": policy, **stat})
                years.extend(yearly)
                for name in ["daily", "orders", "trades", "decisions", "rejections"]:
                    table(f"accounts/{period}/{cost}/{policy}/{name}", account[name])
                write_json(OUT / f"results/accounts/{period}/{cost}/{policy}/terminal.json", account["terminal"])
                actual[policy] = (account, stat)
                print(f"{period}/{cost}/{policy}：净年化{stat['net_cagr']:.2%}，夏普{stat['net_sharpe']:.4f}，完整周期{stat['completed_cycles']}，实际pB{stat['p_times_b']:.4f}。", flush=True)
            primary, primary_stats = actual["FRESH_ORDERED_REPAIR"]
            plain, plain_stats = actual["PRICE_CONFIRMATION"]
            passed = bool(primary_stats["net_cagr"] > max(0., base_stats["net_cagr"], plain_stats["net_cagr"])
                          and primary_stats["net_sharpe"] > max(0., base_stats["net_sharpe"], plain_stats["net_sharpe"])
                          and primary_stats["max_drawdown"] <= .1 and primary_stats["p_times_b"] > 1
                          and primary_stats["standard_expectancy_loss_units"] > 0)
            comparison = {"period": period, "cost": cost, "economic_gate_passed": passed, "comparators": []}
            for name, other, stat in [("A_SAVED_WEIGHT", baseline, base_stats), ("PRICE_CONFIRMATION", plain, plain_stats)]:
                require(pd.DatetimeIndex(other["daily"].date).equals(pd.DatetimeIndex(primary["daily"].date)), "完整账户日历不同。")
                comparison["comparators"].append({"policy": name, "cagr_delta": primary_stats["net_cagr"]-stat["net_cagr"],
                                                 "sharpe_delta": primary_stats["net_sharpe"]-stat["net_sharpe"],
                                                 "ending_equity_delta": primary_stats["ending_equity"]-stat["ending_equity"],
                                                 "intervals": intervals(other["daily"].net_return.to_numpy(), primary["daily"].net_return.to_numpy())})
            comparisons.append(comparison)
    table("完整账户共同口径比较", pd.DataFrame(rows))
    table("逐年净收益与实际交易次数", pd.DataFrame(years))
    table("实际账户资金库存核对", pd.DataFrame(checks))
    economic = all(c["economic_gate_passed"] for c in comparisons)
    stable = all(i["cagr_delta_95"][0] > 0 and i["sharpe_delta_95"][0] > 0 for c in comparisons for r in c["comparators"] for i in r["intervals"])
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "执行期间冻结来源改变。")
    write_json(OUT / "summary.json", {"at": now(), "study": protocol["study"], "technical_decision": "TECH.R161",
               "status": "HISTORICAL_ORDERED_REPAIR_INCREMENT_NOT_INDEPENDENT" if economic and stable else "REJECTED_FIXED_FRESH_REPAIR_ORDER_FULL_ACCOUNT_GATE_FAILED",
               "new_policy_accounts": 8, "original_A_control_replays": 4, "new_model_fits": 0, "new_training_labels": 0,
               "fixed_entry_mechanisms": 1, "original_reference_events": 186,
               "ordered_old_reference_events": int(reference.ordered_repair.sum()),
               "all_four_economic_gates_passed": economic, "historical_stability_gate_passed": stable,
               "comparisons": comparisons, "new_market_requests": 0, "independent_validation": "NOT_ESTABLISHED",
               "overfitting_removed": False, "global_DSR_PBO": "NOT_COMPUTED", "goal_achieved": False,
               "old_frozen_failures_and_forward_preserved": True}, exclusive=True)
    print("唯一当前修复顺序及八个完整账户已一次完成；独立验证和完整目标未成立。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="具体上涨解释反推的当前修复顺序完整账户。")
    parser.add_argument("command", choices=("preflight", "freeze", "run"))
    args = parser.parse_args()
    {"preflight": preflight, "freeze": freeze, "run": run}[args.command]()
