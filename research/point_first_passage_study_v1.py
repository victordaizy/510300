"""唯一首次边界学习实验：执行对照、冻结、一次拟合与完整账户比较。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import point_first_passage_inputs_v1 as learning
from research import point_first_passage_account_v1 as execution
from research.point_account_nr7_inputs_v1 import metrics
from research.point_account_nr7_complement_v1 import verify_account, annual_rows
from research.adaptive_allocation_v1 import normalize_dividends
from research.point_directional_confirmation_study_v1 import intervals

OUT = ROOT / "reports/research/510300_point_first_passage_study_v1"
CURRENT = ROOT / "reports/research/510300_point_current_observation_20261001"
WEIGHT = ROOT / "reports/research/510300_point_weight_information_diagnostic_v1"
CONTROL = ROOT / "reports/research/510300_point_second_weight_comparison_v1/inputs/controls"
PERIODS = {"2015_2019": ("2015-01-05", "2019-12-31"), "2020_2026": ("2020-01-02", "2026-09-30")}
COSTS = ("BASE", "STRESS")
POLICIES = ("TECHNICAL_LOGIT", "MATURE_FREQUENCY")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean(v) for v in value]
    if value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(clean(value), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def load():
    prices = pd.read_parquet(CURRENT / "inputs/candidate_prices.parquet")
    dividends = normalize_dividends(pd.read_csv(WEIGHT / "inputs/dividends.csv"))
    data = learning.features(prices, dividends)
    require(len(data) == 3488 and data.symbol.eq("510300.SH").all(), "原日线研究对象或日期范围改变。")
    risks = pd.read_parquet(WEIGHT / "inputs/risks.parquet")
    require(risks.latest_label_exit_idx.le(risks.idx).all(), "原风险估计有未成熟标签。")
    require(np.array_equal(data.date.iloc[risks.idx.to_numpy(int)].to_numpy(), risks.date.to_numpy()), "风险估计日线错位。")
    recent = pd.read_parquet(WEIGHT / "inputs/parent_signals.parquet").pivot(index="origin", columns="candidate", values="target").reset_index()
    early = pd.read_parquet(WEIGHT / "inputs/earlier_signals.parquet")
    early = early.loc[early.period.eq("earlier_diagnostic")].pivot(index="origin", columns="model", values="target").reset_index()
    return data, dividends, risks, {"2015_2019": early, "2020_2026": recent}


def preflight():
    require(not (OUT / "protocol.json").exists() and not (OUT / "control_preflight.json").exists(), "原A对照已经核验或实验已登记。")
    tests = read(OUT / "tests_receipt.json")
    require(tests["passed"] == 6 and tests["exit_code"] == 0, "必要测试未全部通过。")
    for key, file in [("inputs_sha256", Path(learning.__file__)), ("account_sha256", Path(execution.__file__)),
                      ("test_sha256", ROOT / "tests/test_point_first_passage_v1.py")]:
        require(tests[key] == digest(file), "必要测试版本不同。")
    data, dividends, risks, parents = load()
    results = []
    for period, (start, end) in PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        empty = pd.DataFrame({"date": local.date, "entry_event": False, "event_id": None,
                              "atr": np.nan, "stop_index": np.nan, "target_index": np.nan})
        for cost in COSTS:
            actual = execution.account(local, dividends, parents[period], risks, empty, cost, start, "A_CONTROL")
            check = verify_account(actual)
            saved = CONTROL / period / cost / "A_SAVED_WEIGHT"
            pd.testing.assert_frame_equal(actual["daily"], pd.read_parquet(saved / "daily.parquet"), check_exact=True)
            original_orders = pd.read_parquet(saved / "orders.parquet")
            pd.testing.assert_frame_equal(actual["orders"][original_orders.columns], original_orders, check_exact=True)
            results.append({"period": period, "cost": cost, "exact_saved_daily_and_orders": True, **check})
    write_json(OUT / "control_preflight.json", {"at": now(), "status": "PASS_FOUR_ORIGINAL_A_ACCOUNTS_EXACT",
               "actual_original_control_replays": 4, "new_candidate_accounts": 0, "checks": results}, exclusive=True)
    print("原A四个账户逐日金额、份额及原订单精确一致；新模型和候选账户尚未运行。", flush=True)


def freeze():
    require(not (OUT / "protocol.json").exists(), "首次边界学习已经登记。")
    require(read(OUT / "control_preflight.json")["status"] == "PASS_FOUR_ORIGINAL_A_ACCOUNTS_EXACT", "原A精确对照未通过。")
    paths = [Path(__file__), Path(learning.__file__), Path(execution.__file__), ROOT / "tests/test_point_first_passage_v1.py",
             OUT / "tests_receipt.json", OUT / "control_preflight.json", OUT / "pre_registration_test_failure.json", OUT / "pre_registration_control_failure_01.json",
             CURRENT / "inputs/candidate_prices.parquet", WEIGHT / "inputs/dividends.csv", WEIGHT / "inputs/risks.parquet",
             WEIGHT / "inputs/parent_signals.parquet", WEIGHT / "inputs/earlier_signals.parquet",
             ROOT / "research/point_account_nr7_inputs_v1.py", ROOT / "research/daily_supply_test_v1.py",
             ROOT / "research/point_account_nr7_complement_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/point_directional_confirmation_study_v1.py",
             ROOT / "research/core_auxiliary_drawdown_gate_inputs_v1.py",
             ROOT / "research/return_confirmation_auxiliary_batch_inputs_v1.py",
             ROOT / "research/return_classification_v1.py", ROOT / "research/entry_payoff_gate_inputs_v1.py",
             ROOT / "research/probability_payoff_prediction_stage_v1.py"]
    for period in PERIODS:
        for cost in COSTS:
            paths.extend(CONTROL / period / cost / "A_SAVED_WEIGHT" / name for name in ["daily.parquet", "orders.parquet", "trades.parquet"])
    protocol = {"study": "510300_POINT_FIRST_PASSAGE_LEARNING_V1", "at": now(),
                "registration_decision": "TECH.R157", "result_decision": "TECH.R158",
                "question": "日周线MACD、价量和波动能否预测首次盈利/亏损/到期事件，并在完整账户同时提高原A和无技术特征对照的收益夏普？",
                "known_before_freeze": "已知全部旧技术账户和历史选择。A/B共享同一核心及RETURN_RUNS_STATE辅助，差别为回撤或相关方向准入；不新增A/B切换。此前5/20日二分类、原完整入场收益岭与IF概率幅度均有旧结果。",
                "novel_use_bounded_review": "所核对旧合同为固定期限涨跌、原自然入场终点收益、原退出继续收益或IF节点；此处使用逐原点首次边界类别及其实际扣费回报，成熟时间依赖已发生事件。不是原模型单字段残差，不复活旧分类版本；有限去重不声称穷尽全项目。",
                "features": learning.FEATURES,
                "features_definition": "现金前向平移价用于ATR和MACD；经济含息log收益用于动量和波动；日MACD12/26/9及EMA20、ATR14，量/此前20日中位量，5日有向量占比，RV20/此前252日中位；前完整周MACD12/26/9以周末之后才可用，EMA最小期数26/9，不使用原图谱130周暖机合同。",
                "labels": "决定原点次开入；初始开盘前向经济价下1ATR/上2ATR，ATR只取原点已知14日均TR；每日收盘首达或20个持有收盘到期，下一真实开盘出。十万元原价名义参考份额、压力摩擦/刻度/股息资格。类别不等同实际盈亏，缺后续开盘或未确认权益不成熟。",
                "training": {"schedule": "2015账户首个原点及每月第一个实际收盘", "window_origins": 756,
                             "minimum_mature_known_rows": 252, "minimum_each_class": 10,
                             "maturity": "标签退出与已有权益经济成熟都<=拟合原点；不补未来版本。",
                             "weights": "当前已成熟训练池的区间平均倒数并发数，归一为均值1；权重不是独立事件数。",
                             "estimator": "单一多项Logistic，lbfgs、C1、max_iter2000、tol1e-8；训练加权均值方差标准化、clip5；不收敛NO_VIEW不重试。",
                             "payoff_mapping": "同一成熟池的各类别实际正/负比例和平均正/负贡献，概率混合成预计实际p、B和净期望。",
                             "policies": POLICIES, "parameter_search": False},
                "entry": "有当时可用模型和完整特征，预计p×B>1且预计净期望>0，则空仓时次开尝试。预计pB仅模型门，最终用实际完成账户pB检验。对照仅改为相同成熟池类别频率，不使用技术特征。",
                "exit": "买入后固定实际开盘锚定1ATR/2ATR及20收盘，不随新模型移动目标；收盘触发次可卖开盘退出，风险只减仓，终点不伪造清仓。计划2R是本策略标签几何，不是用户通用门槛或已实现B。",
                "account": "两个独立现金账户时期，各20万元、510300.SH/CASH_CNY；原50%上限、ES5预算2.5%、负10%跳空预算5%和半数剩余回撤余量、DD10停止新入场。风险复用保存成熟值、252日、现金0、T+1/100份/.001刻度。",
                "periods": PERIODS, "costs": {"BASE": [.0002, .0005], "STRESS": [.0004, .001], "minimum_commission": 5},
                "primary": "TECHNICAL_LOGIT作为独立完整政策，相对原A和MATURE_FREQUENCY都比较；没有新来源合并和按结果切换。",
                "economic_gate": "全部两时期两费用：技术政策CAGR及全日历Sharpe都高于原A且高于频率对照并为正，DD<=10%，实际完成交易pB>1、标准期望>0。次数是软目标，未定义质量不填0或无穷。",
                "prediction_diagnostic": "所有当时可知预测的成熟事件logloss和Brier；另固定原账户首日每20原点一记录。只解释，不用预测误差门阻止完整账户运行。",
                "uncertainty": "完整真实账户配对20/252日循环区块各2000，沿用种子510300154；两种尺度全部报告，不选有利区间，不提供独立验证。",
                "necessary_tests_passed": 6, "original_control_replays": 4, "new_candidate_accounts_planned": 8,
                "history_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
                "first_vintage": "原日线与股息供应历史首版未认证；时钟正确不等于物理PIT认证。",
                "no_rescue": "固定窗口、类别、边界、特征、参数、预计门、退出、成本和时期均不据结果修改；原失败和前瞻不变。",
                "implementation_reference": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
                "reference_limit": "官方文档只支持多项分类、正则化和样本权重实现；新标签及交易规则为本地固定设计，来源不证明盈利。",
                "environment": {"sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__},
                "goal_achieved": False, "orders_authorized": False,
                "sources": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths]}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    print("唯一首次边界学习及完整账户政策已冻结，尚未派生真实标签、拟合或运行新账户。", flush=True)


def run():
    require(not (OUT / "RUN_STARTED.json").exists(), "固定学习和账户实验已经开始，不重启。")
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "冻结来源改变："+source["path"])
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_candidate_accounts_planned": 8}, exclusive=True)
    data, dividends, risks, parents = load()
    outcomes = learning.labels(data, dividends)
    fitted, forecasts = learning.forecast(data, outcomes)
    table("原点全部技术特征", data)
    table("原点首次边界参考结果", outcomes)
    table("全部事前模型预测", forecasts)
    for record in fitted:
        write_json(OUT / "models" / f"{record['fit_date']}.json", record)
    table("逐月成熟拟合摘要", pd.DataFrame({k: v for k, v in record.items() if k not in ["model", "training_origins", "class_counts"]} for record in fitted))
    rows, checks, years, comparisons, prediction_rows = [], [], [], [], []
    for period, (start, end) in PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        first = int(np.flatnonzero(data.date.ge(pd.Timestamp(start)))[0])-1
        actual = {}
        for cost in COSTS:
            saved = CONTROL / period / cost / "A_SAVED_WEIGHT"
            baseline = {"daily": pd.read_parquet(saved / "daily.parquet"), "trades": pd.read_parquet(saved / "trades.parquet"),
                        "orders": pd.read_parquet(saved / "orders.parquet")}
            baseline["terminal"] = {"stopped": bool(baseline["daily"].risk_stopped.iloc[-1])}
            base_stats = metrics(baseline)
            rows.append({"period": period, "cost": cost, "policy": "A_SAVED_WEIGHT", **base_stats})
            for policy in POLICIES:
                selected = forecasts.loc[forecasts.policy.eq(policy)].reset_index(drop=True).iloc[:len(local)]
                account = execution.account(local, dividends, parents[period], risks, selected, cost, start, "PASSAGE_ONLY")
                check = verify_account(account)
                checks.append({"period": period, "cost": cost, "policy": policy, **check})
                stat = metrics(account)
                yearly = annual_rows(account, period, policy, cost)
                full_counts = [r["completed_cycles"] for r in yearly if r["full_year"]]
                stat.update(average_full_year_cycles=float(np.mean(full_counts)), zero_trade_full_years=sum(n == 0 for n in full_counts))
                rows.append({"period": period, "cost": cost, "policy": policy, **stat})
                years.extend(yearly)
                for name in ["daily", "orders", "trades", "decisions", "rejections"]:
                    table(f"accounts/{period}/{cost}/{policy}/{name}", account[name])
                write_json(OUT / f"results/accounts/{period}/{cost}/{policy}/terminal.json", account["terminal"])
                actual[(cost, policy)] = (account, stat)
                print(f"{period}/{cost}/{policy}：净年化{stat['net_cagr']:.2%}，夏普{stat['net_sharpe']:.4f}，完整周期{stat['completed_cycles']}。", flush=True)
            main, main_stats = actual[(cost, "TECHNICAL_LOGIT")]
            frequency, freq_stats = actual[(cost, "MATURE_FREQUENCY")]
            passed = bool(main_stats["net_cagr"] > max(0., base_stats["net_cagr"], freq_stats["net_cagr"])
                          and main_stats["net_sharpe"] > max(0., base_stats["net_sharpe"], freq_stats["net_sharpe"])
                          and main_stats["max_drawdown"] <= .1 and main_stats["p_times_b"] > 1
                          and main_stats["standard_expectancy_loss_units"] > 0)
            comparison = {"period": period, "cost": cost, "economic_gate_passed": passed, "comparators": []}
            for name, other, stat in [("A_SAVED_WEIGHT", baseline, base_stats), ("MATURE_FREQUENCY", frequency, freq_stats)]:
                require(pd.DatetimeIndex(other["daily"].date).equals(pd.DatetimeIndex(main["daily"].date)), "对照日历不一致。")
                comparison["comparators"].append({"policy": name, "cagr_delta": main_stats["net_cagr"]-stat["net_cagr"],
                                                 "sharpe_delta": main_stats["net_sharpe"]-stat["net_sharpe"],
                                                 "ending_equity_delta": main_stats["ending_equity"]-stat["ending_equity"],
                                                 "intervals": intervals(other["daily"].net_return.to_numpy(), main["daily"].net_return.to_numpy())})
            comparisons.append(comparison)
        for policy in POLICIES:
            selected = forecasts.loc[forecasts.policy.eq(policy)].reset_index(drop=True)
            merged = selected.merge(outcomes[["origin_index", "status", "event_class", "mature_idx"]].rename(columns={"status": "outcome_status"}), on="origin_index", validate="one_to_one")
            eligible = merged.date.ge(pd.Timestamp(start)) & merged.date.le(pd.Timestamp(end)) & merged.status.eq("AVAILABLE") & merged.outcome_status.eq("MATURE_REFERENCE") & merged.mature_idx.le(len(local)-1)
            for group, mask in [("ALL_OVERLAPPING", eligible), ("FIXED_EVERY_TWENTY_ORIGINS", eligible & ((merged.origin_index-first)%20 == 0))]:
                sample = merged.loc[mask]
                probability = sample[["p_"+name for name in learning.CLASSES]].to_numpy(float)
                truth = np.array([[float(event == name) for name in learning.CLASSES] for event in sample.event_class])
                prediction_rows.append({"period": period, "policy": policy, "sample": group, "mature_forecasts": len(sample),
                                        "multiclass_logloss": float(-np.sum(truth*np.log(np.clip(probability, 1e-15, 1)), axis=1).mean()) if len(sample) else None,
                                        "multiclass_brier": float(np.sum((truth-probability)**2, axis=1).mean()) if len(sample) else None,
                                        "independent_sample_claim": False})
    table("完整账户共同口径比较", pd.DataFrame(rows))
    table("实际账户资金库存核对", pd.DataFrame(checks))
    table("逐年净收益与完整交易次数", pd.DataFrame(years))
    table("首次事件事前预测表现", pd.DataFrame(prediction_rows))
    economic = all(c["economic_gate_passed"] for c in comparisons)
    stable = all(i["cagr_delta_95"][0] > 0 and i["sharpe_delta_95"][0] > 0 for c in comparisons for r in c["comparators"] for i in r["intervals"])
    result = {"at": now(), "study": protocol["study"], "technical_decision": "TECH.R158",
              "status": "HISTORICAL_POINT_INCREMENT_NOT_INDEPENDENT" if economic and stable else "REJECTED_FIXED_FIRST_PASSAGE_LEARNING_FULL_ACCOUNT_GATE_FAILED",
              "new_actual_candidate_accounts": 8, "actual_original_control_replays": 4,
              "new_fitted_models": sum(r["status"] == "FIT_COMPLETE" for r in fitted), "monthly_fit_records": len(fitted),
              "derived_mature_reference_origins": int(outcomes.status.eq("MATURE_REFERENCE").sum()),
              "known_feature_origins": int(data.first_passage_feature_known.sum()),
              "forecast_status_counts": forecasts.groupby("policy").status.value_counts().to_dict(),
              "candidate_model_recipes": 1, "nonfeature_frequency_controls": 1,
              "all_four_economic_gates_passed": economic, "historical_stability_gate_passed": stable,
              "comparisons": comparisons, "new_market_requests": 0,
              "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
              "overfitting_removed": False, "goal_achieved": False, "old_rules_and_forward_protocols_preserved": True}
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "执行期间冻结来源变化。")
    write_json(OUT / "summary.json", result, exclusive=True)
    print("首次边界学习和八个完整候选账户已一次完成；独立验证和完整目标尚未成立。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="唯一首次边界学习及完整现金账户实验。")
    parser.add_argument("command", choices=("preflight", "freeze", "run"))
    args = parser.parse_args()
    {"preflight": preflight, "freeze": freeze, "run": run}[args.command]()
