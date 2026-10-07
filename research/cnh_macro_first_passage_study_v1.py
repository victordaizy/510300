"""先解释具体行情，再一次验证技术、宏观与不同人民币信息的完整账户。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from research import cnh_macro_first_passage_inputs_v1 as model
from research import macro_technical_first_passage_study_v1 as parent
from research import macro_technical_first_passage_clock_adapter_v1 as clock
from research.point_first_passage_study_v1 import read, write_json, digest, now, require
from research.point_account_nr7_inputs_v1 import metrics
from research.point_account_nr7_complement_v1 import verify_account, annual_rows
from research.point_directional_confirmation_study_v1 import intervals

original = parent.original
OUT = ROOT / "reports/research/510300_cnh_macro_first_passage_v1"
CARD = ROOT / "docs/510300_CNH_MACRO_FIRST_PASSAGE_V1.md"
FUNDING = ROOT / "reports/research/510300_macro_2026_source_coverage_intake_v1_clock_adapter/results/完整3488资金源派生_仅扩已有日历不拟合.parquet"
FX_SOURCE = ROOT / "reports/research/510300_cnh_fixing_deviation_source_preflight_v1"
FX = FX_SOURCE / "results/全部3488观察槽_离岸相对定盘偏离与保守钟.parquet"
DATA_NAME = "全部3488当时已知技术宏观与定盘偏离_未知保留"
FORECAST_NAME = "全部三模型事前预测与实际技术宏观外汇路径"
METRICS_NAME = "十六完整账户共同口径比较"
POINT_NAME = "全部实际进出点位与事前技术宏观外汇评分"


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def verify_sources():
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "冻结来源改变：" + source["path"])
    return protocol


def load_inputs():
    data, dividends, risks, parents = original.load()
    pd.testing.assert_frame_equal(data, pd.read_parquet(parent.BASE / "results/原点全部技术特征.parquet"), check_exact=True)
    require(read(parent.CORRECTION)["corrected_margin_sha256"] == digest(parent.MARGIN), "融资核实修正副本变化。")
    macro = model.parent.views(data, pd.read_parquet(parent.ORDERS), pd.read_parquet(FUNDING), pd.read_parquet(parent.MARGIN))
    return model.attach_fx(macro, pd.read_parquet(FX)), dividends, risks, parents


def prepare():
    require(not (OUT / "prepare_summary.json").exists() and not (OUT / "protocol.json").exists(), "同钟准备已经完成，不覆盖。")
    data, _, _, _ = load_inputs()
    table(DATA_NAME, data)
    cases = pd.read_parquet(parent.CASES)
    columns = ["date", "original_episode_id", "daily_dif", "daily_hist", "weekly_hist", "relative_volume", "rv_ratio"]
    joined = cases[columns].merge(data, on="date", how="left", validate="many_to_one")
    require(len(joined) == 240, "四原案例行数改变。")
    table("四原上涨反弹案例_全部同钟技术宏观与定盘背景", joined)
    selected = joined.loc[joined.date.isin(pd.to_datetime(parent.CASE_DATES))].copy()
    require(len(selected) == 17, "原固定关键日不完整。")
    table("原固定17关键日_先解释不评分", selected)
    waves = pd.read_parquet(parent.EPISODES)
    fields = ["date", "orders_reference_period", "orders_available_at", "fund_stat_date", "margin_stat_date",
              "fx_stat_date", "fx_available_at", *model.TECH, *model.MACRO, *model.FX, "joint_features_known"]
    aligned = waves.merge(data[fields].rename(columns={"date": "confirm_up_date"}), on="confirm_up_date", how="left", validate="many_to_one")
    table("原全部61分段及49正式上涨_确认当时16字段背景", aligned)
    coverage = data.assign(year=data.date.dt.year).groupby("year").agg(days=("date", "size"),
        technical_known=("first_passage_feature_known", "sum"), macro_known=("macro_joint_features_known", "sum"),
        fx_both_known=("fx_features_known", "sum"), joint_known=("joint_features_known", "sum")).reset_index()
    table("逐年真实信息覆盖_不缩样本", coverage)
    outcomes = pd.read_parquet(parent.BASE / "results/原点首次边界参考结果.parquet")
    support = [model.pool_record(data, model.common_pool(data, outcomes, index), index) for index in model.schedule(data)]
    write_json(OUT / "共同成熟支持_尚未拟合.json", support, exclusive=True)
    table("142月共同成熟支持_先核对不拟合", pd.DataFrame(
        {key: value for key, value in record.items() if key not in ["models", "training_origins", "training_fit_weights"]} for record in support))
    summary = {"at": now(), "status": "CONCRETE_EXPLANATION_AND_COMMON_MATURE_SUPPORT_COMPLETE_BEFORE_FIT",
        "all_daily_origins": len(data), "macro_joint_known": int(data.macro_joint_features_known.sum()),
        "fx_both_known": int(data.fx_features_known.sum()), "joint_known": int(data.joint_features_known.sum()),
        "monthly_records": len(support), "supported_months": sum(record["common_support_sufficient"] for record in support),
        "case_rows": len(joined), "key_days": len(selected), "episodes": len(waves), "admitted_waves": int(waves.admitted.sum()),
        "admitted_waves_joint_known_at_confirmation": int((aligned.admitted & aligned.joint_features_known.eq(True)).sum()),
        "new_financial_fits": 0, "new_accounts": 0, "new_labels": 0, "new_market_requests": 0}
    write_json(OUT / "prepare_summary.json", summary, exclusive=True)
    note = "# 具体上涨与共同成熟成员：拟合前解释\n\n"
    note += "先看原行情和可知背景，再提出固定评分。正MACD柱代表快慢均线差相对信号线的修复；周柱仅上一完整周。日柱变正而周柱仍负可以属于下跌中的修复，不能只凭日柱称趋势启动。放量表示参与活跃，是否持续由后续价格检验；全国融资、资金和订单不是510300逐笔买卖方向。\n\n"
    note += "2015原反弹处于日柱偏负/周线逐步恶化，给出多头失败对照；2019年初日柱修复可以领先上一周柱；2020春季多次修复须连同后续失败保留；2024年9月量价与日柱改善快于上一完整周柱。具体逐日值和可知宏观、人民币背景如下，尚未给这些日期拟合评分。\n\n"
    note += selected[["date", "daily_hist_atr", "weekly_hist_atr", "relative_volume", "log_rv_ratio", *model.MACRO, *model.FX, "fx_stat_date", "joint_features_known"]].to_markdown(index=False, floatfmt=".5f") + "\n\n"
    note += "人民币正偏离仅表示该离岸报价高于该官方参照价；报价和中间价不同时，不能推导同步市场压力或资金流。各字段重复使用同一期公布或同条报价不是独立事件。全部61分段/49正式上涨、四案例240行、信息缺失和142月训练支持均另表保留。\n\n"
    note += f"原日历{len(data)}，16字段同知{summary['joint_known']}，共同支持月{summary['supported_months']}/{len(support)}。三个模型只在同一组已成熟可知成员和同一权重上拟合；未知月份和现金日均保留。0新金融拟合、账户、股票标签、市场请求。\n"
    (OUT / "具体上涨与共同支持_拟合前.md").write_text(note, encoding="utf-8")
    print(f"拟合前解释完成：{len(data)}日，16字段同知{summary['joint_known']}，共同成熟月{summary['supported_months']}/{len(support)}。", flush=True)


def freeze():
    require(not (OUT / "protocol.json").exists(), "完整金融用途已登记，禁止覆盖。")
    support = read(OUT / "prepare_summary.json")
    tests = read(OUT / "tests_receipt.json")
    require(tests["passed"] == 5 and tests["exit_code"] == 0, "新接入五项必要测试未通过。")
    require(tests["inputs_sha256"] == digest(Path(model.__file__)), "新输入代码与测试不一致。")
    require(tests["test_sha256"] == digest(ROOT / "tests/test_cnh_macro_first_passage_v1.py"), "新测试源码改变。")
    paths = [Path(__file__), Path(model.__file__), CARD, ROOT / "tests/test_cnh_macro_first_passage_v1.py",
        OUT / "tests_receipt.json", OUT / "prepare_summary.json", OUT / "共同成熟支持_尚未拟合.json",
        OUT / "具体上涨与共同支持_拟合前.md", OUT / f"results/{DATA_NAME}.parquet",
        parent.ORDERS, FUNDING, parent.MARGIN, parent.CORRECTION, FX,
        FX_SOURCE / "protocol.json", FX_SOURCE / "summary.json",
        parent.CASES, parent.EPISODES, parent.BASE / "results/原点全部技术特征.parquet",
        parent.BASE / "results/原点首次边界参考结果.parquet", Path(model.parent.__file__), Path(parent.__file__), Path(clock.__file__),
        ROOT / "research/point_first_passage_inputs_v1.py", ROOT / "research/point_first_passage_account_v1.py",
        ROOT / "research/point_first_passage_study_v1.py", ROOT / "research/point_account_nr7_inputs_v1.py",
        ROOT / "research/point_account_nr7_complement_v1.py", ROOT / "research/point_directional_confirmation_study_v1.py",
        ROOT / "research/daily_supply_test_v1.py", original.CURRENT / "inputs/candidate_prices.parquet",
        original.WEIGHT / "inputs/dividends.csv", original.WEIGHT / "inputs/risks.parquet",
        original.WEIGHT / "inputs/parent_signals.parquet", original.WEIGHT / "inputs/earlier_signals.parquet",
        ROOT / "reports/research/510300_macro_funding_source_contract_v2/summary.json",
        ROOT / "reports/research/510300_rmb_residual_state_v1/result.json"]
    for period in original.PERIODS:
        for cost in original.COSTS:
            paths.extend(original.CONTROL / period / cost / "A_SAVED_WEIGHT" / (name + ".parquet") for name in ["daily", "orders", "trades"])
    sources = [{"path": path.absolute().relative_to(ROOT).as_posix(), "sha256": digest(path)} for path in dict.fromkeys(paths)]
    protocol = {"at": now(), "study": "510300_CNH_MACRO_FIRST_PASSAGE_V1", "decision": "TECH.R207",
        "authority": "用户加入不同信息源、多因子评分与既有宏观，允许新增隔离实验。",
        "complete_purpose": CARD.relative_to(ROOT).as_posix(), "features": model.FEATURES,
        "candidate_policy": model.POLICIES[2], "candidate_configurations": 1,
        "controls": ["A_SAVED_WEIGHT", *model.POLICIES[:2]],
        "model": {"depth": 3, "min_leaf": 60, "seed": model.SEED, "window": 756, "minimum_common_rows": 252, "each_class_minimum": 10},
        "pool": "三个模型完全相同的16字段已知成熟成员、重叠唯一性权重和类内净收益映射；原142月更新钟。",
        "labels": "原3488首达上2ATR/下1ATR/20收盘参考及原成熟钟，不新增或更改。",
        "entry": "估计pB>1且估计净期望>0；100乘估计净胜率是未校准评分。",
        "fx_source_clock": "R206同统计日配对，第二自然日23:59及原柱/定盘钟之最大值，源龄<=7，五槽差需连续六槽。",
        "account": "原200000元/50%仓位/原ES5与跳空预算/10%DD新买暂停、T+1/100股/.001档/分红和费用。",
        "periods": original.PERIODS, "costs": original.COSTS,
        "new_candidate_accounts_planned": 4, "new_matched_control_accounts_planned": 8, "original_A_exact_replays_planned": 4,
        "economic_gate": "四场景CAGR和Sharpe>0且严格高于A和两同池控制，DD<=.1、实际净pB>1、标准期望>0。",
        "frequency": "完整逐年，软目标，无年度配额。",
        "bootstrap": {"blocks": [20, 252], "resamples_each": 2000, "seed": 510300154, "comparators_each_scene": 3},
        "pre_fit_explanation_and_support": support, "necessary_new_tests_passed": 5,
        "parent_eight_tests_previously_passed_not_rerun": True,
        "evidence_role": "DEVELOPMENT_CALIBRATION", "first_vintage": "NOT_CERTIFIED",
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "overfitting_removed": False, "goal_achieved": False, "orders_authorized": False,
        "no_rescue": "失败关闭，不按结果调参数、样本、符号、字段、费用、退出或拼接分支。原拒绝及前瞻保持。",
        "sources": sources}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    print("TECH.R207完整用途及共同成熟支持已固定，金融拟合尚未开始。", flush=True)


def run():
    protocol = verify_sources()
    require(not (OUT / "RUN_STARTED.json").exists(), "金融入口已经开始，禁止重跑。")
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "one_candidate_configuration": True}, exclusive=True)
    data = pd.read_parquet(OUT / f"results/{DATA_NAME}.parquet")
    _, dividends, risks, parents = original.load()
    outcomes = pd.read_parquet(parent.BASE / "results/原点首次边界参考结果.parquet")
    control_checks = []
    for period, (start, end) in original.PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        empty = pd.DataFrame({"date": local.date, "entry_event": False, "event_id": None,
                              "atr": np.nan, "stop_index": np.nan, "target_index": np.nan})
        for cost in original.COSTS:
            actual = clock.account(local, dividends, parents[period], risks, empty, cost, start, "A_CONTROL")
            saved = parent.saved_baseline(period, cost)
            pd.testing.assert_frame_equal(actual["daily"], saved["daily"], check_exact=True)
            pd.testing.assert_frame_equal(actual["orders"][saved["orders"].columns], saved["orders"], check_exact=True)
            pd.testing.assert_frame_equal(actual["trades"][saved["trades"].columns], saved["trades"], check_exact=True)
            control_checks.append({"period": period, "cost": cost, **verify_account(actual)})
            print(f"原A精确复现：{period}/{cost}。", flush=True)
    write_json(OUT / "control_preflight.json", {"at": now(), "status": "PASS_FOUR_A_CONTROLS_EXACT", "checks": control_checks}, exclusive=True)
    records, forecasts = model.forecast(data, outcomes)
    prior = read(OUT / "共同成熟支持_尚未拟合.json")
    require(len(records) == len(prior), "拟合时更新时间数改变。")
    for before, fitted in zip(prior, records):
        require(before["training_origins"] == fitted["training_origins"] and before["training_fit_weights"] == fitted["training_fit_weights"], "训练成员或权重与先行共同池不符。")
    write_json(OUT / "saved_models.json", records, exclusive=True)
    table(FORECAST_NAME, forecasts)
    table("142月实际共同训练支持_未知保留", pd.DataFrame(
        {key: value for key, value in record.items() if key not in ["models", "training_origins", "training_fit_weights"]} for record in records))
    rows, checks, years, comparisons, points = [], [], [], [], []
    for period, (start, end) in original.PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        for cost in original.COSTS:
            baseline = parent.saved_baseline(period, cost)
            baseline_stats = metrics(baseline)
            baseline_years = annual_rows(baseline, period, "A_SAVED_WEIGHT", cost)
            counts = [row["completed_cycles"] for row in baseline_years if row["full_year"]]
            baseline_stats.update(average_full_year_cycles=float(np.mean(counts)), zero_trade_full_years=sum(count == 0 for count in counts))
            rows.append({"period": period, "cost": cost, "policy": "A_SAVED_WEIGHT", **baseline_stats})
            years.extend(baseline_years)
            accounts = {}
            for policy in model.POLICIES:
                all_selected = forecasts.loc[forecasts.policy.eq(policy)].reset_index(drop=True)
                require(np.array_equal(all_selected.origin_index.to_numpy(), np.arange(len(data))), "评分原点与账户日历错位。")
                selected = all_selected.iloc[:len(local)].copy()
                account = clock.account(local, dividends, parents[period], risks, selected, cost, start, "PASSAGE_ONLY")
                checks.append({"period": period, "cost": cost, "policy": policy, **verify_account(account)})
                stat = metrics(account)
                yearly = annual_rows(account, period, policy, cost)
                counts = [row["completed_cycles"] for row in yearly if row["full_year"]]
                stat.update(average_full_year_cycles=float(np.mean(counts)), zero_trade_full_years=sum(count == 0 for count in counts))
                rows.append({"period": period, "cost": cost, "policy": policy, **stat})
                years.extend(yearly)
                for name in ["daily", "orders", "trades", "decisions", "rejections"]:
                    table(f"accounts/{period}/{cost}/{policy}/{name}", account[name])
                write_json(OUT / f"results/accounts/{period}/{cost}/{policy}/terminal.json", account["terminal"], exclusive=True)
                accounts[policy] = (account, stat)
                for trade in account["trades"].to_dict("records"):
                    origin = int(data.index[data.date.eq(trade["entry_origin"])][0])
                    score = selected.iloc[origin].to_dict()
                    score_fields = ["score", "fit_index", "predicted_win_probability", "predicted_payoff", "predicted_p_times_b",
                                    "predicted_net_expectation", "technical_features_on_path", "macro_features_on_path", "fx_features_on_path"]
                    background = model.TECH + model.MACRO + model.FX + ["orders_reference_period", "orders_available_at", "fund_stat_date",
                        "margin_stat_date", "weekly_last_date", "fx_stat_date", "fx_available_at", "cnh_bid_close", "fixing_value"]
                    points.append({"period": period, "cost": cost, "policy": policy, **trade,
                                   **{key: score[key] for key in score_fields}, **{key: data.iloc[origin][key] for key in background}})
                print(f"{period}/{cost}/{policy}：净年化{stat['net_cagr']:.3%}、夏普{stat['net_sharpe']:.4f}、完成{stat['completed_cycles']}。", flush=True)
            main, main_stats = accounts[model.POLICIES[2]]
            others = [("A_SAVED_WEIGHT", baseline, baseline_stats),
                      *((policy, *accounts[policy]) for policy in model.POLICIES[:2])]
            economic = bool(main_stats["net_cagr"] > max(0., *(stat["net_cagr"] for _, _, stat in others))
                and main_stats["net_sharpe"] > max(0., *(stat["net_sharpe"] for _, _, stat in others))
                and main_stats["max_drawdown"] <= .1 and main_stats["p_times_b"] > 1.
                and main_stats["standard_expectancy_loss_units"] > 0.)
            comparison = {"period": period, "cost": cost, "economic_gate_passed": economic, "comparators": []}
            for name, other, stat in others:
                require(pd.DatetimeIndex(other["daily"].date).equals(pd.DatetimeIndex(main["daily"].date)), "比较日历不一致。")
                comparison["comparators"].append({"policy": name,
                    "cagr_delta": main_stats["net_cagr"] - stat["net_cagr"],
                    "sharpe_delta": main_stats["net_sharpe"] - stat["net_sharpe"],
                    "intervals": intervals(other["daily"].net_return.to_numpy(), main["daily"].net_return.to_numpy())})
            comparisons.append(comparison)
    table(METRICS_NAME, pd.DataFrame(rows))
    table("十二新账户资金与库存核对", pd.DataFrame(checks))
    table("逐年净收益与真实交易次数", pd.DataFrame(years))
    table(POINT_NAME, pd.DataFrame(points))
    economic = all(row["economic_gate_passed"] for row in comparisons)
    stable = all(interval["cagr_delta_95"][0] > 0 and interval["sharpe_delta_95"][0] > 0
                 for row in comparisons for other in row["comparators"] for interval in other["intervals"])
    wide = forecasts.pivot(index="origin_index", columns="policy", values="predicted_net_expectation")
    known = wide.notna().all(axis=1)
    candidate_forecasts = forecasts.loc[forecasts.policy.eq(model.POLICIES[2]) & forecasts.status.eq("AVAILABLE")]
    difference = {policy: int((~np.isclose(wide.loc[known, policy], wide.loc[known, model.POLICIES[2]], rtol=1e-12, atol=1e-14)).sum())
                  for policy in model.POLICIES[:2]}
    summary = {"at": now(), "study": protocol["study"], "decision": "TECH.R208",
        "status": "HISTORICAL_INCREMENT_NOT_INDEPENDENT" if economic and stable else "REJECTED_FIXED_CNH_MACRO_FIRST_PASSAGE_FULL_ACCOUNT_GATE_FAILED",
        "candidate_configurations": 1, "monthly_fit_records": len(records),
        "paired_months_fitted": sum(record["status"] == "FIT_COMPLETE" for record in records),
        "actual_fit_calls": 3 * sum(record["status"] == "FIT_COMPLETE" for record in records),
        "new_candidate_accounts": 4, "new_matched_control_accounts": 8, "new_accounts": 12, "original_A_exact_replays": 4,
        "common_available_forecasts_per_policy": int(known.sum()), "different_expectation_origins": difference,
        "actual_fx_path_origins": int(candidate_forecasts.fx_features_on_path.ne("").sum()),
        "actual_macro_path_origins": int(candidate_forecasts.macro_features_on_path.ne("").sum()),
        "all_four_economic_gates_passed": economic, "historical_stability_gate_passed": stable,
        "comparisons": comparisons, "new_labels": 0, "new_market_requests": 0, "parameter_grids": 0,
        "first_vintage": "NOT_CERTIFIED", "evidence_role": "DEVELOPMENT_CALIBRATION",
        "independent_validation": "NOT_ESTABLISHED", "global_DSR_PBO": "NOT_COMPUTED",
        "overfitting_removed": False, "goal_achieved": False, "orders_authorized": False}
    verify_sources()
    write_json(OUT / "summary.json", summary, exclusive=True)
    print("TECH.R208：唯一配置、12新完整账户、4原A复现及全部24组固定区块区间完成。", flush=True)


def deliver():
    verify_sources()
    require(not (OUT / "delivery_receipt.json").exists(), "保存结果已交付，不重复。")
    summary = read(OUT / "summary.json")
    data = pd.read_parquet(OUT / f"results/{DATA_NAME}.parquet")
    forecasts = pd.read_parquet(OUT / f"results/{FORECAST_NAME}.parquet")
    stats = pd.read_parquet(OUT / f"results/{METRICS_NAME}.parquet")
    points = pd.read_parquet(OUT / f"results/{POINT_NAME}.parquet")
    checks = []
    for period in original.PERIODS:
        for cost in original.COSTS:
            for policy in model.POLICIES:
                path = OUT / f"results/accounts/{period}/{cost}/{policy}"
                account = {name: pd.read_parquet(path / (name + ".parquet")) for name in ["daily", "orders", "trades", "decisions", "rejections"]}
                account["terminal"] = read(path / "terminal.json")
                checks.append({"period": period, "cost": cost, "policy": policy, **verify_account(account)})
                saved = stats.loc[stats.period.eq(period) & stats.cost.eq(cost) & stats.policy.eq(policy)].iloc[0]
                recalculated = metrics(account)
                for key in ["net_cagr", "net_sharpe", "max_drawdown", "p_times_b", "ending_equity"]:
                    require(np.isclose(saved[key], recalculated[key], rtol=1e-12, atol=1e-12, equal_nan=True), "保存账户指标复算不一致。")
    write_json(OUT / "saved_result_verification.json", {"at": now(), "status": "PASS_TWELVE_SAVED_ACCOUNTS_AND_METRICS", "checks": checks}, exclusive=True)
    cases = pd.read_parquet(OUT / "results/四原上涨反弹案例_全部同钟技术宏观与定盘背景.parquet")
    forecast_fields = ["date", "policy", "status", "score", "entry_event", "predicted_p_times_b", "predicted_net_expectation",
                       "macro_features_on_path", "fx_features_on_path"]
    case_scores = cases.merge(forecasts[forecast_fields], on="date", how="left", validate="one_to_many")
    table("四原案例_同钟背景与全部三模型评分", case_scores)
    key_scores = case_scores.loc[case_scores.date.isin(pd.to_datetime(parent.CASE_DATES))].copy()
    table("原固定17关键日_三模型分数与实际路径", key_scores)
    charts = make_charts(cases, forecasts)
    candidate = forecasts.loc[forecasts.policy.eq(model.POLICIES[2]) & forecasts.status.eq("AVAILABLE")]
    usage = {name: int(candidate.fx_features_on_path.str.split("|").map(lambda fields: name in fields).sum()) for name in model.FX}
    report = "# 技术、宏观与人民币参照偏离：具体点位和完整账户结果\n\n"
    report += "已经把不同人民币信息接入既有量价/MACD/波动率及宏观，先解释原案例，再一次运行固定用途。纯技术8、技术宏观14、技术宏观人民币16字段三模型使用完全相同的成熟成员和权重；因此同池增量来自输入字段差异。原A为保留的原策略账户。\n\n"
    report += "人民币信息来自离岸USD-CNH UTC日BID末价和同统计日官方USD-CNY中间价，实际报价时刻不同。两个对数基点字段不是独立赞成票、同步市场基差、真实资金流或外生意外。PMI按发布钟，资金和融资按滞后合同，周线只用上一完整周。2026融资等缺失保留，完整账户包含全部现金日。历史首版和独立验证均未认证。\n\n"
    report += f"3488原观察槽，16字段同知{int(data.joint_features_known.sum())}；142月中{summary['paired_months_fitted']}有共同成熟支持，实际拟合{summary['actual_fit_calls']}次，共同可评分每模型{summary['common_available_forecasts_per_policy']}。候选相对两个控制预测改变{summary['different_expectation_origins']}；实际外汇路径{summary['actual_fx_path_origins']}原点，各字段路径计数{usage}。重复月度模型、重复报价和唯一性权重都不等于独立样本。\n\n"
    columns = ["period", "cost", "policy", "net_cagr", "net_sharpe", "max_drawdown", "completed_cycles", "win_rate", "payoff", "p_times_b", "standard_expectancy_loss_units", "average_full_year_cycles", "ending_equity"]
    report += stats[columns].to_markdown(index=False, floatfmt=".5f") + "\n\n"
    report += f"四经济门通过{sum(row['economic_gate_passed'] for row in summary['comparisons'])}/4；全部经济门{summary['all_four_economic_gates_passed']}、历史稳定门{summary['historical_stability_gate_passed']}。三个控制、两时期、两费用、20/252两区块的全部24组区间在summary.json，不选择好场景。\n\n"
    report += "实际pB按已完成净收益统计；估计pB和分数均不是实现收益。少数全赢没有亏损均值，B和pB不可估，不能按无穷大利润晋升。开放周期另存，不计已完成勝率。交易频率为完整年退出计数，2026不完整年度单列。\n\n"
    report += key_scores[["date", "policy", "daily_hist_atr", "weekly_hist_atr", "relative_volume", *model.FX, "score", "predicted_p_times_b", "entry_event", "fx_features_on_path"]].to_markdown(index=False, floatfmt=".5f") + "\n\n"
    report += f"全部三模型、两费用、两时期实际周期记录{len(points)}行，保存12新账户复算通过，原A4精确复现，必要新接入测试5项通过；0新股票标签/市场请求/原策略改动。DEVELOPMENT_CALIBRATION、first vintage NOT_CERTIFIED、independent NOT_ESTABLISHED、global DSR/PBO NOT_COMPUTED；去除过拟合和完整收益夏普目标未建立。\n\n"
    report += "[拟合前具体解释](具体上涨与共同支持_拟合前.md)、[全部实际点位](results/全部实际进出点位与事前技术宏观外汇评分.csv)、[全部完整账户](results/十六完整账户共同口径比较.csv)、[逐年次数与收益](results/逐年净收益与真实交易次数.csv)、[四案例720评分行](results/四原案例_同钟背景与全部三模型评分.csv)。\n\n"
    report += ("本固定用途失败并关闭；接受不同信息已接入和同池可比较的事实，拒绝其提升完整账户的假设。失败不等于全部宏观/外汇机制失效。仅在实质不同的信息机制/完整用途或真正新样本成立后另登记，不据本结果调树、窗口、打分线、符号、退出或拼接策略救援。\n" if not (summary["all_four_economic_gates_passed"] and summary["historical_stability_gate_passed"]) else "只支持有限历史开发增量，尚不能称独立成功或去除过拟合。后续必须另登记真正独立检验，不改现有前瞻，不直接交易。\n")
    (OUT / "不同信息多因子评分_全部结论与点位.md").write_text(report, encoding="utf-8")
    write_json(OUT / "delivery_receipt.json", {"at": now(), "status": "PASS_SAVED_DELIVERY", "saved_account_checks": 12,
        "case_rows_before_score": len(cases), "case_score_rows": len(case_scores), "key_day_score_rows": len(key_scores),
        "charts": charts, "fx_path_usage": usage, "report": (OUT / "不同信息多因子评分_全部结论与点位.md").relative_to(ROOT).as_posix(),
        "new_accounts_rerun": 0, "new_fits": 0, "new_labels": 0, "orders_authorized": False}, exclusive=True)
    print("完整结论、原四案例、51关键日评分、全部点位及12保存账户复算已交付。", flush=True)


def make_charts(cases, forecasts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    names = dict(zip(model.POLICIES, ["同池技术8", "同池技术宏观14", "技术宏观人民币16"]))
    charts = []
    for case_id, case in cases.groupby("original_episode_id", sort=True):
        fig, axes = plt.subplots(6, 1, figsize=(14, 15), sharex=True, constrained_layout=True)
        axes[0].plot(case.date, case.close, color="#153e65", label="原收盘价")
        axes[0].set_title(f"原案例{int(case_id)}：量价和日周MACD → 可知宏观与人民币 → 三模型固定评分")
        axes[1].plot(case.date, case.daily_hist_atr, label="日MACD柱/ATR")
        axes[1].plot(case.date, case.weekly_hist_atr, label="上个完整周柱/ATR")
        axes[2].plot(case.date, case.log_relative_volume, label="相对量对数")
        axes[2].plot(case.date, case.log_rv_ratio, label="波动率比对数")
        axes[3].step(case.date, case.pmi_orders_level, where="post", label="已发布订单减50/点")
        axes[3].plot(case.date, case.funding_gap_pp, label="资金政策利差/百分点")
        axes[3].plot(case.date, 100 * case.financing_net_change5, label="前日可知融资五日变化/%")
        axes[4].plot(case.date, case.fixing_deviation_log_bp, label="离岸相对定盘偏离/对数基点")
        axes[4].plot(case.date, case.fixing_deviation_change5_log_bp, label="五ETF槽偏离变化/对数基点")
        for policy in model.POLICIES:
            sub = case[["date"]].merge(forecasts.loc[forecasts.policy.eq(policy), ["date", "score"]], on="date", validate="one_to_one")
            axes[5].plot(sub.date, sub.score, label=names[policy] + "：估计净胜率×100")
        axes[5].set_ylim(0, 100)
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend(loc="upper left", fontsize=8, ncol=3)
        path = OUT / f"原案例{int(case_id)}_量价宏观人民币与三评分.png"
        fig.savefig(path, dpi=125)
        plt.close(fig)
        charts.append(path.relative_to(ROOT).as_posix())
    return charts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="日周线、宏观和定盘偏离的唯一同池完整检验")
    parser.add_argument("action", choices=["prepare", "freeze", "run", "deliver"])
    args = parser.parse_args()
    {"prepare": prepare, "freeze": freeze, "run": run, "deliver": deliver}[args.action]()
