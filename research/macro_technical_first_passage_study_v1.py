"""先解释既有上涨的宏观背景，再一次测量联合非线性点位和完整账户。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from research import macro_technical_first_passage_inputs_v1 as model
from research import point_first_passage_study_v1 as original
from research import point_first_passage_account_v1 as execution
from research.point_first_passage_study_v1 import read, write_json, digest, now, require
from research.point_account_nr7_inputs_v1 import metrics
from research.point_account_nr7_complement_v1 import verify_account, annual_rows
from research.point_directional_confirmation_study_v1 import intervals

OUT = ROOT / "reports/research/510300_macro_technical_first_passage_v1"
BASE = ROOT / "reports/research/510300_point_first_passage_study_v1"
CARD = ROOT / "docs/510300_MACRO_TECHNICAL_FIRST_PASSAGE_V1.md"
ORDERS = ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/pmi_new_orders.parquet"
FUNDING = ROOT / "reports/research/510300_factor96_funding_relief_v1/daily_features_lag1.parquet"
CORRECTION = ROOT / "reports/research/510300_multidim_financing_composition_v1/data_correction/correction_receipt.json"
MARGIN = ROOT / "reports/research/510300_multidim_financing_composition_v1/data_correction/margin_corrected_20240808.parquet"
CASES = ROOT / "reports/research/510300_volume_lead_price_confirm_explanation_v1/results/四原案例逐日全部量价及量先恢复状态.parquet"
EPISODES = ROOT / "reports/research/510300_upward_episode_anatomy_v1/results/上涨段全集.parquet"
CASE_DATES = ["2015-06-24", "2015-06-25", "2015-06-29", "2015-06-30", "2019-01-08", "2019-01-14", "2019-01-18",
              "2020-03-27", "2020-04-01", "2020-04-23", "2020-05-28", "2020-05-29", "2020-06-08",
              "2024-09-23", "2024-09-24", "2024-09-26", "2024-09-30"]


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def verify_sources():
    protocol = read(OUT / "protocol.json")
    for source in protocol["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "冻结来源变化：" + source["path"])
    return protocol


def load_inputs():
    data, dividends, risks, parents = original.load()
    saved = pd.read_parquet(BASE / "results/原点全部技术特征.parquet")
    pd.testing.assert_frame_equal(data, saved, check_exact=True)
    correction = read(CORRECTION)
    require(correction["corrected_margin_sha256"] == digest(MARGIN), "未使用已核实的融资修复副本。")
    joint = model.views(data, pd.read_parquet(ORDERS), pd.read_parquet(FUNDING), pd.read_parquet(MARGIN))
    return joint, dividends, risks, parents


def freeze():
    require(not (OUT / "protocol.json").exists(), "联合用途已经登记，禁止覆盖。")
    tests = read(OUT / "tests_receipt.json")
    require(tests["passed"] == 8 and tests["exit_code"] == 0, "八项必要验证未通过。")
    require(tests["inputs_sha256"] == digest(Path(model.__file__)), "测试对应的输入代码变化。")
    paths = [Path(__file__), Path(model.__file__), CARD, ROOT / "tests/test_macro_technical_first_passage_v1.py",
             OUT / "tests_receipt.json", ORDERS, FUNDING, MARGIN, CORRECTION, CASES, EPISODES,
             BASE / "results/原点全部技术特征.parquet", BASE / "results/原点首次边界参考结果.parquet",
             ROOT / "research/point_first_passage_inputs_v1.py", ROOT / "research/point_first_passage_account_v1.py",
             ROOT / "research/point_first_passage_study_v1.py", ROOT / "research/point_account_nr7_inputs_v1.py",
             ROOT / "research/point_account_nr7_complement_v1.py", ROOT / "research/point_directional_confirmation_study_v1.py",
             ROOT / "research/daily_supply_test_v1.py", original.CURRENT / "inputs/candidate_prices.parquet",
             original.WEIGHT / "inputs/dividends.csv", original.WEIGHT / "inputs/risks.parquet",
             original.WEIGHT / "inputs/parent_signals.parquet", original.WEIGHT / "inputs/earlier_signals.parquet"]
    reviews = []
    for stem, difference in [
        ("conditional_score_policy_v1", "旧综合账户评分用途，不是本次共同池首次边界三分类；原账户失败保持。"),
        ("multidim_nonlinear_score_v1", "旧八变量树预测固定五日净收益，本次直接复用首次边界三类标签及原完整退出。"),
        ("multidim_policy_transmission_score_v1", "旧十变量五日回报用途不重跑或改阈值。"),
        ("multidim_money_surprise_score_v1", "旧月度货币公布事件，与本次每个已知日收盘的首次边界用途不同。"),
        ("integrated_macro_micro_prediction_v1", "旧82模型预测五日回报，本次标签、技术母集及退出保持原首次边界用途。")]:
        p = ROOT / ("reports/research/510300_" + stem) / "result.json"
        result = read(p)
        reviews.append({"path": p.relative_to(ROOT).as_posix(), "status": result.get("status", result.get("state")), "distinction": difference})
        paths.append(p)
    for period in original.PERIODS:
        for cost in original.COSTS:
            for name in ["daily", "orders", "trades"]:
                paths.append(original.CONTROL / period / cost / "A_SAVED_WEIGHT" / (name + ".parquet"))
    sources = []
    for p in dict.fromkeys(paths):
        logical = ROOT / p.absolute().relative_to(ROOT)
        sources.append({"path": logical.relative_to(ROOT).as_posix(), "sha256": digest(logical)})
    write_json(OUT / "protocol.json", {"at": now(), "study": "510300_MACRO_TECHNICAL_FIRST_PASSAGE_V1", "decision": "TECH.R191",
        "authority": "用户明确加入不同信息源、多因子打分和原宏观分析；允许隔离实验，原策略及原前瞻不改。",
        "complete_purpose": CARD.relative_to(ROOT).as_posix(), "technical_features": model.TECH, "macro_features": model.MACRO,
        "candidate_configuration": 1, "new_model_family": "固定三分类树，深度3/每叶60/种子510300191",
        "paired_control": "同日期、同共同成熟训练池、同唯一性权重的纯技术树",
        "schedule": "原142个月度更新时间，756日窗口，至少252共同成熟、每类至少10",
        "label": "复用原3488条首次边界标签；不新增标签，不改变2ATR/1ATR/20收盘退出",
        "score": "100乘估计净胜率，不是已校准概率；进场沿用估计pB>1且估计净期望>0",
        "periods": original.PERIODS, "costs": original.COSTS, "new_candidate_accounts_planned": 4,
        "new_matched_technical_accounts_planned": 4, "original_A_replays_planned": 4,
        "bootstrap": {"blocks": [20, 252], "resamples_each": 2000, "seed": 510300154, "all_comparisons_retained": True},
        "old_uses_finite_review": reviews, "finite_review_not_global_novelty": True,
        "evidence_role": "DEVELOPMENT_CALIBRATION", "first_vintage": "NOT_CERTIFIED", "independent_validation": "NOT_ESTABLISHED",
        "overfitting_removed": False, "goal_achieved": False, "orders_authorized": False, "sources": sources}, exclusive=True)
    print("宏观联合首次边界用途已固定；尚未训练或运行新账户。", flush=True)


def explain():
    verify_sources()
    require(not (OUT / "explanation_summary.json").exists(), "具体解释已经完成，不重复。")
    data, _, _, _ = load_inputs()
    table("全部3488当时已知技术与宏观_未知保留", data)
    cases = pd.read_parquet(CASES)
    keep = ["date", "original_episode_id", "daily_dif", "daily_hist", "weekly_hist", "relative_volume", "rv_ratio"]
    joined = cases[keep].merge(data, on="date", how="left", validate="many_to_one")
    table("四原上涨反弹案例_全部同钟宏观与技术", joined)
    selected = joined.loc[joined.date.isin(pd.to_datetime(CASE_DATES))].copy()
    table("事前固定17关键日期_宏观与技术", selected)
    waves = pd.read_parquet(EPISODES)
    fields = ["date", "orders_reference_period", "orders_available_at", *model.MACRO, "joint_features_known"]
    aligned = waves.merge(data[fields].rename(columns={"date": "confirm_up_date"}), on="confirm_up_date", how="left", validate="many_to_one")
    table("原全部61分段及49正式上涨_确认当时宏观状态", aligned)
    coverage = data.assign(year=data.date.dt.year).groupby("year").agg(days=("date", "size"),
        orders_known=("orders_known", "sum"), funding_known=("funding_known", "sum"),
        margin_known=("margin_known", "sum"), joint_features_known=("joint_features_known", "sum")).reset_index()
    table("逐年真实信息覆盖_不缩样本", coverage)
    summary = {"at": now(), "status": "CONCRETE_CASE_AND_ALL_WAVE_MACRO_CLOCK_EXPLANATION_COMPLETE_BEFORE_FITTING",
               "known_joint_origins": int(data.joint_features_known.sum()), "all_daily_origins": len(data),
               "case_rows": len(joined), "all_episode_rows": len(waves), "admitted_waves": int(waves.admitted.sum()),
               "admitted_waves_macro_known_at_confirmation": int((aligned.admitted & aligned.joint_features_known.eq(True)).sum()),
               "joint_first": data.loc[data.joint_features_known, "date"].min(),
               "joint_last": data.loc[data.joint_features_known, "date"].max(), "new_fits": 0, "new_accounts": 0,
               "source_vintage_limit": "PMI和融资为历史回取；融资已用已核实纠正；不认证当年首次版本。",
               "causal_limit": "宏观状态与上涨的共同观察不证明因果，也不能把全国宏观或汇总融资当510300资金流。"}
    write_json(OUT / "explanation_summary.json", summary, exclusive=True)
    print(f"先行解释完成：{len(data)}原点，联合信息已知{summary['known_joint_origins']}，原61分段/49上涨全保留；新拟合0。", flush=True)


def saved_baseline(period, cost):
    path = original.CONTROL / period / cost / "A_SAVED_WEIGHT"
    result = {name: pd.read_parquet(path / (name + ".parquet")) for name in ["daily", "orders", "trades"]}
    result["terminal"] = {"stopped": bool(result["daily"].risk_stopped.iloc[-1])}
    return result


def run():
    protocol = verify_sources()
    require((OUT / "explanation_summary.json").exists(), "必须先解释具体上涨和所有同钟状态。")
    require(not (OUT / "RUN_STARTED.json").exists(), "本实验已经开始，不能重新拟合或重跑账户。")
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "one_fixed_candidate": True}, exclusive=True)
    data = pd.read_parquet(OUT / "results/全部3488当时已知技术与宏观_未知保留.parquet")
    _, dividends, risks, parents = original.load()
    outcomes = pd.read_parquet(BASE / "results/原点首次边界参考结果.parquet")
    control_checks = []
    for period, (start, end) in original.PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        empty = pd.DataFrame({"date": local.date, "entry_event": False, "event_id": None,
                              "atr": np.nan, "stop_index": np.nan, "target_index": np.nan})
        for cost in original.COSTS:
            actual = execution.account(local, dividends, parents[period], risks, empty, cost, start, "A_CONTROL")
            saved = saved_baseline(period, cost)
            pd.testing.assert_frame_equal(actual["daily"], saved["daily"], check_exact=True)
            pd.testing.assert_frame_equal(actual["orders"][saved["orders"].columns], saved["orders"], check_exact=True)
            control_checks.append({"period": period, "cost": cost, **verify_account(actual)})
            print(f"原A精确复现：{period}/{cost}。", flush=True)
    write_json(OUT / "control_preflight.json", {"status": "PASS_FOUR_A_CONTROLS_EXACT", "checks": control_checks}, exclusive=True)
    records, forecasts = model.forecast(data, outcomes)
    write_json(OUT / "saved_models.json", records, exclusive=True)
    table("全部事前配对模型预测与实际宏观路径", forecasts)
    table("142月共同训练支持_未知保留", pd.DataFrame({k: v for k, v in r.items() if k not in ["models", "training_origins"]} for r in records))
    rows, checks, years, comparisons, point_rows = [], [], [], [], []
    for period, (start, end) in original.PERIODS.items():
        local = data.loc[data.date.le(end)].reset_index(drop=True)
        for cost in original.COSTS:
            base = saved_baseline(period, cost)
            base_stats = metrics(base)
            rows.append({"period": period, "cost": cost, "policy": "A_SAVED_WEIGHT", **base_stats})
            accounts = {}
            for policy in model.POLICIES:
                selected = forecasts.loc[forecasts.policy.eq(policy)].reset_index(drop=True).iloc[:len(local)]
                account = execution.account(local, dividends, parents[period], risks, selected, cost, start, "PASSAGE_ONLY")
                checks.append({"period": period, "cost": cost, "policy": policy, **verify_account(account)})
                stat = metrics(account)
                yearly = annual_rows(account, period, policy, cost)
                full_counts = [r["completed_cycles"] for r in yearly if r["full_year"]]
                stat.update(average_full_year_cycles=float(np.mean(full_counts)), zero_trade_full_years=sum(n == 0 for n in full_counts))
                rows.append({"period": period, "cost": cost, "policy": policy, **stat})
                years.extend(yearly)
                for name in ["daily", "orders", "trades", "decisions", "rejections"]:
                    table(f"accounts/{period}/{cost}/{policy}/{name}", account[name])
                write_json(OUT / f"results/accounts/{period}/{cost}/{policy}/terminal.json", account["terminal"])
                accounts[policy] = (account, stat)
                for trade in account["trades"].to_dict("records"):
                    origin = int(data.index[data.date.eq(trade["entry_origin"])][0])
                    score = selected.iloc[origin].to_dict()
                    point_rows.append({"period": period, "cost": cost, "policy": policy, **trade,
                                       **{k: score[k] for k in ["score", "fit_index", "predicted_win_probability", "predicted_payoff", "predicted_p_times_b", "predicted_net_expectation", "macro_features_on_path"]},
                                       **{k: data.iloc[origin][k] for k in model.MACRO + ["orders_reference_period", "orders_available_at", "fund_stat_date", "margin_stat_date", "weekly_last_date"]}})
                print(f"{period}/{cost}/{policy}：净年化{stat['net_cagr']:.3%}，夏普{stat['net_sharpe']:.4f}，完成{stat['completed_cycles']}。", flush=True)
            main, ms = accounts["MACRO_TECH_TREE"]
            tech, ts = accounts["TECH_COMMON_TREE"]
            passed = bool(ms["net_cagr"] > max(0., base_stats["net_cagr"], ts["net_cagr"])
                          and ms["net_sharpe"] > max(0., base_stats["net_sharpe"], ts["net_sharpe"])
                          and ms["max_drawdown"] <= .1 and ms["p_times_b"] > 1.
                          and ms["standard_expectancy_loss_units"] > 0.)
            comparison = {"period": period, "cost": cost, "economic_gate_passed": passed, "comparators": []}
            for name, other, os in [("A_SAVED_WEIGHT", base, base_stats), ("TECH_COMMON_TREE", tech, ts)]:
                require(pd.DatetimeIndex(other["daily"].date).equals(pd.DatetimeIndex(main["daily"].date)), "比较日历不一致。")
                comparison["comparators"].append({"policy": name, "cagr_delta": ms["net_cagr"] - os["net_cagr"],
                    "sharpe_delta": ms["net_sharpe"] - os["net_sharpe"], "intervals": intervals(other["daily"].net_return.to_numpy(), main["daily"].net_return.to_numpy())})
            comparisons.append(comparison)
    table("十二完整账户共同口径比较", pd.DataFrame(rows))
    table("八新账户资金与库存核对", pd.DataFrame(checks))
    table("逐年净收益与真实交易次数", pd.DataFrame(years))
    table("全部实际进出点位与事前宏观评分", pd.DataFrame(point_rows))
    economic = all(c["economic_gate_passed"] for c in comparisons)
    stable = all(i["cagr_delta_95"][0] > 0 and i["sharpe_delta_95"][0] > 0
                 for c in comparisons for p in c["comparators"] for i in p["intervals"])
    wide = forecasts.pivot(index="origin_index", columns="policy", values="predicted_net_expectation")
    known = wide.notna().all(axis=1)
    difference = ~np.isclose(wide.loc[known, "TECH_COMMON_TREE"], wide.loc[known, "MACRO_TECH_TREE"], atol=1e-14, rtol=1e-12)
    summary = {"at": now(), "study": protocol["study"], "decision": "TECH.R192",
        "status": "HISTORICAL_INCREMENT_NOT_INDEPENDENT" if economic and stable else "REJECTED_FIXED_MACRO_TECH_FIRST_PASSAGE_FULL_ACCOUNT_GATE_FAILED",
        "candidate_configurations": 1, "monthly_fit_records": len(records),
        "paired_months_fitted": sum(r["status"] == "FIT_COMPLETE" for r in records),
        "actual_fit_calls": 2 * sum(r["status"] == "FIT_COMPLETE" for r in records),
        "new_candidate_accounts": 4, "new_matched_technical_accounts": 4, "original_A_exact_replays": 4,
        "common_available_forecasts_per_policy": int(known.sum()), "different_expectation_origins": int(difference.sum()),
        "actual_macro_path_origins": int((forecasts.policy.eq("MACRO_TECH_TREE") & forecasts.macro_features_on_path.ne("")).sum()),
        "all_four_economic_gates_passed": economic, "historical_stability_gate_passed": stable, "comparisons": comparisons,
        "new_labels": 0, "new_market_requests": 0, "parameter_grids": 0, "first_vintage": "NOT_CERTIFIED",
        "evidence_role": "DEVELOPMENT_CALIBRATION", "independent_validation": "NOT_ESTABLISHED",
        "global_DSR_PBO": "NOT_COMPUTED", "overfitting_removed": False, "goal_achieved": False, "orders_authorized": False}
    verify_sources()
    write_json(OUT / "summary.json", summary, exclusive=True)
    print("唯一宏观联合用途、八新账户和全部配对区间实际完成。", flush=True)


def deliver():
    summary = read(OUT / "summary.json")
    verify_sources()
    require(not (OUT / "delivery_receipt.json").exists(), "结果已交付，不重复。")
    data = pd.read_parquet(OUT / "results/全部3488当时已知技术与宏观_未知保留.parquet")
    forecasts = pd.read_parquet(OUT / "results/全部事前配对模型预测与实际宏观路径.parquet")
    metrics_table = pd.read_parquet(OUT / "results/十二完整账户共同口径比较.parquet")
    points = pd.read_parquet(OUT / "results/全部实际进出点位与事前宏观评分.parquet")
    checks = []
    for period in original.PERIODS:
        for cost in original.COSTS:
            for policy in model.POLICIES:
                path = OUT / f"results/accounts/{period}/{cost}/{policy}"
                account = {name: pd.read_parquet(path / (name + ".parquet")) for name in ["daily", "orders", "trades", "decisions", "rejections"]}
                account["terminal"] = read(path / "terminal.json")
                checks.append({"period": period, "cost": cost, "policy": policy, **verify_account(account)})
                saved = metrics_table.loc[metrics_table.period.eq(period) & metrics_table.cost.eq(cost) & metrics_table.policy.eq(policy)].iloc[0]
                recalculated = metrics(account)
                for key in ["net_cagr", "net_sharpe", "max_drawdown", "p_times_b", "ending_equity"]:
                    require(np.isclose(saved[key], recalculated[key], rtol=1e-12, atol=1e-12, equal_nan=True), "保存账户指标复算不一致。")
    joined = pd.read_parquet(OUT / "results/四原上涨反弹案例_全部同钟宏观与技术.parquet")
    joint_scores = forecasts.loc[forecasts.policy.eq("MACRO_TECH_TREE"), ["date", "score", "entry_event", "predicted_p_times_b", "predicted_net_expectation", "macro_features_on_path"]]
    joined = joined.merge(joint_scores, on="date", how="left", validate="many_to_one")
    table("四原案例_原宏观解释与事前联合评分", joined)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    charts = []
    for case_id, case in joined.groupby("original_episode_id", sort=True):
        fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
        axes[0].plot(case.date, case.close, color="#153e65", label="原收盘价")
        axes[0].set_title(f"原案例{case_id}：先解释量价、慢指标与可知宏观，再观察固定模型评分")
        axes[1].plot(case.date, case.daily_hist_atr, label="日柱/ATR")
        axes[1].plot(case.date, case.weekly_hist_atr, label="上一完整周柱/ATR")
        axes[1].plot(case.date, case.log_relative_volume, alpha=.55, label="相对量对数")
        axes[2].step(case.date, case.pmi_orders_level, where="post", label="已发布订单减50/点")
        axes[2].step(case.date, case.pmi_orders_change, where="post", label="连续月订单变化/点")
        axes[3].plot(case.date, case.funding_gap_pp, label="资金政策利差/百分点")
        axes[3].plot(case.date, 100 * case.financing_net_change5, label="前日可知融资五日变化/%")
        axes[4].plot(case.date, case.score, color="#a24419", label="估计净胜率分数，未校准")
        axes[4].set_ylim(0, 100)
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend(loc="upper left", fontsize=8, ncol=3)
        chart = OUT / f"案例{int(case_id)}_技术与宏观联合评分.png"
        fig.savefig(chart, dpi=140)
        plt.close(fig)
        charts.append(chart.relative_to(ROOT).as_posix())
    records = read(OUT / "saved_models.json")
    path_usage = forecasts.loc[forecasts.policy.eq("MACRO_TECH_TREE") & forecasts.status.eq("AVAILABLE")].copy()
    usage = {name: int(path_usage.macro_features_on_path.str.split("|").map(lambda x: name in x).sum()) for name in model.MACRO}
    write_json(OUT / "saved_result_verification.json", {"at": now(), "status": "PASS_EIGHT_SAVED_ACCOUNTS_AND_METRICS", "checks": checks}, exclusive=True)
    chosen_columns = ["period", "cost", "policy", "net_cagr", "net_sharpe", "max_drawdown", "completed_cycles", "win_rate", "payoff", "p_times_b"]
    text = "# 宏观与日周线联合评分：具体解释与完整账户结果\n\n"
    text += "已按新授权接入PMI订单、资金价格和修复后的融资信息；先完成全部3488日及原61分段/49上涨的同钟解释，再运行一个固定非线性模型用途。原策略、退出、风险预算和前瞻不改。\n\n"
    text += "本次六宏观字段不是六张独立赞成票。资金价差变大可以伴随需求改善，订单低于50也不排除政策预期推动上涨；模型在共同训练池中学习交互，没有预设一律利好/利空权重。模型分数=100乘估计净胜率，尚未证明校准，实际完成胜率/B另计。\n\n"
    text += "同钟资料缺口：PMI历史台账139月至2026-07，资金/融资主体止于2025-12-31；2026年未知保留，未沿用旧值，也未删出全账户日历。资金统计滞后一日，PMI按真实发布钟，周线只用上一完整周；融资采用已有官方单行纠正副本，但这些历史回取均不是独立首版。\n\n"
    text += f"共同可评分原点每模型{summary['common_available_forecasts_per_policy']}；两模型预测不同{summary['different_expectation_origins']}；联合模型实际使用宏观路径{summary['actual_macro_path_origins']}。142月中{summary['paired_months_fitted']}月有完整训练支持，实际历史拟合{summary['actual_fit_calls']}次；无支持月不补数。宏观路径使用次数：{usage}。\n\n"
    text += metrics_table[chosen_columns].to_markdown(index=False, floatfmt=".5f") + "\n\n"
    text += f"四经济门通过{sum(c['economic_gate_passed'] for c in summary['comparisons'])}/4；整体经济门{summary['all_four_economic_gates_passed']}，历史稳定门{summary['historical_stability_gate_passed']}。全部20/252日配对区间在summary.json，不选择好时期/好区块。\n\n"
    text += f"实际全部周期{len(points)}，两费用和两模型均保留，开放周期不计已完成胜率。首版NOT_CERTIFIED，历史DEVELOPMENT_CALIBRATION，独立NOT_ESTABLISHED，global DSR/PBO NOT_COMPUTED；收益夏普完整目标及去除过拟合尚未证明。\n\n"
    text += "[固定17关键日](results/事前固定17关键日期_宏观与技术.csv)、[四案例全部同钟状态与模型分数](results/四原案例_原宏观解释与事前联合评分.csv)、[全部真实点位](results/全部实际进出点位与事前宏观评分.csv)、[逐年覆盖](results/逐年真实信息覆盖_不缩样本.csv)、[逐年收益与次数](results/逐年净收益与真实交易次数.csv)。\n\n"
    text += ("本固定用途未通过完整门，拒绝并关闭。不得据此改窗口、树深、评分线、变量方向、样本或退出营救；可接入信息与实际宏观路径的研究事实保留。\n" if not (summary["all_four_economic_gates_passed"] and summary["historical_stability_gate_passed"]) else "支持有限历史增量，尚需单独登记真正独立前瞻；不能改写现有前瞻或直接交易。\n")
    (OUT / "研究结果与下一步.md").write_text(text, encoding="utf-8")
    write_json(OUT / "delivery_receipt.json", {"at": now(), "status": "PASS_SAVED_DELIVERY", "charts": charts,
        "report": (OUT / "研究结果与下一步.md").relative_to(ROOT).as_posix(), "macro_path_usage": usage,
        "saved_account_checks": 8, "new_labels": 0, "new_market_requests": 0, "orders_authorized": False}, exclusive=True)
    print("解释、宏观实际路径、全部点位和八保存账户复算已交付。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="宏观与技术首次边界联合用途，一次固定实验。")
    parser.add_argument("command", choices=("freeze", "explain", "run", "deliver"))
    args = parser.parse_args()
    {"freeze": freeze, "explain": explain, "run": run, "deliver": deliver}[args.command]()
