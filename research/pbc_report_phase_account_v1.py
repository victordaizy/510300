"""央行双文本通道与日周线量价的固定阶段条件实验，完整账户只运行一次。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from research import all_factor_macro_earnings_joint_v1 as io
from research import all_factor_macro_earnings_account_v1 as old_account
from research import macro_technical_first_passage_inputs_v1 as tree_io
from research import point_first_passage_inputs_v1 as reference
from research import point_first_passage_study_v1 as original
from research.macro_technical_first_passage_clock_adapter_v1 import account
from research.point_account_nr7_complement_v1 import annual_rows, verify_account

ROOT = Path(__file__).absolute().parents[1]
OUT = ROOT / "reports/research/510300_pbc_report_phase_account_v1"
SOURCE = ROOT / "reports/research/510300_pbc_report_text_source_reconciliation_v2_1"
DATA = io.OUT / "results/全部3488共同源视图_不足保留.parquet"
LABELS = ROOT / "reports/research/510300_point_first_passage_study_v1/results/原点首次边界参考结果.parquet"
TEXT = SOURCE / "全部3488原点_最新公布两章节变化与未知.parquet"
AGE = ["pbc_report_age_calendar_days"]
ECON = ["pbc_economic_change"]
GUIDANCE = ["pbc_guidance_change"]
POLICIES = {"TECH_AGE_COMMON": reference.FEATURES + AGE,
            "TECH_AGE_ECON_COMMON": reference.FEATURES + AGE + ECON,
            "TECH_AGE_GUIDANCE_COMMON": reference.FEATURES + AGE + GUIDANCE,
            "TECH_AGE_TWO_CHANNEL_JOINT": reference.FEATURES + AGE + ECON + GUIDANCE}
PRIMARY = "TECH_AGE_TWO_CHANNEL_JOINT"


def table(name, frame):
    path = OUT / "results" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)
    frame.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def balance_quarter_weights(pool, quarters):
    """先去重叠，再令报告季度权重相同，不把重复日线当独立报告。"""
    pool = pool.copy()
    pool["report_quarter"] = np.asarray(quarters)
    totals = pool.groupby("report_quarter").uniqueness_weight.transform("sum")
    weight = pool.uniqueness_weight / totals
    pool["fit_weight"] = weight / weight.mean()
    io.require(np.isfinite(pool.fit_weight).all() and pool.fit_weight.gt(0).all(), "季度平衡权重未知。")
    return pool


def training_pool(data, outcomes, index):
    pool = tree_io.common_pool(data, outcomes, index)
    indices = pool.origin_index.to_numpy(int)
    return balance_quarter_weights(pool, data.pbc_quarter.iloc[indices].to_numpy()) if len(pool) else pool


def fit_at(data, outcomes, index):
    pool = training_pool(data, outcomes, index)
    counts = {name: int(pool.event_class.eq(name).sum()) for name in reference.CLASSES}
    reports = sorted(pool.report_quarter.unique().tolist()) if len(pool) else []
    record = {"fit_index": index, "fit_date": data.date.iloc[index], "status": "NO_VIEW_TRAINING_SUPPORT", "models": {},
              "training_origins": pool.origin_index.astype(int).tolist(), "training_rows": len(pool),
              "latest_mature_idx": int(pool.mature_idx.max()) if len(pool) else None,
              "class_counts": counts, "report_quarters": reports, "distinct_report_quarters": len(reports),
              "report_count_is_not_independent_validation": True}
    if len(pool) < reference.MINIMUM_ROWS or min(counts.values()) < 10 or len(reports) < 8:
        return record
    payoffs = tree_io.class_payoffs(pool)
    indices = pool.origin_index.to_numpy(int)
    for policy, columns in POLICIES.items():
        tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=60, random_state=tree_io.SEED)
        x = data.iloc[indices][columns].to_numpy(float)
        tree.fit(x, pool.event_class, sample_weight=pool.fit_weight.to_numpy(float))
        model = tree_io.serialize(tree, columns, payoffs)
        leaves = tree.apply(x)
        model["leaf_distinct_reports"] = {str(int(leaf)): int(pool.loc[leaves == leaf, "report_quarter"].nunique())
                                           for leaf in np.unique(leaves)}
        record["models"][policy] = model
    record["status"] = "FIT_COMPLETE"
    return record


def forecast(data, outcomes):
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp("2015-01-05")))[0]) - 1
    months = data.date.dt.to_period("M")
    cuts = {first, *(int(i) for i in np.flatnonzero(months.ne(months.shift())) if i > first)}
    fits, rows, current = [], [], None
    for i in range(len(data)):
        if i in cuts:
            current = fit_at(data, outcomes, i)
            fits.append(current)
        for policy, columns in POLICIES.items():
            row = {"date": data.date.iloc[i], "origin_index": i, "policy": policy,
                   "fit_index": current["fit_index"] if current else None, "status": "NO_VIEW_NO_MODEL",
                   "score": np.nan, "leaf": np.nan, "leaf_distinct_reports": np.nan,
                   "candidate_quality_pass": False, "text_fields_on_path": "", "report_quarter": data.pbc_quarter.iloc[i]}
            if current and current["status"] == "FIT_COMPLETE":
                row["status"] = "NO_VIEW_CURRENT_INFORMATION"
                if bool(data.joint_features_known.iloc[i]):
                    model = current["models"][policy]
                    probabilities, leaf, path = tree_io.predict(model, data.iloc[i][columns].to_numpy(float))
                    support = model["leaf_distinct_reports"][str(leaf)]
                    row.update(leaf=leaf, leaf_distinct_reports=support,
                               text_fields_on_path="|".join(dict.fromkeys(k for k in path if k in AGE + ECON + GUIDANCE)))
                    row["status"] = "NO_VIEW_LEAF_REPORT_SUPPORT"
                    if support >= 4:
                        row.update(status="AVAILABLE", **reference.quality(model, probabilities),
                                   **dict(zip(["p_" + k for k in reference.CLASSES], probabilities)))
                        row["score"] = 100 * row["predicted_win_probability"]
                        row["candidate_quality_pass"] = bool(row["predicted_p_times_b"] > 1 and row["predicted_net_expectation"] > 0)
            rows.append(row)
    return fits, pd.DataFrame(rows)


def economic_gate(main, controls):
    required = ["net_cagr", "net_sharpe", "max_drawdown", "p_times_b", "standard_expectancy_loss_units", "mean_cycle_net_return"]
    if not np.isfinite([main.get(k, np.nan) for k in required]).all():
        return False
    absolute = (main["net_cagr"] >= .1 and main["net_sharpe"] >= 1.5 and main["max_drawdown"] <= .1
                and main["p_times_b"] > 1 and main["standard_expectancy_loss_units"] > 0 and main["mean_cycle_net_return"] > 0)
    relative = all(np.isfinite([c["net_cagr"], c["net_sharpe"]]).all()
                   and main["net_cagr"] > c["net_cagr"] and main["net_sharpe"] > c["net_sharpe"] for c in controls)
    return bool(absolute and relative)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [DATA, LABELS, TEXT, SOURCE / "protocol.json", SOURCE / "summary.json",
             SOURCE / "原全部30案例_当时报告与原解释等级.parquet", ROOT / "config/510300_high_return_sharpe_goal_v1.json",
             original.CURRENT / "inputs/candidate_prices.parquet", original.WEIGHT / "inputs/dividends.csv",
             original.WEIGHT / "inputs/risks.parquet", original.WEIGHT / "inputs/parent_signals.parquet",
             original.WEIGHT / "inputs/earlier_signals.parquet", ROOT / "research/point_first_passage_inputs_v1.py",
             ROOT / "research/point_first_passage_account_v1.py", ROOT / "research/macro_technical_first_passage_inputs_v1.py",
             ROOT / "research/macro_technical_first_passage_clock_adapter_v1.py", ROOT / "research/all_factor_macro_earnings_account_v1.py",
             ROOT / "research/point_account_nr7_inputs_v1.py", ROOT / "research/point_account_nr7_complement_v1.py",
             ROOT / "research/daily_supply_test_v1.py", ROOT / "tests/test_pbc_report_phase_account_v1.py"]
    for period in original.PERIODS:
        for cost in original.COSTS:
            paths += [original.CONTROL / period / cost / "A_SAVED_WEIGHT" / (name + ".parquet") for name in ("daily", "orders", "trades")]
    io.write(OUT / "protocol.json", {"registered_at": io.now(), "registration": "TECH.R267", "decision": "TECH.R268",
             "hypothesis": "经济描述变化和政策指引变化对不同日周线价量阶段的首次边界净收益具有不同条件关系；季度文本本身不定义方向。",
             "primary": PRIMARY, "policies": POLICIES, "configurations": 1, "parameter_searches": 0,
             "source_admission": "固定完整59季度目录、58原件、3368当时双通道已知原点；120未知原点保留无新进入，实际缺件不补。历史原件第一版未认证，仅开发研究。",
             "basis": "R266全部原30案例同分保留；2024启动与后拥挤共享2024Q2，文本只作阶段条件；政策与经济通道分开是新增信息用途，不重开旧盈利/宏观树裁决。",
             "training": "原月度时钟、756日成熟窗口、252原点且三类各10；至少8不同报告季度；先原重叠唯一性、再季度总权重相等，四模型完全同池同权。",
             "fixed_tree": "深度2、叶至少60日、seed保持旧510300191；实际评分叶须至少4不同报告季度，否则NO_VIEW；不是独立样本数或校准后的实际胜率。",
             "entry": "估计净pB>1、净期望>0才产生请求；只多头510300/现金。相同最高50%请求、ES/跳空预算、10%回撤停止新入场。",
             "exit_and_execution": "保持原次开盘/T+1/100份/0.001/至少5元费用/真实分红；上2ATR下1ATR或20收盘到期、下开盘退出；不拼接CORE或增加仓位。",
             "periods": original.PERIODS, "costs": original.COSTS, "initial_capital": 200000, "annual_days": 252,
             "accounts": "16新完整账户+4已保存A，全空仓日和自然期末持仓计入；120未知不删年份或日历。",
             "acceptance": "原四场景同时净CAGR>=10%、净Sharpe>=1.5、DD<=10%、实际净pB>1、标准净期望及周期均值>0；同场景两指标高于原A和三个同池去组。任一失败固定用途拒绝，不换对照救援。",
             "independence": "全部历史已见development；即使点值通过仍需真正新事前独立验证，不能完成目标。",
             "case_explanatory_grades_changed": False, "new_labels": 0, "new_network_requests": 0,
             "goal_achieved": False, "orders_authorized": False,
             "code_sha256": io.sha(Path(__file__)),
             "sources": [{"path": p.absolute().relative_to(ROOT).as_posix(), "sha256": io.sha(p)} for p in paths]}, exclusive=True)
    print("已登记唯一双文本×量价阶段用途与全部同池对照，四场景10%/1.5/10%门不变。", flush=True)


def run():
    protocol = io.read(OUT / "protocol.json")
    io.require(io.sha(Path(__file__)) == protocol["code_sha256"], "阶段实验登记后源码变化。")
    for source in protocol["sources"]:
        io.require(io.sha(ROOT / source["path"]) == source["sha256"], "阶段实验来源变化：" + source["path"])
    io.write(OUT / "run_started.json", {"at": io.now(), "planned_new_accounts": 16}, exclusive=True)
    try:
        data = pd.read_parquet(DATA)
        text = pd.read_parquet(TEXT)
        extra = ["date", "pbc_quarter", "pbc_two_channel_known", *AGE, *ECON, *GUIDANCE]
        data = data.merge(text[extra], on="date", how="left", validate="one_to_one")
        data["joint_features_known"] = data.source_technical_known & data.pbc_two_channel_known & np.isfinite(data[POLICIES[PRIMARY]].to_numpy(float)).all(axis=1)
        outcomes = pd.read_parquet(LABELS)
        io.require(len(data) == 3488 and np.array_equal(outcomes.origin_index.to_numpy(int), np.arange(len(data))), "原3488日/标签范围变化。")
        io.require(np.array_equal(pd.to_datetime(outcomes.origin).to_numpy(dtype="datetime64[ns]"), data.date.to_numpy(dtype="datetime64[ns]")), "原标签日历不同。")
        fits, predictions = forecast(data, outcomes)
        io.write(OUT / "全部月度模型及原成熟训练索引.json", fits, exclusive=True)
        table("全部3488完整原点_共同文本阶段与未知", data)
        table("全部四模型_阶段评分与未知", predictions)
        raw, dividends, risks, parents = original.load()
        for key in ("date", "open", "high", "low", "close", "ac", "atr14"):
            dtype = "datetime64[ns]" if key == "date" else float
            io.require(np.array_equal(data[key].to_numpy(dtype=dtype), raw[key].to_numpy(dtype=dtype), equal_nan=key != "date"), "原账户价位/时钟变化：" + key)
        rows, checks, years, points, contrasts = [], [], [], [], []
        for period, (start, end) in original.PERIODS.items():
            local = data[data.date.le(end)].reset_index(drop=True)
            for cost in original.COSTS:
                saved = old_account.baseline(period, cost)
                base = old_account.metrics_with_limits(saved)
                rows.append({"period": period, "cost": cost, "policy": "A_SAVED_WEIGHT", **base})
                stats = {}
                for policy in POLICIES:
                    selected = old_account.signals(data, predictions, policy).iloc[:len(local)].reset_index(drop=True)
                    result = account(local, dividends, parents[period], risks, selected, cost, start, "PASSAGE_ONLY")
                    io.require(pd.DatetimeIndex(result["daily"].date).equals(pd.DatetimeIndex(saved["daily"].date)), "完整账户日历不同。")
                    checks.append({"period": period, "cost": cost, "policy": policy, **verify_account(result)})
                    stats[policy] = old_account.metrics_with_limits(result)
                    yearly = annual_rows(result, period, policy, cost)
                    counts = [r["completed_cycles"] for r in yearly if r["full_year"]]
                    stats[policy].update(average_full_year_cycles=float(np.mean(counts)), zero_trade_full_years=sum(c == 0 for c in counts))
                    rows.append({"period": period, "cost": cost, "policy": policy, **stats[policy]})
                    years += yearly
                    for name in ("daily", "orders", "trades", "decisions", "rejections"):
                        table(f"accounts/{period}/{cost}/{policy}/{name}", result[name])
                    io.write(OUT / f"results/accounts/{period}/{cost}/{policy}/terminal.json", result["terminal"], exclusive=True)
                    for trade in result["trades"].to_dict("records"):
                        idx = int(np.flatnonzero(data.date.eq(trade["entry_origin"]))[0])
                        score = selected.iloc[idx]
                        points.append({"period": period, "cost": cost, "policy": policy, **trade,
                                       **{k: score.get(k) for k in ("score", "predicted_p_times_b", "predicted_net_expectation", "text_fields_on_path", "leaf_distinct_reports", "report_quarter")}})
                    stat = stats[policy]
                    print(f"{period}/{cost}/{policy}：净年化{stat['net_cagr']:.3%}，净夏普{stat['net_sharpe']:.4f}，完成{stat['completed_cycles']}。", flush=True)
                main = stats[PRIMARY]
                controls = [base] + [stats[k] for k in POLICIES if k != PRIMARY]
                contrasts.append({"period": period, "cost": cost, "economic_gate_passed": economic_gate(main, controls),
                                  "primary_minus_A_cagr": main["net_cagr"] - base["net_cagr"],
                                  "primary_minus_A_sharpe": main["net_sharpe"] - base["net_sharpe"],
                                  "all_controls": [{"policy": k, "cagr_delta": main["net_cagr"] - stats[k]["net_cagr"],
                                                    "sharpe_delta": main["net_sharpe"] - stats[k]["net_sharpe"]} for k in POLICIES if k != PRIMARY]})
        table("全部20账户_四场景同口径比较", pd.DataFrame(rows))
        table("全部16账户_资金库存与时钟", pd.DataFrame(checks))
        table("全部逐年收益与次数", pd.DataFrame(years))
        table("全部实际点位_当时文本路径", pd.DataFrame(points))
        passed = all(r["economic_gate_passed"] for r in contrasts)
        summary = {"decision": "TECH.R268", "completed_at": io.now(),
                   "status": "HISTORICAL_TEXT_STAGE_INCREMENT_REQUIRES_INDEPENDENT_VALIDATION" if passed else "REJECTED_FIXED_TEXT_STAGE_ACCOUNT_ABSOLUTE_TARGETS_NOT_MET",
                   "actual_fit_calls": 4 * sum(f["status"] == "FIT_COMPLETE" for f in fits),
                   "fit_records": len(fits), "scored_days_per_policy": {k: int(predictions.loc[predictions.policy.eq(k), "status"].eq("AVAILABLE").sum()) for k in POLICIES},
                   "two_channel_known_origins": int(data.pbc_two_channel_known.sum()), "missing_text_origins_kept": int((~data.pbc_two_channel_known).sum()),
                   "new_candidate_accounts": 4, "new_matched_control_accounts": 12, "reused_A_accounts": 4,
                   "new_labels": 0, "new_network_requests": 0, "configurations": 1, "parameter_searches": 0,
                   "all_four_economic_gates_passed": passed, "contrasts": contrasts,
                   "primary_four_scene_metrics": [r for r in rows if r["policy"] == PRIMARY], "necessary_account_checks": checks,
                   "independent_validation": "NOT_ESTABLISHED", "new_independent_completed_points": 0, "overfitting_removed": False,
                   "goal_achieved": False, "orders_authorized": False}
        io.write(OUT / "summary.json", summary, exclusive=True)
        io.write(OUT / "run_completed.json", {"at": io.now(), "terminal": True, "new_accounts": 16}, exclusive=True)
        print("完整双文本阶段实验已保存，四场景全门通过：" + str(passed), flush=True)
    except Exception as exc:
        io.write(OUT / "implementation_failure.json", {"at": io.now(), "terminal": True, "type": type(exc).__name__, "error": str(exc)}, exclusive=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="固定央行双文本与日周量价阶段完整账户实验")
    parser.add_argument("command", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()


if __name__ == "__main__":
    main()
