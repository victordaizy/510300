"""价量、宏观与原始财报广度的共同样本非线性评分；保持旧策略冻结。"""
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
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import macro_technical_first_passage_inputs_v1 as tree_io
from research import point_first_passage_inputs_v1 as reference

OUT = ROOT / "reports/research/510300_all_factor_macro_earnings_joint_v1"
TZ = ZoneInfo("Asia/Shanghai")
TECH = list(reference.FEATURES)
MACRO = list(tree_io.MACRO)
EARNINGS = [
    "profit_growth_median", "core_growth_median", "revenue_growth_median", "cashflow_growth_median",
    "profit_improve_share", "core_improve_share", "revenue_improve_share", "cashflow_improve_share",
    "positive_profit_share", "profit_cash_joint_improve_share", "profit_assets_annual_median",
    "cash_profit_assets_annual_median", "cash_profit_assets_change_median", "fundamental_coverage",
    "report_age_scaled", "recent_disclosure_share",
]
POLICIES = {
    "TECH_COMMON": TECH,
    "TECH_MACRO_COMMON": TECH + MACRO,
    "TECH_EARNINGS_COMMON": TECH + EARNINGS,
    "TECH_MACRO_EARNINGS_JOINT": TECH + MACRO + EARNINGS,
}
PRIMARY = "TECH_MACRO_EARNINGS_JOINT"
INPUTS = {
    "daily": "reports/research/510300_macro_technical_first_passage_v1_clock_adapter/results/全部3488当时已知技术与宏观_未知保留.parquet",
    "earnings": "reports/research/510300_original_earnings_breadth_v1/daily_earnings_breadth.parquet",
    "earnings_receipt": "reports/research/510300_original_earnings_breadth_v1/source_receipt.json",
    "labels": "reports/research/510300_point_first_passage_study_v1/results/原点首次边界参考结果.parquet",
    "cases": "reports/research/510300_all_factor_joint_scorecard_v1/results/全部30历史案例_联合解释分与覆盖.parquet",
    "earnings_old_result": "reports/research/510300_original_earnings_breadth_v1/result.json",
    "macro_old_result": "reports/research/510300_macro_technical_first_passage_v1_clock_adapter/summary.json",
    "score_old_result": "reports/research/510300_multidim_nonlinear_score_v1/result.json",
    "integrated_old_result": "reports/research/510300_integrated_macro_micro_prediction_v1/result.json",
    "reported_context": "reports/research/510300_all_factor_source_binding_v1/current_disclosed_context.json",
    "original_earnings_code": "research/original_earnings_breadth_v1.py",
    "tree_code": "research/macro_technical_first_passage_inputs_v1.py",
    "reference_code": "research/point_first_passage_inputs_v1.py",
    "tests": "tests/test_all_factor_macro_earnings_joint_v1.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(TZ).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, value, exclusive=False):
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as f:
        json.dump(clean(value), f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def aware_before(values, cutoff):
    def check(value, upper):
        if pd.isna(value):
            return False
        point = pd.Timestamp(value)
        return point.tzinfo is not None and point.tz_convert(TZ) <= upper
    return pd.Series([check(v, c) for v, c in zip(values, cutoff)], index=values.index)


def bind(daily, earnings):
    d, e = daily.copy(), earnings.copy()
    for frame in (d, e):
        frame["date"] = pd.to_datetime(frame.date).astype("datetime64[ns]")
        require(frame.date.is_unique, "共同来源日期重复，不能选择有利版本。")
    d = d.sort_values("date").reset_index(drop=True)
    d["decision_at"] = d.date.dt.tz_localize(TZ) + pd.Timedelta(hours=15, minutes=5)
    d["source_technical_known"] = (
        d.first_passage_feature_known.eq(True)
        & pd.to_datetime(d.weekly_last_date).lt(d.date)
        & pd.to_datetime(d.weekly_available_date).le(d.date)
        & np.isfinite(d[TECH].to_numpy(float)).all(axis=1)
    )
    orders_ok = d.orders_known.eq(True) & aware_before(d.orders_available_at, d.decision_at)
    funding_ok = (
        d.funding_known.eq(True)
        & aware_before(d.funding_available_at, d.decision_at)
        & aware_before(d.funding_policy_known_at, d.decision_at)
        & pd.to_datetime(d.fund_stat_date).lt(d.date)
    )
    margin_ok = (
        d.margin_known.eq(True) & aware_before(d.margin_available_at, d.decision_at)
        & pd.to_datetime(d.margin_stat_date).lt(d.date)
    )
    d.loc[~orders_ok, ["pmi_orders_level", "pmi_orders_change"]] = np.nan
    d.loc[~funding_ok, "funding_gap_pp"] = np.nan
    # 按新的15:05可得视图重新计算，防止旧16:00视图的前序值进入差分。
    d["funding_gap_change5"] = d.funding_gap_pp.diff(5).where(
        d.funding_gap_pp.rolling(6, min_periods=6).count().eq(6))
    d.loc[~margin_ok, ["financing_net_change5", "financing_buy_activity"]] = np.nan
    d["source_macro_known"] = (
        orders_ok & funding_ok & margin_ok & np.isfinite(d[MACRO].to_numpy(float)).all(axis=1))
    e = e.rename(columns={"latest_publication_date": "earnings_latest_publication_date",
                          "latest_available_date": "earnings_latest_available_date"})
    d = d.merge(e, on="date", how="left", validate="one_to_one")
    pub = pd.to_datetime(d.earnings_latest_publication_date)
    upper = pd.to_datetime(d.earnings_latest_available_date)
    d["earnings_available_at"] = upper.dt.tz_localize(TZ) + pd.Timedelta(hours=9, minutes=30)
    d["source_earnings_known"] = (
        d.fundamental_valid.eq(True) & d.member_count.eq(300) & d.valid_company_count.ge(240)
        & pub.lt(d.date) & d.earnings_available_at.le(d.decision_at)
        & np.isfinite(d[EARNINGS].to_numpy(float)).all(axis=1)
    )
    d["raw_earnings_context_only"] = d.member_count.notna() & ~d.source_earnings_known
    # 旧原值保留用于解释；预测列只有全部源门通过才可进入模型。
    d["joint_source_known"] = d.source_technical_known & d.source_macro_known & d.source_earnings_known
    d["joint_features_known"] = d.joint_source_known
    d["source_clock_role"] = "ORIGINAL_DOCUMENT_DATES_NEXT_SESSION_OPEN_CONSERVATIVE_REPLAY_NOT_INDEPENDENT"
    return d


def common_pool(data, outcomes, fit_index):
    require(np.array_equal(outcomes.origin_index.to_numpy(int), np.arange(len(data))), "原参考索引改变。")
    require(np.array_equal(pd.to_datetime(outcomes.origin).to_numpy(dtype="datetime64[ns]"),
                           data.date.to_numpy(dtype="datetime64[ns]")), "参考原点与共同日历不一致。")
    # 原窗口、成熟钟和重叠权重算法保持；加入不同的原始财报源后重新形成共同池。
    return tree_io.common_pool(data, outcomes, fit_index)


def fit_at(data, outcomes, fit_index):
    pool = common_pool(data, outcomes, fit_index)
    counts = {name: int(pool.event_class.eq(name).sum()) for name in reference.CLASSES}
    record = {"fit_index": int(fit_index), "fit_date": data.date.iloc[fit_index],
              "training_origins": pool.origin_index.astype(int).tolist(), "training_rows": len(pool),
              "class_counts": counts, "latest_mature_idx": float(pool.mature_idx.max()) if len(pool) else None,
              "sum_uniqueness_weights": float(pool.uniqueness_weight.sum()),
              "uniqueness_is_not_independent_sample_count": True,
              "status": "NO_VIEW_COMMON_TRAINING_SUPPORT", "models": {}}
    if len(pool) < reference.MINIMUM_ROWS or min(counts.values()) < 10:
        return record
    payoffs = tree_io.class_payoffs(pool)
    indices = pool.origin_index.to_numpy(int)
    for policy, columns in POLICIES.items():
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=60, random_state=tree_io.SEED)
        tree.fit(data.iloc[indices][columns].to_numpy(float), pool.event_class,
                 sample_weight=pool.fit_weight.to_numpy(float))
        record["models"][policy] = tree_io.serialize(tree, columns, payoffs)
    record["status"] = "FIT_COMPLETE"
    return record


def forecast(data, outcomes):
    start = np.flatnonzero(data.date.ge(pd.Timestamp("2015-01-05")))
    require(len(start) > 0 and start[0] > 0, "原月度训练起点不完整。")
    first = int(start[0]) - 1
    months = data.date.dt.to_period("M")
    cuts = {first, *(int(i) for i in np.flatnonzero(months.ne(months.shift())) if i > first)}
    fits, rows, current = [], [], None
    for i in range(len(data)):
        if i in cuts:
            current = fit_at(data, outcomes, i)
            fits.append(current)
        for policy, columns in POLICIES.items():
            row = {"date": data.date.iloc[i], "origin_index": i, "policy": policy,
                   "fit_index": current["fit_index"] if current else None,
                   "status": "NO_VIEW_NO_MODEL", "score": np.nan, "leaf": np.nan,
                   "earnings_fields_on_path": "", "macro_fields_on_path": "",
                   "source_joint_known": bool(data.joint_source_known.iloc[i]),
                   "candidate_quality_pass": False}
            if current and current["status"] == "FIT_COMPLETE":
                row["status"] = "NO_VIEW_CURRENT_INFORMATION"
                if bool(data.joint_source_known.iloc[i]):
                    model = current["models"][policy]
                    probabilities, leaf, used = tree_io.predict(model, data.iloc[i][columns].to_numpy(float))
                    row.update(status="AVAILABLE", **reference.quality(model, probabilities),
                               **dict(zip(["p_" + k for k in reference.CLASSES], probabilities)))
                    row.update(score=100 * row["predicted_win_probability"], leaf=leaf,
                               earnings_fields_on_path="|".join(dict.fromkeys(k for k in used if k in EARNINGS)),
                               macro_fields_on_path="|".join(dict.fromkeys(k for k in used if k in MACRO)))
                    row["candidate_quality_pass"] = bool(
                        row["predicted_p_times_b"] > 1 and row["predicted_net_expectation"] > 0)
            rows.append(row)
    return fits, pd.DataFrame(rows)


def evaluate(predictions, outcomes):
    cols = ["origin_index", "status", "event_class", "reference_net_return"]
    d = predictions.merge(outcomes[cols].rename(columns={"status": "reference_status"}),
                          on="origin_index", how="left", validate="many_to_one")
    d["period"] = np.where(d.date.lt(pd.Timestamp("2020-01-01")), "2015_2019", "2020_2026")
    d = d[d.date.ge(pd.Timestamp("2015-01-05"))]
    results = []
    for period in ["2015_2019", "2020_2026"]:
        for policy in POLICIES:
            b = d[d.period.eq(period) & d.policy.eq(policy)]
            b = b[b.status.eq("AVAILABLE") & b.reference_status.eq("MATURE_REFERENCE")]
            result = {"period": period, "policy": policy, "scored_mature_origins": len(b),
                      "brier": np.nan, "net_expectation_mse": np.nan,
                      "history_role": "DEVELOPMENT_ALREADY_OBSERVED",
                      "comparison_status": "NO_VIEW_NO_COMMON_MATURE_PREDICTIONS"}
            if len(b):
                actual = np.column_stack([b.event_class.eq(k).to_numpy(float) for k in reference.CLASSES])
                prediction = b[["p_" + k for k in reference.CLASSES]].to_numpy(float)
                result.update(brier=float(np.mean(np.sum((prediction - actual) ** 2, axis=1))),
                              net_expectation_mse=float(np.mean((b.predicted_net_expectation - b.reference_net_return) ** 2)),
                              comparison_status="COMPUTED_SAME_ORIGINS_NOT_INDEPENDENT")
            results.append(result)
    return pd.DataFrame(results)


def table(name, frame):
    frame.to_parquet(OUT / "results" / (name + ".parquet"), index=False)
    frame.to_csv(OUT / "results" / (name + ".csv"), index=False, encoding="utf-8-sig")


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [{"key": k, "path": v, "sha256": sha(ROOT / v)} for k, v in INPUTS.items()]
    receipt = read(ROOT / INPUTS["earnings_receipt"])
    require(receipt["status"] == "PASS_ORIGINAL_FACTS_CONTRACT_COVERAGE_ENFORCED_PER_DAY", "原始财报来源合同不合格。")
    for source in receipt["sources"]:
        require(sha(ROOT / source["path"]) == source["sha256"], "原始财报链版本变化：" + source["path"])
    write(OUT / "protocol.json", {
        "study_id": "510300_ALL_FACTOR_MACRO_EARNINGS_JOINT_V1", "registered_at": now(),
        "registration": "TECH.R253", "decision": "TECH.R254", "primary": PRIMARY,
        "hypothesis": "原始财报改善广度与资金/信用和日周线量价处于共同状态时，可能提供不同于单独盈利或宏观模型的首次边界收益信息。",
        "sources": sources, "original_document_sources": receipt["sources"],
        "code_sha256": sha(Path(__file__).absolute()), "policies": POLICIES,
        "old_rejections_preserved": True, "finite_novelty_role": "旧首次边界宏观用途无原始财报广度；只确认此有限差异，不宣称全球首次。",
        "cutoff": "每个原交易日15:05，上一已完成周；原财报公布日期后下一交易日09:30才可用",
        "earnings": "16原字段、300当时成员、至少240有效且报告龄不超过200日；等权广度不是指数EPS。",
        "pool": "原756日窗口，至少252共同成熟原点、每类至少10；同原点同权重四模型，深度3叶60、原seed，月度更新。",
        "labels": "原3488首次边界标签，上2ATR/下1ATR/20收盘，次开盘与股息后成熟；零新标签。",
        "scoring": "100×模型估计净胜率，仅开发估计、未校准；四模型完全同源覆盖。未知不填零。",
        "evaluation": "保留2015_2019与2020_2026两原时期的全部状态，Brier及净收益期望MSE全对照；不足标NO_VIEW，不挑期间晋升。",
        "all_factor_limit": "当前30数值字段仅为可绑定核心；83项目录全部保留。当前披露估值不得回填，其他未绑定因素未知。",
        "evidence_role": "DEVELOPMENT_CALIBRATION_NOT_INDEPENDENT", "new_candidate_configurations": 1,
        "matched_controls": 3, "parameter_grids": 0, "new_market_requests": 0,
        "new_accounts": 0, "financial_admission": "NOT_ADMITTED_PREDICTION_AND_SOURCE_STAGE_ONLY",
        "goal_achieved": False, "orders_authorized": False,
    }, exclusive=True)
    print("已登记唯一联合核心评分及三个共同池对照；原策略和独立验证口径保持。")


def run():
    p = read(OUT / "protocol.json")
    require(p["code_sha256"] == sha(Path(__file__).absolute()), "登记后代码变化。")
    for s in p["sources"] + p["original_document_sources"]:
        require(sha(ROOT / s["path"]) == s["sha256"], "来源变化：" + s["path"])
    write(OUT / "run_started.json", {"at": now(), "new_accounts": 0, "new_labels": 0}, exclusive=True)
    try:
        d = bind(pd.read_parquet(ROOT / INPUTS["daily"]), pd.read_parquet(ROOT / INPUTS["earnings"]))
        y = pd.read_parquet(ROOT / INPUTS["labels"])
        fits, predictions = forecast(d, y)
        results = OUT / "results"
        results.mkdir()
        table("全部3488共同源视图_不足保留", d)
        table("全部四模型_共同原点评分及未知", predictions)
        comparison = evaluate(predictions, y)
        table("两原时期_全部同池预测比较", comparison)
        pool_records = [{k: v for k, v in f.items() if k not in ("models", "training_origins")}
                        for f in fits]
        table("全部月度成熟支持_不改原下限", pd.DataFrame(pool_records))
        write(OUT / "saved_models_and_training_origins.json", fits, exclusive=True)
        cases = pd.read_parquet(ROOT / INPUTS["cases"])
        cases["date"] = pd.to_datetime(cases.date).astype("datetime64[ns]")
        fields = ["date", "member_count", "valid_company_count", "source_earnings_known",
                  "earnings_latest_publication_date", "earnings_available_at", "joint_source_known", *EARNINGS]
        cases = cases.merge(d[fields], on="date", how="left", validate="one_to_one")
        table("原全部30案例_原始盈利与共同源状态", cases)
        years = d[d.date.ge(pd.Timestamp("2015-01-05"))].groupby(d.date.dt.year).agg(
            trading_days=("date", "size"), technical_known=("source_technical_known", "sum"),
            macro_known=("source_macro_known", "sum"), earnings_known=("source_earnings_known", "sum"),
            common_known=("joint_source_known", "sum"))
        table("逐年全部共同覆盖_不缩时期", years.reset_index())
        available = predictions[predictions.status.eq("AVAILABLE")]
        supports = [f for f in fits if f["status"] == "FIT_COMPLETE"]
        differences = []
        for period in ["2015_2019", "2020_2026"]:
            part = comparison[comparison.period.eq(period)].set_index("policy")
            for control in list(POLICIES)[:-1]:
                count = int(part.loc[PRIMARY, "scored_mature_origins"])
                differences.append({"period": period, "control": control, "common_mature_origins": count,
                    "joint_minus_control_brier": float(part.loc[PRIMARY, "brier"] - part.loc[control, "brier"]) if count else None,
                    "joint_minus_control_expectation_mse": float(part.loc[PRIMARY, "net_expectation_mse"] - part.loc[control, "net_expectation_mse"]) if count else None})
        score_days = int(available.loc[available.policy.eq(PRIMARY), "date"].nunique())
        summary = {
            "study_id": p["study_id"], "completed_at": now(),
            "status": "COMPLETED_FIXED_JOINT_CORE_PREDICTION_REQUIRES_FULL_ACCOUNT_AND_INDEPENDENT_VALIDATION",
            "total_daily_origins": len(d), "raw_earnings_qualified_origins": int(d.source_earnings_known.sum()),
            "common_source_origins": int(d.joint_source_known.sum()), "case_count": len(cases),
            "case_original_earnings_admissions": int(cases.source_earnings_known.sum()),
            "monthly_fit_records": len(fits), "monthly_common_supported_fits": len(supports),
            "actual_fit_calls": len(supports) * len(POLICIES), "scored_days_per_policy": score_days,
            "first_common_supported_fit": supports[0]["fit_date"] if supports else None,
            "first_scored_day": available.date.min() if len(available) else None,
            "comparisons": comparison.to_dict("records"), "joint_control_differences": differences,
            "new_candidate_configurations": 1, "matched_controls": 3, "parameter_grids": 0,
            "new_market_requests": 0, "new_labels": 0, "new_accounts": 0,
            "new_independent_completed_points": 0, "new_strategy_net_CAGR": "NOT_COMPUTED",
            "new_strategy_net_Sharpe": "NOT_COMPUTED", "all_factor_predictive_score": "NOT_COMPUTED",
            "original_core_explanatory_grades_preserved": True,
            "first_vintage_scope": "原始财报日期和原文链保留；其他旧宏观源首次版本未独立认证。",
            "financial_admission": "NOT_ADMITTED_SOURCE_AND_PREDICTION_STAGE_ONLY",
            "independent_validation": "NOT_ESTABLISHED", "overfitting_removed": False,
            "goal_achieved": False, "orders_authorized": False,
        }
        write(OUT / "summary.json", summary, exclusive=True)
        report = ["# 原始盈利、宏观与量价的共同核心评分", "",
                  "唯一新候选与三个去组对照均使用同一共同成熟池。旧原始盈利、宏观首次边界及固定评分拒绝保持。",
                  "这不是83因素完整预测总分；全部原时期和不足保留，没有新金融账户或独立样本。", "",
                  f"3488完整原点：原始财报可用{summary['raw_earnings_qualified_origins']}，共同源{summary['common_source_origins']}；月度合格拟合{len(supports)}，四模型实际拟合{summary['actual_fit_calls']}。",
                  f"每模型实际评分日{score_days}，原30案例盈利源合格{summary['case_original_earnings_admissions']}。", "",
                  "| 原时期 | 模型 | 共同成熟评分原点 | Brier | 净期望MSE |", "|---|---|---:|---:|---:|"]
        for row in comparison.itertuples():
            brier = f"{row.brier:.6f}" if pd.notna(row.brier) else "NO_VIEW"
            mse = f"{row.net_expectation_mse:.8f}" if pd.notna(row.net_expectation_mse) else "NO_VIEW"
            report.append(f"| {row.period} | {row.policy} | {row.scored_mature_origins} | {brier} | {mse} |")
        report += ["", "2015—2019不足仍保留，不能把后期评分结果外推到早期。模型分数是未校准的开发估计，重叠唯一性权重不等于独立事件数量。",
                   "原始累计财报的等权广度不是指数EPS，报告中的对称增长中位数不是普通同比；240公司和200日报告龄门槛保持。",
                   "官方当前披露估值/权重未回填旧历史。没有按模型成绩更换阶段、窗口、叶子或指标，没有合并重复来源为独立票。",
                   "下一先根据完整固定比较判断是否存在新增预测证据，完整账户的净收益、Sharpe、净pB、回撤、交易次数和独立验证仍需另登记检验。"]
        (OUT / "原始盈利宏观量价_共同评分与全部反例.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        write(OUT / "run_completed.json", {"at": now(), "terminal": True, "new_accounts": 0}, exclusive=True)
        print(json.dumps(clean({k: summary[k] for k in ["status", "common_source_origins", "monthly_common_supported_fits", "scored_days_per_policy", "actual_fit_calls", "new_accounts", "goal_achieved"]}), ensure_ascii=False))
    except Exception as exc:
        write(OUT / "implementation_failure.json", {"at": now(), "terminal": True,
              "error_type": type(exc).__name__, "error": str(exc), "new_accounts": 0}, exclusive=True)
        raise


def main():
    parser = argparse.ArgumentParser(description="原始财报、宏观与日周量价共同核心评分")
    parser.add_argument("action", choices=["freeze", "run"])
    args = parser.parse_args()
    freeze() if args.action == "freeze" else run()


if __name__ == "__main__":
    main()
