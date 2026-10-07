"""隔离诊断：核对原继续收益预测的正负，不训练模型、不创建策略账户。"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends
from research.learned_cycle_exit_v1 import FEATURES, continuation_label
from research.point_account_cashflow_state_v1 import digest, now, require, write_json
from research.within_cycle_exit_inputs_v1 import within_cycle_prediction


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_point_exit_sign_calibration_v1"
CURRENT = Path("reports/research/510300_point_current_observation_20261001")
EARLIER = Path("reports/research/510300_point_core_observation_v1")
E05 = Path("reports/research/510300_point_volatility_unit_exit_v1")
ACCOUNTS = Path("reports/research/510300_point_second_weight_comparison_v1/inputs/controls")
STUDY = "510300_POINT_EXIT_SIGN_CALIBRATION_V1"
PERIODS = {"2015_2019": ("2015-01-05", "2019-12-31"),
           "2020_2026": ("2020-01-02", "2026-09-30")}
COSTS = ("BASE", "STRESS")


def reference_path(period, cost, kind):
    prefix = EARLIER / "results/earlier_diagnostic/references" if period == "2015_2019" else CURRENT / "results/references"
    return ROOT / prefix / f"ENTRY_VINTAGE_{cost}_{kind}.parquet"


def prediction_sign(value):
    if not np.isfinite(value):
        return "NO_VIEW"
    return "NEGATIVE" if value < 0 else "NONNEGATIVE"


def period_of(origin):
    day = pd.Timestamp(origin)
    for key, (start, end) in PERIODS.items():
        if pd.Timestamp(start) <= day <= pd.Timestamp(end):
            return key
    return "OUTSIDE_EVALUATION"


def select_model(records, entry_index):
    indexes = [int(r["fit_index"]) for r in records]
    require(indexes == sorted(set(indexes)), "保存模型月份必须唯一递增。")
    position = bisect_right(indexes, int(entry_index)) - 1
    return records[position] if position >= 0 else None


def natural_match(cycles, entry_index, entry_date):
    """只按真实入场日和全局索引匹配；不按不同参考的周期编号匹配。"""
    rows = cycles.loc[cycles.entry_index.eq(entry_index) & cycles.entry_date.eq(pd.Timestamp(entry_date))]
    require(len(rows) <= 1, "一个入场身份对应多个自然参考周期。")
    return rows.iloc[0] if len(rows) else None


def first_confirmed_events(frame):
    eligible = frame.loc[frame.learned_exit_requested.fillna(False).astype(bool)].copy()
    return eligible.sort_values("origin_index").drop_duplicates(["period", "cost", "cycle_id"], keep="first")


def year_interval(cycle_values, iterations=5000, seed=51030068):
    """完整入场年度成块，仅描述已用开发历史中的不确定性。"""
    years = sorted(cycle_values.entry_year.unique())
    if len(years) < 2:
        return {"status": "NOT_COMPUTED_INSUFFICIENT_ENTRY_YEARS", "entry_years": len(years), "low": None, "high": None}
    blocks = [cycle_values.loc[cycle_values.entry_year.eq(year), "mean_target"].to_numpy(float) for year in years]
    rng = np.random.default_rng(seed)
    values = np.empty(iterations)
    for i in range(iterations):
        selected = rng.integers(0, len(blocks), len(blocks))
        values[i] = np.concatenate([blocks[k] for k in selected]).mean()
    low, high = np.quantile(values, [.025, .975])
    return {"status": "DESCRIPTIVE_REUSED_HISTORY", "entry_years": len(years), "low": float(low), "high": float(high)}


def summarize_group(frame):
    """先在每周期内部求均值，再周期等权；未知行不会补成零。"""
    known = frame.loc[np.isfinite(frame.prediction) & np.isfinite(frame.target)].copy()
    result = {"all_rows": len(frame), "eligible_rows": len(known), "cycles": 0,
              "cycle_equal_mean_prediction": None, "cycle_equal_mean_target": None,
              "cycle_equal_negative_target_rate": None, "cycle_equal_mse": None,
              "cycle_equal_prediction_minus_target": None, "target_mean_interval": None}
    if not len(known):
        return result
    known["negative_target"] = known.target.lt(0).astype(float)
    known["squared_error"] = (known.prediction - known.target) ** 2
    values = known.groupby("cycle_id", sort=True).agg(
        mean_prediction=("prediction", "mean"), mean_target=("target", "mean"),
        negative_rate=("negative_target", "mean"), mse=("squared_error", "mean"),
        entry_year=("entry_year", "first"))
    result.update(cycles=len(values), cycle_equal_mean_prediction=float(values.mean_prediction.mean()),
                  cycle_equal_mean_target=float(values.mean_target.mean()),
                  cycle_equal_negative_target_rate=float(values.negative_rate.mean()),
                  cycle_equal_mse=float(values.mse.mean()),
                  cycle_equal_prediction_minus_target=float((values.mean_prediction-values.mean_target).mean()),
                  target_mean_interval=year_interval(values))
    return result


def sources():
    paths = [CURRENT / "results/training_reference/samples.parquet", CURRENT / "results/training_reference/cycles.parquet",
             CURRENT / "inputs/within_models.json", EARLIER / "inputs/within_models.json",
             CURRENT / "inputs/candidate_features.parquet", CURRENT / "inputs/dividends.csv",
             CURRENT / "inputs/config/within_cycle_exit.json", E05 / "results/同原点配对预测.parquet",
             E05 / "protocol.json", E05 / "prediction_summary.json",
             Path("research/point_exit_sign_calibration_v1.py"), Path("tests/test_point_exit_sign_calibration_v1.py"),
             Path("research/learned_cycle_exit_v1.py"), Path("research/within_cycle_exit_inputs_v1.py"),
             Path("research/entry_vintage_exit_inputs_v1.py"), Path("research/point_monthly_model_inputs_v1.py"),
             Path("research/point_account_cashflow_state_v1.py"), Path("research/adaptive_allocation_v1.py"),
             Path("research/simple_price_entry_exit_v1.py"), Path("docs/510300_CYCLE_SERIAL_ERROR_EXIT_V1.md"),
             Path("docs/510300_POINT_PASS_FIXED_DIAGNOSTIC_V1.md"),
             Path("reports/research/510300_daily_weekly_goal_continuation_20261001/isolated_code_authorization_20261002.json")]
    for period in PERIODS:
        for cost in COSTS:
            paths.extend(reference_path(period, cost, kind).relative_to(ROOT) for kind in ("decisions", "cycles"))
            paths.append(ACCOUNTS / period / cost / "A_SAVED_WEIGHT/daily.parquet")
    paths.append((OUT / "tests_receipt.json").relative_to(ROOT))
    return sorted(set(paths))


def freeze():
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["passed"] == 4 and tests["exit_code"] == 0, "四项必要诊断测试未通过。")
    protocol = {"study": STUDY, "frozen_at": now(), "linked_question": "E06_原退出预测的决策相关校准",
        "hypothesis": "原模型在固定版本和负预测确认的决策附近，是否能识别负的原定义继续收益。",
        "deduplication": "旧周期相邻误差研究改变拟合损失，旧固定稳定性诊断评价账户路径；本次只检查原模型正负与原经济标签的条件关系，不重开其候选。",
        "candidate_configurations": 0, "new_model_fits": 0, "new_strategy_accounts": 0,
        "prediction_partition": "只用原零阈值：NEGATIVE / NONNEGATIVE；未知单列NO_VIEW，不翻号、不选分箱。",
        "canonical_view": "原E05同原点表中仅原模型列，复算原参数预测与原BASE自然标签；原参考退出后的原点不算真实决策。",
        "active_reference_view": "两时期BASE/STRESS保存ENTRY_VINTAGE中有真实学习周期的持仓收盘，保留所有缺预测/缺标签；按入场日及全局索引匹配原自然参考，不能按cycle_id配。",
        "auxiliary_label": "匹配自然参考的原保存状态且t+1<自然退出开盘时，以该活跃参考的原入场份额，复用原continuation_label至匹配自然终点，仍用训练BASE费用。不是学习策略未来真实现金流。",
        "auxiliary_label_limit": "STRESS预测使用STRESS持仓状态，但标签仍是原BASE训练定义；八项状态与自然参考是否相同单列。未匹配入场/没有保存自然状态/未成熟/无更长持有区间均未知，不补标签。",
        "confirmation_view": "每活跃参考周期只取首次原规则learned_exit_requested，两次严格负预测已由旧代码确认；不新增或调整确认规则。",
        "actual_account_overlap": "与保存A收盘shares>0相交仅注明是否影响当时持仓；参考标签不能称A剩余现金流、提前退出增量或策略收益。",
        "periods": PERIODS, "weighting": "各负/非负分组内每自然或活跃参考周期总权重一；费用和时期不合并，行数不是独立样本数。",
        "uncertainty": {"method": "整入场年度块，保留同周期全部行", "iterations": 5000, "seed": 51030068,
                        "purpose": "已用历史描述；不足两个入场年度NOT_COMPUTED；不消除选优偏差"},
        "failure_exit": "无论标签关系好坏，本诊断不生成新交易规则、不反向交易、不改旧截距/阈值/正则化；若没有不同事前信息，不自动进入E02。",
        "history_role": "DEVELOPMENT_CALIBRATION_NOT_INDEPENDENT", "independent_validation": "NOT_ESTABLISHED",
        "goal_achieved": False, "sources": [{"path": str(p), "sha256": digest(ROOT/p)} for p in sources()]}
    write_json(OUT / "protocol.json", protocol, exclusive=True)
    write_json(OUT / "freeze.json", {"frozen_at": now(), "protocol_sha256": digest(OUT/"protocol.json")}, exclusive=True)
    print(f"E06有限校准协议已冻结：{len(protocol['sources'])}份来源，尚未读取本次分组标签结果。", flush=True)


def check_sources():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    require(digest(OUT/"protocol.json") == frozen["protocol_sha256"], "E06冻结协议发生变化。")
    for record in protocol["sources"]:
        require(digest(ROOT / record["path"]) == record["sha256"], "E06来源变化：" + record["path"])
    return protocol


def save_table(name, frame):
    folder = OUT / "results"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / f"{name}.parquet", index=False)
    frame.to_csv(folder / f"{name}.csv", index=False, encoding="utf-8-sig")


def run():
    protocol = check_sources()
    write_json(OUT / "RUN_STARTED.json", {"at": now(), "new_model_fits": 0}, exclusive=True)
    samples = pd.read_parquet(ROOT / CURRENT / "results/training_reference/samples.parquet")
    cycles = pd.read_parquet(ROOT / CURRENT / "results/training_reference/cycles.parquet")
    paired = pd.read_parquet(ROOT / E05 / "results/同原点配对预测.parquet")
    prices = pd.read_parquet(ROOT / CURRENT / "inputs/candidate_features.parquet")
    dividends = normalize_dividends(pd.read_csv(ROOT / CURRENT / "inputs/dividends.csv"))
    cfg = json.loads((ROOT / CURRENT / "inputs/config/within_cycle_exit.json").read_text(encoding="utf-8"))
    models = {period: json.loads((ROOT/(EARLIER if period == "2015_2019" else CURRENT)/"inputs/within_models.json").read_text(encoding="utf-8"))["models"] for period in PERIODS}
    natural_by_key = samples.set_index(["cycle_id", "origin_index"])
    reference_decisions, reference_cycles, account_shares = {}, {}, {}
    for period in PERIODS:
        for cost in COSTS:
            key = (period, cost)
            reference_decisions[key] = pd.read_parquet(reference_path(period, cost, "decisions"))
            reference_cycles[key] = pd.read_parquet(reference_path(period, cost, "cycles"))
            daily = pd.read_parquet(ROOT/ACCOUNTS/period/cost/"A_SAVED_WEIGHT/daily.parquet", columns=["date", "shares"])
            account_shares[key] = daily.set_index("date").shares
    canonical, prediction_errors, label_errors = [], [], []
    paired_index = paired.set_index(["cycle_id", "origin_index"])
    natural_cycle_index = cycles.set_index("cycle_id")
    current_models = models["2020_2026"]
    for row in samples.itertuples():
        cycle = natural_cycle_index.loc[row.cycle_id]
        record = select_model(current_models, cycle.entry_index)
        saved = paired_index.loc[(row.cycle_id, row.origin_index)]
        prediction = np.nan
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= cycle.entry_index <= row.origin_index,
                    "自然参考预测使用未来模型或标签。")
            prediction = within_cycle_prediction(record["model"], [getattr(row, k) for k in FEATURES])
            prediction_errors.append(abs(prediction-float(saved.baseline_prediction)))
        else:
            require(pd.isna(saved.baseline_prediction), "原未知预测被填值。")
        target, _ = continuation_label(prices, dividends, row.reference_quantity, row.early_exit_index, row.exit_index, cfg["costs"]["BASE"], cfg["tick"])
        label_errors.append(abs(target-row.target))
        require(abs(saved.target_original_return-row.target) <= 1e-12, "E05保存标签与原自然样本不同。")
        result = {"period": period_of(row.origin), "cost": "ORIGINAL_BASE_LABEL", "cycle_id": int(row.cycle_id),
                  "origin": row.origin, "origin_index": int(row.origin_index), "entry_index": int(cycle.entry_index),
                  "entry_date": cycle.entry_date, "entry_year": int(cycle.entry_date.year), "mature_date": row.mature_date,
                  "prediction": prediction, "target": target, "prediction_sign": prediction_sign(prediction)}
        for cost in COSTS:
            period = result["period"]
            relation = "OUTSIDE_EVALUATION"
            if period in PERIODS:
                ref_cycle = natural_match(reference_cycles[(period,cost)], cycle.entry_index, cycle.entry_date)
                if ref_cycle is None:
                    relation = "NO_MATCHED_REFERENCE_ENTRY"
                else:
                    decisions = reference_decisions[(period,cost)]
                    active = decisions.learning_cycle_id.eq(ref_cycle.cycle_id) & decisions.origin_index.eq(row.origin_index)
                    if active.any():
                        relation = "ACTIVE_MATCHED_REFERENCE_ENTRY"
                    elif pd.notna(ref_cycle.exit_date) and row.origin >= ref_cycle.exit_date:
                        relation = "AFTER_MATCHED_REFERENCE_EXIT"
                    else:
                        relation = "NO_ACTIVE_MATCHED_REFERENCE_STATE"
            result[f"{cost.lower()}_reference_relation"] = relation
        canonical.append(result)
    require(max(prediction_errors, default=0) <= 1e-12 and max(label_errors, default=0) <= 1e-12, "原预测或自然标签复算不一致。")
    canonical_frame = pd.DataFrame(canonical)
    active_rows, active_prediction_errors = [], []
    for period in PERIODS:
        for cost in COSTS:
            ref_cycles = reference_cycles[(period,cost)].set_index("cycle_id")
            holding = reference_decisions[(period,cost)]
            holding = holding.loc[holding.learning_cycle_id.notna()]
            for row in holding.itertuples():
                cycle_id = int(row.learning_cycle_id)
                cycle = ref_cycles.loc[cycle_id]
                require(row.model_selection_index == cycle.entry_index, "活跃参考没有在真实入场首次收盘固定版本。")
                record = select_model(models[period], cycle.entry_index)
                prediction = float(row.continuation_prediction)
                x = [getattr(row,k) for k in FEATURES]
                if np.isfinite(prediction):
                    require(record and record["status"] == "FIT_COMPLETE" and record["latest_exit_index"] <= record["fit_index"] <= cycle.entry_index <= row.origin_index,
                            "活跃参考预测版本或成熟时钟不合法。")
                    active_prediction_errors.append(abs(prediction-within_cycle_prediction(record["model"], x)))
                    require(pd.Timestamp(row.learning_fit_origin) == pd.Timestamp(record["fit_origin"]), "活跃参考所选原模型日期不一致。")
                else:
                    require(row.learning_status != "PREDICTION_AVAILABLE", "预测可用状态却没有保存数值。")
                natural = natural_match(cycles, cycle.entry_index, cycle.entry_date)
                target, state_error, status, mature_date = np.nan, np.nan, "NO_MATCHED_NATURAL_ENTRY", pd.NaT
                natural_id = None
                if natural is not None:
                    natural_id = int(natural.cycle_id)
                    if pd.isna(natural.exit_date):
                        status = "UNMATURED_NATURAL_REFERENCE"
                    else:
                        end_indexes = np.flatnonzero(prices.date.eq(natural.exit_date))
                        require(len(end_indexes) == 1, "自然退出日期不在已绑定真实日线。")
                        end = int(end_indexes[0])
                        mature_date = prices.date.iloc[end]
                        key = (natural_id, int(row.origin_index))
                        if row.origin_index + 1 >= end:
                            status = "NO_LONGER_CONTINUATION_INTERVAL"
                        elif key not in natural_by_key.index:
                            status = "NO_SAVED_NATURAL_STATE"
                        else:
                            original_state = natural_by_key.loc[key]
                            state_error = float(np.max(np.abs(np.asarray(x,float)-original_state[FEATURES].to_numpy(float))))
                            target, _ = continuation_label(prices, dividends, cycle.entry_quantity, int(row.origin_index)+1, end, cfg["costs"]["BASE"], cfg["tick"])
                            status = "MATCHED_NATURAL_ENDPOINT_AUXILIARY_LABEL"
                active_rows.append({"period": period, "cost": cost, "cycle_id": cycle_id, "origin": row.origin,
                    "origin_index": int(row.origin_index), "entry_date": cycle.entry_date,
                    "entry_index": int(cycle.entry_index), "entry_year": int(cycle.entry_date.year),
                    "natural_cycle_id": natural_id, "mature_date": mature_date, "prediction": prediction,
                    "prediction_sign": prediction_sign(prediction), "learning_status": row.learning_status,
                    "target": target, "label_status": status, "eight_state_max_difference": state_error,
                    "negative_confirmation_count": int(row.negative_confirmation_count),
                    "learned_exit_requested": bool(row.learned_exit_requested), "requested_quantity": row.requested_quantity,
                    "actual_A_shares_at_close": int(account_shares[(period,cost)].get(row.origin,0))})
    require(max(active_prediction_errors, default=0) <= 1e-12, "活跃参考保存预测复算不一致。")
    active_frame = pd.DataFrame(active_rows)
    confirmed = first_confirmed_events(active_frame)
    summaries = []
    for period in PERIODS:
        natural_view = canonical_frame.loc[canonical_frame.period.eq(period)]
        for sign in ("NEGATIVE", "NONNEGATIVE", "NO_VIEW"):
            group = natural_view.loc[natural_view.prediction_sign.eq(sign)]
            summaries.append({"period": period, "cost": "ORIGINAL_BASE_LABEL", "view": "CANONICAL_REFERENCE_ALL_STATES",
                              "prediction_sign": sign, **summarize_group(group)})
        for cost in COSTS:
            views = {"ACTIVE_REFERENCE_ALL": active_frame.loc[active_frame.period.eq(period) & active_frame.cost.eq(cost)]}
            views["ACTIVE_REFERENCE_WITH_A_HOLDING"] = views["ACTIVE_REFERENCE_ALL"].loc[views["ACTIVE_REFERENCE_ALL"].actual_A_shares_at_close.gt(0)]
            views["FIRST_CONFIRMED_NEGATIVE_EVENT"] = confirmed.loc[confirmed.period.eq(period) & confirmed.cost.eq(cost)]
            for view, frame in views.items():
                for sign in ("NEGATIVE", "NONNEGATIVE", "NO_VIEW"):
                    group = frame.loc[frame.prediction_sign.eq(sign)]
                    summaries.append({"period": period, "cost": cost, "view": view, "prediction_sign": sign, **summarize_group(group)})
    save_table("自然参考原预测与标签", canonical_frame)
    save_table("真实活跃参考原预测与辅助标签", active_frame)
    save_table("每周期首次确认负预测", confirmed)
    summary = {"study": STUDY, "completed_at": now(), "status": "COMPLETED_FIXED_ORIGINAL_SIGN_CALIBRATION_DIAGNOSTIC",
        "canonical_rows": len(canonical_frame), "canonical_available_predictions": int(np.isfinite(canonical_frame.prediction).sum()),
        "active_reference_rows": len(active_frame), "active_available_predictions": int(np.isfinite(active_frame.prediction).sum()),
        "active_label_status": active_frame.groupby(["period","cost","label_status"]).size().rename("rows").reset_index().to_dict("records"),
        "canonical_reference_relations": [{"period":period,"cost":cost,"relations": canonical_frame.loc[canonical_frame.period.eq(period),f"{cost.lower()}_reference_relation"].value_counts().to_dict()} for period in PERIODS for cost in COSTS],
        "prediction_recomputation_max_error": max(prediction_errors,default=0),
        "active_prediction_recomputation_max_error": max(active_prediction_errors,default=0),
        "natural_label_recomputation_max_error": max(label_errors,default=0),
        "views": summaries, "new_model_fits": 0, "new_strategy_accounts": 0,
        "account_return_sharpe": "NOT_COMPUTED", "goal_achieved": False,
        "independent_validation": "NOT_ESTABLISHED", "history_role": protocol["history_role"]}
    write_json(OUT/"summary.json", summary, exclusive=True)
    check_sources()
    files = sorted((OUT/"results").glob("*")) + [OUT/"summary.json"]
    write_json(OUT/"verification_receipt.json", {"checked_at": now(), "status": "PASS_SAVED_PREDICTION_AND_ORIGINAL_LABEL_RECOMPUTATION",
        "sources_unchanged": len(protocol["sources"]), "new_model_fits": 0, "new_strategy_accounts": 0,
        "artifacts": [{"path":str(p.relative_to(ROOT)),"sha256":digest(p)} for p in files]}, exclusive=True)
    print(f"E06完成：{len(canonical_frame)}自然状态、{len(active_frame)}活跃参考状态、{len(confirmed)}首次确认事件；原预测和原标签复算一致。",flush=True)


def main():
    parser = argparse.ArgumentParser(description="原退出预测正负与继续收益的隔离诊断")
    parser.add_argument("action", choices=["freeze", "run", "check"])
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "run":
        run()
    else:
        protocol = check_sources()
        print(f"E06冻结来源未变：{len(protocol['sources'])}份。",flush=True)


if __name__ == "__main__":
    main()
