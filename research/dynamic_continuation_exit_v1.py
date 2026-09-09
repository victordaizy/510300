"""在已成熟参考周期上拟合含后续退出选择的继续价值。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import CN, FEATURES, ExitController, chinese_formula, continuation_label, fit_one, training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_dynamic_continuation_exit_v1"
CONFIG = ROOT / "config/510300_dynamic_continuation_exit_v1.json"
P31 = ROOT / "reports/research/510300_learned_cycle_exit_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PRIMARY = "DYNAMIC_CONTINUATION"
CONTROL = "NATURAL_SINGLE_CONFIRM"
NAMES = {PRIMARY: "含后续退出选择的线性模型", CONTROL: "原自然退出标签改为单次确认"}


def transition_samples(data, dividends, samples, cfg):
    result = samples.copy().sort_values(["cycle_id", "origin_index"]).reset_index(drop=True)
    result["natural_exit_target"] = result.target
    result["natural_exit_extra_dividend_cny"] = result.extra_dividend_cny
    result["label_exit_index"] = result.early_exit_index + 1
    result["successor_origin_index"] = -1
    result["successor_scale"] = 0.
    result["one_step_target"] = np.nan
    for cid, group in result.groupby("cycle_id", sort=False):
        require(group.exit_index.nunique() == 1 and group.reference_quantity.nunique() == 1, "同一参考周期的终点或持仓数量发生变化")
        require(group.origin_index.diff().dropna().eq(1).all(), "参考周期状态不连续，不能跳日连接")
        require(int(group.origin_index.iloc[-1]) + 2 == int(group.exit_index.iloc[-1]), "参考周期末状态没有对应原退出边界")
        for pos, row in enumerate(group.itertuples()):
            early, late = int(row.early_exit_index), int(row.label_exit_index)
            require(early == row.origin_index + 1 and late <= row.exit_index, "一步标签开盘时点非法")
            target, extra = continuation_label(data, dividends, int(row.reference_quantity), early, late, cfg["costs"]["BASE"], cfg["tick"])
            result.loc[row.Index, ["one_step_target", "target", "extra_dividend_cny"]] = [target, target, extra]
            if pos + 1 < len(group):
                successor = int(group.origin_index.iloc[pos + 1])
                require(successor == row.origin_index + 1, "后继状态跨越缺失日期")
                result.loc[row.Index, "successor_origin_index"] = successor
                result.loc[row.Index, "successor_scale"] = float(data.open.iloc[late] / data.open.iloc[early])
    require(np.isfinite(result.one_step_target).all(), "一步收益不完整")
    pd.testing.assert_frame_equal(result[FEATURES + ["cycle_id", "origin_index", "exit_index", "mature_date"]],
                                  samples.sort_values(["cycle_id", "origin_index"]).reset_index(drop=True)[FEATURES + ["cycle_id", "origin_index", "exit_index", "mature_date"]])
    return result


def successor_positions(rows):
    lookup = {(int(r.cycle_id), int(r.origin_index)): i for i, r in enumerate(rows.itertuples())}
    result = np.full(len(rows), -1, dtype=int)
    for i, row in enumerate(rows.itertuples()):
        if row.successor_origin_index >= 0:
            key = (int(row.cycle_id), int(row.successor_origin_index))
            require(key in lookup, "训练集合缺少同一成熟周期的后继状态")
            result[i] = lookup[key]
    return result


def make_target(rows, positions, previous_predictions):
    target = rows.one_step_target.to_numpy(float).copy()
    live = positions >= 0
    target[live] += rows.successor_scale.to_numpy(float)[live] * np.maximum(previous_predictions[positions[live]], 0.)
    require(np.isfinite(target).all(), "动态继续价值出现非有限数值")
    return target


def fitted_iteration(rows, cfg):
    positions = successor_positions(rows)
    previous = np.zeros(len(rows), dtype=float)
    rows = rows.copy()
    history = []
    for iteration in range(1, cfg["value_iterations"] + 1):
        rows["target"] = make_target(rows, positions, previous)
        fitted = fit_one(rows, "RIDGE", cfg)
        design = np.clip((rows[FEATURES].to_numpy(float) - np.array(fitted["mean"])) / np.array(fitted["scale"]),
                         -fitted["feature_clip"], fitted["feature_clip"])
        current = fitted["intercept"] + design @ np.array(fitted["coefficients"])
        require(np.isfinite(current).all(), "动态继续模型输出出现非有限数值")
        history.append({"iteration": iteration, "target_min": float(rows.target.min()), "target_max": float(rows.target.max()),
                        "prediction_min": float(current.min()), "prediction_max": float(current.max()),
                        "maximum_prediction_change": float(np.abs(current - previous).max()),
                        "negative_predictions": int((current < 0).sum())})
        previous = current
    return fitted, history, rows.target.to_numpy(float)


def train_models(data, samples, originals, cfg):
    models, receipts, memberships, iterations = [], [], [], []
    chinese = ["# 每月动态继续价值模型的实际中文规则", "", "每次训练固定完成六十次回归，只保存第六十次为当月实际决策模型。后续退出选择属于模型估计，不是已知未来最优退出。", ""]
    for original in originals:
        t = int(original["fit_index"])
        rows, ids = training_rows(samples, t, cfg)
        usable = len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
        require(ids == original["training_cycles"] and len(rows) == original["training_rows"] and usable == (original["status"] == "FIT_COMPLETE"),
                "动态学习改变了原训练周期、样本数量或可训练时点")
        require(not len(rows) or ((rows.label_exit_index <= rows.exit_index) & (rows.exit_index <= t)).all(), "动态模型提前使用未成熟周期")
        fitted, history, targets = fitted_iteration(rows, cfg) if usable else (None, [], np.array([]))
        if usable:
            np.testing.assert_allclose(fitted["mean"], original["model"]["mean"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(fitted["scale"], original["model"]["scale"], rtol=0, atol=1e-12)
        record = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()), "fit_time": data.date.iloc[t] + pd.Timedelta(hours=15, minutes=5),
                  "status": "FIT_COMPLETE" if usable else "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS", "training_cycles": ids,
                  "training_cycle_count": len(ids), "training_rows": len(rows), "regression_fits": len(history),
                  "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None,
                  "latest_exit_date": rows.mature_date.max() if len(rows) else None, "model": fitted}
        models.append(record)
        receipts.append({k: v for k, v in record.items() if k not in ["model", "training_cycles"]})
        iterations.extend({"fit_index": t, **r} for r in history)
        if usable:
            memberships.extend({"fit_index": t, "cycle_id": int(r.cycle_id), "origin_index": int(r.origin_index),
                                "exit_index": int(r.exit_index), "successor_origin_index": int(r.successor_origin_index),
                                "sample_weight": float(r.sample_weight), "final_training_target": float(target)}
                               for r, target in zip(rows.itertuples(), targets))
        chinese.extend([f"## {record['fit_origin']}", "", f"使用{len(ids)}个已结束参考周期、{len(rows)}条状态，完成{len(history)}次回归。", ""])
        chinese.extend(chinese_formula(fitted) + [""] if usable else ["成熟周期或状态不足，保留无模型状态，沿用原价格时间退出。", ""])
        if usable and sum(r["status"] == "FIT_COMPLETE" for r in receipts) % 25 == 0:
            print(f"已完成{sum(r['status']=='FIT_COMPLETE' for r in receipts)}个月度动态继续模型。", flush=True)
    write_json(OUT / "saved_models.json", {"models": models, "feature_names": dict(zip(FEATURES, CN)), "value_iterations": cfg["value_iterations"]})
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(iterations).to_csv(OUT / "all_value_iterations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memberships).to_parquet(OUT / "training_memberships.parquet", index=False)
    (OUT / "每月动态退出模型中文规则.md").write_text("\n".join(chinese) + "\n", encoding="utf-8")
    return models, receipts


def simulate_single(data, dividends, cfg, cost, start, models):
    ledger, decisions, cycles = simulate_rearmed_exit(data, dividends, cfg, cost, start, make_rules(data)["D60_INTRA"], cfg["specification"],
                                                     ExitController(data, models, confirmation_days=1))
    # 原账户的执行逻辑支持控制器自定义确认次数；这里只把固定中文说明同步为本轮实际规则。
    old = "连续两个收盘预测继续持有收益为负，学习条件请求退出"
    new = "当日收盘预测继续持有收益为负，学习条件请求退出"
    for frame, column in [(ledger, "execution_reasons"), (decisions, "exit_reasons"), (cycles, "exit_reasons")]:
        if column in frame:
            frame[column] = frame[column].str.replace(old, new, regex=False)
    return ledger, decisions, cycles


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    training = json.loads((ROOT / "config/510300_learned_cycle_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
                              "specification", "saved_models"]}
    cfg.update({k: training[k] for k in ["recent_cycles", "minimum_cycles", "minimum_rows", "feature_columns", "feature_names", "feature_clip", "ridge_alpha"]})
    cfg.update(study_id="510300_DYNAMIC_CONTINUATION_EXIT_V1", round=52, registered_at=now(), primary=PRIMARY, candidate_configurations=2,
               names=NAMES, value_iterations=60, confirmation_days=1, rules="docs/510300_DYNAMIC_CONTINUATION_EXIT_V1.md",
               samples=str((P31 / "all_reference_samples.parquet").relative_to(ROOT)), position_impact=0)
    paths = [Path(__file__), ROOT / "research/rearmed_cycle_exit_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
             ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / cfg["rules"],
             ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["saved_models"], ROOT / cfg["samples"],
             ROOT / "tests/test_dynamic_continuation_exit_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第52轮已登记动态继续价值退出及原标签单次确认对照，共两个设置。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "动态继续价值登记内容发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    samples = pd.read_parquet(ROOT / cfg["samples"])
    samples = transition_samples(data, dividends, samples[samples.signal == "D60_INTRA"].copy(), cfg)
    samples.to_parquet(OUT / "one_step_reference_samples.parquet", index=False)
    originals = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    models, receipts = train_models(data, samples, originals, cfg)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        for cost_id, cost in cfg["costs"].items():
            folder, accounts = OUT / period / cost_id, {}
            names = dict(NAMES)
            for key, series in [(PRIMARY, models), (CONTROL, originals)]:
                ledger, decisions, cycles = simulate_single(frame, dividends, cfg, cost, start, series)
                save_account(folder, key, ledger, decisions)
                cycles.to_csv(folder / f"{key}_cycles.csv", index=False, encoding="utf-8-sig")
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "动态继续退出账户结算失败")
                require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "动态继续退出违反买入次日可卖")
                holding = decisions[decisions.learning_cycle_id.notna()]
                coverage.append({"period": period, "cost": cost_id, "model": key, "holding_decisions": len(holding),
                                 "model_available_rows": int(holding.learning_status.eq("PREDICTION_AVAILABLE").sum()),
                                 "no_model_rows": int(holding.learning_status.ne("PREDICTION_AVAILABLE").sum()),
                                 "learned_exit_cycles": int(cycles.exit_reasons.str.contains("学习条件", regex=False).sum()),
                                 "completed_round_trips": len(cycles)})
                accounts[key] = ledger
            for key, name in [("REARM_RIDGE", "原自然标签两次确认"), ("REARM_NONE", "原价格时间退出"), ("BUY_HOLD", "买入持有")]:
                accounts[key] = pd.read_parquet(P32 / period / cost_id / f"{key}_ledger.parquet")
                names[key] = name
            accounts["PANIC_LEARNED_HALF"] = pd.read_parquet(ROOT / "reports/research/510300_panic_learned_equal_blend_v1" / period / cost_id / "PANIC_LEARNED_HALF_ledger.parquet")
            names["PANIC_LEARNED_HALF"] = "原高波动急跌与学习退出各半"
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, ledger in accounts.items():
                if key not in NAMES:
                    ledger.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "动态继续退出评价日历不一致")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(ledger, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in ledger.groupby(ledger.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = ledger[ledger.date.between(left, right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：两个单次确认账户及四个原样对照已完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("model_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    completed = sum(r["status"] == "FIT_COMPLETE" for r in receipts)
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "DYNAMIC_CONTINUATION_EXIT_COMPLETE", "candidate_configurations": 2,
              "evaluation_accounts": len(main), "new_accounts_generated": 4, "reused_control_accounts": 8,
              "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 4, "reused_earlier_accounts": 8,
              "new_reference_accounts": 0, "new_model_fits": sum(r["regression_fits"] for r in receipts),
              "monthly_final_models": completed, "no_view_fit_origins": len(receipts) - completed,
              "reference_state_rows": len(samples), "reference_cycles": int(samples.cycle_id.nunique()),
              "all_metrics": main, "earlier_diagnostics": earlier, "model_coverage": coverage,
              "primary": [m for m in main if m["model"] == PRIMARY],
              "post_selected_best_base": max((m for m in main if m["model"] in NAMES and m["cost"] == "BASE"), key=lambda x: x["net_sharpe"]),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] in NAMES),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"月度模型": completed, "实际回归次数": result["new_model_fits"], "主评价": [m for m in main if m["model"] in NAMES],
                      "较早诊断": [m for m in earlier if m["model"] in NAMES]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
