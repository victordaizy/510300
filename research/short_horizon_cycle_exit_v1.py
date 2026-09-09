"""只把参考周期的继续持有标签上限改为五个开盘间隔。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import ExitController, FEATURES, chinese_formula, continuation_label, fit_one, training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_short_horizon_cycle_exit_v1"
CONFIG = ROOT / "config/510300_short_horizon_cycle_exit_v1.json"
P31 = ROOT / "reports/research/510300_learned_cycle_exit_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PRIMARY = "SHORT_HORIZON_RIDGE"


def relabel_samples(data, dividends, samples, config):
    result = samples.copy()
    result["natural_exit_target"] = result.target
    result["natural_exit_extra_dividend_cny"] = result.extra_dividend_cny
    result["label_exit_index"] = np.minimum(result.exit_index, result.early_exit_index + config["label_max_intervals"]).astype(int)
    require(((result.label_exit_index > result.early_exit_index) & (result.label_exit_index <= result.exit_index)).all(), "短期退出标签时间非法")
    result["label_exit_date"] = [data.date.iloc[t] for t in result.label_exit_index]
    targets, distributions = [], []
    for row in result.itertuples():
        value, dividend = continuation_label(data, dividends, int(row.reference_quantity), int(row.early_exit_index),
                                             int(row.label_exit_index), config["costs"]["BASE"], config["tick"])
        targets.append(value)
        distributions.append(dividend)
    result["target"], result["extra_dividend_cny"] = targets, distributions
    require(np.isfinite(result.target).all(), "短期标签出现缺失")
    # 原周期结束时间不改写：较短标签不能让尚未结束的周期提前进入最近周期集合。
    pd.testing.assert_frame_equal(result[FEATURES + ["cycle_id", "origin_index", "exit_index", "mature_date"]],
                                  samples[FEATURES + ["cycle_id", "origin_index", "exit_index", "mature_date"]])
    return result


def train_models(data, samples, originals, cfg):
    models, receipts, memberships = [], [], []
    chinese = ["# 五个开盘间隔内继续持有收益：每月实际线性模型", "",
               "每个原参考周期仍需实际结束后才能进入训练。目标最多继续五个开盘间隔，原参考退出更早时以该退出为终点。", ""]
    for original in originals:
        t = int(original["fit_index"])
        rows, ids = training_rows(samples, t, cfg)
        usable = len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
        require(ids == original["training_cycles"] and len(rows) == original["training_rows"] and usable == (original["status"] == "FIT_COMPLETE"),
                "更换标签意外改变了训练周期、样本行或可训练时点")
        require(not len(rows) or ((rows.label_exit_index <= rows.exit_index) & (rows.exit_index <= t)).all(), "短期模型提前使用了未成熟标签或未结束周期")
        fitted = fit_one(rows, "RIDGE", cfg) if usable else None
        if usable:
            np.testing.assert_allclose(fitted["mean"], original["model"]["mean"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(fitted["scale"], original["model"]["scale"], rtol=0, atol=1e-12)
        record = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()), "fit_time": data.date.iloc[t] + pd.Timedelta(hours=15, minutes=5),
                  "status": "FIT_COMPLETE" if usable else "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS", "training_cycles": ids,
                  "training_cycle_count": len(ids), "training_rows": len(rows),
                  "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None,
                  "latest_exit_date": rows.mature_date.max() if len(rows) else None,
                  "latest_label_exit_index": int(rows.label_exit_index.max()) if len(rows) else None,
                  "shortened_label_rows": int((rows.label_exit_index < rows.exit_index).sum()), "model": fitted}
        models.append(record)
        receipts.append({k: v for k, v in record.items() if k not in ["model", "training_cycles"]})
        if usable:
            memberships += [{"fit_index": t, "cycle_id": int(r.cycle_id), "origin_index": int(r.origin_index), "exit_index": int(r.exit_index),
                             "label_exit_index": int(r.label_exit_index), "sample_weight": float(r.sample_weight)} for r in rows.itertuples()]
        chinese += [f"## {record['fit_origin']}", "", f"已结束参考周期{len(ids)}个，状态{len(rows)}条，其中{record['shortened_label_rows']}条标签期限缩短。", ""]
        chinese += chinese_formula(fitted) + [""] if usable else ["成熟周期或状态不足，保留无模型状态，沿用原价格和时间退出。", ""]
    write_json(OUT / "saved_models.json", {"models": models, "feature_names": cfg["feature_names"], "label_max_intervals": cfg["label_max_intervals"]})
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memberships).to_parquet(OUT / "training_memberships.parquet", index=False)
    (OUT / "每月短期退出模型中文规则.md").write_text("\n".join(chinese) + "\n", encoding="utf-8")
    print(f"五日内标签训练完成：{sum(r['status']=='FIT_COMPLETE' for r in receipts)}次拟合，训练周期、行数及标准化均与原模型一致。", flush=True)
    return models, receipts


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    training = json.loads((ROOT / "config/510300_learned_cycle_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
                              "confirmation_days", "specification", "saved_models"]}
    cfg.update({k: training[k] for k in ["recent_cycles", "minimum_cycles", "minimum_rows", "feature_columns", "feature_names", "feature_clip", "ridge_alpha"]})
    cfg.update(study_id="510300_SHORT_HORIZON_CYCLE_EXIT_V1", round=44, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
               label_max_intervals=5, rules="docs/510300_SHORT_HORIZON_CYCLE_EXIT_V1.md",
               samples=str((P31 / "all_reference_samples.parquet").relative_to(ROOT)), position_impact=0)
    paths = [Path(__file__), ROOT / "research/rearmed_cycle_exit_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
             ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / cfg["rules"],
             ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["saved_models"], ROOT / cfg["samples"],
             ROOT / "tests/test_short_horizon_cycle_exit_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第44轮已登记一个五日内标签退出模型，未重建参考账户或补充数据。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "短期退出登记内容发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    samples = pd.read_parquet(ROOT / cfg["samples"])
    samples = relabel_samples(data, dividends, samples[samples.signal == "D60_INTRA"].copy(), cfg)
    samples.to_parquet(OUT / "short_horizon_reference_samples.parquet", index=False)
    originals = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    models, receipts = train_models(data, samples, originals, cfg)
    main, early, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_rearmed_exit(frame, dividends, cfg, cost, start, make_rules(frame)["D60_INTRA"], cfg["specification"],
                ExitController(frame, models, cfg["confirmation_days"]))
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "短期退出账户结算失败")
            require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "短期退出违反买入次日可卖")
            holding = decisions[decisions.learning_cycle_id.notna()]
            coverage.append({"period": period, "cost": cost_id, "holding_decisions": len(holding),
                             "model_available_rows": int((holding.learning_status == "PREDICTION_AVAILABLE").sum()),
                             "no_model_rows": int((holding.learning_status != "PREDICTION_AVAILABLE").sum()),
                             "learned_exit_cycles": int(cycles.exit_reasons.str.contains("学习条件", regex=False).sum()), "completed_round_trips": len(cycles)})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "五日内继续持有标签的线性退出"}
            for key, name in [("REARM_RIDGE", "原自然退出期限的线性模型"), ("REARM_NONE", "原价格时间退出及等待新机会"), ("BUY_HOLD", "买入持有")]:
                saved = pd.read_parquet(P32 / period / cost_id / f"{key}_ledger.parquet")
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "短期退出完整评价日期不一致")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                destination.append(m)
                if period == "evaluation":
                    for year, group in saved.groupby(saved.date.dt.year):
                        yearly.append({"cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[(saved.date >= left) & (saved.date <= right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：一个短期标签账户及三个原样对照已完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("model_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "SHORT_HORIZON_CYCLE_EXIT_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": 8, "new_accounts_generated": 2, "reused_control_accounts": 6, "earlier_diagnostic_accounts": 8,
              "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 6, "new_reference_accounts": 0,
              "completed_fits": sum(r["status"] == "FIT_COMPLETE" for r in receipts), "no_view_fit_origins": sum(r["status"] != "FIT_COMPLETE" for r in receipts),
              "reference_state_rows": len(samples), "reference_cycles": samples.cycle_id.nunique(),
              "shortened_label_rows": int((samples.label_exit_index < samples.exit_index).sum()),
              "all_metrics": main, "earlier_diagnostics": early, "model_coverage": coverage, "latest_training": receipts[-1],
              "primary": [m for m in main if m["model"] == PRIMARY],
              "post_selected_best_base": next(m for m in main if m["model"] == PRIMARY and m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] == PRIMARY),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"完成拟合": result["completed_fits"], "主评价": result["primary"],
                      "较早": [m for m in early if m["model"] == PRIMARY]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
