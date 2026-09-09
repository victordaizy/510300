"""复用完整成熟周期，按退出时间降低旧周期的训练权重。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import ExitController, training_rows, fit_one, chinese_formula
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_recency_weighted_exit_v1"
CONFIG = ROOT / "config/510300_recency_weighted_exit_v1.json"
P31 = ROOT / "reports/research/510300_learned_cycle_exit_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PRIMARY = "RECENCY_RIDGE"


def recency_weights(rows, fit_index, half_life):
    weighted = rows.copy()
    cycles = rows[["cycle_id", "exit_index"]].drop_duplicates().sort_values("cycle_id").copy()
    require(not cycles.cycle_id.duplicated().any(), "同一参考周期的实际退出时点不唯一")
    require((cycles.exit_index <= fit_index).all(), "样本权重包含未来实际退出")
    cycles["age_trading_days"] = fit_index - cycles.exit_index
    cycles["raw_weight"] = np.exp2(-cycles.age_trading_days.to_numpy(float) / half_life)
    if len(cycles):
        cycles["cycle_weight"] = cycles.raw_weight / cycles.raw_weight.sum() * len(cycles)
        by_id = cycles.set_index("cycle_id").cycle_weight
        weighted["sample_weight"] = weighted.cycle_id.map(by_id) / weighted.groupby("cycle_id").origin_index.transform("count")
        effective = float(cycles.cycle_weight.sum() ** 2 / (cycles.cycle_weight ** 2).sum())
        require(abs(weighted.sample_weight.sum() - len(cycles)) < 1e-9, "新旧模型总样本权重不一致")
    else:
        cycles["cycle_weight"] = pd.Series(dtype=float)
        weighted["sample_weight"] = pd.Series(dtype=float)
        effective = 0.
    return weighted, cycles, effective


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    train = json.loads((ROOT / "config/510300_learned_cycle_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
                              "confirmation_days", "specification", "saved_models"]}
    cfg.update({k: train[k] for k in ["recent_cycles", "minimum_cycles", "minimum_rows", "feature_columns", "feature_names", "feature_clip", "ridge_alpha"]})
    cfg.update(study_id="510300_RECENCY_WEIGHTED_EXIT_V1", round=38, registered_at=now(), primary=PRIMARY,
               candidate_configurations=1, half_life_trading_days=242, rules="docs/510300_RECENCY_WEIGHTED_EXIT_V1.md",
               samples=str((P31 / "all_reference_samples.parquet").relative_to(ROOT)), position_impact=0)
    paths = [Path(__file__), ROOT / "research/rearmed_cycle_exit_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
             ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["saved_models"], ROOT / cfg["samples"],
             ROOT / "tests/test_recency_weighted_exit_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第38轮已登记一个按样本年龄加权的退出模型，半衰期为242个交易日。", flush=True)


def train_models(data, samples, originals, cfg):
    models, receipts, memberships, cycle_weights = [], [], [], []
    chinese = ["# 按交易年龄加权的线性退出模型：每月实际中文规则", "", "权重按已结束参考周期年龄每242个交易日减半，周期总权重合计与原周期数一致。", ""]
    for original in originals:
        t = int(original["fit_index"])
        rows, ids = training_rows(samples, t, cfg)
        require(ids == original["training_cycles"] and len(rows) == original["training_rows"], "新旧退出模型的参考周期集合发生变化")
        weighted, weights, effective = recency_weights(rows, t, cfg["half_life_trading_days"])
        usable = len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
        require(usable == (original["status"] == "FIT_COMPLETE"), "新旧退出模型成熟时点不同")
        age_calendar = (data.date.iloc[t] - rows[["cycle_id", "mature_date"]].drop_duplicates().mature_date).dt.days
        record = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()), "fit_time": data.date.iloc[t] + pd.Timedelta(hours=15, minutes=5),
                  "status": "FIT_COMPLETE" if usable else "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS", "training_cycle_count": len(ids), "training_rows": len(rows),
                  "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None,
                  "oldest_exit_date": rows.mature_date.min() if len(rows) else None, "newest_exit_date": rows.mature_date.max() if len(rows) else None,
                  "oldest_cycle_age_calendar_days": int(age_calendar.max()) if len(rows) else None,
                  "cycles_older_than_365_calendar_days": int((age_calendar > 365).sum()),
                  "cycles_older_than_1095_calendar_days": int((age_calendar > 1095).sum()),
                  "effective_weighted_cycles": effective, "model": fit_one(weighted, "RIDGE", cfg) if usable else None}
        models.append(record)
        receipts.append({k: v for k, v in record.items() if k != "model"})
        cycle_weights += [{"fit_index": t, "fit_origin": record["fit_origin"], "used_in_fit": usable, **w} for w in weights.to_dict("records")]
        if usable:
            memberships += [{"fit_index": t, "cycle_id": int(r.cycle_id), "origin_index": int(r.origin_index), "exit_index": int(r.exit_index),
                             "sample_weight": float(r.sample_weight)} for r in weighted.itertuples()]
        chinese += [f"## {record['fit_origin']}", "", f"已结束参考周期{len(ids)}个，状态{len(rows)}条；按权重计算的有效周期数为{effective:.4f}。", ""]
        chinese += chinese_formula(record["model"]) + [""] if usable else ["成熟样本不足，模型缺失，按原价格及时间条件退出。", ""]
    write_json(OUT / "saved_models.json", {"models": models, "half_life_trading_days": 242, "feature_names": cfg["feature_names"]})
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memberships).to_parquet(OUT / "training_memberships.parquet", index=False)
    pd.DataFrame(cycle_weights).to_csv(OUT / "cycle_weights_and_ages.csv", index=False, encoding="utf-8-sig")
    (OUT / "每月加权退出模型中文规则.md").write_text("\n".join(chinese) + "\n", encoding="utf-8")
    print(f"保持原成熟时点，完成{sum(r['status']=='FIT_COMPLETE' for r in receipts)}次加权训练。", flush=True)
    return models, receipts


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "交易年龄加权登记内容发生变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    samples = pd.read_parquet(ROOT / cfg["samples"])
    samples = samples[samples.signal == "D60_INTRA"].copy()
    originals = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    models, receipts = train_models(data, samples, originals, cfg)
    main, early, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_rearmed_exit(frame, dividends, cfg, cost, start, make_rules(frame)["D60_INTRA"], cfg["specification"],
                ExitController(frame, models, cfg["confirmation_days"]))
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "加权退出账户结算失败")
            if len(cycles):
                require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "加权退出违反买入次日可卖")
            holding = decisions[decisions.learning_cycle_id.notna()]
            coverage.append({"period": period, "cost": cost_id, "holding_decisions": len(holding),
                             "model_available_rows": int((holding.learning_status == "PREDICTION_AVAILABLE").sum()),
                             "no_model_rows": int((holding.learning_status != "PREDICTION_AVAILABLE").sum()),
                             "learned_exit_cycles": int(cycles.exit_reasons.str.contains("学习条件", regex=False).sum()), "completed_round_trips": len(cycles)})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "近期交易权重更高的线性退出"}
            for key, name in [("REARM_RIDGE", "原周期等权线性退出"), ("REARM_NONE", "仅等待新机会，不学习退出"), ("BUY_HOLD", "买入持有")]:
                saved = pd.read_parquet(P32 / period / cost_id / f"{key}_ledger.parquet")
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "加权退出评价日期不完整")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                dest.append(m)
                if period == "evaluation":
                    for year, group in saved.groupby(saved.date.dt.year):
                        yearly.append({"cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[(saved.date >= left) & (saved.date <= right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：一个加权退出和三个原样对照账户完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("model_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "RECENCY_WEIGHTED_EXIT_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": 8, "new_accounts_generated": 2, "reused_control_accounts": 6, "earlier_diagnostic_accounts": 8,
              "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 6, "new_reference_accounts": 0,
              "completed_fits": sum(r["status"] == "FIT_COMPLETE" for r in receipts), "no_view_fit_origins": sum(r["status"] != "FIT_COMPLETE" for r in receipts),
              "all_metrics": main, "earlier_diagnostics": early, "model_coverage": coverage, "latest_training": receipts[-1],
              "primary": [m for m in main if m["model"] == PRIMARY],
              "post_selected_best_base": next(m for m in main if m["model"] == PRIMARY and m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] == PRIMARY),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"状态": result["status"], "新方案": result["primary"], "较早": [m for m in early if m["model"] == PRIMARY],
                      "最新训练年龄": receipts[-1], "持仓学习覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
