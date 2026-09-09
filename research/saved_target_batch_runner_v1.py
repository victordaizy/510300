"""共读数据和保存对照，一次运行事先列明的有限目标候选。"""
import json
import time
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import now, digest, require, write_json


def run_saved_target_batch(root, out, config_path, candidates, parents, controls, build_frames):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    require(cfg["candidate_models"] == list(candidates) and cfg["candidate_configurations"] == len(candidates), "批次固定候选身份或数量不同")
    for item in cfg["frozen_files"]:
        require(digest(root / item["path"]) == item["sha256"], "批次冻结来源改变")
    write_json(out / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(config_path)}, exclusive=True)
    began = time.perf_counter()
    data = pd.read_parquet(root / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(root / cfg["dividends"]))
    main, earlier, yearly, eras, coverage, target_summaries = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        sources = {cost: {model: pd.read_parquet(folder / period / cost / f"{model}_decisions.parquet").assign(source_cost=cost, source_model=model)
            for model, folder in parents.items()} for cost in cfg["costs"]}
        factor_frames, summaries = build_frames(frame, sources, cfg, start)
        target_summaries.extend({"period": period, **row} for row in summaries)
        for cost_id, cost in cfg["costs"].items():
            factors = factor_frames[cost_id]
            folder = out / period / cost_id
            folder.mkdir(parents=True)
            factors.to_parquet(folder / "factors.parquet", index=False)
            accounts, names = {}, dict(candidates)
            for model in candidates:
                targets = factors[model+"_target"].to_numpy(float)
                ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, model,
                    targets=targets, event_mask=np.ones(len(frame), bool))
                decisions = decisions.merge(factors.rename(columns={"date": "origin", "origin_index": "factor_origin_index"}), on="origin", how="left", validate="one_to_one")
                save_account(folder, model, ledger, decisions)
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "批次账户财富或终点退出不同")
                accounts[model] = ledger
                coverage.append({"period": period, "cost": cost_id, "model": model, "holding_closes": int(ledger.shares.gt(0).sum()),
                    "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                    "unknown_target_origins": int(decisions.reference_weight.isna().sum()),
                    "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()), "mean_exposure": float(ledger.exposure.mean())})
            for model, (parent_folder, name) in controls.items():
                saved = pd.read_parquet(parent_folder / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            benchmark = summarize(accounts["BUY_HOLD"], cfg)
            calendar = pd.DatetimeIndex(frame.date[frame.date.ge(start)])
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(calendar), "批次账户与对照日历不同")
                measured = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                measured["annualized_return_excess_vs_buy_hold"] = measured["annualized_return"]-benchmark["annualized_return"]
                measured["meets_point_target"] = measured["net_sharpe"] is not None and measured["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(measured)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：{len(candidates)}条新候选账户和{len(controls)}条保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage), ("target_coverage.csv", target_summaries)]:
        pd.DataFrame(rows).to_csv(out / name, index=False, encoding="utf-8-sig")
    candidate_main = [m for m in main if m["model"] in candidates]
    primary = [m for m in main if m["model"] == cfg["primary"]]
    base = [m for m in candidate_main if m["cost"] == "BASE" and m["net_sharpe"] is not None]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "FINITE_SAVED_TARGET_BATCH_ACCOUNTS_COMPLETE",
        "candidate_configurations": len(candidates), "candidate_models": list(candidates), "evaluation_accounts": len(main),
        "new_accounts_generated": len(candidates)*len(cfg["costs"]), "reused_control_accounts": len(controls)*len(cfg["costs"]),
        "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": len(candidates)*len(cfg["costs"]),
        "reused_earlier_accounts": len(controls)*len(cfg["costs"]), "new_model_fits": 0, "new_reference_accounts": 0,
        "all_metrics": main, "earlier_diagnostics": earlier, "account_coverage": coverage, "target_coverage": target_summaries,
        "primary": primary, "post_selected_best_base": max(base, key=lambda m: m["net_sharpe"]) if base else None,
        "historical_point_target_met": any(m["meets_point_target"] for m in candidate_main), "run_seconds": time.perf_counter()-began,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(out / "result.json", result, exclusive=True)
    print(json.dumps({"新候选主历史": candidate_main, "新候选较早历史": [m for m in earlier if m["model"] in candidates],
        "目标路径": target_summaries, "核心耗时": result["run_seconds"]}, ensure_ascii=False), flush=True)
    return result
