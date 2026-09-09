"""只用既有月度线性模型，在入场前检查假设持仓状态。"""
from bisect import bisect_right
import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import commission, digest, fill_price, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES, ExitController, predict, state_values
from research.consistent_entry_exit_account_v1 import simulate_consistent_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_consistent_entry_exit_v1"
CONFIG = ROOT / "config/510300_consistent_entry_exit_v1.json"
PARENT = ROOT / "reports/research/510300_rearmed_session_exit_v1"


class EntryGate:
    def __init__(self, data, models, cost, config):
        self.data, self.models, self.cost, self.config = data, models, cost, config
        self.fit_indexes = [m["fit_index"] for m in models]

    def __call__(self, t, account, mode, quantity):
        index = bisect_right(self.fit_indexes, t) - 1
        stored = self.models[index] if index >= 0 else None
        row = {"entry_allowed": True, "entry_model_status": "NO_VIEW_NO_MATURE_MODEL", "entry_prediction": None,
               "entry_hypothetical_quantity": quantity, "entry_model_fit_origin": self.data.date.iloc[stored["fit_index"]] if stored else pd.NaT}
        if quantity <= 0:
            row.update(entry_allowed=False, entry_model_status="NO_ENTRY_AFFORDABLE_QUANTITY_ZERO")
            return row
        px = fill_price(float(self.data.close.iloc[t]), 1, self.cost, self.config["tick"])
        total_cost = quantity * px + commission(quantity, px, self.cost)
        require(total_cost <= account.cash + 1e-6, "入场假设份额超出已知现金")
        current_value = quantity * float(self.data.close.iloc[t])
        fake_cycle = {"entry_index": t, "entry_cost_cny": total_cost, "mode": mode}
        values = state_values(self.data, t, fake_cycle, current_value, max(total_cost, current_value))
        row.update({"entry_hypothetical_cost": total_cost, **{"entry_factor_" + key: value for key, value in zip(FEATURES, values)}})
        if stored and stored["status"] == "FIT_COMPLETE" and np.isfinite(values).all():
            require(stored["latest_exit_index"] <= stored["fit_index"] <= t, "入场预测使用了未来模型")
            estimate = predict(stored["model"], values)
            row.update(entry_allowed=estimate >= 0, entry_prediction=estimate, entry_model_status="PREDICTION_AVAILABLE")
        elif stored and stored["status"] == "FIT_COMPLETE":
            row["entry_model_status"] = "NO_VIEW_INCOMPLETE_HYPOTHETICAL_STATE"
        return row


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: v for k, v in old.items() if k not in ["frozen_files", "names"]}
    cfg.update({"study_id": "510300_CONSISTENT_ENTRY_EXIT_V1", "round": 33, "registered_at": now(), "primary": "CONSISTENT_ENTRY_RIDGE",
                "candidate_configurations": 1, "rules": "docs/510300_CONSISTENT_ENTRY_EXIT_V1.md", "new_model_fits": 0})
    paths = [Path(__file__), ROOT / "research/consistent_entry_exit_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["saved_models"],
             ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py",
             ROOT / "tests/test_consistent_entry_exit_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第33轮仅一个入场退出一致设置已登记，复用旧模型。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "入场一致规则或模型发生变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    div = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    main, early, yearly, eras, checks = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        rule = make_rules(frame)["D60_INTRA"]
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_consistent_exit(frame, div, cfg, cost, start, rule, cfg["specification"],
                ExitController(frame, models, cfg["confirmation_days"]), EntryGate(frame, models, cost, cfg))
            save_account(folder, cfg["primary"], ledger, decisions)
            cycles.to_csv(folder / f"{cfg['primary']}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "入场检查账户结算失败")
            if len(cycles):
                require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "入场检查出现当天买卖")
            evaluated = decisions[decisions.entry_model_status.notna()] if "entry_model_status" in decisions else pd.DataFrame()
            checks.append({"period": period, "cost": cost_id, "entry_checks": len(evaluated),
                           "negative_entry_predictions": int((evaluated.entry_prediction < 0).sum()) if len(evaluated) else 0,
                           "no_model_entry_checks": int(evaluated.entry_model_status.str.startswith("NO_VIEW").sum()) if len(evaluated) else 0,
                           "filled_entries": int((ledger.filled_quantity > 0).sum())})
            accounts = {cfg["primary"]: ledger}
            for key in ["REARM_RIDGE", "BUY_HOLD"]:
                saved = pd.read_parquet(PARENT / period / cost_id / f"{key}_ledger.parquet")
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key] = saved
            base = summarize(accounts["BUY_HOLD"], cfg)
            names = {cfg["primary"]: "已有线性模型同时检查入场与退出", "REARM_RIDGE": "原线性退出＋等待新机会", "BUY_HOLD": "买入持有"}
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "入场检查日期不完整")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                dest.append(m)
                if period == "evaluation":
                    for y, group in saved.groupby(saved.date.dt.year):
                        yearly.append({"cost": cost_id, "model": key, "year": int(y), **summarize(group, cfg)})
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[(saved.date >= left) & (saved.date <= right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：一个入场检查账户和两个对照已完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("entry_check_statistics.csv", checks)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "CONSISTENT_ENTRY_EXIT_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": 6, "new_accounts_generated": 2, "reused_control_accounts": 4,
              "earlier_diagnostic_accounts": 6, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 4, "new_model_fits": 0,
              "all_metrics": main, "earlier_diagnostics": early, "entry_check_statistics": checks,
              "primary": [m for m in main if m["model"] == cfg["primary"]],
              "post_selected_best_base": next(m for m in main if m["model"] == cfg["primary"] and m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] == cfg["primary"]),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"状态": result["status"], "新设置结果": result["primary"], "入场检查": checks}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
