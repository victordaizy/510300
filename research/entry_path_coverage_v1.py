"""扩大原退出模型的参考入场路径，以整组成熟时间和组内等权约束历史学习。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.entry_path_reference_v1 import PathInputs, single_entry_path, signal_episodes, mature_episodes, episode_training_rows
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import FEATURES, CN, ExitController, continuation_label, fit_one, chinese_formula
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_session_divergence_v1 import make_rule, specifications

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_entry_path_coverage_v1"
CONFIG = ROOT / "config/510300_entry_path_coverage_v1.json"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
PRIMARY = "ALL_ENTRY_PATHS_RIDGE"
NAME = "扩大进入路径的线性退出及等待新机会"
CONTROLS = {"REARM_RIDGE": (P32, "原学习退出及等待新机会"), "REARM_NONE": (P32, "原自然退出及等待新机会"),
            "PANIC_LEARNED_HALF": (P46, "原急跌回升及学习退出各半"), "BUY_HOLD": (P32, "买入持有")}


def raw_rule(data):
    item = next(x for x in specifications() if x["id"] == "D60_INTRA")
    return make_rule(data, item)


def freeze():
    old = json.loads((ROOT / "config/510300_learned_cycle_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
                              "reference_start", "minimum_rows", "feature_clip", "ridge_alpha", "confirmation_days"]}
    cfg.update(study_id="510300_ENTRY_PATH_COVERAGE_V1", round=59, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
               specification=old["candidate_specs"]["D60_INTRA"], feature_columns=FEATURES, feature_names=CN,
               recent_episodes=20, minimum_episodes=10, reference_cost="BASE", model_refit="MONTH_FIRST_CLOSE",
               reference_origins="ALL_VALID_RAW_ENTRIES_ON_OR_AFTER_REFERENCE_START_WITH_NEXT_OPEN",
               path_first_buy_retry=False, group_weight_rule="EACH_GROUP_ONE_EACH_PATH_EQUAL_EACH_STATE_WITHIN_PATH_EQUAL",
               rules="docs/510300_ENTRY_PATH_COVERAGE_V1.md", python_library_version=sklearn.__version__, position_impact=0,
               goal_achieved=False, independent_validation="NOT_ESTABLISHED")
    paths = [Path(__file__), ROOT / "research/entry_path_reference_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
             ROOT / "research/learned_cycle_exit_account_v1.py", ROOT / "research/rearmed_cycle_exit_account_v1.py",
             ROOT / "research/simple_session_divergence_v1.py", ROOT / "research/simple_price_entry_exit_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
             ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / "tests/test_entry_path_reference_v1.py",
             ROOT / "tests/test_entry_path_coverage_v1.py"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(p / period / cost / f"{key}_ledger.parquet" for key, (p, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第59轮一个扩大参考进入路径的退出设置已登记，尚未生成本轮路径收益或训练模型。", flush=True)


def build_samples(data, dividends, cfg):
    rule, factor = raw_rule(data)
    available = data.feature_valid.fillna(False).to_numpy(bool) & np.isfinite(factor) & np.isfinite(factor.shift(1))
    episodes = signal_episodes(rule["entry"], available)
    episode_map = {t: event["episode_id"] for event in episodes for t in range(event["start_index"], event["end_index"] + 1)}
    first = int(np.flatnonzero(data.date >= cfg["reference_start"])[0])
    origins = [int(t) for t in range(first, len(data) - 1) if rule["entry"][t] and available[t]]
    require(len(origins) > 0, "没有可登记的参考进入机会")
    inputs = PathInputs(data, dividends)
    metadata, ledgers, all_states, labels = [], [], [], []
    mode_spec = cfg["specification"]["modes"]["1"]
    for count, origin in enumerate(origins, 1):
        meta, ledger, states = single_entry_path(inputs, origin, cfg, cfg["costs"][cfg["reference_cost"]], rule["exit"][1], mode_spec)
        group = episode_map[origin]
        meta["episode_id"] = group
        ledger["episode_id"] = group
        states["episode_id"] = group
        metadata.append(meta)
        ledgers.append(ledger)
        if len(states):
            all_states.append(states)
        if meta["natural_exit"]:
            end = int(meta["exit_index"])
            for row in states[states.natural_continue].to_dict("records"):
                t = int(row["origin_index"])
                if t + 1 >= end or not np.isfinite([row[k] for k in FEATURES]).all():
                    continue
                target, extra = continuation_label(data, dividends, meta["entry_quantity"], t + 1, end, cfg["costs"][cfg["reference_cost"]], cfg["tick"])
                labels.append({**row, "early_exit_index": t + 1, "early_exit_date": data.date.iloc[t + 1], "exit_index": end,
                               "exit_date": data.date.iloc[end], "target": target, "extra_dividend_cny": extra,
                               "reference_quantity": meta["entry_quantity"]})
        if count % 200 == 0 or count == len(origins):
            print(f"单次假想进入路径已处理 {count}/{len(origins)}；这些重叠路径不代表独立市场样本。", flush=True)
    paths = pd.DataFrame(metadata)
    groups = mature_episodes(episodes, paths, first)
    for field in ["start_index", "end_index", "closed_index", "group_mature_index"]:
        groups[field.replace("index", "date")] = [data.date.iloc[int(t)] if pd.notna(t) else pd.NaT for t in groups[field]]
    samples = pd.DataFrame(labels)
    require(len(samples) > 0, "参考路径未形成任何自然退出标签")
    samples = samples.merge(groups[["episode_id", "group_mature_index", "group_mature_date"]], on="episode_id", how="left", validate="many_to_one")
    mature_sample_groups = set(samples.loc[samples.group_mature_index.notna(), "episode_id"])
    groups["has_usable_labels"] = groups.episode_id.isin(mature_sample_groups)
    paths.to_csv(OUT / "reference_paths.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(OUT / "reference_episodes.csv", index=False, encoding="utf-8-sig")
    pd.concat(ledgers, ignore_index=True).to_parquet(OUT / "reference_path_ledgers.parquet", index=False)
    pd.concat(all_states, ignore_index=True).to_parquet(OUT / "reference_path_states.parquet", index=False)
    samples.to_parquet(OUT / "all_reference_samples.parquet", index=False)
    pd.DataFrame({"date": data.date, "d60_factor": factor, "raw_entry": rule["entry"], "raw_exit": rule["exit"][1], "signal_available": available}).to_parquet(OUT / "entry_factors.parquet", index=False)
    info = {"attempted_hypothetical_paths": len(paths), "filled_hypothetical_paths": int(paths.entry_index.notna().sum()),
            "naturally_completed_paths": int(paths.natural_exit.sum()), "unresolved_or_terminal_censored_paths": int(paths.resolution_index.isna().sum()),
            "first_buy_unfilled_paths": int(paths.status.eq("首次买入未成交，不重试").sum()), "reference_episodes": len(groups),
            "mature_label_bearing_episodes": len(mature_sample_groups), "reference_state_rows": sum(len(x) for x in all_states),
            "natural_label_rows": len(samples), "group_mature_label_rows": int(samples.group_mature_index.notna().sum()),
            "label_bearing_paths": int(samples.path_id.nunique()), "reference_path_daily_rows": sum(len(x) for x in ledgers),
            "reference_left_truncated_groups": int(groups.reference_left_truncated.sum()),
            "max_reference_accounting_error": float(paths.max_accounting_error.max()),
            "independent_market_sample_count": "NOT_ESTABLISHED_OVERLAPPING_PATHS", "reference_paths_are_not_one_tradable_portfolio": True,
            "reference_nav_stops_at_actual_exit_and_may_contain_unpaid_dividend_receivables": True,
            "label_same_as_round31_and_assumes_hypothetical_early_full_sale": True}
    write_json(OUT / "reference_summary.json", info)
    return samples, info


def fit_record(samples, t, data, cfg):
    rows, groups = episode_training_rows(samples, t, cfg["recent_episodes"])
    usable = len(groups) >= cfg["minimum_episodes"] and len(rows) >= cfg["minimum_rows"]
    require(rows.empty or ((rows.origin_index < rows.exit_index) & (rows.exit_index <= rows.group_mature_index) & (rows.group_mature_index <= t)).all(), "训练包含尚未完全成熟的参考组")
    record = {"signal": "D60_INTRA", "kind": "RIDGE", "fit_index": t, "fit_origin": str(data.date.iloc[t].date()),
              "fit_time": data.date.iloc[t] + pd.Timedelta(hours=15, minutes=5),
              "status": "FIT_COMPLETE" if usable else "NO_VIEW_MINIMUM_MATURE_EPISODES_OR_ROWS",
              "training_episodes": groups, "training_episode_count": len(groups), "training_path_count": int(rows.path_id.nunique()),
              "training_rows": len(rows), "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None,
              "latest_exit_date": str(data.date.iloc[int(rows.exit_index.max())].date()) if len(rows) else None,
              "latest_group_mature_index": int(rows.group_mature_index.max()) if len(rows) else None,
              "latest_group_mature_date": str(data.date.iloc[int(rows.group_mature_index.max())].date()) if len(rows) else None,
              "model": fit_one(rows, "RIDGE", cfg) if usable else None}
    return record, rows


class GroupExitController(ExitController):
    def __init__(self, data, models, confirmation_days=2):
        for record in models:
            if record["status"] == "FIT_COMPLETE":
                require(record["latest_exit_index"] <= record["latest_group_mature_index"] <= record["fit_index"], "扩大路径模型使用了未来整组结果")
        super().__init__(data, models, confirmation_days)


def train_models(data, samples, cfg):
    first = int(np.flatnonzero(data.date >= cfg["earlier_start"])[0]) - 1
    month_first = data.date.dt.to_period("M") != data.date.shift(1).dt.to_period("M")
    schedule = sorted(set([first] + [int(t) for t in np.flatnonzero(month_first) if first <= t < len(data) - 1]))
    models, receipts, memberships = [], [], []
    chinese = ["# 第59轮每月模型的完整中文规则", "", "每月只使用当时整组自然成熟的进入路径；每组总权重相同。八个因子按每次训练保存的均值、标准差和系数计算。", ""]
    for number, t in enumerate(schedule, 1):
        record, rows = fit_record(samples, t, data, cfg)
        models.append(record)
        receipts.append({k: v for k, v in record.items() if k not in ["model", "training_episodes"]})
        chinese += [f"## {record['fit_origin']}", "", f"纳入{record['training_episode_count']}个成熟组、{record['training_path_count']}条假想进入路径、{record['training_rows']}条状态。", ""]
        if record["status"] == "FIT_COMPLETE":
            membership = rows[["episode_id", "path_id", "origin_index", "exit_index", "group_mature_index", "sample_weight"]].copy()
            membership["fit_index"] = t
            memberships.append(membership)
            chinese += [f"最新整组成熟日为{record['latest_group_mature_date']}。", ""] + chinese_formula(record["model"]) + [""]
        else:
            chinese += ["可用成熟组或状态不足，未训练模型；预测保持缺失，账户继续执行原自然退出规则。", ""]
        if number % 30 == 0 or number == len(schedule):
            print(f"月度学习记录已处理 {number}/{len(schedule)}。", flush=True)
    write_json(OUT / "saved_models.json", {"models": {PRIMARY: models}, "feature_names": dict(zip(FEATURES, CN))})
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    (pd.concat(memberships, ignore_index=True) if memberships else pd.DataFrame(columns=["fit_index", "episode_id", "path_id", "origin_index", "exit_index", "group_mature_index", "sample_weight"])).to_parquet(OUT / "training_memberships.parquet", index=False)
    (OUT / "每月模型中文规则.md").write_text("\n".join(chinese) + "\n", encoding="utf-8")
    return models, receipts


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "扩大进入路径登记文件发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    samples, reference = build_samples(data, dividends, cfg)
    models, receipts = train_models(data, samples, cfg)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        rule, _ = raw_rule(frame)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            controller = GroupExitController(frame, models, cfg["confirmation_days"])
            ledger, decisions, cycles = simulate_rearmed_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"], controller)
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            finished = cycles.dropna(subset=["exit_date"])
            require((finished.holding_intervals >= 1).all() and ledger.accounting_error.abs().max() < 1e-6, "扩大路径账户或次日可卖约束不成立")
            require(not ledger.terminal_unliquidated.iloc[-1], "扩大路径账户终点未完成清算")
            holding = decisions[decisions.learning_cycle_id.notna()]
            coverage.append({"period": period, "cost": cost_id, "model": PRIMARY, "holding_decision_rows": len(holding),
                             "model_available_rows": int(holding.learning_status.eq("PREDICTION_AVAILABLE").sum()),
                             "no_model_rows": int(holding.learning_status.ne("PREDICTION_AVAILABLE").sum()),
                             "learned_exit_cycles": int(finished.exit_reasons.str.contains("学习条件", regex=False).sum()),
                             "completed_round_trips": len(finished), "positive_cycles": int(finished.net_profit_cny.gt(0).sum()),
                             "mean_holding_intervals": float(finished.holding_intervals.mean()) if len(finished) else None})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for key, (source, name) in CONTROLS.items():
                saved = pd.read_parquet(source / period / cost_id / f"{key}_ledger.parquet")
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "扩大路径账户未使用相同完整交易日历")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[saved.date.between(left, right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            pd.DataFrame({"date": ledger.date, **{k: a.net_return.to_numpy() for k, a in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：扩大进入路径的退出账户与四个原样对照已完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("model_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "ENTRY_PATH_COVERAGE_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8,
              "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8,
              "new_model_fits": sum(r["status"] == "FIT_COMPLETE" for r in receipts), "no_view_fit_origins": sum(r["status"] != "FIT_COMPLETE" for r in receipts),
              "new_reference_accounts": reference["attempted_hypothetical_paths"], "new_reference_accounts_definition": "每个合格原点一次假想进入尝试，不是独立市场样本或一条可执行组合",
              "reference_summary": reference, "all_metrics": main, "earlier_diagnostics": earlier, "model_coverage": coverage,
              "primary": primary, "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in primary),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"完成模型拟合": result["new_model_fits"], "没有足够成熟样本": result["no_view_fit_origins"], "主评价": primary,
                      "较早评价": [m for m in earlier if m["model"] == PRIMARY]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
