"""第192轮固定二十项周期内模型和四条完整模拟账户。"""
import json
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.holding_market_coupling_exit_inputs_v1 import build_monthly_models, CouplingExitController, PAIRS
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules
from research.joint_account_acceptance_v1 import save_joint_assessment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_holding_market_coupling_exit_v1'
CONFIG = ROOT/'config/510300_holding_market_coupling_exit_v1.json'
PRIMARY = 'HOLDING_MARKET_COUPLING_EXIT'
CONTROLS = {
    'ENTRY_VINTAGE_EXIT': (ROOT/'reports/research/510300_entry_vintage_exit_v1', '第128轮原固定版本退出'),
    'ACCOUNT_VOLATILITY_EXPOSURE': (ROOT/'reports/research/510300_account_volatility_exposure_v1', '第181轮原账户风险预算'),
    'BUY_HOLD': (ROOT/'reports/research/510300_rearmed_session_exit_v1', '买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(), '本轮已冻结或计算')
    OUT.mkdir(parents=True, exist_ok=True)
    require(not (OUT/'tests_receipt.json').exists(), '本轮已有必要测试回执')
    started = time.perf_counter()
    tested = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_holding_market_coupling_exit_v1.py', '-q', '-p', 'no:cacheprovider'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT/'tests_output.txt').write_text(tested.stdout+tested.stderr, encoding='utf-8')
    print(tested.stdout, flush=True)
    require(tested.returncode==0 and '6 passed' in tested.stdout, '本轮六项必要测试未通过')
    write_json(OUT/'tests_receipt.json', {'recorded_at': now(), 'exit_code': tested.returncode, 'passed': 6,
        'seconds': time.perf_counter()-started, 'timing_scope': '完整测试进程墙钟'}, exclusive=True)
    parent_path = ROOT/'config/510300_within_cycle_exit_v1.json'
    old = json.loads(parent_path.read_text(encoding='utf-8'))
    keys = ['evaluation_start', 'data_cutoff', 'initial_capital', 'lot', 'tick', 'limit_fraction', 'annual_days',
        'cash_annual_rate_assumption', 'high_sharpe_target', 'costs', 'features', 'dividends', 'earlier_start', 'earlier_terminal',
        'confirmation_days', 'specification', 'feature_columns', 'feature_names', 'feature_clip', 'ridge_alpha',
        'recent_cycles', 'minimum_cycles', 'minimum_rows']
    cfg = {k:old[k] for k in keys}
    cfg.update(study_id='510300_HOLDING_MARKET_COUPLING_EXIT_V1', round=192, registered_at=now(), primary=PRIMARY,
        candidate_models=[PRIMARY], candidate_configurations=1, annual_return_target=.10,
        source_models='reports/research/510300_within_cycle_exit_v1/saved_models.json',
        samples='reports/research/510300_within_cycle_exit_v1/extended_reference_samples.parquet',
        interaction_pairs=[list(p) for p in PAIRS], total_regressors=20, planned_distinct_fits=25,
        planned_monthly_records=141, planned_eligible_months=114, planned_reused_monthly_fits=89,
        model_selection_clock='FIRST_CLOSE_OF_ACTUAL_FILLED_ENTRY', model_reselection='ONLY_ON_NEW_ACTUAL_CYCLE',
        rules='docs/510300_HOLDING_MARKET_COUPLING_EXIT_V1.md', new_reference_accounts=0, source_budget_cny=0,
        goal_achieved=False, position_impact=0, independent_validation='NOT_ESTABLISHED',
        previous_goal_turn_classification='PROGRESS_ROUND191_VERIFIED_DELIVERED_AND_INDEX_UPDATED')
    with (ROOT/cfg['rules']).open('x', encoding='utf-8') as stream:
        rules=(ROOT/'docs/510300_HOLDING_MARKET_COUPLING_EXIT_NEXT_20260913.md').read_text(encoding='utf-8')
        stream.write(rules.replace('本方案尚未冻结、拟合或计算账户。', '本方案在必要测试通过后、首次新模型拟合和账户计算前冻结。'))
    paths=[Path(__file__), parent_path, ROOT/'research/holding_market_coupling_exit_inputs_v1.py',
        ROOT/'research/learned_cycle_exit_v1.py', ROOT/'research/rearmed_cycle_exit_account_v1.py',
        ROOT/'research/simple_intraday_protection_v1.py', ROOT/'research/simple_session_divergence_v1.py',
        ROOT/'research/simple_price_entry_exit_v1.py', ROOT/'research/adaptive_allocation_v1.py',
        ROOT/'research/intraday_overnight_increment_v1.py', ROOT/'research/joint_account_acceptance_v1.py',
        ROOT/'tests/test_holding_market_coupling_exit_v1.py', ROOT/'tests/test_median_continuation_v1.py',
        ROOT/'tests/test_single_component_exit_v1.py', OUT/'tests_receipt.json', OUT/'tests_output.txt',
        ROOT/'config/510300_research_authority_v6.json']
    paths += [ROOT/cfg[k] for k in ['rules', 'features', 'dividends', 'samples', 'source_models']]
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost in cfg['costs']:
            paths += [folder/period/cost/f'{model}_ledger.parquet' for model,(folder,_) in CONTROLS.items()]
    cfg['frozen_files']=[{'path':str(p.relative_to(ROOT)), 'sha256':digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG,cfg,exclusive=True)
    path=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index=json.loads(path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round']==191, '本轮前序完成状态不同')
    index['running_studies']=[{'round':192,'study':cfg['study_id'],'status':'FROZEN_NOT_STARTED','config':str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True,status='HOLDING_MARKET_COUPLING_EXIT_FROZEN',source=cfg['rules'])
    write_json(path,index)
    print('第192轮固定方案已冻结，尚未计算新历史模型或账户。',flush=True)


def run():
    started = time.perf_counter()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "联合作用退出冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    samples = pd.read_parquet(ROOT / cfg["samples"])
    originals = json.loads((ROOT / cfg["source_models"]).read_text(encoding="utf-8"))["models"]
    require(all(pd.Timestamp(r["fit_time"]) == data.date.iloc[r["fit_index"]] + pd.Timedelta(hours=15, minutes=5) for r in originals), "月度训练时钟不对应完整交易日历")
    models, monthly, memberships, counts = build_monthly_models(samples, originals, cfg)
    require(counts["monthly_records"] == 141 and counts["eligible_monthly_records"] == 114 and counts["distinct_source_input_sets"] == 25, "原训练支持或缓存身份数量改变")
    require(counts["new_model_fits"] <= 25, "相同输入重复拟合")
    write_json(OUT / "saved_models.json", {"recorded_at": now(), "counts": counts, "models": models}, exclusive=True)
    monthly.to_csv(OUT / "monthly_fit_diagnostics.csv", index=False, encoding="utf-8-sig")
    memberships.to_parquet(OUT / "training_memberships.parquet", index=False)
    print(f"141个月度记录完成：{counts['new_model_fits']}次新拟合，{counts['reused_monthly_fits']}个月复用；失败{counts['failed_fits']}次。", flush=True)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        rule = make_rules(frame)["D60_INTRA"]
        pd.DataFrame({"date": frame.date, "entry_condition": rule["entry"], "original_price_exit": rule["exit"][1]}).to_parquet(OUT / f"{period}_entry_exit_conditions.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_rearmed_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"], CouplingExitController(frame, models, cfg["confirmation_days"]))
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "联合作用退出账户未完整结算")
            require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "联合作用退出违反次日可卖")
            holding = decisions[decisions.learning_cycle_id.notna()]
            coverage.append({"period": period, "cost": cost_id, "holding_decisions": len(holding),
                "model_available_rows": int(holding.learning_status.eq("PREDICTION_AVAILABLE").sum()),
                "no_model_rows": int(holding.learning_status.ne("PREDICTION_AVAILABLE").sum()),
                "learned_exit_cycles": int(cycles.exit_reasons.str.contains("学习条件", regex=False).sum()),
                "completed_round_trips": len(cycles), "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "市场状态调节持仓退出"}
            for model, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "联合作用退出与对照完整日历不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                dest.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            pd.DataFrame({"date": ledger.date, **{k: a.net_return.to_numpy() for k, a in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：一个新联合作用退出账户和三个保存对照已完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("model_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "HOLDING_MARKET_COUPLING_EXIT_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 6,
        "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 6,
        "new_reference_accounts": 0, "run_seconds": time.perf_counter() - started, **counts,
        "all_metrics": main, "earlier_diagnostics": earlier, "model_coverage": coverage, "primary": primary,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False,
        "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    save_joint_assessment(OUT, cfg, result)
    print(json.dumps({"主评价": primary, "较早": [m for m in earlier if m["model"] == PRIMARY], "核心秒数": result["run_seconds"], "拟合统计": counts}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    {'prepare':prepare, 'run':run}[sys.argv[1]]()
