"""复用双方法完整账户运行段，加入单次逐日滤波及概率保存。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT / "research/joint_entry_exit_v1.py").read_text(encoding="utf-8")
text = text.replace('"""固定两个联合动作设置，复用历史时钟并运行八个完整账户。"""', '"""逐日行情持续时间预测及全历史对照，固定运行八个完整账户。"""')
text = text.replace('510300_joint_entry_exit_v1', '510300_bayesian_run_length_v1')
text = text.replace('from research.joint_state_action_inputs_v1 import market_state_frame\n', '')
text = text.replace('from research.joint_entry_exit_inputs_v1 import PRIMARY, CONTROL, make_cases, fit_month, JointController',
                    'from research.bayesian_run_length_inputs_v1 import PRIMARY, CONTROL, sequential_forecasts, MeanConfirmationController')
text = text.replace('NAMES = {PRIMARY: "联合进入持有退出", CONTROL: "只看单日奖励进出"}',
                    'NAMES = {PRIMARY: "逐日行情持续时间预测", CONTROL: "始终累积全部历史的预测"}')
left, right = text.index('def freeze():'), text.index('def run():')
freeze = '''def freeze():
    require(not CONFIG.exists(), "逐日阶段方案已登记，不重复冻结")
    previous = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: previous[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 8, "逐日阶段八项必要测试未通过")
    cfg.update(study_id="510300_BAYESIAN_RUN_LENGTH_V1", round=123, registered_at=now(), primary=PRIMARY, candidate_configurations=2,
        methods=NAMES, hazard=1/242, prior={"mu": 0., "kappa": 1., "alpha": 2., "beta": .0001}, warmup_observations=252,
        observation="total_log", full_run_length_posterior=True, confirmation_days=2, fixed_control_hazard=0.,
        source_gap="NO_VIEW_SOURCE_GAP_REMAINDER_NO_RESET_OR_IMPUTATION", numeric_failure="NO_VIEW_NUMERICAL_REMAINDER_NO_SOLVER_RESCUE",
        missing_decision="NO_VIEW_KEEP_ACTUAL_SHARES_WITH_LOCKED_EXIT_CONTINUATION", zero_or_mixed="KEEP_CURRENT_ACTUAL_ASSET",
        terminal="LAST_DAY_OPEN_LIQUIDATION_PRIORITY", old_cooldown_or_rearm=False,
        rules="docs/510300_BAYESIAN_RUN_LENGTH_V1.md", input_receipt="reports/research/510300_bayesian_run_length_preflight_20260909/result.json",
        new_reference_accounts=0, source_budget_cny=0, position_impact=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED",
        previous_goal_turn_classification="PROGRESS_ROUND122_COMPLETED_FOUR_ACCOUNTS_AND_123_METHOD_REVIEW")
    paths = [Path(__file__), ROOT / "research/bayesian_run_length_inputs_v1.py", ROOT / "research/joint_entry_exit_account_v1.py",
        ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / "tests/test_bayesian_run_length_v1.py",
        OUT / "tests_receipt.json", ROOT / "config/510300_research_authority_v6.json", ROOT / "config/510300_rearmed_session_exit_v1.json"]
    paths.extend(ROOT / cfg[k] for k in ["rules", "features", "dividends", "input_receipt"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{name}_ledger.parquet" for name, (parent, label) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第123轮逐日阶段预测与全历史对照已冻结，尚未计算新的真实行情预测或账户收益。", flush=True)


'''
text = text[:left]+freeze+text[right:]
left, right = text.index('    states = market_state_frame(data)'), text.index('    main, earlier, yearly, eras, coverage')
preparation = '''    forecasts, final_states, forecast_coverage = {}, {}, []
    for method in NAMES:
        predicted, archive, final = sequential_forecasts(data, cfg, method)
        forecasts[method], final_states[method] = predicted, final
        predicted.to_parquet(OUT / f"{method}_forecasts.parquet", index=False)
        np.savez_compressed(OUT / f"{method}_posterior.npz", **archive)
        forecast_coverage.append({"method": method, "filter_status": final["status"], "updates": final["observations"],
            "available_forecasts": int(predicted.model_status.eq("PREDICTION_AVAILABLE").sum()),
            "unknown_forecasts": int(predicted.prediction.isna().sum()), "saved_log_probability_cells": int(len(archive["log_probabilities"]))})
        print(f"{NAMES[method]}：{final['observations']}次逐日更新，状态{final['status']}。", flush=True)
    write_json(OUT / "filter_final_states.json", final_states, exclusive=True)
    pd.DataFrame(forecast_coverage).to_csv(OUT / "forecast_coverage.csv", index=False, encoding="utf-8-sig")
'''
text = text[:left]+preparation+text[right:]
text = text.replace('        period_states = states.iloc[:len(frame)].copy()\n', '')
text = text.replace('controller = JointController(period_states, models, method)', 'controller = MeanConfirmationController(forecasts[method].iloc[:len(frame)])')
text = text.replace('"status": "JOINT_ENTRY_EXIT_ACCOUNTS_COMPLETE"', '"status": "BAYESIAN_RUN_LENGTH_ACCOUNTS_COMPLETE"')
text = text.replace('"new_model_fits": len(models)', '"new_model_fits": len(final_states)')
left, right = text.index('        "nominal_one_step_cases":'), text.index('        "all_metrics":')
text = text[:left]+'''        "sequential_updates": sum(m["observations"] for m in final_states.values()), "forecast_coverage": forecast_coverage,
        "monthly_model_fits": 0, "full_length_online_filters": len(final_states),
'''+text[right:]
text = text.replace('联合策略冻结来源改变', '逐日阶段冻结来源改变').replace('联合账户终点或财富核算不符', '阶段预测账户终点或财富核算不符')
path = ROOT / 'research/bayesian_run_length_v1.py'
with path.open('x', encoding='utf-8') as stream:
    stream.write(text)
print('第123轮独立运行器已保存，账户经济循环保持复用。')
