"""复用第117轮成熟时钟和完整账户运行段，单独保存新近邻方法。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
text = (ROOT / "research/market_path_exit_v1.py").read_text(encoding="utf-8")
text = text.replace('market_path_exit_v1', 'cycle_analogue_exit_v1').replace('MARKET_PATH_EXIT', 'CYCLE_ANALOGUE_EXIT')
text = text.replace('from research.market_path_exit_inputs_v1 import FEATURES, CN, fit_market_path_exit, MarketPathExitController',
                    'from research.cycle_analogue_exit_inputs_v1 import FEATURES, CN, fit_cycle_analogue, CycleAnalogueExitController')
text = text.replace('MarketPathExitController', 'CycleAnalogueExitController')
text = text.replace('不含已付买入费用的八项周期内', '不同周期相似状态')
left, right = text.index('def freeze():'), text.index('def run():')
new = '''def freeze():
    require(not CONFIG.exists(), "本轮已登记，不重复冻结")
    old = json.loads((ROOT / "config/510300_market_path_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
        "confirmation_days", "specification", "saved_models", "recent_cycles", "minimum_cycles", "minimum_rows", "feature_clip",
        "saved_market_samples", "saved_market_states", "state_preflight_receipt"]}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 7, "不同周期近邻七项必要测试未通过")
    cfg.update(study_id="510300_CYCLE_ANALOGUE_EXIT_V1", round=122, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
        distinct_cycle_neighbors=5, distance="SQUARED_EUCLIDEAN_AFTER_EQUAL_CYCLE_WEIGHT_STANDARDIZATION",
        tie_order="SQUARED_DISTANCE_THEN_CYCLE_ID_THEN_ORIGIN_INDEX", cycle_limit=1, neighbor_weight="EQUAL",
        feature_columns=FEATURES.copy(), feature_names=CN.copy(), planned_model_fits=114,
        missing_training_features="WHOLE_MONTH_NO_VIEW_PRESERVE_ALL_ORIGINAL_ROWS",
        model_loss="NO_COEFFICIENT_OPTIMIZATION_STORE_STANDARDIZED_MATURE_INSTANCES",
        fit_failure="NO_VIEW_MODEL_FIT_FAILED_KEEP_ORIGINAL_EXITS_NO_RESCUE", rules="docs/510300_CYCLE_ANALOGUE_EXIT_V1.md",
        analogue_support_receipt="reports/research/510300_cycle_analogue_support_20260909/result.json",
        new_reference_accounts=0, source_budget_cny=0, position_impact=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED",
        previous_goal_turn_classification="PROGRESS_ROUND121_COMPLETED_EIGHT_ACCOUNTS_AND_122_INPUT_SUPPORT")
    paths = [Path(__file__), ROOT / "research/cycle_analogue_exit_inputs_v1.py", ROOT / "research/market_path_exit_inputs_v1.py",
        ROOT / "research/market_path_state_v1.py", ROOT / "research/rearmed_cycle_exit_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
        ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py", ROOT / "research/simple_price_entry_exit_v1.py",
        ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py",
        ROOT / "tests/test_cycle_analogue_exit_v1.py", ROOT / "tests/test_median_continuation_v1.py", OUT / "tests_receipt.json",
        ROOT / "config/510300_market_path_exit_v1.json", ROOT / "config/510300_research_authority_v6.json"]
    paths.extend(ROOT / cfg[k] for k in ["rules", "features", "dividends", "saved_models", "saved_market_samples", "saved_market_states",
                                       "state_preflight_receipt", "analogue_support_receipt"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{model}_ledger.parquet" for model, (parent, name) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第122轮一个不同周期相似状态退出设置已冻结，尚未建立新样本模型或读取新账户收益。", flush=True)


def train_models(data, samples, originals, cfg):
    models, receipts, memberships, standardization = [], [], [], []
    for original in originals:
        t = int(original["fit_index"])
        rows, ids = training_rows(samples, t, cfg)
        require(ids == original["training_cycles"] and len(rows) == original["training_rows"], "原模型与近邻方法成熟成员不同")
        eligible = len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
        require(eligible == (original["status"] == "FIT_COMPLETE"), "近邻方法改变了原支持月份")
        stored, failure = None, None
        status = "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS"
        missing_rows = int((~np.isfinite(rows[FEATURES].to_numpy(float)).all(axis=1)).sum())
        if eligible and missing_rows:
            status = "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"
        if eligible and not missing_rows:
            try:
                stored = fit_cycle_analogue(rows, cfg)
                status = "FIT_COMPLETE"
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                status, failure = "NO_VIEW_MODEL_FIT_FAILED", str(error)
        record = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()), "fit_time": data.date.iloc[t]+pd.Timedelta(hours=15, minutes=5),
            "status": status, "eligible_for_fit": eligible, "training_cycles": ids, "training_cycle_count": len(ids), "training_rows": len(rows),
            "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None, "latest_exit_date": str(rows.mature_date.max().date()) if len(rows) else None,
            "failure": failure, "missing_feature_rows": missing_rows, "model": stored}
        require(not len(rows) or (rows.exit_index <= t).all(), "相似参考周期在本次训练时尚未结束")
        models.append(record)
        receipts.append({k: v for k, v in record.items() if k not in ["model", "training_cycles"]})
        if eligible:
            memberships.extend({"fit_index": t, "cycle_id": int(r.cycle_id), "origin_index": int(r.origin_index),
                "exit_index": int(r.exit_index), "sample_weight": float(r.sample_weight), "fit_status": status} for r in rows.itertuples())
        if stored is not None:
            standardization.extend({"拟合收盘": record["fit_origin"], "因子": name, "训练均值": mean, "训练标准差": scale,
                                    "固定不同周期邻居数": stored["neighbors"]}
                for name, mean, scale in zip(CN, stored["mean"], stored["scale"], strict=True))
    require(sum(r["eligible_for_fit"] for r in receipts) == cfg["planned_model_fits"], "成熟样本模型次数不同")
    write_json(OUT / "saved_models.json", {"models": models, "feature_names": cfg["feature_names"], "model_rule": cfg["model_loss"]}, exclusive=True)
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memberships).to_parquet(OUT / "training_memberships.parquet", index=False)
    pd.DataFrame(standardization).to_csv(OUT / "每月样本标准化.csv", index=False, encoding="utf-8-sig")
    print(f"近邻样本模型{sum(r['status']=='FIT_COMPLETE' for r in receipts)}个，原不支持或失败{sum(r['status']!='FIT_COMPLETE' for r in receipts)}个月。", flush=True)
    return models, receipts


'''
text = text[:left]+new+text[right:]
path = ROOT / 'research/cycle_analogue_exit_v1.py'
with path.open('x', encoding='utf-8') as stream:
    stream.write(text)
print('第122轮新运行器已保存，成熟时钟与账户循环复用既有实现。')
