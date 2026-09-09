"""只修正已保存决策的空值比较，保留前三账户字节，不重算已完成账户。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "research/afternoon_entry_v1_output_fix.py"


def main():
    if TARGET.exists():
        raise ValueError("下午进入输出修正入口已存在，不覆盖")
    text = (ROOT / "research/afternoon_entry_v1.py").read_text(encoding="utf-8")
    text = text.replace('OUT = ROOT / "reports/research/510300_afternoon_entry_v1"', 'OUT = ROOT / "reports/research/510300_afternoon_entry_v1_output_fix"\nORIGINAL_OUT = ROOT / "reports/research/510300_afternoon_entry_v1"')
    text = text.replace('CONFIG = ROOT / "config/510300_afternoon_entry_v1.json"', 'CONFIG = ROOT / "config/510300_afternoon_entry_v1_output_fix.json"')
    start, end = text.index("def freeze():"), text.index("\n\ndef run():")
    text = text[:start] + '''def compare_saved_decisions(saved_path, original_path):
    current = pd.read_parquet(saved_path)
    original = pd.read_parquet(original_path)
    pd.testing.assert_frame_equal(current[original.columns], original)
    return len(current)


def freeze():
    import shutil
    require(not CONFIG.exists() and not OUT.exists(), "下午进入输出修正已登记")
    original_config = ROOT / "config/510300_afternoon_entry_v1.json"
    cfg = json.loads(original_config.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "原下午进入冻结内容改变")
    tests_path = ROOT / "reports/research/510300_afternoon_entry_saved_comparison_test.json"
    require(json.loads(tests_path.read_text(encoding="utf-8"))["exit_code"] == 0, "保存空值比较测试未通过")
    cfg.update(source_version="OUTPUT_FIX_PRESERVE_THREE_SAVED_ACCOUNTS", correction_registered_at=now(),
        original_config=str(original_config.relative_to(ROOT)), original_config_sha256=digest(original_config),
        correction="内存NaN与保存None比较失败；经济账户已一致。改为比较两份已保存Parquet，保留缺失，不填零、不改策略。",
        prior_actual_accounts_preserved=3, additional_accounts_to_run=1, evaluated_candidate_source_runs=2)
    original_files = [p for p in ORIGINAL_OUT.rglob("*") if p.is_file() and p.name != "RUN_STARTED.json"]
    cfg["preserved_files"] = [{"path": str(p.relative_to(ORIGINAL_OUT)), "sha256": digest(p)} for p in original_files]
    for path in [Path(__file__), original_config, tests_path, ROOT / "tests/test_afternoon_entry_saved_comparison.py"] + original_files:
        cfg["frozen_files"].append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
    OUT.mkdir(parents=True)
    for p in original_files:
        dest = OUT / p.relative_to(ORIGINAL_OUT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    write_json(CONFIG, cfg, exclusive=True)
    print("只修正保存格式比较，前三账户原字节保留，剩一个较早压力账户待算。", flush=True)
''' + text[end:]
    text = text.replace('    snapshots.to_parquet(OUT / "minute_snapshots.parquet")', '    if not (OUT / "minute_snapshots.parquet").exists():\n        snapshots.to_parquet(OUT / "minute_snapshots.parquet")')
    text = text.replace('    pd.DataFrame(partial_rows).to_csv(OUT / "all_partial_entry_factors.csv", index=False, encoding="utf-8-sig")', '    if not (OUT / "all_partial_entry_factors.csv").exists():\n        pd.DataFrame(partial_rows).to_csv(OUT / "all_partial_entry_factors.csv", index=False, encoding="utf-8-sig")')
    daily = '        pd.DataFrame({"date": frame.date, "d60_factor": factors, "entry_condition": rule["entry"], "original_price_exit": rule["exit"][1]}).to_csv(OUT / f"{period}_daily_factors.csv", index=False, encoding="utf-8-sig")'
    text = text.replace(daily, '        if not (OUT / f"{period}_daily_factors.csv").exists():\n    ' + daily)
    start = text.index("            ledger, decisions, cycles, afternoon = simulate_afternoon_entry(")
    end = text.index("            require(ledger.accounting_error", start)
    text = text[:start] + '''            saved_path = folder / f"{PRIMARY}_ledger.parquet"
            if saved_path.exists():
                ledger = pd.read_parquet(saved_path)
                decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
                cycles = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
                afternoon = pd.read_parquet(folder / "afternoon_decisions.parquet")
            else:
                require(period == "earlier_diagnostic" and cost_id == "STRESS", "修正轮只能补较早压力账户")
                ledger, decisions, cycles, afternoon = simulate_afternoon_entry(frame, div, cfg, cost, start, rule, cfg["specification"],
                    ExitController(frame, models, cfg["confirmation_days"]), EntryPreview(frame), snapshot_dict)
                save_account(folder, PRIMARY, ledger, decisions)
                cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
                afternoon.to_parquet(folder / "afternoon_decisions.parquet", index=False)
                afternoon.to_csv(folder / "afternoon_decisions.csv", index=False, encoding="utf-8-sig")
''' + text[end:]
    old_compare = '                original_decisions = pd.read_parquet(PARENT / period / cost_id / "REARM_RIDGE_decisions.parquet")\n                pd.testing.assert_frame_equal(decisions[original_decisions.columns], original_decisions)'
    text = text.replace(old_compare, '                compare_saved_decisions(folder / f"{PRIMARY}_decisions.parquet", PARENT / period / cost_id / "REARM_RIDGE_decisions.parquet")')
    text = text.replace('                if model != PRIMARY:\n                    saved.to_parquet', '                if model != PRIMARY and not (folder / f"{model}_ledger.parquet").exists():\n                    saved.to_parquet')
    text = text.replace('"goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)',
        '"goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0,\n        "preserved_prior_actual_accounts": 3, "new_accounts_in_correction": 1, "evaluated_candidate_source_runs": 2}, exclusive=True)')
    text = text.replace('    print(json.dumps({"主结果": primary, "下午行为": stats}, ensure_ascii=False), flush=True)',
        '    for item in cfg["preserved_files"]:\n        require(digest(OUT / item["path"]) == item["sha256"], "原保存文件在修正过程中被重写")\n    print(json.dumps({"主结果": primary, "下午行为": stats}, ensure_ascii=False), flush=True)')
    TARGET.write_text(text, encoding="utf-8")
    print("输出比较修正入口已生成，原入口与三个账户保留。")


if __name__ == "__main__":
    main()
