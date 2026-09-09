"""保留首次来源与失败证据，只更正每日实际持仓字段。"""
import json
from pathlib import Path
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "reports/research/510300_staged_entry_v1"
NEW = ROOT / "reports/research/510300_staged_entry_v1_1"


def main():
    require(not (ROOT / "research/staged_entry_v1_1.py").exists(), "修正来源已经建立")
    NEW.mkdir(parents=True, exist_ok=True)
    old_config = ROOT / "config/510300_staged_entry_v1.json"
    old = json.loads(old_config.read_text(encoding="utf-8"))
    for item in old["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "首次冻结来源改变")
    write_json(OLD / "source_defect_receipt.json", {
        "recorded_at": now(), "status": "INVALID_STAGING_SOURCE_DAILY_INVENTORY_FIELD",
        "discovered_by": "scripts/finalize_round81_20260908.py",
        "failed_check": "2020-02-06参考入场财富开盘应为1.6433187614191385，保存为缺失；随后较早2015-01-06也复现。",
        "cause": "shares_after仅实际执行时提供，无成交日缺失；每日实际持仓是shares。首次合成测试没有模拟此缺失。",
        "effect": "遗漏无成交日的参考持有及入场基准，漏掉可能发生在后续日的价格确认。首次分批策略不能评价原冻结规则。",
        "unchanged_controls": "同执行满仓的实际目标和成交仅取原参考状态，未使用分批持仓标记，其账本可以原字节复用。",
        "original_config_sha256": digest(old_config), "original_result_sha256": digest(OLD / "result.json"),
        "failed_verifier_sha256": digest(ROOT / "scripts/finalize_round81_20260908.py"),
        "new_candidate_configurations": 0, "goal_achieved": False, "position_impact": 0}, exclusive=True)
    source = (ROOT / "research/staged_entry_inputs_v1.py").read_text(encoding="utf-8")
    require(source.count("if row.shares_after > 0:") == 1, "实际持仓修正位置不唯一")
    source = source.replace("if row.shares_after > 0:", "if row.shares > 0:")
    source = source.replace('"""参考入场先半仓，实际参考买入后的价格回升确认补足。"""',
        '"""第81轮来源修正：用每日实际持仓维持参考入场基准，交易规则不变。"""')
    (ROOT / "research/staged_entry_inputs_v1_1.py").write_text(source, encoding="utf-8")
    tests = (ROOT / "tests/test_staged_entry_v1.py").read_text(encoding="utf-8")
    tests = tests.replace("research.staged_entry_inputs_v1 import", "research.staged_entry_inputs_v1_1 import")
    tests = tests.replace('"shares_after": [100, 100, 0, 0]', '"shares_after": [100, np.nan, 0, np.nan], "shares": [100, 100, 0, 0]')
    tests += '''

def test_no_trade_missing_execution_balance_preserves_anchor_and_later_upgrade():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=6),
        "wealth": [1., .99, 1.01, .98, .98, .98], "open": [10., 10., 9.9, 10.1, 9.8, 9.8],
        "dividend": np.zeros(6), "previous_close": [np.nan, 10., 9.9, 10.1, 9.8, 9.8]})
    ledger = pd.DataFrame({"date": data.date.iloc[1:].to_numpy(), "shares_before": [0, 100, 100, 100, 0],
        "filled_quantity": [100, 0, 0, -100, 0], "shares_after": [100, np.nan, np.nan, 0, np.nan],
        "shares": [100, 100, 100, 0, 0]})
    f = reference_entry_frame(data, ledger, 1)
    result = staged_targets(f, [1, 1, 1, 1, 0, np.nan], 1)
    assert f.reference_holding.iloc[1:4].all()
    np.testing.assert_allclose(f.reference_entry_wealth_open.iloc[1:4], [1., 1., 1.])
    np.testing.assert_allclose(result.target.iloc[:5], [.5, .5, 1., 1., 0.])
    assert result.price_upgrade.iloc[2] and not result.price_upgrade.iloc[1]
    assert not f.reference_holding.iloc[4] and np.isnan(f.reference_entry_wealth_open.iloc[4])
'''
    (ROOT / "tests/test_staged_entry_v1_1.py").write_text(tests, encoding="utf-8")
    runner = (ROOT / "research/staged_entry_v1.py").read_text(encoding="utf-8")
    runner = runner.replace("import json\n", "import json\nimport shutil\n", 1)
    runner = runner.replace("staged_entry_inputs_v1", "staged_entry_inputs_v1_1")
    runner = runner.replace("510300_staged_entry_v1\"", "510300_staged_entry_v1_1\"")
    runner = runner.replace("510300_staged_entry_v1.json", "510300_staged_entry_v1_1.json")
    runner = runner.replace("test_staged_entry_v1.py", "test_staged_entry_v1_1.py")
    runner = runner.replace('PRIMARY = "PRICE_CONFIRMED_STAGED_ENTRY"',
        'ORIGINAL = ROOT / "reports/research/510300_staged_entry_v1"\nPRIMARY = "PRICE_CONFIRMED_STAGED_ENTRY"')
    runner = runner.replace('study_id="510300_PRICE_CONFIRMED_STAGED_ENTRY_V1",',
        'study_id="510300_PRICE_CONFIRMED_STAGED_ENTRY_V1_1_SOURCE_FIX", source_version=2, new_candidate_configurations=0,')
    runner = runner.replace('rules="docs/510300_STAGED_ENTRY_V1.md", execution_control=EXECUTION_CONTROL,',
        'rules="docs/510300_STAGED_ENTRY_V1.md", source_amendment="docs/510300_STAGED_ENTRY_SOURCE_FIX_20260908.md", execution_control=EXECUTION_CONTROL,')
    marker = '    cfg["frozen_files"] = '
    require(runner.count(marker) == 1, "冻结列表插入位置不唯一")
    runner = runner.replace(marker, '''    paths += [ROOT / cfg["source_amendment"], ROOT / "config/510300_staged_entry_v1.json",
        ROOT / "research/staged_entry_v1.py", ROOT / "research/staged_entry_inputs_v1.py", ORIGINAL / "result.json", ORIGINAL / "source_defect_receipt.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths += [ORIGINAL / period / cost / f"{EXECUTION_CONTROL}_{kind}.parquet" for kind in ["ledger", "decisions"]]
''' + marker)
    block = '''                ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, model, targets=target, event_mask=np.ones(len(frame), bool))
                decisions = decisions.merge(f.rename(columns={"date": "origin", "target": "staged_target"}), on="origin", how="left", validate="one_to_one")
                save_account(folder, model, ledger, decisions)'''
    replacement = '''                current_factors = f.rename(columns={"date": "origin", "target": "staged_target"})
                if model == PRIMARY:
                    ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, model, targets=target, event_mask=np.ones(len(frame), bool))
                    decisions = decisions.merge(current_factors, on="origin", how="left", validate="one_to_one")
                    save_account(folder, model, ledger, decisions)
                else:
                    source_folder = ORIGINAL / period / cost_id
                    ledger = pd.read_parquet(source_folder / f"{model}_ledger.parquet")
                    decisions = pd.read_parquet(source_folder / f"{model}_decisions.parquet")
                    # 成交和实际目标不变；仅更新原记录中不参与满仓对照决策的分批说明列。
                    decisions = decisions.drop(columns=[c for c in current_factors if c != "origin"], errors="ignore")
                    decisions = decisions.merge(current_factors, on="origin", how="left", validate="one_to_one")
                    folder.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_folder / f"{model}_ledger.parquet", folder / f"{model}_ledger.parquet")
                    decisions.to_parquet(folder / f"{model}_decisions.parquet", index=False)
                    require(digest(folder / f"{model}_ledger.parquet") == digest(source_folder / f"{model}_ledger.parquet"), "满仓对照账本未原字节复用")'''
    require(runner.count(block) == 1, "账户运行替换位置不唯一")
    runner = runner.replace(block, replacement)
    runner = runner.replace('"new_accounts_generated": 4, "new_execution_control_accounts": 2, "reused_control_accounts": 8',
        '"new_accounts_generated": 2, "new_execution_control_accounts": 0, "reused_execution_control_accounts": 2, "reused_control_accounts": 10')
    runner = runner.replace('"new_earlier_diagnostic_accounts": 4, "new_earlier_execution_control_accounts": 2, "reused_earlier_accounts": 8',
        '"new_earlier_diagnostic_accounts": 2, "new_earlier_execution_control_accounts": 0, "reused_earlier_execution_control_accounts": 2, "reused_earlier_accounts": 10')
    runner = runner.replace('"candidate_configurations": 1, "evaluation_accounts": 12,',
        '"candidate_configurations": 1, "source_version": 2, "new_candidate_configurations": 0, "evaluation_accounts": 12,')
    runner = runner.replace("第81轮一个分批进入设置及同执行满仓对照已冻结，尚无新账户收益。", "第81轮相同规则的持仓字段修正已冻结；首次收益已见，修正账户尚未运行。")
    (ROOT / "research/staged_entry_v1_1.py").write_text(runner, encoding="utf-8")
    (ROOT / "docs/510300_STAGED_ENTRY_SOURCE_FIX_20260908.md").write_text('''# 第81轮来源修正：每日实际持仓字段

首次账户已经运行，核对时发现真实来源错误，首次结果不能用于评价已冻结的分批规则。原代码、设置、结果和失败核对脚本全部保留，原目录增加缺陷回执。本修正是同一设置的第二个来源版本，不冒充新的独立策略或未看过的历史。

原参考账本中“成交后股数”仅在成交时有值，无成交日缺失；每日实际持仓另有完整字段。原实现错误地在无成交日清除持有标记和参考入场价格，导致后续日的价格确认被漏掉。修正只将每日是否实际持有改为读取每日实际股数，持续持有就保留本周期入场基准，实际卖完才清除。

首次六项合成测试把无成交日的“成交后股数”填成了数字，未覆盖真实来源结构。新版本继承六项并补充一项：真实无成交日的成交后股数缺失、实际仍持有；入场当日不确认，次日上涨确认补足，再在真正卖完时清除基准。

首批50%、确认后100%、严格高于参考入场连续开盘、参考进入退出、下一开盘执行、现金整手、费用和全部日历均按原协议不变。没有根据已见收益改变参数。七项测试通过后冻结修正来源，再只重算两时期两费用共四个分批账户。

首次四个同执行满仓对照只依赖原零一目标，不使用分批标记，其账本原字节复用；说明列重新关联正确阶段，实际目标与请求不变。另16个老参考账本复用。因此修正版主评价12记录，其中仅2新账户，较早同样12记录其中仅2新账户。首次来源的12主记录也保留，总共一个不同设置、两个已运行来源版本、24主评价记录。两来源实际生成账户合计12个，其中首次8个，修正4个。

原第77轮使用成交前股数或成交后股数判断直接风险日，无成交持仓日由完整的成交前股数覆盖；本缺陷不自动使该口径失效，不重跑第77轮。核对聚焦参考持仓、入场时钟、确认阶段和四个修正账户的完整经济结果，不进行全面审计或重新训练。
''', encoding="utf-8")
    print("第81轮原来源已保留，修正代码和七项必要测试已建立。", flush=True)


if __name__ == "__main__":
    main()
