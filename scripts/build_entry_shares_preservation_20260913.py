"""生成保持份额执行方法，原已冻结引擎和研究文件保持原样。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(name, text):
    with (ROOT/name).open('x', encoding='utf-8') as stream:
        stream.write(text)


def main():
    source = (ROOT/'research/event_clock_account_v1.py').read_text(encoding='utf-8')
    source = source.replace('def simulate_event_account(', 'def simulate_preserved_shares(')
    source = source.replace('    require(event_mask is not None',
        '    require(prediction is None and targets is not None and model_id != "BUY_HOLD", "本方法只执行保存目标")\n'
        '    require(event_mask is not None')
    source = source.replace('    first = int(np.flatnonzero',
        '    require(np.asarray(event_mask, bool).all(), "本方法逐日检查已有目标")\n'
        '    first = int(np.flatnonzero', 1)
    old = '                result = choose_order(account, price, value, horizon * variance[t], cost, config) if prediction is not None else target_request(account, price, value, config)'
    assert old in source
    source = source.replace(old, '                result = preservation_request(account, price, value, config)')
    anchor = 'def simulate_preserved_shares('
    helper = '''def preservation_request(account, price, target, config):
    require(np.isfinite(target) and 0 <= target <= 1, "目标仓位超出范围")
    if account.shares > 0 and target > 0:
        return {"requested_quantity": 0, "reference_weight": float(target), "action": "已知正目标，保持本笔实际份额"}
    return target_request(account, price, target, config)


'''
    source = source.replace(anchor, helper+anchor)
    source = source.replace('按事件时钟调整的完整账户', '正目标期间保持初始实际份额的完整模拟账户')
    save('research/entry_shares_preservation_account_v1.py', source)

    batch = (ROOT/'research/saved_target_batch_runner_v1.py').read_text(encoding='utf-8')
    batch = batch.replace('from research.event_clock_account_v1 import simulate_event_account\n', '')
    batch = batch.replace('def run_saved_target_batch(root, out, config_path, candidates, parents, controls, build_frames):',
        'def run_custom_execution_batch(root, out, config_path, candidates, parents, controls, build_frames, simulate_account):')
    batch = batch.replace('simulate_event_account(', 'simulate_account(')
    batch = batch.replace('共读数据和保存对照，一次运行事先列明的有限目标候选', '共读数据和保存对照，由明确传入的执行方法运行有限候选')
    save('research/saved_target_custom_execution_v1.py', batch)

    check = (ROOT/'research/saved_target_account_checks_v1.py').read_text(encoding='utf-8')
    check = check.replace('def verify_saved_target_accounts(', 'def verify_preserved_shares_accounts(')
    check = check.replace('            within_band = known & (targets > 0) & (old_shares > 0) & (np.abs(targets-old_shares*prices/prior) < cfg["weight_band"])',
        '            within_band = known & (targets > 0) & (old_shares > 0)')
    check = check.replace('自身净值和既有带宽', '自身净值和正目标保持份额规则')
    check = check.replace('    """策略只提供独立目标；共用核对不重新运行账户或模型。"""',
        '    """独立目标和实际份额分别核对，不重新运行账户。"""')
    check = check.replace('            measured, stored = summarize(ledger, cfg), metric(result, primary, period, cost_id)',
        '            require(((ledger.filled_quantity <= 0) | (old_shares == 0)).all(), "持仓期间发生追加")\n'
        '            require(((ledger.filled_quantity >= 0) | (ledger.shares == 0)).all(), "出现部分退出")\n'
        '            measured, stored = summarize(ledger, cfg), metric(result, primary, period, cost_id)')
    save('research/saved_entry_shares_checks_v1.py', check)

    inputs = '''"""逐条沿用同费用第181轮目标，执行层单独决定保持实际份额。"""
import numpy as np
from research.intraday_overnight_increment_v1 import require
from research.saved_parent_target_alignment_v1 import aligned_target_frames

PARENT='ACCOUNT_VOLATILITY_EXPOSURE'
MODELS=[PARENT]
PRIMARY='ENTRY_SHARES_PRESERVATION'
CANDIDATES={PRIMARY:'正目标期间保持初次实际买入份额'}


def preservation_frames(data, parents_by_cost, cfg, start):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['holding_positive_policy']=='KEEP_ACTUAL_SHARES', '固定执行规则不同')
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start)
    origins=np.arange(first-1,len(data)-1)
    summaries=[]
    for cost,frame in frames.items():
        source=frame[PARENT+'_parent_target'].to_numpy(float)
        frame[PRIMARY+'_target']=source.copy()
        values=source[origins]
        summaries.append({'model':PRIMARY,'cost':cost,'decision_origins':len(origins),
            'positive_target_origins':int((values>0).sum()),'zero_target_origins':int((values==0).sum()),
            'unknown_target_origins':int(np.isnan(values).sum()),'source_target_changes':0})
    return frames,summaries
'''
    save('research/entry_shares_preservation_inputs_v1.py', inputs)

    runner=(ROOT/'research/thursday_weekly_reduction_v1.py').read_text(encoding='utf-8')
    runner=runner.replace('thursday_weekly_reduction','entry_shares_preservation').replace('THURSDAY_WEEKLY_REDUCTION','ENTRY_SHARES_PRESERVATION')
    runner=runner.replace('189','193').replace('188','192')
    runner=runner.replace('固定周四判断日清仓条件','正目标期间保持实际初始份额')
    runner=runner.replace('weekly_frames','preservation_frames')
    runner=runner.replace('from research.saved_target_account_checks_v1 import verify_saved_target_accounts',
        'from research.saved_entry_shares_checks_v1 import verify_preserved_shares_accounts')
    runner=runner.replace('from research.saved_target_batch_runner_v1 import run_saved_target_batch',
        'from research.saved_target_custom_execution_v1 import run_custom_execution_batch\nfrom research.entry_shares_preservation_account_v1 import simulate_preserved_shares')
    runner=runner.replace("'4 passed'", "'6 passed'").replace("'passed':4", "'passed':6")
    runner=runner.replace('zero_origin_weekday=3', "holding_positive_policy='KEEP_ACTUAL_SHARES'")
    runner=runner.replace('PROGRESS_ROUND192_EIGHT_ACCOUNTS_AND_SESSION_ATTRIBUTION', 'PROGRESS_ROUND192_TRAINED_VERIFIED_AND_DELIVERED')
    runner=runner.replace("ROOT/'research/saved_target_batch_runner_v1.py'", "ROOT/'research/saved_target_custom_execution_v1.py'")
    runner=runner.replace("ROOT/'research/saved_target_account_checks_v1.py'", "ROOT/'research/saved_entry_shares_checks_v1.py'")
    runner=runner.replace("ROOT/'research/event_clock_account_v1.py'", "ROOT/'research/entry_shares_preservation_account_v1.py'")
    runner=runner.replace("ROOT/'tests/test_entry_shares_preservation_v1.py',receipt", "ROOT/'tests/test_entry_shares_preservation_v1.py',ROOT/'tests/test_median_continuation_v1.py',receipt")
    runner=runner.replace('run_saved_target_batch(ROOT,OUT,CONFIG,CANDIDATES,PARENTS,CONTROLS,preservation_frames)',
        'run_custom_execution_batch(ROOT,OUT,CONFIG,CANDIDATES,PARENTS,CONTROLS,preservation_frames,simulate_preserved_shares)')
    runner=runner.replace('target[origin]=0. if frame.date.iloc[origin].weekday()==3 else value', 'target[origin]=value')
    runner=runner.replace('        np.testing.assert_array_equal(factors.origin_weekday,[d.weekday() for d in frame.date])\n', '')
    runner=runner.replace('verify_saved_target_accounts(', 'verify_preserved_shares_accounts(')
    runner=runner.replace('saved_weekly_reduction_checks.csv', 'saved_preserved_shares_checks.csv')
    runner=runner.replace("'maximum_target_error':0.",
        "'maximum_target_error':0.,'intermediate_position_trades':int((filled & ledger.shares_before.gt(0) & ledger.shares.gt(0)).sum())")
    runner=runner.replace("        folder=OUT/period/cost_id", "        require(pd.DatetimeIndex(parent.decision_time).equals(pd.DatetimeIndex(parent.origin)+pd.Timedelta(hours=15,minutes=5)),'来源真实判断时钟不同')\n        folder=OUT/period/cost_id")
    runner=runner.replace("        stream.write((ROOT/'docs/510300_ENTRY_SHARES_PRESERVATION_NEXT_20260913.md').read_text(encoding='utf-8'))",
        "        stream.write((ROOT/'docs/510300_ENTRY_SHARES_PRESERVATION_NEXT_20260913.md').read_text(encoding='utf-8').replace('尚未登记、实现、测试或回测。','本方案在必要测试通过后、首次账户计算前冻结。'))")
    save('research/entry_shares_preservation_v1.py',runner)
    print('第193轮独立执行方法、输入对齐及保存结果核对已生成，尚未回测。')


if __name__=='__main__':
    main()
