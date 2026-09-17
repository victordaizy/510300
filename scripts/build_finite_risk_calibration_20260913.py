"""生成两档公开登记的风险预算校准，共用输入、账本和费用核对。"""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def save(name,text):
    with (ROOT/name).open('x',encoding='utf-8') as stream:
        stream.write(text)


def main():
    inputs='''"""在原174完整账户风险上同时计算12%和15%两档预算。"""
from pathlib import Path
import numpy as np
import pandas as pd
from research.account_volatility_exposure_inputs_v1 import risk_budget
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/research/510300_return_confirmation_auxiliary_batch_v1'
PARENT='EITHER_CONFIRMED_RUNS_AUXILIARY'
MODELS=[PARENT]
PRIMARY='ACCOUNT_RISK_12'
RISK_TARGETS={'ACCOUNT_RISK_12':.12,'ACCOUNT_RISK_15':.15}
CANDIDATES={'ACCOUNT_RISK_12':'12%账户风险预算','ACCOUNT_RISK_15':'15%账户风险预算'}


def calibrated_paths(parent,realized,first,cfg):
    require(cfg['candidate_models']==list(CANDIDATES) and cfg['risk_targets']==RISK_TARGETS
        and cfg['account_risk_window']==60,'固定两档风险预算或窗口不同')
    return {model:risk_budget(parent,realized,first,cfg['annual_days'],60,budget) for model,budget in RISK_TARGETS.items()}


def calibration_frames(data,parents_by_cost,cfg,start):
    frames,first=aligned_target_frames(data,parents_by_cost,MODELS,cfg,start)
    period='evaluation' if pd.Timestamp(start)==pd.Timestamp(cfg['evaluation_start']) else 'earlier_diagnostic'
    origins=np.arange(first-1,len(data)-1)
    rows=[]
    for cost,frame in frames.items():
        ledger=pd.read_parquet(SOURCE/period/cost/f'{PARENT}_ledger.parquet')
        require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(data.date.iloc[first:])),'来源收益日历错位')
        realized=np.full(len(data),np.nan)
        realized[first:]=ledger.net_return.to_numpy(float)
        parent=frame[PARENT+'_parent_target'].to_numpy(float)
        paths=calibrated_paths(parent,realized,first,cfg)
        frame['source_realized_net_return']=realized
        frame['source_account_volatility60']=paths[PRIMARY][1]
        for model,(target,risk,multiplier) in paths.items():
            np.testing.assert_allclose(risk,paths[PRIMARY][1],atol=0,rtol=0,equal_nan=True)
            frame[model+'_target']=target
            frame[model+'_multiplier']=multiplier
            values=target[origins]
            rows.append({'model':model,'cost':cost,'risk_budget':RISK_TARGETS[model],'decision_origins':len(origins),
                'positive_target_origins':int((values>0).sum()),'zero_target_origins':int((values==0).sum()),
                'unknown_target_origins':int(np.isnan(values).sum()),'full_target_origins':int((values==1).sum()),
                'mean_target':float(np.nanmean(values))})
    return frames,rows
'''
    save('research/finite_account_risk_calibration_inputs_v1.py',inputs)
    source=(ROOT/'research/account_volatility_exposure_v1.py').read_text(encoding='utf-8')
    prefix=source[:source.index('def run():')]
    prefix=prefix.replace('account_volatility_exposure_inputs_v1 import CANDIDATES, MODELS, PARENT, PRIMARY, SOURCE, account_volatility_frames',
        'finite_account_risk_calibration_inputs_v1 import CANDIDATES, MODELS, PARENT, PRIMARY, SOURCE, RISK_TARGETS, calibration_frames')
    prefix=prefix.replace('510300_account_volatility_exposure_v1','510300_finite_account_risk_calibration_v1')
    prefix=prefix.replace('510300_ACCOUNT_VOLATILITY_EXPOSURE_V1','510300_FINITE_ACCOUNT_RISK_CALIBRATION_V1')
    prefix=prefix.replace('510300_ACCOUNT_VOLATILITY_EXPOSURE_NEXT','510300_FINITE_ACCOUNT_RISK_CALIBRATION_NEXT')
    prefix=prefix.replace('第181轮','第194轮').replace('round=181','round=194').replace("'round': 181","'round': 194")
    prefix=prefix.replace("== 180", "== 193")
    prefix=prefix.replace('account_risk_target=.10','risk_targets=RISK_TARGETS.copy()')
    prefix=prefix.replace('candidate_configurations=1','candidate_configurations=2')
    prefix=prefix.replace('test_account_volatility_exposure_v1.py','test_finite_account_risk_calibration_v1.py')
    prefix=prefix.replace("'3 passed'", "'5 passed'").replace("'passed': 3", "'passed': 5")
    prefix=prefix.replace('PROGRESS_ROUND180_EIGHT_ACCOUNTS_JOINT_TARGET_FAILED','PROGRESS_ROUND193_FOUR_ACCOUNTS_VERIFIED_DELIVERED')
    prefix=prefix.replace("'ACCOUNT_VOLATILITY_EXPOSURE_FROZEN'","'FINITE_ACCOUNT_RISK_CALIBRATION_FROZEN'")
    prefix=prefix.replace("evidence_class='RETROSPECTIVE_REPLAY_PREVIOUSLY_OBSERVED_HISTORY'", "evidence_class='EXPLICIT_RETROSPECTIVE_FINITE_PARAMETER_CALIBRATION'")
    a,b=prefix.index('CONTROLS ='),prefix.index('\n\n\ndef prepare')
    controls='''CONTROLS = {PARENT:(SOURCE,'第174轮原组合'),
    'ACCOUNT_VOLATILITY_EXPOSURE':(ROOT/'reports/research/510300_account_volatility_exposure_v1','第181轮10%风险预算'),
    'BUY_HOLD':(ROOT/'reports/research/510300_rearmed_session_exit_v1','买入持有')}'''
    prefix=prefix[:a]+controls+prefix[b:]
    prefix=prefix.replace("ROOT/'research/account_volatility_exposure_inputs_v1.py',", "ROOT/'research/finite_account_risk_calibration_inputs_v1.py', ROOT/'research/account_volatility_exposure_inputs_v1.py', ROOT/'research/joint_account_acceptance_v1.py',")
    prefix=prefix.replace('from research.saved_target_batch_runner_v1 import run_saved_target_batch',
        'from research.saved_target_batch_runner_v1 import run_saved_target_batch\nfrom research.joint_account_acceptance_v1 import save_joint_assessment')
    prefix=prefix.replace('第194轮单一风险预算已冻结', '第194轮两档有限风险预算已冻结')
    prefix=prefix.replace("stream.write((ROOT/'docs/510300_FINITE_ACCOUNT_RISK_CALIBRATION_NEXT_20260913.md').read_text(encoding='utf-8'))",
        "stream.write((ROOT/'docs/510300_FINITE_ACCOUNT_RISK_CALIBRATION_NEXT_20260913.md').read_text(encoding='utf-8').replace('尚未登记、实现或回测。','本方案在必要测试通过后、首次账户计算前冻结。'))")
    run='''def run():
    result=run_saved_target_batch(ROOT,OUT,CONFIG,CANDIDATES,PARENTS,CONTROLS,calibration_frames)
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    print(json.dumps(save_joint_assessment(OUT,cfg,result),ensure_ascii=False),flush=True)


'''
    verify=source[source.index('def verify():'):source.index("if __name__ == '__main__':")]
    verify=verify.replace('第181轮','第194轮')
    verify=verify.replace('def expected(period, cost_id, frame, start):','def expected(period, cost_id, frame, start, model):')
    verify=verify.replace("multiplier[t] = .10/risk[t]", "multiplier[t] = cfg['risk_targets'][model]/risk[t]")
    verify=verify.replace('factors.account_risk_multiplier', "factors[model+'_multiplier']")
    verify=verify.replace("factors[PRIMARY+'_target']", "factors[model+'_target']")
    verify=verify.replace("f'{PRIMARY}_ledger.parquet'", "f'{model}_ledger.parquet'")
    verify=verify.replace("checks.append({'period': period", "checks.append({'model':model, 'period': period")
    verify=verify.replace("        risk = np.full(len(frame), np.nan)",
        "        require(pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(frame.date.iloc[indices])), '来源判断日期错位')\n"
        "        require(pd.DatetimeIndex(parent.decision_time).equals(pd.DatetimeIndex(parent.origin)+pd.Timedelta(hours=15,minutes=5)), '来源收盘时钟不同')\n"
        "        risk = np.full(len(frame), np.nan)")
    old='''    accounts, cycles, differences, count = verify_saved_target_accounts(OUT, cfg, result, data, dividends, expected, comparison_models=list(CONTROLS))
    require(len(accounts)==4 and count==5646, '第194轮核对范围不同')'''
    assert old in verify
    new='''    accounts,cycles,differences,count=[],[],[],0
    for model in CANDIDATES:
        def selected(period,cost_id,frame,start):
            return expected(period,cost_id,frame,start,model)
        local_cfg={**cfg,'primary':model}
        a,c,d,n=verify_saved_target_accounts(OUT,local_cfg,result,data,dividends,selected,comparison_models=list(CONTROLS))
        accounts.extend({'model':model,**row} for row in a)
        cycles.extend({'model':model,**row} for row in c)
        differences.extend({'model':model,**row} for row in d)
        count+=n
    for filename,rows in [('saved_account_checks.csv',accounts),('saved_actual_cycles.csv',cycles),('saved_comparison_differences.csv',differences)]:
        pd.DataFrame(rows).to_csv(OUT/filename,index=False,encoding='utf-8-sig')
    require(len(accounts)==8 and count==11292, '第194轮核对范围不同')'''
    verify=verify.replace(old,new)
    verify=verify.replace("'actual_accounts': 4", "'actual_accounts': 8")
    verify=verify.replace('PASS_FOUR_SIMULATED_ACCOUNTS_CAUSAL_RISK_BUDGET_AND_COSTS','PASS_EIGHT_SIMULATED_ACCOUNTS_FINITE_RISK_CALIBRATION_AND_COSTS')
    save('research/finite_account_risk_calibration_v1.py',prefix+run+verify+'''if __name__=='__main__':
    {'prepare':prepare,'run':run,'verify':verify}[sys.argv[1]]()
''')
    print('第194轮两档预算及八账户核对入口已生成，尚未回测。')


if __name__=='__main__':
    main()
