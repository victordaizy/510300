"""第185轮固定保护与恢复重入，比较原账户风险预算及满仓。"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends
from research.protected_recovery_reentry_inputs_v1 import CANDIDATES,MODELS,PARENT,BUDGET_PARENT,PRIMARY,FULL,recovery_frames
from research.intraday_overnight_increment_v1 import digest,now,require,write_json
from research.joint_account_acceptance_v1 import save_joint_assessment
from research.saved_target_account_checks_v1 import verify_saved_target_accounts
from research.saved_target_batch_runner_v1 import run_saved_target_batch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_protected_recovery_reentry_v1'
CONFIG=ROOT/'config/510300_protected_recovery_reentry_v1.json'
SOURCE=ROOT/'reports/research/510300_return_confirmation_auxiliary_batch_v1'
RISK_SOURCE=ROOT/'reports/research/510300_account_volatility_exposure_v1'
PARENTS={PARENT:SOURCE,BUDGET_PARENT:RISK_SOURCE}
CONTROLS={PARENT:(SOURCE,'第174轮原预算'),
    'BINARY_SIGNAL_EXPOSURE':(ROOT/'reports/research/510300_binary_signal_exposure_v1','第183轮不设保护满仓'),
    'ACCOUNT_VOLATILITY_EXPOSURE':(ROOT/'reports/research/510300_account_volatility_exposure_v1','第181轮每日风险预算'),
    'SIGNAL_EPISODE_PROTECTIVE_EXIT':(ROOT/'reports/research/510300_signal_episode_protective_exit_v1','第184轮锁定保护退出'),
    'BUY_HOLD':(ROOT/'reports/research/510300_rearmed_session_exit_v1','买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(),'第185轮已有冻结或运行')
    OUT.mkdir(parents=True,exist_ok=True)
    receipt=OUT/'tests_receipt.json'
    require(not receipt.exists(),'必要测试已有回执')
    began=time.perf_counter()
    t=subprocess.run([sys.executable,'-m','pytest','tests/test_protected_recovery_reentry_v1.py','-q','-p','no:cacheprovider'],
                     cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (OUT/'tests_output.txt').write_text(t.stdout+t.stderr,encoding='utf-8')
    print(t.stdout,flush=True)
    require(t.returncode==0 and '7 passed' in t.stdout,'第185轮测试未通过')
    write_json(receipt,{'recorded_at':now(),'exit_code':t.returncode,'passed':7,'seconds':time.perf_counter()-began,
                       'timing_scope':'完整测试进程墙钟'},exclusive=True)
    source_config=ROOT/'config/510300_return_confirmation_auxiliary_batch_v1.json'
    old=json.loads(source_config.read_text(encoding='utf-8'))
    keys=['evaluation_start','data_cutoff','initial_capital','lot','tick','limit_fraction','annual_days',
          'cash_annual_rate_assumption','high_sharpe_target','costs','features','dividends','earlier_start','earlier_terminal','weight_band']
    cfg={k:old[k] for k in keys}
    cfg.update(study_id='510300_PROTECTED_RECOVERY_REENTRY_V1',round=185,primary=PRIMARY,candidate_models=list(CANDIDATES),
        parent_models=MODELS,candidate_configurations=2,registered_at=now(),decision_clock='15:05:00',
        rules='docs/510300_PROTECTED_RECOVERY_REENTRY_V1.md',recovery_rule='KNOWN_POSITIVE_SOURCE_AND_PREEXIT_HIGH_RECLAIM',protective_multiple=3.,protective_volatility_window=20,
        annual_return_target=.10,new_model_fits=0,new_reference_accounts=0,source_budget_cny=0,position_impact=0,
        goal_achieved=False,independent_validation='NOT_ESTABLISHED',
        evidence_class='RETROSPECTIVE_REPLAY_PREVIOUSLY_OBSERVED_HISTORY',
        previous_goal_turn_classification='PROGRESS_ROUND184_FOUR_ACCOUNTS_JOINT_TARGET_FAILED',
        legacy_meets_point_target_field='SHARPE_ONLY_USE_JOINT_TARGET_ASSESSMENT')
    rules=ROOT/cfg['rules']
    with rules.open('x',encoding='utf-8') as stream:
        stream.write((ROOT/'docs/510300_PROTECTED_RECOVERY_REENTRY_NEXT_20260913.md').read_text(encoding='utf-8'))
        stream.write('\n\n## 来源全部中文因素及进入退出规则\n\n')
        stream.write((ROOT/old['rules']).read_text(encoding='utf-8'))
    paths=[Path(__file__),ROOT/'research/protected_recovery_reentry_inputs_v1.py',ROOT/'research/joint_account_acceptance_v1.py',
        ROOT/'research/saved_target_batch_runner_v1.py',ROOT/'research/saved_parent_target_alignment_v1.py',
        ROOT/'research/saved_target_account_checks_v1.py',ROOT/'research/event_clock_account_v1.py',
        ROOT/'research/adaptive_allocation_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',
        ROOT/'tests/test_protected_recovery_reentry_v1.py',receipt,OUT/'tests_output.txt',rules,
        ROOT/cfg['features'],ROOT/cfg['dividends'],source_config,ROOT/old['rules'],SOURCE/'saved_verification_receipt.json',
        ROOT/'config/510300_research_authority_v6.json',ROOT/'config/510300_account_volatility_exposure_v1.json',
        ROOT/'docs/510300_ACCOUNT_VOLATILITY_EXPOSURE_V1.md',RISK_SOURCE/'saved_verification_receipt.json']
    for period in ['evaluation','earlier_diagnostic']:
        for cost in cfg['costs']:
            paths += [SOURCE/period/cost/f'{PARENT}_decisions.parquet',RISK_SOURCE/period/cost/f'{BUDGET_PARENT}_decisions.parquet']
            paths += [folder/period/cost/f'{model}_ledger.parquet' for model,(folder,_) in CONTROLS.items()]
    cfg['frozen_files']=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG,cfg,exclusive=True)
    p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    i=json.loads(p.read_text(encoding='utf-8'))
    require(i['latest_completed_round']['round']==184,'第185轮前序不同')
    i['running_studies']=[{'round':185,'study':cfg['study_id'],'status':'FROZEN_NOT_STARTED','config':str(CONFIG.relative_to(ROOT))}]
    i['next_work'].update(registered=True,status='PROTECTED_RECOVERY_REENTRY_FROZEN',source=cfg['rules'])
    write_json(p,i)
    print('第185轮已冻结。',flush=True)


def run():
    r=run_saved_target_batch(ROOT,OUT,CONFIG,CANDIDATES,PARENTS,CONTROLS,recovery_frames)
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    print(json.dumps(save_joint_assessment(OUT,cfg,r),ensure_ascii=False),flush=True)


def verify():
    require(not (OUT/'saved_verification_receipt.json').exists(),'第185轮已核对')
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    r=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    for x in cfg['frozen_files']:
        require(digest(ROOT/x['path'])==x['sha256'],'冻结文件改变')
    data=pd.read_parquet(ROOT/cfg['features'])
    dividends=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    checks=[]

    def expected(period,cost_id,frame,start):
        first=int(np.flatnonzero(frame.date.ge(start))[0])
        origins=np.arange(first-1,len(frame)-1)
        source,budget=np.full(len(frame),np.nan),np.full(len(frame),np.nan)
        for name,destination in [(PARENT,source),(BUDGET_PARENT,budget)]:
            parent=pd.read_parquet(PARENTS[name]/period/cost_id/f'{name}_decisions.parquet')
            np.testing.assert_array_equal(parent.origin_index,origins)
            require(pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(frame.date.iloc[origins])),'来源收盘错位')
            require(pd.DatetimeIndex(parent.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[origins+1])),'来源下一开盘错位')
            destination[origins]=parent.reference_weight.to_numpy(float)
        close=frame.close.to_numpy(float)
        cash_div=frame.dividend.to_numpy(float)
        returns=np.r_[np.nan,(close[1:]+cash_div[1:])/close[:-1]-1.]
        wealth=np.cumprod(np.r_[1.,1.+returns[1:]])
        np.testing.assert_allclose(frame.wealth,wealth,atol=1e-12,rtol=1e-12)
        sigma=np.full(len(frame),np.nan)
        for t in range(19,len(frame)):
            values=returns[t-19:t+1]
            if np.isfinite(values).all():
                mean=math.fsum(values)/20
                sigma[t]=math.sqrt(math.fsum((v-mean)**2 for v in values)/19)
        groups=np.cumsum(source==0)
        gate=np.full(len(frame),np.nan)
        gate[source==0]=0.
        width=np.full(len(frame),np.nan)
        peaks=np.full(len(frame),np.nan)
        recovery=np.full(len(frame),np.nan)
        starts,exits,entries,paused=(np.zeros(len(frame),bool) for _ in range(4))
        for group in np.unique(groups):
            positives=np.flatnonzero((groups==group)&(source>0))
            if not len(positives):
                continue
            start_index=int(positives[0])
            starts[start_index]=True
            if not np.isfinite(sigma[start_index]) or sigma[start_index]<=0:
                continue
            last=int(np.flatnonzero(groups==group)[-1])
            distance=3.*sigma[start_index]
            width[start_index:last+1]=distance
            cursor=start_index
            while cursor<=last:
                segment=np.arange(cursor,last+1)
                running_high=np.maximum.accumulate(wealth[segment])
                crossings=np.flatnonzero(wealth[segment]<=running_high*(1.-distance))
                end=cursor+int(crossings[0]) if len(crossings) else last+1
                hold_indices=np.arange(cursor,end)
                gate[hold_indices[source[hold_indices]>0]]=1.
                peaks[hold_indices]=running_high[:len(hold_indices)]
                if end==last+1:
                    break
                exits[end]=True
                threshold=float(running_high[end-cursor])
                future=np.arange(end+1,last+1)
                recovered=future[(wealth[future]>=threshold)&(source[future]>0)]
                next_entry=int(recovered[0]) if len(recovered) else last+1
                paused_indices=np.arange(end,next_entry)
                gate[paused_indices]=0.
                paused[paused_indices]=True
                peaks[paused_indices]=threshold
                recovery[paused_indices]=threshold
                if next_entry==last+1:
                    break
                entries[next_entry]=True
                cursor=next_entry
        gate[-1]=np.nan
        if active_model==FULL:
            target=gate.copy()
        else:
            require(active_model==PRIMARY,'候选身份不同')
            target=np.full(len(frame),np.nan)
            target[gate==0]=0.
            use=(gate==1)&np.isfinite(budget)
            target[use]=budget[use]
        folder=OUT/period/cost_id
        f=pd.read_parquet(folder/'factors.parquet')
        np.testing.assert_allclose(f.market_daily_volatility20,sigma,atol=1e-12,rtol=1e-12,equal_nan=True)
        np.testing.assert_allclose(f.fixed_protection_distance,width,atol=1e-12,rtol=1e-12,equal_nan=True)
        np.testing.assert_allclose(f.observation_peak,peaks,atol=1e-12,rtol=1e-12,equal_nan=True)
        np.testing.assert_allclose(f.fixed_recovery_high,recovery,atol=1e-12,rtol=1e-12,equal_nan=True)
        for key,val in [('positive_episode_start',starts),('protective_exit_trigger',exits),('reentry_trigger',entries),('waiting_recovery',paused)]:
            np.testing.assert_array_equal(f[key],val)
        np.testing.assert_allclose(f.recovery_gate,gate,atol=0,rtol=0,equal_nan=True)
        np.testing.assert_allclose(f[active_model+'_target'],target,atol=0,rtol=0,equal_nan=True)
        require(pd.DatetimeIndex(f.decision_time.iloc[origins]).equals(pd.DatetimeIndex(frame.date.iloc[origins])+pd.Timedelta(hours=15,minutes=5)),'收盘判断时钟不同')
        ledger=pd.read_parquet(folder/f'{active_model}_ledger.parquet')
        require(ledger.cash.ge(-1e-7).all() and ledger.shares.ge(0).all(),'账户出现融资或负份额')
        np.testing.assert_allclose(ledger.open,frame.open.iloc[first:],atol=0,rtol=0)
        np.testing.assert_allclose(ledger.mark.iloc[:-1],frame.close.iloc[first:-1],atol=0,rtol=0)
        filled=ledger.filled_quantity.ne(0)
        qty=ledger.loc[filled,'filled_quantity'].to_numpy()
        op=ledger.loc[filled,'open'].to_numpy()
        cost=cfg['costs'][cost_id]
        unit=op*(1+np.sign(qty)*cost['slippage'])/cfg['tick']
        price=np.where(qty>0,np.ceil(unit-1e-10),np.floor(unit+1e-10))*cfg['tick']
        np.testing.assert_allclose(ledger.loc[filled,'fill_price'],price,atol=1e-12,rtol=0)
        np.testing.assert_allclose(ledger.loc[filled,'commission'],np.maximum(abs(qty)*price*cost['commission'],cost['minimum']),atol=1e-8,rtol=0)
        np.testing.assert_allclose(ledger.loc[filled,'slippage_cost'],abs(qty)*abs(price-op),atol=1e-8,rtol=0)
        checks.append({'period':period,'cost':cost_id,'model':active_model,'fills':int(filled.sum()),
            'protection_triggers':int(exits[origins].sum()),'reentries':int(entries[origins].sum()),'maximum_target_error':0.})
        return target

    accounts,cycles,diffs=[],[],[]
    total_count=0
    for active_model in CANDIDATES:
        aa,cc,dd,count=verify_saved_target_accounts(OUT,{**cfg,'primary':active_model},r,data,dividends,expected,comparison_models=list(CONTROLS))
        accounts.extend({'model':active_model,**x} for x in aa)
        cycles.extend({'model':active_model,**x} for x in cc)
        diffs.extend({'model':active_model,**x} for x in dd)
        total_count+=count
    require(len(accounts)==8 and total_count==11292,'第185轮核对范围不同')
    for name,rows in [('saved_account_checks.csv',accounts),('saved_actual_cycles.csv',cycles),('saved_comparison_differences.csv',diffs),('saved_recovery_reentry_checks.csv',checks)]:
        pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    receipt={'verified_at':now(),'status':'PASS_EIGHT_SIMULATED_ACCOUNTS_PROTECTED_RECOVERY_AND_COSTS',
        'actual_accounts':8,'actual_decisions_checked':total_count,'complete_actual_cycles':len(cycles),
        'simulated_fills_checked':sum(x['fills'] for x in checks),'new_models_or_accounts':0,
        'independent_performance_validation':False,'security_audit_performed':False,'reviewer_source_sha256':digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json',receipt,exclusive=True)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    {'prepare':prepare,'run':run,'verify':verify}[sys.argv[1]]()
