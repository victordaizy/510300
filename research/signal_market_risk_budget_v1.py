"""第187轮直接按市场波动确定正资格仓位。"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends
from research.signal_market_risk_budget_inputs_v1 import CANDIDATES,MODELS,PARENT,PRIMARY,EPISODE,market_frames
from research.intraday_overnight_increment_v1 import digest,now,require,write_json
from research.joint_account_acceptance_v1 import save_joint_assessment
from research.saved_target_account_checks_v1 import verify_saved_target_accounts
from research.saved_target_batch_runner_v1 import run_saved_target_batch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_signal_market_risk_budget_v1'
CONFIG=ROOT/'config/510300_signal_market_risk_budget_v1.json'
SOURCE=ROOT/'reports/research/510300_return_confirmation_auxiliary_batch_v1'
PARENTS={PARENT:SOURCE}
CONTROLS={'ACCOUNT_VOLATILITY_EXPOSURE':(ROOT/'reports/research/510300_account_volatility_exposure_v1','第181轮账户风险预算'),
    'EPISODE_ACCOUNT_RISK_BUDGET':(ROOT/'reports/research/510300_episode_account_risk_budget_v1','第182轮区间固定倍率'),
    'BINARY_SIGNAL_EXPOSURE':(ROOT/'reports/research/510300_binary_signal_exposure_v1','第183轮正资格满仓'),
    'BUY_HOLD':(ROOT/'reports/research/510300_rearmed_session_exit_v1','买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(),'第187轮已有冻结或运行')
    OUT.mkdir(parents=True,exist_ok=True)
    receipt=OUT/'tests_receipt.json'
    require(not receipt.exists(),'必要测试已有回执')
    began=time.perf_counter()
    t=subprocess.run([sys.executable,'-m','pytest','tests/test_signal_market_risk_budget_v1.py','-q','-p','no:cacheprovider'],
                     cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (OUT/'tests_output.txt').write_text(t.stdout+t.stderr,encoding='utf-8')
    print(t.stdout,flush=True)
    require(t.returncode==0 and '5 passed' in t.stdout,'第187轮测试未通过')
    write_json(receipt,{'recorded_at':now(),'exit_code':t.returncode,'passed':5,'seconds':time.perf_counter()-began,
                       'timing_scope':'完整测试进程墙钟'},exclusive=True)
    source_config=ROOT/'config/510300_return_confirmation_auxiliary_batch_v1.json'
    old=json.loads(source_config.read_text(encoding='utf-8'))
    keys=['evaluation_start','data_cutoff','initial_capital','lot','tick','limit_fraction','annual_days',
          'cash_annual_rate_assumption','high_sharpe_target','costs','features','dividends','earlier_start','earlier_terminal','weight_band']
    cfg={k:old[k] for k in keys}
    cfg.update(study_id='510300_SIGNAL_MARKET_RISK_BUDGET_V1',round=187,primary=PRIMARY,candidate_models=list(CANDIDATES),
        parent_models=MODELS,candidate_configurations=2,registered_at=now(),decision_clock='15:05:00',
        rules='docs/510300_SIGNAL_MARKET_RISK_BUDGET_V1.md',market_volatility_window=60,market_risk_budget=.1,
        annual_return_target=.10,new_model_fits=0,new_reference_accounts=0,source_budget_cny=0,position_impact=0,
        goal_achieved=False,independent_validation='NOT_ESTABLISHED',
        evidence_class='RETROSPECTIVE_REPLAY_PREVIOUSLY_OBSERVED_HISTORY',
        previous_goal_turn_classification='PROGRESS_ROUND186_EIGHT_ACCOUNTS_JOINT_TARGET_FAILED',
        legacy_meets_point_target_field='SHARPE_ONLY_USE_JOINT_TARGET_ASSESSMENT')
    rules=ROOT/cfg['rules']
    with rules.open('x',encoding='utf-8') as stream:
        stream.write((ROOT/'docs/510300_SIGNAL_MARKET_RISK_BUDGET_NEXT_20260913.md').read_text(encoding='utf-8'))
        stream.write('\n\n## 来源全部中文因素及进入退出规则\n\n')
        stream.write((ROOT/old['rules']).read_text(encoding='utf-8'))
    paths=[Path(__file__),ROOT/'research/signal_market_risk_budget_inputs_v1.py',ROOT/'research/joint_account_acceptance_v1.py',
        ROOT/'research/saved_target_batch_runner_v1.py',ROOT/'research/saved_parent_target_alignment_v1.py',
        ROOT/'research/saved_target_account_checks_v1.py',ROOT/'research/event_clock_account_v1.py',
        ROOT/'research/adaptive_allocation_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',
        ROOT/'tests/test_signal_market_risk_budget_v1.py',receipt,OUT/'tests_output.txt',rules,
        ROOT/cfg['features'],ROOT/cfg['dividends'],source_config,ROOT/old['rules'],SOURCE/'saved_verification_receipt.json',
        ROOT/'config/510300_research_authority_v6.json']
    for period in ['evaluation','earlier_diagnostic']:
        for cost in cfg['costs']:
            paths += [SOURCE/period/cost/f'{PARENT}_decisions.parquet']
            paths += [folder/period/cost/f'{model}_ledger.parquet' for model,(folder,_) in CONTROLS.items()]
    cfg['frozen_files']=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG,cfg,exclusive=True)
    p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    i=json.loads(p.read_text(encoding='utf-8'))
    require(i['latest_completed_round']['round']==186,'第187轮前序不同')
    i['running_studies']=[{'round':187,'study':cfg['study_id'],'status':'FROZEN_NOT_STARTED','config':str(CONFIG.relative_to(ROOT))}]
    i['next_work'].update(registered=True,status='SIGNAL_MARKET_RISK_BUDGET_FROZEN',source=cfg['rules'])
    write_json(p,i)
    print('第187轮已冻结。',flush=True)


def run():
    r=run_saved_target_batch(ROOT,OUT,CONFIG,CANDIDATES,PARENTS,CONTROLS,market_frames)
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    print(json.dumps(save_joint_assessment(OUT,cfg,r),ensure_ascii=False),flush=True)


def verify():
    require(not (OUT/'saved_verification_receipt.json').exists(),'第187轮已核对')
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
        parent=pd.read_parquet(SOURCE/period/cost_id/f'{PARENT}_decisions.parquet')
        np.testing.assert_array_equal(parent.origin_index,origins)
        require(pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(frame.date.iloc[origins])),'来源收盘错位')
        require(pd.DatetimeIndex(parent.execution_date).equals(pd.DatetimeIndex(frame.date.iloc[origins+1])),'来源下一开盘错位')
        close=frame.close.to_numpy(float)
        div=frame.dividend.to_numpy(float)
        returns=np.r_[np.nan,(close[1:]+div[1:])/close[:-1]-1.]
        risk=np.full(len(frame),np.nan)
        for t in origins:
            window=returns[t-59:t+1]
            require(len(window)==60 and np.isfinite(window).all(),'判断时点六十日市场收益缺失')
            mean=math.fsum(window)/60
            risk[t]=math.sqrt(math.fsum((x-mean)**2 for x in window)/59*242)
        np.testing.assert_allclose(frame.vol60.iloc[origins],risk[origins],atol=1e-12,rtol=0)
        budget=np.full(len(frame),np.nan)
        for t in origins:
            if math.isfinite(risk[t]) and risk[t]>=0:
                budget[t]=1. if risk[t]<=.1 else .1/risk[t]
        source=np.full(len(frame),np.nan)
        source[origins]=parent.reference_weight.to_numpy(float)
        target=np.full(len(frame),np.nan)
        target[source==0]=0.
        fixed=np.full(len(frame),np.nan)
        starts=np.zeros(len(frame),bool)
        groups=np.cumsum(source==0)
        for group in np.unique(groups):
            positive=np.flatnonzero((groups==group)&(source>0))
            if not len(positive):
                continue
            begin=int(positive[0])
            starts[begin]=True
            extent=np.flatnonzero((groups==group)&(np.arange(len(frame))>=begin))
            fixed[extent]=budget[begin]
            if active_model==EPISODE:
                target[positive]=budget[begin]
        if active_model==PRIMARY:
            target[source>0]=budget[source>0]
        else:
            require(active_model==EPISODE,'市场预算候选身份不同')
        folder=OUT/period/cost_id
        factors=pd.read_parquet(folder/'factors.parquet')
        np.testing.assert_allclose(factors[active_model+'_target'],target,atol=1e-12,rtol=0,equal_nan=True)
        np.testing.assert_allclose(factors.frozen_episode_budget,fixed,atol=1e-12,rtol=0,equal_nan=True)
        np.testing.assert_array_equal(factors.positive_episode_start,starts)
        np.testing.assert_allclose(factors.current_market_budget.iloc[origins],budget[origins],atol=1e-12,rtol=0)
        require(pd.DatetimeIndex(factors.decision_time.iloc[origins]).equals(pd.DatetimeIndex(frame.date.iloc[origins])+pd.Timedelta(hours=15,minutes=5)),'收盘判断时钟不同')
        ledger=pd.read_parquet(folder/f'{active_model}_ledger.parquet')
        require(ledger.cash.ge(-1e-7).all() and ledger.shares.ge(0).all(),'账户出现融资或负份额')
        np.testing.assert_allclose(ledger.open,frame.open.iloc[first:],atol=0,rtol=0)
        np.testing.assert_allclose(ledger.mark.iloc[:-1],frame.close.iloc[first:-1],atol=0,rtol=0)
        filled=ledger.filled_quantity.ne(0)
        qty=ledger.loc[filled,'filled_quantity'].to_numpy()
        op=ledger.loc[filled,'open'].to_numpy()
        cost=cfg['costs'][cost_id]
        unit=op*(1+np.sign(qty)*cost['slippage'])/cfg['tick']
        prices=np.where(qty>0,np.ceil(unit-1e-10),np.floor(unit+1e-10))*cfg['tick']
        np.testing.assert_allclose(ledger.loc[filled,'fill_price'],prices,atol=1e-12,rtol=0)
        np.testing.assert_allclose(ledger.loc[filled,'commission'],np.maximum(abs(qty)*prices*cost['commission'],cost['minimum']),atol=1e-8,rtol=0)
        np.testing.assert_allclose(ledger.loc[filled,'slippage_cost'],abs(qty)*abs(prices-op),atol=1e-8,rtol=0)
        checks.append({'period':period,'cost':cost_id,'model':active_model,'fills':int(filled.sum()),'maximum_target_error':float(np.nanmax(np.abs(factors[active_model+'_target'].to_numpy()-target)))})
        return target

    accounts,cycles,diffs=[],[],[]
    total_count=0
    for active_model in CANDIDATES:
        aa,cc,dd,count=verify_saved_target_accounts(OUT,{**cfg,'primary':active_model},r,data,dividends,expected,comparison_models=list(CONTROLS))
        accounts.extend({'model':active_model,**x} for x in aa)
        cycles.extend({'model':active_model,**x} for x in cc)
        diffs.extend({'model':active_model,**x} for x in dd)
        total_count+=count
    require(len(accounts)==8 and total_count==11292,'第187轮核对范围不同')
    for name,rows in [('saved_account_checks.csv',accounts),('saved_actual_cycles.csv',cycles),('saved_comparison_differences.csv',diffs),('saved_market_risk_checks.csv',checks)]:
        pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    receipt={'verified_at':now(),'status':'PASS_EIGHT_SIMULATED_ACCOUNTS_SIGNAL_MARKET_RISK_AND_COSTS',
        'actual_accounts':8,'actual_decisions_checked':total_count,'complete_actual_cycles':len(cycles),
        'simulated_fills_checked':sum(x['fills'] for x in checks),'new_models_or_accounts':0,
        'independent_performance_validation':False,'security_audit_performed':False,'reviewer_source_sha256':digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json',receipt,exclusive=True)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    {'prepare':prepare,'run':run,'verify':verify}[sys.argv[1]]()
