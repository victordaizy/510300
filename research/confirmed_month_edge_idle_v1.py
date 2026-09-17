"""第191轮前一收盘方向确认的原空仓日历补充。"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends
from research.confirmed_month_edge_idle_inputs_v1 import CANDIDATES,MODELS,PARENT,PRIMARY,CALENDAR_SOURCE,confirmed_frames
from research.intraday_overnight_increment_v1 import digest,now,require,write_json
from research.joint_account_acceptance_v1 import save_joint_assessment
from research.saved_target_account_checks_v1 import verify_saved_target_accounts
from research.saved_target_batch_runner_v1 import run_saved_target_batch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_confirmed_month_edge_idle_v1'
CONFIG=ROOT/'config/510300_confirmed_month_edge_idle_v1.json'
SOURCE=ROOT/'reports/research/510300_account_volatility_exposure_v1'
PARENTS={PARENT:SOURCE}
CONTROLS={PARENT:(SOURCE,'第181轮原账户风险预算'),
    'IDLE_MONTH_EDGE_OPPORTUNITY':(ROOT/'reports/research/510300_month_edge_opportunity_v1','第190轮无方向确认主方案'),
    'BUY_HOLD':(ROOT/'reports/research/510300_rearmed_session_exit_v1','买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(),'第191轮已有冻结或运行')
    OUT.mkdir(parents=True,exist_ok=True)
    receipt=OUT/'tests_receipt.json'
    require(not receipt.exists(),'必要测试已有回执')
    began=time.perf_counter()
    t=subprocess.run([sys.executable,'-m','pytest','tests/test_confirmed_month_edge_idle_v1.py','-q','-p','no:cacheprovider'],
                     cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (OUT/'tests_output.txt').write_text(t.stdout+t.stderr,encoding='utf-8')
    print(t.stdout,flush=True)
    require(t.returncode==0 and '5 passed' in t.stdout,'第191轮测试未通过')
    write_json(receipt,{'recorded_at':now(),'exit_code':t.returncode,'passed':5,'seconds':time.perf_counter()-began,
                       'timing_scope':'完整测试进程墙钟'},exclusive=True)
    source_config=ROOT/'config/510300_account_volatility_exposure_v1.json'
    old=json.loads(source_config.read_text(encoding='utf-8'))
    keys=['evaluation_start','data_cutoff','initial_capital','lot','tick','limit_fraction','annual_days',
          'cash_annual_rate_assumption','high_sharpe_target','costs','features','dividends','earlier_start','earlier_terminal','weight_band']
    cfg={k:old[k] for k in keys}
    cfg.update(study_id='510300_CONFIRMED_MONTH_EDGE_IDLE_V1',round=191,primary=PRIMARY,candidate_models=list(CANDIDATES),
        parent_models=MODELS,candidate_configurations=1,registered_at=now(),decision_clock='09:00:00',source_signal_clock='15:05:00',
        rules='docs/510300_CONFIRMED_MONTH_EDGE_IDLE_V1.md',confirmation_window=20,confirmation_rounding_tolerance=1e-12,month_first_sessions=3,month_end_natural_days=5,
        annual_return_target=.10,new_model_fits=0,new_reference_accounts=0,source_budget_cny=0,position_impact=0,
        goal_achieved=False,independent_validation='NOT_ESTABLISHED',
        evidence_class='RETROSPECTIVE_REPLAY_PREVIOUSLY_OBSERVED_HISTORY',
        previous_goal_turn_classification='PROGRESS_ROUND190_EIGHT_ACCOUNTS_JOINT_TARGET_FAILED',
        legacy_meets_point_target_field='SHARPE_ONLY_USE_JOINT_TARGET_ASSESSMENT')
    rules=ROOT/cfg['rules']
    with rules.open('x',encoding='utf-8') as stream:
        stream.write((ROOT/'docs/510300_CONFIRMED_MONTH_EDGE_IDLE_NEXT_20260913.md').read_text(encoding='utf-8'))
        stream.write('\n\n## 来源全部中文因素及进入退出规则\n\n')
        stream.write((ROOT/old['rules']).read_text(encoding='utf-8'))
    paths=[Path(__file__),ROOT/'research/confirmed_month_edge_idle_inputs_v1.py',ROOT/'research/joint_account_acceptance_v1.py',
        ROOT/'research/saved_target_batch_runner_v1.py',ROOT/'research/saved_parent_target_alignment_v1.py',
        ROOT/'research/saved_target_account_checks_v1.py',ROOT/'research/event_clock_account_v1.py',
        ROOT/'research/adaptive_allocation_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',
        ROOT/'tests/test_confirmed_month_edge_idle_v1.py',receipt,OUT/'tests_output.txt',rules,
        ROOT/cfg['features'],ROOT/cfg['dividends'],source_config,ROOT/old['rules'],SOURCE/'saved_verification_receipt.json',
        ROOT/'config/510300_research_authority_v6.json',ROOT/'config/510300_calendar_learned_equal_blend_v1.json',
        CALENDAR_SOURCE/'evaluation_states.parquet',CALENDAR_SOURCE/'earlier_diagnostic_states.parquet']
    for period in ['evaluation','earlier_diagnostic']:
        for cost in cfg['costs']:
            paths += [SOURCE/period/cost/f'{PARENT}_decisions.parquet']
            paths += [folder/period/cost/f'{model}_ledger.parquet' for model,(folder,_) in CONTROLS.items()]
    cfg['frozen_files']=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG,cfg,exclusive=True)
    p=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    i=json.loads(p.read_text(encoding='utf-8'))
    require(i['latest_completed_round']['round']==190,'第191轮前序不同')
    i['running_studies']=[{'round':191,'study':cfg['study_id'],'status':'FROZEN_NOT_STARTED','config':str(CONFIG.relative_to(ROOT))}]
    i['next_work'].update(registered=True,status='CONFIRMED_MONTH_EDGE_IDLE_FROZEN',source=cfg['rules'])
    write_json(p,i)
    print('第191轮已冻结。',flush=True)


def run():
    r=run_saved_target_batch(ROOT,OUT,CONFIG,CANDIDATES,PARENTS,CONTROLS,confirmed_frames)
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    print(json.dumps(save_joint_assessment(OUT,cfg,r),ensure_ascii=False),flush=True)


def verify():
    require(not (OUT/'saved_verification_receipt.json').exists(),'第191轮已核对')
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
        dates=pd.DatetimeIndex(frame.date)
        ordinal=pd.Series(np.arange(len(dates))).groupby(dates.to_period('M'),sort=False).cumcount()+1
        calendar=np.full(len(frame),np.nan)
        confirmation=np.full(len(frame),np.nan)
        target=np.full(len(frame),np.nan)
        for origin,value in zip(origins,parent.reference_weight):
            execution_day=dates[origin+1]
            calendar[origin]=float(ordinal.iloc[origin+1]<=3 or execution_day.days_in_month-execution_day.day<5)
            gross=math.prod((frame.close.iloc[t]+frame.dividend.iloc[t])/frame.close.iloc[t-1] for t in range(origin-19,origin+1))
            require(math.isfinite(gross) and gross>0,'二十日独立复合价格路径无效')
            confirmation[origin]=float(gross>1.+1e-12)
            if value>0:
                target[origin]=value
            elif value==0:
                if calendar[origin]==0 or confirmation[origin]==0:
                    target[origin]=0.
                elif calendar[origin]==1 and confirmation[origin]==1:
                    target[origin]=1.
        folder=OUT/period/cost_id
        factors=pd.read_parquet(folder/'factors.parquet')
        np.testing.assert_allclose(factors.calendar_state,calendar,atol=0,rtol=0,equal_nan=True)
        np.testing.assert_allclose(factors.prior_close_confirmation20,confirmation,atol=0,rtol=0,equal_nan=True)
        np.testing.assert_allclose(factors[PRIMARY+'_target'],target,atol=0,rtol=0,equal_nan=True)
        require(pd.DatetimeIndex(factors.parent_signal_time.iloc[origins]).equals(dates[origins]+pd.Timedelta(hours=15,minutes=5)),'原目标信号时点不同')
        require(pd.DatetimeIndex(factors.decision_time.iloc[origins]).equals(dates[origins+1]+pd.Timedelta(hours=9)),'确认日历最终决定不是执行日九点')
        ledger=pd.read_parquet(folder/f'{PRIMARY}_ledger.parquet')
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
        checks.append({'period':period,'cost':cost_id,'fills':int(filled.sum()),'maximum_target_error':0.})
        return target

    accounts,cycles,diffs,count=verify_saved_target_accounts(OUT,cfg,r,data,dividends,expected,comparison_models=list(CONTROLS))
    require(len(accounts)==4 and count==5646,'第191轮核对范围不同')
    pd.DataFrame(checks).to_csv(OUT/'saved_confirmed_calendar_checks.csv',index=False,encoding='utf-8-sig')
    receipt={'verified_at':now(),'status':'PASS_FOUR_SIMULATED_ACCOUNTS_CONFIRMED_MONTH_EDGE_IDLE_AND_COSTS',
        'actual_accounts':4,'actual_decisions_checked':count,'complete_actual_cycles':len(cycles),
        'simulated_fills_checked':sum(x['fills'] for x in checks),'new_models_or_accounts':0,
        'independent_performance_validation':False,'security_audit_performed':False,'reviewer_source_sha256':digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json',receipt,exclusive=True)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    {'prepare':prepare,'run':run,'verify':verify}[sys.argv[1]]()
