"""第182轮正目标区间固定倍率的完整账户研究。"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends
from research.episode_account_risk_budget_inputs_v1 import CANDIDATES, MODELS, PARENT, PRIMARY, RISK_SOURCE, episode_frames
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.joint_account_acceptance_v1 import save_joint_assessment
from research.saved_target_account_checks_v1 import verify_saved_target_accounts
from research.saved_target_batch_runner_v1 import run_saved_target_batch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_episode_account_risk_budget_v1'
CONFIG = ROOT/'config/510300_episode_account_risk_budget_v1.json'
SOURCE = ROOT/'reports/research/510300_return_confirmation_auxiliary_batch_v1'
PARENTS = {PARENT: SOURCE}
CONTROLS = {PARENT: (SOURCE, '第174轮原预算'),
            'ACCOUNT_VOLATILITY_EXPOSURE': (RISK_SOURCE, '第181轮每日风险预算'),
            'EXPOSURE_EXPANSION_200': (ROOT/'reports/research/510300_unlevered_exposure_expansion_v1', '第180轮固定两倍预算'),
            'BUY_HOLD': (ROOT/'reports/research/510300_rearmed_session_exit_v1', '买入持有')}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(), '第182轮已冻结或已启动，请接续')
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = OUT/'tests_receipt.json'
    require(not receipt.exists(), '必要测试已有回执')
    began = time.perf_counter()
    tested = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_episode_account_risk_budget_v1.py',
                             '-q', '-p', 'no:cacheprovider'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    (OUT/'tests_output.txt').write_text(tested.stdout+tested.stderr, encoding='utf-8')
    print(tested.stdout, flush=True)
    require(tested.returncode==0 and '4 passed' in tested.stdout, '第182轮必要测试未通过')
    write_json(receipt, {'recorded_at': now(), 'passed': 4, 'exit_code': tested.returncode,
                         'seconds': time.perf_counter()-began, 'timing_scope': '完整测试进程墙钟'}, exclusive=True)
    prior_config = ROOT/'config/510300_account_volatility_exposure_v1.json'
    prior = json.loads(prior_config.read_text(encoding='utf-8'))
    cfg = {k:v for k,v in prior.items() if k!='frozen_files'}
    cfg.update(study_id='510300_EPISODE_ACCOUNT_RISK_BUDGET_V1', round=182, primary=PRIMARY,
               candidate_models=list(CANDIDATES), parent_models=MODELS, registered_at=now(),
               rules='docs/510300_EPISODE_ACCOUNT_RISK_BUDGET_V1.md',
               budget_clock='FIRST_POSITIVE_ORIGIN_UNTIL_KNOWN_ZERO',
               previous_goal_turn_classification='PROGRESS_ROUNDS180_181_TWELVE_ACCOUNTS_JOINT_TARGET_FAILED')
    rules = ROOT/cfg['rules']
    original_cfg = ROOT/'config/510300_return_confirmation_auxiliary_batch_v1.json'
    original = json.loads(original_cfg.read_text(encoding='utf-8'))
    with rules.open('x', encoding='utf-8') as stream:
        stream.write((ROOT/'docs/510300_EPISODE_ACCOUNT_RISK_BUDGET_NEXT_20260913.md').read_text(encoding='utf-8'))
        stream.write('\n\n## 来源全部中文因素及进入退出规则\n\n')
        stream.write((ROOT/original['rules']).read_text(encoding='utf-8'))
    files = [Path(__file__), ROOT/'research/episode_account_risk_budget_inputs_v1.py',
             ROOT/'research/joint_account_acceptance_v1.py', ROOT/'research/saved_target_batch_runner_v1.py',
             ROOT/'research/saved_parent_target_alignment_v1.py', ROOT/'research/saved_target_account_checks_v1.py',
             ROOT/'research/event_clock_account_v1.py', ROOT/'research/adaptive_allocation_v1.py',
             ROOT/'research/intraday_overnight_increment_v1.py', ROOT/'tests/test_episode_account_risk_budget_v1.py',
             receipt, OUT/'tests_output.txt', rules, ROOT/cfg['features'], ROOT/cfg['dividends'],
             ROOT/'config/510300_research_authority_v6.json', prior_config, original_cfg, ROOT/original['rules'],
             RISK_SOURCE/'saved_verification_receipt.json', SOURCE/'saved_verification_receipt.json']
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost in cfg['costs']:
            files += [RISK_SOURCE/period/cost/'factors.parquet', SOURCE/period/cost/f'{PARENT}_decisions.parquet']
            files += [folder/period/cost/f'{model}_ledger.parquet' for model,(folder,_) in CONTROLS.items()]
    cfg['frozen_files'] = [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in sorted(set(files))]
    write_json(CONFIG, cfg, exclusive=True)
    ip = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(ip.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round']==181, '前序轮次不同')
    index['running_studies'] = [{'round':182,'study':cfg['study_id'],'status':'FROZEN_NOT_STARTED','config':str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True, status='EPISODE_ACCOUNT_RISK_BUDGET_FROZEN', source=cfg['rules'])
    write_json(ip,index)
    print('第182轮单一设置已冻结。',flush=True)


def run():
    result = run_saved_target_batch(ROOT,OUT,CONFIG,CANDIDATES,PARENTS,CONTROLS,episode_frames)
    cfg = json.loads(CONFIG.read_text(encoding='utf-8'))
    print(json.dumps(save_joint_assessment(OUT,cfg,result),ensure_ascii=False),flush=True)


def verify():
    require(not (OUT/'saved_verification_receipt.json').exists(),'第182轮已经核对')
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    result=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    for item in cfg['frozen_files']:
        require(digest(ROOT/item['path'])==item['sha256'],'冻结来源发生改变')
    data=pd.read_parquet(ROOT/cfg['features'])
    div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    checks=[]

    def expected(period,cost_id,frame,start):
        first=int(np.flatnonzero(frame.date.ge(start))[0])
        origins=np.arange(first-1,len(frame)-1)
        parent=pd.read_parquet(SOURCE/period/cost_id/f'{PARENT}_decisions.parquet')
        old=pd.read_parquet(SOURCE/period/cost_id/f'{PARENT}_ledger.parquet')
        np.testing.assert_array_equal(parent.origin_index,origins)
        require(pd.DatetimeIndex(parent.origin).equals(pd.DatetimeIndex(frame.date.iloc[origins])),'来源收盘错位')
        require(pd.DatetimeIndex(old.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])),'来源收益错位')
        returns=np.r_[np.full(first,np.nan),old.net_return.to_numpy(float)]
        current=np.full(len(frame),np.nan)
        for t in range(len(frame)):
            if t<first+59:
                current[t]=1.
            else:
                values=returns[t-59:t+1]
                if np.isfinite(values).all():
                    avg=math.fsum(values)/60
                    vol=math.sqrt(math.fsum((v-avg)**2 for v in values)/59*cfg['annual_days'])
                    current[t]=.10/vol if vol>0 else 1.
        source=np.full(len(frame),np.nan)
        source[origins]=parent.reference_weight.to_numpy(float)
        groups=np.cumsum(source==0)
        target=np.full(len(frame),np.nan)
        target[source==0]=0.
        fixed=np.full(len(frame),np.nan)
        starts=np.zeros(len(frame),bool)
        for group in np.unique(groups):
            positives=np.flatnonzero((groups==group)&(source>0))
            if len(positives)==0:
                continue
            begin=positives[0]
            starts[begin]=True
            fixed[(groups==group)&(np.arange(len(frame))>=begin)]=current[begin]
            if np.isfinite(current[begin]):
                target[positives]=np.minimum(1.,source[positives]*current[begin])
        folder=OUT/period/cost_id
        f=pd.read_parquet(folder/'factors.parquet')
        np.testing.assert_allclose(f.current_account_risk_multiplier,current,atol=1e-10,rtol=1e-12,equal_nan=True)
        np.testing.assert_allclose(f.fixed_episode_multiplier,fixed,atol=1e-10,rtol=1e-12,equal_nan=True)
        np.testing.assert_array_equal(f.positive_episode_start,starts)
        np.testing.assert_allclose(f[PRIMARY+'_target'],target,atol=1e-12,rtol=1e-12,equal_nan=True)
        ledger=pd.read_parquet(folder/f'{PRIMARY}_ledger.parquet')
        require(ledger.cash.ge(-1e-7).all() and ledger.shares.ge(0).all(),'出现融资或负份额')
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
        checks.append({'period':period,'cost':cost_id,'positive_signal_episodes':int(starts.sum()),
                       'fills':int(filled.sum()),'maximum_target_error':float(np.nanmax(abs(f[PRIMARY+'_target'].to_numpy()-target)))})
        return target

    accounts,cycles,differences,count=verify_saved_target_accounts(OUT,cfg,result,data,div,expected,comparison_models=list(CONTROLS))
    require(len(accounts)==4 and count==5646,'核对账户数量不同')
    pd.DataFrame(checks).to_csv(OUT/'saved_episode_budget_checks.csv',index=False,encoding='utf-8-sig')
    receipt={'verified_at':now(),'status':'PASS_FOUR_SIMULATED_ACCOUNTS_FIXED_EPISODE_RISK_AND_COSTS',
             'actual_accounts':4,'actual_decisions_checked':count,'complete_actual_cycles':len(cycles),
             'simulated_fills_checked':sum(x['fills'] for x in checks),'new_models_or_accounts':0,
             'independent_performance_validation':False,'security_audit_performed':False,
             'reviewer_source_sha256':digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json',receipt,exclusive=True)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    {'prepare':prepare,'run':run,'verify':verify}[sys.argv[1]]()
