"""固定已通过历史门槛的三个候选，只读账本并诊断年份和连续区块敏感性。"""
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import block_indices, digest, now, require, return_metrics, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/research/510300_point_pass_fixed_diagnostic_v1'
CONFIG = ROOT/'config/510300_point_pass_fixed_diagnostic_v1.json'
SOURCE = ROOT/'reports/research/510300_incremental_selected_intent_mix_v1'
MODELS = ['SELECTED_MIX_BAND00_SIMPLE3','SELECTED_MIX_BAND10_SIMPLE2','SELECTED_MIX_BAND00_FULL']
NAMES = dict(zip(MODELS,['事前主方案：零门槛70／15／15','简单方案：十个百分点85／15','数值首位：零门槛五来源']))
PERIODS = ['evaluation','earlier_diagnostic']
COSTS = ['BASE','STRESS']
BLOCKS = [5,20,60]


def batch_indices(rng, count, size, block):
    require(count>0 and size>1 and block>0,'区块抽样尺寸无效')
    starts = rng.integers(0,size,size=(count,math.ceil(size/block)))
    return ((starts[:,:,None]+np.arange(block))%size).reshape(count,-1)[:,:size]


def sampled_metrics(values, annual_days=242):
    require(values.ndim==2 and values.shape[1]>1 and np.isfinite(values).all() and (values>-1).all(),'抽样收益无效')
    average=values.mean(axis=1)
    deviation=values.std(axis=1,ddof=1)
    sharpe=np.divide(average*np.sqrt(annual_days),deviation,out=np.full(len(values),np.nan),where=deviation>1e-15)
    annual=np.expm1(np.log1p(values).sum(axis=1)*annual_days/values.shape[1])
    return np.column_stack([sharpe,annual])


def concentration(profits,total):
    require(np.isfinite(profits).all() and np.isfinite(total) and total>0,'周期利润或分母无效')
    positive=np.sort(profits[profits>0])[::-1]
    return {'complete_cycles':len(profits),'winning_cycles':int((profits>0).sum()),'losing_cycles':int((profits<0).sum()),
        'total_net_profit':float(total),'largest_cycle_net_profit':float(positive[0]) if len(positive) else 0.,
        'top_five_winning_profit':float(positive[:5].sum()),
        'largest_cycle_share_of_net_profit':float(positive[0]/total) if len(positive) else 0.,
        'top_five_share_of_net_profit':float(positive[:5].sum()/total)}


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(),'本项已登记或启动')
    index_path=ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index=json.loads(index_path.read_text(encoding='utf-8'))
    require(index['latest_completed_round']['round']==209 and not index['running_studies'],'前序状态不同')
    OUT.mkdir(parents=True,exist_ok=True)
    began=time.perf_counter()
    tested=subprocess.run([sys.executable,'-m','pytest','tests/test_point_pass_fixed_diagnostic_v1.py','-q','-p','no:cacheprovider'],
        cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (OUT/'tests_output.txt').write_text(tested.stdout+tested.stderr,encoding='utf-8')
    print(tested.stdout,flush=True)
    require(tested.returncode==0 and '3 passed' in tested.stdout,'三项统计计算测试未通过')
    write_json(OUT/'tests_receipt.json',{'recorded_at':now(),'exit_code':tested.returncode,'passed':3,'seconds':time.perf_counter()-began},exclusive=True)
    old_path=ROOT/'config/510300_incremental_selected_intent_mix_v1.json'
    old=json.loads(old_path.read_text(encoding='utf-8'))
    keys=['evaluation_start','data_cutoff','initial_capital','lot','tick','limit_fraction','annual_days','cash_annual_rate_assumption',
        'high_sharpe_target','annual_return_target','costs','features','dividends','earlier_start','earlier_terminal']
    cfg={k:old[k] for k in keys}
    cfg.update(study_id='510300_POINT_PASS_FIXED_DIAGNOSTIC_V1',round=210,registered_at=now(),primary=MODELS[0],
        candidate_models=MODELS,candidate_configurations=0,fixed_existing_candidates=3,new_model_fits=0,new_reference_accounts=0,
        rules='docs/510300_POINT_PASS_FIXED_DIAGNOSTIC_V1.md',source_round=209,source_result=str((SOURCE/'result.json').relative_to(ROOT)),
        block_lengths=BLOCKS,replications=2000,random_seed=209,quantiles=[.025,.5,.975],new_trading_accounts=0,
        independent_validation='NOT_ESTABLISHED',goal_achieved=False,position_impact=0,
        evidence_class='POST_SELECTION_FIXED_CANDIDATE_SAVED_STABILITY_DIAGNOSTIC',
        prior_goal_turn_classification='PROGRESS_ROUNDS207_208_209_COMPLETED_88_ACCOUNTS_AND_EIGHT_HISTORICAL_JOINT_PASSES')
    text=(ROOT/'docs/510300_POINT_PASS_FIXED_DIAGNOSTIC_NEXT_20260913.md').read_text(encoding='utf-8')
    text=text.replace('本项目前只准备规则，尚未实现或运行。','本项在三项必要统计计算测试通过后、首次读取固定诊断结果前冻结。')
    with (ROOT/cfg['rules']).open('x',encoding='utf-8') as stream:
        stream.write(text)
    paths=[Path(__file__),ROOT/'tests/test_point_pass_fixed_diagnostic_v1.py',ROOT/'research/intraday_overnight_increment_v1.py',
        old_path,ROOT/old['rules'],ROOT/cfg['rules'],OUT/'tests_output.txt',OUT/'tests_receipt.json',SOURCE/'result.json',
        SOURCE/'saved_verification_receipt.json',SOURCE/'yearly_metrics.csv',SOURCE/'saved_actual_cycles.csv']
    paths += [SOURCE/p/c/f'{m}_ledger.parquet' for p in PERIODS for c in COSTS for m in MODELS]
    cfg['frozen_files']=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG,cfg,exclusive=True)
    index['running_studies']=[{'round':210,'study':cfg['study_id'],'status':'FROZEN_DIAGNOSTIC_NOT_STARTED','config':str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True,status='POINT_PASS_FIXED_DIAGNOSTIC_FROZEN',source=cfg['rules'])
    write_json(index_path,index)
    print('第210轮诊断已冻结：三个固定候选、零新账户、三种区块各两千次。',flush=True)


def read_config():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    for item in cfg['frozen_files']:
        require(digest(ROOT/item['path'])==item['sha256'],'冻结来源改变')
    return cfg


def run():
    require(not (OUT/'RUN_STARTED.json').exists(),'诊断已经启动，不重复运行')
    cfg=read_config()
    write_json(OUT/'RUN_STARTED.json',{'started_at':now(),'config_sha256':digest(CONFIG),'new_accounts':0},exclusive=True)
    began=time.perf_counter()
    old=json.loads((SOURCE/'result.json').read_text(encoding='utf-8'))
    years=pd.read_csv(SOURCE/'yearly_metrics.csv')
    cycles=pd.read_csv(SOURCE/'saved_actual_cycles.csv')
    accounts,annual,leave_year,concentrated,matrices,dates=[],[],[],[],{},{}
    for period in PERIODS:
        streams=[]
        for model in MODELS:
            for cost in COSTS:
                ledger=pd.read_parquet(SOURCE/period/cost/f'{model}_ledger.parquet')
                values=ledger.net_return.to_numpy(float)
                np.testing.assert_allclose(ledger.equity/np.r_[cfg['initial_capital'],ledger.equity.iloc[:-1]]-1,values,atol=1e-12,rtol=0)
                require(ledger.shares.iloc[-1]==0,'来源终点未清算')
                if period in dates:
                    require(pd.DatetimeIndex(ledger.date).equals(dates[period]),'固定账户日历不同')
                dates[period]=pd.DatetimeIndex(ledger.date)
                measures=return_metrics(values,cfg['annual_days'])
                stored=next(r for r in old['all_metrics' if period=='evaluation' else 'earlier_diagnostics'] if r['model']==model and r['cost']==cost)
                for key,value in measures.items():
                    require(abs(value-stored[key])<1e-9,'完整账户指标不同')
                accounts.append({'period':period,'model':model,'cost':cost,'name':NAMES[model],**measures,'trading_days':len(values),
                    'trade_count':stored['trade_count'],'commission':stored['commission'],'slippage_cost':stored['slippage_cost'],
                    'historical_joint_point_pass':measures['net_sharpe']>=1.2 and measures['annualized_return']>=.1})
                for year in sorted(ledger.date.dt.year.unique()):
                    mask=ledger.date.dt.year.eq(year).to_numpy()
                    measured=return_metrics(values[mask],cfg['annual_days'])
                    row=years[(years.period==period)&(years.model==model)&(years.cost==cost)&(years.year==year)].iloc[0]
                    for key,value in measured.items():
                        require(abs(value-row[key])<1e-9,'逐年指标不同')
                    annual.append({'period':period,'model':model,'cost':cost,'year':int(year),**measured,
                        'trading_days':int(mask.sum()),'trade_count':int(row.trade_count),'partial_year':bool(year==2026)})
                    remaining=return_metrics(values[~mask],cfg['annual_days'])
                    leave_year.append({'period':period,'model':model,'cost':cost,'excluded_year':int(year),**remaining,
                        'remaining_days':int((~mask).sum()),'diagnostic_only_not_executable':True,
                        'remaining_joint_point_pass':remaining['net_sharpe']>=1.2 and remaining['annualized_return']>=.1})
                own=cycles[(cycles.period==period)&(cycles.model==model)&(cycles.cost==cost)]
                profits=own.net_profit.to_numpy(float)
                total=float(ledger.equity.iloc[-1]-cfg['initial_capital'])
                require(abs(profits.sum()-total)<1e-6,'周期利润不能合计到完整净值')
                concentrated.append({'period':period,'model':model,'cost':cost,**concentration(profits,total)})
                streams.append(values)
        matrices[period]=np.column_stack(streams)
    for name,rows in [('account_metrics.csv',accounts),('yearly_metrics.csv',annual),('leave_one_year_statistics.csv',leave_year),('cycle_concentration.csv',concentrated)]:
        pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    rng=np.random.default_rng(cfg['random_seed'])
    summary,paired,stored_samples=[],[],{}
    for period in PERIODS:
        matrix=matrices[period]
        for block in BLOCKS:
            samples=np.empty((cfg['replications'],len(MODELS)*len(COSTS),2))
            hasher=hashlib.sha256()
            for left in range(0,cfg['replications'],200):
                count=min(200,cfg['replications']-left)
                indices=batch_indices(rng,count,len(matrix),block)
                hasher.update(indices.astype('<i4').tobytes())
                for column in range(matrix.shape[1]):
                    samples[left:left+count,column]=sampled_metrics(matrix[:,column][indices],cfg['annual_days'])
            stored_samples[f'{period}_block{block}']=samples
            for number,model in enumerate(MODELS):
                joint_mask=np.ones(cfg['replications'],dtype=bool)
                for offset,cost in enumerate(COSTS):
                    values=samples[:,number*2+offset]
                    joint_mask &= np.isfinite(values).all(axis=1)&(values[:,0]>=1.2)&(values[:,1]>=.1)
                    q=np.nanquantile(values,cfg['quantiles'],axis=0)
                    summary.append({'period':period,'model':model,'cost':cost,'block_length':block,'replications':len(values),
                        'sharpe_lower_2_5':q[0,0],'sharpe_median':q[1,0],'sharpe_upper_97_5':q[2,0],
                        'annual_lower_2_5':q[0,1],'annual_median':q[1,1],'annual_upper_97_5':q[2,1],
                        'defined_sharpe_count':int(np.isfinite(values[:,0]).sum()),'indices_sha256':hasher.hexdigest()})
                paired.append({'period':period,'model':model,'block_length':block,'replications':cfg['replications'],
                    'two_cost_joint_threshold_fraction':float(joint_mask.mean()),'is_future_success_probability':False})
            print(f"{period}、{block}日区块：两千次、三个固定候选和两档费用已完成。",flush=True)
    np.savez_compressed(OUT/'bootstrap_metrics.npz',**stored_samples)
    pd.DataFrame(summary).to_csv(OUT/'bootstrap_intervals.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(paired).to_csv(OUT/'bootstrap_two_cost_threshold_fractions.csv',index=False,encoding='utf-8-sig')
    result={'study_id':cfg['study_id'],'completed_at':now(),'status':'COMPLETED_FIXED_SAVED_DIAGNOSTIC_INDEPENDENT_NOT_ESTABLISHED',
        'candidate_configurations':0,'candidate_models':MODELS,'fixed_existing_candidates':3,
        'evaluation_accounts':6,'new_accounts_generated':0,'reused_control_accounts':6,
        'earlier_diagnostic_accounts':6,'new_earlier_diagnostic_accounts':0,'new_model_fits':0,'new_reference_accounts':0,
        'all_metrics':[r for r in accounts if r['period']=='evaluation'],'earlier_diagnostics':[r for r in accounts if r['period']=='earlier_diagnostic'],
        'primary':MODELS[0],'post_selected_best_base':next(r for r in accounts if r['period']=='evaluation' and r['model']==MODELS[0] and r['cost']=='BASE'),
        'historical_point_target_met':all(r['historical_joint_point_pass'] for r in accounts),'goal_achieved':False,
        'independent_validation':'NOT_ESTABLISHED','position_impact':0,'run_seconds':time.perf_counter()-began,
        'bootstrap_metric_rows':cfg['replications']*3*2*3*2,'bootstrap_paired_fraction_rows':len(paired),
        'leave_year_rows':len(leave_year),'yearly_rows':len(annual),'cycle_concentration_rows':len(concentrated),
        'limitation':'固定候选重采样未校正策略和权重选优，不能当作独立证据；比例不是未来成功概率，留出年份的拼接统计不是新交易账户。'}
    write_json(OUT/'result.json',result,exclusive=True)
    print(json.dumps({'状态':result['status'],'耗时':result['run_seconds'],'新账户':0,'固定账户':len(accounts),
        '重采样指标行数':result['bootstrap_metric_rows'],'逐年留出行数':len(leave_year)},ensure_ascii=False),flush=True)


def verify():
    require(not (OUT/'saved_verification_receipt.json').exists(),'诊断核对已完成')
    cfg=read_config()
    result=json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    intervals=pd.read_csv(OUT/'bootstrap_intervals.csv')
    pairs=pd.read_csv(OUT/'bootstrap_two_cost_threshold_fractions.csv')
    rng=np.random.default_rng(cfg['random_seed'])
    checked=0
    with np.load(OUT/'bootstrap_metrics.npz') as saved:
        for period in PERIODS:
            matrix=np.column_stack([pd.read_parquet(SOURCE/period/c/f'{m}_ledger.parquet').net_return for m in MODELS for c in COSTS])
            for block in BLOCKS:
                indices=np.vstack([block_indices(rng,len(matrix),block) for _ in range(cfg['replications'])])
                hashed=hashlib.sha256(indices.astype('<i4').tobytes()).hexdigest()
                observed=saved[f'{period}_block{block}']
                for col in range(matrix.shape[1]):
                    v=matrix[:,col][indices]
                    sum_return=v.sum(axis=1)
                    variance=((v*v).sum(axis=1)-sum_return*sum_return/v.shape[1])/(v.shape[1]-1)
                    sd=np.sqrt(np.maximum(variance,0))
                    sharpe=np.divide(sum_return*np.sqrt(cfg['annual_days'])/v.shape[1],sd,out=np.full(len(v),np.nan),where=sd>1e-15)
                    annual=np.exp(np.log1p(v).sum(axis=1)*cfg['annual_days']/v.shape[1])-1
                    np.testing.assert_allclose(observed[:,col,0],sharpe,atol=1e-10,rtol=0,equal_nan=True)
                    np.testing.assert_allclose(observed[:,col,1],annual,atol=1e-12,rtol=0)
                    model,cost=MODELS[col//2],COSTS[col%2]
                    row=intervals[(intervals.period==period)&(intervals.model==model)&(intervals.cost==cost)&(intervals.block_length==block)].iloc[0]
                    require(row.indices_sha256==hashed,'两个费用或候选抽样索引不同')
                    quantiles=np.nanquantile(observed[:,col],cfg['quantiles'],axis=0)
                    np.testing.assert_allclose([row.sharpe_lower_2_5,row.sharpe_median,row.sharpe_upper_97_5],quantiles[:,0],atol=1e-12,rtol=0)
                    np.testing.assert_allclose([row.annual_lower_2_5,row.annual_median,row.annual_upper_97_5],quantiles[:,1],atol=1e-12,rtol=0)
                    checked+=len(v)
                for j,model in enumerate(MODELS):
                    values=observed[:,j*2:j*2+2]
                    passed=np.isfinite(values).all(axis=(1,2))&(values[:,:,0]>=1.2).all(axis=1)&(values[:,:,1]>=.1).all(axis=1)
                    row=pairs[(pairs.period==period)&(pairs.model==model)&(pairs.block_length==block)].iloc[0]
                    require(abs(row.two_cost_joint_threshold_fraction-passed.mean())<1e-12,'两费用同时过线比例不同')
    require(checked==result['bootstrap_metric_rows']==72000,'抽样核对范围不同')
    receipt={'verified_at':now(),'status':'PASS_TWELVE_SAVED_ACCOUNTS_AND_72000_BOOTSTRAP_METRIC_ROWS',
        'saved_accounts':12,'bootstrap_metric_rows_checked':checked,'interval_rows_checked':len(intervals),
        'paired_fraction_rows_checked':len(pairs),'new_accounts':0,'new_models':0,'independent_performance_validation':False,
        'reviewer_source_sha256':digest(Path(__file__))}
    write_json(OUT/'saved_verification_receipt.json',receipt,exclusive=True)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':
    {'prepare':prepare,'run':run,'verify':verify}[sys.argv[1]]()
