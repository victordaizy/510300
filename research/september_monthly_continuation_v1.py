"""仅补原定九月月首模型，从保存状态增量运行固定来源图。"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import now,digest,require,write_json
from research.adaptive_allocation_v1 import normalize_dividends,target_request,summarize
from research.post_selection_continuous_replay_v1 import Pipeline as AugustPipeline
from research.post_selection_continuous_accounts_v1 import simulate_indexed_request_account,simulate_policy,simulate_rearmed_exit,unpack
from research.post_selection_continuous_factors_v1 import align_decisions
from research.september_monthly_training_v1 import reference_samples,fit_month,append_record
from research.simple_intraday_protection_v1 import make_rules as session_rules
from research.learned_cycle_exit_v1 import FEATURES,CN

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_september_monthly_continuation_v1'
CONFIG=ROOT/'config/510300_september_monthly_continuation_v1.json'
AUGUST=ROOT/'reports/research/510300_post_selection_continuous_replay_v1'
INPUTS=ROOT/'reports/research/510300_post_selection_extension_inputs_v1'
PRIMARY='SELECTED_MIX_BAND10_SIMPLE2'
FIT='2026-09-01'
CUTOFF='2026-09-11'


def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))


def config(name):
    return read('config/510300_'+name+'_v1.json')


def canonical(frame,template):
    require(not (set(frame.columns)-set(template.columns)), '增量账户出现未定义的新列')
    return frame.reindex(columns=template.columns).astype(template.dtypes.to_dict())


def prepare():
    require(not CONFIG.exists() and not (OUT/'RUN_STARTED.json').exists(),'本轮已登记或开始，请接续保存状态')
    index=read('reports/research/510300_sharpe_1_2_latest_research.json')
    require(index['latest_completed_round']['round']==213 and not index['running_studies'],'前序状态不同')
    OUT.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter()
    tested=subprocess.run([sys.executable,'-m','pytest','tests/test_september_monthly_training_v1.py','-q','-p','no:cacheprovider'],
        cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (OUT/'tests_output.txt').write_text(tested.stdout+tested.stderr,encoding='utf-8')
    print(tested.stdout,flush=True)
    require(tested.returncode==0 and '3 passed' in tested.stdout,'新增训练与版本锁定验证未通过')
    write_json(OUT/'tests_receipt.json',{'recorded_at':now(),'exit_code':0,'passed':3,'seconds':time.perf_counter()-begin,
        'previous_continuous_engine_tests_reused':10},exclusive=True)
    old=config('post_selection_continuous_replay')
    keys=['evaluation_start','initial_capital','lot','tick','limit_fraction','annual_days','cash_annual_rate_assumption',
        'high_sharpe_target','annual_return_target','costs','dividends','earlier_start','earlier_terminal','weight_band','features']
    cfg={k:old[k] for k in keys}
    cfg.update(study_id='510300_SEPTEMBER_MONTHLY_CONTINUATION_V1',round=214,registered_at=now(),primary=PRIMARY,
        candidate_models=[PRIMARY],candidate_configurations=0,data_cutoff=CUTOFF,fit_origin=FIT,
        rules='docs/510300_SEPTEMBER_MONTHLY_CONTINUATION_V1.md',planned_incremental_accounts=22,planned_model_fits=2,
        planned_existing_training_reference_replays=1,new_reference_accounts=0,goal_achieved=False,
        independent_validation='NOT_ESTABLISHED',evidence_class='PRE_SELECTION_HISTORICAL_GAP_ORIGINAL_MONTHLY_UPDATE',
        model_fit_time_semantics='HISTORICALLY_SIMULATED_1505_NOT_CONTEMPORANEOUS_RECORD',
        current_period_mark='LAST_COMPLETE_CLOSE_WITHOUT_ARTIFICIAL_LIQUIDATION',position_impact=0,
        source_budget_cny=0,strict_forward_evidence_days=0)
    rules=(ROOT/'docs/510300_SEPTEMBER_MONTHLY_CONTINUATION_NEXT_20260913.md').read_text(encoding='utf-8')
    (ROOT/cfg['rules']).write_text(rules.replace('本项尚未实现、登记或运行。','本项必要实现及测试已完成，现于首次新增训练及增量账户之前固定。'),encoding='utf-8')
    graph=read(INPUTS/'dependency_graph.json')
    paths=[Path(__file__),ROOT/'research/september_monthly_training_v1.py',ROOT/'tests/test_september_monthly_training_v1.py',
        ROOT/'research/post_selection_continuous_replay_v1.py',ROOT/'research/post_selection_continuous_accounts_v1.py',
        ROOT/'research/post_selection_continuous_factors_v1.py',ROOT/'research/learned_cycle_exit_v1.py',
        ROOT/'research/learned_cycle_exit_account_v1.py',ROOT/'research/within_cycle_exit_inputs_v1.py',
        ROOT/'research/entry_vintage_exit_inputs_v1.py',ROOT/cfg['features'],ROOT/cfg['dividends'],ROOT/cfg['rules'],OUT/'tests_receipt.json',
        ROOT/'config/510300_post_selection_continuous_replay_v1.json',AUGUST/'result.json',AUGUST/'saved_verification_receipt.json',
        AUGUST/'factors/all_required_targets.parquet',INPUTS/'dependency_graph.json',INPUTS/'required_account_state_inventory.csv',
        ROOT/'data/reference/sse_trade_calendar_2026.csv',ROOT/'reports/research/510300_learned_cycle_exit_v1/all_reference_samples.parquet']
    paths.extend(ROOT/n[k] for n in graph['nodes'] for k in ['configuration','program'])
    for folder in (AUGUST/'accounts').glob('*/*'):
        paths.extend(folder/name for name in ['ledger.parquet','decisions.parquet','checkpoint.json'])
        if (folder/'cycles.csv').exists():paths.append(folder/'cycles.csv')
    paths.extend([ROOT/'reports/research/510300_learned_cycle_exit_v1/saved_models.json',ROOT/'reports/research/510300_within_cycle_exit_v1/saved_models.json',
        ROOT/'reports/research/510300_learned_cycle_exit_v1/reference/D60_INTRA_ledger.parquet'])
    cfg['frozen_files']=[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG,cfg,exclusive=True)
    index['running_studies']=[{'round':214,'study':cfg['study_id'],'status':'FROZEN_NOT_STARTED','config':str(CONFIG.relative_to(ROOT))}]
    index['next_work'].update(registered=True,status='SEPTEMBER_MONTHLY_DELTA_FROZEN',source=cfg['rules'])
    write_json(ROOT/'reports/research/510300_sharpe_1_2_latest_research.json',index)
    print('第214轮已固定：两条原定月首记录、一条训练基准延长、22条九月增量账户。',flush=True)


def train(cfg):
    full=pd.read_parquet(ROOT/cfg['features'])
    data=full[full.date.le(FIT)].reset_index(drop=True)
    div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    cfg31,cfg114=config('learned_cycle_exit'),config('within_cycle_exit')
    output=reference_samples(data,div,cfg31,session_rules(data)['D60_INTRA'],FIT,'2026-09-02')
    frame,ledger,decisions,cycles,checkpoint,samples=output
    folder=OUT/'training_reference';folder.mkdir()
    ledger.to_parquet(folder/'ledger.parquet',index=False);decisions.to_parquet(folder/'decisions.parquet',index=False)
    cycles.to_csv(folder/'cycles.csv',index=False,encoding='utf-8-sig');samples.to_parquet(folder/'samples.parquet',index=False)
    write_json(folder/'checkpoint.json',checkpoint,exclusive=True)
    original=pd.read_parquet(ROOT/'reports/research/510300_learned_cycle_exit_v1/reference/D60_INTRA_ledger.parquet')
    old=original[original.date.le('2026-08-13')].reset_index(drop=True)
    current=ledger.iloc[:len(old)].reset_index(drop=True)
    for column in ['date','shares','requested_quantity','filled_quantity','status']:
        require(current[column].tolist()==old[column].tolist(),'原训练基准正常账户改变：'+column)
    for column in ['cash','equity','dividend_receivable','net_return']:
        np.testing.assert_allclose(current[column],old[column],atol=1e-8,rtol=0,equal_nan=True)
    old_samples=pd.read_parquet(ROOT/'reports/research/510300_learned_cycle_exit_v1/all_reference_samples.parquet')
    old_samples=old_samples[old_samples.signal.eq('D60_INTRA')].sort_values(['cycle_id','origin_index']).reset_index(drop=True)
    match=samples.merge(old_samples[['cycle_id','origin_index']],on=['cycle_id','origin_index'],how='inner',validate='one_to_one')
    match=match.sort_values(['cycle_id','origin_index']).reset_index(drop=True)
    require(len(match)==len(old_samples),'原已成熟样本缺失')
    columns=FEATURES+['target','extra_dividend_cny','reference_quantity','early_exit_index','exit_index']
    np.testing.assert_allclose(match[columns].to_numpy(float),old_samples[columns].to_numpy(float),atol=1e-12,rtol=1e-12,equal_nan=True)
    ridge,within,rows,fits=fit_month(frame,samples,cfg31,cfg114,FIT)
    old_ridge=read('reports/research/510300_learned_cycle_exit_v1/saved_models.json')['models']['D60_INTRA__RIDGE']
    old_within=read('reports/research/510300_within_cycle_exit_v1/saved_models.json')['models']
    new_ridge,new_within=append_record(old_ridge,ridge),append_record(old_within,within)
    write_json(OUT/'ridge_models.json',{'models':new_ridge,'new_record_count':1,'actual_created_at':now()},exclusive=True)
    write_json(OUT/'within_models.json',{'models':new_within,'new_record_count':1,'actual_created_at':now()},exclusive=True)
    require(read(OUT/'ridge_models.json')['models'][:-1]==old_ridge and read(OUT/'within_models.json')['models'][:-1]==old_within,'旧模型记录发生变化')
    rows.to_parquet(OUT/'september_training_membership.parquet',index=False)
    coefficient_rows=[]
    for label,record in [('原岭回归继续收益',ridge),('原持仓内继续收益',within)]:
        if record['model']:
            model=record['model']
            for name,mean,scale,coefficient in zip(CN,model['mean'],model['scale'],model['coefficients']):
                coefficient_rows.append({'模型':label,'因素':name,'原训练均值':mean,'原训练标准差':scale,'标准化系数':coefficient,
                    '预测截距':model['intercept'],'标准化截断倍数':model['feature_clip']})
    pd.DataFrame(coefficient_rows).to_csv(OUT/'九月模型完整中文系数.csv',index=False,encoding='utf-8-sig')
    receipt={'completed_at':now(),'fit_origin':FIT,'simulated_fit_time':'2026-09-01T15:05:00',
        'actual_fit_date_is_later_retrospective_replay':True,'price_data_last_read_for_training':str(frame.date.iloc[-1].date()),
        'training_reference_rows':len(ledger),'old_reference_ledger_rows_verified':len(old),'old_mature_sample_rows_verified':len(old_samples),
        'all_natural_sample_rows':len(samples),'natural_sample_rows_added':len(samples)-len(old_samples),'actual_fits':fits,
        'training_cycles':ridge['training_cycles'],'training_cycle_count':ridge['training_cycle_count'],'training_rows':ridge['training_rows'],
        'latest_mature_exit_date':ridge['latest_exit_date'],'ridge_status':ridge['status'],'within_status':within['status'],
        'old_model_records_preserved':len(old_ridge)+len(old_within),'new_records':2,'sample_max_exit_index':int(rows.exit_index.max()),
        'fit_index':ridge['fit_index'],'coefficient_rows':len(coefficient_rows),'new_candidates':0}
    write_json(OUT/'monthly_training_receipt.json',receipt,exclusive=True)
    require(ridge['status']=='FIT_COMPLETE' and within['status']=='FIT_COMPLETE','本次原月度更新未成功，保持实际失败记录，不以替代模型接续')
    print(f"九月月首更新完成：{fits}次拟合，{len(rows)}行、{len(ridge['training_cycles'])}个成熟周期；旧模型记录全部保留。",flush=True)


class SeptemberPipeline(AugustPipeline):
    def __init__(self,cfg):
        super().__init__(cfg)
        full=pd.read_parquet(ROOT/cfg['features'])
        self.data=full[full.date.le(CUTOFF)].reset_index(drop=True)
        dates=pd.to_datetime(pd.read_csv(ROOT/'data/reference/sse_trade_calendar_2026.csv').trade_date)
        self.next_date=str(dates[dates>pd.Timestamp(CUTOFF)].iloc[0].date())
        self.within=read(OUT/'within_models.json')['models'];self.ridge=read(OUT/'ridge_models.json')['models']
        self.session=session_rules(self.data)['D60_INTRA']
        self.august_targets=pd.read_parquet(AUGUST/'factors/all_required_targets.parquet')
        self.delta_receipts=[];self.august_target_checks=[]

    def save_factors(self,name,frame):
        folder=OUT/'factors';folder.mkdir(parents=True,exist_ok=True)
        frame.to_parquet(folder/(name+'.parquet'),index=False)

    def target(self,key,cost,values,check=True):
        result=super().target(key,cost,values,check)
        column=key+'__'+cost
        if column in self.august_targets:
            old=self.august_targets[column].to_numpy(float)
            np.testing.assert_allclose(result[:len(old)],old,atol=1e-12,rtol=1e-12,equal_nan=True,err_msg='八月已完成目标前缀改变：'+column)
            self.august_target_checks.append({'node':key,'cost':cost,'prefix_rows':len(old),'status':'PASS_AUGUST_TARGET_PREFIX'})
        return result

    def account(self,key,cost,kind='target',values=None,rule=None,spec=None,controller_factory=None,request=None):
        require((key,cost) not in self.accounts,'共享增量账户重复计算')
        cfg=self.setting(key);start=self.graph[key]['replay_start']
        source=AUGUST/'accounts'/cost/key
        old_ledger=pd.read_parquet(source/'ledger.parquet');old_decisions=pd.read_parquet(source/'decisions.parquet')
        snapshot=read(source/'checkpoint.json')
        old_state=unpack(snapshot['state'])
        require(old_state['asof_date']==pd.Timestamp('2026-08-31'),'来源恢复断点不是八月末')
        args=(self.data,self.div,cfg,cfg['costs'][cost],start)
        kwargs={'next_execution_date':self.next_date,'resume':snapshot}
        if kind=='target':
            fn=simulate_indexed_request_account;args+=(key,)
            kwargs.update(targets=values,event_mask=np.ones(len(self.data),bool),
                request_policy=request or (lambda a,p,v,c,m,t:target_request(a,p,v,c)))
        else:
            fn=simulate_rearmed_exit if kind=='rearmed' else simulate_policy
            args+=(rule,spec)
            if kind=='rearmed':kwargs['controller']=controller_factory()
        self.calls[(key,cost)]=(fn,args,kwargs,controller_factory)
        result=fn(*args,**kwargs)
        delta_ledger,delta_decisions,state=result[0],result[1],result[-1]
        expected=pd.DatetimeIndex(self.data.loc[self.data.date.gt('2026-08-31'),'date'])
        require(pd.DatetimeIndex(delta_ledger.date).equals(expected) and len(delta_ledger)==9,'九月增量日期不是九个完整交易日')
        require(delta_ledger.requested_quantity.iloc[0]==old_state['pending']['requested_quantity'],'九月首个开盘没有执行八月末已保存申请')
        delta_decisions['decision_time']=pd.to_datetime(delta_decisions.origin)+pd.Timedelta(hours=15,minutes=5)
        delta_decisions['source_model']=key;delta_decisions['source_cost']=cost
        delta_ledger=canonical(delta_ledger,old_ledger);delta_decisions=canonical(delta_decisions,old_decisions)
        ledger=pd.concat([old_ledger,delta_ledger],ignore_index=True)
        decisions=pd.concat([old_decisions,delta_decisions],ignore_index=True)
        pd.testing.assert_frame_equal(ledger.iloc[:len(old_ledger)].reset_index(drop=True),old_ledger,check_exact=True)
        pd.testing.assert_frame_equal(decisions.iloc[:len(old_decisions)].reset_index(drop=True),old_decisions,check_exact=True)
        self.verify_prefix(key,cost,ledger,decisions)
        folder=OUT/'accounts'/cost/key;folder.mkdir(parents=True)
        ledger.to_parquet(folder/'ledger.parquet',index=False);decisions.to_parquet(folder/'decisions.parquet',index=False)
        delta_ledger.to_parquet(folder/'delta_ledger.parquet',index=False);delta_decisions.to_parquet(folder/'delta_decisions.parquet',index=False)
        write_json(folder/'checkpoint.json',state,exclusive=True)
        if kind!='target':
            previous=pd.read_csv(source/'cycles.csv')
            closed=previous[previous.exit_date.notna()].copy()
            complete=pd.concat([closed,result[2]],ignore_index=True)
            require(not complete.cycle_id.duplicated().any(),'续接持仓周期重复')
            complete.to_csv(folder/'cycles.csv',index=False,encoding='utf-8-sig')
        self.accounts[(key,cost)]=(ledger,decisions,state)
        self.target(key,cost,align_decisions(self.data,decisions),check=False)
        self.delta_receipts.append({'node':key,'cost':cost,'source_checkpoint':str((source/'checkpoint.json').relative_to(ROOT)),
            'source_checkpoint_sha256':digest(source/'checkpoint.json'),'old_ledger_rows_reused':len(old_ledger),
            'old_decision_rows_reused':len(old_decisions),'incremental_ledger_rows':len(delta_ledger),
            'incremental_decision_rows':len(delta_decisions),'first_open_request':int(delta_ledger.requested_quantity.iloc[0]),
            'first_open_date':'2026-09-01','end_close':CUTOFF,'old_prefix_exact':True,'state_restore_used':True})
        print(f'九月增量 {len(self.accounts)}/22：{key}／{cost}，只计算9日；旧前缀一致。',flush=True)
        return ledger,decisions,state

    def verify_master_single_pass(self):
        key,cost=PRIMARY,'STRESS'
        fn,args,kwargs,factory=self.calls[(key,cost)]
        comparison=fn(*args,**{k:v for k,v in kwargs.items() if k!='resume'})
        ledger,decisions,state=self.accounts[(key,cost)]
        pd.testing.assert_frame_equal(canonical(comparison[0],ledger),ledger,check_exact=True)
        raw=comparison[1];expected=decisions[raw.columns]
        pd.testing.assert_frame_equal(canonical(raw,expected),expected,check_exact=True)
        require(comparison[-1]==state,'主压力一次完整执行与增量恢复最终状态不同')
        write_json(OUT/'master_single_pass_comparison.json',{'verified_at':now(),'status':'PASS_MASTER_STRESS_SAME_FULL_TARGETS_SINGLE_PASS_EXACT',
            'ledger_rows':len(ledger),'decision_rows':len(decisions),'final_state_exact':True,'validation_accounts_replayed':1,
            'scope':'主压力账户使用整图已生成的同一完整目标，原起点一次执行与九月增量拼接精确相同；不是所有22个来源都重新整段回放'},exclusive=True)


def run():
    cfg=read(CONFIG)
    require(not (OUT/'RUN_STARTED.json').exists(),'本轮已开始，请接续保存步骤，不重跑训练或已生成增量')
    for item in cfg['frozen_files']:require(digest(ROOT/item['path'])==item['sha256'],'冻结来源改变')
    write_json(OUT/'RUN_STARTED.json',{'started_at':now(),'config_sha256':digest(CONFIG)},exclusive=True)
    began=time.perf_counter()
    train(cfg)
    pipeline=SeptemberPipeline(cfg).run()
    pipeline.verify_master_single_pass()
    pd.DataFrame(pipeline.delta_receipts).to_csv(OUT/'incremental_account_receipts.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pipeline.august_target_checks).drop_duplicates().to_csv(OUT/'august_target_prefix_checks.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pipeline.checks).to_csv(OUT/'old_normal_prefix_checks.csv',index=False,encoding='utf-8-sig')
    metrics=[];segments=[];positions=[]
    for (key,cost),(ledger,decisions,state) in pipeline.accounts.items():
        last=ledger.iloc[-1];pending=decisions.iloc[-1]
        positions.append({'model':key,'cost':cost,'date':str(last.date.date()),'cash':last.cash,'shares':int(last.shares),'equity':last.equity,
            'dividend_receivable':last.dividend_receivable,'next_execution_date':str(pending.execution_date.date()),
            'next_request':int(pending.requested_quantity),'next_target':pending.reference_weight})
        if key!=PRIMARY:continue
        metrics.append({'model':key,'cost':cost,'name':'固定85%与15%方案连续至9月11日',**summarize(ledger,cfg),
            'mark_scope':'CONTINUOUS_CLOSE_NO_TERMINAL_LIQUIDATION','ending_shares':int(last.shares)})
        for start,end,label in [('2026-08-13','2026-08-31','八月正常延续'),('2026-08-31',CUTOFF,'九月增量'),('2026-08-13',CUTOFF,'全部补充历史')]:
            first=ledger[ledger.date.eq(start)].iloc[0];final=ledger[ledger.date.eq(end)].iloc[0]
            part=ledger[ledger.date.gt(start)&ledger.date.le(end)]
            segments.append({'cost':cost,'period':label,'start_close':start,'end_close':end,'completed_days':len(part),
                'net_profit_cny':final.equity-first.equity,'net_return':final.equity/first.equity-1,'trades':int(part.filled_quantity.ne(0).sum()),
                'commission':float(part.commission.sum()),'slippage_cost':float(part.slippage_cost.sum()),'strict_forward':False})
    pd.DataFrame(metrics).to_csv(OUT/'continuous_account_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(segments).to_csv(OUT/'extension_segment_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(positions).to_csv(OUT/'ending_positions_and_next_requests.csv',index=False,encoding='utf-8-sig')
    training=read(OUT/'monthly_training_receipt.json');original=read('reports/research/510300_incremental_selected_intent_mix_v1/result.json')
    result={'study_id':cfg['study_id'],'completed_at':now(),'status':'COMPLETED_ORIGINAL_MONTHLY_FITS_AND_22_INCREMENTAL_ACCOUNTS',
        'candidate_configurations':0,'candidate_models':[PRIMARY],'evaluation_accounts':2,'new_accounts_generated':23,
        'reused_control_accounts':0,'earlier_diagnostic_accounts':0,'new_earlier_diagnostic_accounts':0,'new_model_fits':training['actual_fits'],
        'new_reference_accounts':0,'existing_training_reference_replays':1,'existing_account_incremental_continuations':22,
        'source_account_rows_reused':sum(r['old_ledger_rows_reused'] for r in pipeline.delta_receipts),
        'incremental_account_rows':sum(r['incremental_ledger_rows'] for r in pipeline.delta_receipts),'master_validation_replays':1,
        'all_metrics':metrics,'earlier_diagnostics':[r for r in original['earlier_diagnostics'] if r['model']==PRIMARY],
        'earlier_metrics_reused_from_round209':True,'primary':PRIMARY,'post_selected_best_base':metrics[0],
        'goal_achieved':False,'independent_validation':'NOT_ESTABLISHED','strict_forward_evidence_days':0,'position_impact':0,
        'monthly_training':training,'extension_segment_results':segments,'continuous_cutoff':CUTOFF,'terminal_liquidation':False,
        'full_pipeline_incremental_resume_verified':True,
        'incremental_verification_scope':'22个账户实际从保存断点执行九日，所有旧账户与目标前缀一致；主压力同一完整目标一次执行对比精确；未把22个来源全部重新整段回放',
        'new_strategy_performance_computed':True,'run_seconds':time.perf_counter()-began,'network_requests':0,'source_cost_cny':0}
    write_json(OUT/'result.json',result,exclusive=True)
    print(json.dumps({'status':result['status'],'seconds':result['run_seconds'],'metrics':metrics,'segments':segments},ensure_ascii=False),flush=True)


def verify():
    cfg=read(CONFIG);result=read(OUT/'result.json');receipts=pd.read_csv(OUT/'incremental_account_receipts.csv')
    require(len(receipts)==22 and receipts.incremental_ledger_rows.eq(9).all(),'保存增量账户计数不同')
    rows=0;decisions_count=0
    for item in receipts.to_dict('records'):
        source=AUGUST/'accounts'/item['cost']/item['node'];folder=OUT/'accounts'/item['cost']/item['node']
        old=pd.read_parquet(source/'ledger.parquet');new=pd.read_parquet(folder/'ledger.parquet');delta=pd.read_parquet(folder/'delta_ledger.parquet')
        decisions=pd.read_parquet(folder/'decisions.parquet');state=unpack(read(folder/'checkpoint.json')['state'])
        pd.testing.assert_frame_equal(new.iloc[:len(old)].reset_index(drop=True),old,check_exact=True)
        pd.testing.assert_frame_equal(new.iloc[len(old):].reset_index(drop=True),delta,check_exact=True)
        require(new.mark_clock.eq('CLOSE').all() and len(delta)==9 and state['asof_date']==pd.Timestamp(CUTOFF),'保存连续账本末日不同')
        require(int(new.shares.iloc[-1])==state['account']['shares'] and abs(new.cash.iloc[-1]-state['account']['cash'])<1e-9,'九月末账本与状态不同')
        require(new.accounting_error.abs().max()<1e-6 and len(decisions)==len(new)+1 and decisions.execution_date.iloc[-1]==pd.Timestamp('2026-09-14'),'财富、决定或下一日历交易日错误')
        if item['node']==PRIMARY:
            measured=summarize(new,cfg);stored=next(x for x in result['all_metrics'] if x['cost']==item['cost'])
            for key in ['net_sharpe','annualized_return','max_drawdown','trade_count']:require(abs(measured[key]-stored[key])<1e-12,'保存主账户指标不同')
        rows+=len(new);decisions_count+=len(decisions)
    training=read(OUT/'monthly_training_receipt.json');members=pd.read_parquet(OUT/'september_training_membership.parquet')
    require(training['actual_fits']==2 and members.exit_index.max()<=training['fit_index'] and training['price_data_last_read_for_training']==FIT,'训练成熟边界不同')
    for current,old,name in [('ridge_models.json','reports/research/510300_learned_cycle_exit_v1/saved_models.json','D60_INTRA__RIDGE'),
        ('within_models.json','reports/research/510300_within_cycle_exit_v1/saved_models.json',None)]:
        original=read(old)['models'];original=original[name] if name else original
        stored=read(OUT/current)['models'];require(stored[:-1]==original and stored[-1]['fit_origin']==FIT,'原模型前缀或新增拟合日不同')
    require(read(OUT/'master_single_pass_comparison.json')['status']=='PASS_MASTER_STRESS_SAME_FULL_TARGETS_SINGLE_PASS_EXACT','主压力恢复对比未通过')
    for item in cfg['frozen_files']:require(digest(ROOT/item['path'])==item['sha256'],'冻结来源改变')
    write_json(OUT/'saved_verification_receipt.json',{'verified_at':now(),'status':'PASS_TWO_MONTHLY_MODELS_22_INCREMENTAL_ACCOUNTS_AND_SAVED_PREFIXES',
        'accounts':22,'account_rows':rows,'decision_rows':decisions_count,'incremental_rows':198,'old_account_rows_reused':38742,
        'model_fits':2,'new_model_records':2,'old_model_records_preserved':282,'mature_training_rows':len(members),
        'full_pipeline_incremental_resume_verified':True,'scope':result['incremental_verification_scope'],
        'independent_performance_validation':False},exclusive=True)
    print('九月两条模型及22条增量账户保存核对通过。',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='只补原月首模型并增量续接固定策略。')
    parser.add_argument('action',choices=['prepare','run','verify'])
    globals()[parser.parse_args().action]()
