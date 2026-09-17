"""修正准备期缓存的比较范围，复用已拟合模型与已完成增量继续剩余来源。"""
from pathlib import Path
import numpy as np
import pandas as pd

from research.september_monthly_continuation_v1 import ROOT,OUT,CONFIG,AUGUST,PRIMARY,CUTOFF,SeptemberPipeline,read
from research.post_selection_continuous_replay_v1 import Pipeline as AugustPipeline
from research.post_selection_continuous_factors_v1 import align_decisions
from research.post_selection_continuous_accounts_v1 import simulate_indexed_request_account,simulate_policy,simulate_rearmed_exit,unpack
from research.adaptive_allocation_v1 import target_request,summarize
from research.intraday_overnight_increment_v1 import now,digest,require,write_json
import json
import time


class SavedSeptemberPipeline(SeptemberPipeline):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.cached_accounts=0

    def target(self,key,cost,values,check=True):
        result=AugustPipeline.target(self,key,cost,values,check)
        column=key+'__'+cost
        if column in self.august_targets:
            old=self.august_targets[column].to_numpy(float)
            start=self.graph[key]['replay_start']
            first=int(np.flatnonzero(self.data.date.ge(start))[0])
            np.testing.assert_allclose(result[first-1:len(old)],old[first-1:],atol=1e-12,rtol=1e-12,equal_nan=True,
                err_msg='原有效决定区间目标前缀改变：'+column)
            self.august_target_checks.append({'node':key,'cost':cost,'prefix_rows':len(old)-first+1,
                'first_decision_date':str(self.data.date.iloc[first-1].date()),'unused_preparation_rows_excluded':first-1,
                'status':'PASS_AUGUST_VALID_DECISION_TARGET_PREFIX'})
        return result

    def save_factors(self,name,frame):
        path=OUT/'factors'/(name+'.parquet')
        if path.exists():
            saved=pd.read_parquet(path)
            pd.testing.assert_frame_equal(frame,saved,check_exact=True)
            return
        super().save_factors(name,frame)

    def account(self,key,cost,kind='target',values=None,rule=None,spec=None,controller_factory=None,request=None):
        folder=OUT/'accounts'/cost/key
        if not (folder/'checkpoint.json').exists():
            return super().account(key,cost,kind,values,rule,spec,controller_factory,request)
        require((key,cost) not in self.accounts,'已保存增量来源被重复接纳')
        for filename in ['ledger.parquet','decisions.parquet','delta_ledger.parquet','delta_decisions.parquet']:
            require((folder/filename).is_file(),'已保存增量账户不完整：'+key)
        cfg=self.setting(key);start=self.graph[key]['replay_start'];source=AUGUST/'accounts'/cost/key
        snapshot=read(source/'checkpoint.json');old_state=unpack(snapshot['state'])
        args=(self.data,self.div,cfg,cfg['costs'][cost],start)
        kwargs={'next_execution_date':self.next_date,'resume':snapshot}
        if kind=='target':
            fn=simulate_indexed_request_account;args+=(key,)
            kwargs.update(targets=values,event_mask=np.ones(len(self.data),bool),request_policy=request or (lambda a,p,v,c,m,t:target_request(a,p,v,c)))
        else:
            fn=simulate_rearmed_exit if kind=='rearmed' else simulate_policy;args+=(rule,spec)
            if kind=='rearmed':kwargs['controller']=controller_factory()
        self.calls[(key,cost)]=(fn,args,kwargs,controller_factory)
        old_ledger=pd.read_parquet(source/'ledger.parquet');old_decisions=pd.read_parquet(source/'decisions.parquet')
        ledger=pd.read_parquet(folder/'ledger.parquet');decisions=pd.read_parquet(folder/'decisions.parquet')
        delta=pd.read_parquet(folder/'delta_ledger.parquet');delta_decisions=pd.read_parquet(folder/'delta_decisions.parquet');state=read(folder/'checkpoint.json')
        pd.testing.assert_frame_equal(ledger.iloc[:len(old_ledger)].reset_index(drop=True),old_ledger,check_exact=True)
        pd.testing.assert_frame_equal(decisions.iloc[:len(old_decisions)].reset_index(drop=True),old_decisions,check_exact=True)
        pd.testing.assert_frame_equal(ledger.iloc[len(old_ledger):].reset_index(drop=True),delta,check_exact=True)
        require(len(delta)==9 and old_state['asof_date']==pd.Timestamp('2026-08-31') and delta.requested_quantity.iloc[0]==old_state['pending']['requested_quantity'],'保存增量断点或首个申请不同')
        if kind=='target':
            np.testing.assert_allclose(decisions.reference_weight,np.asarray(values)[decisions.origin_index],atol=1e-12,rtol=1e-12,equal_nan=True)
        self.verify_prefix(key,cost,ledger,decisions)
        self.accounts[(key,cost)]=(ledger,decisions,state)
        self.target(key,cost,align_decisions(self.data,decisions),check=False)
        self.delta_receipts.append({'node':key,'cost':cost,'source_checkpoint':str((source/'checkpoint.json').relative_to(ROOT)),
            'source_checkpoint_sha256':digest(source/'checkpoint.json'),'old_ledger_rows_reused':len(old_ledger),
            'old_decision_rows_reused':len(old_decisions),'incremental_ledger_rows':len(delta),'incremental_decision_rows':len(delta_decisions),
            'first_open_request':int(delta.requested_quantity.iloc[0]),'first_open_date':'2026-09-01','end_close':CUTOFF,
            'old_prefix_exact':True,'state_restore_used':True})
        self.cached_accounts+=1
        print(f'复用已保存九月增量 {self.cached_accounts}/5：{key}／{cost}，未重新模拟。',flush=True)
        return ledger,decisions,state


def main():
    require(not (OUT/'result.json').exists() and not (OUT/'valid_origin_comparison_continuation.json').exists(),'当前保存接续已开始或完成')
    cfg=read(CONFIG)
    for item in cfg['frozen_files']:require(digest(ROOT/item['path'])==item['sha256'],'原冻结输入或代码改变')
    training=read(OUT/'monthly_training_receipt.json')
    require(training['actual_fits']==2 and training['ridge_status']==training['within_status']=='FIT_COMPLETE','已保存原月首模型不完整')
    write_json(OUT/'valid_origin_comparison_continuation.json',{'started_at':now(),'entrypoint':str(Path(__file__).relative_to(ROOT)),
        'entrypoint_sha256':digest(Path(__file__)),'ridge_models_sha256':digest(OUT/'ridge_models.json'),'within_models_sha256':digest(OUT/'within_models.json'),
        'previous_failure':'连续参考预算的基础分支在模拟实际账户前保留2013年至2019年的准备期目标，而最终保存的2020年起账户目标在该段为空；比较器把两个处理阶段的未使用准备区间一起比较',
        'evidence':'1606处仅位于2013年5月31日至2019年12月30日，原2019年12月31日起可决策区间零差异',
        'correction':'按每个来源原账户的准备收盘起点开始比较，有效区间内仍保留完整缺失位置与原数值精度；所有实际账本和决定前缀继续逐值核对',
        'strategy_parameters_changed':False,'account_or_model_revised':False,'new_model_fits_on_resume':0,
        'already_saved_incremental_accounts_to_reuse':5},exclusive=True)
    began=time.perf_counter()
    pipeline=SavedSeptemberPipeline(cfg).run()
    require(pipeline.cached_accounts==5,'已保存增量复用数量不同')
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
        'new_strategy_performance_computed':True,'run_seconds':(pd.Timestamp(now())-pd.Timestamp(read(OUT/'RUN_STARTED.json')['started_at'])).total_seconds(),
        'run_seconds_scope':'首次训练启动至最终保存汇总完成，含准备区间比较中止与接续开发；不是纯计算计时',
        'saved_resume_seconds':time.perf_counter()-began,'cached_incremental_accounts':5,'new_incremental_accounts_on_resume':17,
        'august_target_prefix_scope':'每个来源原有效决定区间，未使用的启动前准备目标不参与比较',
        'network_requests':0,'source_cost_cny':0}
    write_json(OUT/'result.json',result,exclusive=True)
    print(json.dumps({'status':result['status'],'seconds':result['run_seconds'],'metrics':metrics,'segments':segments},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
