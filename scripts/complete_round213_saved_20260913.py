"""从已完成的22条账户继续验证和汇总，不重跑整批来源。"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.post_selection_continuous_replay_v1 import ROOT,OUT,CONFIG,PRIMARY,CUTOFF,Pipeline,read
from research.post_selection_continuous_accounts_v1 import simulate_rearmed_exit,unpack
from research.entry_vintage_exit_inputs_v1 import EntryVintageExitController
from research.adaptive_allocation_v1 import summarize
from research.intraday_overnight_increment_v1 import now,digest,require,write_json


def exact_values_and_schema(actual,expected):
    """无成交的None转成原浮点列的NaN，仍逐个数值精确比较。"""
    require(list(actual.columns)==list(expected.columns),'恢复比较的列结构不同')
    converted=actual.astype(expected.dtypes.to_dict())
    pd.testing.assert_frame_equal(converted,expected,check_exact=True)
    return converted


def main():
    require(not (OUT/'result.json').exists() and not (OUT/'saved_validation_continuation.json').exists(), '本轮保存汇总已经开始或完成')
    cfg=read(CONFIG)
    for item in cfg['frozen_files']:require(digest(ROOT/item['path'])==item['sha256'],'冻结来源改变')
    write_json(OUT/'saved_validation_continuation.json',{'started_at':now(),'entrypoint':str(Path(__file__).relative_to(ROOT)),
        'entrypoint_sha256':digest(Path(__file__)),'status':'RESUME_SAVED_ACCOUNTS_ONLY',
        'previous_failure':'真实分段恢复拼接后，无成交fill_price列是object，完整账本该列是float64；数值比较因类型先行中止',
        'correction':'按原账本列类型规范化无成交空值后仍精确比较所有数值、状态和最终恢复状态；不改变账户函数或数据',
        'batch_accounts_rerun':0,'previous_actual_resume_validation_replays':2,'additional_actual_resume_validation_replays':2,
        'new_candidates':0,'new_model_fits':0},exclusive=True)
    began=time.perf_counter()
    pipeline=Pipeline(cfg)
    for (key,cost),item in pipeline.inventory.items():
        folder=OUT/'accounts'/cost/key
        ledger=pd.read_parquet(folder/'ledger.parquet');decisions=pd.read_parquet(folder/'decisions.parquet');state=read(folder/'checkpoint.json')
        pipeline.verify_prefix(key,cost,ledger,decisions)
        pipeline.accounts[(key,cost)]=(ledger,decisions,state)
    targets=pd.read_parquet(OUT/'factors/all_required_targets.parquet')
    for column in targets:
        if column!='date':
            key,cost=column.rsplit('__',1)
            pipeline.target(key,cost,targets[column])
    key,cost='ENTRY_VINTAGE_EXIT','STRESS'
    ledger,decisions,state=pipeline.accounts[(key,cost)]
    snapshot=read(OUT/'actual_held_split_checkpoint.json')
    split=unpack(snapshot['state'])['asof_index']
    source_cfg=pipeline.setting(key)
    args=(pipeline.data,pipeline.div,source_cfg,source_cfg['costs'][cost],'2020-01-02',pipeline.session,source_cfg['specification'])
    factory=lambda:EntryVintageExitController(pipeline.data,pipeline.within,2)
    first=simulate_rearmed_exit(*args,factory(),stop_index=split,next_execution_date=pipeline.next_date)
    require(first[-1]==snapshot,'原先保存的持仓断点状态无法复现')
    resumed=simulate_rearmed_exit(*args,factory(),resume=snapshot,next_execution_date=pipeline.next_date)
    joined=exact_values_and_schema(pd.concat([first[0],resumed[0]],ignore_index=True),ledger)
    raw_decisions=pd.concat([first[1],resumed[1]],ignore_index=True)
    joined_decisions=exact_values_and_schema(raw_decisions,decisions[raw_decisions.columns])
    require(resumed[-1]==state,'恢复后的最终账户及控制器状态不同')
    check_folder=OUT/'resume_validation';check_folder.mkdir()
    joined.to_parquet(check_folder/'joined_ledger.parquet',index=False)
    joined_decisions.to_parquet(check_folder/'joined_decisions.parquet',index=False)
    write_json(check_folder/'ending_checkpoint.json',resumed[-1],exclusive=True)
    write_json(OUT/'actual_resume_verification.json',{'verified_at':now(),'status':'PASS_ACTUAL_HELD_CYCLE_JSON_RESUME_EXACT',
        'source':key,'cost':cost,'split_close':'2026-08-13','split_shares':64200,
        'joined_ledger_rows':len(joined),'joined_decision_rows':len(joined_decisions),
        'account_and_controller_state_exact':True,'canonicalized_empty_execution_price_dtype':True,
        'validation_replays_including_initial_type_failure':4,'new_candidates':0,
        'full_pipeline_incremental_resume_verified':False},exclusive=True)
    pd.DataFrame(pipeline.checks).to_csv(OUT/'old_account_prefix_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pipeline.target_checks).to_csv(OUT/'old_target_prefix_comparison.csv',index=False,encoding='utf-8-sig')
    metrics=[];extension=[];positions=[]
    for (key,cost),(ledger,decisions,state) in pipeline.accounts.items():
        last=ledger.iloc[-1];pending=decisions.iloc[-1]
        positions.append({'model':key,'cost':cost,'date':str(last.date.date()),'cash':last.cash,'shares':int(last.shares),
            'equity':last.equity,'dividend_receivable':last.dividend_receivable,'next_execution_date':str(pending.execution_date.date()),
            'next_request':int(pending.requested_quantity),'next_target':pending.reference_weight})
        if key==PRIMARY:
            metrics.append({'model':key,'cost':cost,'name':'固定85%与15%方案连续至8月31日',**summarize(ledger,cfg),
                'mark_scope':'CONTINUOUS_CLOSE_NO_TERMINAL_LIQUIDATION','ending_shares':int(last.shares)})
            prior=ledger[ledger.date.eq('2026-08-13')].iloc[0];gap=ledger[ledger.date.ge('2026-08-14')]
            extension.append({'model':key,'cost':cost,'start_close':'2026-08-13','end_close':CUTOFF,'completed_days':len(gap),
                'start_equity':prior.equity,'end_equity':last.equity,'net_profit_cny':last.equity-prior.equity,
                'net_return':last.equity/prior.equity-1,'trades':int(gap.filled_quantity.ne(0).sum()),
                'commission':float(gap.commission.sum()),'slippage_cost':float(gap.slippage_cost.sum()),'ending_shares':int(last.shares),
                'strict_forward':False,'annualized_short_period_metrics_computed':False})
    metrics.sort(key=lambda x:list(cfg['costs']).index(x['cost']))
    extension.sort(key=lambda x:list(cfg['costs']).index(x['cost']))
    pd.DataFrame(metrics).to_csv(OUT/'continuous_account_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(extension).to_csv(OUT/'extension_segment_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(positions).to_csv(OUT/'ending_positions_and_next_requests.csv',index=False,encoding='utf-8-sig')
    original=read('reports/research/510300_incremental_selected_intent_mix_v1/result.json')
    start_time=pd.Timestamp(read(OUT/'RUN_STARTED.json')['started_at'])
    completion_time=pd.Timestamp(now())
    result={'study_id':cfg['study_id'],'completed_at':now(),'status':'COMPLETED_OLD_PREFIX_EXACT_CONTINUOUS_ACCOUNTS_TO_AUGUST31',
        'candidate_configurations':0,'candidate_models':[PRIMARY],'evaluation_accounts':2,'new_accounts_generated':22,
        'reused_control_accounts':0,'earlier_diagnostic_accounts':0,'new_earlier_diagnostic_accounts':0,'new_model_fits':0,'new_reference_accounts':0,
        'existing_source_account_replays':20,'existing_continuous_reference_replays_included':2,'resume_validation_replays':4,
        'all_metrics':metrics,'earlier_diagnostics':[r for r in original['earlier_diagnostics'] if r['model']==PRIMARY],
        'earlier_metrics_reused_from_round209':True,'post_selected_best_base':metrics[0],'primary':PRIMARY,
        'goal_achieved':False,'independent_validation':'NOT_ESTABLISHED','position_impact':0,'strict_forward_evidence_days':0,
        'old_normal_account_prefixes_verified':len(pipeline.checks),'old_target_prefixes_verified':len(pipeline.target_checks),
        'extension_segment_results':extension,'continuous_cutoff':CUTOFF,'terminal_liquidation':False,
        'monthly_model_updates_due_in_extension':0,'old_model_last_fit_origin':'2026-08-03',
        'new_strategy_performance_computed':True,'full_pipeline_incremental_resume_verified':False,
        'run_seconds':(completion_time-start_time).total_seconds(),
        'run_seconds_scope':'首次完整计算启动至保存汇总完成的墙钟，包含一次类型核对中止及接续开发，不能称为纯计算用时',
        'saved_verification_and_completion_seconds':time.perf_counter()-began,'source_cost_cny':0,'network_requests':0}
    write_json(OUT/'result.json',result,exclusive=True)
    print(json.dumps({'status':result['status'],'metrics':metrics,'extension':extension,'saved_completion_seconds':result['saved_verification_and_completion_seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
