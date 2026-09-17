"""只延长原六十日日内训练基准，并在原月首生成两条成熟模型记录。"""
from copy import deepcopy
import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require
from research.post_selection_continuous_accounts_v1 import simulate_policy
from research.learned_cycle_exit_v1 import FEATURES,state_values,continuation_label,training_rows,fit_one
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit


def reference_samples(data,dividends,cfg,rule,fit_date,next_date):
    """原回调仅记录状态且不请求退出，因此复用无学习退出的同规则价格账户。"""
    boundary=pd.Timestamp(fit_date)
    frame=data[data.date.le(boundary)].reset_index(drop=True)
    require(len(frame)>0 and frame.date.iloc[-1]==boundary,'训练边界不是已完成交易日')
    selected={'entry':np.asarray(rule['entry'])[:len(frame)],'exit':{int(k):np.asarray(v)[:len(frame)] for k,v in rule['exit'].items()}}
    ledger,decisions,cycles,checkpoint=simulate_policy(frame,dividends,cfg,cfg['costs']['BASE'],cfg['reference_start'],
        selected,cfg['candidate_specs']['D60_INTRA'],next_execution_date=next_date)
    states={}
    for cycle in cycles.to_dict('records'):
        held=ledger[ledger.cycle_id.eq(cycle['cycle_id'])]
        peak=float(cycle['entry_cost_cny']);distribution=0.
        for row in held.itertuples():
            t=int(np.flatnonzero(frame.date.eq(row.date))[0])
            distribution+=float(row.dividend_recognized)
            value=int(row.shares)*float(frame.close.iloc[t])+distribution
            peak=max(peak,value)
            states[t]=(int(cycle['cycle_id']),state_values(frame,t,cycle,value,peak))
    decisions['learning_cycle_id']=np.nan
    decisions['learned_exit_requested']=False
    for column in FEATURES:
        if column not in decisions:decisions[column]=np.nan
    for index,row in decisions.iterrows():
        state=states.get(int(row.origin_index))
        if state is not None:
            decisions.loc[index,'learning_cycle_id']=state[0]
            for column,value in zip(FEATURES,state[1]):decisions.loc[index,column]=value
    samples=[]
    for cycle in cycles.to_dict('records'):
        if pd.isna(cycle.get('exit_date')) or '研究终点' in cycle['exit_reasons']:
            continue
        end=int(np.flatnonzero(frame.date.eq(pd.Timestamp(cycle['exit_date'])))[0])
        held=decisions[decisions.learning_cycle_id.eq(cycle['cycle_id']) & decisions.requested_quantity.eq(0)]
        for row in held.to_dict('records'):
            t=int(row['origin_index'])
            if t+1>=end or not np.isfinite([row[k] for k in FEATURES]).all():continue
            target,extra=continuation_label(frame,dividends,int(cycle['entry_quantity']),t+1,end,cfg['costs']['BASE'],cfg['tick'])
            samples.append({'signal':'D60_INTRA','cycle_id':int(cycle['cycle_id']),'origin_index':t,'origin':frame.date.iloc[t],
                'early_exit_index':t+1,'early_exit_date':frame.date.iloc[t+1],'exit_index':end,'mature_date':frame.date.iloc[end],
                'target':target,'extra_dividend_cny':extra,'reference_quantity':int(cycle['entry_quantity']),**{k:row[k] for k in FEATURES}})
    return frame,ledger,decisions,cycles,checkpoint,pd.DataFrame(samples)


def fit_month(frame,samples,cfg31,cfg114,fit_date):
    t=int(np.flatnonzero(frame.date.eq(pd.Timestamp(fit_date)))[0])
    require(t>0 and frame.date.iloc[t].to_period('M')!=frame.date.iloc[t-1].to_period('M'),'新增拟合不是原月首交易日')
    rows,ids=training_rows(samples,t,cfg31)
    require((rows.exit_index<=t).all() and (rows.early_exit_index<rows.exit_index).all(),'新增训练用了未成熟周期')
    require(cfg31['recent_cycles']==cfg114['recent_cycles']==20 and cfg31['minimum_cycles']==cfg114['minimum_cycles']==10 and
        cfg31['minimum_rows']==cfg114['minimum_rows']==100,'原成熟周期与行数门槛不同')
    eligible=len(ids)>=10 and len(rows)>=100
    missing=int((~np.isfinite(rows[FEATURES].to_numpy(float)).all(axis=1)).sum())
    require(not missing,'新增成熟训练八项因素缺失，禁止删行补救')
    common={'fit_index':t,'fit_origin':str(frame.date.iloc[t].date()),'fit_time':frame.date.iloc[t]+pd.Timedelta(hours=15,minutes=5),
        'training_cycles':ids,'training_cycle_count':len(ids),'training_rows':len(rows),
        'latest_exit_index':int(rows.exit_index.max()) if len(rows) else None,
        'latest_exit_date':str(rows.mature_date.max().date()) if len(rows) else None}
    ridge={**common,'signal':'D60_INTRA','kind':'RIDGE','status':'NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS','model':None}
    within={**common,'status':'NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS','model':None,'eligible_for_fit':eligible,
        'failure':None,'missing_feature_rows':missing}
    fits=0
    if eligible:
        ridge['model']=fit_one(rows,'RIDGE',cfg31);ridge['status']='FIT_COMPLETE';fits+=1
        try:
            within['model']=fit_within_cycle_exit(rows,cfg114);within['status']='FIT_COMPLETE';fits+=1
        except (RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            within.update(status='NO_VIEW_MODEL_FIT_FAILED',failure=str(error))
    return ridge,within,rows,fits


def append_record(original,record):
    require(original and record['fit_index']>original[-1]['fit_index'],'新增模型记录时点没有位于原序列之后')
    return deepcopy(original)+[deepcopy(record)]
