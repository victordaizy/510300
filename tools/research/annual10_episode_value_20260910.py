"""按已结束完整自然周期学习新机会价值；严格季度前序拟合。"""
from __future__ import annotations
import json,sys,time,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.research.annual10_capital_20260910 import load_json,read_frame,aligned,accounting,FOLDERS,MODELS
from research.adaptive_allocation_v1 import normalize_dividends,summarize
from research.event_clock_account_v1 import simulate_event_account
from research.simple_price_entry_exit_v1 import signals,specifications,simulate_policy
from research.simple_volume_reversal_v1 import make_rules,specifications as vspecifications
from research.intraday_overnight_increment_v1 import digest,now,require,write_json
OUT=ROOT/'research_runs/annual10_sharpe12_20260910/episode_value'
FEATURES=['mom1','mom2','mom5','mom10','mom20','mom60','mom120','mom252','sma5','sma20','sma60','sma120','sma200','vol5','vol20','vol60','downvol20','dd20','dd60','dd120','overnight_log_5','overnight_log_20','intraday_log_5','intraday_log_20','range','close_location','volume_ratio','efficiency20','rsi2','z20','vol_ratio']
KINDS=['RIDGE','HGB','EXTRA_TREES','ENSEMBLE']
SOURCES=list(specifications())+list(vspecifications())

def setup():
    cfg=load_json(ROOT/'config/510300_continuous_reference_min_variance_v1.json')
    data=read_frame(ROOT/cfg['features']);div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    return cfg,data,div

def prepare():
    cfg,data,div=setup();OUT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'protocol.json').exists(),'本阶段已登记')
    paths=[Path(__file__),ROOT/cfg['features'],ROOT/cfg['dividends'],ROOT/'research/simple_price_entry_exit_v1.py',ROOT/'research/simple_volume_reversal_v1.py']
    write_json(OUT/'protocol.json',{'registered_at':now(),'remote_protocol_commit':'485f63b2cf64355a2ca8e925fd624a5e53da24b0','features':FEATURES,'sources':SOURCES,
        'models':KINDS,'thresholds':[0.,.01],'sizes':['FULL','VOL10'],'modes':['STANDALONE','IDLE_ONLY','REMAINING'],
        'minimum_cycles':150,'embargo_days':5,'reference_start':'2013-06-03','sample_weight':'inverse_entry_date_multiplicity',
        'source_exit_cooldown':'three origin indices after known zero; equivalent to two closes after scheduled next-open exit',
        'account_count':192,'code_sha256':digest(Path(__file__)),'files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in paths]},exclusive=True)
    specs={**specifications(),**vspecifications()};rules={**signals(data),**make_rules(data)[0]};n=len(data)
    training=[]
    for cost,cost_cfg in cfg['costs'].items():
        states=np.full((n,len(SOURCES)),np.nan);requests=np.zeros((n,len(SOURCES)),bool);days=np.ones((n,len(SOURCES)),int)*60
        for j,key in enumerate(SOURCES):
            led,dec,cy=simulate_policy(data,div,cfg,cost_cfg,'2013-06-03',rules[key],specs[key])
            accounting(led,cfg)
            folder=OUT/'references'/cost;folder.mkdir(parents=True,exist_ok=True)
            led.to_parquet(folder/f'{key}_ledger.parquet',index=False);dec.to_parquet(folder/f'{key}_decisions.parquet',index=False);cy.to_csv(folder/f'{key}_cycles.csv',index=False)
            states[:,j]=aligned(data,dec)
            idx=pd.DatetimeIndex(data.date).get_indexer(dec.origin)
            requests[idx,j]=dec.requested_quantity.gt(0)
            for i,mode in zip(idx,dec.entry_mode):days[i,j]=int(specs[key]['modes'].get(int(mode),{}).get('days') or 60)
            if cost=='BASE':
                for row in cy.to_dict('records'):
                    if pd.isna(row.get('exit_date')):continue
                    entry=int(pd.DatetimeIndex(data.date).get_loc(row['entry_origin']));exit_i=int(pd.DatetimeIndex(data.date).get_loc(row['exit_date']))
                    roi=row['net_profit_cny']/row['entry_cost_cny']
                    training.append({'source':key,'source_index':j,'entry_index':entry,'exit_index':exit_i,'entry_origin':row['entry_origin'],'exit_date':row['exit_date'],'roi':roi,'label':np.clip(roi,-.20,.30)})
        np.savez_compressed(OUT/f'{cost}_sources.npz',states=states,requests=requests,days=days)
    tr=pd.DataFrame(training).sort_values(['exit_index','entry_index','source']).reset_index(drop=True)
    tr.to_parquet(OUT/'training_episodes.parquet',index=False)
    print('REFERENCES',36,'TRAINING_EPISODES',len(tr),'CLIPPED',int((tr.roi!=tr.label).sum()),flush=True)

def models():
    return {'RIDGE':make_pipeline(StandardScaler(),Ridge(alpha=10.)),
        'HGB':HistGradientBoostingRegressor(loss='squared_error',max_iter=120,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=20260910),
        'EXTRA_TREES':ExtraTreesRegressor(n_estimators=128,max_depth=4,min_samples_leaf=15,max_features=1.,bootstrap=False,random_state=20260910,n_jobs=1)}

def fit():
    cfg,data,div=setup();pro=load_json(OUT/'protocol.json');require(pro['code_sha256']==digest(Path(__file__)),'冻结后代码变化')
    require(not (OUT/'predictions.npz').exists(),'预测已完成')
    tr=read_frame(OUT/'training_episodes.parquet');base=data[FEATURES].to_numpy(float)
    X=np.column_stack([base[tr.entry_index],np.eye(len(SOURCES))[tr.source_index]])
    y=tr.label.to_numpy(float);valid=np.isfinite(X).all(axis=1)
    require(valid.all(),'训练特征缺失，不删除源周期')
    cuts=[i for i in range(1,len(data)-1) if data.date.iloc[i].to_period('Q')!=data.date.iloc[i-1].to_period('Q') and data.date.iloc[i]>=pd.Timestamp('2014-01-01')]+[len(data)-1]
    pred=np.full((len(data),len(SOURCES),len(KINDS)),np.nan);records=[]
    with threadpool_limits(limits=1):
        for i,(t,end) in enumerate(zip(cuts[:-1],cuts[1:])):
            take=tr.exit_index.to_numpy()<=t-5;count=int(take.sum())
            rec={'fit_index':t,'fit_origin':data.date.iloc[t],'prediction_end_index':end-1,'training_cycles':count,'status':'NO_VIEW_SUPPORT'}
            if count<150:records.append(rec);continue
            eligible=tr[take];cnt=eligible.entry_index.value_counts();weight=(1/eligible.entry_index.map(cnt)).to_numpy(float)
            mods=models();query=np.column_stack([np.repeat(base[t:end],len(SOURCES),axis=0),np.tile(np.eye(len(SOURCES)),(end-t,1))])
            require(np.isfinite(query).all(),'查询特征有缺失')
            for j,(kind,m) in enumerate(mods.items()):
                if kind=='RIDGE':m.fit(X[take],y[take],standardscaler__sample_weight=weight,ridge__sample_weight=weight)
                else:m.fit(X[take],y[take],sample_weight=weight)
                pred[t:end,:,j]=m.predict(query).reshape(end-t,len(SOURCES))
            pred[t:end,:,3]=pred[t:end,:,:3].mean(axis=2)
            rec.update(status='FIT_COMPLETE',latest_exit_index=int(eligible.exit_index.max()),last_training_exit=str(eligible.exit_date.max()),
                first_training_entry=str(eligible.entry_origin.min()),unique_entry_dates=len(cnt),training_hash=hashlib.sha256(X[take].tobytes()+y[take].tobytes()+weight.tobytes()).hexdigest())
            folder=OUT/'models';folder.mkdir(exist_ok=True);joblib.dump({'record':rec,'models':mods,'features':FEATURES,'sources':SOURCES},folder/f'{t}.joblib',compress=3)
            records.append(rec)
            if i%8==0:print('FIT',str(data.date.iloc[t].date()),count,flush=True)
    np.savez_compressed(OUT/'predictions.npz',predictions=pred)
    write_json(OUT/'training_records.json',{'records':records,'actual_fits':3*sum(r['status']=='FIT_COMPLETE' for r in records),'completed_at':now()})
    print('FITS_DONE',3*sum(r['status']=='FIT_COMPLETE' for r in records),flush=True)

def learned_targets(data,core,states,requests,days,pred,threshold,size,mode,first):
    n=len(data);target=np.full(n,np.nan);chosen=np.full(n,-1,int);selected=-1;allocation=0.;clear=-100000
    for t in range(first-1,n-1):
        if selected>=0:
            v=states[t,selected]
            if not np.isfinite(v):target[t]=np.nan;chosen[t]=selected;continue
            if v==0 or (mode=='IDLE_ONLY' and np.isfinite(core[t]) and core[t]>0):selected=-1;allocation=0.;clear=t
        if selected<0 and t-clear>=3:
            permit=(mode!='IDLE_ONLY') or (np.isfinite(core[t]) and core[t]==0)
            if permit:
                candidates=np.flatnonzero(requests[t]&np.isfinite(pred[t])&(pred[t]>threshold))
                if len(candidates):
                    selected=min(candidates,key=lambda j:(-pred[t,j]/days[t,j],SOURCES[j]))
                    allocation=1. if size=='FULL' else min(1.,.10/data.vol20.iloc[t])
        sleeve=allocation if selected>=0 else 0.
        chosen[t]=selected
        if mode=='STANDALONE':target[t]=sleeve
        elif not np.isfinite(core[t]):target[t]=np.nan
        elif mode=='REMAINING':target[t]=core[t]+(1-core[t])*sleeve
        else:target[t]=core[t] if core[t]>0 else sleeve
    return target,chosen

def evaluate(period,cost_id):
    cfg,data,div=setup();require(load_json(OUT/'protocol.json')['code_sha256']==digest(Path(__file__)),'代码变化')
    start,end=(cfg['evaluation_start'],cfg['data_cutoff']) if period=='evaluation' else (cfg['earlier_start'],cfg['earlier_terminal'])
    data=data[data.date.le(end)].copy();n=len(data);first=int(np.flatnonzero(data.date>=start)[0])
    src=np.load(OUT/f'{cost_id}_sources.npz');pred=np.load(OUT/'predictions.npz')['predictions'][:n]
    states,requests,days=[src[k][:n] for k in ['states','requests','days']]
    old=read_frame(ROOT/'reports/research'/FOLDERS['R150']/period/cost_id/f"{MODELS['R150']}_decisions.parquet")
    core=np.minimum(1.,2*aligned(data,old))
    folder=OUT/period/cost_id;folder.mkdir(parents=True,exist_ok=True);require(not (folder/'metrics.csv').exists(),'不覆盖完成结果')
    rows=[];checks=[]
    for j,kind in enumerate(KINDS):
        for threshold in [0.,.01]:
            for size in ['FULL','VOL10']:
                for mode in ['STANDALONE','IDLE_ONLY','REMAINING']:
                    model=f'{kind}__T{int(threshold*1000)}__{size}__{mode}'
                    target,chosen=learned_targets(data,core,states,requests,days,pred[:,:,j],threshold,size,mode,first)
                    ledger,dec=simulate_event_account(data,div,cfg,cfg['costs'][cost_id],start,model,targets=target,event_mask=np.ones(n,bool))
                    ledger.to_parquet(folder/f'{model}_ledger.parquet',index=False);dec.to_parquet(folder/f'{model}_decisions.parquet',index=False)
                    m={'period':period,'cost':cost_id,'model':model,**summarize(ledger,cfg)}
                    m['point_met']=bool(m['annualized_return']>=.10 and m['net_sharpe'] is not None and m['net_sharpe']>=1.2)
                    rows.append(m);checks.append({'model':model,**accounting(ledger,cfg)})
                    pd.DataFrame({'date':data.date,'chosen_source_index':chosen,'target':target}).to_parquet(folder/f'{model}_choices.parquet',index=False)
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False);pd.DataFrame(checks).to_csv(folder/'checks.csv',index=False)
    print('EVAL',period,cost_id,flush=True)
    print(pd.DataFrame(rows).sort_values('net_sharpe',ascending=False).head(12)[['model','annualized_return','net_sharpe','max_drawdown','point_met']].to_string(index=False),flush=True)
    print('POINT_MET',sum(r['point_met'] for r in rows),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='fit':fit()
    else:evaluate(*sys.argv[1:])
