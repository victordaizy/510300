"""逐日假设交易标签、季度因果盈亏分类与完整账户；不覆盖旧结果。"""
from __future__ import annotations
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.research.annual10_capital_20260910 import load_json,read_frame,aligned,accounting,FOLDERS,MODELS
from tools.research.annual10_episode_value_20260910 import FEATURES
from research.adaptive_allocation_v1 import normalize_dividends,summarize
from research.simple_price_entry_exit_v1 import simulate_policy
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest,now,require,write_json
OUT=ROOT/'research_runs/annual10_sharpe12_20260910/daily_opportunity'
KINDS=['LOGISTIC','HGB','EXTRA_TREES','ENSEMBLE']
SPEC={'cooldown':2,'modes':{1:{'loss':.02,'trail':None,'take':.04,'days':20}}}

def setup():
    cfg=load_json(ROOT/'config/510300_continuous_reference_min_variance_v1.json')
    data=read_frame(ROOT/cfg['features']);div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    return cfg,data,div

def inputs(data):
    dates=pd.DatetimeIndex(data.date);monthpos=pd.Series(1,index=dates).groupby(dates.to_period('M')).cumsum().to_numpy()
    x=np.column_stack([data[FEATURES].to_numpy(float),np.eye(5)[dates.weekday],monthpos<=3,(dates.days_in_month-dates.day)<5])
    return x,FEATURES+[f'weekday_{i}' for i in range(5)]+['month_first_three','month_end_five']

def prepare():
    cfg,data,div=setup();OUT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'protocol.json').exists(),'本阶段已登记，不覆盖')
    x,features=inputs(data)
    files=[Path(__file__),ROOT/cfg['features'],ROOT/cfg['dividends'],ROOT/'research/simple_price_entry_exit_v1.py',ROOT/'research/event_clock_account_v1.py',ROOT/'tools/research/annual10_capital_20260910.py']
    write_json(OUT/'protocol.json',{'registered_at':now(),'remote_protocol_commit':'c06ba274667bbd6fdedf4e78335be6c4dcd352a1','features':features,'specification':SPEC,
      'minimum_training':400,'embargo':5,'models':KINDS,'thresholds':[0.,.005],'sizes':['FULL','VOL10'],'size_clock':'first_positive_reference_intent_origin',
      'modes':['STANDALONE','IDLE_ONLY','REMAINING'],'planned_accounts':192,'code_sha256':digest(Path(__file__)),
      'files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in files]},exclusive=True)
    rows=[]
    for t in range(1,len(data)-22):
        if not np.isfinite(x[t]).all() or data.date.iloc[t]<pd.Timestamp('2013-06-03'):continue
        frame=data.iloc[t:t+23].copy().reset_index(drop=True)
        entry=np.zeros(len(frame),int);entry[0]=1
        rule={'entry':entry,'exit':{1:np.zeros(len(frame),bool)}}
        led,dec,cy=simulate_policy(frame,div,cfg,cfg['costs']['BASE'],str(frame.date.iloc[1].date()),rule,SPEC)
        accounting(led,cfg)
        row={'origin_index':t,'origin':data.date.iloc[t],'status':'NO_VIEW_NO_FILLED_CYCLE'}
        if len(cy)==1 and pd.notna(cy.iloc[0].exit_date):
            c=cy.iloc[0];exit_i=int(pd.DatetimeIndex(data.date).get_loc(c.exit_date))
            # 终点强制清算不能假装自然退出；其标签只记录、不进入训练。
            if '研究终点' not in c.exit_reasons:
                row.update(status='MATURE',entry_index=t+int(c.entry_index),exit_index=exit_i,exit_date=c.exit_date,
                           roi=float(c.net_profit_cny/c.entry_cost_cny),entry_cost=float(c.entry_cost_cny),net_profit=float(c.net_profit_cny),exit_reasons=c.exit_reasons)
        rows.append(row)
        if len(rows)%500==0:print('LABELS',len(rows),flush=True)
    pd.DataFrame(rows).to_parquet(OUT/'labels.parquet',index=False)
    print('LABELS_COMPLETE',len(rows),pd.Series(r['status'] for r in rows).value_counts().to_dict(),flush=True)

def concurrency_weights(labels,n):
    delta=np.zeros(n+1)
    en=labels.entry_index.to_numpy(int);ex=labels.exit_index.to_numpy(int)
    np.add.at(delta,en,1.);np.add.at(delta,ex+1,-1.)
    concurrency=np.cumsum(delta[:-1]);inv=np.divide(1.,concurrency,out=np.zeros(n),where=concurrency>0)
    cumulative=np.r_[0.,np.cumsum(inv)]
    return (cumulative[ex+1]-cumulative[en])/(ex-en+1)

def classifiers():
    return {'LOGISTIC':make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=20260910)),
     'HGB':HistGradientBoostingClassifier(max_iter=120,learning_rate=.05,max_leaf_nodes=7,min_samples_leaf=30,l2_regularization=1.,early_stopping=False,random_state=20260910),
     'EXTRA_TREES':ExtraTreesClassifier(n_estimators=128,max_depth=5,min_samples_leaf=20,max_features=1.0,bootstrap=False,random_state=20260910,n_jobs=1)}

def fit():
    cfg,data,div=setup();require(load_json(OUT/'protocol.json')['code_sha256']==digest(Path(__file__)),'冻结后代码变化')
    require(not (OUT/'predictions.npz').exists(),'本次预测已经存在')
    labels=read_frame(OUT/'labels.parquet');labels=labels[labels.status.eq('MATURE')].copy()
    x,features=inputs(data);prob=np.full((len(data),4),np.nan);pred=prob.copy();records=[]
    cuts=[i for i in range(1,len(data)-1) if data.date.iloc[i].to_period('Q')!=data.date.iloc[i-1].to_period('Q') and data.date.iloc[i]>=pd.Timestamp('2014-01-01')]+[len(data)-1]
    with threadpool_limits(limits=1):
        for k,(t,end) in enumerate(zip(cuts[:-1],cuts[1:])):
            train=labels[labels.exit_index<=t-5];y=train.roi.to_numpy(float);cls=y>0
            record={'fit_index':t,'fit_origin':data.date.iloc[t],'prediction_end_index':end-1,'n':len(train),'status':'NO_VIEW_SUPPORT'}
            if len(train)<400 or min(cls.sum(),(~cls).sum())<30:records.append(record);continue
            weight=concurrency_weights(train,len(data));xt=x[train.origin_index.to_numpy(int)]
            require(np.isfinite(xt).all() and np.isfinite(x[t:end]).all(),'必要特征缺失')
            gain=np.average(y[cls],weights=weight[cls]);loss=np.average(y[~cls],weights=weight[~cls]);mods=classifiers()
            for j,(name,m) in enumerate(mods.items()):
                if name=='LOGISTIC':m.fit(xt,cls,standardscaler__sample_weight=weight,logisticregression__sample_weight=weight)
                else:m.fit(xt,cls,sample_weight=weight)
                prob[t:end,j]=m.predict_proba(x[t:end])[:,1]
            prob[t:end,3]=prob[t:end,:3].mean(axis=1)
            pred[t:end]=prob[t:end]*gain+(1-prob[t:end])*loss
            record.update(status='FIT_COMPLETE',latest_exit_index=int(train.exit_index.max()),gain_mean=float(gain),loss_mean=float(loss),
              training_hash=hashlib.sha256(xt.tobytes()+y.tobytes()+weight.tobytes()).hexdigest())
            folder=OUT/'models';folder.mkdir(exist_ok=True);joblib.dump({'record':record,'models':mods,'features':features},folder/f'{t}.joblib',compress=3)
            records.append(record)
            if k%8==0:print('FIT',str(data.date.iloc[t].date()),len(train),flush=True)
    np.savez_compressed(OUT/'predictions.npz',probability=prob,predictions=pred)
    write_json(OUT/'training_records.json',{'records':records,'actual_fits':sum(r['status']=='FIT_COMPLETE' for r in records)*3,'completed_at':now()})
    print('FIT_COMPLETE',sum(r['status']=='FIT_COMPLETE' for r in records)*3,flush=True)

def scaled_sleeve(state,vol,size):
    out=np.full(len(state),np.nan);allocation=0.;active=False
    for t,v in enumerate(state):
        if not np.isfinite(v):continue
        if v==0:active=False;allocation=0.
        elif not active:
            if not np.isfinite(vol[t]) or vol[t]<=0:continue
            active=True;allocation=1. if size=='FULL' else min(1.,.10/vol[t])
        out[t]=allocation
    return out

def evaluate(period,cost_id):
    cfg,data,div=setup();require(load_json(OUT/'protocol.json')['code_sha256']==digest(Path(__file__)),'冻结后代码变化')
    start,end=(cfg['evaluation_start'],cfg['data_cutoff']) if period=='evaluation' else (cfg['earlier_start'],cfg['earlier_terminal'])
    data=data[data.date.le(end)].copy();n=len(data)
    prediction=np.load(OUT/'predictions.npz')['predictions'][:n]
    old=read_frame(ROOT/'reports/research'/FOLDERS['R150']/period/cost_id/f"{MODELS['R150']}_decisions.parquet");core=np.minimum(1.,2*aligned(data,old))
    folder=OUT/period/cost_id;folder.mkdir(parents=True,exist_ok=True);require(not (folder/'metrics.csv').exists(),'不覆盖完成结果')
    rows=[];checks=[]
    for j,kind in enumerate(KINDS):
        for threshold in [0.,.005]:
            source=f'{kind}__T{int(threshold*1000)}'
            rule={'entry':(np.isfinite(prediction[:,j])&(prediction[:,j]>threshold)).astype(int),'exit':{1:np.zeros(n,bool)}}
            ref,rd,cy=simulate_policy(data,div,cfg,cfg['costs'][cost_id],start,rule,SPEC)
            refout=folder/'sources';refout.mkdir(exist_ok=True)
            ref.to_parquet(refout/f'{source}_ledger.parquet',index=False);rd.to_parquet(refout/f'{source}_decisions.parquet',index=False);cy.to_csv(refout/f'{source}_cycles.csv',index=False)
            accounting(ref,cfg);state=aligned(data,rd)
            for size in ['FULL','VOL10']:
                sleeve=scaled_sleeve(state,data.vol20.to_numpy(),size)
                for mode in ['STANDALONE','IDLE_ONLY','REMAINING']:
                    if mode=='STANDALONE':target=sleeve.copy()
                    elif mode=='IDLE_ONLY':target=np.where(core>0,core,sleeve)
                    else:target=core+(1-core)*sleeve
                    if mode!='STANDALONE':target[~np.isfinite(core)|~np.isfinite(sleeve)]=np.nan
                    model=source+'__'+size+'__'+mode
                    led,dec=simulate_event_account(data,div,cfg,cfg['costs'][cost_id],start,model,targets=target,event_mask=np.ones(n,bool))
                    led.to_parquet(folder/f'{model}_ledger.parquet',index=False);dec.to_parquet(folder/f'{model}_decisions.parquet',index=False)
                    m={'period':period,'cost':cost_id,'model':model,**summarize(led,cfg)}
                    m['point_met']=bool(m['annualized_return']>=.10 and m['net_sharpe'] is not None and m['net_sharpe']>=1.2)
                    rows.append(m);checks.append({'model':model,**accounting(led,cfg)})
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False);pd.DataFrame(checks).to_csv(folder/'checks.csv',index=False)
    print('EVAL',period,cost_id,flush=True)
    print(pd.DataFrame(rows).sort_values('net_sharpe',ascending=False).head(8)[['model','annualized_return','net_sharpe','max_drawdown','point_met']].to_string(index=False),flush=True)
    print('POINT_MET',sum(r['point_met'] for r in rows),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='fit':fit()
    else:evaluate(*sys.argv[1:])
