"""仅检验延迟ETF来源的信息增量；下载时点不冒充原始发布时间。"""
from __future__ import annotations
import sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.research.annual10_capital_20260910 import load_json,read_frame,aligned,accounting,FOLDERS,MODELS
from tools.research.annual10_episode_value_20260910 import FEATURES,models
from research.adaptive_allocation_v1 import normalize_dividends,summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest,now,require,write_json,holding_total_return
OUT=ROOT/'research_runs/annual10_sharpe12_20260910/etf_flow'
KINDS=['RIDGE','HGB','EXTRA_TREES','ENSEMBLE'];SETS=['PRICE','PRICE_FLOW'];HORIZONS=[5,20]
PATHS={'NAV':'data/raw/fund/510300_nav_daily_raw.parquet','SHARE':'data/raw/flow/510300_fund_share_daily_tushare.parquet','MARGIN':'data/raw/flow/510300_margin_detail_daily_tushare.parquet'}

def setup():
    cfg=load_json(ROOT/'config/510300_continuous_reference_min_variance_v1.json');data=read_frame(ROOT/cfg['features']);div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    return cfg,data,div

def read_source(path):
    frame=pd.read_parquet(path)
    require(getattr(frame.date.dtype,'tz',None) is None,'来源交易日期带时区，需另行核对')
    frame['date']=frame.date.astype('datetime64[ns]')
    return frame

def flow_features(data,sources):
    dates=pd.DatetimeIndex(data.date);raw={};coverage=[]
    for name,f in sources.items():
        require(not f.date.duplicated().any(),'ETF来源同日重复')
        coverage.append({'source':name,'raw_rows':len(f),'first':str(f.date.min()),'last':str(f.date.max()),'nontrading_rows':int((~f.date.isin(dates)).sum()),'publication_vintage':'UNVERIFIED'})
        raw[name]=f.set_index('date').reindex(dates)
    # 每项源统一延迟两个交易日；没有前向填补。
    premium=raw['NAV'].close_premium_to_nav.shift(2)
    share=raw['SHARE'].fund_shares.shift(2).where(lambda s:s>0)
    margin=raw['MARGIN'].shift(2);rz=margin.rzye.where(lambda s:s>0)
    close=pd.Series(data.close.to_numpy(float),index=dates).shift(2)
    features={'premium':premium,'premium_mean20':premium.rolling(20).mean(),'premium_std20':premium.rolling(20).std(ddof=1),'premium_range20':premium.rolling(20).max()-premium.rolling(20).min()}
    for w in [1,5,20,60]:features[f'share_log_{w}']=np.log(share/ share.shift(w))
    for w in [3,20]:features[f'financing_log_{w}']=np.log(rz/rz.shift(w))
    features.update(financing_size=rz/(share*close),financing_flow5=margin.financing_net_buy_cny.rolling(5).sum()/rz,short_quantity_size=margin.rqyl/share)
    f=pd.DataFrame(features,index=dates).replace([np.inf,-np.inf],np.nan).reset_index(names='date')
    return f,coverage

def prepare():
    cfg,data,div=setup();OUT.mkdir(parents=True,exist_ok=True);require(not (OUT/'protocol.json').exists(),'不覆盖已登记研究')
    paths=[Path(__file__),ROOT/cfg['features'],ROOT/cfg['dividends']]+[ROOT/v for v in PATHS.items()] if False else [Path(__file__),ROOT/cfg['features'],ROOT/cfg['dividends']]+[ROOT/v for v in PATHS.values()]
    write_json(OUT/'protocol.json',{'registered_at':now(),'remote_protocol_commit':'4a818bcd526197f70db622b4f6f8bb750593c318','features_price':FEATURES,'models':KINDS,'feature_sets':SETS,'horizons':HORIZONS,'minimum_samples':252,'source_lag':2,
      'thresholds':[0.,.01],'modes':['STANDALONE','REMAINING_CORE150X2'],'accounts':256,'publication_vintage':'UNVERIFIED','evidence':'ASSUMED_LAG_PROXY_DISCOVERY_ONLY','code_sha256':digest(Path(__file__)),
      'files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in paths]},exclusive=True)
    f,coverage=flow_features(data,{k:read_source(ROOT/v) for k,v in PATHS.items()});f.to_parquet(OUT/'features.parquet',index=False)
    valid=np.isfinite(f.drop(columns='date').to_numpy(float)).all(axis=1)&np.isfinite(data[FEATURES].to_numpy(float)).all(axis=1)
    write_json(OUT/'source_coverage.json',{'sources':coverage,'valid_origins':int(valid.sum()),'first_valid':str(data.date[valid].min()),'last_valid':str(data.date[valid].max()),'source_availability':'TWO_TRADING_DAY_LAG_ASSUMPTION_NOT_PROVEN_PIT'})
    labels=[]
    for h in HORIZONS:
        for t in np.flatnonzero(valid):
            if t+1+h>=len(data):continue
            value,dist=holding_total_return(data,div,int(t+1),int(t+1+h))
            labels.append({'origin_index':t,'origin':data.date.iloc[t],'horizon':h,'exit_index':t+1+h,'gross_roi':value,'label':np.clip(value,-.20,.30),'dividend_per_share':dist})
    pd.DataFrame(labels).to_parquet(OUT/'labels.parquet',index=False)
    print('FEATURES',int(valid.sum()),'LABELS',len(labels),flush=True)

def fit(h):
    cfg,data,div=setup();require(load_json(OUT/'protocol.json')['code_sha256']==digest(Path(__file__)),'冻结代码改变')
    path=OUT/f'predictions_H{h}.npz';require(not path.exists(),'本期限已完成，不覆盖')
    f=read_frame(OUT/'features.parquet');fx=f.drop(columns='date').to_numpy(float);px=data[FEATURES].to_numpy(float)
    valid=np.isfinite(fx).all(axis=1)&np.isfinite(px).all(axis=1)
    labels=read_frame(OUT/'labels.parquet');labels=labels[labels.horizon.eq(h)]
    pred=np.full((len(data),2,4),np.nan);records=[];cuts=[i for i in range(1,len(data)-1) if data.date.iloc[i].to_period('Q')!=data.date.iloc[i-1].to_period('Q')]+[len(data)-1]
    with threadpool_limits(limits=1):
        for t,end in zip(cuts[:-1],cuts[1:]):
            train=labels[labels.exit_index.le(t-2)];ti=train.origin_index.to_numpy(int);y=train.label.to_numpy(float)
            rec={'fit_index':t,'prediction_end_index':end-1,'training_rows':len(train),'status':'NO_VIEW_SUPPORT','horizon':h}
            if len(train)<252:records.append(rec);continue
            ix=np.arange(t,end)[valid[t:end]]
            for j,name in enumerate(SETS):
                allx=px if name=='PRICE' else np.column_stack([px,fx]);xx=allx[ti];mods=models()
                for k,(kind,m) in enumerate(mods.items()):
                    m.fit(xx,y)
                    if len(ix):pred[ix,j,k]=m.predict(allx[ix])
                pred[ix,j,3]=pred[ix,j,:3].mean(axis=1)
                folder=OUT/'models';folder.mkdir(exist_ok=True)
                item={'fit_index':t,'latest_exit_index':int(train.exit_index.max()),'training_hash':hashlib.sha256(xx.tobytes()+y.tobytes()).hexdigest(),'models':mods,'feature_set':name,'horizon':h}
                joblib.dump(item,folder/f'H{h}_{name}_{t}.joblib',compress=3)
            rec.update(status='FIT_COMPLETE',latest_exit_index=int(train.exit_index.max()));records.append(rec)
    np.savez_compressed(path,predictions=pred)
    write_json(OUT/f'training_H{h}.json',{'records':records,'completed_model_fits':6*sum(r['status']=='FIT_COMPLETE' for r in records),'completed_at':now()})
    print('FIT_DONE',h,6*sum(r['status']=='FIT_COMPLETE' for r in records),flush=True)

def policy_target(prediction,core,cost,threshold,mode):
    barrier=2*(cost['commission']+cost['slippage'])+threshold
    side=np.where(np.isfinite(prediction),(prediction>barrier).astype(float),np.nan)
    if mode=='STANDALONE':return side
    # 来源无观点时保留独立核心意向，不伪造新增观点。
    target=core.copy();known=np.isfinite(side)&np.isfinite(core)
    target[known]=core[known]+(1-core[known])*side[known]
    return target

def evaluate(period,cost_id):
    cfg,data,div=setup();require(load_json(OUT/'protocol.json')['code_sha256']==digest(Path(__file__)),'冻结代码变化')
    start,end=(cfg['evaluation_start'],cfg['data_cutoff']) if period=='evaluation' else (cfg['earlier_start'],cfg['earlier_terminal']);data=data[data.date.le(end)].copy();n=len(data)
    core=np.minimum(1.,2*aligned(data,read_frame(ROOT/'reports/research'/FOLDERS['R150']/period/cost_id/f"{MODELS['R150']}_decisions.parquet")))
    folder=OUT/period/cost_id;folder.mkdir(parents=True,exist_ok=True);require(not (folder/'metrics.csv').exists(),'完成结果不得覆盖')
    rows=[];checks=[]
    for h in HORIZONS:
        pred=np.load(OUT/f'predictions_H{h}.npz')['predictions'][:n]
        for j,fs in enumerate(SETS):
            for k,kind in enumerate(KINDS):
                for threshold in [0.,.01]:
                    for mode in ['STANDALONE','REMAINING_CORE150X2']:
                        model=f'{fs}__H{h}__{kind}__T{int(threshold*1000)}__{mode}'
                        target=policy_target(pred[:,j,k],core,cfg['costs'][cost_id],threshold,mode)
                        led,dec=simulate_event_account(data,div,cfg,cfg['costs'][cost_id],start,model,targets=target,event_mask=np.ones(n,bool))
                        led.to_parquet(folder/f'{model}_ledger.parquet',index=False);dec.to_parquet(folder/f'{model}_decisions.parquet',index=False)
                        m={'period':period,'cost':cost_id,'model':model,**summarize(led,cfg),'publication_vintage':'UNVERIFIED_PUBLICATION_VINTAGE','prediction_origins':int(np.isfinite(pred[:,j,k]).sum())}
                        m['point_met']=bool(m['annualized_return']>=.10 and m['net_sharpe'] is not None and m['net_sharpe']>=1.2)
                        rows.append(m);checks.append({'model':model,**accounting(led,cfg)})
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False);pd.DataFrame(checks).to_csv(folder/'checks.csv',index=False)
    print('EVAL',period,cost_id,flush=True)
    print(pd.DataFrame(rows).sort_values('net_sharpe',ascending=False).head(8)[['model','annualized_return','net_sharpe','point_met']].to_string(index=False),flush=True)
    print('POINT_MET',sum(r['point_met'] for r in rows),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='fit':fit(int(sys.argv[2]))
    else:evaluate(*sys.argv[1:])
