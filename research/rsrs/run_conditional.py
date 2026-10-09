"""RSRS第二批：固定条件预测及同结构去因子对照，禁止下单。"""
from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ.setdefault(k,'1')
import sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.rsrs import run_research as base
from research.point_first_passage_inputs_v1 import FEATURES
from research.point_account_nr7_inputs_v1 import fill,fee
OUT=ROOT/'rsrs_artifacts'/'experiment_v2'
CONFIGS=[(m,h,u) for m in ['RIDGE','HGB'] for h in [5,20] for u in [False,True]]
PRIMARY='HGB_H5_RSRS'

def name_of(m,h,u):return f'{m}_H{h}_'+('RSRS' if u else 'BASE')

def make_features(d):
    x=d[FEATURES].copy()
    f=base.calculate_rsrs(pd.DataFrame({'high':d.high+d.cash_shift,'low':d.low+d.cash_shift}),18,252)
    for c in ['beta','r_squared','zscore','right_skew_rsrs']:x['rsrs_'+c]=f[c]
    x['rsrs_z_change5']=f.zscore.diff(5)
    x['rv20']=((d.close+d.dividend)/d.close.shift()-1).rolling(20,min_periods=20).std(ddof=1)
    return x.replace([np.inf,-np.inf],np.nan)

def label(d,h):
    ans=np.full(len(d),np.nan);op=d.open.to_numpy(float);shift=d.cash_shift.to_numpy(float)
    for t in range(len(d)-h-1):
        a,b=t+1,t+h+1;q=int(100000/op[a]//100)*100
        buy=fill(op[a],1,'STRESS');sell=fill(op[b],-1,'STRESS')
        debit=q*buy+fee(q*buy,'STRESS')
        credit=q*sell-fee(q*sell,'STRESS')+q*(shift[b]-shift[a])
        ans[t]=credit/debit-1
    return ans

def predict(d,x,y,kind,h,use,collect=True):
    all_ok=np.isfinite(x.to_numpy(float)).all(axis=1)
    cols=list(FEATURES)+['rv20']+([c for c in x if c.startswith('rsrs_')] if use else [])
    xx=x[cols].to_numpy(float);vol=x.rv20.to_numpy(float)*np.sqrt(h)
    yy=np.clip(y/np.maximum(vol,1e-6),-3,3)
    months=d.date.dt.to_period('M');refit=(months!=months.shift()).to_numpy()
    for day in ['2014-12-31','2019-12-31']:refit |= d.date.eq(pd.Timestamp(day)).to_numpy()
    model=None;mean=scale=None;pred=np.full(len(d),np.nan);fits=[]
    for i in range(len(d)):
        if refit[i]:
            pool=np.arange(max(0,i-756+1),i-h,dtype=int)
            pool=pool[all_ok[pool]&np.isfinite(yy[pool])]
            if len(pool)<252:model=None;continue
            assert np.max(pool+h+1)<=i
            train=xx[pool];mean=train.mean(axis=0);scale=train.std(axis=0);scale[scale<1e-12]=1
            train=np.clip((train-mean)/scale,-5,5)
            model=(Ridge(alpha=100.,solver='svd') if kind=='RIDGE' else
                   HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=4,max_depth=2,min_samples_leaf=63,
                       l2_regularization=10.,learning_rate=.05,early_stopping=False,random_state=510300))
            model.fit(train,yy[pool])
            if collect:fits.append({'fit_idx':i,'fit_date':str(d.date.iloc[i].date()),'n':len(pool),
                'last_mature_idx':int(np.max(pool+h+1)),'training_origins':pool.tolist(),'columns':cols,
                'mean':mean.tolist(),'scale':scale.tolist()})
        if model is not None and all_ok[i]:pred[i]=model.predict(np.clip((xx[i:i+1]-mean)/scale,-5,5))[0]*vol[i]
    return pred,fits

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol={'registered_at':datetime.now(timezone.utc).isoformat(),'primary':PRIMARY,
        'configs':[name_of(*c) for c in CONFIGS],'extra':['EQUAL_RSRS_MODELS','A_PLUS_EQUAL_RSRS'],
        'candidate_accounts':40,'control_replays':4,'window':756,'minimum_rows':252,
        'refit':'first observed close each month plus preceding decision dates of evaluation periods',
        'base_features':list(FEATURES)+['rv20'],'rsrs':'N18 M252 beta,R2,z,right_score,z_change5',
        'M_choice':'one trading year, enabling a common training pool before 2015; not selected on this batch returns',
        'horizons':[5,20],'target':'next open to H+1 open economic dividend-adjusted net return with original STRESS fills/fees',
        'maturity':'origin+H+1<=fit index; same complete rows for paired models',
        'scaling':'training-only predictor mean/std, clip5; target divided by known RV20*sqrt(H), training clip3',
        'ridge':{'alpha':100.,'solver':'svd'},
        'HGB':{'max_iter':100,'max_leaf_nodes':4,'max_depth':2,'min_samples_leaf':63,'l2_regularization':10.,
               'learning_rate':.05,'early_stopping':False,'random_state':510300},
        'weights':'.5 if predicted after-cost return>0 else0; unknown remainsNaN',
        'equal':'equal four RSRS models when all known',
        'A_plus_equal':'.5*original A target+.5*equal-models target',
        'execution':'unchanged original A_CONTROL with all original risks,fees,lots,T+1',
        'periods':base.PERIODS,'previous_batch_observed':True,'independent_validation':'NOT_ESTABLISHED_HISTORY_ALREADY_USED',
        'goal_achieved':False,'orders_authorized':False,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    base.write(OUT/'protocol.json',protocol)
    d,div,risks,parents=base.load();x=make_features(d)
    labels={h:label(d,h) for h in [5,20]};preds={};audits={}
    with threadpool_limits(limits=1):
        for c in CONFIGS:
            name=name_of(*c);p,f=predict(d,x,labels[c[1]],*c);preds[name]=p;audits[name]=f
            print('FIT_COMPLETE',name,len(f),flush=True)
        cut=2300;small=d.iloc[:cut].reset_index(drop=True)
        pp,_=predict(small,make_features(small),label(small,5),'HGB',5,True,collect=False)
        np.testing.assert_allclose(pp,preds[PRIMARY][:cut],atol=1e-12,rtol=0,equal_nan=True)
    base.write(OUT/'all_fit_audits.json',audits)
    pd.DataFrame(preds).assign(date=d.date).to_csv(OUT/'all_predictions.csv',index=False)
    weights={k:np.where(np.isfinite(p),np.where(p>0,.5,0.),np.nan) for k,p in preds.items()}
    equal=np.mean(np.array([v for k,v in weights.items() if k.endswith('_RSRS')]),axis=0)
    rows=[];checks=[];annual=[]
    for period,(start,end) in base.PERIODS.items():
        local=d[d.date.le(end)].reset_index(drop=True);n=len(local);sig=base.empty_signals(local)
        a=parents[period].set_index('origin')[base.PARENT_A].reindex(pd.DatetimeIndex(d.date)).to_numpy()
        targets={**weights,'EQUAL_RSRS_MODELS':equal,'A_PLUS_EQUAL_RSRS':.5*a+.5*equal}
        for cost in ['BASE','STRESS']:
            ar=base.account(local,div,parents[period],risks,sig,cost,start,'A_CONTROL')
            ref=base.read_frame(base.CONTROL/period/cost/'A_SAVED_WEIGHT/daily.parquet')
            err=float(np.max(np.abs(ar['daily'].equity.to_numpy()-ref.equity.to_numpy())))
            assert err<1e-7;checks.append({'period':period,'cost':cost,'max_nav_error':err});astat=base.metrics(ar)
            for name,w in targets.items():
                par=pd.DataFrame({'origin':local.date,base.PARENT_A:w[:n]})
                res=base.account(local,div,par,risks,sig,cost,start,'A_CONTROL')
                s=base.metrics(res);row={'period':period,'cost':cost,'policy':name,**s,**base.verify(res)}
                row['absolute_gate']=base.evaluate_gate(row)
                row['beats_A_both']=bool(s['net_cagr']>astat['net_cagr'] and s['net_sharpe']>astat['net_sharpe'])
                rows.append(row);base.dump_result(OUT/'accounts'/period/cost/name,res)
                for year,g in res['daily'].groupby(res['daily'].date.dt.year):
                    rr=g.net_return.to_numpy();sd=rr.std(ddof=1)
                    annual.append({'period':period,'cost':cost,'policy':name,'year':int(year),'days':len(g),
                        'return':float(np.prod(1+rr)-1),'sharpe':float(rr.mean()/sd*np.sqrt(252)) if sd>1e-14 else np.nan})
                print('ACCOUNT',period,cost,name,round(s['net_cagr'],6),round(s['net_sharpe'],6),flush=True)
                pd.DataFrame(rows).to_csv(OUT/'all_metrics.csv',index=False)
    table=pd.DataFrame(rows);passed=[];paired=[]
    for name,g in table.groupby('policy'):
        if len(g)==4 and g.absolute_gate.all() and g.beats_A_both.all():passed.append(name)
    for kind,h in [('RIDGE',5),('RIDGE',20),('HGB',5),('HGB',20)]:
        for period in base.PERIODS:
            for cost in ['BASE','STRESS']:
                get=lambda u:table.loc[table.policy.eq(name_of(kind,h,u))&table.period.eq(period)&table.cost.eq(cost)].iloc[0]
                a,b=get(False),get(True)
                paired.append({'kind':kind,'horizon':h,'period':period,'cost':cost,
                    'cagr_delta':float(b.net_cagr-a.net_cagr),'sharpe_delta':float(b.net_sharpe-a.net_sharpe)})
    base.write(OUT/'summary.json',{'completed_at':datetime.now(timezone.utc).isoformat(),'candidate_accounts':40,
        'control_replays':checks,'actual_model_fits':sum(len(v) for v in audits.values()),
        'prefix_check_primary_passed':True,'all_four_passed_policies':passed,'paired_increments':paired,
        'independent_validation':'NOT_ESTABLISHED_HISTORY_ALREADY_USED','goal_achieved':False,'all_results':rows,'orders_authorized':False})
    pd.DataFrame(annual).to_csv(OUT/'annual_returns.csv',index=False)
    print('CONDITIONAL_BATCH_COMPLETE',len(rows),passed,flush=True)

if __name__=='__main__':main()
