"""跨年度条件分布与实盘资格研究。只读原输入，不产生交易授权。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.adaptive_allocation_v1 import normalize_dividends, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import write_json, digest, now, require, Account, execute_order
from tools.research.annual10_capital_20260910 import read_frame, aligned, accounting
OUT = ROOT/'research_runs/stability_distribution_20260910'
CONFIG = ROOT/'config/510300_continuous_reference_min_variance_v1.json'
OLD = ROOT/'research_runs/annual10_sequential_20260910'
OLD_MODEL = 'C__IMMEDIATE__H5__FULL__MAX'
MODELS = ['PRIMARY_YEAR_ROBUST', 'POINT_CONTROL']
REMOTE_PROTOCOL = 'a4ebe512744f930ee4f6246980094fd803441eaa'
PARAM = dict(horizon=5, embargo=5, label_anchor=252, train_window=1260,
             min_train=80, min_years=3, min_year_rows=10, k_min=20,
             gamma=4., target_vol=.10, weights=[0.,.25,.5,.75,1.],
             annual_days=242, bootstrap_block=63, bootstrap_reps=2000,
             bootstrap_seed=20260910, rolling_window=726)
FEATURES = ['mom5_risk', 'mom20_risk', 'mom60_risk', 'drawdown60_risk',
            'session_balance60', 'vol_ratio', 'downside_ratio', 'volume_surprise',
            'signed_participation5', 'range_risk', 'close_location', 'efficiency20']

def load():
    cfg=json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    frame=read_frame(ROOT/cfg['features'])
    div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    return cfg,frame,div

def features(data):
    r=data.total_simple.astype(float); s60=r.rolling(60).std(ddof=1)
    s20=r.rolling(20).std(ddof=1)
    vr=data.volume/data.volume.shift(1).rolling(20).mean()
    diff=data.intraday_log-data.overnight_log
    values={f'mom{h}_risk':data.total_log.rolling(h).sum()/(s60*np.sqrt(h)) for h in [5,20,60]}
    values.update(drawdown60_risk=data.dd60/(s60*np.sqrt(60)),
                  session_balance60=diff.rolling(60).sum()/(diff.rolling(60).std(ddof=1)*np.sqrt(60)),
                  vol_ratio=s20/s60,
                  downside_ratio=np.sqrt(r.clip(upper=0).pow(2).rolling(20).mean())/s20,
                  volume_surprise=np.log(vr),
                  signed_participation5=(r*vr).rolling(5).sum()/(s60*np.sqrt(5)),
                  range_risk=(data.high-data.low)/(data.previous_close*s60),
                  close_location=(data.close-data.low)/(data.high-data.low).replace(0,np.nan),
                  efficiency20=data.efficiency20)
    out=pd.DataFrame(values)[FEATURES]
    out=out.replace([np.inf,-np.inf],np.nan)
    return out,s60.to_numpy(float),s20.to_numpy(float)*np.sqrt(242)

def build_labels(data,div):
    # 登记日与除息日关系先验证，确保每份标签的分红边界没有假设错误。
    dates=pd.DatetimeIndex(data.date)
    for d in div.itertuples():
        if d.record_date in dates and d.ex_date in dates:
            require(dates.get_loc(d.ex_date)==dates.get_loc(d.record_date)+1,'登记/除息不是相邻交易日，禁止近似标签')
    x,sigma,_=features(data)
    rows=[]
    for t in range(PARAM['label_anchor'],len(data)-PARAM['horizon']-1,PARAM['horizon']):
        buy=t+1; sell=t+6
        amount=float(data.dividend.iloc[buy+1:sell+1].sum())
        gross=(float(data.open.iloc[sell])-float(data.open.iloc[buy])+amount)/float(data.open.iloc[buy])
        norm=sigma[t]*np.sqrt(5)
        rows.append(dict(origin_index=t,origin=data.date.iloc[t],buy_index=buy,exit_index=sell,
                         exit_date=data.date.iloc[sell],eligible_index=sell+PARAM['embargo'],
                         year=int(data.date.iloc[t].year),gross_return=gross,
                         normalized_return=gross/norm if norm>0 else np.nan,
                         label_dividend_per_share=amount,feature_valid=bool(np.isfinite(x.iloc[t]).all())))
    out=pd.DataFrame(rows)
    require((out.buy_index.to_numpy()[1:]>=out.exit_index.to_numpy()[:-1]).all(),'持有区间重叠')
    return out

def choose_distribution(train_x,train_y,years,current_x,current_risk,current_vol,close,cfg):
    mean=np.median(train_x,axis=0)
    scale=1.4826*np.median(np.abs(train_x-mean),axis=0)
    scale=np.where(scale>1e-12,scale,1.)
    xx=np.clip((train_x-mean)/scale,-5,5);cur=np.clip((current_x-mean)/scale,-5,5)
    distance=np.square(xx-cur).sum(axis=1)
    k=min(len(train_x),max(PARAM['k_min'],int(np.ceil(2*np.sqrt(len(train_x))))))
    # 输入按时间排序；stable确保等距离以原日期顺序打破平局。
    sel=np.argsort(distance,kind='stable')[:k]
    h=max(float(distance[sel[-1]]),1e-12)
    p=np.exp(-.5*distance[sel]/h);p/=p.sum()
    y=train_y[sel]*current_risk*np.sqrt(5)
    point=float(p@y);var=float(p@np.square(y-point)/(1-np.square(p).sum()))
    leave=[]
    for year in sorted(set(years.tolist())):
        keep=years[sel]!=year
        if int(keep.sum())<10:
            leave.append(dict(year=int(year),mean=None,retained=int(keep.sum())))
        else:
            leave.append(dict(year=int(year),mean=float(p[keep]@y[keep]/p[keep].sum()),retained=int(keep.sum())))
    robust=min([point]+[v['mean'] for v in leave]) if all(v['mean'] is not None for v in leave) else None
    risk_cap=min(1.,.10/current_vol)
    values=sorted(set(min(w,risk_cap) for w in PARAM['weights']))
    choices={};objectives={}
    for name,mu in [('POINT_CONTROL',point),('PRIMARY_YEAR_ROBUST',robust)]:
        if mu is None:choices[name]=np.nan;objectives[name]=[];continue
        scores=[]
        for w in values:
            stress=cfg['costs']['STRESS']
            friction=0. if w==0 else 2*max(w*stress['commission'],stress['minimum']/cfg['initial_capital'])+2*w*(stress['slippage']+cfg['tick']/close)
            score=w*mu-.5*PARAM['gamma']*w*w*var-friction
            scores.append(dict(weight=w,estimated_roundtrip_cost=friction,utility=score))
        best=max(scores,key=lambda v:v['utility'])
        choices[name]=best['weight'];objectives[name]=scores
    return choices,dict(point_mean=point,robust_mean=robust,conditional_variance=var,
                        effective_neighbors=float(1/np.square(p).sum()),neighbor_count=k,
                        risk_cap=risk_cap,leave_year_out=leave,objectives=objectives,
                        feature_median=mean.tolist(),feature_scale=scale.tolist()),sel,p

def predict_all(data,labels,cfg):
    x,sigma,vol=features(data);xx=x.to_numpy(float)
    anchor=int(np.flatnonzero(data.date>=pd.Timestamp(cfg['earlier_start']))[0])-1
    main=int(np.flatnonzero(data.date>=pd.Timestamp(cfg['evaluation_start']))[0])-1 if data.date.iloc[-1]>=pd.Timestamp(cfg['evaluation_start']) else None
    schedule=set(range(anchor,len(data)-1,5))
    if main is not None:schedule.add(main)
    targets={name:np.full(len(data),np.nan) for name in MODELS}
    mask=np.zeros(len(data),bool);records=[];neighbors=[]
    for t in sorted(schedule):
        mask[t]=True
        z=labels[(labels.eligible_index<=t)&(labels.origin_index>=t-PARAM['train_window'])&labels.feature_valid].copy()
        z=z[np.isfinite(z.normalized_return)]
        counts=z.groupby('year').size()
        row=dict(origin_index=t,origin=data.date.iloc[t],training_rows=len(z),
                 years_with_ten=int((counts>=10).sum()),status='NO_VIEW_INSUFFICIENT_HISTORY')
        if len(z)>=80 and (counts>=10).sum()>=3:
            if np.isfinite(xx[t]).all() and np.isfinite([sigma[t],vol[t]]).all() and min(sigma[t],vol[t])>0:
                ids=z.origin_index.to_numpy(int)
                require(int(z.eligible_index.max())<=t,'标签提前使用')
                choice,stats,sel,p=choose_distribution(xx[ids],z.normalized_return.to_numpy(float),z.year.to_numpy(int),xx[t],sigma[t],vol[t],float(data.close.iloc[t]),cfg)
                row.update(stats,status='ESTIMATE_AVAILABLE',latest_label_exit=int(z.exit_index.max()),
                           latest_label_eligible=int(z.eligible_index.max()),training_first=int(ids.min()),training_last=int(ids.max()))
                for name in MODELS:targets[name][t]=choice[name];row[name]=choice[name]
                for j,w in zip(sel,p):
                    v=z.iloc[j]
                    neighbors.append(dict(decision_origin_index=t,source_origin_index=int(v.origin_index),source_exit_index=int(v.exit_index),source_eligible_index=int(v.eligible_index),weight=float(w),source_year=int(v.year),normalized_return=float(v.normalized_return)))
            else:row['status']='NO_VIEW_CURRENT_FEATURES'
        records.append(row)
    return targets,mask,records,pd.DataFrame(neighbors)

def metrics_and_checks(led,cfg):
    info=accounting(led,cfg)
    before=np.r_[0,led.shares.to_numpy()[:-1]]
    require(np.array_equal(led.shares.to_numpy()-before,led.filled_quantity.to_numpy()),'股数变化与成交不等')
    require((np.maximum(-led.filled_quantity.to_numpy(),0)<=before).all(),'卖出超过昨日日终库存')
    np.testing.assert_allclose(led.price_pnl+led.dividend_recognized-led.commission-led.slippage_cost,led.pnl,atol=1e-7,rtol=0)
    m=summarize(led,cfg)
    m['terminal_equity']=float(led.equity.iloc[-1]);m['point_met']=bool(m['annualized_return']>=.10 and m['net_sharpe'] is not None and m['net_sharpe']>=1.2)
    return m,info

def rolling(led):
    r=led.net_return.astype(float);window=PARAM['rolling_window']
    out=pd.DataFrame(dict(date=led.date,cagr=np.expm1(np.log1p(r).rolling(window).sum()*242/window),
                          sharpe=r.rolling(window).mean()/r.rolling(window).std(ddof=1)*np.sqrt(242)))
    return out.iloc[window-1:].reset_index(drop=True)

def bootstrap(led):
    r=led.net_return.to_numpy(float);n=len(r);rng=np.random.default_rng(PARAM['bootstrap_seed']);vals=[]
    for start in range(0,PARAM['bootstrap_reps'],100):
        b=min(100,PARAM['bootstrap_reps']-start)
        ids=(rng.integers(0,n,size=(b,math.ceil(n/63),1))+np.arange(63)[None,None,:])%n
        a=r[ids.reshape(b,-1)[:,:n]]
        c=np.expm1(np.log1p(a).sum(axis=1)*242/n)
        sd=a.std(axis=1,ddof=1)
        s=np.divide(a.mean(axis=1)*np.sqrt(242),sd,out=np.full(b,np.nan),where=sd>1e-15)
        vals.extend(zip(c,s))
    a=np.asarray(vals)
    return dict(cagr_quantiles=np.nanquantile(a[:,0],[.025,.5,.975]).tolist(),sharpe_quantiles=np.nanquantile(a[:,1],[.025,.5,.975]).tolist() if np.isfinite(a[:,1]).any() else [None]*3,
                conditional_only=True,selection_adjusted=False,block=63,repetitions=2000)

def concentration(led):
    prev=0;cycle=None;rows=[]
    for r in led.itertuples():
        if cycle is None and r.shares>0:cycle=dict(entry=str(r.date.date()),profit=0.,holding_days=0)
        if cycle is not None:
            cycle['profit']+=r.pnl;cycle['holding_days']+=int(r.shares>0)
            if prev>0 and r.shares==0:cycle['exit']=str(r.date.date());rows.append(cycle);cycle=None
        prev=r.shares
    net=float(led.pnl.sum());wins=sorted([x['profit'] for x in rows if x['profit']>0],reverse=True)
    return rows,dict(cycles=len(rows),net_profit=net,top_three_profit_share=(sum(wins[:3])/net if net>0 else None),outside_cycle_pnl=net-sum(x['profit'] for x in rows))

def verify_functions(cfg,data,div,labels):
    tests=[]
    def ok(name):tests.append(dict(test=name,status='PASS'))
    require((labels.buy_index.to_numpy()[1:]>=labels.exit_index.to_numpy()[:-1]).all(),'标签重叠');ok('nonoverlapping_holding_intervals')
    require((labels.eligible_index==labels.exit_index+5).all(),'隔离期错误');ok('five_session_embargo')
    # 采用独立原引擎、零费用买持，检查每份分红标签及整数份额折算。
    ids=set(labels[labels.label_dividend_per_share.gt(0)].index.tolist()[:5]+[0,80,200,400])
    for ix in sorted(ids):
        z=labels.iloc[ix];frame=data.iloc[:int(z.exit_index)+1].copy();t=int(z.origin_index)
        zero=dict(commission=0.,minimum=0.,slippage=0.)
        led,_=simulate_event_account(frame,div,cfg,zero,str(frame.date.iloc[t+1].date()),'BUY_HOLD',event_mask=np.zeros(len(frame),bool))
        invested=float(led.notional.iloc[0]);expected=cfg['initial_capital']+invested*float(z.gross_return)
        require(abs(float(led.equity.iloc[-1])-expected)<1e-7,'每份标签与真实分红权益不符')
    ok('unit_labels_with_dividends_replayed_'+str(len(ids)))
    # 合成相近状态，单一暴利年份不能支配稳健均值。
    x=np.zeros((100,12));y=np.r_[np.ones(30)*2,np.ones(30)*-.02,np.zeros(40)];years=np.repeat([2013,2014,2015,2016],[30,30,20,20])
    # 打散日期顺序，使选入的近邻包含多个年份。
    ids=np.argsort(np.arange(100)%10,kind='stable');choices,st,_,_=choose_distribution(x,y[ids],years[ids],np.zeros(12),.01,.2,4.,cfg)
    require(st['robust_mean'] is None or st['robust_mean']<=st['point_mean']+1e-15,'留年下界超过均值');ok('leave_year_out_never_increases_mean')
    # 没有新模型时留空，不伪造成0。
    t,mask,rr,nn=predict_all(data.iloc[:820].copy(),labels[labels.exit_index<820],cfg)
    require(all(np.isnan(v[:632]).all() for v in t.values()),'准备区间被填造');ok('unknown_is_not_cash_signal')
    for cut in [1100,1900,2700,3300]:
        pref=data.iloc[:cut+1].copy();ll=build_labels(pref,div)
        t1,_,r1,n1=predict_all(pref,ll,cfg)
        alt=data.copy();alter=np.arange(len(alt))>cut
        # 改变未来原始价格及收益字段，验证此前输入、标签和目标不变。
        for col in ['open','high','low','close','volume','total_simple','total_log','intraday_log','overnight_log']:
            alt.loc[alter,col]*=1.7
        t2,_,r2,n2=predict_all(alt,build_labels(alt,div),cfg)
        for name in MODELS:np.testing.assert_allclose(t1[name][:-1],t2[name][:cut],rtol=0,atol=0,equal_nan=True)
        ok('future_suffix_does_not_change_past_'+str(cut))
    a=Account(200000.)
    c=cfg['costs']['BASE'];execute_order(a,1000,4.,4.,0.,3,c,cfg)
    before=a.shares;z=execute_order(a,-1000,4.,4.,0.,3,c,cfg)
    require(z['filled_quantity']==0 and a.shares==before,'同日新买可卖');ok('T_plus_one_blocks_same_day_sell')
    z=execute_order(a,-1000,3.6,4.,0.,4,c,cfg);require(z['filled_quantity']==0,'跌停卖出未受阻');ok('directional_limit_blocks_sell')
    return tests

def freeze():
    require(not OUT.exists(),'本轮输出目录已存在，禁止覆盖')
    cfg,data,div=load();labels=build_labels(data,div)
    tests=verify_functions(cfg,data,div,labels)
    paths=[Path(__file__),CONFIG,ROOT/cfg['features'],ROOT/cfg['dividends'],ROOT/'tools/research/annual10_capital_20260910.py',
           ROOT/'research/adaptive_allocation_v1.py',ROOT/'research/event_clock_account_v1.py',ROOT/'research/intraday_overnight_increment_v1.py']
    for period in ['evaluation','earlier_diagnostic']:
        for cost in cfg['costs']:
            paths += [OLD/period/cost/f'{OLD_MODEL}_{s}.parquet' for s in ['ledger','decisions']]
    OUT.mkdir(parents=True)
    write_json(OUT/'protocol.json',dict(study_id='510300_STABILITY_FIRST_DISTRIBUTION_20260910',registered_at=now(),remote_protocol_commit=REMOTE_PROTOCOL,
        source_commit='7b773765575ec0f5a778254a1b84bc974b08fc3f',parameters=PARAM,features=FEATURES,models=MODELS,
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest(p)) for p in paths],
        historical_selection='PREVIOUSLY_OBSERVED_HISTORY',independent_validation='NOT_ESTABLISHED',position_impact=0,live_trading_authorized=False),exclusive=True)
    write_json(OUT/'tests_receipt.json',dict(passed=True,tests=tests,count=len(tests)))
    labels.to_csv(OUT/'labels.csv',index=False)
    print('FROZEN',len(tests),'TESTS',len(labels),'LABELS',flush=True)

def verify():
    pro=json.loads((OUT/'protocol.json').read_text())
    for row in pro['files']:
        p=ROOT/row['path'];require(p.stat().st_size==row['bytes'] and digest(p)==row['sha256'],'来源变动：'+row['path'])

def run():
    verify();require(not (OUT/'RUN_STARTED.json').exists(),'已经开始，不覆盖')
    write_json(OUT/'RUN_STARTED.json',dict(started_at=now()),exclusive=True)
    cfg,data,div=load();labels=build_labels(data,div)
    targets,mask,records,neighbors=predict_all(data,labels,cfg)
    pd.DataFrame(dict(date=data.date,scheduled=mask,**targets)).to_parquet(OUT/'targets.parquet',index=False)
    write_json(OUT/'estimates.json',dict(records=records))
    neighbors.to_csv(OUT/'neighbors.csv',index=False)
    rows=[];checks=[];yr=[];rolls=[];conc=[];boots=[];replays=[]
    for period,start,end in [('evaluation',cfg['evaluation_start'],cfg['data_cutoff']),('earlier_diagnostic',cfg['earlier_start'],cfg['earlier_terminal'])]:
        frame=data[data.date<=pd.Timestamp(end)].copy();n=len(frame)
        for cost,fc in cfg['costs'].items():
            oldled=read_frame(OLD/period/cost/f'{OLD_MODEL}_ledger.parquet')
            olddec=read_frame(OLD/period/cost/f'{OLD_MODEL}_decisions.parquet')
            oldtarget=aligned(frame,olddec)
            for name in [*MODELS,'OLD_POINT_CANDIDATE','BUY_HOLD']:
                ta=oldtarget if name=='OLD_POINT_CANDIDATE' else targets[name][:n] if name in MODELS else np.ones(n)
                ma=np.ones(n,bool) if name in ['OLD_POINT_CANDIDATE','BUY_HOLD'] else mask[:n]
                delays=[0,1] if name!='BUY_HOLD' else [0]
                for delay in delays:
                    tt=np.r_[np.nan,ta[:-1]] if delay else ta
                    mm=np.r_[False,ma[:-1]] if delay else ma
                    model=name if not delay else name+'__DELAY_ONE_SESSION'
                    led,dec=simulate_event_account(frame,div,cfg,fc,start,model,targets=tt,event_mask=mm)
                    if name=='OLD_POINT_CANDIDATE' and not delay:
                        fields=['equity','cash','shares','commission','slippage_cost','net_return','filled_quantity']
                        np.testing.assert_allclose(led[fields].to_numpy(float),oldled[fields].to_numpy(float),atol=1e-8,rtol=0)
                        replays.append(dict(period=period,cost=cost,status='PASS',max_absolute_difference=float(np.max(np.abs(led[fields].to_numpy(float)-oldled[fields].to_numpy(float))))))
                    folder=OUT/period/cost;folder.mkdir(parents=True,exist_ok=True)
                    led.to_parquet(folder/f'{model}_ledger.parquet',index=False);dec.to_parquet(folder/f'{model}_decisions.parquet',index=False)
                    m,ck=metrics_and_checks(led,cfg);key=dict(period=period,cost=cost,model=model)
                    rows.append({**key,**m});checks.append({**key,**ck})
                    for year,g in led.groupby(led.date.dt.year):yr.append({**key,'year':int(year),**summarize(g,cfg)})
                    roll=rolling(led);roll.to_csv(folder/f'{model}_rolling_3y.csv',index=False)
                    valid=roll.sharpe.notna()
                    rolls.append({**key,'windows':len(roll),'defined_sharpe_windows':int(valid.sum()),'cagr_min':roll.cagr.min(),'cagr_median':roll.cagr.median(),'sharpe_min':roll.sharpe.min(),'sharpe_median':roll.sharpe.median(),'joint_pass_fraction':float(((roll.cagr>=.1)&(roll.sharpe>=1.2)).mean())})
                    cyc,co=concentration(led);pd.DataFrame(cyc).to_csv(folder/f'{model}_cycles.csv',index=False);conc.append({**key,**co})
                    if not delay:boots.append({**key,**bootstrap(led)})
            print('COMPLETED',period,cost,flush=True)
    for fn,val in [('metrics',rows),('account_checks',checks),('yearly',yr),('rolling_summary',rolls),('concentration',conc),('old_replays',replays)]:pd.DataFrame(val).to_csv(OUT/f'{fn}.csv',index=False)
    write_json(OUT/'conditional_intervals.json',dict(results=boots))
    m=pd.DataFrame(rows);base=m[~m.model.str.contains('__DELAY')&m.model.ne('BUY_HOLD')]
    ready=[]
    for name in [*MODELS,'OLD_POINT_CANDIDATE']:
        g=base[base.model==name]
        main=g[g.period=='evaluation'];all_hist=bool(len(g)==4 and g.point_met.all())
        ready.append(dict(model=name,historical_all_four_target_met=all_hist,main_both_costs_met=bool(main.point_met.all()),
                          independent_post_freeze_observations=0,independent_validation=False,verified_broker_execution=False,
                          validated_strategy_goal_achieved=False,live_ready=False,
                          state='BLOCKED_HISTORICAL_AND_INDEPENDENT_EVIDENCE' if not all_hist else 'BLOCKED_INDEPENDENT_EVIDENCE'))
    write_json(OUT/'result.json',dict(study_id='510300_STABILITY_FIRST_DISTRIBUTION_20260910',status='RESEARCH_EXECUTED_STABLE_TARGET_NOT_ESTABLISHED',completed_at=now(),
        new_model_policies=2,new_primary_accounts=8,delay_diagnostic_accounts=12,buy_hold_replays=4,old_candidate_replays=4,
        saved_accounts=len(rows),independent_bookkeeping_checks=len(checks),conditional_distribution_estimation_origins=sum(r['status']=='ESTIMATE_AVAILABLE' for r in records),
        label_count=len(labels),neighbor_audit_rows=len(neighbors),goals=dict(cagr=.10,sharpe=1.2),readiness=ready,
        historical_selection_adjusted=False,independent_validation='NOT_ESTABLISHED',goal_achieved=False,live_trading_authorized=False,position_impact=0,
        protocol_sha256=digest(OUT/'protocol.json'),code_sha256=digest(Path(__file__))))
    print(m[['period','cost','model','annualized_return','net_sharpe','max_drawdown','mean_exposure']].to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','run','verify'])
    action=parser.parse_args().action
    {'freeze':freeze,'run':run,'verify':verify}[action]()
