"""全结论驱动：点时成员市场beta中性领涨信息。只做冻结历史研究，不连接券商。"""
from __future__ import annotations
import argparse, hashlib, json, math, os, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.adaptive_allocation_v1 import summarize, normalize_dividends
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, write_json, require

OUT = ROOT / 'research_runs/conclusion_beta_iteration_20260910'
RET = 'data/curated/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1/classified_constituent_returns_remediated.parquet'
MEM = 'data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/000300_daily_pit_membership_20150101_20260814.parquet'
FEATURE = 'reports/research/510300_adaptive_allocation_v1/features.parquet'
DIV = 'data/reference/510300_dividends.csv'
PARENT = 'config/510300_continuous_reference_min_variance_v1.json'
POLICIES = [(x,s) for x in ('RESIDUAL','RAW','PRICE') for s in ('FULL','VOL10')]


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def load_inputs():
    market = pd.read_parquet(ROOT/FEATURE)
    market['date'] = pd.to_datetime(market.date).astype('datetime64[ns]')
    require(market.date.is_monotonic_increasing and market.date.is_unique,'ETF日期重复或无序')
    returns = pd.read_parquet(ROOT/RET)
    returns['date'] = pd.to_datetime(returns.date).astype('datetime64[ns]')
    returns['symbol'] = returns.symbol.astype(str)
    members = pd.read_parquet(ROOT/MEM)
    members['date'] = pd.to_datetime(members.membership_date).astype('datetime64[ns]')
    members['symbol'] = members.symbol.astype(str)
    require(not returns.duplicated(['date','symbol']).any(),'股票收益复合键重复')
    require(not members.duplicated(['date','symbol']).any(),'点时成员重复')
    require(members.groupby('date').size().eq(300).all(),'点时成员不为300')
    q=returns.loc[returns.return_is_usable,'daily_total_shareholder_return'].to_numpy(float)
    require(np.isfinite(q).all() and (q>-1).all(),'声明有效的股票收益不合法')
    returns['usable_return']=returns.daily_total_shareholder_return.where(returns.return_is_usable)
    dates=pd.DatetimeIndex(market.date)
    stocks=returns.pivot(index='date',columns='symbol',values='usable_return').reindex(dates).sort_index(axis=1)
    mem=members.assign(member=True).pivot(index='date',columns='symbol',values='member').reindex(index=dates,columns=stocks.columns).eq(True)
    receipt={'stock_rows':len(returns),'stock_symbols':returns.symbol.nunique(),'membership_rows':len(members),
             'membership_days':members.date.nunique(),'etf_rows':len(market),'stock_first':returns.date.min(),
             'stock_last':returns.date.max(),'source_state_counts':returns.constituent_return_state.value_counts().to_dict(),
             'future_return_targets_used_for_source_admission':False,
             'publication_vintage':'RETROSPECTIVE_OFFICIAL_REPLAY_NOT_CONTEMPORANEOUS_SNAPSHOT_PROOF',
             'market_return_column':'total_simple',
             'large_usable_return_rows_in_estimation_universe':int((returns.return_is_usable & returns.usable_return.abs().gt(.30)).sum()),
             'large_usable_return_current_member_rows':int(returns.merge(members[['date','symbol']],on=['date','symbol']).usable_return.abs().gt(.30).sum())}
    return market,stocks,mem,receipt


def lagged_beta(stocks: np.ndarray, market: np.ndarray, window=126, minimum=80):
    """配对缺失只从该股票滚动回归中排除。所有滚动量严格滞后一日。"""
    x=np.asarray(stocks,float); m=np.asarray(market,float)
    require(x.ndim==2 and len(m)==len(x),'beta输入形状错误')
    valid=np.isfinite(x)&np.isfinite(m[:,None]); a=np.where(valid,x,0.); b=np.where(valid,m[:,None],0.)
    def roll(v): return pd.DataFrame(v).rolling(window,min_periods=1).sum().shift(1).to_numpy(float)
    n=roll(valid.astype(float)); sx=roll(a); sy=roll(b); sxy=roll(a*b); syy=roll(b*b)
    with np.errstate(divide='ignore',invalid='ignore'):
        den=syy-sy*sy/n
        beta=(sxy-sx*sy/n)/den
        var=den/(n-1)
    beta[(n<minimum)|(var<=1e-8)|~np.isfinite(beta)]=np.nan
    return beta,n


def group_factor(stock_return, beta, prior_members, market_return, min_members=270):
    """组别由上一时点信息形成；实际有效成员的回报和beta严格配对。"""
    s=np.asarray(stock_return,float);b=np.asarray(beta,float);members=np.asarray(prior_members,bool)
    eligible=np.flatnonzero(members&np.isfinite(b))
    result={'beta_eligible':len(eligible),'group_n':0,'high_valid':0,'low_valid':0,'raw':np.nan,'residual':np.nan,
            'high_beta':np.nan,'low_beta':np.nan,'high_return':np.nan,'low_return':np.nan,'factor_status':'NO_VIEW_BETA_COVERAGE'}
    if len(eligible)<min_members or not np.isfinite(market_return): return result,[],[]
    order=eligible[np.argsort(b[eligible],kind='stable')];k=len(order)//3
    lo,hi=order[:k],order[-k:]; lv=lo[np.isfinite(s[lo])];hv=hi[np.isfinite(s[hi])]
    result.update(group_n=k,high_valid=len(hv),low_valid=len(lv))
    if min(len(lv),len(hv))<math.ceil(.9*k):
        result['factor_status']='NO_VIEW_CURRENT_RETURN_COVERAGE';return result,hi.tolist(),lo.tolist()
    hr,lr=float(s[hv].mean()),float(s[lv].mean());hb,lb=float(b[hv].mean()),float(b[lv].mean())
    result.update(raw=hr-lr,residual=float((s[hv]-b[hv]*market_return).mean()-(s[lv]-b[lv]*market_return).mean()),
                  high_beta=hb,low_beta=lb,high_return=hr,low_return=lr,factor_status='AVAILABLE')
    return result,hi.tolist(),lo.tolist()


def build_factors(market,stocks,mem,save_groups=False):
    beta,count=lagged_beta(stocks.to_numpy(float),market.total_simple.to_numpy(float))
    prior=mem.shift(1,fill_value=False).to_numpy(bool); rows=[];groups=[]
    values=stocks.to_numpy(float);codes=stocks.columns.to_numpy()
    for t in range(len(market)):
        rec,hi,lo=group_factor(values[t],beta[t],prior[t],float(market.total_simple.iloc[t]))
        rows.append({'date':market.date.iloc[t],**rec})
        if save_groups and hi:
            groups.append({'date':market.date.iloc[t],'high_codes':'|'.join(codes[hi]),'low_codes':'|'.join(codes[lo]),
                           'group_information_end':market.date.iloc[t-1]})
    f=pd.DataFrame(rows)
    f['RESIDUAL']=f.residual.rolling(20,min_periods=20).sum()
    f['RAW']=f.raw.rolling(20,min_periods=20).sum()
    f['PRICE']=np.log1p(market.total_simple).rolling(20,min_periods=20).sum()
    f['common_valid']=np.isfinite(f[['RESIDUAL','RAW','PRICE']]).all(axis=1)
    f['vol20']=pd.Series(market.total_simple).rolling(20,min_periods=20).std(ddof=1)*np.sqrt(242)
    f['update']=market.date.dt.to_period('M').ne(market.date.shift(1).dt.to_period('M'))
    for representation,scale in POLICIES:
        v=np.where(f[representation]>0,1.,0.)
        valid=f.common_valid.to_numpy(bool)
        if scale=='VOL10':
            positive=v>0;vol=f.vol20.to_numpy(float);good=np.isfinite(vol)&(vol>0)
            valid=valid&(~positive|good)
            multiplier=np.divide(.1,vol,out=np.ones(len(vol)),where=good)
            v=v*np.minimum(1,multiplier)
        f[representation+'_'+scale]=np.where(valid & f['update'].to_numpy(bool),v,np.nan)
    return f,pd.DataFrame(groups),{'finite_beta_estimates':int(np.isfinite(beta).sum()),'daily_factor_available':int(f.factor_status.eq('AVAILABLE').sum()),
                                  'common_signal_days':int(f.common_valid.sum()),'available_month_origins':int((f.common_valid&f['update']).sum()),
                                  'first_common_signal_date':f.loc[f.common_valid,'date'].min()}


def basic_tests():
    rng=np.random.default_rng(313);m=rng.normal(0,.015,200);true=np.array([.5,1.,2.]);x=m[:,None]*true
    b,n=lagged_beta(x,m,126,80);np.testing.assert_allclose(b[130:],np.tile(true,(70,1)),atol=1e-12)
    altered=x.copy();altered[150:]*=100
    bp,_=lagged_beta(altered,m,126,80);np.testing.assert_allclose(b[:151],bp[:151],equal_nan=True,atol=0,rtol=0)
    beta=np.linspace(.1,2.5,300);ret=beta*.01
    g,hi,lo=group_factor(ret,beta,np.ones(300,bool),.01);assert abs(g['residual'])<1e-15 and g['raw']>0
    missing=ret.copy();missing[hi[:11]]=np.nan
    bad,_,_=group_factor(missing,beta,np.ones(300,bool),.01);assert np.isnan(bad['residual'])
    g,_,_=group_factor(ret,beta,np.zeros(300,bool),.01);assert g['beta_eligible']==0 and np.isnan(g['raw'])
    xx=x.copy();xx[:130,0]=np.nan;bb,nn=lagged_beta(xx,m);assert np.isnan(bb[:200,0]).all()
    return {'checks':6,'passed':True,'check_names':['linear_beta_identity','current_and_future_do_not_change_prior_beta','market_beta_cancelled','10pct_missing_group_fails','unknown_members_not_backfilled','insufficient_pairs_no_beta']}


def freeze():
    OUT.mkdir(parents=True,exist_ok=True);require(not (OUT/'protocol.json').exists(),'协议已存在，不覆盖')
    tests=basic_tests();write_json(OUT/'tests_receipt.json',tests)
    parent=load_json(ROOT/PARENT)
    cfg={k:parent[k] for k in ['initial_capital','lot','tick','limit_fraction','annual_days','cash_annual_rate_assumption','high_sharpe_target','costs','weight_band']}
    cfg.update(study_id='510300_BETA_NEUTRAL_LEADERSHIP_V1_20260910',registered_at=now(),remote_preregistration_commit='b6e36e1a5f48ddcc5bfc728e91ea36b7f7962992',
               primary='RESIDUAL_FULL',window=126,min_pairs=80,min_members=270,group_fraction='one_third',group_return_coverage=.90,signal_window=20,
               update='FIRST_TRADING_DAY_OF_MONTH_CLOSE_NEXT_OPEN',representations=['RESIDUAL','RAW','PRICE'],scales=['FULL','VOL10'],
               policies=[a+'_'+b for a,b in POLICIES],evaluation_start='2020-01-02',data_cutoff='2026-08-14',earlier_start='2015-01-05',earlier_terminal='2019-12-31',
               source_commit='e5433a24e6aecc2e3333ac12e050fbb84f9169ce',source_branch='codex/conclusion-led-iteration-20260910',
               supervision_return_fits=0,estimated_beta_is_not_a_forecast=False,goal_cagr=.10,goal_sharpe=1.2,
               independent_validation='NOT_ESTABLISHED',position_impact=0,live_trading_authorized=False)
    inputs=[FEATURE,RET,MEM,DIV,PARENT,'research/event_clock_account_v1.py','research/adaptive_allocation_v1.py','research/intraday_overnight_increment_v1.py',str(Path(__file__).relative_to(ROOT))]
    cfg['files']=[{'path':p,'bytes':(ROOT/p).stat().st_size,'sha256':digest(ROOT/p)} for p in inputs]
    write_json(OUT/'protocol.json',cfg,exclusive=True);print('FROZEN',digest(OUT/'protocol.json'),'SIX_SYNTHETIC_CHECKS_PASS',flush=True)


def verify_ledger(l,cfg):
    shares=l.shares.to_numpy(float);mark=l.mark.to_numpy(float);cash=l.cash.to_numpy(float);rec=l.dividend_receivable.to_numpy(float)
    eq=l.equity.to_numpy(float);prev=np.r_[cfg['initial_capital'],eq[:-1]];old=np.r_[0,shares[:-1]]
    np.testing.assert_allclose(eq,cash+shares*mark+rec,atol=1e-7,rtol=0)
    np.testing.assert_allclose(l.net_return.to_numpy(float),eq/prev-1,atol=1e-12,rtol=0)
    np.testing.assert_allclose(shares-old,l.filled_quantity.to_numpy(float),atol=0,rtol=0)
    require((cash>=-1e-7).all() and (shares>=0).all() and np.allclose(shares%100,0),'透支/负股数/非整手')
    require(((-l.filled_quantity.to_numpy(float)).clip(0)<=old).all(),'卖出超过隔日结转股份')
    require(shares[-1]==0,'终点未清算')
    r=eq/prev-1;cagr=float(np.expm1(np.log1p(r).sum()*cfg['annual_days']/len(r)))
    sd=np.std(r,ddof=1);sr=float(np.mean(r)/sd*np.sqrt(cfg['annual_days'])) if sd>0 else None
    saved=summarize(l,cfg)
    np.testing.assert_allclose([cagr],[saved['annualized_return']],atol=1e-12,rtol=0)
    if sr is not None:np.testing.assert_allclose(sr,saved['net_sharpe'],atol=1e-12,rtol=0)
    return {'status':'PASS','rows':len(l),'annualized_return':cagr,'net_sharpe':sr,'max_equity_identity_error':float(np.abs(eq-cash-shares*mark-rec).max())}


def bootstrap(a,b,seed=20260910,block=20,reps=2000):
    rng=np.random.default_rng(seed);n=len(a);results=[]
    for first in range(0,reps,100):
        k=min(100,reps-first);idx=(rng.integers(n,size=(k,math.ceil(n/block),1))+np.arange(block)[None,None,:])%n
        ids=idx.reshape(k,-1)[:,:n]
        results.extend((np.expm1(np.log1p(a[ids]).sum(1)*242/n)-np.expm1(np.log1p(b[ids]).sum(1)*242/n)).tolist())
    return np.quantile(results,[.025,.5,.975]).tolist()


def run():
    cfg=load_json(OUT/'protocol.json')
    for item in cfg['files']:require(digest(ROOT/item['path'])==item['sha256'],'冻结输入/代码改变：'+item['path'])
    require(not (OUT/'RUN_STARTED.json').exists(),'本目录运行已开始，不覆盖')
    write_json(OUT/'RUN_STARTED.json',{'started_at':now(),'protocol_sha256':digest(OUT/'protocol.json')},exclusive=True)
    market,stocks,mem,source=load_inputs();write_json(OUT/'source_receipt.json',source)
    f,groups,factor_receipt=build_factors(market,stocks,mem,True)
    f.to_parquet(OUT/'factors.parquet',index=False);f.to_csv(OUT/'factors.csv',index=False)
    groups.to_parquet(OUT/'groups.parquet',index=False);write_json(OUT/'factor_receipt.json',factor_receipt)
    prefix=[]
    for date in ['2016-12-30','2019-12-31','2023-12-29']:
        n=int(market.date.le(date).sum());ff,_,_=build_factors(market.iloc[:n].copy(),stocks.iloc[:n].copy(),mem.iloc[:n].copy())
        pd.testing.assert_frame_equal(f.iloc[:n].reset_index(drop=True),ff.reset_index(drop=True),check_exact=True)
        prefix.append({'through':date,'rows':n,'status':'PASS'})
    write_json(OUT/'prefix_receipt.json',{'checks':prefix,'future_input_read_does_not_change_checked_prefixes':True})
    print('FACTOR',json.dumps(factor_receipt,default=str),'PREFIX_PASS',flush=True)
    div=normalize_dividends(pd.read_csv(ROOT/DIV)); metrics=[]; checks=[];years=[]; coverage=[];rolls=[];boots=[];controls=[]
    for period,start,end in [('evaluation',cfg['evaluation_start'],cfg['data_cutoff']),('earlier_diagnostic',cfg['earlier_start'],cfg['earlier_terminal'])]:
        mask=market.date.le(end);data=market[mask].reset_index(drop=True);fac=f[mask].reset_index(drop=True)
        first=int(np.flatnonzero(data.date>=start)[0]);ev=fac['update'].to_numpy(bool)
        cov=fac[(fac.date>=start)&(fac.date<end)]
        coverage.append({'period':period,'full_days':len(data)-first,'common_signal_days':int(cov.common_valid.sum()),
                         'scheduled_months':int(cov['update'].sum()),'available_scheduled_months':int((cov['update']&cov.common_valid).sum()),
                         'no_view_scheduled_months':int((cov['update']&~cov.common_valid).sum()),'first_signal_date':cov.loc[cov.common_valid,'date'].min()})
        for cost_id,cost in cfg['costs'].items():
            folder=OUT/period/cost_id;folder.mkdir(parents=True,exist_ok=True);accounts={}
            for model in [*cfg['policies'],'BUY_HOLD']:
                targets=fac[model].to_numpy(float) if model!='BUY_HOLD' else np.ones(len(fac))
                ledger,decisions=simulate_event_account(data,div,cfg,cost,start,model,targets=targets,event_mask=ev)
                ledger.to_parquet(folder/f'{model}_ledger.parquet',index=False);decisions.to_parquet(folder/f'{model}_decisions.parquet',index=False)
                m={'period':period,'cost':cost_id,'model':model,**summarize(ledger,cfg)}
                m['point_met']=bool(m['annualized_return']>=.10 and m['net_sharpe'] is not None and m['net_sharpe']>=1.2)
                metrics.append(m);checks.append({'period':period,'cost':cost_id,'model':model,**verify_ledger(ledger,cfg)})
                accounts[model]=ledger
                if model=='BUY_HOLD':
                    saved=pd.read_parquet(ROOT/'reports/research/510300_continuous_reference_min_variance_v1'/period/cost_id/'BUY_HOLD_ledger.parquet')
                    for col in ['cash','shares','equity','net_return','filled_quantity','commission','slippage_cost']:
                        np.testing.assert_allclose(ledger[col].to_numpy(float),saved[col].to_numpy(float),atol=0,rtol=0)
                    controls.append({'period':period,'cost':cost_id,'status':'SEVEN_FIELDS_EXACT_MATCH'})
                for year,g in ledger.groupby(ledger.date.dt.year):years.append({'period':period,'cost':cost_id,'model':model,'year':int(year),**summarize(g,cfg)})
                r=ledger.net_return.to_numpy(float);w=726
                if len(r)>=w:
                    s=pd.Series(r);cum=np.expm1(pd.Series(np.log1p(r)).rolling(w).sum()*242/w);ss=s.rolling(w).mean()/s.rolling(w).std(ddof=1)*np.sqrt(242)
                    good=np.isfinite(cum)&np.isfinite(ss)
                    rolls.append({'period':period,'cost':cost_id,'model':model,'window_days':w,'window_count':int(good.sum()),'median_cagr':float(cum[good].median()),'min_cagr':float(cum[good].min()),'median_sharpe':float(ss[good].median()),'min_sharpe':float(ss[good].min()),'joint_fraction':float(((cum[good]>=.1)&(ss[good]>=1.2)).mean())})
            for scale in ['FULL','VOL10']:
                a=accounts['RESIDUAL_'+scale]
                for alternative in ['RAW','PRICE']:
                    b=accounts[alternative+'_'+scale];ci=bootstrap(a.net_return.to_numpy(float),b.net_return.to_numpy(float))
                    boots.append({'period':period,'cost':cost_id,'scale':scale,'contrast':'RESIDUAL_minus_'+alternative,
                                  'cagr_difference':summarize(a,cfg)['annualized_return']-summarize(b,cfg)['annualized_return'],
                                  'conditional_low':ci[0],'conditional_median':ci[1],'conditional_high':ci[2],'selection_adjusted':False})
            print(period,cost_id,'SEVEN_ACCOUNTS_COMPLETE',flush=True)
    for name,rows in [('metrics',metrics),('independent_checks',checks),('yearly',years),('coverage',coverage),('rolling_three_year',rolls),('conditional_bootstrap',boots),('buy_hold_replay',controls)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
    d=pd.DataFrame(metrics);new=d[d.model!='BUY_HOLD'];ad=[]
    for model,g in new.groupby('model'):
        mb=g[g.period.eq('evaluation')&g.cost.eq('BASE')].iloc[0]
        ad.append({'model':model,'main_base_met':bool(mb.point_met),'main_both_costs_met':bool(g[g.period.eq('evaluation')].point_met.all()),'all_four_met':bool(g.point_met.all()),'min_cagr':float(g.annualized_return.min()),'min_sharpe':float(g.net_sharpe.min())})
    pd.DataFrame(ad).to_csv(OUT/'adjudications.csv',index=False,encoding='utf-8-sig')
    write_json(OUT/'result.json',{'study_id':cfg['study_id'],'completed_at':now(),'status':'FIXED_NEW_INFORMATION_OBJECT_EVALUATED',
                                'policy_accounts':len(new),'control_accounts':len(controls),'independent_checks':len(checks),
                                'new_predictive_model_fits':0,'rolling_factor_beta_estimates':factor_receipt['finite_beta_estimates'],
                                'main_base_any_met':bool(new[new.period.eq('evaluation')&new.cost.eq('BASE')].point_met.any()),
                                'all_four_any_met':any(a['all_four_met'] for a in ad),'primary':d[d.model.eq(cfg['primary'])].to_dict('records'),
                                'goal_achieved':False,'independent_validation':'NOT_ESTABLISHED','position_impact':0,'live_trading_authorized':False,
                                'protocol_sha256':digest(OUT/'protocol.json'),'code_sha256':digest(Path(__file__))},exclusive=True)
    print(d[['period','cost','model','annualized_return','net_sharpe','max_drawdown','mean_exposure']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);args=p.parse_args()
    {'freeze':freeze,'run':run}[args.action]()
