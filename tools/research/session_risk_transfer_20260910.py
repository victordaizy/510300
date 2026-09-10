"""时段风险补偿、T+1桥接与开盘折价吸收。固定研究，不连接券商。"""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.intraday_overnight_increment_v1 import (Account, execute_order, affordable_quantity,
    fill_price, normalize_dividends, return_metrics, require, write_json, now, digest)
from research.event_clock_account_v1 import simulate_event_account
OUT = ROOT/'research_runs/session_risk_transfer_20260910'
FEATURES = 'reports/research/510300_adaptive_allocation_v1/features.parquet'
MINUTE = 'data/curated/510300_stk_mins_source_admission_v1/510300_1min.parquet'
QUALITY = 'data/curated/510300_stk_mins_source_admission_v1/daily_quality_ledger.parquet'
CONFIG = 'config/510300_continuous_reference_min_variance_v1.json'
DIVIDENDS = 'data/reference/510300_dividends.csv'
PERIODS = {'earlier': ('2015-01-05','2019-12-31'), 'main': ('2020-01-02','2026-08-14')}
SLOTS = ('09:30','09:31','09:32','10:00','10:02','14:56')
PRIMARY = 'GAP_ABSORBED_H1_FULL'

@dataclass(frozen=True)
class Policy:
    name: str
    mode: str
    hold: int
    size: str = 'FULL'
    phase: int = 0

POLICIES = ([Policy(f'BRIDGE_H{h}_P{p}', 'BRIDGE', h, phase=p) for h in (2,5) for p in (0,1)]
    + [Policy(f'{m}_H{h}_{s}', m, h, s) for m in ('GAP_DIRECT','GAP_ABSORBED') for h in (1,3) for s in ('FULL','VOL10')])

def read_data():
    d=pd.read_parquet(ROOT/FEATURES).copy()
    d['date']=pd.to_datetime(d.date).astype('datetime64[ns]')
    cfg=json.loads((ROOT/CONFIG).read_text(encoding='utf-8'))
    div=normalize_dividends(pd.read_csv(ROOT/DIVIDENDS))
    return d, div, cfg


def minute_inputs(d):
    raw=pd.read_parquet(ROOT/MINUTE).copy()
    raw['trade_time']=pd.to_datetime(raw.trade_time).astype('datetime64[ns]')
    require(raw.ts_code.eq('510300.SH').all(), '分钟代码错误')
    require(not raw.trade_time.duplicated().any(), '分钟时间重复')
    raw=raw.sort_values('trade_time').reset_index(drop=True)
    raw['date']=raw.trade_time.dt.normalize()
    raw['slot']=raw.trade_time.dt.strftime('%H:%M')
    req=['open','high','low','close','vol']
    require(np.isfinite(raw[req].to_numpy(float)).all(), '必要分钟字段不完整')
    require(raw[['open','high','low','close']].gt(0).all().all() and raw.vol.ge(0).all(), '分钟价格/量非法')
    require((raw.high>=raw[['open','close']].max(axis=1)).all() and (raw.low<=raw[['open','close']].min(axis=1)).all(), '分钟OHLC顺序错误')
    panel={}
    for slot in SLOTS:
        x=raw[raw.slot.eq(slot)].set_index('date')[req].reindex(pd.DatetimeIndex(d.date))
        x.index=d.index
        panel[slot]=x
    agg=raw.groupby('date').agg(minute_open=('open','first'),minute_high=('high','max'),minute_low=('low','min'),minute_close=('close','last'),minute_volume=('vol','sum'),rows=('slot','size'))
    j=d[['date','open','high','low','close','volume']].merge(agg,left_on='date',right_index=True,how='left')
    for k in ('open','high','low','close'): j[f'{k}_difference']=j[f'minute_{k}']-j[k]
    j['max_ohlc_difference']=j[[f'{k}_difference' for k in ('open','high','low','close')]].abs().max(axis=1)
    j.loc[j.rows.isna(),'max_ohlc_difference']=np.nan
    j['ohlc_over_one_tick']=j.max_ohlc_difference.gt(.001000001)
    j['volume_difference']=j.minute_volume-j.volume
    known=j.rows.notna()
    require(j.loc[known,'open_difference'].abs().max()<1e-12,'开盘价不一致，阻断分钟进入信号')
    require(j.loc[known,'rows'].eq(241).all(),'源有不完整日，不静默删除')
    # 日线信号用原日线，分钟仅用于指定时点报价。不把149天差异洗掉。
    receipt={'status':'PRICE_VOLUME_RESEARCH_DIAGNOSTIC_ONLY', 'raw_rows':len(raw),'raw_first':raw.trade_time.iloc[0],
      'raw_last':raw.trade_time.iloc[-1],'cutoff':d.date.iloc[-1], 'overlap_days':int(known.sum()),
      'ohlc_over_one_tick_days':int(j.ohlc_over_one_tick.sum()),'max_ohlc_difference':float(j.max_ohlc_difference.max()),
      'max_open_difference':float(j.open_difference.abs().max()), 'strict_full_ohlc_admission':False,
      'minute_volume_unit':'shares; original source implementation and ledger minute_vol_sum_shares',
      'bar_semantics':'BAR_END; 10:02 open belongs to (10:01,10:02], after the 10:00 observation',
      'no_new_external_market_data':True,'post_cutoff_rows_used_for_policy':0,
      'fill_model':'conditional bar-open plus adverse tick/slippage, <=1% realized minute shares; not L2 execution proof'}
    return panel,j,receipt


def signals(d,panel):
    # 先前60个隔夜，不含当前开盘。分母为样本标准差。
    sd=d.overnight_log.rolling(60,min_periods=60).std(ddof=1).shift(1)
    risk=d.total_simple.rolling(20,min_periods=20).std(ddof=1).shift(1)*np.sqrt(242)
    o=panel['09:30'].open
    gap=np.log((o+d.dividend)/d.previous_close)
    valid=np.isfinite(gap)&np.isfinite(sd)&sd.gt(0)
    event=valid & gap.le(-sd)
    direct=event & panel['09:31'].close.notna()
    absorbed=event & panel['10:00'].close.gt(o)
    size=np.minimum(1.,.10/risk)
    size=size.where(risk.gt(0)&np.isfinite(risk))
    return pd.DataFrame({'date':d.date,'observed_open':o,'gap':gap,'previous_overnight_sd':sd,
       'previous_total_vol20':risk,'entry_multiplier':size,'GAP_DIRECT':direct,'GAP_ABSORBED':absorbed,
       'minute_origin_available':valid})


def dividend_maps(div,dates):
    record, ex, pay = {}, {}, {}
    for k,e in enumerate(div.itertuples()):
        for name,mapping in [('record_date',record),('ex_date',ex)]:
            day=getattr(e,name)
            if day in dates: mapping.setdefault(dates.get_loc(day),[]).append((k,e.cash_dividend_per_share))
        i=dates.searchsorted(e.payment_date)
        if i<len(dates): pay.setdefault(i,[]).append((k,e.payment_date==dates[i]))
    return record,ex,pay


def simulate(d,div,cfg,cost,start,policy,panel=None,sig=None):
    """独立实际现金账户，盘中请求从已知观察价格计算，不预知后续成交量。"""
    dates=pd.DatetimeIndex(d.date); first=int(np.flatnonzero(d.date.ge(pd.Timestamp(start)))[0]);last=len(d)-1
    rec,ex,pay=dividend_maps(div,dates)
    a=Account(cfg['initial_capital']); previous_nav=a.cash;previous_mark=float(d.close.iloc[first-1])
    ledger=[];trades=[];decisions=[];entry_index=None;deadline=None;cycle=0
    next_entry=first+policy.phase
    for i in range(first,last+1):
        r=d.iloc[i];terminal=i==last;old=a.shares;before_qty=a.shares;recognized=paid=0.
        for k,amount in ex.get(i,[]):
            value=a.entitlements.get(k,0)*amount;a.receivables[k]=value;recognized+=value
        for k,same in pay.get(i,[]):
            if not same:
                value=a.receivables.pop(k,0.);a.cash+=value;paid+=value
        day_fills=[]
        def trade(q,raw_price,clock,origin,cap=None):
            req=int(q);limited=req
            if cap is not None:
                maxq=max(0,int(math.floor(float(cap)*.01/cfg['lot']))*cfg['lot'])
                limited=int(np.sign(req))*min(abs(req),maxq)
            x=execute_order(a,limited,float(raw_price),float(r.previous_close),float(r.dividend),i,cost,cfg)
            x.update(date=r.date,execution_clock=clock,decision_origin=origin,original_request=req,
                capacity_limited=abs(limited)<abs(req),minute_volume=cap,cycle=cycle,
                min_holding_days=(i-entry_index if entry_index is not None else None))
            if req and limited==0:x['status']='UNFILLED_MINUTE_CAPACITY'
            trades.append(x);day_fills.append(x)
            return x
        if terminal:
            trade(-a.shares,r.open,'09:25_TERMINAL_PRECOMMITTED',dates[i-1]+pd.Timedelta(hours=15))
        elif policy.mode=='BUY_HOLD':
            if i==first:
                q=affordable_quantity(a.cash,fill_price(previous_mark,1,cost,cfg['tick']),cost,cfg['lot'])
                trade(q,r.open,'09:25',dates[i-1]+pd.Timedelta(hours=15))
        elif policy.mode=='BRIDGE':
            if not a.shares and i>=next_entry:
                q=affordable_quantity(a.cash,fill_price(previous_mark,1,cost,cfg['tick']),cost,cfg['lot'])
                x=trade(q,r.open,'09:25',dates[i-1]+pd.Timedelta(hours=15))
                if x['filled_quantity']>0:entry_index=i;deadline=i+policy.hold-1;cycle+=1
            if a.shares and i>=deadline:
                trade(-a.shares,r.close,'15:00_PRECOMMITTED',dates[entry_index]+pd.Timedelta(hours=9,minutes=25))
                if not a.shares:entry_index=deadline=None;next_entry=i+1
        else:
            # 开盘时已有仓位，则本日不得新开仓；预定退出只在14:55执行槽尝试。
            if not a.shares:
                event=bool(sig[policy.mode].iloc[i]);mult=1. if policy.size=='FULL' else float(sig.entry_multiplier.iloc[i])
                slot,obs=('09:32','09:31') if policy.mode=='GAP_DIRECT' else ('10:02','10:00')
                reference=float(panel[obs].close.iloc[i]);px=float(panel[slot].open.iloc[i]);vol=float(panel[slot].vol.iloc[i])
                valid=bool(np.isfinite([reference,mult]).all())
                decisions.append({'date':r.date,'origin':r.date+pd.Timedelta(hours=int(obs[:2]),minutes=int(obs[3:])),
                    'entry_event':event,'multiplier':mult,'observation_price':reference,'input_valid':valid,
                    'gap':sig.gap.iloc[i],'previous_overnight_sd':sig.previous_overnight_sd.iloc[i]})
                if event and valid and np.isfinite([px,vol]).all():
                    nav=a.value(reference)
                    q=int(math.floor(mult*nav/reference/cfg['lot']))*cfg['lot']
                    q=min(q,affordable_quantity(a.cash,fill_price(reference,1,cost,cfg['tick']),cost,cfg['lot']))
                    x=trade(q,px,slot+'_BAR_OPEN',decisions[-1]['origin'],vol)
                    if x['filled_quantity']>0:entry_index=i;deadline=i+policy.hold;cycle+=1
            elif deadline is not None and i>=deadline:
                px=float(panel['14:56'].open.iloc[i]);vol=float(panel['14:56'].vol.iloc[i])
                if np.isfinite([px,vol]).all():
                    trade(-a.shares,px,'14:56_BAR_OPEN',dates[entry_index]+pd.Timedelta(hours=10,minutes=1),vol)
                    if not a.shares:entry_index=deadline=None
                else:
                    decisions.append({'date':r.date,'input_valid':False,'action':'MISSING_EXIT_SLOT_KEEP_REQUEST'})
        if not terminal:
            for k,same in pay.get(i,[]):
                if same:value=a.receivables.pop(k,0.);a.cash+=value;paid+=value
            for k,amount in rec.get(i,[]):a.entitlements[k]=a.shares
        mark=float(r.open if terminal else r.close)
        fee=sum(x['commission'] for x in day_fills);slip=sum(x['slippage_cost'] for x in day_fills)
        fq=sum(x['filled_quantity'] for x in day_fills);requested=sum(x['original_request'] for x in day_fills)
        # 对任意日内成交时刻都成立：旧股*(终标-昨日标)+各成交股*(终标-原执行标)。
        price_pnl=old*(mark-previous_mark)+sum(x['filled_quantity']*(mark-x['open_price']) for x in day_fills)
        nav=a.value(mark);error=nav-previous_nav-price_pnl-recognized+fee+slip
        require(abs(error)<1e-6,'逐日财富不守恒')
        require(a.shares==before_qty+fq,'股数不守恒');a.assert_valid()
        ledger.append({'date':r.date,'open':r.open,'mark':mark,'mark_clock':'OPEN_TERMINAL' if terminal else 'CLOSE',
            'cash':a.cash,'shares':a.shares,'dividend_receivable':a.receivable(),'equity':nav,'net_return':nav/previous_nav-1,
            'pnl':nav-previous_nav,'price_pnl':price_pnl,'dividend_recognized':recognized,'dividend_paid':paid,
            'commission':fee,'slippage_cost':slip,'filled_quantity':fq,'requested_quantity':requested,
            'trade_count':sum(x['filled_quantity']!=0 for x in day_fills),'exposure':a.shares*mark/nav,
            'accounting_error':error,'terminal_unliquidated':bool(terminal and a.shares),
            'risk_day':bool(old or a.shares or any(x['filled_quantity']!=0 for x in day_fills)),
            'cycle':cycle if (old or a.shares) else None})
        previous_nav=nav;previous_mark=mark
    require(not ledger[-1]['terminal_unliquidated'],'终点未清仓，不将残仓当作成功')
    return pd.DataFrame(ledger),pd.DataFrame(trades),pd.DataFrame(decisions)


def metrics(l):
    m=return_metrics(l.net_return.to_numpy(float),242)
    m.update(trading_days=len(l),terminal_equity=float(l.equity.iloc[-1]),mean_exposure=float(l.exposure.mean()),
       trade_count=int(l.trade_count.sum()),commission=float(l.commission.sum()),slippage_cost=float(l.slippage_cost.sum()),
       holding_closes=int(l.shares.gt(0).sum()),risk_days=int(l.risk_day.sum()),
       max_accounting_error=float(l.accounting_error.abs().max()))
    m['point_met']=bool(m['annualized_return']>=.10 and m['net_sharpe'] is not None and m['net_sharpe']>=1.2)
    return m


def independent_check(l,t):
    np.testing.assert_allclose(l.equity,l.cash+l.shares*l.mark+l.dividend_receivable,atol=1e-7,rtol=0)
    np.testing.assert_allclose(l.equity.diff().fillna(l.equity.iloc[0]-200000),l.pnl,atol=1e-7,rtol=0)
    np.testing.assert_allclose(l.net_return,l.equity/l.equity.shift().fillna(200000)-1,atol=1e-12,rtol=0)
    np.testing.assert_allclose(l.shares.diff().fillna(l.shares.iloc[0]),l.filled_quantity,atol=0,rtol=0)
    require(l.cash.ge(-1e-7).all() and l.shares.ge(0).all(),'现金股份非法')
    filled=t[t.filled_quantity.ne(0)]
    if len(filled):
        expected=200000-np.cumsum(l.commission)+np.cumsum(l.dividend_paid)-filled.assign(cash_delta=filled.filled_quantity*filled.fill_price).groupby('date').cash_delta.sum().reindex(l.date).fillna(0).cumsum().to_numpy()
        np.testing.assert_allclose(l.cash,expected,atol=1e-6,rtol=0)
        sold=filled[filled.filled_quantity.lt(0)]
        check=sold[sold.min_holding_days.notna()]
        require(check.min_holding_days.ge(1).all(),'卖出了当日买入份额')
        bounded=filled[filled.minute_volume.notna()]
        require((bounded.filled_quantity.abs()<=bounded.minute_volume*.01+1e-7).all(),'超过容量上限')
    vals=l.net_return.to_numpy(float)
    sd=vals.std(ddof=1)*np.sqrt(242); sr=vals.mean()*242/sd if sd>1e-15 else None
    cagr=(l.equity.iloc[-1]/200000)**(242/len(l))-1
    m=metrics(l)
    require(abs(cagr-m['annualized_return'])<1e-10,'CAGR独立复算不一致')
    if sr is not None:require(abs(sr-m['net_sharpe'])<1e-10,'夏普独立复算不一致')
    return {'status':'PASS','rows':len(l),'trade_rows':len(t),'filled_legs':len(filled)}


def simple_tests():
    d,div,cfg=read_data();a=Account(200000);cost=cfg['costs']['BASE']
    execute_order(a,1000,4,4,0,1,cost,cfg)
    require(execute_order(a,-1000,4,4,0,1,cost,cfg)['filled_quantity']==0,'T+1失败')
    require(execute_order(a,-1000,4,4,0,2,cost,cfg)['filled_quantity']==-1000,'次日可卖失败')
    a=Account(200000)
    require(execute_order(a,1000,4.4,4,0,1,cost,cfg)['filled_quantity']==0,'涨停方向约束失败')
    # 十个合成日足够覆盖期初、次日到期、现金结算与期限。
    x=d.iloc[600:612].copy().reset_index(drop=True)
    for k in ('open','close','previous_close','high','low'):x[k]=4.
    x['dividend']=0.
    for p in (Policy('T','BRIDGE',2),Policy('T','BRIDGE',5)):
        l,t,_=simulate(x,div.iloc[:0],cfg,cost,str(x.date.iloc[1].date()),p)
        independent_check(l,t)
        require(l.equity.iloc[-1]<200000,'平价有成本账户竟然盈利')
        sells=t[t.filled_quantity.lt(0)]
        require(sells.min_holding_days.ge(1).all(),'桥接违反T+1')
    # 容量、未知退出、隔日持有检查使用合成面板，不读取策略收益。
    panel={k:pd.DataFrame({'open':4.,'high':4.,'low':4.,'close':4.,'vol':100000.},index=x.index) for k in SLOTS}
    sig=pd.DataFrame({'GAP_DIRECT':[False,True]+[False]*(len(x)-2),'GAP_ABSORBED':[False,True]+[False]*(len(x)-2),
        'entry_multiplier':1.,'gap':-.02,'previous_overnight_sd':.01},index=x.index)
    policy=Policy('CAP','GAP_ABSORBED',1)
    l,t,_=simulate(x,div.iloc[:0],cfg,cost,str(x.date.iloc[1].date()),policy,panel,sig)
    require(int(t[t.filled_quantity.gt(0)].filled_quantity.iloc[0])==1000,'容量未约束买入')
    independent_check(l,t)
    panel['14:56'].loc[2,['open','vol']]=np.nan
    ll,tt,_=simulate(x,div.iloc[:0],cfg,cost,str(x.date.iloc[1].date()),policy,panel,sig)
    require(ll.shares.iloc[1]==1000 and ll.shares.iloc[2]==0,'缺失退出槽没有保持请求')
    independent_check(ll,tt)
    # 登记日持有、次日除息、随后到账，零成本相等财富不能凭空增加。
    y=x.copy();y.loc[2:,'close']=3.9;y.loc[2:,'open']=3.9;y.loc[2,'dividend']=.1
    y.loc[3:,'previous_close']=3.9
    dv=pd.DataFrame({'record_date':[y.date.iloc[1]],'ex_date':[y.date.iloc[2]],
        'payment_date':[y.date.iloc[3]],'cash_dividend_per_share':[.1]})
    zero={'commission':0.,'minimum':0.,'slippage':0.}
    l,t,_=simulate(y,dv,cfg,zero,str(y.date.iloc[1].date()),Policy('DIV','BRIDGE',2))
    require(abs(l.equity.iloc[1]-200000)<1e-7 and l.dividend_recognized.sum()==5000.,'分红记账重复或漏记')
    independent_check(l,t)
    return {'status':'PASS','categories':['T_PLUS_ONE','DIRECTIONAL_LIMIT','BRIDGE_H2_H5','COST_ON_FLAT_PRICE','MINUTE_CAPACITY','MISSING_EXIT_SLOT','DIVIDEND_ENTITLEMENT'], 'categories_count':7}


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    require(not (OUT/'protocol.json').exists(),'本轮已经冻结，不覆盖')
    d,div,cfg=read_data();panel,comparison,receipt=minute_inputs(d)
    tests=simple_tests()
    files=[FEATURES,MINUTE,QUALITY,CONFIG,DIVIDENDS,'research/intraday_overnight_increment_v1.py',
           'research/event_clock_account_v1.py','research/adaptive_allocation_v1.py',str(Path(__file__).relative_to(ROOT))]
    comparison.to_csv(OUT/'source_daily_comparison.csv',index=False)
    write_json(OUT/'source_receipt.json',receipt)
    write_json(OUT/'tests_before_freeze.json',tests)
    write_json(OUT/'protocol.json',{'study_id':'510300_SESSION_RISK_TRANSFER_20260910','frozen_at':now(),
      'remote_protocol_commit':'ca5c02f7bba7dced5a2d8d6d51bd652fca408084','policies':[asdict(p) for p in POLICIES],
      'primary':PRIMARY,'periods':PERIODS,'costs':cfg['costs'],'source_admission':receipt['status'],
      'new_historical_policy_accounts_before_freeze':0,'underlying_history_already_observed':True,
      'fills':'one entry attempt; exit at 14:56 BAR_END open on every eligible day until flat; 1% minute volume',
      'minute_source_strict_full_ohlc_pass':False,'goal_cagr':.1,'goal_sharpe':1.2,
      'position_impact':0,'live_trading_authorized':False,
      'files':[{'path':p,'bytes':(ROOT/p).stat().st_size,'sha256':digest(ROOT/p)} for p in files]},exclusive=True)
    print('FROZEN',digest(OUT/'protocol.json'),flush=True)


def validate():
    p=json.loads((OUT/'protocol.json').read_text())
    for r in p['files']:require(digest(ROOT/r['path'])==r['sha256'],'冻结输入改变 '+r['path'])
    return p


def decomposition(d):
    records=[]; ci=[]
    for period,(start,end) in PERIODS.items():
        f=d[d.date.between(start,end)]
        for label,g in [(period,f)]+[(str(y),x) for y,x in f.groupby(f.date.dt.year)]:
            for col in ('intraday_log','overnight_log','total_log'):
                z=g[col].to_numpy(float); records.append({'period':period,'window':label,'component':col,'days':len(z),
                  'annualized_log_mean':z.mean()*242,'mean_bps_per_day':z.mean()*10000,
                  'mean_positive':bool(z.mean()>0),'not_a_tradable_strategy':True})
        rng=np.random.default_rng(20260910);z=f[['intraday_log','overnight_log']].to_numpy(float); n=len(z)
        b=[]
        for _ in range(20):
            ids=(rng.integers(n,size=(100,math.ceil(n/20),1))+np.arange(20))%n
            ids=ids.reshape(100,-1)[:,:n];b.append(z[ids].mean(axis=1)*242)
        c=np.concatenate(b)
        for k,col in enumerate(('intraday_log','overnight_log')):
            lo,hi=np.quantile(c[:,k],[.025,.975]);ci.append({'period':period,'component':col,
                'mean_log_annualized':z[:,k].mean()*242,'conditional_low':lo,'conditional_high':hi,
                'block':20,'repetitions':2000,'selection_adjusted':False})
    pd.DataFrame(records).to_csv(OUT/'session_components.csv',index=False)
    pd.DataFrame(ci).to_csv(OUT/'session_component_intervals.csv',index=False)


def run_period(period):
    validate();d,div,cfg=read_data();panel,_,_=minute_inputs(d);s=signals(d,panel)
    start,end=PERIODS[period];f=d[d.date.le(end)].copy();part={k:v.iloc[:len(f)].copy() for k,v in panel.items()};ss=s.iloc[:len(f)]
    dest=OUT/period;dest.mkdir(parents=True,exist_ok=True)
    require(not (dest/'metrics.csv').exists(),'已运行本情景，不覆盖')
    rows=[];checks=[];years=[];subperiod=[];gross=[];replays=[]
    for cid,cost in cfg['costs'].items():
        folder=dest/cid;folder.mkdir(exist_ok=True)
        for p in POLICIES+[Policy('BUY_HOLD','BUY_HOLD',0)]:
            l,t,dec=simulate(f,div,cfg,cost,start,p,part,ss)
            for suffix,frame in [('ledger',l),('trades',t),('decisions',dec)]: frame.to_parquet(folder/f'{p.name}_{suffix}.parquet',index=False)
            m=metrics(l);row={'period':period,'cost':cid,'model':p.name,'mode':p.mode,
               'historical_evidence_class':'DAILY_FILL_ASSUMPTION' if p.mode in ('BRIDGE','BUY_HOLD') else 'MINUTE_SOURCE_DIAGNOSTIC_ONLY',**m}
            rows.append(row);checks.append({'period':period,'cost':cid,'model':p.name,**independent_check(l,t)})
            for y,g in l.groupby(l.date.dt.year):years.append({'period':period,'cost':cid,'model':p.name,'year':int(y),**metrics(g)})
            ranges=[('2017-2019_SOURCE_SUBPERIOD','2017-01-03','2019-12-31')] if period=='earlier' else [('2020-2021','2020-01-01','2021-12-31'),('2022-2023','2022-01-01','2023-12-31'),('2024-END','2024-01-01',end)]
            for name,a,b in ranges:
                g=l[l.date.between(a,b)]
                if len(g)>1:subperiod.append({'period':period,'cost':cid,'model':p.name,'era':name,**metrics(g)})
            # 原份额不变，返还累计费用，不进行免费新路径选优。
            free=l.copy();free['equity']=l.equity+(l.commission+l.slippage_cost).cumsum()
            free['net_return']=free.equity/free.equity.shift().fillna(cfg['initial_capital'])-1
            gmetric=return_metrics(free.net_return.to_numpy(),242)
            gross.append({'period':period,'cost':cid,'model':p.name,'same_quantity_path':True,**gmetric})
            if p.mode=='BUY_HOLD':
                orig,_=simulate_event_account(f,div,cfg,cost,start,'BUY_HOLD',event_mask=np.ones(len(f),bool))
                cols=['cash','shares','equity','net_return','commission','slippage_cost','dividend_receivable']
                np.testing.assert_allclose(l[cols].to_numpy(),orig[cols].to_numpy(),atol=1e-7,rtol=0)
                replays.append({'period':period,'cost':cid,'max_difference':float(np.max(np.abs(l[cols].to_numpy()-orig[cols].to_numpy()))),'status':'PASS'})
            print(period,cid,p.name,round(m['annualized_return']*100,4),m['net_sharpe'],flush=True)
    for name,records in [('metrics',rows),('account_checks',checks),('yearly',years),('eras',subperiod),('fixed_quantity_cost_attribution',gross),('buy_hold_replay',replays)]:pd.DataFrame(records).to_csv(dest/f'{name}.csv',index=False)


def prefix_tests():
    d,div,cfg=read_data();panel,_,_=minute_inputs(d);s=signals(d,panel);rows=[]
    # 截断计算与相同前缀逐单元格比较；终点清算日除外。
    for dt in ('2019-06-03','2021-06-01','2024-06-03'):
        last=int(np.flatnonzero(d.date.le(dt))[-1]);f=d.iloc[:last+1].copy();p={k:v.iloc[:last+1].copy() for k,v in panel.items()}
        ss=signals(f,p)
        pd.testing.assert_frame_equal(ss,s.iloc[:last+1],check_exact=True)
        start='2017-01-03';policy=Policy(PRIMARY,'GAP_ABSORBED',1)
        small,_,_=simulate(f,div,cfg,cfg['costs']['BASE'],start,policy,p,ss)
        full,_,_=simulate(d,div,cfg,cfg['costs']['BASE'],start,policy,panel,s)
        pd.testing.assert_frame_equal(small.iloc[:-1].reset_index(drop=True),full.iloc[:len(small)-1].reset_index(drop=True),check_exact=True)
        rows.append({'cutoff':dt,'signals_exact':True,'ledger_before_terminal_exact':True,'counts_as_new_policy':False})
    write_json(OUT/'prefix_tests.json',{'status':'PASS','checks':rows})


def finish():
    validate();d,_,_=read_data();decomposition(d);prefix_tests()
    frames={}
    for name in ('metrics','account_checks','yearly','eras','fixed_quantity_cost_attribution','buy_hold_replay'):
        frames[name]=pd.concat([pd.read_csv(OUT/p/(name+'.csv')) for p in PERIODS],ignore_index=True)
        frames[name].to_csv(OUT/(name+'.csv'),index=False)
    s=signals(d,minute_inputs(d)[0]);s.to_parquet(OUT/'signals.parquet',index=False)
    missing=[]
    for period,(a,b) in PERIODS.items():
        g=s[s.date.between(a,b)]
        missing.append({'period':period,'days':len(g),'no_minute_origin_days':int((~g.minute_origin_available).sum()),
            'gap_direct_days':int(g.GAP_DIRECT.sum()),'gap_absorbed_days':int(g.GAP_ABSORBED.sum())})
    pd.DataFrame(missing).to_csv(OUT/'opportunity_coverage.csv',index=False)
    f=frames['metrics'];eligible=f[~f.model.eq('BUY_HOLD')]
    by=eligible.groupby('model').agg(minimum_cagr=('annualized_return','min'),minimum_sharpe=('net_sharpe','min'),all_four_met=('point_met','all'))
    by.to_csv(OUT/'four_scenario_adjudications.csv')
    result={'study_id':'510300_SESSION_RISK_TRANSFER_20260910','completed_at':now(),'status':'COMPUTED_RESEARCH_ONLY',
      'policies':12,'policy_accounts':48,'buy_hold_controls':4,'account_checks':len(frames['account_checks']),
      'primary':PRIMARY,'primary_metrics':f[f.model.eq(PRIMARY)].to_dict('records'),
      'main_base_point_met':bool(eligible[eligible.period.eq('main')&eligible.cost.eq('BASE')].point_met.any()),
      'all_four_scenarios_met':bool(by.all_four_met.any()),'strict_minute_source_admission':False,
      'new_model_fits':0,'independent_validation':'NOT_ESTABLISHED','goal_achieved':False,'live_ready':False,'position_impact':0}
    write_json(OUT/'result.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['freeze','earlier','main','finish','verify'])
    a=parser.parse_args().action
    if a=='freeze':freeze()
    elif a=='finish':finish()
    elif a=='verify':validate();print('INPUT_HASHES_PASS')
    else:run_period(a)
if __name__=='__main__':main()
