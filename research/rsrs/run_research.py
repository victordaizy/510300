"""RSRS固定研究批次：复用原资金引擎、保留全部结果，不连接券商。"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.rsrs.rsrs import calculate_rsrs, rsrs_target_weight
from research.point_first_passage_inputs_v1 import features
from research.point_first_passage_account_v1 import account
from research.point_account_nr7_inputs_v1 import PARENT_A, metrics

OUT = ROOT / 'rsrs_artifacts' / 'experiment_v1'
CURRENT = ROOT / 'reports/research/510300_point_current_observation_20261001/inputs/candidate_prices.parquet'
WEIGHT = ROOT / 'reports/research/510300_point_weight_information_diagnostic_v1/inputs'
CONTROL = ROOT / 'reports/research/510300_point_second_weight_comparison_v1/inputs/controls'
PERIODS = {'2015_2019': ('2015-01-05','2019-12-31'), '2020_2026': ('2020-01-02','2026-09-30')}
POLICIES = ['RSRS_RIGHT_18_600','RSRS_Z_18_600','RSRS_R2_18_600',
            'RSRS_RIGHT_16_600','RSRS_RIGHT_20_600','RSRS_RIGHT_18_252',
            'RSRS_VOL8','RSRS_CONTINUOUS','RSRS_TREND200','RSRS_MOM60',
            'A_RSRS_CONFIRM','A_RSRS_TILT']
PRIMARY = POLICIES[0]


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple,np.ndarray)): return [clean(v) for v in x]
    if x is pd.NaT or x is pd.NA: return None
    if isinstance(x, (pd.Timestamp,datetime)): return x.isoformat()
    if isinstance(x, (bool,np.bool_)): return bool(x)
    if isinstance(x, (int,np.integer)): return int(x)
    if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
    return x


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def read_frame(path):
    if importlib.util.find_spec('pyarrow') is not None:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(str(path)+'.csv',float_precision='round_trip')
    for c in frame.columns:
        if c in {'date','origin','execution_date','entry_origin','entry_date','exit_date','record_date','ex_date','payment_date'}:
            frame[c] = pd.to_datetime(frame[c]).astype('datetime64[ns]')
    return frame


def load():
    p = read_frame(CURRENT)
    div = pd.read_csv(WEIGHT/'dividends.csv')
    for c in ['record_date','ex_date','payment_date']:
        div[c] = pd.to_datetime(div[c]).dt.normalize()
    assert not div.ex_date.duplicated().any()
    assert (div.record_date < div.ex_date).all() and (div.ex_date <= div.payment_date).all()
    assert div.cash_dividend_per_share.gt(0).all()
    div = div.sort_values('ex_date').reset_index(drop=True)
    d = features(p,div)
    assert len(d)==3488 and d.symbol.eq('510300.SH').all()
    risks = read_frame(WEIGHT/'risks.parquet')
    assert risks.latest_label_exit_idx.le(risks.idx).all()
    assert np.array_equal(d.date.iloc[risks.idx.astype(int)].to_numpy(),risks.date.to_numpy())
    recent = read_frame(WEIGHT/'parent_signals.parquet').pivot(index='origin',columns='candidate',values='target').reset_index()
    early = read_frame(WEIGHT/'earlier_signals.parquet')
    early = early.loc[early.period.eq('earlier_diagnostic')].pivot(index='origin',columns='model',values='target').reset_index()
    return d,div,risks,{'2015_2019':early,'2020_2026':recent}


def empty_signals(d):
    return pd.DataFrame({'date':d.date,'entry_event':False,'event_id':None,'atr':np.nan,'stop_index':np.nan,'target_index':np.nan})


def verify(result):
    d = result['daily']
    np.testing.assert_allclose(d.equity,d.cash+d.receivable+d.shares*d.close,atol=1e-7,rtol=0)
    np.testing.assert_allclose(d.net_return,d.equity.to_numpy()/np.r_[200000.,d.equity.to_numpy()[:-1]]-1,atol=1e-13,rtol=0)
    assert d.cash.ge(-1e-7).all() and d.receivable.ge(-1e-7).all() and d.shares.mod(100).eq(0).all()
    assert d.accounting_error.abs().max()<1e-6
    orders=result['orders']
    if len(orders):
        assert orders.origin.lt(orders.date).all()
        signed=np.where(orders.side.eq('BUY'),orders.quantity,-orders.quantity)
        inventory=pd.Series(signed,index=pd.DatetimeIndex(orders.date)).groupby(level=0).sum().reindex(pd.DatetimeIndex(d.date),fill_value=0).cumsum()
        np.testing.assert_array_equal(inventory,d.shares)
    for t in result['trades'].loc[result['trades'].status.eq('COMPLETE')].itertuples():
        assert t.entry_origin<t.entry_date<t.exit_date
    return {'max_accounting_error':float(d.accounting_error.abs().max()),'ledger_checks_passed':True}


def target_set(d,parents):
    bars=pd.DataFrame({'high':d.high+d.cash_shift,'low':d.low+d.cash_shift},index=d.index)
    f=calculate_rsrs(bars,18,600)
    state=rsrs_target_weight(f.right_skew_rsrs,long_weight=.5)
    ans={PRIMARY:state,
         'RSRS_Z_18_600':rsrs_target_weight(f.zscore,long_weight=.5),
         'RSRS_R2_18_600':rsrs_target_weight(f.zscore*f.r_squared,long_weight=.5)}
    for n,m in [(16,600),(20,600),(18,252)]:
        g=calculate_rsrs(bars,n,m)
        ans[f'RSRS_RIGHT_{n}_{m}']=rsrs_target_weight(g.right_skew_rsrs,long_weight=.5)
    returns=(d.close+d.dividend)/d.close.shift()-1
    rv=returns.rolling(20,min_periods=20).std(ddof=1)*np.sqrt(252)
    ans['RSRS_VOL8']=state*(.08/rv/.5).clip(upper=1)
    ans['RSRS_CONTINUOUS']=.5*((f.right_skew_rsrs+.7)/1.4).clip(0,1)
    ans['RSRS_TREND200']=state.where(d.ac>d.ac.rolling(200,min_periods=200).mean(),0.)
    ans['RSRS_MOM60']=state.where(d.ac>d.ac.shift(60),0.)
    a=parents.set_index('origin')[PARENT_A].reindex(pd.DatetimeIndex(d.date)).reset_index(drop=True)
    ans['A_RSRS_CONFIRM']=a.where(state>0,0.).where(a.notna())
    ans['A_RSRS_TILT']=a*(.5+state)
    for name,x in ans.items():
        known=x.dropna()
        assert known.ge(0).all() and known.le(1).all(),name
    return ans,f


def evaluate_gate(s):
    fields=['net_sharpe','net_cagr','max_drawdown','p_times_b','standard_expectancy_loss_units','mean_cycle_net_return']
    if not all(np.isfinite(s.get(k,np.nan)) for k in fields): return False
    return bool(s['net_sharpe']>=1.5 and s['net_cagr']>=.1 and s['max_drawdown']<=.1
                and s['p_times_b']>1 and s['standard_expectancy_loss_units']>0 and s['mean_cycle_net_return']>0)


def dump_result(folder,result):
    folder.mkdir(parents=True,exist_ok=True)
    for name in ['daily','orders','trades','decisions','rejections']:
        result[name].to_csv(folder/(name+'.csv'),index=False,encoding='utf-8-sig')
    write(folder/'terminal.json',result['terminal'])


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol={'registered_at':datetime.now(timezone.utc).isoformat(),'primary':PRIMARY,'policies':POLICIES,
              'rules':{'n':18,'m':600,'enter':.7,'exit':-.7,'right_score':'z * R_squared * beta',
                       'price_adjustment':'raw OHLC plus cumulative already-ex-date cash distributions',
                       'vol_target':.08,'vol_window':20,'continuous':'.5 * clip((score+.7)/1.4,0,1)',
                       'A_confirm':'original A target times binary RSRS permission',
                       'A_tilt':'original A target times (.5 + RSRS target); multiplier .5 or 1',
                       'trend_filter':200,'momentum_filter':60},
              'periods':PERIODS,'costs':{'BASE':[.0002,.0005],'STRESS':[.0004,.001]},
              'capital':200000,'max_stock_weight':.5,'cash_return_and_sharpe_reference':0.,'annual_days':252,
              'account_engine':'unchanged point_first_passage_account_v1.account(mode=A_CONTROL)',
              'risk':'original ES, gap budget, 10pct drawdown stop, 100-share lots, tick .001, min fee 5, T+1',
              'rebalance':'original 10 percentage point band and risk-only reductions retained',
              'planned_candidate_accounts':48,'planned_baseline_replays':4,'planned_constant50_controls':4,
              'history_status':'already examined in prior research; development only, NOT new independent OOS',
              'independent_validation':'NOT_ESTABLISHED','goal_achieved':False,'orders_authorized':False,
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write(OUT/'protocol.json',protocol)
    d,div,risks,parents=load()
    write(OUT/'data_audit.json',{'rows':len(d),'start':d.date.iloc[0],'end':d.date.iloc[-1],
                               'dividend_events':len(div),'risks':len(risks),'risk_maturity_passed':True})
    # 导出可直接读取的文本，原Parquet和哈希保持不变。
    export=ROOT/'rsrs_artifacts'/'csv_inputs'
    manifest=json.loads((ROOT/'rsrs_artifacts/source_manifest.json').read_text())
    for r in manifest:
        p=ROOT/r['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
        if p.suffix=='.parquet':
            dest=export/(r['path']+'.csv');dest.parent.mkdir(parents=True,exist_ok=True)
            pd.read_parquet(p).to_csv(dest,index=False,encoding='utf-8-sig')
    d.to_csv(OUT/'features.csv',index=False,encoding='utf-8-sig')
    rows=[];checks=[];yearly=[]
    for period,(start,end) in PERIODS.items():
        local=d.loc[d.date.le(end)].reset_index(drop=True)
        sig=empty_signals(local)
        targets,f=target_set(local,parents[period])
        f.assign(date=local.date).to_csv(OUT/(period+'_rsrs.csv'),index=False)
        cuts=[int(len(local)*.65),int(len(local)*.8)]
        for cut in cuts:
            small=local.iloc[:cut].copy()
            smaller,_=target_set(small,parents[period])
            for name in POLICIES:
                np.testing.assert_allclose(targets[name].iloc[:cut],smaller[name],rtol=0,atol=1e-12,equal_nan=True)
        for cost in ['BASE','STRESS']:
            actual=account(local,div,parents[period],risks,sig,cost,start,'A_CONTROL')
            reference=read_frame(CONTROL/period/cost/'A_SAVED_WEIGHT/daily.parquet')
            assert np.array_equal(actual['daily'].date.to_numpy(),reference.date.to_numpy())
            errors={}
            for col in ['equity','cash','shares','receivable','net_return','commission','slippage']:
                errors[col]=float(np.max(np.abs(actual['daily'][col].to_numpy()-reference[col].to_numpy())))
                assert errors[col]<1e-7,(period,cost,col,errors[col])
            checks.append({'period':period,'cost':cost,'baseline_errors':errors,**verify(actual)})
            baseline_stat=metrics(actual)
            base_row={'period':period,'cost':cost,'policy':'A_CONTROL',**baseline_stat}
            base_row['absolute_gate']=evaluate_gate(base_row)
            rows.append(base_row)
            dump_result(OUT/'accounts'/period/cost/'A_CONTROL',actual)
            policies={'CONSTANT_50':pd.Series(.5,index=local.index),**targets}
            for name,target in policies.items():
                weights=pd.DataFrame({'origin':local.date,PARENT_A:target.to_numpy()})
                result=account(local,div,weights,risks,sig,cost,start,'A_CONTROL')
                check=verify(result)
                s=metrics(result)
                row={'period':period,'cost':cost,'policy':name,**s,**check}
                row['absolute_gate']=evaluate_gate(row)
                row['beats_A_both']=bool(row['net_cagr']>baseline_stat['net_cagr'] and row['net_sharpe']>baseline_stat['net_sharpe'])
                rows.append(row)
                dump_result(OUT/'accounts'/period/cost/name,result)
                for year,g in result['daily'].groupby(result['daily'].date.dt.year):
                    r=g.net_return.to_numpy();sd=r.std(ddof=1)
                    yearly.append({'period':period,'cost':cost,'policy':name,'year':int(year),'days':len(g),
                                   'return':float(np.prod(1+r)-1),'sharpe':float(r.mean()/sd*np.sqrt(252)) if sd>1e-14 else np.nan})
                print('ACCOUNT_RESULT',json.dumps(clean(row),ensure_ascii=False,allow_nan=False),flush=True)
                pd.DataFrame(rows).to_csv(OUT/'all_metrics.csv',index=False)
    table=pd.DataFrame(rows)
    passed=[]
    for name in POLICIES:
        v=table[table.policy.eq(name)]
        if len(v)==4 and v.absolute_gate.all() and v.beats_A_both.all(): passed.append(name)
    summary={'completed_at':datetime.now(timezone.utc).isoformat(),'accounts':len(rows),'candidate_accounts':48,
             'baseline_replays':4,'constant50_controls':4,'all_four_passed_policies':passed,
             'primary_results':[r for r in rows if r['policy']==PRIMARY],
             'baseline_reproduction':checks,'prefix_causality_checks':48,
             'independent_validation':'NOT_ESTABLISHED_HISTORY_ALREADY_USED','goal_achieved':False,
             'all_results':rows,'orders_authorized':False}
    write(OUT/'summary.json',summary)
    pd.DataFrame(yearly).to_csv(OUT/'annual_returns.csv',index=False)
    print('FINAL_RESEARCH_SUMMARY',json.dumps(clean({k:v for k,v in summary.items() if k!='all_results'}),ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
