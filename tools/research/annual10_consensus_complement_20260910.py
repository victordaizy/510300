"""固定旧共识的边际资金机会；不改旧信号，不以不同账户拼接目标。"""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.adaptive_allocation_v1 import normalize_dividends,summarize
from research.event_clock_account_v1 import simulate_event_account
from research.simple_price_entry_exit_v1 import signals,specifications,simulate_policy
from research.simple_volume_reversal_v1 import make_rules as volume_rules,specifications as volume_specs
from research.simple_session_divergence_v1 import make_rule,specifications as session_specs
from research.simple_signal_blend_v1 import targets,decision_state,NAMES
from research.intraday_overnight_increment_v1 import digest,now,require,write_json
OUT=ROOT/'research_runs/annual10_consensus_complement_20260910'
P150=ROOT/'reports/research/510300_monotone_episode_budget_v1'
SOURCES=['S1_TREND_REBOUND','D60_INTRA','V1_CLIMAX_RECOVERY']

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def read_frame(path):
    frame=pd.read_parquet(path)
    for col in frame:
        if getattr(frame[col].dtype,'kind',None)=='M':frame[col]=frame[col].astype('datetime64[ns]')
    return frame

def align(data,dec):
    ids=pd.DatetimeIndex(data.date).get_indexer(dec.origin)
    require((ids>=0).all() and (ids<len(data)-1).all() and len(set(ids))==len(ids),'意向日期不合法')
    require(pd.DatetimeIndex(dec.execution_date).equals(pd.DatetimeIndex(data.date.iloc[ids+1])),'并非下一开盘执行')
    return decision_state(data,dec)

def combine(core,sleeve,mode,a):
    known=np.isfinite(core)&np.isfinite(sleeve);out=np.full(len(core),np.nan)
    if mode=='IDLE_ONLY':out[known]=np.where(core[known]==0,a*sleeve[known],core[known])
    elif mode=='REMAINING':out[known]=core[known]+(1-core[known])*a*sleeve[known]
    else:raise ValueError('未知合成方式')
    require((np.isnan(out)|((out>=0)&(out<=1))).all(),'超出无杠杆范围')
    return out

def check(ledger,cfg):
    np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable,ledger.equity,atol=1e-7,rtol=0)
    np.testing.assert_allclose(ledger.pnl,ledger.price_pnl+ledger.dividend_recognized-ledger.commission-ledger.slippage_cost,atol=1e-7,rtol=0)
    prior=np.r_[cfg['initial_capital'],ledger.equity.to_numpy()[:-1]]
    np.testing.assert_allclose(ledger.net_return,ledger.equity.to_numpy()/prior-1,atol=1e-13,rtol=0)
    require(ledger.shares.iloc[-1]==0 and ledger.cash.min()>=-1e-7 and (ledger.shares%100==0).all(),'股数、现金或终点无效')
    r=ledger.net_return.to_numpy();m=summarize(ledger,cfg)
    require(abs(np.expm1(np.log1p(r).sum()*242/len(r))-m['annualized_return'])<1e-12,'CAGR复算不符')
    if r.std(ddof=1)>1e-15:require(abs(r.mean()/r.std(ddof=1)*np.sqrt(242)-m['net_sharpe'])<1e-12,'夏普复算不符')
    return {'status':'PASS','rows':len(ledger),'maximum_accounting_error':float(ledger.accounting_error.abs().max())}

def main():
    require(not (OUT/'protocol.json').exists(),'本组已登记，不覆盖')
    cfg=read_json(ROOT/'config/510300_continuous_reference_min_variance_v1.json')
    files=[Path(__file__),ROOT/cfg['features'],ROOT/cfg['dividends']]+list((ROOT/'research').glob('*.py'))+list(P150.glob('*/*/*_decisions.parquet'))
    write_json(OUT/'protocol.json',{'registered_at':now(),'protocol_commit':'38334a89318b2ffa85e9f74dc6bd440998eb8cd2','sources':SOURCES,'legacy_formulas':list(NAMES),'core_multipliers':[2,3],'allocations':[.5,1.],'modes':['IDLE_ONLY','REMAINING'],'new_policies':64,'planned_new_accounts':256,'planned_reference_accounts':6,'planned_control_accounts':40,'code_sha256':digest(Path(__file__)),'files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in files]},exclusive=True)
    c=np.array([np.nan,0.,.4,1.]);s=np.array([1.,.8,.6,.9])
    np.testing.assert_allclose(combine(c,s,'IDLE_ONLY',.5),[np.nan,.4,.4,1.],equal_nan=True)
    np.testing.assert_allclose(combine(c,s,'REMAINING',.5),[np.nan,.4,.58,1.],equal_nan=True)
    data=read_frame(ROOT/cfg['features']);div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    expected=pd.read_csv(ROOT/'reports/research/510300_simple_signal_blend_v1/metrics.csv')
    metrics=[];checks=[];controls=[];replays=[]
    for period,start,end in [('evaluation',cfg['evaluation_start'],cfg['data_cutoff']),('earlier_diagnostic',cfg['earlier_start'],cfg['earlier_terminal'])]:
        frame=data[data.date.le(end)].copy();rule_set=signals(frame);spec=specifications()
        rule_set['D60_INTRA']=make_rule(frame,next(r for r in session_specs() if r['id']=='D60_INTRA'))[0]
        spec['D60_INTRA']=read_json(ROOT/'config/510300_simple_session_divergence_v1.json')['common_specification']
        rule_set['V1_CLIMAX_RECOVERY']=volume_rules(frame)[0]['V1_CLIMAX_RECOVERY'];spec['V1_CLIMAX_RECOVERY']=volume_specs()['V1_CLIMAX_RECOVERY']
        states=[]
        for name in SOURCES:
            led,dec,cy=simulate_policy(frame,div,cfg,cfg['costs']['BASE'],start,rule_set[name],spec[name])
            folder=OUT/period/'references';folder.mkdir(parents=True,exist_ok=True)
            led.to_parquet(folder/f'{name}_ledger.parquet',index=False);dec.to_parquet(folder/f'{name}_decisions.parquet',index=False);cy.to_csv(folder/f'{name}_cycles.csv',index=False)
            checks.append({'period':period,'cost':'BASE','model':name,'kind':'reference',**check(led,cfg)});states.append(align(frame,dec))
        consensus=targets(np.column_stack(states),frame.vol20.to_numpy())
        for cost,cost_cfg in cfg['costs'].items():
            folder=OUT/period/cost;folder.mkdir(parents=True,exist_ok=True)
            for name,tar in consensus.items():
                led,dec=simulate_event_account(frame,div,cfg,cost_cfg,start,name,targets=tar,event_mask=np.ones(len(frame),bool))
                m={'period':period,'cost':cost,'model':name,**summarize(led,cfg)};controls.append(m)
                checks.append({'period':period,'cost':cost,'model':name,'kind':'control',**check(led,cfg)})
                if period=='evaluation' and cost=='BASE':
                    e=expected[expected.model.eq(name)&expected.cost.eq('BASE')].iloc[0]
                    cols=['annualized_return','net_sharpe','max_drawdown','mean_exposure','trade_count','commission','slippage_cost']
                    np.testing.assert_allclose([m[x] for x in cols],e[cols].to_numpy(float),atol=1e-9,rtol=1e-10)
                    replays.append({'model':name,'status':'PASS'})
                led.to_parquet(folder/f'CONTROL_{name}_ledger.parquet',index=False)
            parent=align(frame,read_frame(P150/period/cost/'EPISODE_BUDGET_NONINCREASING_decisions.parquet'))
            for k in [2,3]:
                core=np.minimum(1.,k*parent)
                led,dec=simulate_event_account(frame,div,cfg,cost_cfg,start,f'CORE{k}',targets=core,event_mask=np.ones(len(frame),bool))
                controls.append({'period':period,'cost':cost,'model':f'CORE{k}',**summarize(led,cfg)});checks.append({'period':period,'cost':cost,'model':f'CORE{k}','kind':'control',**check(led,cfg)})
                led.to_parquet(folder/f'CONTROL_CORE{k}_ledger.parquet',index=False)
                for name,sleeve in consensus.items():
                    for mode in ['IDLE_ONLY','REMAINING']:
                        for a in [.5,1.]:
                            model=f'CORE{k}__{name}__{mode}__A{int(a*100)}';tar=combine(core,sleeve,mode,a);cut=len(frame)//2
                            np.testing.assert_allclose(tar[:cut],combine(core[:cut],sleeve[:cut],mode,a),atol=0,rtol=0,equal_nan=True)
                            led,dec=simulate_event_account(frame,div,cfg,cost_cfg,start,model,targets=tar,event_mask=np.ones(len(frame),bool))
                            m={'period':period,'cost':cost,'model':model,**summarize(led,cfg)};m['point_met']=bool(m['annualized_return']>=.10 and m['net_sharpe'] is not None and m['net_sharpe']>=1.2)
                            metrics.append(m);checks.append({'period':period,'cost':cost,'model':model,'kind':'new',**check(led,cfg)})
                            led.to_parquet(folder/f'{model}_ledger.parquet',index=False);dec.to_parquet(folder/f'{model}_decisions.parquet',index=False)
            print('完成',period,cost,'新账户累计',len(metrics),flush=True)
    table=pd.DataFrame(metrics);table.to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(controls).to_csv(OUT/'controls.csv',index=False);pd.DataFrame(checks).to_csv(OUT/'checks.csv',index=False);pd.DataFrame(replays).to_csv(OUT/'original_replays.csv',index=False)
    primary=table[table.period.eq('evaluation')&table.cost.eq('BASE')]
    result={'status':'COMPUTED','completed_at':now(),'new_policies':64,'new_accounts':len(metrics),'reference_accounts':6,'control_accounts':len(controls),'account_checks':len(checks),'main_base_point_met':bool(primary.point_met.any()),'new_model_fits':0,'point_pass_rows':table[table.point_met].to_dict('records'),'best_main_cagr':primary.sort_values('annualized_return',ascending=False).iloc[0].to_dict(),'best_main_sharpe':primary.sort_values('net_sharpe',ascending=False).iloc[0].to_dict(),'independent_validation':'NOT_ESTABLISHED','position_impact':0,'live_trading_authorized':False}
    write_json(OUT/'result.json',result)
    print(primary.sort_values('annualized_return',ascending=False)[['model','annualized_return','net_sharpe','max_drawdown','point_met']].head(12).to_string(index=False));print('主期同时达标',result['main_base_point_met'])

if __name__=='__main__':main()
