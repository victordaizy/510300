"""用固定半仓对照，区分减少风险与信号择时带来的差异。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_allocation_v1 import normalize_dividends,save_account,summarize
from research.intraday_overnight_increment_v1 import now,write_json,block_indices,interval

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_simple_pair_static_risk_attribution_v1'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/510300_simple_signal_blend_v1.json').read_text(encoding='utf-8'))
    write_json(OUT/'protocol.json',{'registered_at':now(),'candidate':'B2_TREND_SESSION','counterfactual':'固定50%仓位，每日收盘检查，沿用10个百分点免调仓区间',
               'periods':['2020-01-02至2026-08-14开盘','2015-01-05至2019-12-31开盘'],'costs':['BASE','STRESS'],
               'blocks':[20,60],'repetitions':2000,'seed':20260907,'annual_days':242,'fit_new_model':False,
               'purpose':'诊断择时与静态风险暴露，不新增可部署策略，不以区间诊断替代独立验证'},exclusive=True)
    data=pd.read_parquet(ROOT/cfg['features']);div=normalize_dividends(pd.read_csv(ROOT/cfg['dividends']))
    metrics=[];increments=[]
    for period,start,terminal in [('evaluation',cfg['evaluation_start'],cfg['data_cutoff']),('earlier_diagnostic',cfg['earlier_start'],cfg['earlier_terminal'])]:
        frame=data[data.date<=terminal].copy()
        for cost_id,cost in cfg['costs'].items():
            ledger,decisions=simulate_event_account(frame,div,cfg,cost,start,'STATIC_HALF',targets=np.full(len(frame),.5),event_mask=np.ones(len(frame),bool))
            save_account(OUT/period/cost_id,'STATIC_HALF',ledger,decisions)
            m={'period':period,'cost':cost_id,'model':'STATIC_HALF',**summarize(ledger,cfg)};metrics.append(m)
            candidate=pd.read_parquet(ROOT/'reports/research/510300_simple_signal_blend_v1'/period/cost_id/'B2_TREND_SESSION_ledger.parquet')
            assert pd.DatetimeIndex(candidate.date).equals(pd.DatetimeIndex(ledger.date))
            a,b=candidate.net_return.to_numpy(),ledger.net_return.to_numpy()
            for block in [20,60]:
                rng=np.random.default_rng(20260907+block);vals=[]
                for _ in range(2000):
                    idx=block_indices(rng,len(a),block);vals.append(float((a[idx]-b[idx]).mean()*242))
                increments.append({'period':period,'cost':cost_id,'block':block,'annualized_arithmetic_timing_increment':float((a-b).mean()*242),
                                   'interval_95':interval(vals),'account_cagr_difference':summarize(candidate,cfg)['annualized_return']-m['annualized_return'],
                                   'multiple_selection_adjusted':False})
    write_json(OUT/'result.json',{'completed_at':now(),'status':'STATIC_HALF_RISK_ATTRIBUTION_COMPLETE','static_accounts':metrics,'increments':increments,
               'new_diagnostic_accounts':4,'new_candidate_configurations':0,'goal_achieved':False,'position_impact':0},exclusive=True)
    print(json.dumps({'固定半仓':metrics,'择时增量':increments},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
