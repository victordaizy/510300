"""从已保存账户计算固定二信号组合超额的不确定性，不重新训练或改仓位。"""
from pathlib import Path
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import now,write_json,block_indices,return_metrics,interval

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_simple_pair_saved_uncertainty_v1'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'protocol.json',{'registered_at':now(),'model':'B2_TREND_SESSION','periods':['evaluation','earlier_diagnostic'],
               'costs':['BASE','STRESS'],'blocks':[20,60],'bootstrap_repetitions':2000,'seed':20260907,'annual_days':242,
               'purpose':'已有历史筛选结果的未校正不确定性诊断；不改规则、来源、期间或持仓，不构成独立验证'},exclusive=True)
    rows=[]
    for period in ['evaluation','earlier_diagnostic']:
        for cost in ['BASE','STRESS']:
            p=ROOT/'reports/research/510300_simple_signal_blend_v1'/period/cost
            a=pd.read_parquet(p/'B2_TREND_SESSION_ledger.parquet')
            bp=(ROOT/'reports/research/510300_simple_price_entry_exit_v1/evaluation'/cost/'BUY_HOLD_ledger.parquet') if period=='evaluation' else p/'BUY_HOLD_ledger.parquet'
            b=pd.read_parquet(bp)
            assert pd.DatetimeIndex(a.date).equals(pd.DatetimeIndex(b.date))
            ar,br=a.net_return.to_numpy(),b.net_return.to_numpy()
            for block in [20,60]:
                rng=np.random.default_rng(20260907+block);samples=[]
                for k in range(2000):
                    idx=block_indices(rng,len(ar),block)
                    samples.append({'sample':k,'net_sharpe':return_metrics(ar[idx],242)['net_sharpe'],
                                    'annualized_arithmetic_excess':float((ar[idx]-br[idx]).mean()*242)})
                s=pd.DataFrame(samples);s.to_parquet(OUT/f'{period}_{cost}_block{block}_samples.parquet',index=False)
                rows.append({'period':period,'cost':cost,'block':block,'net_sharpe':return_metrics(ar,242)['net_sharpe'],
                             'annualized_arithmetic_excess':float((ar-br).mean()*242),
                             'sharpe_95_interval':interval(s.net_sharpe.to_numpy()),
                             'annualized_arithmetic_excess_95_interval':interval(s.annualized_arithmetic_excess.to_numpy()),
                             'multiple_selection_adjustment':False})
    write_json(OUT/'result.json',{'completed_at':now(),'status':'SAVED_PAIR_ACCOUNT_UNCERTAINTY_COMPLETE','results':rows,
               'new_accounts_generated':0,'new_models_fit':0,'goal_achieved':False,'independent_validation':'NOT_ESTABLISHED'},exclusive=True)
    print(rows,flush=True)

if __name__=='__main__':main()
