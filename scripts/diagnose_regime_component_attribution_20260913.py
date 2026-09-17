"""按收盘前可知的市场状态，归因保存策略在两段历史的实际账户表现。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_regime_component_attribution_v1'
FEATURES=ROOT/'reports/research/510300_adaptive_allocation_v1/features.parquet'
SOURCES={
 'TREND_NOISE_REFERENCE_BLEND':ROOT/'reports/research/510300_trend_noise_reference_blend_v1',
 'RETURN_RUNS_STATE':ROOT/'reports/research/510300_return_runs_state_v1',
 'RUNS_OPPORTUNITY_CAPPED_SUM':ROOT/'reports/research/510300_runs_opportunity_union_v1',
 'EITHER_CONFIRMED_RUNS_AUXILIARY':ROOT/'reports/research/510300_return_confirmation_auxiliary_batch_v1',
 'BUY_HOLD':ROOT/'reports/research/510300_rearmed_session_exit_v1',
}

def state_frame(data):
    x=data[['date','wealth','vol20']].copy()
    x['wealth_sma120']=x.wealth.rolling(120,min_periods=120).mean()
    x['peak60']=x.wealth.rolling(60,min_periods=60).max()
    x['drawdown60']=x.wealth/x.peak60-1
    trend=x.wealth>x.wealth_sma120
    stress=x.drawdown60<=-.05
    high_vol=x.vol20>.20
    x['state']=np.select([stress,trend&high_vol,trend&~high_vol,~trend],['压力回撤','上升高波动','上升稳定','非上升'],default='未知')
    return x[['date','state','drawdown60','vol20']]

def summary(group):
    r=group.net_return.to_numpy(float)
    daily=float(np.mean(r)); vol=float(np.std(r,ddof=1)) if len(r)>1 else np.nan
    return {'days':len(group),'mean_daily_return':daily,'annualized_arithmetic_return':daily*242,
        'annualized_volatility':vol*np.sqrt(242) if np.isfinite(vol) else np.nan,
        'conditional_sharpe':daily/vol*np.sqrt(242) if vol>0 else np.nan,
        'net_pnl':float(group.pnl.sum()),'trade_days':int(group.filled_quantity.ne(0).sum()),
        'mean_exposure':float(group.exposure.mean())}

def main():
    if OUT.exists() and any(OUT.iterdir()): raise RuntimeError('归因目录已存在，不覆盖')
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_parquet(FEATURES)
    states=state_frame(data)
    rows=[]
    for period,start,end in [('主历史','2020-01-02','2026-08-14'),('较早历史','2015-01-05','2019-12-31')]:
        state=states.copy()
        for cost in ['BASE','STRESS']:
            for model,folder in SOURCES.items():
                ledger=pd.read_parquet(folder/('evaluation' if period=='主历史' else 'earlier_diagnostic')/cost/f'{model}_ledger.parquet')
                merged=ledger.merge(state.rename(columns={'date':'origin'}),on='origin',how='left',validate='many_to_one')
                if merged.state.isna().any(): raise RuntimeError('状态日期未完全对齐')
                for label,g in merged.groupby('state',sort=False): rows.append({'period':period,'cost':cost,'model':model,'state':label,**summary(g)})
    result=pd.DataFrame(rows);result.to_csv(OUT/'state_component_metrics.csv',index=False,encoding='utf-8-sig')
    order=['压力回撤','非上升','上升高波动','上升稳定','未知']
    lines=['# 510300 保存策略的市场状态归因','',
      '本诊断不产生新账户、不选择新策略。状态只使用每个执行日之前收盘已知的含分红累计财富、一百二十日均线、六十日回撤和二十日年化波动率。',
      '状态定义：压力回撤为六十日回撤不高于负5%；上升高波动为累计财富高于120日均线且年化波动率高于20%；上升稳定为高于120日均线且波动率不高于20%；其余为非上升。',
      '条件夏普是该状态内实际账户日收益的均值除以标准差后年化，状态天数较少或不连续时不能与完整历史夏普直接等同。','']
    for period in ['主历史','较早历史']:
      lines+=['## '+period,'']
      for cost in ['BASE','STRESS']:
        lines+=['### '+('基础费用' if cost=='BASE' else '压力费用'),'','|状态|策略|天数|条件夏普|状态内净损益（元）|交易日|平均仓位|','|---|---|---:|---:|---:|---:|---:|']
        part=result[(result.period==period)&(result.cost==cost)].copy();part['state']=pd.Categorical(part.state,order,ordered=True)
        for _,r in part.sort_values(['state','model']).iterrows(): lines.append(f"|{r.state}|{r.model}|{r.days}|{r.conditional_sharpe:.3f}|{r.net_pnl:.2f}|{r.trade_days}|{r.mean_exposure:.2%}|")
        lines.append('')
    (OUT/'状态归因报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'records':len(result),'output':str(OUT)},ensure_ascii=False))
if __name__=='__main__': main()
