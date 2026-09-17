"""只读保存账户，按前一收盘的核心与辅助资格分组核算损益。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import digest,now,require,write_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/research/510300_signal_stage_attribution_through183'
SOURCES={
    'EITHER_CONFIRMED_RUNS_AUXILIARY':('510300_return_confirmation_auxiliary_batch_v1','原174'),
    'EXPOSURE_EXPANSION_200':('510300_unlevered_exposure_expansion_v1','两倍180'),
    'ACCOUNT_VOLATILITY_EXPOSURE':('510300_account_volatility_exposure_v1','每日181'),
    'EPISODE_ACCOUNT_RISK_BUDGET':('510300_episode_account_risk_budget_v1','区间182'),
    'BINARY_SIGNAL_EXPOSURE':('510300_binary_signal_exposure_v1','满仓183'),
}


def main():
    require(not (OUT/'result.json').exists(),'该保存流水归因已经完成')
    OUT.mkdir(parents=True,exist_ok=True)
    rows,files=[],[Path(__file__)]
    for period in ['evaluation','earlier_diagnostic']:
        for cost in ['BASE','STRESS']:
            factor_path=ROOT/'reports/research/510300_return_confirmation_auxiliary_batch_v1'/period/cost/'factors.parquet'
            files.append(factor_path)
            factors=pd.read_parquet(factor_path)
            core=factors.TREND_NOISE_REFERENCE_BLEND_parent_target.to_numpy(float)
            auxiliary=factors.EITHER_CONFIRMED_RUNS_AUXILIARY_effective_auxiliary_target.to_numpy(float)
            known=np.isfinite(core)&np.isfinite(auxiliary)
            category=np.full(len(factors),'未知目标',dtype=object)
            category[known&(core==0)&(auxiliary==0)]='零目标或退出过渡'
            category[known&(core>0)&(auxiliary==0)]='核心独有'
            category[known&(core==0)&(auxiliary>0)]='辅助独有'
            category[known&(core>0)&(auxiliary>0)]='核心辅助同时'
            keys=pd.DataFrame({'origin':factors.date,'prior_close_signal_stage':category})
            for model,(name,label) in SOURCES.items():
                path=ROOT/'reports/research'/name/period/cost/f'{model}_ledger.parquet'
                files.append(path)
                ledger=pd.read_parquet(path)
                joined=ledger.merge(keys,on='origin',how='left',validate='one_to_one')
                require(len(joined)==len(ledger) and joined.prior_close_signal_stage.notna().all(),'来源状态与真实收益日错位')
                net_sum=0.
                for stage,group in joined.groupby('prior_close_signal_stage',sort=False):
                    gross=float(group.price_pnl.sum()+group.dividend_recognized.sum())
                    fees=float(group.commission.sum()+group.slippage_cost.sum())
                    net=float(group.pnl.sum())
                    require(abs(gross-fees-net)<1e-6,'分组资金损益不平')
                    net_sum+=net
                    rows.append({'period':period,'cost':cost,'model':model,'label':label,'stage':stage,
                        'calendar_days':len(group),'holding_closes':int(group.shares.gt(0).sum()),
                        'gross_price_dividend_profit':gross,'commission_and_slippage':fees,'net_profit':net,
                        'loss_day_profit_sum':float(group.loc[group.pnl.lt(0),'pnl'].sum()),
                        'positive_day_profit_sum':float(group.loc[group.pnl.gt(0),'pnl'].sum()),
                        'arithmetic_return_contribution':float(group.net_return.sum()),
                        'filled_trade_days':int(group.filled_quantity.ne(0).sum())})
                require(abs(net_sum-(float(ledger.equity.iloc[-1])-200000.))<1e-5,'分组损益未还原完整账户')
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'stage_pnl.csv',index=False,encoding='utf-8-sig')
    report=['# 至第183轮：按前一收盘信号来源分组的保存账户损益','',
        '本归因只读取五个既有策略、两段历史和两档费用的20条保存账户，不新跑账户、训练或选择新参数。各组以收益发生前一交易日收盘的原174核心和已确认辅助目标分类。',
        '分组金额加总等于完整账户净利润，但分组不是纯粹的模块增量收益。当前日还会承担旧份额的隔夜收益、调整费用、整手及分红，尤其“零目标或退出过渡”包含收到退出信号后至下一开盘的旧份额损益。',
        '这是对已观察历史的描述，不能将分组标签当作独立样本或未来预测。不得按分组结果拼接有利年份，不能据此宣称某模块单独必然产生这些收益。','',
        '|历史段|费用|方案|前一收盘资格|天数|价格及分红损益|费用|净利润|',
        '|---|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        report.append(f"|{'主' if row['period']=='evaluation' else '较早'}|{'基础' if row['cost']=='BASE' else '压力'}|{row['label']}|{row['stage']}|{row['calendar_days']}|{row['gross_price_dividend_profit']:.2f}|{row['commission_and_slippage']:.2f}|{row['net_profit']:.2f}|")
    (OUT/'信号阶段损益归因.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    receipt={'recorded_at':now(),'status':'COMPLETED_SAVED_LEDGER_DESCRIPTIVE_ATTRIBUTION_ONLY',
        'accounts':20,'groups':len(rows),'new_account_runs':0,'new_model_fits':0,'independent_validation':False,
        'all_group_profits_reconcile_to_account':True,
        'files':[{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in sorted(set(files))]}
    write_json(OUT/'result.json',receipt,exclusive=True)
    print(df.loc[df.cost.eq('STRESS'),['period','label','stage','net_profit','commission_and_slippage']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
