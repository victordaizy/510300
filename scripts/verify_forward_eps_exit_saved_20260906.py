"""核对保存的入场、退出触发、真实成交、逐周期损益及完整账户。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def close(a,b,tolerance=1e-10):
    if a is None:a=np.nan
    if b is None:b=np.nan
    np.testing.assert_allclose(a,b,atol=tolerance,rtol=0,equal_nan=True)


def verify(root,study):
    stem='510300_forward_eps_'+study;out=root/'reports/research'/stem
    config=read(root/'config'/(stem+'.json'));result=read(out/'result.json')
    price=pd.read_parquet(root/'reports/research/510300_adaptive_allocation_v1/features.parquet')
    origin_map={pd.Timestamp(day):i for i,day in enumerate(price.date)}
    account_count=0;cycle_count=0;trigger_count=0;blocked_exits=0
    for metric in result['all_metrics']:
        model=metric['model'];cost=metric['cost']
        ledger=pd.read_parquet(out/'evaluation'/cost/(model+'_ledger.parquet'))
        decisions=pd.read_parquet(out/'evaluation'/cost/(model+'_decisions.parquet'))
        r=ledger.net_return.to_numpy(float);wealth=np.r_[1,np.cumprod(1+r)]
        average=r.mean()*242;vol=r.std(ddof=1)*np.sqrt(242)
        for key,value in {'cumulative_return':wealth[-1]-1,'annualized_return':wealth[-1]**(242/len(r))-1,
                          'annualized_arithmetic_mean':average,'annualized_volatility':vol,'net_sharpe':average/vol if vol>1e-15 else None,
                          'max_drawdown':(wealth/np.maximum.accumulate(wealth)-1).min(),'commission':ledger.commission.sum(),'slippage_cost':ledger.slippage_cost.sum()}.items():close(value,metric[key])
        close(ledger.equity,ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable,1e-6)
        close(ledger.equity,200000*wealth[1:],1e-6)
        close(ledger.pnl,ledger.price_pnl+ledger.dividend_recognized-ledger.commission-ledger.slippage_cost,1e-6)
        assert not ledger.terminal_unliquidated.iloc[-1]
        daily=model.startswith(('X1_','Y1_','Y3_'))
        if daily:
            cycle_file=out/(f'{cost}_{model}_逐次完整进出场.csv' if study=='forecast_entry_exit_v1' else f'{cost}_逐次完整进出场.csv')
            cycles=pd.read_csv(cycle_file);cycle_count+=len(cycles)
            for cycle in cycles.itertuples():
                a=pd.Timestamp(cycle.entry_date);b=pd.Timestamp(cycle.exit_date)
                buy=ledger.loc[ledger.date.eq(a)].iloc[0];sell=ledger.loc[ledger.date.eq(b)].iloc[0]
                assert buy.filled_quantity==cycle.entry_quantity and sell.filled_quantity==-cycle.entry_quantity
                assert b>a
                close(cycle.entry_cost_cny,buy.notional+buy.commission,1e-6)
                close(cycle.exit_net_proceeds_cny,sell.notional-sell.commission,1e-6)
                div=ledger.loc[ledger.date.between(a,b),'dividend_recognized'].sum()
                close(cycle.dividend_entitlement_cny,div,1e-6)
                profit=sell.notional-sell.commission+div-buy.notional-buy.commission
                close(cycle.cycle_net_profit_cny,profit,1e-6)
                close(cycle.cycle_net_return,profit/cycle.entry_cost_cny)
                assert cycle.holding_open_to_open_trading_intervals==origin_map[b]-origin_map[a]
            close(cycles.cycle_net_profit_cny.sum(),ledger.pnl.sum(),1e-6)
            by_date=ledger.set_index('date');dec=decisions.set_index('origin')
            for row in decisions.itertuples():
                t=int(row.origin_index)
                if row.requested_quantity>0:
                    assert row.entry_signal==1 and t-row.last_exit_index>=5
                    if row.origin in by_date.index:assert by_date.loc[row.origin,'shares']==0
                if row.origin not in by_date.index:continue
                day=by_date.loc[row.origin]
                if day.shares:
                    triggers=(day.cycle_value_cny/day.cycle_entry_cost_cny-1<=-.08
                              or day.cycle_value_cny/day.cycle_peak_value_cny-1<=-.12
                              or day.holding_days>=60
                              or (t>0 and np.isfinite(price.sma120.iloc[t-1:t+1]).all() and price.sma120.iloc[t-1:t+1].le(0).all())
                              or (pd.notna(row.entry_signal) and row.entry_signal==0))
                    if triggers:assert row.requested_quantity==-day.shares
                    if row.new_exit_trigger:
                        assert triggers and row.exit_reasons;trigger_count+=1
                if day.requested_quantity<0 and day.filled_quantity==0 and day.shares:
                    assert row.requested_quantity==-day.shares;blocked_exits+=1
            for buy in ledger.loc[ledger.filled_quantity.gt(0)].itertuples():
                origin=price.date.iloc[origin_map[buy.date]-1]
                assert dec.loc[origin,'requested_quantity']>=buy.filled_quantity
                assert dec.loc[origin,'entry_signal']==1
        account_count+=1
    for cost in config['costs']:
        for block in config['bootstrap_day_blocks']:
            samples=pd.read_parquet(out/f'{cost}_block{block}_saved_bootstrap_statistics.parquet');assert len(samples)==2000
            for column in samples:
                values=samples[column].dropna().to_numpy();got=np.quantile(values,[.025,.975]) if len(values) else [None,None]
                expected=result['uncertainty'][cost][str(block)][column+'_95_interval']
                for a,b in zip(got,expected):close(a,b)
    assert account_count==result['evaluation_accounts']
    return {'status':'PASS_SAVED_ENTRY_TRIGGER_FILL_CYCLE_AND_ACCOUNT_VERIFICATION','study':stem,
            'complete_accounts_checked':account_count,'completed_position_cycles_checked':cycle_count,'first_exit_triggers_checked':trigger_count,
            'unfilled_exit_continuations_checked':blocked_exits,'saved_bootstrap_files_checked':4,
            'new_accounts_generated':0,'new_models_fit':0,'new_downloads':0,'random_samples_regenerated':0,'security_audit_performed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--study',choices=['explicit_entry_exit_v2','forecast_entry_exit_v1'],required=True);parser.add_argument('--save',action='store_true')
    args=parser.parse_args();result=verify(args.root,args.study)
    if args.save:
        path=args.root/'reports/research'/('510300_forward_eps_'+args.study)/'saved_numerical_verification.json'
        with path.open('x',encoding='utf-8') as handle:json.dump(result,handle,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False),flush=True)
