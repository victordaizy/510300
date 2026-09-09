"""明确月末入场、每日退出、退出待成交及重新入场的完整研究账户。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import Account,affordable_quantity,fill_price,execute_order,require


def exit_reasons(current_value,entry_cost,peak_value,holding_days,trend_broken,monthly_reverse,config):
    reasons=[]
    if current_value/entry_cost-1<=-config['loss_stop_fraction']:reasons.append('含分红持仓收益触及止损')
    if current_value/peak_value-1<=-config['trailing_stop_fraction']:reasons.append('持仓价值从最高值回落触及退出线')
    if trend_broken:reasons.append('连续两个收盘低于一百二十日均线')
    if holding_days>=config['maximum_holding_trading_days']:reasons.append('最长持有期到期')
    if monthly_reverse:reasons.append('月末盈利或趋势条件明确不成立')
    return reasons


def simulate_entry_exit_account(data,dividends,config,cost,start,entry_signals,event_mask):
    first=int(np.flatnonzero(data.date>=pd.Timestamp(start))[0]);last=len(data)-1;anchor=first-1
    dates=pd.DatetimeIndex(data.date);op=data.open.to_numpy(float);cl=data.close.to_numpy(float)
    previous=data.previous_close.to_numpy(float);distribution=data.dividend.to_numpy(float);trend=data.sma120.to_numpy(float)
    require(len(entry_signals)==len(data) and len(event_mask)==len(data),'入场时钟长度不符')
    record_events={};ex_events={};pay_events={}
    for key,event in enumerate(dividends.itertuples()):
        for field,mapping in [('record_date',record_events),('ex_date',ex_events)]:
            when=getattr(event,field)
            if when in dates:mapping.setdefault(dates.get_loc(when),[]).append((key,event.cash_dividend_per_share))
        idx=dates.searchsorted(event.payment_date)
        if idx<len(dates):pay_events.setdefault(idx,[]).append((key,event.payment_date==dates[idx]))
    account=Account(config['initial_capital']);previous_nav=config['initial_capital'];previous_mark=cl[anchor]
    records=[];decisions=[];cycles=[];cycle=None;cycle_number=0;last_exit=-1000000
    pending_exit=False;pending_reasons=[];peak_value=np.nan;cycle_dividends=0.

    def decide(t):
        nonlocal pending_exit,pending_reasons
        scheduled=t==anchor or bool(event_mask[t]);signal=float(entry_signals[t])
        new_trigger=False;reasons=[];reference=1. if account.shares else 0.
        if account.shares:
            current_value=account.shares*cl[t]+cycle_dividends
            broken=t>0 and np.isfinite(trend[t-1:t+1]).all() and (trend[t-1:t+1]<=0).all()
            if not pending_exit:
                reasons=exit_reasons(current_value,cycle['entry_cost_cny'],peak_value,t-cycle['entry_index']+1,
                                     bool(broken),scheduled and np.isfinite(signal) and signal==0,config)
                if reasons:pending_exit=True;pending_reasons=reasons;new_trigger=True
            if pending_exit:
                quantity=-account.shares;reference=0.;state='EXPLICIT_EXIT_POLICY_WAITING_FOR_FILL';action='按独立退出规则清仓，未成交则继续请求'
                reasons=pending_reasons
            else:
                quantity=0;state='HOLD_WITH_DAILY_EXIT_CHECK';action='保持已有份额并每日检查退出，不在持仓周期内加仓'
                if scheduled and not np.isfinite(signal):state='NO_EPS_VIEW_HOLD_UNDER_EXISTING_TIME_AND_RISK_RULES'
        elif scheduled and np.isfinite(signal) and signal==1:
            if t-last_exit>=config['reentry_cooldown_trading_days']:
                quantity=affordable_quantity(account.cash,fill_price(cl[t],1,cost,config['tick']),cost,config['lot'])
                reference=1.;state='EXPLICIT_MONTH_END_ENTRY';action='盈利与趋势条件成立，下一开盘入场'
            else:quantity=0;state='EXPLICIT_REENTRY_COOLDOWN';action='等待卖出后的冷静期结束'
        else:quantity=0;state='WAIT_FOR_NEW_ELIGIBLE_MONTH_END_ENTRY';action='空仓等待下一次符合条件的月末'
        row={'origin':dates[t],'execution_date':dates[t+1] if t+1<len(dates) else pd.NaT,'origin_index':t,
             'simulation_only':True,'requested_quantity':int(quantity),'reference_weight':reference,'signal_state':state,'action':action,
             'entry_signal':signal,'exit_reasons':'；'.join(reasons),'new_exit_trigger':new_trigger,
             'cycle_id':cycle['cycle_id'] if cycle else None,'last_exit_index':last_exit}
        decisions.append(row);return row

    pending=decide(anchor)
    for day in range(first,last+1):
        old_shares=account.shares;recognized=0.;paid=0.
        for key,amount in ex_events.get(day,[]):
            value=account.entitlements.get(key,0)*amount;account.receivables[key]=value;recognized+=value
        if old_shares:cycle_dividends+=recognized
        for key,same in pay_events.get(day,[]):
            if not same:value=account.receivables.pop(key,0.);paid+=value;account.cash+=value
        terminal=day==last;quantity=-account.shares if terminal else int(pending['requested_quantity'])
        before=account.value(op[day]);execution=execute_order(account,quantity,op[day],previous[day],distribution[day],day,cost,config)
        mark=op[day] if terminal else cl[day]
        if old_shares==0 and account.shares>0:
            cycle_number+=1;cycle_dividends=0.;pending_exit=False;pending_reasons=[]
            cycle={'cycle_id':cycle_number,'entry_date':str(dates[day].date()),'entry_index':day,'entry_signal_origin':str(pending['origin'].date()),
                   'entry_quantity':account.shares,'entry_fill_price':execution['fill_price'],'entry_commission':execution['commission'],
                   'entry_cost_cny':execution['notional']+execution['commission']}
            peak_value=cycle['entry_cost_cny']
        elif old_shares>0 and account.shares==0:
            proceeds=execution['notional']-execution['commission']
            cycle.update({'exit_date':str(dates[day].date()),'exit_index':day,'exit_quantity':-execution['filled_quantity'],
                          'exit_fill_price':execution['fill_price'],'exit_commission':execution['commission'],'exit_net_proceeds_cny':proceeds,
                          'dividend_entitlement_cny':cycle_dividends,'holding_open_to_open_trading_intervals':day-cycle['entry_index'],
                          'cycle_net_profit_cny':proceeds+cycle_dividends-cycle['entry_cost_cny'],
                          'cycle_net_return':(proceeds+cycle_dividends)/cycle['entry_cost_cny']-1,
                          'exit_signal_origin':str(pending['origin'].date()) if not terminal else str(dates[day].date()),
                          'exit_reasons':pending['exit_reasons'] if not terminal else '研究终点统一开盘退出'})
            cycles.append(cycle);cycle=None;last_exit=day;pending_exit=False;pending_reasons=[];peak_value=np.nan;cycle_dividends=0.
        elif account.shares and execution['filled_quantity']:
            raise ValueError('本策略持仓周期内禁止增加或部分减少份额')
        if not terminal:
            for key,same in pay_events.get(day,[]):
                if same:value=account.receivables.pop(key,0.);paid+=value;account.cash+=value
            for key,amount in record_events.get(day,[]):account.entitlements[key]=account.shares
        if account.shares:peak_value=max(peak_value,account.shares*mark+cycle_dividends)
        nav=account.value(mark);price_pnl=old_shares*(op[day]-previous_mark)+account.shares*(mark-op[day])
        error=nav-previous_nav-price_pnl-recognized+execution['commission']+execution['slippage_cost']
        require(abs(error)<1e-6,'明确进出场账户未通过净值恒等式');account.assert_valid()
        records.append({'date':dates[day],'open':op[day],'mark':mark,'mark_clock':'OPEN_TERMINAL' if terminal else 'CLOSE',
                        'cash':account.cash,'shares':account.shares,'dividend_receivable':account.receivable(),'equity':nav,
                        'net_return':nav/previous_nav-1,'pnl':nav-previous_nav,'price_pnl':price_pnl,'dividend_recognized':recognized,'dividend_paid':paid,
                        'exposure':account.shares*mark/nav,'accounting_error':error,'origin':dates[day-1],
                        'terminal_unliquidated':bool(terminal and account.shares),'turnover':execution['notional']/before,
                        'cycle_id':cycle['cycle_id'] if cycle else None,'holding_days':day-cycle['entry_index']+1 if cycle else 0,
                        'cycle_value_cny':account.shares*mark+cycle_dividends if cycle else None,
                        'cycle_entry_cost_cny':cycle['entry_cost_cny'] if cycle else None,'cycle_peak_value_cny':peak_value,
                        'cycle_dividend_entitlement_cny':cycle_dividends,'execution_exit_reasons':pending['exit_reasons'] if quantity<0 and not terminal else '研究终点统一退出' if terminal else '',**execution})
        previous_nav=nav;previous_mark=mark
        if not terminal:pending=decide(day)
    if cycle:
        cycles.append({**cycle,'exit_date':None,'exit_reasons':'研究终点仍未成交，保留未平仓状态'})
    return pd.DataFrame(records),pd.DataFrame(decisions),pd.DataFrame(cycles)
