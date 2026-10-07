"""从原始价和公司行为直接复算新低窗口，并只读核对T02保存账户。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a,b,tol=1e-10):
    np.testing.assert_allclose(np.asarray(a,dtype=float),np.asarray(b,dtype=float),atol=tol,rtol=0,equal_nan=True)


def verify_members(root,market):
    legacy=pd.read_parquet(root/"raw/legacy_daily.parquet").rename(columns={"con_code":"symbol",**{"raw_"+k:k for k in ["open","high","low","close"]}})
    fresh=pd.read_parquet(root/"raw/fresh_daily.parquet").rename(columns={"ts_code":"symbol","trade_date":"date"})
    fresh["date"]=pd.to_datetime(fresh.date,format="%Y%m%d")
    cols=["date","symbol","open","high","low","close"]
    raw=pd.concat([legacy[cols],fresh[cols]],ignore_index=True).drop_duplicates(["date","symbol"],keep="last").sort_values(["date","symbol"]).reset_index(drop=True)
    saved=pd.read_parquet(root/"normalized_raw_prices.parquet").reset_index(drop=True)
    pd.testing.assert_frame_equal(raw,saved,check_dtype=False)
    q=pd.read_parquet(root/"inputs/classified.parquet").merge(raw,on=["date","symbol"],how="left",validate="one_to_one")
    suspension=q.return_is_usable&q.constituent_return_state.eq("OFFICIAL_SUSPENSION")
    assert q.loc[suspension,"corporate_action_status"].eq("NONE_CONFIRMED").all()
    assert q.loc[suspension,"previous_unadjusted_close"].gt(0).all()
    q.loc[suspension,"unadjusted_close"]=q.loc[suspension,"previous_unadjusted_close"]
    for c in ["open","high","low","close"]:
        q.loc[suspension,c]=q.loc[suspension,"unadjusted_close"]
    gross=(q.unadjusted_close*q.post_to_pre_share_ratio+q.cash_distribution_per_pre_event_share-q.subscription_cash_outflow_per_pre_event_share)/q.previous_unadjusted_close
    close(q.loc[q.return_is_usable,"daily_total_shareholder_return"],gross[q.return_is_usable]-1)
    good=(q.return_is_usable&q.constituent_return_state.isin(["TRADED_VALID","OFFICIAL_SUSPENSION"])
        &np.isclose(q.close,q.unadjusted_close,atol=1e-9,rtol=0)&q.low.gt(0)&q.low.le(q.close)&q.low.le(q.open)
        &q.high.ge(q.open)&q.high.ge(q.close)&q.high.ge(q.low)&np.isfinite(gross)&gross.gt(0))
    membership=pd.read_parquet(root/"inputs/membership.parquet").rename(columns={"membership_date":"date"})
    membership["date"]=pd.to_datetime(membership.date)
    symbols=sorted(membership.symbol.unique())
    dates=pd.DatetimeIndex(market.date)
    mask=membership.assign(yes=True).pivot(index="date",columns="symbol",values="yes").reindex(index=dates,columns=symbols).eq(True)
    mask.columns.name=None
    pd.testing.assert_frame_equal(mask,pd.read_parquet(root/"membership_mask.parquet"))
    def wide(values):
        return q.assign(value=values).pivot(index="date",columns="symbol",values="value").reindex(index=dates,columns=symbols).to_numpy(float)
    cash=q.cash_distribution_per_pre_event_share-q.subscription_cash_outflow_per_pre_event_share
    ratios=wide(((q.low*q.post_to_pre_share_ratio+cash)/(q.close*q.post_to_pre_share_ratio+cash)).where(good))
    gross_array=wide(gross.where(good))
    good_array=np.isfinite(ratios)&np.isfinite(gross_array)&(ratios>0)&(gross_array>0)
    flags=np.zeros(gross_array.shape,dtype=bool)
    valid=np.zeros(gross_array.shape,dtype=bool)
    # 每个21日窗口从1重新起算，没有用缺口零填后的长财富序列。
    for end in range(20,len(dates)):
        start=end-20
        ok=good_array[start:end+1].all(axis=0)
        valid[end]=ok
        if ok.any():
            relative=np.cumprod(gross_array[start:end+1,ok],axis=0)*ratios[start:end+1,ok]
            flags[end,ok]=relative[-1]<relative[:-1].min(axis=0)*(1-1e-12)
    assert np.array_equal(valid,pd.read_parquet(root/"member_low_valid.parquet").to_numpy())
    assert np.array_equal(flags,pd.read_parquet(root/"member_new_low_flags.parquet").to_numpy())
    coverage=pd.read_parquet(root/"member_daily_coverage.parquet")
    close(coverage.members,mask.sum(axis=1))
    close(coverage.complete_low20_members,(mask.to_numpy()&valid).sum(axis=1))
    close(coverage.new_low_members,(mask.to_numpy()&flags).sum(axis=1))
    assert coverage.known.equals((coverage.members.eq(300)&coverage.complete_low20_members.ge(294)).rename("known"))
    return mask,pd.DataFrame(flags,index=dates,columns=symbols),pd.DataFrame(valid,index=dates,columns=symbols),coverage


def verify_events(root,market,mask,flags,valid,coverage):
    features={lag:pd.read_parquet(root/f"daily_features_lag{lag}.parquet") for lag in [0,1]}
    f=features[0]
    total_log=np.log((market.close+market.dividend)/market.close.shift())
    wealth=np.exp(total_log.fillna(0).cumsum())
    low=wealth*(market.low+market.dividend)/(market.close+market.dividend)
    close(f.wealth,wealth)
    close(f.low_w,low)
    close(f.r,total_log)
    events=pd.read_parquet(root/"event_registry.parquet")
    first_used=set()
    for row in events.itertuples():
        a,b=int(row.first_test_idx),int(row.second_test_idx)
        assert a not in first_used
        first_used.add(a)
        assert row.first_confirm_idx==a+2 and 3<=b-a<=10
        assert low.iloc[a]<low.iloc[a-20:a].min()*(1-1e-12)
        assert low.iloc[a+1]>low.iloc[a] and low.iloc[a+2]>low.iloc[a]
        assert low.iloc[b]<low.iloc[b-20:b].min()*(1-1e-12)
        close(row.frozen_low,low.iloc[b-20:b].min())
        close(row.frozen_low,low.iloc[a])
        assert abs(low.iloc[b]-low.iloc[a])<.5*f.atr20.iloc[a-1]
        close(row.frozen_stop,row.frozen_low-.5*f.atr20.iloc[b-1])
        shared=mask.iloc[a]&mask.iloc[b]
        usable=shared&valid.iloc[a]&valid.iloc[b]
        assert row.usable_symbols=="|".join(mask.columns[usable])
        count=int(usable.sum())
        left,right=int(flags.iloc[a][usable].sum()),int(flags.iloc[b][usable].sum())
        known=count>=294 and coverage.known.iloc[a] and coverage.known.iloc[b]
        assert row.common_members==shared.sum() and row.common_usable_members==count
        assert row.first_new_low_count==left and row.second_new_low_count==right
        assert row.internal_known==known and row.internal_divergence==bool(known and right<left)
        close(row.E02_change,(right-left)/count if count else np.nan)
        candidates=np.flatnonzero(wealth.iloc[b:min(len(f),b+11)].gt(row.frozen_low).to_numpy())
        if len(candidates):
            c=b+int(candidates[0])
            assert row.outcome=="RECLAIM_CONFIRMED" and row.reclaim_idx==c
            assert f.price_signal.iloc[c] and f.event_id.iloc[c]==row.event_id
            assert f.internal_known.iloc[c]==known and f.internal_divergence.iloc[c]==row.internal_divergence
        else:
            assert pd.isna(row.reclaim_idx) and row.outcome!="RECLAIM_CONFIRMED"
    assert f.price_signal.sum()==events.outcome.eq("RECLAIM_CONFIRMED").sum()
    delayed=features[1]
    expected=f.price_signal.shift(fill_value=False)&f.wealth.gt(f.frozen_low.shift())
    assert delayed.price_signal.equals(expected.rename("price_signal"))
    for col in ["internal_known","internal_divergence"]:
        assert delayed[col].equals(f[col].shift(fill_value=False))
    return features,events


def metrics(z,capital):
    nav=np.r_[capital,z.equity]
    r=nav[1:]/nav[:-1]-1
    vol=r.std(ddof=1)*np.sqrt(242)
    return {"net_sharpe":r.mean()*242/vol if vol>1e-15 else np.nan,
        "cagr":(nav[-1]/capital)**(242/len(r))-1,"max_drawdown":1-(nav/np.maximum.accumulate(nav)).min(),
        "mean_exposure":z.exposure.mean()}


def verify(root):
    freeze=read(root/"freeze.json")
    for row in freeze["files"]:
        assert digest(root/row["path"])==row["sha256"],row["path"]
    started=read(root/"run_started.json")
    assert digest(root/"freeze.json")==started["freeze_sha256"]
    assert pd.Timestamp(freeze["at"])<pd.Timestamp(started["at"])
    market=pd.read_parquet(root/"inputs/market.parquet")
    market=market[market.date.le("2025-12-31")].reset_index(drop=True)
    mask,flags,valid,coverage=verify_members(root,market)
    features,events=verify_events(root,market,mask,flags,valid,coverage)
    print("原始低价、公司行为与每个21日新低窗口已独立复算。",flush=True)
    div=pd.read_csv(root/"inputs/dividends.csv",parse_dates=["record_date","ex_date","payment_date"])
    stored=pd.read_csv(root/"metrics.csv")
    annual=pd.read_csv(root/"annual_metrics.csv")
    assert len(stored)==40 and not stored.duplicated(["period","capital","cost","lag","policy"]).any()
    returns,rows,count={},0,0
    for row in stored.itertuples():
        folder=root/f"accounts/{row.period}/{row.capital}/{row.cost}/LAG{row.lag}/{row.policy}"
        z=pd.read_parquet(folder/"ledger.parquet")
        d=pd.read_parquet(folder/"decisions.parquet")
        nav=np.r_[row.capital,z.equity]
        close(z.net_return,nav[1:]/nav[:-1]-1,1e-12)
        close(z.equity,z.cash+z.shares*z.mark+z.dividend_receivable-z.terminal_exit_reserve,1e-7)
        close(z.shares,z.filled_quantity.cumsum(),0)
        fill=z.fill_price.astype(float).fillna(0.)
        close(z.cash,row.capital+(-z.filled_quantity*fill-z.commission+z.dividend_paid).cumsum(),1e-7)
        reserve=z.terminal_exit_reserve.diff().fillna(z.terminal_exit_reserve.iloc[0])
        close(np.diff(nav),z.price_pnl+z.dividend_recognized-z.commission-z.slippage_cost-reserve,1e-7)
        close(z.drawdown,1-nav[1:]/np.maximum.accumulate(nav)[1:],1e-12)
        assert z.cash.min()>=-1e-7 and z.shares.min()>=0 and z.shares.mod(100).eq(0).all()
        assert (d.origin<d.date).all() and d.date.equals(z.date)
        assert d.filled_quantity.equals(z.filled_quantity)
        assert (z.loc[z.filled_quantity.lt(0),"filled_quantity"].abs()<=z.loc[z.filled_quantity.lt(0),"sellable_before"]).all()
        rate,slip=(.0002,.0005) if row.cost=="BASE" else (.0004,.001)
        close(z.commission,np.where(z.filled_quantity.ne(0),np.maximum(abs(z.filled_quantity)*fill*rate,5.),0.))
        expected=np.where(z.filled_quantity.gt(0),np.ceil((z.open*(1+slip)-1e-12)/.001)*.001,np.floor((z.open*(1-slip)+1e-12)/.001)*.001)
        traded=z.filled_quantity.ne(0)
        close(fill[traded],expected[traded])
        close(z.slippage_cost,abs(z.filled_quantity)*abs(fill-z.open),1e-8)
        source=features[row.lag].iloc[d.origin_idx].reset_index(drop=True)
        assert d.origin.equals(source.date.rename("origin"))
        expected_signal=source.price_signal.copy()
        if row.policy!="PRICE_ALL":
            expected_signal&=source.internal_known
        if row.policy=="FULL":
            expected_signal&=source.internal_divergence
        assert d.active_signal.equals(expected_signal.rename("active_signal"))
        assert d.loc[d.filled_quantity.gt(0),"active_signal"].all()
        close(d.es95_5d,source.es95)
        for name,value in metrics(z,row.capital).items():
            close(getattr(row,name),value)
        aa=annual[(annual.period==row.period)&(annual.capital==row.capital)&(annual.cost==row.cost)&(annual.lag==row.lag)&(annual.policy==row.policy)]
        previous=row.capital
        for year,part in z.groupby(z.date.dt.year):
            s=aa[aa.year.eq(year)]
            assert len(s)==1
            for name,value in metrics(part,previous).items():
                close(s.iloc[0][name],value)
            previous=float(part.equity.iloc[-1])
        recognized,paid=np.zeros(len(z)),np.zeros(len(z))
        for event in div.itertuples():
            holding=z.loc[z.date.eq(event.record_date),"shares"]
            value=float(holding.iloc[0])*event.cash_dividend_per_share if len(holding) else 0.
            recognized[z.date.eq(event.ex_date)]=value
            paid[z.date.eq(event.payment_date)]=value
        close(z.dividend_recognized,recognized,1e-7)
        close(z.dividend_paid,paid,1e-7)
        close(z.dividend_receivable,np.cumsum(recognized-paid),1e-7)
        assert row.closed_cycles==int(((z.shares.shift(fill_value=0)>0)&z.shares.eq(0)).sum())
        returns[row.period,row.capital,row.cost,row.lag,row.policy]=z.net_return.to_numpy()
        rows+=len(z)
        count+=1
        if count%10==0:
            print(f"已复核{count}份T02保存账户。",flush=True)
    labels=pd.read_parquet(root/"inputs/mature_risk_labels.parquet")
    risk=read(root/"inputs/risk_training_records.json")
    for record in risk:
        ids=np.array(record["selected_indices"],int)
        t=record["decision_idx"]
        assert ids.max()+6<=t and market.date.iloc[ids.min()]>=market.date.iloc[t]-pd.DateOffset(years=2)
        expected=max(0.,-np.sort(labels.gross_return5.iloc[ids])[:int(np.ceil(.05*len(ids)))].mean())
        close([record["es95_5d"],features[0].es95.iloc[t]],[expected,expected],1e-12)
    indices=np.load(root/"bootstrap_indices.npz")["indices"]
    diff=returns["MAIN",200000,"STRESS",0,"FULL"]-returns["MAIN",200000,"STRESS",0,"PRICE_COMMON"]
    assert indices.shape==(4000,len(diff))
    draws=diff[indices].mean(axis=1)*242
    increment=read(root/"paired_increment.json")
    close([increment["annual_arithmetic_increment"],increment["ci95_low"],increment["ci95_high"]],
        [diff.mean()*242,*np.quantile(draws,[.025,.975])],1e-12)
    selected=stored[(stored.period=="MAIN")&(stored.cost=="STRESS")&(stored.lag==0)&(stored.policy=="FULL")]
    point=bool(((selected.net_sharpe>=1.2)&(selected.cagr>=.1)&(selected.max_drawdown<=.1)).all())
    assert point==read(root/"result.json")["historical_joint_point_pass"]
    return {"status":"PASS_SAVED_RAW_LOW_WINDOW_EVENT_AND_ACCOUNT_RECOMPUTATION","ledgers":count,"ledger_rows":rows,
        "symbol_count":mask.shape[1],"member_window_cells":int(mask.size),"events":len(events),"risk_records":len(risk),
        "new_strategy_accounts":0,"new_random_draws":0,"network_requests":0,"external_review":"NOT_PERFORMED",
        "independent_forward_validation":False,"scope":"逐个窗口、事件与账本的保存结果复核，不是新回测或经济有效性证明。"}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--receipt",type=Path)
    args=parser.parse_args()
    result=verify(args.root)
    if args.receipt:
        args.receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))
