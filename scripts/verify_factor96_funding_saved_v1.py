"""只读复核T14保存来源、时钟、特征和完整账户；不运行策略或重新抽样。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, tol=1e-10):
    np.testing.assert_allclose(np.asarray(actual,dtype=float),np.asarray(expected,dtype=float),rtol=0,atol=tol,equal_nan=True)


def source_and_features(root,market):
    """从原始银行响应和已冻结公开政策记录独立构建对齐列。"""
    dr=pd.read_parquet(root/"inputs/dr007.parquet")
    pieces=[]
    for path in sorted((root/"raw").glob("repo_daily*.json")):
        raw=load(path)["data"]
        f=pd.DataFrame(raw["items"],columns=raw["fields"])
        assert f.ts_code.eq("DR007.IB").all() and f.repo_maturity.eq("DR007").all()
        pieces.append(f)
    raw=pd.concat(pieces).sort_values("trade_date").reset_index(drop=True)
    assert pd.to_datetime(raw.trade_date,format="%Y%m%d").tolist()==dr.date.tolist()
    close(raw.weight,dr.dr007)
    policy=pd.read_parquet(root/"inputs/policy_rate_ledger.parquet")
    original=pd.read_parquet(root/"inputs/operation_changes.parquet")
    published=pd.read_parquet(root/"inputs/published_operation_rates.parquet")
    assert len(policy)==len(original)+1==26
    for row in policy.itertuples():
        assert digest(root/"raw"/Path(row.raw_path).name)==row.raw_sha256
        assert row.effective_date<=row.known_at.tz_localize(None).normalize()
        if row.record_kind=="PUBLISHED_OPERATION_RATE":
            matched=published[published.notice_date.eq(row.notice_date)]
            assert len(matched)==1
            close(matched.seven_day_rate_percent,[row.rate])
        else:
            node=next(r for r in load(root/"inputs/policy_nodes.json") if r["node_id"]=="R02")
            assert row.known_at==pd.Timestamp(node["source_available_upper"])
            assert row.effective_date==pd.Timestamp(node["underlying_execution_date"])
            close(row.rate,node["amount"])
    dates=pd.DatetimeIndex(market.date)
    bank_dates=pd.DatetimeIndex(dr.date)
    ends=bank_dates.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=23,minutes=59,seconds=59)
    policy_ids=np.searchsorted(pd.DatetimeIndex(policy.known_at).asi8,ends.asi8,side="right")-1
    assert (policy_ids>=0).all()
    available_ids=np.searchsorted(dates.values,bank_dates.values,side="right")
    eligible=available_ids<len(dates)
    bank=dr.loc[eligible,["date","dr007"]].reset_index(drop=True).rename(columns={"date":"fund_stat_date"})
    policy_ids=policy_ids[eligible]
    bank["available_at"]=dates[available_ids[eligible]].tz_localize("Asia/Shanghai")+pd.Timedelta(hours=9,minutes=30)
    bank["policy_known_at"]=policy.known_at.iloc[policy_ids].reset_index(drop=True)
    bank["rate"]=policy.rate.iloc[policy_ids].to_numpy()
    bank["effective_date"]=policy.effective_date.iloc[policy_ids].to_numpy()
    bank["gap_pp"]=bank.dr007-bank.rate
    stored=pd.read_parquet(root/"aligned_bank_source.parquet").reset_index(drop=True)
    for col in ["fund_stat_date","available_at","policy_known_at","effective_date"]:
        assert stored[col].equals(bank[col]),col
    close(stored[["dr007","rate","gap_pp"]],bank[["dr007","rate","gap_pp"]])
    decision=pd.DataFrame({"date":dates,"decision_time":dates.tz_localize("Asia/Shanghai")+pd.Timedelta(hours=16)})
    expected=pd.merge_asof(decision,bank.sort_values(["available_at","fund_stat_date"]),
        left_on="decision_time",right_on="available_at",direction="backward")
    age=(expected.date-expected.fund_stat_date).dt.days
    expected.loc[~age.between(1,10),"gap_pp"]=np.nan
    expected["K05"]=expected.gap_pp.rolling(5,min_periods=5).mean()
    expected["K05_q90"]=expected.K05.shift().rolling(252,min_periods=120).quantile(.9)
    expected["K05_q50"]=expected.K05.shift().rolling(252,min_periods=120).quantile(.5)
    features={}
    for lag in [1,2]:
        e=expected.copy()
        cols=[c for c in e.columns if c not in ["date","decision_time"]]
        if lag==2:
            e[cols]=e[cols].shift()
        age=(e.date-e.fund_stat_date).dt.days
        known=np.isfinite(e[["K05","K05_q90","K05_q50"]]).all(axis=1)&age.between(1,10)
        high=known&e.K05.gt(e.K05_q90)
        relief=known&high.shift(fill_value=False).rolling(5,min_periods=5).max().eq(1)&e.K05.lt(e.K05_q50)
        saved=pd.read_parquet(root/f"daily_features_lag{lag}.parquet")
        assert saved.date.equals(e.date)
        for col in ["fund_stat_date","available_at","policy_known_at","effective_date"]:
            assert saved[col].equals(e[col]),(lag,col)
        close(saved[["K05","K05_q90","K05_q50","gap_pp"]],e[["K05","K05_q90","K05_q50","gap_pp"]])
        for col,value in [("fund_known",known),("fund_high",high),("fund_relief",relief)]:
            assert saved[col].equals(value.rename(col)),(lag,col)
        assert (saved.loc[known,"fund_stat_date"]<saved.loc[known,"date"]).all()
        assert (saved.loc[known,"available_at"]<=saved.loc[known,"decision_time"]).all()
        features[lag]=saved
    r=np.log((market.close+market.dividend)/market.close.shift())
    wealth=np.exp(r.fillna(0.).cumsum())
    low=wealth*(market.low+market.dividend)/(market.close+market.dividend)
    close(features[1].r,r)
    close(features[1].wealth,wealth)
    close(features[1].low_w,low)
    close(features[1].rv20,r.rolling(20).std(ddof=1))
    assert features[1].shock.equals(r.lt(-2*r.rolling(20).std(ddof=1).shift()).rename("shock"))
    confirmations=features[1][features[1].price_confirmation]
    for row in confirmations.itertuples():
        origin=int(row.shock_origin)
        assert 1<=row.Index-origin<=10 and row.r>0
        assert not r.iloc[origin+1:row.Index].gt(0).any()
        close(row.shock_low,low.iloc[origin])
    return features,len(dr),len(policy)


def account_metrics(z,capital):
    nav=np.r_[float(capital),z.equity.to_numpy()]
    returns=nav[1:]/nav[:-1]-1
    vol=returns.std(ddof=1)*np.sqrt(242)
    return {"net_sharpe":returns.mean()*242/vol if vol>1e-15 else np.nan,
        "cagr":(nav[-1]/capital)**(242/len(returns))-1,
        "max_drawdown":1-np.min(nav/np.maximum.accumulate(nav)),"mean_exposure":z.exposure.mean()}


def verify(root):
    frozen=load(root/"freeze.json")
    for row in frozen["files"]:
        assert digest(root/row["path"])==row["sha256"],row["path"]
    started=load(root/"run_started.json")
    assert started["freeze_sha256"]==digest(root/"freeze.json")
    assert pd.Timestamp(frozen["at"])<pd.Timestamp(started["at"])
    market=pd.read_parquet(root/"inputs/market.parquet")
    market=market[market.date.le("2025-12-31")].reset_index(drop=True)
    features,dr_count,policy_count=source_and_features(root,market)
    dividends=pd.read_csv(root/"inputs/dividends.csv",parse_dates=["record_date","ex_date","payment_date"])
    metrics=pd.read_csv(root/"metrics.csv")
    annual=pd.read_csv(root/"annual_metrics.csv")
    assert len(metrics)==56 and not metrics.duplicated(["period","capital","cost","lag","policy"]).any()
    returns,ledgers,total_rows={},0,0
    for row in metrics.itertuples():
        folder=root/f"accounts/{row.period}/{row.capital}/{row.cost}/LAG{row.lag}/{row.policy}"
        z=pd.read_parquet(folder/"ledger.parquet")
        d=pd.read_parquet(folder/"decisions.parquet")
        nav=np.r_[float(row.capital),z.equity]
        close(z.net_return,nav[1:]/nav[:-1]-1,1e-12)
        close(z.equity,z.cash+z.shares*z.mark+z.dividend_receivable-z.terminal_exit_reserve,1e-7)
        close(z.shares,z.filled_quantity.cumsum(),0)
        fill=z.fill_price.astype(float).fillna(0.)
        close(z.cash,row.capital+(-z.filled_quantity*fill-z.commission+z.dividend_paid).cumsum(),1e-7)
        reserve_change=z.terminal_exit_reserve.diff().fillna(z.terminal_exit_reserve.iloc[0])
        close(np.diff(nav),z.price_pnl+z.dividend_recognized-z.commission-z.slippage_cost-reserve_change,1e-7)
        close(z.drawdown,1-nav[1:]/np.maximum.accumulate(nav)[1:],1e-12)
        assert z.cash.min()>=-1e-7 and z.shares.min()>=0 and z.shares.mod(100).eq(0).all()
        assert (d.origin<d.date).all() and z.date.equals(d.date)
        assert z.filled_quantity.equals(d.filled_quantity)
        assert (z.loc[z.filled_quantity.lt(0),"filled_quantity"].abs()<=z.loc[z.filled_quantity.lt(0),"sellable_before"]).all()
        cost_rate,slip=(.0002,.0005) if row.cost=="BASE" else (.0004,.001)
        traded=z.filled_quantity.ne(0)
        fees=np.where(traded,np.maximum(abs(z.filled_quantity)*fill*cost_rate,5.),0.)
        close(z.commission,fees,1e-9)
        sign=np.sign(z.filled_quantity)
        expected_fill=np.where(sign>0,np.ceil((z.open*(1+slip)-1e-12)/.001)*.001,
            np.floor((z.open*(1-slip)+1e-12)/.001)*.001)
        close(fill[traded],expected_fill[traded],1e-10)
        close(z.slippage_cost,abs(z.filled_quantity)*abs(fill-z.open),1e-8)
        source=features[row.lag].iloc[d.origin_idx].reset_index(drop=True)
        assert source.date.equals(d.origin.rename("date"))
        close(d.es95_5d,source.es95)
        for col in ["price_confirmation","fund_relief","fund_known","fund_high"]:
            assert d[col].equals(source[col]),col
        expected_signal=source.price_confirmation&(source.fund_relief if row.policy in ["ENTRY_ONLY","FULL"] else True)
        assert d.active_signal.equals(expected_signal.rename("active_signal"))
        buys=d.filled_quantity.gt(0)
        assert d.loc[buys,"active_signal"].all()
        if row.policy in ["ENTRY_ONLY","FULL"]:
            assert d.loc[buys,"fund_known"].all()
            assert (d.loc[buys,"fund_stat_date"]<d.loc[buys,"origin"]).all()
        calculated=account_metrics(z,row.capital)
        for name,value in calculated.items():
            close(getattr(row,name),value)
        group=annual[(annual.period==row.period)&(annual.capital==row.capital)&(annual.cost==row.cost)&(annual.lag==row.lag)&(annual.policy==row.policy)]
        prior=row.capital
        for year,part in z.groupby(z.date.dt.year):
            saved=group[group.year.eq(year)]
            assert len(saved)==1
            for name,value in account_metrics(part,prior).items():
                close(saved.iloc[0][name],value)
            prior=part.equity.iloc[-1]
        expected_recognized=np.zeros(len(z))
        expected_paid=np.zeros(len(z))
        for event in dividends.itertuples():
            shares=z.loc[z.date.eq(event.record_date),"shares"]
            amount=float(shares.iloc[0])*event.cash_dividend_per_share if len(shares) else 0.
            expected_recognized[z.date.eq(event.ex_date)]=amount
            expected_paid[z.date.eq(event.payment_date)]=amount
        close(z.dividend_recognized,expected_recognized,1e-7)
        close(z.dividend_paid,expected_paid,1e-7)
        close(z.dividend_receivable,np.cumsum(expected_recognized-expected_paid),1e-7)
        closed=int(((z.shares.shift(fill_value=0)>0)&z.shares.eq(0)).sum())
        assert closed==row.closed_cycles
        returns[row.period,row.capital,row.cost,row.lag,row.policy]=z.net_return.to_numpy()
        total_rows+=len(z)
        ledgers+=1
        if ledgers%16==0:
            print(f"已复核{ledgers}份T14保存账户。",flush=True)
    labels=pd.read_parquet(root/"inputs/mature_risk_labels.parquet")
    risk=load(root/"inputs/risk_training_records.json")
    for record in risk:
        selected=np.array(record["selected_indices"],int)
        t=record["decision_idx"]
        assert selected.max()+6<=t
        assert market.date.iloc[selected.min()]>=market.date.iloc[t]-pd.DateOffset(years=2)
        expected=max(0.,-np.sort(labels.gross_return5.iloc[selected])[:int(np.ceil(.05*len(selected)))].mean())
        close([record["es95_5d"],features[1].es95.iloc[t]],[expected,expected],1e-12)
    for row in labels.dropna().itertuples():
        entry,exit_=int(row.origin_idx)+1,int(row.exit_idx)
        cash=dividends.loc[dividends.record_date.ge(market.date.iloc[entry])&dividends.record_date.lt(market.date.iloc[exit_]),"cash_dividend_per_share"].sum()
        close(row.gross_return5,(market.open.iloc[exit_]+cash)/market.open.iloc[entry]-1,1e-12)
    indices=np.load(root/"bootstrap_indices.npz")["indices"]
    assert indices.shape==(4000,len(returns["MAIN",200000,"STRESS",1,"FULL"]))
    diff=returns["MAIN",200000,"STRESS",1,"FULL"]-returns["MAIN",200000,"STRESS",1,"PRICE_ONLY"]
    boot=diff[indices].mean(axis=1)*242
    increment=load(root/"paired_increment.json")
    close([increment["annual_arithmetic_increment"],increment["ci95_low"],increment["ci95_high"]],
        [diff.mean()*242,*np.quantile(boot,[.025,.975])],1e-12)
    selected=metrics[(metrics.period=="MAIN")&(metrics.cost=="STRESS")&(metrics.lag==1)&(metrics.policy=="FULL")]
    joint=bool(((selected.net_sharpe>=1.2)&(selected.cagr>=.1)&(selected.max_drawdown<=.1)).all())
    assert joint==load(root/"result.json")["historical_joint_point_pass"]
    return {"status":"PASS_SAVED_SOURCE_CLOCK_FEATURE_ACCOUNT_METRIC_RECOMPUTATION",
        "ledgers":ledgers,"ledger_rows":total_rows,"dr007_raw_rows":dr_count,"policy_clock_records":policy_count,
        "mature_risk_records":len(risk),"new_accounts":0,"new_random_draws":0,"network_requests":0,
        "independent_forward_validation":False,"external_review":"NOT_PERFORMED",
        "scope":"保存来源、时序与账户算术复核；不证明策略有效或历史首次版本真实性。"}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--receipt",type=Path)
    args=parser.parse_args()
    result=verify(args.root)
    if args.receipt:
        args.receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))
