"""只读复算候选库首批保存结果；不重跑账户，不发请求，不抽取新样本。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(root):
    active = root/"completed_run"
    checks, ledgers, rows = 0, 0, 0
    completion=json.loads((active/"data_completion_freeze.json").read_text(encoding="utf-8"))
    assert digest(root.parents[2]/"research/factor96_margin_source_repair_v1.py") == completion["driver_sha256"]
    assert digest(root/"freeze.json") == completion["original_freeze_sha256"]
    assert digest(root/"data_repair/source_receipt.json") == completion["source_receipt_sha256"]
    checks+=3
    for scope in [root, active]:
        frozen = json.loads((scope/"freeze.json").read_text(encoding="utf-8"))
        for r in frozen["files"]:
            assert digest(scope/r["path"]) == r["sha256"], r["path"]
            checks += 1
    prices = pd.read_parquet(active/"inputs/market.parquet")
    prices = prices[prices.date.le("2025-12-31")].reset_index(drop=True)
    div = pd.read_csv(active/"inputs/dividends.csv",parse_dates=["record_date","ex_date","payment_date"])
    margin = pd.read_parquet(active/"inputs/margin.parquet")
    assert margin.date.tolist() == prices.loc[prices.date.ge("2015-01-05"),"date"].tolist()
    assert len(margin) == 2674 and not margin.date.duplicated().any()
    missing=json.loads((root/"run_failure_01.json").read_text(encoding="utf-8"))["missing"]
    sh = pd.read_parquet(root/"data_repair/sse_local_official_matching_days.parquet").set_index("date")
    mm=margin.set_index("date")
    for day in missing:
        payload=json.loads((root/"data_repair"/("szse_"+day.replace("-","")+".json")).read_text(encoding="utf-8"))[0]
        assert payload["metadata"]["subname"] == day
        for name,key in [("rzye","jrrzye"),("rzmre","jrrzmr")]:
            value=float(payload["data"][0][key].replace(",",""))*1e8
            np.testing.assert_allclose(mm.loc[day,"market_"+name],sh.loc[day,name]+value,rtol=0,atol=.01)
            checks += 1
    stored=pd.read_csv(active/"metrics.csv")
    returns={}
    for row in stored.itertuples():
        folder=active/f"accounts/{row.period}/{row.capital}/{row.cost}/LAG{row.lag}/{row.policy}"
        z=pd.read_parquet(folder/"ledger.parquet")
        decision=pd.read_parquet(folder/"decisions.parquet")
        returns[row.period,row.capital,row.cost,row.lag,row.policy]=z.net_return.to_numpy()
        nav=np.r_[float(row.capital),z.equity.to_numpy()]
        r=nav[1:]/nav[:-1]-1
        np.testing.assert_allclose(z.net_return,r,atol=1e-12,rtol=0)
        np.testing.assert_allclose(z.equity,z.cash+z.shares*z.mark+z.dividend_receivable-z.terminal_exit_reserve,atol=1e-7,rtol=0)
        np.testing.assert_allclose(z.shares,np.cumsum(z.filled_quantity),atol=0,rtol=0)
        fill=z.fill_price.astype(float).fillna(0.)
        cash=row.capital+np.cumsum(-z.filled_quantity*fill-z.commission+z.dividend_paid)
        np.testing.assert_allclose(z.cash,cash,atol=1e-7,rtol=0)
        np.testing.assert_allclose(z.equity.diff().iloc[1:],(z.price_pnl+z.dividend_recognized-z.commission-z.slippage_cost-z.terminal_exit_reserve.diff()).iloc[1:],atol=1e-7,rtol=0)
        assert z.cash.min() >= -1e-7 and z.shares.min() >= 0
        assert z.shares.mod(100).eq(0).all()
        assert (decision.origin < decision.date).all()
        assert z.date.equals(decision.date)
        assert (z.loc[z.filled_quantity.lt(0),"filled_quantity"].abs() <= z.loc[z.filled_quantity.lt(0),"sellable_before"]).all()
        annual_mean, vol = r.mean()*242, r.std(ddof=1)*np.sqrt(242)
        calculated={"net_sharpe":annual_mean/vol if vol>1e-15 else np.nan,
            "cagr":(nav[-1]/row.capital)**(242/len(r))-1,
            "max_drawdown":1-np.min(nav/np.maximum.accumulate(nav)),"mean_exposure":z.exposure.mean()}
        for name,value in calculated.items():
            np.testing.assert_allclose(getattr(row,name),value,atol=1e-10,rtol=0,equal_nan=True)
        rate=.0002 if row.cost=="BASE" else .0004
        fee=np.where(z.filled_quantity.ne(0),np.maximum(abs(z.filled_quantity)*fill*rate,5.),0.)
        np.testing.assert_allclose(z.commission,fee,atol=1e-9,rtol=0)
        if row.policy in ["T03","T05"]:
            buys=decision.filled_quantity.gt(0)
            assert decision.loc[buys,"active_signal"].all()
            assert decision.loc[buys,"margin_known"].all()
            assert (decision.loc[buys,"margin_stat_date"] < decision.loc[buys,"origin"]).all()
            if row.policy=="T03":
                assert (decision.loc[buys,"F01"] < 0).all() and (decision.loc[buys,"F02"] > 0).all()
            else:
                assert (decision.loc[buys,"F06_implied_repay5"] > decision.loc[buys,"F06_q80"]).all()
        marks=z.set_index("date")
        entitlements={}
        outstanding=0.
        for record in z.itertuples():
            expected_recognized,expected_paid=0.,0.
            for j,event in enumerate(div.itertuples()):
                if event.record_date == record.date:
                    entitlements[j]=record.shares*event.cash_dividend_per_share
                if event.ex_date == record.date:
                    expected_recognized += entitlements.get(j,0.)
                if event.payment_date == record.date:
                    expected_paid += entitlements.get(j,0.)
            outstanding += expected_recognized-expected_paid
            np.testing.assert_allclose([record.dividend_recognized,record.dividend_paid,record.dividend_receivable],
                [expected_recognized,expected_paid,outstanding],atol=1e-7,rtol=0)
        ledgers+=1
        rows+=len(z)
        checks+=len(z)*10+20
        if ledgers % 16 == 0:
            print(f"已复算{ledgers}份保存账户。",flush=True)
    features=pd.read_parquet(active/"daily_features_lag1.parquet")
    labels=pd.read_parquet(active/"mature_risk_labels.parquet")
    recorded=json.loads((active/"risk_training_records.json").read_text(encoding="utf-8"))
    for r in recorded:
        selected=np.array(r["selected_indices"],int)
        t=r["decision_idx"]
        assert selected.max()+6 <= t
        assert prices.date.iloc[selected.min()] >= prices.date.iloc[t]-pd.DateOffset(years=2)
        realized=labels.gross_return5.iloc[selected].to_numpy()
        expected=max(0.,-np.sort(realized)[:int(np.ceil(.05*len(selected)))].mean())
        np.testing.assert_allclose([r["es95_5d"],features.es95.iloc[t]],[expected,expected],atol=1e-12,rtol=0)
        checks+=4
    for r in labels.dropna().itertuples():
        a,b=int(r.origin_idx)+1,int(r.exit_idx)
        cash=div.loc[div.record_date.ge(prices.date.iloc[a]) & div.record_date.lt(prices.date.iloc[b]),"cash_dividend_per_share"].sum()
        value=(prices.open.iloc[b]+cash)/prices.open.iloc[a]-1
        np.testing.assert_allclose(r.gross_return5,value,atol=1e-12,rtol=0)
        checks+=1
    index=np.load(active/"bootstrap_indices.npz")["indices"]
    comparisons=json.loads((active/"paired_increment.json").read_text(encoding="utf-8"))
    for r in comparisons:
        diff=returns["MAIN",200000,"STRESS",1,r["policy"]]-returns["MAIN",200000,"STRESS",1,r["policy"]+"_PRICE"]
        boot=diff[index].mean(axis=1)*242
        expected=[diff.mean()*242,*np.quantile(boot,[.025,.975])]
        np.testing.assert_allclose([r["annual_arithmetic_increment"],r["ci95_low"],r["ci95_high"]],expected,atol=1e-12,rtol=0)
        checks+=3
    result={"status":"PASS_SAVED_SOURCES_CLOCK_ACCOUNT_AND_METRIC_RECOMPUTATION","ledgers":ledgers,
        "ledger_rows":rows,"risk_records":len(recorded),"checks":checks,"new_accounts":0,
        "new_random_draws":0,"network_requests":0,"independent_forward_validation":False,
        "scope":"核对保存数据和算术，不等于策略有效或外部审阅"}
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--receipt",type=Path)
    args=parser.parse_args()
    result=verify(args.root)
    if args.receipt:
        args.receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))
