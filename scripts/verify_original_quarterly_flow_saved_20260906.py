"""只读复算第十轮已保存的原始数值、时钟、预测、账户和区间。"""
from __future__ import annotations
import json
import sys
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.original_quarterly_flow_policy_v1 import (
    OUT,SOURCE,PARENT,MANIFEST,CONFIG,MODELS,RULES,FLOW,FEATURES,
    physical,sha,eligible_training,flow_values,source_event_date,
)
from research.original_fund_quarterly_facts_v1 import available_clock
from research.intraday_overnight_increment_v1 import normalize_dividends,holding_total_return,interval
from research.adaptive_allocation_v1 import summarize


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_saved():
    config=read(CONFIG)
    result=read(OUT/"result.json")
    manifest=read(MANIFEST)
    assert sha(MANIFEST)==result["manifest_sha256"]
    for row in manifest["files"]:
        assert sha(ROOT/row["path"])==row["sha256"],row["path"]
    amendment=read(SOURCE/"catalog_title_and_missing_source_amendment_v1_1.json")
    assert amendment["script_sha256"]==sha(ROOT/"scripts/reconcile_original_fund_quarterly_catalog_20260906.py")
    data=pd.read_parquet(PARENT/"features.parquet")
    dividends=normalize_dividends(pd.read_csv(physical(ROOT/config["inputs"]["dividends"])))
    calendar=pd.read_parquet(physical(ROOT/config["inputs"]["calendar"]))
    days=pd.DatetimeIndex(calendar.loc[calendar.is_open,"trade_date"])
    facts=pd.read_parquet(SOURCE/"quarterly_share_flow_facts_v1_1.parquet")
    assert len(facts)==56 and facts.period.is_unique and int(facts.source_usable.sum())==55
    assert facts.quarter_transition_residual.iloc[1:].abs().le(.01).all()
    sources={r["period"]:r for r in read(SOURCE/"quarterly_download_v1_1_result.json")["rows"]}
    factmap={r["period"]:r for r in facts.to_dict("records")}
    for row in facts.to_dict("records"):
        record=read(SOURCE/"quarterly_fact_records_v1_1"/(row["period"]+".json"))
        table=record["flow_table"]
        raw=table["raw_rows"]
        exact={k:(Decimal(0) if k=="split_delta_units" and v["raw_cell"].strip() in ["-","—","－"] else Decimal(v["raw_cell"].replace(",","").strip())) for k,v in raw.items()}
        for key,value in exact.items():
            assert float(value)==row[key]==table["values"][key]
        assert exact["beginning_units"]+exact["gross_subscription_units"]-exact["gross_redemption_units"]+exact["split_delta_units"]==exact["ending_units"]
        source=sources[row["period"]]
        assert sha(ROOT/source["raw_path"])==row["source_sha256"]==source["sha256"]
        clock=available_clock(source,record["identity"]["reported_send_date"],record["pdf_metadata"],days)
        assert clock["source_usable"]==row["source_usable"]
        assert (clock["feature_available_session"] is None and pd.isna(row["feature_available_session"])) or clock["feature_available_session"]==row["feature_available_session"]
    expected=(data.close+data.dividend)/data.close.shift(1)-1
    wealth=(1+expected.fillna(0)).cumprod()
    rebuilt={"mom60":np.log1p(expected).rolling(60).sum(),"sma120":wealth/wealth.rolling(120).mean()-1,
             "vol60":expected.rolling(60).std(ddof=1)*np.sqrt(242)}
    for name,series in rebuilt.items():
        np.testing.assert_allclose(series,data[name],rtol=0,atol=1e-12,equal_nan=True)
    events=pd.read_parquet(OUT/"evaluation_event_features.parquet")
    quarters=pd.read_parquet(OUT/"quarter_feature_origins.parquet")
    labels=pd.read_parquet(OUT/"quarterly_labels.parquet")
    signals=pd.read_parquet(OUT/"event_signals.parquet")
    assert len(events)==28 and events.origin_index.is_unique and events.all_features_valid.all()
    for frame in [quarters,events]:
        for _,row in frame.iterrows():
            source=factmap[row.period]
            prior=factmap.get(str(pd.Period(row.period,freq="Q")-1))
            if row.is_quarter_training_origin:
                expected_date=source_event_date(source,days)
                assert (pd.isna(expected_date) and pd.isna(row.origin)) or expected_date==row.origin
            if row.all_features_valid:
                values,state=flow_values(source,prior,row.origin)
                assert state.startswith("PASS_")
                np.testing.assert_allclose([values[k] for k in FLOW],row[FLOW].astype(float),rtol=0,atol=1e-12)
                for name in rebuilt:
                    assert abs(row[name]-rebuilt[name].iloc[int(row.origin_index)])<1e-12
    for _,row in labels.loc[labels.Y60.notna()].iterrows():
        t=int(row.origin_index)
        assert row.label_exit_date==data.date.iloc[t+61]
        assert abs(row.Y60-holding_total_return(data,dividends,t+1,t+61)[0])<1e-12
    receipts=read(OUT/"training_receipts.json")["rows"]
    assert len(receipts)==84 and all(r["status"].startswith("TRAINED_") for r in receipts)
    for receipt in receipts:
        origin=pd.Timestamp(receipt["origin"])
        row=events.loc[events.origin==origin].iloc[0]
        train=eligible_training(labels,origin)
        assert len(train)>=12 and len(train)==receipt["training_samples"]
        assert train.period.tolist()==receipt["training_periods"]
        assert str(train.label_exit_date.max().date())==receipt["latest_label_exit_date"]
        model_path=ROOT/receipt["model_path"]
        assert sha(model_path)==receipt["model_sha256"]
        model=joblib.load(model_path)
        columns=MODELS[receipt["model"]]
        np.testing.assert_allclose(model[0].mean_,train[columns].mean(),rtol=0,atol=1e-12)
        value=float(model.predict(pd.DataFrame([row[columns].to_dict()],columns=columns))[0])
        assert abs(value-receipt["prediction"])<1e-12
        assert value==signals.loc[int(row.origin_index),receipt["model"]]
    for _,row in events.iterrows():
        actual=[float(row.net_units_ratio>0),float(row.net_units_ratio<0),float(row.sma120>0),float(row.net_units_ratio>0 and row.sma120>0)]
        np.testing.assert_array_equal(signals.loc[int(row.origin_index),RULES].astype(float),actual)
    assert set(signals.index[signals.event_mask])==set(events.origin_index)
    for key in list(MODELS)+RULES:
        assert signals.loc[~signals.event_mask,key].isna().all()
    metrics=pd.read_csv(OUT/"metrics.csv")
    maximum=0.0
    assert len(metrics)==16
    for row in metrics.to_dict("records"):
        ledger=pd.read_parquet(OUT/"evaluation"/row["cost"]/(row["model"]+"_ledger.parquet"))
        decisions=pd.read_parquet(OUT/"evaluation"/row["cost"]/(row["model"]+"_decisions.parquet"))
        actual=summarize(ledger,config)
        for key,value in actual.items():
            if isinstance(value,(float,int)) and not isinstance(value,bool) and np.isfinite(value):
                error=abs(value-float(row[key]))
                assert error<1e-7,(row["model"],key,error)
                maximum=max(maximum,error)
        assert len(ledger)==1604 and str(ledger.date.iloc[0].date())=="2020-01-02" and str(ledger.date.iloc[-1].date())=="2026-08-14"
        assert ledger.accounting_error.abs().max()<1e-6 and ledger.shares.iloc[-1]==0
        np.testing.assert_allclose(ledger.equity,ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable,rtol=0,atol=1e-6)
        np.testing.assert_allclose(ledger.net_return,ledger.equity.to_numpy()/np.r_[200000.,ledger.equity.to_numpy()[:-1]]-1,rtol=0,atol=1e-12)
        allowed=decisions.origin_index.isin(events.origin_index)
        assert decisions.loc[~allowed,"requested_quantity"].eq(0).all()
        assert (decisions.execution_date>decisions.origin).all()
        assert decisions.loc[decisions.signal_state.eq("NO_VIEW_KEEP_EXISTING_SHARES"),"requested_quantity"].eq(0).all()
        if row["model"]=="BUY_HOLD":
            baseline=pd.read_parquet(PARENT/"evaluation"/row["cost"]/"BUY_HOLD_ledger.parquet")
            for name in ["equity","shares","cash","net_return","dividend_receivable","commission","slippage_cost"]:
                np.testing.assert_array_equal(ledger[name],baseline[name])
    uncertainty=read(OUT/"uncertainty.json")
    for cost in ["BASE","STRESS"]:
        samples=pd.read_parquet(OUT/f"{cost}_saved_bootstrap_statistics.parquet")
        assert len(samples)==2000
        for col in samples:
            np.testing.assert_allclose(interval(samples[col].tolist()),uncertainty[cost][col+"_95_interval"],rtol=0,atol=1e-12)
    return {"status":"PASS_SAVED_ORIGINAL_FACTS_CLOCKS_PREDICTIONS_ACCOUNTS_AND_INTERVALS",
            "original_quarterly_pdfs":56,"verified_official_clocks":55,"adjacent_transitions":55,
            "independently_rebuilt_price_features":3,"trained_models_checked_without_refit":84,
            "evaluation_accounts":16,"maximum_metric_recomputation_error":maximum,
            "buy_hold_daily_parity":"PASS","event_only_orders":"PASS","frozen_files":len(manifest["files"]),
            "new_model_fits":0,"new_account_simulations":0,"new_downloads":0,"new_bootstrap_samples":0,
            "security_audit_performed":False}


if __name__=="__main__":
    print(json.dumps(verify_saved(),ensure_ascii=False),flush=True)
