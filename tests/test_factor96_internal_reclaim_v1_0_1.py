"""二次试低、同成员比较、公司行为及下一开盘账户的关键反例。"""
import numpy as np
import pandas as pd

from research import factor96_internal_reclaim_v1_0_1 as study
from test_factor96_margin_repair_v1 import fixture,get_engine


def stock_inputs(n=70):
    dates=pd.bdate_range("2015-01-01",periods=n)
    names=[f"{i:06d}.SZ" for i in range(300)]
    q=pd.MultiIndex.from_product([dates,names],names=["date","symbol"]).to_frame(index=False)
    q["unadjusted_close"],q["daily_total_shareholder_return"]=100.,0.
    q["previous_unadjusted_close"],q["corporate_action_status"]=100.,"NONE_CONFIRMED"
    q["cash_distribution_per_pre_event_share"],q["subscription_cash_outflow_per_pre_event_share"]=0.,0.
    q["post_to_pre_share_ratio"]=1.
    q["return_is_usable"],q["constituent_return_state"]=True,"TRADED_VALID"
    raw=q[["date","symbol"]].copy()
    raw["open"],raw["high"],raw["low"],raw["close"]=100.,100.1,99.9,100.
    membership=q[["date","symbol"]].rename(columns={"date":"membership_date"})
    return pd.DataFrame({"date":dates}),q,raw,membership


def event_inputs(n=100):
    d,_,_=fixture(n)
    f=study.core.price_features(d)
    f["low_w"],f["high_w"],f["wealth"],f["atr20"],f["r"]=100.,101.,100.5,2.,0.
    values={30:(90.,91.,90.4),31:(91.,92.,91.5),32:(91.2,92.,91.5),33:(91.1,92.,91.5),
        34:(89.5,90.1,89.8),35:(89.7,91.,90.2),36:(90.1,91.,90.5)}
    for i,(low,high,close) in values.items():
        f.loc[i,["low_w","high_w","wealth"]]=[low,high,close]
    return d,f


def test_share_split_does_not_create_new_low():
    d,q,raw,m=stock_inputs()
    after=q.date.ge(d.date.iloc[30])
    q.loc[after,"unadjusted_close"]=50.
    q.loc[q.date.eq(d.date.iloc[30]),"post_to_pre_share_ratio"]=2.
    for c in ["open","high","low","close"]:
        raw.loc[after,c]/=2
    flags,valid,_,_,_=study.member_new_lows(d,q,raw,m)
    assert valid.iloc[30].all() and not flags.iloc[30].any()
    raw.loc[raw.date.eq(d.date.iloc[40]),"low"]=49.
    flags,_,_,_,_=study.member_new_lows(d,q,raw,m)
    assert flags.iloc[40].all()


def test_unknown_gap_is_not_zero_and_requires_fresh_full_window():
    d,q,raw,m=stock_inputs()
    q.loc[q.date.eq(d.date.iloc[25]),"return_is_usable"]=False
    q.loc[q.date.eq(d.date.iloc[25]),"daily_total_shareholder_return"]=np.nan
    flags,valid,_,coverage,_=study.member_new_lows(d,q,raw,m)
    assert not valid.iloc[25:46].to_numpy().any()
    assert valid.iloc[46].all() and not flags.iloc[25:46].to_numpy().any()
    assert not coverage.known.iloc[25:46].any()


def test_official_suspension_is_flat_without_fake_new_low():
    d,q,raw,m=stock_inputs()
    mask=q.date.eq(d.date.iloc[30])
    q.loc[mask,"constituent_return_state"]="OFFICIAL_SUSPENSION"
    raw.loc[mask,["open","high","low","close"]]=np.nan
    flags,valid,_,_,_=study.member_new_lows(d,q,raw,m)
    assert valid.iloc[30].all() and not flags.iloc[30].any()


def test_real_suspension_schema_has_no_observed_close_but_has_prior_mark():
    d,q,raw,m=stock_inputs()
    mask=q.date.eq(d.date.iloc[30])
    q.loc[mask,"constituent_return_state"]="OFFICIAL_SUSPENSION"
    q.loc[mask,"unadjusted_close"]=np.nan
    raw.loc[mask,["open","high","low","close"]]=np.nan
    flags,valid,_,coverage,_=study.member_new_lows(d,q,raw,m)
    assert valid.iloc[30].all() and coverage.known.iloc[30]
    assert not flags.iloc[30].any()
    q.loc[mask,"return_is_usable"]=False
    _,valid,_,coverage,_=study.member_new_lows(d,q,raw,m)
    assert not valid.iloc[30].any() and not coverage.known.iloc[30]


def test_two_day_confirmation_first_reclaim_and_future_invariance():
    d,state=event_inputs()
    f,events=study.price_reclaims(d,state)
    assert len(events)==1
    assert events.first_test_idx.tolist()==[30] and events.first_confirm_idx.tolist()==[32]
    assert events.second_test_idx.tolist()==[34] and events.reclaim_idx.tolist()==[35]
    assert f.index[f.price_signal].tolist()==[35]
    altered=state.copy()
    altered.loc[50:,"wealth"]*=.7
    g,_=study.price_reclaims(d,altered)
    pd.testing.assert_frame_equal(f.iloc[:50],g.iloc[:50])
    prefix,_=study.price_reclaims(d.iloc[:35],state.iloc[:35])
    pd.testing.assert_frame_equal(f.iloc[:35],prefix)


def test_non_reclaimed_test_stays_in_event_registry():
    d,state=event_inputs()
    state.loc[35:44,"wealth"]=89.9
    f,events=study.price_reclaims(d,state)
    assert events.outcome.tolist()==["EXPIRED_WITHOUT_RECLAIM"]
    assert not f.price_signal.any() and events.reclaim_idx.isna().all()


def test_comparison_uses_same_members_and_missing_stays_unknown():
    d,state=event_inputs()
    f,events=study.price_reclaims(d,state)
    names=[f"{i:06d}.SZ" for i in range(300)]
    flags=pd.DataFrame(False,index=d.date,columns=names)
    valid=pd.DataFrame(True,index=d.date,columns=names)
    member=valid.copy()
    flags.iloc[30,:100]=True
    flags.iloc[34,:90]=True
    cover=pd.DataFrame({"known":True},index=d.index)
    valid.iloc[34,:6]=False
    g,pairs=study.attach_internal(f,events,flags,valid,member,cover)
    assert pairs.common_usable_members.iloc[0]==294
    assert pairs.first_new_low_count.iloc[0]==94 and pairs.second_new_low_count.iloc[0]==84
    assert g.internal_divergence.iloc[35]
    valid.iloc[34,6]=False
    g,pairs=study.attach_internal(f,events,flags,valid,member,cover)
    assert not pairs.internal_known.iloc[0] and not g.internal_divergence.iloc[35]


def test_delay_stale_reclaim_and_account_control_separation():
    d,state=event_inputs()
    d[["open","close","high","low","previous_close"]]=4.
    f,events=study.price_reclaims(d,state)
    f["internal_known"],f["internal_divergence"]=False,False
    f["E02_change"],f["common_usable_members"],f["es95"]=np.nan,0,.04
    empty=pd.DataFrame(columns=["record_date","ex_date","payment_date","cash_dividend_per_share"])
    runs={key:study.simulate(d,f,empty,key,20000,"STRESS",d.date.iloc[25],d.date.iloc[-1],get_engine())[0] for key in study.POLICIES}
    assert runs["PRICE_ALL"].loc[runs["PRICE_ALL"].filled_quantity.gt(0),"date"].tolist()==[d.date.iloc[36]]
    assert runs["PRICE_ALL"].loc[runs["PRICE_ALL"].filled_quantity.lt(0),"date"].iloc[0]==d.date.iloc[41]
    assert not runs["FULL"].filled_quantity.ne(0).any()
    assert not runs["PRICE_COMMON"].filled_quantity.ne(0).any()
    f.loc[35,["internal_known","internal_divergence","E02_change","common_usable_members"]]=[True,True,-.1,300]
    delayed=study.lag_features(f,1)
    assert delayed.index[delayed.price_signal].tolist()==[36]
    f.loc[36,"wealth"]=89.
    assert not study.lag_features(f,1).price_signal.any()
    z,decisions=study.simulate(d,f,empty,"FULL",20000,"STRESS",d.date.iloc[25],d.date.iloc[-1],get_engine())
    assert (decisions.origin<decisions.date).all() and z.accounting_error.abs().max()<1e-7
