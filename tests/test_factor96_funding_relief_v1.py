"""T14的来源可得时钟、事件确认及进入退出分解验证。"""
import numpy as np
import pandas as pd

from research import factor96_funding_relief_v1 as study
from test_factor96_margin_repair_v1 import fixture,get_engine


def inputs(n=500):
    d,_,div=fixture(n)
    dr=pd.DataFrame({"date":d.date,"dr007":2+np.sin(np.arange(n)/4)*.8})
    p=pd.DataFrame({"known_at":[pd.Timestamp("2014-01-01",tz="Asia/Shanghai")],
        "effective_date":[pd.Timestamp("2014-01-01")],"rate":[2.]})
    return d,dr,p,div


def test_funding_future_changes_cannot_change_past():
    d,dr,p,_=inputs()
    before,_=study.funding_features(d,dr,p)
    changed=dr.copy()
    changed.loc[400:,"dr007"]=99.
    after,_=study.funding_features(d,changed,p)
    pd.testing.assert_frame_equal(before.iloc[:401],after.iloc[:401])
    extra=pd.DataFrame({"known_at":[d.date.iloc[450].tz_localize("Asia/Shanghai")],"effective_date":[d.date.iloc[450]],"rate":[1.]})
    after,_=study.funding_features(d,dr,pd.concat([p,extra],ignore_index=True))
    pd.testing.assert_frame_equal(before.iloc[:450],after.iloc[:450])


def test_extra_delay_and_weekend_publication():
    d,dr,p,_=inputs()
    a,_=study.funding_features(d,dr,p,1)
    b,_=study.funding_features(d,dr,p,2)
    np.testing.assert_allclose(b.K05.iloc[300],a.K05.iloc[299])
    assert a.fund_stat_date.iloc[300]==d.date.iloc[299]
    assert b.fund_stat_date.iloc[300]==d.date.iloc[298]
    saturday=pd.Timestamp("2015-02-07")
    extra=pd.DataFrame({"date":[saturday],"dr007":[3.]})
    c,_=study.funding_features(d,pd.concat([dr,extra],ignore_index=True),p,1)
    row=c[c.date.eq("2015-02-09")].iloc[0]
    assert row.fund_stat_date==saturday and row.dr007==3.


def test_effective_record_not_applied_before_its_public_date():
    d,dr,p,_=inputs()
    effective=d.date.iloc[300]
    late=pd.DataFrame({"effective_date":[effective],"known_at":[(effective+pd.Timedelta(days=2)).tz_localize("Asia/Shanghai")],"rate":[1.]})
    f,_=study.funding_features(d,dr,pd.concat([p,late],ignore_index=True),1)
    assert f.loc[f.date.eq(effective),"rate"].iloc[0]==2.
    assert f.loc[f.date.eq(d.date.iloc[301]),"rate"].iloc[0]==2.


def test_policy_ledger_normalizes_fixed_offset_and_named_timezone():
    raw=pd.DataFrame({"notice_date":[pd.Timestamp("2024-07-22")],
        "published_at":[pd.Timestamp("2024-07-22 09:20:00")],
        "seven_day_rate_percent":[1.7],"previous_rate_percent":[1.8],
        "source_url":["https://www.pbc.gov.cn/"],"raw_path":["raw.html"],"raw_sha256":["source"]})
    nodes=[{"node_id":"R02","economic_event_date":"2024-09-27",
        "source_available_upper":"2024-09-27T23:59:59+08:00","amount":1.5,
        "source_url":"https://app.www.gov.cn/","source_path":"raw/effective.html",
        "source_sha256":"effective","underlying_execution_date":"2024-09-27"}]
    result=study.policy_ledger(raw,nodes)
    assert str(result.known_at.dt.tz)=="Asia/Shanghai"
    assert result.known_at.iloc[-1]==pd.Timestamp("2024-09-27 23:59:59",tz="Asia/Shanghai")
    assert result.rate.tolist()==[1.7,1.5]


def test_shock_first_up_confirmation_and_cooldown():
    d,_,_,_=inputs(100)
    x=study.core.price_features(d)
    x["r"],x["rv20"]=-.001,.01
    x.loc[30,"r"]=-.04
    x.loc[31,"r"]=-.01
    x.loc[32,"r"]=.003
    x.loc[33,"r"]=.004
    x.loc[34,"r"]=-.04
    result=study.price_events(x)
    assert result.index[result.price_confirmation].tolist()==[32]
    assert result.shock_origin.iloc[32]==30


def test_factorial_entry_and_exit_separate():
    d,dr,p,div=inputs(100)
    d[["open","high","low","close","previous_close"]]=4.
    x=study.price_events(study.core.price_features(d))
    fund,_=study.funding_features(d,dr,p,1)
    x=pd.concat([x,fund.drop(columns="date")],axis=1)
    x["es95"],x["wealth"],x["shock_low"]=.04,1.,.5
    x["price_confirmation"],x["fund_relief"],x["fund_high"]=False,False,False
    x["fund_known"]=True
    x.loc[20,"price_confirmation"]=True
    x.loc[22,"fund_high"]=True
    runs={name:study.simulate(d,x,div,name,20000,"STRESS",d.date.iloc[20],d.date.iloc[-1],get_engine())[0] for name in study.POLICIES}
    assert runs["FULL"].filled_quantity.eq(0).all()
    assert runs["ENTRY_ONLY"].filled_quantity.eq(0).all()
    assert runs["PRICE_ONLY"].loc[runs["PRICE_ONLY"].filled_quantity.lt(0),"date"].iloc[0]==d.date.iloc[26]
    assert runs["EXIT_ONLY"].loc[runs["EXIT_ONLY"].filled_quantity.lt(0),"date"].iloc[0]==d.date.iloc[23]
    x.loc[20,"fund_relief"]=True
    full,decision=study.simulate(d,x,div,"FULL",20000,"STRESS",d.date.iloc[20],d.date.iloc[-1],get_engine())
    assert full.loc[full.filled_quantity.gt(0),"date"].tolist()==[d.date.iloc[21]]
    assert full.loc[full.filled_quantity.lt(0),"date"].iloc[0]==d.date.iloc[23]
    assert full.accounting_error.abs().max()<1e-7
    assert (decision.origin<decision.date).all()
