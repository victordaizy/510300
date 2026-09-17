"""公告可用时钟、重复事件和当日成员身份的实质边界。"""
import numpy as np
import pandas as pd
from research.unlock_announcement_increment_v1 import announcement_density


def members(dates, special="600000.SH"):
    codes=[special]+[f"{100000+i:06d}.SZ" for i in range(299)]
    return pd.DataFrame([(day,code) for day in dates for code in codes],columns=["membership_date","symbol"])


def test_day_delay_and_weekend_do_not_allow_same_day_information():
    dates=pd.bdate_range("2024-01-04",periods=5)
    events=pd.DataFrame({"secCode":["600000"],"announcement_date":["2024-01-05"]})
    density,_,lineage=announcement_density(dates,events,members(dates),1,2)
    assert np.array_equal(density>0,[False,False,True,False,False])
    assert lineage.available_at_assumed.iloc[0]==pd.Timestamp("2024-01-07")


def test_repeated_disclosures_count_one_issuer_in_window():
    dates=pd.bdate_range("2024-01-02",periods=6)
    events=pd.DataFrame({"secCode":["600000"]*3,"announcement_date":["2024-01-02","2024-01-02","2024-01-03"]})
    density,_,_=announcement_density(dates,events,members(dates),2,2)
    assert density.max()==1/300


def test_future_membership_cannot_create_past_activity():
    dates=pd.bdate_range("2024-01-02",periods=5)
    m=members(dates)
    m.loc[m.membership_date.lt(dates[3])&m.symbol.eq("600000.SH"),"symbol"]="600001.SH"
    events=pd.DataFrame({"secCode":["600000"],"announcement_date":["2024-01-02"]})
    density,_,_=announcement_density(dates,events,m,3,2)
    assert density[2]==0 and density[3]==1/300


def test_later_announcements_cannot_change_earlier_features():
    dates=pd.bdate_range("2024-01-02",periods=8)
    a=pd.DataFrame({"secCode":["600000"],"announcement_date":["2024-01-02"]})
    b=pd.concat([a,pd.DataFrame({"secCode":["600000"],"announcement_date":["2024-01-09"]})],ignore_index=True)
    left,_,_=announcement_density(dates,a,members(dates),2,2)
    right,_,_=announcement_density(dates,b,members(dates),2,2)
    assert np.array_equal(left[:7],right[:7])
