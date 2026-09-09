"""验证季度信息时钟、成熟样本和未来记录不能影响当时拟合。"""
import numpy as np
import pandas as pd
from research.original_quarterly_flow_policy_v1 import flow_values,source_event_date,eligible_training,fixed_fit


def source(period,begin,end,sub,red,available):
    return {"period":period,"source_usable":True,"beginning_units":begin,"ending_units":end,
            "gross_subscription_units":sub,"gross_redemption_units":red,"split_delta_units":0,
            "feature_available_session":available}


def test_flow_uses_actual_prior_quarter_and_never_future_or_missing_zero():
    prior=source("2020Q1",100,110,30,20,"2020-04-23")
    current=source("2020Q2",110,99,10,21,"2020-07-22")
    value,state=flow_values(current,prior,pd.Timestamp("2020-07-22"))
    assert state.startswith("PASS_")
    np.testing.assert_allclose([value['net_units_ratio'],value['net_units_change']],[-.1,-.2])
    assert not flow_values(current,prior,pd.Timestamp("2020-07-21"))[1].startswith("PASS_")
    missing={**prior,"source_usable":False}
    value,state=flow_values(current,missing,pd.Timestamp("2020-07-22"))
    assert state=="NO_VIEW_PREVIOUS_SOURCE_UNUSABLE" and all(np.isnan(x) for x in value.values())
    older={**prior,"period":"2019Q4"}
    assert not flow_values(current,older,pd.Timestamp("2020-07-22"))[1].startswith("PASS_")


def test_publication_next_session_and_later_send_date():
    dates=pd.to_datetime(["2020-07-17","2020-07-20","2020-07-21","2020-07-22"])
    assert source_event_date({"publication_date":"2020-07-17","reported_send_date":"2020-07-20"},dates)==pd.Timestamp("2020-07-21")
    assert pd.isna(source_event_date({"publication_date":None,"reported_send_date":"2020-07-20"},dates))


def test_mature_quarter_selection_and_future_append_invariance():
    origin=pd.Timestamp("2020-07-22")
    records=pd.DataFrame({"origin":pd.to_datetime(["2020-01-21","2020-04-22","2020-07-22"]),
                          "label_exit_date":pd.to_datetime(["2020-04-21","2020-07-23","2020-10-21"]),
                          "is_quarter_training_origin":[True,True,True],"all_features_valid":[True,True,True],
                          "Y60":[.02,100,200],"x":[1.,2.,999.]})
    selected=eligible_training(records,origin)
    assert len(selected)==1 and selected.index.tolist()==[0]
    row=pd.Series({"x":1.5})
    a=fixed_fit(selected,row,["x"],10)[1]
    changed=records.copy()
    changed.loc[1:,["x","Y60"]]=1e12
    b=fixed_fit(eligible_training(changed,origin),row,["x"],10)[1]
    assert a==b==.02
