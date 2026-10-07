"""确认延迟、当前尾点替换、未知状态和未来后缀隔离。"""
import numpy as np
import pandas as pd
from research.confirmed_structure_inputs_v1 import structure_states, account_signals


def market():
    prices = [11.,10.,9.,10.,11.,12.,11.,10.,9.5,10.,11.,13.,12.,11.,10.5,11.,12.,14.,13.,12.,11.5,12.,13.]
    return pd.DataFrame({"date":pd.date_range("2020-01-02",periods=len(prices),freq="B"),
                         "close":prices,"dividend":0.,"atr20":.1})


def test_four_pivots_only_create_structure_after_two_day_confirmation():
    d=market()
    states, pivots=structure_states(d)
    assert not states.confirmed_low_today.iloc[:4].any()
    assert states.confirmed_low_today.iloc[4]
    assert not states.structure_onset.iloc[:13].any()
    assert states.structure_onset.iloc[13]
    r=states.iloc[13]
    assert [r.L1_index,r.H1_index,r.L2_index,r.H2_index]==[2,5,8,11]
    assert [r.L1_confirmation_index,r.H1_confirmation_index,r.L2_confirmation_index,r.H2_confirmation_index]==[4,7,10,13]
    assert r.latest_confirmed_low_price==9.5
    assert (pivots.confirmation_index-pivots.center_index).eq(2).all()


def test_same_kind_more_extreme_only_replaces_current_tail():
    d=market().iloc[:11].copy()
    d["close"]=[11.,10.,9.,10.,11.,11.,11.,10.,8.,10.,11.]
    full,pivots=structure_states(d)
    assert full.confirmed_sequence_length.iloc[4]==1
    assert full.confirmed_sequence_length.iloc[10]==1
    assert pivots.iloc[-1].action=="REPLACED_CURRENT_TAIL_MORE_EXTREME"
    assert pivots.iloc[-1].replaced_tail_center_index==2
    prefix,_=structure_states(d.iloc[:10])
    pd.testing.assert_frame_equal(prefix,full.iloc[:10].reset_index(drop=True),check_exact=True)
    weak=d.copy()
    weak.loc[8,"close"]=9.5
    _,weak_pivots=structure_states(weak)
    assert weak_pivots.iloc[-1].action=="IGNORED_SAME_KIND_LESS_EXTREME"


def test_unknown_does_not_force_exit_or_fabricate_recovery_birth():
    d=market()
    d.loc[14,"close"]=np.nan
    states,_=structure_states(d)
    signal=account_signals(d,states,"CONFIRMED_RISING_STRUCTURE")
    assert states.structure_status.iloc[14]=="NO_VIEW"
    assert not signal.rule_exit.iloc[14]
    assert states.structure_active.iloc[15]
    assert not states.structure_onset.iloc[15]
    unknown_dividend=market()
    unknown_dividend.loc[14,"dividend"]=np.nan
    a,_=structure_states(unknown_dividend)
    assert a.structure_status.iloc[14:].eq("NO_VIEW").all()


def test_future_suffix_and_result_columns_do_not_rewrite_past():
    d=market()
    d.loc[7,"dividend"]=.05
    full,_=structure_states(d)
    d["future_cycle_return"]=np.arange(len(d))*100.
    d["retrospective_low"]=np.arange(len(d))[::-1]
    pd.testing.assert_frame_equal(structure_states(d)[0],full,check_exact=True)
    for n in range(1,len(d)+1):
        short,_=structure_states(d.iloc[:n].reset_index(drop=True))
        pd.testing.assert_frame_equal(short,full.iloc[:n].reset_index(drop=True),check_exact=True)
        other=d.copy()
        other.loc[n:,"close"]=np.arange(len(d)-n)+100.
        other.loc[n:,"dividend"]=5.
        altered,_=structure_states(other)
        pd.testing.assert_frame_equal(altered.iloc[:n].reset_index(drop=True),short,check_exact=True)
