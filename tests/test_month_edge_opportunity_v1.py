"""必要边界：原预算优先、独立日历进入、未知、时钟和未来隔离。"""
import numpy as np
import pandas as pd

from research.month_edge_opportunity_inputs_v1 import CANDIDATES,FULL,MODELS,PARENT,PRIMARY,calendar_frames,calendar_targets


def test_primary_preserves_positive_parent_and_only_fills_cash():
    result=calendar_targets([.3,.7,0.,0.],[1.,0.,1.,0.])
    np.testing.assert_array_equal(result[PRIMARY],[.3,.7,1.,0.])
    np.testing.assert_array_equal(result[FULL],[1.,.7,1.,0.])


def test_unknown_and_independent_calendar_entry():
    result=calendar_targets([np.nan,np.nan,.4,0.,1.],[1.,0.,np.nan,np.nan,np.nan])
    np.testing.assert_allclose(result[PRIMARY],[np.nan,np.nan,.4,np.nan,1.],equal_nan=True)
    np.testing.assert_allclose(result[FULL],[1.,np.nan,np.nan,np.nan,1.],equal_nan=True)


def test_calendar_exit_returns_to_parent_instead_of_forced_cash():
    result=calendar_targets([.3,.3,0.,np.nan],[1.,0.,0.,0.])
    np.testing.assert_allclose(result[FULL],[1.,.3,0.,np.nan],equal_nan=True)


def test_two_clocks_and_terminal_boundary(monkeypatch):
    dates=pd.DatetimeIndex(['2026-03-31','2026-04-01','2026-04-02','2026-04-03','2026-04-07'])
    data=pd.DataFrame({'date':dates})
    execution=pd.Series(dates).shift(-1)
    saved=pd.DataFrame({'market_date':dates,'execution_date':execution,'decision_time':execution+pd.Timedelta(hours=9),
        'calendar_state':[1.,1.,1.,0.,np.nan]})
    monkeypatch.setattr(pd,'read_parquet',lambda path:saved.copy())
    parent=pd.DataFrame({'source_cost':'BASE','source_model':PARENT,'origin_index':np.arange(4),
        'origin':dates[:-1],'execution_date':dates[1:],'reference_weight':[.3,0.,.6,0.]})
    cfg={'candidate_models':list(CANDIDATES),'decision_clock':'09:00:00','source_signal_clock':'15:05:00',
        'month_first_sessions':3,'month_end_natural_days':5,'weight_band':.1,'costs':{'BASE':{}},'evaluation_start':'2026-04-01'}
    frames,_=calendar_frames(data,{'BASE':{PARENT:parent}},cfg,'2026-04-01')
    f=frames['BASE']
    np.testing.assert_array_equal(f.parent_signal_time.iloc[:-1],dates[:-1]+pd.Timedelta(hours=15,minutes=5))
    np.testing.assert_array_equal(f.decision_time.iloc[:-1],dates[1:]+pd.Timedelta(hours=9))
    assert pd.isna(f.decision_time.iloc[-1]) and np.isnan(f[FULL+'_target'].iloc[-1])


def test_future_does_not_change_previous_targets():
    source=np.array([.2,0.,np.nan,.5])
    calendar=np.array([1.,0.,1.,1.])
    old=calendar_targets(source[:3],calendar[:3])
    source[-1]=1.
    calendar[-1]=0.
    new=calendar_targets(source,calendar)
    for model in CANDIDATES:
        np.testing.assert_allclose(new[model][:3],old[model],equal_nan=True)
