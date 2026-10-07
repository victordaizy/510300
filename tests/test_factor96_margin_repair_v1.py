"""验证首批研究的时钟、风险标签及现金账户约束。"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from research import factor96_margin_repair_v1 as study


def fixture(n=760):
    dates = pd.bdate_range("2015-01-05", periods=n)
    rng = np.random.default_rng(96)
    close = 4*np.exp(np.cumsum(rng.normal(.0002, .013, n)))
    op = close*np.exp(rng.normal(0, .003, n))
    d = pd.DataFrame({"date": dates, "open": op, "close": close,
        "high": np.maximum(op, close)*1.005, "low": np.minimum(op, close)*.995,
        "amount": 1e8*(1+np.arange(n)%5), "dividend": 0., "previous_close": np.r_[close[0], close[:-1]]})
    margin = pd.DataFrame({"date": dates, "market_rzye": 1e10*np.exp(np.cumsum(rng.normal(0,.004,n))),
                           "market_rzmre": 2e8})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    for k in ["record_date", "ex_date", "payment_date"]:
        dividends[k] = pd.to_datetime(dividends[k])
    return d, margin, dividends


def get_engine():
    path = Path(study.ROOT)/"research/intraday_overnight_increment_v1.py"
    spec = importlib.util.spec_from_file_location("factor96_test_engine", path)
    e = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = e
    spec.loader.exec_module(e)
    return e


def test_feature_future_invariance():
    d, _, _ = fixture()
    before = study.price_features(d)
    changed = d.copy()
    changed.loc[500:, ["open", "high", "low", "close", "amount"]] *= 11
    after = study.price_features(changed)
    pd.testing.assert_frame_equal(before.iloc[:500], after.iloc[:500])
    pd.testing.assert_frame_equal(before.iloc[:500].reset_index(drop=True), study.price_features(d.iloc[:500]))


def test_margin_delay_and_gap():
    d, m, _ = fixture()
    f = study.margin_features(d, m, 1)
    np.testing.assert_allclose(f.F01.iloc[300], m.market_rzye.iloc[299]/m.market_rzye.iloc[294]-1)
    assert f.margin_stat_date.iloc[300] == d.date.iloc[299]
    g = study.margin_features(d, m, 2)
    assert g.margin_stat_date.iloc[300] == d.date.iloc[298]
    missing = m.drop(index=298)
    h = study.margin_features(d, missing, 1)
    assert not h.margin_known.iloc[300]


def test_risk_labels_only_mature_and_two_years():
    d, _, div = fixture()
    x = study.price_features(d)
    risk, _, saved = study.mature_risk(d, x, div)
    changed = d.copy()
    changed.loc[601:, ["open", "high", "low", "close"]] *= 9
    other, _, _ = study.mature_risk(changed, study.price_features(changed), div)
    np.testing.assert_allclose(risk[:601], other[:601], equal_nan=True)
    for record in saved:
        t = record["decision_idx"]
        assert max(record["selected_indices"])+6 <= t
        assert d.date.iloc[min(record["selected_indices"])] >= d.date.iloc[t]-pd.DateOffset(years=2)


def test_t_plus_one_limit_and_cash():
    e, cost, cfg = get_engine(), study.COSTS["STRESS"], {"tick": .001, "lot":100, "limit_fraction":.1}
    a = e.Account(20000.)
    trade = e.execute_order(a, 1000, 4, 4, 0, 5, cost, cfg)
    assert trade["filled_quantity"] == 1000
    assert e.execute_order(a,-1000,4,4,0,5,cost,cfg)["filled_quantity"] == 0
    assert e.execute_order(a,-1000,3.6,4,0,6,cost,cfg)["filled_quantity"] == 0
    assert e.execute_order(a,-1000,3.8,4,0,7,cost,cfg)["filled_quantity"] == -1000
    a.assert_valid()


def test_simulation_has_no_same_day_signal_and_reconciles():
    d, m, div = fixture(100)
    x = pd.concat([study.price_features(d), study.margin_features(d, m, 1)], axis=1)
    x["es95"] = .04
    x["T03_price_signal"] = False
    x.loc[20, "T03_price_signal"] = True
    x["T03_stop"], x["T03_take"] = 0., 999.
    ledger, decisions = study.simulate(d,x,div,"T03_PRICE",20000,"STRESS",d.date.iloc[20],d.date.iloc[-1],get_engine())
    assert ledger.iloc[0].filled_quantity == 0
    assert ledger.iloc[1].filled_quantity > 0
    assert ledger.accounting_error.abs().max() < 1e-7
    assert (decisions.origin < decisions.date).all()
    assert (ledger.loc[ledger.filled_quantity.lt(0), "filled_quantity"].abs() <= ledger.loc[ledger.filled_quantity.lt(0), "sellable_before"]).all()
    assert ledger.filled_quantity.gt(0).sum() == 1


def test_dividend_record_ex_and_pay_are_distinct():
    d, m, _ = fixture(100)
    d[["open","close","high","low","previous_close"]] = 4.
    d.loc[22:, ["open","close","high","low"]] = 3.9
    d["previous_close"] = d.close.shift().fillna(4.)
    d.loc[22,"dividend"] = .1
    div = pd.DataFrame({"record_date":[d.date.iloc[21]],"ex_date":[d.date.iloc[22]],
        "payment_date":[d.date.iloc[30]],"cash_dividend_per_share":[.1]})
    x = pd.concat([study.price_features(d), study.margin_features(d,m,1)],axis=1)
    x["es95"] = .04
    x["T03_price_signal"] = False
    x.loc[20,"T03_price_signal"] = True
    x["T03_stop"],x["T03_take"] = 0.,999.
    ledger,_ = study.simulate(d,x,div,"T03_PRICE",20000,"BASE",d.date.iloc[20],d.date.iloc[-1],get_engine())
    shares = ledger.loc[ledger.date.eq(d.date.iloc[21]),"shares"].iloc[0]
    assert shares > 0
    assert ledger.loc[ledger.date.eq(d.date.iloc[22]),"dividend_receivable"].iloc[0] == shares*.1
    assert ledger.loc[ledger.date.eq(d.date.iloc[30]),"dividend_paid"].iloc[0] == shares*.1
    assert ledger.loc[ledger.date.eq(d.date.iloc[30]),"shares"].iloc[0] == 0
    assert ledger.accounting_error.abs().max() < 1e-7


def test_cash_sharpe_is_undefined():
    d,m,div=fixture(100)
    x=pd.concat([study.price_features(d),study.margin_features(d,m,1)],axis=1)
    x["es95"]=.04
    ledger,_=study.simulate(d,x,div,"CASH",20000,"STRESS",d.date.iloc[20],d.date.iloc[-1],get_engine())
    assert study.metrics(ledger,20000)["net_sharpe"] is None
    assert ledger.equity.eq(20000).all()
