"""验证首次边界时钟、完整周信息、成熟训练和下一开盘资金约束。"""
import numpy as np
import pandas as pd

from research import point_first_passage_inputs_v1 as model
from research.point_first_passage_account_v1 import account
from research.point_account_nr7_inputs_v1 import PARENT_A


def empty_dividends():
    return pd.DataFrame({"record_date": pd.Series(dtype="datetime64[ns]"), "ex_date": pd.Series(dtype="datetime64[ns]"),
                         "payment_date": pd.Series(dtype="datetime64[ns]"), "cash_dividend_per_share": pd.Series(dtype=float)})


def label_data(n=25):
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"date": dates, "open": 10., "high": 20., "low": 5., "close": 10.,
                         "ac": 10., "cash_shift": 0., "dividend": 0., "atr14": .1})


def test_first_passage_uses_close_and_matures_after_next_open():
    data = label_data()
    data.loc[2, "ac"] = 10.21
    result = model.labels(data, empty_dividends()).iloc[0]
    assert result.event_class == "PROFIT"
    assert result.entry_idx == 1 and result.barrier_close_idx == 2 and result.exit_idx == 3 and result.mature_idx == 3
    # 全程盘中高低早已越界；没有日收盘确认时只会在20个收盘到期。
    flat = model.labels(label_data(), empty_dividends()).iloc[0]
    assert flat.event_class == "TIMEOUT" and flat.barrier_close_idx == 20 and flat.exit_idx == 21


def test_terminal_is_unmatured_and_owned_dividend_delays_maturity():
    data = label_data(4)
    data.loc[3, "ac"] = 10.21
    assert model.labels(data, empty_dividends()).iloc[0].status == "UNMATURED"
    data = label_data(6)
    data.loc[1, "ac"] = 10.21
    div = pd.DataFrame({"record_date": [data.date.iloc[1]], "ex_date": [data.date.iloc[4]],
                        "payment_date": [data.date.iloc[5]], "cash_dividend_per_share": [.1]})
    result = model.labels(data, div).iloc[0]
    assert result.exit_idx == 2 and result.mature_idx == 4
    assert result.reference_net_return > 0


def test_daily_weekly_features_are_prefix_invariant():
    dates = pd.bdate_range("2012-01-02", periods=400).astype("datetime64[ms]")
    prices = 10+np.arange(400)*.002+np.sin(np.arange(400)/7)*.05
    data = pd.DataFrame({"date": dates, "open": prices-.003, "close": prices,
                         "high": prices+.04, "low": prices-.04, "volume": 1000+np.arange(400)%17})
    full = model.features(data, empty_dividends())
    cut = model.features(data.iloc[:333], empty_dividends())
    pd.testing.assert_frame_equal(full.iloc[:333][model.FEATURES].reset_index(drop=True), cut[model.FEATURES], check_exact=True)
    assert full.loc[full.weekly_last_date.notna(), "weekly_last_date"].lt(full.loc[full.weekly_last_date.notna(), "date"]).all()


def synthetic_training():
    rng = np.random.default_rng(123)
    n = 330
    data = pd.DataFrame({"date": pd.bdate_range("2013-01-01", periods=n), "first_passage_feature_known": True})
    for name in model.FEATURES:
        data[name] = rng.normal(size=n)
    outcomes = pd.DataFrame({"origin_index": np.arange(n), "status": "MATURE_REFERENCE",
                             "entry_idx": np.arange(n)+1, "exit_idx": np.arange(n)+3,
                             "mature_idx": np.arange(n)+3, "event_class": np.array(model.CLASSES)[np.arange(n)%3],
                             "reference_net_return": np.where(np.arange(n)%3 == 1, .025, -.01)})
    return data, outcomes


def test_training_excludes_unmatured_and_corrects_overlap_weights():
    data, outcomes = synthetic_training()
    pool = model.training_pool(data, outcomes, 300)
    assert pool.mature_idx.max() == 300
    assert pool.origin_index.max() == 297
    assert pool.uniqueness_weight.between(0, 1).all()
    assert abs(pool.fit_weight.mean()-1) < 1e-14


def test_future_labels_cannot_change_fitted_model():
    data, outcomes = synthetic_training()
    first = model.fit_at(data, outcomes, 300)
    assert first["status"] == "FIT_COMPLETE"
    changed = outcomes.copy()
    changed.loc[changed.mature_idx.gt(300), ["reference_net_return", "event_class"]] = [.99, "PROFIT"]
    second = model.fit_at(data, changed, 300)
    assert first == second
    values = data.iloc[300][model.FEATURES].to_numpy(float)
    p = model.probability(first["model"], values, "TECHNICAL_LOGIT")
    assert (p > 0).all() and abs(p.sum()-1) < 1e-14


def test_first_open_quantity_ignores_entry_day_close_and_exit_obeys_t_plus_one():
    data = label_data(5)
    data["high"], data["low"] = 10.05, 9.95
    parents = pd.DataFrame({"origin": data.date, PARENT_A: 0.})
    risks = pd.DataFrame({"idx": np.arange(len(data)), "es95": .01})
    signals = pd.DataFrame({"date": data.date, "entry_event": [True, False, False, False, False],
                            "event_id": ["TEST_0", None, None, None, None], "atr": .1,
                            "stop_index": np.nan, "target_index": np.nan})
    data.loc[1, ["ac", "close"]] = [10.25, 10.25]
    a = account(data, empty_dividends(), parents, risks, signals, "STRESS", str(data.date.iloc[1].date()), "PASSAGE_ONLY")
    changed = data.copy()
    changed.loc[1, ["ac", "close"]] = [9.8, 9.8]
    b = account(changed, empty_dividends(), parents, risks, signals, "STRESS", str(data.date.iloc[1].date()), "PASSAGE_ONLY")
    buy_a, buy_b = a["orders"].query("side=='BUY'").iloc[0], b["orders"].query("side=='BUY'").iloc[0]
    assert buy_a.quantity == buy_b.quantity and buy_a.fill_price == buy_b.fill_price
    trade = a["trades"].iloc[0]
    assert trade.entry_origin < trade.entry_date < trade.exit_date
    assert trade.exit_reason == "FIRST_PASSAGE_PROFIT_CLOSE"
    np.testing.assert_allclose(a["daily"].accounting_error, 0., atol=1e-6)
