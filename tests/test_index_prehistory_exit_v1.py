"""只检查本轮新增的价格训练样本、成熟日期和合并训练计算。"""
import numpy as np
import pandas as pd
import pytest
from research.index_prehistory_exit_inputs_v1 import price_features, natural_price_episodes, pool_samples, choose_mature_samples
from research.learned_cycle_exit_v1 import FEATURES, fit_one, predict

SPEC = {"cooldown": 2, "modes": {"1": {"loss": .06, "trail": .08, "days": 60}}}


def episode_data():
    frame = pd.DataFrame({"date": pd.bdate_range("2010-01-01", periods=11), "open": [100., 101., 102., 103., 105., 106., 104., 105., 104., 105., 106.],
        "close": [100., 102., 103., 104., 106., 107., 104., 105., 104., 105., 106.],
        "entry_condition": [True] + [False] * 5 + [True] * 5, "price_exit_condition": [False] * 4 + [True] + [False] * 6})
    for key in ["mom5", "mom20", "sma120", "vol20"]:
        frame[key] = .01
    return frame


def test_price_features_use_only_previous_prices_and_252_return_warmup():
    n = 320
    close = 100 * np.exp(np.cumsum(.001 + .01 * np.sin(np.arange(n))))
    prices = pd.DataFrame({"date": pd.bdate_range("2005-04-08", periods=n), "open": close * np.exp(.003 * np.cos(np.arange(n))), "close": close})
    prices["high"] = prices[["open", "close"]].max(axis=1) * 1.01
    prices["low"] = prices[["open", "close"]].min(axis=1) * .99
    full, prefix = price_features(prices), price_features(prices.iloc[:280])
    pd.testing.assert_frame_equal(full.iloc[:280], prefix)
    assert not full.feature_valid.iloc[:252].any() and full.feature_valid.iloc[252:].all()
    np.testing.assert_allclose(full.intraday_log.iloc[1:] + full.overnight_log.iloc[1:], np.log(close[1:] / close[:-1]), atol=1e-14)
    difference = full.intraday_log - full.overnight_log
    expected = difference.iloc[201:261].sum() / difference.iloc[201:261].std(ddof=1) / np.sqrt(60)
    assert abs(full.d60.iloc[260] - expected) < 1e-12


def test_natural_labels_use_later_open_and_censor_unfinished_cycle():
    data = episode_data()
    episodes, samples = natural_price_episodes(data, SPEC)
    assert episodes.iloc[0].entry_index == 1 and episodes.iloc[0].exit_index == 5
    assert episodes.iloc[1].entry_index == 8 and episodes.iloc[1].status == "NO_VIEW_UNCLOSED_AT_SOURCE_CUTOFF"
    assert samples.origin_index.to_list() == [1, 2, 3]
    np.testing.assert_allclose(samples.target, 106 / data.open.iloc[[2, 3, 4]].to_numpy() - 1)
    assert samples.mature_date.eq(data.date.iloc[5]).all()
    assert samples.cycle_id.eq(1).all()
    assert samples.iloc[0].log_holding_days == np.log(2)


def test_unfinished_future_cycles_do_not_create_early_training_rows():
    frame = episode_data()
    _, before_exit = natural_price_episodes(frame.iloc[:5], SPEC)
    assert before_exit.empty
    _, complete = natural_price_episodes(frame, SPEC)
    assert not complete.empty


def sample_row(source_date, mature_date, cycle, origin_index, exit_index):
    origin = pd.Timestamp(source_date)
    return {"cycle_id": cycle, "origin": origin, "early_exit_date": origin + pd.Timedelta(days=1), "mature_date": pd.Timestamp(mature_date),
        "origin_index": origin_index, "exit_index": exit_index, "target": .01, **{key: .1 for key in FEATURES}}


def test_pool_uses_source_unique_cycles_and_calendar_dates():
    dates = pd.bdate_range("2012-05-28", periods=10)
    index = pd.DataFrame([sample_row("2010-01-04", "2010-01-08", 1, 1, 5)])
    etf = pd.DataFrame([sample_row("2012-05-28", "2012-06-01", 1, 0, 4)])
    pooled = pool_samples(index, etf, dates)
    assert pooled.cycle_id.nunique() == 2
    assert pooled.exit_index.to_list() == [-1, 4]
    early, ids = choose_mature_samples(pooled, "2012-05-31", 20)
    assert len(ids) == 1 and early.source.eq("INDEX_PRICE").all()
    full, ids = choose_mature_samples(pooled, "2012-06-01", 1)
    assert len(ids) == 1 and full.source.eq("ETF_REFERENCE").all()
    with pytest.raises(ValueError, match="上市前"):
        pool_samples(etf, etf, dates)


def test_maturity_order_and_cycle_equal_weights_ignore_origin_local_indices():
    rows = []
    for cycle, maturity, count in [("Z", "2010-01-20", 3), ("A", "2011-01-20", 1), ("FUTURE", "2013-01-20", 2)]:
        for i in range(count):
            rows.append({"cycle_id": cycle, "mature_date": pd.Timestamp(maturity), "origin": pd.Timestamp(maturity) - pd.Timedelta(days=10-i), "row_id": f"{cycle}_{i}"})
    data = pd.DataFrame(rows)
    selected, ids = choose_mature_samples(data, "2012-01-01", 2)
    assert ids == ["Z", "A"]
    np.testing.assert_allclose(selected.groupby("cycle_id").sample_weight.sum(), 1.)
    assert choose_mature_samples(data, "2012-01-01", 1)[1] == ["A"]


def test_pooled_ridge_matches_weighted_normal_equations():
    rng = np.random.default_rng(93)
    x = rng.normal(size=(30, 8))
    x[:, 3] = 1
    weights = np.r_[np.repeat(1/10, 10), np.repeat(1/20, 20)]
    y = rng.normal(scale=.02, size=30)
    rows = pd.DataFrame(x, columns=FEATURES)
    rows["target"], rows["sample_weight"] = y, weights
    cfg = {"feature_clip": 5., "ridge_alpha": 1.}
    model = fit_one(rows, "RIDGE", cfg)
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale[scale <= 1e-12] = 1
    design = np.c_[np.ones(len(x)), np.clip((x-mean)/scale, -5, 5)]
    penalty = np.diag([0.] + [1.] * 8)
    beta = np.linalg.solve(design.T @ (weights[:, None]*design) + penalty, design.T @ (weights*y))
    np.testing.assert_allclose([model["intercept"], *model["coefficients"]], beta, atol=1e-12, rtol=0)
    np.testing.assert_allclose([predict(model, row) for row in x], design @ beta, atol=1e-12, rtol=0)
