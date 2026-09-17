"""检验单因子顺序学习的未来隔离、标签成熟及成交目标边界。"""
from copy import deepcopy
import numpy as np
import pandas as pd

from research.monthly_single_factor_walkforward_v1 import read, CONFIG, monthly_samples, make_signals, single_ridge, signal_target


def synthetic():
    rng = np.random.default_rng(7419)
    dates = pd.bdate_range('2010-01-01', periods=1800)
    close = 100 * np.exp(np.cumsum(rng.normal(.0002, .012, len(dates))))
    data = pd.DataFrame({'date': dates, 'close': close, 'open': close * np.exp(rng.normal(0, .002, len(dates)))})
    returns = data.close.pct_change()
    data['mom20'] = np.log(data.close / data.close.shift(20))
    data['z20'] = (data.close - data.close.rolling(20).mean()) / data.close.rolling(20).std(ddof=1)
    data['vol20'] = returns.rolling(20).std(ddof=1) * np.sqrt(242)
    data['logvol20'] = np.log(data.vol20)
    dividends = pd.DataFrame({name: pd.Series(dtype='datetime64[ns]') for name in ['record_date', 'ex_date', 'payment_date']})
    dividends['cash_dividend_per_share'] = pd.Series(dtype=float)
    return data, dividends


def test_future_prices_and_future_features_do_not_change_previous_forecasts():
    cfg = read(CONFIG)
    data, dividends = synthetic()
    cutoff = pd.Timestamp('2014-07-15')
    first = make_signals(data, monthly_samples(data, dividends), cfg, '2012-01-01')[0]
    changed = data.copy()
    changed.loc[changed.date > cutoff, ['open', 'close']] *= 7
    changed.loc[changed.date > cutoff, ['mom20', 'z20', 'logvol20']] += 100
    second = make_signals(changed, monthly_samples(changed, dividends), cfg, '2012-01-01')[0]
    columns = ['model', 'origin', 'training_count', 'training_last_maturity', 'training_mean', 'training_scale',
               'coefficient', 'intercept', 'prediction', 'target']
    pd.testing.assert_frame_equal(first.loc[first.origin <= cutoff, columns].reset_index(drop=True),
                                  second.loc[second.origin <= cutoff, columns].reset_index(drop=True))


def test_dividend_record_ownership_and_late_label_maturity():
    data, _ = synthetic()
    data = data[data.date.between('2016-01-01', '2016-06-30')].reset_index(drop=True)
    dividends = pd.DataFrame({'record_date': pd.to_datetime(['2016-01-29', '2016-02-15', '2016-02-29', '2016-03-01']),
                              'ex_date': pd.to_datetime(['2016-02-01', '2016-02-16', '2016-03-02', '2016-03-02']),
                              'cash_dividend_per_share': [1.0, .5, .7, .9]})
    row = monthly_samples(data, dividends).iloc[0]
    assert row.entry_date == pd.Timestamp('2016-02-01')
    assert row.exit_date == pd.Timestamp('2016-03-01')
    assert row.mature_date == pd.Timestamp('2016-03-02')
    assert abs(row.earned_dividend_per_share - 1.2) < 1e-12
    assert abs(row.label - ((row.exit_price + 1.2) / row.entry_price - 1)) < 1e-12


def test_training_members_are_mature_and_months_do_not_overlap():
    data, dividends = synthetic()
    samples = monthly_samples(data, dividends)
    forecasts, members, _, _ = make_signals(data, samples, read(CONFIG), '2012-01-01')
    assert len(members) > 0 and members.mature_date.le(members.fit_origin).all()
    assert members.sample_origin.lt(members.fit_origin).all()
    assert forecasts.training_count.le(60).all()
    assert forecasts.loc[forecasts.fit_status.eq('FIT_READY'), 'training_count'].ge(24).all()
    assert samples.exit_date.iloc[:-1].reset_index(drop=True).equals(samples.entry_date.iloc[1:].reset_index(drop=True))
    assert pd.isna(samples.label.iloc[-1])
    changed = samples.copy()
    current_origin = forecasts.origin.iloc[20]
    changed.loc[changed.mature_date.gt(current_origin) | changed.mature_date.isna(), 'label'] = 999
    second = make_signals(data, changed, read(CONFIG), '2012-01-01')[0]
    np.testing.assert_allclose(forecasts.loc[forecasts.origin <= current_origin, 'prediction'],
                               second.loc[second.origin <= current_origin, 'prediction'], equal_nan=True)


def test_scalar_ridge_matches_independent_augmented_least_squares():
    cfg = read(CONFIG)
    rng = np.random.default_rng(531)
    x = np.r_[rng.normal(size=28), 20, -18]
    y = .03 + .005 * x + rng.normal(0, .02, len(x))
    fit = single_ridge(x, y, .7, cfg)
    z = np.clip((x - x.mean()) / x.std(ddof=1), -3, 3)
    design = np.column_stack([np.ones(len(x)), z])
    design = np.vstack([design, [0, np.sqrt(len(x))]])
    coefficients = np.linalg.lstsq(design, np.r_[y, 0], rcond=None)[0]
    np.testing.assert_allclose([fit['intercept'], fit['coefficient']], coefficients, atol=1e-12, rtol=0)


def test_strict_buffer_risk_cap_and_unknown_targets():
    cfg = read(CONFIG)
    assert signal_target(.004, .2, cfg) == 0
    assert signal_target(.004001, .2, cfg) == .5
    assert signal_target(.01, .05, cfg) == 1
    assert signal_target(-.1, .2, cfg) == 0
    assert np.isnan(signal_target(np.nan, .2, cfg))
    assert np.isnan(signal_target(.02, 0, cfg))
