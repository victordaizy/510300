"""风险标签的全日分母、时间隔离、残差转换和预测损失检查。"""
import numpy as np
import pandas as pd

from research.monthly_downside_forecast_v1 import (
    CONFIG, read, samples_from_data, predict_months, fit_risk, loss, bootstrap_indices,
)


def synthetic():
    rng = np.random.default_rng(8907)
    dates = pd.bdate_range('2010-01-01', '2017-04-12')
    values = rng.normal(.0003, .01, len(dates))
    data = pd.DataFrame({'date': dates, 'total_simple': values})
    data['vol20'] = data.total_simple.rolling(20).std(ddof=1) * np.sqrt(242)
    return data


def test_monthly_downside_label_includes_up_days_in_denominator_and_excludes_incomplete_month():
    cfg = read(CONFIG)
    data = synthetic()
    data.loc[data.date.dt.to_period('M').eq(pd.Period('2015-02')), 'total_simple'] = .03
    selected = data.index[data.date.dt.to_period('M').eq(pd.Period('2015-02'))]
    data.loc[selected[:2], 'total_simple'] = [-.02, -.01]
    samples = samples_from_data(data, cfg)
    label = samples[samples.target_month.eq('2015-02')].iloc[0]
    expected = (.02 ** 2 + .01 ** 2) / len(selected) * 242
    np.testing.assert_allclose(label.label, expected, atol=1e-14, rtol=0)
    assert label.target_days == len(selected)
    assert samples.iloc[-1].target_month == '2017-04'
    assert pd.isna(samples.iloc[-1].label) and pd.isna(samples.iloc[-1].target_end)


def test_future_returns_and_features_cannot_change_earlier_predictions():
    cfg = read(CONFIG)
    cfg['evaluation_first_origin'] = '2012-01-01'
    data = synthetic()
    cutoff = pd.Timestamp('2015-07-15')
    first = predict_months(samples_from_data(data, cfg), cfg)[0]
    modified = data.copy()
    modified.loc[modified.date > cutoff, 'total_simple'] = -.4
    modified['vol20'] = modified.total_simple.rolling(20).std(ddof=1) * np.sqrt(242)
    second = predict_months(samples_from_data(modified, cfg), cfg)[0]
    columns = ['origin', 'training_count', 'training_last_maturity', 'coefficient', 'intercept',
               'smearing', 'LOG_RISK_RIDGE', 'PAST_MONTHLY_MEAN', 'RECENT_DOWNSIDE20']
    pd.testing.assert_frame_equal(first.loc[first.origin <= cutoff, columns].reset_index(drop=True),
                                  second.loc[second.origin <= cutoff, columns].reset_index(drop=True))


def test_mature_month_labels_and_training_eligibility():
    cfg = read(CONFIG)
    data = synthetic()
    samples = samples_from_data(data, cfg)
    forecasts, members = predict_months(samples, cfg)
    assert members.target_end.le(members.fit_origin).all()
    assert members.sample_origin.lt(members.fit_origin).all()
    complete = samples[samples.target_end.notna()]
    assert (complete.target_start.iloc[1:].to_numpy() > complete.target_end.iloc[:-1].to_numpy()).all()
    assert forecasts.training_count.between(24, 60).all()
    origin = forecasts.origin.iloc[5]
    changed = samples.copy()
    changed.loc[changed.target_end.gt(origin) | changed.target_end.isna(), 'label'] = 1234
    second = predict_months(changed, cfg)[0]
    np.testing.assert_allclose(forecasts.loc[forecasts.origin <= origin, 'LOG_RISK_RIDGE'],
                               second.loc[second.origin <= origin, 'LOG_RISK_RIDGE'])


def test_ridge_and_residual_smearing_match_separate_least_squares_calculation():
    cfg = read(CONFIG)
    x = np.r_[np.linspace(-6, -1, 38), -15, 10]
    y = np.exp(.3 * x - 3 + np.sin(x) / 4)
    fit = fit_risk(x, y, -2, cfg)
    z = np.clip((x - x.mean()) / x.std(ddof=1), -3, 3)
    design = np.column_stack([np.ones(len(x)), z])
    augmented = np.vstack([design, [0, np.sqrt(len(x))]])
    coefficient = np.linalg.lstsq(augmented, np.r_[np.log(y), 0], rcond=None)[0]
    smear = np.exp(np.log(y) - design @ coefficient).mean()
    current = np.clip((-2 - x.mean()) / x.std(ddof=1), -3, 3)
    expected = np.exp(coefficient @ [1, current]) * smear
    np.testing.assert_allclose(fit['prediction'], expected, atol=1e-13, rtol=0)


def test_loss_accepts_zero_risk_and_block_draws_preserve_contiguous_months():
    cfg = read(CONFIG)
    observed, forecast = np.array([0, .04, .02]), np.array([.01, .02, .04])
    np.testing.assert_allclose(loss(observed, forecast), np.log(forecast) + observed / forecast)
    matched = .025
    assert loss([matched], [matched])[0] < loss([matched], [.04])[0]
    draw = bootstrap_indices(29, cfg)
    assert draw.shape == (5000, 29)
    np.testing.assert_array_equal(draw, bootstrap_indices(29, cfg))
    for start in range(0, 24, 6):
        assert ((np.diff(draw[:, start:start + 6], axis=1)) % 29 == 1).all()
