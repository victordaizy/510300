"""固定单因子月度下行风险预测：只在预测门通过后开放账户阶段。"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import digest, now, write_json
from research.monthly_single_factor_walkforward_v1 import read, csv, single_ridge


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/510300_monthly_downside_forecast_v1.json'
OUT = ROOT / 'reports/research/510300_monthly_downside_forecast_v1'
INDEX = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
CANDIDATE = 'LOG_RISK_RIDGE'
BASELINES = ['PAST_MONTHLY_MEAN', 'RECENT_DOWNSIDE20']


def inputs(cfg):
    data = pd.read_parquet(ROOT / cfg['features'])
    data['date'] = pd.to_datetime(data.date)
    assert data.date.is_monotonic_increasing and not data.date.duplicated().any()
    assert data.date.iloc[-1] == pd.Timestamp(cfg['data_cutoff'])
    return data


def samples_from_data(data, cfg):
    """每个标签恰好覆盖下个已完成自然月，保留最后一个未成熟预测对象。"""
    floor = cfg['positive_risk_floor']
    month = data.date.dt.to_period('M')
    origins = np.flatnonzero(month.iloc[:-1].to_numpy() != month.iloc[1:].to_numpy())
    downside_daily = data.total_simple.clip(upper=0).pow(2)
    recent_downside = downside_daily.rolling(cfg['factor_window_days']).mean() * cfg['annual_days']
    factor = np.log(data.vol20.pow(2).clip(lower=floor))
    rows = []
    for j, origin in enumerate(origins):
        first = int(origin + 1)
        row = {'origin': data.date.iloc[origin], 'origin_index': int(origin),
               'target_start': data.date.iloc[first], 'target_end': pd.NaT,
               'target_month': str(month.iloc[first]), 'target_days': 0,
               'feature': float(factor.iloc[origin]),
               'recent_downside': float(recent_downside.iloc[origin]), 'label': np.nan}
        if j + 1 < len(origins):
            last = int(origins[j + 1])
            values = downside_daily.iloc[first:last + 1].to_numpy(float)
            assert month.iloc[first] == month.iloc[last]
            assert np.isfinite(values).all()
            row.update(target_end=data.date.iloc[last], target_days=len(values),
                       label=float(values.mean() * cfg['annual_days']))
        rows.append(row)
    samples = pd.DataFrame(rows)
    for name in ['origin', 'target_start', 'target_end']:
        samples[name] = pd.to_datetime(samples[name])
    return samples


def fit_risk(x, y, current, cfg):
    log_labels = np.log(np.maximum(y, cfg['positive_risk_floor']))
    fit = single_ridge(x, log_labels, current, cfg)
    standardized = np.clip((np.asarray(x) - fit['training_mean']) / fit['training_scale'],
                           -cfg['standardized_clip'], cfg['standardized_clip'])
    residual = log_labels - (fit['intercept'] + fit['coefficient'] * standardized)
    smearing = float(np.exp(residual).mean())
    prediction = float(np.exp(fit['prediction']) * smearing)
    assert np.isfinite(prediction) and prediction > 0 and np.isfinite(smearing)
    return {**fit, 'log_prediction': fit['prediction'], 'smearing': smearing,
            'prediction': max(prediction, cfg['positive_risk_floor'])}


def predict_months(samples, cfg):
    forecasts, members = [], []
    valid = np.isfinite(samples[['feature', 'label']].to_numpy(float)).all(axis=1)
    for row in samples.itertuples():
        if row.origin < pd.Timestamp(cfg['evaluation_first_origin']):
            continue
        train = samples[valid & samples.target_end.le(row.origin)].tail(cfg['training_months'])
        supported = len(train) >= cfg['minimum_training_months'] and np.isfinite(row.feature)
        base = {'origin': row.origin, 'origin_index': row.origin_index,
                'target_month': row.target_month, 'target_start': row.target_start,
                'target_end': row.target_end, 'target_days': row.target_days,
                'observed_risk': row.label, 'training_count': len(train),
                'training_first_origin': train.origin.min(), 'training_last_origin': train.origin.max(),
                'training_last_maturity': train.target_end.max(), 'feature': row.feature,
                'fit_status': 'FIT_READY' if supported else 'INSUFFICIENT_PAST_INFORMATION'}
        if supported:
            assert train.target_end.le(row.origin).all() and train.origin.lt(row.origin).all()
            fit = fit_risk(train.feature.to_numpy(), train.label.to_numpy(), row.feature, cfg)
            predictions = {
                CANDIDATE: fit['prediction'],
                BASELINES[0]: max(float(train.label.mean()), cfg['positive_risk_floor']),
                BASELINES[1]: max(row.recent_downside, cfg['positive_risk_floor']) if np.isfinite(row.recent_downside) else np.nan,
            }
            members.extend({'fit_origin': row.origin, 'sample_origin': item.origin,
                            'target_start': item.target_start, 'target_end': item.target_end}
                           for item in train.itertuples())
        else:
            fit = {key: np.nan for key in ['training_mean', 'training_scale', 'coefficient',
                                          'intercept', 'current_standardized', 'smearing', 'log_prediction']}
            predictions = {model: np.nan for model in [CANDIDATE] + BASELINES}
        forecasts.append({**base, **{key: value for key, value in fit.items() if key != 'prediction'}, **predictions})
    return pd.DataFrame(forecasts), pd.DataFrame(members)


def loss(observed, forecast):
    observed, forecast = np.asarray(observed, float), np.asarray(forecast, float)
    assert np.isfinite(observed).all() and (observed >= 0).all()
    assert np.isfinite(forecast).all() and (forecast > 0).all()
    return np.log(forecast) + observed / forecast


def bootstrap_indices(n, cfg):
    rng = np.random.default_rng(cfg['random_seed'])
    block = cfg['bootstrap_block_months']
    starts = rng.integers(0, n, size=(cfg['bootstrap_repetitions'], math.ceil(n / block)))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(len(starts), -1)[:, :n]


def assess(forecasts, cfg):
    predictions, period_rows, annual_rows = [], [], []
    for period, (start, end, _) in cfg['periods'].items():
        piece = forecasts[forecasts.target_start.ge(start) & forecasts.target_end.le(end)
                          & forecasts.fit_status.eq('FIT_READY')].copy()
        assert piece[[CANDIDATE] + BASELINES].notna().all().all()
        piece['period'] = period
        for model in [CANDIDATE] + BASELINES:
            piece[model + '_loss'] = loss(piece.observed_risk, piece[model])
            piece[model + '_squared_error'] = (piece.observed_risk - piece[model]) ** 2
        for baseline in BASELINES:
            piece[baseline + '_improvement'] = piece[baseline + '_loss'] - piece[CANDIDATE + '_loss']
            period_rows.append({'period': period, 'baseline': baseline, 'months': len(piece),
                                'candidate_loss': float(piece[CANDIDATE + '_loss'].mean()),
                                'baseline_loss': float(piece[baseline + '_loss'].mean()),
                                'mean_loss_improvement': float(piece[baseline + '_improvement'].mean()),
                                'candidate_mse': float(piece[CANDIDATE + '_squared_error'].mean()),
                                'baseline_mse': float(piece[baseline + '_squared_error'].mean())})
        for year, group in piece.groupby(piece.target_start.dt.year):
            for baseline in BASELINES:
                annual_rows.append({'period': period, 'year': int(year), 'partial_year': int(year) == 2026,
                                    'baseline': baseline, 'months': len(group),
                                    'mean_loss_improvement': float(group[baseline + '_improvement'].mean())})
        predictions.append(piece)
    paired = pd.concat(predictions).sort_values('target_start').reset_index(drop=True)
    assert not paired.target_month.duplicated().any()
    draws = bootstrap_indices(len(paired), cfg)
    bootstrap, distributions = [], {}
    for baseline in BASELINES:
        distribution = paired[baseline + '_improvement'].to_numpy()[draws].mean(axis=1)
        lower = float(np.quantile(distribution, cfg['bootstrap_lower_quantile']))
        distributions[baseline] = distribution
        bootstrap.append({'baseline': baseline, 'months': len(paired),
                          'mean_loss_improvement': float(paired[baseline + '_improvement'].mean()),
                          'lower_5pct': lower, 'upper_95pct': float(np.quantile(distribution, .95)),
                          'lower_positive': lower > 0})
    gates = []
    for row in period_rows:
        gates.append({'gate': row['period'] + '_minimum_months_' + row['baseline'],
                      'pass': row['months'] >= cfg['gate']['minimum_paired_months_each_period'], 'value': row['months']})
        gates.append({'gate': row['period'] + '_positive_increment_' + row['baseline'],
                      'pass': row['mean_loss_improvement'] > 0, 'value': row['mean_loss_improvement']})
    for row in bootstrap:
        gates.append({'gate': 'bootstrap_positive_lower_' + row['baseline'],
                      'pass': row['lower_positive'], 'value': row['lower_5pct']})
    for baseline in BASELINES:
        complete = [r for r in annual_rows if r['baseline'] == baseline and not r['partial_year']]
        assert {r['year'] for r in complete} == set(range(2015, 2026))
        assert all(r['months'] == 12 for r in complete)
        wins = sum(r['mean_loss_improvement'] > 0 for r in complete)
        gates.append({'gate': 'full_year_wins_' + baseline,
                      'pass': wins >= cfg['gate']['complete_year_wins_minimum_vs_each_comparator'],
                      'value': wins, 'full_years': len(complete)})
    return paired, period_rows, annual_rows, bootstrap, pd.DataFrame(distributions), gates


def freeze():
    cfg = read(CONFIG)
    assert not (OUT / 'protocol.json').exists(), '本轮已登记，不重复登记'
    tests = read(OUT / 'tests_receipt.json')
    assert tests['passed'] == 5 and tests['failed'] == 0
    paths = [CONFIG, ROOT / cfg['rules'], Path(__file__), ROOT / cfg['features'], ROOT / cfg['dividends'],
             ROOT / 'tests/test_monthly_downside_forecast_v1.py',
             ROOT / 'research/monthly_single_factor_walkforward_v1.py',
             ROOT / 'research/adaptive_allocation_v1.py',
             ROOT / 'research/post_selection_continuous_accounts_v1.py']
    protocol = {**cfg, 'registered_at': now(), 'tests_passed': tests['passed'],
                'candidate_forecast_results_read_before_registration': False,
                'frozen_files': [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in paths]}
    write_json(OUT / 'protocol.json', protocol, exclusive=True)
    index = read(INDEX)
    index['updated_at'] = now()
    index['running_studies'].append({'study': cfg['study_id'], 'status': 'FROZEN_READY_FOR_PREDICTION',
                                    'configuration': str(CONFIG.relative_to(ROOT))})
    index['next_work'].update(registered=True, next_historical_question_status='FROZEN_READY_TO_RUN',
                              planned_settings=1, planned_new_accounts=0,
                              conditional_new_accounts_if_prediction_passes=12)
    index['last_goal_turn_classification'] = 'PROGRESS_MONTHLY_DOWNSIDE_STUDY_REGISTERED'
    write_json(INDEX, index)
    print('一个风险预测方法及两个基准已登记，尚未读取本轮预测成绩。', flush=True)


def check_protocol():
    cfg = read(OUT / 'protocol.json')
    for item in cfg['frozen_files']:
        assert digest(ROOT / item['path']) == item['sha256'], '本轮固定输入或方法发生变化：' + item['path']
    return cfg


def predict():
    cfg = check_protocol()
    assert not (OUT / 'prediction_result.json').exists(), '本轮预测已完成，不重复拟合'
    began = time.perf_counter()
    samples = samples_from_data(inputs(cfg), cfg)
    forecasts, members = predict_months(samples, cfg)
    paired, periods, annual, bootstrap, distributions, gates = assess(forecasts, cfg)
    for name, value in [('monthly_samples.csv', samples), ('forecasts.csv', forecasts),
                        ('training_members.csv', members), ('paired_losses.csv', paired),
                        ('period_metrics.csv', periods), ('yearly_metrics.csv', annual),
                        ('bootstrap_means.csv', distributions)]:
        csv(OUT / name, value)
    passed = all(row['pass'] for row in gates)
    result = {'study_id': cfg['study_id'], 'completed_at': now(),
              'status': 'PREDICTION_PASS_ACCOUNT_STAGE_PENDING' if passed else 'REJECTED_FIXED_RISK_FORECAST_NO_STABLE_INCREMENT',
              'new_candidate_methods': 1, 'new_model_fits': int(forecasts.fit_status.eq('FIT_READY').sum()),
              'paired_months': len(paired), 'prediction_gate_pass': passed,
              'gates': gates, 'period_metrics': periods, 'bootstrap': bootstrap,
              'account_stage': 'READY_TO_REGISTER' if passed else 'NOT_RUN_PREDICTION_GATE_FAILED',
              'new_accounts': 0, 'net_sharpe': 'NOT_COMPUTED', 'net_annual_return': 'NOT_COMPUTED',
              'strict_forward_evidence_days': 0, 'independent_validation': False,
              'goal_achieved': False, 'position_impact': 0, 'run_seconds': time.perf_counter() - began}
    write_json(OUT / 'prediction_result.json', result, exclusive=True)
    print(f'完成{result["new_model_fits"]}次单因子风险拟合、{len(paired)}个成熟月检验；预测门通过：{passed}。', flush=True)


def verify():
    cfg = check_protocol()
    result = read(OUT / 'prediction_result.json')
    samples = pd.read_csv(OUT / 'monthly_samples.csv', parse_dates=['origin', 'target_start', 'target_end'])
    forecasts = pd.read_csv(OUT / 'forecasts.csv', parse_dates=['origin', 'training_last_maturity', 'target_end'])
    members = pd.read_csv(OUT / 'training_members.csv', parse_dates=['fit_origin', 'sample_origin', 'target_start', 'target_end'])
    paired = pd.read_csv(OUT / 'paired_losses.csv')
    fitted = forecasts[forecasts.fit_status.eq('FIT_READY')]
    assert members.target_end.le(members.fit_origin).all()
    assert members.sample_origin.lt(members.fit_origin).all()
    assert fitted.training_last_maturity.le(fitted.origin).all()
    counts = members.groupby('fit_origin').size()
    assert len(fitted) == result['new_model_fits']
    for row in fitted.itertuples():
        assert counts.loc[row.origin] == row.training_count and 24 <= row.training_count <= 60
    data = inputs(cfg)
    complete = samples[samples.target_end.notna()]
    previous_end = None
    for row in complete.itertuples():
        raw = data.loc[data.date.between(row.target_start, row.target_end), 'total_simple'].to_numpy(float)
        expected = sum(min(float(r), 0) ** 2 for r in raw) / len(raw) * cfg['annual_days']
        np.testing.assert_allclose(expected, row.label, atol=1e-12, rtol=0)
        assert len(raw) == row.target_days
        assert previous_end is None or row.target_start > previous_end
        previous_end = row.target_end
    for model in [CANDIDATE] + BASELINES:
        expected_loss = np.log(paired[model].to_numpy()) + paired.observed_risk.to_numpy() / paired[model].to_numpy()
        np.testing.assert_allclose(expected_loss, paired[model + '_loss'], atol=1e-10, rtol=0)
    distribution = pd.read_csv(OUT / 'bootstrap_means.csv')
    for row in result['bootstrap']:
        np.testing.assert_allclose(np.quantile(distribution[row['baseline']], .05), row['lower_5pct'], atol=1e-12, rtol=0)
    assert result['prediction_gate_pass'] == all(row['pass'] for row in result['gates'])
    write_json(OUT / 'saved_verification_receipt.json', {
        'verified_at': now(), 'status': 'PASS_SAVED_LABEL_TIMING_AND_PREDICTION_LOSS_CHECKS',
        'fitted_models': len(fitted), 'paired_months': len(paired), 'complete_risk_labels': len(complete),
        'training_members': len(members), 'new_fits_generated_by_verification': 0,
        'new_accounts': 0, 'independent_validation': False, 'goal_achieved': False}, exclusive=True)
    print('保存结果核对通过：月度风险标签、样本成熟时间和预测损失一致。', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='固定单因子下行风险检验')
    parser.add_argument('stage', choices=['freeze', 'predict', 'verify'])
    args = parser.parse_args()
    {'freeze': freeze, 'predict': predict, 'verify': verify}[args.stage]()
