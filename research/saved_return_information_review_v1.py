"""配对旧的逆回购及季度申赎预测，保留全部排除理由，不重新拟合。"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_saved_return_information_review_v1'
INDEX = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
RULES = ROOT / 'docs/510300_SAVED_RETURN_INFORMATION_REVIEW_V1.md'
REPO = ROOT / 'reports/research/510300_total_reverse_repo_v2'
FLOW = ROOT / 'reports/research/510300_original_quarterly_flow_policy_v1'
PAIRS = [
    ('REPO_ALL_VS_PRICE', '逆回购各期限及买断式披露对价格', 'REPO', 'T1_PRIMARY_ALL', 'T5_PRICE'),
    ('REPO_ALL_VS_SEVEN', '逆回购各期限及买断式披露对七天量', 'REPO', 'T1_PRIMARY_ALL', 'T3_SEVEN_DISCLOSURE'),
    ('REPO_REGULAR_VS_SEVEN', '常规逆回购全期限对七天量', 'REPO', 'T2_REGULAR_ALL_TENOR', 'T3_SEVEN_DISCLOSURE'),
    ('FUND_FLOW_VS_PRICE', '价格加季度申赎对价格', 'FLOW', 'M1_PRICE_FLOW', 'M2_PRICE_ONLY'),
]
REPO_COMPONENTS = {
    'T1_PRIMARY_ALL': ['ALL_RIDGE_H20', 'ALL_ET_H20'],
    'T2_REGULAR_ALL_TENOR': ['REGULAR_RIDGE_H20', 'REGULAR_ET_H20'],
    'T3_SEVEN_DISCLOSURE': ['SEVEN_RIDGE_H20', 'SEVEN_ET_H20'],
    'T5_PRICE': ['PRICE_RIDGE_H20', 'PRICE_ET_H20'],
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def save(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    else:
        temporary = path.with_name(path.name + '.information_review.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(path)


def csv(name, data):
    pd.DataFrame(data).to_csv(OUT / name, index=False, encoding='utf-8-sig')


def inventory(index):
    rows = []
    selected = set(range(1, 12)) | {14, 25, 31, 32, 86, 116}
    for entry in index['completed_rounds']:
        if entry['round'] not in selected:
            continue
        source = ROOT / entry['result']
        result = read(source)
        files = sorted(p.name for p in source.parent.iterdir() if p.is_file())
        evidence = [name for name in files if any(word in name.lower() for word in ['prediction', 'probabilit', 'signal', 'training', 'saved_model', 'label'])]
        rows.append({'original_round': entry['round'], 'study_id': entry['study'],
                     'original_status': result.get('status'), 'result_path': str(source.relative_to(ROOT)),
                     'evidence_files': evidence, 'independent_validation_original': result.get('independent_validation'),
                     'original_result_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    assert len(rows) == 17
    return rows


def register():
    assert not (OUT / 'protocol.json').exists(), '本次已登记，不能重新登记'
    index = read(INDEX)
    records = inventory(index)
    paths = [RULES, Path(__file__), ROOT / 'data/reference/510300_dividends.csv',
             ROOT / 'research/total_reverse_repo_v2.py', ROOT / 'research/original_quarterly_flow_policy_v1.py',
             ROOT / 'research/intraday_overnight_increment_v1.py',
             ROOT / 'reports/research/510300_adaptive_allocation_v1/features.parquet']
    paths += [REPO / name for name in ['predictions.parquet', 'labels.parquet', 'features.parquet', 'training_receipts.json', 'source_reconstruction_receipt.json']]
    paths += [FLOW / name for name in ['event_signals.parquet', 'quarterly_labels.parquet', 'evaluation_event_features.parquet', 'training_receipts.json', 'source_receipt.json']]
    protocol = {
        'study_id': '510300_SAVED_RETURN_INFORMATION_REVIEW_V1', 'registered_at': now(),
        'rules': str(RULES.relative_to(ROOT)), 'pairs': PAIRS,
        'original_cutoff': '2026-08-14', 'bootstrap_repetitions': 5000, 'seed': 20260915,
        'periods': {'overall': ['2020-01-02', '2026-08-14'],
                    'first': ['2020-01-02', '2022-12-31'], 'second': ['2023-01-01', '2026-08-14']},
        'new_strategy_candidates': 0, 'new_model_fits': 0, 'new_accounts': 0,
        'original_history_previously_seen': True,
        'frozen_files': [{'path': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
    }
    save(OUT / 'protocol.json', protocol, True)
    save(OUT / 'original_evidence_inventory.json', {'rows': records, 'count': len(records)}, True)
    index['running_studies'].append({'study': protocol['study_id'], 'status': 'REGISTERED_SAVED_COMPARISONS_READY'})
    index['last_goal_turn_classification'] = 'PROGRESS_RETURN_INFORMATION_RECORDS_LOCATED_AND_REVIEW_REGISTERED'
    index['updated_at'] = now()
    save(INDEX, index)
    print('17项原研究证据已定位，四项固定旧预测配对已登记。', flush=True)


def verify_label(data, dividends, origin_index, horizon, stored):
    entry = origin_index + 1
    end = origin_index + horizon + 1
    if end >= len(data):
        assert not np.isfinite(stored)
        return
    if not np.isfinite(stored):
        return
    start_date, end_date = data.date.iloc[entry], data.date.iloc[end]
    entitled = dividends[(dividends.record_date >= start_date) & (dividends.record_date < end_date)
                         & (dividends.ex_date <= end_date)]
    expected = (float(data.open.iloc[end]) + float(entitled.cash_dividend_per_share.sum())) / float(data.open.iloc[entry]) - 1
    np.testing.assert_allclose(expected, stored, atol=1e-12, rtol=0)


def sources():
    repo_predictions = pd.read_parquet(REPO / 'predictions.parquet')
    repo_labels = pd.read_parquet(REPO / 'labels.parquet')
    data = pd.read_parquet(REPO / 'features.parquet')
    data['date'] = pd.to_datetime(data.date)
    dividends = pd.read_csv(ROOT / 'data/reference/510300_dividends.csv', parse_dates=['record_date', 'ex_date'])
    np.testing.assert_array_equal(repo_predictions.date, data.date)
    np.testing.assert_array_equal(repo_labels.date, data.date)
    repo = repo_predictions.copy()
    repo['origin_index'] = np.arange(len(repo))
    repo['label'] = repo_labels.Y20
    repo['entry_date'] = data.date.shift(-1)
    repo['exit_date'] = data.date.shift(-21)
    repo['label_origin_saved'] = True
    for t, stored in enumerate(repo.label):
        verify_label(data, dividends, t, 20, stored)
    receipts = pd.DataFrame(read(REPO / 'training_receipts.json'))
    for name in ['fit_origin', 'last_label_exit']:
        receipts[name] = pd.to_datetime(receipts[name]).astype('datetime64[ns]')
    assert receipts.last_label_exit.le(receipts.fit_origin).all()
    for model, components in REPO_COMPONENTS.items():
        np.testing.assert_allclose(repo[model], repo[components].mean(axis=1), atol=1e-12, rtol=0, equal_nan=True)
        valid = repo[repo[model].notna()].copy()
        valid['date'] = valid.date.astype('datetime64[ns]')
        for component in components:
            fits = receipts[receipts.model.eq(component)].sort_values('fit_origin')
            joined = pd.merge_asof(valid[['date']], fits[['fit_origin', 'last_label_exit']],
                                   left_on='date', right_on='fit_origin', direction='backward')
            assert joined.fit_origin.notna().all() and joined.last_label_exit.le(joined.date).all()
    raw_flow = pd.read_parquet(FLOW / 'event_signals.parquet')
    flow_labels = pd.read_parquet(FLOW / 'quarterly_labels.parquet')
    flow = raw_flow[raw_flow.event_mask].copy()
    flow = flow.merge(flow_labels[['origin', 'origin_index', 'Y60', 'label_exit_date']],
                      left_on='date', right_on='origin', how='left', validate='one_to_one')
    location = pd.Series(np.arange(len(data)), index=data.date)
    flow['origin_index'] = flow.date.map(location).astype(int)
    flow['entry_date'] = flow.origin_index.map(lambda t: data.date.iloc[t + 1] if t + 1 < len(data) else pd.NaT)
    flow['label'] = flow.Y60
    flow['exit_date'] = pd.to_datetime(flow.label_exit_date)
    flow['label_origin_saved'] = flow.origin.notna()
    for row in flow.itertuples():
        verify_label(data, dividends, row.origin_index, 60, row.label)
    flow_receipts = pd.DataFrame(read(FLOW / 'training_receipts.json')['rows'])
    for name in ['origin', 'latest_label_exit_date']:
        flow_receipts[name] = pd.to_datetime(flow_receipts[name])
    fitted = flow_receipts[flow_receipts.status.eq('TRAINED_POINT_IN_TIME_QUARTER_MODEL')]
    assert fitted.latest_label_exit_date.le(fitted.origin).all()
    for model in ['M1_PRICE_FLOW', 'M2_PRICE_ONLY']:
        check = raw_flow.loc[raw_flow[model].notna(), ['date', model]].merge(
            fitted[fitted.model.eq(model)][['origin', 'prediction', 'training_samples']],
            left_on='date', right_on='origin', how='left', validate='one_to_one')
        assert check.origin.notna().all() and check.training_samples.ge(12).all()
        np.testing.assert_allclose(check[model], check.prediction, atol=1e-12, rtol=0)
    return {'REPO': repo, 'FLOW': flow}, {
        'repo_training_receipts': len(receipts), 'flow_training_receipts': len(fitted),
        'existing_labels_checked': int(repo.label.notna().sum() + flow.label.notna().sum()),
        'new_labels_created': 0, 'new_predictions_created': 0,
        'scope': '复核保存训练记录的日期和已有标签；没有重新重建历史外部来源到达时钟',
    }


def run():
    cfg = read(OUT / 'protocol.json')
    assert not (OUT / 'result.json').exists(), '已完成本轮固定配对，不重复运行'
    amendment_path = OUT / 'datetime_precision_amendment.json'
    amendment = read(amendment_path) if amendment_path.exists() else None
    for item in cfg['frozen_files']:
        expected = item['sha256']
        if amendment and item['path'] == amendment['program']:
            assert amendment['original_sha256'] == expected
            expected = amendment['corrected_sha256']
        assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == expected, item['path']
    inputs, verification = sources()
    period_rows, outcomes, all_exclusions, all_paired, bootstraps = [], [], [], [], []
    for pair_id, title, family, candidate, baseline in PAIRS:
        raw = inputs[family].copy()
        raw = raw[raw.date.ge('2019-12-31')].copy()
        raw['exclusion_reason'] = ''
        raw.loc[~raw.label_origin_saved, 'exclusion_reason'] = 'NO_SAME_ORIGIN_SAVED_LABEL'
        raw.loc[raw.label_origin_saved & raw.label.isna(), 'exclusion_reason'] = 'LABEL_NOT_MATURE_OR_ORIGINAL_INPUT_UNAVAILABLE'
        raw.loc[raw.exclusion_reason.eq('') & (raw[candidate].isna() | raw[baseline].isna()), 'exclusion_reason'] = 'ORIGINAL_FORECAST_UNAVAILABLE'
        for row in raw[raw.exclusion_reason.ne('')].itertuples():
            all_exclusions.append({'pair': pair_id, 'origin': row.date, 'reason': row.exclusion_reason})
        valid = raw[raw.exclusion_reason.eq('')].copy()
        valid['candidate_squared_error'] = (valid[candidate] - valid.label) ** 2
        valid['baseline_squared_error'] = (valid[baseline] - valid.label) ** 2
        valid['loss_improvement'] = valid.baseline_squared_error - valid.candidate_squared_error
        for period, (start, end) in cfg['periods'].items():
            piece = valid[valid.entry_date.ge(start) & valid.exit_date.le(end)].copy()
            monthly = piece.groupby(piece.date.dt.to_period('M')).loss_improvement.mean()
            candidate_mse, baseline_mse = float(piece.candidate_squared_error.mean()), float(piece.baseline_squared_error.mean())
            period_rows.append({'pair': pair_id, 'title': title, 'period': period, 'origins': len(piece),
                                'observed_origin_months': len(monthly), 'candidate_mse': candidate_mse,
                                'baseline_mse': baseline_mse, 'relative_mse_improvement': 1 - candidate_mse / baseline_mse,
                                'equal_month_mean_loss_improvement': float(monthly.mean())})
            if period == 'overall':
                all_paired.extend({'pair': pair_id, 'origin': row.date, 'entry': row.entry_date, 'exit': row.exit_date,
                                   'label': row.label, 'candidate_prediction': getattr(row, candidate),
                                   'baseline_prediction': getattr(row, baseline),
                                   'loss_improvement': row.loss_improvement} for row in piece.itertuples())
                block = 6 if family == 'REPO' else 4
                rng = np.random.default_rng(cfg['seed'])
                starts = rng.integers(0, len(monthly), size=(cfg['bootstrap_repetitions'], math.ceil(len(monthly) / block)))
                positions = ((starts[:, :, None] + np.arange(block)) % len(monthly)).reshape(len(starts), -1)[:, :len(monthly)]
                distribution = monthly.to_numpy()[positions].mean(axis=1)
                lower, upper = np.quantile(distribution, [.05, .95])
                bootstraps.extend({'pair': pair_id, 'draw': i, 'mean_loss_improvement': value} for i, value in enumerate(distribution))
                uncertainty = {'lower_5pct': float(lower), 'upper_95pct': float(upper),
                               'block_observed_months': block, 'observed_months': len(monthly)}
        split = [r for r in period_rows if r['pair'] == pair_id and r['period'] != 'overall']
        min_months = 12 if family == 'REPO' else 8
        sufficient = uncertainty['observed_months'] >= 24 and all(r['observed_origin_months'] >= min_months for r in split)
        consistent = sufficient and all(r['relative_mse_improvement'] > 0 for r in split) and uncertainty['lower_5pct'] > 0
        outcomes.append({'pair': pair_id, 'title': title, 'family': family, 'sufficient_observation_months': sufficient,
                         'both_segments_mse_improved': all(r['relative_mse_improvement'] > 0 for r in split),
                         'historical_paired_increment_consistent': consistent, 'uncertainty': uncertainty,
                         'earlier_2015_2019_prediction_evidence': 'NO_SAVED_FORECAST_COVERAGE',
                         'old_failed_strategy_promoted': False, 'independent_validation': False})
    csv('paired_predictions_and_losses.csv', all_paired)
    csv('excluded_forecast_origins.csv', all_exclusions)
    csv('period_prediction_metrics.csv', period_rows)
    csv('bootstrap_saved_statistics.csv', bootstraps)
    save(OUT / 'saved_evidence_verification.json', {'status': 'PASS_SAVED_LABELS_AND_TRAINING_RECEIPT_TIMING', **verification}, True)
    result = {'study_id': cfg['study_id'], 'completed_at': now(), 'status': 'COMPLETED_FOUR_SAVED_RETURN_INFORMATION_COMPARISONS',
              'original_studies_in_inventory': 17, 'fixed_comparisons': 4, 'outcomes': outcomes, 'period_metrics': period_rows,
              'consistent_historical_pairs': sum(row['historical_paired_increment_consistent'] for row in outcomes),
              'new_model_fits': 0, 'new_accounts': 0, 'new_strategy_candidates': 0,
              'independent_validation': False, 'goal_achieved': False, 'position_impact': 0}
    save(OUT / 'result.json', result, True)
    print(json.dumps({'consistent_pairs': result['consistent_historical_pairs'], 'outcomes': outcomes, 'period_metrics': period_rows}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='旧收益信息证据的固定配对')
    parser.add_argument('stage', choices=['register', 'run'])
    argument = parser.parse_args()
    {'register': register, 'run': run}[argument.stage]()
