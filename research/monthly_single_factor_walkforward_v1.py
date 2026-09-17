"""三个固定单因子的按月顺序学习与完整账户历史检验。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import now, digest, write_json
from research.adaptive_allocation_v1 import normalize_dividends, target_request
from research.post_selection_continuous_accounts_v1 import simulate_indexed_request_account
from research.strategy_review_diagnostics_v1 import metrics, cycles

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/510300_monthly_single_factor_walkforward_v1.json'
OUT = ROOT / 'reports/research/510300_monthly_single_factor_walkforward_v1'
OLD = ROOT / 'reports/research/510300_simple_core_window_diagnostic_v1/comparators'
FEATURES = ['mom20', 'z20', 'logvol20']
COMPARATORS = ['MONTHLY_MEAN', 'MONTHLY_VOL10']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False, encoding='utf-8-sig')


def inputs(cfg):
    data = pd.read_parquet(ROOT / cfg['features'])
    data['date'] = pd.to_datetime(data.date)
    assert data.date.is_monotonic_increasing and not data.date.duplicated().any()
    assert data.date.iloc[-1] == pd.Timestamp(cfg['data_cutoff'])
    data['logvol20'] = np.log(data.vol20.where(data.vol20 > 0))
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg['dividends']))
    return data, dividends


def monthly_samples(data, dividends):
    """只在已观察到下月首日的月末建记录，保留尚未成熟的最后一个标签。"""
    month = data.date.dt.to_period('M')
    origins = np.flatnonzero(month.iloc[:-1].to_numpy() != month.iloc[1:].to_numpy())
    rows = []
    for position, origin in enumerate(origins):
        entry = int(origin + 1)
        row = {'origin_index': int(origin), 'origin': data.date.iloc[origin],
               'entry_date': data.date.iloc[entry], 'entry_price': float(data.open.iloc[entry]),
               'exit_date': pd.NaT, 'mature_date': pd.NaT, 'exit_price': np.nan,
               'earned_dividend_per_share': np.nan, 'label': np.nan,
               **{feature: float(data[feature].iloc[origin]) for feature in FEATURES}}
        if position + 1 < len(origins):
            exit_index = int(origins[position + 1] + 1)
            exit_date = data.date.iloc[exit_index]
            earned = dividends[(dividends.record_date >= row['entry_date']) & (dividends.record_date < exit_date)]
            income = float(earned.cash_dividend_per_share.sum())
            maturity = max(exit_date, earned.ex_date.max()) if len(earned) else exit_date
            row.update(exit_date=exit_date, mature_date=maturity, exit_price=float(data.open.iloc[exit_index]),
                       earned_dividend_per_share=income,
                       label=(float(data.open.iloc[exit_index]) + income) / row['entry_price'] - 1)
        rows.append(row)
    result = pd.DataFrame(rows)
    for name in ['origin', 'entry_date', 'exit_date', 'mature_date']:
        result[name] = pd.to_datetime(result[name])
    return result


def single_ridge(x, y, current, cfg):
    """平均平方误差加固定系数平方惩罚；标准化仅依赖训练窗口。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mean, scale = float(x.mean()), float(x.std(ddof=1))
    scale = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
    z = np.clip((x - mean) / scale, -cfg['standardized_clip'], cfg['standardized_clip'])
    centered = z - z.mean()
    coefficient = float(centered @ (y - y.mean()) / (centered @ centered + len(x) * cfg['ridge_mean_loss_penalty']))
    intercept = float(y.mean() - coefficient * z.mean())
    current_z = float(np.clip((current - mean) / scale, -cfg['standardized_clip'], cfg['standardized_clip']))
    return {'training_mean': mean, 'training_scale': scale, 'coefficient': coefficient,
            'intercept': intercept, 'current_standardized': current_z,
            'prediction': float(intercept + coefficient * current_z)}


def signal_target(prediction, volatility, cfg):
    if not np.isfinite(prediction) or not np.isfinite(volatility) or volatility <= 0:
        return float('nan')
    return min(1.0, cfg['target_volatility'] / volatility) if prediction > cfg['entry_prediction_buffer'] else 0.0


def make_signals(data, samples, cfg, evaluation_origin='2014-12-31'):
    forecasts, members = [], []
    events = np.zeros(len(data), bool)
    models = list(cfg['candidate_models']) + COMPARATORS
    targets = {model: np.full(len(data), np.nan) for model in models}
    valid = np.isfinite(samples[FEATURES + ['label']].to_numpy(float)).all(axis=1)
    for current in samples.itertuples():
        origin, index = current.origin, current.origin_index
        if origin < pd.Timestamp(evaluation_origin):
            continue
        events[index] = True
        train = samples[valid & samples.mature_date.le(origin)].tail(cfg['training_months'])
        assert train.empty or (train.origin.lt(origin).all() and train.mature_date.le(origin).all())
        volatility = float(data.vol20.iloc[index])
        targets['MONTHLY_VOL10'][index] = min(1.0, cfg['target_volatility'] / volatility) if np.isfinite(volatility) and volatility > 0 else np.nan
        supported = len(train) >= cfg['minimum_training_months']
        mean_prediction = float(train.label.mean()) if supported else np.nan
        targets['MONTHLY_MEAN'][index] = signal_target(mean_prediction, volatility, cfg)
        if supported:
            members.extend({'fit_origin': origin, 'sample_origin': row.origin, 'entry_date': row.entry_date,
                            'exit_date': row.exit_date, 'mature_date': row.mature_date}
                           for row in train.itertuples())
        for model, feature in cfg['candidate_models'].items():
            current_value = float(getattr(current, feature))
            supported_model = supported and np.isfinite(current_value)
            values = single_ridge(train[feature], train.label, current_value, cfg) if supported_model else {
                key: np.nan for key in ['training_mean', 'training_scale', 'coefficient', 'intercept', 'current_standardized', 'prediction']}
            target = signal_target(values['prediction'], volatility, cfg)
            targets[model][index] = target
            forecasts.append({'model': model, 'feature': feature, 'origin': origin, 'origin_index': index,
                              'entry_date': current.entry_date, 'label_exit_date': current.exit_date,
                              'label_mature_date': current.mature_date, 'observed_label': current.label,
                              'training_count': len(train), 'training_first_origin': train.origin.min(),
                              'training_last_origin': train.origin.max(), 'training_last_maturity': train.mature_date.max(),
                              'fit_status': 'FIT_READY' if supported_model else 'INSUFFICIENT_PAST_INFORMATION',
                              'mean_only_prediction': mean_prediction, 'raw_feature': current_value,
                              'volatility': volatility, 'target': target, **values})
    return pd.DataFrame(forecasts), pd.DataFrame(members), targets, events


def freeze():
    cfg = read(CONFIG)
    assert not (OUT / 'protocol.json').exists(), '本轮已登记，直接继续未完成运行'
    test_receipt = read(OUT / 'tests_receipt.json')
    assert test_receipt['passed'] == 5
    data, dividends = inputs(cfg)
    paths = [CONFIG, ROOT / cfg['rules'], Path(__file__), ROOT / cfg['features'], ROOT / cfg['dividends'],
             ROOT / 'tests/test_monthly_single_factor_walkforward_v1.py',
             ROOT / 'research/post_selection_continuous_accounts_v1.py', ROOT / 'research/adaptive_allocation_v1.py',
             ROOT / 'research/intraday_overnight_increment_v1.py', ROOT / 'research/strategy_review_diagnostics_v1.py']
    for period in cfg['periods']:
        for model in ['BUY_HOLD', 'ETF_VOL10']:
            for cost in cfg['costs']:
                paths.append(OLD / f'{period}_{model}_{cost}' / 'ledger.parquet')
    protocol = {**cfg, 'registered_at': now(), 'tests_passed': 5, 'historical_account_results_read_for_this_study': False,
                'input_rows': len(data), 'dividend_events': len(dividends),
                'frozen_files': [{'path': str(p.relative_to(ROOT)), 'sha256': digest(p)} for p in paths]}
    write_json(OUT / 'protocol.json', protocol, exclusive=True)
    index_path = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
    index = read(index_path)
    assert not index['running_studies']
    index['historical_mining_authorization_20260914'] = {
        'received_at': now(), 'user_instruction': cfg['user_authorization'], 'rules': cfg['rules'],
        'historical_research_enabled': True, 'reducing_overfitting_required': True,
        'supersedes_wait_only_restriction': True, 'old_strategy_disposition_preserved': True}
    index['running_studies'] = [{'study': cfg['study_id'], 'configuration': str(CONFIG.relative_to(ROOT)), 'status': 'FROZEN_READY_TO_RUN'}]
    index['next_work']['historical_mining_allowed'] = True
    index['next_work']['historical_mining_rules'] = cfg['rules']
    index['next_work']['status'] = 'FINITE_HISTORICAL_MINING_AND_FIXED_DAILY_OBSERVATION'
    index['next_work']['focus'] = '完成已登记三个单因子的按月顺序学习及全部账户；原每日固定观察独立保留'
    index['last_goal_turn_classification'] = 'PROGRESS_FINITE_HISTORICAL_STUDY_REGISTERED'
    index['consecutive_external_data_blocked_goal_turns'] = 0
    index['updated_at'] = now()
    write_json(index_path, index)
    print('三个单因子、20份新账户和8份复用对照已登记，随后直接执行。', flush=True)


def check_protocol():
    protocol = read(OUT / 'protocol.json')
    for item in protocol['frozen_files']:
        assert digest(ROOT / item['path']) == item['sha256'], item['path']
    return protocol


def run():
    cfg = check_protocol()
    assert not (OUT / 'result.json').exists(), '已完成，不重复运行'
    began = time.perf_counter()
    data, dividends = inputs(cfg)
    signal_file = OUT / 'signals.npz'
    if not signal_file.exists():
        samples = monthly_samples(data, dividends)
        forecasts, members, targets, events = make_signals(data, samples, cfg)
        csv(OUT / 'monthly_samples.csv', samples)
        csv(OUT / 'monthly_forecasts.csv', forecasts)
        csv(OUT / 'training_members.csv', members)
        np.savez_compressed(signal_file, events=events, **targets)
        print(f'顺序学习完成：{int(forecasts.fit_status.eq("FIT_READY").sum())}次单因子拟合；开始完整账户。', flush=True)
    with np.load(signal_file) as stored:
        events = stored['events'].copy()
        targets = {model: stored[model].copy() for model in list(cfg['candidate_models']) + COMPARATORS}
    all_metrics = []
    for period, (start, end, next_date) in cfg['periods'].items():
        frame = data[data.date.le(end)].reset_index(drop=True)
        for model in list(cfg['candidate_models']) + COMPARATORS:
            for cost_id, cost in cfg['costs'].items():
                folder = OUT / 'accounts' / period / cost_id / model
                complete = folder / 'completed.json'
                if complete.exists():
                    all_metrics.append(read(complete))
                    continue
                ledger, decisions, state = simulate_indexed_request_account(
                    frame, dividends, cfg, cost, start, model, targets=targets[model][:len(frame)],
                    event_mask=events[:len(frame)], next_execution_date=next_date,
                    request_policy=lambda account, price, value, config, identity, t: target_request(account, price, value, config))
                indices = decisions.origin_index.to_numpy(int)
                decisions['monthly_event'] = events[indices]
                folder.mkdir(parents=True, exist_ok=True)
                ledger.to_parquet(folder / 'ledger.parquet', index=False)
                decisions.to_parquet(folder / 'decisions.parquet', index=False)
                csv(folder / 'trades.csv', ledger[ledger.filled_quantity.ne(0)])
                write_json(folder / 'checkpoint.json', state)
                item = {'period': period, 'model': model, 'cost': cost_id, 'reused': False, **metrics(ledger),
                        'ending_shares': int(ledger.shares.iloc[-1]),
                        'unfilled_requests': int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum())
                        if 'requested_quantity' in ledger else None}
                write_json(complete, item, exclusive=True)
                all_metrics.append(item)
                print('账户完成：' + period + '／' + cost_id + '／' + model, flush=True)
        for model in ['BUY_HOLD', 'ETF_VOL10']:
            for cost_id in cfg['costs']:
                source = OLD / f'{period}_{model}_{cost_id}' / 'ledger.parquet'
                ledger = pd.read_parquet(source)
                assert ledger.date.iloc[0] == pd.Timestamp(start) and ledger.date.iloc[-1] == pd.Timestamp(end)
                assert ledger.mark_clock.eq('CLOSE').all()
                all_metrics.append({'period': period, 'model': model, 'cost': cost_id, 'reused': True,
                                    'source': str(source.relative_to(ROOT)), **metrics(ledger),
                                    'ending_shares': int(ledger.shares.iloc[-1])})
    csv(OUT / 'account_metrics.csv', all_metrics)
    write_json(OUT / 'accounts_completed.json', {'completed_at': now(), 'new_accounts': 20, 'reused_accounts': 8,
               'run_seconds': time.perf_counter() - began, 'metrics': all_metrics}, exclusive=True)


def summarize():
    cfg = check_protocol()
    completed = read(OUT / 'accounts_completed.json')
    rows = pd.DataFrame(completed['metrics'])
    yearly, concentration, completed_cycles = [], [], []
    for row in completed['metrics']:
        source = ROOT / row['source'] if row['reused'] else OUT / 'accounts' / row['period'] / row['cost'] / row['model'] / 'ledger.parquet'
        ledger = pd.read_parquet(source)
        identity = {key: row[key] for key in ['period', 'model', 'cost']}
        for year, piece in ledger.groupby(ledger.date.dt.year):
            initial = float(piece.equity.iloc[0] / (1 + piece.net_return.iloc[0]))
            yearly.append({**identity, 'year': int(year), 'partial_year': int(year) == 2026, **metrics(piece, initial)})
        cycle_rows, active = cycles(ledger)
        winners = cycle_rows[cycle_rows.profit > 0].sort_values('profit', ascending=False) if len(cycle_rows) else cycle_rows
        unfinished = float(ledger.equity.iloc[-1] - active['start_equity']) if active else 0.0
        np.testing.assert_allclose((float(cycle_rows.profit.sum()) if len(cycle_rows) else 0.0) + unfinished, row['profit'], atol=1e-6)
        concentration.append({**identity, 'complete_cycles': len(cycle_rows), 'total_profit': row['profit'],
                              'largest_profit_share': float(winners.profit.iloc[0] / row['profit']) if len(winners) and row['profit'] > 0 else None,
                              'top_five_profit_share': float(winners.profit.head(5).sum() / row['profit']) if len(winners) and row['profit'] > 0 else None,
                              'unfinished_profit': unfinished})
        completed_cycles.extend({**identity, **item} for item in cycle_rows.to_dict('records'))
    csv(OUT / 'yearly_metrics.csv', yearly)
    csv(OUT / 'cycle_concentration.csv', concentration)
    csv(OUT / 'completed_cycles.csv', completed_cycles)
    forecasts = pd.read_csv(OUT / 'monthly_forecasts.csv', parse_dates=['origin', 'entry_date', 'label_mature_date'])
    predictions, outcomes = [], []
    for model in cfg['candidate_models']:
        model_predictions = []
        for period, (start, end, _) in cfg['periods'].items():
            piece = forecasts[forecasts.model.eq(model) & forecasts.entry_date.ge(start) & forecasts.label_mature_date.le(end)
                              & forecasts.fit_status.eq('FIT_READY')].copy()
            mse = float(((piece.prediction - piece.observed_label) ** 2).mean())
            reference_mse = float(((piece.mean_only_prediction - piece.observed_label) ** 2).mean())
            result = {'model': model, 'period': period, 'mature_prediction_months': len(piece),
                      'mse': mse, 'mean_only_mse': reference_mse,
                      'mse_improvement': reference_mse - mse, 'out_of_fit_r_squared': 1 - mse / reference_mse}
            predictions.append(result)
            model_predictions.append(result)
        comparison, target_pass = [], []
        for period in cfg['periods']:
            for cost in cfg['costs']:
                candidate = rows[rows.model.eq(model) & rows.period.eq(period) & rows.cost.eq(cost)].iloc[0]
                baseline = rows[rows.model.eq('MONTHLY_MEAN') & rows.period.eq(period) & rows.cost.eq(cost)].iloc[0]
                comparison.append(bool(candidate.annual_return > baseline.annual_return and pd.notna(candidate.sharpe)
                                       and pd.notna(baseline.sharpe) and candidate.sharpe > baseline.sharpe))
                target_pass.append(bool(candidate.annual_return >= .1 and pd.notna(candidate.sharpe) and candidate.sharpe >= 1.2))
        annual = pd.DataFrame(yearly)
        a = annual[annual.model.eq(model) & annual.cost.eq('STRESS') & ~annual.partial_year].set_index('year')
        b = annual[annual.model.eq('MONTHLY_MEAN') & annual.cost.eq('STRESS') & ~annual.partial_year].set_index('year')
        assert len(a) == len(b) == 11
        wins = int((a.annual_return > b.annual_return).sum())
        prediction_pass = all(p['mse_improvement'] > 0 for p in model_predictions)
        eligible = all(comparison) and wins >= cfg['continuation_full_year_outperformance_count_min'] and prediction_pass
        outcomes.append({'model': model, 'four_scenarios_better_than_mean': all(comparison),
                         'pressure_full_year_wins_over_mean': wins, 'full_years': 11,
                         'both_period_prediction_mse_improved': prediction_pass,
                         'eligible_for_further_mechanism_research': eligible,
                         'all_four_historical_point_targets_pass': all(target_pass),
                         'independent_validation': False, 'goal_achieved': False})
    csv(OUT / 'prediction_quality.csv', predictions)
    write_json(OUT / 'candidate_outcomes.json', {'candidates': outcomes, 'selected_model': None, 'goal_achieved': False})
    result = {'study_id': cfg['study_id'], 'completed_at': now(),
              'status': 'COMPLETED_FINITE_HISTORICAL_MONTHLY_WALKFORWARD', 'new_candidates': 3,
              'new_accounts': 20, 'reused_accounts': 8,
              'new_model_fits': int(forecasts.fit_status.eq('FIT_READY').sum()),
              'historical_data_previously_seen': True, 'strict_forward_evidence_days': 0,
              'selected_model': None, 'goal_achieved': False, 'position_impact': 0,
              'run_seconds': completed['run_seconds'], 'metrics': completed['metrics'], 'candidate_outcomes': outcomes}
    write_json(OUT / 'result.json', result, exclusive=True)
    print(rows[rows.cost.eq('STRESS')][['period', 'model', 'annual_return', 'sharpe']].to_string(index=False), flush=True)
    print(json.dumps(outcomes, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='有限单因子历史挖掘：登记、运行、汇总')
    parser.add_argument('stage', choices=['freeze', 'run', 'summarize'])
    args = parser.parse_args()
    {'freeze': freeze, 'run': run, 'summarize': summarize}[args.stage]()
