"""只读核对保存账户、退出触发及区块结果，不重跑策略或生成随机样本。"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.financial_annual_components_v1 import read, save, now
from research.adaptive_allocation_v1 import summarize
from research.intraday_overnight_increment_v1 import interval

OUT = ROOT / 'reports/research/510300_forward_eps_utility_exit_v1'
SOURCE = ROOT / 'reports/research/510300_forward_eps_monthly_policy_v2_csi'
config = read(ROOT / 'config/510300_forward_eps_utility_exit_v1.json')
result = read(OUT / 'result.json')
data = pd.read_parquet(ROOT / 'reports/research/510300_adaptive_allocation_v1/features.parquet')
signals = pd.read_parquet(SOURCE / 'signals.parquet')
assert data.date.tolist() == signals.date.tolist()
trigger_count = 0
for metric in result['all_metrics']:
    folder = OUT / 'evaluation' / metric['cost']
    ledger = pd.read_parquet(folder / (metric['model'] + '_ledger.parquet'))
    decisions = pd.read_parquet(folder / (metric['model'] + '_decisions.parquet'))
    actual = summarize(ledger, config)
    for key, value in actual.items():
        expected = metric[key]
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            np.testing.assert_allclose(value, expected, atol=1e-10, rtol=1e-10, equal_nan=True)
        else:
            assert value == expected
    np.testing.assert_allclose(ledger.equity, ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, atol=1e-7, rtol=0)
    previous = np.r_[config['initial_capital'], ledger.equity.to_numpy()[:-1]]
    np.testing.assert_allclose(ledger.net_return, ledger.equity.to_numpy()/previous-1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(ledger.pnl, ledger.price_pnl+ledger.dividend_recognized-ledger.commission-ledger.slippage_cost, atol=1e-7, rtol=0)
    if metric['model'] == 'BUY_HOLD':
        continue
    model = metric['model']
    allowed = set(signals.loc[signals.event_mask].index)
    assert decisions.loc[decisions.requested_quantity.gt(0), 'origin_index'].isin(allowed).all()
    for decision in decisions.itertuples():
        t = int(decision.origin_index)
        if pd.notna(decision.latest_valid_forecast_origin):
            known = signals.loc[:t]
            valid = known.loc[known.event_mask & np.isfinite(known.E3_FORWARD_EPS)]
            assert decision.latest_valid_forecast_origin == valid.date.iloc[-1]
            assert decision.forecast_age_trading_days == t-valid.index[-1]
        if not decision.new_exit_trigger:
            continue
        trigger_count += 1
        if '六十交易日' in decision.exit_reasons:
            assert decision.forecast_age_trading_days >= 60
        if '一百二十日均线' in decision.exit_reasons:
            assert t > 0 and (data.sma120.iloc[t-1:t+1] <= 0).all()
        next_ledger = ledger.loc[ledger.date.eq(decision.execution_date)]
        assert len(next_ledger) == 1
        assert int(next_ledger.requested_quantity.iloc[0]) == int(decision.requested_quantity) < 0
    if model in ['Z0_ORIGINAL_MONTHLY', 'Z1_FORECAST_EXPIRY']:
        original = pd.read_parquet(SOURCE / 'evaluation' / metric['cost'] / 'E3_FORWARD_EPS_ledger.parquet')
        for key in ['equity','shares','net_return','commission','slippage_cost']:
            np.testing.assert_array_equal(ledger[key], original[key])
for cost in config['costs']:
    for block in config['bootstrap_day_blocks']:
        saved = pd.read_parquet(OUT / f'{cost}_block{block}_saved_bootstrap_statistics.parquet')
        assert len(saved) == config['bootstrap_repetitions']
        for column in saved:
            np.testing.assert_allclose(interval(saved[column].dropna().tolist()),
                                       result['uncertainty'][cost][str(block)][column+'_95_interval'], atol=1e-12, rtol=0)
verification = {'checked_at':now(), 'status':'PASS_SAVED_UTILITY_ACCOUNTS_EXIT_CLOCK_AND_INTERVALS',
                'complete_accounts_checked':len(result['all_metrics']), 'first_exit_triggers_checked':trigger_count,
                'original_and_expiry_accounts_identical_to_original':True, 'bootstrap_files_checked':4,
                'new_accounts_generated':0, 'new_models_fit':0, 'new_downloads':0,
                'random_samples_regenerated':0, 'security_audit_performed':False}
save(OUT / 'saved_numerical_verification.json', verification, exclusive=True)
print('八条账户、十八次退出触发和保存区间的只读数值核对通过。', flush=True)
