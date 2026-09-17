"""静态复用共同账户与测试场景，为排序信号替换必要定义。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
entry = (ROOT/'research/drawdown_depth_risk_v1.py').read_text(encoding='utf-8')
entry = entry.replace('drawdown_depth_risk', 'session_signed_rank').replace('drawdown_depth_frames', 'session_signed_rank_frames')
entry = entry.replace('DRAWDOWN_DEPTH_RISK', 'SESSION_SIGNED_RANK').replace('第161轮', '第162轮').replace('round=161', 'round=162')
entry = entry.replace('趋势与持续回撤双上限', '日内隔夜带符号排序')
entry = entry.replace('510300_sequential_return_state_v1.json', '510300_drawdown_depth_risk_v1.json')
start, end = entry.index('        trend_window=120,'), entry.index('        new_model_fits=0,')
entry = entry[:start]+'''        rank_window=60, volatility_window=20, risk_target=.1, entry_threshold=1.96, exit_threshold=0., confirmation_closes=2,
        rank_method="NONZERO_ABSOLUTE_DIFFERENCE_EXACT_TIE_AVERAGE_SIGNED_SUM_OVER_RANK_SQUARE_ROOT",
        zero_difference="KNOWN_TIE_EXCLUDED_FROM_RANK_ALL_ZERO_KNOWN_SCORE_ZERO", continuity_correction=False,
        missing_window="WHOLE_WINDOW_UNKNOWN_RESET_DIRECTION_AND_CONFIRMATION_COUNTERS",
        direction="TWO_SCORES_ABOVE_ENTRY_ALLOW_TWO_BELOW_ZERO_DISALLOW_OTHERWISE_PREVIOUS",
        initial_direction=0, signal_parameters_fitted=False,
''' + entry[end:]
entry = entry.replace('PROGRESS_ROUND160_COMPLETED', 'PROGRESS_ROUND161_COMPLETED')
with (ROOT/'research/session_signed_rank_v1.py').open('x', encoding='utf-8') as stream:
    stream.write(entry)

source = (ROOT/'tests/test_drawdown_depth_risk_v1.py').read_text(encoding='utf-8')
header = source[:source.index('def test_known_drawdown_depth')]
header = header.replace('局部先后高点、持续回撤、双上限', '带符号排序、连续确认、波动仓位')
header = header.replace('from research.drawdown_depth_risk_inputs_v1 import drawdown_depth_factors, drawdown_depth_frames, PRIMARY, CANDIDATES',
    'from scipy.stats import rankdata, wilcoxon\nfrom research.session_signed_rank_inputs_v1 import signed_rank_summary, confirmed_rank_directions, session_signed_rank_factors, session_signed_rank_frames, PRIMARY, CANDIDATES')
header = header.replace("'drawdown_window': 60,", "'rank_window': 60, 'entry_threshold': 1.96, 'exit_threshold': 0., 'confirmation_closes': 2,")
header = header.replace(", 'drawdown_budget': .04", '')
header = header.replace("    data['vol20'] = .2", "    data['intraday_log'] = np.log((data.close+data.dividend)/(data.open+data.dividend))\n    data['overnight_log'] = np.log((data.open+data.dividend)/data.previous_close)\n    data['vol20'] = .2", 1)
tests = '''def test_known_signed_ranks_ties_zero_independent_score_and_extreme_rank():
    values = np.array([1., -2., 2., 0., 3.])
    got = signed_rank_summary(values)
    assert got['nonzero_count'] == 4 and got['positive_rank_sum'] == 7.5 and got['negative_rank_sum'] == 2.5
    assert got['rank_square_sum'] == 29.5 and got['rank_score60'] == pytest.approx(5/np.sqrt(29.5))
    ranks = rankdata(abs(values[values != 0]), method='average')
    expected = np.dot(np.sign(values[values != 0]), ranks)/np.linalg.norm(ranks)
    assert got['rank_score60'] == pytest.approx(expected)
    assert got['rank_score60'] == pytest.approx(wilcoxon(values, alternative='greater', method='asymptotic', zero_method='wilcox', correction=False).zstatistic)
    values[-1] = 3000.
    assert signed_rank_summary(values) == got
    assert signed_rank_summary(-values)['rank_score60'] == -got['rank_score60']
    assert signed_rank_summary(np.zeros(60))['rank_score60'] == 0.
    with pytest.raises(ValueError):
        signed_rank_summary([1., np.nan])


def test_strict_threshold_two_closes_state_retention_and_missing_reset():
    f = confirmed_rank_directions([1.96, 2., 2., 0., -.1, -.1, 2., np.nan, 2., 2.])
    np.testing.assert_allclose(f.positive_direction, [0, 0, 1, 1, 1, 0, 0, np.nan, 0, 1], equal_nan=True)
    np.testing.assert_allclose(f.positive_confirmation_count, [0, 1, 2, 0, 0, 0, 1, np.nan, 1, 2], equal_nan=True)
    data, parents, start = fixture()
    data.loc[150, 'intraday_log'] = np.nan
    f = session_signed_rank_factors(data)
    assert f.rank_score60.iloc[150:210].isna().all() and pd.notna(f.rank_score60.iloc[210])
    frames, _ = session_signed_rank_frames(data, parents, CFG, start)
    assert frames['BASE'][PRIMARY+'_target'].iloc[150:210].isna().all()


def test_ordinary_risk_cap_unknown_positive_and_known_flat_state():
    data, parents, start = fixture()
    data.loc[140:143, 'vol20'] = [np.nan, 0., .4, .05]
    frames, _ = session_signed_rank_frames(data, parents, CFG, start)
    target = frames['BASE'][PRIMARY+'_target']
    assert target.iloc[139] == .5 and target.iloc[140:142].isna().all()
    assert target.iloc[142] == .25 and target.iloc[143] == 1.
    flat = data.copy(); flat[['intraday_log', 'overnight_log']] = 0.; flat['vol20'] = np.nan
    frames, _ = session_signed_rank_frames(flat, parents, CFG, start)
    assert frames['BASE'][PRIMARY+'_target'].iloc[124:-1].eq(0.).all()


def test_ex_dividend_session_difference_zero_and_initial_band_exception():
    raw = np.r_[np.full(125, 10.), np.full(20, 9.)]
    cash = np.zeros(len(raw)); cash[125] = 1.
    data = prices(raw, cash); data.loc[125, 'open'] = 9.
    data['intraday_log'] = np.log((data.close+data.dividend)/(data.open+data.dividend))
    data['overnight_log'] = np.log((data.open+data.dividend)/data.previous_close)
    data['vol20'] = .2
    f = session_signed_rank_factors(data)
    assert f.session_difference.iloc[125] == 0. and f.rank_score60.iloc[60:].eq(0.).all()
    account = Account(200000.)
    assert target_request(account, 10., .05, CFG)['requested_quantity'] == 1000
    account.cash, account.shares = 100000., 10000
    assert target_request(account, 10., .55, CFG)['requested_quantity'] == 0
    assert target_request(account, 10., .7, CFG)['requested_quantity'] == 4000
    assert target_request(account, 10., 0., CFG)['requested_quantity'] == -10000


'''
tail = source[source.index('def test_future_prefix'):]
tail = tail.replace('drawdown_depth_frames', 'session_signed_rank_frames')
tail = tail.replace("changed.loc[300:, 'wealth'] *= 2.", "changed.loc[300:, 'intraday_log'] *= -1.")
tail = tail.replace("    data['vol20'] = .2; data.loc[145, 'vol20'] = np.nan", "    data['intraday_log'] = np.log((data.close+data.dividend)/(data.open+data.dividend))\n    data['overnight_log'] = np.log((data.open+data.dividend)/data.previous_close)\n    data['vol20'] = .2; data.loc[145, 'vol20'] = np.nan")
with (ROOT/'tests/test_session_signed_rank_v1.py').open('x', encoding='utf-8') as stream:
    stream.write(header+tests+tail)
print('第162轮入口与六项必要测试已生成，尚未冻结或计算新历史账户。')
