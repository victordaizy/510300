"""从已核对结构生成独立的固定加仓条件投入批次，旧研究入口保持不变。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('待替换结构数量不同：'+old[:100])
    return text.replace(old, new)


def save(path, text):
    with (ROOT/path).open('x', encoding='utf-8') as stream:
        stream.write(text)


def main():
    if (ROOT/'research/addition_gate_exposure_batch_v1.py').exists():
        raise ValueError('第206轮入口已存在，不能重复生成')
    source = (ROOT/'research/addition_only_return_gate_batch_inputs_v1.py').read_text(encoding='utf-8')
    left, right = source.index('SETTINGS, CANDIDATES = {}, {}'), source.index('\n\ndef market_factors')
    source = source[:left]+'''SETTINGS, CANDIDATES = {}, {}
for percent in [105, 110, 115, 120]:
    model = f'ADD_GATE_EXPOSURE_{percent}'
    SETTINGS[model] = {'parent': 'INTENT_MIX_BAND_20', 'band': .2, 'direction': 'LOWER',
        'return_threshold': .01, 'exposure_multiplier': percent/100}
    CANDIDATES[model] = f'固定1%加仓条件、原目标乘{percent/100:.2f}并限制100%'
PRIMARY = 'ADD_GATE_EXPOSURE_110'
'''+source[right:]
    source = replace_once(source, "values = frame[setting['parent']+'_parent_target'].to_numpy(float)",
        "values = np.minimum(1., frame[setting['parent']+'_parent_target'].to_numpy(float)*setting['exposure_multiplier'])")
    source = replace_once(source, "filter_direction=setting['direction'], return_threshold=float(setting['return_threshold']),",
        "exposure_multiplier=float(setting['exposure_multiplier']), filter_direction=setting['direction'], return_threshold=float(setting['return_threshold']),")
    source = source.replace('原初次入场与退出保持，只对实际持股后的正加仓申请应用收益条件。', '固定加仓条件，在100%以内小幅放大目标投入，并重新计算完整账户。')
    save('research/addition_gate_exposure_batch_inputs_v1.py', source)

    test = (ROOT/'tests/test_addition_only_return_gate_batch_v1.py').read_text(encoding='utf-8')
    for old, new in [('addition_only_return_gate_batch', 'addition_gate_exposure_batch'),
        ('ADD_GATE_BAND00_LOWER01','ADD_GATE_EXPOSURE_110'), ('ADD_GATE_BAND20_UPPER00','ADD_GATE_EXPOSURE_120'),
        ('ADD_GATE_BAND00_LOWER00','ADD_GATE_EXPOSURE_105'), ('len(SETTINGS) == 8 and len(summaries) == 16','len(SETTINGS) == 4 and len(summaries) == 8'),
        ("frames['BASE'][model+'_target'].iloc[:-1], .5)", "frames['BASE'][model+'_target'].iloc[:-1], .5*setting['exposure_multiplier'])"),
        ('test_eight_mappings_cost_identity_and_market_factor_selection', 'test_four_mappings_cost_identity_and_market_factor_selection')]:
        test = test.replace(old, new)
    test += '''

def test_multiplier_caps_full_position_and_preserves_zero_unknown_and_cost_path():
    data, cfg, parents = fixture()[0], settings(), {}
    raw = np.array([.5, .96, 1., 0., np.nan, .2, .8, .9, .3])
    for cost in cfg['costs']:
        values = raw if cost == 'BASE' else raw*.8
        parents[cost] = {m: pd.DataFrame({'origin': data.date.iloc[:-1].to_numpy(),
            'execution_date': data.date.iloc[1:].to_numpy(), 'origin_index': np.arange(len(data)-1),
            'reference_weight': values, 'source_cost': cost, 'source_model': m}) for m in MODELS}
    frames, _ = gate_frames(data, parents, cfg, str(data.date.iloc[1].date()))
    for cost, frame in frames.items():
        for model, setting in SETTINGS.items():
            expected = np.minimum(1., (raw if cost == 'BASE' else raw*.8)*setting['exposure_multiplier'])
            np.testing.assert_allclose(frame[model+'_target'].iloc[:-1], expected, equal_nan=True)
            assert frame[model+'_target'].iloc[3] == 0 and np.isnan(frame[model+'_target'].iloc[4])
            assert frame[model+'_target'].dropna().le(1).all()
'''
    save('tests/test_addition_gate_exposure_batch_v1.py', test)

    main = (ROOT/'research/addition_only_return_gate_batch_v1.py').read_text(encoding='utf-8')
    for old, new in [('addition_only_return_gate_batch','addition_gate_exposure_batch'),
        ('ADDITION_ONLY_RETURN_GATE_BATCH','ADDITION_GATE_EXPOSURE_BATCH'),
        ('round=205','round=206'), ("'round': 205", "'round': 206"), ("['round'] == 204", "['round'] == 205"),
        ('第205轮八套仅加仓过滤','第206轮四套投入倍率'), ('candidate_configurations=8','candidate_configurations=4'),
        ('八套仅加仓过滤','四套固定加仓条件投入倍率'), ('八套实际冻结设置','四套实际冻结设置'),
        ('三十二条账户','十六条账户'), ('三十二账户','十六账户'), ('len(accounts) == 32 and count == 45168','len(accounts) == 16 and count == 22584'),
        ("'actual_accounts': 32", "'actual_accounts': 16"),
        ('PASS_THIRTY_TWO_ADDITION_ONLY_ACCOUNTS_AND_UNFILTERED_INITIAL_ENTRIES','PASS_SIXTEEN_CAPPED_EXPOSURE_ACCOUNTS_AND_UNFILTERED_INITIAL_ENTRIES'),
        ('eleven_setting_joint_comparison.csv','eight_setting_joint_comparison.csv'),
        ('RETROSPECTIVE_ENTRY_PRESERVED_ADDITION_ONLY_RETURN_FILTERS','RETROSPECTIVE_FIXED_ADDITION_GATE_CAPPED_EXPOSURE_MULTIPLIERS'),
        ('PROGRESS_ROUND203_204_COMPLETED_32_ACCOUNTS_AND_ENTRY_ADDITION_SEPARATION_EVIDENCE','PROGRESS_ROUND205_COMPLETED_32_ACCOUNTS_AND_JOINT_GAP_IMPROVED'),
        ("'5 passed'", "'6 passed'"), ("'passed': 5", "'passed': 6"), ('五项','六项'),
        ('|已有持仓加仓的当日收益条件|', '|已有持仓加仓的当日收益条件|目标倍率|'),
        ('|---|---:|---:|','|---|---:|---:|---:|'),
        ("{s['return_threshold']:.0%}|\\n", "{s['return_threshold']:.0%}|{s['exposure_multiplier']:.2f}|\\n"),
        ('本批只选择零与二十个百分点门槛，仅增加上述已有持仓加仓过滤，初次入场完整沿用原规则；不加入第202轮亏损保护。',
         '本批固定二十个百分点门槛及至少1%的加仓准入，只对原始目标乘候选倍率并限制100%；空仓按新目标正常进入，退出结构保持，不加入第202轮亏损保护。')]:
        main = main.replace(old, new)
    main = replace_once(main, "CONTROLS = {MODELS[0]:", "CONTROLS = {'ADD_GATE_BAND20_LOWER01': (ROOT/'reports/research/510300_addition_only_return_gate_batch_v1', '原205仅强势加仓'), MODELS[0]:")
    main = replace_once(main, "for model in [*CANDIDATES, *MODELS, 'TWO_CLOSE_ZERO_EXIT']:",
        "for model in [*CANDIDATES, 'ADD_GATE_BAND20_LOWER01', *MODELS, 'TWO_CLOSE_ZERO_EXIT']:")
    main = replace_once(main, "raw = originals[setting['parent']]", "raw = np.minimum(1., originals[setting['parent']]*setting['exposure_multiplier'])")
    main = replace_once(main, "prior, shares = np.r_[cfg['initial_capital'], ledger.equity.iloc[:-1]], np.r_[0, ledger.shares.iloc[:-1]].astype(int)",
        "require(decisions.loc[known, 'exposure_multiplier'].eq(setting['exposure_multiplier']).all(), '投入倍率不同')\n        prior, shares = np.r_[cfg['initial_capital'], ledger.equity.iloc[:-1]], np.r_[0, ledger.shares.iloc[:-1]].astype(int)")
    save('research/addition_gate_exposure_batch_v1.py', main)
    print('第206轮四套倍率的独立入口与六项必要测试已生成，尚未冻结或运行。', flush=True)


if __name__ == '__main__':
    main()
