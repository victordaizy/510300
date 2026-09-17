"""生成独立的正目标减仓批次，复用账户组织结构并独立核对新申请。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def change(text, old, new):
    if text.count(old) != 1:
        raise ValueError('替换结构数量不唯一：'+old[:100])
    return text.replace(old, new)


def save(path, text):
    with (ROOT/path).open('x', encoding='utf-8') as stream:
        stream.write(text)


def main():
    if (ROOT/'research/positive_target_reduction_batch_v1.py').exists():
        raise ValueError('第207轮入口已存在，不重复生成')
    inputs = (ROOT/'research/addition_gate_exposure_batch_inputs_v1.py').read_text(encoding='utf-8')
    left, right = inputs.index('SETTINGS, CANDIDATES = {}, {}'), inputs.index('\n\ndef market_factors')
    inputs = inputs[:left]+'''SETTINGS, CANDIDATES = {}, {}
for percent in [100, 105]:
    for mode, label in [('UPPER00', '非正日普通减仓'), ('LOWER00', '非负日普通减仓'),
                        ('HOLD', '暂不普通减仓'), ('HALF', '普通减仓一半')]:
        model = f'REDUCE_{percent}_{mode}'
        SETTINGS[model] = {'parent': 'INTENT_MIX_BAND_20', 'band': .2, 'direction': 'LOWER',
            'return_threshold': .01, 'exposure_multiplier': percent/100, 'reduction_mode': mode}
        CANDIDATES[model] = f'目标{percent/100:.2f}倍、{label}'
PRIMARY = 'REDUCE_105_HALF'
'''+inputs[right:]
    inputs = change(inputs, 'def gate_request(account, price, raw, cfg, model, allowed, daily_return):', '''def reduction_request(quantity, raw, lot, mode, upper_zero, lower_zero):
    if not (raw > 0 and quantity < 0):
        return quantity
    if mode == 'HALF':
        return -int((-quantity)//(2*lot))*lot
    permitted = upper_zero if mode == 'UPPER00' else lower_zero if mode == 'LOWER00' else False
    return quantity if permitted else 0


def gate_request(account, price, raw, cfg, model, allowed, daily_return, upper_zero=False, lower_zero=False):''')
    inputs = change(inputs, "result.update(output_candidate=model, budget_source=setting['parent'], used_band=float(setting['band']),", '''adjusted = reduction_request(result['requested_quantity'], raw, cfg['lot'], setting['reduction_mode'], upper_zero, lower_zero)
    ordinary = raw > 0 and quantity < 0
    modified = ordinary and adjusted != quantity
    result['requested_quantity'] = adjusted
    if modified:
        result['action'] = '原目标仍为正，按本候选条件暂缓或减半普通减仓'
    result.update(reduction_mode=setting['reduction_mode'], ordinary_reduction=float(ordinary),
        reduction_modified=float(modified), reduction_upper_zero=float(upper_zero), reduction_lower_zero=float(lower_zero),
        output_candidate=model, budget_source=setting['parent'], used_band=float(setting['band']),''')
    inputs = change(inputs, "returns = market.daily_total_simple.to_numpy(float)", "returns = market.daily_total_simple.to_numpy(float)\n    upper = market.buy_allowed_upper_r00.to_numpy(bool)\n    lower = market.buy_allowed_lower_r00.to_numpy(bool)")
    inputs = change(inputs, 'bool(allowed[origin_index]), returns[origin_index])', 'bool(allowed[origin_index]), returns[origin_index], bool(upper[origin_index]), bool(lower[origin_index]))')
    inputs = inputs.replace('固定加仓条件，在100%以内小幅放大目标投入，并重新计算完整账户。', '正目标普通减仓使用条件或半量，明确零退出及原加仓条件保持。')
    save('research/positive_target_reduction_batch_inputs_v1.py', inputs)

    main = (ROOT/'research/addition_gate_exposure_batch_v1.py').read_text(encoding='utf-8')
    for old, new in [('addition_gate_exposure_batch','positive_target_reduction_batch'),
        ('ADDITION_GATE_EXPOSURE_BATCH','POSITIVE_TARGET_REDUCTION_BATCH'),
        ('round=206','round=207'), ("'round': 206", "'round': 207"), ("['round'] == 205", "['round'] == 206"),
        ('第206轮四套投入倍率','第207轮八套正目标减仓'), ('candidate_configurations=4','candidate_configurations=8'),
        ('四套固定加仓条件投入倍率','八套正目标减仓'), ('四套实际冻结设置','八套实际冻结设置'),
        ('十六条账户','三十二条账户'), ('十六账户','三十二账户'),
        ('len(accounts) == 16 and count == 22584','len(accounts) == 32 and count == 45168'),
        ("'actual_accounts': 16", "'actual_accounts': 32"),
        ('PASS_SIXTEEN_CAPPED_EXPOSURE_ACCOUNTS_AND_UNFILTERED_INITIAL_ENTRIES','PASS_THIRTY_TWO_POSITIVE_TARGET_REDUCTION_ACCOUNTS_AND_ZERO_EXIT_PRIORITY'),
        ('eight_setting_joint_comparison.csv','thirteen_setting_joint_comparison.csv'),
        ('RETROSPECTIVE_FIXED_ADDITION_GATE_CAPPED_EXPOSURE_MULTIPLIERS','RETROSPECTIVE_POSITIVE_TARGET_REDUCTION_WITH_ZERO_EXIT_PRESERVED'),
        ('PROGRESS_ROUND205_COMPLETED_32_ACCOUNTS_AND_JOINT_GAP_IMPROVED','PROGRESS_ROUND205_206_COMPLETED_48_ACCOUNTS_AND_REDUCTION_REQUEST_INVENTORY'),
        ("'6 passed'", "'7 passed'"), ("'passed': 6", "'passed': 7"), ('六项','七项'),
        ('|目标倍率|', '|目标倍率|普通减仓方式|'), ('|---|---:|---:|---:|','|---|---:|---:|---:|---|'),
        ("{s['exposure_multiplier']:.2f}|\\n", "{s['exposure_multiplier']:.2f}|{CANDIDATES[model].split('、')[-1]}|\\n"),
        ('本批固定二十个百分点门槛及至少1%的加仓准入，只对原始目标乘候选倍率并限制100%；空仓按新目标正常进入，退出结构保持，不加入第202轮亏损保护。',
         '本批固定二十个百分点门槛及至少1%的加仓准入，目标倍率1.00或1.05，上限100%；普通正目标减仓使用上述方式，明确零及终点全部退出优先。')]:
        main = main.replace(old, new)
    main = change(main, "CONTROLS = {'ADD_GATE_BAND20_LOWER01':", "CONTROLS = {'ADD_GATE_EXPOSURE_105': (ROOT/'reports/research/510300_addition_gate_exposure_batch_v1', '原206目标1.05倍'), 'ADD_GATE_BAND20_LOWER01':")
    main = change(main, "for model in [*CANDIDATES, 'ADD_GATE_BAND20_LOWER01', *MODELS, 'TWO_CLOSE_ZERO_EXIT']:",
        "for model in [*CANDIDATES, 'ADD_GATE_EXPOSURE_105', 'ADD_GATE_BAND20_LOWER01', *MODELS, 'TWO_CLOSE_ZERO_EXIT']:")
    main = change(main, "blocked = (normal > 0) & (shares > 0) & ~allowed", "blocked = (normal > 0) & (shares > 0) & ~allowed\n        ordinary = (raw > 0) & (normal < 0)\n        upper = market.buy_allowed_upper_r00.iloc[indices].to_numpy(bool)\n        lower = market.buy_allowed_lower_r00.iloc[indices].to_numpy(bool)\n        reduced = independent_reduction(normal, raw, cfg['lot'], setting['reduction_mode'], upper, lower)\n        require(decisions.loc[known, 'reduction_mode'].eq(setting['reduction_mode']).all(), '减仓身份不同')\n        np.testing.assert_array_equal(decisions.loc[known, 'ordinary_reduction'], ordinary[known].astype(float))\n        np.testing.assert_array_equal(decisions.loc[known, 'reduction_modified'], (ordinary & (reduced != normal))[known].astype(float))\n        np.testing.assert_array_equal(decisions.loc[known, 'reduction_upper_zero'], upper[known].astype(float))\n        np.testing.assert_array_equal(decisions.loc[known, 'reduction_lower_zero'], lower[known].astype(float))")
    main = change(main, "checks.append({'model': model, 'period': period, 'cost': cost, 'suppressed_buy_requests': int(blocked.sum()),",
        "checks.append({'model': model, 'period': period, 'cost': cost, 'ordinary_reduction_requests': int(ordinary.sum()),\n            'modified_reduction_requests': int((ordinary & (reduced != normal)).sum()),\n            'zero_exit_requests': int(((raw == 0) & (normal < 0)).sum()), 'suppressed_buy_requests': int(blocked.sum()),")
    main = change(main, 'return targets, allowed', 'return targets, allowed, upper, lower')
    main = change(main, 'targets, allowed = expected(period, cost, frame, start, model)', 'targets, allowed, upper, lower = expected(period, cost, frame, start, model)')
    main = change(main, "active['allowed'] = allowed", "active.update(allowed=allowed, upper=upper, lower=lower)")
    main = change(main, "requests[(requests > 0) & (shares > 0) & ~active['allowed']] = 0\n            return requests", "requests[(requests > 0) & (shares > 0) & ~active['allowed']] = 0\n            return independent_reduction(requests, targets, settings['lot'], SETTINGS[model]['reduction_mode'], active['upper'], active['lower'])")
    main = change(main, 'def verify():', '''def independent_reduction(requests, targets, lot, mode, upper, lower):
    output = requests.copy()
    rows = (targets > 0) & (requests < 0)
    if mode == 'HALF':
        output[rows] = -np.floor(np.abs(requests[rows])/lot/2).astype(int)*lot
    elif mode == 'HOLD':
        output[rows] = 0
    elif mode == 'UPPER00':
        output[rows & ~upper] = 0
    elif mode == 'LOWER00':
        output[rows & ~lower] = 0
    else:
        raise ValueError('未知减仓方式')
    return output


def verify():''')
    main = change(main, "'simulated_fills_checked': sum(r['fills'] for r in checks), 'suppressed_buy_requests_checked': len(suppressions),",
        "'simulated_fills_checked': sum(r['fills'] for r in checks), 'suppressed_buy_requests_checked': len(suppressions),\n        'modified_reduction_requests_checked': sum(r['modified_reduction_requests'] for r in checks),\n        'zero_exit_requests_checked': sum(r['zero_exit_requests'] for r in checks),")
    save('research/positive_target_reduction_batch_v1.py', main)
    print('第207轮八套正目标减仓入口已生成，尚未冻结或计算。', flush=True)


if __name__ == '__main__':
    main()
