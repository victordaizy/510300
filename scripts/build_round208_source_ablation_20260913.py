"""生成固定加仓条件下的来源删减批次，来源计划独立重建。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def change(text, old, new):
    if text.count(old) != 1:
        raise ValueError('待替换结构不唯一：'+old[:100])
    return text.replace(old, new)


def save(path, text):
    with (ROOT/path).open('x', encoding='utf-8') as stream:
        stream.write(text)


def main():
    if (ROOT/'research/gated_source_ablation_batch_v1.py').exists():
        raise ValueError('第208轮已存在，不重复生成')
    source = (ROOT/'research/addition_gate_exposure_batch_inputs_v1.py').read_text(encoding='utf-8')
    source = source.replace('import numpy as np', 'from pathlib import Path\nimport numpy as np\nimport pandas as pd\nfrom research.three_source_order_intent_mix_inputs_v1 import SOURCES, source_plan')
    left, right = source.index("MODELS = ['INTENT_MIX_BAND_00'"), source.index('\n\ndef market_factors')
    source = source[:left]+'''ROOT = Path(__file__).resolve().parents[1]
MODELS = list(SOURCES)
MIXES = {'NO_EPISODE': ([16/17, 0., 1/17], '去掉区间固定预算'),
         'NO_REBOUND': ([16/19, 3/19, 0.], '去掉反弹辅助'),
         'CORE_ONLY': ([1., 0., 0.], '只保留两次零确认核心')}
SETTINGS, CANDIDATES = {}, {}
for percent in [100, 105]:
    for mix, (weights, label) in MIXES.items():
        model = f'ABLATE_{percent}_{mix}'
        SETTINGS[model] = {'parent': mix, 'band': .2, 'direction': 'LOWER', 'return_threshold': .01,
            'exposure_multiplier': percent/100, 'source_weights': dict(zip(MODELS, weights))}
        CANDIDATES[model] = f'目标{percent/100:.2f}倍、{label}'
PRIMARY = 'ABLATE_105_NO_EPISODE'
'''+source[right:]
    left, right = source.index('def gate_frames'), source.index('\n\ndef gate_request')
    source = source[:left]+'''def gate_frames(data, parents_by_cost, cfg, start, ledger_loader=None):
    require(cfg['candidate_models'] == list(CANDIDATES) and cfg['candidate_settings'] == SETTINGS
        and cfg['source_folders'] == SOURCES, '来源删减设置不同')
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    indices = np.arange(first-1, len(data)-1)
    period = 'evaluation' if pd.Timestamp(start) == pd.Timestamp(cfg['evaluation_start']) else 'earlier_diagnostic'
    if ledger_loader is None:
        def ledger_loader(source_period, cost, model):
            return pd.read_parquet(ROOT/'reports/research'/SOURCES[model]/source_period/cost/f'{model}_ledger.parquet')
    market, summaries = market_factors(data), []
    for cost, frame in frames.items():
        for column in market:
            frame[column] = market[column].to_numpy()
        for parent in MODELS:
            plan = source_plan(data, parents_by_cost[cost][parent], ledger_loader(period, cost, parent), cfg, first)
            for column in plan:
                values = np.full(len(data), np.nan)
                values[indices] = plan[column]
                frame[parent+'_'+column] = values
        for model, setting in SETTINGS.items():
            target = np.zeros(len(data))
            for parent, weight in setting['source_weights'].items():
                if weight > 0:
                    target += weight*frame[parent+'_planned_weight'].to_numpy(float)
            target = np.minimum(1., target*setting['exposure_multiplier'])
            frame[model+'_target'] = target
            active = target[indices]
            summaries.append({'model': model, 'cost': cost, 'decision_origins': len(indices),
                'positive_source_origins': int((active > 0).sum()), 'zero_source_origins': int((active == 0).sum()),
                'unknown_source_origins': int(np.isnan(active).sum()), 'source_mix': setting['parent'],
                'exposure_multiplier': setting['exposure_multiplier']})
    return frames, summaries
'''+source[right:]
    source = source.replace('固定加仓条件，在100%以内小幅放大目标投入，并重新计算完整账户。', '从收盘计划份额删去辅助来源，保留已有加仓准入及正常减仓退出。')
    save('research/gated_source_ablation_batch_inputs_v1.py', source)

    main = (ROOT/'research/addition_gate_exposure_batch_v1.py').read_text(encoding='utf-8')
    for old, new in [('addition_gate_exposure_batch', 'gated_source_ablation_batch'),
        ('ADDITION_GATE_EXPOSURE_BATCH', 'GATED_SOURCE_ABLATION_BATCH'),
        ('round=206', 'round=208'), ("'round': 206", "'round': 208"), ("['round'] == 205", "['round'] == 207"),
        ('第206轮四套投入倍率', '第208轮六套来源删减'), ('candidate_configurations=4', 'candidate_configurations=6'),
        ('四套固定加仓条件投入倍率', '六套固定加仓条件来源删减'), ('四套实际冻结设置', '六套实际冻结设置'),
        ('十六条账户', '二十四条账户'), ('十六账户', '二十四账户'),
        ('len(accounts) == 16 and count == 22584', 'len(accounts) == 24 and count == 33876'),
        ("'actual_accounts': 16", "'actual_accounts': 24"),
        ('PASS_SIXTEEN_CAPPED_EXPOSURE_ACCOUNTS_AND_UNFILTERED_INITIAL_ENTRIES', 'PASS_TWENTY_FOUR_SOURCE_ABLATION_ACCOUNTS_AND_INDEPENDENT_PLANS'),
        ('eight_setting_joint_comparison.csv', 'eleven_setting_joint_comparison.csv'),
        ('RETROSPECTIVE_FIXED_ADDITION_GATE_CAPPED_EXPOSURE_MULTIPLIERS', 'RETROSPECTIVE_GATED_SOURCE_ABLATION_WITH_PLANNED_EXPOSURE'),
        ('PROGRESS_ROUND205_COMPLETED_32_ACCOUNTS_AND_JOINT_GAP_IMPROVED', 'PROGRESS_ROUND207_COMPLETED_32_ACCOUNTS_AND_REDUCTION_BRANCH_REJECTED'),
        ('CANDIDATES, MODELS, PRIMARY, SETTINGS, allowance_column', 'CANDIDATES, MODELS, PRIMARY, SETTINGS, SOURCES, allowance_column'),
        ('本批固定二十个百分点门槛及至少1%的加仓准入，只对原始目标乘候选倍率并限制100%；空仓按新目标正常进入，退出结构保持，不加入第202轮亏损保护。',
         '本批固定二十个百分点门槛及至少1%的加仓准入，三来源按上述保留权重重建计划目标，再乘1.00或1.05并限制100%；普通减仓恢复原规则，明确零及终点全部退出保持。')]:
        main = main.replace(old, new)
    left, right = main.index('PARENTS = '), main.index('\n\n\ndef prepare')
    main = main[:left]+'''PARENTS = {m: ROOT/'reports/research'/folder for m, folder in SOURCES.items()}
CONTROLS = {
    'ADD_GATE_EXPOSURE_105': (ROOT/'reports/research/510300_addition_gate_exposure_batch_v1', '原206目标1.05倍'),
    'ADD_GATE_BAND20_LOWER01': (ROOT/'reports/research/510300_addition_only_return_gate_batch_v1', '原205仅强势加仓'),
    'INTENT_MIX_BAND_00': (SOURCE, '原201零门槛合并'),
    'INTENT_MIX_BAND_20': (SOURCE, '原201二十个百分点合并'),
    'TWO_CLOSE_ZERO_EXIT': (ROOT/'reports/research/510300_two_close_zero_exit_v1', '原198两次归零确认'),
    'BUY_HOLD': (ROOT/'reports/research/510300_rearmed_session_exit_v1', '买入持有')}
'''+main[right:]
    main = change(main, 'candidate_settings=SETTINGS, parent_models=MODELS,', 'candidate_settings=SETTINGS, parent_models=MODELS, source_folders=SOURCES,')
    main = change(main, "for model in [*CANDIDATES, 'ADD_GATE_BAND20_LOWER01', *MODELS, 'TWO_CLOSE_ZERO_EXIT']:",
        "for model in [*CANDIDATES, *[m for m in CONTROLS if m != 'BUY_HOLD']]:")
    main = change(main, "ROOT/'research/saved_parent_target_alignment_v1.py', ROOT/'research/adaptive_allocation_v1.py',",
        "ROOT/'research/saved_parent_target_alignment_v1.py', ROOT/'research/adaptive_allocation_v1.py',\n        ROOT/'research/three_source_order_intent_mix_inputs_v1.py', ROOT/'tests/test_three_source_order_intent_mix_v1.py',")
    main = change(main, "paths += [folder/p/c/f'{model}_decisions.parquet' for p in ['evaluation', 'earlier_diagnostic'] for c in cfg['costs']]",
        "paths += [folder/p/c/f'{model}_{kind}.parquet' for p in ['evaluation', 'earlier_diagnostic'] for c in cfg['costs'] for kind in ['decisions', 'ledger']]\n        paths += [folder/'saved_verification_receipt.json', ROOT/'config'/f'{SOURCES[model]}.json']")
    main = change(main, "originals[parent] = source.reference_weight.to_numpy(float)", '''source_ledger = pd.read_parquet(old/period/cost/f'{parent}_ledger.parquet')
                require(pd.DatetimeIndex(source_ledger.date).equals(pd.DatetimeIndex(frame.date.iloc[indices+1])), '来源账本日期不同')
                source_equity = np.r_[cfg['initial_capital'], source_ledger.equity.iloc[:-1]].astype(float)
                source_shares = np.r_[0, source_ledger.shares.iloc[:-1]].astype(float)
                planned = source_shares+source.requested_quantity.to_numpy(float)
                source_price = frame.close.iloc[indices].to_numpy(float)
                require((planned >= 0).all() and (planned % cfg['lot'] == 0).all(), '来源计划份额无效')
                weights = planned*source_price/source_equity
                require((weights >= -1e-12).all() and (weights <= 1+1e-12).all(), '来源计划比例越界')
                weights = np.clip(weights, 0., 1.)
                weights[source.reference_weight.isna().to_numpy()] = np.nan
                originals[parent] = weights''')
    main = change(main, "factors = pd.read_parquet(folder/'factors.parquet')", "factors = pd.read_parquet(folder/'factors.parquet')\n            for parent, weights in originals.items():\n                np.testing.assert_allclose(factors.loc[indices, parent+'_planned_weight'], weights, atol=0, rtol=0, equal_nan=True)")
    main = change(main, "raw = np.minimum(1., originals[setting['parent']]*setting['exposure_multiplier'])", "raw = np.zeros(len(indices))\n        for parent, weight in setting['source_weights'].items():\n            if weight > 0:\n                raw += weight*originals[parent]\n        raw = np.minimum(1., raw*setting['exposure_multiplier'])")
    save('research/gated_source_ablation_batch_v1.py', main)
    print('第208轮六套来源删减入口已生成，尚未冻结或计算。', flush=True)


if __name__ == '__main__':
    main()
