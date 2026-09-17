"""把新增量组合的固定权重转成独立计划合并入口，不重算来源账户。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def change(text, old, new):
    if text.count(old) != 1:
        raise ValueError('替换结构数量不同：'+old[:120])
    return text.replace(old, new)


def save(path, text):
    with (ROOT/path).open('x', encoding='utf-8') as stream:
        stream.write(text)


def main():
    if (ROOT/'research/incremental_selected_intent_mix_v1.py').exists():
        raise ValueError('第209轮入口已存在')
    source = (ROOT/'research/gated_source_ablation_batch_inputs_v1.py').read_text(encoding='utf-8')
    source = source.replace('from pathlib import Path', 'import json\nfrom pathlib import Path').replace('import SOURCES, source_plan', 'import source_plan')
    left, right = source.index('ROOT = Path'), source.index('\n\ndef market_factors')
    source = source[:left]+'''ROOT = Path(__file__).resolve().parents[1]
MIX_PATH = ROOT/'reports/research/510300_incremental_saved_mix_through208/result.json'
SOURCES = {'ADD_GATE_EXPOSURE_115': '510300_addition_gate_exposure_batch_v1',
           'ADD_GATE_EXPOSURE_110': '510300_addition_gate_exposure_batch_v1',
           'RUNS_OPPORTUNITY_CAPPED_SUM': '510300_runs_opportunity_union_v1',
           'REDUCE_105_HOLD': '510300_positive_target_reduction_batch_v1',
           'MEAN_REBOUND_AUX_50': '510300_mean_rebound_auxiliary_v1'}
MODELS = list(SOURCES)
saved = json.loads(MIX_PATH.read_text(encoding='utf-8'))
require([r['model'] for r in saved['best_weights']] == MODELS, '已保存五来源身份不同')
weights = np.array([r['weight'] for r in saved['best_weights']])
weights = weights/weights.sum()
top = np.r_[weights[:3]/weights[:3].sum(), 0., 0.]
MIXES = {'FULL': (weights, '已保存五来源'), 'TOP3': (top, '前三来源归一化'),
         'SIMPLE3': (np.array([.70,.15,.15,0.,0.]), '七成、一成半、一成半'),
         'SIMPLE2': (np.array([.85,0.,.15,0.,0.]), '八成半、一成半两来源')}
SETTINGS, CANDIDATES = {}, {}
for band in [0,10]:
    for mix, (vector,label) in MIXES.items():
        model = f'SELECTED_MIX_BAND{band:02d}_{mix}'
        SETTINGS[model] = {'parent': mix, 'band': band/100, 'direction': 'LOWER', 'return_threshold': .01,
            'exposure_multiplier': 1., 'source_weights': dict(zip(MODELS, vector.tolist()))}
        CANDIDATES[model] = f'{band}个百分点调仓、{label}'
PRIMARY = 'SELECTED_MIX_BAND00_SIMPLE3'
'''+source[right:]
    source = change(source, "allowed = market[allowance_column(SETTINGS[model])].to_numpy(bool)", "allowed = np.ones(len(data), dtype=bool)")
    source = source.replace('从收盘计划份额删去辅助来源，保留已有加仓准入及正常减仓退出。', '合并新筛选来源的收盘计划，外层只按目标调仓，不再次叠加来源过滤。')
    save('research/incremental_selected_intent_mix_inputs_v1.py', source)

    main = (ROOT/'research/gated_source_ablation_batch_v1.py').read_text(encoding='utf-8')
    for old, new in [('gated_source_ablation_batch','incremental_selected_intent_mix'),
        ('GATED_SOURCE_ABLATION_BATCH','INCREMENTAL_SELECTED_INTENT_MIX'),
        ('round=208','round=209'), ("'round': 208", "'round': 209"), ("['round'] == 207", "['round'] == 208"),
        ('第208轮六套来源删减','第209轮八套筛选计划合并'), ('candidate_configurations=6','candidate_configurations=8'),
        ('六套固定加仓条件来源删减','八套筛选计划合并'), ('六套实际冻结设置','八套实际冻结设置'),
        ('二十四条账户','三十二条账户'), ('二十四账户','三十二账户'),
        ('len(accounts) == 24 and count == 33876','len(accounts) == 32 and count == 45168'),
        ("'actual_accounts': 24", "'actual_accounts': 32"),
        ('PASS_TWENTY_FOUR_SOURCE_ABLATION_ACCOUNTS_AND_INDEPENDENT_PLANS','PASS_THIRTY_TWO_SELECTED_INTENT_ACCOUNTS_AND_NO_DUPLICATE_GATES'),
        ('eleven_setting_joint_comparison.csv','thirteen_setting_joint_comparison.csv'),
        ('RETROSPECTIVE_GATED_SOURCE_ABLATION_WITH_PLANNED_EXPOSURE','RETROSPECTIVE_INCREMENTAL_SELECTED_PLANNED_INTENT_MIX'),
        ('PROGRESS_ROUND207_COMPLETED_32_ACCOUNTS_AND_REDUCTION_BRANCH_REJECTED','PROGRESS_ROUND207_208_COMPLETED_56_ACCOUNTS_AND_NEW_VIRTUAL_MIX_POINT_PASS'),
        ('CANDIDATES, MODELS, PRIMARY, SETTINGS, SOURCES, allowance_column','CANDIDATES, MIX_PATH, MODELS, PRIMARY, SETTINGS, SOURCES, allowance_column')]:
        main = main.replace(old,new)
    left, right = main.index("    text += '\\n\\n## 八套实际冻结设置"), main.index("    with (ROOT/cfg['rules']).open")
    main = main[:left]+'''    text += '\\n\\n## 八套实际冻结权重与外层门槛\\n\\n|方案|外层门槛|1.15倍来源|1.10倍来源|168相加封顶|207暂不减仓|195反弹辅助|\\n|---|---:|---:|---:|---:|---:|---:|\\n'
    for model,s in SETTINGS.items():
        text += '|'+CANDIDATES[model]+f"|{s['band']:.0%}|"+'|'.join(f"{s['source_weights'][m]:.10%}" for m in MODELS)+'|\\n'
    text += '\\n\\n## 来源因素与内部进出场完整说明\\n\\n以下为来源的原冻结说明；209的外层权重及调仓规则以上文为准，不重复叠加来源的加仓过滤或减仓限制。\\n\\n'
    appended_rules = [ROOT/'docs/510300_ADDITION_GATE_EXPOSURE_BATCH_V1.md',
        ROOT/'docs/510300_RUNS_OPPORTUNITY_UNION_V1.md', ROOT/'docs/510300_POSITIVE_TARGET_REDUCTION_BATCH_NEXT_20260913.md']
    for rule in appended_rules:
        text += '\\n\\n'+rule.read_text(encoding='utf-8')
'''+main[right:]
    main = change(main, "paths = [Path(__file__), ROOT/'research/incremental_selected_intent_mix_inputs_v1.py',", "paths = [Path(__file__), MIX_PATH, *appended_rules, ROOT/'research/incremental_selected_intent_mix_inputs_v1.py',")
    main = change(main, "allowed = market[allowance_column(setting)].iloc[indices].to_numpy(bool)", "allowed = np.ones(len(indices), dtype=bool)")
    save('research/incremental_selected_intent_mix_v1.py', main)
    print('第209轮八套实际计划合并入口已生成，尚未冻结或运行。', flush=True)


if __name__ == '__main__':
    main()
