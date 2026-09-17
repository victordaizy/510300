"""集中交付十四套新参数和四套旧参数，保留完整失败并准备成本调整。"""
import json

import pandas as pd

from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json
from research.risk_window_clock_batch_v1 import CANDIDATES, CONFIG, OUT, ROOT


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    require(not any(v['four_scenario_joint_pass'] for v in joint['candidates'].values()), '本批出现联合点值通过，需另行报告')
    rows = []
    settings = [(model, result, False) for model in CANDIDATES]
    settings += [(model, result, True) for model in ['ACCOUNT_VOLATILITY_EXPOSURE', 'EPISODE_ACCOUNT_RISK_BUDGET']]
    old = json.loads((ROOT/'reports/research/510300_finite_account_risk_calibration_v1/result.json').read_text(encoding='utf-8'))
    settings += [(model, old, True) for model in ['ACCOUNT_RISK_12', 'ACCOUNT_RISK_15']]
    for model, source, reused in settings:
        row = {'model': model, 'reused_old_setting': reused}
        ratios = []
        for period, key in [('main', 'all_metrics'), ('earlier', 'earlier_diagnostics')]:
            for cost in ['BASE', 'STRESS']:
                m = next(m for m in source[key] if m['model'] == model and m['cost'] == cost)
                row['name'] = m['name']
                row.update({period+'_'+cost+'_'+k: m[k] for k in ['net_sharpe', 'annualized_return', 'max_drawdown']})
                ratios += [m['net_sharpe']/1.2, m['annualized_return']/.1]
        row['minimum_joint_ratio'] = min(ratios)
        rows.append(row)
    ranked = pd.DataFrame(rows).sort_values(['minimum_joint_ratio', 'model'], ascending=[False, True])
    ranked.to_csv(OUT/'all_eighteen_setting_comparison.csv', index=False, encoding='utf-8-sig')
    require(ranked.iloc[0].model == 'ACCOUNT_VOLATILITY_EXPOSURE', '旧比较候选排名发生变化')
    decision = ('第196轮十四套未运行过的风险窗口与预算时点设置已全部完成，共五十六条新账户，四套旧设置直接复用。'
        '没有一套新设置达到四个场景的净夏普1.2与年化10%联合目标，完整十八套中原181仍排在联合比较首位。'
        '结束本次有限网格，不追加风险窗口或预算。下一项只比较两种部分调仓规则，检查成本节约是否足以补偿收益变化。')
    detail = [
        f"五项必要测试通过。五十六条新账户核心计算{result['run_seconds']:.2f}秒；耗时不含开发、测试、核对与交付。所有候选一次构建因素和批量运行，零新增预测模型、零新外部来源。",
        f"保存结果集中核对了{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整周期和{checked['simulated_fills_checked']}次模拟成交，已核对来源时点、窗口、固定倍率、实际份额、分红与成本。",
        '本轮主候选“六十日风险、12%预算、区间固定”：主历史基础／压力夏普1.194／1.122、年化8.33%／7.79%；较早夏普1.154／1.151、年化9.50%／9.57%。没有获得所需提升。',
        '新设置按八项指标相对各自门槛的最弱比值排序，排在首位的是“一百二十日风险、12%预算、每日调整”：主夏普1.281／1.197、年化9.01%／8.37%；较早夏普0.996／0.991、年化9.75%／9.78%。这是事后排序，不是独立验证；它仍弱于原181。',
        '较早历史最高夏普的新设置是“一百二十日风险、10%预算、区间固定”：较早夏普1.179／1.176、年化10.57%／10.66%，但主历史年化只有8.10%／7.54%。不能挑某段历史单独宣布达标。',
        '',
        '## 十八套设置的统一比较', '',
        '|设置|本轮计算|主基础夏普／年化|主压力夏普／年化|较早基础夏普／年化|较早压力夏普／年化|最弱比值|',
        '|---|---|---|---|---|---|---:|',
    ]
    for row in ranked.to_dict('records'):
        cells = [f"{row[p+'_'+c+'_net_sharpe']:.3f}／{row[p+'_'+c+'_annualized_return']:.2%}"
                 for p, c in [('main', 'BASE'), ('main', 'STRESS'), ('earlier', 'BASE'), ('earlier', 'STRESS')]]
        detail.append('|'+row['name']+'|'+('复用旧结果' if row['reused_old_setting'] else '新计算')+'|'+'|'.join(cells)+f"|{row['minimum_joint_ratio']:.3f}|")
    detail += ['', '## 本次研究提速已落实', '',
        '先用32.29秒批量筛查224条不同策略的保存收益，再用15.32秒一次计算56条新完整账户。前者是虚拟收益诊断，后者才是包含整手、现金、分红、成本与进出场的模拟账户；两个时间都是对应计算阶段耗时。',
        '复用现有输入和账户程序，将十四个小参数合并成一轮集中测试、核对和交付。全部规则仍用中文，保留失败记录，无新增GPT审阅数值包。',
        '下一项保留181原信号和十个百分点偏离带，只比较“调到允许区间边缘”和“调整一半差额”两套执行规则。明确零仍全部退出；首次进入仍按完整目标。当前只准备方案，尚未实现或回测。',
        '目标尚未完成，研究继续。',
    ]
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'candidates': {model: 'CLOSED_FINITE_GRID_FULL_JOINT_TARGET_NOT_MET' for model in CANDIDATES},
        'retained_joint_comparison_candidate': 'ACCOUNT_VOLATILITY_EXPOSURE'}, exclusive=True)
    nxt = ROOT/'docs/510300_TARGET_BAND_PARTIAL_REBALANCE_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300策略提速与批量比较_第196轮_20260913/策略提速_14套批量结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '十四套风险窗口与预算时点批量校准',
        'COMPLETED_FOURTEEN_SETTING_BATCH_NO_JOINT_IMPROVEMENT', decision, '\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'TARGET_BAND_PARTIAL_REBALANCE_PREPARED',
        '保留181信号，偏离超过原带宽后只调到边缘或只调整一半差额')
    p = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(p.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=197, registered=False, planned_settings=2, planned_new_accounts=8,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        research_class='RETROSPECTIVE_FINITE_EXECUTION_RULE_COMPARISON')
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    index['latest_finite_grid_comparison'] = str((OUT/'all_eighteen_setting_comparison.csv').relative_to(ROOT))
    index['supplementary_saved_return_screen'] = {'result': 'reports/research/510300_saved_combination_fast_screen_20260913/result.json',
        'input_records': 564, 'distinct_complete_paths': 224, 'optimizer_starts': 2, 'optimizer_objective_evaluations': 32,
        'new_accounts': 0, 'independent_validation': False}
    for key in ['current_best_four_scenario_comparison_candidate', 'latest_main_sharpe_pass_candidate', 'latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 196
    write_json(p, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
