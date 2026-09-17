"""交付第181轮动态账户风险预算的联合目标结果。"""
import json

from research.account_volatility_exposure_v1 import CONFIG, OUT, PRIMARY, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    r = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    a = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    v = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    require(not a['main_two_cost_joint_pass'] and not a['earlier_two_cost_joint_pass'], '联合目标关闭结论不同')
    decision = ('第181轮按原完整账户最近六十日已实现波动，每天确定股票资金预算。'
                '主历史基础／压力净夏普1.306／1.223、复合年化9.66%／8.99%；'
                '较早历史净夏普1.069／1.056、复合年化9.88%／9.84%。'
                '四个场景均没有同时达到夏普1.2和复合年化10%，关闭本固定结构。')
    detail = [
        '|历史段|费用|净夏普|复合年化|最大回撤|成交次数|平均股票仓位|',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for period, key in [('主历史', 'all_metrics'), ('较早历史', 'earlier_diagnostics')]:
        for row in r[key]:
            if row['model']==PRIMARY:
                detail.append(f"|{period}|{'基础' if row['cost']=='BASE' else '压力'}|{row['net_sharpe']:.3f}|{row['annualized_return']:.2%}|{row['max_drawdown']:.2%}|{row['trade_count']}|{row['mean_exposure']:.2%}|")
    detail += ['', '三个必要测试通过；四条新模拟账户核心计算1.48秒，不含开发、测试及核对。',
        f"独立重建六十日风险、每日倍率、完整目标和{v['actual_decisions_checked']}个收盘决定，核对{v['complete_actual_cycles']}段实际模拟持仓和{v['simulated_fills_checked']}次模拟成交。资金、费用、分红、下一开盘与终点清算一致，没有重新训练模型。",
        '',
        '相对第180轮固定两倍预算，主历史基础／压力年化只提高约0.28／0.17个百分点，主历史最大回撤扩大到6.72%／7.15%，每档费用成交从95次增加到115次。它并没有实现收益、风险与费用的共同改善。',
        '',
        '较早历史压力费用夏普由固定两倍的0.976改善到1.056，但仍低于1.2；年化从10.51%回落至9.84%。不能将主历史较高收益和另一个方案较早历史较高收益拼成一条策略。',
        '',
        '六十日波动包含全部现金日，风险预算10%是目标尺度，实际账户波动不保证恰为10%。股票仓位100%上限、信号间歇、未知窗口处理、整手和调仓带都会改变实现路径。',
        '',
        '旧字段meets_point_target仅代表夏普一点二检查；joint_target_assessment.json和joint_target_metrics.csv才同时检查净夏普与复合年化。全部四个场景联合门槛为未通过。',
        '',
        '历史段仍为已观察过的研究回放，独立代码复核只证明计算符合规则。没有新的独立收益验证，完整目标保持未完成。',
        '',
        '下一轮固定区间开始时的预算倍率，用真实账户比较减少预算调整是否改善成本；保持原信号与退出资格，不再增加风险预算或修改窗口。']
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        PRIMARY: 'CLOSED_NO_SCENARIO_JOINT_SHARPE12_CAGR10'}, exclusive=True)
    nxt = ROOT/'docs/510300_EPISODE_ACCOUNT_RISK_BUDGET_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300账户风险预算_第181轮_20260913/账户风险预算_结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '以六十日完整账户波动确定每日股票预算',
        'CLOSED_ACCOUNT_VOLATILITY_JOINT_TARGET_NOT_MET', decision, '\n'.join(detail), nxt,
        nxt.read_text(encoding='utf-8').splitlines(), 'EPISODE_ACCOUNT_RISK_BUDGET_PREPARED',
        '正目标区间开始时确定风险倍率，区间内固定，信号归零退出')
    p = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    i = json.loads(p.read_text(encoding='utf-8'))
    i['next_work'].update(candidate_round=182, registered=False, planned_settings=1, planned_new_accounts=4,
                          planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False)
    i['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    for key in ['current_best_four_scenario_comparison_candidate','latest_main_sharpe_pass_candidate','latest_all_four_scenario_excess_candidate']:
        i[key]['last_compared_completed_round'] = 181
    i['previous_main_return_improvement_candidate_before181'] = i['latest_main_return_improvement_candidate']
    i['latest_main_return_improvement_candidate'] = {
        'round': 181, 'model': PRIMARY, 'source_result': str((OUT/'result.json').relative_to(ROOT)),
        'selection': 'HIGHER_MAIN_CAGR_THAN_FIXED_TWO_MULTIPLIER_WITH_HIGHER_DRAWDOWN',
        'status': 'MAIN_CAGR_BELOW10_EARLIER_SHARPE_BELOW12', 'goal_achieved': False}
    write_json(p, i)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__=='__main__':
    main()
