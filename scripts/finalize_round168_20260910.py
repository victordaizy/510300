"""交付两种机会合并的更高收益与较早夏普不足，保持完整目标未完成。"""
import json
from research.runs_opportunity_union_v1 import ROOT, OUT, CONFIG, CANDIDATES
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    main_rows = [r for r in result['all_metrics'] if r['model'] in CANDIDATES]
    early_rows = [r for r in result['earlier_diagnostics'] if r['model'] in CANDIDATES]
    require(all(r['net_sharpe'] >= 1.2 for r in main_rows) and all(r['net_sharpe'] < 1.2 for r in early_rows), '两种合并夏普结论不同')
    require(all(r['annualized_return_excess_vs_buy_hold'] > 0 for r in main_rows+early_rows), '两种合并年化超额结论不同')
    decision = ('第168轮两种进场机会合并完成。较大目标方案主基础／压力净夏普1.585／1.511，较早0.960／0.945；'
        '相加封顶方案主1.589／1.516，较早1.037／1.027。两方案各自四个场景的年化收益都超过买入持有，'
        '但较早夏普仍不足1.2，稳定超额及独立证据未建立，完整目标未完成，关闭这两套固定方法。')
    detail = ('六项必要测试3.57秒通过，八个新账户核心计算2.675370秒，不含开发、测试、核对及交付。'
        '零新增模型训练、零新增参考账户，没有重新运行父策略。五套保存对照在两段两费用中共二十条记录，加八个新账户，共二十八条指标。\n\n'
        '独立核对两种合成公式，合成目标最大误差为零。11292条不同的父收盘目标用于两个新策略，共22584次目标比对；'
        '11292个实际决定、184段完整持仓、614次开盘成交和全部资金、分红、费用通过核对。'
        '完整目标和真实账户的计算核对不等于已经完成独立收益验证。\n\n'
        '较大目标方案主年化7.34%／6.97%，较早6.02%／5.98%；主最大回撤3.38%／3.66%，较早6.58%／6.96%。'
        '相加封顶方案主年化7.63%／7.25%，较早7.72%／7.70%；主最大回撤3.38%／3.66%，较早6.69%／7.10%。'
        '相加封顶相对买入持有，主年化领先4.001／3.643个百分点，较早领先3.829／3.829个百分点。\n\n'
        '较大目标方案的主基础净赚119786.66元、压力112633.83元，较早基础68490.00元、压力68008.80元；'
        '对应佣金加滑点7133.14、12983.47、4869.00、9286.00元。相加封顶主基础净赚125530.18元、压力118093.89元，'
        '较早基础90947.23元、压力90591.27元；对应费用7349.32、13395.81、5705.77、10899.03元。\n\n'
        '两方案每个主账户均30段持仓、438个持仓收盘；较大目标各91次成交，相加封顶各93次成交。'
        '较早各16段，基础448个、压力454个持仓收盘；较大目标各59次成交，相加封顶各64次。'
        '八账户均没有未知目标或受阻未成交请求，终点全部清仓。\n\n'
        '两套父目标只在主历史11个、较早49个判断日同时为正。两种方案在仅一套有信号时使用同一父目标，'
        '差别主要来自两套同时有信号的持有规模及后续本账户净资产、整手和调仓带。'
        '主平均股票市值比例约12.86%至13.00%，较早约20.31%至22.13%，高于固定各半组合。'
        '提高有效仓位带来更多收益，也带来更大波动与回撤，较早风险收益效率没有达到要求。\n\n'
        '相加封顶相对143，四场景年化收益都更高；主夏普提高0.298／0.284，较早基础提高0.015、较早压力降低0.017。'
        '其回撤更大，不能宣称各项风险指标全面优于143。它是本轮两套方案中事后看到的较好收益组合，尚非经过独立验证的可用策略。'
        '143保留为原均衡比较基准，167保留主历史较高夏普记录，168相加封顶另记录四场景正超额及其回撤代价。\n\n'
        '下一项只检验过去完整账户平均净收益为正的部分，能否每月指导父策略预算；历史非正则不给新收益预算。'
        '不补EPS等慢来源，也不修改已关闭方法救回。')
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        **{model: 'CLOSED_MAIN_SHARPE_AND_FOUR_SCENARIO_EXCESS_PASS_EARLIER_SHARPE_BELOW12' for model in CANDIDATES}}, exclusive=True)
    next_path = ROOT/'docs/510300_RUNS_NET_PROFIT_BUDGET_NEXT_20260910.md'
    document = ROOT/'deliverables/510300进场机会合并_第168轮_20260910/进场机会合并_结果及全部中文规则.md'
    delivery = deliver_round(ROOT, OUT, CONFIG, document, '两套父目标取较大值及相加封顶',
        'CLOSED_RUNS_OPPORTUNITY_UNION_FULL_GOAL_NOT_MET', decision, detail,
        next_path, next_path.read_text(encoding='utf-8').splitlines(),
        'RUNS_NET_PROFIT_BUDGET_FINITE_CANDIDATE_PREPARED', '每月按两父过去242日完整账户平均净收益的正值分配预算')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=169, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        planned_saved_control_models=['RUNS_OPPORTUNITY_CAPPED_SUM', 'RUNS_COVARIANCE_BUDGET', 'RUNS_REFERENCE_BLEND',
            'TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE', 'BUY_HOLD'], planned_main_metric_rows=14, planned_earlier_metric_rows=14)
    index['current_best_four_scenario_comparison_candidate'].update(last_compared_completed_round=168,
        comparison_basis='SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO168')
    index['latest_main_sharpe_pass_candidate'].update(last_compared_completed_round=168,
        selection_note='保留167作为主历史两档费用夏普较高的比较记录；168新合并主段也通过但夏普略低，较早均未达标')
    combined = [r for r in main_rows+early_rows if r['model'] == 'RUNS_OPPORTUNITY_CAPPED_SUM']
    index['latest_all_four_scenario_excess_candidate'] = {'round': 168, 'study': result['study_id'],
        'model': 'RUNS_OPPORTUNITY_CAPPED_SUM', 'selection': 'POST_SELECTED_WITHIN_TWO_PRESPECIFIED_CANDIDATES',
        'source_result': str((OUT/'result.json').relative_to(ROOT)),
        'status': 'FOUR_WHOLE_PERIOD_EXCESSES_POSITIVE_EARLIER_SHARPE_BELOW12_AND_DRAWDOWN_HIGHER',
        'main_base_sharpe': combined[0]['net_sharpe'], 'main_stress_sharpe': combined[1]['net_sharpe'],
        'earlier_base_sharpe': combined[2]['net_sharpe'], 'earlier_stress_sharpe': combined[3]['net_sharpe'],
        'independent_validation': 'NOT_ESTABLISHED', 'goal_achieved': False}
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
