"""交付方向确认辅助仓位的局部改善，准备一次有限批量比较。"""
import json
from research.sign_confirmed_runs_auxiliary_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    main_rows = [r for r in result['all_metrics'] if r['model'] == PRIMARY]
    early_rows = [r for r in result['earlier_diagnostics'] if r['model'] == PRIMARY]
    require(all(r['net_sharpe'] >= 1.2 for r in main_rows) and all(r['net_sharpe'] < 1.2 for r in early_rows),
        '第173轮分段夏普结论不同')
    require(all(r['annualized_return_excess_vs_buy_hold'] > 0 for r in main_rows+early_rows), '第173轮正超额结论不同')
    decision = ('第173轮保留核心、以上涨方向确认辅助仓位完成。主基础／压力净夏普1.469／1.390，'
        '较早1.080／1.072；年化收益主6.77%／6.38%，较早7.99%／7.98%。'
        '四场景年化超额均为正，较早夏普仍不足1.2，完整目标未完成。'
        '相对第168轮不加方向条件的相加封顶，较早表现改善，主历史表现下降；关闭本轮固定方案的达标申请，保留局部改善记录。')
    detail = ('七项必要测试7.02秒通过，四个新账户核心计算2.825542秒，不含开发、测试、核对与交付。'
        '复用三套已保存来源和五套账户对照，不重跑旧模型；主与较早历史各十二条指标中，各只有两个新账户。\n\n'
        '独立核对16938个保存来源目标及5646个方向值，再以另一段实现重建组合目标，目标最大误差为零。'
        '5646个本组合决定、94段实际持仓、314次历史模拟开盘成交，以及全部资金、费用、分红与终点退出通过核对。'
        '实现符合固定规则，不代表独立收益证据已成立。\n\n'
        '主每账户31段持仓、419个持仓收盘、95次成交；较早各16段，基础444个、压力450个持仓收盘，均62次成交。'
        '主426个正目标、1178个零目标；较早基础445正774零，压力451正768零。'
        '正目标不等于持仓收盘：极小正目标整手取整、调仓带和固定终点清算均影响实际份额。'
        '四账户没有未知目标，终点均清仓。\n\n'
        '在主历史，辅助原有144个正目标中，方向允许125个，拒绝19个；较早196个中允许191个、拒绝5个。'
        '主159个、较早基础121个和压力127个收盘，即使方向不允许，核心仍保留正目标。'
        '这次过滤对两个时期的作用不同，不能只看较早改善就认为新增条件稳定有效。\n\n'
        '主基础价格加分红利润116084.00元、费用7358.87元、净赚108725.13元；压力分别114781.80、'
        '13441.83、101339.97元。较早基础分别100056.60、5545.84、94510.76元；压力分别104978.80、'
        '10586.87、94391.93元。不同费用下的仓位和后续收益路径各自独立计算。\n\n'
        '主最大回撤3.30%／3.56%，较早6.30%／6.30%。相对168相加封顶，四场景回撤均下降；'
        '主终值减少16805.04／16753.93元，夏普下降0.120／0.126；较早终值增加3563.53／3800.66元，'
        '夏普提高0.044／0.045。主减少的利润主要是价格收益和分红减少，不能解释为费用节约不足。\n\n'
        '相对买入持有，主年化超额3.143／2.771个百分点，较早4.090／4.107个百分点。'
        '相对143，四场景夏普和年化收益均提高，但四场景最大回撤也更大。'
        '四场景最低夏普由143的1.022提高到本轮1.072，作为事后历史比较记录，不能据此宣布稳定目标完成。\n\n'
        '167仍保留为较高主历史夏普的比较，168仍保留为不加方向条件的相加封顶比较。'
        '下一轮一次冻结三种方案：相邻相关方向确认、两方向共同确认、任一方向确认。'
        '共同数据和来源一次读取、十二个新账户一次计算，减少重复开发和逐轮等待。')
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        PRIMARY: 'CLOSED_MAIN_SHARPE_PASS_EARLIER_BELOW12_FOUR_EXCESSES_POSITIVE_LOCAL_EARLIER_GAIN'}, exclusive=True)
    next_path = ROOT/'docs/510300_RETURN_CONFIRMATION_AUXILIARY_BATCH_NEXT_20260911.md'
    document = ROOT/'deliverables/510300方向确认辅助仓位_第173轮_20260911/方向确认辅助仓位_结果及全部中文规则.md'
    delivery = deliver_round(ROOT, OUT, CONFIG, document, '核心加上涨方向确认的辅助仓位',
        'CLOSED_SIGN_CONFIRMED_RUNS_AUXILIARY_FULL_GOAL_NOT_MET', decision, detail, next_path,
        next_path.read_text(encoding='utf-8').splitlines(), 'RETURN_CONFIRMATION_AUXILIARY_BATCH_PREPARED',
        '保留核心，一次比较相邻相关确认、双方向共同确认和任一方向确认的三个辅助组合')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=174, registered=False, planned_settings=3, planned_new_accounts=12,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        planned_saved_control_models=[PRIMARY, 'RUNS_OPPORTUNITY_CAPPED_SUM', 'TREND_NOISE_REFERENCE_BLEND',
            'RETURN_RUNS_STATE', 'BUY_HOLD'], planned_main_metric_rows=16, planned_earlier_metric_rows=16)
    m, e = ({r['cost']: r for r in group} for group in [main_rows, early_rows])
    index['previous_balanced_candidate_before_round173'] = index['current_best_four_scenario_comparison_candidate']
    index['current_best_four_scenario_comparison_candidate'] = {
        'study': result['study_id'], 'model': PRIMARY, 'round': 173,
        'status': 'POST_SELECTED_FOUR_SCENARIO_MINIMUM_GAIN_EARLIER_BELOW12_DRAWDOWN_HIGHER_THAN143',
        'source_result': str((OUT/'result.json').relative_to(ROOT)),
        'comparison_basis': 'SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO173',
        'base_main_sharpe': m['BASE']['net_sharpe'], 'stress_main_sharpe': m['STRESS']['net_sharpe'],
        'base_earlier_sharpe': e['BASE']['net_sharpe'], 'stress_earlier_sharpe': e['STRESS']['net_sharpe'],
        'minimum_four_scenario_sharpe': min(r['net_sharpe'] for r in main_rows+early_rows),
        'four_full_period_excess_returns_positive': True, 'every_year_stable_excess_established': False,
        'independent_validation': 'NOT_ESTABLISHED', 'goal_achieved': False,
        'four_sharpe_and_cagr_improve_vs143': True, 'four_drawdowns_larger_than143': True,
        'main_sharpe_lower_than168_capped_sum': True, 'earlier_sharpe_higher_than168_capped_sum': True,
        'last_compared_completed_round': 173}
    index['latest_main_sharpe_pass_candidate'].update(last_compared_completed_round=173,
        selection_note='保留167作为主历史两费用较高夏普比较记录；173改善较早但两费用仍不足1.2')
    index['previous_all_four_scenario_excess_candidate_before_round173'] = index['latest_all_four_scenario_excess_candidate']
    index['latest_all_four_scenario_excess_candidate'] = {
        'round': 173, 'study': result['study_id'], 'model': PRIMARY,
        'selection': 'POST_SELECTED_HISTORICAL_COMPARISON_NOT_INDEPENDENT',
        'source_result': str((OUT/'result.json').relative_to(ROOT)),
        'status': 'FOUR_WHOLE_PERIOD_EXCESSES_POSITIVE_EARLIER_SHARPE_BELOW12_DRAWDOWN_HIGHER_THAN143',
        'main_base_sharpe': m['BASE']['net_sharpe'], 'main_stress_sharpe': m['STRESS']['net_sharpe'],
        'earlier_base_sharpe': e['BASE']['net_sharpe'], 'earlier_stress_sharpe': e['STRESS']['net_sharpe'],
        'independent_validation': 'NOT_ESTABLISHED', 'goal_achieved': False, 'last_compared_completed_round': 173}
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
