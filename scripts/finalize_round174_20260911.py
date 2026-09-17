"""交付三种方向确认结果，保留任一确认的改善并准备进入资格研究。"""
import json
import pandas as pd
from research.return_confirmation_auxiliary_batch_v1 import ROOT, OUT, CONFIG, PRIMARY, CANDIDATES
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    main_rows = [r for r in result['all_metrics'] if r['model'] in CANDIDATES]
    early_rows = [r for r in result['earlier_diagnostics'] if r['model'] in CANDIDATES]
    require(all(r['net_sharpe'] >= 1.2 for r in main_rows) and all(r['net_sharpe'] < 1.2 for r in early_rows),
        '第174轮分段夏普结论不同')
    require(all(r['annualized_return_excess_vs_buy_hold'] > 0 for r in main_rows+early_rows), '第174轮正超额结论不同')
    selected = 'EITHER_CONFIRMED_RUNS_AUXILIARY'
    decision = ('第174轮三种方向确认辅助组合一次完成。预先指定的主方案“两方向共同确认”主基础／压力净夏普'
        '1.523／1.440，较早0.828／0.823；“仅相关方向确认”分别1.551／1.474和0.849／0.844；'
        '“任一方向确认”分别1.497／1.423和1.098／1.089。全部十二个场景年化超额为正，'
        '但三套方案的较早历史夏普均不足1.2，完整目标未完成。'
        '任一确认在四场景夏普与年化收益上均优于第173轮，保留这一局部改善；它是三方案中的事后比较选择，主方案身份不变。')
    detail = ('七项必要测试3.19秒通过，十二个新账户共用一次数据和来源读取，核心计算2.757116秒。'
        '该时间不含开发、测试、核对与交付。没有重复训练旧模型或新建参考账户，'
        '两段历史各十六条指标，其中各六条为新账户，其余来自五套保存对照。\n\n'
        '独立核对22584个不同的保存来源目标和11292个不同的方向值，三方案合计67752次来源目标比较。'
        '16938个本轮收盘决定、274段完整实际持仓、888次历史模拟开盘成交，及全部账户资金、费用、分红和终点清算通过核对。'
        '组合目标的最大误差为零。核对证明实现符合固定规则，不能替代独立样本的收益验证。\n\n'
        '共同确认在主历史保留辅助94个正目标、拒绝50个；较早保留94个、拒绝102个。'
        '仅相关确认主保留104个、拒绝40个；较早保留95个、拒绝101个。'
        '任一确认主保留135个、拒绝9个；较早保留192个、拒绝4个。'
        '要求更多条件同时成立，大幅削减了较早历史的辅助机会，并未提升该段夏普。\n\n'
        '任一确认主年化收益7.05%／6.69%，较早8.13%／8.11%；年化超额主3.429／3.078个百分点，'
        '较早4.231／4.245个百分点。主最大回撤3.38%／3.65%，较早6.30%／6.30%。'
        '主两账户各30段持仓、429个持仓收盘、93次成交；较早各16段、445／451个持仓收盘、62次成交。\n\n'
        '任一确认相对173，主终值增加5509.70／5796.68元，较早增加1944.25／1894.72元；'
        '夏普分别提高0.028／0.033和0.018／0.017。主回撤略有扩大，较早最大回撤相同。'
        '四场景最低夏普由173的1.072提高至1.089，距离1.2仍有约0.111。'
        '这只是已反复研究历史中的点值差距，不能线性外推成即将达标。\n\n'
        '共同确认相对173，较早终值少39998.97／39550.83元；仅相关确认少38344.72／37876.20元。'
        '两者虽然费用较少，但价格和分红利润减少得更多。任一确认主历史仍弱于168相加封顶的夏普，'
        '保留167的较高主历史夏普和168的原机会合并比较，不以单一最佳值覆盖不同取舍。\n\n'
        '以下列出十二个独立账户的实际损益构成，单位为元。价格加分红为扣除费用之前的损益；'
        '费用包括佣金和滑点，三者在各自账户上核对，不把其他费用情景的净值混合。\n\n')
    table = ['|方案|历史段|费用|持仓段数|价格加分红|佣金及滑点|净利润|',
        '|---|---|---|---:|---:|---:|---:|']
    for row in pd.read_csv(OUT/'saved_account_checks.csv').to_dict('records'):
        table.append(f"|{CANDIDATES[row['model']]}|{'主' if row['period']=='evaluation' else '较早'}|"
            f"{'基础' if row['cost']=='BASE' else '压力'}|{row['cycles']}|{row['gross_price_dividend_profit']:.2f}|"
            f"{row['commission_and_slippage']:.2f}|{row['net_profit']:.2f}|")
    detail += '\n'.join(table)+'\n\n下一轮固定比较段首确认和段内等候两种辅助资格：确认负责进入，辅助原规则负责退出；不改变旧方案结果。'
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'selection_scope': 'PRIMARY_DUAL_UNCHANGED_EITHER_POST_SELECTED_FOR_FOUR_SCENARIO_COMPARISON',
        **{model: ('CLOSED_MAIN_SHARPE_PASS_EARLIER_BELOW12_FOUR_EXCESSES_POSITIVE_LOCAL_FOUR_SCENARIO_GAIN'
            if model == selected else 'CLOSED_MAIN_SHARPE_PASS_EARLIER_BELOW12_FOUR_EXCESSES_POSITIVE_EARLIER_WEAKER_THAN173')
            for model in CANDIDATES}}, exclusive=True)
    next_path = ROOT/'docs/510300_CONFIRMED_AUXILIARY_EPISODE_BATCH_NEXT_20260911.md'
    document = ROOT/'deliverables/510300三种方向确认_第174轮_20260911/三种方向确认_结果及全部中文规则.md'
    delivery = deliver_round(ROOT, OUT, CONFIG, document, '三个固定方向条件一次比较辅助仓位',
        'CLOSED_RETURN_CONFIRMATION_AUXILIARY_BATCH_FULL_GOAL_NOT_MET', decision, detail, next_path,
        next_path.read_text(encoding='utf-8').splitlines(), 'CONFIRMED_AUXILIARY_EPISODE_BATCH_PREPARED',
        '段首确认与段内等候两个辅助资格方案，取得资格后直到辅助原目标归零才退出')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=175, registered=False, planned_settings=2, planned_new_accounts=8,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        planned_saved_control_models=[selected, 'SIGN_CONFIRMED_RUNS_AUXILIARY', 'RUNS_OPPORTUNITY_CAPPED_SUM',
            'TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE', 'BUY_HOLD'],
        planned_main_metric_rows=16, planned_earlier_metric_rows=16)
    m, e = ({r['cost']: r for r in group if r['model'] == selected} for group in [main_rows, early_rows])
    index['previous_balanced_candidate_before_round174'] = index['current_best_four_scenario_comparison_candidate']
    index['current_best_four_scenario_comparison_candidate'] = {
        'study': result['study_id'], 'model': selected, 'round': 174,
        'status': 'POST_SELECTED_WITHIN_THREE_FOUR_SCENARIO_GAIN_EARLIER_BELOW12_NOT_INDEPENDENT',
        'selection': 'POST_SELECTED_FOUR_SCENARIO_COMPARISON_PRIMARY_REMAINS_DUAL_CONFIRMED_RUNS_AUXILIARY',
        'source_result': str((OUT/'result.json').relative_to(ROOT)),
        'comparison_basis': 'SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO174',
        'base_main_sharpe': m['BASE']['net_sharpe'], 'stress_main_sharpe': m['STRESS']['net_sharpe'],
        'base_earlier_sharpe': e['BASE']['net_sharpe'], 'stress_earlier_sharpe': e['STRESS']['net_sharpe'],
        'minimum_four_scenario_sharpe': min(r['net_sharpe'] for r in list(m.values())+list(e.values())),
        'four_full_period_excess_returns_positive': True, 'every_year_stable_excess_established': False,
        'independent_validation': 'NOT_ESTABLISHED', 'goal_achieved': False,
        'four_sharpe_and_cagr_improve_vs173': True, 'main_drawdowns_larger_than173': True,
        'earlier_drawdowns_same_as173': True, 'last_compared_completed_round': 174}
    index['latest_main_sharpe_pass_candidate'].update(last_compared_completed_round=174,
        selection_note='保留167作为主历史两费用较高夏普比较记录；174任一确认提高四场景但较早仍不足1.2')
    index['previous_all_four_scenario_excess_candidate_before_round174'] = index['latest_all_four_scenario_excess_candidate']
    index['latest_all_four_scenario_excess_candidate'] = {
        'round': 174, 'study': result['study_id'], 'model': selected,
        'selection': 'POST_SELECTED_WITHIN_THREE_PRESPECIFIED_CANDIDATES',
        'source_result': str((OUT/'result.json').relative_to(ROOT)),
        'status': 'FOUR_WHOLE_PERIOD_EXCESSES_POSITIVE_EARLIER_SHARPE_BELOW12',
        'main_base_sharpe': m['BASE']['net_sharpe'], 'main_stress_sharpe': m['STRESS']['net_sharpe'],
        'earlier_base_sharpe': e['BASE']['net_sharpe'], 'earlier_stress_sharpe': e['STRESS']['net_sharpe'],
        'independent_validation': 'NOT_ESTABLISHED', 'goal_achieved': False, 'last_compared_completed_round': 174}
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
