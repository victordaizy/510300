"""交付主段高夏普及完整目标未达的组合结果，保留整手退出说明。"""
import json
import shutil
from research.runs_reference_blend_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    primary = [row for row in result['all_metrics'] if row['model'] == 'RUNS_REFERENCE_BLEND']
    earlier = [row for row in result['earlier_diagnostics'] if row['model'] == 'RUNS_REFERENCE_BLEND']
    require(all(row['net_sharpe'] >= 1.2 for row in primary) and all(row['net_sharpe'] < 1.2 for row in earlier), '本轮主段与较早夏普判断不同')
    require(next(row for row in primary if row['cost'] == 'STRESS')['annualized_return_excess_vs_buy_hold'] < 0, '主压力年化超额判断不同')
    decision = ('第166轮两套既定目标各半组合完成。主基础／压力净夏普1.587／1.519，较早1.053／1.040；'
        '年化收益主3.75%／3.58%、较早4.09%／4.07%。主两项夏普通过1.2数值线，较早两项仍不足，'
        '且主压力年化收益略低于买入持有，完整目标未实现，关闭这一固定各半组合。')
    clarification = (ROOT/'docs/510300_RUNS_REFERENCE_BLEND_EXECUTION_CLARIFICATION_20260910.md').read_text(encoding='utf-8').split('\n', 2)[2]
    detail = ('六项必要测试4.03秒通过，零新增模型训练或参考账户，四个新组合账户核心计算1.960122秒，不含开发、测试、核对及交付。'
        '独立核对11292条父收盘目标和5646个合成决定，合成比例最大误差为零；92段完整持仓、246次真实开盘成交及全部资金、费用、分红通过核对。'
        '原父账户仅作为保存来源读取，本轮没有重新运行，也没有平均两条净值线代替实际成交。\n\n'
        '主每账户30段持仓、443个持仓收盘、72次成交；较早各16段，基础448个、压力454个持仓收盘，均51次成交。'
        '主445个正目标、1159个零目标；较早基础449正770零、压力455正764零。四账户无未知目标或受阻未成交请求，终点均完整清仓。'
        '正目标数与实际持仓数的差异由整手取整及固定终点清算产生，下文单列说明。\n\n'
        '主基础价格加分红利润58120.10元、费用2892.59元、净赚55227.51元；压力分别57789.20、5272.90、52516.30元。'
        '较早基础价格加分红利润47119.70元、费用2393.22元、净赚44726.48元；压力分别49049.70、4580.48、44469.22元。'
        '主最大回撤1.65%／1.79%，较早3.44%／3.64%。平均股票市值占净资产比例主约6.43%，较早11.09%／11.29%。\n\n'
        '相对143，主夏普提高0.296／0.287，较早基础提高0.030，较早压力降低约0.004；四场景年化收益都更低。'
        '主终值少11174.56／10233.53元，较早少14088.00／16890.64元。相对买入持有，主基础年化领先0.122个百分点、'
        '主压力落后0.028个百分点；较早基础与压力领先0.192／0.196个百分点。三个整段正超额仍不等于逐年稳定超额或独立验证。\n\n'
        '两套父目标在主历史只有11个判断日同时为正，只有143为正301日、只有165为正133日；较早同时为正49日。'
        '两父完整日净收益的历史相关系数主约0.061，较早约0.28。这与主历史较强的分散效果一致，'
        '但属于已观察历史的描述，不证明未来相关性或收益会保持，也未用于改变本轮固定各半预算。\n\n'
        '均衡比较候选继续保留143，另记录166为主历史两费用夏普通过的研究候选，明确标识较早夏普和主压力超额未达标。'
        '下一项每月按截至当天的两父完整净收益风险更新预算，固定二百四十二日窗口，独立检验是否改善不同阶段的资金分配。\n\n'
        '## 实际退出口径补充\n\n'+clarification)
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'RUNS_REFERENCE_BLEND': 'CLOSED_MAIN_TWO_SHARPE_PASS_EARLIER_BELOW12_AND_MAIN_STRESS_EXCESS_NEGATIVE'}, exclusive=True)
    next_path = ROOT/'docs/510300_RUNS_COVARIANCE_BUDGET_NEXT_20260910.md'
    document = ROOT/'deliverables/510300固定各半组合_第166轮_20260910/固定各半组合_结果及全部中文规则.md'
    delivery = deliver_round(ROOT, OUT, CONFIG, document,
        '第143轮趋势波动参考与第165轮收益连续段固定各半组合', 'CLOSED_RUNS_REFERENCE_BLEND_FULL_GOAL_NOT_MET',
        decision, detail, next_path, next_path.read_text(encoding='utf-8').splitlines(),
        'RUNS_COVARIANCE_BUDGET_FINITE_CANDIDATE_PREPARED', '按两套基础费用父账户过去242日方差协方差，每月首交易日更新预算')
    shutil.copy2(ROOT/'docs/510300_RUNS_REFERENCE_BLEND_EXECUTION_CLARIFICATION_20260910.md', document.parent/'实际退出口径补充.md')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=167, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        planned_saved_control_models=['RUNS_REFERENCE_BLEND', 'TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE', 'BUY_HOLD'],
        planned_main_metric_rows=10, planned_earlier_metric_rows=10)
    index['current_best_four_scenario_comparison_candidate'].update(last_compared_completed_round=166,
        comparison_basis='SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO166')
    index['latest_main_sharpe_pass_candidate'] = {'round': 166, 'study': '510300_RUNS_REFERENCE_BLEND_V1', 'model': 'RUNS_REFERENCE_BLEND',
        'source_result': str((OUT/'result.json').relative_to(ROOT)),
        'status': 'HISTORICAL_MAIN_TWO_COST_SHARPE_PASS_EARLIER_AND_FULL_EXCESS_GATES_FAILED',
        'main_base_sharpe': primary[0]['net_sharpe'], 'main_stress_sharpe': primary[1]['net_sharpe'],
        'earlier_base_sharpe': earlier[0]['net_sharpe'], 'earlier_stress_sharpe': earlier[1]['net_sharpe'],
        'all_four_period_excess_returns_positive': False, 'independent_validation': 'NOT_ESTABLISHED', 'goal_achieved': False}
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
