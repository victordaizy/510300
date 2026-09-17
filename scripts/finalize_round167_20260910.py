"""交付月度风险预算的主段改善和较早失败，保存下一有限合并方案。"""
import json
from research.runs_covariance_budget_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    primary = {r['cost']: r for r in result['all_metrics'] if r['model'] == PRIMARY}
    earlier = {r['cost']: r for r in result['earlier_diagnostics'] if r['model'] == PRIMARY}
    require(all(r['net_sharpe'] >= 1.2 and r['annualized_return_excess_vs_buy_hold'] > 0 for r in primary.values()),
        '第167轮主段夏普及超额结论不同')
    require(all(r['net_sharpe'] < 1.2 and r['annualized_return_excess_vs_buy_hold'] < 0 for r in earlier.values()),
        '第167轮较早夏普及超额结论不同')
    decision = ('第167轮月度历史风险预算完成。主基础／压力净夏普1.666／1.596、年化4.32%／4.13%，'
        '两项夏普达到1.2且年化超过买入持有；较早净夏普0.931／0.923、年化3.69%／3.68%，'
        '两项夏普不足1.2且年化低于买入持有。较早表现比固定各半组合更弱，完整目标未实现，关闭这套固定月度最小方差方法。')
    detail = ('七项最终必要测试3.91秒通过，四个新账户核心计算2.746808秒，不含开发、测试、核对和交付。'
        '零新增模型训练、零新增参考账户，父账户只读取保存结果。本轮计算四个新账户，另使用十六条两段两费用的保存对照记录。\n\n'
        '独立从父净资产重新计算日收益，再用另一套样本统计核对142条月度预算和初始化记录，其中93次完整风险更新。'
        '预算最大误差为三点三三乘十的负十六次方；11292条父收盘目标、5646个合成决定、90段完整持仓、245次开盘成交，'
        '以及全部资金、费用、分红和终点退出通过核对。该独立计算核对说明实现符合事前规则，不代表独立样本已证明收益目标。\n\n'
        '主历史80次月首尝试，50次更新、12次窗口不足、18次父风险退化；较早60次尝试，43次更新、12次窗口不足、5次风险退化。'
        '两段另各有一条初始各半记录。所有未更新的月首均按事前规则保留上一已知预算。\n\n'
        '第143轮的平均预算主历史57.20%、较早41.99%。较早2016年4月1日计算出的第143轮预算降为零，'
        '其后五次月首因为另一父策略窗口净收益方差为零而保留预算，持续至2016年9月30日，共125个判断日。'
        '这是最低历史方差目标与保留预算规则共同产生的实际分配，不是缺失收益被补成零。'
        '主历史两套预算均未到零或一的边界。不能据此事后修改本轮边界或窗口救回。\n\n'
        '主每账户30段持仓、442个持仓收盘、73次成交；较早各15段，基础388个持仓收盘、49次成交，'
        '压力394个持仓收盘、50次成交。主445个正目标、1159个零目标；较早基础389正830零、压力395正824零。'
        '四账户均没有未知目标或受阻未成交请求，终点均完整清仓。正目标与持仓日不完全相同，仍以实际整手和终点规则为准。\n\n'
        '主基础价格加分红利润68047.30元、费用3351.08元、净赚64696.22元；压力分别67659.30、6106.74、61552.56元。'
        '较早基础分别42531.10、2473.43、40057.67元；压力分别44690.40、4744.42、39945.98元。'
        '主最大回撤2.24%／2.45%，较早3.90%／4.11%；平均股票市值占净资产主约6.46%／6.45%，较早10.42%／10.59%。\n\n'
        '相对第166轮固定各半，主夏普提高0.079／0.077、终值多9468.71／9036.26元；较早夏普降低0.122／0.117、'
        '终值少4668.81／4523.24元。相对买入持有，主年化领先0.693／0.523个百分点，较早落后0.205／0.189个百分点。'
        '相对143，主夏普更高但终值仍少1705.85／1197.27元；较早终值少18756.80／21413.88元。\n\n'
        '均衡比较候选仍保留143，最新主历史两费用夏普通过候选记录为167，同时明确较早夏普与超额失败、独立验证未建立。'
        '下一轮固定两套进场机会合并规则，直接检验只有一套信号时使用原有目标规模的效果，不补新数据或训练旧模型。')
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        PRIMARY: 'CLOSED_MAIN_TWO_SHARPE_AND_EXCESS_PASS_EARLIER_SHARPE_AND_EXCESS_FAILED'}, exclusive=True)
    next_path = ROOT/'docs/510300_RUNS_OPPORTUNITY_UNION_NEXT_20260910.md'
    document = ROOT/'deliverables/510300月度历史风险预算_第167轮_20260910/月度历史风险预算_结果及全部中文规则.md'
    delivery = deliver_round(ROOT, OUT, CONFIG, document,
        '两套父策略每月按历史方差协方差分配预算', 'CLOSED_RUNS_COVARIANCE_BUDGET_FULL_GOAL_NOT_MET',
        decision, detail, next_path, next_path.read_text(encoding='utf-8').splitlines(),
        'RUNS_OPPORTUNITY_UNION_FINITE_CANDIDATES_PREPARED', '两套父目标取较大值及相加至百分之一百，两种进场合并方案')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=168, registered=False, planned_settings=2, planned_new_accounts=8,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        planned_saved_control_models=['RUNS_COVARIANCE_BUDGET', 'RUNS_REFERENCE_BLEND', 'TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE', 'BUY_HOLD'],
        planned_main_metric_rows=14, planned_earlier_metric_rows=14)
    index['current_best_four_scenario_comparison_candidate'].update(last_compared_completed_round=167,
        comparison_basis='SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO167')
    index['latest_main_sharpe_pass_candidate'] = {'round': 167, 'study': result['study_id'], 'model': PRIMARY,
        'source_result': str((OUT/'result.json').relative_to(ROOT)),
        'status': 'HISTORICAL_MAIN_TWO_COST_SHARPE_AND_EXCESS_PASS_EARLIER_GATES_FAILED',
        'main_base_sharpe': primary['BASE']['net_sharpe'], 'main_stress_sharpe': primary['STRESS']['net_sharpe'],
        'earlier_base_sharpe': earlier['BASE']['net_sharpe'], 'earlier_stress_sharpe': earlier['STRESS']['net_sharpe'],
        'all_four_period_excess_returns_positive': False, 'independent_validation': 'NOT_ESTABLISHED', 'goal_achieved': False}
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
