"""交付相邻收益相关规则的失败结果，准备实际上涨日优势的新方案。"""
import json
from research.return_lag_state_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    rows = [r for r in result['all_metrics']+result['earlier_diagnostics'] if r['model'] == PRIMARY]
    require(all(r['net_sharpe'] < 1.2 and r['annualized_return_excess_vs_buy_hold'] < 0 for r in rows), '第171轮夏普或超额结论不同')
    decision = ('第171轮相邻收益相关与累计方向完成。主基础／压力净夏普0.322／0.242，较早0.156／0.075；'
        '年化收益主1.58%／1.15%，较早0.81%／0.27%。四场景夏普不足1.2，年化收益均低于买入持有，'
        '也未改善143和165，关闭这一固定每日市场状态方法，完整目标未完成。')
    detail = ('六项必要测试5.77秒通过，四个新账户核心计算1.625821秒，不含开发、测试、核对和交付。'
        '没有预测模型训练或新增参考账户；三套保存对照两段两费用共十二条，加四新账户共十六条指标记录。\n\n'
        '独立由原始收盘与分红重建日收益，再用独立统计实现核对3396个完整六十日窗口及其中的相邻配对相关。'
        '相关度最大误差约一点六七乘十的负十六次方，完整状态和三个连续计数一致。'
        '5646个实际决定、100段完整持仓、314次真实开盘成交以及全部资金、费用、分红和终点退出通过核对。'
        '该核对说明实现符合规则，不代表独立收益样本已证明策略目标。\n\n'
        '主每账户26段实际持仓、384个持仓收盘、79次成交；较早各24段、361个持仓收盘、78次成交。'
        '主384个正目标、1220个零目标；较早362正857零，较早正目标比持仓收盘多一天来自固定终点开盘清算。'
        '四账户均没有未知目标或受阻未成交请求，终点全部清仓。\n\n'
        '主进入连续计数达到两天29次，相关度退出确认44次，累计方向退出确认36次；较早分别31、31、24次。'
        '确认次数不等于实际交易周期：允许状态保持期间条件短暂中断后重新满足，也会再次达到两天确认。\n\n'
        '主基础价格加分红利润28434.60元、费用6595.36元、净赚21839.24元；压力分别27706.60、11994.45、15712.15元。'
        '较早基础分别13778.40、5473.63、8304.77元；压力分别13245.40、10463.73、2781.67元。'
        '主最大回撤13.36%／14.20%，较早10.89%／11.36%；平均股票市值占净资产主约14.95%／14.94%，较早20.65%／20.66%。'
        '其毛利润本来就不高，费用进一步压低净利润，较早压力情景尤其明显。\n\n'
        '相对买入持有，主年化落后2.050／2.461个百分点，较早落后3.085／3.595个百分点。'
        '相对143，主终值少44562.83／47037.69元，较早少50509.71／58578.19元，四场景夏普均更低。'
        '不修改相关系数方向、阈值或窗口救回本轮。\n\n'
        '143的原均衡比较记录、167的较高主段夏普和168相加封顶的正超额记录保持。'
        '下一项单独检验实际上涨日是否多于下跌日、且累计收益是否为正，以固定的频率和幅度双条件形成进出场。')
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        PRIMARY: 'CLOSED_ALL_FOUR_SHARPE_BELOW12_ALL_EXCESSES_NEGATIVE_AND_BELOW_REFERENCE'}, exclusive=True)
    next_path = ROOT/'docs/510300_RETURN_SIGN_BALANCE_NEXT_20260910.md'
    document = ROOT/'deliverables/510300相邻收益相关_第171轮_20260910/相邻收益相关_结果及全部中文规则.md'
    delivery = deliver_round(ROOT, OUT, CONFIG, document, '相邻收益相关与六十日累计方向的独立价格状态',
        'CLOSED_RETURN_LAG_STATE_FULL_GOAL_NOT_MET', decision, detail, next_path,
        next_path.read_text(encoding='utf-8').splitlines(), 'RETURN_SIGN_BALANCE_FINITE_CANDIDATE_PREPARED',
        '六十日实际上涨日占优及累计方向，两日进入和分别两日退出的独立市场状态')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=172, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        planned_saved_control_models=['TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE', 'BUY_HOLD'],
        planned_main_metric_rows=8, planned_earlier_metric_rows=8)
    index['current_best_four_scenario_comparison_candidate'].update(last_compared_completed_round=171,
        comparison_basis='SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO171')
    index['latest_main_sharpe_pass_candidate'].update(last_compared_completed_round=171,
        selection_note='保留167作为主历史两费用较高夏普比较记录；171四场景不足且超额均为负')
    index['latest_all_four_scenario_excess_candidate']['last_compared_completed_round'] = 171
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
