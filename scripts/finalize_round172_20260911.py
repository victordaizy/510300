"""交付上涨日优势规则结果，准备核心与方向确认辅助仓位的新组合。"""
import json
from research.return_sign_balance_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    main_rows = [r for r in result['all_metrics'] if r['model'] == PRIMARY]
    early_rows = [r for r in result['earlier_diagnostics'] if r['model'] == PRIMARY]
    require(all(r['net_sharpe'] < 1.2 for r in main_rows+early_rows), '第172轮夏普结论不同')
    require(all(r['annualized_return_excess_vs_buy_hold'] < 0 for r in main_rows)
        and all(r['annualized_return_excess_vs_buy_hold'] > 0 for r in early_rows), '第172轮分段超额结论不同')
    decision = ('第172轮上涨日优势与累计方向完成。主基础／压力净夏普为－0.180／－0.238，'
        '较早为0.593／0.543；年化收益主－1.66%／－2.09%，较早4.72%／4.28%。'
        '四场景夏普不足1.2，主历史亏损且落后买入持有，较早历史超额为正但风险调整后表现仍不足。'
        '关闭这一固定独立策略，完整目标未完成。')
    detail = ('六项必要测试8.23秒通过，四个新账户核心计算2.101708秒，不含开发、测试、核对和交付。'
        '没有预测模型训练或新增参考账户；三套保存对照加新方案形成主、较早历史各八条指标。\n\n'
        '独立由原始收盘及分红重建日收益，逐个核对3396个完整六十日窗口的上涨、下跌、平盘天数，'
        '上涨日优势最大误差为零。方向状态及三个独立连续计数一致。'
        '5646个收盘决定、84段完整实际持仓、368次历史模拟开盘成交及资金、费用、分红和终点退出通过核对。'
        '这说明实现与固定规则一致，不构成独立收益样本验证。\n\n'
        '主每账户29段持仓、678个持仓收盘、110次成交；较早每账户13段、647个持仓收盘、74次成交。'
        '主678个正目标、926个零目标；较早648正、571零。较早正目标比持仓收盘多一天来自固定终点开盘清算。'
        '四账户均没有未知目标，终点已全部清仓。主647个正目标受波动率约束、31个满仓目标；'
        '较早分别543个和105个。\n\n'
        '进入确认达到两天的主记录为34次，上涨日优势退出确认28次，累计方向退出确认36次；'
        '较早分别17、20、24次。确认次数与实际交易周期不同，持仓期间条件再次连续成立也会产生确认记录。\n\n'
        '主基础价格加分红损益－14056.10元、费用6903.95元、净亏20960.05元；压力分别－13752.00、'
        '12401.57、－26153.57元。主历史扣费前已经亏损，不能把失败只归因于手续费。'
        '较早基础价格加分红利润57069.10元、费用4730.14元、净赚52338.96元；压力分别55999.40、'
        '8986.00、47013.40元。\n\n'
        '主最大回撤23.96%／25.25%，较早12.84%／13.65%；平均股票市值占净资产主27.23%／27.20%，'
        '较早36.15%／36.16%。相对买入持有的年化超额，主为－5.282／－5.701个百分点，'
        '较早为正0.827／0.410个百分点。相对143，四场景夏普和终值均较低；'
        '主终值少87362.12／88903.41元，较早少6475.51／14346.46元。\n\n'
        '保留143的原均衡比较记录、167的较高主段夏普和168相加封顶的正超额记录。'
        '下一项保留143核心机会，只用本轮原方向决定165辅助目标是否加入。'
        '这是事先固定的新条件组合，不更改172的方向、阈值、窗口、费用或失败结果。')
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        PRIMARY: 'CLOSED_ALL_FOUR_SHARPE_BELOW12_MAIN_LOSS_EARLIER_EXCESS_POSITIVE'}, exclusive=True)
    next_path = ROOT/'docs/510300_SIGN_CONFIRMED_RUNS_AUXILIARY_NEXT_20260911.md'
    document = ROOT/'deliverables/510300上涨日优势_第172轮_20260911/上涨日优势_结果及全部中文规则.md'
    delivery = deliver_round(ROOT, OUT, CONFIG, document, '实际上涨日优势与六十日累计方向',
        'CLOSED_RETURN_SIGN_BALANCE_FULL_GOAL_NOT_MET', decision, detail, next_path,
        next_path.read_text(encoding='utf-8').splitlines(), 'SIGN_CONFIRMED_RUNS_AUXILIARY_FINITE_CANDIDATE_PREPARED',
        '保留143核心，只在172原方向允许时加入165辅助目标，相加封顶并独立执行进退场')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=173, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        planned_saved_control_models=['RUNS_OPPORTUNITY_CAPPED_SUM', 'TREND_NOISE_REFERENCE_BLEND',
            'RETURN_RUNS_STATE', 'RETURN_SIGN_BALANCE', 'BUY_HOLD'],
        planned_main_metric_rows=12, planned_earlier_metric_rows=12)
    index['current_best_four_scenario_comparison_candidate'].update(last_compared_completed_round=172,
        comparison_basis='SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO172')
    index['latest_main_sharpe_pass_candidate'].update(last_compared_completed_round=172,
        selection_note='保留167作为主历史两费用较高夏普比较记录；172四场景夏普不足且主历史亏损')
    index['latest_all_four_scenario_excess_candidate']['last_compared_completed_round'] = 172
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
