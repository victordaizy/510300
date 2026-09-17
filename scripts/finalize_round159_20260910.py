"""交付八曲线退出的完整结果，关闭失败方法并转向新进入结构。"""
import json
import shutil
import pandas as pd
from research.marginal_monotone_exit_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    summaries = []
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost in ['BASE', 'STRESS']:
            folder = OUT / period / cost
            decisions = pd.read_parquet(folder / f'{PRIMARY}_decisions.parquet')
            available = decisions[decisions.learning_status.eq('PREDICTION_AVAILABLE')]
            cycles = pd.read_csv(folder / f'{PRIMARY}_cycles.csv')
            ledger = pd.read_parquet(folder / f'{PRIMARY}_ledger.parquet')
            summaries.append({'period': period, 'cost': cost, 'two_day_cycles': int(cycles.holding_intervals.eq(2).sum()),
                'available_predictions': len(available), 'negative_predictions': int(available.continuation_prediction.lt(0).sum()),
                **{key: float(ledger[key].sum()) for key in ['price_pnl', 'dividend_recognized', 'commission', 'slippage_cost']}})
    write_json(OUT / 'saved_prediction_path_summary.json', {'recorded_at': now(), 'saved_only': True,
        'new_models': 0, 'new_accounts': 0, 'records': summaries}, exclusive=True)
    decision = ('第159轮八个单因素曲线等权退出已完成。主基础／压力净夏普−0.057／−0.130，较早0.642／0.621；'
        '年化收益主−0.79%／−1.38%，较早7.78%／7.48%。四项夏普均低于1.2、128及143，主账户亏损，关闭这套固定方法。完整目标尚未实现。')
    detail = ('九项必要测试5.48秒；25组实际训练输入各拟合一次，后来89个月复用，保留141个月度时钟与114个可用月份。'
        '全部拟合成功，共200条曲线，其中175次非恒定方向保序求解、25条恒定进入类别曲线。训练和四账户核心计算4.230426秒，'
        '不含开发、测试、结果核对与交付时间。\n\n'
        '独立SciPy求解核对25组模型的全部加权曲线，最大支撑点预测误差3.78乘十的负十四次方。'
        '独立手工插值核对652次实际预测；76个实际周期、1060个持仓状态、5646个收盘判断、152次成交及完整费用分红净值均通过。'
        '这是数值与账户一致性核对，不是独立样本有效性证明。全部25组的200条曲线参数和3158个必要断点均列在本文后部，'
        '包括均值尺度、成熟协方差方向、固定权重和每个断点的实际预测贡献。\n\n'
        '主各29周期、190个持仓收盘、58次成交；26周期触发学习退出，22周期仅持有两个交易日。'
        '较早各9周期、340个持仓收盘、18次成交；136个状态可预测、204个状态在入场时没有成熟模型；'
        '没有周期因学习条件退出，也没有仅持有两日的周期。四账户全部清仓，没有受阻未成交请求。'
        '同一类学习方法在两段历史形成了明显不同的退出频率，不能将主区间频繁卖出解释为已学到普遍有效的退出能力。\n\n'
        '主基础价格损益−9066.00元、分红7978.20元、佣金与滑点9117.60元，净亏10205.40元；'
        '压力价格损益−8769.00元、分红7803.90元、费用16641.29元，净亏17606.39元。'
        '两档主账户在扣费前的价格与分红合计已经亏损。较早净利润91740.77／87649.08元。'
        '主最大回撤18.99%／21.68%，较早13.79%／13.92%。\n\n'
        '相对128，主终值少122758.67／121573.67元，较早少13326.41／20607.57元。'
        '主价格与分红合计相对128少122277.60／120704.00元，额外费用仅481.07／869.67元，主要问题是持仓时机。'
        '较早费用还少47.19／96.23元，但收益仍下降。四项夏普与年化收益均低于上一158轮。'
        '主年化落后买入持有4.41／4.99个百分点，较早高出3.89／3.61个百分点；相对143主年化更低、较早更高。\n\n'
        '25个模型中，二十日波动的曲线方向全部为下降；进入类别全部恒定。其他因素方向存在随成熟资料改变的情况。'
        '这些方向是模型训练结果，不是独立证实的因果关系，也不是把某个因素反向交易即可获得超额的依据。'
        '下一项转向每天累计上涨与下跌证据决定进入和退出，配固定波动预算；不继续更换这套标签的退出拟合器。')
    write_json(OUT / 'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        PRIMARY: 'CLOSED_FOUR_SHARPE_BELOW12_128_143_MAIN_LOSSES_AND_FREQUENT_TWO_DAY_EXIT'}, exclusive=True)
    next_path = ROOT / 'docs/510300_SEQUENTIAL_RETURN_STATE_NEXT_20260910.md'
    document = ROOT / 'deliverables/510300八曲线退出_第159轮_20260910/八曲线退出_结果及全部中文规则.md'
    delivery = deliver_round(ROOT, OUT, CONFIG, document, '原八因素单调曲线等权与固定版本退出',
        'CLOSED_MARGINAL_MONOTONE_EXIT_FULL_GOAL_NOT_MET', decision, detail, next_path,
        next_path.read_text(encoding='utf-8').splitlines(), 'SEQUENTIAL_RETURN_STATE_FINITE_CANDIDATE_PREPARED',
        '完整市场日历双向累积收益证据决定进入退出，按当时波动控制仓位')
    parameters = OUT / '全部已拟合模型参数.md'
    shutil.copy2(parameters, document.parent / parameters.name)
    with document.open('a', encoding='utf-8') as stream:
        stream.write('\n'+parameters.read_text(encoding='utf-8'))
    index_path = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(index_path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=160, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False)
    index['current_best_four_scenario_comparison_candidate'].update(last_compared_completed_round=159,
        comparison_basis='SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO159')
    write_json(index_path, index)
    require(document.read_text(encoding='utf-8').count('## 首次拟合：') == 25, '报告未包含全部25组曲线')
    require(len(pd.read_csv(document.parent / 'saved_all_factor_curve_parameters.csv')) == 200 and
        len(pd.read_csv(document.parent / 'saved_all_curve_knots.csv')) == 3158, '全部因素与曲线断点数量不同')
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
