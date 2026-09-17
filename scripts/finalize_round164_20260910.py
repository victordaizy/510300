"""交付价格顺序分布四账户结果与全部中文规则，接续收益强弱连续段。"""
import json
import pandas as pd
from research.ordinal_entropy_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, write_json


def main():
    cycles = pd.read_csv(OUT/'saved_actual_cycles.csv')
    cycle_lines = ['## 全部实际持仓周期', '',
        '每行从实际空仓进入至实际全部退出，期间可能加减仓。以下所有周期均在历史终点前结束，终点没有残留持仓。净利润含实际费用，本轮实际分红权利为零。', '',
        '|历史|费用|实际进入|实际退出|持仓收盘数|买入次数|卖出次数|净利润（元）|',
        '|---|---|---|---|---:|---:|---:|---:|']
    for row in cycles.itertuples():
        cycle_lines.append(f"|{'主历史' if row.period == 'evaluation' else '较早历史'}|{'基础' if row.cost == 'BASE' else '压力'}|{row.entry_date}|{row.exit_date}|{row.held_closes}|{row.buy_trades}|{row.sell_trades}|{row.net_profit:.2f}|")
    decision = ('第164轮价格顺序分布与上涨动量完成。主基础／压力净夏普0.237／0.215，较早0.674／0.661；'
        '年化收益主0.77%／0.69%、较早2.76%／2.70%。四项夏普均低于1.2和143，四段年化均落后买入持有，完整目标未实现，关闭这套固定结构。')
    detail = ('六项必要测试11.23秒通过，零模型训练，一套规则四账户核心计算2.458592秒，不含开发、测试、核对及交付。'
        '独立由原始收盘和分红重建财富，对3397个完整六十日窗口逐一枚举五十八组三日顺序，六类频数完全一致，熵最大数值误差为零。'
        '四账户5646个收盘决定、18段完整持仓、68次真实开盘成交及全部资金、费用和分红权利已核对。\n\n'
        '主每账户5段持仓、213个持仓收盘、21次成交；较早各4段、158个持仓收盘、13次成交。'
        '主213个正目标、1391个零目标；较早158个正目标、1061个零目标。'
        '没有未知目标或受阻未成交请求，到固定终点均为空仓。平均股票市值占净资产比例主约7.75%／7.76%，较早约7.20%。\n\n'
        '主基础价格利润11804.90元、费用1434.66元、净赚10370.24元；压力分别11896.40、2560.50、9335.90元。'
        '较早基础价格利润30138.60元、费用764.42元、净赚29374.18元；压力分别30126.90、1430.62、28696.28元。'
        '本轮四账户在分红登记日都没有获得应计分红权利，因此实际分红利润为零，分红源和登记、除息、到账逻辑均正常保留。'
        '主最大回撤4.59%／4.62%，较早6.80%／6.86%。较小回撤同时伴随较低平均持仓和收益，不能直接称为目标已实现。\n\n'
        '相对143，主终值少56031.83／53413.93元，较早少29440.30／32663.58元。'
        '四段年化分别落后买入持有2.86／2.92／1.14／1.17个百分点。'
        '较早基础第一段2015年2月9日至5月19日净赚33027.96元，已经超过整段历史净赚29374.18元，后续周期合计反而减少利润。'
        '主基础2025年6月27日至8月6日一段净赚9181.43元，而全期只净赚10370.24元；收益仍集中在少数区间。\n\n'
        '因此关闭这套完整结构，不放宽熵阈值或按年份拼接救回。下一项检验收益相对其窗口中位数的强弱是否连续成段，'
        '再与上涨动量共同决定进入退出，保持现有免费来源和无需训练的流程。\n\n'+'\n'.join(cycle_lines))
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'ORDINAL_ENTROPY': 'CLOSED_FOUR_SHARPE_BELOW12_AND143_FOUR_CAGR_BELOW_BUY_HOLD_CONCENTRATED_CYCLE_PROFIT'}, exclusive=True)
    next_path = ROOT/'docs/510300_RETURN_RUNS_STATE_NEXT_20260910.md'
    delivery = deliver_round(ROOT, OUT, CONFIG,
        ROOT/'deliverables/510300价格顺序分布_第164轮_20260910/价格顺序分布_结果及全部中文规则.md',
        '三日价格顺序的六十日分布与上涨动量', 'CLOSED_ORDINAL_ENTROPY_FULL_GOAL_NOT_MET',
        decision, detail, next_path, next_path.read_text(encoding='utf-8').splitlines(),
        'RETURN_RUNS_STATE_FINITE_CANDIDATE_PREPARED', '收益高于或低于六十日中位数的连续段与上涨动量，分别确认进出场')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=165, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False)
    index['current_best_four_scenario_comparison_candidate'].update(last_compared_completed_round=164,
        comparison_basis='SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO164')
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
