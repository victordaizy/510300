"""交付双向市场状态的四账户结果和完整中文进出场规则。"""
import json
import pandas as pd
from research.sequential_return_state_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, write_json


def main():
    alarms = []
    for period in ['evaluation', 'earlier_diagnostic']:
        factors = pd.read_parquet(OUT/period/'BASE'/'factors.parquet')
        active = factors[factors.decision_time.notna() & factors.direction_trigger.isin([-1, 1])]
        for row in active.itertuples():
            alarms.append({'历史': '主历史' if period == 'evaluation' else '较早历史', '收盘日期': str(row.date.date()),
                '方向触发': '上涨' if row.direction_trigger == 1 else '下跌', '当天标准化收益': row.standardized_daily_return,
                '触发前上涨证据': row.up_evidence_before_reset, '触发前下跌证据': row.down_evidence_before_reset,
                '触发后方向': '上涨' if row.positive_direction == 1 else '非上涨'})
    pd.DataFrame(alarms).to_csv(OUT/'全部方向触发记录.csv', index=False, encoding='utf-8-sig')
    cycles = pd.read_csv(OUT/'saved_actual_cycles.csv')
    cycle_lines = ['## 全部实际持仓周期', '', '持仓期间可以加减仓，以下一行代表从实际空仓进入、到实际全部退出的完整一段。单段净利润含实际费用和分红；末段包含按固定终点开盘退出。', '',
        '|历史|费用|实际进入|实际退出|持仓收盘数|买入次数|卖出次数|净利润（元）|', '|---|---|---|---|---:|---:|---:|---:|']
    for row in cycles.itertuples():
        cycle_lines.append(f"|{'主历史' if row.period == 'evaluation' else '较早历史'}|{'基础' if row.cost == 'BASE' else '压力'}|{row.entry_date}|{row.exit_date}|{row.held_closes}|{row.buy_trades}|{row.sell_trades}|{row.net_profit:.2f}|")
    decision = ('第160轮双向累计收益证据与普通波动预算完成。主基础／压力净夏普0.579／0.564，较早0.770／0.748；'
        '年化收益主4.25%／4.13%、较早6.03%／5.85%。四段复合年化收益均超过买入持有，但四项夏普均低于1.2和143，完整目标未实现，关闭这套固定结构。')
    detail = ('六项必要测试2.88秒通过，零新模型训练，一套规则四账户核心计算0.978856秒；该时间不含开发、测试、结果核对与交付。'
        '独立从原始收盘及分红还原收益、前二十日波动与当前波动，并顺序复算3435个可用市场日状态；'
        '四账户5646个收盘决定、12段完整持仓、198次真实开盘成交和全部权益分红费用已核对。'
        '主、较早两个区间没有未知目标，两档费用共用相同市场方向和目标比例。\n\n'
        '主每账户3段完整持仓、664个持仓收盘、58次成交，其中28次买入及30次卖出；较早各3段、642个收盘、41次成交。'
        '主共有8次上涨触发和5次下跌触发，较早7次上涨和4次下跌触发；同方向重复触发不一定产生新持仓周期。'
        '主665个正目标和939个零目标，较早642个正目标和577个零目标。主正目标比实际持仓收盘多一，是因为终点开盘清算优先。'
        '四账户无受阻未成交请求，均完整清仓。\n\n'
        '主基础价格利润58872.20元、分红6970.80元、费用2245.13元、净利润63597.87元；压力分别58732.30、6933.90、4126.97、61539.23元。'
        '较早基础价格利润63544.40元、分红6708.00元、费用1635.32元、净利润68617.08元；压力分别62709.70、6677.70、3122.87、66264.53元。'
        '主最大回撤9.12%／9.35%，较早13.59%／13.70%。\n\n'
        '相对143，主终值少2804.21／1210.61元，较早多9802.60／4904.67元。四项夏普仍比143低0.712／0.668／0.252／0.296。'
        '较早收益提高，同时承担更深回撤，不能把较高收益直接称为夏普改善。两段两费用的复合年化超额分别为0.63／0.52／2.13／1.98个百分点，'
        '这是已观察历史中的点估计，没有新的独立验证。\n\n'
        '主区间三段进入分别在2020年7月3日、2024年9月25日、2025年6月26日，持续持有261、126、277个收盘。'
        '较早三段持有159、363、120个收盘，仍是少量长期区间贡献整体结果。下一项改用市场局部回撤的深度和持续时间限制仓位，'
        '与普通波动上限取较低值，保持一套无需训练的明确进入退出规则。\n\n'+'\n'.join(cycle_lines))
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'SEQUENTIAL_RETURN_STATE': 'CLOSED_FOUR_SHARPE_BELOW12_AND143_ONLY_THREE_ACTUAL_CYCLES_PER_SCENARIO'}, exclusive=True)
    next_path = ROOT/'docs/510300_DRAWDOWN_DEPTH_RISK_NEXT_20260910.md'
    delivery = deliver_round(ROOT, OUT, CONFIG,
        ROOT/'deliverables/510300双向收益状态_第160轮_20260910/双向收益状态_结果及全部中文规则.md',
        '完整市场日历双向收益证据、明确进出场及风险仓位', 'CLOSED_SEQUENTIAL_RETURN_STATE_FULL_GOAL_NOT_MET',
        decision, detail, next_path, next_path.read_text(encoding='utf-8').splitlines(),
        'DRAWDOWN_DEPTH_RISK_FINITE_CANDIDATE_PREPARED', '一百二十日趋势准入，普通波动与局部持续回撤风险上限取较小值')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=161, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False)
    index['current_best_four_scenario_comparison_candidate'].update(last_compared_completed_round=160,
        comparison_basis='SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO160')
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
