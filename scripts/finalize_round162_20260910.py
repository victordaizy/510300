"""交付排序方法失败结果，保留全部持仓，并接续已有数据的价量检验。"""
import json
import pandas as pd
from research.session_signed_rank_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, write_json


def main():
    cycles = pd.read_csv(OUT/'saved_actual_cycles.csv')
    cycle_lines = ['## 全部实际持仓周期', '',
        '每行从实际空仓进入至实际全部退出，期间可能加减仓。净利润含实际费用与分红；最后一段按固定终点开盘清算。', '',
        '|历史|费用|实际进入|实际退出|持仓收盘数|买入次数|卖出次数|净利润（元）|',
        '|---|---|---|---|---:|---:|---:|---:|']
    for row in cycles.itertuples():
        cycle_lines.append(f"|{'主历史' if row.period == 'evaluation' else '较早历史'}|{'基础' if row.cost == 'BASE' else '压力'}|{row.entry_date}|{row.exit_date}|{row.held_closes}|{row.buy_trades}|{row.sell_trades}|{row.net_profit:.2f}|")
    decision = ('第162轮日内隔夜带符号排序完成。主基础／压力净夏普0.253／0.232，较早0.444／0.422；'
        '年化收益主1.41%／1.27%、较早3.17%／2.99%。四项夏普均低于1.2和143，四段年化收益均落后买入持有，完整目标未实现，关闭这套固定结构。')
    detail = ('六项必要测试4.26秒通过，零模型训练，一套规则四账户核心计算1.464960秒，不含开发、测试、核对和交付。'
        '独立由原始开收盘及分红重建日内、隔夜和差值，再用独立平均名次排序复算3396个完整窗口；保存分数最大误差为零。'
        '四账户5646个收盘决定、18段实际完整持仓、184次原始开盘成交及净资产、费用、分红均已核对。\n\n'
        '主每账户5段持仓、498个持仓收盘、51次成交；较早各4段、605个持仓收盘、41次成交。'
        '主499个正目标和1105个零目标，较早605个正目标和614个零目标；主正目标比持仓收盘多一来自固定终点开盘清算。'
        '所有评价窗口没有未知目标、全部为零的差值窗口或受阻未成交请求，四账户均已完整清仓。\n\n'
        '主基础价格加分红利润21426.40元、费用2048.62元、净赚19377.78元；压力分别21219.20、3735.51、17483.69元。'
        '较早基础价格加分红利润35577.40元、费用1504.56元、净赚34072.84元；压力分别34875.70、2843.45、32032.25元。'
        '主最大回撤14.90%／15.25%，较早16.10%／16.25%。\n\n'
        '相对143，主终值少47024.30／45266.14元，较早少24741.63／29327.61元。'
        '主基础价格损益少50674.00元，分红多1716.80元、费用少1932.90元，仍不能抵消价格路径损失；压力亦相同。'
        '四段年化收益分别落后买入持有2.22／2.34／0.72／0.88个百分点。较早夏普虽高于买入持有，收益仍较低，不能混称四项指标都改善。\n\n'
        '排序方法相对161只在主区间改善，较早反而下降；相对160则四项夏普与年化收益都更低。'
        '因此不挑选年份组合这些方案。下一项直接利用已齐全的免费成交量，检验上涨动量与价量正相关共同准入、'
        '相关转负或动量消失时退出的一套固定规则。\n\n'+'\n'.join(cycle_lines))
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'SESSION_SIGNED_RANK': 'CLOSED_FOUR_SHARPE_BELOW12_AND143_FOUR_CAGR_BELOW_BUY_HOLD'}, exclusive=True)
    next_path = ROOT/'docs/510300_PRICE_VOLUME_COHERENCE_NEXT_20260910.md'
    delivery = deliver_round(ROOT, OUT, CONFIG,
        ROOT/'deliverables/510300日内隔夜排序_第162轮_20260910/日内隔夜排序_结果及全部中文规则.md',
        '日内隔夜带符号排序、连续进出确认与普通波动仓位', 'CLOSED_SESSION_SIGNED_RANK_FULL_GOAL_NOT_MET',
        decision, detail, next_path, next_path.read_text(encoding='utf-8').splitlines(),
        'PRICE_VOLUME_COHERENCE_FINITE_CANDIDATE_PREPARED', '二十日上涨动量与价量相关确认进入，相关转负或动量消失确认退出')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=163, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False)
    index['current_best_four_scenario_comparison_candidate'].update(last_compared_completed_round=162,
        comparison_basis='SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO162')
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
