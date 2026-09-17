"""交付收益强弱连续段的完整失败结果，接续固定等额组合。"""
import json
import pandas as pd
from research.return_runs_state_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, write_json


def main():
    cycles = pd.read_csv(OUT/'saved_actual_cycles.csv')
    cycle_lines = ['## 全部实际持仓周期', '',
        '每行从实际空仓进入至实际全部退出，期间可能加减仓。净利润含实际费用与分红，较早历史最后一段按固定终点开盘清算。', '',
        '|历史|费用|实际进入|实际退出|持仓收盘数|买入次数|卖出次数|净利润（元）|',
        '|---|---|---|---|---:|---:|---:|---:|']
    for row in cycles.itertuples():
        cycle_lines.append(f"|{'主历史' if row.period == 'evaluation' else '较早历史'}|{'基础' if row.cost == 'BASE' else '压力'}|{row.entry_date}|{row.exit_date}|{row.held_closes}|{row.buy_trades}|{row.sell_trades}|{row.net_profit:.2f}|")
    decision = ('第165轮收益强弱连续段与上涨动量完成。主基础／压力净夏普1.007／0.963，较早0.612／0.553；'
        '年化收益主3.10%／2.96%、较早2.66%／2.38%。四项夏普均低于1.2和143，四段年化均落后买入持有，完整目标未实现，关闭这套独立规则。')
    detail = ('六项必要测试7.45秒通过，零模型训练，一套规则四账户核心计算2.138294秒，不含开发、测试、核对及交付。'
        '独立由原始收盘及分红重建收益，另行排序求窗口中位数并按连续同类分组，复算3396个完整六十日窗口，段数分数最大误差为零。'
        '四账户5646个收盘决定、34段完整持仓、114次真实开盘成交及完整资金、费用、分红均已核对。\n\n'
        '主每账户8段持仓、144个持仓收盘、27次成交；较早各9段、195个持仓收盘、30次成交。'
        '主144个正目标、1460个零目标；较早196个正目标、1023个零目标，较早多一个正目标来自固定终点开盘清算。'
        '没有未知目标或受阻未成交请求，四账户终点均无残留持仓。平均股票市值占净资产比例主约6.87%／6.88%，较早约10.70%／10.71%。\n\n'
        '主基础价格利润42914.40元、分红4336.00元、费用2418.51元、净赚44831.89元；压力分别42706.80、4329.60、4403.55、42632.85元。'
        '较早基础价格利润30384.70元、分红644.00元、费用2792.42元、净赚28236.28元；压力分别29767.80、644.00、5244.07、25167.73元。'
        '主最大回撤3.48%／3.64%，较早6.85%／7.26%。\n\n'
        '相对143，主终值少21570.18／20116.98元，较早少30578.19／36192.13元。'
        '四段年化分别落后买入持有0.53／0.65／1.24／1.49个百分点。虽然主夏普超过最近三轮，'
        '较早基础首周期净赚22205.38元，占全期净赚28236.28元的大部分，较早阶段没有同步改善。\n\n'
        '这套方法衡量收益相对局部中位数的连续性，不能将较强和较弱误写成正收益和负收益。'
        '本轮确认了因素和实际执行一致，但没有证明未来稳定超额。下一项把第143轮与本轮已定目标各取一半，'
        '单独建立新组合账户检验互补性；两个原方法的规则、来源和失败结论不变，也不平均夏普作为组合结果。\n\n'+'\n'.join(cycle_lines))
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'RETURN_RUNS_STATE': 'CLOSED_FOUR_SHARPE_BELOW12_AND143_FOUR_CAGR_BELOW_BUY_HOLD'}, exclusive=True)
    next_path = ROOT/'docs/510300_RUNS_REFERENCE_BLEND_NEXT_20260910.md'
    delivery = deliver_round(ROOT, OUT, CONFIG,
        ROOT/'deliverables/510300收益强弱连续段_第165轮_20260910/收益强弱连续段_结果及全部中文规则.md',
        '收益相对六十日中位数的强弱连续段及上涨动量', 'CLOSED_RETURN_RUNS_STATE_FULL_GOAL_NOT_MET',
        decision, detail, next_path, next_path.read_text(encoding='utf-8').splitlines(),
        'RUNS_REFERENCE_BLEND_FINITE_CANDIDATE_PREPARED', '第143轮与第165轮已定收盘目标各取一半，独立资金账户检验互补性')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=166, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False)
    index['current_best_four_scenario_comparison_candidate'].update(last_compared_completed_round=165,
        comparison_basis='SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO165')
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
