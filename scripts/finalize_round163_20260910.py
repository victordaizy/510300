"""交付价量相关的完整失败结果及中文规则，接续顺序分布的有限检验。"""
import json
from research.price_volume_coherence_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, write_json


def main():
    decision = ('第163轮价量相关与上涨动量确认完成。主基础／压力净夏普0.167／0.103，较早0.572／0.476；'
        '年化收益主0.91%／0.47%、较早3.56%／2.92%。四项夏普均低于1.2和143，四段年化均落后买入持有，完整目标未实现，关闭这套固定结构。')
    detail = ('六项必要测试3.82秒通过，零模型训练，一套规则四账户核心计算1.471371秒，不含开发、测试、核对和交付。'
        '独立由原始收盘、分红和成交份额重建两个变化序列，再用独立皮尔逊函数核对3436个完整窗口；相关系数最大数值差4.44乘十的负十六次方。'
        '四账户5646个收盘决定、116段完整持仓、369次真实开盘成交及全部资金、费用、分红已核对。\n\n'
        '主每账户30段持仓、519个持仓收盘、99次成交，较早各28段、471个持仓收盘，基础85次、压力86次成交。'
        '同样的目标比例因本账户净资产、费用与整手约束而形成不同实际成交次数，这不代表两档费用使用了不同因素。'
        '主519个正目标、1085个零目标；较早472个正目标、747个零目标。较早正目标比实际持仓收盘多一来自终点开盘清算。'
        '四场景没有未知目标或受阻未成交请求，最后均完整清仓。全部116段持仓明细见同目录保存的实际持仓周期表。\n\n'
        '主基础价格加分红利润19784.40元、费用7447.51元、净赚12336.89元；压力分别19942.60、13644.41、6298.19元。'
        '较早基础价格加分红利润45490.60元、费用6943.50元、净赚38547.10元；压力分别44539.80、13344.41、31195.39元。'
        '主最大回撤15.67%／17.27%，较早9.94%／10.52%。费用压低收益，但也不能把不足完全归因于费用：相对143，四个账户的价格损益本来就更低。\n\n'
        '相对143，主终值少54065.18／56451.64元，较早少20267.37／30164.48元。'
        '主基础价格损益少52279.10元，分红多1679.90元、费用多3465.98元；三个组成项合计少54065.18元。'
        '四段年化分别落后买入持有2.72／3.14／0.34／0.95个百分点。较早夏普高于买入持有但年化收益较低，不能只摘取夏普报喜。\n\n'
        '主、较早持仓周期中位数分别13与14.5个持仓收盘。只有主2段、较早1段持有至多两个收盘，'
        '因此不能笼统把所有失败解释成两天内反复出入；完整持有路径本身也没有形成足够利润。'
        '下一项改用三日价格顺序在六十日内的集中程度，配合上涨动量定义进入与退出，继续复用已有数据和四账户运行器。')
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'PRICE_VOLUME_COHERENCE': 'CLOSED_FOUR_SHARPE_BELOW12_AND143_FOUR_CAGR_BELOW_BUY_HOLD'}, exclusive=True)
    next_path = ROOT/'docs/510300_ORDINAL_ENTROPY_NEXT_20260910.md'
    delivery = deliver_round(ROOT, OUT, CONFIG,
        ROOT/'deliverables/510300价量相关_第163轮_20260910/价量相关_结果及全部中文规则.md',
        '二十日价量相关与上涨动量确认进入及分别确认退出', 'CLOSED_PRICE_VOLUME_COHERENCE_FULL_GOAL_NOT_MET',
        decision, detail, next_path, next_path.read_text(encoding='utf-8').splitlines(),
        'ORDINAL_ENTROPY_FINITE_CANDIDATE_PREPARED', '三日含分红价格顺序的六十日排列熵与上涨动量，分别确认进出场')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=164, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False)
    index['current_best_four_scenario_comparison_candidate'].update(last_compared_completed_round=163,
        comparison_basis='SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO163')
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
