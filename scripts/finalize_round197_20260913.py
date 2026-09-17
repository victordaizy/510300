"""完成部分调仓比较，并保存明确零退出确认的后续方案。"""
import json

from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json
from research.target_band_partial_rebalance_v1 import CANDIDATES, CONFIG, OUT, ROOT


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    require(not any(v['four_scenario_joint_pass'] for v in joint['candidates'].values()), '本轮存在不同的联合结论')
    decision = ('第197轮两种部分调仓均未完成夏普1.2、年化10%的联合目标。调到区间边缘的主历史基础／压力夏普1.279／1.203、年化9.36%／8.76%；'
        '调整一半差额的主夏普1.242／1.166、年化9.11%／8.51%。较早历史年化提高到约10.1%，但夏普仍约1.10。'
        '结束本次执行规则试验；原181继续作为跨场景联合比较基准。')
    detail = [
        f"十四项必要测试最终通过，八条新账户核心计算{result['run_seconds']:.2f}秒，不含开发、测试和核对。保存核对覆盖{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整周期及{checked['simulated_fills_checked']}次模拟成交；零新模型和外部数据。",
        '调到区间边缘较早基础／压力夏普1.101／1.095、年化10.15%／10.15%；调整一半差额较早夏普1.111／1.101、年化10.11%／10.09%。局部年化过线没有补上较早夏普和主历史收益的缺口。',
        '边缘方案相对181在主压力场景节约佣金滑点2499.49元，但价格和分红利润减少7548.30元，净利润最终少5048.81元。较早压力场景则节约896.64元、价格和分红利润增加3792.90元，净利润多4689.54元。成本与收益改善并未在两段历史同时出现。',
        '部分调整也没有减少成交笔数：原181主历史两档各115次，边缘方案138／136次，一半差额147／148次。每次调整变小后可能反复触及偏离带，因此不能将“部分调整”直接理解为“更少交易”。',
        '新增通用账户入口只把申请数量规则作为明确函数传入，保留原记账流程；用原申请函数接入时，与原账户逐行一致的测试通过。后续执行规则可复用它，减少重复开发。初次测试出现未知记录引起的字段类型差异，已修正记录类型后全部通过，初次输出保留。',
        '退出诊断另保存原181的明确零退出和下一、下两个交易日纯价格变化。多等一个收盘在两段历史都有正价格差线索，但它不是完整账户利润；下一项单独检验两个收盘明确归零才退出，完整计入后续资金与再入场。',
        '下一项只准备一套规则、四条账户，尚未实现、冻结或回测。本轮结果属于已经观察过历史上的策略比较，独立验证尚未建立，目标保持进行中。',
    ]
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'candidates': {m: 'EARLIER_LOCAL_GAIN_MAIN_DEGRADATION_FULL_JOINT_TARGET_NOT_MET_CLOSED' for m in CANDIDATES}}, exclusive=True)
    nxt = ROOT/'docs/510300_TWO_CLOSE_ZERO_EXIT_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300部分调仓_第197轮_20260913/部分调仓_结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '偏离带边缘和一半差额的部分调仓',
        'COMPLETED_PARTIAL_REBALANCE_EARLIER_GAIN_MAIN_DEGRADATION', decision, '\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'TWO_CLOSE_ZERO_EXIT_PREPARED',
        '首次明确零保留份额，连续第二个明确零收盘后下一开盘全退；正目标维持181原规则')
    p = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(p.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=198, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        source_preflight='reports/research/510300_saved_zero_target_exit_preflight_20260913/result.json',
        research_class='RETROSPECTIVE_EXIT_TIMING_SELECTED_AFTER_SAVED_PRICE_DIAGNOSTIC')
    index['latest_saved_zero_target_exit_preflight'] = 'reports/research/510300_saved_zero_target_exit_preflight_20260913/result.json'
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    for key in ['current_best_four_scenario_comparison_candidate', 'latest_main_sharpe_pass_candidate', 'latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 197
    write_json(p, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
