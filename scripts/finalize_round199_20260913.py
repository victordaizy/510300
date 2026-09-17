"""集中交付68条账户，记录六个较早通过以及尚未解决的主历史缺口。"""
import json

import pandas as pd

from research.confirmed_exit_existing_budget_batch_v1 import CANDIDATES, CONFIG, OUT, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    ranking = pd.read_csv(OUT/'eighteen_confirmed_budget_comparison.csv')
    early = [model for model, row in joint['candidates'].items() if row['earlier_two_cost_joint_pass']]
    require(len(early) == 6 and not any(r['main_two_cost_joint_pass'] for r in joint['candidates'].values()), '本批联合结果不同')
    require(ranking.iloc[0].model == 'TWO_CLOSE_ZERO_EXIT', '联合比较首位不同')
    decision = ('第199轮十七套新组合、六十八条完整账户已完成。六套组合在较早历史两档成本同时达到夏普1.2和年化10%，'
        '但十七套在主历史两档成本均未联合通过；没有改善原198的四场景联合最弱指标。结束本次预算与退出规则组合，原198继续作为最接近目标的完整账户候选。')
    detail = [
        f"五项集中必要测试通过。六十八条新账户核心计算{result['run_seconds']:.2f}秒，不含开发、测试与核对；零新增预测模型、来源账户或外部数据。",
        f"保存核对完成：{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整持仓周期、{checked['simulated_fills_checked']}次模拟成交。来源预算、日期、明确零等待与实际申请数量已分别检查，三个对照直接复用。",
        '本轮预先指定的主候选“一百二十日10%预算、每日调整、两次零确认”，主基础／压力夏普1.230／1.149、年化9.03%／8.38%；较早夏普1.174／1.185、年化11.31%／11.52%，仍未解决主历史收益。',
        '六个较早两成本联合通过的组合均为区间固定预算：六十日10%、12%、15%，以及一百二十日10%、12%、15%。例如一百二十日12%较早夏普1.279／1.292、年化12.01%／12.25%，但主历史仅1.149／1.079、8.80%／8.22%。不能按年份事后切换规则来拼出达标结果。',
        '新候选中最弱门槛比值最高的是三十日12%每日预算加两次零确认，为0.9268；仍低于原198的0.9676。原198主基础／压力夏普1.230／1.161、年化10.60%／9.93%，较早夏普1.183／1.187、年化11.21%／11.33%。目标尚未完成。',
        '', '## 十七套新组合与198基准的统一比较', '',
        '|组合|主基础夏普／年化|主压力夏普／年化|较早基础夏普／年化|较早压力夏普／年化|最弱门槛比值|',
        '|---|---|---|---|---|---:|',
    ]
    for row in ranking.to_dict('records'):
        cells = [f"{row[tag+'_net_sharpe']:.3f}／{row[tag+'_annualized_return']:.2%}" for tag in ['main_base', 'main_stress', 'earlier_base', 'earlier_stress']]
        detail.append('|'+row['name']+'|'+'|'.join(cells)+f"|{row['minimum_joint_ratio']:.3f}|")
    mix_path = ROOT/'reports/research/510300_incremental_saved_mix_through199/result.json'
    mix = json.loads(mix_path.read_text(encoding='utf-8'))
    detail += ['', '## 增量组合筛选与下一步', '',
        f"复用224条旧收益矩阵，仅新增读取136份账本，形成258条不同完整路径，两个固定起点共31次目标函数评价，用时{mix['run_seconds']:.2f}秒。没有重跑旧回测。",
        '找到的最佳虚拟组合仍未达标：主压力夏普1.181、年化9.84%，较早夏普约1.181；最弱门槛比值0.9838。它只是日净收益的理想加权，不是本轮已成交的模拟账户，不能用来替代原198成绩或独立验证。',
        '虚拟解由原198、199的三十日10%区间固定加两次零确认、195的二分之一反弹辅助构成，权重约77.74%、14.55%、7.72%。下一项把这三个来源的实际买卖意图合并到一个账户，比较80%／15%／5%简单权重与保存虚拟权重，检查交易合并和费用是否有实际帮助。',
        '合并时不能直接平均来源原目标：198和199第一次归零时仍保持持仓。应从来源收盘份额加已保存的下一开盘申请量，得到计划份额，再计算来源计划股票比例，保留确认期间的持仓。新账户只按合并后的净差额成交。',
        '下一项目前已准备完整规则，尚未实现、测试、冻结或回测。所有结果属于重复使用历史后的策略研究，独立验证未建立，目标保持进行中。',
    ]
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'earlier_two_cost_joint_pass_candidates': early,
        'candidates': {model: 'EARLIER_JOINT_PASS_MAIN_FAIL_CLOSED' if model in early else 'FULL_JOINT_TARGET_NOT_MET_CLOSED' for model in CANDIDATES},
        'retained_joint_candidate': 'TWO_CLOSE_ZERO_EXIT'}, exclusive=True)
    nxt = ROOT/'docs/510300_THREE_SOURCE_ORDER_INTENT_MIX_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300退出确认与预算批量组合_第199轮_20260913/17套退出确认与预算组合_全部结果及中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '十七套已有预算与两次零确认批量组合',
        'COMPLETED_SIX_EARLIER_PASSES_NO_MAIN_JOINT_PASS_NO_GLOBAL_IMPROVEMENT', decision, '\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'THREE_SOURCE_ORDER_INTENT_MIX_PREPARED',
        '按来源实际计划份额合并三个来源，只交易合并后的差额；两套固定权重')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=200, registered=False, planned_settings=2, planned_new_accounts=8,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        source_preflight=str(mix_path.relative_to(ROOT)), research_class='RETROSPECTIVE_THREE_SOURCE_PLANNED_EXPOSURE_NETTING')
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    index['latest_confirmed_budget_comparison'] = str((OUT/'eighteen_confirmed_budget_comparison.csv').relative_to(ROOT))
    index['latest_incremental_saved_mix'] = str(mix_path.relative_to(ROOT))
    index['supplementary_incremental_return_screen'] = {'result': str(mix_path.relative_to(ROOT)), 'reused_paths': 224,
        'new_paths': 34, 'total_paths': 258, 'optimizer_starts': 2, 'optimizer_objective_evaluations': 31, 'new_accounts': 0}
    for key in ['current_best_joint_comparison_candidate', 'latest_main_return_improvement_candidate',
                'current_best_four_scenario_comparison_candidate', 'latest_main_sharpe_pass_candidate', 'latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 199
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
