"""交付涨幅上限过滤的完整失败结果，登记强势准入为新的事后方向研究。"""
import json

import pandas as pd

from research.close_return_buy_gate_batch_v1 import CANDIDATES, CONFIG, OUT, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    ranking = pd.read_csv(OUT/'seven_setting_joint_comparison.csv')
    checks = pd.read_csv(OUT/'saved_buy_gate_checks.csv')
    require(not any(r['main_two_cost_joint_pass'] for r in joint['candidates'].values()), '主历史联合结果不同')
    require(ranking.iloc[0].model == 'INTENT_MIX_BAND_00', '最接近候选不同')
    early = [m for m, r in joint['candidates'].items() if r['earlier_two_cost_joint_pass']]
    require(early == ['BUY_GATE_BAND20_R00'], '较早联合通过集合不同')
    decision = ('第203轮四套收盘涨幅上限过滤、十六条完整账户已完成，没有改善原201联合目标。禁止在强上涨日新增买入明显损失主历史收益，'
        '结束本批上限过滤。原201零门槛合并继续保留为最接近目标的历史比较候选，目标尚未完成。')
    detail = [
        f"五项必要测试通过。十六条新账户核心计算{result['run_seconds']:.2f}秒，不含开发、测试和核对。无新模型、参考账户或外部数据。",
        f"保存核对完成：{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整持仓周期、{checked['simulated_fills_checked']}次模拟成交，以及{checked['suppressed_buy_requests_checked']}次被拦买入申请。分别重算含分红日收益、十进制边界、正常申请和最终申请，原目标未被错误补成零。",
        '预定主方案为零个百分点调仓、当日涨幅不超过1%才新增买入。主基础／压力夏普0.8455／0.7528、年化4.8025%／4.2506%；较早夏普1.1401／1.1418、年化10.3120%／10.4077%，没有达到联合目标。',
        '主方案在主历史每档费用各拦27次正申请，其中19次发生于实际空仓、8次为已有持仓加仓；较早每档各拦13次，其中9次为空仓买入、4次为加仓。同一等待过程可能在多个收盘被拦，次数不等于独立交易机会数。',
        '主压力相对原201零门槛合并，价格与分红毛利润少114302.00元，费用节省5702.60元，最终净资产少108599.40元。强势日期被拦后的整条资金路径损失，远大于交易节费。',
        '二十个百分点门槛且只允许非上涨日新增的方案，在较早两档费用夏普1.2553／1.2440、年化11.0815%／11.0572%，仍通过较早点值；但主压力夏普0.6220、年化3.3254%，不能据较早结果推广为成功策略。',
        '', '|策略|主基础夏普／年化|主压力夏普／年化|较早基础夏普／年化|较早压力夏普／年化|最弱门槛比值|',
        '|---|---|---|---|---|---:|',
    ]
    for row in ranking.to_dict('records'):
        cells = [f"{row[tag+'_net_sharpe']:.4f}／{row[tag+'_annualized_return']:.3%}" for tag in ['main_base', 'main_stress', 'earlier_base', 'earlier_stress']]
        detail.append('|'+row['name']+'|'+'|'.join(cells)+f"|{row['minimum_joint_ratio']:.4f}|")
    detail += ['', '## 实际被拦买入', '', '|策略|区间|费用|被拦正申请|其中空仓买入|其中加仓|', '|---|---|---|---:|---:|---:|']
    for row in checks.to_dict('records'):
        detail.append(f"|{CANDIDATES[row['model']]}|{'主历史' if row['period']=='evaluation' else '较早历史'}|{'基础' if row['cost']=='BASE' else '压力'}|{row['suppressed_buy_requests']}|{row['suppressed_initial_entry_requests']}|{row['suppressed_addition_requests']}|")
    detail += ['', '## 下一批新方向', '',
        '观察本轮失败后，下一批单独检验当日含分红收益至少为零或至少1%时才新增买入，两种原调仓门槛共四套、十六条账户。明确记录这是事后选择的强势准入方向，不将旧上限规则反向改写或宣称独立验证。',
        '下一批只准备完整规则，尚未实现、测试、冻结或回测。继续使用原201目标及完整资金规则，所有旧失败保留，目标保持进行中。',
    ]
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'earlier_two_cost_joint_pass_candidates': early,
        'candidates': {m: 'EARLIER_PASS_MAIN_FAIL_UPPER_GATE_CLOSED' if m in early else 'UPPER_GATE_FULL_JOINT_FAIL_CLOSED' for m in CANDIDATES},
        'retained_joint_candidate': 'INTENT_MIX_BAND_00'}, exclusive=True)
    nxt = ROOT/'docs/510300_CLOSE_RETURN_BUY_STRENGTH_GATE_BATCH_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300收盘买入过滤批量比较_第203轮_20260913/四套买入过滤_结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '四套收盘涨幅上限买入过滤',
        'COMPLETED_UPPER_RETURN_BUY_GATES_FAILED_RETAIN_ROUND201', decision, '\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'CLOSE_RETURN_BUY_STRENGTH_GATE_BATCH_PREPARED',
        '事后新方向检验：当日含分红收益至少为零或1%才新增买入，卖出退出照常')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=204, registered=False, planned_settings=4, planned_new_accounts=16,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        research_class='RETROSPECTIVE_CLOSE_STRENGTH_DIRECTION_AFTER_UPPER_GATE_FAILURE')
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    index['latest_close_return_buy_gate_comparison'] = str((OUT/'seven_setting_joint_comparison.csv').relative_to(ROOT))
    for key in ['current_best_joint_comparison_candidate', 'latest_main_return_improvement_candidate',
                'current_best_four_scenario_comparison_candidate', 'latest_main_sharpe_pass_candidate', 'latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 203
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
