"""交付六套保护退出的失败证据，保留原201并接续买入条件研究。"""
import json

import pandas as pd

from research.account_cycle_loss_exit_batch_v1 import CANDIDATES, CONFIG, OUT, PRIMARY, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    ranking = pd.read_csv(OUT/'nine_setting_joint_comparison.csv')
    checks = pd.read_csv(OUT/'saved_protection_checks.csv')
    require(not any(r['main_two_cost_joint_pass'] or r['earlier_two_cost_joint_pass'] for r in joint['candidates'].values()), '本批联合结果不同')
    require(ranking.iloc[0].model == 'INTENT_MIX_BAND_00', '原201不再是本批最接近候选')
    decision = ('第202轮六套完整账户周期亏损退出、二十四条账户已完成，全部未达到夏普与年化联合目标。保护退出及等待原目标归零明显损失后续收益，'
        '特别是较早历史；结束这批1%、2%、3%亏损退出，不继续追加阈值。原201零门槛合并仍为当前最接近目标的历史比较候选，目标尚未完成。')
    detail = [
        f"六项必要测试通过。二十四条新账户核心计算{result['run_seconds']:.2f}秒，不含开发、测试与核对；无新模型、参考账户或外部数据。",
        f"保存核对完成：{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整持仓周期、{checked['simulated_fills_checked']}次模拟成交、{checked['protection_triggers_checked']}次保护触发。独立从实际份额变化和前序净资产重建入场基准、卖出锁、等待归零与实际请求。",
        '预定主方案为零个百分点调仓、完整账户周期亏损2%触发退出。主基础／压力夏普1.1914／1.1224，年化9.7320%／9.1068%；较早基础／压力夏普0.6055／0.3715，年化4.3843%／2.4578%，四场景均弱于未加保护的原201零门槛合并。',
        '该主方案在主历史每档费用触发六次保护，每档有51次收盘处于等待来源归零状态；较早基础触发四次、等待152次收盘，较早压力触发六次、等待176次收盘。等待次数包含触发当天尚未卖完的收盘，不等于同等天数全程空仓。',
        '较早压力主方案相对原201零门槛合并：价格与分红毛利润少115429.50元，佣金与滑点合计节省4455.40元，最终净资产少110974.10元。节费远小于减少持仓带来的利润损失。此比较是两条完整重算资金路径的差额，不能把其中某些日期单独删除后声称可获得剩余收益。',
        '原201零门槛合并的主压力夏普1.1728、年化9.8216%，较早夏普1.1840／1.1874、年化10.7959%／10.9133%。它仍未全面达标，不把本批新策略更差解释成原策略已经成功。',
        '', '## 全部六套亏损退出与原对照', '',
        '|策略|主基础夏普／年化|主压力夏普／年化|较早基础夏普／年化|较早压力夏普／年化|最弱门槛比值|',
        '|---|---|---|---|---|---:|',
    ]
    for row in ranking.to_dict('records'):
        cells = [f"{row[tag+'_net_sharpe']:.4f}／{row[tag+'_annualized_return']:.3%}" for tag in ['main_base', 'main_stress', 'earlier_base', 'earlier_stress']]
        detail.append('|'+row['name']+'|'+'|'.join(cells)+f"|{row['minimum_joint_ratio']:.4f}|")
    detail += ['', '## 实际保护次数', '', '|方案|区间|费用|保护触发|等待原目标归零的收盘数|', '|---|---|---|---:|---:|']
    for row in checks.to_dict('records'):
        detail.append(f"|{CANDIDATES[row['model']]}|{'主历史' if row['period']=='evaluation' else '较早历史'}|{'基础' if row['cost']=='BASE' else '压力'}|{row['triggers']}|{row['waiting_zero_origins']}|")
    detail += ['', '## 下一项', '',
        '下一批保留原201的进出场与持仓规则，只检验当天涨幅较大时暂缓新增买入：零及二十个百分点调仓，各比较含分红日收益不超过零、不超过1%两档准入条件，四套、十六条账户。',
        '这是一项收盘已知涨幅过滤，下一开盘仍按原成交与费用假设执行；不是限价单，不能保证更低买价，不会看见下一开盘后回改申请。卖出、减仓、明确零退出及终点清仓不受该买入过滤限制。',
        '下一批目前仅准备完整规则，尚未实现、测试、冻结或回测。所有已观察历史仍是策略研究，独立验证未建立，持续目标保持进行中。',
    ]
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'candidates': {m: 'REJECTED_CYCLE_LOSS_PROTECTION_AND_ZERO_REARM_FULL_JOINT_FAIL' for m in CANDIDATES},
        'retained_joint_candidate': 'INTENT_MIX_BAND_00', 'protection_triggers': checked['protection_triggers_checked']}, exclusive=True)
    nxt = ROOT/'docs/510300_CLOSE_RETURN_BUY_GATE_BATCH_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300账户亏损退出批量比较_第202轮_20260913/六套亏损退出_结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '六套完整账户周期亏损退出',
        'COMPLETED_ALL_CYCLE_LOSS_EXITS_FAILED_RETAIN_ROUND201', decision, '\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'CLOSE_RETURN_BUY_GATE_BATCH_PREPARED',
        '仅按收盘已知含分红涨幅限制新增买入，两种原调仓方式各检验两个上限')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=203, registered=False, planned_settings=4, planned_new_accounts=16,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        research_class='RETROSPECTIVE_CLOSE_RETURN_GATE_ON_POSITIVE_REQUESTS')
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    index['latest_account_cycle_loss_comparison'] = str((OUT/'nine_setting_joint_comparison.csv').relative_to(ROOT))
    index['latest_account_cycle_loss_outcomes'] = str((OUT/'candidate_outcomes.json').relative_to(ROOT))
    for key in ['current_best_joint_comparison_candidate', 'latest_main_return_improvement_candidate',
                'current_best_four_scenario_comparison_candidate', 'latest_main_sharpe_pass_candidate', 'latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 202
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
