"""交付八套调仓门槛和日龄证据，更新联合比较候选。"""
import json

import pandas as pd

from research.fast_round_delivery_v1 import deliver_round
from research.finite_rebalance_band_batch_v1 import CANDIDATES, CONFIG, OUT, ROOT
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    checked = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    ranking = pd.read_csv(OUT/'ten_setting_joint_comparison.csv')
    selected = ranking.iloc[0].to_dict()
    require(selected['model'] == 'INTENT_MIX_BAND_00', '联合比较首位与已读结果不同')
    require(not any(r['four_scenario_joint_pass'] for r in joint['candidates'].values()), '完整通过状态不同')
    early = [m for m, r in joint['candidates'].items() if r['earlier_two_cost_joint_pass']]
    require(set(early) == {'ZERO_CONFIRM_BAND_20', 'INTENT_MIX_BAND_20'}, '较早联合通过集合不同')
    decision = ('第201轮八套调仓门槛、三十二条完整账户已完成。简单三来源合并在每个收盘计算调至目标中心时，四场景联合最弱门槛比值由原198的0.9676提高至0.9773，'
        '成为当前最接近目标的历史比较候选；主压力夏普仍为1.1728、年化9.8216%，较早两档夏普仍低于1.2，整体目标未完成。')
    detail = [
        f"六项必要测试通过。三十二条新账户核心计算{result['run_seconds']:.2f}秒，不含开发、测试与核对。八套共读数据与两个父目标，四种对照直接复用；零新模型、参考账户或外部来源。",
        f"保存核对完成：{checked['actual_decisions_checked']}个决定、{checked['complete_actual_cycles']}个完整持仓周期、{checked['simulated_fills_checked']}次模拟成交。按每个候选自己的门槛和退出类型核对请求、净值、份额、分红及费用。",
        '本批主候选预先指定为简单合并的十五个百分点门槛，主基础／压力夏普1.216／1.150、年化10.41%／9.78%，未达标。最终保留的零门槛是八套实际结果中的事后比较选择，不能把它改称事前主方案。',
        '零门槛合并的四场景夏普均略高于原198，四场景年化则均低于原198。它改善的是当前最弱夏普距离，不能称为全部指标改善；原198作为比较保留。',
        '相对第200轮原十个百分点合并，零门槛的主压力夏普从1.1586提高到1.1728，年化从9.8308%降至9.8216%。模拟成交从143次增至218次，费用增加639.19元、价格与分红毛利润增加431.60元，净资产少207.59元。更频繁调仓没有带来节费，夏普变化来自收益与波动路径。',
        '二十个百分点门槛的两个候选均在较早历史两档费用通过。简单合并较早夏普1.251／1.240、年化11.29%／11.26%，但主压力夏普1.158、年化9.88%，仍未全面通过。',
        '', '## 八套新方案及两个原十个百分点对照', '',
        '|策略|主基础夏普／年化|主压力夏普／年化|较早基础夏普／年化|较早压力夏普／年化|最弱门槛比值|',
        '|---|---|---|---|---|---:|',
    ]
    for row in ranking.to_dict('records'):
        cells = [f"{row[tag+'_net_sharpe']:.4f}／{row[tag+'_annualized_return']:.3%}" for tag in ['main_base', 'main_stress', 'earlier_base', 'earlier_stress']]
        detail.append('|'+row['name']+'|'+'|'.join(cells)+f"|{row['minimum_joint_ratio']:.4f}|")
    diagnostic = ROOT/'reports/research/510300_saved_entry_age_attribution_through201/result.json'
    require(diagnostic.is_file(), '日龄诊断缺失')
    detail += ['', '## 入场日龄证据与下一项', '',
        '读取最接近候选四份已保存账本，按实际入场首日、第2至5日、第6至10日、第11日以后归因。所有阶段的合计净利润在四个场景都为正。主压力首日合计5246.58元，第2至5日128719.25元，较早首日7613.24元；不支持简单删除首日或统一在持仓后半段退出。',
        '该诊断没有新账户，不是延后入场或缩短持有期的回测。阶段的资金规模、交易日数和市场机会不同，不作显著性或因果结论，也不能只选择事后亏损交易。归因表与完整周期表保存在reports/research/510300_saved_entry_age_attribution_through201。',
        '下一项改为可当时判断的完整账户周期亏损退出：零与二十个百分点合并分别检验亏损达到1%、2%、3%时触发下一开盘卖出，六套、二十四条账户。定义实际入场前基准、部分调仓不重置、受阻退出持续和原目标归零后再入场。现阶段仅准备规则，尚未实现、测试、冻结或回测。',
        '所有已观察历史仍是策略研究回放。当前最接近目标不等于已验证达到目标，独立验证未建立；持续目标保持进行中。',
    ]
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'earlier_two_cost_joint_pass_candidates': early,
        'post_selected_joint_candidate': selected['model'], 'minimum_joint_ratio': selected['minimum_joint_ratio'],
        'candidates': {m: 'EARLIER_TWO_COST_JOINT_PASS_MAIN_FAIL' if m in early else 'FULL_JOINT_TARGET_NOT_MET' for m in CANDIDATES}}, exclusive=True)
    nxt = ROOT/'docs/510300_ACCOUNT_CYCLE_LOSS_EXIT_BATCH_NEXT_20260913.md'
    doc = ROOT/'deliverables/510300调仓门槛批量比较_第201轮_20260913/八套调仓门槛_结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '两套现有策略的八种调仓门槛',
        'COMPLETED_JOINT_GAP_IMPROVED_FULL_TARGET_NOT_MET', decision, '\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'ACCOUNT_CYCLE_LOSS_EXIT_BATCH_PREPARED',
        '按新账户实际持仓周期亏损触发退出，两种已保存调仓方式各检验三档阈值')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    old = index['current_best_joint_comparison_candidate']
    require(selected['minimum_joint_ratio'] > old['minimum_joint_ratio'], '新候选没有缩小最弱门槛差距')
    index['previous_best_joint_comparison_candidate_before201'] = old
    index['current_best_joint_comparison_candidate'] = {'round': 201, 'study': result['study_id'], 'model': selected['model'],
        'source_result': str((OUT/'result.json').relative_to(ROOT)), 'minimum_joint_ratio': selected['minimum_joint_ratio'],
        'goal_achieved': False, 'independent_validation': 'NOT_ESTABLISHED',
        'status': 'POST_SELECTED_FOUR_SCENARIO_JOINT_GAP_IMPROVEMENT_MAIN_STRESS_AND_EARLIER_SHARPE_FAIL',
        'last_compared_completed_round': 201,
        **{k: selected[k] for k in selected if k.endswith(('_net_sharpe', '_annualized_return', '_max_drawdown'))}}
    index['next_work'].update(candidate_round=202, registered=False, planned_settings=6, planned_new_accounts=24,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        source_preflight=str(diagnostic.relative_to(ROOT)), research_class='RETROSPECTIVE_COMPLETE_ACCOUNT_CYCLE_LOSS_EXITS')
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    index['latest_rebalance_band_comparison'] = str((OUT/'ten_setting_joint_comparison.csv').relative_to(ROOT))
    index['latest_saved_entry_age_attribution'] = str(diagnostic.relative_to(ROOT))
    index['supplementary_saved_entry_age_attribution'] = {'source': str(diagnostic.relative_to(ROOT)), 'ledger_files_read': 4,
        'complete_cycles': 134, 'new_accounts': 0, 'independent_validation': False}
    for key in ['latest_main_return_improvement_candidate', 'current_best_four_scenario_comparison_candidate',
                'latest_main_sharpe_pass_candidate', 'latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 201
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
