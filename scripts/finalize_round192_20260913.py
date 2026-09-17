"""交付本轮完整模型、中文规则及失败归因，保留继续研究的准确状态。"""
import json
import shutil

from research.holding_market_coupling_exit_v1 import CONFIG, OUT, PRIMARY, ROOT
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    joint = json.loads((OUT/'joint_target_assessment.json').read_text(encoding='utf-8'))
    verified = json.loads((OUT/'saved_verification_receipt.json').read_text(encoding='utf-8'))
    require(not any(joint['candidates'][PRIMARY].values()), '本轮联合门槛结论不同')
    decision = ('第192轮持仓与市场联合作用未实现完整目标：主历史基础／压力净夏普0.763／0.714，年化6.35%／5.91%；'
        '较早夏普0.775／0.789，年化8.86%／9.09%。主历史明显弱于原固定版本，较早增量很小，结束该二十项结构，不继续调参。')
    detail = [
        f"六项必要测试通过。141个月度时点中114个支持成熟，只对25组不同输入拟合，其余89个月复用此前相同输入；27个原支持不足时点保持。没有求解失败。模型与四条模拟账户核心计算{result['run_seconds']:.2f}秒，不含开发、测试、核对和交付。",
        '',
        f"独立核对25组系数与两级尺度，二十项正规方程最大误差{verified['maximum_normal_equation_error']:.3g}；核对{verified['full_decisions_checked']}个决定、{verified['actual_holding_states']}个持仓状态、{verified['predictions_recomputed']}个有效预测、{verified['complete_cycles']}个完整周期和{verified['actual_fills_checked']}次模拟成交。",
        '相对原第128轮固定版本，主基础／压力终值分别少11680.53元／11320.10元；较早分别多1674.48元／1721.25元。主历史净夏普分别下降0.121／0.118；较早仅增加约0.009。账本核对未发现本轮模型或账户实现差错，这不是通过少算费用形成的收益差。',
        '两套基础账户2021年2月3日进入的对应交易，原版于2月19日退出、盈利11215.21元，本轮于3月1日退出、亏损5506.93元。本轮该年累计收益相对原版少7.05个百分点。不同账户已有净资产路径不同，这一对交易用于解释时点变化，不把利润差直接说成固定本金的单一因果效应。',
        '较早基础与压力的年化均有所增加，但整体夏普仍远低于1.2。压力费用可能通过本账户浮盈、回撤输入改变退出日期，所以较早压力收益略高于基础并不自动代表费用核算错误；两条独立账户的份额、价格、分红及费用均已核对。',
        '',
        '下一项回到第181轮保存方向与预算，只检验正目标期间保留初始实际份额，减少持仓中途追加或减持；它不同于第182轮固定倍率但仍逐日改变份额。计划零新模型、四条账户，具体规则已事前列明。',
        '本轮仍是反复观察过的历史模拟，未建立新的独立验证，目标保持进行中。',
    ]
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        PRIMARY: 'CLOSED_HOLDING_MARKET_COUPLING_MAIN_DEGRADATION_AND_JOINT_TARGET_FAILED'}, exclusive=True)
    next_path = ROOT/'docs/510300_ENTRY_SHARES_PRESERVATION_NEXT_20260913.md'
    document = ROOT/'deliverables/510300持仓与市场联合作用_第192轮_20260913/持仓与市场联合作用_结果及全部中文规则.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, document, '市场环境调节持仓退出',
        'CLOSED_HOLDING_MARKET_COUPLING_JOINT_TARGET_NOT_MET', decision, '\n'.join(detail), next_path,
        next_path.read_text(encoding='utf-8').splitlines(), 'ENTRY_SHARES_PRESERVATION_PREPARED',
        '第181轮正目标期间保持初次实际买入份额，归零退出，不改变来源计算')
    model_doc = OUT/'全部已拟合模型系数.md'
    shutil.copy2(model_doc, document.parent/model_doc.name)
    with document.open('a', encoding='utf-8') as stream:
        stream.write('\n\n## 所有实际拟合参数\n\n')
        stream.write(model_doc.read_text(encoding='utf-8'))
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=193, registered=False, planned_settings=1,
        planned_new_accounts=4, planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False)
    index['latest_joint_target_assessment'] = str((OUT/'joint_target_assessment.json').relative_to(ROOT))
    for key in ['current_best_four_scenario_comparison_candidate', 'latest_main_sharpe_pass_candidate',
                'latest_all_four_scenario_excess_candidate']:
        index[key]['last_compared_completed_round'] = 192
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
