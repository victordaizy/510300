"""将旧收益信息的实际配对检查写成中文说明并更新下一步。"""
import json
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/research/510300_saved_return_information_review_v1'
REPORT = ROOT / 'deliverables/510300有限历史挖掘_旧收益信息配对检查_20260914/逆回购和季度申赎是否真的增加预测信息.md'
INDEX = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
TITLES = {
    1: '条件切换与滚动预测', 2: '收益分类', 3: '全球盘后信息', 4: '央行披露规模与资金压力',
    5: '直接优化交易效用', 6: '日历效应', 7: '全期限逆回购披露', 8: '已公告盈利广度',
    9: '已有策略的条件跟踪', 10: '原始季度申赎', 11: '前瞻EPS月度旧版本',
    14: '前瞻EPS预测进入退出', 25: '日内与隔夜差异', 31: '学习退出',
    32: '重新触发与学习退出', 86: '成交额价格冲击与退出', 116: '成交方向量与退出',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def link(path, title):
    return f'[{title}](<{path.resolve().as_posix()}>)'


def save(path, content):
    temporary = path.with_name(path.name + '.return_review_delivery.tmp')
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    result = read(OUT / 'result.json')
    inventory = read(OUT / 'original_evidence_inventory.json')
    verification = read(OUT / 'saved_evidence_verification.json')
    protocol = read(OUT / 'protocol.json')
    assert result['consistent_historical_pairs'] == 0
    assert verification['status'] == 'PASS_SAVED_LABELS_AND_TRAINING_RECEIPT_TIMING'
    pairs = {r['pair']: r for r in result['outcomes']}
    metrics = {(r['pair'], r['period']): r for r in result['period_metrics']}
    content = [
        '# 逆回购和季度申赎是否真的增加预测信息',
        '',
        '完成日期：2026年9月14日。检查的原始预测与标签截止2026年8月14日。',
        '',
        '**对四项已保存的预测配对检查后，没有找到跨时期一致的收益预测增量。全期限逆回购主预测相对原价格预测，整体平方误差增加14.49%；价格模型加入季度申赎后，平方误差增加19.55%。这些结果不支持继续把对应信息叠入策略来追求高夏普。**',
        '',
        '本次没有新拟合或重跑账户。先定位17项原研究的结果和证据文件，再直接检查已经保存的逆回购、季度申赎预测及其随后实际收益。所有失败结果保留，没有反转方向或更改窗口。夏普1.2、年化10%的目标仍未完成。',
        '',
        '## 一、为什么要做这个检查',
        '',
        '上一轮风险预测更准确，但完整账户没有变好。因此这次回到收益信息本身：给模型加上一个看似合理的因素，是否真的比只看价格预测得更好？先做这个对照，可以减少把无效因素叠加进策略造成的复杂度。',
        '',
        '本次只评价原模型的平均平方预测误差，不能据此证明任何其他用法都无效。预测误差和交易收益也不是同一个指标；某个方法即使预测较好，仍需通过完整买卖与费用检验。原模型未达标的账户结果不会因本次检查而改变。',
        '',
        '## 二、四项固定比较的结果',
        '',
        '表中数字为候选的平方预测误差相对基准的变化，正数代表误差变大、预测变差，负数代表误差减少。它们不是账户亏损、年化收益或夏普的变化。',
        '',
        '| 已保存的比较 | 整体误差变化 | 2020—2022年 | 2023年至截止日 | 跨时期一致增量 |',
        '|---|---:|---:|---:|---|',
    ]
    for key, outcome in pairs.items():
        values = [-metrics[key, p]['relative_mse_improvement'] for p in ['overall', 'first', 'second']]
        content.append(f"| {outcome['title']} | {values[0]:+.2%} | {values[1]:+.2%} | {values[2]:+.2%} | 未获得 |")
    content += [
        '',
        '逆回购每项比较有1575个成熟预测起点，覆盖80个预测起点月份；这些是同一批预测起点的三个比较，不是三批独立样本。季度申赎有26个成熟公告起点。',
        '',
        '常规逆回购全期限相对七天量，在2023年以来误差略降约0.67%，但前段更差，整体仍更差，不能只保留后段来宣布有效。全期限及买断式公开量相对价格、季度申赎相对价格，都在两个分段变差。',
        '',
        '## 三、没有把重叠的收益标签当成独立试验',
        '',
        '逆回购是每日给出20日收益预测，相邻预测的持有期大量重叠。先按预测起点月聚合误差差值，再用连续6个观测月为区块进行5000次重采样。季度申赎使用连续4个公告观测月为区块，一般相当于约一年的报告次数。种子固定为20260915。',
        '',
        '下表为“基准平方误差减候选平方误差”的重采样第5与95百分位，正数才表示候选改善。数值单位为收益小数的平方。四项比较的下界都没有大于零。',
        '',
        '| 比较 | 第5百分位 | 第95百分位 |',
        '|---|---:|---:|',
    ]
    for outcome in result['outcomes']:
        uncertainty = outcome['uncertainty']
        content.append(f"| {outcome['title']} | {uncertainty['lower_5pct']:+.8f} | {uncertainty['upper_95pct']:+.8f} |")
    content += [
        '',
        '这是已见历史上的诊断，并未校正整个项目以前的全部试验与选择，也没有重新证明历史数据的真实到达时间。它不能当作独立未来验证。',
        '',
        '## 四、两个必须保留的口径问题',
        '',
        '**逆回购的期限已经做过扩展，但披露量仍不等于实际每日净投放。** 原资料包括7、14、21、28、63天常规操作，以及另列的买断式计划和月度实际披露。原始记录为3261条常规操作、44条买断式披露。全部实际操作日与到期证据未完全还原，因此不能把这些披露数字直接相加后叫作每日净现金注入。',
        '',
        '用户指出“不能只看七天”是正确的数据口径要求。完整保留其他期限仍然必要；本次也没有因为七天对照某段表现较好，就把其他期限删掉。数据口径修正是否改善收益预测，需要单独用实际结果检验。',
        '',
        '**季度申赎反映原基金报告的份额变化。** 它不是已经测得的全市场公募净现金流。26个公告起点也不能被复制到每个交易日、当作上千个独立资金信息样本。',
        '',
        '## 五、早期数据主要用于训练，不能称为早期预测验证',
        '',
        '这两组旧模型保存的预测从2019年12月31日账户初始化附近开始，对应2020年起的未来收益。2015—2019年虽然出现在训练记录中，但没有这两组模型对应的早期预测序列。本次明确标记为“无该段保存预测覆盖”，不重训填补后再称独立验证。',
        '',
        '只将候选、基准和已成熟实际收益三者位于同一判断起点的行配对。季度申赎的2019年12月31日初始化事件没有同日起点的原季度标签，单列为未配对；没有为它新造标签。其他没有成熟标签或没有预测的行也逐行保留排除原因。',
        '',
        '整体包含截止日前所有成熟持有期；2020—2022年与2023年以来两分段只收录完整落在本段的持有期。跨分段边界的标签留在整体，不列入任一分段，避免跨期混用。',
        '',
        '## 六、本次定位的17项旧研究',
        '',
        '下表仅是本次指定的17项证据定位，不代表全项目只有17项研究。文件中保留各项原始状态、结果哈希及预测、标签和训练记录的位置；没有根据回测高低挑一个恢复为策略。前瞻EPS旧版本只保留其原记录，不代替后来修正的版本。',
        '',
        '| 原轮次 | 研究内容 | 原结果入口 |',
        '|---:|---|---|',
    ]
    for row in inventory['rows']:
        content.append(f"| {row['original_round']} | {TITLES[row['original_round']]} | {link(ROOT / row['result_path'], '查看原结果')} |")
    content += [
        '',
        '## 七、实际完成的工作与后续方向',
        '',
        f"完成四项固定配对，复核{verification['existing_labels_checked']}个已保存收益标签与原价格、分红规则的一致性，并检查{verification['repo_training_receipts']}条逆回购训练记录和{verification['flow_training_receipts']}条已拟合申赎训练记录的时间关系。没有新模型、新账户或新策略。",
        '',
        '旧文件读取时出现毫秒与微秒日期精度不一致，已统一为纳秒精度后继续；原程序副本及修正记录保留。修正没有改变时间点、预测值、标签、配对或统计规则。',
        '',
        '四项比较都没有获得本次定义的跨时期一致增量，因此不据此恢复旧逆回购、申赎策略，也不继续给它们添加参数。本次没有支持“已经找到稳定收益信息”的结论。',
        '',
        '下一步优先核对低复杂度的直接交易学习是否已被覆盖：只使用一个已知日收益因素，学习一个浅层买入／退出规则，用过去训练期的扣费后交易价值决定规则，而非仅优化预测误差。如果原研究已经等价测试，就保留原结果；若确有不同的完整方法，再事前固定训练、进出场、成本、比较对象及试验数量后执行。这个方法目前尚未登记、尚未运行。',
        '',
        '## 八、完整结果文件',
        '',
        '- ' + link(ROOT / 'docs/510300_SAVED_RETURN_INFORMATION_REVIEW_V1.md', '计算前写明的中文配对规则'),
        '- ' + link(OUT / 'original_evidence_inventory.json', '17项原研究的证据位置与状态'),
        '- ' + link(OUT / 'period_prediction_metrics.csv', '四项比较的整体及分时期误差'),
        '- ' + link(OUT / 'paired_predictions_and_losses.csv', '所有配对起点、预测、实际收益和误差'),
        '- ' + link(OUT / 'excluded_forecast_origins.csv', '全部未配对起点与原因'),
        '- ' + link(OUT / 'bootstrap_saved_statistics.csv', '全部保存的重采样统计量'),
        '- ' + link(OUT / 'saved_evidence_verification.json', '旧标签和训练记录的核对结果'),
        '',
        f"登记时间：{protocol['registered_at']}。结果完成时间：{result['completed_at']}。",
        '',
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(content), encoding='utf-8')
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
    record = {
        'study_id': result['study_id'], 'status': 'COMPLETED_NO_CONSISTENT_INCREMENT_IN_FOUR_SAVED_PAIRS',
        'completed_at': result['completed_at'], 'original_studies_in_inventory': 17,
        'fixed_comparisons': 4, 'consistent_historical_pairs': 0,
        'result': str((OUT / 'result.json').relative_to(ROOT)).replace('\\', '/'),
        'report': str(REPORT.relative_to(ROOT)).replace('\\', '/'),
        'result_sha256': hashlib.sha256((OUT / 'result.json').read_bytes()).hexdigest(),
        'report_sha256': hashlib.sha256(REPORT.read_bytes()).hexdigest(),
        'new_candidates': 0, 'new_accounts': 0, 'new_model_fits': 0,
        'new_labels': 0, 'goal_achieved': False, 'independent_validation': False,
        'old_frozen_results_preserved': True, 'position_impact': 0,
    }
    index = read(INDEX)
    index['updated_at'] = stamp
    index['status'] = 'SAVED_RETURN_INFORMATION_REVIEW_COMPLETED_NO_CONSISTENT_INCREMENT'
    index['goal_achieved'] = False
    index['running_studies'] = [r for r in index['running_studies'] if r.get('study', r.get('study_id')) != result['study_id']]
    index['latest_saved_return_information_review'] = record
    index['last_goal_turn_classification'] = 'PROGRESS_FOUR_SAVED_PREDICTION_COMPARISONS_COMPLETED'
    index['consecutive_external_data_blocked_goal_turns'] = 0
    index['latest_continuation_note'] = record['report']
    index['next_work'].update({
        'status': 'FINITE_LOW_COMPLEXITY_DIRECT_DECISION_REVIEW_AND_FIXED_DAILY_OBSERVATION',
        'source': record['report'],
        'focus': '核对单一日收益因素的浅层直接交易学习是否已等价测试；若未覆盖，事前登记一个完整方法并以过去训练、完整扣费账户检验',
        'registered': False, 'planned_settings': 0, 'planned_new_accounts': 0, 'planned_new_model_fits': 0,
        'historical_mining_allowed': True, 'external_data_required': False,
        'implementation_remaining': True,
        'implementation_remaining_scope': '下一低复杂度直接交易方法的等价性检查和事前登记；四项旧收益信息配对已完成',
        'next_historical_question_status': 'SINGLE_FACTOR_SHALLOW_DIRECT_DECISION_NOVELTY_CHECK_PENDING',
        'next_historical_question': '单一已知日收益因素、浅层交易规则、仅在过去训练期学习能否改善随后扣费后的买卖价值？',
        'known_direct_utility_prior_method': 'config/510300_direct_policy_utility_v1.json',
        'old_reverse_repo_and_quarterly_flow_failures_not_reopened': True,
    })
    save(INDEX, index)
    save(OUT / 'delivery_receipt.json', {'completed_at': stamp, 'status': 'CHINESE_REPORT_AND_INDEX_UPDATED',
         'report': record['report'], 'report_sha256': record['report_sha256'],
         'new_accounts': 0, 'new_model_fits': 0, 'goal_achieved': False})
    print(json.dumps({'status': 'DELIVERED', 'report': record['report'], 'consistent_pairs': 0}, ensure_ascii=True))


if __name__ == '__main__':
    main()
