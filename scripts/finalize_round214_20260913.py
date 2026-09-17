"""交付九月增量结果与原月首系数，保留已完成计算和历史比较口径。"""
import json
import shutil

import pandas as pd

from research.fast_round_delivery_v1 import deliver_round
from research.september_monthly_continuation_v1 import ROOT, OUT, CONFIG, PRIMARY, read
from research.intraday_overnight_increment_v1 import now, write_json, require


def main():
    result = read(OUT / 'result.json')
    verified = read(OUT / 'saved_verification_receipt.json')
    require(verified['accounts'] == 22 and verified['incremental_rows'] == 198, '保存核对范围不同')
    require(result['continuous_cutoff'] == '2026-09-11' and result['new_model_fits'] == 2, '本轮模型或截止日不同')
    require(result['full_pipeline_incremental_resume_verified'], '全图实际增量尚未通过')
    unchanged = {}
    for label, filename in [('岭回归', 'ridge_models.json'), ('持仓内模型', 'within_models.json')]:
        records = read(OUT / filename)['models']
        unchanged[label] = records[-1]['model'] == records[-2]['model']
    require(all(unchanged.values()), '新旧模型系数关系不同，需要按真实结果解释')
    decision = ('固定85%与15%两来源方案已经连续到9月11日收盘。基础费用净夏普1.2848、年化10.91%；压力费用净夏普1.2205、年化10.31%，历史点值继续达到目标。'
                '本轮复用38742条旧账户记录，仅增算198条；原定两条九月模型已补齐，全部22条账户实际恢复并通过保存核对。')
    detail = [
        '主历史为2020年1月2日至2026年9月11日收盘，1624个交易日；真实收盘计价，未人为指定终点清仓。较早历史表直接引用209保存的2015年至2019年结果，本轮没有重跑。', '',
        '## 新增九月实际上发生了什么', '',
        '9月1日至9月11日共9个交易日，主策略两档费用都保持现金：收益为零，买卖次数为零，新增佣金和滑点为零。基础期末研究资产400689.27元，压力期末386452.42元。九天现金等待日全部计入全期指标，因此夏普和年化较八月末略低；没有删除等待日，也没有为了提高回测而改规则。', '',
        '|延续区间（按收盘净值比较）|费用|交易日数|净利润|净收益|成交次数|',
        '|---|---|---:|---:|---:|---:|',
    ]
    for row in result['extension_segment_results']:
        detail.append(f"|{row['start_close']}至{row['end_close']}|{'基础' if row['cost']=='BASE' else '压力'}|{row['completed_days']}|{row['net_profit_cny']:,.2f}元|{row['net_return']:.4%}|{row['trades']}|")
    detail += [
        '', '上表各区间已包含实际模拟买卖费用，未把短区间收益年化。全部补充历史的成交来自八月自然退出，九月没有新增主策略交易。主策略现金状态不能代替内部来源状态：两个学习退出来源及一个下行风险参考仍有持仓，其余来源和预算也必须正常更新。', '',
        '## 原定九月模型与全部八项因素', '',
        '本轮实际生成两条模型记录：原岭回归继续收益模型、原持仓内继续收益模型。各自保留原141条历史记录，新增1条。模型的研究拟合时点是9月1日15:05，文件实际在9月13日生成，属于事后按原规则补算，不能当作9月1日已实时保存的记录。', '',
        '两条模型均用最近20个已自然结束的持仓周期，共790行状态；最晚成熟退出日为7月20日。此次没有新增已成熟训练样本，所以两条九月模型的参数分别与各自八月模型完全相同。新记录补齐原月度日程，并未产生新的预测信息或改善因子。', '',
        '|因素|中文计算规则|', '|---|---|',
        '|持仓时间对数|已经持有的交易日数加一，再取自然对数。|',
        '|含分红浮盈浮亏|把已享有分红计入当前持仓价值，除以该周期入场总成本，再减一；入场总成本包含原费用。|',
        '|持仓回撤|当前含分红持仓价值相对本周期此前最高价值的跌幅；处于新高时为零。|',
        '|入场模式|使用入场时保存的原模式编号；本次训练样本均为模式一，不事后重新分类。|',
        '|五日涨跌|每日含分红简单收益加一并取自然对数，再把最近五个交易日的结果相加。|',
        '|二十日涨跌|每日含分红简单收益加一并取自然对数，再把最近二十个交易日的结果相加。|',
        '|一百二十日均线偏离|当前含分红财富水平相对于其一百二十日均线的偏离。|',
        '|二十日波动|最近二十个交易日含分红简单收益的样本标准差，乘以二百四十二的平方根。|', '',
        '每项因素先减去下表训练均值，再除以训练标准差；结果超过正五或负五时限制在该范围，乘以对应系数后相加，最后加上该模型截距，得到继续持有相对下一开盘退出的预测收益。岭回归截距为−0.009292494115068821，持仓内模型截距为−0.009294441778808393。表格便于阅读保留十位小数，完整精度保存在同目录“九月模型完整中文系数.csv”。', '',
        '|模型|因素|训练均值|训练标准差|标准化系数|', '|---|---|---:|---:|---:|',
    ]
    coefficients = pd.read_csv(OUT / '九月模型完整中文系数.csv')
    for row in coefficients.to_dict('records'):
        detail.append(f"|{row['模型']}|{row['因素']}|{row['原训练均值']:.10f}|{row['原训练标准差']:.10f}|{row['标准化系数']:.10f}|")
    detail += [
        '', '入场模式系数约为零，原因是本组训练样本模式相同；不能据此宣称该因素已产生择时贡献。持仓内模型保留原周期内去均值、岭惩罚一、各周期独立截距及平均预测截距；两个模型都保持每个成熟周期总训练权重一。', '',
        '## 怎样进入，怎样退出', '',
        '主方案把“206加仓过滤来源”的计划持有比例乘85%，把“168机会补充来源”的计划持有比例乘15%，再相加。计划持有比例包含来源当前份额及已提出的下一开盘申请，按各自实际研究净值换算。主方案从现金进入时按原目标及整百份约束买入；持仓后的普通调仓使用十个百分点门槛，明确零目标则全部退出。',
        '206内部仍使用原两成调仓门槛、原目标放大1.15倍并最高限制为全仓；已有持仓增加买入时，要求当日含分红收益至少1%，首次入场和减仓按各自原规则处理。主方案没有额外重复这一加仓过滤。',
        '学习退出来源先满足原六十日日内相对隔夜因素的连续两日入场条件；已持仓时，预测继续收益连续两个收盘为负触发下一开盘退出，原信号反转、亏损、回撤、持有期限和再入场条件继续生效。持仓内退出来源在首次实际买入时锁定模型，月首更新不会替换该笔已有持仓的版本。',
        '全部父来源的趋势、连续段、相邻日相关、涨跌日数、波动及下行预算、反弹、机会补充等详细进入和退出阈值，见同目录“固定两来源方案_完整中文执行说明.md”和“必要来源因素及规则.md”。这两份从已验证交付复制，全部用中文，不能用本段概述替代其细则。', '',
        '## 提速与必要核对', '',
        '只新增两次原定月首拟合，22条既有账户各接9个日期，共198条账户日记录；直接复用38742条旧记录。另延长一条原训练基准用于判断周期是否成熟，没有新增参考策略。共享来源只算一次，纯目标层不运行多余账户。',
        '三项新增月首训练测试通过，复用之前十项连续引擎与因素测试。全部22条保存状态已实际恢复，旧账本和有效决定区间目标均一致；主策略压力账户使用同一完整目标进行一次连续执行核对，也与增量结果一致。没有为证明增量而把所有22条来源重新全历史计算。',
        '第一次检查比较了账户启动前的准备期缓存，因该区间一个处理阶段有预算、另一个阶段为空值而中止；实际可决策区间没有差异。修正比较起点后复用已经生成的两模型和五条增量账户，仅继续其余十七条。未改任何价格、模型、交易和收益。原中止与接续记录保留。',
        f"保存总耗时{result['run_seconds']:.2f}秒，包括该次检查中止和接续开发，不作为纯计算耗时。此次新增账户行占最终38940条的{198/38940:.2%}；这一比例说明复用了多少历史，不能直接换算成同等倍数的运行加速。", '',
        '## 接下来直接做什么', '',
        '固定当前22条完整研究状态及9月11日已经形成的申请，准备下一官方交易日的单次增量续算。9月14日之前没有新的完整交易日可补；每日不重复拟合，非月首只处理新增日期。旧历史和已经完成的诊断不重跑，EPS、公募及估值等慢数据继续暂停。',
        '历史净夏普1.2和年化10%的点值目标已经通过。现有补充行情均发生在9月13日方案正式冻结之前，严格前瞻收益仍为零；独立稳定表现没有建立，因此持续研究目标保持进行中。冻结后只记录真实取得的新数据和实际生成时间，不能把事后补算日期称为及时前瞻。',
    ]
    write_json(OUT / 'candidate_outcomes.json', {
        'recorded_at': now(), 'status': 'FIXED_STRATEGY_MONTHLY_DELTA_HISTORICAL_POINT_PASS',
        'new_candidates': 0, 'new_model_fits': 2, 'monthly_models_unchanged_coefficients': unchanged,
        'continuous_two_cost_point_pass': all(r['net_sharpe'] >= 1.2 and r['annualized_return'] >= .1 for r in result['all_metrics']),
        'goal_achieved': False, 'independent_validation': 'NOT_ESTABLISHED',
        'next': '固定研究延续起点，新增完整交易日只做单次增量'}, exclusive=True)
    nxt = ROOT / 'docs/510300_FIXED_INCREMENTAL_RESEARCH_NEXT_20260913.md'
    doc = ROOT / 'deliverables/510300九月增量与月首模型_第214轮_20260913/固定策略至9月11日_收益与全部月首因素.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '固定策略九月增量及月首模型实际结果',
        'COMPLETED_TWO_MONTHLY_MODELS_22_DELTA_ACCOUNTS_HISTORICAL_POINT_PASS', decision, '\n'.join(detail),
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'FIXED_RESEARCH_STATE_AND_SINGLE_DATE_CONTINUATION_PREPARED',
        '固定9月11日研究状态和已有申请，参数化单日增量入口；新收盘数据到齐后只处理新增日期')
    content = doc.read_text(encoding='utf-8').replace('## 主历史：2020年1月2日至2026年8月14日开盘', '## 主历史连续账户：2020年1月2日至2026年9月11日收盘')
    doc.write_text(content, encoding='utf-8')
    previous = ROOT / 'deliverables/510300连续续算实际结果_第213轮_20260913'
    for name in ['固定两来源方案_完整中文执行说明.md', '必要来源因素及规则.md']:
        shutil.copy2(previous / name, doc.parent / name)
    path = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
    index = read(path)
    for item in [index['latest_completed_round'], index['completed_rounds'][-1]]:
        item['configuration'] = str(CONFIG.relative_to(ROOT))
        item['metric_period_note'] = '主历史连续至2026-09-11收盘；209原四情景联合比較仍保留各自旧日期口径'
    index['next_work'].update(candidate_round=215, registered=False, planned_settings=0, planned_new_accounts=22,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=True,
        planned_existing_account_incremental_continuations=22, research_class='FIXED_RESEARCH_STATE_AND_SINGLE_DATE_CONTINUATION',
        next_official_trading_day='2026-09-14', source_cutoff='2026-09-11')
    index.update(status='ROUND214_HISTORICAL_POINT_PASS_TO_SEPTEMBER11_FULL_GRAPH_DELTA_VERIFIED',
        latest_monthly_incremental_continuation_result=str((OUT / 'result.json').relative_to(ROOT)),
        latest_continuous_account_cutoff='2026-09-11', supplemental_strategy_used_cutoff='2026-09-11',
        full_pipeline_incremental_resume_verified=True, strict_forward_evidence_days=0, independent_validation='NOT_ESTABLISHED')
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
