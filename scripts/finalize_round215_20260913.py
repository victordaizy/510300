"""交付通用日期入口的实际提速结果，下一项只接免费新增收盘输入。"""
import json
import shutil

from research.fast_round_delivery_v1 import deliver_round
from research.fixed_date_continuation_validation_v1 import ROOT, OUT, CONFIG, read
from research.intraday_overnight_increment_v1 import now, write_json, require


def main():
    result = read(OUT / 'result.json')
    checked = read(OUT / 'saved_verification_receipt.json')
    require(checked['accounts'] == 22 and result['idempotent_repeat_no_recomputation'], '新增日入口保存核对未完成')
    decision = (f"通用增量入口已完成。先接1日、再接8日，22条完整账户、决定及最终内部状态与214结果精确相同。"
        f"单日22条账户核心耗时{result['first_segment_core_seconds']:.2f}秒，八日部分{result['remaining_segment_core_seconds']:.2f}秒；"
        f"两段加等价及复用核对共{result['run_seconds']:.2f}秒。相同日期再次提交直接复用，没有重新运行账户。")
    text = '\n'.join([
        '上面指标直接引用第214轮已完成的固定策略，主历史连续至2026年9月11日收盘。较早历史仍引用209。215没有新增市场日期或策略收益，不能把工程检验当成新的达标结果。', '',
        '## 已经落实的提速', '',
        '|操作|实际计算量|本机本次耗时|', '|---|---:|---:|',
        f"|从8月31日状态接9月1日|22账户，各1日|{result['first_segment_core_seconds']:.2f}秒|",
        f"|从9月1日状态接至9月11日|22账户，各8日|{result['remaining_segment_core_seconds']:.2f}秒|",
        f"|两段运行、保存等价和重复日期复用核对|44段、198条增量行|{result['run_seconds']:.2f}秒|", '',
        '前两项为各段账户和共用因素计算及保存；总耗时还包括比较及重复调用。未计入程序开发和六项测试；这不是未来每次运行耗时的保证。旧账户历史直接读取拼接，未把22条完整历史重新回放。', '',
        '新入口接受明确的上一目录、新截止日和新输入文件，一天或连续多天使用同一程序。中断后复用已完整保存的来源；同一截止日已完成则直接返回保存结果。共享来源一次计算，纯目标层没有额外账户，每日不重复训练月度模型。', '',
        '## 怎样保证没有因提速改变结果', '',
        '22条账户最终38940条账本、38962条决定，以及现金、份额、分红应收、买入批次、持仓周期、待执行申请和模型控制器状态，均与214保存结果精确一致。存在周期文件的来源，周期记录也一致。',
        '六项新入口测试通过，覆盖可变日数、周末后的下一官方交易日、缺失日、未完成收盘、分红覆盖不足、月首记录缺项和活进程锁。原十三项引擎和训练测试直接复用，没有重复运行。',
        '原始研究起点仍是214的9月11日收盘状态，已于9月13日10:49:31固定。此次用于检验的两个临时日期段不替代正式研究起点。', '',
        '## 因素和进出场规则', '',
        '主方案仍将206来源计划持有比例乘85%，168来源乘15%后相加；普通正目标调仓采用十个百分点门槛，明确零目标全部退出。206内部的1%已有持仓加仓过滤、1.15倍目标放大及两成调仓门槛原样保留，不重复叠加到主方案。',
        '内部趋势、连续段、相邻日相关、涨跌日数、普通与下行风险预算、反弹、学习退出和入场模型锁定均保持。八项学习退出因素与两模型系数详见复制的“最新历史结果与全部月首因素.md”；全部来源中文规则见“固定两来源方案_完整中文执行说明.md”和“必要来源因素及规则.md”。', '',
        '## 接下来与当前限制', '',
        '下一项把原免费来源获取与输入接纳接到这个入口，按新增日期取得一次数据，收盘完整且分红覆盖通过后直接续算。官方日历显示下一交易日为9月14日；当前没有新的完整收盘日，因此本轮网络请求为零、新增独立交易日为零。',
        '历史基础净夏普1.2848、年化10.91%；压力净夏普1.2205、年化10.31%。独立稳定表现仍未建立，继续研究目标保持进行中。入口会分别保存规则模拟时钟与实际记录时间，迟到记录不回填成及时前瞻。',
    ])
    write_json(OUT / 'candidate_outcomes.json', {'recorded_at': now(), 'status': 'ENGINE_SPEED_AND_EQUIVALENCE_VERIFIED',
        'new_candidates': 0, 'new_model_fits': 0, 'new_performance_accounts': 0, 'new_independent_days': 0,
        'historical_point_pass_preserved': True, 'goal_achieved': False, 'independent_validation': 'NOT_ESTABLISHED'}, exclusive=True)
    nxt = ROOT / 'docs/510300_NEW_DAILY_INPUT_ADAPTER_NEXT_20260913.md'
    doc = ROOT / 'deliverables/510300按日续算提速_第215轮_20260913/按日续算提速_实际耗时与结果一致性.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '按新增日期续算的实际提速结果',
        'COMPLETED_DATE_PARAMETERIZED_ENTRY_AND_22_ACCOUNT_EXACT_EQUIVALENCE', decision, text,
        nxt, nxt.read_text(encoding='utf-8').splitlines(), 'NEW_DAILY_FREE_INPUT_ADAPTER_PREPARED',
        '接上既有免费日线与官方分红接纳，新增完整收盘出现后直接调用已验证增量入口')
    content = doc.read_text(encoding='utf-8').replace('## 主历史：2020年1月2日至2026年8月14日开盘',
        '## 引用214的主历史：2020年1月2日至2026年9月11日收盘')
    doc.write_text(content, encoding='utf-8')
    previous = ROOT / 'deliverables/510300九月增量与月首模型_第214轮_20260913'
    for name in ['固定两来源方案_完整中文执行说明.md', '必要来源因素及规则.md']:
        shutil.copy2(previous / name, doc.parent / name)
    shutil.copy2(previous / '固定策略至9月11日_收益与全部月首因素.md', doc.parent / '最新历史结果与全部月首因素.md')
    index_path = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
    index = read(index_path)
    for record in [index['latest_completed_round'], index['completed_rounds'][-1]]:
        record['configuration'] = str(CONFIG.relative_to(ROOT))
        record['metric_period_note'] = '指标全部复用214；日期入口检验未新增策略绩效'
    index['next_work'].update(candidate_round=216, registered=False, planned_settings=0, planned_new_accounts=0,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=True,
        planned_existing_account_incremental_continuations_when_new_data_available=22,
        next_official_trading_day='2026-09-14', source_cutoff='2026-09-11',
        research_class='NEW_DAILY_FREE_INPUT_ADAPTER_AND_FIXED_RESEARCH_CONTINUATION')
    index.update(status='ROUND215_DATE_INCREMENTAL_ENTRY_VERIFIED_NEW_DAILY_INPUT_ADAPTER_NEXT',
        latest_date_parameterized_continuation_validation=str((OUT / 'result.json').relative_to(ROOT)),
        date_parameterized_continuation_entrypoint='research/fixed_date_continuation_v1.py',
        strict_forward_evidence_days=0, independent_validation='NOT_ESTABLISHED')
    write_json(index_path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
