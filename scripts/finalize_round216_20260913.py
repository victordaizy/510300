"""交付免费新增输入的直接续算入口，明确真正剩余的外部日期条件。"""
import json
import shutil

from research.fast_round_delivery_v1 import deliver_round
from research.new_daily_input_adapter_validation_v1 import ROOT, OUT, CONFIG, RUNTIME, read
from research.intraday_overnight_increment_v1 import now, write_json, require


def main():
    result = read(OUT / 'result.json')
    verified = read(OUT / 'saved_verification_receipt.json')
    require(verified['new_tests'] == 10 and result['network_requests'] == result['new_observed_trading_days'] == 0, '当前实现或真实检查范围不同')
    decision = ('免费行情与官方分红已经接到固定策略按日续算入口。十项兼容与边界验证通过；实际当前检查显示没有新的完整收盘日，'
        '因此没有请求网络、生成新账户或增加绩效样本。下一有意义检查为9月14日15:05以后。')
    text = '\n'.join([
        '上面历史指标全部引用214已完成结果，主区间为2020年1月2日至2026年9月11日收盘。215和216仅完成提速与数据流程，不是新的策略收益或独立验证。', '',
        '## 已经连通的流程', '',
        '出现新的官方完整收盘日后，入口一次取得两路免费行情与官方分红覆盖；先检查日期、价格、成交量金额精度、旧重叠行情及分红完整性，再追加新因素并直接续接22条必要研究账户。无需再拆开若干轮资料准备和回测。',
        '相同截止日的已通过来源和已完成账户复用。失败保留实际记录，未通过来源至少五分钟后才以新尝试编号重试；旧全局输入、旧规则和原结果不覆盖。',
        '主策略仍为85%与15%两来源、十个百分点普通调仓、明确零全部退出。非月首不训练；原内部进出场、费用、分红及模型锁定全部保留。实际研究持仓不代表用户账户，也不产生实际下单。', '',
        '## 本轮实际验证结果', '',
        '|验证项目|实际结果|', '|---|---|',
        '|新浪、腾讯保存原始响应|各自解析与原30日表精确一致|',
        '|两路行情单位及精度|180项比较符合原定义，开高低收一致|',
        '|原价格加已知延续日期|3476行、15列全部精确复现212|',
        '|全部因素|3476行、63列全部精确复现212|',
        '|周末和未完整收盘|零网络请求、零账户计算|',
        '|已通过来源重复使用|直接读原响应，不重复请求|',
        '|失败来源重试|原失败保留，未到五分钟拒绝重试|', '',
        '十项测试耗时9.29秒，原十九项连续引擎、因素与训练测试以及215全图等价结果直接复用。本轮没有重跑215账户，也没有外部新数据请求。', '',
        '## 现在为什么没有新增收益', '',
        '2026年9月13日11:18:37实际运行了新入口：已接纳行情截止9月11日，官方日历中的最后完整交易日也为9月11日。入口明确返回“暂无新完整交易日”，网络请求零、新账户行零、新模型零。',
        '这是完成后退出的日期检查，不是一个仍在后台等待的计算进程。周日继续重复旧历史不能增加独立证据；下一官方交易日是9月14日，完整收盘检查从15:05后开始。', '',
        '## 下一步与结果边界', '',
        '后续直接执行现成输入入口，新数据通过后自动衔接现成日期账户入口。若出现真实来源失败、月首模型或新分红缺项，就处理该具体问题；目前这些情况尚未出现。',
        '每个新日期都保存实际申请形成时间和模拟成交收益，迟到日保留标记，不回填、不删除。历史基础净夏普1.2848、年化10.91%；压力净夏普1.2205、年化10.31%，独立稳定表现仍未建立。目标保持进行中。',
        '完整中文因素和进出场规则见复制的“固定两来源方案_完整中文执行说明.md”“必要来源因素及规则.md”和“最新历史结果与全部月首因素.md”。当前实现不新增策略因素。',
    ])
    write_json(OUT / 'candidate_outcomes.json', {'recorded_at': now(), 'status': 'INPUT_ADAPTER_READY_WAITING_NEW_COMPLETE_DAY',
        'new_candidates':0, 'new_model_fits':0, 'new_performance_accounts':0, 'new_independent_days':0,
        'goal_achieved':False, 'independent_validation':'NOT_ESTABLISHED', 'runtime_ready':True}, exclusive=True)
    nxt = ROOT / 'docs/510300_WAIT_NEW_COMPLETE_DAILY_DATA_20260913.md'
    doc = ROOT / 'deliverables/510300免费输入直接续算_第216轮_20260913/免费输入接入完成_实际验证及下一检查时间.md'
    delivered = deliver_round(ROOT, OUT, CONFIG, doc, '免费新增输入直接续算已接通',
        'COMPLETED_FREE_INPUT_ADAPTER_VERIFIED_NO_NEW_COMPLETE_DAY', decision, text, nxt,
        nxt.read_text(encoding='utf-8').splitlines(), 'RUNTIME_READY_WAITING_NEXT_COMPLETE_OFFICIAL_DAY',
        '9月14日15:05后直接执行已验证免费输入与账户续算入口；之前没有新数据不重复计算')
    content = doc.read_text(encoding='utf-8').replace('## 主历史：2020年1月2日至2026年8月14日开盘',
        '## 引用214的主历史：2020年1月2日至2026年9月11日收盘')
    doc.write_text(content, encoding='utf-8')
    previous = ROOT / 'deliverables/510300按日续算提速_第215轮_20260913'
    for name in ['固定两来源方案_完整中文执行说明.md', '必要来源因素及规则.md', '最新历史结果与全部月首因素.md']:
        shutil.copy2(previous / name, doc.parent / name)
    shutil.copy2(RUNTIME, doc.parent / '按日研究入口设置.json')
    index_path = ROOT / 'reports/research/510300_sharpe_1_2_latest_research.json'
    index = read(index_path)
    for record in [index['latest_completed_round'], index['completed_rounds'][-1]]:
        record['configuration'] = str(CONFIG.relative_to(ROOT))
        record['metric_period_note'] = '全部绩效指标复用214；输入接入与无新日期检查未增加收益样本'
    index['next_work'].update(candidate_round=217, registered=False, planned_settings=0, planned_new_accounts=0,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=True,
        planned_existing_account_incremental_continuations_when_new_data_available=22,
        next_official_trading_day='2026-09-14', next_useful_check_after='2026-09-14T15:05:00+08:00',
        source_cutoff='2026-09-11', entrypoint_ready=True, implementation_remaining=False,
        runtime_entrypoint=result['runtime_entrypoint'], runtime_settings=str(RUNTIME.relative_to(ROOT)),
        research_class='FIXED_RESEARCH_CONTINUATION_WAITING_NEW_EXTERNAL_DATES')
    index.update(status='ROUND216_DAILY_RUNTIME_READY_EXTERNAL_NEW_DAY_REQUIRED',
        latest_daily_input_adapter_result=str((OUT/'result.json').relative_to(ROOT)),
        daily_research_runtime_settings=str(RUNTIME.relative_to(ROOT)),
        latest_actual_no_new_day_check=str((OUT/'actual_calendar_check.json').relative_to(ROOT)),
        strict_forward_evidence_days=0, independent_validation='NOT_ESTABLISHED',
        last_goal_turn_classification='PROGRESS_INPUT_ADAPTER_IMPLEMENTED_TESTED_AND_CURRENT_CHECK_COMPLETED',
        consecutive_external_data_blocked_goal_turns=0)
    write_json(index_path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
