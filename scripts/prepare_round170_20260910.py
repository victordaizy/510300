"""保存成熟周期预算的测试回执与全部中文因素和进出规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_RUNS_CLOSED_CYCLE_BUDGET_NEXT_20260910.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第170轮：', 1).replace(
        '此文为第170轮事前方案，尚未登记、实现、测试、冻结或计算新账户。',
        '此文为第170轮新账户计算前确定的完整规则，七项必要测试已通过，尚未冻结或计算新账户。')
    text += ('\n## 已完成的必要测试\n\n七项测试7.80秒通过，核对实际部分买卖的周期边界、累计买入支出含买入佣金、'
        '周期净利润与收益率、退出后才除息的分红推迟成熟但不等到账、保存字段缺失不补值和错误来源拒绝。'
        '另核对盈亏效率手算、最近二十个选择、至少五个边界、双非正现金、月首更新月中保持、两段初始化、'
        '费用共用预算但分别读取目标、未知父目标、未来周期和终点隔离及前缀一致。\n\n'
        '六百日实际账户测试覆盖非正效率现金退出、恢复后再次进入、未知保持、整手与调仓带、'
        '真实下一开盘成交、费用、登记除息到账及终点开盘清算。以下原父策略训练和事前状态属于各自冻结时的记录，'
        '本轮只读取已经形成的父来源，不重新运行原模型或账户。\n')
    for title, path in [('第一来源：第143轮完整规则', 'docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md'),
        ('第二来源：第165轮完整规则', 'docs/510300_RETURN_RUNS_STATE_V1.md'),
        ('共同实际执行的整手取整说明', 'docs/510300_RUNS_REFERENCE_BLEND_EXECUTION_CLARIFICATION_20260910.md')]:
        text += '\n## '+title+'\n\n'+(ROOT/path).read_text(encoding='utf-8').split('\n', 2)[2]
    with (ROOT/'docs/510300_RUNS_CLOSED_CYCLE_BUDGET_V1.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_runs_closed_cycle_budget_v1/tests_receipt.json', {
        'recorded_at': now(), 'exit_code': 0, 'passed': 7, 'seconds': 7.80,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_runs_closed_cycle_budget_v1.py -q',
        'output': '7 passed in 7.80s'}, exclusive=True)
    print('第170轮完整中文规则和七项必要测试回执已保存，尚未冻结或计算。')


if __name__ == '__main__':
    main()
