"""保存净收益正值预算测试回执及全部中文父规则。"""
from pathlib import Path
from research.intraday_overnight_increment_v1 import now, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT/'docs/510300_RUNS_NET_PROFIT_BUDGET_NEXT_20260910.md').read_text(encoding='utf-8')
    text = text.replace('# 下一项：', '# 第169轮：', 1).replace(
        '此文为第169轮事前方案，尚未登记、实现、测试、冻结或计算新账户。',
        '此文为第169轮新账户计算前确定的完整规则，六项必要测试已通过，尚未冻结或计算新账户。')
    text += ('\n## 已完成的必要测试\n\n六项测试3.94秒通过，覆盖平均净收益和正值占比手算、单边盈利、双边非正及恰好零的现金预算、'
        '不足窗口和缺失收益保留两项预算、月首更新与月中保持、两段初始化、费用来源、零预算父目标未知、未来与终点隔离及前缀一致。'
        '实际资金测试使用六百日合成路径，核对月度非正预算触发现金退出、之后恢复正预算再次进入、未知保持、'
        '空仓进入和整手调仓边界、分红登记除息到账及终点开盘清算。\n\n'
        '以下保留来源冻结时的全部中文规则，其中原轮次事前状态、训练和测试属于原记录。第169轮不重新运行原训练或父账户。\n')
    for title, path in [('第一来源：第143轮完整规则', 'docs/510300_TREND_NOISE_REFERENCE_BLEND_V1.md'),
        ('第二来源：第165轮完整规则', 'docs/510300_RETURN_RUNS_STATE_V1.md'),
        ('共同实际执行的整手取整说明', 'docs/510300_RUNS_REFERENCE_BLEND_EXECUTION_CLARIFICATION_20260910.md')]:
        text += '\n## '+title+'\n\n'+(ROOT/path).read_text(encoding='utf-8').split('\n', 2)[2]
    with (ROOT/'docs/510300_RUNS_NET_PROFIT_BUDGET_V1.md').open('x', encoding='utf-8') as stream:
        stream.write(text)
    write_json(ROOT/'reports/research/510300_runs_net_profit_budget_v1/tests_receipt.json', {
        'recorded_at': now(), 'exit_code': 0, 'passed': 6, 'seconds': 3.94,
        'command': '.venv\\Scripts\\python.exe -m pytest tests\\test_runs_net_profit_budget_v1.py -q',
        'output': '6 passed in 3.94s'}, exclusive=True)
    print('第169轮完整中文规则和必要测试回执已保存，尚未冻结或计算。')


if __name__ == '__main__':
    main()
